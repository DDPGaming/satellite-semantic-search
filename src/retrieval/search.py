"""Natural Language Text Search Engine for Satellite Tile Retrieval.

Composes the M3 offline CLIP vision-language text encoder with the M4 FAISS
IndexFlatIP vector index:
- Validates and sanitizes natural language text queries
- Generates 512-D L2-normalized float32 query embeddings offline via CLIPEmbedder
- Performs exact cosine similarity search over indexed satellite tiles
- Supports exact exhaustive modality filtering ("optical", "sar", or None) without index duplication
- Enforces strictly non-increasing score ranking with deterministic tie-breaking (tile_id)
- Preserves the authoritative M4 tile metadata contract while surfacing acquisition datetime
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.embeddings.base import BaseEmbedder
from src.embeddings.clip_embedder import CLIPEmbedder
from src.retrieval.index import VectorIndex


def get_default_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


class TextSearchEngine:
    """
    Search engine executing natural-language text queries against a FAISS vector index.
    """

    def __init__(
        self,
        vector_index: Optional[VectorIndex] = None,
        embedder: Optional[BaseEmbedder] = None,
        index_file: Optional[Union[Path, str]] = None,
        metadata_file: Optional[Union[Path, str]] = None,
        model_dir: Optional[Union[Path, str]] = None,
        device: str = "cpu",
    ):
        """
        Initialize the TextSearchEngine.

        Args:
            vector_index: Pre-loaded VectorIndex instance (optional).
            embedder: Pre-loaded BaseEmbedder instance (optional).
            index_file: Path to faiss.index file if vector_index is not provided.
            metadata_file: Path to metadata.json file if vector_index is not provided.
            model_dir: Path to offline model directory if embedder is not provided.
            device: Inference device for CLIPEmbedder ('cpu', 'cuda', etc.).
        """
        root = get_default_project_root()

        # 1. Initialize or load VectorIndex
        if vector_index is not None:
            self.vector_index = vector_index
        else:
            idx_path = Path(index_file) if index_file else root / "data" / "index" / "faiss.index"
            meta_path = Path(metadata_file) if metadata_file else root / "data" / "index" / "metadata.json"
            if not idx_path.exists():
                raise FileNotFoundError(f"FAISS index file not found at: {idx_path}")
            if not meta_path.exists():
                raise FileNotFoundError(f"Index metadata file not found at: {meta_path}")
            self.vector_index = VectorIndex.load(idx_path, meta_path)

        # 2. Initialize or inject CLIPEmbedder
        if embedder is not None:
            self.embedder = embedder
        else:
            m_dir = Path(model_dir) if model_dir else root / "models" / "clip-rsicd-v2"
            if not m_dir.exists():
                raise FileNotFoundError(f"Model directory not found at: {m_dir}")
            self.embedder = CLIPEmbedder(model_dir=m_dir, device=device, local_files_only=True)

    @classmethod
    def from_staged_artifacts(
        cls,
        project_root: Optional[Union[Path, str]] = None,
        device: str = "cpu",
    ) -> "TextSearchEngine":
        """
        Factory helper to initialize TextSearchEngine directly from default staged directories.
        """
        root = Path(project_root).resolve() if project_root else get_default_project_root()
        idx_path = root / "data" / "index" / "faiss.index"
        meta_path = root / "data" / "index" / "metadata.json"
        model_path = root / "models" / "clip-rsicd-v2"

        return cls(
            index_file=idx_path,
            metadata_file=meta_path,
            model_dir=model_path,
            device=device,
        )

    @property
    def total_vectors(self) -> int:
        """Return the total number of vectors in the underlying index."""
        return self.vector_index.total_vectors

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.vector_index.dimension

    def encode_query(self, query_text: str) -> np.ndarray:
        """
        Validate and encode a natural language text query into a 1D float32 normalized embedding.

        Args:
            query_text: Natural language query string.

        Returns:
            1D float32 array of shape (512,) with unit L2 norm.
        """
        if not isinstance(query_text, str):
            raise TypeError(f"Query must be a string, got {type(query_text).__name__}.")

        cleaned = query_text.strip()
        if not cleaned:
            raise ValueError("Query text cannot be empty or whitespace-only.")

        emb = self.embedder.encode_text(cleaned)

        # Validate vector integrity
        if emb.ndim != 1 or emb.shape[0] != self.vector_index.dimension:
            raise ValueError(
                f"Expected query vector of shape ({self.vector_index.dimension},), got {emb.shape}"
            )

        norm = float(np.linalg.norm(emb))
        if not np.isclose(norm, 1.0, rtol=1e-3, atol=1e-3):
            if norm == 0.0:
                raise ValueError("Query embedding vector norm is zero.")
            emb = emb / norm

        return emb.astype(np.float32)

    def search(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 10,
        modality: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search satellite tiles by text query or pre-computed embedding vector.

        Args:
            query: Natural language string or 1D/2D float32 query embedding.
            top_k: Maximum number of nearest neighbors to retrieve (positive integer).
            modality: Optional modality filter: 'optical', 'sar', or None (all).

        Returns:
            Ranked list of hit dictionaries formatted according to the M5 result contract:
            - rank (1-indexed int)
            - tile_id (str)
            - similarity_score (float, cosine similarity in [-1.0, 1.0])
            - row_id (int)
            - scene_id (str)
            - modality (str)
            - bounds_wgs84 (list of 4 floats: [min_lon, min_lat, max_lon, max_lat])
            - tile_directory (str)
            - acquisition_datetime_utc (str or None)
            - metadata (dict: authoritative M4 tile metadata)
        """
        # 1. Validate top_k
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}.")
        if top_k < 1:
            raise ValueError(f"top_k must be a positive integer >= 1, got {top_k}.")

        # 2. Validate modality filter
        target_modality: Optional[str] = None
        if modality is not None:
            if not isinstance(modality, str):
                raise TypeError(f"Modality must be a string or None, got {type(modality).__name__}.")
            norm_modality = modality.strip().lower()
            if norm_modality not in ("optical", "sar"):
                raise ValueError(f"Unsupported modality '{modality}'. Must be 'optical', 'sar', or None.")
            target_modality = norm_modality

        # 3. Handle query embedding
        if isinstance(query, str):
            query_emb = self.encode_query(query)
        elif isinstance(query, np.ndarray):
            q_arr = np.asarray(query, dtype=np.float32)
            if q_arr.ndim == 2:
                if q_arr.shape[0] != 1 or q_arr.shape[1] != self.vector_index.dimension:
                    raise ValueError(f"Expected 2D query shape (1, {self.vector_index.dimension}), got {q_arr.shape}")
                query_emb = q_arr[0]
            elif q_arr.ndim == 1:
                if q_arr.shape[0] != self.vector_index.dimension:
                    raise ValueError(f"Expected 1D query shape ({self.vector_index.dimension},), got {q_arr.shape}")
                query_emb = q_arr
            else:
                raise ValueError(f"Query array must be 1D or 2D, got ndim={q_arr.ndim}")
            # Verify unit norm
            q_norm = float(np.linalg.norm(query_emb))
            if q_norm == 0.0:
                raise ValueError("Query array norm is zero.")
            if not np.isclose(q_norm, 1.0, rtol=1e-3, atol=1e-3):
                query_emb = query_emb / q_norm
        else:
            raise TypeError(f"Query must be a string or numpy ndarray, got {type(query).__name__}.")

        total_available = self.vector_index.total_vectors
        if total_available == 0:
            return []

        # 4. Search and modality filtering logic
        if target_modality is None:
            # Query FAISS directly with top_k (clamped to total_available)
            raw_hits = self.vector_index.search(query_emb, top_k=top_k)
            # Enforce deterministic tie-breaking: primary key = -score, secondary key = tile_id
            candidates = sorted(raw_hits, key=lambda h: (-h["similarity_score"], h["tile_id"]))
        else:
            # Exact exhaustive candidate pool: search all indexed vectors to avoid missing relevant items
            all_hits = self.vector_index.search(query_emb, top_k=total_available)
            # Filter strictly by modality
            filtered = [h for h in all_hits if str(h.get("modality", "")).lower() == target_modality]
            # Deterministic tie-breaking
            filtered.sort(key=lambda h: (-h["similarity_score"], h["tile_id"]))
            candidates = filtered[:top_k]

        # 5. Build standard M5 result contract
        results: List[Dict[str, Any]] = []
        for rank_idx, hit in enumerate(candidates, start=1):
            meta = hit.get("metadata", {})
            acq_dt = meta.get("acquisition_datetime_utc")

            results.append({
                "rank": rank_idx,
                "tile_id": hit["tile_id"],
                "similarity_score": hit["similarity_score"],
                "row_id": hit["row_id"],
                "scene_id": hit["scene_id"],
                "modality": hit["modality"],
                "bounds_wgs84": hit["bounds_wgs84"],
                "tile_directory": hit["tile_directory"],
                "acquisition_datetime_utc": acq_dt,
                "metadata": meta,
            })

        return results
