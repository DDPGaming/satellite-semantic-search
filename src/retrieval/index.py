"""FAISS Vector Index for Satellite Tile Embeddings.

Provides a deterministic, exact nearest-neighbor vector search index over L2-normalized
tile embeddings:
- Uses FAISS IndexFlatIP (Inner Product)
- Since embeddings are unit L2-normalized, Inner Product is strictly equivalent to Cosine Similarity:
    <u, v> = ||u||_2 * ||v||_2 * cos(theta) = 1.0 * 1.0 * cos(theta) = cos(theta)
- Exact flat indexing guarantees 100% precision with zero approximation/quantization error
- Preserves float32 vectors
- 1-to-1 mapping between FAISS row IDs and rich tile metadata
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import faiss
import numpy as np


class VectorIndex:
    """
    Local FAISS Vector Index supporting exact cosine similarity search over normalized embeddings.
    """

    def __init__(self, dimension: int = 512):
        """
        Initialize an empty FAISS IndexFlatIP.

        Args:
            dimension: Dimensionality of the embedding vectors (default: 512).
        """
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")

        self.dimension = int(dimension)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_entries: List[Dict[str, Any]] = []
        self._tile_id_to_row_id: Dict[str, int] = {}

    @property
    def total_vectors(self) -> int:
        """Return the total number of vectors in the FAISS index."""
        return int(self.index.ntotal)

    def add(
        self,
        embeddings: np.ndarray,
        metadata_entries: Sequence[Dict[str, Any]],
    ) -> None:
        """
        Add a batch of L2-normalized embedding vectors and their corresponding metadata.

        Args:
            embeddings: 2D NumPy array of shape (N, dimension) and float32 dtype.
            metadata_entries: Sequence of N metadata dictionaries aligned with embeddings.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D array [N, {self.dimension}], got ndim={embeddings.ndim}")

        num_vectors, dim = embeddings.shape
        if dim != self.dimension:
            raise ValueError(f"Mismatched embedding dimension: expected {self.dimension}, got {dim}")

        if num_vectors != len(metadata_entries):
            raise ValueError(
                f"Count mismatch: received {num_vectors} vectors but {len(metadata_entries)} metadata entries."
            )

        if num_vectors == 0:
            return

        # Ensure contiguous float32 array
        vecs = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Verify L2 normalization (allow small numerical tolerance)
        norms = np.linalg.norm(vecs, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-3, atol=1e-3):
            raise ValueError(
                "Vectors passed to IndexFlatIP must be unit L2-normalized for cosine equivalence. "
                f"Min norm: {norms.min():.4f}, Max norm: {norms.max():.4f}"
            )

        start_row_id = self.total_vectors

        for idx, entry in enumerate(metadata_entries):
            row_id = start_row_id + idx
            entry_copy = dict(entry)
            entry_copy["row_id"] = row_id

            tile_id = entry_copy.get("tile_id")
            if tile_id:
                self._tile_id_to_row_id[str(tile_id)] = row_id

            self.metadata_entries.append(entry_copy)

        # Add to FAISS index
        self.index.add(vecs)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Perform exact nearest-neighbor search for a query embedding.

        Args:
            query_embedding: 1D or 2D NumPy array of dimension (512,).
            top_k: Number of nearest neighbors to retrieve (must be positive).

        Returns:
            Ranked list of hit dictionaries containing:
            - rank (1-indexed)
            - tile_id
            - similarity_score (cosine similarity in [-1.0, 1.0])
            - row_id (FAISS row index)
            - scene_id
            - modality
            - bounds_wgs84
            - tile_directory
            - metadata (full tile metadata dict)
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        if self.total_vectors == 0:
            return []

        # Validate query shape
        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim == 1:
            if q.shape[0] != self.dimension:
                raise ValueError(f"Expected query dimension {self.dimension}, got {q.shape[0]}")
            q = q.reshape(1, self.dimension)
        elif q.ndim == 2:
            if q.shape[0] != 1 or q.shape[1] != self.dimension:
                raise ValueError(f"Expected query shape (1, {self.dimension}), got {q.shape}")
        else:
            raise ValueError(f"query_embedding must be 1D or 2D, got ndim={q.ndim}")

        # Ensure unit L2 normalization for query
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            raise ValueError("Query embedding cannot be all zeros.")
        if not np.isclose(q_norm, 1.0, rtol=1e-3, atol=1e-3):
            q = q / q_norm

        q = np.ascontiguousarray(q, dtype=np.float32)
        actual_k = min(top_k, self.total_vectors)

        scores, indices = self.index.search(q, actual_k)

        hits: List[Dict[str, Any]] = []
        for rank, (score, row_id) in enumerate(zip(scores[0], indices[0]), start=1):
            if row_id < 0:
                continue

            row_int = int(row_id)
            meta = self.metadata_entries[row_int]

            hits.append({
                "rank": rank,
                "tile_id": meta.get("tile_id", f"row_{row_int}"),
                "similarity_score": round(float(score), 6),
                "row_id": row_int,
                "scene_id": meta.get("scene_id"),
                "modality": meta.get("modality"),
                "bounds_wgs84": meta.get("bounds_wgs84"),
                "tile_directory": meta.get("tile_directory"),
                "metadata": meta,
            })

        return hits

    def get_metadata_by_row_id(self, row_id: int) -> Dict[str, Any]:
        """Retrieve metadata for a specific FAISS row ID."""
        if 0 <= row_id < len(self.metadata_entries):
            return self.metadata_entries[row_id]
        raise IndexError(f"row_id {row_id} out of range [0, {len(self.metadata_entries)})")

    def get_metadata_by_tile_id(self, tile_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a specific tile ID."""
        row_id = self._tile_id_to_row_id.get(str(tile_id))
        if row_id is not None:
            return self.metadata_entries[row_id]
        return None

    def save(
        self,
        index_file: Union[Path, str],
        metadata_file: Union[Path, str],
    ) -> None:
        """
        Persist the FAISS index binary and metadata mapping to local disk.

        Args:
            index_file: Path to faiss.index file.
            metadata_file: Path to metadata.json file.
        """
        idx_p = Path(index_file).resolve()
        meta_p = Path(metadata_file).resolve()

        idx_p.parent.mkdir(parents=True, exist_ok=True)
        meta_p.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(idx_p))

        meta_doc = {
            "index_type": "IndexFlatIP",
            "metric": "InnerProduct (Cosine similarity for L2-normalized vectors)",
            "dimension": self.dimension,
            "total_vectors": self.total_vectors,
            "entries": self.metadata_entries,
        }

        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(meta_doc, f, indent=2)

    @classmethod
    def load(
        cls,
        index_file: Union[Path, str],
        metadata_file: Union[Path, str],
    ) -> "VectorIndex":
        """
        Load a persisted FAISS index and metadata mapping from disk.

        Args:
            index_file: Path to faiss.index.
            metadata_file: Path to metadata.json.

        Returns:
            Populated VectorIndex instance.
        """
        idx_p = Path(index_file).resolve()
        meta_p = Path(metadata_file).resolve()

        if not idx_p.exists():
            raise FileNotFoundError(f"FAISS index file not found at: {idx_p}")
        if not meta_p.exists():
            raise FileNotFoundError(f"Metadata file not found at: {meta_p}")

        with open(meta_p, "r", encoding="utf-8") as f:
            meta_doc = json.load(f)

        dimension = int(meta_doc.get("dimension", 512))
        entries = meta_doc.get("entries", [])

        loaded_faiss = faiss.read_index(str(idx_p))

        if loaded_faiss.ntotal != len(entries):
            raise ValueError(
                f"Integrity error: FAISS index contains {loaded_faiss.ntotal} vectors "
                f"but metadata contains {len(entries)} entries."
            )
        if loaded_faiss.d != dimension:
            raise ValueError(
                f"Dimension mismatch: FAISS index has d={loaded_faiss.d}, metadata has dimension={dimension}"
            )

        instance = cls(dimension=dimension)
        instance.index = loaded_faiss
        instance.metadata_entries = entries
        instance._tile_id_to_row_id = {
            str(e["tile_id"]): int(e.get("row_id", idx))
            for idx, e in enumerate(entries)
            if "tile_id" in e
        }

        return instance
