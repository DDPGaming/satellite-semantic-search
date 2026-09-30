"""Natural Language Text Search Engine with Metadata Filtering.

Composes the M3 offline CLIP vision-language text encoder with the M4 FAISS
IndexFlatIP vector index:
- Validates and sanitizes natural language text queries
- Generates 512-D L2-normalized float32 query embeddings offline via CLIPEmbedder
- Performs exact cosine similarity search over indexed satellite tiles
- Supports exact exhaustive metadata filtering (modality, date_from, date_to, scene_id, bbox)
- Evaluates active metadata predicates in strict logical conjunction (AND)
- Enforces strictly non-increasing score ranking with deterministic tie-breaking (tile_id)
- Preserves the authoritative M4 tile metadata contract while surfacing acquisition datetime
"""

import collections.abc
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.embeddings.base import BaseEmbedder
from src.embeddings.clip_embedder import CLIPEmbedder
from src.retrieval.index import VectorIndex


def get_default_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def _parse_filter_datetime(val: Any, is_end_of_day: bool) -> Optional[datetime]:
    """
    Parse and normalize a date/datetime filter into a timezone-aware UTC datetime.

    Rules:
    - None -> None
    - datetime.datetime: aware is converted to UTC; naive is interpreted as UTC.
    - datetime.date: normalized to 00:00:00.000000 UTC (date_from) or 23:59:59.999999 UTC (date_to).
    - ISO str:
        - Date-only (e.g. '2022-01-23'): normalized to start/end of that UTC day.
        - ISO datetime: parsed and converted/assigned to UTC.
    - Malformed str -> ValueError
    - Non-supported type -> TypeError
    """
    if val is None:
        return None

    if isinstance(val, bool):
        raise TypeError(f"Expected str, date, or datetime, got bool.")

    if isinstance(val, datetime):
        if val.tzinfo is not None and val.tzinfo.utcoffset(val) is not None:
            return val.astimezone(timezone.utc)
        return val.replace(tzinfo=timezone.utc)

    if isinstance(val, date):
        if is_end_of_day:
            return datetime(val.year, val.month, val.day, 23, 59, 59, 999999, tzinfo=timezone.utc)
        return datetime(val.year, val.month, val.day, 0, 0, 0, 0, tzinfo=timezone.utc)

    if isinstance(val, str):
        cleaned = val.strip()
        if not cleaned:
            raise ValueError("Date filter string cannot be empty or whitespace-only.")

        # 1. Attempt parsing as date-only ISO string
        try:
            d = date.fromisoformat(cleaned)
            if is_end_of_day:
                return datetime(d.year, d.month, d.day, 23, 59, 59, 999999, tzinfo=timezone.utc)
            return datetime(d.year, d.month, d.day, 0, 0, 0, 0, tzinfo=timezone.utc)
        except ValueError:
            pass

        # 2. Attempt parsing as full ISO datetime string
        try:
            s = cleaned[:-1] + "+00:00" if cleaned.endswith(("Z", "z")) else cleaned
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None:
                return dt.astimezone(timezone.utc)
            return dt.replace(tzinfo=timezone.utc)
        except Exception as err:
            raise ValueError(f"Invalid date/datetime string '{val}': {err}") from err

    raise TypeError(f"Expected str, date, or datetime, got {type(val).__name__}.")


def _parse_tile_datetime(dt_val: Any) -> Optional[datetime]:
    """Parse tile acquisition datetime into UTC datetime, returning None on failure."""
    if dt_val is None:
        return None
    if isinstance(dt_val, datetime):
        if dt_val.tzinfo is not None and dt_val.tzinfo.utcoffset(dt_val) is not None:
            return dt_val.astimezone(timezone.utc)
        return dt_val.replace(tzinfo=timezone.utc)
    if isinstance(dt_val, str):
        cleaned = dt_val.strip()
        if not cleaned:
            return None
        try:
            s = cleaned[:-1] + "+00:00" if cleaned.endswith(("Z", "z")) else cleaned
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None:
                return dt.astimezone(timezone.utc)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def _bbox_intersects(
    query_bbox: Tuple[float, float, float, float],
    tile_bounds: Optional[Sequence[float]],
) -> bool:
    """
    Check if a tile's axis-aligned WGS84 bounding box intersects the query bbox.
    Includes boundary-touching intersections.
    """
    if tile_bounds is None or len(tile_bounds) != 4:
        return False
    try:
        t_min_lon = float(tile_bounds[0])
        t_min_lat = float(tile_bounds[1])
        t_max_lon = float(tile_bounds[2])
        t_max_lat = float(tile_bounds[3])
    except (ValueError, TypeError):
        return False

    q_min_lon, q_min_lat, q_max_lon, q_max_lat = query_bbox
    return (
        max(q_min_lon, t_min_lon) <= min(q_max_lon, t_max_lon)
        and max(q_min_lat, t_min_lat) <= min(q_max_lat, t_max_lat)
    )


class TextSearchEngine:
    """
    Search engine executing natural-language text queries against a FAISS vector index
    with support for structured metadata filtering.
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
        date_from: Optional[Union[str, date, datetime]] = None,
        date_to: Optional[Union[str, date, datetime]] = None,
        scene_id: Optional[str] = None,
        bbox: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search satellite tiles by text query or pre-computed embedding vector with metadata filtering.

        Args:
            query: Natural language string or 1D/2D float32 query embedding.
            top_k: Maximum number of nearest neighbors to retrieve (positive integer).
            modality: Optional modality filter: 'optical', 'sar', or None (all).
            date_from: Optional start UTC date/datetime filter (inclusive).
            date_to: Optional end UTC date/datetime filter (inclusive).
            scene_id: Optional exact scene identifier filter.
            bbox: Optional WGS84 bounding box sequence [min_lon, min_lat, max_lon, max_lat].

        Returns:
            Ranked list of hit dictionaries formatted according to the result contract:
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

        # 3. Validate and normalize date filters
        min_datetime = _parse_filter_datetime(date_from, is_end_of_day=False)
        max_datetime = _parse_filter_datetime(date_to, is_end_of_day=True)
        if min_datetime is not None and max_datetime is not None:
            if min_datetime > max_datetime:
                raise ValueError("Invalid date range: date_from cannot be after date_to.")

        # 4. Validate scene_id filter
        target_scene_id: Optional[str] = None
        if scene_id is not None:
            if not isinstance(scene_id, str):
                raise TypeError(f"scene_id must be a string or None, got {type(scene_id).__name__}.")
            cleaned_scene = scene_id.strip()
            if not cleaned_scene:
                raise ValueError("scene_id cannot be empty or whitespace-only.")
            target_scene_id = cleaned_scene

        # 5. Validate bbox filter
        parsed_bbox: Optional[Tuple[float, float, float, float]] = None
        if bbox is not None:
            if not isinstance(bbox, (list, tuple, np.ndarray, collections.abc.Sequence)) or isinstance(
                bbox, (str, bytes, dict)
            ):
                raise TypeError(f"bbox must be a sequence of 4 floats, got {type(bbox).__name__}.")
            if len(bbox) != 4:
                raise ValueError(
                    f"bbox must contain exactly 4 coordinates [min_lon, min_lat, max_lon, max_lat], got {len(bbox)} elements."
                )

            cleaned_coords = []
            for c in bbox:
                if isinstance(c, bool) or not isinstance(c, (int, float, np.floating, np.integer)):
                    raise TypeError(f"All bbox coordinates must be numeric, got {type(c).__name__}.")
                cleaned_coords.append(float(c))

            min_lon, min_lat, max_lon, max_lat = cleaned_coords
            if min_lon > max_lon:
                raise ValueError(f"Invalid bbox: min_lon ({min_lon}) cannot be greater than max_lon ({max_lon}).")
            if min_lat > max_lat:
                raise ValueError(f"Invalid bbox: min_lat ({min_lat}) cannot be greater than max_lat ({max_lat}).")
            parsed_bbox = (min_lon, min_lat, max_lon, max_lat)

        # 6. Handle query embedding
        if isinstance(query, str):
            query_emb = self.encode_query(query)
        elif isinstance(query, np.ndarray):
            q_arr = np.asarray(query, dtype=np.float32)
            if q_arr.ndim == 2:
                if q_arr.shape[0] != 1 or q_arr.shape[1] != self.vector_index.dimension:
                    raise ValueError(
                        f"Expected 2D query shape (1, {self.vector_index.dimension}), got {q_arr.shape}"
                    )
                query_emb = q_arr[0]
            elif q_arr.ndim == 1:
                if q_arr.shape[0] != self.vector_index.dimension:
                    raise ValueError(
                        f"Expected 1D query shape ({self.vector_index.dimension},), got {q_arr.shape}"
                    )
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

        # 7. Search and metadata filtering logic
        has_filters = (
            target_modality is not None
            or min_datetime is not None
            or max_datetime is not None
            or target_scene_id is not None
            or parsed_bbox is not None
        )

        if not has_filters:
            # Query FAISS directly with top_k (clamped to total_available)
            raw_hits = self.vector_index.search(query_emb, top_k=top_k)
            candidates = sorted(raw_hits, key=lambda h: (-h["similarity_score"], h["tile_id"]))
        else:
            # Exact exhaustive candidate pool: search all indexed vectors to avoid missing relevant items
            all_hits = self.vector_index.search(query_emb, top_k=total_available)
            filtered = []
            for hit in all_hits:
                meta = hit.get("metadata", {})

                # Predicate 1: Modality filter
                if target_modality is not None:
                    hit_mod = str(hit.get("modality", "")).strip().lower()
                    if hit_mod != target_modality:
                        continue

                # Predicate 2: Date filters (inclusive UTC)
                if min_datetime is not None or max_datetime is not None:
                    tile_dt = _parse_tile_datetime(
                        hit.get("acquisition_datetime_utc") or meta.get("acquisition_datetime_utc")
                    )
                    if tile_dt is None:
                        continue
                    if min_datetime is not None and tile_dt < min_datetime:
                        continue
                    if max_datetime is not None and tile_dt > max_datetime:
                        continue

                # Predicate 3: Scene filter
                if target_scene_id is not None:
                    hit_scene = hit.get("scene_id") or meta.get("scene_id")
                    if hit_scene != target_scene_id:
                        continue

                # Predicate 4: Bounding Box intersection
                if parsed_bbox is not None:
                    hit_bounds = hit.get("bounds_wgs84") or meta.get("bounds_wgs84")
                    if not _bbox_intersects(parsed_bbox, hit_bounds):
                        continue

                filtered.append(hit)

            # Enforce deterministic tie-breaking: primary key = -score, secondary key = tile_id
            filtered.sort(key=lambda h: (-h["similarity_score"], h["tile_id"]))
            candidates = filtered[:top_k]

        # 8. Build standard result contract
        results: List[Dict[str, Any]] = []
        for rank_idx, hit in enumerate(candidates, start=1):
            meta = hit.get("metadata", {})
            acq_dt = meta.get("acquisition_datetime_utc") or hit.get("acquisition_datetime_utc")

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
