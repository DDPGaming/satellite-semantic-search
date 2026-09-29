"""Builder for Constructing FAISS Vector Indexes from Staged Embeddings.

Scans the M3 embedding directories (data/embeddings/) and corresponding M2 tile
metadata (data/tiles/) to construct an exact, deterministic FAISS index:
- Deterministic ordering by modality, scene_id, and tile_id
- 1-to-1 linkage between embedding rows and tile geospatial metadata
- Preserves all 226 existing tile vectors without regeneration
- Zero model downloads, zero external API calls
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.retrieval.index import VectorIndex


def get_default_embeddings_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "data" / "embeddings"


def get_default_tiles_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "data" / "tiles"


def build_index_from_embeddings(
    embeddings_root: Path,
    tiles_root: Path,
    dimension: int = 512,
) -> VectorIndex:
    """
    Build a VectorIndex from all staged M3 embedding manifests and M2 tile metadata.

    Args:
        embeddings_root: Path to data/embeddings/ root.
        tiles_root: Path to data/tiles/ root.
        dimension: Embedding dimension (default: 512).

    Returns:
        Fully populated VectorIndex instance.
    """
    emb_root = Path(embeddings_root).resolve()
    t_root = Path(tiles_root).resolve()

    if not emb_root.exists():
        raise FileNotFoundError(f"Embeddings directory not found at: {emb_root}")
    if not t_root.exists():
        raise FileNotFoundError(f"Tiles directory not found at: {t_root}")

    # Discover all scene embedding manifests
    manifest_paths = sorted(emb_root.rglob("manifest.json"))
    if not manifest_paths:
        raise FileNotFoundError(f"No embedding manifests found under: {emb_root}")

    collected_vectors: List[np.ndarray] = []
    collected_metadata: List[Dict[str, Any]] = []

    for m_path in manifest_paths:
        with open(m_path, "r", encoding="utf-8") as f:
            emb_manifest = json.load(f)

        scene_id = emb_manifest["source_scene_id"]
        modality = emb_manifest["modality"]
        emb_file = m_path.parent / emb_manifest["embeddings_file"]

        if not emb_file.exists():
            raise FileNotFoundError(f"Embeddings array file missing at: {emb_file}")

        scene_vectors = np.load(emb_file)
        tile_ids = emb_manifest.get("tile_ids", [])

        if len(scene_vectors) != len(tile_ids):
            raise ValueError(
                f"Count mismatch in {m_path}: {len(scene_vectors)} vectors vs {len(tile_ids)} tile IDs"
            )

        # Determine tile directory under data/tiles/
        # S2 is in data/tiles/sentinel2/<scene_id>/, S1 is in data/tiles/sentinel1/<scene_id>/
        sensor_folder = "sentinel2" if modality == "optical" else "sentinel1"
        scene_tiles_base = t_root / sensor_folder / scene_id

        for idx, tile_id in enumerate(tile_ids):
            tile_dir = scene_tiles_base / tile_id
            tile_meta_file = tile_dir / "metadata.json"

            tile_meta: Dict[str, Any] = {}
            if tile_meta_file.exists():
                with open(tile_meta_file, "r", encoding="utf-8") as tf:
                    tile_meta = json.load(tf)

            georef = tile_meta.get("georeferencing", {})
            source_meta = tile_meta.get("source_metadata", {})
            derived_meta = tile_meta.get("derived_tile_metadata", {})

            # Resolve geographic bounds (authoritative for S2 optical; derived approximation for S1 SAR)
            bounds_wgs84 = georef.get("bounds_wgs84")
            if bounds_wgs84 is None and modality == "sar":
                approx = georef.get("derived_local_affine_approximation", {})
                coeffs = approx.get("coefficients")
                if coeffs and len(coeffs) == 6:
                    a, b, c, d, e, f_c = coeffs
                    # Tile corners at (0, 0), (256, 0), (256, 256), (0, 256)
                    corners_col = [0, 256, 256, 0]
                    corners_row = [0, 0, 256, 256]
                    lons = [c + a * x + b * y for x, y in zip(corners_col, corners_row)]
                    lats = [f_c + d * x + e * y for x, y in zip(corners_col, corners_row)]
                    bounds_wgs84 = [round(min(lons), 6), round(min(lats), 6), round(max(lons), 6), round(max(lats), 6)]

            # Clean, standardized metadata entry
            entry = {
                "tile_id": tile_id,
                "scene_id": scene_id,
                "modality": modality,
                "tile_directory": str(tile_dir.relative_to(t_root.parent)).replace("\\", "/"),
                "bounds_wgs84": bounds_wgs84,
                "bounds_projected": georef.get("bounds_projected"),
                "crs": georef.get("crs"),
                "acquisition_datetime_utc": (
                    source_meta.get("acquisition_datetime_utc")
                    or source_meta.get("acquisition_datetime")
                ),
                "grid_index": derived_meta.get("grid_index"),
                "quality_diagnostic": tile_meta.get("quality_diagnostic"),
            }

            collected_vectors.append(scene_vectors[idx])
            collected_metadata.append(entry)

    all_vectors = np.vstack(collected_vectors).astype(np.float32)

    index = VectorIndex(dimension=dimension)
    index.add(all_vectors, collected_metadata)

    return index
