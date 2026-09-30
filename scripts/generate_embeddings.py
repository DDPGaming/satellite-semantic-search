"""CLI Script to Generate Embeddings from Staged M2 Satellite Tiles.

Processes M2 tiles from `data/tiles/` into fixed-dimensional, L2-normalized float32
embeddings and saves them in `data/embeddings/`:
- Optical Sentinel-2 tiles: true-color RGB surface reflectance composition
- SAR Sentinel-1 tiles: calibrated dual-polarization (VV/VH) composition
- Completely local and offline execution
- Separate storage from raw imagery and tile rasters
- Scene-level manifest recording tile order, shapes, hashes, and model provenance
- Verification of M1 and M2 data immutability

Note: M3 provides the shared image/text embedding capability required for future
retrieval stages. It does not implement vector indexing, search APIs, or ranking (M4/M5).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.embeddings.clip_embedder import CLIPEmbedder
from src.embeddings.preprocessing import load_image_to_pil
from src.ingestion.sentinel1 import compute_file_sha256


def get_default_tiles_dir() -> Path:
    return ROOT_DIR / "data" / "tiles"


def get_default_output_dir() -> Path:
    return ROOT_DIR / "data" / "embeddings"


def get_default_model_dir() -> Path:
    return ROOT_DIR / "models" / "clip-rsicd-v2"


def generate_scene_embeddings(
    scene_tiles_dir: Path,
    scene_out_dir: Path,
    embedder: CLIPEmbedder,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Generate and persist embeddings for all tiles in a single scene.

    Args:
        scene_tiles_dir: Path to directory containing scene tiles and tiles_manifest.json.
        scene_out_dir: Destination path under data/embeddings/<sensor>/<scene_id>/.
        embedder: Instantiated CLIPEmbedder.
        batch_size: Inference batch size.

    Returns:
        Summary dict containing tile count, file path, size, and SHA-256.
    """
    manifest_file = scene_tiles_dir / "tiles_manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Scene tiles manifest not found at: {manifest_file}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        tiles_manifest = json.load(f)

    tiles_info = tiles_manifest.get("tiles", [])
    if not tiles_info:
        raise ValueError(f"No tiles found in manifest: {manifest_file}")

    source_scene_id = tiles_manifest.get("source_scene_id", scene_tiles_dir.name)
    modality = tiles_manifest.get("modality", "unknown")

    tile_ids: List[str] = []
    tile_paths: List[Path] = []

    for t in tiles_info:
        t_id = t["tile_id"]
        t_dir_name = t.get("tile_directory", t_id)
        t_path = scene_tiles_dir / t_dir_name
        if not t_path.exists():
            raise FileNotFoundError(f"Tile directory does not exist: {t_path}")
        tile_ids.append(t_id)
        tile_paths.append(t_path)

    num_tiles = len(tile_paths)
    print(f"  Encoding {num_tiles} tiles for scene '{source_scene_id}' (modality={modality})...")

    t0 = time.perf_counter()
    embeddings = embedder.encode_images(tile_paths, batch_size=batch_size)
    encode_duration = time.perf_counter() - t0

    assert embeddings.shape[0] == num_tiles, f"Mismatched count: {embeddings.shape[0]} vs {num_tiles}"
    assert embeddings.shape[1] == embedder.embedding_dim, f"Mismatched dim: {embeddings.shape[1]}"

    # Ensure output directory exists
    scene_out_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings array as .npy
    emb_file = scene_out_dir / "embeddings.npy"
    np.save(emb_file, embeddings)
    emb_file_size = emb_file.stat().st_size
    emb_sha256 = compute_file_sha256(emb_file)

    # Write scene embeddings manifest
    manifest_doc = {
        "source_scene_id": source_scene_id,
        "modality": modality,
        "total_tiles": num_tiles,
        "embedding_dimensions": [num_tiles, embedder.embedding_dim],
        "dtype": str(embeddings.dtype),
        "normalization": "L2",
        "embeddings_file": "embeddings.npy",
        "embeddings_size_bytes": emb_file_size,
        "embeddings_sha256": emb_sha256,
        "tile_ids": tile_ids,
        "model_metadata": embedder.model_metadata,
        "generation_metrics": {
            "duration_seconds": round(encode_duration, 3),
            "latency_ms_per_tile": round((encode_duration / max(num_tiles, 1)) * 1000.0, 2),
            "batch_size": batch_size,
        },
        "generation_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    emb_manifest_file = scene_out_dir / "manifest.json"
    with open(emb_manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_doc, f, indent=2)

    total_bytes = emb_file_size + emb_manifest_file.stat().st_size

    return {
        "scene_id": source_scene_id,
        "modality": modality,
        "total_tiles": num_tiles,
        "embedding_shape": list(embeddings.shape),
        "file_size_bytes": total_bytes,
        "embeddings_sha256": emb_sha256,
        "duration_s": round(encode_duration, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate fixed-dimensional image embeddings for staged M2 tiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tiles-dir",
        type=Path,
        default=get_default_tiles_dir(),
        help="Root directory containing M2 tiles",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=get_default_output_dir(),
        help="Root directory to store generated embeddings",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=get_default_model_dir(),
        help="Path to staged model directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Target inference device (cpu, cuda, or auto)",
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default="all",
        choices=["all", "sentinel2", "sentinel1"],
        help="Sensor modality to embed",
    )
    args = parser.parse_args()

    tiles_dir = args.tiles_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_dir = args.model_dir.resolve()

    if not tiles_dir.exists():
        print(f"Error: Tiles directory '{tiles_dir}' not found.", file=sys.stderr)
        return 1

    if not model_dir.exists():
        print(f"Error: Model directory '{model_dir}' not found.", file=sys.stderr)
        return 1

    print("=================================================================")
    print("M3: IMAGE EMBEDDING GENERATION PIPELINE")
    print(f"Tiles Directory:   {tiles_dir}")
    print(f"Output Directory:  {output_dir}")
    print(f"Model Directory:   {model_dir}")
    print(f"Target Device:     {args.device}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Target Modality:   {args.sensor}")
    print("=================================================================\n")

    print("Loading embedding model offline...")
    t0_load = time.perf_counter()
    embedder = CLIPEmbedder(model_dir=model_dir, device=args.device, local_files_only=True)
    load_time = time.perf_counter() - t0_load
    print(f"Model loaded successfully in {load_time:.2f} s")
    print(f"Model: {embedder.model_metadata['model_name']} | Dim: {embedder.embedding_dim}\n")

    sensors_to_process = ["sentinel2", "sentinel1"] if args.sensor == "all" else [args.sensor]
    summary_results: List[Dict[str, Any]] = []
    total_tiles_embedded = 0
    total_bytes_written = 0

    t_start = time.perf_counter()

    for sensor in sensors_to_process:
        sensor_tiles_dir = tiles_dir / sensor
        if not sensor_tiles_dir.exists():
            continue

        sensor_out_dir = output_dir / sensor

        for scene_dir in sorted(sensor_tiles_dir.iterdir()):
            if scene_dir.is_dir() and (scene_dir / "tiles_manifest.json").exists():
                print(f"--- Processing Scene: {scene_dir.name} ---")
                scene_out = sensor_out_dir / scene_dir.name
                res = generate_scene_embeddings(
                    scene_tiles_dir=scene_dir,
                    scene_out_dir=scene_out,
                    embedder=embedder,
                    batch_size=args.batch_size,
                )
                summary_results.append(res)
                total_tiles_embedded += res["total_tiles"]
                total_bytes_written += res["file_size_bytes"]
                print(f"  -> Generated {res['total_tiles']} embeddings ({res['file_size_bytes'] / 1024:.1f} KB) in {res['duration_s']} s\n")

    total_duration = time.perf_counter() - t_start

    print("=================================================================")
    print("M3 EMBEDDING GENERATION COMPLETE")
    print(f"Total Scenes Processed:   {len(summary_results)}")
    print(f"Total Tiles Embedded:     {total_tiles_embedded}")
    print(f"Total Output Footprint:   {total_bytes_written / 1024:.2f} KB ({total_bytes_written / (1024*1024):.2f} MB)")
    print(f"Total Embedding Duration: {total_duration:.2f} s")
    print("=================================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
