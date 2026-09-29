"""CLI Script to Build, Inspect, and Query Local FAISS Vector Index.

Builds a deterministic, exact FAISS IndexFlatIP index over all M3 tile embeddings:
- Reads existing embeddings from data/embeddings/ without regenerating
- Preserves 1-to-1 linkage between FAISS row IDs and M2 tile metadata
- Saves index binary to data/index/faiss.index and mapping to data/index/metadata.json
- Supports inspecting index statistics and executing nearest-neighbor tile queries
"""

import argparse
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

from src.retrieval.builder import build_index_from_embeddings
from src.retrieval.index import VectorIndex


def get_default_embeddings_dir() -> Path:
    return ROOT_DIR / "data" / "embeddings"


def get_default_tiles_dir() -> Path:
    return ROOT_DIR / "data" / "tiles"


def get_default_index_file() -> Path:
    return ROOT_DIR / "data" / "index" / "faiss.index"


def get_default_metadata_file() -> Path:
    return ROOT_DIR / "data" / "index" / "metadata.json"


def inspect_index(index_file: Path, metadata_file: Path) -> None:
    """Print detailed inspection statistics for a persisted FAISS index."""
    if not index_file.exists() or not metadata_file.exists():
        print(f"Error: Persisted index files not found.\n  Index: {index_file}\n  Metadata: {metadata_file}", file=sys.stderr)
        sys.exit(1)

    index = VectorIndex.load(index_file, metadata_file)
    idx_size_kb = index_file.stat().st_size / 1024.0
    meta_size_kb = metadata_file.stat().st_size / 1024.0

    # Count by modality and scene
    modality_counts: Dict[str, int] = {}
    scene_counts: Dict[str, int] = {}
    for entry in index.metadata_entries:
        mod = entry.get("modality", "unknown")
        scene = entry.get("scene_id", "unknown")
        modality_counts[mod] = modality_counts.get(mod, 0) + 1
        scene_counts[scene] = scene_counts.get(scene, 0) + 1

    sep = "=" * 65
    print(f"\n{sep}")
    print("FAISS VECTOR INDEX INSPECTION REPORT")
    print(sep)
    print(f"  Index File:          {index_file} ({idx_size_kb:.1f} KB)")
    print(f"  Metadata File:       {metadata_file} ({meta_size_kb:.1f} KB)")
    print(f"  Index Type:          IndexFlatIP (Exact Inner Product)")
    print(f"  Metric:              Cosine Similarity (L2-normalized vectors)")
    print(f"  Embedding Dimension: {index.dimension}")
    print(f"  Total Vectors:       {index.total_vectors}")
    print("\n  Modality Breakdown:")
    for mod, count in sorted(modality_counts.items()):
        print(f"    - {mod:<12}: {count} tiles")
    print("\n  Scene Breakdown:")
    for scene, count in sorted(scene_counts.items()):
        print(f"    - {scene}: {count} tiles")
    print(f"{sep}\n")


def search_by_tile_id(
    index_file: Path,
    metadata_file: Path,
    embeddings_dir: Path,
    target_tile_id: str,
    top_k: int = 5,
) -> None:
    """Find nearest-neighbor tiles for a specific existing tile ID."""
    index = VectorIndex.load(index_file, metadata_file)
    target_meta = index.get_metadata_by_tile_id(target_tile_id)
    if not target_meta:
        print(f"Error: Tile ID '{target_tile_id}' not found in index.", file=sys.stderr)
        sys.exit(1)

    row_id = target_meta["row_id"]

    # Reconstruct query vector directly from FAISS index or embedding file
    query_vector = index.index.reconstruct(row_id)

    hits = index.search(query_vector, top_k=top_k)

    sep = "=" * 80
    print(f"\n{sep}")
    print(f"NEAREST NEIGHBOR SEARCH RESULTS FOR TILE: {target_tile_id}")
    print(f"Modality: {target_meta.get('modality')} | Scene: {target_meta.get('scene_id')}")
    print(sep)
    print(f"{'Rank':<6} {'Score':<10} {'Modality':<10} {'Tile ID':<42} {'Scene'}")
    print("-" * 80)
    for h in hits:
        print(
            f"{h['rank']:<6} {h['similarity_score']:<10.4f} {str(h['modality']):<10} "
            f"{h['tile_id']:<42} {str(h['scene_id'])[:15]}..."
        )
    print(f"{sep}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build, inspect, or query local FAISS vector search index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=get_default_embeddings_dir(),
        help="Root directory containing M3 embeddings",
    )
    parser.add_argument(
        "--tiles-dir",
        type=Path,
        default=get_default_tiles_dir(),
        help="Root directory containing M2 tiles",
    )
    parser.add_argument(
        "--index-file",
        type=Path,
        default=get_default_index_file(),
        help="Destination path for faiss.index",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=get_default_metadata_file(),
        help="Destination path for index metadata.json",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect existing index and print statistics",
    )
    parser.add_argument(
        "--search-tile",
        type=str,
        default=None,
        help="Run nearest-neighbor similarity search for an existing tile ID",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest neighbors to retrieve",
    )
    args = parser.parse_args()

    index_file = args.index_file.resolve()
    metadata_file = args.metadata_file.resolve()
    embeddings_dir = args.embeddings_dir.resolve()
    tiles_dir = args.tiles_dir.resolve()

    if args.inspect:
        inspect_index(index_file, metadata_file)
        return 0

    if args.search_tile:
        search_by_tile_id(index_file, metadata_file, embeddings_dir, args.search_tile, top_k=args.top_k)
        return 0

    # Build and persist the index
    print("=================================================================")
    print("M4: LOCAL VECTOR SEARCH / FAISS INDEXING PIPELINE")
    print(f"Embeddings Directory: {embeddings_dir}")
    print(f"Tiles Directory:      {tiles_dir}")
    print(f"Index Output File:    {index_file}")
    print(f"Metadata Output File: {metadata_file}")
    print("=================================================================\n")

    t0 = time.perf_counter()
    print("Building FAISS IndexFlatIP from staged embeddings...")
    index = build_index_from_embeddings(embeddings_dir, tiles_dir, dimension=512)
    build_time = time.perf_counter() - t0
    print(f"Index constructed in {build_time:.3f} s: {index.total_vectors} vectors (dim={index.dimension})")

    t1 = time.perf_counter()
    print("Persisting index and metadata to disk...")
    index.save(index_file, metadata_file)
    save_time = time.perf_counter() - t1
    print(f"Saved successfully in {save_time:.3f} s")

    idx_size_kb = index_file.stat().st_size / 1024.0
    meta_size_kb = metadata_file.stat().st_size / 1024.0

    print("\n=================================================================")
    print("M4 FAISS INDEXING COMPLETE")
    print(f"Total Vectors Indexed:  {index.total_vectors}")
    print(f"Vector Dimensionality:  {index.dimension}")
    print(f"FAISS Index Binary Size:{idx_size_kb:.1f} KB")
    print(f"Metadata JSON Size:     {meta_size_kb:.1f} KB")
    print(f"Total Disk Footprint:   {(idx_size_kb + meta_size_kb):.1f} KB")
    print("=================================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
