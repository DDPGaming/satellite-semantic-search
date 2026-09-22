"""Deterministic Spatial Tiling Script for M2.

Tiles all locally staged M1 Sentinel-2 and Sentinel-1 scenes into `data/tiles/`
using the approved M2 deterministic spatial tiling rules:
- Native resolutions preserved (10m optical, 20m SCL, 10m radar spacing; zero resampling)
- Deterministic 256x256 pixel windows with 'shift' boundary strategy
- Complete provenance and georeferencing preserved
- Verification that M1 source rasters remain completely unmodified
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.sentinel1 import compute_file_sha256
from src.tiling.sentinel1_tiler import tile_sentinel1_scene
from src.tiling.sentinel2_tiler import tile_sentinel2_scene


def verify_m1_source_unmodified(scene_dir: Path) -> bool:
    """Verify that M1 source rasters and metadata have not been modified."""
    meta_path = scene_dir / "metadata.json"
    if not meta_path.exists():
        return False
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    for band_key, bmeta in meta.get("bands", {}).items():
        fname = bmeta.get("local_filename")
        if fname:
            fpath = scene_dir / fname
            if not fpath.exists():
                return False
            expected_sha = bmeta.get("local_sha256")
            if expected_sha:
                current_sha = compute_file_sha256(fpath)
                if current_sha != expected_sha:
                    return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic spatial tiling for M2.")
    parser.add_argument(
        "--s2-dir",
        type=Path,
        default=ROOT_DIR / "data" / "raw" / "sentinel2",
        help="Directory containing staged Sentinel-2 scenes",
    )
    parser.add_argument(
        "--s1-dir",
        type=Path,
        default=ROOT_DIR / "data" / "raw" / "sentinel1",
        help="Directory containing staged Sentinel-1 scenes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "data" / "tiles",
        help="Base directory for generated tiles",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Tile window dimension in pixels (default: 256)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Tile stride in pixels (default: None -> equals tile_size)",
    )
    parser.add_argument(
        "--boundary-strategy",
        type=str,
        default="shift",
        choices=["shift", "truncate", "pad"],
        help="Boundary handling strategy (default: shift)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    s2_out = output_dir / "sentinel2"
    s1_out = output_dir / "sentinel1"

    print("=================================================================")
    print("M2: DETERMINISTIC SPATIAL TILING & METADATA GENERATION")
    print(f"Tile Size:          {args.tile_size} x {args.tile_size} px")
    print(f"Stride:             {args.stride or args.tile_size} px")
    print(f"Boundary Strategy:  {args.boundary_strategy}")
    print("Boundary Policy:    stride=256 used; shift may introduce overlap only at terminal boundary")
    print("                    tiles to maintain fixed 256x256 dimensions without synthetic padding.")
    print(f"Output Directory:   {output_dir}")
    print("=================================================================")

    t0 = time.perf_counter()
    summary = {
        "sentinel2": {},
        "sentinel1": {},
        "total_tiles": 0,
        "total_bytes": 0,
    }

    # 1. Tile Sentinel-2 Scenes
    s2_dir = args.s2_dir.resolve()
    if s2_dir.exists():
        for scene_path in sorted(s2_dir.iterdir()):
            if scene_path.is_dir() and (scene_path / "metadata.json").exists():
                print(f"\n--- Tiling Sentinel-2 Scene: {scene_path.name} ---")
                res = tile_sentinel2_scene(
                    scene_dir=scene_path,
                    output_dir=s2_out,
                    tile_size=args.tile_size,
                    stride=args.stride,
                    boundary_strategy=args.boundary_strategy,
                )
                summary["sentinel2"][scene_path.name] = {
                    "total_tiles": res["total_tiles"],
                    "total_bytes": res["total_bytes"],
                    "manifest": str(res["manifest_path"]),
                }
                summary["total_tiles"] += res["total_tiles"]
                summary["total_bytes"] += res["total_bytes"]
                print(f"  -> Generated {res['total_tiles']} tiles ({res['total_bytes'] / (1024*1024):.2f} MB)")

    # 2. Tile Sentinel-1 Scenes
    s1_dir = args.s1_dir.resolve()
    if s1_dir.exists():
        for scene_path in sorted(s1_dir.iterdir()):
            if scene_path.is_dir() and (scene_path / "metadata.json").exists():
                print(f"\n--- Tiling Sentinel-1 Scene: {scene_path.name} ---")
                res = tile_sentinel1_scene(
                    scene_dir=scene_path,
                    output_dir=s1_out,
                    tile_size=args.tile_size,
                    stride=args.stride,
                    boundary_strategy=args.boundary_strategy,
                )
                summary["sentinel1"][scene_path.name] = {
                    "total_tiles": res["total_tiles"],
                    "total_bytes": res["total_bytes"],
                    "manifest": str(res["manifest_path"]),
                }
                summary["total_tiles"] += res["total_tiles"]
                summary["total_bytes"] += res["total_bytes"]
                print(f"  -> Generated {res['total_tiles']} tiles ({res['total_bytes'] / (1024*1024):.2f} MB)")

    elapsed = time.perf_counter() - t0

    # 3. Verify M1 Source Rasters remain unmodified
    print("\n--- Verifying M1 Source Ingestion Integrity ---")
    all_unmodified = True
    if s2_dir.exists():
        for sp in sorted(s2_dir.iterdir()):
            if sp.is_dir() and (sp / "metadata.json").exists():
                ok = verify_m1_source_unmodified(sp)
                print(f"  Sentinel-2 {sp.name}: Source SHA-256 Unmodified = {ok}")
                all_unmodified = all_unmodified and ok

    if s1_dir.exists():
        for sp in sorted(s1_dir.iterdir()):
            if sp.is_dir() and (sp / "metadata.json").exists():
                ok = verify_m1_source_unmodified(sp)
                print(f"  Sentinel-1 {sp.name}: Source SHA-256 Unmodified = {ok}")
                all_unmodified = all_unmodified and ok

    print("\n=================================================================")
    print("M2 TILING COMPLETE")
    print(f"Total Tiles Generated: {summary['total_tiles']}")
    print(f"Total Disk Footprint:  {summary['total_bytes'] / (1024*1024):.2f} MB")
    print(f"Execution Time:        {elapsed:.2f} s")
    print(f"M1 Data Integrity:     {'ALL UNMODIFIED (VERIFIED)' if all_unmodified else 'INTEGRITY MISMATCH'}")
    print("=================================================================\n")

    return 0 if all_unmodified else 1


if __name__ == "__main__":
    sys.exit(main())
