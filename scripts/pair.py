"""CLI Script for Multi-Temporal Pairing and Spatial Alignment.

Resolves corresponding satellite observations across time:
- Pairs specific tiles to their canonical temporal counterparts
- Pairs all corresponding tiles between two scenes
- Reports verified pixel alignment status and spatial IoU diagnostics
- Formats results as human-readable tables or machine-readable JSON
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.pairing import TemporalPair, TemporalPairer


def get_default_metadata_file() -> Path:
    return ROOT_DIR / "data" / "index" / "metadata.json"


def print_single_pair_table(pair: TemporalPair) -> None:
    """Print a single TemporalPair in a clean formatted ASCII summary."""
    sep = "=" * 96
    print(f"\n{sep}")
    print(f"TEMPORAL PAIR: {pair.pair_id}")
    print(f"Modality: {pair.modality.upper()} | Grid Index: r{pair.grid_index[0]:02d}_c{pair.grid_index[1]:02d}")
    print(sep)

    aligned_str = "YES (native_pixel_aligned)" if pair.is_pixel_aligned else f"NO ({pair.alignment_type})"
    crs_str = pair.alignment.crs if pair.alignment.crs else "None (GCP referenced)"

    print(f"{'Temporal Separation:':<24} {pair.temporal_delta_days:.2f} days")
    print(f"{'Pixel Aligned:':<24} {aligned_str}")
    print(f"{'Raster CRS:':<24} {crs_str}")
    print(f"{'Spatial IoU (WGS84):':<24} {pair.spatial_iou_wgs84:.6f}")
    print("-" * 96)
    print(f"{'REFERENCE (T0):':<18} {pair.reference_tile_id}")
    print(f"{'  Scene:':<18} {pair.reference_scene_id}")
    print(f"{'  Acquisition UTC:':<18} {pair.reference_datetime_utc}")
    print(f"{'  Directory:':<18} {pair.reference_tile_dir}")
    print("-" * 96)
    print(f"{'COMPARISON (T1):':<18} {pair.comparison_tile_id}")
    print(f"{'  Scene:':<18} {pair.comparison_scene_id}")
    print(f"{'  Acquisition UTC:':<18} {pair.comparison_datetime_utc}")
    print(f"{'  Directory:':<18} {pair.comparison_tile_dir}")
    print(sep + "\n")


def print_pairs_summary_table(pairs: List[TemporalPair], title: str = "TEMPORAL PAIRS") -> None:
    """Print a summary table of multiple temporal pairs."""
    sep = "=" * 104
    print(f"\n{sep}")
    print(f"{title} (Total: {len(pairs)})")
    print(sep)

    if not pairs:
        print("No matching temporal pairs found.")
        print(sep + "\n")
        return

    header = f"{'Grid':<8} {'Modality':<9} {'Delta (Days)':<14} {'Aligned?':<12} {'IoU':<9} {'Ref Tile ID':<26} {'Comp Tile ID'}"
    print(header)
    print("-" * 104)

    for p in pairs:
        grid_str = f"r{p.grid_index[0]:02d}_c{p.grid_index[1]:02d}"
        aligned = "YES" if p.is_pixel_aligned else "NO"
        ref_short = p.reference_tile_id[:24] + ".." if len(p.reference_tile_id) > 24 else p.reference_tile_id
        comp_short = p.comparison_tile_id[:24] + ".." if len(p.comparison_tile_id) > 24 else p.comparison_tile_id

        print(
            f"{grid_str:<8} {p.modality:<9} {p.temporal_delta_days:<14.2f} {aligned:<12} {p.spatial_iou_wgs84:<9.4f} {ref_short:<26} {comp_short}"
        )

    print(sep + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve multi-temporal pairing and spatial alignment across satellite observations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--tile-id",
        "-t",
        type=str,
        help="Resolve temporal counterpart for a specific tile identifier",
    )
    group.add_argument(
        "--scene-a",
        type=str,
        help="Reference scene identifier (must be paired with --scene-b)",
    )
    group.add_argument(
        "--modality",
        "-m",
        type=str,
        choices=["optical", "sar"],
        help="List all pairs for a specific modality",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Enumerate all temporal pairs across all modalities",
    )

    parser.add_argument(
        "--scene-b",
        type=str,
        default=None,
        help="Comparison scene identifier (required when --scene-a is provided)",
    )
    parser.add_argument(
        "--target-scene",
        type=str,
        default=None,
        help="Restrict tile counterpart candidate search to this specific scene",
    )
    parser.add_argument(
        "--min-days",
        type=float,
        default=0.0,
        help="Minimum required temporal delta in days",
    )
    parser.add_argument(
        "--max-days",
        type=float,
        default=None,
        help="Maximum allowable temporal delta in days",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="table",
        choices=["table", "json"],
        help="Output presentation format",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=get_default_metadata_file(),
        help="Path to index metadata JSON file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.scene_a and not args.scene_b:
        print("Error: --scene-b is required when --scene-a is specified.", file=sys.stderr)
        return 1

    try:
        pairer = TemporalPairer(metadata_file=args.metadata_file)

        if args.tile_id:
            pair = pairer.get_pair_for_tile(
                tile_id=args.tile_id,
                target_scene_id=args.target_scene,
                min_temporal_delta_days=args.min_days,
                max_temporal_delta_days=args.max_days,
            )
            if pair is None:
                if args.format == "json":
                    print(json.dumps({"tile_id": args.tile_id, "pair": None, "message": "No counterpart found"}, indent=2))
                else:
                    print(f"\nNo valid temporal counterpart found for tile: {args.tile_id}\n")
                return 0

            if args.format == "json":
                print(json.dumps(pair.to_dict(), indent=2))
            else:
                print_single_pair_table(pair)
            return 0

        elif args.scene_a and args.scene_b:
            pairs = pairer.find_pairs_for_scenes(scene_id_a=args.scene_a, scene_id_b=args.scene_b)
            if args.format == "json":
                payload = {
                    "scene_a": args.scene_a,
                    "scene_b": args.scene_b,
                    "total_pairs": len(pairs),
                    "pairs": [p.to_dict() for p in pairs],
                }
                print(json.dumps(payload, indent=2))
            else:
                print_pairs_summary_table(pairs, title=f"SCENE PAIR: {args.scene_a} <-> {args.scene_b}")
            return 0

        elif args.modality or args.all:
            target_mod = args.modality if args.modality else None
            pairs = pairer.find_all_pairs(modality=target_mod)
            if args.format == "json":
                payload = {
                    "modality": target_mod or "all",
                    "total_pairs": len(pairs),
                    "pairs": [p.to_dict() for p in pairs],
                }
                print(json.dumps(payload, indent=2))
            else:
                title = f"ALL TEMPORAL PAIRS ({target_mod.upper() if target_mod else 'ALL'})"
                print_pairs_summary_table(pairs, title=title)
            return 0

        return 0

    except Exception as e:
        print(f"Error during pairing: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
