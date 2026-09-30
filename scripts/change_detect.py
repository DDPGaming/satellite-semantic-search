"""CLI Script for Primary Raw Change Detection.

Computes physical multi-temporal change detection across corresponding satellite observations:
- Detects change for a specific tile by resolving its canonical counterpart (M7 -> M8)
- Detects change across all corresponding tiles between two scenes
- Reports continuous spectral distance magnitude distribution statistics
- Supports optional materialization of continuous change GeoTIFF rasters
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

from src.pairing import TemporalPairer
from src.change_detection import ChangeDetectionConfig, ChangeDetector, ChangeResult


def print_single_result_card(result: ChangeResult) -> None:
    """Print a single ChangeResult in a clean formatted ASCII summary."""
    sep = "=" * 96
    print(f"\n{sep}")
    print(f"CHANGE DETECTION RESULT: {result.pair_id}")
    print(f"Modality: {result.modality.upper()} | Grid Index: r{result.grid_index[0]:02d}_c{result.grid_index[1]:02d}")
    print(sep)

    print(f"{'Execution Status:':<24} {result.status}")
    print(f"{'Detection Method:':<24} {result.method}")
    print(f"{'Temporal Separation:':<24} {result.temporal_delta_days:.2f} days")

    if result.summary_statistics is not None:
        stats = result.summary_statistics
        print("-" * 96)
        print("SUMMARY STATISTICS (Continuous Spectral Change Magnitude):")
        print(f"  {'Mean Change:':<22} {stats['mean']:.6f}")
        print(f"  {'Median Change:':<22} {stats['median']:.6f}")
        print(f"  {'95th Percentile:':<22} {stats['p95']:.6f}")
        print(f"  {'99th Percentile:':<22} {stats['p99']:.6f}")
        print(f"  {'Max Change:':<22} {stats['max']:.6f}")
        print(f"  {'Std Deviation:':<22} {stats['std']:.6f}")
        print(f"  {'Valid Pixels:':<22} {stats['valid_pixels']} / {stats['total_pixels']} ({stats['valid_pixel_ratio'] * 100:.2f}%)")

    if result.change_map_path:
        print("-" * 96)
        print(f"{'Saved Change Map:':<24} {result.change_map_path}")

    if result.warnings:
        print("-" * 96)
        print("WARNINGS / DIAGNOSTICS:")
        for w in result.warnings:
            print(f"  - {w}")

    print("-" * 96)
    print(f"{'REFERENCE (T0):':<18} {result.reference_tile_id}")
    print(f"{'  Scene:':<18} {result.reference_scene_id}")
    print(f"{'  Acquisition UTC:':<18} {result.reference_datetime_utc}")
    print("-" * 96)
    print(f"{'COMPARISON (T1):':<18} {result.comparison_tile_id}")
    print(f"{'  Scene:':<18} {result.comparison_scene_id}")
    print(f"{'  Acquisition UTC:':<18} {result.comparison_datetime_utc}")
    print(sep + "\n")


def print_results_summary_table(results: List[ChangeResult], title: str = "CHANGE DETECTION RESULTS") -> None:
    """Print a summary table of multiple change detection results."""
    sep = "=" * 108
    print(f"\n{sep}")
    print(f"{title} (Total: {len(results)})")
    print(sep)

    if not results:
        print("No change detection results to display.")
        print(sep + "\n")
        return

    header = f"{'Grid':<8} {'Modality':<9} {'Status':<16} {'Mean':<10} {'P95':<10} {'Valid %':<10} {'Ref Tile ID':<22} {'Comp Tile ID'}"
    print(header)
    print("-" * 108)

    for r in results:
        grid_str = f"r{r.grid_index[0]:02d}_c{r.grid_index[1]:02d}"
        if r.summary_statistics:
            mean_str = f"{r.summary_statistics['mean']:.4f}"
            p95_str = f"{r.summary_statistics['p95']:.4f}"
            valid_pct = f"{r.summary_statistics['valid_pixel_ratio'] * 100:.1f}%"
        else:
            mean_str = "N/A"
            p95_str = "N/A"
            valid_pct = "0.0%"

        ref_short = r.reference_tile_id[:20] + ".." if len(r.reference_tile_id) > 20 else r.reference_tile_id
        comp_short = r.comparison_tile_id[:20] + ".." if len(r.comparison_tile_id) > 20 else r.comparison_tile_id

        print(
            f"{grid_str:<8} {r.modality:<9} {r.status:<16} {mean_str:<10} {p95_str:<10} {valid_pct:<10} {ref_short:<22} {comp_short}"
        )

    print(sep + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run primary raw change detection on multi-temporal satellite tile pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--tile-id",
        type=str,
        help="Query tile ID to resolve counterpart and detect change (e.g., s2_S2A_43QBB_20220127_0_L2A_10m_r03_c04).",
    )
    group.add_argument(
        "--scene-a",
        type=str,
        help="Reference scene ID (must be paired with --scene-b).",
    )

    parser.add_argument(
        "--scene-b",
        type=str,
        default=None,
        help="Comparison scene ID (required when --scene-a is provided).",
    )
    parser.add_argument(
        "--save-map",
        action="store_true",
        help="Save continuous change magnitude raster as GeoTIFF.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for saved change GeoTIFFs (defaults to data/changes).",
    )
    parser.add_argument(
        "--no-clouds",
        action="store_true",
        help="Disable SCL cloud and shadow quality masking.",
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.10,
        help="Minimum valid pixel ratio required for statistics (default: 0.10).",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format: 'text' (default) or 'json'.",
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default=None,
        help="Custom path to data/index/metadata.json.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Custom project root directory.",
    )

    args = parser.parse_args()
    if args.scene_a and not args.scene_b:
        parser.error("--scene-b is required when --scene-a is provided.")

    return args


def main() -> int:
    """Main CLI execution flow."""
    args = parse_args()

    project_root = Path(args.project_root).resolve() if args.project_root else ROOT_DIR
    metadata_file = Path(args.metadata_file).resolve() if args.metadata_file else None

    # Initialize M7 Pairer and M8 Detector
    try:
        pairer = TemporalPairer(metadata_file=metadata_file, project_root=project_root)
        detector = ChangeDetector(project_root=project_root)
    except Exception as exc:
        print(f"Initialization error: {exc}", file=sys.stderr)
        return 1

    config = ChangeDetectionConfig(
        mask_clouds=not args.no_clouds,
        minimum_valid_pixel_ratio=args.min_valid_ratio,
        save_change_map=args.save_map,
        output_dir=args.output_dir,
    )

    # Mode 1: Single tile pairing and change detection
    if args.tile_id:
        try:
            pair = pairer.get_pair_for_tile(args.tile_id)
        except Exception as exc:
            print(f"Error during tile pairing: {exc}", file=sys.stderr)
            return 1

        if pair is None:
            if args.format == "json":
                print(json.dumps({"tile_id": args.tile_id, "result": None, "message": "No temporal counterpart found."}))
            else:
                print(f"\nNo temporal counterpart found for tile: '{args.tile_id}'")
            return 0

        try:
            result = detector.detect_change(pair, config=config)
        except Exception as exc:
            print(f"Error during change detection: {exc}", file=sys.stderr)
            return 1

        if args.format == "json":
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print_single_result_card(result)

        return 0

    # Mode 2: Scene-to-scene change detection
    if args.scene_a and args.scene_b:
        try:
            pairs = pairer.find_pairs_for_scenes(scene_id_a=args.scene_a, scene_id_b=args.scene_b)
        except Exception as exc:
            print(f"Error during scene pairing: {exc}", file=sys.stderr)
            return 1

        results = list(detector.detect_changes(pairs, config=config))

        if args.format == "json":
            out_data = {
                "scene_a": args.scene_a,
                "scene_b": args.scene_b,
                "total_results": len(results),
                "results": [r.to_dict() for r in results],
            }
            print(json.dumps(out_data, indent=2))
        else:
            title = f"SCENE CHANGE DETECTION: {args.scene_a} <-> {args.scene_b}"
            print_results_summary_table(results, title=title)

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
