"""CLI Script for False-Alarm Suppression and Confidence Estimation.

Executes end-to-end multi-temporal satellite change analysis:
- Resolves temporal counterpart pairs via M7 TemporalPairer
- Computes primary raw change magnitude via M8 ChangeDetector
- Post-processes change magnitudes with M9 FalseAlarmSuppressor:
    * Non-parametric robust noise estimation (median + MAD)
    * Adaptive candidate thresholding
    * 8-neighbor spatial count suppression
    * 8-connectivity minimum connected component area filtering
    * Deterministic heuristic confidence scoring
- Supports optional materialization of uint8 confirmed masks and float32 confidence GeoTIFFs
- Formats output as human-readable tables or machine-readable JSON
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
from src.change_detection import ChangeDetectionConfig, ChangeDetector
from src.suppression import SuppressionConfig, SuppressedChangeResult, FalseAlarmSuppressor


def print_single_suppressed_result_card(result: SuppressedChangeResult) -> None:
    """Print a single SuppressedChangeResult in a clean formatted ASCII summary."""
    sep = "=" * 96
    print(f"\n{sep}")
    print(f"CHANGE SUPPRESSION & CONFIDENCE RESULT: {result.pair_id}")
    print(f"Modality: {result.modality.upper()} | Grid Index: r{result.grid_index[0]:02d}_c{result.grid_index[1]:02d}")
    print(sep)

    print(f"{'Execution Status:':<24} {result.status}")
    print(f"{'Suppression Method:':<24} {result.method}")
    print(f"{'Temporal Separation:':<24} {result.temporal_delta_days:.2f} days")

    if result.status == "success":
        print("-" * 96)
        print("ROBUST NOISE STATISTICS & ADAPTIVE THRESHOLD:")
        print(f"  {'Noise Median:':<24} {result.noise_median:.6f}" if result.noise_median is not None else "  Noise Median: N/A")
        print(f"  {'Noise MAD:':<24} {result.noise_mad:.6f}" if result.noise_mad is not None else "  Noise MAD: N/A")
        if result.noise_mad is not None:
            print(f"  {'Robust Scale (1.4826*MAD):':<24} {1.4826 * result.noise_mad:.6f}")
        print(f"  {'Threshold Used:':<24} {result.threshold_used:.6f}" if result.threshold_used is not None else "  Threshold Used: N/A")

        print("-" * 96)
        print("SPATIAL SUPPRESSION & PIXEL COUNTS:")
        print(f"  {'Candidate Pixels:':<24} {result.candidate_pixels_count}")
        print(f"  {'Suppressed (False Alarms):':<24} {result.suppressed_pixels_count}")
        print(f"  {'Confirmed Pixels:':<24} {result.confirmed_pixels_count}")
        print(f"  {'Confirmed Change Ratio:':<24} {result.confirmed_change_ratio * 100:.3f}%")

        print("-" * 96)
        print("HEURISTIC CONFIDENCE (Confirmed Change Pixels):")
        mean_c_str = f"{result.mean_confidence_on_change:.4f}" if result.mean_confidence_on_change is not None else "N/A"
        max_c_str = f"{result.max_confidence:.4f}" if result.max_confidence is not None else "N/A"
        print(f"  {'Mean Confidence:':<24} {mean_c_str}")
        print(f"  {'Max Confidence:':<24} {max_c_str}")

    if result.mask_raster_path or result.confidence_raster_path:
        print("-" * 96)
        print("MATERIALIZED RASTERS:")
        if result.mask_raster_path:
            print(f"  {'Confirmed Mask (GeoTIFF):':<26} {result.mask_raster_path}")
        if result.confidence_raster_path:
            print(f"  {'Confidence Map (GeoTIFF):':<26} {result.confidence_raster_path}")

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


def print_suppressed_results_summary_table(
    results: List[SuppressedChangeResult],
    title: str = "CHANGE SUPPRESSION RESULTS",
) -> None:
    """Print a summary table of multiple change suppression results."""
    sep = "=" * 116
    print(f"\n{sep}")
    print(f"{title} (Total: {len(results)})")
    print(sep)

    if not results:
        print("No change suppression results to display.")
        print(sep + "\n")
        return

    header = (
        f"{'Grid':<8} {'Modality':<9} {'Status':<16} {'Median':<9} {'Thresh':<9} "
        f"{'Cand':<7} {'Suppr':<7} {'Conf':<7} {'Ratio %':<9} {'Mean Conf'}"
    )
    print(header)
    print("-" * 116)

    for r in results:
        grid_str = f"r{r.grid_index[0]:02d}_c{r.grid_index[1]:02d}"
        if r.status == "success":
            med_str = f"{r.noise_median:.4f}" if r.noise_median is not None else "N/A"
            thr_str = f"{r.threshold_used:.4f}" if r.threshold_used is not None else "N/A"
            cand_str = str(r.candidate_pixels_count)
            supp_str = str(r.suppressed_pixels_count)
            conf_str = str(r.confirmed_pixels_count)
            ratio_str = f"{r.confirmed_change_ratio * 100:.2f}%"
            mean_c_str = f"{r.mean_confidence_on_change:.3f}" if r.mean_confidence_on_change is not None else "N/A"
        else:
            med_str = "N/A"
            thr_str = "N/A"
            cand_str = "-"
            supp_str = "-"
            conf_str = "-"
            ratio_str = "0.0%"
            mean_c_str = "N/A"

        print(
            f"{grid_str:<8} {r.modality:<9} {r.status:<16} {med_str:<9} {thr_str:<9} "
            f"{cand_str:<7} {supp_str:<7} {conf_str:<7} {ratio_str:<9} {mean_c_str}"
        )

    print(sep + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run false-alarm suppression and confidence estimation on satellite change pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--tile-id",
        type=str,
        help="Query tile ID to resolve counterpart, detect change, and suppress false alarms.",
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

    # M9 Suppression parameters
    parser.add_argument(
        "-k", "--sensitivity-k",
        type=float,
        default=3.0,
        help="Sensitivity multiplier k for robust scale thresholding (default: 3.0).",
    )
    parser.add_argument(
        "--min-threshold-offset",
        type=float,
        default=0.0,
        help="Minimum offset added to median threshold when MAD is zero or tiny (default: 0.0).",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=2,
        help="Minimum 8-neighbor count required to retain a candidate pixel (default: 2).",
    )
    parser.add_argument(
        "--min-region-area",
        type=int,
        default=4,
        help="Minimum connected component area in pixels for 8-connectivity (default: 4).",
    )
    parser.add_argument(
        "--min-valid-pixels",
        type=int,
        default=100,
        help="Minimum valid pixels required for robust statistics (default: 100).",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save confirmed change mask and confidence map as GeoTIFFs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for saved GeoTIFFs (defaults to data/changes).",
    )

    # M8 Detection options
    parser.add_argument(
        "--save-change-map",
        action="store_true",
        help="Also save raw continuous change magnitude GeoTIFF from M8.",
    )
    parser.add_argument(
        "--no-clouds",
        action="store_true",
        help="Disable SCL cloud and shadow quality masking during raw change detection.",
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.10,
        help="Minimum valid pixel ratio required for M8 change statistics (default: 0.10).",
    )

    # Output formatting and environment
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

    # Initialize M7 Pairer, M8 Detector, and M9 Suppressor
    try:
        pairer = TemporalPairer(metadata_file=metadata_file, project_root=project_root)
        detector = ChangeDetector(project_root=project_root)
        suppressor = FalseAlarmSuppressor(project_root=project_root)
    except Exception as exc:
        print(f"Initialization error: {exc}", file=sys.stderr)
        return 1

    try:
        m8_config = ChangeDetectionConfig(
            mask_clouds=not args.no_clouds,
            minimum_valid_pixel_ratio=args.min_valid_ratio,
            save_change_map=args.save_change_map,
            output_dir=args.output_dir,
        )
        m9_config = SuppressionConfig(
            sensitivity_k=args.sensitivity_k,
            min_threshold_offset=args.min_threshold_offset,
            min_neighbors=args.min_neighbors,
            min_region_area=args.min_region_area,
            min_valid_pixels=args.min_valid_pixels,
            save_masks=args.save_masks,
            output_dir=args.output_dir,
        )
    except ValueError as val_err:
        print(f"Configuration error: {val_err}", file=sys.stderr)
        return 1

    # Mode 1: Single tile pairing, detection, and suppression
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
            raw_change = detector.detect_change(pair, config=m8_config)
            suppressed = suppressor.suppress(raw_change, config=m9_config)
        except Exception as exc:
            print(f"Error during change suppression: {exc}", file=sys.stderr)
            return 1

        if args.format == "json":
            print(json.dumps(suppressed.to_dict(), indent=2))
        else:
            print_single_suppressed_result_card(suppressed)

        return 0

    # Mode 2: Scene-to-scene change detection and suppression
    if args.scene_a and args.scene_b:
        try:
            pairs = pairer.find_pairs_for_scenes(scene_id_a=args.scene_a, scene_id_b=args.scene_b)
        except Exception as exc:
            print(f"Error during scene pairing: {exc}", file=sys.stderr)
            return 1

        raw_changes = list(detector.detect_changes(pairs, config=m8_config))
        suppressed_results = list(suppressor.suppress_batch(raw_changes, config=m9_config))

        if args.format == "json":
            out_data = {
                "scene_a": args.scene_a,
                "scene_b": args.scene_b,
                "total_results": len(suppressed_results),
                "results": [r.to_dict() for r in suppressed_results],
            }
            print(json.dumps(out_data, indent=2))
        else:
            title = f"SCENE CHANGE SUPPRESSION: {args.scene_a} <-> {args.scene_b}"
            print_suppressed_results_summary_table(suppressed_results, title=title)

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
