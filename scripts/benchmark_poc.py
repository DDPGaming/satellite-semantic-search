"""End-to-End PoC Functional Benchmark CLI (M5 -> M6 -> M7 -> M8 -> M9).

Exercises the complete multi-temporal satellite change analysis pipeline:
  Natural-language query
    -> M5: Semantic Retrieval
    -> M6: Metadata Filtering
    -> M7: Temporal Pairing
    -> M8: Optical Change Detection
    -> M9: False-Alarm Suppression & Confidence Estimation

Measures actual wall-clock elapsed timings for each stage and end-to-end.
Evaluates deterministic execution across consecutive query evaluations.
Saves a standardized machine-readable JSON baseline report to data/evaluation/poc_benchmark.json.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import PoCPipeline, PoCPipelineResult

DEFAULT_QUERIES = [
    "urban development around Navi Mumbai",
    "built-up area around Navi Mumbai",
    "vegetation change around Navi Mumbai",
]


def get_git_commit_hash(project_root: Path) -> str:
    """Retrieve current Git commit hash or return placeholder if unavailable."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def verify_determinism(run1: PoCPipelineResult, run2: PoCPipelineResult) -> Tuple[bool, List[str]]:
    """
    Verify determinism between two consecutive pipeline executions on the same query.

    Checks:
    - Same status
    - Same retrieved tile IDs and ordering
    - Same similarity scores within numerical tolerance (1e-5)
    - Same filtered tile IDs
    - Same resolved temporal pair ID
    - Same M8 summary statistics
    - Same M9 candidate, suppressed, and confirmed pixel counts
    - Same M9 threshold and heuristic confidence summary
    """
    failures = []

    # 1. Status check
    if run1.status != run2.status:
        failures.append(f"Status mismatch: run1='{run1.status}' vs run2='{run2.status}'")

    # 2. Retrieval hits check
    r1_ids = [h["tile_id"] for h in run1.retrieval_hits]
    r2_ids = [h["tile_id"] for h in run2.retrieval_hits]
    if r1_ids != r2_ids:
        failures.append(f"Retrieval tile IDs mismatch: run1={r1_ids} vs run2={r2_ids}")

    for i, (h1, h2) in enumerate(zip(run1.retrieval_hits, run2.retrieval_hits)):
        s1, s2 = float(h1["similarity_score"]), float(h2["similarity_score"])
        if abs(s1 - s2) > 1e-5:
            failures.append(f"Retrieval score mismatch at rank {i+1}: {s1} vs {s2}")

    # 3. Filtered hits check
    f1_ids = [h["tile_id"] for h in run1.filtered_hits]
    f2_ids = [h["tile_id"] for h in run2.filtered_hits]
    if f1_ids != f2_ids:
        failures.append(f"Filtered tile IDs mismatch: run1={f1_ids} vs run2={f2_ids}")

    # 4. Temporal pair check
    p1_id = run1.temporal_pair.pair_id if run1.temporal_pair else None
    p2_id = run2.temporal_pair.pair_id if run2.temporal_pair else None
    if p1_id != p2_id:
        failures.append(f"Temporal pair mismatch: run1='{p1_id}' vs run2='{p2_id}'")

    # 5. M8 Summary statistics check
    if run1.change_result and run2.change_result:
        s1 = run1.change_result.summary_statistics or {}
        s2 = run2.change_result.summary_statistics or {}
        for key in ("mean", "median", "p95", "valid_pixels"):
            v1, v2 = s1.get(key), s2.get(key)
            if v1 != v2:
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    if abs(v1 - v2) > 1e-5:
                        failures.append(f"M8 statistic '{key}' mismatch: {v1} vs {v2}")
                else:
                    failures.append(f"M8 statistic '{key}' mismatch: {v1} vs {v2}")

    # 6. M9 Suppression counts and confidence check
    if run1.suppressed_result and run2.suppressed_result:
        m9_1 = run1.suppressed_result
        m9_2 = run2.suppressed_result
        if m9_1.candidate_pixels_count != m9_2.candidate_pixels_count:
            failures.append(f"Candidate count mismatch: {m9_1.candidate_pixels_count} vs {m9_2.candidate_pixels_count}")
        if m9_1.suppressed_pixels_count != m9_2.suppressed_pixels_count:
            failures.append(f"Suppressed count mismatch: {m9_1.suppressed_pixels_count} vs {m9_2.suppressed_pixels_count}")
        if m9_1.confirmed_pixels_count != m9_2.confirmed_pixels_count:
            failures.append(f"Confirmed count mismatch: {m9_1.confirmed_pixels_count} vs {m9_2.confirmed_pixels_count}")

        thr1 = m9_1.threshold_used
        thr2 = m9_2.threshold_used
        if thr1 is not None and thr2 is not None and abs(thr1 - thr2) > 1e-5:
            failures.append(f"Threshold mismatch: {thr1} vs {thr2}")

        c1 = m9_1.mean_confidence_on_change
        c2 = m9_2.mean_confidence_on_change
        if c1 is not None and c2 is not None and abs(c1 - c2) > 1e-5:
            failures.append(f"Mean confidence mismatch: {c1} vs {c2}")

    is_deterministic = len(failures) == 0
    return is_deterministic, failures


def print_benchmark_table(query_data_list: List[Dict[str, Any]]) -> None:
    """Print human-readable summary table for benchmark test queries."""
    sep = "=" * 120
    print(f"\n{sep}")
    print("END-TO-END PoC FUNCTIONAL BENCHMARK (M5 -> M6 -> M7 -> M8 -> M9)")
    print(sep)

    header = (
        f"{'Query':<36} {'Retrieval':<10} {'Pair ID':<22} {'M8 Stat':<9} {'M9 Conf':<9} "
        f"{'Ratio %':<9} {'Run 1 E2E':<11} {'Repeat E2E'}"
    )
    print(header)
    print("-" * 120)

    for item in query_data_list:
        q_short = item["query"][:34] + ".." if len(item["query"]) > 34 else item["query"]
        top_hit = item["retrieval"]["top_tile_id"]
        top_short = top_hit[:8] + ".." if top_hit else "None"
        pair_id = item["temporal_pair"]["pair_id"] if item["temporal_pair"] else "None"
        pair_short = pair_id.replace("pair__", "")[:20] + ".." if len(pair_id) > 22 else pair_id

        m8_stat = item["change_detection"]["status"] if item["change_detection"] else "N/A"
        m9_conf = str(item["suppression"]["confirmed_pixels"]) if item["suppression"] else "N/A"
        ratio_pct = f"{item['suppression']['confirmed_change_ratio'] * 100:.2f}%" if item["suppression"] else "0.0%"

        t1_e2e = f"{item['first_run']['timing']['total_s']:.3f} s"
        t2_e2e = f"{item['repeat_run']['timing']['total_s']:.3f} s"

        print(
            f"{q_short:<36} {top_short:<10} {pair_short:<22} {m8_stat:<9} {m9_conf:<9} "
            f"{ratio_pct:<9} {t1_e2e:<11} {t2_e2e}"
        )

    print(sep)


def print_aggregate_timing(aggregate: Dict[str, Any]) -> None:
    """Print aggregate pipeline timing statistics."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("PIPELINE TIMING BASELINE (Mean Wall-Clock Elapsed Time)")
    print(sep)
    print(f"  {'Mean First-Run Total (Cold):':<36} {aggregate['mean_first_run_total_s']:.4f} s")
    print(f"  {'Mean Repeat-Run Total (Warm):':<36} {aggregate['mean_repeat_run_total_s']:.4f} s")
    print("-" * 72)
    print("  Per-Stage Breakdown (Mean Across All Runs):")
    print(f"    {'M5 Semantic Retrieval:':<34} {aggregate['mean_retrieval_s']:.4f} s")
    print(f"    {'M6 Metadata Filtering:':<34} {aggregate['mean_filtering_s']:.4f} s")
    print(f"    {'M7 Temporal Pairing:':<34} {aggregate['mean_pairing_s']:.4f} s")
    print(f"    {'M8 Optical Change Detection:':<34} {aggregate['mean_change_detection_s']:.4f} s")
    print(f"    {'M9 False-Alarm Suppression:':<34} {aggregate['mean_suppression_s']:.4f} s")
    print(sep)
    print("Note: Wall-clock timing is environment-dependent and serves as a functional PoC baseline.\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for benchmark."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end PoC functional pipeline benchmark (M5 -> M6 -> M7 -> M8 -> M9).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Custom query to benchmark (can be specified multiple times; defaults to Navi Mumbai set).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save machine-readable JSON benchmark report (defaults to data/evaluation/poc_benchmark.json).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top candidates to retrieve per query (default: 5).",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="optical",
        help="Metadata filter for modality (default: optical).",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Custom project root directory.",
    )

    return parser.parse_args()


def run_benchmark(
    queries: List[str],
    output_path: Optional[Path] = None,
    top_k: int = 5,
    modality: Optional[str] = "optical",
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute the benchmark programmatically across queries.

    Returns the complete structured benchmark dictionary.
    """
    root = project_root if project_root else ROOT_DIR
    commit_hash = get_git_commit_hash(root)
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    print("\nInitializing End-to-End PoC Pipeline (loading models and indexes)...")
    t_init_start = time.perf_counter()
    pipeline = PoCPipeline(project_root=root)
    init_duration_s = time.perf_counter() - t_init_start
    print(f"Pipeline initialized in {init_duration_s:.3f} s.\n")

    query_results = []
    overall_determinism = True

    first_run_totals = []
    repeat_run_totals = []
    all_retrievals = []
    all_filterings = []
    all_pairings = []
    all_changes = []
    all_suppressions = []

    for q_idx, query in enumerate(queries, start=1):
        print(f"Evaluating query [{q_idx}/{len(queries)}]: \"{query}\"")

        # Run 1: First Run (Cold)
        res1 = pipeline.run(query=query, top_k=top_k, modality=modality)
        first_run_totals.append(res1.timing.total_s if res1.timing else 0.0)

        # Run 2: Repeat Run (Warm)
        res2 = pipeline.run(query=query, top_k=top_k, modality=modality)
        repeat_run_totals.append(res2.timing.total_s if res2.timing else 0.0)

        # Collect stage timings
        for res in (res1, res2):
            if res.timing:
                all_retrievals.append(res.timing.retrieval_s)
                all_filterings.append(res.timing.filtering_s)
                all_pairings.append(res.timing.pairing_s)
                all_changes.append(res.timing.change_detection_s)
                all_suppressions.append(res.timing.suppression_s)

        # Verify determinism
        is_det, failures = verify_determinism(res1, res2)
        if not is_det:
            overall_determinism = False

        query_item = {
            "query": query,
            "determinism": "PASS" if is_det else "FAIL",
            "determinism_issues": failures,
            "first_run": {
                "status": res1.status,
                "timing": res1.timing.to_dict() if res1.timing else None,
            },
            "repeat_run": {
                "status": res2.status,
                "timing": res2.timing.to_dict() if res2.timing else None,
            },
            "retrieval": {
                "top_tile_id": res1.selected_tile_id,
                "hits_count": len(res1.retrieval_hits),
                "top_k_hits": [
                    {
                        "rank": i + 1,
                        "tile_id": h["tile_id"],
                        "similarity_score": round(float(h["similarity_score"]), 6),
                        "scene_id": h["scene_id"],
                        "modality": h["modality"],
                    }
                    for i, h in enumerate(res1.retrieval_hits[:top_k])
                ],
            },
            "temporal_pair": {
                "pair_id": res1.temporal_pair.pair_id if res1.temporal_pair else None,
                "reference_tile_id": res1.temporal_pair.reference_tile_id if res1.temporal_pair else None,
                "comparison_tile_id": res1.temporal_pair.comparison_tile_id if res1.temporal_pair else None,
                "reference_scene_id": res1.temporal_pair.reference_scene_id if res1.temporal_pair else None,
                "comparison_scene_id": res1.temporal_pair.comparison_scene_id if res1.temporal_pair else None,
                "temporal_delta_days": round(res1.temporal_pair.temporal_delta_days, 5) if res1.temporal_pair else None,
                "alignment_type": res1.temporal_pair.alignment_type if res1.temporal_pair else None,
                "is_pixel_aligned": res1.temporal_pair.is_pixel_aligned if res1.temporal_pair else None,
            } if res1.temporal_pair else None,
            "change_detection": {
                "status": res1.change_result.status if res1.change_result else None,
                "method": res1.change_result.method if res1.change_result else None,
                "summary_statistics": res1.change_result.summary_statistics if res1.change_result else None,
            } if res1.change_result else None,
            "suppression": {
                "status": res1.suppressed_result.status if res1.suppressed_result else None,
                "method": res1.suppressed_result.method if res1.suppressed_result else None,
                "threshold_used": res1.suppressed_result.threshold_used if res1.suppressed_result else None,
                "candidate_pixels": res1.suppressed_result.candidate_pixels_count if res1.suppressed_result else 0,
                "suppressed_pixels": res1.suppressed_result.suppressed_pixels_count if res1.suppressed_result else 0,
                "confirmed_pixels": res1.suppressed_result.confirmed_pixels_count if res1.suppressed_result else 0,
                "confirmed_change_ratio": res1.suppressed_result.confirmed_change_ratio if res1.suppressed_result else 0.0,
                "mean_confidence_on_change": res1.suppressed_result.mean_confidence_on_change if res1.suppressed_result else None,
                "max_confidence": res1.suppressed_result.max_confidence if res1.suppressed_result else None,
            } if res1.suppressed_result else None,
        }
        query_results.append(query_item)

    aggregate_timing = {
        "mean_first_run_total_s": round(float(sum(first_run_totals) / len(first_run_totals)), 6),
        "mean_repeat_run_total_s": round(float(sum(repeat_run_totals) / len(repeat_run_totals)), 6),
        "mean_retrieval_s": round(float(sum(all_retrievals) / len(all_retrievals)), 6) if all_retrievals else 0.0,
        "mean_filtering_s": round(float(sum(all_filterings) / len(all_filterings)), 6) if all_filterings else 0.0,
        "mean_pairing_s": round(float(sum(all_pairings) / len(all_pairings)), 6) if all_pairings else 0.0,
        "mean_change_detection_s": round(float(sum(all_changes) / len(all_changes)), 6) if all_changes else 0.0,
        "mean_suppression_s": round(float(sum(all_suppressions) / len(all_suppressions)), 6) if all_suppressions else 0.0,
    }

    report = {
        "benchmark_type": "end-to-end PoC functional pipeline baseline",
        "timestamp_utc": timestamp_utc,
        "repository_commit": commit_hash,
        "test_queries_count": len(queries),
        "test_queries": queries,
        "determinism_overall": "PASS" if overall_determinism else "FAIL",
        "aggregate_timing": aggregate_timing,
        "results": query_results,
    }

    # Save JSON report
    out_file = output_path if output_path else root / "data" / "evaluation" / "poc_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print summary table and aggregate
    print_benchmark_table(query_results)
    print_aggregate_timing(aggregate_timing)

    det_status_str = "[PASS] All queries executed deterministically across repeated runs." if overall_determinism else "[FAIL] Determinism mismatch detected."
    print(f"Determinism Check: {det_status_str}")
    print(f"Benchmark Report Saved: {out_file}\n")

    return report


def main() -> int:
    """Main CLI execution flow."""
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else ROOT_DIR
    queries = args.queries if args.queries else DEFAULT_QUERIES
    output_path = Path(args.output).resolve() if args.output else project_root / "data" / "evaluation" / "poc_benchmark.json"

    try:
        report = run_benchmark(
            queries=queries,
            output_path=output_path,
            top_k=args.top_k,
            modality=args.modality,
            project_root=project_root,
        )
        if report["determinism_overall"] != "PASS":
            return 1
        return 0
    except Exception as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
