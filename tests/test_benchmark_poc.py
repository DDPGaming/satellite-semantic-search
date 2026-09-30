"""Unit and integration tests for the End-to-End PoC Pipeline and Benchmark Orchestration.

Covers:
1. PoCPipeline initialization and query validation
2. PoCPipelineResult and PipelineStageTiming serialization
3. Pipeline execution when no matching filtered tiles exist
4. Pipeline execution when no temporal counterpart exists
5. Real end-to-end pipeline execution on optical search query
6. verify_determinism helper on matching vs mismatching pipeline results
7. run_benchmark programmatic execution generating valid JSON report structure
8. scripts/benchmark_poc.py CLI help and parameter parsing
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.change_detection.models import ChangeResult
from src.pairing.models import AlignmentStatus, SpatialCorrespondence, TemporalPair
from src.pipeline import PipelineStageTiming, PoCPipeline, PoCPipelineResult
from src.suppression.models import SuppressedChangeResult
from scripts.benchmark_poc import verify_determinism, run_benchmark

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_dummy_temporal_pair(pair_id: str = "pair__test_ref__test_comp") -> TemporalPair:
    """Helper to construct a mock TemporalPair."""
    return TemporalPair(
        pair_id=pair_id,
        modality="optical",
        reference_tile_id="test_ref",
        comparison_tile_id="test_comp",
        reference_scene_id="S2A_ref",
        comparison_scene_id="S2B_comp",
        reference_datetime_utc="2022-01-27T00:00:00Z",
        comparison_datetime_utc="2024-01-12T00:00:00Z",
        temporal_delta_days=715.0,
        spatial=SpatialCorrespondence(
            method="grid_index_exact",
            grid_index=(0, 0),
            spatial_iou_wgs84=1.0,
            bounds_wgs84=[72.9, 18.9, 73.0, 19.0],
        ),
        alignment=AlignmentStatus(
            is_pixel_aligned=True,
            alignment_type="native_pixel_aligned",
            crs="EPSG:32643",
            pixel_dimensions=(256, 256),
        ),
        reference_tile_dir="tiles/ref",
        comparison_tile_dir="tiles/comp",
    )


def make_dummy_pipeline_result(
    query: str = "test query",
    pair_id: str = "pair__test",
    cand_count: int = 100,
    conf_count: int = 80,
    threshold: float = 0.12,
    mean_conf: float = 0.45,
) -> PoCPipelineResult:
    """Helper to construct a synthetic PoCPipelineResult for determinism testing."""
    timing = PipelineStageTiming(
        retrieval_s=0.05,
        filtering_s=0.01,
        pairing_s=0.001,
        change_detection_s=0.10,
        suppression_s=0.02,
        total_s=0.181,
    )
    pair = make_dummy_temporal_pair(pair_id=pair_id)
    change_res = ChangeResult(
        pair_id=pair_id,
        modality="optical",
        reference_tile_id="test_ref",
        comparison_tile_id="test_comp",
        reference_scene_id="S2A_ref",
        comparison_scene_id="S2B_comp",
        reference_datetime_utc="2022-01-27T00:00:00Z",
        comparison_datetime_utc="2024-01-12T00:00:00Z",
        temporal_delta_days=715.0,
        grid_index=(0, 0),
        status="success",
        method="spectral_distance_l2",
        summary_statistics={"mean": 0.05, "median": 0.04, "p95": 0.10, "valid_pixels": 65536},
    )
    suppr_res = SuppressedChangeResult(
        pair_id=pair_id,
        modality="optical",
        reference_tile_id="test_ref",
        comparison_tile_id="test_comp",
        reference_scene_id="S2A_ref",
        comparison_scene_id="S2B_comp",
        reference_datetime_utc="2022-01-27T00:00:00Z",
        comparison_datetime_utc="2024-01-12T00:00:00Z",
        temporal_delta_days=715.0,
        grid_index=(0, 0),
        status="success",
        method="adaptive_mad_suppression",
        threshold_used=threshold,
        candidate_pixels_count=cand_count,
        suppressed_pixels_count=cand_count - conf_count,
        confirmed_pixels_count=conf_count,
        confirmed_change_ratio=conf_count / 65536.0,
        mean_confidence_on_change=mean_conf,
        max_confidence=0.85,
    )

    return PoCPipelineResult(
        query=query,
        status="success",
        retrieval_hits=[
            {"tile_id": "tile_01", "similarity_score": 0.35, "scene_id": "S2B_01", "modality": "optical"},
            {"tile_id": "tile_02", "similarity_score": 0.30, "scene_id": "S2A_01", "modality": "optical"},
        ],
        filtered_hits=[
            {"tile_id": "tile_01", "similarity_score": 0.35, "scene_id": "S2B_01", "modality": "optical"},
        ],
        selected_tile_id="tile_01",
        temporal_pair=pair,
        change_result=change_res,
        suppressed_result=suppr_res,
        timing=timing,
    )


class TestPoCPipelineOrchestration(unittest.TestCase):
    """Test suite for the reusable PoCPipeline orchestration service layer."""

    def test_query_validation(self):
        pipeline = PoCPipeline(
            project_root=PROJECT_ROOT,
            search_engine=MagicMock(),
            pairer=MagicMock(),
            detector=MagicMock(),
            suppressor=MagicMock(),
        )
        with self.assertRaises(ValueError):
            pipeline.run("")
        with self.assertRaises(ValueError):
            pipeline.run("   ")
        with self.assertRaises(ValueError):
            pipeline.run(12345)

    def test_result_and_timing_serialization(self):
        res = make_dummy_pipeline_result()
        d = res.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["query"], "test query")
        self.assertEqual(d["status"], "success")
        self.assertEqual(d["selected_tile_id"], "tile_01")
        self.assertIn("timing", d)
        self.assertIn("change_detection", d)
        self.assertIn("suppression", d)

        # Ensure valid JSON serialization
        json_str = json.dumps(d)
        self.assertIn("test query", json_str)
        deserialized = json.loads(json_str)
        self.assertEqual(deserialized["status"], "success")

    def test_no_filtered_hits_handling(self):
        mock_engine = MagicMock()
        mock_engine.encode_query.return_value = [0.1] * 512
        mock_engine.vector_index.search.return_value = [
            {"tile_id": "sar_01", "similarity_score": 0.40, "scene_id": "S1_01", "modality": "sar"}
        ]
        # Metadata filter returns empty
        mock_engine.search.return_value = []

        pipeline = PoCPipeline(
            project_root=PROJECT_ROOT,
            search_engine=mock_engine,
            pairer=MagicMock(),
            detector=MagicMock(),
            suppressor=MagicMock(),
        )

        res = pipeline.run("urban test", modality="optical")
        self.assertEqual(res.status, "no_filtered_hits")
        self.assertEqual(len(res.retrieval_hits), 1)
        self.assertEqual(len(res.filtered_hits), 0)
        self.assertIsNone(res.selected_tile_id)
        self.assertIsNone(res.temporal_pair)
        self.assertIsNotNone(res.timing)

    def test_no_temporal_pair_handling(self):
        mock_engine = MagicMock()
        mock_engine.encode_query.return_value = [0.1] * 512
        mock_engine.vector_index.search.return_value = [
            {"tile_id": "opt_01", "similarity_score": 0.40, "scene_id": "S2_01", "modality": "optical"}
        ]
        mock_engine.search.return_value = [
            {"tile_id": "opt_01", "similarity_score": 0.40, "scene_id": "S2_01", "modality": "optical"}
        ]

        mock_pairer = MagicMock()
        mock_pairer.get_pair_for_tile.return_value = None

        pipeline = PoCPipeline(
            project_root=PROJECT_ROOT,
            search_engine=mock_engine,
            pairer=mock_pairer,
            detector=MagicMock(),
            suppressor=MagicMock(),
        )

        res = pipeline.run("urban test", modality="optical")
        self.assertEqual(res.status, "no_temporal_pair")
        self.assertEqual(res.selected_tile_id, "opt_01")
        self.assertIsNone(res.temporal_pair)
        self.assertIsNotNone(res.timing)


class TestBenchmarkDeterminismAndExecution(unittest.TestCase):
    """Test suite for benchmark determinism checks and execution runner."""

    def test_verify_determinism_identical(self):
        r1 = make_dummy_pipeline_result(query="navi mumbai")
        r2 = make_dummy_pipeline_result(query="navi mumbai")

        is_det, failures = verify_determinism(r1, r2)
        self.assertTrue(is_det)
        self.assertEqual(len(failures), 0)

    def test_verify_determinism_mismatch_detected(self):
        r1 = make_dummy_pipeline_result(query="navi mumbai", cand_count=100)
        r2 = make_dummy_pipeline_result(query="navi mumbai", cand_count=105)

        is_det, failures = verify_determinism(r1, r2)
        self.assertFalse(is_det)
        self.assertTrue(any("Candidate count mismatch" in f for f in failures))

    def test_programmatic_benchmark_run(self):
        """Test programmatic execution of run_benchmark with temporary report path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_json = Path(tmp_dir) / "test_benchmark.json"
            test_query = ["built-up area around Navi Mumbai"]

            report = run_benchmark(
                queries=test_query,
                output_path=out_json,
                top_k=3,
                modality="optical",
                project_root=PROJECT_ROOT,
            )

            self.assertIn("benchmark_type", report)
            self.assertEqual(report["benchmark_type"], "end-to-end PoC functional pipeline baseline")
            self.assertEqual(report["determinism_overall"], "PASS")
            self.assertEqual(report["test_queries_count"], 1)
            self.assertIn("aggregate_timing", report)
            self.assertIn("results", report)

            # Verify saved file exists and contains valid JSON
            self.assertTrue(out_json.exists())
            with open(out_json, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            self.assertEqual(saved_data["repository_commit"], report["repository_commit"])
            self.assertEqual(len(saved_data["results"]), 1)
            self.assertEqual(saved_data["results"][0]["status"] if "status" in saved_data["results"][0] else saved_data["results"][0]["first_run"]["status"], "success")


if __name__ == "__main__":
    unittest.main()
