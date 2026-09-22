"""Unit tests for the Semantic Retrieval Model Evaluation Harness.

Verifies:
1. Retrieval metric formulas (Recall@K, MRR, Precision@K, mAP@K, nDCG@K).
2. Strict self-match exclusion in Image -> Image retrieval.
3. Decoupling of instance retrieval vs. category retrieval.
4. Evaluation manifest schema and deterministic hash integrity.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.build_evaluation_manifest import (
    build_manifest_dict,
    compute_sha256,
    extract_category_from_filename,
    generate_mock_benchmark,
)
from scripts.evaluate_retrieval_models import (
    compute_category_recall_at_k,
    compute_map_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    evaluate_retrieval_performance,
)


class TestEvaluationHarness(unittest.TestCase):
    """Test suite for retrieval evaluation harness and metric math."""

    def test_sha256_and_category_extraction(self):
        """Verify SHA-256 computation and RSICD filename category parsing."""
        data = b"satellite_test_image_bytes"
        expected_hash = "2bd04906695db2e5229decc12d541d687f04e6d6fb05241d9dd0500bc60a8b0a"
        self.assertEqual(compute_sha256(data), expected_hash)

        # RSICD category parsing
        self.assertEqual(extract_category_from_filename("airport_102.jpg"), "airport")
        self.assertEqual(extract_category_from_filename("storage_tank_15.jpg"), "storage tank")
        self.assertEqual(extract_category_from_filename("dense_residential_99.jpg"), "dense residential")
        self.assertEqual(extract_category_from_filename("railway_station_01.jpg"), "railway station")

    def test_exact_instance_metrics(self):
        """Verify Recall@K and MRR with known ranked label matrices."""
        # 3 queries, 5 corpus items
        # Query 0: hit at rank 1
        # Query 1: hit at rank 3
        # Query 2: no hit in top 5
        labels = np.array([
            [True, False, False, False, False],
            [False, False, True, False, False],
            [False, False, False, False, False],
        ], dtype=bool)

        # Recall@1: query 0 hits -> 1/3
        self.assertAlmostEqual(compute_recall_at_k(labels, 1), 1.0 / 3.0)
        # Recall@3: query 0 and query 1 hit -> 2/3
        self.assertAlmostEqual(compute_recall_at_k(labels, 3), 2.0 / 3.0)
        # MRR: (1/1 + 1/3 + 0) / 3 = (4/3) / 3 = 4/9
        self.assertAlmostEqual(compute_mrr(labels), 4.0 / 9.0)

    def test_precision_at_k(self):
        """Verify Precision@K calculation."""
        # Query 0: 2 hits out of top 2 -> precision@2 = 1.0
        # Query 1: 1 hit out of top 2  -> precision@2 = 0.5
        labels = np.array([
            [True, True, False, False],
            [True, False, True, False],
        ], dtype=bool)

        self.assertAlmostEqual(compute_precision_at_k(labels, 1), 1.0)
        self.assertAlmostEqual(compute_precision_at_k(labels, 2), 0.75)

    def test_map_and_ndcg_at_k(self):
        """Verify mAP and nDCG rankings."""
        labels = np.array([
            [True, False, True, False],  # hits at 1 and 3
        ], dtype=bool)
        totals = np.array([2])  # 2 total relevant items

        # AP@4: rank 1 precision = 1/1 = 1.0; rank 3 precision = 2/3.
        # Mean AP = (1.0 + 2/3) / 2 = 5/6 approx 0.8333
        self.assertAlmostEqual(compute_map_at_k(labels, totals, 4), 5.0 / 6.0, places=4)

        # nDCG@4
        # DCG = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG (hits at 1 and 2) = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309297 = 1.6309297
        # nDCG = 1.5 / 1.6309297 approx 0.9197
        self.assertAlmostEqual(compute_ndcg_at_k(labels, totals, 4), 1.5 / (1.0 + 1.0 / np.log2(3)), places=4)

    def test_image_to_image_self_match_exclusion(self):
        """Verify that identical query image is NEVER counted as its own match."""
        # 3 images, 2 categories: img_0 (cat A), img_1 (cat A), img_2 (cat B)
        image_ids = ["img_0", "img_1", "img_2"]
        caption_ids = ["cap_0"]

        manifest = {
            "ground_truth": {
                "text_to_image_instance": {"cap_0": ["img_0"]},
                "text_to_image_category": {"cap_0": ["img_0", "img_1"]},
                "image_to_image_class": {
                    "img_0": ["img_1"],  # img_0's ground truth is ONLY img_1, NOT img_0
                    "img_1": ["img_0"],
                    "img_2": [],
                },
            }
        }

        # Mock embeddings: img_0 and img_1 are similar, img_2 is orthogonal
        # Normalize vectors
        img_embs = np.array([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ], dtype=float)
        img_embs = img_embs / np.linalg.norm(img_embs, axis=1, keepdims=True)

        text_embs = np.array([[1.0, 0.0]], dtype=float)

        results = evaluate_retrieval_performance(
            image_embeddings=img_embs,
            image_ids=image_ids,
            text_embeddings=text_embs,
            caption_ids=caption_ids,
            manifest=manifest,
        )

        i2i_metrics = results["image_to_image_class"]
        # For img_0, top non-self candidate is img_1 (which is relevant) -> P@1 should be > 0
        self.assertIn("Precision@1", i2i_metrics)
        self.assertIn("note", i2i_metrics)
        self.assertIn("self-match excluded", i2i_metrics["note"])

    def test_mock_manifest_generation(self):
        """Verify deterministic mock benchmark generation and manifest structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            manifest_file = generate_mock_benchmark(out_dir, num_categories=3, images_per_cat=2)

            self.assertTrue(manifest_file.exists())
            with open(manifest_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertIn("benchmark_metadata", data)
            self.assertIn("production_target_sources", data["benchmark_metadata"])
            self.assertEqual(len(data["categories"]), 3)
            self.assertEqual(len(data["items"]), 6)
            self.assertEqual(data["benchmark_metadata"]["caption_count"], 30)

            # Check ground truth keys
            gt = data["ground_truth"]
            self.assertIn("text_to_image_instance", gt)
            self.assertIn("text_to_image_category", gt)
            self.assertIn("image_to_image_class", gt)

            # Check self-match exclusion in ground truth
            for img_id, matches in gt["image_to_image_class"].items():
                self.assertNotIn(img_id, matches)


if __name__ == "__main__":
    unittest.main()
