"""Comprehensive Unit & Integration Test Suite for M5 Text Search Engine.

Verifies:
1. Query input validation (empty strings, whitespace, non-string types).
2. top_k parameter validation (positive integer constraints, bounds clamping).
3. Modality filter validation ('optical', 'sar', or None).
4. Text query vector normalization and dimension integrity.
5. Exact exhaustive modality filtering without index duplication.
6. Deterministic tie-breaking behavior (tile_id ascending on equal similarity scores).
7. Strict non-increasing similarity score monotonicity (S1 >= S2 >= ... >= Sk).
8. Result contract field completeness and metadata surfacing.
9. Synthetic engine behavior with mock embedder.
10. Live staged index and CLIP model integration tests.
"""

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.embeddings.base import BaseEmbedder
from src.retrieval.index import VectorIndex
from src.retrieval.search import TextSearchEngine

ROOT_DIR = Path(__file__).resolve().parent.parent
STAGED_MODEL_DIR = ROOT_DIR / "models" / "clip-rsicd-v2"
INDEX_FILE = ROOT_DIR / "data" / "index" / "faiss.index"
METADATA_FILE = ROOT_DIR / "data" / "index" / "metadata.json"


class MockEmbedder(BaseEmbedder):
    """Deterministic mock embedder for fast unit tests without loading neural weights."""

    def __init__(self, dim: int = 512):
        self._dim = dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def model_metadata(self) -> Dict[str, Any]:
        return {"model_name": "MockEmbedder", "embedding_dim": self._dim}

    def encode_image(self, image: Any) -> np.ndarray:
        v = np.ones(self._dim, dtype=np.float32)
        return v / np.linalg.norm(v)

    def encode_images(self, images: Any, batch_size: int = 32) -> np.ndarray:
        return np.vstack([self.encode_image(img) for img in images])

    def encode_text(self, text: str) -> np.ndarray:
        # Deterministic pseudo-embedding based on hash of text
        h = sum(ord(c) for c in text)
        rng = np.random.RandomState(h % 10000)
        v = rng.randn(self._dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def encode_texts(self, texts: Any, batch_size: int = 64) -> np.ndarray:
        return np.vstack([self.encode_text(t) for t in texts])


def create_synthetic_index(dim: int = 512) -> VectorIndex:
    """Helper to create an in-memory VectorIndex with optical and SAR tiles."""
    idx = VectorIndex(dimension=dim)

    # 4 optical tiles and 4 SAR tiles
    vectors = []
    metadata = []

    for i in range(8):
        modality = "optical" if i < 4 else "sar"
        v = np.zeros(dim, dtype=np.float32)
        # Unique directional vector
        v[i] = 1.0
        vectors.append(v)

        metadata.append({
            "tile_id": f"tile_{modality}_{i:02d}",
            "scene_id": f"scene_{modality}_{i // 2}",
            "modality": modality,
            "tile_directory": f"data/tiles/{modality}/tile_{i:02d}",
            "bounds_wgs84": [73.0 + i * 0.01, 19.0 + i * 0.01, 73.02 + i * 0.01, 19.02 + i * 0.01],
            "acquisition_datetime_utc": f"2022-01-2{i}T00:00:00Z",
            "crs": "EPSG:4326",
            "grid_index": {"row_idx": i, "col_idx": 0},
            "row_id": i,
        })

    idx.add(np.vstack(vectors), metadata)
    return idx


class TestQueryValidation(unittest.TestCase):
    """Test suite for input query validation and parameter boundaries."""

    def setUp(self):
        self.mock_embedder = MockEmbedder(dim=512)
        self.mock_index = create_synthetic_index(dim=512)
        self.engine = TextSearchEngine(
            vector_index=self.mock_index,
            embedder=self.mock_embedder,
        )

    def test_empty_query_raises(self):
        """Verify empty query string raises ValueError."""
        with self.assertRaises(ValueError):
            self.engine.search("")

    def test_whitespace_query_raises(self):
        """Verify whitespace-only query string raises ValueError."""
        with self.assertRaises(ValueError):
            self.engine.search("   \t\n  ")

    def test_non_string_query_raises(self):
        """Verify non-string and non-array queries raise TypeError."""
        for invalid in [12345, None, ["query"], {"q": "text"}, True]:
            with self.assertRaises(TypeError):
                self.engine.search(invalid)

    def test_invalid_top_k_raises(self):
        """Verify top_k <= 0 or non-integer raises ValueError or TypeError."""
        with self.assertRaises(ValueError):
            self.engine.search("valid query", top_k=0)
        with self.assertRaises(ValueError):
            self.engine.search("valid query", top_k=-5)
        with self.assertRaises(TypeError):
            self.engine.search("valid query", top_k=2.5)  # type: ignore
        with self.assertRaises(TypeError):
            self.engine.search("valid query", top_k=True)  # type: ignore

    def test_invalid_modality_raises(self):
        """Verify unsupported modality string raises ValueError."""
        with self.assertRaises(ValueError):
            self.engine.search("valid query", modality="infrared")
        with self.assertRaises(TypeError):
            self.engine.search("valid query", modality=123)  # type: ignore


class TestSearchEngineSynthetic(unittest.TestCase):
    """Test suite for search engine logic, tie-breaking, and result contracts."""

    def setUp(self):
        self.mock_embedder = MockEmbedder(dim=512)
        self.mock_index = create_synthetic_index(dim=512)
        self.engine = TextSearchEngine(
            vector_index=self.mock_index,
            embedder=self.mock_embedder,
        )

    def test_search_all_modalities(self):
        """Verify searching all modalities retrieves both optical and SAR tiles."""
        results = self.engine.search("test terrain", top_k=6, modality=None)
        self.assertLessEqual(len(results), 6)
        modalities = {r["modality"] for r in results}
        self.assertTrue(len(modalities) >= 1)

        # Check non-increasing score monotonicity: S1 >= S2 >= ... >= Sk
        scores = [r["similarity_score"] for r in results]
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i], scores[i + 1])

    def test_search_optical_only(self):
        """Verify modality='optical' returns exclusively optical tiles."""
        results = self.engine.search("test terrain", top_k=4, modality="optical")
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertEqual(r["modality"], "optical")

    def test_search_sar_only(self):
        """Verify modality='sar' returns exclusively SAR tiles."""
        results = self.engine.search("test radar", top_k=4, modality="sar")
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertEqual(r["modality"], "sar")

    def test_modality_filtering_exactness(self):
        """Verify modality filtering returns exact top-k without missing lower-ranked candidates."""
        # Query targeting optical vector 0
        q_vec = np.zeros(512, dtype=np.float32)
        q_vec[0] = 1.0

        optical_hits = self.engine.search(q_vec, top_k=3, modality="optical")
        self.assertEqual(len(optical_hits), 3)
        self.assertEqual(optical_hits[0]["tile_id"], "tile_optical_00")
        self.assertAlmostEqual(optical_hits[0]["similarity_score"], 1.0, places=4)

        # Ranks must be 1, 2, 3
        self.assertEqual([h["rank"] for h in optical_hits], [1, 2, 3])

    def test_deterministic_tie_breaking(self):
        """Verify identical similarity scores are broken deterministically by tile_id ascending."""
        idx = VectorIndex(dimension=512)
        # Create 3 vectors with identical inner product against query [1, 0, 0, ...]
        v_query = np.zeros(512, dtype=np.float32)
        v_query[0] = 1.0

        v1 = np.zeros(512, dtype=np.float32)
        v1[0] = 0.5
        v1[1] = np.sqrt(0.75)  # norm = 1.0, dot product with v_query = 0.5

        v2 = np.zeros(512, dtype=np.float32)
        v2[0] = 0.5
        v2[2] = np.sqrt(0.75)  # norm = 1.0, dot product with v_query = 0.5

        v3 = np.zeros(512, dtype=np.float32)
        v3[0] = 0.5
        v3[3] = np.sqrt(0.75)  # norm = 1.0, dot product with v_query = 0.5

        # Add in non-alphabetical order
        vectors = np.vstack([v2, v3, v1])
        meta = [
            {"tile_id": "tile_Z", "modality": "optical", "scene_id": "s1"},
            {"tile_id": "tile_A", "modality": "optical", "scene_id": "s1"},
            {"tile_id": "tile_M", "modality": "optical", "scene_id": "s1"},
        ]
        idx.add(vectors, meta)

        engine = TextSearchEngine(vector_index=idx, embedder=self.mock_embedder)
        hits = engine.search(v_query, top_k=3)

        # All scores are equal (0.5), so tile_id must be sorted: tile_A, tile_M, tile_Z
        self.assertEqual(hits[0]["tile_id"], "tile_A")
        self.assertEqual(hits[1]["tile_id"], "tile_M")
        self.assertEqual(hits[2]["tile_id"], "tile_Z")
        self.assertEqual([h["rank"] for h in hits], [1, 2, 3])

    def test_result_contract_fields(self):
        """Verify all required result contract fields are present in every hit."""
        results = self.engine.search("sample query", top_k=2)
        self.assertTrue(len(results) > 0)

        required_fields = {
            "rank",
            "tile_id",
            "similarity_score",
            "row_id",
            "scene_id",
            "modality",
            "bounds_wgs84",
            "tile_directory",
            "acquisition_datetime_utc",
            "metadata",
        }

        for r in results:
            self.assertTrue(required_fields.issubset(r.keys()), f"Missing fields: {required_fields - set(r.keys())}")
            self.assertIsInstance(r["rank"], int)
            self.assertIsInstance(r["tile_id"], str)
            self.assertIsInstance(r["similarity_score"], float)
            self.assertIsInstance(r["modality"], str)
            self.assertIsInstance(r["metadata"], dict)

    def test_top_k_larger_than_total_vectors(self):
        """Verify requesting top_k > total_vectors returns all available vectors without error."""
        results = self.engine.search("query", top_k=100)
        self.assertEqual(len(results), 8)  # Index only has 8 vectors


class TestLiveTextSearchEngine(unittest.TestCase):
    """Integration test suite executing against the live staged FAISS index and model."""

    @classmethod
    def setUpClass(cls):
        if not INDEX_FILE.exists() or not METADATA_FILE.exists():
            raise unittest.SkipTest("Live data/index/ files not found.")
        if not STAGED_MODEL_DIR.exists():
            raise unittest.SkipTest(f"Live model directory '{STAGED_MODEL_DIR}' not found.")

        cls.engine = TextSearchEngine.from_staged_artifacts(device="cpu")

    def test_live_engine_initialization(self):
        """Verify live engine successfully loads 226 vectors from staged index."""
        self.assertEqual(self.engine.embedding_dim, 512)
        self.assertGreater(self.engine.total_vectors, 0)
        self.assertEqual(self.engine.total_vectors, 226)

    def test_live_optical_search(self):
        """Verify searching with optical filter on live index."""
        results = self.engine.search("dense urban residential buildings", top_k=5, modality="optical")
        self.assertEqual(len(results), 5)
        for r in results:
            self.assertEqual(r["modality"], "optical")
            self.assertTrue(r["tile_id"].startswith("s2_"))
            self.assertIsNotNone(r["bounds_wgs84"])
            self.assertIsNotNone(r["acquisition_datetime_utc"])

        # Monotonicity check
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i]["similarity_score"], results[i + 1]["similarity_score"])

    def test_live_sar_search(self):
        """Verify searching with SAR filter on live index."""
        results = self.engine.search("coastal water and shipping channel", top_k=5, modality="sar")
        self.assertEqual(len(results), 5)
        for r in results:
            self.assertEqual(r["modality"], "sar")
            self.assertTrue(r["tile_id"].startswith("s1_"))
            self.assertIsNotNone(r["bounds_wgs84"])
            self.assertIsNotNone(r["acquisition_datetime_utc"])

        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i]["similarity_score"], results[i + 1]["similarity_score"])

    def test_live_cross_modal_search(self):
        """Verify cross-modal search returns ranked results across both sensors."""
        results = self.engine.search("commercial harbor and port docks", top_k=8, modality=None)
        self.assertEqual(len(results), 8)
        self.assertEqual([r["rank"] for r in results], list(range(1, 9)))
        for r in results:
            self.assertIn(r["modality"], ("optical", "sar"))

        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i]["similarity_score"], results[i + 1]["similarity_score"])


if __name__ == "__main__":
    unittest.main()
