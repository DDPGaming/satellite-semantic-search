"""Comprehensive Test Suite for M6 Metadata Filtering in Natural Language Text Search.

Verifies:
1. Modality filtering ('optical', 'sar', None)
2. date_from filtering (UTC inclusive)
3. date_to filtering (UTC inclusive)
4. Single-day inclusive date range
5. Datetime boundary behavior (exact timestamps, tz-aware, naive)
6. scene_id filtering (exact matching)
7. Bounding box intersection (axis-aligned WGS84)
8. Bounding box boundary-touching behavior
9. Multi-predicate logical conjunction (AND across all 5 filters)
10. Zero-result query handling
11. Fewer results than top_k
12. top_k greater than available results
13. Monotonically non-increasing similarity score ranking
14. Deterministic tie-breaking on identical scores (tile_id ascending)
15. Invalid modality error handling
16. Invalid date format error handling
17. Inverted date range error handling (date_from > date_to)
18. Invalid bbox error handling (format, length, numeric, coordinate order)
19. Invalid scene_id type error handling
20. Regression verification for precomputed ndarray queries
21. Dynamic live index integration tests (zero hard-coded dataset sizes)
"""

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import unittest

import numpy as np

from src.embeddings.base import BaseEmbedder
from src.retrieval.index import VectorIndex
from src.retrieval.search import TextSearchEngine, _bbox_intersects, _parse_filter_datetime

ROOT_DIR = Path(__file__).resolve().parent.parent
STAGED_MODEL_DIR = ROOT_DIR / "models" / "clip-rsicd-v2"
INDEX_FILE = ROOT_DIR / "data" / "index" / "faiss.index"
METADATA_FILE = ROOT_DIR / "data" / "index" / "metadata.json"


class MockEmbedder(BaseEmbedder):
    """Deterministic mock embedder for fast unit tests without neural inference overhead."""

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
        # Deterministic vector based on query string
        h = sum(ord(c) for c in text)
        rng = np.random.RandomState(h % 10000)
        v = rng.randn(self._dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def encode_texts(self, texts: Any, batch_size: int = 64) -> np.ndarray:
        return np.vstack([self.encode_text(t) for t in texts])


def create_synthetic_index(dim: int = 512) -> VectorIndex:
    """
    Create an in-memory VectorIndex with 8 synthetic tiles spanning diverse
    modalities, dates, scenes, and bounding boxes.
    """
    idx = VectorIndex(dimension=dim)

    # 4 optical tiles, 4 SAR tiles
    vectors = []
    metadata = []

    # Tile specifications:
    # 0: optical, scene_opt_1, 2022-01-20T10:00:00Z, [73.0, 19.0, 73.1, 19.1]
    # 1: optical, scene_opt_1, 2022-01-21T10:00:00Z, [73.1, 19.0, 73.2, 19.1]
    # 2: optical, scene_opt_2, 2024-01-10T12:00:00Z, [73.2, 19.0, 73.3, 19.1]
    # 3: optical, scene_opt_2, 2024-01-11T12:00:00Z, [73.3, 19.0, 73.4, 19.1]
    # 4: sar,     scene_sar_1, 2022-01-20T10:00:00Z, [73.0, 19.1, 73.1, 19.2]
    # 5: sar,     scene_sar_1, 2022-01-21T10:00:00Z, [73.1, 19.1, 73.2, 19.2]
    # 6: sar,     scene_sar_2, 2024-01-10T12:00:00Z, [73.2, 19.1, 73.3, 19.2]
    # 7: sar,     scene_sar_2, 2024-01-11T12:00:00Z, [73.3, 19.1, 73.4, 19.2]
    tile_defs = [
        ("tile_opt_0", "optical", "scene_opt_1", "2022-01-20T10:00:00Z", [73.0, 19.0, 73.1, 19.1]),
        ("tile_opt_1", "optical", "scene_opt_1", "2022-01-21T10:00:00Z", [73.1, 19.0, 73.2, 19.1]),
        ("tile_opt_2", "optical", "scene_opt_2", "2024-01-10T12:00:00Z", [73.2, 19.0, 73.3, 19.1]),
        ("tile_opt_3", "optical", "scene_opt_2", "2024-01-11T12:00:00Z", [73.3, 19.0, 73.4, 19.1]),
        ("tile_sar_4", "sar",     "scene_sar_1", "2022-01-20T10:00:00Z", [73.0, 19.1, 73.1, 19.2]),
        ("tile_sar_5", "sar",     "scene_sar_1", "2022-01-21T10:00:00Z", [73.1, 19.1, 73.2, 19.2]),
        ("tile_sar_6", "sar",     "scene_sar_2", "2024-01-10T12:00:00Z", [73.2, 19.1, 73.3, 19.2]),
        ("tile_sar_7", "sar",     "scene_sar_2", "2024-01-11T12:00:00Z", [73.3, 19.1, 73.4, 19.2]),
    ]

    for i, (tid, mod, scn, acq, bbox) in enumerate(tile_defs):
        v = np.zeros(dim, dtype=np.float32)
        v[i] = 1.0  # Orthogonal basis vector
        vectors.append(v)

        metadata.append({
            "tile_id": tid,
            "scene_id": scn,
            "modality": mod,
            "tile_directory": f"tiles/{mod}/{scn}/{tid}",
            "bounds_wgs84": bbox,
            "acquisition_datetime_utc": acq,
            "crs": "EPSG:4326",
            "grid_index": {"row_idx": i // 2, "col_idx": i % 2},
            "row_id": i,
        })

    idx.add(np.vstack(vectors), metadata)
    return idx


class TestSyntheticMetadataFiltering(unittest.TestCase):
    """Synthetic unit tests verifying all filter predicates, combinations, and boundaries."""

    def setUp(self):
        self.mock_embedder = MockEmbedder(dim=512)
        self.synth_index = create_synthetic_index(dim=512)
        self.engine = TextSearchEngine(
            vector_index=self.synth_index,
            embedder=self.mock_embedder,
        )

    def test_modality_filtering(self):
        """Verify modality filtering restricts hits strictly to optical or SAR."""
        opt_results = self.engine.search("test query", top_k=10, modality="optical")
        self.assertEqual(len(opt_results), 4)
        for r in opt_results:
            self.assertEqual(r["modality"], "optical")

        sar_results = self.engine.search("test query", top_k=10, modality="sar")
        self.assertEqual(len(sar_results), 4)
        for r in sar_results:
            self.assertEqual(r["modality"], "sar")

        # Case insensitivity
        opt_upper = self.engine.search("test query", top_k=10, modality="OPTICAL")
        self.assertEqual(len(opt_upper), 4)

    def test_date_from_filtering(self):
        """Verify date_from filters out tiles acquired before the threshold."""
        # 2023-01-01 should include only tiles from 2024 (tiles 2, 3, 6, 7)
        results = self.engine.search("test query", top_k=10, date_from="2023-01-01")
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertTrue(r["acquisition_datetime_utc"].startswith("2024"))

    def test_date_to_filtering(self):
        """Verify date_to filters out tiles acquired after the end of the specified date."""
        # 2023-01-01 should include only tiles from 2022 (tiles 0, 1, 4, 5)
        results = self.engine.search("test query", top_k=10, date_to="2023-01-01")
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertTrue(r["acquisition_datetime_utc"].startswith("2022"))

    def test_single_day_inclusive_range(self):
        """Verify date_from == date_to covering a single day includes all tiles from that day."""
        # 2022-01-20 has 2 tiles: tile_opt_0 and tile_sar_4
        results = self.engine.search(
            "test query",
            top_k=10,
            date_from="2022-01-20",
            date_to="2022-01-20",
        )
        self.assertEqual(len(results), 2)
        retrieved_ids = {r["tile_id"] for r in results}
        self.assertEqual(retrieved_ids, {"tile_opt_0", "tile_sar_4"})

    def test_datetime_boundary_behavior(self):
        """Verify exact boundary timestamps and timezone conversions."""
        # Exact second match
        results = self.engine.search(
            "test query",
            top_k=10,
            date_from="2022-01-20T10:00:00Z",
            date_to="2022-01-20T10:00:00Z",
        )
        self.assertEqual(len(results), 2)

        # 1 second after excludes the 10:00:00 tile
        results_after = self.engine.search(
            "test query",
            top_k=10,
            date_from="2022-01-20T10:00:01Z",
            date_to="2022-01-20T23:59:59Z",
        )
        self.assertEqual(len(results_after), 0)

        # Timezone aware datetime input (UTC+5:30)
        # 15:30:00+05:30 == 10:00:00Z
        ist_dt = datetime(2022, 1, 20, 15, 30, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
        results_ist = self.engine.search(
            "test query",
            top_k=10,
            date_from=ist_dt,
            date_to=ist_dt,
        )
        self.assertEqual(len(results_ist), 2)

        # Naive datetime interpreted as UTC
        naive_dt = datetime(2022, 1, 20, 10, 0, 0)
        results_naive = self.engine.search(
            "test query",
            top_k=10,
            date_from=naive_dt,
            date_to=naive_dt,
        )
        self.assertEqual(len(results_naive), 2)

        # datetime.date object
        date_obj = date(2022, 1, 20)
        results_date = self.engine.search(
            "test query",
            top_k=10,
            date_from=date_obj,
            date_to=date_obj,
        )
        self.assertEqual(len(results_date), 2)

    def test_scene_id_filtering(self):
        """Verify exact matching on scene_id."""
        results = self.engine.search("test query", top_k=10, scene_id="scene_opt_1")
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r["scene_id"], "scene_opt_1")

        sar_scene = self.engine.search("test query", top_k=10, scene_id="scene_sar_2")
        self.assertEqual(len(sar_scene), 2)
        for r in sar_scene:
            self.assertEqual(r["scene_id"], "scene_sar_2")

    def test_bbox_intersection(self):
        """Verify axis-aligned bounding box spatial intersection."""
        # Query bbox covering tile 0 and tile 1 [73.05, 19.05, 73.15, 19.15]
        # Intersects tile 0, 1, 4, 5
        results = self.engine.search(
            "test query",
            top_k=10,
            bbox=[73.05, 19.05, 73.15, 19.15],
        )
        self.assertEqual(len(results), 4)
        retrieved_ids = {r["tile_id"] for r in results}
        self.assertEqual(retrieved_ids, {"tile_opt_0", "tile_opt_1", "tile_sar_4", "tile_sar_5"})

        # Disjoint bbox returns 0 tiles
        disjoint_results = self.engine.search(
            "test query",
            top_k=10,
            bbox=[80.0, 30.0, 81.0, 31.0],
        )
        self.assertEqual(len(disjoint_results), 0)

    def test_bbox_boundary_touching_behavior(self):
        """Verify that touching the boundary of a tile is considered an intersection."""
        # Tile 0 bounds: [73.0, 19.0, 73.1, 19.1]
        # Bbox touching east edge: [73.1, 19.0, 73.2, 19.1]
        # This touches tile 0 at lon 73.1 and tile 1 at lon 73.1..73.2
        results = self.engine.search(
            "test query",
            top_k=10,
            bbox=[73.1, 19.0, 73.2, 19.1],
        )
        retrieved_ids = {r["tile_id"] for r in results}
        self.assertIn("tile_opt_0", retrieved_ids)
        self.assertIn("tile_opt_1", retrieved_ids)

        # Touching at a single corner vertex (lon=73.1, lat=19.1)
        results_corner = self.engine.search(
            "test query",
            top_k=10,
            bbox=[73.1, 19.1, 73.2, 19.2],
        )
        # Should touch tile 0 (northeast corner), tile 1 (northwest corner),
        # tile 4 (southeast corner), tile 5 (southwest corner)
        corner_ids = {r["tile_id"] for r in results_corner}
        self.assertIn("tile_opt_0", corner_ids)
        self.assertIn("tile_opt_1", corner_ids)
        self.assertIn("tile_sar_4", corner_ids)
        self.assertIn("tile_sar_5", corner_ids)

    def test_all_five_filters_combined_and(self):
        """Verify all five filters active simultaneously in logical conjunction (AND)."""
        results = self.engine.search(
            "test query",
            top_k=10,
            modality="optical",
            date_from="2022-01-01",
            date_to="2022-12-31",
            scene_id="scene_opt_1",
            bbox=[73.0, 19.0, 73.05, 19.05],
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["tile_id"], "tile_opt_0")
        self.assertEqual(results[0]["rank"], 1)

    def test_zero_result_query(self):
        """Verify that zero matches returns an empty list without error."""
        results = self.engine.search(
            "test query",
            top_k=5,
            scene_id="nonexistent_scene_xyz",
        )
        self.assertEqual(results, [])

    def test_fewer_results_than_top_k(self):
        """Verify that when fewer tiles match than top_k, all matching tiles are returned."""
        results = self.engine.search(
            "test query",
            top_k=10,
            scene_id="scene_opt_1",
        )
        # Exactly 2 matching tiles
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[1]["rank"], 2)

    def test_top_k_greater_than_available_results(self):
        """Verify requesting top_k greater than all indexed tiles does not error."""
        results = self.engine.search(
            "test query",
            top_k=1000,
        )
        self.assertEqual(len(results), 8)

    def test_ranking_remains_non_increasing(self):
        """Verify returned similarity scores are monotonically non-increasing."""
        results = self.engine.search("general terrain query", top_k=8)
        self.assertGreater(len(results), 1)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(
                results[i]["similarity_score"],
                results[i + 1]["similarity_score"],
            )

    def test_deterministic_tie_breaking(self):
        """Verify tiles with identical scores are ordered deterministically by tile_id ascending."""
        tie_idx = VectorIndex(dimension=512)
        vec = np.zeros(512, dtype=np.float32)
        vec[0] = 1.0  # Identical vector for all tiles

        vectors = np.tile(vec, (3, 1))
        metadata = [
            {"tile_id": "tile_charlie", "scene_id": "s1", "modality": "optical", "tile_directory": "t3", "bounds_wgs84": [0, 0, 1, 1], "acquisition_datetime_utc": "2024-01-01T00:00:00Z", "crs": "EPSG:4326", "grid_index": {"row_idx": 0, "col_idx": 0}, "row_id": 0},
            {"tile_id": "tile_alpha", "scene_id": "s1", "modality": "optical", "tile_directory": "t1", "bounds_wgs84": [0, 0, 1, 1], "acquisition_datetime_utc": "2024-01-01T00:00:00Z", "crs": "EPSG:4326", "grid_index": {"row_idx": 0, "col_idx": 1}, "row_id": 1},
            {"tile_id": "tile_bravo", "scene_id": "s1", "modality": "optical", "tile_directory": "t2", "bounds_wgs84": [0, 0, 1, 1], "acquisition_datetime_utc": "2024-01-01T00:00:00Z", "crs": "EPSG:4326", "grid_index": {"row_idx": 0, "col_idx": 2}, "row_id": 2},
        ]
        tie_idx.add(vectors, metadata)

        engine = TextSearchEngine(vector_index=tie_idx, embedder=self.mock_embedder)
        results = engine.search(query=vec, top_k=3, modality="optical")

        # All scores identical
        self.assertAlmostEqual(results[0]["similarity_score"], 1.0, places=5)
        self.assertAlmostEqual(results[1]["similarity_score"], 1.0, places=5)
        self.assertAlmostEqual(results[2]["similarity_score"], 1.0, places=5)

        # Alphabetical tie-break
        self.assertEqual(results[0]["tile_id"], "tile_alpha")
        self.assertEqual(results[1]["tile_id"], "tile_bravo")
        self.assertEqual(results[2]["tile_id"], "tile_charlie")

    def test_invalid_modality_raises(self):
        """Verify unsupported modality string or invalid type raises ValueError or TypeError."""
        with self.assertRaises(ValueError):
            self.engine.search("query", modality="infrared")
        with self.assertRaises(TypeError):
            self.engine.search("query", modality=123)

    def test_invalid_date_raises(self):
        """Verify malformed date strings or invalid types raise ValueError or TypeError."""
        with self.assertRaises(ValueError):
            self.engine.search("query", date_from="not-a-date")
        with self.assertRaises(ValueError):
            self.engine.search("query", date_to="2022-99-99")
        with self.assertRaises(ValueError):
            self.engine.search("query", date_from="   ")
        with self.assertRaises(TypeError):
            self.engine.search("query", date_from=12345)
        with self.assertRaises(TypeError):
            self.engine.search("query", date_to=True)

    def test_inverted_date_range_raises(self):
        """Verify date_from > date_to raises specific ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.engine.search(
                "query",
                date_from="2024-01-01",
                date_to="2022-01-01",
            )
        self.assertIn("Invalid date range: date_from cannot be after date_to.", str(ctx.exception))

    def test_invalid_bbox_raises(self):
        """Verify invalid bbox shapes, types, and coordinate orders raise ValueError or TypeError."""
        # Non-sequence
        with self.assertRaises(TypeError):
            self.engine.search("query", bbox="73.0, 19.0, 74.0, 20.0")
        # Length != 4
        with self.assertRaises(ValueError):
            self.engine.search("query", bbox=[73.0, 19.0, 74.0])
        with self.assertRaises(ValueError):
            self.engine.search("query", bbox=[73.0, 19.0, 74.0, 20.0, 25.0])
        # Non-numeric
        with self.assertRaises(TypeError):
            self.engine.search("query", bbox=[73.0, "19.0", 74.0, 20.0])
        with self.assertRaises(TypeError):
            self.engine.search("query", bbox=[True, 19.0, 74.0, 20.0])
        with self.assertRaises(TypeError):
            self.engine.search("query", bbox=[73.0, None, 74.0, 20.0])
        # Inverted coordinates
        with self.assertRaises(ValueError):
            self.engine.search("query", bbox=[74.0, 19.0, 73.0, 20.0])  # min_lon > max_lon
        with self.assertRaises(ValueError):
            self.engine.search("query", bbox=[73.0, 20.0, 74.0, 19.0])  # min_lat > max_lat

    def test_invalid_scene_id_type_raises(self):
        """Verify non-string non-None scene_id raises TypeError."""
        with self.assertRaises(TypeError):
            self.engine.search("query", scene_id=12345)
        with self.assertRaises(TypeError):
            self.engine.search("query", scene_id=["scene_opt_1"])
        with self.assertRaises(TypeError):
            self.engine.search("query", scene_id=False)
        with self.assertRaises(ValueError):
            self.engine.search("query", scene_id="   ")

    def test_regression_ndarray_query(self):
        """Verify precomputed 1D and 2D numpy arrays work with metadata filters."""
        query_vec = np.zeros(512, dtype=np.float32)
        query_vec[0] = 1.0

        # 1D array with filters
        res_1d = self.engine.search(
            query=query_vec,
            top_k=5,
            modality="optical",
            date_from="2022-01-01",
        )
        self.assertGreater(len(res_1d), 0)
        self.assertEqual(res_1d[0]["modality"], "optical")

        # 2D array of shape (1, 512)
        query_vec_2d = query_vec.reshape(1, 512)
        res_2d = self.engine.search(
            query=query_vec_2d,
            top_k=5,
            scene_id="scene_opt_1",
        )
        self.assertEqual(len(res_2d), 2)


class TestLiveMetadataFiltering(unittest.TestCase):
    """Integration test suite executing queries against staged index with dynamic expectations."""

    @classmethod
    def setUpClass(cls):
        if not (INDEX_FILE.exists() and METADATA_FILE.exists() and STAGED_MODEL_DIR.exists()):
            raise unittest.SkipTest("Staged artifacts not available for live integration test.")

        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            cls.live_meta = json.load(f)

        cls.engine = TextSearchEngine(
            index_file=INDEX_FILE,
            metadata_file=METADATA_FILE,
            model_dir=STAGED_MODEL_DIR,
            device="cpu",
        )

    def test_live_dynamic_modality_filtering(self):
        """Verify live optical and SAR tile counts match metadata dynamically."""
        total_vectors = self.engine.total_vectors
        optical_count = sum(1 for e in self.live_meta["entries"] if e["modality"] == "optical")
        sar_count = sum(1 for e in self.live_meta["entries"] if e["modality"] == "sar")
        self.assertEqual(optical_count + sar_count, total_vectors)

        # Optical filter retrieving all available
        opt_results = self.engine.search("port and cargo vessels", top_k=total_vectors, modality="optical")
        self.assertEqual(len(opt_results), optical_count)
        for r in opt_results:
            self.assertEqual(r["modality"], "optical")

        # SAR filter retrieving all available
        sar_results = self.engine.search("radar water backscatter", top_k=total_vectors, modality="sar")
        self.assertEqual(len(sar_results), sar_count)
        for r in sar_results:
            self.assertEqual(r["modality"], "sar")

    def test_live_scene_filtering(self):
        """Verify filtering by an existing scene matches that scene's tile count dynamically."""
        target_scene = self.live_meta["entries"][0]["scene_id"]
        expected_count = sum(1 for e in self.live_meta["entries"] if e["scene_id"] == target_scene)

        results = self.engine.search(
            "coastal water",
            top_k=self.engine.total_vectors,
            scene_id=target_scene,
        )
        self.assertEqual(len(results), expected_count)
        for r in results:
            self.assertEqual(r["scene_id"], target_scene)

    def test_live_date_range_filtering(self):
        """Verify live date filtering against known acquisition dates."""
        # Find earliest and latest acquisition dates dynamically
        acq_dates = sorted(
            datetime.fromisoformat(e["acquisition_datetime_utc"][:-1] + "+00:00")
            for e in self.live_meta["entries"]
            if e.get("acquisition_datetime_utc")
        )
        midpoint = acq_dates[len(acq_dates) // 2]

        results = self.engine.search(
            "urban development",
            top_k=self.engine.total_vectors,
            date_from=midpoint,
        )
        for r in results:
            r_dt = datetime.fromisoformat(r["acquisition_datetime_utc"][:-1] + "+00:00")
            self.assertGreaterEqual(r_dt, midpoint)

    def test_live_bbox_filtering(self):
        """Verify spatial bbox filtering on live tiles."""
        # Pick the bounds of the first entry and expand slightly
        sample_bounds = self.live_meta["entries"][0]["bounds_wgs84"]
        query_bbox = [
            sample_bounds[0] - 0.001,
            sample_bounds[1] - 0.001,
            sample_bounds[2] + 0.001,
            sample_bounds[3] + 0.001,
        ]

        results = self.engine.search(
            "satellite view",
            top_k=20,
            bbox=query_bbox,
        )
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertTrue(_bbox_intersects(tuple(query_bbox), r["bounds_wgs84"]))

    def test_live_combined_conjunction(self):
        """Verify multi-predicate query combining modality, scene, and bbox on live data."""
        sample_entry = self.live_meta["entries"][0]
        mod = sample_entry["modality"]
        scn = sample_entry["scene_id"]
        b = sample_entry["bounds_wgs84"]

        results = self.engine.search(
            "water and land boundary",
            top_k=5,
            modality=mod,
            scene_id=scn,
            bbox=b,
        )
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r["modality"], mod)
            self.assertEqual(r["scene_id"], scn)
            self.assertTrue(_bbox_intersects(tuple(b), r["bounds_wgs84"]))


if __name__ == "__main__":
    unittest.main()
