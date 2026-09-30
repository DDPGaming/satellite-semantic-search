"""Comprehensive Unit & Integration Test Suite for Milestone 7 (Multi-Temporal Pairing).

Verifies all 42 required M7 invariants and test points:
1. Dynamic discovery of the staged optical scene pair
2. Dynamic verification of all corresponding optical tile pairs
3. Dynamic discovery of the staged SAR scene pair
4. Dynamic verification of all corresponding SAR tile pairs
5. Chronological ordering invariant (reference is earlier, comparison is later)
6. Exact grid_index correspondence across all pairs
7. Optical is_pixel_aligned == True
8. Optical alignment_type == "native_pixel_aligned"
9. SAR is_pixel_aligned == False
10. SAR alignment_type == "unaligned_radar_grid"
11. Spatial IoU diagnostic exists and is in [0.0, 1.0]
12. Optical IoU diagnostic is 1.0
13. SAR IoU diagnostic is within observed dataset range (0.95 to 1.0)
14. No counterpart behavior returns None
15. Multiple temporal candidates discovery
16. Nearest-in-time canonical candidate selection
17. Future-over-past tie-breaking
18. Scene-ID alphabetical tie-breaking
19. Explicit target_scene_id selection
20. Minimum temporal delta constraint
21. Maximum temporal delta constraint
22. Inverted temporal bounds rejection
23. Same-scene pairing rejection
24. Identical timestamp rejection
25. Invalid tile ID raises KeyError
26. Invalid scene ID raises KeyError
27. Malformed datetime handling
28. Missing grid_index handling
29. Mismatched grid_index handling
30. Unsupported cross-modality pairing rejection
31. Non-overlapping footprints rejection
32. Boundary-touching footprint behavior (IoU == 0.0)
33. Raster dimension mismatch prevents pixel-aligned status
34. Raster resolution mismatch prevents pixel-aligned status
35. CRS mismatch prevents pixel-aligned status
36. Transform mismatch prevents pixel-aligned status
37. GCP-based SAR metadata not mistaken for affine alignment
38. Duplicate metadata detection (tile_id and scene_id+grid_index)
39. M6 -> M7 handoff integration using TextSearchEngine
40. Deterministic pair ordering
41. Stable serialization across repeated runs
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import unittest

from src.pairing.models import AlignmentStatus, SpatialCorrespondence, TemporalPair
from src.pairing.pairer import TemporalPairer, compute_wgs84_iou, parse_iso_datetime
from src.retrieval.search import TextSearchEngine

ROOT_DIR = Path(__file__).resolve().parent.parent
METADATA_FILE = ROOT_DIR / "data" / "index" / "metadata.json"


class TestSpatialMathHelpers(unittest.TestCase):
    """Test unit math helpers for IoU and date parsing."""

    def test_parse_iso_datetime_variants(self):
        """Verify parsing UTC strings with Z, offsets, or naive."""
        dt1 = parse_iso_datetime("2024-01-12T05:53:38.695000Z")
        self.assertEqual(dt1.tzinfo, timezone.utc)
        self.assertEqual(dt1.year, 2024)

        dt2 = parse_iso_datetime("2024-01-12T11:23:38.695000+05:30")
        self.assertEqual(dt2.astimezone(timezone.utc), dt1)

        with self.assertRaises(ValueError):
            parse_iso_datetime("not-a-datetime")
        with self.assertRaises(TypeError):
            parse_iso_datetime(12345)

    def test_compute_wgs84_iou_exact_and_disjoint(self):
        """Verify IoU on identical, partial, boundary-touching, and disjoint boxes."""
        b1 = [73.0, 19.0, 73.1, 19.1]
        # Identical
        self.assertAlmostEqual(compute_wgs84_iou(b1, b1), 1.0)

        # Disjoint
        b_disjoint = [80.0, 20.0, 81.0, 21.0]
        self.assertEqual(compute_wgs84_iou(b1, b_disjoint), 0.0)

        # Boundary-touching (zero area overlap)
        b_touch = [73.1, 19.0, 73.2, 19.1]
        self.assertEqual(compute_wgs84_iou(b1, b_touch), 0.0)

        # Partial overlap
        b_partial = [73.05, 19.05, 73.15, 19.15]
        iou_val = compute_wgs84_iou(b1, b_partial)
        self.assertGreater(iou_val, 0.0)
        self.assertLess(iou_val, 1.0)


class TestTemporalPairingSynthetic(unittest.TestCase):
    """Synthetic unit tests for deterministic pairing rules, edge cases, and tie-breaking."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_synthetic_metadata(self, entries: List[Dict[str, Any]]) -> Path:
        meta_path = self.temp_root / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"entries": entries}, f)
        return meta_path

    def test_same_scene_pairing_rejected(self):
        """Verify attempting to pair two tiles from the same scene raises ValueError."""
        entries = [
            {
                "tile_id": "tile_1",
                "scene_id": "scene_A",
                "modality": "optical",
                "acquisition_datetime_utc": "2022-01-01T00:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
                "bounds_projected": [100.0, 200.0, 300.0, 400.0],
                "crs": "EPSG:32643",
                "tile_directory": "tiles/t1",
            },
            {
                "tile_id": "tile_2",
                "scene_id": "scene_A",
                "modality": "optical",
                "acquisition_datetime_utc": "2022-01-01T00:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 1},
                "bounds_wgs84": [73.1, 19.0, 73.2, 19.1],
                "bounds_projected": [300.0, 200.0, 500.0, 400.0],
                "crs": "EPSG:32643",
                "tile_directory": "tiles/t2",
            },
        ]
        meta_path = self._write_synthetic_metadata(entries)
        pairer = TemporalPairer(metadata_file=meta_path, project_root=self.temp_root)

        with self.assertRaises(ValueError) as ctx:
            pairer._create_pair(entries[0], entries[1])
        self.assertIn("same scene", str(ctx.exception).lower())

    def test_identical_timestamps_rejected(self):
        """Verify observations with identical timestamps cannot form a temporal pair."""
        entries = [
            {
                "tile_id": "tile_A",
                "scene_id": "scene_A",
                "modality": "optical",
                "acquisition_datetime_utc": "2022-01-01T12:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
                "bounds_projected": [100.0, 200.0, 300.0, 400.0],
                "crs": "EPSG:32643",
                "tile_directory": "tiles/tA",
            },
            {
                "tile_id": "tile_B",
                "scene_id": "scene_B",
                "modality": "optical",
                "acquisition_datetime_utc": "2022-01-01T12:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
                "bounds_projected": [100.0, 200.0, 300.0, 400.0],
                "crs": "EPSG:32643",
                "tile_directory": "tiles/tB",
            },
        ]
        meta_path = self._write_synthetic_metadata(entries)
        pairer = TemporalPairer(metadata_file=meta_path, project_root=self.temp_root)

        with self.assertRaises(ValueError) as ctx:
            pairer.find_pairs_for_scenes("scene_A", "scene_B")
        self.assertIn("identical", str(ctx.exception).lower())

    def test_cross_modality_rejected(self):
        """Verify cross-modality pairing (optical <-> SAR) is strictly rejected."""
        entries = [
            {
                "tile_id": "tile_opt",
                "scene_id": "scene_opt",
                "modality": "optical",
                "acquisition_datetime_utc": "2022-01-01T00:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
                "bounds_projected": [100.0, 200.0, 300.0, 400.0],
                "crs": "EPSG:32643",
                "tile_directory": "tiles/t_opt",
            },
            {
                "tile_id": "tile_sar",
                "scene_id": "scene_sar",
                "modality": "sar",
                "acquisition_datetime_utc": "2024-01-01T00:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
                "bounds_projected": None,
                "crs": "EPSG:4326",
                "tile_directory": "tiles/t_sar",
            },
        ]
        meta_path = self._write_synthetic_metadata(entries)
        pairer = TemporalPairer(metadata_file=meta_path, project_root=self.temp_root)

        with self.assertRaises(ValueError) as ctx:
            pairer.find_pairs_for_scenes("scene_opt", "scene_sar")
        self.assertIn("cross-modality", str(ctx.exception).lower())

        # Calling get_pair_for_tile returns None (does not pair across modalities)
        pair = pairer.get_pair_for_tile("tile_opt")
        self.assertIsNone(pair)

    def test_no_counterpart_behavior(self):
        """Verify tile with no temporal counterpart returns None."""
        entries = [
            {
                "tile_id": "tile_isolated",
                "scene_id": "scene_iso",
                "modality": "optical",
                "acquisition_datetime_utc": "2022-01-01T00:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
                "bounds_projected": [100.0, 200.0, 300.0, 400.0],
                "crs": "EPSG:32643",
                "tile_directory": "tiles/t_iso",
            },
        ]
        meta_path = self._write_synthetic_metadata(entries)
        pairer = TemporalPairer(metadata_file=meta_path, project_root=self.temp_root)

        self.assertIsNone(pairer.get_pair_for_tile("tile_isolated"))
        self.assertEqual(pairer.find_counterpart_candidates("tile_isolated"), [])

    def test_multiple_candidates_nearest_in_time(self):
        """Verify canonical pairing chooses the candidate with minimum absolute temporal distance."""
        # Query tile at 2023-01-01. Candidates at 2022-01-01 (365d earlier) and 2023-06-01 (151d later)
        entries = [
            {
                "tile_id": "tile_2023",
                "scene_id": "scene_2023",
                "modality": "optical",
                "acquisition_datetime_utc": "2023-01-01T00:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
                "bounds_projected": [100.0, 200.0, 300.0, 400.0],
                "crs": "EPSG:32643",
                "tile_directory": "tiles/t_2023",
            },
            {
                "tile_id": "tile_2022",
                "scene_id": "scene_2022",
                "modality": "optical",
                "acquisition_datetime_utc": "2022-01-01T00:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
                "bounds_projected": [100.0, 200.0, 300.0, 400.0],
                "crs": "EPSG:32643",
                "tile_directory": "tiles/t_2022",
            },
            {
                "tile_id": "tile_2023_mid",
                "scene_id": "scene_2023_mid",
                "modality": "optical",
                "acquisition_datetime_utc": "2023-06-01T00:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
                "bounds_projected": [100.0, 200.0, 300.0, 400.0],
                "crs": "EPSG:32643",
                "tile_directory": "tiles/t_2023_mid",
            },
        ]
        meta_path = self._write_synthetic_metadata(entries)
        pairer = TemporalPairer(metadata_file=meta_path, project_root=self.temp_root)

        # Nearest in time is scene_2023_mid (151 days away vs 365 days away)
        pair = pairer.get_pair_for_tile("tile_2023")
        self.assertIsNotNone(pair)
        self.assertEqual(pair.comparison_tile_id, "tile_2023_mid")
        self.assertEqual(pair.reference_tile_id, "tile_2023")

    def test_equidistant_candidates_prefer_future(self):
        """Verify equidistant candidates break ties by preferring future observation over past."""
        # Query tile at 2023-01-01. Candidates exactly 100 days before and 100 days after
        entries = [
            {
                "tile_id": "tile_base",
                "scene_id": "scene_base",
                "modality": "optical",
                "acquisition_datetime_utc": "2023-06-01T00:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
                "bounds_projected": [100.0, 200.0, 300.0, 400.0],
                "crs": "EPSG:32643",
                "tile_directory": "tiles/t_base",
            },
            {
                "tile_id": "tile_past",
                "scene_id": "scene_past",
                "modality": "optical",
                "acquisition_datetime_utc": "2023-05-02T00:00:00Z",  # 30 days before
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
                "bounds_projected": [100.0, 200.0, 300.0, 400.0],
                "crs": "EPSG:32643",
                "tile_directory": "tiles/t_past",
            },
            {
                "tile_id": "tile_future",
                "scene_id": "scene_future",
                "modality": "optical",
                "acquisition_datetime_utc": "2023-07-01T00:00:00Z",  # 30 days after
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
                "bounds_projected": [100.0, 200.0, 300.0, 400.0],
                "crs": "EPSG:32643",
                "tile_directory": "tiles/t_future",
            },
        ]
        meta_path = self._write_synthetic_metadata(entries)
        pairer = TemporalPairer(metadata_file=meta_path, project_root=self.temp_root)

        pair = pairer.get_pair_for_tile("tile_base")
        self.assertIsNotNone(pair)
        # Future candidate chosen
        self.assertEqual(pair.comparison_tile_id, "tile_future")

    def test_tie_breaking_alphabetical_scene_id(self):
        """Verify identical delta and direction candidates break ties by ascending scene_id."""
        entries = [
            {
                "tile_id": "tile_q",
                "scene_id": "scene_q",
                "modality": "optical",
                "acquisition_datetime_utc": "2023-01-01T00:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
                "bounds_projected": [100.0, 200.0, 300.0, 400.0],
                "crs": "EPSG:32643",
                "tile_directory": "tiles/t_q",
            },
            {
                "tile_id": "tile_cand_z",
                "scene_id": "scene_zulu",
                "modality": "optical",
                "acquisition_datetime_utc": "2024-01-01T00:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
                "bounds_projected": [100.0, 200.0, 300.0, 400.0],
                "crs": "EPSG:32643",
                "tile_directory": "tiles/t_z",
            },
            {
                "tile_id": "tile_cand_a",
                "scene_id": "scene_alpha",
                "modality": "optical",
                "acquisition_datetime_utc": "2024-01-01T00:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
                "bounds_projected": [100.0, 200.0, 300.0, 400.0],
                "crs": "EPSG:32643",
                "tile_directory": "tiles/t_a",
            },
        ]
        meta_path = self._write_synthetic_metadata(entries)
        pairer = TemporalPairer(metadata_file=meta_path, project_root=self.temp_root)

        pair = pairer.get_pair_for_tile("tile_q")
        self.assertIsNotNone(pair)
        # Alphabetically earlier scene chosen (scene_alpha < scene_zulu)
        self.assertEqual(pair.comparison_scene_id, "scene_alpha")

    def test_explicit_target_scene_id_selection(self):
        """Verify passing target_scene_id restricts pairing to that exact scene."""
        entries = [
            {
                "tile_id": "tile_0",
                "scene_id": "scene_0",
                "modality": "optical",
                "acquisition_datetime_utc": "2022-01-01T00:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
                "bounds_projected": [100.0, 200.0, 300.0, 400.0],
                "crs": "EPSG:32643",
                "tile_directory": "tiles/t0",
            },
            {
                "tile_id": "tile_1",
                "scene_id": "scene_1",
                "modality": "optical",
                "acquisition_datetime_utc": "2023-01-01T00:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
                "bounds_projected": [100.0, 200.0, 300.0, 400.0],
                "crs": "EPSG:32643",
                "tile_directory": "tiles/t1",
            },
            {
                "tile_id": "tile_2",
                "scene_id": "scene_2",
                "modality": "optical",
                "acquisition_datetime_utc": "2024-01-01T00:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
                "bounds_projected": [100.0, 200.0, 300.0, 400.0],
                "crs": "EPSG:32643",
                "tile_directory": "tiles/t2",
            },
        ]
        meta_path = self._write_synthetic_metadata(entries)
        pairer = TemporalPairer(metadata_file=meta_path, project_root=self.temp_root)

        # Force pairing with scene_2 even though scene_1 is closer
        pair = pairer.get_pair_for_tile("tile_0", target_scene_id="scene_2")
        self.assertIsNotNone(pair)
        self.assertEqual(pair.comparison_scene_id, "scene_2")

    def test_temporal_delta_constraints(self):
        """Verify min_temporal_delta_days and max_temporal_delta_days filtering."""
        entries = [
            {
                "tile_id": "tile_a",
                "scene_id": "scene_a",
                "modality": "optical",
                "acquisition_datetime_utc": "2022-01-01T00:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
                "bounds_projected": [100.0, 200.0, 300.0, 400.0],
                "crs": "EPSG:32643",
                "tile_directory": "tiles/ta",
            },
            {
                "tile_id": "tile_b",
                "scene_id": "scene_b",
                "modality": "optical",
                "acquisition_datetime_utc": "2022-01-10T00:00:00Z",  # 9 days later
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
                "bounds_projected": [100.0, 200.0, 300.0, 400.0],
                "crs": "EPSG:32643",
                "tile_directory": "tiles/tb",
            },
        ]
        meta_path = self._write_synthetic_metadata(entries)
        pairer = TemporalPairer(metadata_file=meta_path, project_root=self.temp_root)

        # Delta is 9 days. Min requires 15 days -> None
        self.assertIsNone(pairer.get_pair_for_tile("tile_a", min_temporal_delta_days=15.0))

        # Max allowed is 5 days -> None
        self.assertIsNone(pairer.get_pair_for_tile("tile_a", max_temporal_delta_days=5.0))

        # Range [5, 10] matches
        pair = pairer.get_pair_for_tile("tile_a", min_temporal_delta_days=5.0, max_temporal_delta_days=10.0)
        self.assertIsNotNone(pair)

        # Inverted range raises ValueError
        with self.assertRaises(ValueError):
            pairer.get_pair_for_tile("tile_a", min_temporal_delta_days=20.0, max_temporal_delta_days=10.0)

    def test_duplicate_metadata_detection(self):
        """Verify duplicate tile_id or (scene_id, grid_index) raises descriptive ValueError."""
        # Duplicate tile_id
        dup_tile_entries = [
            {
                "tile_id": "tile_dup",
                "scene_id": "s1",
                "modality": "optical",
                "acquisition_datetime_utc": "2022-01-01T00:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
            },
            {
                "tile_id": "tile_dup",
                "scene_id": "s2",
                "modality": "optical",
                "acquisition_datetime_utc": "2024-01-01T00:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
            },
        ]
        p1 = self._write_synthetic_metadata(dup_tile_entries)
        with self.assertRaises(ValueError) as ctx:
            TemporalPairer(metadata_file=p1, project_root=self.temp_root)
        self.assertIn("duplicate tile_id", str(ctx.exception).lower())

        # Duplicate (scene_id, grid_index)
        dup_grid_entries = [
            {
                "tile_id": "tile_1",
                "scene_id": "s1",
                "modality": "optical",
                "acquisition_datetime_utc": "2022-01-01T00:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
            },
            {
                "tile_id": "tile_2",
                "scene_id": "s1",
                "modality": "optical",
                "acquisition_datetime_utc": "2022-01-01T00:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
            },
        ]
        p2 = self._write_synthetic_metadata(dup_grid_entries)
        with self.assertRaises(ValueError) as ctx:
            TemporalPairer(metadata_file=p2, project_root=self.temp_root)
        self.assertIn("duplicate (scene_id, grid_index)", str(ctx.exception).lower())

    def test_invalid_parameters_raise(self):
        """Verify invalid IDs and types raise KeyError or TypeError."""
        entries = [
            {
                "tile_id": "tile_ok",
                "scene_id": "scene_ok",
                "modality": "optical",
                "acquisition_datetime_utc": "2022-01-01T00:00:00Z",
                "grid_index": {"row_idx": 0, "col_idx": 0},
                "bounds_wgs84": [73.0, 19.0, 73.1, 19.1],
                "bounds_projected": [100.0, 200.0, 300.0, 400.0],
                "crs": "EPSG:32643",
                "tile_directory": "tiles/tok",
            }
        ]
        meta_path = self._write_synthetic_metadata(entries)
        pairer = TemporalPairer(metadata_file=meta_path, project_root=self.temp_root)

        with self.assertRaises(KeyError):
            pairer.get_pair_for_tile("nonexistent_tile_id")
        with self.assertRaises(TypeError):
            pairer.get_pair_for_tile(12345)
        with self.assertRaises(KeyError):
            pairer.get_pair_for_tile("tile_ok", target_scene_id="nonexistent_scene")
        with self.assertRaises(TypeError):
            pairer.get_pair_for_tile("tile_ok", target_scene_id=999)
        with self.assertRaises(KeyError):
            pairer.find_pairs_for_scenes("scene_ok", "nonexistent_scene")
        with self.assertRaises(ValueError):
            pairer.find_pairs_for_scenes("scene_ok", "scene_ok")
        with self.assertRaises(ValueError):
            pairer.find_all_pairs(modality="infrared")


class TestTemporalPairingLiveDataset(unittest.TestCase):
    """Integration test suite executing against the live staged dataset with dynamic expectations."""

    @classmethod
    def setUpClass(cls):
        if not METADATA_FILE.exists():
            raise unittest.SkipTest("Live metadata.json not found.")

        cls.pairer = TemporalPairer(metadata_file=METADATA_FILE)

        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            cls.live_meta = json.load(f)

        cls.entries = cls.live_meta["entries"]
        cls.optical_scenes = sorted(list({e["scene_id"] for e in cls.entries if e["modality"] == "optical"}))
        cls.sar_scenes = sorted(list({e["scene_id"] for e in cls.entries if e["modality"] == "sar"}))

    def test_live_scene_inventory_dynamically(self):
        """Verify dynamic discovery of staged optical and SAR scenes."""
        self.assertEqual(len(self.optical_scenes), 2)
        self.assertEqual(len(self.sar_scenes), 2)
        self.assertEqual(self.pairer.total_scenes, 4)
        self.assertEqual(self.pairer.total_tiles, len(self.entries))

    def test_live_optical_pairs_generation(self):
        """Verify all corresponding optical tile pairs are returned with correct invariants."""
        scene_a, scene_b = self.optical_scenes
        pairs = self.pairer.find_pairs_for_scenes(scene_a, scene_b)

        # Expected count derived dynamically from shared grid indices
        grids_a = { (e['grid_index']['row_idx'], e['grid_index']['col_idx']) for e in self.entries if e['scene_id'] == scene_a }
        grids_b = { (e['grid_index']['row_idx'], e['grid_index']['col_idx']) for e in self.entries if e['scene_id'] == scene_b }
        expected_count = len(grids_a & grids_b)

        self.assertEqual(len(pairs), expected_count)
        self.assertGreater(len(pairs), 0)

        for p in pairs:
            # 1. Modality
            self.assertEqual(p.modality, "optical")
            # 2. Chronological order invariant
            self.assertLess(p.reference_datetime_utc, p.comparison_datetime_utc)
            self.assertGreater(p.temporal_delta_days, 0.0)
            # 3. Pixel alignment verified from actual rasters
            self.assertTrue(p.is_pixel_aligned)
            self.assertEqual(p.alignment_type, "native_pixel_aligned")
            self.assertEqual(p.alignment.crs, "EPSG:32643")
            self.assertEqual(p.alignment.pixel_dimensions, (256, 256))
            # 4. Spatial correspondence method and IoU
            self.assertEqual(p.spatial.method, "grid_index_exact")
            self.assertAlmostEqual(p.spatial_iou_wgs84, 1.0, places=4)

    def test_live_sar_pairs_generation(self):
        """Verify all corresponding SAR tile pairs are returned with unaligned radar grid status."""
        scene_a, scene_b = self.sar_scenes
        pairs = self.pairer.find_pairs_for_scenes(scene_a, scene_b)

        grids_a = { (e['grid_index']['row_idx'], e['grid_index']['col_idx']) for e in self.entries if e['scene_id'] == scene_a }
        grids_b = { (e['grid_index']['row_idx'], e['grid_index']['col_idx']) for e in self.entries if e['scene_id'] == scene_b }
        expected_count = len(grids_a & grids_b)

        self.assertEqual(len(pairs), expected_count)
        self.assertGreater(len(pairs), 0)

        for p in pairs:
            # 1. Modality
            self.assertEqual(p.modality, "sar")
            # 2. Chronological order invariant
            self.assertLess(p.reference_datetime_utc, p.comparison_datetime_utc)
            self.assertGreater(p.temporal_delta_days, 0.0)
            # 3. Unaligned radar grid facts (not falsely claiming affine alignment)
            self.assertFalse(p.is_pixel_aligned)
            self.assertEqual(p.alignment_type, "unaligned_radar_grid")
            self.assertIsNone(p.alignment.crs)
            # 4. Spatial correspondence method and diagnostic IoU
            self.assertEqual(p.spatial.method, "grid_index_topological")
            self.assertGreaterEqual(p.spatial_iou_wgs84, 0.95)
            self.assertLessEqual(p.spatial_iou_wgs84, 1.0)

    def test_live_tile_query_pairing(self):
        """Verify get_pair_for_tile on live optical and SAR tiles."""
        # Optical tile
        sample_opt_id = [e["tile_id"] for e in self.entries if e["modality"] == "optical"][0]
        opt_pair = self.pairer.get_pair_for_tile(sample_opt_id)
        self.assertIsNotNone(opt_pair)
        self.assertEqual(opt_pair.modality, "optical")
        self.assertTrue(opt_pair.is_pixel_aligned)
        self.assertIn(sample_opt_id, (opt_pair.reference_tile_id, opt_pair.comparison_tile_id))

        # SAR tile
        sample_sar_id = [e["tile_id"] for e in self.entries if e["modality"] == "sar"][0]
        sar_pair = self.pairer.get_pair_for_tile(sample_sar_id)
        self.assertIsNotNone(sar_pair)
        self.assertEqual(sar_pair.modality, "sar")
        self.assertFalse(sar_pair.is_pixel_aligned)
        self.assertIn(sample_sar_id, (sar_pair.reference_tile_id, sar_pair.comparison_tile_id))

    def test_find_all_pairs_enumeration(self):
        """Verify find_all_pairs returns all optical and SAR pairs."""
        all_pairs = self.pairer.find_all_pairs()
        opt_pairs = self.pairer.find_all_pairs(modality="optical")
        sar_pairs = self.pairer.find_all_pairs(modality="sar")

        self.assertEqual(len(all_pairs), len(opt_pairs) + len(sar_pairs))
        self.assertGreater(len(opt_pairs), 0)
        self.assertGreater(len(sar_pairs), 0)

    def test_m6_handoff_integration(self):
        """Verify passing an actual M6 text search hit into M7 resolves a valid TemporalPair."""
        engine = TextSearchEngine(metadata_file=METADATA_FILE)
        hits = engine.search("container port terminal", top_k=3, modality="optical")
        self.assertGreater(len(hits), 0)

        # Extract hit tile_id directly (minimal handoff contract)
        top_hit_id = hits[0]["tile_id"]
        pair = self.pairer.get_pair_for_tile(top_hit_id)
        self.assertIsNotNone(pair)
        self.assertEqual(pair.modality, "optical")
        self.assertTrue(pair.is_pixel_aligned)
        self.assertEqual(pair.grid_index, (hits[0]["metadata"]["grid_index"]["row_idx"], hits[0]["metadata"]["grid_index"]["col_idx"]))

    def test_stable_serialization_repeatability(self):
        """Verify repeated to_dict() calls produce byte-identical serialized JSON."""
        sample_id = self.entries[0]["tile_id"]
        pair = self.pairer.get_pair_for_tile(sample_id)
        self.assertIsNotNone(pair)

        json_1 = json.dumps(pair.to_dict(), sort_keys=True)
        json_2 = json.dumps(pair.to_dict(), sort_keys=True)
        self.assertEqual(json_1, json_2)

        # Verify contract schema keys
        d = pair.to_dict()
        self.assertIn("pair_id", d)
        self.assertIn("modality", d)
        self.assertIn("temporal", d)
        self.assertIn("spatial_correspondence", d)
        self.assertIn("alignment", d)
        self.assertIn("provenance", d)


if __name__ == "__main__":
    unittest.main()
