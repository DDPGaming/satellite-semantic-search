"""Unit and integration tests for Milestone 9: False-Alarm Suppression and Confidence.

Covers:
1. Config validation (parameter constraints, rejection of invalid inputs)
2. Zero-change array (all 0.0 -> zero candidates, zero confirmed)
3. Uniform-change array (identical positive value -> zero candidates)
4. MAD == 0 with minority anomaly (dominant median, anomalies become candidates)
5. Normal synthetic distribution (Gaussian noise + injected clustered change)
6. Exact threshold calculation with known numerical inputs
7. min_threshold_offset enforcement when scale is zero or tiny
8. min_valid_pixels guardrail -> insufficient_valid_pixels status
9. All-invalid pixels -> no_valid_pixels status
10. NaN and Inf handling (exclusion from stats, zero confidence)
11. Isolated candidate pixel suppression (neighbor_count < min_neighbors)
12. Coherent 4x4 region retention
13. Minimum region area filtering (connected component area < min_region_area)
14. Configurable min_neighbors setting
15. Thin line suppression / endpoint behavior
16. Border and corner candidates (safe padding without IndexError)
17. Multiple disconnected components with independent area filtering
18. Confidence bounds: confirmed in [0.05, 1.0], unconfirmed/invalid == 0.0
19. Confidence monotonicity (higher magnitude -> higher conf; more neighbors -> higher conf)
20. Zero denominator safety in spectral margin
21. SAR deferred behavior (unsupported_sar_unaligned status pass-through)
22. M8 upstream error-status pass-through
23. SuppressedChangeResult JSON serialization (to_dict, exclusion of arrays)
24. Provenance tracking (algorithm, parameters, noise stats, counts)
25. GeoTIFF mask and confidence export (CRS, transform, dimensions, nodata)
26. Deterministic repeated execution
27. End-to-end M7 -> M8 -> M9 live optical tile integration
28. Batch streaming execution (suppress_batch)
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.change_detection import ChangeDetectionConfig, ChangeDetector, ChangeResult
from src.pairing.pairer import TemporalPairer
from src.suppression import (
    FalseAlarmSuppressor,
    SuppressedChangeResult,
    SuppressionConfig,
)
from src.suppression.suppressor import (
    count_8_neighbors,
    filter_min_region_area_8,
    label_connected_components_8,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_dummy_change_result(
    change_magnitude: Optional[np.ndarray],
    status: str = "success",
    modality: str = "optical",
    pair_id: str = "test_optical_pair_r03_c04",
    change_map_path: Optional[str] = None,
    warnings: Tuple[str, ...] = (),
) -> ChangeResult:
    """Helper to construct an immutable M8 ChangeResult for testing."""
    return ChangeResult(
        pair_id=pair_id,
        modality=modality,
        reference_tile_id="s2_ref_tile_01",
        comparison_tile_id="s2_comp_tile_01",
        reference_scene_id="S2A_ref_scene",
        comparison_scene_id="S2B_comp_scene",
        reference_datetime_utc="2022-01-27T05:30:00+00:00",
        comparison_datetime_utc="2024-01-12T05:30:00+00:00",
        temporal_delta_days=715.0,
        grid_index=(3, 4),
        status=status,
        method="spectral_distance_l2" if status == "success" else "unsupported",
        summary_statistics={"mean": 0.05, "median": 0.05, "valid_pixel_ratio": 1.0} if status == "success" else None,
        change_map_path=change_map_path,
        change_magnitude=change_magnitude,
        warnings=warnings,
        provenance={"upstream_test": True},
    )


class TestSuppressionConfig(unittest.TestCase):
    """Test suite for SuppressionConfig validation and serialization."""

    def test_default_config_valid(self):
        cfg = SuppressionConfig()
        self.assertEqual(cfg.sensitivity_k, 3.0)
        self.assertEqual(cfg.min_threshold_offset, 0.0)
        self.assertEqual(cfg.min_neighbors, 2)
        self.assertEqual(cfg.min_region_area, 4)
        self.assertEqual(cfg.min_valid_pixels, 100)
        self.assertFalse(cfg.save_masks)
        self.assertIsNone(cfg.output_dir)

    def test_invalid_sensitivity_k(self):
        with self.assertRaises(ValueError):
            SuppressionConfig(sensitivity_k=-1.0)
        with self.assertRaises(ValueError):
            SuppressionConfig(sensitivity_k=float("nan"))
        with self.assertRaises(ValueError):
            SuppressionConfig(sensitivity_k=float("inf"))
        with self.assertRaises(TypeError):
            SuppressionConfig(sensitivity_k="invalid")
        with self.assertRaises(TypeError):
            SuppressionConfig(sensitivity_k=True)

    def test_invalid_min_threshold_offset(self):
        with self.assertRaises(ValueError):
            SuppressionConfig(min_threshold_offset=-0.05)
        with self.assertRaises(ValueError):
            SuppressionConfig(min_threshold_offset=float("nan"))
        with self.assertRaises(ValueError):
            SuppressionConfig(min_threshold_offset=float("inf"))
        with self.assertRaises(TypeError):
            SuppressionConfig(min_threshold_offset="invalid")
        with self.assertRaises(TypeError):
            SuppressionConfig(min_threshold_offset=False)

    def test_invalid_min_neighbors(self):
        with self.assertRaises(ValueError):
            SuppressionConfig(min_neighbors=-1)
        with self.assertRaises(ValueError):
            SuppressionConfig(min_neighbors=9)
        with self.assertRaises(TypeError):
            SuppressionConfig(min_neighbors=2.5)
        with self.assertRaises(TypeError):
            SuppressionConfig(min_neighbors=True)

    def test_invalid_min_region_area(self):
        with self.assertRaises(ValueError):
            SuppressionConfig(min_region_area=0)
        with self.assertRaises(TypeError):
            SuppressionConfig(min_region_area=True)
        with self.assertRaises(TypeError):
            SuppressionConfig(min_region_area=4.5)

    def test_invalid_min_valid_pixels(self):
        with self.assertRaises(ValueError):
            SuppressionConfig(min_valid_pixels=0)
        with self.assertRaises(TypeError):
            SuppressionConfig(min_valid_pixels=True)
        with self.assertRaises(TypeError):
            SuppressionConfig(min_valid_pixels="100")

    def test_invalid_save_masks_and_output_dir(self):
        with self.assertRaises(TypeError):
            SuppressionConfig(save_masks="True")
        with self.assertRaises(TypeError):
            SuppressionConfig(output_dir=12345)

    def test_to_dict_serialization(self):
        cfg = SuppressionConfig(sensitivity_k=2.5, min_threshold_offset=0.01, min_neighbors=3)
        d = cfg.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["sensitivity_k"], 2.5)
        self.assertEqual(d["min_threshold_offset"], 0.01)
        self.assertEqual(d["min_neighbors"], 3)
        # Verify JSON serializability
        serialized = json.dumps(d)
        self.assertIn("sensitivity_k", serialized)


class TestSuppressionAlgorithms(unittest.TestCase):
    """Test suite for spatial neighborhood and connected component algorithms."""

    def test_count_8_neighbors_isolated_and_dense(self):
        mask = np.zeros((5, 5), dtype=bool)
        # Isolated pixel
        mask[0, 0] = True
        # 3x3 block in bottom right
        mask[2:5, 2:5] = True

        counts = count_8_neighbors(mask)
        # Top-left isolated corner has 0 neighbors
        self.assertEqual(counts[0, 0], 0)
        # Center of 3x3 block has 8 neighbors
        self.assertEqual(counts[3, 3], 8)
        # Corner of 3x3 block has 3 neighbors
        self.assertEqual(counts[2, 2], 3)
        self.assertEqual(counts[4, 4], 3)
        # Edge of 3x3 block has 5 neighbors
        self.assertEqual(counts[3, 2], 5)

    def test_label_and_filter_connected_components(self):
        mask = np.zeros((10, 10), dtype=bool)
        # Component 1: 2x2 = 4 pixels
        mask[1:3, 1:3] = True
        # Component 2: 1 pixel
        mask[6, 6] = True
        # Component 3: 2 pixels diagonally adjacent (8-connectivity)
        mask[8, 8] = True
        mask[9, 9] = True

        labeled, num_components = label_connected_components_8(mask)
        self.assertEqual(num_components, 3)

        # Filter area >= 4: only Component 1 should survive
        filtered = filter_min_region_area_8(mask, min_area=4)
        self.assertEqual(np.sum(filtered), 4)
        self.assertTrue(np.all(filtered[1:3, 1:3]))
        self.assertFalse(filtered[6, 6])
        self.assertFalse(filtered[8, 8])
        self.assertFalse(filtered[9, 9])


class TestFalseAlarmSuppressor(unittest.TestCase):
    """Comprehensive test suite for FalseAlarmSuppressor covering all 27+ requirements."""

    def setUp(self):
        self.suppressor = FalseAlarmSuppressor(project_root=PROJECT_ROOT)

    def test_02_zero_change_array(self):
        """Zero-change array produces zero candidates and zero confirmed change."""
        arr = np.zeros((256, 256), dtype=np.float32)
        cr = make_dummy_change_result(arr)
        res = self.suppressor.suppress(cr)

        self.assertEqual(res.status, "success")
        self.assertEqual(res.noise_median, 0.0)
        self.assertEqual(res.noise_mad, 0.0)
        self.assertEqual(res.threshold_used, 0.0)
        self.assertEqual(res.candidate_pixels_count, 0)
        self.assertEqual(res.confirmed_pixels_count, 0)
        self.assertEqual(res.suppressed_pixels_count, 0)
        self.assertEqual(res.confirmed_change_ratio, 0.0)
        self.assertIsNone(res.mean_confidence_on_change)

    def test_03_uniform_change_array(self):
        """Uniform positive change array produces zero candidates (x > median is False everywhere)."""
        arr = np.full((256, 256), 0.25, dtype=np.float32)
        cr = make_dummy_change_result(arr)
        res = self.suppressor.suppress(cr)

        self.assertEqual(res.status, "success")
        self.assertEqual(res.noise_median, 0.25)
        self.assertEqual(res.noise_mad, 0.0)
        self.assertEqual(res.threshold_used, 0.25)
        self.assertEqual(res.candidate_pixels_count, 0)
        self.assertEqual(res.confirmed_pixels_count, 0)
        self.assertEqual(res.suppressed_pixels_count, 0)
        self.assertEqual(res.confirmed_change_ratio, 0.0)

    def test_04_mad_zero_with_minority_anomaly(self):
        """When MAD is 0 but an anomaly exists, threshold is median and anomaly becomes candidate."""
        arr = np.full((256, 256), 0.05, dtype=np.float32)
        # Inject a 4x4 block of 0.30 (16 pixels < 50% so median remains 0.05)
        arr[50:54, 50:54] = 0.30
        cr = make_dummy_change_result(arr)
        res = self.suppressor.suppress(cr)

        self.assertEqual(res.status, "success")
        self.assertEqual(res.noise_median, 0.05)
        self.assertEqual(res.noise_mad, 0.0)
        self.assertEqual(res.threshold_used, 0.05)
        self.assertEqual(res.candidate_pixels_count, 16)
        self.assertEqual(res.confirmed_pixels_count, 16)
        self.assertEqual(res.suppressed_pixels_count, 0)
        self.assertGreater(res.mean_confidence_on_change, 0.05)

    def test_05_normal_synthetic_distribution(self):
        """Normal synthetic background noise with injected clustered change."""
        rng = np.random.default_rng(42)
        # Background: normal noise around 0.05 with small std 0.005
        arr = np.clip(rng.normal(loc=0.05, scale=0.005, size=(256, 256)).astype(np.float32), 0.0, None)
        # Inject cohesive change cluster of 8x8 pixels at value 0.20
        arr[100:108, 100:108] = 0.20

        cr = make_dummy_change_result(arr)
        res = self.suppressor.suppress(cr)

        self.assertEqual(res.status, "success")
        self.assertAlmostEqual(res.noise_median, 0.05, delta=0.002)
        self.assertGreater(res.noise_mad, 0.0)
        self.assertGreater(res.threshold_used, res.noise_median)
        # The 64 change pixels should be confirmed
        self.assertGreaterEqual(res.confirmed_pixels_count, 64)

    def test_06_exact_threshold_calculation(self):
        """Explicit numerical threshold calculation matches mathematical formula."""
        # Create an array where median and MAD are exactly known
        # Array with 70% 0.10, 15% 0.08, 15% 0.12
        arr = np.full((100, 100), 0.10, dtype=np.float32)
        arr[:15, :100] = 0.08
        arr[85:, :100] = 0.12
        # median is 0.10, abs deviations: 70% 0.0, 30% 0.02 -> MAD is 0.0
        # Let's adjust so MAD is 0.02: 50% 0.10, 50% 0.12 -> median is 0.11, abs dev is 0.01
        arr = np.array([0.08, 0.09, 0.10, 0.11, 0.12] * 200, dtype=np.float32).reshape(100, 10)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        expected_thr = med + max(2.0 * 1.4826 * mad, 0.0)

        cfg = SuppressionConfig(sensitivity_k=2.0, min_threshold_offset=0.0)
        cr = make_dummy_change_result(arr)
        res = self.suppressor.suppress(cr, config=cfg)

        self.assertAlmostEqual(res.threshold_used, expected_thr, places=5)

    def test_07_min_threshold_offset(self):
        """min_threshold_offset is enforced when k * robust_scale < min_threshold_offset."""
        arr = np.full((256, 256), 0.10, dtype=np.float32)
        cfg = SuppressionConfig(sensitivity_k=3.0, min_threshold_offset=0.08)
        cr = make_dummy_change_result(arr)
        res = self.suppressor.suppress(cr, config=cfg)

        # Since MAD is 0, k * robust_scale = 0 < 0.08, so threshold = median + 0.08 = 0.18
        self.assertEqual(res.threshold_used, 0.18)

    def test_08_min_valid_pixels_guardrail(self):
        """Fewer valid pixels than min_valid_pixels returns insufficient_valid_pixels."""
        arr = np.full((256, 256), np.nan, dtype=np.float32)
        # Only 50 valid pixels
        arr[:5, :10] = 0.15
        cr = make_dummy_change_result(arr)
        cfg = SuppressionConfig(min_valid_pixels=100)
        res = self.suppressor.suppress(cr, config=cfg)

        self.assertEqual(res.status, "insufficient_valid_pixels")
        self.assertEqual(res.candidate_pixels_count, 0)
        self.assertEqual(res.confirmed_pixels_count, 0)
        self.assertTrue(any("below minimum" in w for w in res.warnings))

    def test_09_all_invalid_pixels(self):
        """All NaN pixels return no_valid_pixels status."""
        arr = np.full((256, 256), np.nan, dtype=np.float32)
        cr = make_dummy_change_result(arr)
        res = self.suppressor.suppress(cr)

        self.assertEqual(res.status, "no_valid_pixels")
        self.assertEqual(res.candidate_pixels_count, 0)
        self.assertEqual(res.confirmed_pixels_count, 0)
        self.assertTrue(any("Zero valid pixels" in w for w in res.warnings))

    def test_10_nan_inf_handling(self):
        """NaN and Inf values are safely excluded from statistics and masks."""
        arr = np.full((256, 256), 0.05, dtype=np.float32)
        arr[0, 0] = np.nan
        arr[0, 1] = np.inf
        arr[0, 2] = -np.inf
        # Cohesive change cluster elsewhere
        arr[50:54, 50:54] = 0.30

        cr = make_dummy_change_result(arr)
        res = self.suppressor.suppress(cr)

        self.assertEqual(res.status, "success")
        self.assertFalse(res.confirmed_mask[0, 0])
        self.assertFalse(res.confirmed_mask[0, 1])
        self.assertFalse(res.confirmed_mask[0, 2])
        self.assertEqual(res.confidence_map[0, 0], 0.0)
        self.assertEqual(res.confidence_map[0, 1], 0.0)
        self.assertEqual(res.confidence_map[0, 2], 0.0)

    def test_11_isolated_pixel_suppression(self):
        """Isolated single candidate pixels (neighbor_count < 2) are suppressed."""
        arr = np.full((256, 256), 0.05, dtype=np.float32)
        # Single isolated candidate at (20, 20)
        arr[20, 20] = 0.50

        cr = make_dummy_change_result(arr)
        res = self.suppressor.suppress(cr)

        self.assertEqual(res.status, "success")
        self.assertEqual(res.candidate_pixels_count, 1)
        self.assertEqual(res.suppressed_pixels_count, 1)
        self.assertEqual(res.confirmed_pixels_count, 0)
        self.assertFalse(res.confirmed_mask[20, 20])

    def test_12_coherent_4x4_region_retention(self):
        """Coherent 4x4 candidate region (16 pixels) is retained."""
        arr = np.full((256, 256), 0.05, dtype=np.float32)
        arr[10:14, 10:14] = 0.40  # 4x4 block

        cr = make_dummy_change_result(arr)
        res = self.suppressor.suppress(cr)

        self.assertEqual(res.status, "success")
        self.assertEqual(res.candidate_pixels_count, 16)
        self.assertEqual(res.confirmed_pixels_count, 16)
        self.assertEqual(res.suppressed_pixels_count, 0)
        self.assertTrue(np.all(res.confirmed_mask[10:14, 10:14]))

    def test_13_minimum_region_area_filter(self):
        """Small region (area 3 < min_region_area 4) is completely removed by area filter."""
        arr = np.full((256, 256), 0.05, dtype=np.float32)
        # Line of 3 pixels (each intermediate pixel has 2 neighbors, but total area is 3)
        # Triangle of 3 pixels: (10, 10), (10, 11), (11, 10) - each has 2 neighbors!
        arr[10, 10] = 0.40
        arr[10, 11] = 0.40
        arr[11, 10] = 0.40

        cfg = SuppressionConfig(min_neighbors=2, min_region_area=4)
        cr = make_dummy_change_result(arr)
        res = self.suppressor.suppress(cr, config=cfg)

        self.assertEqual(res.status, "success")
        self.assertEqual(res.candidate_pixels_count, 3)
        self.assertEqual(res.suppressed_pixels_count, 3)
        self.assertEqual(res.confirmed_pixels_count, 0)

    def test_14_minimum_neighbor_setting(self):
        """Higher min_neighbors suppresses sparse pixels and retains dense cores."""
        arr = np.full((256, 256), 0.05, dtype=np.float32)
        # 3x3 block (corners have 3 neighbors, edges have 5, center has 8)
        arr[10:13, 10:13] = 0.40

        # With min_neighbors=4 and min_region_area=1: corners (3 neighbors) should be suppressed
        cfg = SuppressionConfig(min_neighbors=4, min_region_area=1)
        cr = make_dummy_change_result(arr)
        res = self.suppressor.suppress(cr, config=cfg)

        self.assertEqual(res.candidate_pixels_count, 9)
        self.assertEqual(res.confirmed_pixels_count, 5)  # 1 center + 4 edges
        self.assertEqual(res.suppressed_pixels_count, 4)  # 4 corners suppressed

    def test_15_thin_line_behavior(self):
        """1-pixel wide line of length 10: intermediate pixels retained, endpoints suppressed."""
        arr = np.full((256, 256), 0.05, dtype=np.float32)
        # Horizontal line from col 20 to 29 (length 10) at row 50
        arr[50, 20:30] = 0.40

        cfg = SuppressionConfig(min_neighbors=2, min_region_area=4)
        cr = make_dummy_change_result(arr)
        res = self.suppressor.suppress(cr, config=cfg)

        # Endpoints at col 20 and 29 have only 1 neighbor -> suppressed by min_neighbors=2
        # Cols 21 to 28 (8 pixels) have 2 neighbors and area 8 >= 4 -> retained!
        self.assertEqual(res.candidate_pixels_count, 10)
        self.assertEqual(res.confirmed_pixels_count, 8)
        self.assertEqual(res.suppressed_pixels_count, 2)
        self.assertFalse(res.confirmed_mask[50, 20])
        self.assertFalse(res.confirmed_mask[50, 29])
        self.assertTrue(np.all(res.confirmed_mask[50, 21:29]))

    def test_16_border_and_corner_behavior(self):
        """Candidates on tile corners and borders do not raise IndexError or wrap around."""
        arr = np.full((256, 256), 0.05, dtype=np.float32)
        # 2x2 on top-left corner
        arr[0:2, 0:2] = 0.40
        # 2x2 on bottom-right corner
        arr[254:256, 254:256] = 0.40

        cfg = SuppressionConfig(min_neighbors=2, min_region_area=4)
        cr = make_dummy_change_result(arr)
        res = self.suppressor.suppress(cr, config=cfg)

        self.assertEqual(res.status, "success")
        self.assertEqual(res.candidate_pixels_count, 8)
        self.assertEqual(res.confirmed_pixels_count, 8)
        self.assertTrue(np.all(res.confirmed_mask[0:2, 0:2]))
        self.assertTrue(np.all(res.confirmed_mask[254:256, 254:256]))

    def test_17_multiple_connected_components(self):
        """Multiple disconnected components are filtered independently based on area."""
        arr = np.full((256, 256), 0.05, dtype=np.float32)
        # Component 1: 3x2 = 6 pixels (>= 4)
        arr[10:13, 10:12] = 0.40
        # Component 2: 2x1 = 2 pixels (< 4)
        arr[50:52, 50] = 0.40

        cfg = SuppressionConfig(min_neighbors=1, min_region_area=4)
        cr = make_dummy_change_result(arr)
        res = self.suppressor.suppress(cr, config=cfg)

        self.assertEqual(res.candidate_pixels_count, 8)
        self.assertEqual(res.confirmed_pixels_count, 6)
        self.assertEqual(res.suppressed_pixels_count, 2)
        self.assertTrue(np.all(res.confirmed_mask[10:13, 10:12]))
        self.assertFalse(np.any(res.confirmed_mask[50:52, 50]))

    def test_18_confidence_bounds(self):
        """Confirmed pixels have confidence in [0.05, 1.0]; unconfirmed/invalid pixels are 0.0."""
        arr = np.full((256, 256), 0.05, dtype=np.float32)
        arr[20:25, 20:25] = 0.35  # 5x5 block
        cr = make_dummy_change_result(arr)
        res = self.suppressor.suppress(cr)

        conf = res.confidence_map
        mask = res.confirmed_mask
        self.assertTrue(np.all(conf[mask] >= 0.05))
        self.assertTrue(np.all(conf[mask] <= 1.0))
        self.assertTrue(np.all(conf[~mask] == 0.0))

    def test_19_confidence_monotonicity(self):
        """Confidence increases monotonically with spectral magnitude and spatial density."""
        arr = np.full((256, 256), 0.05, dtype=np.float32)
        # Two 4x4 blocks at different magnitudes: 0.20 vs 0.80
        arr[20:24, 20:24] = 0.20
        arr[40:44, 40:44] = 0.80

        cr = make_dummy_change_result(arr)
        res = self.suppressor.suppress(cr)

        # Center pixel of block 1 vs center pixel of block 2 (identical neighbor density = 8)
        conf_low = res.confidence_map[21, 21]
        conf_high = res.confidence_map[41, 41]
        self.assertGreater(conf_high, conf_low)

        # Within the same block (0.80), center (8 neighbors) has higher confidence than corner (3 neighbors)
        conf_corner = res.confidence_map[40, 40]
        self.assertGreater(conf_high, conf_corner)

    def test_20_zero_denominator_safety(self):
        """Division by zero is safely handled when delta + threshold is zero."""
        arr = np.zeros((256, 256), dtype=np.float32)
        cr = make_dummy_change_result(arr)
        res = self.suppressor.suppress(cr)

        self.assertEqual(res.status, "success")
        self.assertTrue(np.all(res.confidence_map == 0.0))

    def test_21_sar_deferred_behavior(self):
        """SAR ChangeResult returns unsupported_sar_unaligned without computing false alarms."""
        arr = np.ones((256, 256), dtype=np.float32)
        cr = make_dummy_change_result(arr, modality="sar", status="unsupported_sar_unaligned")
        res = self.suppressor.suppress(cr)

        self.assertEqual(res.status, "unsupported_sar_unaligned")
        self.assertEqual(res.method, "unsupported")
        self.assertEqual(res.candidate_pixels_count, 0)
        self.assertEqual(res.confirmed_pixels_count, 0)
        self.assertIsNone(res.threshold_used)
        self.assertTrue(any("SAR modality is unsupported" in w for w in res.warnings))
        self.assertFalse(res.provenance["suppression_applied"])

    def test_22_m8_error_status_pass_through(self):
        """Non-success M8 status passes through cleanly without attempting suppression."""
        cr = make_dummy_change_result(None, status="dimension_mismatch", warnings=("Dimensions differ",))
        res = self.suppressor.suppress(cr)

        self.assertEqual(res.status, "dimension_mismatch")
        self.assertEqual(res.method, "unsupported")
        self.assertIsNone(res.threshold_used)
        self.assertFalse(res.provenance["suppression_applied"])
        self.assertIn("Dimensions differ", res.warnings)

    def test_23_result_serialization(self):
        """to_dict() produces valid JSON-serializable dictionary without numpy arrays."""
        arr = np.full((256, 256), 0.05, dtype=np.float32)
        arr[10:14, 10:14] = 0.30
        cr = make_dummy_change_result(arr)
        res = self.suppressor.suppress(cr)

        d = res.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["pair_id"], res.pair_id)
        self.assertEqual(d["grid_index"], {"row_idx": 3, "col_idx": 4})
        self.assertEqual(d["candidate_pixels_count"], 16)
        self.assertEqual(d["confirmed_pixels_count"], 16)

        # Ensure json.dumps succeeds and no array strings are present
        serialized = json.dumps(d)
        self.assertNotIn("ndarray", serialized)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized["pair_id"], res.pair_id)

    def test_24_provenance_tracking(self):
        """Provenance tracks all configuration and statistical variables."""
        arr = np.full((256, 256), 0.05, dtype=np.float32)
        arr[10:14, 10:14] = 0.30
        cr = make_dummy_change_result(arr)
        res = self.suppressor.suppress(cr)

        prov = res.provenance
        self.assertEqual(prov["suppression_method"], "adaptive_mad_suppression")
        self.assertEqual(prov["sensitivity_k"], 3.0)
        self.assertEqual(prov["min_region_area"], 4)
        self.assertEqual(prov["valid_pixels_count"], 65536)
        self.assertEqual(prov["candidate_pixels_count"], 16)
        self.assertEqual(prov["confirmed_pixels_count"], 16)
        self.assertTrue(prov["suppression_applied"])

    def test_25_geotiff_export(self):
        """GeoTIFF export writes valid mask and confidence rasters with correct metadata."""
        temp_dir = tempfile.mkdtemp()
        try:
            arr = np.full((256, 256), 0.05, dtype=np.float32)
            arr[10:15, 10:15] = 0.40  # 5x5 block = 25 pixels
            cr = make_dummy_change_result(arr)

            cfg = SuppressionConfig(save_masks=True, output_dir=temp_dir)
            res = self.suppressor.suppress(cr, config=cfg)

            self.assertIsNotNone(res.mask_raster_path)
            self.assertIsNotNone(res.confidence_raster_path)

            mask_path = Path(temp_dir) / f"mask__{cr.pair_id}.tif"
            conf_path = Path(temp_dir) / f"confidence__{cr.pair_id}.tif"

            self.assertTrue(mask_path.exists())
            self.assertTrue(conf_path.exists())

            with rasterio.open(mask_path) as src_mask:
                self.assertEqual(src_mask.shape, (256, 256))
                self.assertEqual(src_mask.dtypes[0], "uint8")
                self.assertEqual(src_mask.nodata, 0)
                mask_data = src_mask.read(1)
                self.assertEqual(np.sum(mask_data), 25)

            with rasterio.open(conf_path) as src_conf:
                self.assertEqual(src_conf.shape, (256, 256))
                self.assertEqual(src_conf.dtypes[0], "float32")
                self.assertEqual(src_conf.nodata, 0.0)
                conf_data = src_conf.read(1)
                self.assertTrue(np.all(conf_data[10:15, 10:15] >= 0.05))
                self.assertEqual(conf_data[0, 0], 0.0)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_25b_geotiff_export_failure_handling(self):
        """Export failure produces a structured warning rather than crashing or corrupting result."""
        # Using an invalid path that cannot be written to as a directory (e.g. an existing regular file)
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        try:
            arr = np.full((256, 256), 0.05, dtype=np.float32)
            arr[10:14, 10:14] = 0.40
            cr = make_dummy_change_result(arr)

            # Using the file as output_dir causes mkdir to fail with NotADirectoryError / FileExistsError
            cfg = SuppressionConfig(save_masks=True, output_dir=temp_file.name)
            res = self.suppressor.suppress(cr, config=cfg)

            # Result is still returned successfully with valid counts and status
            self.assertEqual(res.status, "success")
            self.assertEqual(res.confirmed_pixels_count, 16)
            self.assertIsNone(res.mask_raster_path)
            self.assertIsNone(res.confidence_raster_path)
            self.assertTrue(any("GeoTIFF export failed" in w for w in res.warnings))
        finally:
            Path(temp_file.name).unlink(missing_ok=True)

    def test_26_deterministic_repeated_execution(self):
        """Running suppression twice on identical input produces bitwise identical results."""
        arr = np.full((256, 256), 0.05, dtype=np.float32)
        arr[30:35, 30:35] = 0.45
        cr = make_dummy_change_result(arr)

        res1 = self.suppressor.suppress(cr)
        res2 = self.suppressor.suppress(cr)

        self.assertEqual(res1.to_dict(), res2.to_dict())
        np.testing.assert_array_equal(res1.confirmed_mask, res2.confirmed_mask)
        np.testing.assert_array_equal(res1.confidence_map, res2.confidence_map)

    def test_27_live_optical_m7_m8_m9_integration(self):
        """End-to-end integration test: M7 TemporalPairer -> M8 ChangeDetector -> M9 FalseAlarmSuppressor."""
        pairer = TemporalPairer(project_root=PROJECT_ROOT)
        detector = ChangeDetector(project_root=PROJECT_ROOT)

        # Query a real optical tile in the repository
        tile_id = "s2_S2A_43QBB_20220127_0_L2A_10m_r03_c04"
        pair = pairer.get_pair_for_tile(tile_id)
        self.assertIsNotNone(pair)

        # Run M8 change detection
        m8_cfg = ChangeDetectionConfig(mask_clouds=True, minimum_valid_pixel_ratio=0.10)
        raw_change = detector.detect_change(pair, config=m8_cfg)
        self.assertEqual(raw_change.status, "success")
        self.assertIsNotNone(raw_change.change_magnitude)

        # Run M9 false-alarm suppression
        m9_cfg = SuppressionConfig(sensitivity_k=3.0, min_neighbors=2, min_region_area=4)
        suppressed = self.suppressor.suppress(raw_change, config=m9_cfg)

        self.assertEqual(suppressed.status, "success")
        self.assertEqual(suppressed.modality, "optical")
        self.assertIsNotNone(suppressed.noise_median)
        self.assertIsNotNone(suppressed.noise_mad)
        self.assertIsNotNone(suppressed.threshold_used)
        self.assertGreaterEqual(suppressed.candidate_pixels_count, 0)
        self.assertGreaterEqual(suppressed.confirmed_pixels_count, 0)
        self.assertGreaterEqual(suppressed.suppressed_pixels_count, 0)
        self.assertEqual(
            suppressed.candidate_pixels_count,
            suppressed.confirmed_pixels_count + suppressed.suppressed_pixels_count,
        )

        # Verify JSON serializability
        d = suppressed.to_dict()
        self.assertEqual(d["status"], "success")
        json_str = json.dumps(d)
        self.assertIn("adaptive_mad_suppression", json_str)

    def test_28_batch_streaming_execution(self):
        """Batch processing yields matching SuppressedChangeResult items in identical order."""
        arr1 = np.full((256, 256), 0.05, dtype=np.float32)
        arr1[10:14, 10:14] = 0.30
        arr2 = np.full((256, 256), 0.08, dtype=np.float32)
        arr2[20:24, 20:24] = 0.40

        cr1 = make_dummy_change_result(arr1, pair_id="pair_01")
        cr2 = make_dummy_change_result(arr2, pair_id="pair_02")

        batch_results = list(self.suppressor.suppress_batch([cr1, cr2]))
        self.assertEqual(len(batch_results), 2)
        self.assertEqual(batch_results[0].pair_id, "pair_01")
        self.assertEqual(batch_results[1].pair_id, "pair_02")
        self.assertEqual(batch_results[0].confirmed_pixels_count, 16)
        self.assertEqual(batch_results[1].confirmed_pixels_count, 16)


if __name__ == "__main__":
    unittest.main()
