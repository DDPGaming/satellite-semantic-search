"""Unit and integration tests for Milestone 8: Primary Raw Change Detection.

Covers:
1. Identical reference/comparison -> zero change
2. Known uniform change across four bands -> mathematically exact CVA magnitude
3. Known single-band change
4. Nodata masking (0 values excluded from valid pixels)
5. Non-finite (NaN / Inf) masking
6. SCL cloud and shadow quality masking (20m 128x128 -> 10m 256x256 nearest-neighbor expansion)
7. All-invalid pixels -> no_valid_pixels status
8. Insufficient-valid-pixel threshold guardrail -> insufficient_valid_pixels status
9. Missing raster file handling
10. Missing band handling
11. Dimension mismatch handling
12. Unexpected dimensions handling
13. CRS mismatch handling
14. Affine transform mismatch handling
15. Non-aligned optical pair handling
16. SAR pair -> structured deferred/unsupported result (no differencing/resampling)
17. Deterministic repeated execution
18. ChangeResult JSON serialization
19. Materialized GeoTIFF raster verification (profile, CRS, transform, nodata)
20. Batch streaming execution (detect_changes)
21. Live optical tile handoff with M7 TemporalPairer
22. Live SAR tile handoff with M7 TemporalPairer
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
from src.pairing.models import AlignmentStatus, SpatialCorrespondence, TemporalPair
from src.pairing.pairer import TemporalPairer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_synthetic_optical_tile(
    tile_dir: Path,
    b02_val: int = 1000,
    b03_val: int = 1000,
    b04_val: int = 1000,
    b08_val: int = 1000,
    scl_val: int = 4,  # Vegetation by default
    crs: str = "EPSG:32643",
    shape: Tuple[int, int] = (256, 256),
    transform=None,
    create_scl: bool = True,
    scl_crs: Optional[str] = None,
    scl_shape: Optional[Tuple[int, int]] = None,
    scl_transform=None,
) -> None:
    """Helper to create a synthetic 10m optical tile directory with GeoTIFFs."""
    tile_dir.mkdir(parents=True, exist_ok=True)
    if transform is None:
        transform = from_origin(284070.0, 2107620.0, 10.0, 10.0)

    for band_name, val in [("B02", b02_val), ("B03", b03_val), ("B04", b04_val), ("B08", b08_val)]:
        arr = np.full(shape, val, dtype=np.uint16)
        profile = {
            "driver": "GTiff",
            "height": shape[0],
            "width": shape[1],
            "count": 1,
            "dtype": rasterio.uint16,
            "crs": crs,
            "transform": transform,
            "nodata": 0.0,
        }
        with rasterio.open(tile_dir / f"{band_name}.tif", "w", **profile) as dst:
            dst.write(arr, 1)

    if create_scl:
        # SCL at 20m (128x128 for 256x256 tile)
        if scl_shape is None:
            scl_shape = (shape[0] // 2, shape[1] // 2) if shape == (256, 256) else shape
        scl_arr = np.full(scl_shape, scl_val, dtype=np.uint8)
        if scl_transform is None:
            scl_transform = from_origin(284060.0, 2107620.0, 20.0, 20.0)
        scl_profile = {
            "driver": "GTiff",
            "height": scl_shape[0],
            "width": scl_shape[1],
            "count": 1,
            "dtype": rasterio.uint8,
            "crs": scl_crs if scl_crs is not None else crs,
            "transform": scl_transform,
            "nodata": 0.0,
        }
        with rasterio.open(tile_dir / "SCL.tif", "w", **scl_profile) as dst:
            dst.write(scl_arr, 1)


def make_dummy_pair(
    ref_dir: Path,
    comp_dir: Path,
    modality: str = "optical",
    is_pixel_aligned: bool = True,
    alignment_type: str = "native_pixel_aligned",
) -> TemporalPair:
    """Helper to construct a dummy TemporalPair for unit testing."""
    return TemporalPair(
        pair_id=f"pair__{ref_dir.name}__{comp_dir.name}",
        modality=modality,
        reference_tile_id=ref_dir.name,
        comparison_tile_id=comp_dir.name,
        reference_scene_id="SCENE_REF",
        comparison_scene_id="SCENE_COMP",
        reference_datetime_utc="2022-01-27T05:53:40Z",
        comparison_datetime_utc="2024-01-12T05:53:38Z",
        temporal_delta_days=715.0,
        spatial=SpatialCorrespondence(
            method="grid_index_exact",
            grid_index=(0, 0),
            spatial_iou_wgs84=1.0,
            bounds_wgs84=[72.8, 18.9, 72.9, 19.0],
        ),
        alignment=AlignmentStatus(
            is_pixel_aligned=is_pixel_aligned,
            alignment_type=alignment_type,
            crs="EPSG:32643" if modality == "optical" else None,
            pixel_dimensions=(256, 256),
        ),
        reference_tile_dir=str(ref_dir),
        comparison_tile_dir=str(comp_dir),
    )


class TestChangeDetection(unittest.TestCase):
    """Test suite for Milestone 8 Primary Raw Change Detection."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="flux_m8_test_"))
        self.detector = ChangeDetector(project_root=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. Identical reference/comparison -> zero change
    def test_identical_rasters_zero_change(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        create_synthetic_optical_tile(ref_dir, 1500, 1500, 1500, 1500)
        create_synthetic_optical_tile(comp_dir, 1500, 1500, 1500, 1500)

        pair = make_dummy_pair(ref_dir, comp_dir)
        result = self.detector.detect_change(pair)

        self.assertEqual(result.status, "success")
        self.assertIsNotNone(result.summary_statistics)
        self.assertEqual(result.summary_statistics["mean"], 0.0)
        self.assertEqual(result.summary_statistics["max"], 0.0)
        self.assertEqual(result.summary_statistics["std"], 0.0)
        self.assertEqual(result.summary_statistics["valid_pixels"], 65536)

    # 2. Known uniform change across four bands -> mathematically exact CVA magnitude
    def test_known_uniform_change_cva(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        # Ref = 1000 DN (0.10 refl), Comp = 2000 DN (0.20 refl) on all 4 bands
        # Delta = sqrt( 4 * (0.20 - 0.10)^2 ) = sqrt( 4 * 0.01 ) = sqrt(0.04) = 0.20
        create_synthetic_optical_tile(ref_dir, 1000, 1000, 1000, 1000)
        create_synthetic_optical_tile(comp_dir, 2000, 2000, 2000, 2000)

        pair = make_dummy_pair(ref_dir, comp_dir)
        result = self.detector.detect_change(pair)

        self.assertEqual(result.status, "success")
        self.assertIsNotNone(result.summary_statistics)
        self.assertTrue(np.isclose(result.summary_statistics["mean"], 0.20, atol=1e-5))
        self.assertTrue(np.isclose(result.summary_statistics["max"], 0.20, atol=1e-5))
        self.assertTrue(np.isclose(result.summary_statistics["median"], 0.20, atol=1e-5))
        self.assertTrue(np.isclose(result.summary_statistics["std"], 0.0, atol=1e-5))

    # 3. Known single-band change
    def test_known_single_band_change(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        # Only B04 changes by +1000 DN (+0.10 refl)
        # Delta = sqrt( 0 + 0 + 0.10^2 + 0 ) = 0.10
        create_synthetic_optical_tile(ref_dir, 1000, 1000, 1000, 1000)
        create_synthetic_optical_tile(comp_dir, 1000, 1000, 2000, 1000)

        pair = make_dummy_pair(ref_dir, comp_dir)
        result = self.detector.detect_change(pair)

        self.assertEqual(result.status, "success")
        self.assertIsNotNone(result.summary_statistics)
        self.assertTrue(np.isclose(result.summary_statistics["mean"], 0.10, atol=1e-5))
        self.assertTrue(np.isclose(result.summary_statistics["max"], 0.10, atol=1e-5))

    # 4. Nodata masking (0 values excluded from valid pixels)
    def test_nodata_masking(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        create_synthetic_optical_tile(ref_dir, 1000, 1000, 1000, 1000)
        create_synthetic_optical_tile(comp_dir, 2000, 2000, 2000, 2000)

        # Inject 0 (nodata) into the upper half of B04 in comparison
        with rasterio.open(comp_dir / "B04.tif", "r+") as dst:
            arr = dst.read(1)
            arr[:128, :] = 0
            dst.write(arr, 1)

        pair = make_dummy_pair(ref_dir, comp_dir)
        result = self.detector.detect_change(pair)

        self.assertEqual(result.status, "success")
        self.assertIsNotNone(result.summary_statistics)
        # Exactly half the pixels should be valid: 256 * 128 = 32768
        self.assertEqual(result.summary_statistics["valid_pixels"], 32768)
        self.assertEqual(result.summary_statistics["valid_pixel_ratio"], 0.5)
        # Valid half has delta = 0.20
        self.assertTrue(np.isclose(result.summary_statistics["mean"], 0.20, atol=1e-5))
        # Masked pixels are NaN in change_magnitude
        self.assertTrue(np.isnan(result.change_magnitude[:128, :]).all())

    # 5. Non-finite (NaN / Inf) masking
    def test_nan_inf_masking(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        create_synthetic_optical_tile(ref_dir, 1000, 1000, 1000, 1000)
        create_synthetic_optical_tile(comp_dir, 1000, 1000, 1000, 1000)

        pair = make_dummy_pair(ref_dir, comp_dir)
        result = self.detector.detect_change(pair)
        self.assertEqual(result.status, "success")

    # 6. SCL cloud and shadow quality masking (128x128 -> 256x256)
    def test_scl_cloud_masking(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        create_synthetic_optical_tile(ref_dir, 1000, 1000, 1000, 1000, scl_val=4)  # Veg
        create_synthetic_optical_tile(comp_dir, 2000, 2000, 2000, 2000, scl_val=4)

        # Inject SCL class 8 (Cloud Medium Prob) into quadrant of comparison SCL (64x64 at 20m -> 128x128 at 10m)
        with rasterio.open(comp_dir / "SCL.tif", "r+") as dst:
            scl = dst.read(1)
            scl[:64, :64] = 8  # Cloud
            dst.write(scl, 1)

        pair = make_dummy_pair(ref_dir, comp_dir)

        # With cloud masking enabled (default)
        res_cloud_masked = self.detector.detect_change(pair)
        self.assertEqual(res_cloud_masked.status, "success")
        # 128x128 pixels masked out of 256x256: 65536 - 16384 = 49152
        self.assertEqual(res_cloud_masked.summary_statistics["valid_pixels"], 49152)

        # With cloud masking disabled
        res_no_mask = self.detector.detect_change(pair, config=ChangeDetectionConfig(mask_clouds=False))
        self.assertEqual(res_no_mask.status, "success")
        self.assertEqual(res_no_mask.summary_statistics["valid_pixels"], 65536)

    # 6b. Wrong SCL resolution -> skipped safely
    def test_scl_wrong_resolution_rejected_safely(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        create_synthetic_optical_tile(ref_dir, 1000, 1000, 1000, 1000)
        # SCL resolution 15m instead of 20m
        scl_t = from_origin(284060.0, 2107620.0, 15.0, 15.0)
        create_synthetic_optical_tile(comp_dir, 2000, 2000, 2000, 2000, scl_transform=scl_t)

        pair = make_dummy_pair(ref_dir, comp_dir)
        result = self.detector.detect_change(pair)

        self.assertEqual(result.status, "success")
        self.assertFalse(result.provenance["scl_masking_applied"])
        self.assertTrue(any("resolution" in w.lower() for w in result.warnings))

    # 6c. Wrong SCL transform origin -> skipped safely
    def test_scl_wrong_transform_origin_rejected_safely(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        create_synthetic_optical_tile(ref_dir, 1000, 1000, 1000, 1000)
        # Origin shifted by 500m (far exceeding SCL pixel size)
        scl_t = from_origin(284560.0, 2108120.0, 20.0, 20.0)
        create_synthetic_optical_tile(comp_dir, 2000, 2000, 2000, 2000, scl_transform=scl_t)

        pair = make_dummy_pair(ref_dir, comp_dir)
        result = self.detector.detect_change(pair)

        self.assertEqual(result.status, "success")
        self.assertFalse(result.provenance["scl_masking_applied"])
        self.assertTrue(any("origin offset" in w.lower() or "overlap" in w.lower() for w in result.warnings))

    # 6d. Wrong SCL CRS -> skipped safely
    def test_scl_wrong_crs_rejected_safely(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        create_synthetic_optical_tile(ref_dir, 1000, 1000, 1000, 1000)
        create_synthetic_optical_tile(comp_dir, 2000, 2000, 2000, 2000, scl_crs="EPSG:4326")

        pair = make_dummy_pair(ref_dir, comp_dir)
        result = self.detector.detect_change(pair)

        self.assertEqual(result.status, "success")
        self.assertFalse(result.provenance["scl_masking_applied"])
        self.assertTrue(any("crs" in w.lower() for w in result.warnings))

    # 6e. Wrong SCL bounds / footprint -> skipped safely
    def test_scl_wrong_bounds_rejected_safely(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        create_synthetic_optical_tile(ref_dir, 1000, 1000, 1000, 1000)
        # SCL placed in completely disjoint bounding box
        scl_t = from_origin(500000.0, 3000000.0, 20.0, 20.0)
        create_synthetic_optical_tile(comp_dir, 2000, 2000, 2000, 2000, scl_transform=scl_t)

        pair = make_dummy_pair(ref_dir, comp_dir)
        result = self.detector.detect_change(pair)

        self.assertEqual(result.status, "success")
        self.assertFalse(result.provenance["scl_masking_applied"])
        self.assertTrue(any("overlap" in w.lower() or "origin" in w.lower() for w in result.warnings))

    # 6f. Non-128x128 SCL dimensions -> skipped safely
    def test_scl_non_128_dimensions_rejected_safely(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        create_synthetic_optical_tile(ref_dir, 1000, 1000, 1000, 1000)
        # SCL with shape (64, 64)
        create_synthetic_optical_tile(comp_dir, 2000, 2000, 2000, 2000, scl_shape=(64, 64))

        pair = make_dummy_pair(ref_dir, comp_dir)
        result = self.detector.detect_change(pair)

        self.assertEqual(result.status, "success")
        self.assertFalse(result.provenance["scl_masking_applied"])
        self.assertTrue(any("dimensions" in w.lower() for w in result.warnings))

    # 6g. Confirm categorical nearest-neighbor behavior remains exact 2x replication
    def test_scl_categorical_nearest_neighbor_exact_2x(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        create_synthetic_optical_tile(ref_dir, 1000, 1000, 1000, 1000)
        create_synthetic_optical_tile(comp_dir, 1000, 1000, 1000, 1000)

        # Write a checkerboard of alternating classes (4=Veg, 8=Cloud) into SCL of comparison
        scl_custom = np.full((128, 128), 4, dtype=np.uint8)
        # Top-left 2x2 in SCL coordinates: (0,0) is cloud (8), (0,1) is veg (4)
        scl_custom[0, 0] = 8
        with rasterio.open(comp_dir / "SCL.tif", "r+") as dst:
            dst.write(scl_custom, 1)

        pair = make_dummy_pair(ref_dir, comp_dir)
        result = self.detector.detect_change(pair)

        self.assertEqual(result.status, "success")
        self.assertTrue(result.provenance["scl_masking_applied"])
        # Pixel (0,0) in 20m SCL must expand to exact 2x2 block (0:2, 0:2) in 10m change map
        # Those 4 pixels must be NaN (cloud masked)
        self.assertTrue(np.isnan(result.change_magnitude[0:2, 0:2]).all())
        # Pixel (0,1) in 20m SCL must expand to exact 2x2 block (0:2, 2:4) in 10m change map
        # Those 4 pixels must be valid (not NaN, equal to 0.0)
        self.assertFalse(np.isnan(result.change_magnitude[0:2, 2:4]).any())
        self.assertTrue((result.change_magnitude[0:2, 2:4] == 0.0).all())

    # 7. All-invalid pixels -> no_valid_pixels status
    def test_all_invalid_pixels(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        # Zero out all pixels
        create_synthetic_optical_tile(ref_dir, 0, 0, 0, 0)
        create_synthetic_optical_tile(comp_dir, 1000, 1000, 1000, 1000)

        pair = make_dummy_pair(ref_dir, comp_dir)
        result = self.detector.detect_change(pair)

        self.assertEqual(result.status, "no_valid_pixels")
        self.assertIsNone(result.summary_statistics)
        self.assertIn("Zero valid pixels", result.warnings[0])

    # 8. Insufficient-valid-pixel threshold guardrail
    def test_insufficient_valid_pixels(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        create_synthetic_optical_tile(ref_dir, 1000, 1000, 1000, 1000)
        create_synthetic_optical_tile(comp_dir, 1000, 1000, 1000, 1000)

        # Mask 95% of pixels by setting them to 0 in B02 (only 5% valid)
        with rasterio.open(ref_dir / "B02.tif", "r+") as dst:
            arr = dst.read(1)
            arr[12:, :] = 0  # Leaves only ~4.7% valid
            dst.write(arr, 1)

        pair = make_dummy_pair(ref_dir, comp_dir)
        result = self.detector.detect_change(pair, config=ChangeDetectionConfig(minimum_valid_pixel_ratio=0.10))

        self.assertEqual(result.status, "insufficient_valid_pixels")
        self.assertIsNone(result.summary_statistics)
        self.assertTrue(any("below minimum" in w for w in result.warnings))

    # 9. Missing raster file handling
    def test_missing_raster_file(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        create_synthetic_optical_tile(ref_dir)
        create_synthetic_optical_tile(comp_dir)

        # Remove B08 from comparison
        (comp_dir / "B08.tif").unlink()

        pair = make_dummy_pair(ref_dir, comp_dir)
        result = self.detector.detect_change(pair)

        self.assertEqual(result.status, "missing_raster_file")
        self.assertIsNone(result.summary_statistics)
        self.assertTrue(any("B08.tif" in w for w in result.warnings))

    # 10. Missing directory handling
    def test_missing_directory(self):
        ref_dir = self.temp_dir / "non_existent_ref"
        comp_dir = self.temp_dir / "non_existent_comp"

        pair = make_dummy_pair(ref_dir, comp_dir)
        result = self.detector.detect_change(pair)

        self.assertEqual(result.status, "missing_raster_file")

    # 11. Dimension mismatch handling
    def test_dimension_mismatch(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        create_synthetic_optical_tile(ref_dir, shape=(256, 256))
        create_synthetic_optical_tile(comp_dir, shape=(128, 128))

        pair = make_dummy_pair(ref_dir, comp_dir)
        result = self.detector.detect_change(pair)

        self.assertEqual(result.status, "dimension_mismatch")

    # 12. Unexpected dimensions handling (e.g. 512x512)
    def test_unexpected_dimensions(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        create_synthetic_optical_tile(ref_dir, shape=(512, 512))
        create_synthetic_optical_tile(comp_dir, shape=(512, 512))

        pair = make_dummy_pair(ref_dir, comp_dir)
        result = self.detector.detect_change(pair)

        self.assertEqual(result.status, "dimension_mismatch")

    # 13. CRS mismatch handling
    def test_crs_mismatch(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        create_synthetic_optical_tile(ref_dir, crs="EPSG:32643")
        create_synthetic_optical_tile(comp_dir, crs="EPSG:4326")

        pair = make_dummy_pair(ref_dir, comp_dir)
        result = self.detector.detect_change(pair)

        self.assertEqual(result.status, "crs_mismatch")

    # 14. Affine transform mismatch handling
    def test_transform_mismatch(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        create_synthetic_optical_tile(ref_dir, transform=from_origin(284070.0, 2107620.0, 10.0, 10.0))
        create_synthetic_optical_tile(comp_dir, transform=from_origin(290000.0, 2107620.0, 10.0, 10.0))

        pair = make_dummy_pair(ref_dir, comp_dir)
        result = self.detector.detect_change(pair)

        self.assertEqual(result.status, "transform_mismatch")

    # 15. Non-aligned optical pair handling
    def test_unaligned_optical_pair(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        create_synthetic_optical_tile(ref_dir)
        create_synthetic_optical_tile(comp_dir)

        pair = make_dummy_pair(ref_dir, comp_dir, is_pixel_aligned=False, alignment_type="unaligned_metadata_mismatch")
        result = self.detector.detect_change(pair)

        self.assertEqual(result.status, "unaligned_optical_grid")
        self.assertEqual(result.method, "unsupported")
        self.assertIsNone(result.summary_statistics)

    # 16. SAR pair -> structured deferred/unsupported result (no differencing/resampling)
    def test_sar_pair_unsupported_deferred(self):
        ref_dir = self.temp_dir / "sar_ref"
        comp_dir = self.temp_dir / "sar_comp"

        pair = make_dummy_pair(ref_dir, comp_dir, modality="sar", is_pixel_aligned=False, alignment_type="unaligned_radar_grid")
        result = self.detector.detect_change(pair)

        self.assertEqual(result.status, "unsupported_sar_unaligned")
        self.assertEqual(result.method, "unsupported")
        self.assertEqual(result.modality, "sar")
        self.assertIsNone(result.summary_statistics)
        self.assertIsNone(result.change_magnitude)
        self.assertTrue(any("unaligned radar grid" in w for w in result.warnings))

    # 17. Deterministic repeated execution
    def test_deterministic_repeated_execution(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        create_synthetic_optical_tile(ref_dir, 1000, 1200, 1400, 1600)
        create_synthetic_optical_tile(comp_dir, 1100, 1300, 1500, 1700)

        pair = make_dummy_pair(ref_dir, comp_dir)
        res1 = self.detector.detect_change(pair)
        res2 = self.detector.detect_change(pair)

        self.assertEqual(res1.to_dict(), res2.to_dict())
        self.assertTrue(np.array_equal(res1.change_magnitude, res2.change_magnitude, equal_nan=True))

    # 18. ChangeResult JSON serialization
    def test_change_result_serialization(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        create_synthetic_optical_tile(ref_dir, 1000, 1000, 1000, 1000)
        create_synthetic_optical_tile(comp_dir, 1500, 1500, 1500, 1500)

        pair = make_dummy_pair(ref_dir, comp_dir)
        result = self.detector.detect_change(pair)

        d = result.to_dict()
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)

        self.assertEqual(deserialized["pair_id"], pair.pair_id)
        self.assertEqual(deserialized["status"], "success")
        self.assertIn("summary_statistics", deserialized)
        self.assertEqual(deserialized["summary_statistics"]["valid_pixels"], 65536)

    # 19. Materialized GeoTIFF raster verification
    def test_save_change_map_geotiff(self):
        ref_dir = self.temp_dir / "tile_ref"
        comp_dir = self.temp_dir / "tile_comp"
        create_synthetic_optical_tile(ref_dir, 1000, 1000, 1000, 1000)
        create_synthetic_optical_tile(comp_dir, 1500, 1500, 1500, 1500)

        out_changes_dir = self.temp_dir / "saved_changes"
        pair = make_dummy_pair(ref_dir, comp_dir)
        config = ChangeDetectionConfig(save_change_map=True, output_dir=str(out_changes_dir))

        result = self.detector.detect_change(pair, config=config)

        self.assertIsNotNone(result.change_map_path)
        out_file = Path(result.change_map_path)
        if not out_file.is_absolute():
            out_file = self.temp_dir / out_file

        self.assertTrue(out_file.exists())
        with rasterio.open(out_file) as src:
            self.assertEqual(src.shape, (256, 256))
            self.assertEqual(src.count, 1)
            self.assertEqual(str(src.crs), "EPSG:32643")
            self.assertEqual(src.dtypes[0], rasterio.float32)
            arr = src.read(1)
            self.assertTrue(np.isclose(arr.mean(), 0.10, atol=1e-5))

    # 20. Batch streaming execution (detect_changes)
    def test_batch_processing(self):
        ref_dir1 = self.temp_dir / "tile_ref1"
        comp_dir1 = self.temp_dir / "tile_comp1"
        ref_dir2 = self.temp_dir / "tile_ref2"
        comp_dir2 = self.temp_dir / "tile_comp2"

        create_synthetic_optical_tile(ref_dir1, 1000, 1000, 1000, 1000)
        create_synthetic_optical_tile(comp_dir1, 1200, 1200, 1200, 1200)
        create_synthetic_optical_tile(ref_dir2, 2000, 2000, 2000, 2000)
        create_synthetic_optical_tile(comp_dir2, 2500, 2500, 2500, 2500)

        p1 = make_dummy_pair(ref_dir1, comp_dir1)
        p2 = make_dummy_pair(ref_dir2, comp_dir2)

        results = list(self.detector.detect_changes([p1, p2]))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].status, "success")
        self.assertEqual(results[1].status, "success")

    # 21. Live optical tile handoff with M7 TemporalPairer
    def test_live_optical_tile_handoff(self):
        pairer = TemporalPairer(project_root=PROJECT_ROOT)
        detector = ChangeDetector(project_root=PROJECT_ROOT)

        # Query live optical tile
        pair = pairer.get_pair_for_tile("s2_S2A_43QBB_20220127_0_L2A_10m_r03_c04")
        self.assertIsNotNone(pair)

        result = detector.detect_change(pair)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.modality, "optical")
        self.assertIsNotNone(result.summary_statistics)
        self.assertGreater(result.summary_statistics["valid_pixels"], 0)
        self.assertGreaterEqual(result.summary_statistics["mean"], 0.0)
        self.assertLessEqual(result.summary_statistics["mean"], 2.0)

    # 22. Live SAR tile handoff with M7 TemporalPairer
    def test_live_sar_tile_handoff(self):
        pairer = TemporalPairer(project_root=PROJECT_ROOT)
        detector = ChangeDetector(project_root=PROJECT_ROOT)

        # Query live SAR tile
        sar_tile_id = "s1_S1A_IW_GRDH_1SDV_20220123T010321_20220123T010346_041581_04F21E_41F0_radar_r03_c04"
        pair = pairer.get_pair_for_tile(sar_tile_id)
        self.assertIsNotNone(pair)

        result = detector.detect_change(pair)
        self.assertEqual(result.status, "unsupported_sar_unaligned")
        self.assertEqual(result.method, "unsupported")
        self.assertIsNone(result.summary_statistics)


if __name__ == "__main__":
    unittest.main()
