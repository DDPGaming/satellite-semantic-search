"""Unit and integration tests for Sentinel-2 Level-2A data ingestion (M1 Stage 1).

Covers:
1. Data containers: SatelliteBandRaster, SatelliteScene
2. Statistical functions: compute_band_statistics, compute_scl_distribution
3. Integrity hashing: compute_file_sha256
4. Validation suite: validate_sentinel2_scene (valid and defect cases)
5. Ingested scene integration tests (when raw data is present)
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from src.ingestion.sentinel2 import (
    SCL_CLASSES,
    SatelliteBandRaster,
    SatelliteScene,
    compute_band_statistics,
    compute_file_sha256,
    compute_scl_distribution,
    load_sentinel2_scene,
    validate_sentinel2_scene,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_SENTINEL2_DIR = ROOT_DIR / "data" / "raw" / "sentinel2"


class TestBandStatistics(unittest.TestCase):
    """Test compute_band_statistics on synthetic arrays."""

    def test_statistics_without_nodata(self):
        data = np.array([[100, 200], [300, 400]], dtype=np.uint16)
        stats = compute_band_statistics(data)
        self.assertEqual(stats["min"], 100.0)
        self.assertEqual(stats["max"], 400.0)
        self.assertEqual(stats["mean"], 250.0)
        self.assertEqual(stats["valid_count"], 4)
        self.assertEqual(stats["total_pixels"], 4)

    def test_statistics_with_nodata(self):
        data = np.array([[0, 200], [0, 400]], dtype=np.uint16)
        stats = compute_band_statistics(data, nodata=0)
        self.assertEqual(stats["min"], 200.0)
        self.assertEqual(stats["max"], 400.0)
        self.assertEqual(stats["mean"], 300.0)
        self.assertEqual(stats["valid_count"], 2)
        self.assertEqual(stats["total_pixels"], 4)

    def test_all_nodata(self):
        data = np.zeros((4, 4), dtype=np.uint16)
        stats = compute_band_statistics(data, nodata=0)
        self.assertEqual(stats["valid_count"], 0)
        self.assertEqual(stats["total_pixels"], 16)
        self.assertEqual(stats["min"], 0.0)
        self.assertEqual(stats["count_above_10000"], 0)
        self.assertEqual(stats["pct_above_10000"], 0.0)

    def test_statistics_values_above_10000(self):
        data = np.array([[5000, 9000], [12000, 15000]], dtype=np.uint16)
        stats = compute_band_statistics(data)
        self.assertEqual(stats["count_above_10000"], 2)
        self.assertEqual(stats["pct_above_10000"], 50.0)
        self.assertEqual(stats["max"], 15000.0)
        self.assertEqual(stats["min"], 5000.0)


class TestSclDistribution(unittest.TestCase):
    """Test compute_scl_distribution diagnostic calculations."""

    def test_scl_distribution_counts(self):
        # 4 pixels: 1 vegetation (4), 1 water (6), 1 cloud medium (8), 1 bare soil (5)
        scl = np.array([[4, 6], [8, 5]], dtype=np.uint8)
        dist = compute_scl_distribution(scl)

        self.assertEqual(dist["total_pixels"], 4)
        diag = dist["diagnostic_summary"]
        self.assertEqual(diag["vegetation_pct"], 25.0)
        self.assertEqual(diag["water_pct"], 25.0)
        self.assertEqual(diag["bare_soil_pct"], 25.0)
        self.assertEqual(diag["cloud_pixels_pct"], 25.0)
        self.assertEqual(diag["cloud_shadow_pct"], 0.0)

        # Verify all 12 standard SCL class keys are present
        self.assertEqual(len(dist["classes"]), len(SCL_CLASSES))


class TestValidationSuite(unittest.TestCase):
    """Test 11-point validation suite on synthetic mock scenes."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create synthetic GeoTIFF files
        self.band_files = {}
        transform = Affine(10.0, 0.0, 284070.0, 0.0, -10.0, 2107620.0)
        crs = "EPSG:32643"

        for bid in ["B02", "B03", "B04", "B08"]:
            fpath = self.temp_path / f"{bid}.tif"
            data = np.full((10, 10), 1000, dtype=np.uint16)
            profile = {
                "driver": "GTiff",
                "height": 10,
                "width": 10,
                "count": 1,
                "dtype": "uint16",
                "crs": crs,
                "transform": transform,
                "nodata": 0,
            }
            with rasterio.open(fpath, "w", **profile) as dst:
                dst.write(data, 1)
            self.band_files[bid] = fpath

        # SCL 20m raster (5x5)
        scl_transform = Affine(20.0, 0.0, 284060.0, 0.0, -20.0, 2107620.0)
        scl_path = self.temp_path / "SCL.tif"
        scl_data = np.full((5, 5), 4, dtype=np.uint8)  # all vegetation
        scl_profile = {
            "driver": "GTiff",
            "height": 5,
            "width": 5,
            "count": 1,
            "dtype": "uint8",
            "crs": crs,
            "transform": scl_transform,
            "nodata": None,
        }
        with rasterio.open(scl_path, "w", **scl_profile) as dst:
            dst.write(scl_data, 1)
        self.band_files["SCL"] = scl_path

        # Metadata dictionary
        self.metadata = {
            "scene_metadata": {
                "scene_id": "MOCK_S2A_TEST",
                "spacecraft": "Sentinel-2A",
                "sensor": "MSI",
                "processing_level": "Level-2A",
                "mgrs_tile": "43QBB",
                "acquisition_datetime": "2022-01-27T05:53:40Z",
                "cloud_cover_tile_pct": 0.01,
                "sun_elevation": 45.0,
                "sun_azimuth": 150.0,
            },
            "provenance": {
                "tier1_copernicus_source": {"authority": "ESA Copernicus"},
                "tier2_element84_aws_cog_distribution": {"curator": "AWS Open Data"},
                "tier3_local_aoi_crop_artifact": {"temporal_baseline": "Approximately 23.5 months / 716 days"},
            },
            "bands": {
                bid: {
                    "band_identity": bid,
                    "pixel_resolution_m": 10.0,
                    "local_filename": f"{bid}.tif",
                    "local_sha256": compute_file_sha256(self.band_files[bid]),
                    "dimensions": {"height": 10, "width": 10},
                }
                for bid in ["B02", "B03", "B04", "B08"]
            },
        }
        self.metadata["bands"]["SCL"] = {
            "band_identity": "SCL",
            "pixel_resolution_m": 20.0,
            "local_filename": "SCL.tif",
            "local_sha256": compute_file_sha256(scl_path),
            "dimensions": {"height": 5, "width": 5},
        }

        # Write metadata.json
        with open(self.temp_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_synthetic_scene_validation(self):
        scene, report = load_sentinel2_scene(self.temp_path)
        self.assertEqual(report["status"], "PASSED")
        self.assertEqual(len(report["errors"]), 0)
        self.assertEqual(len(scene.bands), 5)
        self.assertEqual(scene.bands["B02"].shape, (10, 10))
        self.assertEqual(scene.bands["SCL"].shape, (5, 5))

    def test_missing_band_fails(self):
        # Remove B08
        (self.temp_path / "B08.tif").unlink()
        scene, report = load_sentinel2_scene(self.temp_path)
        self.assertEqual(report["status"], "FAILED")
        self.assertTrue(any("Missing expected band: B08" in err for err in report["errors"]))

    def test_corrupted_sha256_detected(self):
        # Tamper with B02 file
        with open(self.band_files["B02"], "ab") as f:
            f.write(b"tamper")
        scene, report = load_sentinel2_scene(self.temp_path)
        self.assertEqual(report["status"], "FAILED")
        self.assertTrue(any("Local SHA-256 hash mismatch" in err for err in report["errors"]))

    def test_values_above_10000_does_not_fail_validation(self):
        # Overwrite B02 with values > 10,000 (valid physical reflectance e.g. specular glint/clouds)
        data = np.full((10, 10), 12500, dtype=np.uint16)
        profile = {
            "driver": "GTiff",
            "height": 10,
            "width": 10,
            "count": 1,
            "dtype": "uint16",
            "crs": "EPSG:32643",
            "transform": Affine(10.0, 0.0, 284070.0, 0.0, -10.0, 2107620.0),
            "nodata": 0,
        }
        with rasterio.open(self.band_files["B02"], "w", **profile) as dst:
            dst.write(data, 1)

        # Update metadata hash to match
        self.metadata["bands"]["B02"]["local_sha256"] = compute_file_sha256(self.band_files["B02"])
        with open(self.temp_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)

        scene, report = load_sentinel2_scene(self.temp_path)
        self.assertEqual(report["status"], "PASSED")
        self.assertEqual(len(report["errors"]), 0)
        self.assertEqual(report["band_statistics"]["B02"]["count_above_10000"], 100)
        self.assertEqual(report["band_statistics"]["B02"]["pct_above_10000"], 100.0)


class TestIngestedSentinel2Data(unittest.TestCase):
    """Integration tests verifying the actual ingested sample scenes (if materialized)."""

    def setUp(self):
        self.scene_ids = [
            "S2A_43QBB_20220127_0_L2A",
            "S2B_43QBB_20240112_0_L2A",
        ]

    def test_ingested_scenes_if_present(self):
        for scene_id in self.scene_ids:
            scene_dir = RAW_SENTINEL2_DIR / scene_id
            if not (scene_dir / "metadata.json").exists():
                self.skipTest(f"Scene {scene_id} not yet fully ingested into {scene_dir}")

            scene, report = load_sentinel2_scene(scene_dir)

            # Ingestion validation must pass
            self.assertEqual(
                report["status"],
                "PASSED",
                f"Scene {scene_id} validation failed: {report['errors']}",
            )

            # Check required bands
            for bid in ["B02", "B03", "B04", "B08", "SCL"]:
                self.assertIn(bid, scene.bands)

            # 10m optical band geometries and reflectance statistics
            for bid in ["B02", "B03", "B04", "B08"]:
                b = scene.bands[bid]
                self.assertEqual(b.crs, "EPSG:32643")
                self.assertEqual(b.shape, (1679, 1599))
                self.assertEqual(b.dtype, "uint16")
                self.assertEqual(b.resolution, 10.0)

                # Reflectance range validation: non-negative and within uint16 storage bounds
                stats = report["band_statistics"][bid]
                self.assertGreaterEqual(stats["min"], 0.0)
                self.assertLessEqual(stats["max"], 65535.0)
                self.assertIn("count_above_10000", stats)
                self.assertIn("pct_above_10000", stats)
                self.assertIn("p1", stats)
                self.assertIn("p50", stats)
                self.assertIn("p99", stats)

            # 20m SCL band geometries
            scl_b = scene.bands["SCL"]
            self.assertEqual(scl_b.crs, "EPSG:32643")
            self.assertEqual(scl_b.shape, (840, 800))
            self.assertEqual(scl_b.dtype, "uint8")
            self.assertEqual(scl_b.resolution, 20.0)
            scl_stats = report["band_statistics"]["SCL"]
            self.assertGreaterEqual(scl_stats["min"], 0.0)
            self.assertLessEqual(scl_stats["max"], 255.0)

            # Provenance checks
            prov = scene.provenance
            self.assertIn("tier1_copernicus_source", prov)
            self.assertIn("tier2_element84_aws_cog_distribution", prov)
            self.assertIn("tier3_local_aoi_crop_artifact", prov)
            self.assertEqual(
                prov["tier3_local_aoi_crop_artifact"]["temporal_baseline"],
                "Approximately 23.5 months / 716 days",
            )
            self.assertEqual(
                prov["tier1_copernicus_source"]["mgrs_tile"],
                "43QBB",
            )


if __name__ == "__main__":
    unittest.main()
