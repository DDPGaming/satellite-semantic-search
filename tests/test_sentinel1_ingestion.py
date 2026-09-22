"""Unit and integration tests for Sentinel-1 Level-1 GRD data ingestion (M1 Stage 2).

Covers:
1. Data containers: SatelliteSARBandRaster, SatelliteSARScene
2. Statistical functions: compute_sar_band_statistics
3. GCP handling: translate_gcps, derive_local_affine
4. Validation suite: validate_sentinel1_scene (valid and defect cases)
5. Ingested SAR scene integration tests (when raw data is present)
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint

from src.ingestion.sentinel1 import (
    SatelliteSARBandRaster,
    SatelliteSARScene,
    compute_file_sha256,
    compute_sar_band_statistics,
    derive_local_affine,
    load_sentinel1_scene,
    translate_gcps,
    validate_sentinel1_scene,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_SENTINEL1_DIR = ROOT_DIR / "data" / "raw" / "sentinel1"


class TestSARBandStatistics(unittest.TestCase):
    """Test compute_sar_band_statistics on synthetic arrays."""

    def test_statistics_without_nodata(self):
        data = np.array([[50, 100], [150, 200]], dtype=np.uint16)
        stats = compute_sar_band_statistics(data)
        self.assertEqual(stats["min"], 50.0)
        self.assertEqual(stats["max"], 200.0)
        self.assertEqual(stats["mean"], 125.0)
        self.assertEqual(stats["valid_count"], 4)
        self.assertEqual(stats["total_pixels"], 4)

    def test_statistics_with_nodata(self):
        data = np.array([[0, 100], [0, 200]], dtype=np.uint16)
        stats = compute_sar_band_statistics(data, nodata=0)
        self.assertEqual(stats["min"], 100.0)
        self.assertEqual(stats["max"], 200.0)
        self.assertEqual(stats["mean"], 150.0)
        self.assertEqual(stats["valid_count"], 2)


class TestGCPHandling(unittest.TestCase):
    """Test GCP translation and local affine derivation."""

    def test_translate_gcps(self):
        gcps = [
            GroundControlPoint(row=4000, col=6500, x=73.0, y=19.0, z=10.0, id="1"),
            GroundControlPoint(row=10000, col=10000, x=71.0, y=18.0, z=5.0, id="2"),  # outside
        ]
        # Crop: col_off=6300, row_off=3800, width=500, height=500
        translated = translate_gcps(gcps, col_off=6300, row_off=3800, width=500, height=500)
        self.assertEqual(len(translated), 1)
        g = translated[0]
        self.assertEqual(g["id"], "1")
        self.assertEqual(g["source_col"], 6500.0)
        self.assertEqual(g["source_row"], 4000.0)
        self.assertEqual(g["local_col"], 200.0)  # 6500 - 6300
        self.assertEqual(g["local_row"], 200.0)  # 4000 - 3800

    def test_derive_local_affine_labeling(self):
        gcp_dicts = [
            {"local_col": 0.0, "local_row": 0.0, "lon": 73.1, "lat": 19.1},
            {"local_col": 100.0, "local_row": 0.0, "lon": 73.0, "lat": 19.1},
            {"local_col": 0.0, "local_row": 100.0, "lon": 73.1, "lat": 19.0},
            {"local_col": 100.0, "local_row": 100.0, "lon": 73.0, "lat": 19.0},
        ]
        aff = derive_local_affine(gcp_dicts)
        self.assertEqual(aff["label"], "derived_local_affine_approximation")
        self.assertIn("not an authoritative source transform", aff["disclaimer"])
        self.assertEqual(len(aff["coefficients"]), 6)


class TestSARValidationSuite(unittest.TestCase):
    """Test 11-point SAR validation suite on synthetic mock scenes."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create mock synthetic VV and VH GeoTIFF files
        self.band_files = {}
        for pol in ["vv", "vh"]:
            fpath = self.temp_path / f"{pol}.tif"
            data = np.full((20, 20), 500, dtype=np.uint16)
            profile = {
                "driver": "GTiff",
                "height": 20,
                "width": 20,
                "count": 1,
                "dtype": "uint16",
                "nodata": 0,
            }
            with rasterio.open(fpath, "w", **profile) as dst:
                dst.write(data, 1)
            self.band_files[pol.upper()] = fpath

        self.metadata = {
            "scene_metadata": {
                "scene_id": "MOCK_S1A_TEST",
                "platform": "Sentinel-1A",
                "sensor": "C-SAR",
                "instrument_mode": "IW",
                "product_type": "GRD",
                "resolution_type": "high",
                "polarizations": ["VV", "VH"],
                "orbit_direction": "descending",
                "relative_orbit": 34,
                "absolute_orbit": 41581,
                "acquisition_datetime_utc": "2022-01-23T01:03:33Z",
            },
            "source_raster_metadata": {
                "dimensions": {"height": 16000, "width": 25000},
                "total_source_gcps": 210,
                "gcp_crs": "EPSG:4326",
            },
            "crop_window": {
                "col_off": 6000,
                "row_off": 3000,
                "width": 20,
                "height": 20,
                "safety_margin_pixels": 50,
            },
            "local_gcps": [
                {
                    "id": "1",
                    "source_col": 6010.0,
                    "source_row": 3010.0,
                    "local_col": 10.0,
                    "local_row": 10.0,
                    "lon": 73.0,
                    "lat": 19.0,
                    "elevation_m": 0.0,
                }
            ],
            "derived_local_affine_approximation": {
                "label": "derived_local_affine_approximation",
                "coefficients": [1, 0, 0, 0, 1, 0],
            },
            "provenance": {
                "tier1_copernicus_source": {"authority": "ESA Copernicus"},
                "tier2_aws_cog_distribution": {"curator": "AWS Open Data"},
                "tier3_http_range_window_read": {"safety_margin_pixels": 50},
                "tier4_local_native_radar_crop": {"coordinate_space": "native_radar_range_azimuth"},
            },
            "bands": {
                pol: {
                    "polarization": pol,
                    "local_filename": f"{pol.lower()}.tif",
                    "local_sha256": compute_file_sha256(self.band_files[pol]),
                    "dimensions": {"height": 20, "width": 20},
                }
                for pol in ["VV", "VH"]
            },
        }

        with open(self.temp_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_synthetic_scene(self):
        scene, report = load_sentinel1_scene(self.temp_path)
        self.assertEqual(report["status"], "PASSED")
        self.assertEqual(len(report["errors"]), 0)
        self.assertEqual(len(scene.bands), 2)
        self.assertEqual(scene.bands["VV"].shape, (20, 20))
        self.assertEqual(scene.bands["VH"].shape, (20, 20))

    def test_missing_band_fails(self):
        (self.temp_path / "vh.tif").unlink()
        scene, report = load_sentinel1_scene(self.temp_path)
        self.assertEqual(report["status"], "FAILED")
        self.assertTrue(any("Missing required polarization band: VH" in err for err in report["errors"]))

    def test_corrupted_hash_fails(self):
        with open(self.band_files["VV"], "ab") as f:
            f.write(b"corruption")
        scene, report = load_sentinel1_scene(self.temp_path)
        self.assertEqual(report["status"], "FAILED")
        self.assertTrue(any("Local SHA-256 hash mismatch" in err for err in report["errors"]))


class TestIngestedSentinel1Data(unittest.TestCase):
    """Integration tests verifying the actual ingested Sentinel-1 sample scenes (if materialized)."""

    def setUp(self):
        self.scene_ids = [
            "S1A_IW_GRDH_1SDV_20220123T010321_20220123T010346_041581_04F21E_41F0",
            "S1A_IW_GRDH_1SDV_20240113T010330_20240113T010355_052081_064B6D_A8B7",
        ]

    def test_ingested_scenes_if_present(self):
        for scene_id in self.scene_ids:
            scene_dir = RAW_SENTINEL1_DIR / scene_id
            if not (scene_dir / "metadata.json").exists():
                self.skipTest(f"Scene {scene_id} not yet fully ingested into {scene_dir}")

            scene, report = load_sentinel1_scene(scene_dir)

            # Ingestion validation must pass
            self.assertEqual(
                report["status"],
                "PASSED",
                f"Scene {scene_id} validation failed: {report['errors']}",
            )

            # Check required bands VV and VH
            self.assertIn("VV", scene.bands)
            self.assertIn("VH", scene.bands)

            # Check identical shapes between polarizations
            self.assertEqual(scene.bands["VV"].shape, scene.bands["VH"].shape)
            self.assertEqual(scene.bands["VV"].dtype, "uint16")
            self.assertEqual(scene.bands["VH"].dtype, "uint16")

            # Check safe window dimensions
            if "20220123" in scene_id:
                self.assertEqual(scene.bands["VV"].shape, (1999, 1976))
            elif "20240113" in scene_id:
                self.assertEqual(scene.bands["VV"].shape, (1997, 1976))

            # Non-negative values and within uint16 storage bounds
            for pol in ["VV", "VH"]:
                stats = report["band_statistics"][pol]
                self.assertGreaterEqual(stats["min"], 0.0)
                self.assertLessEqual(stats["max"], 65535.0)

            # Provenance checks
            prov = scene.provenance
            self.assertIn("tier1_copernicus_source", prov)
            self.assertIn("tier2_aws_cog_distribution", prov)
            self.assertIn("tier3_http_range_window_read", prov)
            self.assertIn("tier4_local_native_radar_crop", prov)

            self.assertEqual(prov["tier3_http_range_window_read"]["safety_margin_pixels"], 50)
            self.assertEqual(prov["tier4_local_native_radar_crop"]["coordinate_space"], "native_radar_range_azimuth")
            self.assertFalse(prov["tier4_local_native_radar_crop"]["reprojected_or_resampled"])

            # Local affine must be explicitly labeled as derived approximation
            aff = scene.derived_local_affine_approximation
            self.assertEqual(aff["label"], "derived_local_affine_approximation")
            self.assertIn("not an authoritative source transform", aff["disclaimer"])


if __name__ == "__main__":
    unittest.main()
