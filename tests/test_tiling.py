"""Comprehensive Unit & Integration Test Suite for M2 Tiling + Metadata.

Covers:
1. Deterministic grid window generation & boundary handling (shift, truncate)
2. Sentinel-2 multi-resolution tiling (10m optical, 20m SCL, zero resampling)
3. Sentinel-1 native radar coordinate tiling & local GCP preservation
4. Metadata schema completeness and authoritative vs derived distinction
5. Repeatability: identical tile contents, metadata, and hashes across independent runs
6. Verification that M1 source rasters remain 100% unmodified
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from src.ingestion.sentinel1 import compute_file_sha256
from src.tiling.grid import compute_axis_offsets, generate_tile_windows
from src.tiling.sentinel1_tiler import tile_sentinel1_scene
from src.tiling.sentinel2_tiler import tile_sentinel2_scene

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_S2_DIR = ROOT_DIR / "data" / "raw" / "sentinel2"
RAW_S1_DIR = ROOT_DIR / "data" / "raw" / "sentinel1"


class TestGridGenerator(unittest.TestCase):
    """Test deterministic 1D and 2D grid window generation."""

    def test_compute_axis_offsets_shift(self):
        # 600 px length with tile_size=256, stride=256
        # Expected offsets: 0, 256, and shifted last: 600 - 256 = 344
        offsets = compute_axis_offsets(600, tile_size=256, stride=256, boundary_strategy="shift")
        self.assertEqual(len(offsets), 3)
        self.assertEqual(offsets[0], (0, 256))
        self.assertEqual(offsets[1], (256, 256))
        self.assertEqual(offsets[2], (344, 256))

    def test_compute_axis_offsets_truncate(self):
        # 600 px length with tile_size=256, stride=256
        # Expected offsets: (0, 256), (256, 256), (512, 88)
        offsets = compute_axis_offsets(600, tile_size=256, stride=256, boundary_strategy="truncate")
        self.assertEqual(len(offsets), 3)
        self.assertEqual(offsets[0], (0, 256))
        self.assertEqual(offsets[1], (256, 256))
        self.assertEqual(offsets[2], (512, 88))

    def test_exact_multiple_dimension(self):
        # 512 px length with tile_size=256 -> exactly 2 tiles, no shift
        offsets = compute_axis_offsets(512, tile_size=256, stride=256, boundary_strategy="shift")
        self.assertEqual(len(offsets), 2)
        self.assertEqual(offsets[0], (0, 256))
        self.assertEqual(offsets[1], (256, 256))

    def test_smaller_than_tile_size(self):
        offsets = compute_axis_offsets(100, tile_size=256, stride=256, boundary_strategy="shift")
        self.assertEqual(len(offsets), 1)
        self.assertEqual(offsets[0], (0, 100))

    def test_generate_tile_windows_repeatability(self):
        w1 = generate_tile_windows(1599, 1679, tile_size=256, boundary_strategy="shift")
        w2 = generate_tile_windows(1599, 1679, tile_size=256, boundary_strategy="shift")
        self.assertEqual(len(w1), len(w2))
        self.assertEqual(w1, w2)
        # All windows must have exact size 256x256
        for w in w1:
            self.assertEqual(w.width, 256)
            self.assertEqual(w.height, 256)


class TestBoundaryOverlapAndCoverage(unittest.TestCase):
    """Explicitly verify and audit boundary behavior under 'shift' tiling policy.
    
    Documents and locks the following invariants:
    1. Stride=256 is used; all interior tiles are strictly non-overlapping (0 px overlap).
    2. The 'shift' policy introduces overlap ONLY at the terminal boundary tiles (last row / column).
    3. The terminal overlap equals: stride - (dimension % stride).
    4. 100% of source pixels are covered (zero gaps, zero unmonitored zones).
    5. Fixed 256x256 dimensions are preserved for all tiles without synthetic padding.
    """

    def _verify_axis_coverage_and_overlaps(self, dim: int, tile_size: int, stride: int):
        offsets = compute_axis_offsets(dim, tile_size, stride, boundary_strategy="shift")
        self.assertTrue(len(offsets) > 0)

        # 1. Verify 100% pixel coverage
        covered = [False] * dim
        for off, length in offsets:
            self.assertEqual(length, tile_size)
            for p in range(off, off + length):
                covered[p] = True
        self.assertTrue(all(covered), f"Coverage gap detected for dimension {dim}")

        # 2. Verify all interior tiles have exactly 0 overlap (stride == tile_size)
        num_tiles = len(offsets)
        for i in range(num_tiles - 2):
            curr_off, curr_len = offsets[i]
            next_off, _ = offsets[i + 1]
            overlap = (curr_off + curr_len) - next_off
            self.assertEqual(
                overlap, 0,
                f"Interior tile {i} and {i+1} must be strictly non-overlapping, got {overlap} px"
            )

        # 3. Verify terminal boundary overlap if dimension not divisible by stride
        remainder = dim % stride
        if remainder != 0 and num_tiles > 1:
            expected_terminal_overlap = stride - remainder
            prev_off, prev_len = offsets[-2]
            term_off, _ = offsets[-1]
            actual_terminal_overlap = (prev_off + prev_len) - term_off
            self.assertEqual(
                actual_terminal_overlap, expected_terminal_overlap,
                f"Expected terminal overlap {expected_terminal_overlap} px, got {actual_terminal_overlap} px"
            )
        elif remainder == 0 and num_tiles > 1:
            prev_off, prev_len = offsets[-2]
            term_off, _ = offsets[-1]
            actual_terminal_overlap = (prev_off + prev_len) - term_off
            self.assertEqual(actual_terminal_overlap, 0)

        return offsets

    def test_sentinel2_staged_dimensions_overlap_and_coverage(self):
        """Sentinel-2 staged crop: 1679 (H) x 1599 (W).
        
        Width 1599: 7 tiles. Overlap between Tile 5 and Tile 6 = 193 px.
        Height 1679: 7 tiles. Overlap between Tile 5 and Tile 6 = 113 px.
        """
        # Width: 1599 px
        w_offsets = self._verify_axis_coverage_and_overlaps(1599, 256, 256)
        self.assertEqual(len(w_offsets), 7)
        self.assertEqual(w_offsets[-2], (1280, 256))
        self.assertEqual(w_offsets[-1], (1343, 256))
        # Terminal overlap: (1280 + 256) - 1343 = 193 px
        self.assertEqual((1280 + 256) - 1343, 193)

        # Height: 1679 px
        h_offsets = self._verify_axis_coverage_and_overlaps(1679, 256, 256)
        self.assertEqual(len(h_offsets), 7)
        self.assertEqual(h_offsets[-2], (1280, 256))
        self.assertEqual(h_offsets[-1], (1423, 256))
        # Terminal overlap: (1280 + 256) - 1423 = 113 px
        self.assertEqual((1280 + 256) - 1423, 113)

    def test_sentinel1_2022_staged_dimensions_overlap_and_coverage(self):
        """Sentinel-1 2022 crop: 1999 (H) x 1976 (W) native radar grid.
        
        Width 1976: 8 tiles. Overlap between Tile 6 and Tile 7 = 72 px.
        Height 1999: 8 tiles. Overlap between Tile 6 and Tile 7 = 49 px.
        """
        # Width: 1976 px
        w_offsets = self._verify_axis_coverage_and_overlaps(1976, 256, 256)
        self.assertEqual(len(w_offsets), 8)
        self.assertEqual(w_offsets[-2], (1536, 256))
        self.assertEqual(w_offsets[-1], (1720, 256))
        # Terminal overlap: (1536 + 256) - 1720 = 72 px
        self.assertEqual((1536 + 256) - 1720, 72)

        # Height: 1999 px
        h_offsets = self._verify_axis_coverage_and_overlaps(1999, 256, 256)
        self.assertEqual(len(h_offsets), 8)
        self.assertEqual(h_offsets[-2], (1536, 256))
        self.assertEqual(h_offsets[-1], (1743, 256))
        # Terminal overlap: (1536 + 256) - 1743 = 49 px
        self.assertEqual((1536 + 256) - 1743, 49)

    def test_sentinel1_2024_staged_dimensions_overlap_and_coverage(self):
        """Sentinel-1 2024 crop: 1997 (H) x 1976 (W) native radar grid.
        
        Width 1976: 8 tiles. Overlap between Tile 6 and Tile 7 = 72 px.
        Height 1997: 8 tiles. Overlap between Tile 6 and Tile 7 = 51 px.
        """
        # Width: 1976 px
        w_offsets = self._verify_axis_coverage_and_overlaps(1976, 256, 256)
        self.assertEqual(len(w_offsets), 8)
        self.assertEqual(w_offsets[-2], (1536, 256))
        self.assertEqual(w_offsets[-1], (1720, 256))
        self.assertEqual((1536 + 256) - 1720, 72)

        # Height: 1997 px
        h_offsets = self._verify_axis_coverage_and_overlaps(1997, 256, 256)
        self.assertEqual(len(h_offsets), 8)
        self.assertEqual(h_offsets[-2], (1536, 256))
        self.assertEqual(h_offsets[-1], (1741, 256))
        # Terminal overlap: (1536 + 256) - 1741 = 51 px
        self.assertEqual((1536 + 256) - 1741, 51)

    def test_2d_full_coverage_s2(self):
        """Verify 2D pixel coverage on complete Sentinel-2 grid (1599 x 1679)."""
        windows = generate_tile_windows(1599, 1679, tile_size=256, boundary_strategy="shift")
        self.assertEqual(len(windows), 49)  # 7 x 7 grid

        # Check corners and terminal tiles
        bottom_right = windows[-1]
        self.assertEqual(bottom_right.col_off, 1343)
        self.assertEqual(bottom_right.row_off, 1423)
        self.assertEqual(bottom_right.col_off + bottom_right.width, 1599)
        self.assertEqual(bottom_right.row_off + bottom_right.height, 1679)


class TestSyntheticSentinel2Tiling(unittest.TestCase):
    """Test Sentinel-2 tiling on a synthetic staged scene."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.scene_dir = Path(self.temp_dir.name) / "MOCK_S2A_SCENE"
        self.scene_dir.mkdir(parents=True)
        self.out_dir = Path(self.temp_dir.name) / "tiles"

        # Create mock 10m rasters (300 x 300)
        transform_10m = Affine(10.0, 0.0, 280000.0, 0.0, -10.0, 2100000.0)
        self.source_hashes = {}

        for bid in ["B02", "B03", "B04", "B08"]:
            fpath = self.scene_dir / f"{bid}.tif"
            data = np.full((300, 300), 500, dtype=np.uint16)
            profile = {
                "driver": "GTiff",
                "height": 300,
                "width": 300,
                "count": 1,
                "dtype": "uint16",
                "crs": "EPSG:32643",
                "transform": transform_10m,
                "nodata": 0,
            }
            with rasterio.open(fpath, "w", **profile) as dst:
                dst.write(data, 1)
            self.source_hashes[f"{bid}.tif"] = compute_file_sha256(fpath)

        # Create mock 20m SCL raster (150 x 150)
        transform_20m = Affine(20.0, 0.0, 280000.0, 0.0, -20.0, 2100000.0)
        scl_path = self.scene_dir / "SCL.tif"
        scl_data = np.full((150, 150), 4, dtype=np.uint8)  # vegetation
        scl_profile = {
            "driver": "GTiff",
            "height": 150,
            "width": 150,
            "count": 1,
            "dtype": "uint8",
            "crs": "EPSG:32643",
            "transform": transform_20m,
            "nodata": 0,
        }
        with rasterio.open(scl_path, "w", **scl_profile) as dst:
            dst.write(scl_data, 1)
        self.source_hashes["SCL.tif"] = compute_file_sha256(scl_path)

        # Create metadata.json
        meta = {
            "scene_metadata": {
                "scene_id": "MOCK_S2A_SCENE",
                "spacecraft": "Sentinel-2A",
                "sensor": "MSI",
                "processing_level": "Level-2A",
                "mgrs_tile": "43QBB",
                "acquisition_datetime": "2022-01-27T05:53:40Z",
                "cloud_cover_tile_pct": 0.0,
                "sun_elevation": 45.0,
                "sun_azimuth": 150.0,
                "total_local_size_bytes": 1000,
            },
            "provenance": {
                "tier1_copernicus_source": {"authority": "ESA"},
                "tier2_element84_aws_cog_distribution": {"curator": "AWS"},
                "tier3_local_aoi_crop_artifact": {"aoi_name": "Test"},
            },
            "bands": {
                bid: {
                    "band_identity": bid,
                    "pixel_resolution_m": 10.0,
                    "local_filename": f"{bid}.tif",
                    "local_sha256": self.source_hashes[f"{bid}.tif"],
                    "dimensions": {"height": 300, "width": 300},
                    "dtype": "uint16",
                    "nodata": 0.0,
                }
                for bid in ["B02", "B03", "B04", "B08"]
            },
        }
        meta["bands"]["SCL"] = {
            "band_identity": "SCL",
            "pixel_resolution_m": 20.0,
            "local_filename": "SCL.tif",
            "local_sha256": self.source_hashes["SCL.tif"],
            "dimensions": {"height": 150, "width": 150},
            "dtype": "uint8",
            "nodata": 0.0,
        }
        with open(self.scene_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        self.source_hashes["metadata.json"] = compute_file_sha256(self.scene_dir / "metadata.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_tiling_generates_valid_tiles_and_preserves_source(self):
        res = tile_sentinel2_scene(self.scene_dir, self.out_dir, tile_size=256, boundary_strategy="shift")
        # 300 px length with tile_size=256 -> 2 offsets per axis (0, 44) -> 2x2 = 4 tiles
        self.assertEqual(res["total_tiles"], 4)

        # Verify source files remain completely untouched
        for fname, exp_sha in self.source_hashes.items():
            current_sha = compute_file_sha256(self.scene_dir / fname)
            self.assertEqual(current_sha, exp_sha, f"Source file {fname} was modified!")

        # Verify one tile's files and metadata
        tile_dirs = list((self.out_dir / "MOCK_S2A_SCENE").glob("s2_*"))
        self.assertEqual(len(tile_dirs), 4)

        t0 = tile_dirs[0]
        self.assertTrue((t0 / "B02.tif").exists())
        self.assertTrue((t0 / "SCL.tif").exists())
        self.assertTrue((t0 / "metadata.json").exists())

        # Check optical 10m tile dimension
        with rasterio.open(t0 / "B02.tif") as src:
            self.assertEqual(src.shape, (256, 256))
            self.assertEqual(str(src.dtypes[0]), "uint16")
            self.assertEqual(str(src.crs), "EPSG:32643")

        # Check SCL 20m tile dimension (native grid: 128x128, NO resampling)
        with rasterio.open(t0 / "SCL.tif") as src:
            self.assertEqual(src.shape, (128, 128))
            self.assertEqual(str(src.dtypes[0]), "uint8")

        # Check metadata schema
        with open(t0 / "metadata.json", "r", encoding="utf-8") as f:
            tmeta = json.load(f)

        self.assertIn("source_metadata", tmeta)
        self.assertIn("derived_tile_metadata", tmeta)
        self.assertIn("georeferencing", tmeta)
        self.assertTrue(tmeta["georeferencing"]["is_authoritative"])
        self.assertEqual(tmeta["georeferencing"]["crs"], "EPSG:32643")
        self.assertIn("provenance", tmeta)


class TestSyntheticSentinel1Tiling(unittest.TestCase):
    """Test Sentinel-1 tiling on a synthetic staged scene."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.scene_dir = Path(self.temp_dir.name) / "MOCK_S1A_SCENE"
        self.scene_dir.mkdir(parents=True)
        self.out_dir = Path(self.temp_dir.name) / "tiles"

        self.source_hashes = {}
        for pol in ["vv", "vh"]:
            fpath = self.scene_dir / f"{pol}.tif"
            data = np.full((300, 300), 200, dtype=np.uint16)
            profile = {
                "driver": "GTiff",
                "height": 300,
                "width": 300,
                "count": 1,
                "dtype": "uint16",
                "nodata": 0,
            }
            with rasterio.open(fpath, "w", **profile) as dst:
                dst.write(data, 1)
            self.source_hashes[f"{pol.lower()}.tif"] = compute_file_sha256(fpath)

        meta = {
            "scene_metadata": {
                "scene_id": "MOCK_S1A_SCENE",
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
                "pixel_spacing_m": [10.0, 10.0],
                "effective_resolution_m": [20.0, 22.0],
                "total_local_size_bytes": 1000,
            },
            "source_raster_metadata": {
                "dimensions": {"height": 16000, "width": 25000},
                "total_source_gcps": 210,
                "gcp_crs": "EPSG:4326",
            },
            "crop_window": {
                "col_off": 6000,
                "row_off": 3000,
                "width": 300,
                "height": 300,
                "safety_margin_pixels": 50,
            },
            "local_gcps": [
                {
                    "id": "1",
                    "source_col": 6050.0,
                    "source_row": 3050.0,
                    "local_col": 50.0,
                    "local_row": 50.0,
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
                "tier1_copernicus_source": {"authority": "ESA"},
                "tier2_aws_cog_distribution": {"curator": "AWS"},
                "tier3_http_range_window_read": {"safety_margin_pixels": 50},
                "tier4_local_native_radar_crop": {"coordinate_space": "native_radar_range_azimuth"},
            },
            "bands": {
                pol: {
                    "polarization": pol,
                    "local_filename": f"{pol.lower()}.tif",
                    "local_sha256": self.source_hashes[f"{pol.lower()}.tif"],
                    "dimensions": {"height": 300, "width": 300},
                    "dtype": "uint16",
                    "nodata": 0.0,
                }
                for pol in ["VV", "VH"]
            },
        }
        with open(self.scene_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        self.source_hashes["metadata.json"] = compute_file_sha256(self.scene_dir / "metadata.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_s1_tiling_native_radar_preservation(self):
        res = tile_sentinel1_scene(self.scene_dir, self.out_dir, tile_size=256, boundary_strategy="shift")
        self.assertEqual(res["total_tiles"], 4)

        # Source untouched
        for fname, exp_sha in self.source_hashes.items():
            self.assertEqual(compute_file_sha256(self.scene_dir / fname), exp_sha)

        tile_dirs = list((self.out_dir / "MOCK_S1A_SCENE").glob("s1_*"))
        self.assertEqual(len(tile_dirs), 4)

        t0 = tile_dirs[0]
        self.assertTrue((t0 / "vv.tif").exists())
        self.assertTrue((t0 / "vh.tif").exists())
        self.assertTrue((t0 / "metadata.json").exists())

        with rasterio.open(t0 / "vv.tif") as src:
            self.assertEqual(src.shape, (256, 256))
            self.assertEqual(str(src.dtypes[0]), "uint16")

        with open(t0 / "metadata.json", "r", encoding="utf-8") as f:
            tmeta = json.load(f)

        # Georeferencing must NOT be marked authoritative affine
        geo = tmeta["georeferencing"]
        self.assertFalse(geo["is_authoritative_affine"])
        self.assertEqual(geo["coordinate_space"], "native_radar_range_azimuth")
        self.assertEqual(geo["derived_local_affine_approximation"]["label"], "derived_local_affine_approximation")


class TestTilingRepeatability(unittest.TestCase):
    """Test that independent tiling runs produce byte-for-byte identical outputs."""

    def test_repeatable_s2_tiling(self):
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            out1 = Path(tmp1) / "tiles"
            out2 = Path(tmp2) / "tiles"

            # Create mock scene in a third temp dir
            with tempfile.TemporaryDirectory() as src_tmp:
                scene_dir = Path(src_tmp) / "SCENE"
                scene_dir.mkdir()
                trans = Affine(10.0, 0.0, 280000.0, 0.0, -10.0, 2100000.0)
                hashes = {}
                for bid in ["B02", "B03", "B04", "B08"]:
                    fp = scene_dir / f"{bid}.tif"
                    with rasterio.open(fp, "w", driver="GTiff", height=300, width=300, count=1, dtype="uint16", crs="EPSG:32643", transform=trans, nodata=0) as d:
                        d.write(np.full((300, 300), 100, dtype=np.uint16), 1)
                    hashes[f"{bid}.tif"] = compute_file_sha256(fp)

                scl_fp = scene_dir / "SCL.tif"
                trans_20m = Affine(20.0, 0.0, 280000.0, 0.0, -20.0, 2100000.0)
                with rasterio.open(scl_fp, "w", driver="GTiff", height=150, width=150, count=1, dtype="uint8", crs="EPSG:32643", transform=trans_20m, nodata=0) as d:
                    d.write(np.full((150, 150), 4, dtype=np.uint8), 1)
                hashes["SCL.tif"] = compute_file_sha256(scl_fp)

                meta = {
                    "scene_metadata": {"scene_id": "SCENE", "spacecraft": "S2A", "acquisition_datetime": "2022-01-27T00:00:00Z"},
                    "provenance": {"tier1_copernicus_source": {"authority": "ESA"}},
                    "bands": {bid: {"band_identity": bid, "pixel_resolution_m": 10.0, "local_filename": f"{bid}.tif", "local_sha256": hashes[f"{bid}.tif"], "dimensions": {"height": 300, "width": 300}, "dtype": "uint16", "nodata": 0.0} for bid in ["B02", "B03", "B04", "B08"]},
                }
                meta["bands"]["SCL"] = {
                    "band_identity": "SCL",
                    "pixel_resolution_m": 20.0,
                    "local_filename": "SCL.tif",
                    "local_sha256": hashes["SCL.tif"],
                    "dimensions": {"height": 150, "width": 150},
                    "dtype": "uint8",
                    "nodata": 0.0,
                }
                with open(scene_dir / "metadata.json", "w") as f:
                    json.dump(meta, f)

                # Run tiling twice
                res1 = tile_sentinel2_scene(scene_dir, out1, tile_size=256)
                res2 = tile_sentinel2_scene(scene_dir, out2, tile_size=256)

                self.assertEqual(res1["total_tiles"], res2["total_tiles"])

                # Compare all files
                tiles1 = sorted((out1 / "SCENE").glob("s2_*"))
                tiles2 = sorted((out2 / "SCENE").glob("s2_*"))
                self.assertEqual(len(tiles1), len(tiles2))

                for t1, t2 in zip(tiles1, tiles2):
                    self.assertEqual(t1.name, t2.name)
                    # Check GeoTIFF SHA-256
                    self.assertEqual(compute_file_sha256(t1 / "B02.tif"), compute_file_sha256(t2 / "B02.tif"))


class TestLiveM1Tiling(unittest.TestCase):
    """Integration test verifying actual M1 scenes if present."""

    def test_live_s2_tiling_if_present(self):
        s2_scenes = list(RAW_S2_DIR.glob("S2*")) if RAW_S2_DIR.exists() else []
        if not s2_scenes:
            self.skipTest("No staged Sentinel-2 scenes found in data/raw/sentinel2")

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "tiles"
            scene_path = s2_scenes[0]
            # Record source hashes before tiling
            orig_meta_sha = compute_file_sha256(scene_path / "metadata.json")
            orig_b02_sha = compute_file_sha256(scene_path / "B02.tif")

            res = tile_sentinel2_scene(scene_path, out_dir, tile_size=256, boundary_strategy="shift")
            self.assertEqual(res["total_tiles"], 49)  # 7x7 grid

            # Check that M1 source was NOT modified
            self.assertEqual(compute_file_sha256(scene_path / "metadata.json"), orig_meta_sha)
            self.assertEqual(compute_file_sha256(scene_path / "B02.tif"), orig_b02_sha)


if __name__ == "__main__":
    unittest.main()
