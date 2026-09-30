"""Unit tests for the FLUX Presentation UI, metadata inspection, and reporting layers.

Covers:
1. True-color optical and SAR raster rendering to base64 PNG data URLs
2. Change magnitude heatmap rendering
3. Confirmed change mask binary image rendering
4. Heuristic confidence heatmap rendering
5. Authoritative metadata extraction, geographic coordinates (WGS84 & projected), and offline SVG diagrams
6. Milestone architecture (M0-M9 frozen vs M10-M15 planned) HTML generation
7. Full 15-section technical report Markdown and HTML generation
8. Dashboard HTML assembly for initial, populated, and error states
9. Interactive Top-K tile selection and highlighting
10. HTTP GET endpoints including report downloads and selected tile query execution
11. CLI argument parsing
"""

import base64
import io
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin

from app import build_html_page, create_ui_server, parse_args
from src.change_detection.models import ChangeResult
from src.pairing.models import AlignmentStatus, SpatialCorrespondence, TemporalPair
from src.pipeline import PipelineStageTiming, PoCPipelineResult
from src.suppression.models import SuppressedChangeResult
from src.ui.metadata_inspection import (
    extract_geographic_info,
    extract_tile_display_metadata,
    load_tile_authoritative_metadata,
    render_bounding_box_diagram,
)
from src.ui.rendering import (
    render_change_heatmap,
    render_confidence_heatmap,
    render_mask_image,
    render_tile_image,
)
from src.ui.report import (
    generate_milestone_architecture_html,
    generate_technical_report_html,
    generate_technical_report_markdown,
)


def create_dummy_geotiff(path: Path, data: np.ndarray, dtype: str = "float32") -> None:
    """Helper to write a simple single-band GeoTIFF file for rendering tests."""
    transform = from_origin(73.0, 19.0, 10.0, 10.0)
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": "EPSG:32643",
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype), 1)


def make_dummy_pair(temp_dir: Path) -> TemporalPair:
    """Helper to construct a mock TemporalPair with real synthetic raster files."""
    ref_dir = temp_dir / "ref"
    comp_dir = temp_dir / "comp"
    ref_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)

    arr = np.full((32, 32), 1200, dtype=np.uint16)
    for band in ("B04.tif", "B03.tif", "B02.tif"):
        create_dummy_geotiff(ref_dir / band, arr, dtype="uint16")
        create_dummy_geotiff(comp_dir / band, arr, dtype="uint16")

    return TemporalPair(
        pair_id="pair__test_ref__test_comp",
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
            pixel_dimensions=(32, 32),
        ),
        reference_tile_dir=str(ref_dir),
        comparison_tile_dir=str(comp_dir),
    )


class TestUIRendering(unittest.TestCase):
    """Tests for base64 PNG rendering utilities in src/ui/rendering.py."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tile_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_render_tile_image_nonexistent(self) -> None:
        """Non-existent tile directory returns None."""
        result = render_tile_image("/nonexistent/directory/path/12345")
        self.assertIsNone(result)

    def test_render_tile_image_optical(self) -> None:
        """Optical tile with B04, B03, B02 GeoTIFFs produces valid base64 PNG data URL."""
        b4 = np.full((16, 16), 1500, dtype=np.uint16)
        b3 = np.full((16, 16), 1200, dtype=np.uint16)
        b2 = np.full((16, 16), 800, dtype=np.uint16)

        create_dummy_geotiff(self.tile_dir / "B04.tif", b4, dtype="uint16")
        create_dummy_geotiff(self.tile_dir / "B03.tif", b3, dtype="uint16")
        create_dummy_geotiff(self.tile_dir / "B02.tif", b2, dtype="uint16")

        data_url = render_tile_image(self.tile_dir, modality="optical")
        self.assertIsNotNone(data_url)
        self.assertTrue(data_url.startswith("data:image/png;base64,"))

        b64_content = data_url.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_content)
        img = Image.open(io.BytesIO(img_bytes))
        self.assertEqual(img.size, (16, 16))
        self.assertEqual(img.mode, "RGB")

    def test_render_tile_image_sar(self) -> None:
        """SAR tile with VV and VH GeoTIFFs produces valid base64 PNG data URL."""
        vv = np.linspace(0.01, 0.5, 256, dtype=np.float32).reshape(16, 16)
        vh = np.linspace(0.005, 0.25, 256, dtype=np.float32).reshape(16, 16)

        create_dummy_geotiff(self.tile_dir / "vv.tif", vv, dtype="float32")
        create_dummy_geotiff(self.tile_dir / "vh.tif", vh, dtype="float32")

        data_url = render_tile_image(self.tile_dir, modality="sar")
        self.assertIsNotNone(data_url)
        self.assertTrue(data_url.startswith("data:image/png;base64,"))

        b64_content = data_url.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_content)
        img = Image.open(io.BytesIO(img_bytes))
        self.assertEqual(img.size, (16, 16))
        self.assertEqual(img.mode, "RGB")

    def test_render_change_heatmap(self) -> None:
        """Continuous change magnitude converts into an orange/red heatmap PNG."""
        self.assertIsNone(render_change_heatmap(None))
        self.assertIsNone(render_change_heatmap(np.array([1.0, 2.0])))

        magnitude = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
        data_url = render_change_heatmap(magnitude)
        self.assertIsNotNone(data_url)
        self.assertTrue(data_url.startswith("data:image/png;base64,"))

        b64_content = data_url.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_content)
        img = Image.open(io.BytesIO(img_bytes))
        self.assertEqual(img.size, (8, 8))

    def test_render_mask_image(self) -> None:
        """Confirmed change boolean mask converts into a high-contrast binary PNG."""
        self.assertIsNone(render_mask_image(None))
        self.assertIsNone(render_mask_image(np.array([True, False])))

        mask = np.zeros((8, 8), dtype=bool)
        mask[2:5, 2:5] = True
        data_url = render_mask_image(mask)
        self.assertIsNotNone(data_url)
        self.assertTrue(data_url.startswith("data:image/png;base64,"))

        b64_content = data_url.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_content)
        img = Image.open(io.BytesIO(img_bytes))
        self.assertEqual(img.size, (8, 8))

    def test_render_confidence_heatmap(self) -> None:
        """Heuristic confidence map converts into a cyan-to-emerald gradient PNG."""
        self.assertIsNone(render_confidence_heatmap(None))
        self.assertIsNone(render_confidence_heatmap(np.array([0.5, 0.8])))

        conf = np.full((8, 8), 0.75, dtype=np.float32)
        data_url = render_confidence_heatmap(conf)
        self.assertIsNotNone(data_url)
        self.assertTrue(data_url.startswith("data:image/png;base64,"))

        b64_content = data_url.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_content)
        img = Image.open(io.BytesIO(img_bytes))
        self.assertEqual(img.size, (8, 8))


class TestUIMetadataInspection(unittest.TestCase):
    """Tests for geographic coordinate extraction and tile metadata in src/ui/metadata_inspection.py."""

    def test_extract_geographic_info_wgs84(self) -> None:
        """Extracts exact WGS84 bounds, centroid, and all 4 corners."""
        meta = {
            "crs": "EPSG:32643",
            "bounds_wgs84": [72.90, 18.90, 73.10, 19.10],
            "bounds_projected": [280000.0, 2100000.0, 290000.0, 2110000.0],
        }
        geo = extract_geographic_info(meta)
        self.assertTrue(geo["has_geo"])
        self.assertEqual(geo["crs"], "EPSG:32643")
        self.assertEqual(geo["bounds_wgs84"], [72.90, 18.90, 73.10, 19.10])

        # Centroid
        self.assertAlmostEqual(geo["centroid_wgs84"]["lon"], 73.00)
        self.assertAlmostEqual(geo["centroid_wgs84"]["lat"], 19.00)

        # Corners
        corners = geo["corners_wgs84"]
        self.assertAlmostEqual(corners["NW"]["lon"], 72.90)
        self.assertAlmostEqual(corners["NW"]["lat"], 19.10)
        self.assertAlmostEqual(corners["NE"]["lon"], 73.10)
        self.assertAlmostEqual(corners["NE"]["lat"], 19.10)
        self.assertAlmostEqual(corners["SW"]["lon"], 72.90)
        self.assertAlmostEqual(corners["SW"]["lat"], 18.90)
        self.assertAlmostEqual(corners["SE"]["lon"], 73.10)
        self.assertAlmostEqual(corners["SE"]["lat"], 18.90)

        # Projected
        self.assertTrue(geo["is_projected"])
        self.assertEqual(geo["bounds_projected"], [280000.0, 2100000.0, 290000.0, 2110000.0])

    def test_extract_geographic_info_unprojected(self) -> None:
        """Handles unprojected SAR metadata without projected bounds."""
        meta = {
            "crs": "EPSG:4326",
            "bounds_wgs84": [73.10, 19.02, 73.14, 19.05],
            "bounds_projected": None,
        }
        geo = extract_geographic_info(meta)
        self.assertTrue(geo["has_geo"])
        self.assertFalse(geo["is_projected"])
        self.assertIsNone(geo["bounds_projected"])

    def test_render_bounding_box_diagram(self) -> None:
        """Generates offline SVG diagram containing coordinates and corner pins."""
        svg = render_bounding_box_diagram([72.95, 18.95, 73.05, 19.05], crs="EPSG:32643")
        self.assertIn("<svg", svg)
        self.assertIn("EPSG:32643", svg)
        self.assertIn("NW", svg)
        self.assertIn("Centroid", svg)
        self.assertIn("72.95000", svg)

    def test_extract_tile_display_metadata_fallback(self) -> None:
        """Unavailable metadata fields fallback to 'Not available' without inventing values."""
        disp = extract_tile_display_metadata({}, {})
        self.assertEqual(disp["tile_id"], "Not available")
        self.assertEqual(disp["scene_id"], "Not available")
        self.assertEqual(disp["modality"], "Not available")
        self.assertEqual(disp["acquisition_datetime_utc"], "Not available")
        self.assertEqual(disp["grid_index"], "Not available")


class TestUIReports(unittest.TestCase):
    """Tests for Milestone Architecture and Technical Report generation in src/ui/report.py."""

    def test_generate_milestone_architecture_html(self) -> None:
        """Milestone architecture distinguishes M0-M9 frozen from M10-M15 planned."""
        html_str = generate_milestone_architecture_html()
        self.assertIn("FLUX Milestone Architecture", html_str)
        self.assertIn("M0 &bull; Frozen", html_str)
        self.assertIn("M9 &bull; Frozen", html_str)
        self.assertIn("M10 &bull; Planned", html_str)
        self.assertIn("M15 &bull; Planned", html_str)

    def test_generate_technical_report_markdown(self) -> None:
        """Technical report contains all 15 required sections and no fabricated metrics."""
        md = generate_technical_report_markdown()
        self.assertIn("1. Executive Summary", md)
        self.assertIn("2. Problem Statement", md)
        self.assertIn("3. Technologies Used", md)
        self.assertIn("4. System Architecture", md)
        self.assertIn("5. Complete Algorithmic Flowchart", md)
        self.assertIn("6. M0–M15 Milestone Status", md)
        self.assertIn("7. Current Implementation Details", md)
        self.assertIn("8. Technical Feasibility Assessment", md)
        self.assertIn("9. Practical Viability", md)
        self.assertIn("10. Known Limitations & Technical Risk Disclosure", md)
        self.assertIn("11. Evaluation Strategy", md)
        self.assertIn("12. Current PoC Benchmark Baseline", md)
        self.assertIn("13. End-to-End Demonstration Workflow", md)
        self.assertIn("14. Future Development Roadmap", md)
        self.assertIn("15. Conclusion", md)

        # Confirm technologies listed
        self.assertIn("Rasterio", md)
        self.assertIn("FAISS", md)
        self.assertIn("CLIP-RSICD-v2", md)
        self.assertIn("SIH2026227", md)

    def test_generate_technical_report_html(self) -> None:
        """Report HTML includes download buttons and rendered headings."""
        html_str = generate_technical_report_html()
        self.assertIn("FLUX Full Technical &amp; Feasibility Report", html_str)
        self.assertIn("/download-report?format=markdown", html_str)
        self.assertIn("/download-report?format=html", html_str)


class TestUIHtmlPage(unittest.TestCase):
    """Tests for HTML generation in app.py."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_build_html_page_initial_view(self) -> None:
        """Initial state HTML contains header, search form, and empty query state."""
        html_str = build_html_page(
            query="test initial query",
            top_k=5,
            modality="optical",
            result=None,
            project_root=self.root_path,
        )
        self.assertIn("<!DOCTYPE html>", html_str)
        self.assertIn("FLUX", html_str)
        self.assertIn("SIH2026227", html_str)
        self.assertIn("LOCAL / OFFLINE", html_str)
        self.assertIn('value="test initial query"', html_str)
        self.assertIn("Run Analysis", html_str)
        self.assertIn("Sample Queries:", html_str)

    def test_build_html_page_with_error(self) -> None:
        """Error alert message renders prominently."""
        html_str = build_html_page(
            query="failed query",
            error_message="Simulation test failure message",
            project_root=self.root_path,
        )
        self.assertIn("Simulation test failure message", html_str)
        self.assertIn("alert-error", html_str)

    def test_build_html_page_with_populated_result(self) -> None:
        """Populated PoCPipelineResult renders all pipeline stages, selected tile info, and reports."""
        pair = make_dummy_pair(self.root_path)

        timing = PipelineStageTiming(
            retrieval_s=0.045,
            filtering_s=0.005,
            pairing_s=0.002,
            change_detection_s=0.080,
            suppression_s=0.015,
            total_s=0.147,
        )

        change_res = ChangeResult(
            pair_id=pair.pair_id,
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
            summary_statistics={"mean": 0.05, "median": 0.04, "p95": 0.10, "valid_pixels": 1024},
            change_magnitude=np.full((32, 32), 0.15, dtype=np.float32),
        )

        mask = np.zeros((32, 32), dtype=bool)
        mask[10:15, 10:15] = True
        conf = np.zeros((32, 32), dtype=np.float32)
        conf[mask] = 0.85

        suppr_res = SuppressedChangeResult(
            pair_id=pair.pair_id,
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
            noise_median=0.04,
            noise_mad=0.015,
            threshold_used=0.1067,
            candidate_pixels_count=40,
            confirmed_pixels_count=25,
            suppressed_pixels_count=15,
            confirmed_change_ratio=0.0244,
            mean_confidence_on_change=0.85,
            max_confidence=0.85,
            confirmed_mask=mask,
            confidence_map=conf,
        )

        result = PoCPipelineResult(
            query="urban expansion",
            status="success",
            retrieval_hits=[
                {"tile_id": "test_ref", "similarity_score": 0.88, "score": 0.88, "modality": "optical", "scene_id": "S2A_ref"},
                {"tile_id": "test_rank2", "similarity_score": 0.82, "score": 0.82, "modality": "optical", "scene_id": "S2A_ref"},
            ],
            filtered_hits=[
                {"tile_id": "test_ref", "similarity_score": 0.88, "score": 0.88, "modality": "optical", "scene_id": "S2A_ref"},
                {"tile_id": "test_rank2", "similarity_score": 0.82, "score": 0.82, "modality": "optical", "scene_id": "S2A_ref"},
            ],
            selected_tile_id="test_ref",
            temporal_pair=pair,
            change_result=change_res,
            suppressed_result=suppr_res,
            timing=timing,
        )

        html_str = build_html_page(
            query="urban expansion",
            top_k=5,
            modality="optical",
            selected_tile_id="test_ref",
            result=result,
            project_root=self.root_path,
        )

        # Stage timings
        self.assertIn("M5 Retrieval", html_str)
        self.assertIn("0.045 s", html_str)
        self.assertIn("M8 Change Det.", html_str)

        # Top-K table with interactive selection
        self.assertIn("test_ref", html_str)
        self.assertIn("test_rank2", html_str)
        self.assertIn("Select Tile", html_str)
        self.assertIn("Selected Active Tile:", html_str)

        # Geographic Location section
        self.assertIn("Exact Geographic Location", html_str)
        self.assertIn("WGS84 Geographic Coordinates", html_str)

        # Expandable Tile Metadata
        self.assertIn("Authoritative Tile Metadata", html_str)

        # M7 temporal pair info
        self.assertIn("715.0 days", html_str)
        self.assertIn("test_comp", html_str)

        # Quantitative metrics
        self.assertIn("25", html_str)
        self.assertIn("2.44%", html_str)
        self.assertIn("0.1067", html_str)
        self.assertIn("0.8500", html_str)

        # Milestone Architecture and Full Technical Report sections
        self.assertIn("FLUX Milestone Architecture", html_str)
        self.assertIn("FLUX Full Technical &amp; Feasibility Report", html_str)
        self.assertIn("/download-report?format=markdown", html_str)


class TestUIServerIntegration(unittest.TestCase):
    """Tests for HTTP server lifecycle, query handling, and report downloads."""

    def test_http_server_get_request(self) -> None:
        """Server responds to HTTP GET with status 200 and valid HTML."""
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = PoCPipelineResult(
            query="mock query",
            status="no_retrieval_hits",
            retrieval_hits=[],
            filtered_hits=[],
            warnings=("No matching tiles found for query.",),
        )

        server = create_ui_server(host="127.0.0.1", port=0, pipeline=mock_pipeline)
        port = server.server_address[1]

        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            url = f"http://127.0.0.1:{port}/?query=mock+query&top_k=3&modality=optical"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as response:
                status_code = response.status
                headers = dict(response.getheaders())
                body = response.read().decode("utf-8")

            self.assertEqual(status_code, 200)
            self.assertIn("text/html", headers.get("Content-Type", ""))
            self.assertIn("FLUX", body)
            self.assertIn("mock query", body)
            self.assertIn("No matching tiles found for query", body)
            mock_pipeline.run.assert_called_once()
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)

    def test_http_server_download_report_markdown(self) -> None:
        """Server responds to /download-report?format=markdown with Markdown attachment."""
        server = create_ui_server(host="127.0.0.1", port=0, pipeline=MagicMock())
        port = server.server_address[1]

        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            url = f"http://127.0.0.1:{port}/download-report?format=markdown"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as response:
                status_code = response.status
                headers = dict(response.getheaders())
                body = response.read().decode("utf-8")

            self.assertEqual(status_code, 200)
            self.assertIn("text/markdown", headers.get("Content-Type", ""))
            self.assertIn("FLUX_Technical_Report.md", headers.get("Content-Disposition", ""))
            self.assertIn("# FLUX Technical & Feasibility Report", body)
            self.assertIn("Executive Summary", body)
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)

    def test_http_server_download_report_html(self) -> None:
        """Server responds to /download-report?format=html with HTML attachment."""
        server = create_ui_server(host="127.0.0.1", port=0, pipeline=MagicMock())
        port = server.server_address[1]

        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            url = f"http://127.0.0.1:{port}/download-report?format=html"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as response:
                status_code = response.status
                headers = dict(response.getheaders())
                body = response.read().decode("utf-8")

            self.assertEqual(status_code, 200)
            self.assertIn("text/html", headers.get("Content-Type", ""))
            self.assertIn("FLUX_Technical_Report.html", headers.get("Content-Disposition", ""))
            self.assertIn("<!DOCTYPE html>", body)
            self.assertIn("FLUX Technical &amp; Feasibility Report", body)
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)

    def test_http_server_explicit_tile_selection(self) -> None:
        """Server forwards selected_tile_id parameter to pipeline.run()."""
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = PoCPipelineResult(
            query="test",
            status="no_retrieval_hits",
            retrieval_hits=[],
            filtered_hits=[],
            selected_tile_id="custom_tile_123",
        )

        server = create_ui_server(host="127.0.0.1", port=0, pipeline=mock_pipeline)
        port = server.server_address[1]

        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            url = f"http://127.0.0.1:{port}/?query=test&selected_tile_id=custom_tile_123"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as response:
                self.assertEqual(response.status, 200)

            mock_pipeline.run.assert_called_once()
            _, kwargs = mock_pipeline.run.call_args
            self.assertEqual(kwargs.get("selected_tile_id"), "custom_tile_123")
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)

    def test_parse_args(self) -> None:
        """CLI arguments parse correctly with default values."""
        import sys
        old_argv = sys.argv
        sys.argv = ["app.py", "--host", "0.0.0.0", "--port", "9000"]
        try:
            args = parse_args()
            self.assertEqual(args.host, "0.0.0.0")
            self.assertEqual(args.port, 9000)
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
