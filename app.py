"""FLUX Presentation UI: Local Offline Multi-Temporal Satellite Semantic Search & Change Analysis.

SIH Problem Statement: SIH2026227
Demonstrates the complete end-to-end pipeline:
  Natural-language query
    ↓
  M5: Semantic Retrieval (Vector Index Similarity)
    ↓
  M6: Metadata Filtering (Modality, Dates, Scene, BBox)
    ↓
  Interactive Top-K Tile Selection & Geographic Location Inspection
    ↓
  M7: Temporal Pairing (Counterpart Resolution & Spatial Alignment)
    ↓
  M8: Spectral Change Detection (Continuous Euclidean Distance)
    ↓
  M9: Adaptive False-Alarm Suppression (Robust MAD Noise Thresholding & Spatial Filtering)
    ↓
  Interactive Visual Results (True-color Satellite Imagery, Change Heatmaps, Confirmed Masks)
    ↓
  Milestone Architecture & Full Technical / Feasibility Report

Architecture Rule:
Directly invokes `src.pipeline.PoCPipeline` and consumes `PoCPipelineResult`.
Does not duplicate or reimplement M1-M9 milestone logic.
Runs 100% locally and offline without external CDN, mapping tiles, or cloud dependencies.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html
import json
from pathlib import Path
import sys
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import PoCPipeline, PoCPipelineResult
from src.ui.change_evidence import render_change_evidence_panel
from src.ui.flowchart import generate_m1_m9_flowchart_html
from src.ui.gauges import (
    render_cloud_cover_gauge,
    render_confidence_gauge,
    render_numeric_gauge,
    render_similarity_gauge,
    render_spectral_distance_gauge,
    render_suppression_ratio_gauge,
    render_valid_ratio_gauge,
)
from src.ui.grid_visualization import render_spatial_tile_grid_html
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

DEFAULT_QUERY = "urban development around Navi Mumbai"


def build_html_page(
    query: str = DEFAULT_QUERY,
    top_k: int = 5,
    modality: str = "optical",
    date_from: str = "",
    date_to: str = "",
    selected_tile_id: str = "",
    result: Optional[PoCPipelineResult] = None,
    error_message: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> str:
    """Generate the full responsive HTML/CSS dashboard for the FLUX presentation UI."""
    root = project_root if project_root else ROOT_DIR

    # Render imagery if result is available
    ref_img_data = None
    comp_img_data = None
    change_heatmap_data = None
    mask_img_data = None
    conf_heatmap_data = None

    if result and result.temporal_pair:
        pair = result.temporal_pair
        ref_img_data = render_tile_image(pair.reference_tile_dir, pair.modality, project_root=root)
        comp_img_data = render_tile_image(pair.comparison_tile_dir, pair.modality, project_root=root)

    if result and result.change_result:
        change_heatmap_data = render_change_heatmap(result.change_result.change_magnitude)

    if result and result.suppressed_result:
        mask_img_data = render_mask_image(result.suppressed_result.confirmed_mask)
        conf_heatmap_data = render_confidence_heatmap(result.suppressed_result.confidence_map)

    # HTML template with pure inline CSS (zero external network dependencies)
    css = """
    :root {
        --bg-main: #0c0f17;
        --bg-card: #151a24;
        --bg-card-hover: #1c2331;
        --border-color: #242c3d;
        --border-focus: #3b82f6;
        --text-main: #f1f5f9;
        --text-muted: #94a3b8;
        --accent-blue: #3b82f6;
        --accent-cyan: #06b6d4;
        --accent-emerald: #10b981;
        --accent-amber: #f59e0b;
        --accent-red: #ef4444;
        --badge-bg: #1e293b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        background-color: var(--bg-main);
        color: var(--text-main);
        line-height: 1.5;
        padding-bottom: 60px;
    }
    .container { max-width: 1300px; margin: 0 auto; padding: 0 24px; }

    /* Header */
    header {
        background-color: var(--bg-card);
        border-bottom: 1px solid var(--border-color);
        padding: 20px 0;
        margin-bottom: 30px;
    }
    .header-content { display: flex; justify-content: space-between; align-items: center; }
    .brand-title { font-size: 24px; font-weight: 800; letter-spacing: -0.5px; display: flex; align-items: center; gap: 10px; }
    .brand-title .logo { color: var(--accent-cyan); }
    .brand-subtitle { font-size: 13px; color: var(--text-muted); margin-top: 2px; }
    .badges { display: flex; gap: 8px; align-items: center; }
    .badge {
        font-size: 11px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 9999px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-sih { background: rgba(59, 130, 246, 0.2); color: var(--accent-blue); border: 1px solid rgba(59, 130, 246, 0.4); }
    .badge-offline { background: rgba(16, 185, 129, 0.2); color: var(--accent-emerald); border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-m { background: var(--badge-bg); color: var(--text-muted); border: 1px solid var(--border-color); }

    /* Cards */
    .card {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .card-title {
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
        color: var(--text-main);
    }
    .card-title .icon { color: var(--accent-blue); }

    /* Search Form */
    .form-grid { display: grid; grid-template-columns: 1fr auto auto auto auto; gap: 12px; align-items: flex-end; }
    .form-group { display: flex; flex-direction: column; gap: 6px; }
    .form-group label { font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
    .form-control {
        background-color: var(--bg-main);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-main);
        padding: 10px 14px;
        font-size: 14px;
        outline: none;
        transition: border-color 0.15s;
    }
    .form-control:focus { border-color: var(--border-focus); }
    .btn-submit {
        background-color: var(--accent-blue);
        color: #fff;
        border: none;
        border-radius: 8px;
        padding: 10px 22px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.15s;
    }
    .btn-submit:hover { background-color: #2563eb; }

    /* Quick query chips */
    .chips { display: flex; gap: 8px; margin-top: 14px; align-items: center; flex-wrap: wrap; }
    .chips-label { font-size: 12px; color: var(--text-muted); }
    .chip {
        background-color: var(--badge-bg);
        border: 1px solid var(--border-color);
        color: var(--text-main);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        cursor: pointer;
        text-decoration: none;
        transition: all 0.15s;
    }
    .chip:hover { border-color: var(--accent-blue); color: var(--accent-blue); }

    /* Timings Bar */
    .timing-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-bottom: 24px; }
    .timing-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 12px 14px;
        text-align: center;
    }
    .timing-label { font-size: 11px; color: var(--text-muted); font-weight: 600; text-transform: uppercase; margin-bottom: 4px; }
    .timing-value { font-size: 18px; font-weight: 700; color: var(--accent-cyan); font-family: monospace; }
    .timing-total .timing-value { color: var(--accent-emerald); }

    /* Side by side display */
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
    .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }

    .image-box {
        background: var(--bg-main);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        overflow: hidden;
        text-align: center;
        padding: 12px;
    }
    .image-box img {
        width: 100%;
        max-width: 256px;
        height: auto;
        aspect-ratio: 1/1;
        border-radius: 4px;
        display: block;
        margin: 0 auto 10px auto;
        image-rendering: pixelated;
    }
    .image-title { font-size: 13px; font-weight: 700; color: var(--text-main); margin-bottom: 4px; }
    .image-meta { font-size: 11px; color: var(--text-muted); font-family: monospace; }
    .image-placeholder {
        width: 100%;
        max-width: 256px;
        height: 256px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 0 auto 10px auto;
        background: #11141c;
        border: 1px dashed var(--border-color);
        border-radius: 4px;
        color: var(--text-muted);
        font-size: 12px;
        padding: 16px;
    }

    /* Metric Cards */
    .metric-card {
        background: var(--bg-main);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 14px;
    }
    .metric-label { font-size: 11px; color: var(--text-muted); font-weight: 600; text-transform: uppercase; margin-bottom: 6px; }
    .metric-val { font-size: 20px; font-weight: 800; color: var(--text-main); font-family: monospace; }
    .metric-sub { font-size: 11px; color: var(--text-muted); margin-top: 2px; }

    /* Results Table */
    .table-container { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
    th { background: var(--bg-main); color: var(--text-muted); font-weight: 600; padding: 10px 14px; border-bottom: 1px solid var(--border-color); font-size: 11px; text-transform: uppercase; }
    td { padding: 10px 14px; border-bottom: 1px solid var(--border-color); }
    tr.selected { background: rgba(59, 130, 246, 0.12); font-weight: 600; }
    tr.selected td:first-child { border-left: 3px solid var(--accent-blue); }

    /* Buttons & Interactive items */
    .btn-select {
        background: #2563eb;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 4px;
        text-decoration: none;
        font-size: 11px;
        font-weight: 600;
        display: inline-block;
        transition: background 0.15s;
    }
    .btn-select:hover { background: #1d4ed8; }

    .btn-download {
        background: var(--accent-blue);
        color: #ffffff;
        padding: 8px 16px;
        border-radius: 6px;
        text-decoration: none;
        font-size: 12px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        transition: background 0.15s;
    }
    .btn-download:hover { background: #2563eb; }
    .btn-download-alt {
        background: #1e293b;
        color: var(--accent-cyan);
        border: 1px solid var(--accent-cyan);
    }
    .btn-download-alt:hover { background: #24344d; }

    .active-tile-banner {
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 16px;
        font-size: 13px;
        color: var(--text-main);
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
    }

    /* Top-K Tile Navigator Controls */
    .navigator-bar {
        background: var(--bg-main);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .navigator-controls {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .nav-btn {
        background: var(--bg-card);
        color: var(--text-main);
        border: 1px solid var(--border-color);
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        text-decoration: none;
        transition: all 0.15s ease;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .nav-btn:hover { background: var(--bg-card-hover); border-color: var(--accent-blue); }
    .nav-btn.disabled { opacity: 0.35; pointer-events: none; }
    .rank-btn {
        background: var(--bg-card);
        color: var(--text-muted);
        border: 1px solid var(--border-color);
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
        text-decoration: none;
        transition: all 0.15s ease;
    }
    .rank-btn:hover { color: var(--text-main); border-color: var(--accent-blue); }
    .rank-btn.active {
        background: var(--accent-blue);
        color: #ffffff;
        border-color: var(--accent-blue);
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.4);
    }

    /* Top-K Gallery Cards */
    .topk-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 14px;
        margin-bottom: 20px;
    }
    .topk-card {
        background: var(--bg-main);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        transition: all 0.2s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .topk-card.selected {
        border: 2px solid var(--accent-cyan);
        background: rgba(56, 189, 248, 0.08);
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.2);
    }
    .topk-card img {
        width: 100%;
        max-width: 180px;
        height: auto;
        aspect-ratio: 1/1;
        border-radius: 4px;
        display: block;
        margin: 0 auto 8px auto;
        border: 1px solid var(--border-color);
        image-rendering: pixelated;
    }

    /* Milestone boxes */
    .milestone-box {
        background: var(--bg-main);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 14px;
    }
    .milestone-box.implemented {
        border-left: 3px solid var(--accent-emerald);
    }
    .milestone-box.planned {
        border-left: 3px solid var(--accent-blue);
        border-style: dashed;
    }
    .milestone-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
        font-size: 13px;
    }
    .milestone-desc {
        font-size: 12px;
        color: var(--text-muted);
        line-height: 1.45;
    }

    /* Metadata details table */
    .metadata-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .metadata-table th { width: 220px; background: #11141c; color: var(--text-muted); font-weight: 600; padding: 8px 12px; border: 1px solid var(--border-color); }
    .metadata-table td { background: var(--bg-main); color: var(--text-main); padding: 8px 12px; border: 1px solid var(--border-color); font-family: monospace; }

    /* Alert / Warnings */
    .alert {
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 13px;
        margin-bottom: 20px;
        border: 1px solid transparent;
    }
    .alert-warning { background: rgba(245, 158, 11, 0.1); border-color: rgba(245, 158, 11, 0.3); color: var(--accent-amber); }
    .alert-error { background: rgba(239, 68, 68, 0.1); border-color: rgba(239, 68, 68, 0.3); color: var(--accent-red); }
    .alert-info { background: rgba(6, 182, 212, 0.1); border-color: rgba(6, 182, 212, 0.3); color: var(--accent-cyan); }

    details summary { cursor: pointer; color: var(--accent-blue); font-size: 13px; font-weight: 600; padding: 8px 0; outline: none; }
    pre { background: var(--bg-main); padding: 14px; border-radius: 6px; font-size: 12px; color: #cbd5e1; overflow-x: auto; border: 1px solid var(--border-color); margin-top: 8px; }
    """

    # Escape inputs for HTML safety
    q_safe = html.escape(query)
    from_safe = html.escape(date_from or "")
    to_safe = html.escape(date_to or "")
    sel_safe = html.escape(selected_tile_id or "")

    # Header HTML
    header_html = """
    <header>
        <div class="container header-content">
            <div>
                <div class="brand-title">
                    <span class="logo">&#9672;</span> FLUX
                </div>
                <div class="brand-subtitle">Semantic Retrieval &amp; Multi-Temporal Change Analysis</div>
            </div>
            <div class="badges">
                <span class="badge badge-sih">SIH2026227</span>
                <span class="badge badge-offline">&#9679; LOCAL / OFFLINE</span>
                <span class="badge badge-m">M5 &rarr; M9 Frozen</span>
            </div>
        </div>
    </header>
    """

    # Query Panel HTML
    opt_sel = "selected" if modality.lower() == "optical" else ""
    sar_sel = "selected" if modality.lower() == "sar" else ""
    all_sel = "selected" if modality.lower() == "all" else ""

    k3_sel = "selected" if top_k == 3 else ""
    k5_sel = "selected" if top_k == 5 else ""
    k10_sel = "selected" if top_k == 10 else ""

    query_panel_html = f"""
    <div class="card">
        <div class="card-title"><span class="icon">&#9881;</span> Natural Language Query &amp; Filter Panel</div>
        <form method="GET" action="/">
            <input type="hidden" name="selected_tile_id" value="" />
            <div class="form-grid">
                <div class="form-group">
                    <label for="query">Natural-Language Query</label>
                    <input type="text" id="query" name="query" class="form-control" value="{q_safe}" placeholder="e.g. urban development around Navi Mumbai" required />
                </div>
                <div class="form-group">
                    <label for="modality">Modality</label>
                    <select id="modality" name="modality" class="form-control">
                        <option value="optical" {opt_sel}>Optical (Sentinel-2)</option>
                        <option value="sar" {sar_sel}>SAR (Sentinel-1)</option>
                        <option value="all" {all_sel}>All Modalities</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="top_k">Top-K</label>
                    <select id="top_k" name="top_k" class="form-control">
                        <option value="3" {k3_sel}>3</option>
                        <option value="5" {k5_sel}>5</option>
                        <option value="10" {k10_sel}>10</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="date_from">Date From (UTC)</label>
                    <input type="date" id="date_from" name="date_from" class="form-control" value="{from_safe}" />
                </div>
                <div class="form-group">
                    <button type="submit" class="btn-submit">Run Analysis</button>
                </div>
            </div>
        </form>
        <div class="chips">
            <span class="chips-label">Sample Queries:</span>
            <a class="chip" href="/?query=urban+development+around+Navi+Mumbai&amp;modality=optical">urban development around Navi Mumbai</a>
            <a class="chip" href="/?query=built-up+area+around+Navi+Mumbai&amp;modality=optical">built-up area around Navi Mumbai</a>
            <a class="chip" href="/?query=vegetation+change+around+Navi+Mumbai&amp;modality=optical">vegetation change around Navi Mumbai</a>
            <a class="chip" href="/?query=coastal+water+radar+reflectance&amp;modality=sar">coastal water radar reflectance (SAR)</a>
        </div>
    </div>
    """

    # Error banner if any
    alert_html = ""
    if error_message:
        alert_html = f'<div class="alert alert-error">{html.escape(error_message)}</div>'

    # Results content
    results_html = ""
    if result:
        # Check warnings
        if result.warnings:
            w_text = "<br>".join([f"&bull; {html.escape(w)}" for w in result.warnings])
            alert_html += f'<div class="alert alert-warning"><strong>Pipeline Diagnostics:</strong><br>{w_text}</div>'

        # 1. Pipeline Execution Timings Bar
        if result.timing:
            t = result.timing
            results_html += f"""
            <div class="timing-grid">
                <div class="timing-card">
                    <div class="timing-label">M5 Retrieval</div>
                    <div class="timing-value">{t.retrieval_s:.3f} s</div>
                </div>
                <div class="timing-card">
                    <div class="timing-label">M6 Filtering</div>
                    <div class="timing-value">{t.filtering_s:.3f} s</div>
                </div>
                <div class="timing-card">
                    <div class="timing-label">M7 Pairing</div>
                    <div class="timing-value">{t.pairing_s:.3f} s</div>
                </div>
                <div class="timing-card">
                    <div class="timing-label">M8 Change Det.</div>
                    <div class="timing-value">{t.change_detection_s:.3f} s</div>
                </div>
                <div class="timing-card">
                    <div class="timing-label">M9 Suppression</div>
                    <div class="timing-value">{t.suppression_s:.3f} s</div>
                </div>
                <div class="timing-card timing-total">
                    <div class="timing-label">Total E2E Pipeline</div>
                    <div class="timing-value">{t.total_s:.3f} s</div>
                </div>
            </div>
            """

        # 2. Semantic retrieval Top-K Navigator, Visual Gallery, and Table
        if result.filtered_hits or result.retrieval_hits:
            table_hits = result.filtered_hits if result.filtered_hits else result.retrieval_hits
            num_hits = min(len(table_hits), top_k)

            # Identify active selection index
            active_idx = 0
            for i, h in enumerate(table_hits[:top_k]):
                if h.get("tile_id") == result.selected_tile_id:
                    active_idx = i
                    break
            active_rank = active_idx + 1
            active_hit = table_hits[active_idx]
            active_tile_id = active_hit.get("tile_id", "")
            active_score = float(active_hit.get("similarity_score", active_hit.get("score", 0.0)))
            active_tile_name = html.escape(active_tile_id)

            # Previous / Next navigation URLs
            prev_idx = max(0, active_idx - 1)
            next_idx = min(num_hits - 1, active_idx + 1)
            prev_tile_id = table_hits[prev_idx].get("tile_id", "")
            next_tile_id = table_hits[next_idx].get("tile_id", "")

            def make_tile_url(t_id: str) -> str:
                p = {
                    "query": query,
                    "modality": modality,
                    "top_k": str(top_k),
                    "date_from": date_from,
                    "date_to": date_to,
                    "selected_tile_id": t_id,
                }
                return f"/?{urllib.parse.urlencode({k: v for k, v in p.items() if v})}"

            prev_url = make_tile_url(prev_tile_id)
            next_url = make_tile_url(next_tile_id)
            prev_cls = "nav-btn" if active_idx > 0 else "nav-btn disabled"
            next_cls = "nav-btn" if active_idx < num_hits - 1 else "nav-btn disabled"

            # Rank buttons [Rank 1] [Rank 2] ... [Rank K]
            rank_buttons_html = ""
            for i, h in enumerate(table_hits[:top_k]):
                r_num = i + 1
                t_id = h.get("tile_id", "")
                r_url = make_tile_url(t_id)
                r_cls = "rank-btn active" if i == active_idx else "rank-btn"
                rank_buttons_html += f'<a href="{r_url}" class="{r_cls}">Rank {r_num}</a>'

            # Build Gallery Cards and Table Rows
            gallery_cards_html = ""
            table_rows = []
            for i, h in enumerate(table_hits[:top_k]):
                r_num = i + 1
                t_id = h.get("tile_id", "")
                is_sel = (i == active_idx)
                score = float(h.get("similarity_score", h.get("score", 0.0)))
                t_url = make_tile_url(t_id)
                t_modality = h.get("modality", "optical")
                acq_dt = h.get("acquisition_datetime_utc", "")[:19].replace("T", " ") if h.get("acquisition_datetime_utc") else "N/A"
                scene_display = html.escape(h.get("scene_id", "")[:20])

                # Thumbnail image
                thumb_b64 = render_tile_image(t_id, t_modality, project_root=root)
                if thumb_b64:
                    thumb_img = f'<img src="{thumb_b64}" alt="Tile thumbnail" />'
                else:
                    thumb_img = '<div style="width:100%; max-width:180px; height:120px; display:flex; align-items:center; justify-content:center; margin:0 auto 8px auto; background:#11141c; border:1px dashed var(--border-color); border-radius:4px; font-size:10px; color:var(--text-muted);">No Thumbnail</div>'

                card_sel_cls = "topk-card selected" if is_sel else "topk-card"
                card_badge = '<span class="badge badge-sih" style="padding:2px 6px; font-size:9px;">Active</span>' if is_sel else f'<span class="badge badge-m" style="padding:2px 6px; font-size:9px;">Rank #{r_num}</span>'
                btn_action = (
                    '<span class="badge badge-offline" style="padding:5px 10px; font-size:11px; width:100%;">Selected Target</span>'
                    if is_sel
                    else f'<a href="{t_url}" class="btn-select" style="text-align:center; display:block; padding:5px 10px; font-size:11px;">Select Tile &rarr;</a>'
                )

                gallery_cards_html += f"""
                <div class="{card_sel_cls}">
                    <div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span style="font-size:12px; font-weight:800; color:var(--text-main);">#{r_num}</span>
                            {card_badge}
                        </div>
                        {thumb_img}
                        <div style="font-size:11px; font-weight:700; color:var(--text-main); font-family:monospace; word-break:break-all; margin-bottom:4px;">
                            {html.escape(t_id[-16:])}
                        </div>
                        <div style="font-size:10px; color:var(--text-muted); margin-bottom:6px;">{acq_dt} UTC</div>
                        <div style="margin-bottom:8px;">
                            {render_similarity_gauge(score)}
                        </div>
                    </div>
                    <div>
                        {btn_action}
                    </div>
                </div>
                """

                # Table row
                row_cls = ' class="selected"' if is_sel else ""
                table_action_col = (
                    '<span class="badge badge-offline" style="padding: 4px 8px;">Selected</span>'
                    if is_sel
                    else f'<a href="{t_url}" class="btn-select">Select Tile</a>'
                )
                table_rows.append(
                    f"<tr{row_cls}>"
                    f"<td>{r_num}{' <span class=\"badge badge-sih\" style=\"padding:2px 6px; font-size:9px;\">Active</span>' if is_sel else ''}</td>"
                    f"<td><strong>{score:.4f}</strong></td>"
                    f"<td>{html.escape(t_modality.upper())}</td>"
                    f"<td><code>{html.escape(t_id)}</code></td>"
                    f"<td>{acq_dt}</td>"
                    f"<td><code>{scene_display}..</code></td>"
                    f"<td>{table_action_col}</td>"
                    f"</tr>"
                )

            # Active selection callout banner & Top-K Navigator Section HTML
            results_html += f"""
            <div class="card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
                    <div class="card-title" style="margin-bottom:0;">
                        <span class="icon">&#9670;</span> Top-K Semantic Retrieval Navigator (M5 Discovery &rarr; M6 Filtering)
                    </div>
                    <span class="badge badge-sih">Rank {active_rank} of {num_hits} selected</span>
                </div>

                <!-- Navigator Controls Bar -->
                <div class="navigator-bar">
                    <div class="navigator-controls">
                        <a href="{prev_url}" class="{prev_cls}">&#9664; Previous Tile</a>
                        {rank_buttons_html}
                        <a href="{next_url}" class="{next_cls}">Next Tile &#9654;</a>
                    </div>
                    <div style="display:flex; align-items:center; gap:12px;">
                        <div style="font-size:13px; color:var(--text-main);">
                            <strong>Selected Active Tile:</strong> <code>{active_tile_name}</code>
                        </div>
                        <span class="badge badge-offline">Target for M7-M9</span>
                    </div>
                </div>

                <!-- Active Similarity Gauge Banner -->
                <div style="margin-bottom:18px;">
                    {render_similarity_gauge(active_score)}
                </div>

                <!-- Top-K Visual Gallery -->
                <div class="topk-gallery">
                    {gallery_cards_html}
                </div>

                <!-- Tabular Comparison List -->
                <details>
                    <summary style="font-size:13px; font-weight:700; color:var(--text-muted); cursor:pointer;">
                        View Tabular Top-K Comparison List ({num_hits} tiles)
                    </summary>
                    <div class="table-container" style="margin-top:10px;">
                        <table>
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>Similarity Score</th>
                                    <th>Modality</th>
                                    <th>Tile ID</th>
                                    <th>Acquisition UTC</th>
                                    <th>Scene</th>
                                    <th>Action</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join(table_rows)}
                            </tbody>
                        </table>
                    </div>
                </details>
            </div>
            """

        # 3. Spatial Tile Grid (Actual relative grid placement and 3x3 local neighborhood)
        if result.selected_tile_id:
            results_html += render_spatial_tile_grid_html(
                result.selected_tile_id,
                query_params={
                    "query": query,
                    "modality": modality,
                    "top_k": str(top_k),
                    "date_from": date_from,
                    "date_to": date_to,
                },
                project_root=root,
            )

        # 4. Exact Geographic Location & Spatial Footprint (for selected tile)
        if result.selected_tile_id:
            active_id = result.selected_tile_id
            active_hit = result.selected_hit
            authoritative_meta = load_tile_authoritative_metadata(active_id, root)
            geo_info = extract_geographic_info(authoritative_meta or active_hit)
            tile_disp_meta = extract_tile_display_metadata(authoritative_meta, active_hit)

            crs_disp = html.escape(str(geo_info["crs"]))
            bounds_wgs = geo_info["bounds_wgs84"]
            centroid = geo_info["centroid_wgs84"]
            corners = geo_info["corners_wgs84"]
            bounds_proj = geo_info["bounds_projected"]

            centroid_str = f"{centroid['lon']:.6f}&deg; E, {centroid['lat']:.6f}&deg; N" if centroid else "Not available"
            min_lon_str = f"{bounds_wgs[0]:.6f}&deg; E" if bounds_wgs else "Not available"
            min_lat_str = f"{bounds_wgs[1]:.6f}&deg; N" if bounds_wgs else "Not available"
            max_lon_str = f"{bounds_wgs[2]:.6f}&deg; E" if bounds_wgs else "Not available"
            max_lat_str = f"{bounds_wgs[3]:.6f}&deg; N" if bounds_wgs else "Not available"

            if corners:
                nw_str = f"({corners['NW']['lon']:.5f}&deg; E, {corners['NW']['lat']:.5f}&deg; N)"
                ne_str = f"({corners['NE']['lon']:.5f}&deg; E, {corners['NE']['lat']:.5f}&deg; N)"
                sw_str = f"({corners['SW']['lon']:.5f}&deg; E, {corners['SW']['lat']:.5f}&deg; N)"
                se_str = f"({corners['SE']['lon']:.5f}&deg; E, {corners['SE']['lat']:.5f}&deg; N)"
            else:
                nw_str = ne_str = sw_str = se_str = "Not available"

            if bounds_proj and len(bounds_proj) == 4:
                projected_html = f"""
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px;">
                    <div><span style="color:var(--text-muted);">Projected CRS:</span> <code>{html.escape(str(geo_info.get('projected_crs', 'EPSG:32643')))}</code></div>
                    <div><span style="color:var(--text-muted);">Unit:</span> Metres (Cartesian Easting/Northing)</div>
                    <div><span style="color:var(--text-muted);">Min X (Easting):</span> <code>{bounds_proj[0]:.1f} m</code></div>
                    <div><span style="color:var(--text-muted);">Min Y (Northing):</span> <code>{bounds_proj[1]:.1f} m</code></div>
                    <div><span style="color:var(--text-muted);">Max X (Easting):</span> <code>{bounds_proj[2]:.1f} m</code></div>
                    <div><span style="color:var(--text-muted);">Max Y (Northing):</span> <code>{bounds_proj[3]:.1f} m</code></div>
                </div>
                """
            else:
                projected_html = '<div style="color:var(--text-muted); font-size:12px;">Not available (Unprojected WGS84 native coordinate reference system).</div>'

            svg_diagram = render_bounding_box_diagram(bounds_wgs, geo_info["crs"], active_id)

            results_html += f"""
            <div class="card">
                <div class="card-title"><span class="icon">&#127758;</span> Exact Geographic Location &amp; Spatial Footprint</div>
                <div class="grid-2">
                    <div>
                        <div style="background:var(--bg-main); border:1px solid var(--border-color); border-radius:8px; padding:16px; margin-bottom:14px;">
                            <div style="font-size:12px; font-weight:700; color:var(--accent-cyan); text-transform:uppercase; margin-bottom:8px; letter-spacing:0.5px;">
                                WGS84 Geographic Coordinates (Degrees &deg;)
                            </div>
                            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; font-size:13px;">
                                <div><span style="color:var(--text-muted);">CRS:</span> <code>{crs_disp}</code></div>
                                <div><span style="color:var(--text-muted);">Centroid:</span> <code>{centroid_str}</code></div>
                                <div><span style="color:var(--text-muted);">Min Longitude:</span> <code>{min_lon_str}</code></div>
                                <div><span style="color:var(--text-muted);">Min Latitude:</span> <code>{min_lat_str}</code></div>
                                <div><span style="color:var(--text-muted);">Max Longitude:</span> <code>{max_lon_str}</code></div>
                                <div><span style="color:var(--text-muted);">Max Latitude:</span> <code>{max_lat_str}</code></div>
                            </div>
                            <div style="margin-top:14px; font-size:12px; border-top:1px solid var(--border-color); padding-top:10px;">
                                <span style="color:var(--text-muted); font-weight:600;">Corner Coordinates (WGS84):</span>
                                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:6px; margin-top:6px; font-family:monospace; font-size:11px;">
                                    <div><strong>NW:</strong> {nw_str}</div>
                                    <div><strong>NE:</strong> {ne_str}</div>
                                    <div><strong>SW:</strong> {sw_str}</div>
                                    <div><strong>SE:</strong> {se_str}</div>
                                </div>
                            </div>
                        </div>
                        <div style="background:var(--bg-main); border:1px solid var(--border-color); border-radius:8px; padding:16px;">
                            <div style="font-size:12px; font-weight:700; color:var(--accent-emerald); text-transform:uppercase; margin-bottom:8px; letter-spacing:0.5px;">
                                Projected Coordinates (Metres X/Y)
                            </div>
                            {projected_html}
                            <div style="margin-top:10px; font-size:11px; color:var(--text-muted); line-height:1.4;">
                                <em>Distinction: WGS84 coordinates represent angular geographic latitude/longitude on the WGS84 ellipsoid. Projected coordinates represent Cartesian planar metres on the conformal UTM grid.</em>
                            </div>
                        </div>
                    </div>
                    <div>
                        {svg_diagram}
                    </div>
                </div>
            </div>
            """

            # 5. Expandable Tile Metadata Panel (with cloud cover gauge)
            cloud_val = None
            try:
                if authoritative_meta and "cloud_cover_percentage" in authoritative_meta:
                    cloud_val = float(authoritative_meta["cloud_cover_percentage"])
                elif active_hit and "cloud_cover_percentage" in active_hit:
                    cloud_val = float(active_hit["cloud_cover_percentage"])
            except Exception:
                pass

            cloud_gauge_html = render_cloud_cover_gauge(cloud_val) if cloud_val is not None else ""

            results_html += f"""
            <div class="card">
                <details>
                    <summary style="font-size:15px; font-weight:700; color:var(--text-main);"><span class="icon">&#128196;</span> Authoritative Tile Metadata (M2 / M4 / M6 Contract)</summary>
                    <div style="margin-top:16px;">
                        {cloud_gauge_html}
                        <table class="metadata-table">
                            <tbody>
                                <tr><th>Tile ID</th><td><code>{html.escape(tile_disp_meta['tile_id'])}</code></td></tr>
                                <tr><th>Source Scene ID</th><td><code>{html.escape(tile_disp_meta['scene_id'])}</code></td></tr>
                                <tr><th>Modality / Sensor</th><td>{html.escape(tile_disp_meta['modality'])} &bull; {html.escape(tile_disp_meta['platform'])} ({html.escape(tile_disp_meta['sensor'])})</td></tr>
                                <tr><th>Processing Level</th><td>{html.escape(tile_disp_meta['processing_level'])}</td></tr>
                                <tr><th>Acquisition Timestamp (UTC)</th><td>{html.escape(tile_disp_meta['acquisition_datetime_utc'])}</td></tr>
                                <tr><th>Grid Index</th><td>{html.escape(tile_disp_meta['grid_index'])}</td></tr>
                                <tr><th>Tile Dimensions</th><td>{tile_disp_meta['dimensions']}</td></tr>
                                <tr><th>Tile Filesystem Path</th><td><code>{html.escape(tile_disp_meta['tile_directory'])}</code></td></tr>
                                <tr><th>Available Bands</th><td>{html.escape(tile_disp_meta['bands_available'])}</td></tr>
                                <tr><th>Quality Diagnostics</th><td>{html.escape(tile_disp_meta['quality_diagnostic'])}</td></tr>
                            </tbody>
                        </table>
                    </div>
                </details>
            </div>
            """

        # 6. Multi-Temporal Change Analysis Evidence Panel (M7 -> M8 -> M9)
        results_html += render_change_evidence_panel(result, project_root=root)

        # 7. Machine-Readable Provenance expander
        prov_dict = result.to_dict()
        prov_json = html.escape(json.dumps(prov_dict, indent=2))
        results_html += f"""
        <div class="card">
            <details>
                <summary>Inspect Complete Machine-Readable Provenance &amp; Contract (JSON)</summary>
                <pre><code>{prov_json}</code></pre>
            </details>
        </div>
        """

    # 8. M1-M9 Processing Pipeline Architecture Flowchart (prominently displayed)
    flowchart_html = generate_m1_m9_flowchart_html()

    # 9. FLUX Milestone Architecture (Roadmap & Status)
    milestone_arch_html = generate_milestone_architecture_html()

    # 10. FLUX Full Technical & Feasibility Report (Integrated)
    technical_report_html = generate_technical_report_html()

    return f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>FLUX - Satellite Semantic Retrieval &amp; Change Analysis</title>
        <style>{css}</style>
    </head>
    <body>
        {header_html}
        <div class="container">
            {alert_html}
            {query_panel_html}
            {results_html}
            {flowchart_html}
            {milestone_arch_html}
            {technical_report_html}
        </div>
    </body>
    </html>
    """


class FLUXRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler serving the local FLUX Presentation UI and Report downloads."""

    pipeline: Optional[PoCPipeline] = None
    project_root: Optional[Path] = None

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress standard log spam to stdout."""
        pass

    def do_GET(self) -> None:
        """Handle GET requests for the dashboard UI and report downloads."""
        parsed_url = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed_url.query)

        # 1. Handle Technical Report Downloads
        if parsed_url.path == "/download-report":
            fmt = params.get("format", ["markdown"])[0].lower().strip()
            if fmt in ("html", "htm"):
                md = generate_technical_report_markdown()
                body = generate_technical_report_html(md)
                page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>FLUX Technical &amp; Feasibility Report - SIH2026227</title>
<style>
body {{ background:#0c0f17; color:#f1f5f9; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; line-height:1.6; padding:40px; }}
.card {{ background:#151a24; border:1px solid #242c3d; border-radius:12px; padding:28px; max-width:1100px; margin:0 auto; }}
table {{ width:100%; border-collapse:collapse; margin:16px 0; font-size:13px; }}
th, td {{ border:1px solid #242c3d; padding:10px 14px; text-align:left; }}
th {{ background:#0c0f17; color:#94a3b8; font-weight:600; }}
code {{ font-family:monospace; color:#38bdf8; }}
pre {{ background:#0c0f17; border:1px solid #242c3d; border-radius:6px; padding:16px; overflow-x:auto; }}
</style>
</head>
<body>
{body}
</body>
</html>"""
                body_bytes = page.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="FLUX_Technical_Report.html"')
                self.send_header("Content-Length", str(len(body_bytes)))
                self.end_headers()
                self.wfile.write(body_bytes)
                return
            else:
                # Default: Markdown
                md = generate_technical_report_markdown()
                md_bytes = md.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/markdown; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="FLUX_Technical_Report.md"')
                self.send_header("Content-Length", str(len(md_bytes)))
                self.end_headers()
                self.wfile.write(md_bytes)
                return

        # 2. Handle Interactive Dashboard Requests
        query = params.get("query", [DEFAULT_QUERY])[0].strip()
        modality = params.get("modality", ["optical"])[0].strip()
        top_k_str = params.get("top_k", ["5"])[0].strip()
        date_from = params.get("date_from", [""])[0].strip()
        date_to = params.get("date_to", [""])[0].strip()
        selected_tile_id = params.get("selected_tile_id", [""])[0].strip()

        try:
            top_k = int(top_k_str)
        except ValueError:
            top_k = 5

        # Execute pipeline if query is provided
        result: Optional[PoCPipelineResult] = None
        error_msg: Optional[str] = None

        if query:
            try:
                modality_filter = modality if modality.lower() in ("optical", "sar") else None
                result = self.pipeline.run(
                    query=query,
                    top_k=top_k,
                    modality=modality_filter,
                    date_from=date_from if date_from else None,
                    date_to=date_to if date_to else None,
                    selected_tile_id=selected_tile_id if selected_tile_id else None,
                )
            except Exception as exc:
                error_msg = f"Pipeline execution failed: {exc}"

        # Generate HTML
        html_content = build_html_page(
            query=query,
            top_k=top_k,
            modality=modality,
            date_from=date_from,
            date_to=date_to,
            selected_tile_id=selected_tile_id,
            result=result,
            error_message=error_msg,
            project_root=self.project_root,
        )

        body_bytes = html_content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()
        self.wfile.write(body_bytes)


def create_ui_server(
    host: str = "127.0.0.1",
    port: int = 8501,
    project_root: Optional[Path] = None,
    pipeline: Optional[PoCPipeline] = None,
) -> ThreadingHTTPServer:
    """Create and configure the local ThreadingHTTPServer instance."""
    root = project_root if project_root else ROOT_DIR
    pipe = pipeline if pipeline is not None else PoCPipeline(project_root=root)

    class CustomHandler(FLUXRequestHandler):
        pass

    CustomHandler.pipeline = pipe
    CustomHandler.project_root = root

    server = ThreadingHTTPServer((host, port), CustomHandler)
    return server


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the FLUX Presentation UI."""
    parser = argparse.ArgumentParser(
        description="Launch local FLUX presentation UI for SIH multi-temporal change analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8501, help="Port to listen on (default: 8501).")
    parser.add_argument("--project-root", type=str, default=None, help="Custom project root directory.")

    return parser.parse_args()


def main() -> int:
    """Main CLI entrypoint."""
    args = parse_args()
    root = Path(args.project_root).resolve() if args.project_root else ROOT_DIR

    print("\n" + "=" * 70)
    print("  FLUX — Semantic Retrieval & Multi-Temporal Change Analysis")
    print("  SIH Problem Statement: SIH2026227")
    print("  Status: LOCAL / OFFLINE (M5 -> M9 Complete)")
    print("=" * 70)
    print("Initializing PoCPipeline (loading CLIP model & FAISS index)...")

    pipeline = PoCPipeline(project_root=root)
    server = create_ui_server(
        host=args.host,
        port=args.port,
        project_root=root,
        pipeline=pipeline,
    )

    url = f"http://{args.host}:{args.port}"
    print(f"\n[OK] Presentation UI is running at: {url}")
    print("Press Ctrl+C to terminate the local server.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down FLUX presentation UI...")
        server.server_close()
        return 0


if __name__ == "__main__":
    sys.exit(main())
