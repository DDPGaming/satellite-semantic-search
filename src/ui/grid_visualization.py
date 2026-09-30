"""Spatial tile grid visualization derived from authoritative dataset metadata.

Computes and renders the true relative geographic grid placement for any selected satellite
tile, extracting neighboring tiles from actual indexed scene metadata without fabricating
non-existent neighbors or using external mapping services.
"""

import html
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.parse


def get_default_project_root() -> Path:
    """Return repository root path."""
    return Path(__file__).resolve().parent.parent.parent


def parse_tile_row_col(tile_id: str) -> Optional[Tuple[int, int]]:
    """
    Extract (row, col) indices from a standard tile ID string.

    Example: 's2_S2B_43QBB_20240112_0_L2A_10m_r01_c04' -> (1, 4)
    """
    match = re.search(r"_r(\d+)_c(\d+)", tile_id)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def load_scene_tile_matrix(
    scene_id: str,
    project_root: Optional[Path] = None,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Load all actual indexed tiles for a given scene from data/index/metadata.json.

    Returns:
        Dictionary mapping (row, col) integer tuples to authoritative tile metadata entries.
    """
    root = project_root if project_root else get_default_project_root()
    index_meta_path = root / "data" / "index" / "metadata.json"
    if not index_meta_path.exists():
        return {}

    try:
        data = json.loads(index_meta_path.read_text(encoding="utf-8"))
        entries = data.get("entries", []) if isinstance(data, dict) else []
    except Exception:
        return {}

    scene_matrix: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for e in entries:
        if e.get("scene_id") == scene_id:
            tile_id = e.get("tile_id", "")
            coords = parse_tile_row_col(tile_id)
            if coords:
                scene_matrix[coords] = e

    return scene_matrix


def get_tile_neighborhood_data(
    selected_tile_id: str,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Derive the 3x3 local neighborhood and scene bounds for a selected tile.

    Returns:
        Structured dictionary containing:
        - selected_tile_id
        - selected_coords (row, col)
        - scene_id
        - crs
        - neighborhood_grid: 3x3 list of rows, each containing cell dictionaries
        - scene_extent: (min_row, max_row, min_col, max_col)
        - total_scene_tiles: int
    """
    root = project_root if project_root else get_default_project_root()
    index_meta_path = root / "data" / "index" / "metadata.json"

    # 1. Locate selected tile entry
    selected_entry: Optional[Dict[str, Any]] = None
    if index_meta_path.exists():
        try:
            data = json.loads(index_meta_path.read_text(encoding="utf-8"))
            for e in data.get("entries", []):
                if e.get("tile_id") == selected_tile_id:
                    selected_entry = e
                    break
        except Exception:
            pass

    coords = parse_tile_row_col(selected_tile_id)
    if not coords:
        return {
            "selected_tile_id": selected_tile_id,
            "error": f"Cannot parse grid row/column indices from tile ID '{selected_tile_id}'.",
        }

    sel_row, sel_col = coords
    scene_id = selected_entry.get("scene_id", "") if selected_entry else ""
    crs = selected_entry.get("crs", "EPSG:32643") if selected_entry else "EPSG:32643"

    # If scene_id not in entry, attempt to parse from tile_id
    if not scene_id and "s2_" in selected_tile_id:
        # e.g. s2_S2B_43QBB_20240112_0_L2A_10m_r01_c04 -> S2B_43QBB_20240112_0_L2A
        parts = selected_tile_id.split("_10m_")
        if len(parts) == 2:
            scene_id = parts[0].replace("s2_", "")

    scene_matrix = load_scene_tile_matrix(scene_id, root) if scene_id else {}

    # Determine scene row/col extents from available tiles
    if scene_matrix:
        rows = [r for r, _ in scene_matrix.keys()]
        cols = [c for _, c in scene_matrix.keys()]
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
    else:
        min_row, max_row = 0, 6
        min_col, max_col = 0, 6

    # 2. Build 3x3 local neighborhood array
    grid_rows: List[List[Dict[str, Any]]] = []
    for r in [sel_row - 1, sel_row, sel_row + 1]:
        row_cells: List[Dict[str, Any]] = []
        for c in [sel_col - 1, sel_col, sel_col + 1]:
            is_center = (r == sel_row and c == sel_col)
            tile_data = scene_matrix.get((r, c))

            if tile_data is not None:
                # Real neighbor tile exists in dataset
                b_wgs = tile_data.get("bounds_wgs84", [])
                centroid = (
                    f"{(b_wgs[0] + b_wgs[2])/2:.4f}°E, {(b_wgs[1] + b_wgs[3])/2:.4f}°N"
                    if len(b_wgs) == 4
                    else "N/A"
                )
                row_cells.append({
                    "row": r,
                    "col": c,
                    "status": "selected" if is_center else "available",
                    "tile_id": tile_data.get("tile_id", ""),
                    "bounds_wgs84": b_wgs,
                    "centroid_str": centroid,
                    "is_center": is_center,
                })
            elif r < min_row or r > max_row or c < min_col or c > max_col:
                # Outside scene physical boundaries
                row_cells.append({
                    "row": r,
                    "col": c,
                    "status": "out_of_bounds",
                    "label": "Out of Scene Extent",
                    "is_center": False,
                })
            else:
                # In scene extent but not present in dataset
                row_cells.append({
                    "row": r,
                    "col": c,
                    "status": "missing",
                    "label": "Tile Not In Archive",
                    "is_center": False,
                })
        grid_rows.append(row_cells)

    return {
        "selected_tile_id": selected_tile_id,
        "selected_row": sel_row,
        "selected_col": sel_col,
        "scene_id": scene_id,
        "crs": crs,
        "selected_entry": selected_entry,
        "neighborhood_grid": grid_rows,
        "scene_matrix": scene_matrix,
        "min_row": min_row,
        "max_row": max_row,
        "min_col": min_col,
        "max_col": max_col,
        "total_scene_tiles": len(scene_matrix),
    }


def render_spatial_tile_grid_html(
    selected_tile_id: str,
    query_params: Optional[Dict[str, str]] = None,
    project_root: Optional[Path] = None,
) -> str:
    """
    Render the complete Spatial Tile Grid section as self-contained HTML.

    Includes:
    - Orientation arrow (North ↑)
    - 3x3 local neighborhood with highlighted active tile, row/column indices, and neighbor links
    - Selected tile geographic footprint summary (WGS84 bounds, centroid, CRS, row, col)
    - Full-scene mini-map matrix showing exact position in the parent scene
    - Zero fabricated coordinates or tiles
    """
    if isinstance(query_params, Path):
        project_root = query_params
        query_params = None

    data = get_tile_neighborhood_data(selected_tile_id, project_root)
    if "error" in data:

        return f'<div class="alert alert-warning">{html.escape(data["error"])}</div>'

    sel_row = data["selected_row"]
    sel_col = data["selected_col"]
    scene_id = html.escape(data["scene_id"] or "Unknown Scene")
    crs = html.escape(data["crs"])
    total_tiles = data["total_scene_tiles"]
    grid = data["neighborhood_grid"]
    params = query_params or {}

    # Extract bounds and centroid for selected tile
    sel_entry = data.get("selected_entry") or {}
    bounds = sel_entry.get("bounds_wgs84", [])
    if len(bounds) == 4:
        min_lon, min_lat, max_lon, max_lat = bounds
        centroid_str = f"{(min_lon + max_lon)/2:.6f}° E, {(min_lat + max_lat)/2:.6f}° N"
        bbox_str = f"[{min_lon:.5f}° E, {min_lat:.5f}° N, {max_lon:.5f}° E, {max_lat:.5f}° N]"
    else:
        centroid_str = "Not available"
        bbox_str = "Not available"

    # Build 3x3 HTML grid cells
    cells_html = ""
    for r_idx, row in enumerate(grid):
        for c_idx, cell in enumerate(row):
            st = cell["status"]
            r = cell["row"]
            c = cell["col"]

            if st == "selected":
                # Active selected tile
                cells_html += f"""
                <div style="background:rgba(56, 189, 248, 0.15); border:2px solid var(--accent-cyan); border-radius:8px; padding:12px; text-align:center; box-shadow:0 0 12px rgba(56,189,248,0.25);">
                    <div style="font-size:10px; font-weight:800; color:var(--accent-cyan); text-transform:uppercase; letter-spacing:0.5px;">&#9670; ACTIVE SELECTED</div>
                    <div style="font-size:16px; font-weight:800; color:#ffffff; font-family:monospace; margin:4px 0;">r{r:02d} c{c:02d}</div>
                    <div style="font-size:10px; color:#cbd5e1; font-family:monospace; word-break:break-all; line-height:1.2;">{html.escape(cell['tile_id'][-15:])}</div>
                    <div style="font-size:10px; color:var(--accent-cyan); font-family:monospace; margin-top:4px;">{cell.get('centroid_str', '')}</div>
                </div>
                """
            elif st == "available":
                # Valid neighbor tile
                tile_id = cell["tile_id"]
                # Build switch URL
                nav_params = dict(params)
                nav_params["selected_tile_id"] = tile_id
                nav_url = f"/?{urllib.parse.urlencode({k: v for k, v in nav_params.items() if v})}"

                cells_html += f"""
                <a href="{nav_url}" style="text-decoration:none; display:block; background:var(--bg-main); border:1px solid var(--border-color); border-radius:8px; padding:12px; text-align:center; transition:all 0.2s ease;" onmouseover="this.style.borderColor='var(--accent-blue)';" onmouseout="this.style.borderColor='var(--border-color)';">
                    <div style="font-size:10px; font-weight:700; color:var(--accent-emerald); text-transform:uppercase;">Neighbor Tile</div>
                    <div style="font-size:15px; font-weight:700; color:var(--text-main); font-family:monospace; margin:4px 0;">r{r:02d} c{c:02d}</div>
                    <div style="font-size:10px; color:var(--text-muted); font-family:monospace; word-break:break-all; line-height:1.2;">{html.escape(tile_id[-15:])}</div>
                    <div style="font-size:10px; color:var(--text-muted); font-family:monospace; margin-top:4px;">{cell.get('centroid_str', '')}</div>
                    <div style="font-size:10px; color:var(--accent-blue); font-weight:600; margin-top:6px;">Switch &rarr;</div>
                </a>
                """
            else:
                # Out of bounds or missing
                label = html.escape(cell.get("label", "Unavailable"))
                cells_html += f"""
                <div style="background:#0a0c12; border:1px dashed #1e2638; border-radius:8px; padding:12px; text-align:center; opacity:0.4;">
                    <div style="font-size:10px; font-weight:600; color:var(--text-muted); text-transform:uppercase;">r{r:02d} c{c:02d}</div>
                    <div style="font-size:11px; color:var(--text-muted); margin-top:10px; font-style:italic;">{label}</div>
                </div>
                """

    # Build Scene Mini-Map (7x7 / 8x8 full scene grid)
    scene_matrix = data.get("scene_matrix", {})
    min_r = data.get("min_row", 0)
    max_r = data.get("max_row", 6)
    min_c = data.get("min_col", 0)
    max_c = data.get("max_col", 6)

    minimap_rows = ""
    for r in range(min_r, max_r + 1):
        row_cells = ""
        for c in range(min_c, max_c + 1):
            is_active = (r == sel_row and c == sel_col)
            is_neighbor = abs(r - sel_row) <= 1 and abs(c - sel_col) <= 1
            has_tile = (r, c) in scene_matrix

            if is_active:
                bg = "var(--accent-cyan)"
                border = "var(--accent-cyan)"
            elif is_neighbor and has_tile:
                bg = "rgba(16, 185, 129, 0.4)"
                border = "var(--accent-emerald)"
            elif has_tile:
                bg = "#1e2638"
                border = "#242c3d"
            else:
                bg = "#080a0f"
                border = "transparent"

            row_cells += f'<div style="width:16px; height:16px; background:{bg}; border:1px solid {border}; border-radius:2px;" title="r{r:02d} c{c:02d}"></div>'
        minimap_rows += f'<div style="display:flex; gap:3px;">{row_cells}</div>'

    html_out = f"""
    <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
            <div class="card-title" style="margin-bottom:0;">
                <span class="icon">&#9881;</span> Spatial Tile Grid: Local Neighborhood &amp; Relative Placement
            </div>
            <div>
                <span class="badge badge-offline">&#9679; Authoritative Scene Grid</span>
                <span class="badge badge-sih" style="margin-left:6px;">Scene: {scene_id}</span>
            </div>
        </div>

        <div class="grid-2" style="gap:24px; align-items:start;">
            <!-- Left: 3x3 Local Grid with Compass Orientation -->
            <div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <div style="font-size:12px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">
                        Local 3&times;3 Tile Neighborhood
                    </div>
                    <div style="display:flex; align-items:center; gap:6px; font-size:12px; font-weight:800; color:var(--accent-cyan);">
                        <span>&#8593; NORTH</span>
                    </div>
                </div>

                <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:10px; background:var(--bg-main); border:1px solid var(--border-color); border-radius:10px; padding:12px;">
                    {cells_html}
                </div>
                <div style="font-size:11px; color:var(--text-muted); margin-top:8px; text-align:center;">
                    <em>Relative positioning strictly follows scene grid row (North &rarr; South) and column (West &rarr; East) indices. Click any available neighbor to switch analysis.</em>
                </div>
            </div>

            <!-- Right: Exact Selected Tile Footprint & Scene Mini-Map -->
            <div>
                <div style="background:var(--bg-main); border:1px solid var(--border-color); border-radius:10px; padding:16px; margin-bottom:16px;">
                    <div style="font-size:12px; font-weight:700; color:var(--accent-cyan); text-transform:uppercase; margin-bottom:10px; letter-spacing:0.5px;">
                        Selected Tile Placement &amp; Footprint
                    </div>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; font-size:12px;">
                        <div><span style="color:var(--text-muted);">Tile Row:</span> <strong style="font-family:monospace; color:var(--text-main);">r{sel_row:02d}</strong></div>
                        <div><span style="color:var(--text-muted);">Tile Column:</span> <strong style="font-family:monospace; color:var(--text-main);">c{sel_col:02d}</strong></div>
                        <div><span style="color:var(--text-muted);">CRS:</span> <code>{crs}</code></div>
                        <div><span style="color:var(--text-muted);">Grid Resolution:</span> <code>10m &times; 10m (256&times;256)</code></div>
                        <div style="grid-column:span 2;"><span style="color:var(--text-muted);">Centroid (WGS84):</span> <code>{centroid_str}</code></div>
                        <div style="grid-column:span 2;"><span style="color:var(--text-muted);">Bounding Box:</span> <code>{bbox_str}</code></div>
                    </div>
                </div>

                <!-- Scene Mini-Map -->
                <div style="background:var(--bg-main); border:1px solid var(--border-color); border-radius:10px; padding:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <div style="font-size:12px; font-weight:700; color:var(--accent-emerald); text-transform:uppercase; letter-spacing:0.5px;">
                            Full Scene Tile Index ({total_tiles} Tiles)
                        </div>
                        <div style="display:flex; gap:10px; font-size:10px; color:var(--text-muted);">
                            <span><span style="color:var(--accent-cyan); font-weight:800;">&#9632;</span> Active</span>
                            <span><span style="color:var(--accent-emerald); font-weight:800;">&#9632;</span> 3&times;3 Scope</span>
                            <span><span style="color:#1e2638; font-weight:800;">&#9632;</span> Scene</span>
                        </div>
                    </div>
                    <div style="display:inline-flex; flex-direction:column; gap:3px; padding:8px; background:#0c0f17; border:1px solid var(--border-color); border-radius:6px;">
                        {minimap_rows}
                    </div>
                    <div style="font-size:11px; color:var(--text-muted); margin-top:8px;">
                        Shows selected tile placement inside parent acquisition scene (100km &times; 100km UTM tile).
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    return html_out
