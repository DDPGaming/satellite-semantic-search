"""Tile metadata inspection, geographic coordinate parsing, and offline visualization.

Extracts authoritative metadata from M2/M4/M6 tile metadata contracts and renders
clean offline SVG bounding-box location diagrams without remote map dependencies.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.ui.rendering import _resolve_tile_dir, get_default_project_root


def load_tile_authoritative_metadata(
    tile_dir_or_id: Union[str, Path],
    project_root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load authoritative tile metadata JSON produced during ingestion/tiling (M2/M6).

    Args:
        tile_dir_or_id: Tile directory path or tile ID.
        project_root: Optional project repository root.

    Returns:
        Loaded metadata dictionary, or None if unavailable.
    """
    root = project_root if project_root else get_default_project_root()
    tile_dir = _resolve_tile_dir(tile_dir_or_id, root)

    # 1. Check for tile-local metadata.json inside the tile directory
    if tile_dir and tile_dir.is_dir():
        local_meta_path = tile_dir / "metadata.json"
        if local_meta_path.exists():
            try:
                with open(local_meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

    # 2. Fallback to vector index metadata.json
    index_meta_path = root / "data" / "index" / "metadata.json"
    if index_meta_path.exists():
        try:
            with open(index_meta_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            entries = index_data.get("entries", [])
            target_id = Path(tile_dir_or_id).name
            for entry in entries:
                if entry.get("tile_id") == target_id:
                    return entry
        except Exception:
            pass

    return None


def extract_geographic_info(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract exact geographic and projected coordinate information from tile metadata.

    Distinguishes WGS84 geographic coordinates (latitude/longitude in degrees)
    from projected coordinates (X/Y in metres).
    """
    if not meta:
        return {
            "has_geo": False,
            "crs": "Not available",
            "bounds_wgs84": None,
            "centroid_wgs84": None,
            "corners_wgs84": None,
            "bounds_projected": None,
            "projected_crs": None,
            "is_projected": False,
        }

    georef = meta.get("georeferencing", {})

    # CRS
    crs = georef.get("crs") or meta.get("crs") or "Not available"

    # WGS84 bounds: [min_lon, min_lat, max_lon, max_lat]
    bounds_wgs = georef.get("bounds_wgs84") or meta.get("bounds_wgs84")

    centroid = None
    corners = None

    if bounds_wgs and len(bounds_wgs) == 4:
        min_lon = float(bounds_wgs[0])
        min_lat = float(bounds_wgs[1])
        max_lon = float(bounds_wgs[2])
        max_lat = float(bounds_wgs[3])

        centroid = {
            "lon": (min_lon + max_lon) / 2.0,
            "lat": (min_lat + max_lat) / 2.0,
        }

        corners = {
            "NW": {"lon": min_lon, "lat": max_lat},
            "NE": {"lon": max_lon, "lat": max_lat},
            "SW": {"lon": min_lon, "lat": min_lat},
            "SE": {"lon": max_lon, "lat": min_lat},
        }

    # Projected bounds
    bounds_proj = georef.get("bounds_projected") or meta.get("bounds_projected")
    is_projected = bounds_proj is not None and len(bounds_proj) == 4 and str(crs).upper().startswith("EPSG:32")

    return {
        "has_geo": bounds_wgs is not None and len(bounds_wgs) == 4,
        "crs": crs,
        "bounds_wgs84": bounds_wgs,
        "centroid_wgs84": centroid,
        "corners_wgs84": corners,
        "bounds_projected": bounds_proj,
        "projected_crs": crs if is_projected else None,
        "is_projected": is_projected,
    }


def render_bounding_box_diagram(
    bounds_wgs84: Optional[Sequence[float]],
    crs: str = "EPSG:4326",
    tile_id: str = "",
) -> str:
    """
    Render a clean, offline SVG diagram of the tile footprint and bounding box coordinates.
    Zero external network or map tile requests.
    """
    if not bounds_wgs84 or len(bounds_wgs84) != 4:
        return '<div class="image-placeholder">Geographic coordinates not available for diagram.</div>'

    min_lon = float(bounds_wgs84[0])
    min_lat = float(bounds_wgs84[1])
    max_lon = float(bounds_wgs84[2])
    max_lat = float(bounds_wgs84[3])

    c_lon = (min_lon + max_lon) / 2.0
    c_lat = (min_lat + max_lat) / 2.0

    d_lon = max_lon - min_lon
    d_lat = max_lat - min_lat

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 540 280" width="100%" height="280" style="background:#0c0f17; border-radius:8px; border:1px solid #242c3d; font-family:monospace;">
        <defs>
            <linearGradient id="tileGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#3b82f6" stop-opacity="0.25"/>
                <stop offset="100%" stop-color="#06b6d4" stop-opacity="0.10"/>
            </linearGradient>
            <pattern id="grid" width="30" height="30" patternUnits="userSpaceOnUse">
                <path d="M 30 0 L 0 0 0 30" fill="none" stroke="#1a2233" stroke-width="1"/>
            </pattern>
        </defs>

        <!-- Background grid -->
        <rect width="100%" height="100%" fill="url(#grid)" />

        <!-- Title & CRS info -->
        <text x="24" y="28" fill="#94a3b8" font-size="11" font-weight="600" letter-spacing="1">LOCAL FOOTPRINT BOUNDING BOX</text>
        <rect x="390" y="14" width="126" height="22" rx="4" fill="#1e293b" stroke="#3b82f6" stroke-width="1"/>
        <text x="453" y="29" fill="#3b82f6" font-size="10" font-weight="700" text-anchor="middle">CRS: {crs}</text>

        <!-- Footprint Polygon -->
        <rect x="90" y="60" width="360" height="150" rx="4" fill="url(#tileGrad)" stroke="#38bdf8" stroke-width="2" stroke-dasharray="6 3"/>

        <!-- Diagonal crosshair lines -->
        <line x1="90" y1="60" x2="450" y2="210" stroke="#38bdf8" stroke-opacity="0.15" stroke-width="1"/>
        <line x1="90" y1="210" x2="450" y2="60" stroke="#38bdf8" stroke-opacity="0.15" stroke-width="1"/>

        <!-- Center centroid pin -->
        <circle cx="270" cy="135" r="5" fill="#f59e0b" stroke="#ffffff" stroke-width="1.5"/>
        <circle cx="270" cy="135" r="12" fill="none" stroke="#f59e0b" stroke-opacity="0.4" stroke-width="1"/>
        <rect x="175" y="146" width="190" height="20" rx="4" fill="#151a24" stroke="#f59e0b" stroke-opacity="0.6"/>
        <text x="270" y="160" fill="#fcd34d" font-size="10" font-weight="bold" text-anchor="middle">Centroid: {c_lon:.5f}&deg;E, {c_lat:.5f}&deg;N</text>

        <!-- NW Corner -->
        <circle cx="90" cy="60" r="4" fill="#38bdf8"/>
        <text x="86" y="52" fill="#e2e8f0" font-size="10" font-weight="bold" text-anchor="end">NW ({min_lon:.5f}&deg;E, {max_lat:.5f}&deg;N)</text>

        <!-- NE Corner -->
        <circle cx="450" cy="60" r="4" fill="#38bdf8"/>
        <text x="454" y="52" fill="#e2e8f0" font-size="10" font-weight="bold" text-anchor="start">NE ({max_lon:.5f}&deg;E, {max_lat:.5f}&deg;N)</text>

        <!-- SW Corner -->
        <circle cx="90" cy="210" r="4" fill="#38bdf8"/>
        <text x="86" y="226" fill="#e2e8f0" font-size="10" font-weight="bold" text-anchor="end">SW ({min_lon:.5f}&deg;E, {min_lat:.5f}&deg;N)</text>

        <!-- SE Corner -->
        <circle cx="450" cy="210" r="4" fill="#38bdf8"/>
        <text x="454" y="226" fill="#e2e8f0" font-size="10" font-weight="bold" text-anchor="start">SE ({max_lon:.5f}&deg;E, {min_lat:.5f}&deg;N)</text>

        <!-- Dimension annotations -->
        <text x="270" y="78" fill="#94a3b8" font-size="10" text-anchor="middle">&Delta;lon: {d_lon:.5f}&deg; (&sim;{d_lon * 111.32 * 0.94:.2f} km)</text>
        <text x="458" y="139" fill="#94a3b8" font-size="10" text-anchor="start">&Delta;lat: {d_lat:.5f}&deg; (&sim;{d_lat * 110.57:.2f} km)</text>

        <!-- Footer note -->
        <text x="270" y="260" fill="#64748b" font-size="9" text-anchor="middle">WGS84 Axis-Aligned Geographic Extent &bull; Offline Vector Footprint Diagram</text>
    </svg>
    """
    return svg


def extract_tile_display_metadata(meta: Optional[Dict[str, Any]], hit: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract comprehensive metadata dictionary for display in the UI details panel.
    Returns authoritative values where present, or "Not available".
    """
    m = meta or {}
    h = hit or {}

    tile_id = m.get("tile_id") or h.get("tile_id") or "Not available"

    # Source metadata
    src_meta = m.get("source_metadata", {})
    scene_id = src_meta.get("source_scene_id") or m.get("scene_id") or h.get("scene_id") or "Not available"
    modality = src_meta.get("modality") or m.get("modality") or h.get("modality") or "Not available"
    platform = src_meta.get("platform") or "Not available"
    sensor = src_meta.get("sensor") or "Not available"
    processing_level = src_meta.get("processing_level") or src_meta.get("product_type") or "Not available"

    acq_dt = (
        src_meta.get("acquisition_datetime_utc")
        or m.get("acquisition_datetime_utc")
        or h.get("acquisition_datetime_utc")
        or "Not available"
    )

    # Derived tile metadata
    derived = m.get("derived_tile_metadata", {})
    grid_idx = derived.get("grid_index") or m.get("grid_index") or h.get("grid_index")
    if isinstance(grid_idx, dict):
        grid_idx_str = f"Row {grid_idx.get('row_idx')}, Col {grid_idx.get('col_idx')}"
    elif isinstance(grid_idx, (tuple, list)) and len(grid_idx) == 2:
        grid_idx_str = f"Row {grid_idx[0]}, Col {grid_idx[1]}"
    else:
        grid_idx_str = "Not available"

    dims = (
        derived.get("tile_dimensions_10m")
        or derived.get("tile_dimensions")
        or {"height": 256, "width": 256}
    )
    dim_str = f"{dims.get('width', 256)} &times; {dims.get('height', 256)} px" if isinstance(dims, dict) else "Not available"

    # Geographic / Projection
    geo_info = extract_geographic_info(m if m else h)

    # Available Bands
    bands_dict = m.get("bands", {})
    if bands_dict:
        bands_list = list(bands_dict.keys())
        bands_str = ", ".join(bands_list)
    elif modality.lower() == "optical":
        bands_str = "B02, B03, B04, B08, SCL (Sentinel-2 10m/20m)"
    elif modality.lower() == "sar":
        bands_str = "VV, VH (Sentinel-1 C-SAR)"
    else:
        bands_str = "Not available"

    # Quality diagnostics
    quality = m.get("quality_diagnostic") or h.get("quality_diagnostic")
    quality_summary = "Not available"
    if quality and isinstance(quality, dict):
        scl_sum = quality.get("scl_summary") or quality.get("diagnostic_summary")
        if scl_sum and isinstance(scl_sum, dict):
            c_pct = scl_sum.get("cloud_pixels_pct", 0.0)
            w_pct = scl_sum.get("water_pct", 0.0)
            v_pct = scl_sum.get("vegetation_pct", 0.0)
            s_pct = scl_sum.get("bare_soil_pct", 0.0)
            quality_summary = f"Cloud: {c_pct:.1f}%, Water: {w_pct:.1f}%, Vegetation: {v_pct:.1f}%, Bare Soil: {s_pct:.1f}%"

    tile_path = m.get("tile_directory") or h.get("tile_directory") or "Not available"

    return {
        "tile_id": tile_id,
        "scene_id": scene_id,
        "modality": modality.upper() if modality != "Not available" else "Not available",
        "platform": platform,
        "sensor": sensor,
        "processing_level": processing_level,
        "acquisition_datetime_utc": acq_dt,
        "grid_index": grid_idx_str,
        "dimensions": dim_str,
        "tile_directory": tile_path,
        "crs": geo_info["crs"],
        "bounds_wgs84": geo_info["bounds_wgs84"],
        "centroid_wgs84": geo_info["centroid_wgs84"],
        "corners_wgs84": geo_info["corners_wgs84"],
        "bounds_projected": geo_info["bounds_projected"],
        "is_projected": geo_info["is_projected"],
        "bands_available": bands_str,
        "quality_diagnostic": quality_summary,
        "raw_metadata": m,
    }
