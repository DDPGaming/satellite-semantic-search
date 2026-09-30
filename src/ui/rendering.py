"""Image rendering and base64 encoding utilities for the FLUX presentation UI.

Provides helper routines to convert raw GeoTIFF rasters and in-memory NumPy arrays
into browser-renderable base64 PNG data URLs without external web hosting or CDN dependencies.
"""

import base64
import io
from pathlib import Path
from typing import Optional, Union
import numpy as np
from PIL import Image
import rasterio


def get_default_project_root() -> Path:
    """Return project root based on repository structure."""
    return Path(__file__).resolve().parent.parent.parent


def _resolve_tile_dir(tile_dir_or_id: Union[str, Path], project_root: Optional[Path] = None) -> Optional[Path]:
    """Resolve a tile directory path from a relative path, absolute path, or tile ID."""
    root = project_root if project_root else get_default_project_root()
    p = Path(tile_dir_or_id)

    if p.is_absolute() and p.exists():
        return p

    cand1 = root / p
    if cand1.exists():
        return cand1

    cand2 = root / "data" / p
    if cand2.exists():
        return cand2

    # Search in data/tiles for tile_id
    tile_name = p.name
    tiles_base = root / "data" / "tiles"
    if tiles_base.exists():
        for d in tiles_base.glob(f"**/{tile_name}"):
            if d.is_dir():
                return d

    return None


def render_tile_image(
    tile_dir_or_id: Union[str, Path],
    modality: str = "optical",
    project_root: Optional[Path] = None,
) -> Optional[str]:
    """
    Render a true-color (optical) or backscatter (SAR) tile raster to a base64 PNG data URL.

    Args:
        tile_dir_or_id: Tile directory path or tile ID.
        modality: "optical" or "sar".
        project_root: Optional repository root path.

    Returns:
        Base64 PNG data URL string, or None if files cannot be found or read.
    """
    tile_dir = _resolve_tile_dir(tile_dir_or_id, project_root)
    if not tile_dir:
        return None

    try:
        if modality.lower() == "optical":
            b04_path = tile_dir / "B04.tif"
            b03_path = tile_dir / "B03.tif"
            b02_path = tile_dir / "B02.tif"

            if not (b04_path.exists() and b03_path.exists() and b02_path.exists()):
                return None

            with rasterio.open(b04_path) as s4, rasterio.open(b03_path) as s3, rasterio.open(b02_path) as s2:
                r = s4.read(1).astype(np.float32)
                g = s3.read(1).astype(np.float32)
                b = s2.read(1).astype(np.float32)

            rgb = np.stack([r, g, b], axis=-1)
            # Sentinel-2 L2A surface reflectance visual scaling (0 - 2500 typical visual dynamic range)
            rgb_scaled = np.clip((rgb / 2500.0) * 255.0, 0, 255).astype(np.uint8)
            img = Image.fromarray(rgb_scaled)

        elif modality.lower() == "sar":
            # Search for vv.tif / vh.tif (case-insensitive)
            vv_path = None
            vh_path = None
            for f in tile_dir.iterdir():
                if f.name.lower() == "vv.tif":
                    vv_path = f
                elif f.name.lower() == "vh.tif":
                    vh_path = f

            if not vv_path or not vv_path.exists():
                return None

            with rasterio.open(vv_path) as s_vv:
                vv = s_vv.read(1).astype(np.float32)

            # Normalization for radar amplitude/backscatter
            p99 = float(np.percentile(vv[vv > 0], 99)) if np.any(vv > 0) else 1.0
            p99 = max(p99, 1e-3)
            vv_norm = np.clip((vv / p99) * 255.0, 0, 255).astype(np.uint8)

            if vh_path and vh_path.exists():
                with rasterio.open(vh_path) as s_vh:
                    vh = s_vh.read(1).astype(np.float32)
                p99_vh = float(np.percentile(vh[vh > 0], 99)) if np.any(vh > 0) else 1.0
                vh_norm = np.clip((vh / max(p99_vh, 1e-3)) * 255.0, 0, 255).astype(np.uint8)
                ratio_norm = np.clip(((vv / (vh + 1e-4)) / 10.0) * 255.0, 0, 255).astype(np.uint8)
                rgb_sar = np.stack([vv_norm, vh_norm, ratio_norm], axis=-1)
                img = Image.fromarray(rgb_sar)
            else:
                img = Image.fromarray(vv_norm).convert("RGB")

        else:
            return None

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    except Exception:
        return None


def render_change_heatmap(change_magnitude: Optional[np.ndarray]) -> Optional[str]:
    """
    Render a continuous change magnitude array to an orange/red heatmap PNG data URL.

    Args:
        change_magnitude: 2D float32 numpy array.

    Returns:
        Base64 PNG data URL string, or None if array is missing or invalid.
    """
    if change_magnitude is None or change_magnitude.ndim != 2:
        return None

    try:
        valid_mask = np.isfinite(change_magnitude) & (~np.isnan(change_magnitude))
        h, w = change_magnitude.shape

        # Default dark slate background for invalid / nodata pixels
        rgb = np.full((h, w, 3), (18, 22, 30), dtype=np.uint8)

        if np.any(valid_mask):
            valid_vals = change_magnitude[valid_mask]
            # Use 99th percentile for visual dynamic range scaling
            p99 = float(np.percentile(valid_vals, 99)) if np.any(valid_vals > 0) else 0.5
            p99 = max(p99, 0.05)
            norm = np.clip(change_magnitude[valid_mask] / p99, 0.0, 1.0)

            # Warm magma/hot colormap gradient (black/slate -> purple -> orange -> bright yellow)
            r = (norm * 255).astype(np.uint8)
            g = (np.power(norm, 1.4) * 200).astype(np.uint8)
            b = (np.power(norm, 2.5) * 80).astype(np.uint8)

            rgb[valid_mask, 0] = r
            rgb[valid_mask, 1] = g
            rgb[valid_mask, 2] = b

        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    except Exception:
        return None


def render_mask_image(confirmed_mask: Optional[np.ndarray]) -> Optional[str]:
    """
    Render a confirmed change boolean mask to a high-contrast binary PNG data URL.

    Args:
        confirmed_mask: 2D boolean numpy array.

    Returns:
        Base64 PNG data URL string, or None if mask is missing.
    """
    if confirmed_mask is None or confirmed_mask.ndim != 2:
        return None

    try:
        h, w = confirmed_mask.shape
        # Dark slate background for unchanged pixels
        rgb = np.full((h, w, 3), (18, 22, 30), dtype=np.uint8)

        # High-contrast bright coral red for confirmed change pixels
        change_color = (255, 65, 54)
        rgb[confirmed_mask] = change_color

        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    except Exception:
        return None


def render_confidence_heatmap(confidence_map: Optional[np.ndarray]) -> Optional[str]:
    """
    Render a heuristic confidence map to a cyan-to-emerald gradient PNG data URL.

    Args:
        confidence_map: 2D float32 numpy array with values in [0.0, 1.0].

    Returns:
        Base64 PNG data URL string, or None if confidence map is missing.
    """
    if confidence_map is None or confidence_map.ndim != 2:
        return None

    try:
        h, w = confidence_map.shape
        rgb = np.full((h, w, 3), (18, 22, 30), dtype=np.uint8)

        conf_mask = np.isfinite(confidence_map) & (confidence_map > 0.0)
        if np.any(conf_mask):
            conf_vals = np.clip(confidence_map[conf_mask], 0.0, 1.0)
            # Gradient: Cyan (0, 210, 255) to Bright Emerald Green (0, 255, 130)
            r = (np.power(1.0 - conf_vals, 2.0) * 40).astype(np.uint8)
            g = (180 + conf_vals * 75).astype(np.uint8)
            b = (255 - conf_vals * 125).astype(np.uint8)

            rgb[conf_mask, 0] = r
            rgb[conf_mask, 1] = g
            rgb[conf_mask, 2] = b

        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    except Exception:
        return None
