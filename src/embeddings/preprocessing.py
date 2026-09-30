"""Deterministic Preprocessing for Satellite Imagery and Multimodal Tiles.

Provides deterministic, sensor-aware preprocessing for optical and SAR satellite tiles:
- Sentinel-2: True-color RGB composition (B04/B03/B02) with calibrated surface reflectance scaling
- Sentinel-1: Calibrated dual-polarization (VV/VH) false-color composition with logarithmic scaling
- Generic images: Safe PIL Image loading and format normalization
- Guarantees 100% deterministic output for identical inputs (no random crops or augmentations)
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import rasterio
from PIL import Image

# Default surface reflectance visual ceiling for Sentinel-2 Level-2A (B04, B03, B02)
# Standard Earth observation display range: reflectance 0.0 to 0.35 covers >98% of natural terrain
S2_REFLECTANCE_SCALE = 10000.0
S2_VISUAL_MAX_REFLECTANCE = 0.35

# Sentinel-1 amplitude normalization constants (log10 scale)
S1_LOG_VV_MAX = 3.6
S1_LOG_VH_MAX = 3.2


def preprocess_sentinel2_tile(
    tile_dir: Union[Path, str],
    visual_max_reflectance: float = S2_VISUAL_MAX_REFLECTANCE,
) -> Image.Image:
    """
    Deterministically preprocess a Sentinel-2 optical tile into an RGB PIL Image.

    Reads 10m native bands:
    - B04.tif (Red, 665 nm)
    - B03.tif (Green, 560 nm)
    - B02.tif (Blue, 490 nm)

    Scales DN values by 10,000 (L2A reflectance) and applies deterministic linear
    visual contrast stretching up to visual_max_reflectance.

    Args:
        tile_dir: Path to tile directory containing B02.tif, B03.tif, B04.tif.
        visual_max_reflectance: Upper surface reflectance bound mapped to 255.

    Returns:
        RGB PIL Image of size (width, height).
    """
    tile_path = Path(tile_dir)
    b04_file = tile_path / "B04.tif"
    b03_file = tile_path / "B03.tif"
    b02_file = tile_path / "B02.tif"

    if not b04_file.exists() or not b03_file.exists() or not b02_file.exists():
        raise FileNotFoundError(
            f"Sentinel-2 tile at '{tile_dir}' is missing required optical bands (B04, B03, B02)."
        )

    with rasterio.open(b04_file) as src_r:
        red = src_r.read(1).astype(np.float32)
    with rasterio.open(b03_file) as src_g:
        green = src_g.read(1).astype(np.float32)
    with rasterio.open(b02_file) as src_b:
        blue = src_b.read(1).astype(np.float32)

    # Stack into [H, W, 3] RGB
    rgb = np.stack([red, green, blue], axis=-1)

    # Convert native DN to surface reflectance [0.0, 1.0]
    rgb_refl = np.clip(rgb / S2_REFLECTANCE_SCALE, 0.0, 1.0)

    # Calibrated contrast scaling mapped to [0, 255]
    rgb_scaled = np.clip(rgb_refl / visual_max_reflectance, 0.0, 1.0) * 255.0
    rgb_uint8 = np.round(rgb_scaled).astype(np.uint8)

    return Image.fromarray(rgb_uint8, mode="RGB")


def preprocess_sentinel1_tile(
    tile_dir: Union[Path, str],
) -> Image.Image:
    """
    Deterministically preprocess a Sentinel-1 SAR tile into a 3-channel PIL Image.

    Reads native radar amplitude bands:
    - vv.tif (Co-polarized, vertical transmit / vertical receive)
    - vh.tif (Cross-polarized, vertical transmit / horizontal receive)

    Applies logarithmic amplitude compression and maps VV/VH to false-color RGB channels:
    - R: VV normalized amplitude
    - G: VH normalized amplitude
    - B: Mean of (VV, VH) normalized amplitude

    Args:
        tile_dir: Path to tile directory containing vv.tif and vh.tif.

    Returns:
        RGB PIL Image of size (width, height).
    """
    tile_path = Path(tile_dir)
    vv_file = tile_path / "vv.tif"
    vh_file = tile_path / "vh.tif"

    if not vv_file.exists() or not vh_file.exists():
        raise FileNotFoundError(
            f"Sentinel-1 tile at '{tile_dir}' is missing required SAR polarization bands (vv.tif, vh.tif)."
        )

    with rasterio.open(vv_file) as src_vv:
        vv = src_vv.read(1).astype(np.float32)
    with rasterio.open(vh_file) as src_vh:
        vh = src_vh.read(1).astype(np.float32)

    # Non-negative guard
    vv = np.maximum(vv, 0.0)
    vh = np.maximum(vh, 0.0)

    # Logarithmic compression
    vv_log = np.log10(1.0 + vv)
    vh_log = np.log10(1.0 + vh)

    # Normalized channels [0.0, 1.0]
    vv_norm = np.clip(vv_log / S1_LOG_VV_MAX, 0.0, 1.0)
    vh_norm = np.clip(vh_log / S1_LOG_VH_MAX, 0.0, 1.0)
    mean_norm = np.clip(0.5 * (vv_norm + vh_norm), 0.0, 1.0)

    # False-color 3-channel composite
    sar_composite = np.stack([vv_norm, vh_norm, mean_norm], axis=-1)
    sar_uint8 = np.round(sar_composite * 255.0).astype(np.uint8)

    return Image.fromarray(sar_uint8, mode="RGB")


def load_image_to_pil(image_input: Union[Image.Image, np.ndarray, Path, str]) -> Image.Image:
    """
    Safely and deterministically resolve diverse input representations into a PIL RGB Image.

    Supports:
    - Existing PIL Image (ensures RGB mode)
    - Path to a Sentinel-2 tile directory (containing B02, B03, B04)
    - Path to a Sentinel-1 tile directory (containing vv, vh)
    - Path to an image file (JPEG, PNG, TIFF)
    - NumPy array (2D grayscale, 3D channel-first or channel-last RGB)

    Args:
        image_input: Input image representation.

    Returns:
        Standard PIL Image in "RGB" mode.
    """
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB") if image_input.mode != "RGB" else image_input

    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if p.is_dir():
            # Check for Sentinel-2 tile
            if (p / "B04.tif").exists() and (p / "B02.tif").exists():
                return preprocess_sentinel2_tile(p)
            # Check for Sentinel-1 tile
            if (p / "vv.tif").exists() and (p / "vh.tif").exists():
                return preprocess_sentinel1_tile(p)
            raise ValueError(f"Directory '{p}' does not match recognized Sentinel-2 or Sentinel-1 tile structures.")
        elif p.is_file():
            with Image.open(p) as img:
                return img.convert("RGB")
        else:
            raise FileNotFoundError(f"Input path '{p}' does not exist.")

    if isinstance(image_input, np.ndarray):
        arr = image_input
        if arr.ndim == 2:
            # Grayscale: replicate across 3 channels
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            stacked = np.stack([arr, arr, arr], axis=-1)
            return Image.fromarray(stacked, mode="RGB")
        elif arr.ndim == 3:
            # Handle channel-first [3, H, W] vs channel-last [H, W, 3]
            if arr.shape[0] == 3 and arr.shape[2] != 3:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr[:, :, :3], mode="RGB")
        else:
            raise ValueError(f"Unsupported NumPy array shape for image preprocessing: {arr.shape}")

    raise TypeError(f"Unsupported image input type: {type(image_input)}")
