"""Deterministic spatial tile window generator.

Reusable, sensor-agnostic tiling logic:
- Computes deterministic grid windows (col_off, row_off, width, height)
- Configurable tile size and stride (default: stride=tile_size=256)
- Deterministic boundary handling:
    - 'shift' (default): Shifts the final edge tile so it aligns flush with the image boundary.
      When image dimensions are not exact multiples of stride, stride=tile_size is used for all
      interior tiles (which remain strictly non-overlapping), while the 'shift' policy may introduce
      overlap only at the terminal boundary tiles in order to maintain fixed (tile_size, tile_size)
      dimensions without synthetic padding. 100% of source pixels remain covered.
    - 'truncate': Keeps partial edge tiles at their truncated dimensions (strictly non-overlapping).
    - 'pad': Extends beyond boundary with fixed dimensions.
- Zero dependency on specific geography / AOI.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class TileWindow:
    """Deterministic 2D pixel window in a raster."""
    row_idx: int
    col_idx: int
    col_off: int
    row_off: int
    width: int
    height: int


def compute_axis_offsets(
    dimension_size: int,
    tile_size: int,
    stride: int,
    boundary_strategy: str = "shift",
) -> List[Tuple[int, int]]:
    """
    Compute 1D offsets and sizes along an axis.

    Returns:
        List of (offset, length) tuples.
    """
    if dimension_size <= 0 or tile_size <= 0:
        raise ValueError(f"dimension_size ({dimension_size}) and tile_size ({tile_size}) must be positive")
    if stride <= 0:
        raise ValueError(f"stride ({stride}) must be positive")

    # If the image is smaller than the requested tile size
    if dimension_size <= tile_size:
        if boundary_strategy in ("shift", "truncate"):
            return [(0, dimension_size)]
        else:
            return [(0, tile_size)]

    offsets: List[Tuple[int, int]] = []
    current = 0

    while current + tile_size <= dimension_size:
        offsets.append((current, tile_size))
        current += stride

    # Handle remaining edge
    if current < dimension_size:
        if boundary_strategy == "shift":
            shifted_offset = dimension_size - tile_size
            # Only add if it's not identical to the last added offset
            if not offsets or offsets[-1][0] != shifted_offset:
                offsets.append((shifted_offset, tile_size))
        elif boundary_strategy == "truncate":
            rem_len = dimension_size - current
            offsets.append((current, rem_len))
        elif boundary_strategy == "pad":
            offsets.append((current, tile_size))
        else:
            raise ValueError(f"Unknown boundary_strategy: '{boundary_strategy}'. Must be 'shift', 'truncate', or 'pad'.")

    return offsets


def generate_tile_windows(
    image_width: int,
    image_height: int,
    tile_size: int = 256,
    stride: Optional[int] = None,
    boundary_strategy: str = "shift",
) -> List[TileWindow]:
    """
    Generate a deterministic 2D grid of TileWindow instances in row-major order (top to bottom, left to right).

    Args:
        image_width: Width of source raster in pixels
        image_height: Height of source raster in pixels
        tile_size: Window size in pixels (default: 256)
        stride: Step between consecutive tiles (default: None -> stride = tile_size)
        boundary_strategy: 'shift' (default), 'truncate', or 'pad'

    Returns:
        Deterministic list of TileWindow objects.
    """
    if stride is None:
        stride = tile_size

    x_offsets = compute_axis_offsets(image_width, tile_size, stride, boundary_strategy)
    y_offsets = compute_axis_offsets(image_height, tile_size, stride, boundary_strategy)

    windows: List[TileWindow] = []
    for r_idx, (row_off, height) in enumerate(y_offsets):
        for c_idx, (col_off, width) in enumerate(x_offsets):
            windows.append(
                TileWindow(
                    row_idx=r_idx,
                    col_idx=c_idx,
                    col_off=col_off,
                    row_off=row_off,
                    width=width,
                    height=height,
                )
            )

    return windows
