"""Spatial tiling package for satellite imagery.

Supports:
- Generic deterministic spatial grid window generation (grid.py)
- Sentinel-2 Level-2A multi-resolution optical tiling (sentinel2_tiler.py)
- Sentinel-1 Level-1 GRD native radar coordinate tiling (sentinel1_tiler.py)
"""

from src.tiling.grid import TileWindow, generate_tile_windows
from src.tiling.sentinel1_tiler import tile_sentinel1_scene
from src.tiling.sentinel2_tiler import tile_sentinel2_scene

__all__ = [
    "TileWindow",
    "generate_tile_windows",
    "tile_sentinel2_scene",
    "tile_sentinel1_scene",
]
