"""Sentinel-2 M2 Spatial Tiling Module.

Tiles M1 Level-2A local artifacts while strictly preserving:
- Native resolutions: 10m for optical bands (B02, B03, B04, B08), 20m for SCL (NO resampling)
- Full authoritative georeferencing (EPSG:32643, affine transform, bounding boxes)
- Source scene provenance and SCL quality diagnostics per tile
- Zero modification to M1 source rasters
- Deterministic tile IDs, windows, and repeatable generation
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from affine import Affine

from src.ingestion.sentinel2 import (
    compute_band_statistics,
    compute_file_sha256,
    compute_scl_distribution,
    load_sentinel2_scene,
)
from src.tiling.grid import TileWindow, generate_tile_windows


def tile_sentinel2_scene(
    scene_dir: Path,
    output_dir: Path,
    tile_size: int = 256,
    stride: Optional[int] = None,
    boundary_strategy: str = "shift",
) -> Dict[str, Any]:
    """
    Tile a locally staged Sentinel-2 scene.

    Args:
        scene_dir: Path to staged M1 Sentinel-2 scene directory
        output_dir: Base output directory for tiles (e.g. data/tiles/sentinel2)
        tile_size: Window size in pixels for native 10m bands (default: 256)
        stride: Stride in pixels (default: None -> stride = tile_size)
        boundary_strategy: 'shift' (default), 'truncate', or 'pad'

    Returns:
        Summary dict containing scene_id, tile_count, output_dir, and tile manifest.
    """
    scene_dir = Path(scene_dir).resolve()
    scene, val_report = load_sentinel2_scene(scene_dir)
    if val_report["status"] != "PASSED":
        raise ValueError(f"Cannot tile unvalidated Sentinel-2 scene {scene.scene_id}: {val_report['errors']}")

    # 10m optical raster reference (B02)
    ref_10m = scene.bands["B02"]
    h_10m, w_10m = ref_10m.shape
    transform_10m = ref_10m.transform
    crs_str = ref_10m.crs

    # Generate deterministic 10m windows
    windows_10m = generate_tile_windows(
        image_width=w_10m,
        image_height=h_10m,
        tile_size=tile_size,
        stride=stride,
        boundary_strategy=boundary_strategy,
    )

    scene_tile_dir = output_dir / scene.scene_id
    scene_tile_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load data in read-only mode (without modifying M1 source files)
    optical_data = {bid: scene.bands[bid].read_data() for bid in ["B02", "B03", "B04", "B08"]}
    scl_band = scene.bands.get("SCL")
    scl_data = scl_band.read_data() if scl_band else None

    tiles_manifest_entries: List[Dict[str, Any]] = []
    total_tile_bytes = 0

    for win in windows_10m:
        tile_id = f"s2_{scene.scene_id}_10m_r{win.row_idx:02d}_c{win.col_idx:02d}"
        tile_dir = scene_tile_dir / tile_id
        tile_dir.mkdir(parents=True, exist_ok=True)

        # Derived 10m affine transform
        # X = col * a + row * b + c
        # Y = col * d + row * e + f
        tile_transform_10m = transform_10m @ Affine.translation(win.col_off, win.row_off)
        tile_bounds_proj = [
            tile_transform_10m.c,
            tile_transform_10m.f + win.height * tile_transform_10m.e,
            tile_transform_10m.c + win.width * tile_transform_10m.a,
            tile_transform_10m.f,
        ]

        # Calculate WGS84 bounding box via pyproj / rasterio.warp
        from rasterio.warp import transform_bounds
        bounds_wgs84 = list(transform_bounds(crs_str, "EPSG:4326", *tile_bounds_proj))

        tile_bands_meta: Dict[str, Any] = {}

        # 1. Write 10m Optical Bands
        for bid in ["B02", "B03", "B04", "B08"]:
            band_crop = optical_data[bid][win.row_off : win.row_off + win.height, win.col_off : win.col_off + win.width]
            target_tif = tile_dir / f"{bid}.tif"

            profile = {
                "driver": "GTiff",
                "height": win.height,
                "width": win.width,
                "count": 1,
                "dtype": band_crop.dtype,
                "crs": crs_str,
                "transform": tile_transform_10m,
                "nodata": scene.bands[bid].nodata,
                "compress": "deflate",
                "tiled": True,
                "blockxsize": 256 if win.width >= 256 else 16,
                "blockysize": 256 if win.height >= 256 else 16,
            }

            with rasterio.open(target_tif, "w", **profile) as dst:
                dst.write(band_crop, 1)

            file_size = target_tif.stat().st_size
            total_tile_bytes += file_size
            sha256 = compute_file_sha256(target_tif)
            stats = compute_band_statistics(band_crop, scene.bands[bid].nodata)

            tile_bands_meta[bid] = {
                "band_identity": bid,
                "pixel_resolution_m": 10.0,
                "local_filename": f"{bid}.tif",
                "file_size_bytes": file_size,
                "local_sha256": sha256,
                "dimensions": {"height": win.height, "width": win.width},
                "dtype": str(band_crop.dtype),
                "nodata": scene.bands[bid].nodata,
                "statistics": stats,
            }

        # 2. Corresponding 20m SCL Layer (preserves native 20m resolution, NO resampling)
        scl_tile_meta: Optional[Dict[str, Any]] = None
        scl_diag_summary: Optional[Dict[str, Any]] = None

        if scl_data is not None and scl_band is not None:
            # Map 10m window to native 20m pixel window
            scl_col_off = win.col_off // 2
            scl_row_off = win.row_off // 2
            scl_width = win.width // 2
            scl_height = win.height // 2

            scl_crop = scl_data[scl_row_off : scl_row_off + scl_height, scl_col_off : scl_col_off + scl_width]
            scl_tif = tile_dir / "SCL.tif"

            scl_transform = scl_band.transform @ Affine.translation(scl_col_off, scl_row_off)
            scl_profile = {
                "driver": "GTiff",
                "height": scl_height,
                "width": scl_width,
                "count": 1,
                "dtype": scl_crop.dtype,
                "crs": crs_str,
                "transform": scl_transform,
                "nodata": scl_band.nodata,
                "compress": "deflate",
                "tiled": True,
                "blockxsize": 128 if scl_width >= 128 else 16,
                "blockysize": 128 if scl_height >= 128 else 16,
            }

            with rasterio.open(scl_tif, "w", **scl_profile) as dst:
                dst.write(scl_crop, 1)

            scl_size = scl_tif.stat().st_size
            total_tile_bytes += scl_size
            scl_sha256 = compute_file_sha256(scl_tif)
            scl_dist = compute_scl_distribution(scl_crop)
            scl_diag_summary = scl_dist["diagnostic_summary"]

            scl_tile_meta = {
                "band_identity": "SCL",
                "pixel_resolution_m": 20.0,
                "local_filename": "SCL.tif",
                "file_size_bytes": scl_size,
                "local_sha256": scl_sha256,
                "dimensions": {"height": scl_height, "width": scl_width},
                "dtype": str(scl_crop.dtype),
                "nodata": scl_band.nodata,
                "source_window_20m": {
                    "col_off": scl_col_off,
                    "row_off": scl_row_off,
                    "width": scl_width,
                    "height": scl_height,
                },
                "diagnostic_summary": scl_diag_summary,
            }
            tile_bands_meta["SCL"] = scl_tile_meta

        # Assemble stable tile metadata schema
        tile_metadata = {
            "tile_id": tile_id,
            "source_metadata": {
                "source_scene_id": scene.scene_id,
                "modality": "optical",
                "platform": scene.spacecraft,
                "sensor": scene.sensor,
                "processing_level": scene.processing_level,
                "mgrs_tile": scene.mgrs_tile,
                "acquisition_datetime_utc": scene.acquisition_datetime,
                "source_scene_directory": str(scene_dir),
                "source_raster_dimensions_10m": {"height": h_10m, "width": w_10m},
            },
            "derived_tile_metadata": {
                "grid_index": {"row_idx": win.row_idx, "col_idx": win.col_idx},
                "tile_dimensions_10m": {"height": win.height, "width": win.width},
                "source_window_10m": {
                    "col_off": win.col_off,
                    "row_off": win.row_off,
                    "width": win.width,
                    "height": win.height,
                },
                "spatial_coverage_metres": {
                    "width_m": win.width * 10.0,
                    "height_m": win.height * 10.0,
                },
            },
            "georeferencing": {
                "crs": crs_str,
                "is_authoritative": True,
                "affine_transform": [
                    tile_transform_10m.a,
                    tile_transform_10m.b,
                    tile_transform_10m.c,
                    tile_transform_10m.d,
                    tile_transform_10m.e,
                    tile_transform_10m.f,
                ],
                "bounds_projected": tile_bounds_proj,
                "bounds_wgs84": bounds_wgs84,
            },
            "quality_diagnostic": {
                "scl_summary": scl_diag_summary,
            },
            "bands": tile_bands_meta,
            "provenance": {
                "source_provenance": scene.provenance,
                "tiling_process": {
                    "generator": "src/tiling/sentinel2_tiler.py",
                    "tile_size": tile_size,
                    "stride": stride or tile_size,
                    "boundary_strategy": boundary_strategy,
                    "boundary_policy_note": (
                        "stride=256 is used; 'shift' boundary policy may introduce overlap only at terminal boundary "
                        "tiles in order to maintain fixed 256x256 dimensions without synthetic padding."
                    ),
                    "multi_resolution_policy": "Native grids preserved: 10m optical (256x256), 20m SCL (128x128). Zero resampling.",
                    "creation_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            },
        }

        # Write tile metadata.json
        tile_meta_path = tile_dir / "metadata.json"
        with open(tile_meta_path, "w", encoding="utf-8") as f:
            json.dump(tile_metadata, f, indent=2)

        total_tile_bytes += tile_meta_path.stat().st_size

        tiles_manifest_entries.append({
            "tile_id": tile_id,
            "grid_index": [win.row_idx, win.col_idx],
            "source_window_10m": [win.col_off, win.row_off, win.width, win.height],
            "bounds_projected": tile_bounds_proj,
            "bounds_wgs84": bounds_wgs84,
            "tile_directory": f"{tile_id}",
            "metadata_sha256": compute_file_sha256(tile_meta_path),
        })

    # Write scene-level manifest
    manifest_doc = {
        "source_scene_id": scene.scene_id,
        "modality": "optical",
        "platform": scene.spacecraft,
        "acquisition_datetime_utc": scene.acquisition_datetime,
        "tiling_config": {
            "tile_size": tile_size,
            "stride": stride or tile_size,
            "boundary_strategy": boundary_strategy,
            "boundary_policy_note": (
                "stride=256 is used; 'shift' boundary policy may introduce overlap only at terminal boundary "
                "tiles in order to maintain fixed 256x256 dimensions without synthetic padding."
            ),
            "total_tiles": len(tiles_manifest_entries),
            "grid_shape": [len(windows_10m) // len(set(w.col_off for w in windows_10m)), len(set(w.col_off for w in windows_10m))],
        },
        "total_scene_tiles_bytes": total_tile_bytes,
        "tiles": tiles_manifest_entries,
    }

    manifest_path = scene_tile_dir / "tiles_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_doc, f, indent=2)

    return {
        "scene_id": scene.scene_id,
        "total_tiles": len(tiles_manifest_entries),
        "total_bytes": total_tile_bytes,
        "scene_tile_dir": scene_tile_dir,
        "manifest_path": manifest_path,
    }
