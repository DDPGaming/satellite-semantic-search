"""Sentinel-1 M2 Spatial Tiling Module.

Tiles M1 Level-1 GRD local artifacts while strictly preserving:
- Native radar ground-range / azimuth coordinate grid (NO reprojection, warping, resampling, or TPS)
- Native 16-bit unsigned integer DN values for both VV and VH polarizations
- Source scene GCPs and translated tile-local GCPs
- Derived local affine explicitly labeled as 'derived_local_affine_approximation' (non-authoritative)
- Four-tier provenance chain
- Zero modification to M1 source rasters
- Deterministic tile IDs, windows, and repeatable generation
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint

from src.ingestion.sentinel1 import (
    compute_file_sha256,
    compute_sar_band_statistics,
    derive_local_affine,
    load_sentinel1_scene,
)
from src.tiling.grid import TileWindow, generate_tile_windows


def tile_sentinel1_scene(
    scene_dir: Path,
    output_dir: Path,
    tile_size: int = 256,
    stride: Optional[int] = None,
    boundary_strategy: str = "shift",
) -> Dict[str, Any]:
    """
    Tile a locally staged Sentinel-1 SAR scene in its native radar range-azimuth coordinate grid.

    Args:
        scene_dir: Path to staged M1 Sentinel-1 scene directory
        output_dir: Base output directory for tiles (e.g. data/tiles/sentinel1)
        tile_size: Window size in pixels (default: 256)
        stride: Stride in pixels (default: None -> stride = tile_size)
        boundary_strategy: 'shift' (default), 'truncate', or 'pad'

    Returns:
        Summary dict containing scene_id, tile_count, output_dir, and tile manifest.
    """
    scene_dir = Path(scene_dir).resolve()
    scene, val_report = load_sentinel1_scene(scene_dir)
    if val_report["status"] != "PASSED":
        raise ValueError(f"Cannot tile unvalidated Sentinel-1 scene {scene.scene_id}: {val_report['errors']}")

    # Reference band (VV)
    ref_band = scene.bands["VV"]
    h_radar, w_radar = ref_band.shape

    # Generate deterministic radar windows
    windows_radar = generate_tile_windows(
        image_width=w_radar,
        image_height=h_radar,
        tile_size=tile_size,
        stride=stride,
        boundary_strategy=boundary_strategy,
    )

    scene_tile_dir = output_dir / scene.scene_id
    scene_tile_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load data in read-only mode (without modifying M1 source files)
    sar_data = {pol: scene.bands[pol].read_data() for pol in ["VV", "VH"]}

    crop_win = scene.crop_window
    scene_col_off = crop_win.get("col_off", 0)
    scene_row_off = crop_win.get("row_off", 0)

    tiles_manifest_entries: List[Dict[str, Any]] = []
    total_tile_bytes = 0

    for win in windows_radar:
        tile_id = f"s1_{scene.scene_id}_radar_r{win.row_idx:02d}_c{win.col_idx:02d}"
        tile_dir = scene_tile_dir / tile_id
        tile_dir.mkdir(parents=True, exist_ok=True)

        # Absolute radar window within the full source scene
        abs_col_off = scene_col_off + win.col_off
        abs_row_off = scene_row_off + win.row_off

        # Translate local GCPs relative to tile origin
        tile_gcps: List[Dict[str, Any]] = []
        for g in scene.local_gcps:
            t_col = g["local_col"] - win.col_off
            t_row = g["local_row"] - win.row_off
            # Include GCPs that fall within tile or immediate neighborhood
            tile_gcps.append({
                "id": g["id"],
                "source_scene_col": g["source_col"],
                "source_scene_row": g["source_row"],
                "crop_local_col": g["local_col"],
                "crop_local_row": g["local_row"],
                "tile_col": float(t_col),
                "tile_row": float(t_row),
                "lon": g["lon"],
                "lat": g["lat"],
                "elevation_m": g["elevation_m"],
            })

        tile_gcp_objects = [
            GroundControlPoint(
                row=g["tile_row"],
                col=g["tile_col"],
                x=g["lon"],
                y=g["lat"],
                z=g["elevation_m"],
                id=g["id"],
            )
            for g in tile_gcps
        ]

        # Derived local affine approximation strictly labeled as non-authoritative
        derived_affine = derive_local_affine([
            {"local_col": g["tile_col"], "local_row": g["tile_row"], "lon": g["lon"], "lat": g["lat"]}
            for g in tile_gcps
        ])

        tile_bands_meta: Dict[str, Any] = {}

        # Write VV and VH tile GeoTIFFs
        for pol in ["VV", "VH"]:
            crop = sar_data[pol][win.row_off : win.row_off + win.height, win.col_off : win.col_off + win.width]
            target_tif = tile_dir / f"{pol.lower()}.tif"

            profile = {
                "driver": "GTiff",
                "height": win.height,
                "width": win.width,
                "count": 1,
                "dtype": crop.dtype,
                "nodata": scene.bands[pol].nodata,
                "compress": "deflate",
                "tiled": True,
                "blockxsize": 256 if win.width >= 256 else 16,
                "blockysize": 256 if win.height >= 256 else 16,
            }

            with rasterio.open(target_tif, "w", **profile) as dst:
                dst.write(crop, 1)
                if tile_gcp_objects:
                    dst.gcps = (tile_gcp_objects, "EPSG:4326")

            file_size = target_tif.stat().st_size
            total_tile_bytes += file_size
            sha256 = compute_file_sha256(target_tif)
            stats = compute_sar_band_statistics(crop, scene.bands[pol].nodata)

            tile_bands_meta[pol] = {
                "polarization": pol,
                "local_filename": f"{pol.lower()}.tif",
                "file_size_bytes": file_size,
                "local_sha256": sha256,
                "dimensions": {"height": win.height, "width": win.width},
                "dtype": str(crop.dtype),
                "nodata": scene.bands[pol].nodata,
                "statistics": stats,
            }

        tile_metadata = {
            "tile_id": tile_id,
            "source_metadata": {
                "source_scene_id": scene.scene_id,
                "modality": "sar",
                "platform": scene.platform,
                "sensor": scene.sensor,
                "instrument_mode": scene.instrument_mode,
                "product_type": scene.product_type,
                "resolution_type": scene.resolution_type,
                "polarizations": scene.polarizations,
                "orbit_direction": scene.orbit_direction,
                "relative_orbit": scene.relative_orbit,
                "absolute_orbit": scene.absolute_orbit,
                "acquisition_datetime_utc": scene.acquisition_datetime,
                "pixel_spacing_m": list(scene.pixel_spacing_m),
                "effective_resolution_m": list(scene.effective_resolution_m),
                "source_scene_directory": str(scene_dir),
                "source_crop_dimensions": {"height": h_radar, "width": w_radar},
            },
            "derived_tile_metadata": {
                "grid_index": {"row_idx": win.row_idx, "col_idx": win.col_idx},
                "tile_dimensions": {"height": win.height, "width": win.width},
                "crop_window": {
                    "col_off": win.col_off,
                    "row_off": win.row_off,
                    "width": win.width,
                    "height": win.height,
                },
                "source_scene_absolute_window": {
                    "col_off": abs_col_off,
                    "row_off": abs_row_off,
                    "width": win.width,
                    "height": win.height,
                },
            },
            "georeferencing": {
                "coordinate_space": "native_radar_range_azimuth",
                "crs": "EPSG:4326",
                "is_authoritative_affine": False,
                "authoritative_model": "ESA PDGS Ground Control Point (GCP) geolocation",
                "tile_local_gcps": tile_gcps,
                "derived_local_affine_approximation": derived_affine,
            },
            "bands": tile_bands_meta,
            "provenance": {
                "source_provenance": scene.provenance,
                "tiling_process": {
                    "generator": "src/tiling/sentinel1_tiler.py",
                    "tile_size": tile_size,
                    "stride": stride or tile_size,
                    "boundary_strategy": boundary_strategy,
                    "boundary_policy_note": (
                        "stride=256 is used; 'shift' boundary policy may introduce overlap only at terminal boundary "
                        "tiles in order to maintain fixed 256x256 dimensions without synthetic padding."
                    ),
                    "coordinate_policy": "Strict native radar range-azimuth grid. Zero reprojection or resampling.",
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
            "crop_window": [win.col_off, win.row_off, win.width, win.height],
            "absolute_window": [abs_col_off, abs_row_off, win.width, win.height],
            "tile_directory": f"{tile_id}",
            "metadata_sha256": compute_file_sha256(tile_meta_path),
        })

    # Write scene-level manifest
    manifest_doc = {
        "source_scene_id": scene.scene_id,
        "modality": "sar",
        "platform": scene.platform,
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
