"""Windowed Sentinel-2 Level-2A Sample Ingestion Script for M1 Stage 1.

CRITICAL ACQUISITION REQUIREMENT:
    Uses the remote Cloud-Optimized GeoTIFF (COG) geospatial transform/window
    to stream ONLY the required AOI window over HTTP Range requests.
    Does NOT download or materialize the full 100km x 100km tile or .SAFE archive.

Target Scenes (Navi Mumbai / JNPT AOI):
    1. S2A_43QBB_20220127_0_L2A (Sentinel-2A, 2022-01-27T05:53:40Z)
    2. S2B_43QBB_20240112_0_L2A (Sentinel-2B, 2024-01-12T05:53:38Z)
    Temporal baseline: Approximately 23.5 months / 716 days
    Spacecraft: Distinct (S2A vs S2B) sharing spatial MGRS grid tile 43QBB

Bands Preserved:
    - Native 10m grid: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
    - Native 20m grid: SCL (Scene Classification Layer)
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.sentinel2 import (
    compute_band_statistics,
    compute_file_sha256,
    compute_scl_distribution,
    load_sentinel2_scene,
)

# STAC asset map for the two verified scenes
SCENE_SPECS = {
    "S2A_43QBB_20220127_0_L2A": {
        "spacecraft": "Sentinel-2A",
        "sensor": "MSI",
        "processing_level": "Level-2A",
        "mgrs_tile": "43QBB",
        "acquisition_datetime": "2022-01-27T05:53:40.959000Z",
        "cloud_cover_tile_pct": 0.001058,
        "sun_elevation": 46.3963,
        "sun_azimuth": 148.8782,
        "stac_item_url": "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2A_43QBB_20220127_0_L2A",
        "assets": {
            "B02": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/Q/BB/2022/1/S2A_43QBB_20220127_0_L2A/B02.tif",
            "B03": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/Q/BB/2022/1/S2A_43QBB_20220127_0_L2A/B03.tif",
            "B04": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/Q/BB/2022/1/S2A_43QBB_20220127_0_L2A/B04.tif",
            "B08": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/Q/BB/2022/1/S2A_43QBB_20220127_0_L2A/B08.tif",
            "SCL": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/Q/BB/2022/1/S2A_43QBB_20220127_0_L2A/SCL.tif",
        },
    },
    "S2B_43QBB_20240112_0_L2A": {
        "spacecraft": "Sentinel-2B",
        "sensor": "MSI",
        "processing_level": "Level-2A",
        "mgrs_tile": "43QBB",
        "acquisition_datetime": "2024-01-12T05:53:38.695000Z",
        "cloud_cover_tile_pct": 0.000969,
        "sun_elevation": 44.0842,
        "sun_azimuth": 152.4957,
        "stac_item_url": "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2B_43QBB_20240112_0_L2A",
        "assets": {
            "B02": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/Q/BB/2024/1/S2B_43QBB_20240112_0_L2A/B02.tif",
            "B03": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/Q/BB/2024/1/S2B_43QBB_20240112_0_L2A/B03.tif",
            "B04": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/Q/BB/2024/1/S2B_43QBB_20240112_0_L2A/B04.tif",
            "B08": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/Q/BB/2024/1/S2B_43QBB_20240112_0_L2A/B08.tif",
            "SCL": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/Q/BB/2024/1/S2B_43QBB_20240112_0_L2A/SCL.tif",
        },
    },
}

# Derived exact crop windows for Navi Mumbai/JNPT AOI in EPSG:32643
GRID_CONFIGS = {
    10.0: {
        "col_off": 8409,
        "row_off": 9240,
        "width": 1599,
        "height": 1679,
        "bounds": [284070.0, 2090830.0, 300060.0, 2107620.0],
    },
    20.0: {
        "col_off": 4204,
        "row_off": 4620,
        "width": 800,
        "height": 840,
        "bounds": [284060.0, 2090820.0, 300060.0, 2107620.0],
    },
}


def ingest_scene_bands(
    scene_id: str,
    spec: Dict[str, Any],
    output_base_dir: Path,
) -> Dict[str, Any]:
    """
    Perform windowed reads from remote COGs and write compressed local GeoTIFF crops.
    """
    scene_dir = output_base_dir / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========================================================")
    print(f"Ingesting Scene: {scene_id} ({spec['spacecraft']})")
    print(f"Acquisition:     {spec['acquisition_datetime']}")
    print(f"Destination:     {scene_dir}")
    print(f"========================================================")

    bands_metadata: Dict[str, Any] = {}
    total_scene_bytes = 0

    for band_id, remote_url in spec["assets"].items():
        res = 20.0 if band_id == "SCL" else 10.0
        grid = GRID_CONFIGS[res]

        col_off = grid["col_off"]
        row_off = grid["row_off"]
        width = grid["width"]
        height = grid["height"]

        target_tif = scene_dir / f"{band_id}.tif"
        window = Window(col_off=col_off, row_off=row_off, width=width, height=height)

        print(f"  Streaming window for {band_id:<4} ({res}m, {width}x{height})...")
        t0 = time.perf_counter()

        with rasterio.open(remote_url) as src:
            # Read strictly the windowed data
            data = src.read(1, window=window)
            src_crs = str(src.crs)
            src_nodata = src.nodata
            # Derived transform for the cropped window
            crop_transform = src.window_transform(window)
            crop_bounds = rasterio.windows.bounds(window, src.transform)

        dtype_str = str(data.dtype)

        # Write cropped GeoTIFF with DEFLATE compression
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": data.dtype,
            "crs": src_crs,
            "transform": crop_transform,
            "nodata": src_nodata,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }

        with rasterio.open(target_tif, "w", **profile) as dst:
            dst.write(data, 1)

        elapsed = time.perf_counter() - t0
        file_size = target_tif.stat().st_size
        total_scene_bytes += file_size
        sha256 = compute_file_sha256(target_tif)

        # Compute raster statistics
        stats = compute_band_statistics(data, src_nodata)

        bands_metadata[band_id] = {
            "band_identity": band_id,
            "pixel_resolution_m": res,
            "local_filename": f"{band_id}.tif",
            "file_size_bytes": file_size,
            "local_sha256": sha256,
            "crs": src_crs,
            "affine_transform": [
                crop_transform.a,
                crop_transform.b,
                crop_transform.c,
                crop_transform.d,
                crop_transform.e,
                crop_transform.f,
            ],
            "dimensions": {"height": height, "width": width},
            "bounds_utm43n": [crop_bounds[0], crop_bounds[1], crop_bounds[2], crop_bounds[3]],
            "dtype": dtype_str,
            "nodata": src_nodata,
            "source_asset_key": band_id,
            "source_uri": remote_url,
            "download_latency_seconds": round(elapsed, 2),
            "statistics": stats,
        }
        print(f"    -> Saved {target_tif.name} ({file_size / (1024*1024):.2f} MB, {elapsed:.1f}s, SHA256={sha256[:8]}...)")

    # Assemble provenance and complete scene metadata
    provenance_doc = {
        "tier1_copernicus_source": {
            "authority": "European Space Agency (ESA) Copernicus Programme",
            "constellation": "Sentinel-2",
            "spacecraft": spec["spacecraft"],
            "sensor": spec["sensor"],
            "product_level": spec["processing_level"],
            "mgrs_tile": spec["mgrs_tile"],
            "orbit_note": "Different spacecraft acquisitions sharing MGRS tile 43QBB grid footprint",
            "acquisition_datetime_utc": spec["acquisition_datetime"],
            "cloud_cover_tile_pct": spec["cloud_cover_tile_pct"],
            "sun_elevation_deg": spec["sun_elevation"],
            "sun_azimuth_deg": spec["sun_azimuth"],
            "license": "Copernicus Open Access Policy",
        },
        "tier2_element84_aws_cog_distribution": {
            "curator": "Element84 AWS Open Data",
            "stac_item_url": spec["stac_item_url"],
            "format": "Cloud-Optimized GeoTIFF (COG) with HTTP Range support",
        },
        "tier3_local_aoi_crop_artifact": {
            "generator": "scripts/ingest_sentinel2_sample.py",
            "aoi_name": "Navi Mumbai / JNPT",
            "requested_bbox_wgs84": [72.95, 18.90, 73.10, 19.05],
            "projected_crs": "EPSG:32643",
            "temporal_baseline": "Approximately 23.5 months / 716 days",
            "creation_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }

    scene_metadata = {
        "scene_metadata": {
            "scene_id": scene_id,
            "spacecraft": spec["spacecraft"],
            "sensor": spec["sensor"],
            "processing_level": spec["processing_level"],
            "mgrs_tile": spec["mgrs_tile"],
            "acquisition_datetime": spec["acquisition_datetime"],
            "cloud_cover_tile_pct": spec["cloud_cover_tile_pct"],
            "sun_elevation": spec["sun_elevation"],
            "sun_azimuth": spec["sun_azimuth"],
            "total_local_size_bytes": total_scene_bytes,
        },
        "provenance": provenance_doc,
        "bands": bands_metadata,
    }

    # Write metadata.json
    meta_path = scene_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(scene_metadata, f, indent=2)

    meta_sha256 = compute_file_sha256(meta_path)
    print(f"  -> Generated metadata.json ({meta_path.stat().st_size} bytes, SHA256={meta_sha256[:8]}...)")

    return scene_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest windowed Sentinel-2 L2A crops for M1 Stage 1.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "data" / "raw" / "sentinel2",
        help="Destination directory for ingested scenes",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("M1 STAGE 1: SENTINEL-2 LOCAL-FIRST WINDOWED INGESTION")
    print(f"Target Directory: {output_dir}")
    print("=================================================================")

    t_start = time.perf_counter()
    grand_total_bytes = 0

    for scene_id, spec in SCENE_SPECS.items():
        meta = ingest_scene_bands(scene_id, spec, output_dir)
        grand_total_bytes += meta["scene_metadata"]["total_local_size_bytes"]

    total_time = time.perf_counter() - t_start

    print("\n=================================================================")
    print("INGESTION COMPLETE — VALIDATING SCENES VIA API")
    print("=================================================================")

    for scene_id in SCENE_SPECS.keys():
        scene_dir = output_dir / scene_id
        scene, val_report = load_sentinel2_scene(scene_dir)

        print(f"\nValidation Report for {scene_id} ({scene.spacecraft}):")
        print(f"  Status: {val_report['status']}")
        for bid, stats in val_report["band_statistics"].items():
            k10 = f">10k: {stats['count_above_10000']} ({stats['pct_above_10000']}%)" if "count_above_10000" in stats else ""
            print(f"  Band {bid:<4}: min={stats['min']:<6.1f} p50={stats['p50']:<6.1f} p99={stats['p99']:<6.1f} max={stats['max']:<7.1f} {k10}")

        scl = val_report.get("scl_quality_report")
        if scl:
            diag = scl["diagnostic_summary"]
            print("  SCL Diagnostic Summary:")
            print(f"    Cloud pixels:      {diag['cloud_pixels_pct']}%")
            print(f"    Cloud shadow:      {diag['cloud_shadow_pct']}%")
            print(f"    Water:             {diag['water_pct']}%")
            print(f"    Vegetation:        {diag['vegetation_pct']}%")
            print(f"    Bare soil:         {diag['bare_soil_pct']}%")

    print("\n=================================================================")
    print(f"TOTAL MEASURED LOCAL DISK FOOTPRINT: {grand_total_bytes / (1024*1024):.2f} MB")
    print(f"TOTAL EXECUTION TIME:               {total_time:.1f} s")
    print("=================================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
