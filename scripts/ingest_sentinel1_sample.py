"""Windowed Sentinel-1 Level-1 GRD Sample Ingestion Script for M1 Stage 2.

CRITICAL ACQUISITION RULES:
    1. Preserves Sentinel-1 crop strictly in its NATIVE RADAR / RANGE-AZIMUTH pixel grid.
    2. Zero reprojection, warping, resampling, orthorectification, or TPS during ingestion.
    3. Stream ONLY the verified safe window (with explicit 50-pixel safety margin) over HTTP Range.
    4. Preserve native DN values (uint16), source raster dimensions, and source GCPs.
    5. Translate local GCPs relative to crop origin (col_off, row_off).
    6. Any locally derived affine is explicitly labeled:
       'derived_local_affine_approximation' (not authoritative source transform).
    7. Multi-tier provenance:
       Copernicus S1 L1 GRD -> AWS/Element84 COG -> HTTP Range window read -> Local native radar crop

Approved Scenes (Navi Mumbai / JNPT AOI):
    1. S1A_IW_GRDH_1SDV_20220123T010321_20220123T010346_041581_04F21E_41F0
       Window: col_off=6345, row_off=3770, width=1976, height=1999
    2. S1A_IW_GRDH_1SDV_20240113T010330_20240113T010355_052081_064B6D_A8B7
       Window: col_off=6334, row_off=4908, width=1976, height=1997
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.windows import Window

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.sentinel1 import (
    compute_file_sha256,
    compute_sar_band_statistics,
    derive_local_affine,
    load_sentinel1_scene,
    translate_gcps,
)

SCENE_SPECS = {
    "S1A_IW_GRDH_1SDV_20220123T010321_20220123T010346_041581_04F21E_41F0": {
        "platform": "Sentinel-1A",
        "sensor": "C-SAR",
        "frequency_ghz": 5.405,
        "instrument_mode": "IW",
        "product_type": "GRD",
        "resolution_type": "high",
        "polarizations": ["VV", "VH"],
        "orbit_direction": "descending",
        "relative_orbit": 34,
        "absolute_orbit": 41581,
        "acquisition_datetime": "2022-01-23T01:03:33.956041Z",
        "pixel_spacing_m": [10.0, 10.0],
        "effective_resolution_m": [20.0, 22.0],
        "stac_item_url": "https://earth-search.aws.element84.com/v1/collections/sentinel-1-grd/items/S1A_IW_GRDH_1SDV_20220123T010321_20220123T010346_041581_04F21E",
        "safe_window": {
            "col_off": 6345,
            "row_off": 3770,
            "width": 1976,
            "height": 1999,
            "safety_margin_pixels": 50,
            "safety_margin_metres": 500,
        },
        "assets": {
            "VV": "https://sentinel-s1-l1c.s3.eu-central-1.amazonaws.com/GRD/2022/1/23/IW/DV/S1A_IW_GRDH_1SDV_20220123T010321_20220123T010346_041581_04F21E_41F0/measurement/iw-vv.tiff",
            "VH": "https://sentinel-s1-l1c.s3.eu-central-1.amazonaws.com/GRD/2022/1/23/IW/DV/S1A_IW_GRDH_1SDV_20220123T010321_20220123T010346_041581_04F21E_41F0/measurement/iw-vh.tiff",
        },
    },
    "S1A_IW_GRDH_1SDV_20240113T010330_20240113T010355_052081_064B6D_A8B7": {
        "platform": "Sentinel-1A",
        "sensor": "C-SAR",
        "frequency_ghz": 5.405,
        "instrument_mode": "IW",
        "product_type": "GRD",
        "resolution_type": "high",
        "polarizations": ["VV", "VH"],
        "orbit_direction": "descending",
        "relative_orbit": 34,
        "absolute_orbit": 52081,
        "acquisition_datetime": "2024-01-13T01:03:43.218375Z",
        "pixel_spacing_m": [10.0, 10.0],
        "effective_resolution_m": [20.0, 22.0],
        "stac_item_url": "https://earth-search.aws.element84.com/v1/collections/sentinel-1-grd/items/S1A_IW_GRDH_1SDV_20240113T010330_20240113T010355_052081_064B6D",
        "safe_window": {
            "col_off": 6334,
            "row_off": 4908,
            "width": 1976,
            "height": 1997,
            "safety_margin_pixels": 50,
            "safety_margin_metres": 500,
        },
        "assets": {
            "VV": "https://sentinel-s1-l1c.s3.eu-central-1.amazonaws.com/GRD/2024/1/13/IW/DV/S1A_IW_GRDH_1SDV_20240113T010330_20240113T010355_052081_064B6D_A8B7/measurement/iw-vv.tiff",
            "VH": "https://sentinel-s1-l1c.s3.eu-central-1.amazonaws.com/GRD/2024/1/13/IW/DV/S1A_IW_GRDH_1SDV_20240113T010330_20240113T010355_052081_064B6D_A8B7/measurement/iw-vh.tiff",
        },
    },
}


def ingest_sar_scene(
    scene_id: str,
    spec: Dict[str, Any],
    output_base_dir: Path,
) -> Dict[str, Any]:
    """Perform windowed HTTP Range reads for Sentinel-1 VV/VH and store native radar crops."""
    scene_dir = output_base_dir / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    win = spec["safe_window"]
    col_off = win["col_off"]
    row_off = win["row_off"]
    width = win["width"]
    height = win["height"]

    print(f"\n========================================================")
    print(f"Ingesting SAR Scene: {scene_id}")
    print(f"Platform:            {spec['platform']} ({spec['instrument_mode']} {spec['product_type']})")
    print(f"Acquisition:         {spec['acquisition_datetime']}")
    print(f"Orbit:               {spec['orbit_direction']}, Relative Orbit {spec['relative_orbit']}")
    print(f"Safe Window:         col_off={col_off}, row_off={row_off}, {width}x{height} (margin={win['safety_margin_pixels']}px)")
    print(f"Destination:         {scene_dir}")
    print(f"========================================================")

    bands_metadata: Dict[str, Any] = {}
    total_scene_bytes = 0
    source_gcp_list = None
    source_raster_dims = None
    gcp_crs_str = "EPSG:4326"

    window = Window(col_off=col_off, row_off=row_off, width=width, height=height)

    for pol, remote_url in spec["assets"].items():
        target_tif = scene_dir / f"{pol.lower()}.tif"
        print(f"  Streaming safe window for {pol:<3} (native radar grid, {width}x{height})...")
        t0 = time.perf_counter()

        with rasterio.open(remote_url) as src:
            data = src.read(1, window=window)
            src_nodata = src.nodata
            if source_gcp_list is None:
                source_gcps_raw, gcp_crs = src.gcps
                source_gcp_list = source_gcps_raw
                gcp_crs_str = str(gcp_crs) if gcp_crs else "EPSG:4326"
                source_raster_dims = {"height": src.height, "width": src.width}

        elapsed = time.perf_counter() - t0

        # Translate GCPs relative to crop origin
        local_gcps_dicts = translate_gcps(source_gcp_list, col_off, row_off, width, height, margin=1200)
        local_gcp_objects = [
            GroundControlPoint(
                row=g["local_row"],
                col=g["local_col"],
                x=g["lon"],
                y=g["lat"],
                z=g["elevation_m"],
                id=g["id"],
            )
            for g in local_gcps_dicts
        ]

        # Write local GeoTIFF preserving native radar geometry and embedding local GCPs
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": data.dtype,
            "nodata": src_nodata,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }

        with rasterio.open(target_tif, "w", **profile) as dst:
            dst.write(data, 1)
            # Embed local GCPs into GeoTIFF metadata
            if local_gcp_objects:
                dst.gcps = (local_gcp_objects, gcp_crs_str)

        file_size = target_tif.stat().st_size
        total_scene_bytes += file_size
        sha256 = compute_file_sha256(target_tif)

        # Compute statistics
        stats = compute_sar_band_statistics(data, src_nodata)

        bands_metadata[pol] = {
            "polarization": pol,
            "local_filename": f"{pol.lower()}.tif",
            "file_size_bytes": file_size,
            "local_sha256": sha256,
            "dimensions": {"height": height, "width": width},
            "dtype": str(data.dtype),
            "nodata": src_nodata,
            "source_asset_key": pol,
            "source_uri": remote_url,
            "download_latency_seconds": round(elapsed, 2),
            "statistics": stats,
        }
        print(f"    -> Saved {target_tif.name} ({file_size / (1024*1024):.2f} MB, {elapsed:.1f}s, SHA256={sha256[:8]}...)")

    # Derive local affine approximation strictly labeled as approximation
    local_gcps_dicts = translate_gcps(source_gcp_list, col_off, row_off, width, height, margin=1200)
    derived_affine_doc = derive_local_affine(local_gcps_dicts)

    provenance_doc = {
        "tier1_copernicus_source": {
            "authority": "European Space Agency (ESA) Copernicus Programme",
            "constellation": "Sentinel-1",
            "spacecraft": spec["platform"],
            "sensor": spec["sensor"],
            "instrument_mode": spec["instrument_mode"],
            "product_type": spec["product_type"],
            "resolution_type": spec["resolution_type"],
            "orbit_direction": spec["orbit_direction"],
            "relative_orbit": spec["relative_orbit"],
            "absolute_orbit": spec["absolute_orbit"],
            "acquisition_datetime_utc": spec["acquisition_datetime"],
            "license": "Copernicus Open Access Policy",
        },
        "tier2_aws_cog_distribution": {
            "curator": "AWS Open Data Registry / Element84",
            "stac_item_url": spec["stac_item_url"],
            "format": "Cloud-Optimized GeoTIFF (COG) with HTTP Range support",
        },
        "tier3_http_range_window_read": {
            "retrieval_protocol": "HTTP Range Partial Content (206)",
            "safety_margin_pixels": win["safety_margin_pixels"],
            "safety_margin_metres": win["safety_margin_metres"],
            "safety_rationale": "All four AOI corners and all 80 sampled AOI perimeter points were verified inside the crop with >500 m margin under the GCP-based geolocation model",
        },
        "tier4_local_native_radar_crop": {
            "generator": "scripts/ingest_sentinel1_sample.py",
            "aoi_name": "Navi Mumbai / JNPT",
            "requested_bbox_wgs84": [72.95, 18.90, 73.10, 19.05],
            "coordinate_space": "native_radar_range_azimuth",
            "reprojected_or_resampled": False,
            "temporal_baseline": "Approximately 23.6 months / 720 days",
            "creation_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }

    scene_metadata = {
        "scene_metadata": {
            "scene_id": scene_id,
            "platform": spec["platform"],
            "sensor": spec["sensor"],
            "frequency_ghz": spec["frequency_ghz"],
            "instrument_mode": spec["instrument_mode"],
            "product_type": spec["product_type"],
            "resolution_type": spec["resolution_type"],
            "polarizations": spec["polarizations"],
            "orbit_direction": spec["orbit_direction"],
            "relative_orbit": spec["relative_orbit"],
            "absolute_orbit": spec["absolute_orbit"],
            "acquisition_datetime_utc": spec["acquisition_datetime"],
            "pixel_spacing_m": spec["pixel_spacing_m"],
            "effective_resolution_m": spec["effective_resolution_m"],
            "total_local_size_bytes": total_scene_bytes,
        },
        "source_raster_metadata": {
            "dimensions": source_raster_dims,
            "total_source_gcps": len(source_gcp_list) if source_gcp_list else 0,
            "gcp_crs": gcp_crs_str,
            "geolocation_model": "ESA PDGS Ground-Range Detected slant-to-ground GCP grid",
        },
        "crop_window": win,
        "local_gcps": local_gcps_dicts,
        "derived_local_affine_approximation": derived_affine_doc,
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
    parser = argparse.ArgumentParser(description="Ingest native windowed Sentinel-1 GRD crops for M1 Stage 2.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "data" / "raw" / "sentinel1",
        help="Destination directory for ingested Sentinel-1 scenes",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("M1 STAGE 2: SENTINEL-1 LOCAL-FIRST NATIVE RADAR INGESTION")
    print(f"Target Directory: {output_dir}")
    print("Geospatial Rule:  Native radar coordinate grid strictly preserved")
    print("=================================================================")

    t_start = time.perf_counter()
    grand_total_bytes = 0

    for scene_id, spec in SCENE_SPECS.items():
        meta = ingest_sar_scene(scene_id, spec, output_dir)
        grand_total_bytes += meta["scene_metadata"]["total_local_size_bytes"]

    total_time = time.perf_counter() - t_start

    print("\n=================================================================")
    print("INGESTION COMPLETE — VALIDATING SCENES VIA API")
    print("=================================================================")

    for scene_id in SCENE_SPECS.keys():
        scene_dir = output_dir / scene_id
        scene, val_report = load_sentinel1_scene(scene_dir)

        print(f"\nValidation Report for {scene_id} ({scene.platform}):")
        print(f"  Status: {val_report['status']}")
        for pol, stats in val_report["band_statistics"].items():
            print(f"  Polarization {pol:<3}: min={stats['min']:<6.1f} p50={stats['p50']:<6.1f} p99={stats['p99']:<6.1f} max={stats['max']:<7.1f}")
        gcp_sum = val_report.get("gcp_summary", {})
        print(f"  Local GCP Count: {gcp_sum.get('gcp_count')} in CRS {gcp_sum.get('source_gcp_crs')}")
        aff = gcp_sum.get("derived_local_affine_approximation", {})
        print(f"  Derived Affine RMSE: {aff.get('rmse_metres')} m ({aff.get('label')})")

    print("\n=================================================================")
    print(f"TOTAL MEASURED LOCAL DISK FOOTPRINT: {grand_total_bytes / (1024*1024):.2f} MB")
    print(f"TOTAL EXECUTION TIME:               {total_time:.1f} s")
    print("=================================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
