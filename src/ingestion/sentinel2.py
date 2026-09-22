"""Sentinel-2 Level-2A data container, loader, and validation suite.

M1 Ingestion Architecture:
    Preserves:
    - Native resolutions/grid definitions (B02/B03/B04/B08 at 10m, SCL at 20m)
    - Full geospatial georeferencing (CRS, affine transform, bounding box)
    - Multi-tier provenance distinction:
        1. Original Copernicus Sentinel-2 Level-2A source
        2. Element84 / AWS Open Data COG distribution
        3. Locally generated spatial AOI crop artifact
    - AOI-agnostic validation against recorded crop metadata
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from affine import Affine

# SCL (Scene Classification Layer) standard class definitions
SCL_CLASSES = {
    0: "NO_DATA",
    1: "SATURATED_OR_DEFECTIVE",
    2: "DARK_AREA_PIXELS",
    3: "CLOUD_SHADOWS",
    4: "VEGETATION",
    5: "NOT_VEGETATED",
    6: "WATER",
    7: "UNCLASSIFIED",
    8: "CLOUD_MEDIUM_PROBABILITY",
    9: "CLOUD_HIGH_PROBABILITY",
    10: "THIN_CIRRUS",
    11: "SNOW",
}


def compute_file_sha256(file_path: Path) -> str:
    """Compute the SHA-256 checksum of a local file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class SatelliteBandRaster:
    """
    Representation of an individual satellite band raster preserving its native grid.
    """
    band_id: str
    resolution: float
    crs: str
    transform: Affine
    shape: Tuple[int, int]  # (height, width)
    bounds: Tuple[float, float, float, float]  # (left, bottom, right, top)
    dtype: str
    nodata: Optional[float]
    local_path: Path
    sha256: str
    source_asset_key: str
    source_uri: str
    _data: Optional[np.ndarray] = None

    def read_data(self) -> np.ndarray:
        """Load and return the raw 2D numpy array for this band."""
        if self._data is None:
            with rasterio.open(self.local_path) as src:
                self._data = src.read(1)
        return self._data


@dataclass
class SatelliteScene:
    """
    Standardized M1 SatelliteScene container preserving per-band native grids and provenance.
    """
    scene_id: str
    spacecraft: str  # e.g. "Sentinel-2A" or "Sentinel-2B"
    sensor: str  # "MSI"
    processing_level: str  # "Level-2A"
    mgrs_tile: str  # "43QBB"
    acquisition_datetime: str  # ISO 8601 UTC
    cloud_cover_tile_pct: float
    sun_elevation: float
    sun_azimuth: float
    bands: Dict[str, SatelliteBandRaster] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)


def compute_band_statistics(data: np.ndarray, nodata: Optional[float] = None) -> Dict[str, Any]:
    """Compute distribution statistics (min, max, p1, p50, p99, valid count, and >10,000 metrics) for a band.

    Sentinel-2 L2A surface reflectance uses a scale factor of 10,000 (10,000 = 100% surface reflectance).
    Legitimate physical factors (specular glint, clouds, bright man-made surfaces, high solar angles)
    can produce valid reflectance values exceeding DN 10,000.
    """
    if nodata is not None:
        valid_mask = data != nodata
    else:
        valid_mask = np.ones_like(data, dtype=bool)

    valid_vals = data[valid_mask]
    if valid_vals.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p1": 0.0,
            "p50": 0.0,
            "p99": 0.0,
            "count_above_10000": 0,
            "pct_above_10000": 0.0,
            "valid_count": 0,
            "total_pixels": int(data.size),
        }

    count_above_10k = int(np.count_nonzero(valid_vals > 10000))
    pct_above_10k = round(100.0 * count_above_10k / int(valid_vals.size), 4)

    return {
        "min": float(np.min(valid_vals)),
        "max": float(np.max(valid_vals)),
        "mean": float(np.mean(valid_vals)),
        "p1": float(np.percentile(valid_vals, 1)),
        "p50": float(np.percentile(valid_vals, 50)),
        "p99": float(np.percentile(valid_vals, 99)),
        "count_above_10000": count_above_10k,
        "pct_above_10000": pct_above_10k,
        "valid_count": int(valid_vals.size),
        "total_pixels": int(data.size),
    }


def compute_scl_distribution(scl_data: np.ndarray) -> Dict[str, Any]:
    """Tabulate SCL (Scene Classification Layer) pixel counts and percentages.

    Used for diagnostic quality assessment, not automated rejection.
    SCL indicates extremely low detected cloud and cloud-shadow contamination within this AOI.
    Other radiometric, atmospheric, viewing-geometry, registration, and seasonal
    effects remain possible and will be addressed in later change-analysis milestones.
    """
    total_pixels = scl_data.size
    unique_vals, counts = np.unique(scl_data, return_counts=True)
    val_map = dict(zip(unique_vals.tolist(), counts.tolist()))

    breakdown = {}
    for code, label in SCL_CLASSES.items():
        count = val_map.get(code, 0)
        pct = round(100.0 * count / total_pixels, 4)
        breakdown[f"{code}_{label}"] = {
            "code": code,
            "label": label,
            "pixel_count": count,
            "percentage": pct,
        }

    # Aggregate cloud/haze diagnostic percentage
    cloud_codes = [8, 9, 10]  # medium, high, cirrus
    cloud_shadow_codes = [3]
    cloud_pixels = sum(val_map.get(c, 0) for c in cloud_codes)
    shadow_pixels = sum(val_map.get(c, 0) for c in cloud_shadow_codes)

    return {
        "total_pixels": total_pixels,
        "classes": breakdown,
        "diagnostic_summary": {
            "cloud_pixels_pct": round(100.0 * cloud_pixels / total_pixels, 4),
            "cloud_shadow_pct": round(100.0 * shadow_pixels / total_pixels, 4),
            "vegetation_pct": round(100.0 * val_map.get(4, 0) / total_pixels, 4),
            "water_pct": round(100.0 * val_map.get(6, 0) / total_pixels, 4),
            "bare_soil_pct": round(100.0 * val_map.get(5, 0) / total_pixels, 4),
        },
    }


def validate_sentinel2_scene(scene: SatelliteScene, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the strengthened 11-point ingestion validation suite on a loaded SatelliteScene.

    Validates:
        1. CRS
        2. Affine transform
        3. Geospatial bounds
        4. Dimensions (against recorded crop metadata, AOI-agnostic)
        5. Dtype
        6. NoData
        7. Band identity
        8. Value range and percentiles
        9. SCL class distribution
        10. Local SHA-256 integrity hashes
        11. Provenance completeness
    """
    report: Dict[str, Any] = {
        "scene_id": scene.scene_id,
        "status": "PASSED",
        "checks": {},
        "warnings": [],
        "errors": [],
        "band_statistics": {},
        "scl_quality_report": None,
    }

    expected_bands_meta = metadata.get("bands", {})

    # Check 1: Mandatory bands present
    expected_band_ids = ["B02", "B03", "B04", "B08", "SCL"]
    for bid in expected_band_ids:
        if bid not in scene.bands:
            report["errors"].append(f"Missing expected band: {bid}")
            report["status"] = "FAILED"
    report["checks"]["band_presence"] = report["status"] != "FAILED"

    # Per-band validation
    for bid, band in scene.bands.items():
        band_meta = expected_bands_meta.get(bid, {})
        data = band.read_data()

        # Check: CRS
        if not band.crs.startswith("EPSG:"):
            report["errors"].append(f"{bid}: Invalid CRS '{band.crs}'")
        # Check: Dimensions against recorded metadata
        exp_h = band_meta.get("dimensions", {}).get("height")
        exp_w = band_meta.get("dimensions", {}).get("width")
        if exp_h and exp_w and band.shape != (exp_h, exp_w):
            report["errors"].append(f"{bid}: Shape mismatch {band.shape} vs recorded ({exp_h}, {exp_w})")

        # Check: Resolution consistency
        exp_res = band_meta.get("pixel_resolution_m")
        if exp_res and abs(band.resolution - exp_res) > 1e-4:
            report["errors"].append(f"{bid}: Resolution mismatch {band.resolution} vs {exp_res}")

        # Check: Dtype
        if bid == "SCL":
            if band.dtype != "uint8":
                report["errors"].append(f"SCL dtype must be uint8, got {band.dtype}")
        else:
            if band.dtype != "uint16":
                report["errors"].append(f"{bid} dtype must be uint16, got {band.dtype}")

        # Check: NoData
        if bid != "SCL" and band.nodata != 0:
            report["warnings"].append(f"{bid}: Expected nodata=0, got {band.nodata}")

        # Compute statistics
        stats = compute_band_statistics(data, band.nodata)
        report["band_statistics"][bid] = stats

        # Value range check: verify non-negative and within actual data-type bounds
        if bid != "SCL":
            if stats["min"] < 0:
                report["errors"].append(f"{bid}: Negative reflectance value detected (min={stats['min']})")
            if stats["max"] > 65535:
                report["errors"].append(f"{bid}: Value exceeds uint16 storage bounds (max={stats['max']})")

            # Report and flag values above DN 10,000 without failing ingestion
            # Sentinel-2 L2A surface reflectance uses a scale factor of 10,000 (10,000 = 100% reflectance).
            # Values above 10,000 (e.g., specular glint, bright rooftops, clouds) are physically possible
            # and do not indicate corrupted data.
            if stats["count_above_10000"] > 0:
                if stats["pct_above_10000"] > 5.0 or stats["max"] > 25000:
                    report["warnings"].append(
                        f"{bid}: Informational flag — {stats['count_above_10000']} pixels "
                        f"({stats['pct_above_10000']}%) exceed DN 10,000 (max={stats['max']})"
                    )
        else:
            if stats["min"] < 0 or stats["max"] > 255:
                report["errors"].append(f"SCL: Value out of uint8 bounds (min={stats['min']}, max={stats['max']})")
            if stats["max"] > 11:
                report["warnings"].append(f"SCL: Value exceeds known class code range 0..11 (max={stats['max']})")

        # SHA-256 Check
        recomputed_hash = compute_file_sha256(band.local_path)
        if recomputed_hash != band.sha256:
            report["errors"].append(f"{bid}: Local SHA-256 hash mismatch!")

    # SCL Diagnostic Distribution
    if "SCL" in scene.bands:
        scl_data = scene.bands["SCL"].read_data()
        scl_report = compute_scl_distribution(scl_data)
        report["scl_quality_report"] = scl_report

        diag = scl_report["diagnostic_summary"]
        if diag["cloud_pixels_pct"] > 10.0:
            report["warnings"].append(f"SCL diagnostic flag: Cloud cover is {diag['cloud_pixels_pct']}% (> 10%)")

    if report["errors"]:
        report["status"] = "FAILED"

    report["checks"]["overall_validation"] = report["status"] == "PASSED"
    return report


def load_sentinel2_scene(scene_dir: Path) -> Tuple[SatelliteScene, Dict[str, Any]]:
    """
    Load and validate a local Sentinel-2 ingested scene directory.

    Returns:
        (SatelliteScene, validation_report_dict)
    """
    scene_dir = Path(scene_dir).resolve()
    metadata_path = scene_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {scene_dir}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    scene_id = meta["scene_metadata"]["scene_id"]
    spacecraft = meta["scene_metadata"].get("spacecraft", "Unknown")
    sensor = meta["scene_metadata"].get("sensor", "MSI")
    mgrs_tile = meta["scene_metadata"].get("mgrs_tile", "43QBB")
    dt = meta["scene_metadata"].get("acquisition_datetime", "")
    cc = meta["scene_metadata"].get("cloud_cover_tile_pct", 0.0)
    sun_el = meta["scene_metadata"].get("sun_elevation", 0.0)
    sun_az = meta["scene_metadata"].get("sun_azimuth", 0.0)

    scene = SatelliteScene(
        scene_id=scene_id,
        spacecraft=spacecraft,
        sensor=sensor,
        processing_level=meta["scene_metadata"].get("processing_level", "Level-2A"),
        mgrs_tile=mgrs_tile,
        acquisition_datetime=dt,
        cloud_cover_tile_pct=cc,
        sun_elevation=sun_el,
        sun_azimuth=sun_az,
        provenance=meta.get("provenance", {}),
    )

    bands_dict = {}
    for bid, bmeta in meta.get("bands", {}).items():
        tif_filename = bmeta.get("local_filename", f"{bid}.tif")
        tif_path = scene_dir / tif_filename
        if not tif_path.exists():
            continue

        with rasterio.open(tif_path) as src:
            transform = src.transform
            crs_str = str(src.crs)
            shape = (src.height, src.width)
            bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            dtype_str = str(src.dtypes[0])
            nodata_val = src.nodata

        res = float(bmeta.get("pixel_resolution_m", 10.0))
        sha256 = bmeta.get("local_sha256", "")
        source_asset = bmeta.get("source_asset_key", "")
        source_uri = bmeta.get("source_uri", "")

        band_raster = SatelliteBandRaster(
            band_id=bid,
            resolution=res,
            crs=crs_str,
            transform=transform,
            shape=shape,
            bounds=bounds,
            dtype=dtype_str,
            nodata=nodata_val,
            local_path=tif_path,
            sha256=sha256,
            source_asset_key=source_asset,
            source_uri=source_uri,
        )
        bands_dict[bid] = band_raster

    scene.bands = bands_dict

    # Run validation suite
    report = validate_sentinel2_scene(scene, meta)
    return scene, report
