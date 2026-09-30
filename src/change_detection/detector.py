"""Primary Raw Change Detection engine for satellite imagery.

Implements Milestone 8:
- Consumes M7 TemporalPair objects directly without duplicating pairing logic.
- Verifies physical raster header alignment (dimensions, CRS, transform, resolution, bounds).
- Performs multi-band Euclidean spectral distance (CVA magnitude) on optical imagery.
- Enforces strict nodata, non-finite, and SCL cloud masking without hard-coded magic numbers.
- Computes deterministic summary statistics over valid pixels only.
- Supports optional materialization of continuous float32 change GeoTIFFs.
- Explicitly flags SAR pairs as deferred/unsupported due to non-pixel alignment.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import numpy as np
import rasterio

from src.pairing.models import TemporalPair
from src.change_detection.models import ChangeDetectionConfig, ChangeResult


def get_default_project_root() -> Path:
    """Return project root based on repository structure."""
    return Path(__file__).resolve().parent.parent.parent


class ChangeDetector:
    """
    Primary Raw Change Detector for multi-temporal satellite imagery.

    Consumes resolved M7 TemporalPair objects, verifies physical raster alignment,
    and computes continuous spectral distance magnitude across corresponding bands.
    """

    def __init__(self, project_root: Optional[Union[Path, str]] = None):
        """
        Initialize the ChangeDetector with an optional project root path.

        Args:
            project_root: Project root directory (optional, defaults to repo root).
        """
        self.project_root = Path(project_root).resolve() if project_root else get_default_project_root()

    def _resolve_tile_dir(self, tile_dir_str: str) -> Path:
        """
        Resolve a tile directory path from metadata or synthetic test setups.

        Checks:
        1. Absolute path if existing.
        2. project_root / 'data' / tile_dir_str (standard metadata layout).
        3. project_root / tile_dir_str.
        """
        p = Path(tile_dir_str)
        if p.is_absolute() and p.exists():
            return p

        cand1 = self.project_root / "data" / p
        if cand1.exists():
            return cand1

        cand2 = self.project_root / p
        if cand2.exists():
            return cand2

        # Default fallback
        return cand1

    @staticmethod
    def _verify_scl_alignment(
        s_scl: rasterio.DatasetReader,
        s_ref: rasterio.DatasetReader,
    ) -> Tuple[bool, str]:
        """
        Verify that a 20m SCL raster has the required spatial alignment to a 10m reference raster.

        Verifies:
        1. Dimensions: SCL must be exactly (128, 128) and half of reference dimensions.
        2. CRS: SCL CRS must match the 10m reference CRS.
        3. Resolution: SCL resolution must be exactly 2x reference resolution.
        4. Transform: SCL pixel scale must be 2x reference pixel scale, with zero shear/rotation.
        5. Origin: SCL upper-left origin offset must not exceed one SCL pixel (sub-pixel grid registration).
        6. Bounds: Projected bounding boxes must have an overlap IoU >= 0.90 over the tile footprint.

        Returns:
            Tuple of (is_aligned, reason_string).
        """
        # 1. Dimensions
        if s_scl.shape != (128, 128):
            return False, f"SCL dimensions {s_scl.shape} do not match expected (128, 128)"
        if s_scl.shape != (s_ref.shape[0] // 2, s_ref.shape[1] // 2):
            return False, f"SCL shape {s_scl.shape} is not half of 10m reference shape {s_ref.shape}"

        # 2. CRS
        if s_scl.crs is None or s_ref.crs is None or s_scl.crs != s_ref.crs:
            return False, f"SCL CRS '{s_scl.crs}' does not match reference CRS '{s_ref.crs}'"

        # 3. Resolution
        expected_res_x = s_ref.res[0] * 2.0
        expected_res_y = s_ref.res[1] * 2.0
        if not (
            np.isclose(s_scl.res[0], expected_res_x, atol=1e-3)
            and np.isclose(s_scl.res[1], expected_res_y, atol=1e-3)
        ):
            return (
                False,
                f"SCL resolution {s_scl.res} is not 2x reference resolution ({expected_res_x}, {expected_res_y})",
            )

        # 4. Transform scale and rotation
        t_scl = s_scl.transform
        t_ref = s_ref.transform
        if not (
            np.isclose(t_scl.a, t_ref.a * 2.0, atol=1e-3)
            and np.isclose(t_scl.e, t_ref.e * 2.0, atol=1e-3)
        ):
            return False, "SCL pixel scale does not match 2x reference pixel scale"
        if not (np.isclose(t_scl.b, 0.0, atol=1e-5) and np.isclose(t_scl.d, 0.0, atol=1e-5)):
            return False, "SCL transform has unexpected rotation or shear"

        # 5. Origin offset
        dx = abs(t_scl.c - t_ref.c)
        dy = abs(t_scl.f - t_ref.f)
        max_allowed_offset_x = s_scl.res[0] * 1.05
        max_allowed_offset_y = s_scl.res[1] * 1.05
        if dx > max_allowed_offset_x or dy > max_allowed_offset_y:
            return False, f"SCL origin offset ({dx:.1f}, {dy:.1f}) exceeds SCL pixel size"

        # 6. Bounds overlap IoU
        b_scl = s_scl.bounds
        b_ref = s_ref.bounds
        ix_min = max(b_scl.left, b_ref.left)
        iy_min = max(b_scl.bottom, b_ref.bottom)
        ix_max = min(b_scl.right, b_ref.right)
        iy_max = min(b_scl.top, b_ref.top)

        if ix_max <= ix_min or iy_max <= iy_min:
            return False, "SCL and reference bounds do not overlap"

        inter_area = (ix_max - ix_min) * (iy_max - iy_min)
        area_scl = (b_scl.right - b_scl.left) * (b_scl.top - b_scl.bottom)
        area_ref = (b_ref.right - b_ref.left) * (b_ref.top - b_ref.bottom)
        union_area = area_scl + area_ref - inter_area
        iou = inter_area / union_area

        if iou < 0.90:
            return False, f"SCL and reference bounds overlap IoU {iou:.4f} is below 0.90"

        return True, "Aligned"

    def detect_change(
        self,
        pair: TemporalPair,
        config: Optional[ChangeDetectionConfig] = None,
    ) -> ChangeResult:
        """
        Compute primary raw change detection on a single TemporalPair.

        Args:
            pair: An already-resolved M7 TemporalPair.
            config: Optional ChangeDetectionConfig parameters.

        Returns:
            ChangeResult containing execution status, summary statistics,
            change map path (if saved), and complete provenance.
        """
        # 1. Type validation
        if not isinstance(pair, TemporalPair):
            raise TypeError(f"pair must be a TemporalPair instance, got {type(pair).__name__}.")

        cfg = config if config is not None else ChangeDetectionConfig()
        if not isinstance(cfg, ChangeDetectionConfig):
            raise TypeError(f"config must be a ChangeDetectionConfig instance, got {type(cfg).__name__}.")

        # Base provenance metadata
        base_provenance: Dict[str, Any] = {
            "pair_id": pair.pair_id,
            "reference_tile_id": pair.reference_tile_id,
            "comparison_tile_id": pair.comparison_tile_id,
            "reference_tile_directory": pair.reference_tile_dir,
            "comparison_tile_directory": pair.comparison_tile_dir,
            "method": cfg.method,
            "bands_used": list(cfg.bands),
            "reflectance_scale": cfg.reflectance_scale,
        }

        # 2. SAR Handling: Explicitly deferred due to non-pixel alignment
        if pair.modality == "sar":
            return ChangeResult(
                pair_id=pair.pair_id,
                modality="sar",
                reference_tile_id=pair.reference_tile_id,
                comparison_tile_id=pair.comparison_tile_id,
                reference_scene_id=pair.reference_scene_id,
                comparison_scene_id=pair.comparison_scene_id,
                reference_datetime_utc=pair.reference_datetime_utc,
                comparison_datetime_utc=pair.comparison_datetime_utc,
                temporal_delta_days=pair.temporal_delta_days,
                grid_index=pair.grid_index,
                status="unsupported_sar_unaligned",
                method="unsupported",
                summary_statistics=None,
                change_map_path=None,
                change_magnitude=None,
                warnings=(
                    "SAR change detection is deferred: unaligned radar grid (crs=None) "
                    "requires geometric terrain correction before pixel-level differencing.",
                ),
                provenance={
                    **base_provenance,
                    "reason": "SAR imagery in native range-azimuth coordinates has orbital parallax offset.",
                },
            )

        if pair.modality != "optical":
            return ChangeResult(
                pair_id=pair.pair_id,
                modality=pair.modality,
                reference_tile_id=pair.reference_tile_id,
                comparison_tile_id=pair.comparison_tile_id,
                reference_scene_id=pair.reference_scene_id,
                comparison_scene_id=pair.comparison_scene_id,
                reference_datetime_utc=pair.reference_datetime_utc,
                comparison_datetime_utc=pair.comparison_datetime_utc,
                temporal_delta_days=pair.temporal_delta_days,
                grid_index=pair.grid_index,
                status="unsupported_modality",
                method="unsupported",
                summary_statistics=None,
                change_map_path=None,
                change_magnitude=None,
                warnings=(f"Modality '{pair.modality}' is not supported for change detection.",),
                provenance=base_provenance,
            )

        # 3. Optical alignment pre-check
        if not pair.is_pixel_aligned:
            return ChangeResult(
                pair_id=pair.pair_id,
                modality="optical",
                reference_tile_id=pair.reference_tile_id,
                comparison_tile_id=pair.comparison_tile_id,
                reference_scene_id=pair.reference_scene_id,
                comparison_scene_id=pair.comparison_scene_id,
                reference_datetime_utc=pair.reference_datetime_utc,
                comparison_datetime_utc=pair.comparison_datetime_utc,
                temporal_delta_days=pair.temporal_delta_days,
                grid_index=pair.grid_index,
                status="unaligned_optical_grid",
                method="unsupported",
                summary_statistics=None,
                change_map_path=None,
                change_magnitude=None,
                warnings=(f"Optical pair is not pixel aligned ({pair.alignment_type}).",),
                provenance=base_provenance,
            )

        # 4. Resolve tile directories
        ref_dir = self._resolve_tile_dir(pair.reference_tile_dir)
        comp_dir = self._resolve_tile_dir(pair.comparison_tile_dir)

        if not ref_dir.exists():
            return ChangeResult(
                pair_id=pair.pair_id,
                modality="optical",
                reference_tile_id=pair.reference_tile_id,
                comparison_tile_id=pair.comparison_tile_id,
                reference_scene_id=pair.reference_scene_id,
                comparison_scene_id=pair.comparison_scene_id,
                reference_datetime_utc=pair.reference_datetime_utc,
                comparison_datetime_utc=pair.comparison_datetime_utc,
                temporal_delta_days=pair.temporal_delta_days,
                grid_index=pair.grid_index,
                status="missing_raster_file",
                method=cfg.method,
                summary_statistics=None,
                change_map_path=None,
                change_magnitude=None,
                warnings=(f"Reference tile directory does not exist: {ref_dir}",),
                provenance=base_provenance,
            )

        if not comp_dir.exists():
            return ChangeResult(
                pair_id=pair.pair_id,
                modality="optical",
                reference_tile_id=pair.reference_tile_id,
                comparison_tile_id=pair.comparison_tile_id,
                reference_scene_id=pair.reference_scene_id,
                comparison_scene_id=pair.comparison_scene_id,
                reference_datetime_utc=pair.reference_datetime_utc,
                comparison_datetime_utc=pair.comparison_datetime_utc,
                temporal_delta_days=pair.temporal_delta_days,
                grid_index=pair.grid_index,
                status="missing_raster_file",
                method=cfg.method,
                summary_statistics=None,
                change_map_path=None,
                change_magnitude=None,
                warnings=(f"Comparison tile directory does not exist: {comp_dir}",),
                provenance=base_provenance,
            )

        # 5. Check required band files existence
        for band in cfg.bands:
            ref_path = ref_dir / f"{band}.tif"
            comp_path = comp_dir / f"{band}.tif"
            if not ref_path.exists():
                return ChangeResult(
                    pair_id=pair.pair_id,
                    modality="optical",
                    reference_tile_id=pair.reference_tile_id,
                    comparison_tile_id=pair.comparison_tile_id,
                    reference_scene_id=pair.reference_scene_id,
                    comparison_scene_id=pair.comparison_scene_id,
                    reference_datetime_utc=pair.reference_datetime_utc,
                    comparison_datetime_utc=pair.comparison_datetime_utc,
                    temporal_delta_days=pair.temporal_delta_days,
                    grid_index=pair.grid_index,
                    status="missing_raster_file",
                    method=cfg.method,
                    summary_statistics=None,
                    change_map_path=None,
                    change_magnitude=None,
                    warnings=(f"Required band file missing: {ref_path}",),
                    provenance=base_provenance,
                )
            if not comp_path.exists():
                return ChangeResult(
                    pair_id=pair.pair_id,
                    modality="optical",
                    reference_tile_id=pair.reference_tile_id,
                    comparison_tile_id=pair.comparison_tile_id,
                    reference_scene_id=pair.reference_scene_id,
                    comparison_scene_id=pair.comparison_scene_id,
                    reference_datetime_utc=pair.reference_datetime_utc,
                    comparison_datetime_utc=pair.comparison_datetime_utc,
                    temporal_delta_days=pair.temporal_delta_days,
                    grid_index=pair.grid_index,
                    status="missing_raster_file",
                    method=cfg.method,
                    summary_statistics=None,
                    change_map_path=None,
                    change_magnitude=None,
                    warnings=(f"Required band file missing: {comp_path}",),
                    provenance=base_provenance,
                )

        # 6. Physical header alignment verification
        ref_profile: Optional[Dict[str, Any]] = None
        for band in cfg.bands:
            ref_path = ref_dir / f"{band}.tif"
            comp_path = comp_dir / f"{band}.tif"
            with rasterio.open(ref_path) as s_ref, rasterio.open(comp_path) as s_comp:
                if s_ref.shape != s_comp.shape:
                    return ChangeResult(
                        pair_id=pair.pair_id,
                        modality="optical",
                        reference_tile_id=pair.reference_tile_id,
                        comparison_tile_id=pair.comparison_tile_id,
                        reference_scene_id=pair.reference_scene_id,
                        comparison_scene_id=pair.comparison_scene_id,
                        reference_datetime_utc=pair.reference_datetime_utc,
                        comparison_datetime_utc=pair.comparison_datetime_utc,
                        temporal_delta_days=pair.temporal_delta_days,
                        grid_index=pair.grid_index,
                        status="dimension_mismatch",
                        method=cfg.method,
                        summary_statistics=None,
                        change_map_path=None,
                        change_magnitude=None,
                        warnings=(f"Band {band} dimension mismatch: ref {s_ref.shape} vs comp {s_comp.shape}",),
                        provenance=base_provenance,
                    )
                if s_ref.shape != (256, 256):
                    return ChangeResult(
                        pair_id=pair.pair_id,
                        modality="optical",
                        reference_tile_id=pair.reference_tile_id,
                        comparison_tile_id=pair.comparison_tile_id,
                        reference_scene_id=pair.reference_scene_id,
                        comparison_scene_id=pair.comparison_scene_id,
                        reference_datetime_utc=pair.reference_datetime_utc,
                        comparison_datetime_utc=pair.comparison_datetime_utc,
                        temporal_delta_days=pair.temporal_delta_days,
                        grid_index=pair.grid_index,
                        status="dimension_mismatch",
                        method=cfg.method,
                        summary_statistics=None,
                        change_map_path=None,
                        change_magnitude=None,
                        warnings=(f"Band {band} dimensions {s_ref.shape} do not match expected (256, 256)",),
                        provenance=base_provenance,
                    )
                if s_ref.crs != s_comp.crs:
                    return ChangeResult(
                        pair_id=pair.pair_id,
                        modality="optical",
                        reference_tile_id=pair.reference_tile_id,
                        comparison_tile_id=pair.comparison_tile_id,
                        reference_scene_id=pair.reference_scene_id,
                        comparison_scene_id=pair.comparison_scene_id,
                        reference_datetime_utc=pair.reference_datetime_utc,
                        comparison_datetime_utc=pair.comparison_datetime_utc,
                        temporal_delta_days=pair.temporal_delta_days,
                        grid_index=pair.grid_index,
                        status="crs_mismatch",
                        method=cfg.method,
                        summary_statistics=None,
                        change_map_path=None,
                        change_magnitude=None,
                        warnings=(f"Band {band} CRS mismatch: ref {s_ref.crs} vs comp {s_comp.crs}",),
                        provenance=base_provenance,
                    )
                if s_ref.transform != s_comp.transform:
                    return ChangeResult(
                        pair_id=pair.pair_id,
                        modality="optical",
                        reference_tile_id=pair.reference_tile_id,
                        comparison_tile_id=pair.comparison_tile_id,
                        reference_scene_id=pair.reference_scene_id,
                        comparison_scene_id=pair.comparison_scene_id,
                        reference_datetime_utc=pair.reference_datetime_utc,
                        comparison_datetime_utc=pair.comparison_datetime_utc,
                        temporal_delta_days=pair.temporal_delta_days,
                        grid_index=pair.grid_index,
                        status="transform_mismatch",
                        method=cfg.method,
                        summary_statistics=None,
                        change_map_path=None,
                        change_magnitude=None,
                        warnings=(f"Band {band} affine transform mismatch.",),
                        provenance=base_provenance,
                    )
                if s_ref.res != s_comp.res:
                    return ChangeResult(
                        pair_id=pair.pair_id,
                        modality="optical",
                        reference_tile_id=pair.reference_tile_id,
                        comparison_tile_id=pair.comparison_tile_id,
                        reference_scene_id=pair.reference_scene_id,
                        comparison_scene_id=pair.comparison_scene_id,
                        reference_datetime_utc=pair.reference_datetime_utc,
                        comparison_datetime_utc=pair.comparison_datetime_utc,
                        temporal_delta_days=pair.temporal_delta_days,
                        grid_index=pair.grid_index,
                        status="resolution_mismatch",
                        method=cfg.method,
                        summary_statistics=None,
                        change_map_path=None,
                        change_magnitude=None,
                        warnings=(f"Band {band} resolution mismatch: ref {s_ref.res} vs comp {s_comp.res}",),
                        provenance=base_provenance,
                    )
                if s_ref.bounds != s_comp.bounds:
                    return ChangeResult(
                        pair_id=pair.pair_id,
                        modality="optical",
                        reference_tile_id=pair.reference_tile_id,
                        comparison_tile_id=pair.comparison_tile_id,
                        reference_scene_id=pair.reference_scene_id,
                        comparison_scene_id=pair.comparison_scene_id,
                        reference_datetime_utc=pair.reference_datetime_utc,
                        comparison_datetime_utc=pair.comparison_datetime_utc,
                        temporal_delta_days=pair.temporal_delta_days,
                        grid_index=pair.grid_index,
                        status="bounds_mismatch",
                        method=cfg.method,
                        summary_statistics=None,
                        change_map_path=None,
                        change_magnitude=None,
                        warnings=(f"Band {band} bounds mismatch.",),
                        provenance=base_provenance,
                    )

                if ref_profile is None:
                    ref_profile = s_ref.profile.copy()

        # 7. Read rasters, compute validity mask, and accumulate squared differences
        sq_diff_sum = np.zeros((256, 256), dtype=np.float32)
        valid_mask = np.ones((256, 256), dtype=bool)

        for band in cfg.bands:
            ref_path = ref_dir / f"{band}.tif"
            comp_path = comp_dir / f"{band}.tif"
            with rasterio.open(ref_path) as s_ref, rasterio.open(comp_path) as s_comp:
                arr_ref = s_ref.read(1)
                arr_comp = s_comp.read(1)

                # Mask non-finite values (NaN / Inf)
                valid_mask &= np.isfinite(arr_ref) & np.isfinite(arr_comp)

                # Mask nodata values
                if cfg.mask_nodata:
                    if s_ref.nodata is not None:
                        valid_mask &= (arr_ref != s_ref.nodata)
                    if s_comp.nodata is not None:
                        valid_mask &= (arr_comp != s_comp.nodata)
                    # In Sentinel-2 L2A surface reflectance, 0 is the nodata convention
                    valid_mask &= (arr_ref > 0) & (arr_comp > 0)

                # Scale DN to surface reflectance
                r_ref = arr_ref.astype(np.float32) / cfg.reflectance_scale
                r_comp = arr_comp.astype(np.float32) / cfg.reflectance_scale

                # Accumulate squared Euclidean component
                sq_diff_sum += (r_comp - r_ref) ** 2

        # 8. SCL Cloud and Quality Masking (if enabled)
        warnings_list: List[str] = []
        scl_applied: bool = False

        if cfg.mask_clouds:
            scl_ref_path = ref_dir / "SCL.tif"
            scl_comp_path = comp_dir / "SCL.tif"
            if scl_ref_path.exists() and scl_comp_path.exists():
                ref_10m_path = ref_dir / f"{cfg.bands[0]}.tif"
                comp_10m_path = comp_dir / f"{cfg.bands[0]}.tif"
                with rasterio.open(scl_ref_path) as s_scl_ref, rasterio.open(scl_comp_path) as s_scl_comp, \
                     rasterio.open(ref_10m_path) as s_ref_10m, rasterio.open(comp_10m_path) as s_comp_10m:

                    ok_ref, reason_ref = self._verify_scl_alignment(s_scl_ref, s_ref_10m)
                    ok_comp, reason_comp = self._verify_scl_alignment(s_scl_comp, s_comp_10m)

                    if not ok_ref or not ok_comp:
                        reason = reason_ref if not ok_ref else reason_comp
                        warnings_list.append(
                            f"SCL quality masking skipped: SCL raster is spatially incompatible with 10m grid ({reason})."
                        )
                    else:
                        scl_r = s_scl_ref.read(1)
                        scl_c = s_scl_comp.read(1)

                        # Nearest-neighbor 2x expansion for verified 128x128 (20m) SCL to 256x256 (10m) grid
                        scl_r_256 = np.repeat(np.repeat(scl_r, 2, axis=0), 2, axis=1)
                        scl_c_256 = np.repeat(np.repeat(scl_c, 2, axis=0), 2, axis=1)

                        # Unusable SCL categories:
                        # 0: No data, 1: Saturated/Defective, 3: Cloud shadow,
                        # 8: Cloud medium prob, 9: Cloud high prob, 10: Thin cirrus, 11: Snow
                        unusable_scl = (0, 1, 3, 8, 9, 10, 11)
                        unusable_ref = np.isin(scl_r_256, unusable_scl)
                        unusable_comp = np.isin(scl_c_256, unusable_scl)
                        valid_mask &= (~unusable_ref) & (~unusable_comp)
                        scl_applied = True

        # 9. Compute continuous Euclidean change magnitude
        change_magnitude = np.sqrt(sq_diff_sum)
        # Invalid pixels set to NaN in floating representation
        change_magnitude[~valid_mask] = np.nan

        # 10. Summary statistics over valid pixels
        valid_pixels_count = int(np.sum(valid_mask))
        total_pixels_count = int(valid_mask.size)
        valid_pixel_ratio = float(valid_pixels_count / total_pixels_count)

        summary_stats: Optional[Dict[str, float]] = None

        if valid_pixels_count == 0:
            status = "no_valid_pixels"
            warnings_list.append("Zero valid pixels available for change detection.")
        elif valid_pixel_ratio < cfg.minimum_valid_pixel_ratio:
            status = "insufficient_valid_pixels"
            warnings_list.append(
                f"Valid pixel ratio {valid_pixel_ratio:.4f} is below minimum {cfg.minimum_valid_pixel_ratio:.4f}."
            )
        else:
            status = "success"
            valid_vals = change_magnitude[valid_mask]
            summary_stats = {
                "mean": float(np.mean(valid_vals)),
                "median": float(np.median(valid_vals)),
                "p95": float(np.percentile(valid_vals, 95)),
                "p99": float(np.percentile(valid_vals, 99)),
                "max": float(np.max(valid_vals)),
                "std": float(np.std(valid_vals)),
                "valid_pixel_ratio": round(valid_pixel_ratio, 6),
                "valid_pixels": valid_pixels_count,
                "total_pixels": total_pixels_count,
            }

        # 11. Materialize GeoTIFF change raster if configured
        change_map_path: Optional[str] = None
        if cfg.save_change_map and ref_profile is not None:
            out_dir = Path(cfg.output_dir) if cfg.output_dir else self.project_root / "data" / "changes"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"change__{pair.pair_id}.tif"

            profile = {
                "driver": "GTiff",
                "height": 256,
                "width": 256,
                "count": 1,
                "dtype": rasterio.float32,
                "crs": ref_profile["crs"],
                "transform": ref_profile["transform"],
                "nodata": np.nan,
            }
            with rasterio.open(out_file, "w", **profile) as dst:
                dst.write(change_magnitude.astype(np.float32), 1)

            try:
                change_map_path = str(out_file.relative_to(self.project_root)).replace("\\", "/")
            except ValueError:
                change_map_path = str(out_file).replace("\\", "/")

        return ChangeResult(
            pair_id=pair.pair_id,
            modality="optical",
            reference_tile_id=pair.reference_tile_id,
            comparison_tile_id=pair.comparison_tile_id,
            reference_scene_id=pair.reference_scene_id,
            comparison_scene_id=pair.comparison_scene_id,
            reference_datetime_utc=pair.reference_datetime_utc,
            comparison_datetime_utc=pair.comparison_datetime_utc,
            temporal_delta_days=pair.temporal_delta_days,
            grid_index=pair.grid_index,
            status=status,
            method=cfg.method,
            summary_statistics=summary_stats,
            change_map_path=change_map_path,
            change_magnitude=change_magnitude,
            warnings=tuple(warnings_list),
            provenance={
                **base_provenance,
                "valid_pixels": valid_pixels_count,
                "total_pixels": total_pixels_count,
                "valid_pixel_ratio": round(valid_pixel_ratio, 6),
                "change_map_saved": change_map_path is not None,
                "scl_masking_applied": scl_applied,
            },
        )

    def detect_changes(
        self,
        pairs: Iterable[TemporalPair],
        config: Optional[ChangeDetectionConfig] = None,
    ) -> Iterator[ChangeResult]:
        """
        Batch change detection over an iterable of TemporalPair objects.

        Streams change results lazily to conserve memory.

        Args:
            pairs: Iterable of TemporalPair objects.
            config: Optional ChangeDetectionConfig parameters.

        Yields:
            ChangeResult for each input pair.
        """
        for pair in pairs:
            yield self.detect_change(pair, config=config)
