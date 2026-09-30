"""False-Alarm Suppression and Confidence Estimation engine for satellite change detection.

Implements Milestone 9:
- Consumes M8 ChangeResult objects directly downstream without modifying M1-M8 components.
- Computes non-parametric robust noise statistics (median + MAD) on continuous change magnitudes.
- Establishes an adaptive candidate threshold using configurable scale multiplier k and offset.
- Performs deterministic 8-neighbor spatial count suppression and 8-connectivity minimum-area filtering.
- Generates a bounded, deterministic heuristic confidence map combining spectral margin and spatial density.
- Supports optional materialization of uint8 confirmed change masks and float32 confidence GeoTIFFs.
- Explicitly flags SAR pairs as deferred/unsupported and safely passes through upstream error statuses.
"""

from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import numpy as np
import rasterio

from src.change_detection.models import ChangeResult
from src.suppression.models import SuppressionConfig, SuppressedChangeResult


def get_default_project_root() -> Path:
    """Return project root based on repository structure."""
    return Path(__file__).resolve().parent.parent.parent


def count_8_neighbors(mask: np.ndarray) -> np.ndarray:
    """
    Count the number of 8-connected neighbors for each pixel in a 2D boolean mask.

    Uses constant zero-padding to deterministically handle corners, borders,
    and edges without periodic wrapping.

    Args:
        mask: 2D boolean numpy array.

    Returns:
        2D int32 array containing neighbor count in [0, 8] for each pixel.
    """
    pad = np.pad(mask.astype(np.int32), 1, mode="constant", constant_values=0)
    neighbor_count = (
        pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:]
        + pad[1:-1, :-2]                 + pad[1:-1, 2:]
        + pad[2:, :-2]  + pad[2:, 1:-1]  + pad[2:, 2:]
    )
    return neighbor_count


def label_connected_components_8(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Deterministically label connected components using 8-connectivity.

    Implements a fast pure Python/NumPy breadth-first search (BFS) queue.
    Avoids external heavy dependencies while ensuring exact 8-neighbor adjacency.

    Args:
        mask: 2D boolean numpy array.

    Returns:
        Tuple of (labeled_array, num_components) where labeled_array is int32.
    """
    h, w = mask.shape
    labeled = np.zeros((h, w), dtype=np.int32)
    if not np.any(mask):
        return labeled, 0

    rows, cols = np.nonzero(mask)
    visited = np.zeros((h, w), dtype=bool)
    offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    )

    current_label = 0
    queue = deque()

    for r, c in zip(rows, cols):
        if not visited[r, c]:
            current_label += 1
            visited[r, c] = True
            labeled[r, c] = current_label
            queue.append((r, c))

            while queue:
                cr, cc = queue.popleft()
                for dr, dc in offsets:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            labeled[nr, nc] = current_label
                            queue.append((nr, nc))

    return labeled, current_label


def filter_min_region_area_8(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    Filter a 2D boolean mask by removing 8-connected components smaller than min_area.

    Args:
        mask: 2D boolean numpy array.
        min_area: Minimum pixel area required for a component to be retained.

    Returns:
        2D boolean numpy array with small components removed.
    """
    if min_area <= 1:
        return mask.copy()

    labeled, num_components = label_connected_components_8(mask)
    if num_components == 0:
        return np.zeros_like(mask, dtype=bool)

    counts = np.bincount(labeled.ravel())
    # Exclude background label 0
    valid_labels = np.where(counts >= min_area)[0]
    valid_labels = valid_labels[valid_labels > 0]

    if valid_labels.size == 0:
        return np.zeros_like(mask, dtype=bool)

    return np.isin(labeled, valid_labels)


class FalseAlarmSuppressor:
    """
    Engine for post-processing raw change detection rasters to suppress false alarms.

    Consumes M8 ChangeResult objects, derives robust statistical candidate thresholds,
    applies spatial neighborhood and connected-component filtering, and computes
    heuristic confidence scores for confirmed change pixels.
    """

    def __init__(self, project_root: Optional[Union[Path, str]] = None):
        """
        Initialize the FalseAlarmSuppressor with an optional project root path.

        Args:
            project_root: Project root directory (optional, defaults to repo root).
        """
        self.project_root = Path(project_root).resolve() if project_root else get_default_project_root()

    def _resolve_spatial_profile(self, change_result: ChangeResult) -> Optional[Dict[str, Any]]:
        """
        Retrieve spatial reference profile (CRS, Affine transform, dimensions).

        Checks:
        1. Materialized M8 change_map_path if available.
        2. Reference tile rasters on disk under data/tiles.
        """
        if change_result.change_map_path:
            p = Path(change_result.change_map_path)
            if not p.is_absolute():
                p = self.project_root / p
            if p.exists():
                try:
                    with rasterio.open(p) as src:
                        return {
                            "crs": src.crs,
                            "transform": src.transform,
                            "height": src.height,
                            "width": src.width,
                        }
                except Exception:
                    pass

        ref_tile_id = change_result.reference_tile_id
        tiles_dir = self.project_root / "data" / "tiles"
        if tiles_dir.exists():
            for tif in tiles_dir.glob(f"**/{ref_tile_id}/*.tif"):
                try:
                    with rasterio.open(tif) as src:
                        if src.shape == (256, 256):
                            return {
                                "crs": src.crs,
                                "transform": src.transform,
                                "height": src.height,
                                "width": src.width,
                            }
                except Exception:
                    pass

        return None

    def suppress(
        self,
        change_result: ChangeResult,
        config: Optional[SuppressionConfig] = None,
    ) -> SuppressedChangeResult:
        """
        Apply false-alarm suppression and confidence estimation to an M8 ChangeResult.

        Args:
            change_result: M8 ChangeResult instance.
            config: Optional SuppressionConfig. If None, default configuration is used.

        Returns:
            SuppressedChangeResult instance.
        """
        cfg = config if config is not None else SuppressionConfig()

        base_provenance: Dict[str, Any] = {
            "m8_method": change_result.method,
            "m8_status": change_result.status,
            "suppression_method": "adaptive_mad_suppression",
            "sensitivity_k": cfg.sensitivity_k,
            "min_threshold_offset": cfg.min_threshold_offset,
            "min_neighbors": cfg.min_neighbors,
            "min_region_area": cfg.min_region_area,
            "min_valid_pixels": cfg.min_valid_pixels,
        }
        if change_result.summary_statistics:
            base_provenance["m8_summary_statistics"] = change_result.summary_statistics

        # 1. SAR Modality / Non-aligned handling
        if change_result.modality == "sar" or change_result.status == "unsupported_sar_unaligned":
            sar_warnings = list(change_result.warnings)
            sar_warnings.append(
                "SAR modality is unsupported for change detection and suppression due to lack of sub-pixel alignment."
            )
            return SuppressedChangeResult(
                pair_id=change_result.pair_id,
                modality="sar",
                reference_tile_id=change_result.reference_tile_id,
                comparison_tile_id=change_result.comparison_tile_id,
                reference_scene_id=change_result.reference_scene_id,
                comparison_scene_id=change_result.comparison_scene_id,
                reference_datetime_utc=change_result.reference_datetime_utc,
                comparison_datetime_utc=change_result.comparison_datetime_utc,
                temporal_delta_days=change_result.temporal_delta_days,
                grid_index=change_result.grid_index,
                status="unsupported_sar_unaligned",
                method="unsupported",
                threshold_used=None,
                noise_median=None,
                noise_mad=None,
                candidate_pixels_count=0,
                suppressed_pixels_count=0,
                confirmed_pixels_count=0,
                confirmed_change_ratio=0.0,
                mean_confidence_on_change=None,
                max_confidence=None,
                mask_raster_path=None,
                confidence_raster_path=None,
                confirmed_mask=None,
                confidence_map=None,
                warnings=tuple(sar_warnings),
                provenance={**base_provenance, "suppression_applied": False},
            )

        # 2. Check for upstream non-success status
        if change_result.status != "success":
            return SuppressedChangeResult(
                pair_id=change_result.pair_id,
                modality=change_result.modality,
                reference_tile_id=change_result.reference_tile_id,
                comparison_tile_id=change_result.comparison_tile_id,
                reference_scene_id=change_result.reference_scene_id,
                comparison_scene_id=change_result.comparison_scene_id,
                reference_datetime_utc=change_result.reference_datetime_utc,
                comparison_datetime_utc=change_result.comparison_datetime_utc,
                temporal_delta_days=change_result.temporal_delta_days,
                grid_index=change_result.grid_index,
                status=change_result.status,
                method="unsupported",
                threshold_used=None,
                noise_median=None,
                noise_mad=None,
                candidate_pixels_count=0,
                suppressed_pixels_count=0,
                confirmed_pixels_count=0,
                confirmed_change_ratio=0.0,
                mean_confidence_on_change=None,
                max_confidence=None,
                mask_raster_path=None,
                confidence_raster_path=None,
                confirmed_mask=None,
                confidence_map=None,
                warnings=change_result.warnings,
                provenance={**base_provenance, "suppression_applied": False},
            )

        # 3. Retrieve change magnitude array
        change_magnitude = change_result.change_magnitude
        if change_magnitude is None:
            if change_result.change_map_path:
                map_path = Path(change_result.change_map_path)
                if not map_path.is_absolute():
                    map_path = self.project_root / map_path
                if map_path.exists():
                    try:
                        with rasterio.open(map_path) as src:
                            change_magnitude = src.read(1).astype(np.float32)
                    except Exception as err:
                        return SuppressedChangeResult(
                            pair_id=change_result.pair_id,
                            modality=change_result.modality,
                            reference_tile_id=change_result.reference_tile_id,
                            comparison_tile_id=change_result.comparison_tile_id,
                            reference_scene_id=change_result.reference_scene_id,
                            comparison_scene_id=change_result.comparison_scene_id,
                            reference_datetime_utc=change_result.reference_datetime_utc,
                            comparison_datetime_utc=change_result.comparison_datetime_utc,
                            temporal_delta_days=change_result.temporal_delta_days,
                            grid_index=change_result.grid_index,
                            status="missing_raster_file",
                            method=change_result.method,
                            warnings=(f"Failed to read change map raster: {err}",),
                            provenance={**base_provenance, "suppression_applied": False},
                        )
                else:
                    return SuppressedChangeResult(
                        pair_id=change_result.pair_id,
                        modality=change_result.modality,
                        reference_tile_id=change_result.reference_tile_id,
                        comparison_tile_id=change_result.comparison_tile_id,
                        reference_scene_id=change_result.reference_scene_id,
                        comparison_scene_id=change_result.comparison_scene_id,
                        reference_datetime_utc=change_result.reference_datetime_utc,
                        comparison_datetime_utc=change_result.comparison_datetime_utc,
                        temporal_delta_days=change_result.temporal_delta_days,
                        grid_index=change_result.grid_index,
                        status="missing_raster_file",
                        method=change_result.method,
                        warnings=(f"Change map file does not exist: {map_path}",),
                        provenance={**base_provenance, "suppression_applied": False},
                    )
            else:
                return SuppressedChangeResult(
                    pair_id=change_result.pair_id,
                    modality=change_result.modality,
                    reference_tile_id=change_result.reference_tile_id,
                    comparison_tile_id=change_result.comparison_tile_id,
                    reference_scene_id=change_result.reference_scene_id,
                    comparison_scene_id=change_result.comparison_scene_id,
                    reference_datetime_utc=change_result.reference_datetime_utc,
                    comparison_datetime_utc=change_result.comparison_datetime_utc,
                    temporal_delta_days=change_result.temporal_delta_days,
                    grid_index=change_result.grid_index,
                    status="missing_change_data",
                    method=change_result.method,
                    warnings=("Neither in-memory change magnitude nor change_map_path is available.",),
                    provenance={**base_provenance, "suppression_applied": False},
                )

        # 4. Valid pixel extraction
        valid_mask = np.isfinite(change_magnitude) & (~np.isnan(change_magnitude))
        valid_pixels_count = int(np.sum(valid_mask))
        total_pixels_count = int(valid_mask.size)

        warnings_list = list(change_result.warnings)

        if valid_pixels_count == 0:
            warnings_list.append("Zero valid pixels available for false-alarm suppression.")
            return SuppressedChangeResult(
                pair_id=change_result.pair_id,
                modality=change_result.modality,
                reference_tile_id=change_result.reference_tile_id,
                comparison_tile_id=change_result.comparison_tile_id,
                reference_scene_id=change_result.reference_scene_id,
                comparison_scene_id=change_result.comparison_scene_id,
                reference_datetime_utc=change_result.reference_datetime_utc,
                comparison_datetime_utc=change_result.comparison_datetime_utc,
                temporal_delta_days=change_result.temporal_delta_days,
                grid_index=change_result.grid_index,
                status="no_valid_pixels",
                method="adaptive_mad_suppression",
                threshold_used=None,
                noise_median=None,
                noise_mad=None,
                candidate_pixels_count=0,
                suppressed_pixels_count=0,
                confirmed_pixels_count=0,
                confirmed_change_ratio=0.0,
                mean_confidence_on_change=None,
                max_confidence=None,
                mask_raster_path=None,
                confidence_raster_path=None,
                confirmed_mask=np.zeros_like(valid_mask, dtype=bool),
                confidence_map=np.zeros_like(change_magnitude, dtype=np.float32),
                warnings=tuple(warnings_list),
                provenance={
                    **base_provenance,
                    "valid_pixels_count": 0,
                    "total_pixels_count": total_pixels_count,
                    "suppression_applied": False,
                },
            )

        if valid_pixels_count < cfg.min_valid_pixels:
            warnings_list.append(
                f"Valid pixel count {valid_pixels_count} is below minimum {cfg.min_valid_pixels}."
            )
            return SuppressedChangeResult(
                pair_id=change_result.pair_id,
                modality=change_result.modality,
                reference_tile_id=change_result.reference_tile_id,
                comparison_tile_id=change_result.comparison_tile_id,
                reference_scene_id=change_result.reference_scene_id,
                comparison_scene_id=change_result.comparison_scene_id,
                reference_datetime_utc=change_result.reference_datetime_utc,
                comparison_datetime_utc=change_result.comparison_datetime_utc,
                temporal_delta_days=change_result.temporal_delta_days,
                grid_index=change_result.grid_index,
                status="insufficient_valid_pixels",
                method="adaptive_mad_suppression",
                threshold_used=None,
                noise_median=None,
                noise_mad=None,
                candidate_pixels_count=0,
                suppressed_pixels_count=0,
                confirmed_pixels_count=0,
                confirmed_change_ratio=0.0,
                mean_confidence_on_change=None,
                max_confidence=None,
                mask_raster_path=None,
                confidence_raster_path=None,
                confirmed_mask=np.zeros_like(valid_mask, dtype=bool),
                confidence_map=np.zeros_like(change_magnitude, dtype=np.float32),
                warnings=tuple(warnings_list),
                provenance={
                    **base_provenance,
                    "valid_pixels_count": valid_pixels_count,
                    "total_pixels_count": total_pixels_count,
                    "suppression_applied": False,
                },
            )

        # 5. Robust noise statistics and thresholding
        x_valid = change_magnitude[valid_mask]
        median_val = float(np.median(x_valid))
        abs_dev = np.abs(x_valid - median_val)
        mad_val = float(np.median(abs_dev))

        robust_scale = 1.4826 * mad_val
        threshold_val = median_val + max(cfg.sensitivity_k * robust_scale, cfg.min_threshold_offset)

        # Strict inequality on valid pixels
        candidate_mask = valid_mask & (change_magnitude > threshold_val)
        candidate_pixels_count = int(np.sum(candidate_mask))

        # 6. Spatial neighborhood suppression (8-connectivity)
        neighbor_count = count_8_neighbors(candidate_mask)
        neighborhood_pass = candidate_mask & (neighbor_count >= cfg.min_neighbors)

        # 7. Connected component minimum region area filtering (8-connectivity)
        confirmed_mask = filter_min_region_area_8(neighborhood_pass, cfg.min_region_area)
        confirmed_pixels_count = int(np.sum(confirmed_mask))
        suppressed_pixels_count = candidate_pixels_count - confirmed_pixels_count
        confirmed_change_ratio = float(confirmed_pixels_count / valid_pixels_count)

        # 8. Deterministic heuristic confidence estimation
        confidence_map = np.zeros_like(change_magnitude, dtype=np.float32)
        mean_conf: Optional[float] = None
        max_conf: Optional[float] = None

        if confirmed_pixels_count > 0:
            delta = change_magnitude[confirmed_mask]
            tau = threshold_val
            denom = delta + tau
            # Safe division: if denom == 0, spectral_margin is 0.0
            spectral_margin = np.where(denom > 0.0, (delta - tau) / denom, 0.0)
            spatial_density = neighbor_count[confirmed_mask] / 8.0

            raw_conf = 0.5 * spectral_margin + 0.5 * spatial_density
            clamped_conf = np.clip(raw_conf, 0.05, 1.0)
            confidence_map[confirmed_mask] = clamped_conf.astype(np.float32)

            mean_conf = float(np.mean(clamped_conf))
            max_conf = float(np.max(clamped_conf))

        # 9. GeoTIFF Materialization if configured
        mask_raster_path: Optional[str] = None
        conf_raster_path: Optional[str] = None

        if cfg.save_masks:
            try:
                out_dir = Path(cfg.output_dir) if cfg.output_dir else self.project_root / "data" / "changes"
                out_dir.mkdir(parents=True, exist_ok=True)

                mask_file = out_dir / f"mask__{change_result.pair_id}.tif"
                conf_file = out_dir / f"confidence__{change_result.pair_id}.tif"

                spatial_prof = self._resolve_spatial_profile(change_result)
                crs = spatial_prof["crs"] if spatial_prof else None
                transform = spatial_prof["transform"] if spatial_prof else rasterio.Affine.identity()
                height, width = change_magnitude.shape

                mask_profile = {
                    "driver": "GTiff",
                    "height": height,
                    "width": width,
                    "count": 1,
                    "dtype": rasterio.uint8,
                    "crs": crs,
                    "transform": transform,
                    "nodata": 0,
                }
                with rasterio.open(mask_file, "w", **mask_profile) as dst:
                    dst.write(confirmed_mask.astype(np.uint8), 1)

                conf_profile = {
                    "driver": "GTiff",
                    "height": height,
                    "width": width,
                    "count": 1,
                    "dtype": rasterio.float32,
                    "crs": crs,
                    "transform": transform,
                    "nodata": 0.0,
                }
                with rasterio.open(conf_file, "w", **conf_profile) as dst:
                    dst.write(confidence_map.astype(np.float32), 1)

                try:
                    mask_raster_path = str(mask_file.relative_to(self.project_root)).replace("\\", "/")
                except ValueError:
                    mask_raster_path = str(mask_file).replace("\\", "/")

                try:
                    conf_raster_path = str(conf_file.relative_to(self.project_root)).replace("\\", "/")
                except ValueError:
                    conf_raster_path = str(conf_file).replace("\\", "/")
            except Exception as io_err:
                warnings_list.append(f"GeoTIFF export failed: {io_err}")
                mask_raster_path = None
                conf_raster_path = None

        provenance = {
            **base_provenance,
            "threshold_used": round(threshold_val, 6),
            "noise_median": round(median_val, 6),
            "noise_mad": round(mad_val, 6),
            "valid_pixels_count": valid_pixels_count,
            "total_pixels_count": total_pixels_count,
            "candidate_pixels_count": candidate_pixels_count,
            "suppressed_pixels_count": suppressed_pixels_count,
            "confirmed_pixels_count": confirmed_pixels_count,
            "confirmed_change_ratio_valid": round(confirmed_change_ratio, 6),
            "confirmed_change_ratio_total": round(confirmed_pixels_count / total_pixels_count, 6),
            "masks_saved": mask_raster_path is not None,
            "suppression_applied": True,
        }

        return SuppressedChangeResult(
            pair_id=change_result.pair_id,
            modality=change_result.modality,
            reference_tile_id=change_result.reference_tile_id,
            comparison_tile_id=change_result.comparison_tile_id,
            reference_scene_id=change_result.reference_scene_id,
            comparison_scene_id=change_result.comparison_scene_id,
            reference_datetime_utc=change_result.reference_datetime_utc,
            comparison_datetime_utc=change_result.comparison_datetime_utc,
            temporal_delta_days=change_result.temporal_delta_days,
            grid_index=change_result.grid_index,
            status="success",
            method="adaptive_mad_suppression",
            threshold_used=round(threshold_val, 6),
            noise_median=round(median_val, 6),
            noise_mad=round(mad_val, 6),
            candidate_pixels_count=candidate_pixels_count,
            suppressed_pixels_count=suppressed_pixels_count,
            confirmed_pixels_count=confirmed_pixels_count,
            confirmed_change_ratio=round(confirmed_change_ratio, 6),
            mean_confidence_on_change=round(mean_conf, 6) if mean_conf is not None else None,
            max_confidence=round(max_conf, 6) if max_conf is not None else None,
            mask_raster_path=mask_raster_path,
            confidence_raster_path=conf_raster_path,
            confirmed_mask=confirmed_mask,
            confidence_map=confidence_map,
            warnings=tuple(warnings_list),
            provenance=provenance,
        )

    def suppress_batch(
        self,
        change_results: Iterable[ChangeResult],
        config: Optional[SuppressionConfig] = None,
    ) -> Iterator[SuppressedChangeResult]:
        """
        Deterministically process a batch of ChangeResult objects.

        Preserves input ordering and yields SuppressedChangeResult items one by one.

        Args:
            change_results: Iterable of M8 ChangeResult objects.
            config: Optional SuppressionConfig.

        Yields:
            SuppressedChangeResult instances in identical order to input.
        """
        for cr in change_results:
            yield self.suppress(cr, config=config)
