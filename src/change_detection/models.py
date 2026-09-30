"""Data models for Milestone 8: Primary Raw Change Detection.

Defines:
- ChangeDetectionConfig: Immutable configuration for change detection operations.
- ChangeResult: Immutable result contract preserving full provenance, continuous
  change magnitude statistics, spatial references, and execution status.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class ChangeDetectionConfig:
    """
    Configuration parameters for raw change detection.

    All fields are explicit and documented. No magic thresholds or hard-coded
    dataset constraints are embedded.
    """

    method: str = "spectral_distance_l2"
    bands: Tuple[str, ...] = ("B02", "B03", "B04", "B08")
    reflectance_scale: float = 10000.0  # Sentinel-2 L2A surface reflectance scale factor
    mask_nodata: bool = True  # Mask zero / nodata values
    mask_clouds: bool = True  # Mask clouds and shadows via SCL when available
    minimum_valid_pixel_ratio: float = 0.10  # Minimum valid coverage required for statistics
    save_change_map: bool = False  # Whether to materialize GeoTIFF raster to disk
    output_dir: Optional[str] = None  # Custom output directory for saved change rasters

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to JSON-serializable dictionary."""
        return {
            "method": self.method,
            "bands": list(self.bands),
            "reflectance_scale": self.reflectance_scale,
            "mask_nodata": self.mask_nodata,
            "mask_clouds": self.mask_clouds,
            "minimum_valid_pixel_ratio": self.minimum_valid_pixel_ratio,
            "save_change_map": self.save_change_map,
            "output_dir": self.output_dir,
        }


@dataclass(frozen=True)
class ChangeResult:
    """
    Standardized, immutable result contract for primary raw change detection.

    Carries continuous change statistics, spatial alignment references, processing
    status, and complete provenance without baking in empirical classification thresholds.
    """

    pair_id: str
    modality: str
    reference_tile_id: str
    comparison_tile_id: str
    reference_scene_id: str
    comparison_scene_id: str
    reference_datetime_utc: str
    comparison_datetime_utc: str
    temporal_delta_days: float
    grid_index: Tuple[int, int]

    # Execution status and method
    status: str  # e.g., "success", "unsupported_sar_unaligned", "unaligned_optical_grid", etc.
    method: str  # e.g., "spectral_distance_l2", "unsupported"

    # Statistical summary over valid pixels only
    summary_statistics: Optional[Dict[str, float]] = None

    # Spatial output reference
    change_map_path: Optional[str] = None

    # In-memory raster (omitted from comparison, repr, and JSON serialization)
    change_magnitude: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    # Diagnostics and complete provenance
    warnings: Tuple[str, ...] = ()
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to standardized, JSON-serializable dictionary.

        Guarantees stable key ordering and excludes in-memory numpy arrays.
        """
        return {
            "pair_id": self.pair_id,
            "modality": self.modality,
            "reference_tile_id": self.reference_tile_id,
            "comparison_tile_id": self.comparison_tile_id,
            "reference_scene_id": self.reference_scene_id,
            "comparison_scene_id": self.comparison_scene_id,
            "reference_datetime_utc": self.reference_datetime_utc,
            "comparison_datetime_utc": self.comparison_datetime_utc,
            "temporal_delta_days": round(self.temporal_delta_days, 5),
            "grid_index": {
                "row_idx": self.grid_index[0],
                "col_idx": self.grid_index[1],
            },
            "status": self.status,
            "method": self.method,
            "summary_statistics": self.summary_statistics,
            "change_map_path": self.change_map_path,
            "warnings": list(self.warnings),
            "provenance": self.provenance,
        }
