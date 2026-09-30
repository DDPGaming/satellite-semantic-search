"""Data models for Milestone 9: False-Alarm Suppression and Confidence.

Defines:
- SuppressionConfig: Immutable configuration for false-alarm suppression and confidence estimation.
- SuppressedChangeResult: Standardized immutable result contract preserving upstream M7/M8
  provenance, robust noise statistics, candidate/suppressed/confirmed pixel counts,
  and heuristic confidence metrics without serialized NumPy arrays.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class SuppressionConfig:
    """
    Configuration parameters for robust false-alarm suppression and confidence estimation.

    Parameters are explicit and validated upon initialization. No undocumented magic
    numbers or hard-coded assumptions are embedded.
    """

    sensitivity_k: float = 3.0
    min_threshold_offset: float = 0.0
    min_neighbors: int = 2
    min_region_area: int = 4
    min_valid_pixels: int = 100
    save_masks: bool = False
    output_dir: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters strictly against invalid types, NaN, inf, and out-of-range values."""
        import math
        from pathlib import Path

        # Validate sensitivity_k
        if isinstance(self.sensitivity_k, bool) or not isinstance(self.sensitivity_k, (int, float)):
            raise TypeError(f"sensitivity_k must be a real number, got {type(self.sensitivity_k).__name__}")
        if math.isnan(self.sensitivity_k) or math.isinf(self.sensitivity_k) or self.sensitivity_k < 0.0:
            raise ValueError(f"sensitivity_k must be a finite non-negative number, got {self.sensitivity_k}")

        # Validate min_threshold_offset
        if isinstance(self.min_threshold_offset, bool) or not isinstance(self.min_threshold_offset, (int, float)):
            raise TypeError(f"min_threshold_offset must be a real number, got {type(self.min_threshold_offset).__name__}")
        if math.isnan(self.min_threshold_offset) or math.isinf(self.min_threshold_offset) or self.min_threshold_offset < 0.0:
            raise ValueError(f"min_threshold_offset must be a finite non-negative number, got {self.min_threshold_offset}")

        # Validate min_neighbors
        if isinstance(self.min_neighbors, bool) or not isinstance(self.min_neighbors, int):
            raise TypeError(f"min_neighbors must be an integer, got {type(self.min_neighbors).__name__}")
        if not (0 <= self.min_neighbors <= 8):
            raise ValueError(f"min_neighbors must be an integer between 0 and 8 inclusive, got {self.min_neighbors}")

        # Validate min_region_area
        if isinstance(self.min_region_area, bool) or not isinstance(self.min_region_area, int):
            raise TypeError(f"min_region_area must be an integer, got {type(self.min_region_area).__name__}")
        if self.min_region_area < 1:
            raise ValueError(f"min_region_area must be >= 1, got {self.min_region_area}")

        # Validate min_valid_pixels
        if isinstance(self.min_valid_pixels, bool) or not isinstance(self.min_valid_pixels, int):
            raise TypeError(f"min_valid_pixels must be an integer, got {type(self.min_valid_pixels).__name__}")
        if self.min_valid_pixels < 1:
            raise ValueError(f"min_valid_pixels must be >= 1, got {self.min_valid_pixels}")

        # Validate save_masks
        if not isinstance(self.save_masks, bool):
            raise TypeError(f"save_masks must be a boolean, got {type(self.save_masks).__name__}")

        # Validate output_dir
        if self.output_dir is not None and not isinstance(self.output_dir, (str, Path)):
            raise TypeError(f"output_dir must be a string, Path, or None, got {type(self.output_dir).__name__}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to JSON-serializable dictionary."""
        return {
            "sensitivity_k": self.sensitivity_k,
            "min_threshold_offset": self.min_threshold_offset,
            "min_neighbors": self.min_neighbors,
            "min_region_area": self.min_region_area,
            "min_valid_pixels": self.min_valid_pixels,
            "save_masks": self.save_masks,
            "output_dir": self.output_dir,
        }


@dataclass(frozen=True)
class SuppressedChangeResult:
    """
    Standardized, immutable result contract for post-processed change detection.

    Carries robust noise statistics, candidate/suppressed/confirmed pixel counts,
    heuristic confidence metrics, spatial output references, processing status,
    and complete M7/M8 provenance without baking in empirical classification claims.
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
    status: str  # e.g., "success", "unsupported_sar_unaligned", "no_valid_pixels", etc.
    method: str  # e.g., "adaptive_mad_suppression", "unsupported"

    # Robust noise and thresholding statistics
    threshold_used: Optional[float] = None
    noise_median: Optional[float] = None
    noise_mad: Optional[float] = None

    # Spatial pixel counts
    candidate_pixels_count: int = 0
    suppressed_pixels_count: int = 0
    confirmed_pixels_count: int = 0
    confirmed_change_ratio: float = 0.0

    # Heuristic confidence metrics over confirmed change pixels
    mean_confidence_on_change: Optional[float] = None
    max_confidence: Optional[float] = None

    # Optional raster artifact paths on disk
    mask_raster_path: Optional[str] = None
    confidence_raster_path: Optional[str] = None

    # In-memory arrays (omitted from comparison, repr, and JSON serialization)
    confirmed_mask: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    confidence_map: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

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
            "threshold_used": self.threshold_used,
            "noise_median": self.noise_median,
            "noise_mad": self.noise_mad,
            "candidate_pixels_count": self.candidate_pixels_count,
            "suppressed_pixels_count": self.suppressed_pixels_count,
            "confirmed_pixels_count": self.confirmed_pixels_count,
            "confirmed_change_ratio": round(self.confirmed_change_ratio, 6),
            "mean_confidence_on_change": (
                round(self.mean_confidence_on_change, 6)
                if self.mean_confidence_on_change is not None
                else None
            ),
            "max_confidence": (
                round(self.max_confidence, 6)
                if self.max_confidence is not None
                else None
            ),
            "mask_raster_path": self.mask_raster_path,
            "confidence_raster_path": self.confidence_raster_path,
            "warnings": list(self.warnings),
            "provenance": self.provenance,
        }
