"""Data models for Multi-Temporal Pairing & Spatial Alignment.

Defines immutable data contracts representing:
- Spatial correspondence between two observations
- Pixel alignment status and raster coordinate verification
- Fully resolved TemporalPair data structure
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SpatialCorrespondence:
    """Geospatial correspondence metadata between two temporal observations."""

    method: str  # "grid_index_exact" or "grid_index_topological"
    grid_index: Tuple[int, int]  # (row_idx, col_idx)
    spatial_iou_wgs84: float  # Quantitative diagnostic IoU
    bounds_wgs84: List[float]  # [min_lon, min_lat, max_lon, max_lat]


@dataclass(frozen=True)
class AlignmentStatus:
    """Raster pixel-grid alignment status."""

    is_pixel_aligned: bool  # True for optical; False for unaligned radar
    alignment_type: str  # "native_pixel_aligned" or "unaligned_radar_grid"
    crs: Optional[str]  # "EPSG:32643" for optical; None for unprojected SAR
    pixel_dimensions: Tuple[int, int]  # (height, width)


@dataclass(frozen=True)
class TemporalPair:
    """Immutable multi-temporal observation pair over the same geographic location."""

    pair_id: str  # "pair__{ref_tile_id}__{comp_tile_id}"
    modality: str  # "optical" or "sar"
    reference_tile_id: str  # Earlier observation
    comparison_tile_id: str  # Later observation
    reference_scene_id: str
    comparison_scene_id: str
    reference_datetime_utc: str  # ISO-8601 UTC timestamp
    comparison_datetime_utc: str  # ISO-8601 UTC timestamp
    temporal_delta_days: float  # Strictly positive float (comparison - reference)
    spatial: SpatialCorrespondence
    alignment: AlignmentStatus
    reference_tile_dir: str  # Relative path to reference tile directory
    comparison_tile_dir: str  # Relative path to comparison tile directory

    @property
    def is_pixel_aligned(self) -> bool:
        """Return True if corresponding rasters share an identical affine pixel grid."""
        return self.alignment.is_pixel_aligned

    @property
    def alignment_type(self) -> str:
        """Return the qualitative alignment classification."""
        return self.alignment.alignment_type

    @property
    def grid_index(self) -> Tuple[int, int]:
        """Return the shared grid index (row_idx, col_idx)."""
        return self.spatial.grid_index

    @property
    def spatial_iou_wgs84(self) -> float:
        """Return the spatial intersection-over-union diagnostic."""
        return self.spatial.spatial_iou_wgs84

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to the standardized, machine-readable M7 Result Contract.

        Guarantees stable key ordering and exact schema compliance for downstream
        consumption by M8 (Change Detection) and the post-M8 Proof-of-Concept.
        """
        return {
            "pair_id": self.pair_id,
            "modality": self.modality,
            "temporal": {
                "reference_datetime_utc": self.reference_datetime_utc,
                "comparison_datetime_utc": self.comparison_datetime_utc,
                "temporal_delta_days": round(self.temporal_delta_days, 5),
                "chronological_order": True,
            },
            "spatial_correspondence": {
                "method": self.spatial.method,
                "grid_index": {
                    "row_idx": self.spatial.grid_index[0],
                    "col_idx": self.spatial.grid_index[1],
                },
                "spatial_iou_wgs84": round(self.spatial.spatial_iou_wgs84, 6),
                "bounds_wgs84": list(self.spatial.bounds_wgs84),
            },
            "alignment": {
                "is_pixel_aligned": self.alignment.is_pixel_aligned,
                "alignment_type": self.alignment.alignment_type,
                "crs": self.alignment.crs,
                "pixel_dimensions": {
                    "height": self.alignment.pixel_dimensions[0],
                    "width": self.alignment.pixel_dimensions[1],
                },
            },
            "provenance": {
                "reference_tile_id": self.reference_tile_id,
                "comparison_tile_id": self.comparison_tile_id,
                "reference_scene_id": self.reference_scene_id,
                "comparison_scene_id": self.comparison_scene_id,
                "reference_tile_directory": self.reference_tile_dir,
                "comparison_tile_directory": self.comparison_tile_dir,
            },
        }
