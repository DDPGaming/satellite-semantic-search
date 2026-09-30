"""Multi-Temporal Pairing & Spatial Alignment Module (Milestone 7).

Provides:
- TemporalPair: Immutable data contract representing corresponding observations
- SpatialCorrespondence: Spatial matching diagnostics and grid index
- AlignmentStatus: Certified pixel-grid alignment status
- TemporalPairer: Pairing engine resolving correspondences and canonical pairs
"""

from src.pairing.models import AlignmentStatus, SpatialCorrespondence, TemporalPair
from src.pairing.pairer import TemporalPairer, compute_wgs84_iou

__all__ = [
    "TemporalPairer",
    "TemporalPair",
    "SpatialCorrespondence",
    "AlignmentStatus",
    "compute_wgs84_iou",
]
