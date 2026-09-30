"""Change Detection module for satellite semantic search (Milestone 8).

Exports:
- ChangeDetector: Core engine for primary raw change detection.
- ChangeDetectionConfig: Immutable configuration dataclass.
- ChangeResult: Standardized, immutable result contract.
"""

from src.change_detection.models import ChangeDetectionConfig, ChangeResult
from src.change_detection.detector import ChangeDetector

__all__ = [
    "ChangeDetector",
    "ChangeDetectionConfig",
    "ChangeResult",
]
