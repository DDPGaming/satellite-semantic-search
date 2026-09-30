"""Milestone 9: False-Alarm Suppression and Confidence Estimation module.

Provides:
- SuppressionConfig: Configuration dataclass for false-alarm suppression.
- SuppressedChangeResult: Standardized immutable result contract.
- FalseAlarmSuppressor: Post-processing suppression engine.
"""

from src.suppression.models import SuppressionConfig, SuppressedChangeResult
from src.suppression.suppressor import FalseAlarmSuppressor

__all__ = [
    "SuppressionConfig",
    "SuppressedChangeResult",
    "FalseAlarmSuppressor",
]
