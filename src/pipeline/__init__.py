"""End-to-End PoC Pipeline module for Satellite Semantic Search and Change Analysis.

Provides a unified functional pipeline composing M5 through M9:
- M5: Semantic text retrieval over vector index
- M6: Structured metadata filtering
- M7: Multi-temporal tile pairing
- M8: Optical change detection
- M9: False-alarm suppression and heuristic confidence estimation
"""

from src.pipeline.poc_pipeline import (
    PipelineStageTiming,
    PoCPipeline,
    PoCPipelineResult,
)

__all__ = [
    "PipelineStageTiming",
    "PoCPipeline",
    "PoCPipelineResult",
]
