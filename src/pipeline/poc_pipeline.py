"""Reusable End-to-End Functional Pipeline Service Layer (M5 -> M6 -> M7 -> M8 -> M9).

Composes:
- M5: TextSearchEngine (natural-language query embedding and vector similarity retrieval)
- M6: Metadata filtering (modality, dates, scene, bbox)
- M7: TemporalPairer (counterpart tile resolution)
- M8: ChangeDetector (optical spectral distance magnitude differencing)
- M9: FalseAlarmSuppressor (adaptive MAD noise suppression and heuristic confidence)

This service layer provides the single canonical pipeline implementation for both
the end-to-end PoC benchmark and the subsequent barebones presentation UI,
preventing duplicate pipeline implementations.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.change_detection import ChangeDetectionConfig, ChangeDetector, ChangeResult
from src.pairing import TemporalPairer
from src.pairing.models import TemporalPair
from src.retrieval.search import TextSearchEngine
from src.suppression import (
    FalseAlarmSuppressor,
    SuppressedChangeResult,
    SuppressionConfig,
)


def get_default_project_root() -> Path:
    """Return project root based on repository structure."""
    return Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class PipelineStageTiming:
    """Per-stage wall-clock execution timings in seconds."""

    retrieval_s: float
    filtering_s: float
    pairing_s: float
    change_detection_s: float
    suppression_s: float
    total_s: float

    def to_dict(self) -> Dict[str, float]:
        """Convert timing to a JSON-serializable dictionary with rounded float values."""
        return {
            "retrieval_s": round(self.retrieval_s, 6),
            "filtering_s": round(self.filtering_s, 6),
            "pairing_s": round(self.pairing_s, 6),
            "change_detection_s": round(self.change_detection_s, 6),
            "suppression_s": round(self.suppression_s, 6),
            "total_s": round(self.total_s, 6),
        }


@dataclass(frozen=True)
class PoCPipelineResult:
    """
    Structured result contract for the complete end-to-end PoC pipeline.

    Preserves outputs and diagnostics across all stages (M5 through M9).
    """

    query: str
    status: str  # "success", "no_retrieval_hits", "no_filtered_hits", "no_temporal_pair", etc.
    retrieval_hits: List[Dict[str, Any]]
    filtered_hits: List[Dict[str, Any]]
    selected_tile_id: Optional[str] = None
    temporal_pair: Optional[TemporalPair] = None
    change_result: Optional[ChangeResult] = None
    suppressed_result: Optional[SuppressedChangeResult] = None
    timing: Optional[PipelineStageTiming] = None
    warnings: Tuple[str, ...] = ()

    @property
    def confirmed_pixels_count(self) -> int:
        """Return confirmed change pixel count from suppression stage."""
        return self.suppressed_result.confirmed_pixels_count if self.suppressed_result else 0

    @property
    def confirmed_change_ratio(self) -> float:
        """Return confirmed change ratio from suppression stage."""
        return self.suppressed_result.confirmed_change_ratio if self.suppressed_result else 0.0

    @property
    def mean_confidence(self) -> Optional[float]:
        """Return mean heuristic confidence over confirmed change pixels."""
        return self.suppressed_result.mean_confidence_on_change if self.suppressed_result else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""
        return {
            "query": self.query,
            "status": self.status,
            "selected_tile_id": self.selected_tile_id,
            "retrieval_hits_count": len(self.retrieval_hits),
            "retrieval_top_k": [
                {
                    "rank": i + 1,
                    "tile_id": h["tile_id"],
                    "similarity_score": round(float(h["similarity_score"]), 6),
                    "scene_id": h["scene_id"],
                    "modality": h["modality"],
                    "acquisition_datetime_utc": h.get("acquisition_datetime_utc"),
                }
                for i, h in enumerate(self.retrieval_hits[:5])
            ],
            "filtered_hits_count": len(self.filtered_hits),
            "filtered_top_k": [
                {
                    "rank": i + 1,
                    "tile_id": h["tile_id"],
                    "similarity_score": round(float(h["similarity_score"]), 6),
                    "scene_id": h["scene_id"],
                    "modality": h["modality"],
                    "acquisition_datetime_utc": h.get("acquisition_datetime_utc"),
                }
                for i, h in enumerate(self.filtered_hits[:5])
            ],
            "temporal_pair": self.temporal_pair.to_dict() if self.temporal_pair else None,
            "change_detection": self.change_result.to_dict() if self.change_result else None,
            "suppression": self.suppressed_result.to_dict() if self.suppressed_result else None,
            "timing": self.timing.to_dict() if self.timing else None,
            "warnings": list(self.warnings),
        }


class PoCPipeline:
    """
    End-to-End Functional PoC Pipeline.

    Orchestrates the complete flow from natural-language query to post-processed
    change detection and heuristic confidence estimation without duplicate code paths.
    """

    def __init__(
        self,
        project_root: Optional[Union[Path, str]] = None,
        search_engine: Optional[TextSearchEngine] = None,
        pairer: Optional[TemporalPairer] = None,
        detector: Optional[ChangeDetector] = None,
        suppressor: Optional[FalseAlarmSuppressor] = None,
        device: str = "cpu",
    ):
        """
        Initialize the PoCPipeline.

        Components can be injected directly (for testing or memory efficiency)
        or initialized once from staged artifacts.
        """
        self.project_root = Path(project_root).resolve() if project_root else get_default_project_root()

        self.search_engine = (
            search_engine
            if search_engine is not None
            else TextSearchEngine.from_staged_artifacts(project_root=self.project_root, device=device)
        )
        self.pairer = (
            pairer
            if pairer is not None
            else TemporalPairer(project_root=self.project_root)
        )
        self.detector = (
            detector
            if detector is not None
            else ChangeDetector(project_root=self.project_root)
        )
        self.suppressor = (
            suppressor
            if suppressor is not None
            else FalseAlarmSuppressor(project_root=self.project_root)
        )

    def run(
        self,
        query: str,
        top_k: int = 5,
        modality: Optional[str] = "optical",
        date_from: Optional[Union[str, date, datetime]] = None,
        date_to: Optional[Union[str, date, datetime]] = None,
        scene_id: Optional[str] = None,
        bbox: Optional[Sequence[float]] = None,
        change_config: Optional[ChangeDetectionConfig] = None,
        suppression_config: Optional[SuppressionConfig] = None,
    ) -> PoCPipelineResult:
        """
        Execute the complete functional pipeline for a natural language query.

        Args:
            query: Natural language query string.
            top_k: Maximum number of candidate tiles to retrieve.
            modality: Metadata filter for modality (default 'optical' for change analysis).
            date_from: Optional start UTC date/datetime filter.
            date_to: Optional end UTC date/datetime filter.
            scene_id: Optional exact scene filter.
            bbox: Optional WGS84 bounding box sequence [min_lon, min_lat, max_lon, max_lat].
            change_config: Optional M8 ChangeDetectionConfig.
            suppression_config: Optional M9 SuppressionConfig.

        Returns:
            PoCPipelineResult carrying outputs and diagnostics from all stages.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query string must be a non-empty string.")

        t_start = time.perf_counter()

        # ---------------------------------------------------------
        # Stage 1: M5 Semantic Retrieval
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        query_emb = self.search_engine.encode_query(query)
        raw_hits = self.search_engine.vector_index.search(query_emb, top_k=top_k)
        raw_ranked = sorted(raw_hits, key=lambda h: (-h["similarity_score"], h["tile_id"]))
        t1 = time.perf_counter()
        retrieval_s = t1 - t0

        if not raw_ranked:
            total_s = time.perf_counter() - t_start
            timing = PipelineStageTiming(
                retrieval_s=retrieval_s,
                filtering_s=0.0,
                pairing_s=0.0,
                change_detection_s=0.0,
                suppression_s=0.0,
                total_s=total_s,
            )
            return PoCPipelineResult(
                query=query,
                status="no_retrieval_hits",
                retrieval_hits=[],
                filtered_hits=[],
                timing=timing,
                warnings=("No vectors found in search index.",),
            )

        # ---------------------------------------------------------
        # Stage 2: M6 Metadata Filtering
        # ---------------------------------------------------------
        t1_filter = time.perf_counter()
        filtered_hits = self.search_engine.search(
            query=query_emb,
            top_k=top_k,
            modality=modality,
            date_from=date_from,
            date_to=date_to,
            scene_id=scene_id,
            bbox=bbox,
        )
        t2 = time.perf_counter()
        filtering_s = t2 - t1_filter

        if not filtered_hits:
            total_s = time.perf_counter() - t_start
            timing = PipelineStageTiming(
                retrieval_s=retrieval_s,
                filtering_s=filtering_s,
                pairing_s=0.0,
                change_detection_s=0.0,
                suppression_s=0.0,
                total_s=total_s,
            )
            return PoCPipelineResult(
                query=query,
                status="no_filtered_hits",
                retrieval_hits=raw_ranked,
                filtered_hits=[],
                timing=timing,
                warnings=("No tiles matched the metadata filter constraints.",),
            )

        selected_tile = filtered_hits[0]
        selected_tile_id = selected_tile["tile_id"]

        # ---------------------------------------------------------
        # Stage 3: M7 Temporal Pairing
        # ---------------------------------------------------------
        t2_pair = time.perf_counter()
        pair = self.pairer.get_pair_for_tile(selected_tile_id)
        t3 = time.perf_counter()
        pairing_s = t3 - t2_pair

        if pair is None:
            total_s = time.perf_counter() - t_start
            timing = PipelineStageTiming(
                retrieval_s=retrieval_s,
                filtering_s=filtering_s,
                pairing_s=pairing_s,
                change_detection_s=0.0,
                suppression_s=0.0,
                total_s=total_s,
            )
            return PoCPipelineResult(
                query=query,
                status="no_temporal_pair",
                retrieval_hits=raw_ranked,
                filtered_hits=filtered_hits,
                selected_tile_id=selected_tile_id,
                timing=timing,
                warnings=(f"No temporal counterpart resolved for tile '{selected_tile_id}'.",),
            )

        # ---------------------------------------------------------
        # Stage 4: M8 Optical Change Detection
        # ---------------------------------------------------------
        t3_change = time.perf_counter()
        change_result = self.detector.detect_change(pair, config=change_config)
        t4 = time.perf_counter()
        change_detection_s = t4 - t3_change

        # ---------------------------------------------------------
        # Stage 5: M9 False-Alarm Suppression
        # ---------------------------------------------------------
        t4_suppress = time.perf_counter()
        suppressed_result = self.suppressor.suppress(change_result, config=suppression_config)
        t5 = time.perf_counter()
        suppression_s = t5 - t4_suppress

        total_s = t5 - t_start

        timing = PipelineStageTiming(
            retrieval_s=retrieval_s,
            filtering_s=filtering_s,
            pairing_s=pairing_s,
            change_detection_s=change_detection_s,
            suppression_s=suppression_s,
            total_s=total_s,
        )

        return PoCPipelineResult(
            query=query,
            status=suppressed_result.status,
            retrieval_hits=raw_ranked,
            filtered_hits=filtered_hits,
            selected_tile_id=selected_tile_id,
            temporal_pair=pair,
            change_result=change_result,
            suppressed_result=suppressed_result,
            timing=timing,
            warnings=suppressed_result.warnings,
        )
