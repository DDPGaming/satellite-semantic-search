"""UI presentation module for FLUX Satellite Semantic Search and Change Analysis."""

from src.ui.metadata_inspection import (
    extract_geographic_info,
    extract_tile_display_metadata,
    load_tile_authoritative_metadata,
    render_bounding_box_diagram,
)
from src.ui.rendering import (
    render_change_heatmap,
    render_confidence_heatmap,
    render_mask_image,
    render_tile_image,
)
from src.ui.report import (
    generate_milestone_architecture_html,
    generate_technical_report_html,
    generate_technical_report_markdown,
)

__all__ = [
    "render_tile_image",
    "render_change_heatmap",
    "render_mask_image",
    "render_confidence_heatmap",
    "load_tile_authoritative_metadata",
    "extract_geographic_info",
    "extract_tile_display_metadata",
    "render_bounding_box_diagram",
    "generate_milestone_architecture_html",
    "generate_technical_report_markdown",
    "generate_technical_report_html",
]
