"""UI presentation module for FLUX Satellite Semantic Search and Change Analysis."""

from src.ui.change_evidence import render_change_evidence_panel
from src.ui.flowchart import generate_m1_m9_flowchart_html
from src.ui.gauges import (
    render_change_ratio_gauge,
    render_cloud_cover_gauge,
    render_confidence_gauge,
    render_numeric_gauge,
    render_similarity_gauge,
    render_spectral_distance_gauge,
    render_suppression_ratio_gauge,
    render_valid_ratio_gauge,
)
from src.ui.grid_visualization import (
    get_tile_neighborhood_data,
    parse_tile_row_col,
    render_spatial_tile_grid_html,
)
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
    "generate_m1_m9_flowchart_html",
    "render_spatial_tile_grid_html",
    "get_tile_neighborhood_data",
    "parse_tile_row_col",
    "render_change_evidence_panel",
    "render_numeric_gauge",
    "render_similarity_gauge",
    "render_confidence_gauge",
    "render_cloud_cover_gauge",
    "render_valid_ratio_gauge",
    "render_change_ratio_gauge",
    "render_suppression_ratio_gauge",
    "render_spectral_distance_gauge",
]
