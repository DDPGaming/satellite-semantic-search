"""Numeric range visualization and gauge utilities for the FLUX presentation UI.

Provides helper functions to render bounded numeric metrics with compact horizontal
range bars, indicator pins, min/max domains, units, and clear distinction between
bounded physical/normalized metrics and unbounded quantities. Zero external dependencies.
"""

import html
from typing import Optional, Union


def render_numeric_gauge(
    label: str,
    value: Optional[Union[int, float]],
    min_val: Optional[Union[int, float]],
    max_val: Optional[Union[int, float]],
    unit: str = "",
    precision: int = 3,
    description: Optional[str] = None,
    color_accent: str = "cyan",
) -> str:
    """
    Render a horizontal range bar or domain badge for a numeric metric.

    Args:
        label: Human-readable metric title.
        value: Current metric value (or None if unavailable).
        min_val: Lower bound of the defined domain (or None if unbounded).
        max_val: Upper bound of the defined domain (or None if unbounded).
        unit: Unit of measurement (e.g. '%', 'deg', 'days', 'Δρ', or '').
        precision: Decimal display precision.
        description: Optional technical explanation of the metric domain.
        color_accent: Accent color theme ('cyan', 'emerald', 'amber', 'purple', 'blue').

    Returns:
        Safe HTML string containing the rendered gauge card.
    """
    label_safe = html.escape(label)
    unit_safe = html.escape(unit)
    desc_html = (
        f'<div style="font-size:11px; color:var(--text-muted); margin-top:4px;">{html.escape(description)}</div>'
        if description
        else ""
    )

    color_map = {
        "cyan": "var(--accent-cyan, #38bdf8)",
        "emerald": "var(--accent-emerald, #10b981)",
        "amber": "var(--accent-amber, #f59e0b)",
        "purple": "#a855f7",
        "blue": "var(--accent-blue, #3b82f6)",
    }
    accent_css = color_map.get(color_accent, "var(--accent-cyan, #38bdf8)")

    if value is None:
        return f"""
        <div style="background:var(--bg-main, #0c0f17); border:1px solid var(--border-color, #242c3d); border-radius:8px; padding:12px 14px; margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:12px; font-weight:600; color:var(--text-muted, #94a3b8);">{label_safe}</span>
                <span style="font-size:12px; font-weight:700; color:var(--text-muted, #94a3b8);">Not available</span>
            </div>
            {desc_html}
        </div>
        """

    # Format the current value
    sep = "" if unit_safe in ("%", "°", "deg", "") else " "
    val_str = f"{value:.{precision}f}{sep}{unit_safe}".strip()

    # Bounded metric case
    if min_val is not None and max_val is not None and max_val > min_val:
        pct = max(0.0, min(100.0, ((float(value) - float(min_val)) / (float(max_val) - float(min_val))) * 100.0))
        min_str = f"{min_val:.{precision}f}{sep}{unit_safe}".strip()
        max_str = f"{max_val:.{precision}f}{sep}{unit_safe}".strip()

        return f"""
        <div style="background:var(--bg-main, #0c0f17); border:1px solid var(--border-color, #242c3d); border-radius:8px; padding:12px 14px; margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-size:12px; font-weight:600; color:var(--text-muted, #94a3b8);">{label_safe}</span>
                <span style="font-size:14px; font-weight:700; color:{accent_css}; font-family:monospace;">{val_str}</span>
            </div>
            <div style="position:relative; background:#1e2638; height:8px; border-radius:4px; margin:8px 0 4px 0; overflow:hidden;">
                <div style="background:{accent_css}; width:{pct:.1f}%; height:100%; border-radius:4px; transition:width 0.3s ease;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--text-muted, #94a3b8); font-family:monospace;">
                <span>{min_str}</span>
                <span style="color:{accent_css}; font-weight:700;">&#9650; {pct:.1f}%</span>
                <span>{max_str}</span>
            </div>
            {desc_html}
        </div>
        """

    # Unbounded or single-sided metric case (e.g. >= 0)
    bound_note = (
        f"Physical Domain: &ge; {min_val} {unit_safe}"
        if min_val is not None
        else "Range: Unbounded / Not Mathematically Bounded"
    )

    return f"""
    <div style="background:var(--bg-main, #0c0f17); border:1px solid var(--border-color, #242c3d); border-radius:8px; padding:12px 14px; margin-bottom:10px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:12px; font-weight:600; color:var(--text-muted, #94a3b8);">{label_safe}</span>
            <span style="font-size:14px; font-weight:700; color:{accent_css}; font-family:monospace;">{val_str}</span>
        </div>
        <div style="font-size:10px; color:var(--text-muted, #94a3b8); font-family:monospace; margin-top:4px;">
            <span class="badge badge-m" style="padding:1px 6px; font-size:9px;">{bound_note}</span>
        </div>
        {desc_html}
    </div>
    """


def render_similarity_gauge(score: Optional[float]) -> str:
    """Render cosine similarity score gauge in domain [0.000, 1.000]."""
    return render_numeric_gauge(
        label="Cosine Similarity Score",
        value=score,
        min_val=0.0,
        max_val=1.0,
        unit="",
        precision=4,
        description="Inner product between L2-normalized CLIP text and visual tile embeddings [0.0, 1.0].",
        color_accent="cyan",
    )


def render_confidence_gauge(confidence: Optional[float]) -> str:
    """Render change confidence percentage gauge in domain [0%, 100%]."""
    val = confidence * 100.0 if confidence is not None else None
    return render_numeric_gauge(
        label="Mean Change Confidence",
        value=val,
        min_val=0.0,
        max_val=100.0,
        unit="%",
        precision=1,
        description="M9 statistical confidence over confirmed change pixels scaled beyond noise floor [0%, 100%].",
        color_accent="emerald",
    )


def render_cloud_cover_gauge(cloud_pct: Optional[float]) -> str:
    """Render scene cloud cover percentage gauge in domain [0%, 100%]."""
    return render_numeric_gauge(
        label="Scene Cloud Cover",
        value=cloud_pct,
        min_val=0.0,
        max_val=100.0,
        unit="%",
        precision=1,
        description="Authoritative scene metadata cloud obscuration percentage [0%, 100%].",
        color_accent="amber",
    )


def render_valid_ratio_gauge(valid_ratio: Optional[float]) -> str:
    """Render SCL valid pixel coverage gauge in domain [0%, 100%]."""
    val = valid_ratio * 100.0 if valid_ratio is not None else None
    return render_numeric_gauge(
        label="SCL Usable Pixel Coverage",
        value=val,
        min_val=0.0,
        max_val=100.0,
        unit="%",
        precision=1,
        description="Percentage of tile pixels free of cloud, shadow, cirrus, and defective sensors [0%, 100%].",
        color_accent="emerald",
    )


def render_change_ratio_gauge(change_ratio: Optional[float]) -> str:
    """Render confirmed change ratio gauge in domain [0%, 100%]."""
    val = change_ratio * 100.0 if change_ratio is not None else None
    return render_numeric_gauge(
        label="Confirmed Change Extent",
        value=val,
        min_val=0.0,
        max_val=100.0,
        unit="%",
        precision=2,
        description="Confirmed change pixels as a fraction of usable valid tile surface [0%, 100%].",
        color_accent="amber",
    )


def render_suppression_ratio_gauge(suppressed_pixels: int, candidate_pixels: int) -> str:
    """Render false-alarm suppression percentage gauge in domain [0%, 100%]."""
    pct = (suppressed_pixels / candidate_pixels * 100.0) if candidate_pixels > 0 else 0.0
    return render_numeric_gauge(
        label="False-Alarm Suppression Rate",
        value=pct,
        min_val=0.0,
        max_val=100.0,
        unit="%",
        precision=1,
        description=f"Pruned {suppressed_pixels:,} of {candidate_pixels:,} raw candidate pixels as isolated noise.",
        color_accent="purple",
    )


def render_spectral_distance_gauge(val: Optional[float], label: str = "Mean Spectral Distance") -> str:
    """Render Euclidean spectral distance gauge in domain [0.000, 2.000] Δρ."""
    return render_numeric_gauge(
        label=label,
        value=val,
        min_val=0.0,
        max_val=2.0,
        unit="Δρ",
        precision=4,
        description="Euclidean distance across B02, B03, B04, B08 surface reflectance bands (max 2.0).",
        color_accent="cyan",
    )
