"""M1–M9 Processing Pipeline Flowchart and Technical Milestone Breakdown for FLUX UI.

Renders an interactive, connected flowchart representing the frozen M1 through M9 milestones
using the authoritative implementation definitions, showing inputs, processing steps,
outputs, contributions to subsequent milestones, and demo execution status.
"""

from typing import Any, Dict, List


MILESTONES_DATA: List[Dict[str, Any]] = [
    {
        "id": "M1",
        "title": "Data Ingestion & Tile Pipeline",
        "category": "staged",
        "status_badge": "Offline Asset &bull; Frozen",
        "purpose": "Ingest raw Copernicus Sentinel-1/2 products, partition into uniform geospatial tiles, and preserve authoritative metadata.",
        "input": "Raw Copernicus Sentinel-2 Level-2A (SAFE) and Sentinel-1 Level-1 GRD (SAFE) archives.",
        "processing": "Automated raster tiling into 256&times;256 (10m) and 128&times;128 (20m) GeoTIFFs; extraction of EPSG:32643 CRS, UTM bounds, acquisition datetimes, and platform diagnostics.",
        "output": "Directory tree of standardized multi-band GeoTIFF tiles (B02, B03, B04, B08, SCL, VV, VH) accompanied by tile-level metadata.json.",
        "contribution": "Supplies standardized, georeferenced raster patches ready for radiometric harmonization and cloud masking in M2.",
    },
    {
        "id": "M2",
        "title": "GeoTIFF Preprocessing & SCL Masking",
        "category": "staged",
        "status_badge": "Offline Asset &bull; Frozen",
        "purpose": "Harmonize spatial resolutions across multispectral bands and extract ESA Scene Classification Layer (SCL) quality masks.",
        "input": "M1 GeoTIFF band tiles with disparate native resolutions (10m vs 20m).",
        "processing": "2&times; nearest-neighbor upsampling of 20m SCL to match 10m pixel grid; binary masking of cloud, shadow, snow, and defective pixels; affine transform preservation.",
        "output": "Harmonized, quality-screened surface reflectance arrays with verified geospatial co-registration.",
        "contribution": "Delivers clean, cloud-screened optical RGB tiles for visual embedding in M3 and quality masks for change detection in M8.",
    },
    {
        "id": "M3",
        "title": "Offline Vision-Language Feature Embeddings",
        "category": "staged",
        "status_badge": "Offline Asset &bull; Frozen",
        "purpose": "Generate semantic visual embeddings using a local domain-adapted remote-sensing CLIP model without cloud dependencies.",
        "input": "Quality-checked optical RGB tile composites from M2.",
        "processing": "Batch forward inference through local flax-community/clip-rsicd-v2 vision transformer backbone; L2 normalization to 512-dimensional unit hypersphere.",
        "output": "Serialized 512-dimensional float32 visual feature embedding vectors per satellite tile.",
        "contribution": "Provides dense multi-modal feature vectors required to construct the FAISS vector similarity index in M4.",
    },
    {
        "id": "M4",
        "title": "FAISS Vector Search Index",
        "category": "staged",
        "status_badge": "Offline Index &bull; Frozen",
        "purpose": "Construct an exhaustive, deterministic vector similarity search index over all cataloged tile embeddings.",
        "input": "M3 512-dimensional visual embedding vectors and tile manifest entries.",
        "processing": "FAISS IndexFlatIP index construction (inner product on L2-normalized vectors is mathematically equivalent to exact cosine similarity); catalog metadata mapping.",
        "output": "Serialized FAISS index (index.faiss) and indexed tile catalog (data/index/metadata.json).",
        "contribution": "Enables sub-millisecond similarity search over the entire tile catalog for real-time natural-language queries in M5.",
    },
    {
        "id": "M5",
        "title": "Natural-Language Semantic Search",
        "category": "live",
        "status_badge": "Live Execution &bull; Frozen",
        "purpose": "Encode free-form natural-language queries into the shared joint embedding space and retrieve Top-K candidate locations.",
        "input": "Natural-language query text (e.g., 'urban development around Navi Mumbai').",
        "processing": "Query tokenization; CLIP text transformer inference; L2 normalization; FAISS index.search() inner product cosine similarity retrieval with deterministic tie-breaking.",
        "output": "Ranked list of Top-K candidate tile hits with similarity scores and geospatial metadata.",
        "contribution": "Supplies semantically relevant candidate locations for downstream metadata filtering in M6.",
    },
    {
        "id": "M6",
        "title": "Multi-Criteria Metadata Filtering",
        "category": "live",
        "status_badge": "Live Execution &bull; Frozen",
        "purpose": "Apply strict boolean filters to narrow candidate retrievals by sensor modality, date range, or spatial bounds.",
        "input": "M5 Top-K candidate list + query filter criteria (modality: optical/sar, date_from, date_to).",
        "processing": "Sequential boolean evaluation: sensor modality match, ISO-8601 acquisition datetime window filtering, and spatial bounding-box intersection checks.",
        "output": "Refined list of candidate tile hits satisfying all operational analyst constraints.",
        "contribution": "Ensures subsequent temporal pairing in M7 operates solely on valid, analyst-constrained candidate tiles.",
    },
    {
        "id": "M7",
        "title": "Multi-Temporal Baseline Pairing",
        "category": "live",
        "status_badge": "Live Execution &bull; Frozen",
        "purpose": "Automatically discover and co-register the historical baseline observation over the exact same geographic footprint.",
        "input": "Target tile from M6 (or analyst-selected Top-K tile) + scene catalog metadata.",
        "processing": "Spatial footprint matching (row/col grid correspondence); temporal epoch search (earlier observation); affine transform and CRS co-registration verification.",
        "output": "Verified TemporalPair contract (earlier reference tile T0 + later comparison tile T1, delta days, spatial IoU).",
        "contribution": "Guarantees identical affine pixel grids and valid temporal separation before pixel differencing in M8.",
    },
    {
        "id": "M8",
        "title": "Optical Spectral Change Detection",
        "category": "live",
        "status_badge": "Live Execution &bull; Frozen",
        "purpose": "Compute physical surface reflectance distance across multi-spectral bands with SCL quality masking.",
        "input": "M7 TemporalPair with reference and comparison band rasters (B02, B03, B04, B08, SCL).",
        "processing": "Multi-band Change Vector Analysis (CVA) Euclidean distance: Δρ = sqrt(Σ(ρ_T1 - ρ_T0)²); joint SCL validity gating; summary statistics (mean, median, p95, valid ratio).",
        "output": "ChangeResult containing raw continuous change magnitude array and radiometric statistics.",
        "contribution": "Supplies the continuous spectral change signal for statistical thresholding and spatial noise suppression in M9.",
    },
    {
        "id": "M9",
        "title": "Adaptive False-Alarm Suppression & Confidence",
        "category": "live",
        "status_badge": "Live Execution &bull; Frozen",
        "purpose": "Separate genuine physical change from atmospheric and coregistration noise using robust statistics and spatial clustering.",
        "input": "M8 ChangeResult with raw change magnitude array and SCL validity mask.",
        "processing": "Median Absolute Deviation (MAD) noise floor estimation: σ_hat = 1.4826 &times; MAD, T = median + 3.0 &times; σ_hat; 8-neighbor BFS spatial clustering (min_neighbors=2, min_area=4); sigmoid confidence mapping.",
        "output": "SuppressedChangeResult containing confirmed change mask, pixel counts, and confidence map.",
        "contribution": "Delivers final, analyst-ready verified change detections with quantifiable confidence metrics for presentation.",
    },
]


def generate_m1_m9_flowchart_html() -> str:
    """
    Generate the complete, responsive M1–M9 Processing Pipeline flowchart.

    Includes:
    - Status banner showing current PoC implementation state
    - Connected flowchart layout with data-flow connectors
    - Expandable milestone detail cards with inputs, processing, outputs, and contributions
    """
    cards_html = ""
    for idx, m in enumerate(MILESTONES_DATA):
        is_live = m["category"] == "live"
        badge_cls = "badge-offline" if not is_live else "badge-sih"
        accent_color = "var(--accent-cyan)" if is_live else "var(--accent-emerald)"
        arrow_html = (
            '<div style="display:flex; justify-content:center; align-items:center; color:var(--accent-cyan); font-size:18px; font-weight:800; padding:4px 0;">&#8595;</div>'
            if idx < len(MILESTONES_DATA) - 1
            else ""
        )

        cards_html += f"""
        <div style="background:var(--bg-main); border:1px solid var(--border-color); border-radius:10px; padding:16px; margin-bottom:12px; transition:all 0.2s ease;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:10px;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="background:{accent_color}; color:#000000; font-weight:800; font-size:12px; padding:3px 8px; border-radius:4px; font-family:monospace;">
                        {m['id']}
                    </span>
                    <span style="font-size:14px; font-weight:700; color:var(--text-main);">
                        {m['title']}
                    </span>
                </div>
                <div>
                    <span class="badge {badge_cls}" style="font-size:10px; padding:3px 8px;">{m['status_badge']}</span>
                </div>
            </div>

            <div style="font-size:12px; color:var(--text-muted); line-height:1.5; margin-bottom:12px;">
                {m['purpose']}
            </div>

            <details style="background:#0c0f17; border:1px solid var(--border-color); border-radius:6px; padding:10px 14px;">
                <summary style="font-size:11px; font-weight:700; color:var(--accent-cyan); text-transform:uppercase; letter-spacing:0.5px; cursor:pointer;">
                    View Technical Breakdown (Input &bull; Processing &bull; Output &bull; Next Stage)
                </summary>
                <div style="display:grid; grid-template-columns:1fr; gap:8px; margin-top:10px; font-size:11px; line-height:1.5; border-top:1px solid var(--border-color); padding-top:8px;">
                    <div><strong style="color:var(--text-muted);">Input:</strong> {m['input']}</div>
                    <div><strong style="color:var(--accent-cyan);">Processing:</strong> {m['processing']}</div>
                    <div><strong style="color:var(--accent-emerald);">Output:</strong> {m['output']}</div>
                    <div><strong style="color:var(--accent-amber);">Contribution to Next Milestone:</strong> {m['contribution']}</div>
                </div>
            </details>
        </div>
        {arrow_html}
        """

    return f"""
    <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
            <div class="card-title" style="margin-bottom:0;">
                <span class="icon">&#9881;</span> M1&ndash;M9 Processing Pipeline Architecture
            </div>
            <div>
                <span class="badge badge-offline">&#9679; M1&ndash;M4 Pre-Computed</span>
                <span class="badge badge-sih" style="margin-left:6px;">&#9679; M5&ndash;M9 Live Execution</span>
            </div>
        </div>

        <!-- Current PoC Status Indicator -->
        <div style="background:var(--bg-main); border:1px solid var(--border-color); border-radius:8px; padding:14px 18px; margin-bottom:18px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
            <div style="font-size:12px; color:var(--text-main);">
                <strong>Current PoC Scope &amp; Demo Status:</strong>
                <span style="color:var(--text-muted); margin-left:6px;">
                    Milestones M1&ndash;M9 are implemented, validated, and frozen. This demo exercises real-time execution across M5 &rarr; M6 &rarr; M7 &rarr; M8 &rarr; M9.
                </span>
            </div>
            <div style="display:flex; gap:8px;">
                <span class="badge badge-offline">M1&ndash;M4 Staged &amp; Indexed</span>
                <span class="badge badge-sih">M5&ndash;M9 Active In Demo</span>
                <span class="badge badge-m">M10&ndash;M15 Planned Roadmap</span>
            </div>
        </div>

        <!-- Connected Flowchart Nodes -->
        <div style="max-width:900px; margin:0 auto;">
            {cards_html}
        </div>
    </div>
    """
