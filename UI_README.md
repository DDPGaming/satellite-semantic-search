# FLUX Presentation UI — Local Offline Demonstration
**SIH Problem Statement**: SIH2026227<br>
**Branch**: `poc/ui`<br>
**Pipeline Coverage**: M5 (Retrieval) &rarr; M6 (Filtering) &rarr; M7 (Temporal Pairing) &rarr; M8 (Optical Change Detection) &rarr; M9 (Adaptive False-Alarm Suppression)

---

## 1. Overview

The FLUX Presentation UI is a lightweight, zero-external-dependency local web interface designed to showcase the complete end-to-end multi-temporal satellite semantic search and change analysis pipeline.

### Architectural Rules
- **Direct Pipeline Re-use**: Directly invokes `src.pipeline.PoCPipeline` and consumes `PoCPipelineResult`. It does **not** duplicate or re-implement any M1–M9 milestone logic.
- **100% Local & Offline**: Built using Python's standard library `http.server.ThreadingHTTPServer` and pure inline CSS. Requires no external CDNs, web fonts, cloud map tiles, or external JavaScript libraries.
- **Zero Raw Raster File Copies**: Raster imagery is decoded from GeoTIFFs (using `rasterio` and `Pillow`) directly in memory and served as base64-encoded PNG data URLs.

---

## 2. Quickstart

### Launch the Presentation Server
From the project root:

```bash
python app.py
```

By default, the server will initialize the staged CLIP model and FAISS vector index, then listen locally on:
```
http://127.0.0.1:8501
```

### CLI Options
```
python app.py --help

options:
  --host HOST          Host interface (default: 127.0.0.1)
  --port PORT          Port to listen on (default: 8501)
  --project-root PATH  Custom repository root directory (default: repo root)
```

---

## 3. User Interface Features

1. **Natural Language Query Panel**:
   - Query input with preset sample queries (e.g., `"urban development around Navi Mumbai"`, `"built-up area around Navi Mumbai"`, `"vegetation change around Navi Mumbai"`).
   - Filter dropdowns: Modality (`Optical`, `SAR`, `All`), Top-K (`3`, `5`, `10`), and UTC Date Range.

2. **Per-Stage Wall-Clock Timing Bar**:
   - Displays real execution timings for M5 Semantic Retrieval, M6 Metadata Filtering, M7 Temporal Pairing, M8 Change Detection, M9 False-Alarm Suppression, and Total E2E pipeline duration.

3. **M5/M6 Retrieval & Ranking Table**:
   - Lists ranked candidate tiles with similarity scores, modality, tile IDs, UTC timestamps, and scene identifiers. Highlights the top selected tile.

4. **M7 Temporal Observation Pair Visualizer**:
   - Displays side-by-side true-color RGB composites (Sentinel-2 B04/B03/B02) for Reference ($T_0$) and Comparison ($T_1$) observations with acquisition dates and delta days.

5. **M8 & M9 Change Analysis Visualizer**:
   - **M8 Spectral Distance Heatmap**: Continuous Euclidean magnitude mapped to a magma gradient.
   - **M9 Confirmed Change Mask**: High-contrast binary mask showing confirmed 8-connected change clusters.
   - **M9 Heuristic Confidence Map**: Cyan-to-emerald gradient reflecting bounded candidate confidence $[0.05, 1.0]$.

6. **Quantitative Metric Cards**:
   - Confirmed Changed Pixels, Confirmed Change Ratio (%), Suppressed False Alarms, Candidate Exceedances, Noise Median, Noise MAD, and Adaptive Threshold ($\tau$).

7. **Structured Machine-Readable Provenance**:
   - An expandable JSON inspector disclosing the complete `PoCPipelineResult` contract.

---

## 4. Testing

To run the UI test suite:
```bash
python -m unittest tests/test_ui.py -v
```

To run the entire repository test suite:
```bash
python -m unittest discover tests
```
