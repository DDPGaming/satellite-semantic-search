# TODO

## Ingestion
- [ ] Read GeoTIFF
- [ ] Extract metadata
- [ ] Tile images

## Semantic Search
- [ ] Select embedding model
- [ ] Generate embeddings
- [ ] Create vector index: Explicitly mandate FAISS (highly efficient for offline CPU execution) or ChromaDB running in local persistent mode.
- [ ] Create download_weights.py to cache model weights locally.
- [ ] Text search: : Specify FastAPI. It is lightweight, runs locally, and handles asynchronous requests perfectly.

## Change Detection
- [ ] Match locations by coordinates
- [ ] Align images
- [ ] Calculate change
- [ ] Quality filtering

## UI using 
- [ ] Search box
- [ ] Specify Streamlit or Gradio. They are perfect for fast machine learning UIs, support local map rendering, and require zero external hosting.
- [ ] Map
- [ ] Before/after viewer
- [ ] Confidence
