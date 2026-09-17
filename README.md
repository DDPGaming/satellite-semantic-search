# satellite-semantic-search

# Satellite Semantic Retrieval & Change Analysis

## Goal

Build an offline system that allows users to:

1. Search satellite imagery using natural language.
2. Search for visually similar satellite locations.
3. Compare locations across different dates.
4. Detect and classify meaningful changes.
5. Display results on a map with provenance.

## MVP

- Satellite imagery ingestion
- Image tiling
- Image embeddings
- Text-to-image semantic search
- Metadata filtering
- Basic temporal change detection
- Before/after visualization
- Map interface

## Constraints

- Must work offline during evaluation.
- Must support GeoTIFF/COG.
- Must preserve geospatial metadata.
- Models must be available locally.
