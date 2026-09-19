# Architecture

## Technology Stack

### Language
- Python 3.12

### Satellite imagery
- Rasterio
- GDAL where required
- GeoTIFF / Cloud Optimized GeoTIFF (COG)

### Numerical processing
- NumPy
- SciPy where required

### Machine learning
- PyTorch
- Pretrained remote-sensing/vision model
- Model selection TBD after evaluation

### Vector search
- FAISS for the initial MVP

### Metadata
- SQLite for the initial MVP

### Backend
- FastAPI

### Frontend
- Streamlit for the initial MVP
- PyDeck/Folium/appropriate map component for visualization

### Testing
- pytest

### Packaging
- requirements.txt

### Version control
- Git + GitHub
