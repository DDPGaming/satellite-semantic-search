"""Embeddings Package for Satellite Semantic Search.

Exports:
- BaseEmbedder: Abstract contract for image/text embedding models
- CLIPEmbedder: Pretrained vision-language embedder with offline loading
- preprocess_sentinel2_tile: Deterministic optical tile preprocessor
- preprocess_sentinel1_tile: Deterministic SAR tile preprocessor
- load_image_to_pil: Format-agnostic image resolution helper
"""

from src.embeddings.base import BaseEmbedder
from src.embeddings.clip_embedder import CLIPEmbedder
from src.embeddings.preprocessing import (
    load_image_to_pil,
    preprocess_sentinel1_tile,
    preprocess_sentinel2_tile,
)

__all__ = [
    "BaseEmbedder",
    "CLIPEmbedder",
    "preprocess_sentinel2_tile",
    "preprocess_sentinel1_tile",
    "load_image_to_pil",
]
