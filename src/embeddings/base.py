"""Abstract Base Class for Vision-Language and Image Embedders.

Defines the standard interface for embedding generation across modalities:
- Fixed-dimensional L2-normalized float32 outputs
- Single-instance and batch inference methods
- Clean metadata and provenance inspection
- Hardware-agnostic (CPU / GPU) contract
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import numpy as np
from PIL import Image


class BaseEmbedder(ABC):
    """Abstract base class for satellite image and text embedding models."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the fixed output embedding dimension."""
        pass

    @property
    @abstractmethod
    def model_metadata(self) -> Dict[str, Any]:
        """Return structured model metadata and provenance."""
        pass

    @abstractmethod
    def encode_image(self, image: Union[Image.Image, np.ndarray, Path, str]) -> np.ndarray:
        """
        Encode a single image into a 1D L2-normalized float32 embedding vector.

        Args:
            image: PIL Image, RGB NumPy array, or path to an image / tile directory.

        Returns:
            1D NumPy array of shape (embedding_dim,) and dtype float32.
        """
        pass

    @abstractmethod
    def encode_images(
        self,
        images: Sequence[Union[Image.Image, np.ndarray, Path, str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode a sequence of images into a 2D L2-normalized float32 embedding matrix.

        Args:
            images: Sequence of PIL Images, RGB NumPy arrays, or paths.
            batch_size: Number of images per inference batch.

        Returns:
            2D NumPy array of shape (N, embedding_dim) and dtype float32.
        """
        pass

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text query into a 1D L2-normalized float32 embedding vector.

        Args:
            text: Query string.

        Returns:
            1D NumPy array of shape (embedding_dim,) and dtype float32.
        """
        pass

    @abstractmethod
    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Encode a sequence of text queries into a 2D L2-normalized float32 embedding matrix.

        Args:
            texts: Sequence of text query strings.
            batch_size: Number of texts per inference batch.

        Returns:
            2D NumPy array of shape (N, embedding_dim) and dtype float32.
        """
        pass
