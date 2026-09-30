"""CLIP Vision-Language Embedder Implementation.

Loads pretrained, remote-sensing fine-tuned CLIP models (e.g. clip-rsicd-v2)
from local offline storage for deterministic multimodal embedding generation:
- Fully offline execution via local_files_only=True
- Shared 512-dimensional L2-normalized latent space for images and text
- Provides the underlying embedding capability required for future retrieval stages (M4/M5)
- Deterministic batch and single-item inference with numerical consistency
- Hardware selection (CPU / GPU) with CPU-first optimization
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.embeddings.base import BaseEmbedder
from src.embeddings.preprocessing import load_image_to_pil


def get_default_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def extract_tensor_features(output: Any) -> torch.Tensor:
    """Safely extract 2D feature tensor from diverse Hugging Face model outputs."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


class CLIPEmbedder(BaseEmbedder):
    """
    Offline Vision-Language Embedder using pretrained CLIP weights.
    """

    def __init__(
        self,
        model_dir: Optional[Union[Path, str]] = None,
        device: str = "cpu",
        local_files_only: bool = True,
    ):
        """
        Initialize the CLIP embedder from a local offline model directory.

        Args:
            model_dir: Directory containing model weights and configs (default: models/clip-rsicd-v2).
            device: Target device ('cpu', 'cuda', or 'auto').
            local_files_only: Enforce strict offline loading without network access.
        """
        if model_dir is None:
            model_path = get_default_project_root() / "models" / "clip-rsicd-v2"
        else:
            model_path = Path(model_dir).resolve()

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model directory '{model_path}' not found.\n"
                "Please stage weights first using:\n"
                "    python src/download_weights.py"
            )

        if local_files_only:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"

        # Determine target device
        if device == "auto":
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = device.lower()

        self.device = torch.device(self.device_str)
        self.model_dir = model_path
        self.local_files_only = local_files_only

        # Load weights and processor offline
        self.model = CLIPModel.from_pretrained(
            str(self.model_dir),
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()

        self.processor = CLIPProcessor.from_pretrained(
            str(self.model_dir),
            local_files_only=local_files_only,
        )

        self._dim = int(getattr(self.model.config, "projection_dim", 512))

    @property
    def embedding_dim(self) -> int:
        """Return the fixed output embedding dimension."""
        return self._dim

    @property
    def model_metadata(self) -> Dict[str, Any]:
        """Return structured model metadata and provenance."""
        return {
            "model_name": self.model_dir.name,
            "architecture": "ViT-B/32",
            "model_type": "CLIPModel",
            "source_directory": str(self.model_dir),
            "device": self.device_str,
            "embedding_dim": self._dim,
            "normalization": "L2",
            "dtype": "float32",
            "offline_mode": self.local_files_only,
        }

    def encode_image(self, image: Union[Image.Image, np.ndarray, Path, str]) -> np.ndarray:
        """
        Encode a single image into a 1D L2-normalized float32 embedding vector.

        Guarantees exact numerical equivalence with encode_images([image])[0].
        """
        batch_out = self.encode_images([image], batch_size=1)
        return batch_out[0]

    def encode_images(
        self,
        images: Sequence[Union[Image.Image, np.ndarray, Path, str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode a sequence of images into a 2D L2-normalized float32 embedding matrix.

        Args:
            images: Sequence of PIL Images, NumPy arrays, or file/tile paths.
            batch_size: Number of images per inference batch.

        Returns:
            2D NumPy array of shape (N, 512) and dtype float32.
        """
        if not images:
            return np.empty((0, self._dim), dtype=np.float32)

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        num_images = len(images)
        all_embeddings: List[np.ndarray] = []

        for i in range(0, num_images, batch_size):
            batch_inputs = images[i : i + batch_size]
            batch_pil = [load_image_to_pil(img) for img in batch_inputs]

            inputs = self.processor(images=batch_pil, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.get_image_features(**inputs)
                feats = extract_tensor_features(out)
                # L2 normalize
                norm_feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                all_embeddings.append(norm_feats.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text query into a 1D L2-normalized float32 embedding vector.
        """
        batch_out = self.encode_texts([text], batch_size=1)
        return batch_out[0]

    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Encode a sequence of text queries into a 2D L2-normalized float32 embedding matrix.

        Args:
            texts: Sequence of query strings.
            batch_size: Number of queries per inference batch.

        Returns:
            2D NumPy array of shape (N, 512) and dtype float32.
        """
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        num_texts = len(texts)
        all_embeddings: List[np.ndarray] = []

        for i in range(0, num_texts, batch_size):
            batch_slice = list(texts[i : i + batch_size])
            inputs = self.processor(
                text=batch_slice,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                out = self.model.get_text_features(**inputs)
                feats = extract_tensor_features(out)
                norm_feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                all_embeddings.append(norm_feats.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)
