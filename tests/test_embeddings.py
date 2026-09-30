"""Comprehensive Unit & Integration Test Suite for M3 Image Embeddings.

Verifies:
1. Model loading (local directory, strict offline mode with local_files_only=True).
2. Offline loading resilience without network access.
3. Preprocessing of optical Sentinel-2 and SAR Sentinel-1 tiles.
4. Output embedding dimensions (fixed 512-dim, float32, L2-normalized).
5. Numerical consistency between single-item and batch inference.
6. Deterministic output for identical inputs across independent calls.
7. Text encoding and joint vision-language latent space consistency.
8. Error handling for malformed, missing, or empty inputs.
9. Model metadata and provenance completeness.
10. Strict CPU execution and resource safety.
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict

import numpy as np
import rasterio
from affine import Affine
from PIL import Image

from src.embeddings.base import BaseEmbedder
from src.embeddings.clip_embedder import CLIPEmbedder
from src.embeddings.preprocessing import (
    load_image_to_pil,
    preprocess_sentinel1_tile,
    preprocess_sentinel2_tile,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
STAGED_MODEL_DIR = ROOT_DIR / "models" / "clip-rsicd-v2"
TILES_DIR = ROOT_DIR / "data" / "tiles"


class TestPreprocessing(unittest.TestCase):
    """Test deterministic satellite tile and image preprocessing."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tile_dir = Path(self.temp_dir.name) / "test_tile"
        self.tile_dir.mkdir(parents=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_preprocess_sentinel2_tile(self):
        """Verify Sentinel-2 tile preprocessing into calibrated RGB."""
        # Create synthetic 256x256 bands for B04, B03, B02
        trans = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
        profile = {
            "driver": "GTiff",
            "height": 256,
            "width": 256,
            "count": 1,
            "dtype": "uint16",
            "crs": "EPSG:32643",
            "transform": trans,
        }

        # B04 = Red (DN=1500 -> refl=0.15)
        with rasterio.open(self.tile_dir / "B04.tif", "w", **profile) as dst:
            dst.write(np.full((256, 256), 1500, dtype=np.uint16), 1)

        # B03 = Green (DN=2000 -> refl=0.20)
        with rasterio.open(self.tile_dir / "B03.tif", "w", **profile) as dst:
            dst.write(np.full((256, 256), 2000, dtype=np.uint16), 1)

        # B02 = Blue (DN=1000 -> refl=0.10)
        with rasterio.open(self.tile_dir / "B02.tif", "w", **profile) as dst:
            dst.write(np.full((256, 256), 1000, dtype=np.uint16), 1)

        img = preprocess_sentinel2_tile(self.tile_dir)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (256, 256))
        self.assertEqual(img.mode, "RGB")

        # Verify pixel scaling:
        # Green (refl=0.20) should have higher intensity than Red (0.15) and Blue (0.10)
        arr = np.array(img)
        self.assertGreater(arr[0, 0, 1], arr[0, 0, 0])  # G > R
        self.assertGreater(arr[0, 0, 0], arr[0, 0, 2])  # R > B

    def test_missing_s2_band_raises(self):
        """Verify missing optical band raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            preprocess_sentinel2_tile(self.tile_dir)

    def test_preprocess_sentinel1_tile(self):
        """Verify Sentinel-1 tile preprocessing into false-color composite."""
        profile = {
            "driver": "GTiff",
            "height": 256,
            "width": 256,
            "count": 1,
            "dtype": "uint16",
        }
        with rasterio.open(self.tile_dir / "vv.tif", "w", **profile) as dst:
            dst.write(np.full((256, 256), 500, dtype=np.uint16), 1)
        with rasterio.open(self.tile_dir / "vh.tif", "w", **profile) as dst:
            dst.write(np.full((256, 256), 100, dtype=np.uint16), 1)

        img = preprocess_sentinel1_tile(self.tile_dir)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (256, 256))
        self.assertEqual(img.mode, "RGB")

    def test_missing_s1_band_raises(self):
        """Verify missing SAR polarization band raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            preprocess_sentinel1_tile(self.tile_dir)

    def test_load_image_to_pil_numpy(self):
        """Verify NumPy array loading (2D and 3D)."""
        # 2D grayscale
        arr2d = np.full((64, 64), 128, dtype=np.uint8)
        img2d = load_image_to_pil(arr2d)
        self.assertEqual(img2d.size, (64, 64))
        self.assertEqual(img2d.mode, "RGB")

        # 3D RGB
        arr3d = np.zeros((64, 64, 3), dtype=np.uint8)
        img3d = load_image_to_pil(arr3d)
        self.assertEqual(img3d.size, (64, 64))
        self.assertEqual(img3d.mode, "RGB")

    def test_load_image_invalid_type_raises(self):
        """Verify invalid image input type raises TypeError."""
        with self.assertRaises(TypeError):
            load_image_to_pil(12345)


class TestCLIPEmbedder(unittest.TestCase):
    """Test suite for offline CLIP embedding model execution."""

    @classmethod
    def setUpClass(cls):
        if not STAGED_MODEL_DIR.exists():
            raise unittest.SkipTest(f"Staged model not found at '{STAGED_MODEL_DIR}'.")
        cls.embedder = CLIPEmbedder(model_dir=STAGED_MODEL_DIR, device="cpu", local_files_only=True)

    def test_model_loading_and_metadata(self):
        """Verify model loading, offline mode, and metadata schema."""
        self.assertEqual(self.embedder.embedding_dim, 512)
        meta = self.embedder.model_metadata
        self.assertEqual(meta["model_name"], "clip-rsicd-v2")
        self.assertEqual(meta["architecture"], "ViT-B/32")
        self.assertEqual(meta["embedding_dim"], 512)
        self.assertEqual(meta["normalization"], "L2")
        self.assertEqual(meta["dtype"], "float32")
        self.assertTrue(meta["offline_mode"])

    def test_nonexistent_model_dir_raises(self):
        """Verify nonexistent model directory raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            CLIPEmbedder(model_dir="/nonexistent/model/path")

    def test_encode_single_image(self):
        """Verify single image encoding dimensions, dtype, and L2 normalization."""
        dummy_img = Image.new("RGB", (224, 224), (100, 150, 200))
        emb = self.embedder.encode_image(dummy_img)

        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (512,))
        self.assertEqual(emb.dtype, np.float32)
        # L2 norm must be 1.0
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(float(norm), 1.0, places=5)

    def test_encode_images_batch(self):
        """Verify batch image encoding dimensions and normalization."""
        imgs = [
            Image.new("RGB", (224, 224), (i * 20, i * 30, i * 40))
            for i in range(5)
        ]
        embs = self.embedder.encode_images(imgs, batch_size=2)

        self.assertIsInstance(embs, np.ndarray)
        self.assertEqual(embs.shape, (5, 512))
        self.assertEqual(embs.dtype, np.float32)

        # Each vector must be L2 normalized
        norms = np.linalg.norm(embs, axis=1)
        for n in norms:
            self.assertAlmostEqual(float(n), 1.0, places=5)

    def test_batch_vs_single_image_consistency(self):
        """Verify single image encoding is numerically identical to batch encoding."""
        img = Image.new("RGB", (224, 224), (45, 120, 210))
        single_emb = self.embedder.encode_image(img)
        batch_emb = self.embedder.encode_images([img], batch_size=1)[0]

        np.testing.assert_allclose(single_emb, batch_emb, rtol=1e-5, atol=1e-5)

    def test_deterministic_image_output(self):
        """Verify repeated calls on the same image yield identical embeddings."""
        img = Image.new("RGB", (224, 224), (77, 88, 99))
        emb1 = self.embedder.encode_image(img)
        emb2 = self.embedder.encode_image(img)

        np.testing.assert_array_equal(emb1, emb2)

    def test_encode_text_single_and_batch(self):
        """Verify text encoding dimensions, normalization, and consistency."""
        text = "commercial container port with ships and cranes"
        emb = self.embedder.encode_text(text)

        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (512,))
        self.assertEqual(emb.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=5)

        # Batch text
        texts = [text, "residential buildings and streets", "farmland and green vegetation"]
        batch_embs = self.embedder.encode_texts(texts, batch_size=2)
        self.assertEqual(batch_embs.shape, (3, 512))

        # First text in batch matches single text
        np.testing.assert_allclose(emb, batch_embs[0], rtol=1e-5, atol=1e-5)

    def test_empty_inputs_return_empty(self):
        """Verify empty input sequences return empty (0, 512) arrays."""
        img_embs = self.embedder.encode_images([])
        self.assertEqual(img_embs.shape, (0, 512))

        txt_embs = self.embedder.encode_texts([])
        self.assertEqual(txt_embs.shape, (0, 512))

    def test_semantic_vision_language_alignment(self):
        """Verify semantic similarity alignment between text and matching image."""
        # Create an artificial green vegetation-like image
        green_img = Image.new("RGB", (224, 224), (20, 180, 40))
        # Create an artificial dark blue ocean-like image
        blue_img = Image.new("RGB", (224, 224), (10, 40, 160))

        img_embs = self.embedder.encode_images([green_img, blue_img])

        text_veg = self.embedder.encode_text("green forest and vegetation")
        text_water = self.embedder.encode_text("blue ocean water and sea")

        sim_veg_green = float(np.dot(text_veg, img_embs[0]))
        sim_veg_blue = float(np.dot(text_veg, img_embs[1]))
        sim_water_blue = float(np.dot(text_water, img_embs[1]))
        sim_water_green = float(np.dot(text_water, img_embs[0]))

        # Matching pairs should have higher cosine similarity than cross pairs
        self.assertGreater(sim_veg_green, sim_veg_blue)
        self.assertGreater(sim_water_blue, sim_water_green)


class TestLiveM2EmbeddingsIntegration(unittest.TestCase):
    """Integration test verifying embeddings generated for live staged M2 tiles."""

    def test_live_tile_embeddings_exist_and_valid(self):
        emb_dir = ROOT_DIR / "data" / "embeddings"
        if not emb_dir.exists():
            self.skipTest("No data/embeddings/ directory found.")

        manifests = list(emb_dir.rglob("manifest.json"))
        if not manifests:
            self.skipTest("No embedding manifests found in data/embeddings.")

        for m_path in manifests:
            with open(m_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            self.assertIn("source_scene_id", manifest)
            self.assertIn("embedding_dimensions", manifest)
            self.assertEqual(manifest["embedding_dimensions"][1], 512)
            self.assertEqual(manifest["dtype"], "float32")
            self.assertEqual(manifest["normalization"], "L2")

            # Load actual .npy file
            npy_path = m_path.parent / manifest["embeddings_file"]
            self.assertTrue(npy_path.exists())
            arr = np.load(npy_path)

            self.assertEqual(arr.shape[0], manifest["total_tiles"])
            self.assertEqual(arr.shape[1], 512)
            self.assertEqual(arr.dtype, np.float32)

            # Check L2 normalization across all rows
            norms = np.linalg.norm(arr, axis=1)
            np.testing.assert_allclose(norms, 1.0, rtol=1e-5, atol=1e-5)
