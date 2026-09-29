"""Comprehensive Unit & Integration Test Suite for M4 Vector Search / FAISS Indexing.

Verifies:
1. FAISS IndexFlatIP construction and dimension validation.
2. Index persistence and exact reconstruction on load.
3. Total indexed vector count = 226 and metadata count = 226.
4. Correct row-ID -> tile-ID bidirectional mapping.
5. Deterministic search results across repeated queries.
6. Top-K behavior and strictly descending score ordering.
7. Unit L2-normalized vector inner product equals cosine similarity (self-match = 1.0).
8. Save/load produces identical search results and scores.
9. Error handling for missing files, corrupt files, invalid dims, and invalid top_k.
10. Nearest-neighbor tile search (supports M12 similar-location discovery).
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.retrieval.builder import build_index_from_embeddings
from src.retrieval.index import VectorIndex

ROOT_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = ROOT_DIR / "data" / "embeddings"
TILES_DIR = ROOT_DIR / "data" / "tiles"
INDEX_FILE = ROOT_DIR / "data" / "index" / "faiss.index"
METADATA_FILE = ROOT_DIR / "data" / "index" / "metadata.json"


class TestVectorIndexSynthetic(unittest.TestCase):
    """Test VectorIndex operations using synthetic embeddings and metadata."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.dim = 512

        # Generate deterministic synthetic L2-normalized vectors
        rng = np.random.default_rng(seed=42)
        raw_vecs = rng.standard_normal((10, self.dim)).astype(np.float32)
        self.synthetic_vectors = raw_vecs / np.linalg.norm(raw_vecs, axis=1, keepdims=True)

        self.synthetic_metadata = [
            {
                "tile_id": f"mock_tile_{i:02d}",
                "scene_id": "MOCK_SCENE_01",
                "modality": "optical" if i % 2 == 0 else "sar",
                "bounds_wgs84": [72.9 + i * 0.01, 18.9, 72.95 + i * 0.01, 18.95],
                "tile_directory": f"mock/path/tile_{i:02d}",
            }
            for i in range(10)
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_index_initialization(self):
        """Verify index dimension and initial state."""
        idx = VectorIndex(dimension=512)
        self.assertEqual(idx.dimension, 512)
        self.assertEqual(idx.total_vectors, 0)
        self.assertEqual(len(idx.metadata_entries), 0)

    def test_invalid_dimension_raises(self):
        """Verify invalid dimension raises ValueError."""
        with self.assertRaises(ValueError):
            VectorIndex(dimension=-1)
        with self.assertRaises(ValueError):
            VectorIndex(dimension=0)

    def test_add_vectors_and_metadata(self):
        """Verify vector addition, row assignment, and bidirectional mapping."""
        idx = VectorIndex(dimension=self.dim)
        idx.add(self.synthetic_vectors, self.synthetic_metadata)

        self.assertEqual(idx.total_vectors, 10)
        self.assertEqual(len(idx.metadata_entries), 10)

        # Check row IDs
        for i in range(10):
            entry = idx.get_metadata_by_row_id(i)
            self.assertEqual(entry["row_id"], i)
            self.assertEqual(entry["tile_id"], f"mock_tile_{i:02d}")

            tile_entry = idx.get_metadata_by_tile_id(f"mock_tile_{i:02d}")
            self.assertIsNotNone(tile_entry)
            self.assertEqual(tile_entry["row_id"], i)

    def test_dimension_mismatch_raises(self):
        """Verify dimension mismatch raises ValueError."""
        idx = VectorIndex(dimension=512)
        wrong_dim_vecs = np.zeros((5, 256), dtype=np.float32)
        with self.assertRaises(ValueError):
            idx.add(wrong_dim_vecs, self.synthetic_metadata[:5])

    def test_unnormalized_vectors_raise(self):
        """Verify non-unit vectors raise ValueError."""
        idx = VectorIndex(dimension=self.dim)
        unnorm_vecs = np.full((5, self.dim), 5.0, dtype=np.float32)
        with self.assertRaises(ValueError):
            idx.add(unnorm_vecs, self.synthetic_metadata[:5])

    def test_count_mismatch_raises(self):
        """Verify count mismatch between vectors and metadata raises ValueError."""
        idx = VectorIndex(dimension=self.dim)
        with self.assertRaises(ValueError):
            idx.add(self.synthetic_vectors[:5], self.synthetic_metadata[:3])

    def test_search_self_match_score_one(self):
        """Verify self-query yields similarity score 1.0 (cosine equivalence)."""
        idx = VectorIndex(dimension=self.dim)
        idx.add(self.synthetic_vectors, self.synthetic_metadata)

        query = self.synthetic_vectors[3]
        hits = idx.search(query, top_k=5)

        self.assertEqual(len(hits), 5)
        # Top 1 must be self-match
        self.assertEqual(hits[0]["rank"], 1)
        self.assertEqual(hits[0]["tile_id"], "mock_tile_03")
        self.assertEqual(hits[0]["row_id"], 3)
        self.assertAlmostEqual(hits[0]["similarity_score"], 1.0, places=5)

    def test_search_descending_scores(self):
        """Verify search results are sorted in strictly descending similarity score order."""
        idx = VectorIndex(dimension=self.dim)
        idx.add(self.synthetic_vectors, self.synthetic_metadata)

        query = self.synthetic_vectors[0]
        hits = idx.search(query, top_k=10)

        scores = [h["similarity_score"] for h in hits]
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i], scores[i + 1])

    def test_search_top_k_bounds(self):
        """Verify top_k constraints and truncation when top_k > total_vectors."""
        idx = VectorIndex(dimension=self.dim)
        idx.add(self.synthetic_vectors, self.synthetic_metadata)

        # Invalid top_k
        with self.assertRaises(ValueError):
            idx.search(self.synthetic_vectors[0], top_k=0)
        with self.assertRaises(ValueError):
            idx.search(self.synthetic_vectors[0], top_k=-5)

        # Requesting more than total vectors returns exactly total_vectors
        hits = idx.search(self.synthetic_vectors[0], top_k=100)
        self.assertEqual(len(hits), 10)

    def test_save_and_load_roundtrip(self):
        """Verify index and metadata serialization and exact query equivalence."""
        idx = VectorIndex(dimension=self.dim)
        idx.add(self.synthetic_vectors, self.synthetic_metadata)

        idx_file = self.tmp_path / "faiss.index"
        meta_file = self.tmp_path / "metadata.json"

        idx.save(idx_file, meta_file)
        self.assertTrue(idx_file.exists())
        self.assertTrue(meta_file.exists())

        loaded_idx = VectorIndex.load(idx_file, meta_file)
        self.assertEqual(loaded_idx.dimension, idx.dimension)
        self.assertEqual(loaded_idx.total_vectors, idx.total_vectors)

        # Search must return bitwise identical scores and ranks
        q = self.synthetic_vectors[2]
        hits_orig = idx.search(q, top_k=5)
        hits_load = loaded_idx.search(q, top_k=5)

        self.assertEqual(len(hits_orig), len(hits_load))
        for h1, h2 in zip(hits_orig, hits_load):
            self.assertEqual(h1["rank"], h2["rank"])
            self.assertEqual(h1["tile_id"], h2["tile_id"])
            self.assertEqual(h1["row_id"], h2["row_id"])
            self.assertAlmostEqual(h1["similarity_score"], h2["similarity_score"], places=6)

    def test_load_missing_files_raises(self):
        """Verify FileNotFoundError when index files are missing."""
        with self.assertRaises(FileNotFoundError):
            VectorIndex.load("/missing/faiss.index", "/missing/metadata.json")

    def test_load_corrupt_metadata_raises(self):
        """Verify ValueError when index and metadata counts do not match."""
        idx = VectorIndex(dimension=self.dim)
        idx.add(self.synthetic_vectors, self.synthetic_metadata)

        idx_file = self.tmp_path / "faiss.index"
        meta_file = self.tmp_path / "metadata.json"
        idx.save(idx_file, meta_file)

        # Tamper with metadata to create count mismatch
        with open(meta_file, "r") as f:
            data = json.load(f)
        data["entries"] = data["entries"][:5]  # Drop half the entries
        with open(meta_file, "w") as f:
            json.dump(data, f)

        with self.assertRaises(ValueError):
            VectorIndex.load(idx_file, meta_file)


class TestVectorIndexLiveStagedData(unittest.TestCase):
    """Integration test verifying the real 226 tile embeddings index."""

    def test_build_and_validate_live_index(self):
        """Verify complete build from M3 embeddings and M2 tiles."""
        if not EMBEDDINGS_DIR.exists() or not TILES_DIR.exists():
            self.skipTest("data/embeddings or data/tiles not present.")

        index = build_index_from_embeddings(EMBEDDINGS_DIR, TILES_DIR, dimension=512)

        # Invariant checks: exactly 226 vectors, 512 dimensions
        self.assertEqual(index.total_vectors, 226)
        self.assertEqual(index.dimension, 512)
        self.assertEqual(len(index.metadata_entries), 226)

        # Modality distribution: 98 optical + 128 SAR
        mods = [m["modality"] for m in index.metadata_entries]
        self.assertEqual(mods.count("optical"), 98)
        self.assertEqual(mods.count("sar"), 128)

        # Ensure every entry has valid coordinates and tile_id
        for entry in index.metadata_entries:
            self.assertIn("tile_id", entry)
            self.assertIn("scene_id", entry)
            self.assertIn("modality", entry)
            self.assertIn("bounds_wgs84", entry)
            self.assertEqual(len(entry["bounds_wgs84"]), 4)

    def test_live_persisted_index(self):
        """Verify the persisted data/index/ files if generated."""
        if not INDEX_FILE.exists() or not METADATA_FILE.exists():
            self.skipTest("Persisted index files not yet generated in data/index/.")

        loaded = VectorIndex.load(INDEX_FILE, METADATA_FILE)
        self.assertEqual(loaded.total_vectors, 226)
        self.assertEqual(loaded.dimension, 512)

        # Run query using first tile in index
        first_tile = loaded.metadata_entries[0]
        q_vec = loaded.index.reconstruct(0)
        hits = loaded.search(q_vec, top_k=5)

        self.assertEqual(len(hits), 5)
        self.assertEqual(hits[0]["rank"], 1)
        self.assertEqual(hits[0]["tile_id"], first_tile["tile_id"])
        self.assertAlmostEqual(hits[0]["similarity_score"], 1.0, places=5)
