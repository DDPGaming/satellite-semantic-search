"""Unit tests for src/download_weights.py.

These tests use the Python standard library unittest framework,
ensuring they run offline without requiring external network access
or extra packages installed.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure src is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.download_weights import (
    APPROVED_MODEL_NAME,
    APPROVED_REPO_ID,
    download_weights,
    format_file_size,
    get_default_model_dir,
    get_project_root,
    is_model_staged,
    list_staged_files,
    parse_args,
)


class TestDownloadWeights(unittest.TestCase):
    """Test suite for model weight download and offline staging."""

    def test_format_file_size(self):
        """Verify file size formatting in human-readable units."""
        self.assertEqual(format_file_size(500), "500 B")
        self.assertEqual(format_file_size(1024), "1.0 KB")
        self.assertEqual(format_file_size(1024 * 1024), "1.0 MB")
        self.assertEqual(format_file_size(600 * 1024 * 1024), "600.0 MB")
        self.assertEqual(format_file_size(1024 * 1024 * 1024), "1.0 GB")

    def test_get_project_root_and_paths(self):
        """Verify project root detection and default model directory."""
        root = get_project_root()
        self.assertTrue((root / "src").is_dir())
        default_dir = get_default_model_dir(APPROVED_MODEL_NAME)
        self.assertEqual(default_dir, root / "models" / APPROVED_MODEL_NAME)

    def test_list_staged_files_and_is_model_staged(self):
        """Verify detection of staged model weight files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)

            # Initially empty
            self.assertEqual(list_staged_files(model_dir), [])
            self.assertFalse(is_model_staged(model_dir))

            # Non-weight file (e.g. dummy text)
            readme_file = model_dir / "README.md"
            readme_file.write_text("dummy info", encoding="utf-8")
            self.assertEqual(len(list_staged_files(model_dir)), 1)
            self.assertFalse(is_model_staged(model_dir))

            # Valid weight file (.bin)
            bin_file = model_dir / "pytorch_model.bin"
            bin_file.write_bytes(b"\x00" * 100)
            self.assertTrue(is_model_staged(model_dir))

            staged = list_staged_files(model_dir)
            filenames = [fname for fname, _ in staged]
            self.assertIn("README.md", filenames)
            self.assertIn("pytorch_model.bin", filenames)

    def test_check_only_mode(self):
        """Verify --check-only returns appropriate boolean without calling download."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)

            # Empty dir: check_only should return False
            result_empty = download_weights(
                destination_dir=model_dir,
                check_only=True,
            )
            self.assertFalse(result_empty)

            # Add a safetensors weight file
            weight_file = model_dir / "model.safetensors"
            weight_file.write_bytes(b"\x00" * 50)

            # Populated dir: check_only should return True
            result_staged = download_weights(
                destination_dir=model_dir,
                check_only=True,
            )
            self.assertTrue(result_staged)

    def test_safe_skip_if_already_staged(self):
        """Verify script skips downloading when files are already present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "pytorch_model.bin").write_bytes(b"\x00" * 100)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")

            # Without force, should return True immediately without attempting to import huggingface_hub
            success = download_weights(
                destination_dir=model_dir,
                force=False,
            )
            self.assertTrue(success)

    def test_missing_huggingface_hub_error_handling(self):
        """Verify clean error handling when huggingface_hub is not installed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = Path(tmpdir) / "subfolder"

            # Hide huggingface_hub from sys.modules
            with patch.dict(sys.modules, {"huggingface_hub": None}):
                success = download_weights(
                    destination_dir=empty_dir,
                    force=True,
                )
                self.assertFalse(success)

    def test_download_mocked_success(self):
        """Verify snapshot_download is called with correct parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest_dir = Path(tmpdir) / "clip-rsicd-v2"

            mock_snapshot = MagicMock(return_value=str(dest_dir))

            # Simulate dummy module
            mock_hf_hub = MagicMock()
            mock_hf_hub.snapshot_download = mock_snapshot

            def fake_download(*args, **kwargs):
                # Simulate creating files
                dest_dir.mkdir(parents=True, exist_ok=True)
                (dest_dir / "pytorch_model.bin").write_bytes(b"\x00" * 200)
                (dest_dir / "config.json").write_text("{}", encoding="utf-8")
                return str(dest_dir)

            mock_snapshot.side_effect = fake_download

            with patch.dict(sys.modules, {"huggingface_hub": mock_hf_hub}):
                success = download_weights(
                    repo_id=APPROVED_REPO_ID,
                    model_name=APPROVED_MODEL_NAME,
                    destination_dir=dest_dir,
                    force=True,
                )
                self.assertTrue(success)
                self.assertTrue(is_model_staged(dest_dir))
                mock_snapshot.assert_called()
                call_kwargs = mock_snapshot.call_args[1]
                self.assertEqual(call_kwargs["repo_id"], APPROVED_REPO_ID)
                self.assertEqual(Path(call_kwargs["local_dir"]).resolve(), dest_dir.resolve())

    def test_cli_argument_parsing(self):
        """Verify default CLI options and custom argument overrides."""
        args_default = parse_args([])
        self.assertEqual(args_default.repo_id, APPROVED_REPO_ID)
        self.assertEqual(args_default.model_name, APPROVED_MODEL_NAME)
        self.assertFalse(args_default.force)
        self.assertFalse(args_default.check_only)
        self.assertIsNone(args_default.output_dir)

        args_custom = parse_args(["--check-only", "--force", "--model-name", "test-model"])
        self.assertTrue(args_custom.check_only)
        self.assertTrue(args_custom.force)
        self.assertEqual(args_custom.model_name, "test-model")


if __name__ == "__main__":
    unittest.main()
