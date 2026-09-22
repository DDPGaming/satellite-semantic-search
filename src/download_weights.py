"""Model Weights Downloader for Offline Satellite Semantic Search.

This script pre-stages the approved pretrained vision-language model
weights for offline use.

Why this script exists:
    In competition and operational environments, the application must run
    completely offline without internet access. This script is run ONCE during
    environment setup to download the approved model weights into:
        models/<model_name>/

    At runtime, the application will load model weights exclusively from the
    local 'models/' folder and will NOT attempt any network connections.

Approved Model:
    Repository:  flax-community/clip-rsicd-v2
    Model Name:  clip-rsicd-v2
    Description: Vision-Language CLIP model fine-tuned on the Remote Sensing
                 Image Captioning Dataset (RSICD) specifically for satellite
                 and aerial imagery retrieval.

Usage:
    # Standard download / verification of approved model:
    python src/download_weights.py

    # Verify if model is already staged (offline-safe check):
    python src/download_weights.py --check-only

    # Force re-download or refresh from Hugging Face Hub:
    python src/download_weights.py --force
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Approved default vision-language model for satellite semantic search.
# flax-community/clip-rsicd-v2 provides aligned text and satellite imagery embeddings.
APPROVED_REPO_ID = "flax-community/clip-rsicd-v2"
APPROVED_MODEL_NAME = "clip-rsicd-v2"

# File extensions that signify actual model weight files
WEIGHT_EXTENSIONS = {".bin", ".safetensors", ".msgpack", ".pt"}


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_default_model_dir(model_name: str) -> Path:
    """Return the destination path under models/<model_name>/."""
    return get_project_root() / "models" / model_name


def format_file_size(size_bytes: int) -> str:
    """Format file size into a human-readable string (e.g., 598.2 MB)."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def list_staged_files(model_dir: Path) -> List[Tuple[str, int]]:
    """
    List all regular files inside model_dir recursively.

    Returns:
        A sorted list of tuples (relative_path_string, size_in_bytes).
    """
    if not model_dir.exists() or not model_dir.is_dir():
        return []

    results: List[Tuple[str, int]] = []
    for item in sorted(model_dir.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(model_dir).as_posix()
            try:
                size = item.stat().st_size
            except OSError:
                size = 0
            results.append((rel_path, size))
    return results


def is_model_staged(model_dir: Path) -> bool:
    """
    Check whether model weights are already present locally.

    Returns True if the directory contains at least one recognized model weight file.
    """
    files = list_staged_files(model_dir)
    if not files:
        return False
    return any(Path(rel_path).suffix.lower() in WEIGHT_EXTENSIONS for rel_path, _ in files)


def print_summary_report(
    model_name: str,
    repo_id: str,
    dest_dir: Path,
    staged_files: List[Tuple[str, int]],
    status_msg: str,
) -> None:
    """Print a clean, clear summary report of the staged model files."""
    separator = "=" * 60
    print(f"\n{separator}")
    print("MODEL STAGING REPORT")
    print(separator)
    print(f"  Model Name:             {model_name}")
    print(f"  Source Repository:      {repo_id}")
    print(f"  Destination Directory:  {dest_dir}")
    print(f"  Status:                 {status_msg}")
    print(separator)

    if staged_files:
        print("  Staged Files:")
        total_size = 0
        for rel_path, size in staged_files:
            total_size += size
            print(f"    - {rel_path} ({format_file_size(size)})")
        print(f"\n  Total Files:  {len(staged_files)}")
        print(f"  Total Size:   {format_file_size(total_size)}")
    else:
        print("  No model files found in destination directory.")

    print(f"{separator}\n")


def download_weights(
    repo_id: str = APPROVED_REPO_ID,
    model_name: str = APPROVED_MODEL_NAME,
    destination_dir: Optional[Path] = None,
    force: bool = False,
    check_only: bool = False,
) -> bool:
    """
    Ensure model weights are staged under models/<model_name>/ for offline use.

    Args:
        repo_id: The Hugging Face repository identifier.
        model_name: The local folder name to store under models/.
        destination_dir: Explicit destination path (defaults to models/<model_name>/).
        force: If True, re-downloads even if files are already present.
        check_only: If True, only inspects and reports existing files without downloading.

    Returns:
        True if the model is ready for offline use, False otherwise.
    """
    if destination_dir is None:
        dest_dir = get_default_model_dir(model_name)
    else:
        dest_dir = destination_dir.resolve()

    # 1. Check if model files are already staged
    staged = is_model_staged(dest_dir)
    staged_files = list_staged_files(dest_dir)

    # 2. Check-only mode: report current status and return
    if check_only:
        if staged:
            status = "Model is staged and ready for offline use."
        else:
            status = "Model is NOT staged yet. Run without --check-only to download."
        print_summary_report(model_name, repo_id, dest_dir, staged_files, status)
        return staged

    # 3. If already staged and not forcing, skip unnecessary download
    if staged and not force:
        status = "Model files are already present locally. Skipping download."
        print_summary_report(model_name, repo_id, dest_dir, staged_files, status)
        print("Tip: Use --force to re-download or check for updates from Hugging Face.")
        return True

    # 4. Check for huggingface_hub dependency
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "Error: 'huggingface_hub' is required to download model weights.\n"
            "Please install it before running this script:\n"
            "    pip install huggingface_hub",
            file=sys.stderr,
        )
        return False

    # 5. Read authentication token from environment if available (never hard-coded)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print(f"Starting download for '{repo_id}' into: {dest_dir}")
    print("Connecting to Hugging Face Hub (this may take a few minutes depending on connection)...")

    # 6. Ensure destination directory exists
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        print(f"Error creating destination directory '{dest_dir}': {err}", file=sys.stderr)
        return False

    # 7. Download files using snapshot_download
    download_kwargs = {
        "repo_id": repo_id,
        "local_dir": str(dest_dir),
    }
    if token:
        download_kwargs["token"] = token

    try:
        # local_dir_use_symlinks=False ensures actual weight files are written on all operating systems
        try:
            snapshot_download(**download_kwargs, local_dir_use_symlinks=False)
        except TypeError:
            # For newer versions of huggingface_hub where local_dir_use_symlinks is removed
            snapshot_download(**download_kwargs)
    except Exception as err:
        print(f"\nDownload failed with error: {err}", file=sys.stderr)

        # Graceful check: if we already have valid local files, inform the user
        current_files = list_staged_files(dest_dir)
        if is_model_staged(dest_dir):
            print(
                "\nNotice: Network error occurred, but local model weights were found in the destination directory.",
                file=sys.stderr,
            )
            print_summary_report(
                model_name,
                repo_id,
                dest_dir,
                current_files,
                "Network download failed, but existing local weights remain available.",
            )
            return True

        print(
            "\nTroubleshooting:\n"
            "  1. Verify your internet connection.\n"
            "  2. Verify that the Hugging Face repository exists and is accessible.\n"
            "  3. If accessing a private or gated model, set the HF_TOKEN environment variable.",
            file=sys.stderr,
        )
        return False

    # 8. Report final downloaded files
    final_files = list_staged_files(dest_dir)
    print_summary_report(
        model_name=model_name,
        repo_id=repo_id,
        dest_dir=dest_dir,
        staged_files=final_files,
        status_msg="Successfully downloaded and staged for offline use!",
    )
    return True


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and stage pretrained vision-language model weights for offline satellite semantic search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=APPROVED_REPO_ID,
        help="Source repository on Hugging Face Hub",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=APPROVED_MODEL_NAME,
        help="Subdirectory name under models/ to store model weights",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit destination directory (overrides models/<model_name>/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if files are already present locally",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check and report if model files are already staged without initiating download",
    )
    return parser.parse_args(args)


def main() -> int:
    """Main CLI entry point."""
    args = parse_args()
    success = download_weights(
        repo_id=args.repo_id,
        model_name=args.model_name,
        destination_dir=args.output_dir,
        force=args.force,
        check_only=args.check_only,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
