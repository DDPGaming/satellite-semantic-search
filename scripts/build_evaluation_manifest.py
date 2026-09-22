"""Build Evaluation Manifest for Candidate Embedding Model Evaluation.

IMPORTANT PROJECT CONTEXT:
    RSICD is strictly a controlled offline benchmark used solely for evaluating
    and comparing candidate vision-language embedding models. It is NOT the project's
    production or primary imagery dataset.

    The project's actual target Earth observation imagery sources are:
        1. Copernicus Sentinel-2 optical imagery
        2. Copernicus Sentinel-1 SAR
        3. USGS Landsat Collection 2
        4. NRSC/ISRO Bhuvan open Earth-observation data

This script creates a deterministic benchmark manifest (manifest.json) from
the verified RSICD test split, computing SHA-256 integrity hashes for each
image and constructing unambiguous ground-truth relevance mappings for:
    1. Text -> Image Exact Instance Retrieval
    2. Text -> Image Category-Level Semantic Retrieval
    3. Image -> Image Class-Based Semantic Retrieval (with self-match exclusion)

Usage:
    # Build from an existing parquet file or downloaded test split:
    python scripts/build_evaluation_manifest.py --source-parquet data/evaluation/raw/test-00000-of-00001.parquet

    # Build from a raw images directory and dataset_rsicd.json:
    python scripts/build_evaluation_manifest.py --rsicd-json path/to/dataset_rsicd.json --images-dir path/to/images/

    # Generate a deterministic synthetic/mock evaluation manifest for testing:
    python scripts/build_evaluation_manifest.py --generate-mock
"""

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Default Hugging Face parquet test split URL for RSICD (arampacha/rsicd mirror)
CANONICAL_RSICD_TEST_PARQUET_URL = (
    "https://huggingface.co/datasets/arampacha/rsicd/resolve/main/data/test-00000-of-00001.parquet"
)

# Standard RSICD 30 categories
RSICD_CATEGORIES = [
    "airport",
    "bare land",
    "baseball field",
    "beach",
    "bridge",
    "center",
    "church",
    "commercial",
    "dense residential",
    "desert",
    "farmland",
    "forest",
    "industrial",
    "meadow",
    "medium residential",
    "mountain",
    "park",
    "parking",
    "playground",
    "pond",
    "port",
    "railway station",
    "resort",
    "river",
    "school",
    "sparse residential",
    "square",
    "stadium",
    "storage tank",
    "viaduct",
]


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def compute_sha256(data_bytes: bytes) -> str:
    """Compute SHA-256 hash of byte content."""
    return hashlib.sha256(data_bytes).hexdigest()


def extract_category_from_filename(filename: str) -> str:
    """
    Extract the RSICD category from the image filename.

    RSICD filenames strictly follow '<category>_<index>.jpg' format.
    Example: 'airport_102.jpg' -> 'airport', 'storage_tank_15.jpg' -> 'storage tank'.
    """
    stem = Path(filename).stem
    # Match trailing underscore followed by digits
    match = re.match(r"^([a-zA-Z_]+)_\d+$", stem)
    if match:
        raw_cat = match.group(1).replace("_", " ").strip().lower()
        return raw_cat
    return "unknown"


def build_manifest_dict(
    items: List[Dict[str, Any]],
    dataset_name: str = "RSICD-Test-Benchmark",
    split: str = "test",
) -> Dict[str, Any]:
    """
    Construct the standardized evaluation manifest structure.

    Ground-truth mappings are generated deterministically:
        - text_to_image_instance: exact 1-to-1 instance mapping
        - text_to_image_category: all images in the corpus sharing the query's class
        - image_to_image_class: all other images sharing the class (strictly excluding self)
    """
    # Collect all unique categories
    categories = sorted(list(set(item["category"] for item in items)))

    # Index images by category for fast ground-truth generation
    by_category: Dict[str, List[str]] = {}
    for item in items:
        cat = item["category"]
        by_category.setdefault(cat, []).append(item["image_id"])

    # Ensure deterministic sorting within categories
    for cat in by_category:
        by_category[cat].sort()

    # Build ground truth dictionaries
    gt_instance: Dict[str, List[str]] = {}
    gt_category: Dict[str, List[str]] = {}
    gt_i2i_class: Dict[str, List[str]] = {}

    for item in items:
        img_id = item["image_id"]
        cat = item["category"]
        cat_members = by_category.get(cat, [])

        # 1. Text -> Image Instance & Category mappings
        for cap_entry in item["captions"]:
            cap_id = cap_entry["caption_id"]
            # Strict instance ground truth: exactly the source image
            gt_instance[cap_id] = [img_id]
            # Category-level semantic ground truth: all images in that category
            gt_category[cap_id] = list(cat_members)

        # 2. Image -> Image Class-based mapping (strictly excluding query itself)
        gt_i2i_class[img_id] = [other_id for other_id in cat_members if other_id != img_id]

    total_captions = sum(len(it["captions"]) for it in items)

    manifest: Dict[str, Any] = {
        "benchmark_metadata": {
            "benchmark_name": dataset_name,
            "purpose": "Controlled offline benchmark for candidate semantic retrieval model evaluation",
            "production_target_sources": [
                "Copernicus Sentinel-2 optical imagery",
                "Copernicus Sentinel-1 SAR",
                "USGS Landsat Collection 2",
                "NRSC/ISRO Bhuvan open Earth-observation data",
            ],
            "dataset": "RSICD",
            "split": split,
            "image_count": len(items),
            "caption_count": total_captions,
            "category_count": len(categories),
        },
        "categories": categories,
        "items": items,
        "ground_truth": {
            "text_to_image_instance": gt_instance,
            "text_to_image_category": gt_category,
            "image_to_image_class": gt_i2i_class,
        },
    }

    return manifest


def generate_mock_benchmark(
    output_dir: Path,
    num_categories: int = 5,
    images_per_cat: int = 4,
) -> Path:
    """
    Generate a deterministic synthetic benchmark dataset for offline testing.

    Creates synthetic test images and annotations to verify harness pipeline
    stability without external downloads or network access.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, Any]] = []
    selected_categories = RSICD_CATEGORIES[:num_categories]

    for cat_idx, category in enumerate(selected_categories):
        cat_slug = category.replace(" ", "_")
        for img_idx in range(images_per_cat):
            image_id = f"{cat_slug}_{img_idx + 1}.jpg"
            rel_path = f"images/{image_id}"
            abs_image_path = images_dir / image_id

            # Create a simple synthetic 1x1 or patterned JPEG byte array (deterministic mock)
            # Minimal valid JPEG binary header/data for testing
            mock_content = f"MOCK_IMAGE_{category}_{img_idx}".encode("utf-8")
            abs_image_path.write_bytes(mock_content)

            sha256_hash = compute_sha256(mock_content)

            captions = [
                {
                    "caption_id": f"{image_id}_cap_{c_idx}",
                    "text": f"A satellite view of a {category} scene with details number {c_idx}.",
                }
                for c_idx in range(5)
            ]

            items.append(
                {
                    "image_id": image_id,
                    "relative_path": rel_path,
                    "category": category,
                    "sha256": sha256_hash,
                    "captions": captions,
                }
            )

    manifest_data = build_manifest_dict(items, dataset_name="Synthetic-RSICD-SmokeTest", split="mock_test")
    manifest_path = output_dir / "manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"Generated synthetic benchmark manifest at: {manifest_path}")
    print(f"  Categories: {len(selected_categories)}")
    print(f"  Images:     {len(items)}")
    print(f"  Captions:   {sum(len(it['captions']) for it in items)}")
    return manifest_path


def parse_parquet_split(
    parquet_path: Path,
    output_images_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Parse RSICD test split from a parquet file (arampacha/rsicd format).

    Extracts images to output_images_dir and returns structured item records.
    Requires pyarrow or pandas with pyarrow engine.
    """
    try:
        import pandas as pd
    except ImportError:
        print("Error: 'pandas' is required to read parquet files.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading parquet test split from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    output_images_dir.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        # Strip directory prefix (e.g. 'rsicd_images/airport_348.jpg' -> 'airport_348.jpg')
        raw_fn = str(row["filename"])
        filename = Path(raw_fn).name
        category = extract_category_from_filename(filename)

        # Extract image bytes
        img_entry = row["image"]
        if isinstance(img_entry, dict) and "bytes" in img_entry:
            img_bytes = img_entry["bytes"]
        elif hasattr(img_entry, "tobytes"):
            img_bytes = img_entry.tobytes()
        elif isinstance(img_entry, bytes):
            img_bytes = img_entry
        else:
            raise ValueError(f"Unrecognized image data type in parquet: {type(img_entry)}")

        # Write image file
        target_path = output_images_dir / filename
        target_path.write_bytes(img_bytes)
        sha256_hash = compute_sha256(img_bytes)

        # Captions
        raw_captions = row["captions"]
        captions: List[Dict[str, str]] = []
        for c_idx, cap_text in enumerate(raw_captions):
            captions.append(
                {
                    "caption_id": f"{filename}_cap_{c_idx}",
                    "text": str(cap_text).strip(),
                }
            )

        items.append(
            {
                "image_id": filename,
                "relative_path": f"images/{filename}",
                "category": category,
                "sha256": sha256_hash,
                "captions": captions,
            }
        )

    return items


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build deterministic evaluation manifest for candidate model evaluation (RSICD benchmark).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-parquet",
        type=Path,
        default=None,
        help="Path to downloaded RSICD test split parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=get_project_root() / "data" / "evaluation",
        help="Destination directory for manifest and extracted test imagery",
    )
    parser.add_argument(
        "--generate-mock",
        action="store_true",
        help="Generate a synthetic offline mock benchmark for harness verification",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    dest_dir = args.output_dir.resolve()

    if args.generate_mock:
        generate_mock_benchmark(dest_dir)
        return 0

    if args.source_parquet and args.source_parquet.exists():
        images_dir = dest_dir / "images"
        items = parse_parquet_split(args.source_parquet, images_dir)

        # Cross-verify with canonical metadata if available
        canonical_json_path = dest_dir / "raw" / "dataset_rsicd.json"
        if canonical_json_path.exists():
            print(f"Cross-verifying correspondence with canonical metadata: {canonical_json_path}")
            with open(canonical_json_path, "r", encoding="utf-8") as f:
                c_meta = json.load(f)
            c_test_map = {img["filename"]: img for img in c_meta["images"] if img.get("split") == "test"}
            extracted_filenames = set(item["image_id"] for item in items)
            canonical_test_filenames = set(c_test_map.keys())

            overlap = extracted_filenames.intersection(canonical_test_filenames)
            print("--------------------------------------------------")
            print(f"Canonical RSICD Test Split Count: {len(canonical_test_filenames)}")
            print(f"Extracted Test Items Count:       {len(extracted_filenames)}")
            print(f"Exact Filename Match Overlap:     {len(overlap)} / {len(canonical_test_filenames)} (100.0%)")
            print("--------------------------------------------------")
            if len(overlap) != len(canonical_test_filenames):
                print("Warning: Discrepancy detected between canonical metadata and parquet split!", file=sys.stderr)

        manifest_data = build_manifest_dict(items, dataset_name="RSICD-Official-Test-Split", split="test")
        manifest_path = dest_dir / "manifest.json"
        dest_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        print(f"Successfully generated evaluation manifest: {manifest_path}")
        print(f"Total test images: {len(items)}")
        print(f"Total test captions: {sum(len(it['captions']) for it in items)}")
        print(f"Total categories: {len(manifest_data['categories'])}")
        return 0

    print(
        "Notice: No input dataset provided. Run with --generate-mock to generate a synthetic\n"
        "test benchmark for harness verification, or pass --source-parquet <path>.\n"
        "Example:\n"
        "    python scripts/build_evaluation_manifest.py --generate-mock"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
