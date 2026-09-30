"""Evaluation Harness for Candidate Semantic Retrieval Models.

IMPORTANT PROJECT CONTEXT:
    RSICD is strictly a controlled offline evaluation benchmark used to assess
    and compare candidate vision-language embedding models. It is NOT the project's
    primary or production imagery dataset.

    Production satellite and Earth observation imagery sources for the system are:
        1. Copernicus Sentinel-2 optical imagery
        2. Copernicus Sentinel-1 SAR
        3. USGS Landsat Collection 2
        4. NRSC/ISRO Bhuvan open Earth-observation data

This harness computes:
    1. Retrieval Quality Suite:
       - Text -> Image Exact Instance Retrieval (R@1, R@5, R@10, MRR)
       - Text -> Image Category-Level Semantic Retrieval (R@5, R@10, R@20, mAP@20, nDCG@20)
       - Image -> Image Class-Based Semantic Retrieval (P@1, P@5, P@10, mAP@50, R@10)
         * Note: Image -> Image is strictly class-based semantic retrieval and does not
           measure exhaustive pixel-level visual similarity. Self-matches are strictly excluded.
    2. Engineering & Deployment Suite:
       - Disk footprint (MB)
       - Peak memory consumption (MB)
       - Image tile encoding latency (ms/tile)
       - Text query encoding latency (ms/query)
       - Model load time (s)

Usage:
    # Dry-run evaluation on synthetic embeddings (offline, no weights needed):
    python scripts/evaluate_retrieval_models.py --dry-run

    # Evaluate a staged model:
    python scripts/evaluate_retrieval_models.py --candidate openai/clip-vit-base-patch32 --model-dir models/clip-vit-base-patch32
"""

import argparse
import json
import math
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Candidate specifications for reference
KNOWN_CANDIDATES = {
    "openai/clip-vit-base-patch32": {
        "architecture": "ViT-B/32",
        "dim": 512,
        "loader": "transformers",
        "license": "MIT",
        "training_domain": "General Web (WIT, ~400M pairs)",
        "overlap_status": "Cross-domain zero-shot (unreleased WIT web crawl)",
    },
    "flax-community/clip-rsicd-v2": {
        "architecture": "ViT-B/32",
        "dim": 512,
        "loader": "transformers",
        "license": "Unspecified (community sprint)",
        "training_domain": "Remote Sensing (RSICD fine-tuned)",
        "overlap_status": "In-domain test evaluation (trained on RSICD domain)",
    },
    "chendelong/RemoteCLIP": {
        "architecture": "ViT-B/32",
        "dim": 512,
        "loader": "open_clip / torch",
        "license": "Apache 2.0 (code) / Academic research data constraints",
        "training_domain": "Remote Sensing (RET-10M: RSICD, UCM, NWPU, etc.)",
        "overlap_status": "In-domain test evaluation (trained with RSICD train split)",
    },
    "Zilun/GeoRSCLIP": {
        "architecture": "ViT-B/32",
        "dim": 512,
        "loader": "open_clip",
        "license": "Creative Commons / CC BY-NC-SA 4.0",
        "training_domain": "Remote Sensing (RS5M, ~5M pairs)",
        "overlap_status": "Cross-domain zero-shot (RS5M explicitly excludes RSICD)",
    },
}


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_directory_size_mb(path: Path) -> float:
    """Compute total disk size of a directory in Megabytes."""
    if not path.exists():
        return 0.0
    total_bytes = 0
    if path.is_file():
        total_bytes = path.stat().st_size
    else:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total_bytes += p.stat().st_size
                except OSError:
                    pass
    return round(total_bytes / (1024.0 * 1024.0), 2)


# =====================================================================
# Metric Calculations (Decoupled, Vectorized, Pure NumPy)
# =====================================================================

def compute_recall_at_k(ranked_labels: np.ndarray, k: int) -> float:
    """
    Compute Recall@K for single-item (instance) retrieval.
    ranked_labels: boolean 2D array [num_queries, num_corpus_items],
                   True where retrieved item is relevant.
    """
    top_k = ranked_labels[:, :k]
    hits = np.any(top_k, axis=1)
    return float(np.mean(hits))


def compute_mrr(ranked_labels: np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    ranked_labels: boolean 2D array [num_queries, num_corpus_items].
    """
    num_queries = ranked_labels.shape[0]
    reciprocal_ranks = []
    for i in range(num_queries):
        hit_indices = np.where(ranked_labels[i])[0]
        if len(hit_indices) > 0:
            first_hit_rank = hit_indices[0] + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / first_hit_rank)
        else:
            reciprocal_ranks.append(0.0)
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def compute_category_recall_at_k(ranked_labels: np.ndarray, total_relevant: np.ndarray, k: int) -> float:
    """
    Compute category Recall@K (proportion of total relevant category items found in top K).
    """
    top_k = ranked_labels[:, :k]
    hits_per_query = np.sum(top_k, axis=1)
    # Avoid division by zero
    safe_totals = np.maximum(total_relevant, 1)
    recall_per_query = hits_per_query / safe_totals
    return float(np.mean(recall_per_query))


def compute_map_at_k(ranked_labels: np.ndarray, total_relevant: np.ndarray, k: int) -> float:
    """
    Compute Mean Average Precision at rank K (mAP@K).
    """
    num_queries = ranked_labels.shape[0]
    aps = []
    for i in range(num_queries):
        labels = ranked_labels[i, :k]
        tot = total_relevant[i]
        if tot == 0:
            continue
        hits = 0
        prec_sum = 0.0
        for rank, is_rel in enumerate(labels, start=1):
            if is_rel:
                hits += 1
                prec_sum += hits / rank
        aps.append(prec_sum / min(tot, k))
    return float(np.mean(aps)) if aps else 0.0


def compute_ndcg_at_k(ranked_labels: np.ndarray, total_relevant: np.ndarray, k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at rank K (nDCG@K).
    """
    num_queries = ranked_labels.shape[0]
    ndcgs = []
    for i in range(num_queries):
        labels = ranked_labels[i, :k]
        tot = total_relevant[i]
        if tot == 0:
            continue
        # DCG
        dcg = 0.0
        for rank, is_rel in enumerate(labels, start=1):
            if is_rel:
                dcg += 1.0 / math.log2(rank + 1)
        # IDCG
        idcg = 0.0
        num_ideal = min(tot, k)
        for rank in range(1, num_ideal + 1):
            idcg += 1.0 / math.log2(rank + 1)

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def compute_precision_at_k(ranked_labels: np.ndarray, k: int) -> float:
    """
    Compute Precision@K (proportion of top-K results that are relevant).
    """
    top_k = ranked_labels[:, :k]
    return float(np.mean(np.sum(top_k, axis=1) / k))


# =====================================================================
# Evaluation Execution Engine
# =====================================================================

def evaluate_retrieval_performance(
    image_embeddings: np.ndarray,
    image_ids: List[str],
    text_embeddings: np.ndarray,
    caption_ids: List[str],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the full retrieval evaluation across all three modalities.

    All embeddings are assumed to be L2-normalized.
    """
    num_images = len(image_ids)
    num_captions = len(caption_ids)
    img_to_idx = {img_id: idx for idx, img_id in enumerate(image_ids)}

    gt_instance = manifest["ground_truth"]["text_to_image_instance"]
    gt_category = manifest["ground_truth"]["text_to_image_category"]
    gt_i2i_class = manifest["ground_truth"]["image_to_image_class"]

    # 1. Similarity matrix: Text x Image [num_captions, num_images]
    text_image_sim = np.dot(text_embeddings, image_embeddings.T)

    # Ranked image indices for each text query (descending order)
    ranked_img_indices = np.argsort(-text_image_sim, axis=1)

    # -------------------------------------------------------------
    # Task 1: Text -> Image Exact Instance Retrieval
    # -------------------------------------------------------------
    instance_labels = np.zeros((num_captions, num_images), dtype=bool)
    for q_idx, cap_id in enumerate(caption_ids):
        target_imgs = gt_instance.get(cap_id, [])
        if target_imgs and target_imgs[0] in img_to_idx:
            target_idx = img_to_idx[target_imgs[0]]
            rank_pos = np.where(ranked_img_indices[q_idx] == target_idx)[0][0]
            instance_labels[q_idx, rank_pos] = True

    t2i_instance_r1 = compute_recall_at_k(instance_labels, 1)
    t2i_instance_r5 = compute_recall_at_k(instance_labels, 5)
    t2i_instance_r10 = compute_recall_at_k(instance_labels, 10)
    t2i_instance_mrr = compute_mrr(instance_labels)

    # -------------------------------------------------------------
    # Task 2: Text -> Image Category-Level Semantic Retrieval
    # -------------------------------------------------------------
    category_labels = np.zeros((num_captions, 20), dtype=bool)
    category_totals = np.zeros(num_captions, dtype=int)
    for q_idx, cap_id in enumerate(caption_ids):
        target_imgs = gt_category.get(cap_id, [])
        target_indices = {img_to_idx[t] for t in target_imgs if t in img_to_idx}
        category_totals[q_idx] = len(target_indices)
        top_20 = ranked_img_indices[q_idx, :20]
        for rank_pos, img_idx in enumerate(top_20):
            if img_idx in target_indices:
                category_labels[q_idx, rank_pos] = True

    t2i_cat_r5 = compute_category_recall_at_k(category_labels, category_totals, 5)
    t2i_cat_r10 = compute_category_recall_at_k(category_labels, category_totals, 10)
    t2i_cat_r20 = compute_category_recall_at_k(category_labels, category_totals, 20)
    t2i_cat_map20 = compute_map_at_k(category_labels, category_totals, 20)
    t2i_cat_ndcg20 = compute_ndcg_at_k(category_labels, category_totals, 20)

    # -------------------------------------------------------------
    # Task 3: Image -> Image Class-Based Semantic Retrieval
    # (WITH MANDATORY SELF-MATCH EXCLUSION)
    # -------------------------------------------------------------
    # Similarity matrix: Image x Image [num_images, num_images]
    img_img_sim = np.dot(image_embeddings, image_embeddings.T)

    # Strictly exclude self-match by setting diagonal similarity to -infinity
    np.fill_diagonal(img_img_sim, -np.inf)

    # Ranked indices for image queries (top 50 excluding self)
    ranked_i2i_indices = np.argsort(-img_img_sim, axis=1)[:, :50]

    i2i_labels = np.zeros((num_images, 50), dtype=bool)
    i2i_totals = np.zeros(num_images, dtype=int)

    for q_idx, img_id in enumerate(image_ids):
        target_imgs = gt_i2i_class.get(img_id, [])
        target_indices = {img_to_idx[t] for t in target_imgs if t in img_to_idx}
        i2i_totals[q_idx] = len(target_indices)
        top_50 = ranked_i2i_indices[q_idx]
        for rank_pos, cand_idx in enumerate(top_50):
            if cand_idx in target_indices:
                i2i_labels[q_idx, rank_pos] = True

    i2i_p1 = compute_precision_at_k(i2i_labels, 1)
    i2i_p5 = compute_precision_at_k(i2i_labels, 5)
    i2i_p10 = compute_precision_at_k(i2i_labels, 10)
    i2i_map50 = compute_map_at_k(i2i_labels, i2i_totals, 50)
    i2i_r10 = compute_category_recall_at_k(i2i_labels, i2i_totals, 10)

    results = {
        "text_to_image_instance": {
            "Recall@1": round(t2i_instance_r1, 4),
            "Recall@5": round(t2i_instance_r5, 4),
            "Recall@10": round(t2i_instance_r10, 4),
            "MRR": round(t2i_instance_mrr, 4),
        },
        "text_to_image_category": {
            "Recall@5": round(t2i_cat_r5, 4),
            "Recall@10": round(t2i_cat_r10, 4),
            "Recall@20": round(t2i_cat_r20, 4),
            "mAP@20": round(t2i_cat_map20, 4),
            "nDCG@20": round(t2i_cat_ndcg20, 4),
        },
        "image_to_image_class": {
            "Precision@1": round(i2i_p1, 4),
            "Precision@5": round(i2i_p5, 4),
            "Precision@10": round(i2i_p10, 4),
            "mAP@50": round(i2i_map50, 4),
            "Recall@10": round(i2i_r10, 4),
            "note": "Class-based semantic grouping benchmark with self-match excluded; does not evaluate pixel-level similarity.",
        },
    }

    return results


def run_synthetic_benchmark(manifest_path: Path) -> Dict[str, Any]:
    """
    Run evaluation using synthetic embeddings to verify metric math and pipeline integrity.
    """
    print(f"Loading manifest from: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    items = manifest["items"]
    image_ids = [it["image_id"] for it in items]
    caption_ids = [cap["caption_id"] for it in items for cap in it["captions"]]

    num_images = len(image_ids)
    num_captions = len(caption_ids)
    embedding_dim = 512

    print(f"Running synthetic evaluation: {num_images} images, {num_captions} captions (dim={embedding_dim})...")

    # Start memory tracing
    tracemalloc.start()
    t0 = time.perf_counter()

    # Generate deterministic synthetic L2-normalized embeddings
    rng = np.random.default_rng(seed=42)
    raw_img = rng.standard_normal((num_images, embedding_dim))
    img_embs = raw_img / np.linalg.norm(raw_img, axis=1, keepdims=True)

    raw_text = rng.standard_normal((num_captions, embedding_dim))
    text_embs = raw_text / np.linalg.norm(raw_text, axis=1, keepdims=True)

    retrieval_metrics = evaluate_retrieval_performance(
        image_embeddings=img_embs,
        image_ids=image_ids,
        text_embeddings=text_embs,
        caption_ids=caption_ids,
        manifest=manifest,
    )

    elapsed = time.perf_counter() - t0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    engineering_metrics = {
        "disk_footprint_mb": 0.0,
        "peak_ram_mb": round(peak_mem / (1024.0 * 1024.0), 2),
        "tile_encode_latency_ms": 0.0,
        "query_encode_latency_ms": 0.0,
        "evaluation_duration_s": round(elapsed, 3),
    }

    return {
        "candidate": "Synthetic-Mock-Baseline",
        "retrieval_metrics": retrieval_metrics,
        "engineering_metrics": engineering_metrics,
    }


def extract_tensor_features(out: Any) -> Any:
    """Extract 2D feature tensor from diverse Hugging Face model output formats."""
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds
    if hasattr(out, "text_embeds") and out.text_embeds is not None:
        return out.text_embeds
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def run_live_model_evaluation(
    candidate: str,
    model_dir: Path,
    manifest_path: Path,
    batch_size: int = 32,
    device: str = "cpu",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a staged vision-language model on the RSICD evaluation benchmark.
    """
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    print(f"Loading manifest from: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    items = manifest["items"]
    if max_samples and max_samples < len(items):
        print(f"Subsampling evaluation to first {max_samples} images...")
        items = items[:max_samples]

    eval_base_dir = manifest_path.parent
    image_ids = [it["image_id"] for it in items]
    caption_ids = [cap["caption_id"] for it in items for cap in it["captions"]]

    num_images = len(image_ids)
    num_captions = len(caption_ids)
    print(f"Evaluating '{candidate}' on {num_images} images, {num_captions} captions...")

    tracemalloc.start()
    t_load_0 = time.perf_counter()

    # Load model and processor offline
    model = CLIPModel.from_pretrained(str(model_dir), local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model.eval()
    load_duration = time.perf_counter() - t_load_0
    print(f"Model loaded offline in {load_duration:.2f} s")

    disk_size_mb = get_directory_size_mb(model_dir)

    # 1. Encode images
    print(f"Encoding {num_images} images in batches of {batch_size}...")
    t_img_0 = time.perf_counter()
    image_embeds_list = []
    for i in range(0, num_images, batch_size):
        batch_items = items[i : i + batch_size]
        batch_imgs = []
        for it in batch_items:
            img_path = eval_base_dir / it["relative_path"]
            with Image.open(img_path) as img:
                batch_imgs.append(img.convert("RGB"))

        inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            raw_feats = extract_tensor_features(model.get_image_features(**inputs))
            norm_feats = raw_feats / raw_feats.norm(p=2, dim=-1, keepdim=True)
            image_embeds_list.append(norm_feats.cpu().numpy().astype(np.float32))

    img_embs = np.concatenate(image_embeds_list, axis=0)
    total_img_time = time.perf_counter() - t_img_0
    tile_latency_ms = (total_img_time / max(num_images, 1)) * 1000.0

    # 2. Encode text captions
    all_captions = [cap["text"] for it in items for cap in it["captions"]]
    text_batch_size = batch_size * 2
    print(f"Encoding {num_captions} text captions in batches of {text_batch_size}...")
    t_txt_0 = time.perf_counter()
    text_embeds_list = []
    for i in range(0, num_captions, text_batch_size):
        batch_texts = all_captions[i : i + text_batch_size]
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            raw_feats = extract_tensor_features(model.get_text_features(**inputs))
            norm_feats = raw_feats / raw_feats.norm(p=2, dim=-1, keepdim=True)
            text_embeds_list.append(norm_feats.cpu().numpy().astype(np.float32))

    text_embs = np.concatenate(text_embeds_list, axis=0)
    total_txt_time = time.perf_counter() - t_txt_0
    query_latency_ms = (total_txt_time / max(num_captions, 1)) * 1000.0

    # 3. Compute retrieval metrics
    print("Computing retrieval quality suite metrics...")
    retrieval_metrics = evaluate_retrieval_performance(
        image_embeddings=img_embs,
        image_ids=image_ids,
        text_embeddings=text_embs,
        caption_ids=caption_ids,
        manifest=manifest,
    )

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    engineering_metrics = {
        "disk_footprint_mb": disk_size_mb,
        "peak_ram_mb": round(peak_mem / (1024.0 * 1024.0), 2),
        "tile_encode_latency_ms": round(tile_latency_ms, 2),
        "query_encode_latency_ms": round(query_latency_ms, 2),
        "model_load_time_s": round(load_duration, 2),
        "evaluation_duration_s": round(total_img_time + total_txt_time, 3),
        "embedding_dimension": int(img_embs.shape[1]),
    }

    return {
        "candidate": candidate,
        "retrieval_metrics": retrieval_metrics,
        "engineering_metrics": engineering_metrics,
    }


def print_evaluation_report(results: Dict[str, Any]) -> None:
    """Print an unbiased, cleanly formatted evaluation report."""
    sep = "=" * 70
    candidate = results.get("candidate", "Unknown")
    rm = results.get("retrieval_metrics", {})
    em = results.get("engineering_metrics", {})

    print(f"\n{sep}")
    print(f"CANDIDATE RETRIEVAL EVALUATION REPORT: {candidate}")
    print(sep)
    print("Benchmark: RSICD Controlled Evaluation Benchmark (Test Split)")
    print("Production Targets: Copernicus Sentinel-1/2, Landsat Coll 2, Bhuvan")
    print(sep)

    print("\n1. RETRIEVAL QUALITY SUITE")
    print("-" * 50)
    print("Task A: Text -> Image Exact Instance Retrieval:")
    for k, v in rm.get("text_to_image_instance", {}).items():
        print(f"   {k:<15}: {v}")

    print("\nTask B: Text -> Image Category-Level Semantic Retrieval:")
    for k, v in rm.get("text_to_image_category", {}).items():
        print(f"   {k:<15}: {v}")

    print("\nTask C: Image -> Image Class-Based Semantic Retrieval:")
    for k, v in rm.get("image_to_image_class", {}).items():
        if k != "note":
            print(f"   {k:<15}: {v}")
    print(f"   Note: {rm.get('image_to_image_class', {}).get('note', '')}")

    print("\n2. ENGINEERING & DEPLOYMENT SUITE")
    print("-" * 50)
    for k, v in em.items():
        print(f"   {k:<26}: {v}")

    print(f"{sep}\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate candidate vision-language embedding models on the RSICD controlled retrieval benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default="clip-rsicd-v2",
        help="Candidate model identifier",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Local path to staged model weights directory",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=get_project_root() / "data" / "evaluation" / "manifest.json",
        help="Path to benchmark evaluation manifest.json",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of images to evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device (cpu or cuda)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run dry-run evaluation using synthetic embeddings (validates harness math and pipeline)",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()

    if not args.manifest.exists():
        print(
            f"Error: Manifest not found at '{args.manifest}'.\n"
            "Generate the manifest first using:\n"
            "    python scripts/build_evaluation_manifest.py --generate-mock",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        results = run_synthetic_benchmark(args.manifest)
        print_evaluation_report(results)
        return 0

    model_dir = args.model_dir
    if model_dir is None:
        default_dir = get_project_root() / "models" / args.candidate
        if default_dir.exists():
            model_dir = default_dir

    if model_dir is None or not model_dir.exists():
        print(
            f"Error: Model directory not found for candidate '{args.candidate}'.\n"
            f"Expected staged model at: {model_dir or (get_project_root() / 'models' / args.candidate)}\n"
            "Stage model weights first using:\n"
            "    python src/download_weights.py",
            file=sys.stderr,
        )
        return 1

    results = run_live_model_evaluation(
        candidate=args.candidate,
        model_dir=model_dir,
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        device=args.device,
        max_samples=args.max_samples,
    )
    print_evaluation_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
