"""CLI Script for Natural Language Text Search with Metadata Filtering over Satellite Tiles.

Executes offline text-to-image semantic search against the FAISS vector index:
- Uses the locally staged CLIP model (models/clip-rsicd-v2)
- Queries the local FAISS index (data/index/faiss.index)
- Supports structured metadata filtering:
    * Modality ('optical', 'sar', or all)
    * Acquisition date range (--date-from, --date-to)
    * Exact scene identifier (--scene-id)
    * WGS84 bounding box intersection (--bbox min_lon min_lat max_lon max_lat)
- Formats results as human-readable tables or machine-readable JSON
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.retrieval.search import TextSearchEngine


def get_default_index_file() -> Path:
    return ROOT_DIR / "data" / "index" / "faiss.index"


def get_default_metadata_file() -> Path:
    return ROOT_DIR / "data" / "index" / "metadata.json"


def get_default_model_dir() -> Path:
    return ROOT_DIR / "models" / "clip-rsicd-v2"


def print_results_table(
    query: str,
    results: List[Dict[str, Any]],
    modality: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    scene_id: Optional[str] = None,
    bbox: Optional[Sequence[float]] = None,
    duration_s: float = 0.0,
) -> None:
    """Print search results in a clean formatted ASCII table with active filters."""
    sep = "=" * 100
    print(f"\n{sep}")
    print(f"SEARCH QUERY: \"{query}\"")

    filters_desc = []
    if modality:
        filters_desc.append(f"Modality={modality.upper()}")
    if date_from or date_to:
        d_range = f"{date_from or '*'}..{date_to or '*'}"
        filters_desc.append(f"Date=[{d_range}]")
    if scene_id:
        filters_desc.append(f"Scene={scene_id}")
    if bbox:
        filters_desc.append(f"BBox=[{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]")

    filter_str = ", ".join(filters_desc) if filters_desc else "None (Unconstrained)"
    print(f"Active Filters: {filter_str}")
    print(f"Retrieved: {len(results)} tiles in {duration_s:.3f} s")
    print(sep)

    if not results:
        print("No matching tiles found.")
        print(sep + "\n")
        return

    header = f"{'Rank':<5} {'Score':<8} {'Modality':<9} {'Tile ID':<48} {'Scene / Date'}"
    print(header)
    print("-" * 100)

    for hit in results:
        rank = hit["rank"]
        score = hit["similarity_score"]
        mod = hit["modality"]
        tile_id = hit["tile_id"]
        acq_date = hit.get("acquisition_datetime_utc", "")
        date_short = acq_date[:10] if acq_date else ""
        scene_raw = hit["scene_id"]
        scene_short = scene_raw[:18] + "..." if len(scene_raw) > 18 else scene_raw
        scene_display = f"{scene_short} ({date_short})" if date_short else scene_short

        print(f"{rank:<5} {score:<8.4f} {mod:<9} {tile_id:<48} {scene_display}")

    print(sep)
    print(f"Top result: {results[0]['tile_id']} (similarity: {results[0]['similarity_score']:.4f})")
    bounds = results[0].get("bounds_wgs84")
    if bounds:
        print(f"Top result WGS84 Bounding Box: [lon: {bounds[0]:.4f}..{bounds[2]:.4f}, lat: {bounds[1]:.4f}..{bounds[3]:.4f}]")
    print(sep + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search satellite image tiles using natural language queries with metadata filtering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        required=True,
        help="Natural language search query (e.g. 'cargo ships at port', 'mangrove wetlands')",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of nearest tiles to retrieve",
    )
    parser.add_argument(
        "--modality",
        "-m",
        type=str,
        default=None,
        choices=["optical", "sar"],
        help="Optional modality filter (default: all indexed modalities)",
    )
    parser.add_argument(
        "--date-from",
        type=str,
        default=None,
        help="Filter tiles acquired on or after this UTC date or ISO datetime (e.g. '2024-01-01')",
    )
    parser.add_argument(
        "--date-to",
        type=str,
        default=None,
        help="Filter tiles acquired on or before this UTC date or ISO datetime (e.g. '2024-01-31')",
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        default=None,
        help="Filter tiles matching this exact scene identifier",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=None,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Filter tiles intersecting this WGS84 bounding box (min_lon min_lat max_lon max_lat)",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="table",
        choices=["table", "json"],
        help="Output presentation format",
    )
    parser.add_argument(
        "--index-file",
        type=Path,
        default=get_default_index_file(),
        help="Path to FAISS index binary file",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=get_default_metadata_file(),
        help="Path to index metadata JSON file",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=get_default_model_dir(),
        help="Path to staged CLIP model directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device ('cpu', 'cuda', etc.)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.index_file.exists():
        print(f"Error: FAISS index file not found at: {args.index_file}", file=sys.stderr)
        print("Please build the index first using:\n    python scripts/build_index.py", file=sys.stderr)
        return 1

    if not args.metadata_file.exists():
        print(f"Error: Index metadata file not found at: {args.metadata_file}", file=sys.stderr)
        return 1

    if not args.model_dir.exists():
        print(f"Error: Model directory not found at: {args.model_dir}", file=sys.stderr)
        return 1

    try:
        engine = TextSearchEngine(
            index_file=args.index_file,
            metadata_file=args.metadata_file,
            model_dir=args.model_dir,
            device=args.device,
        )

        t0 = time.perf_counter()
        results = engine.search(
            query=args.query,
            top_k=args.top_k,
            modality=args.modality,
            date_from=args.date_from,
            date_to=args.date_to,
            scene_id=args.scene_id,
            bbox=args.bbox,
        )
        duration_s = time.perf_counter() - t0

        if args.format == "json":
            payload = {
                "query": args.query,
                "modality_filter": args.modality,
                "filters": {
                    "modality": args.modality,
                    "date_from": args.date_from,
                    "date_to": args.date_to,
                    "scene_id": args.scene_id,
                    "bbox": args.bbox,
                },
                "total_retrieved": len(results),
                "duration_seconds": round(duration_s, 4),
                "hits": results,
            }
            print(json.dumps(payload, indent=2))
        else:
            print_results_table(
                query=args.query,
                results=results,
                modality=args.modality,
                date_from=args.date_from,
                date_to=args.date_to,
                scene_id=args.scene_id,
                bbox=args.bbox,
                duration_s=duration_s,
            )

        return 0

    except Exception as e:
        print(f"Error during search: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
