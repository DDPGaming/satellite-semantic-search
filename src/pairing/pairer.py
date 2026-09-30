"""Multi-temporal pairing engine resolving spatial correspondence across observations.

Implements M7 responsibilities:
- Resolves temporal counterparts for retrieved/indexed satellite tiles
- Enforces strict chronological normalization (reference is earlier, comparison is later)
- Uses M2 deterministic grid_index as the primary spatial correspondence mechanism
- Calculates WGS84 bounding-box IoU as a quantitative spatial diagnostic
- Reports verified raster pixel-grid alignment for optical imagery without reprojecting
- Accurately classifies Sentinel-1 SAR as unaligned radar grids (GCP georeferenced)
- Explicitly rejects cross-modality pairing (optical <-> SAR)
- Provides deterministic canonical counterpart selection (nearest-in-time, future over past, scene_id)
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import rasterio

from src.pairing.models import AlignmentStatus, SpatialCorrespondence, TemporalPair


def get_default_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def parse_iso_datetime(dt_str: str) -> datetime:
    """Parse an ISO-8601 datetime string into a UTC timezone-aware datetime."""
    if not isinstance(dt_str, str):
        raise TypeError(f"Expected ISO datetime string, got {type(dt_str).__name__}.")
    cleaned = dt_str.strip()
    if not cleaned:
        raise ValueError("Datetime string cannot be empty or whitespace.")
    try:
        s = cleaned[:-1] + "+00:00" if cleaned.endswith(("Z", "z")) else cleaned
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None:
            return dt.astimezone(timezone.utc)
        return dt.replace(tzinfo=timezone.utc)
    except Exception as err:
        raise ValueError(f"Invalid ISO datetime string '{dt_str}': {err}") from err


def compute_wgs84_iou(b1: Sequence[float], b2: Sequence[float]) -> float:
    """
    Calculate the 2D Axis-Aligned Intersection-over-Union (IoU) of two WGS84 bounding boxes.

    Boxes are represented as [min_lon, min_lat, max_lon, max_lat].
    Returns 0.0 if boxes are disjoint or share only a zero-area boundary touch.
    """
    if len(b1) != 4 or len(b2) != 4:
        raise ValueError(f"Bounding boxes must have 4 elements, got {len(b1)} and {len(b2)}.")

    inter_min_lon = max(b1[0], b2[0])
    inter_min_lat = max(b1[1], b2[1])
    inter_max_lon = min(b1[2], b2[2])
    inter_max_lat = min(b1[3], b2[3])

    if inter_min_lon >= inter_max_lon or inter_min_lat >= inter_max_lat:
        return 0.0

    inter_area = (inter_max_lon - inter_min_lon) * (inter_max_lat - inter_min_lat)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0

    return float(inter_area / union_area)


class TemporalPairer:
    """Engine resolving multi-temporal correspondences and spatial alignment status."""

    def __init__(
        self,
        metadata_file: Optional[Union[Path, str]] = None,
        project_root: Optional[Union[Path, str]] = None,
    ):
        """
        Initialize the TemporalPairer from authoritative metadata.

        Args:
            metadata_file: Path to data/index/metadata.json (optional).
            project_root: Project root directory (optional).
        """
        self.project_root = Path(project_root).resolve() if project_root else get_default_project_root()
        self.metadata_path = Path(metadata_file) if metadata_file else self.project_root / "data" / "index" / "metadata.json"

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at: {self.metadata_path}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data.get("entries")
        if entries is None or not isinstance(entries, list):
            raise ValueError(f"Metadata file {self.metadata_path} missing 'entries' list.")

        self._tiles_by_id: Dict[str, Dict[str, Any]] = {}
        self._tiles_by_scene_grid: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
        self._scenes: Dict[str, Dict[str, Any]] = {}

        self._index_metadata(entries)

    def _index_metadata(self, entries: List[Dict[str, Any]]) -> None:
        """Validate and index all tile entries in memory with duplicate detection."""
        for entry in entries:
            # 1. Validate required fields
            tile_id = entry.get("tile_id")
            scene_id = entry.get("scene_id")
            modality = entry.get("modality")
            dt_str = entry.get("acquisition_datetime_utc")
            grid = entry.get("grid_index")
            bounds = entry.get("bounds_wgs84")

            if not tile_id or not isinstance(tile_id, str):
                raise ValueError(f"Invalid or missing tile_id: {tile_id}")
            if not scene_id or not isinstance(scene_id, str):
                raise ValueError(f"Invalid or missing scene_id: {scene_id}")
            if not modality or modality not in ("optical", "sar"):
                raise ValueError(f"Unsupported modality '{modality}' in tile {tile_id}")
            if not dt_str:
                raise ValueError(f"Tile {tile_id} missing acquisition_datetime_utc.")
            if not grid or not isinstance(grid, dict) or "row_idx" not in grid or "col_idx" not in grid:
                raise ValueError(f"Tile {tile_id} missing valid grid_index dict.")
            if not bounds or len(bounds) != 4:
                raise ValueError(f"Tile {tile_id} missing valid bounds_wgs84.")

            row_idx = grid["row_idx"]
            col_idx = grid["col_idx"]
            if not isinstance(row_idx, int) or not isinstance(col_idx, int) or isinstance(row_idx, bool):
                raise ValueError(f"Tile {tile_id} grid_index row/col must be integers.")

            # 2. Duplicate detection
            if tile_id in self._tiles_by_id:
                raise ValueError(f"Duplicate tile_id found in metadata: '{tile_id}'")

            scene_grid_key = (scene_id, row_idx, col_idx)
            if scene_grid_key in self._tiles_by_scene_grid:
                raise ValueError(
                    f"Duplicate (scene_id, grid_index) found in metadata for scene '{scene_id}' at ({row_idx}, {col_idx})"
                )

            # 3. Parse datetime and validate bounds
            parsed_dt = parse_iso_datetime(dt_str)
            if bounds[0] > bounds[2] or bounds[1] > bounds[3]:
                raise ValueError(f"Tile {tile_id} has inverted bounds_wgs84: {bounds}")

            record = dict(entry)
            record["parsed_datetime"] = parsed_dt
            record["grid_tuple"] = (row_idx, col_idx)

            self._tiles_by_id[tile_id] = record
            self._tiles_by_scene_grid[scene_grid_key] = record

            # 4. Aggregate scene metadata
            if scene_id not in self._scenes:
                self._scenes[scene_id] = {
                    "scene_id": scene_id,
                    "modality": modality,
                    "acquisition_datetime_utc": dt_str,
                    "parsed_datetime": parsed_dt,
                    "bounds_wgs84": list(bounds),
                    "grid_indices": {(row_idx, col_idx)},
                    "tile_count": 1,
                    "crs": entry.get("crs"),
                }
            else:
                scn = self._scenes[scene_id]
                scn["grid_indices"].add((row_idx, col_idx))
                scn["tile_count"] += 1
                # Expand envelope
                scn["bounds_wgs84"][0] = min(scn["bounds_wgs84"][0], bounds[0])
                scn["bounds_wgs84"][1] = min(scn["bounds_wgs84"][1], bounds[1])
                scn["bounds_wgs84"][2] = max(scn["bounds_wgs84"][2], bounds[2])
                scn["bounds_wgs84"][3] = max(scn["bounds_wgs84"][3], bounds[3])

    @property
    def total_tiles(self) -> int:
        """Total number of indexed tiles."""
        return len(self._tiles_by_id)

    @property
    def total_scenes(self) -> int:
        """Total number of indexed scenes."""
        return len(self._scenes)

    def get_tile(self, tile_id: str) -> Dict[str, Any]:
        """Retrieve a tile record by tile_id, raising KeyError if not found."""
        if not isinstance(tile_id, str):
            raise TypeError(f"tile_id must be a string, got {type(tile_id).__name__}.")
        if tile_id not in self._tiles_by_id:
            raise KeyError(f"Tile ID '{tile_id}' not found in metadata.")
        return self._tiles_by_id[tile_id]

    def get_scene(self, scene_id: str) -> Dict[str, Any]:
        """Retrieve a scene record by scene_id, raising KeyError if not found."""
        if not isinstance(scene_id, str):
            raise TypeError(f"scene_id must be a string, got {type(scene_id).__name__}.")
        if scene_id not in self._scenes:
            raise KeyError(f"Scene ID '{scene_id}' not found in metadata.")
        return self._scenes[scene_id]

    def _check_optical_alignment(
        self,
        ref_tile: Dict[str, Any],
        comp_tile: Dict[str, Any],
    ) -> Tuple[bool, str, Optional[str], Tuple[int, int]]:
        """
        Verify raster pixel-grid alignment between two optical tiles.

        Inspects raster headers on disk (shape, CRS, affine transform, resolution, bounds)
        when available, or checks authoritative metadata fields.
        """
        ref_dir = self.project_root / ref_tile.get("tile_directory", "")
        comp_dir = self.project_root / comp_tile.get("tile_directory", "")

        # Target 10m reference band
        ref_tif = ref_dir / "B04.tif"
        comp_tif = comp_dir / "B04.tif"

        if ref_tif.exists() and comp_tif.exists():
            with rasterio.open(ref_tif) as src_ref, rasterio.open(comp_tif) as src_comp:
                if src_ref.shape != src_comp.shape:
                    return False, "unaligned_raster_dimension_mismatch", str(src_ref.crs), src_ref.shape
                if src_ref.crs != src_comp.crs:
                    return False, "unaligned_crs_mismatch", str(src_ref.crs), src_ref.shape
                if src_ref.transform != src_comp.transform:
                    return False, "unaligned_transform_mismatch", str(src_ref.crs), src_ref.shape
                if src_ref.res != src_comp.res:
                    return False, "unaligned_resolution_mismatch", str(src_ref.crs), src_ref.shape
                if src_ref.bounds != src_comp.bounds:
                    return False, "unaligned_bounds_mismatch", str(src_ref.crs), src_ref.shape

                return True, "native_pixel_aligned", str(src_ref.crs), src_ref.shape

        # Fallback to metadata-level verification if GeoTIFFs are not on disk (e.g. synthetic unit tests)
        crs_ref = ref_tile.get("crs")
        crs_comp = comp_tile.get("crs")
        proj_ref = ref_tile.get("bounds_projected")
        proj_comp = comp_tile.get("bounds_projected")

        if crs_ref and crs_ref == crs_comp and proj_ref and proj_ref == proj_comp:
            return True, "native_pixel_aligned", crs_ref, (256, 256)

        return False, "unaligned_metadata_mismatch", crs_ref, (256, 256)

    def _are_scenes_compatible(
        self,
        scene_a: Dict[str, Any],
        scene_b: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Check if two scenes can form temporal pairs using structured metadata."""
        if scene_a["scene_id"] == scene_b["scene_id"]:
            return False, "Same scene"

        if scene_a["modality"] != scene_b["modality"]:
            return False, f"Modality mismatch: '{scene_a['modality']}' vs '{scene_b['modality']}'"

        if scene_a["acquisition_datetime_utc"] == scene_b["acquisition_datetime_utc"]:
            return False, "Identical acquisition timestamps"

        # Check geographic overlap
        bA = scene_a["bounds_wgs84"]
        bB = scene_b["bounds_wgs84"]
        inter_lon = max(bA[0], bB[0]) < min(bA[2], bB[2])
        inter_lat = max(bA[1], bB[1]) < min(bA[3], bB[3])

        if not (inter_lon and inter_lat):
            return False, "Non-overlapping scene footprints"

        # Check grid index overlap
        shared_grids = scene_a["grid_indices"] & scene_b["grid_indices"]
        if not shared_grids:
            return False, "No shared grid indices"

        return True, "Compatible"

    def _create_pair(
        self,
        tile_a: Dict[str, Any],
        tile_b: Dict[str, Any],
    ) -> TemporalPair:
        """
        Construct a normalized, chronologically ordered TemporalPair from two tile records.
        """
        # 1. Enforce modality equivalence
        if tile_a["modality"] != tile_b["modality"]:
            raise ValueError(
                f"Cross-modality pairing is not supported in M7. Cannot pair '{tile_a['modality']}' with '{tile_b['modality']}'."
            )

        # 2. Reject same-scene pairing
        if tile_a["scene_id"] == tile_b["scene_id"]:
            raise ValueError("Cannot pair tiles from the same scene.")

        # 3. Reject identical timestamps
        dt_a = tile_a["parsed_datetime"]
        dt_b = tile_b["parsed_datetime"]
        if dt_a == dt_b:
            raise ValueError(
                f"Observations have identical acquisition datetimes ({tile_a['acquisition_datetime_utc']})."
            )

        # 4. Enforce grid_index match
        grid_a = tile_a["grid_tuple"]
        grid_b = tile_b["grid_tuple"]
        if grid_a != grid_b:
            raise ValueError(f"Tiles have mismatched grid_index: {grid_a} vs {grid_b}.")

        # 5. Chronological normalization: reference is earlier, comparison is later
        if dt_a < dt_b:
            ref_tile = tile_a
            comp_tile = tile_b
        else:
            ref_tile = tile_b
            comp_tile = tile_a

        delta_seconds = (comp_tile["parsed_datetime"] - ref_tile["parsed_datetime"]).total_seconds()
        delta_days = delta_seconds / 86400.0

        # 6. Calculate WGS84 IoU diagnostic
        iou = compute_wgs84_iou(ref_tile["bounds_wgs84"], comp_tile["bounds_wgs84"])

        # 7. Alignment determination
        modality = ref_tile["modality"]
        if modality == "optical":
            is_aligned, align_type, crs_val, dims = self._check_optical_alignment(ref_tile, comp_tile)
            spatial_method = "grid_index_exact" if is_aligned else "grid_index_topological"
            # If bounds match down to floating tolerance, normalize IoU to 1.0
            if is_aligned and ref_tile.get("bounds_wgs84") == comp_tile.get("bounds_wgs84"):
                iou = 1.0
        else:
            # SAR rasters are GCP-georeferenced in native radar range-azimuth coordinates
            is_aligned = False
            align_type = "unaligned_radar_grid"
            crs_val = None
            dims = (256, 256)
            spatial_method = "grid_index_topological"

        spatial = SpatialCorrespondence(
            method=spatial_method,
            grid_index=ref_tile["grid_tuple"],
            spatial_iou_wgs84=round(iou, 6),
            bounds_wgs84=list(ref_tile["bounds_wgs84"]),
        )

        alignment = AlignmentStatus(
            is_pixel_aligned=is_aligned,
            alignment_type=align_type,
            crs=crs_val,
            pixel_dimensions=dims,
        )

        pair_id = f"pair__{ref_tile['tile_id']}__{comp_tile['tile_id']}"

        return TemporalPair(
            pair_id=pair_id,
            modality=modality,
            reference_tile_id=ref_tile["tile_id"],
            comparison_tile_id=comp_tile["tile_id"],
            reference_scene_id=ref_tile["scene_id"],
            comparison_scene_id=comp_tile["scene_id"],
            reference_datetime_utc=ref_tile["acquisition_datetime_utc"],
            comparison_datetime_utc=comp_tile["acquisition_datetime_utc"],
            temporal_delta_days=delta_days,
            spatial=spatial,
            alignment=alignment,
            reference_tile_dir=ref_tile["tile_directory"],
            comparison_tile_dir=comp_tile["tile_directory"],
        )

    def get_pair_for_tile(
        self,
        tile_id: str,
        target_scene_id: Optional[str] = None,
        min_temporal_delta_days: float = 0.0,
        max_temporal_delta_days: Optional[float] = None,
    ) -> Optional[TemporalPair]:
        """
        Resolve the canonical temporal counterpart for a tile deterministically.

        Rules:
        1. Validates tile_id and parameters.
        2. Discovers compatible counterpart tiles with identical grid_index and modality.
        3. Applies temporal constraints [min_temporal_delta_days, max_temporal_delta_days].
        4. If target_scene_id is specified, restricts candidates to that scene.
        5. If multiple candidates remain:
           - Chooses minimum absolute temporal distance (|t_cand - t_query|).
           - Tie-breaks by preferring future over past (t_cand > t_query).
           - Tie-breaks by ascending alphabetical scene_id.
        6. Returns None if no candidate satisfies criteria.
        """
        # 1. Parameter validations
        if not isinstance(tile_id, str):
            raise TypeError(f"tile_id must be a string, got {type(tile_id).__name__}.")
        if tile_id not in self._tiles_by_id:
            raise KeyError(f"Tile ID '{tile_id}' not found in metadata.")

        if target_scene_id is not None:
            if not isinstance(target_scene_id, str):
                raise TypeError(f"target_scene_id must be a string, got {type(target_scene_id).__name__}.")
            if target_scene_id not in self._scenes:
                raise KeyError(f"Target scene ID '{target_scene_id}' not found in metadata.")

        if isinstance(min_temporal_delta_days, bool) or not isinstance(min_temporal_delta_days, (int, float)):
            raise TypeError(f"min_temporal_delta_days must be a number, got {type(min_temporal_delta_days).__name__}.")
        if min_temporal_delta_days < 0.0:
            raise ValueError(f"min_temporal_delta_days cannot be negative ({min_temporal_delta_days}).")

        if max_temporal_delta_days is not None:
            if isinstance(max_temporal_delta_days, bool) or not isinstance(max_temporal_delta_days, (int, float)):
                raise TypeError(f"max_temporal_delta_days must be a number, got {type(max_temporal_delta_days).__name__}.")
            if max_temporal_delta_days < 0.0:
                raise ValueError(f"max_temporal_delta_days cannot be negative ({max_temporal_delta_days}).")
            if min_temporal_delta_days > max_temporal_delta_days:
                raise ValueError(
                    f"min_temporal_delta_days ({min_temporal_delta_days}) cannot be greater than max_temporal_delta_days ({max_temporal_delta_days})."
                )

        q_tile = self._tiles_by_id[tile_id]
        q_dt = q_tile["parsed_datetime"]
        q_grid = q_tile["grid_tuple"]
        q_modality = q_tile["modality"]
        q_scene = q_tile["scene_id"]

        # 2. Collect eligible candidates
        candidates: List[Dict[str, Any]] = []
        for scene_id, scn in self._scenes.items():
            if scene_id == q_scene:
                continue
            if scn["modality"] != q_modality:
                continue
            if target_scene_id is not None and scene_id != target_scene_id:
                continue

            scene_grid_key = (scene_id, q_grid[0], q_grid[1])
            if scene_grid_key not in self._tiles_by_scene_grid:
                continue

            cand_tile = self._tiles_by_scene_grid[scene_grid_key]
            cand_dt = cand_tile["parsed_datetime"]
            delta_sec = abs((cand_dt - q_dt).total_seconds())
            delta_days = delta_sec / 86400.0

            if delta_days == 0.0:
                # Same acquisition timestamp
                continue
            if delta_days < min_temporal_delta_days:
                continue
            if max_temporal_delta_days is not None and delta_days > max_temporal_delta_days:
                continue

            candidates.append(cand_tile)

        if not candidates:
            return None

        # 3. Deterministic canonical selection
        # Key: (min absolute temporal delta, future over past (-1 vs 0), ascending scene_id)
        candidates.sort(
            key=lambda c: (
                abs((c["parsed_datetime"] - q_dt).total_seconds()),
                -(1 if c["parsed_datetime"] > q_dt else 0),
                c["scene_id"],
            )
        )

        best_cand = candidates[0]
        return self._create_pair(q_tile, best_cand)

    def find_counterpart_candidates(
        self,
        tile_id: str,
    ) -> List[TemporalPair]:
        """
        Enumerate all valid temporal counterparts for a tile in deterministic order.
        """
        if not isinstance(tile_id, str):
            raise TypeError(f"tile_id must be a string, got {type(tile_id).__name__}.")
        if tile_id not in self._tiles_by_id:
            raise KeyError(f"Tile ID '{tile_id}' not found in metadata.")

        q_tile = self._tiles_by_id[tile_id]
        q_dt = q_tile["parsed_datetime"]
        q_grid = q_tile["grid_tuple"]
        q_modality = q_tile["modality"]
        q_scene = q_tile["scene_id"]

        candidates: List[Dict[str, Any]] = []
        for scene_id, scn in self._scenes.items():
            if scene_id == q_scene:
                continue
            if scn["modality"] != q_modality:
                continue

            scene_grid_key = (scene_id, q_grid[0], q_grid[1])
            if scene_grid_key in self._tiles_by_scene_grid:
                cand_tile = self._tiles_by_scene_grid[scene_grid_key]
                if cand_tile["parsed_datetime"] != q_dt:
                    candidates.append(cand_tile)

        candidates.sort(
            key=lambda c: (
                abs((c["parsed_datetime"] - q_dt).total_seconds()),
                -(1 if c["parsed_datetime"] > q_dt else 0),
                c["scene_id"],
            )
        )

        return [self._create_pair(q_tile, c) for c in candidates]

    def find_pairs_for_scenes(
        self,
        scene_id_a: str,
        scene_id_b: str,
    ) -> List[TemporalPair]:
        """
        Generate all aligned tile pairs between two compatible scenes.
        """
        if not isinstance(scene_id_a, str):
            raise TypeError(f"scene_id_a must be a string, got {type(scene_id_a).__name__}.")
        if not isinstance(scene_id_b, str):
            raise TypeError(f"scene_id_b must be a string, got {type(scene_id_b).__name__}.")

        if scene_id_a not in self._scenes:
            raise KeyError(f"Scene ID '{scene_id_a}' not found in metadata.")
        if scene_id_b not in self._scenes:
            raise KeyError(f"Scene ID '{scene_id_b}' not found in metadata.")

        if scene_id_a == scene_id_b:
            raise ValueError(f"Cannot pair scene '{scene_id_a}' with itself.")

        scene_a = self._scenes[scene_id_a]
        scene_b = self._scenes[scene_id_b]

        compat, reason = self._are_scenes_compatible(scene_a, scene_b)
        if not compat:
            if "Modality mismatch" in reason:
                raise ValueError(f"Cross-modality pairing is not supported in M7: {reason}")
            if "Identical" in reason:
                raise ValueError(f"Cannot pair scenes with identical timestamps: {reason}")
            if "Non-overlapping" in reason:
                return []
            raise ValueError(f"Scenes '{scene_id_a}' and '{scene_id_b}' are incompatible: {reason}")

        shared_grids = sorted(scene_a["grid_indices"] & scene_b["grid_indices"])
        pairs: List[TemporalPair] = []

        for r, c in shared_grids:
            tile_a = self._tiles_by_scene_grid[(scene_id_a, r, c)]
            tile_b = self._tiles_by_scene_grid[(scene_id_b, r, c)]
            pair = self._create_pair(tile_a, tile_b)
            pairs.append(pair)

        pairs.sort(key=lambda p: (p.spatial.grid_index[0], p.spatial.grid_index[1], p.pair_id))
        return pairs

    def find_all_pairs(
        self,
        modality: Optional[str] = None,
    ) -> List[TemporalPair]:
        """
        Enumerate all available temporal pairs across the indexed dataset.
        """
        if modality is not None:
            if not isinstance(modality, str):
                raise TypeError(f"modality must be a string or None, got {type(modality).__name__}.")
            norm_mod = modality.strip().lower()
            if norm_mod not in ("optical", "sar"):
                raise ValueError(f"Unsupported modality '{modality}'. Must be 'optical', 'sar', or None.")
            target_modalities = [norm_mod]
        else:
            target_modalities = ["optical", "sar"]

        all_pairs: List[TemporalPair] = []

        for mod in target_modalities:
            scenes = sorted(
                [s for s in self._scenes.values() if s["modality"] == mod],
                key=lambda s: (s["parsed_datetime"], s["scene_id"]),
            )

            # Generate all pairs of distinct scenes (i < j)
            for i in range(len(scenes)):
                for j in range(i + 1, len(scenes)):
                    s1 = scenes[i]
                    s2 = scenes[j]
                    compat, _ = self._are_scenes_compatible(s1, s2)
                    if compat:
                        pairs = self.find_pairs_for_scenes(s1["scene_id"], s2["scene_id"])
                        all_pairs.extend(pairs)

        all_pairs.sort(
            key=lambda p: (
                p.modality,
                p.reference_scene_id,
                p.comparison_scene_id,
                p.spatial.grid_index[0],
                p.spatial.grid_index[1],
                p.pair_id,
            )
        )
        return all_pairs
