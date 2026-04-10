"""
Path optimization utilities for per-layer toolpaths.

This module provides a simple nearest-neighbor ordering heuristic to
reduce travel moves between disconnected contours/infill segments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np

from cxzb_slicer.core.types import Layer, SlicerConfig, TOOLPATH_DTYPE
from cxzb_slicer.toolpath.contour_to_path import DefaultContourPathConverter, merge_layer_toolpaths
from cxzb_slicer.toolpath.infill import InfillGenerator


def _path_endpoints_world(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (start_xyz, end_xyz) for a TOOLPATH_DTYPE polyline."""

    p = np.asarray(path, dtype=TOOLPATH_DTYPE)
    if len(p) == 0:
        raise ValueError("Empty path.")
    start = np.array([p["x"][0], p["y"][0], p["z"][0]], dtype=np.float64)
    end = np.array([p["x"][-1], p["y"][-1], p["z"][-1]], dtype=np.float64)
    return start, end


def _reverse_path(path: np.ndarray) -> np.ndarray:
    """Reverse a TOOLPATH_DTYPE polyline (preserving dtype)."""

    p = np.asarray(path, dtype=TOOLPATH_DTYPE)
    return p[::-1].copy()


class PathOptimizer(ABC):
    """Abstract base class for path optimizers."""

    @abstractmethod
    def optimize_layer_paths(self, paths: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Reorder and possibly reverse per-layer paths to reduce travel."""


class NearestNeighborPathOptimizer(PathOptimizer):
    """Greedy nearest-neighbor path ordering with optional reversal."""

    def optimize_layer_paths(self, paths: Sequence[np.ndarray]) -> List[np.ndarray]:
        candidates = [np.asarray(p, dtype=TOOLPATH_DTYPE) for p in paths if len(p) > 0]
        if not candidates:
            return []

        unused = list(range(len(candidates)))
        ordered: List[np.ndarray] = []

        # Seed: choose path whose start is nearest to origin (in world coords).
        starts = np.array([_path_endpoints_world(candidates[i])[0] for i in unused], dtype=np.float64)
        seed_idx = int(np.argmin(np.linalg.norm(starts, axis=1)))
        current_i = unused.pop(seed_idx)
        ordered.append(candidates[current_i])

        _cur_start, cur_end = _path_endpoints_world(ordered[-1])

        while unused:
            # Compute distances from current end to each candidate start/end.
            cand_starts = np.array([_path_endpoints_world(candidates[i])[0] for i in unused], dtype=np.float64)
            cand_ends = np.array([_path_endpoints_world(candidates[i])[1] for i in unused], dtype=np.float64)

            d_start = np.linalg.norm(cand_starts - cur_end[None, :], axis=1)
            d_end = np.linalg.norm(cand_ends - cur_end[None, :], axis=1)

            use_reverse = d_end < d_start
            best_dist = np.minimum(d_start, d_end)
            best_j = int(np.argmin(best_dist))

            chosen_i = unused.pop(best_j)
            chosen = candidates[chosen_i]
            if bool(use_reverse[best_j]):
                chosen = _reverse_path(chosen)

            ordered.append(chosen)
            _cur_start, cur_end = _path_endpoints_world(chosen)

        return ordered


def merge_optimized_paths(paths: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate optimized paths and insert travel markers at joins."""

    arrays = [np.asarray(p, dtype=TOOLPATH_DTYPE) for p in paths if len(p) > 0]
    if not arrays:
        return np.empty(0, dtype=TOOLPATH_DTYPE)

    # Mark the first point of each subsequent segment as travel.
    for i in range(1, len(arrays)):
        arrays[i] = arrays[i].copy()
        arrays[i]["move_type"][0] = "travel"
    return np.concatenate(arrays, axis=0)


def build_layer_toolpath(
    contours: Sequence[np.ndarray],
    layer: Layer,
    normals: np.ndarray | None,
    config: SlicerConfig,
    infill_generator: InfillGenerator | None = None,
    optimizer: PathOptimizer | None = None,
    infill_angle: float = 0.0,
) -> np.ndarray:
    """Build a layer toolpath from contours, optional infill, and optional optimization."""

    converter = DefaultContourPathConverter()
    contour_tp = converter.contours_to_toolpath(contours=contours, layer=layer, normals=normals, config=config)

    pieces: List[np.ndarray] = []
    if len(contour_tp) > 0:
        pieces.append(contour_tp)

    if infill_generator is not None:
        infill_tp = infill_generator.generate_infill(
            contours=contours,
            layer_index=layer.index,
            config=config,
            angle=infill_angle,
        )
        if len(infill_tp) > 0:
            pieces.append(infill_tp)

    if optimizer is None:
        return merge_layer_toolpaths(pieces)

    # Split pieces by contiguous runs of same move_type? For now treat each
    # piece as an independent path chunk.
    optimized = optimizer.optimize_layer_paths(pieces)
    return merge_optimized_paths(optimized)

