"""
Infill generation for per-layer contours.

This phase implements a baseline **rectilinear infill** generator for
planar layers, operating in the XY plane at a fixed Z.

Given a closed contour polygon (outer boundary) in world coordinates,
rectilinear infill is produced by generating scanlines at spacing
approximately equal to the extrusion width and clipping them against the
polygon using an even-odd (parity) rule.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np
import numpy.typing as npt

from cxzb_slicer.core.types import SlicerConfig, TOOLPATH_DTYPE

nparray_f64 = npt.NDArray[np.float64]


def _rotation_matrix(theta: float) -> nparray_f64:
    """2D rotation matrix."""

    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _ensure_closed_xy(contour_xy: nparray_f64) -> nparray_f64:
    """Ensure a 2D contour is closed (first point repeated at end)."""

    if contour_xy.shape[0] < 2:
        return contour_xy
    if not np.allclose(contour_xy[0], contour_xy[-1]):
        return np.vstack([contour_xy, contour_xy[0]])
    return contour_xy


class InfillGenerator(ABC):
    """Abstract base class for infill generators."""

    @abstractmethod
    def generate_infill(
        self,
        contours: Sequence[nparray_f64],
        layer_index: int,
        config: SlicerConfig,
        angle: float = 0.0,
    ) -> np.ndarray:
        """Generate an infill toolpath for a layer."""


class RectilinearInfill(InfillGenerator):
    """Baseline rectilinear infill for planar layers.

    Notes
    -----
    - This implementation uses only the **outermost contour** (largest
      polygon by area proxy) and ignores holes for now.
    - Contours are assumed to lie in a plane of constant Z (planar layers).
    """

    def __init__(self, spacing: float | None = None) -> None:
        self.spacing = spacing  # if None, uses config.extrusion_width

    def generate_infill(
        self,
        contours: Sequence[nparray_f64],
        layer_index: int,
        config: SlicerConfig,
        angle: float = 0.0,
    ) -> np.ndarray:
        if not contours:
            return np.empty(0, dtype=TOOLPATH_DTYPE)

        # Pick the contour with largest XY bounding-box area as a proxy for outer boundary.
        areas = []
        valid: List[nparray_f64] = []
        for c in contours:
            arr = np.asarray(c, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 3:
                continue
            xy = arr[:, :2]
            bb = xy.max(axis=0) - xy.min(axis=0)
            areas.append(float(bb[0] * bb[1]))
            valid.append(arr)

        if not valid:
            return np.empty(0, dtype=TOOLPATH_DTYPE)

        outer = valid[int(np.argmax(np.asarray(areas)))]

        z0 = float(np.median(outer[:, 2]))
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # Rotate into infill frame (so scanlines are axis-aligned).
        R = _rotation_matrix(-float(angle))
        xy = outer[:, :2] @ R.T
        xy = _ensure_closed_xy(xy)

        x_min, y_min = xy.min(axis=0)
        x_max, y_max = xy.max(axis=0)

        spacing = float(self.spacing if self.spacing is not None else config.extrusion_width)
        if spacing <= 0.0:
            raise ValueError("Infill spacing must be positive.")

        # Scanline y positions (inclusive range).
        n_lines = int(np.floor((y_max - y_min) / spacing)) + 1
        ys = y_min + spacing * np.arange(n_lines, dtype=np.float64)

        # Precompute polygon edges in rotated frame.
        x0 = xy[:-1, 0]
        y0e = xy[:-1, 1]
        x1 = xy[1:, 0]
        y1e = xy[1:, 1]

        segments_xy: List[nparray_f64] = []

        for i, y in enumerate(ys):
            # Find edges crossing this scanline (half-open rule to avoid duplicates).
            y_low = np.minimum(y0e, y1e)
            y_high = np.maximum(y0e, y1e)
            crosses = (y_low <= y) & (y < y_high) & (y_high > y_low)

            if not np.any(crosses):
                continue

            # Compute intersection x positions for crossing edges.
            denom = (y1e[crosses] - y0e[crosses])
            t = (y - y0e[crosses]) / denom
            xs = x0[crosses] + t * (x1[crosses] - x0[crosses])

            xs_sorted = np.sort(xs)
            if xs_sorted.size < 2:
                continue

            # Pair intersections into inside intervals (even-odd rule).
            pairs = xs_sorted[: (xs_sorted.size // 2) * 2].reshape(-1, 2)
            for (xa, xb) in pairs:
                if xb <= xa:
                    continue
                # Build segment endpoints in rotated XY.
                seg = np.array([[xa, y], [xb, y]], dtype=np.float64)
                if i % 2 == 1:
                    seg = seg[::-1]  # alternate direction for continuity
                segments_xy.append(seg)

        if not segments_xy:
            return np.empty(0, dtype=TOOLPATH_DTYPE)

        # Convert segments back to world coordinates.
        R_back = _rotation_matrix(float(angle))

        total_pts = sum(seg.shape[0] for seg in segments_xy)
        tp = np.zeros(total_pts, dtype=TOOLPATH_DTYPE)

        offset = 0
        for sidx, seg in enumerate(segments_xy):
            pts2 = seg @ R_back.T
            n_pts = pts2.shape[0]
            sl = slice(offset, offset + n_pts)

            tp["x"][sl] = pts2[:, 0]
            tp["y"][sl] = pts2[:, 1]
            tp["z"][sl] = z0

            tp["nx"][sl] = normal[0]
            tp["ny"][sl] = normal[1]
            tp["nz"][sl] = normal[2]
            tp["tilt_angle"][sl] = 0.0

            tp["layer_index"][sl] = int(layer_index)
            tp["move_type"][sl] = "extrude"

            # Mark the first point of each segment as a travel to indicate a jump
            # from the previous segment (downstream optimizer can reorder).
            if sidx > 0:
                tp["move_type"][offset] = "travel"

            offset += n_pts

        return tp

