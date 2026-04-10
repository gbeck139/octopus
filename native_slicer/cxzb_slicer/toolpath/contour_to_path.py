"""
Conversion from geometric contours to structured toolpath arrays.

A contour for a given layer is represented as a polyline in world
coordinates, typically obtained from the intersection of the object
boundary :math:`\\varphi(x) = 0` and a layer surface
``f_layer(x) = c``. This module converts such polylines into NumPy
structured arrays with :data:`cxzb_slicer.core.types.TOOLPATH_DTYPE`
for downstream kinematics, extrusion, and G-code generation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np
import numpy.typing as npt

from cxzb_slicer.core.types import Layer, SlicerConfig, TOOLPATH_DTYPE

nparray_f64 = npt.NDArray[np.float64]


class ContourPathConverter(ABC):
    """Abstract base class for contour-to-toolpath converters."""

    @abstractmethod
    def contours_to_toolpath(
        self,
        contours: Sequence[nparray_f64],
        layer: Layer,
        normals: nparray_f64 | None,
        config: SlicerConfig,
    ) -> np.ndarray:
        """Convert world-space contours for one layer into a toolpath array."""


class DefaultContourPathConverter(ContourPathConverter):
    """Baseline contour-to-toolpath converter.

    Responsibilities:

    - Ensure each contour is treated as a closed loop by repeating the
      first vertex at the end if necessary.
    - Populate world coordinates (``x,y,z``) directly from contour
      vertices.
    - Fill layer metadata (``layer_index``, normals, ``tilt_angle``).
    - Mark contour segments as ``move_type='extrude'`` and insert
      ``'travel'`` points when jumping between disconnected contours.

    Machine coordinates and extrusion/feed fields (``x_m,z_m,c,b,e,f``)
    are left at zero and will be computed in later phases.
    """

    def contours_to_toolpath(
        self,
        contours: Sequence[nparray_f64],
        layer: Layer,
        normals: nparray_f64 | None,
        config: SlicerConfig,  # noqa: ARG002 - reserved for future use
    ) -> np.ndarray:
        if not contours:
            return np.empty(0, dtype=TOOLPATH_DTYPE)

        # Normalize contour list and ensure 2D arrays.
        clean_contours: List[nparray_f64] = []
        for c in contours:
            arr = np.asarray(c, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
                continue
            # Close loop if not already.
            if not np.allclose(arr[0], arr[-1]):
                arr = np.vstack([arr, arr[0]])
            clean_contours.append(arr)

        if not clean_contours:
            return np.empty(0, dtype=TOOLPATH_DTYPE)

        # Compute per-point normals if not provided.
        if normals is None:
            # Default to vertical normals; specific layer types can
            # provide more accurate normals when needed.
            total_points = sum(c.shape[0] for c in clean_contours)
            normals_arr = np.zeros((total_points, 3), dtype=np.float64)
            normals_arr[:, 2] = 1.0
        else:
            normals_arr = np.asarray(normals, dtype=np.float64)

        # Allocate structured array for all points across all contours.
        total_points = sum(c.shape[0] for c in clean_contours)
        tp = np.zeros(total_points, dtype=TOOLPATH_DTYPE)

        offset = 0
        for c in clean_contours:
            n_pts = c.shape[0]
            sl = slice(offset, offset + n_pts)

            tp["x"][sl] = c[:, 0]
            tp["y"][sl] = c[:, 1]
            tp["z"][sl] = c[:, 2]

            if normals is not None:
                n_local = normals_arr[sl]
            else:
                n_local = normals_arr[sl]

            # Normalise and assign normals and tilt.
            lengths = np.linalg.norm(n_local, axis=1, keepdims=True)
            lengths = np.clip(lengths, 1e-9, None)
            n_unit = n_local / lengths
            tp["nx"][sl] = n_unit[:, 0]
            tp["ny"][sl] = n_unit[:, 1]
            tp["nz"][sl] = n_unit[:, 2]
            tp["tilt_angle"][sl] = np.arccos(np.clip(n_unit[:, 2], -1.0, 1.0))

            tp["layer_index"][sl] = layer.index
            tp["move_type"][sl] = "extrude"

            offset += n_pts

        # Insert travel markers between contours by setting the first
        # point of each contour after the first to move_type='travel'.
        offset = 0
        for idx, c in enumerate(clean_contours):
            n_pts = c.shape[0]
            if idx > 0:
                tp["move_type"][offset] = "travel"
            offset += n_pts

        return tp


def merge_layer_toolpaths(paths: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate multiple per-layer toolpath chunks.

    Parameters
    ----------
    paths:
        Sequence of arrays with :data:`TOOLPATH_DTYPE`.

    Returns
    -------
    numpy.ndarray
        Single concatenated toolpath array. If ``paths`` is empty,
        returns an empty array with :data:`TOOLPATH_DTYPE`.
    """

    arrays = [np.asarray(p, dtype=TOOLPATH_DTYPE) for p in paths if len(p) > 0]
    if not arrays:
        return np.empty(0, dtype=TOOLPATH_DTYPE)
    return np.concatenate(arrays, axis=0)

