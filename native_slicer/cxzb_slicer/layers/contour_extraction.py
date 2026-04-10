"""
Contour extraction from the intersection of ``φ(x) = 0`` and a layer surface.

Given a voxel SDF grid :class:`~cxzb_slicer.sdf.grid.SDFGrid` that
approximates :math:`\\varphi(x)` and a layer surface defined by an
implicit function :math:`f_{layer}(x) = c`, the intersection curve is

.. math::

    \\{ x \\mid \\varphi(x) = 0 \\ \\text{and}\\ f_{layer}(x) = c \\}.

This module computes that curve in two stages:

1. Use Marching Cubes on the SDF grid to extract a triangle mesh
   approximation of the isosurface :math:`\\varphi(x) = 0`.
2. For each triangle, evaluate :math:`g(x) = f_{layer}(x) - c` at the
   vertices and linearly interpolate along edges where ``g`` changes
   sign to obtain intersection segments. These are then stitched into
   polylines representing contours.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import numpy.typing as npt
from skimage.measure import marching_cubes

from cxzb_slicer.core.types import Layer
from cxzb_slicer.sdf.grid import SDFGrid


def _extract_zero_isosurface(grid: SDFGrid) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Return vertices and faces of the :math:`\\varphi(x) = 0` isosurface.

    Uses :func:`skimage.measure.marching_cubes` on the voxel grid. The
    resulting vertex coordinates are scaled by :attr:`SDFGrid.spacing`
    and shifted by :attr:`SDFGrid.origin` to obtain world coordinates.
    """

    spacing = tuple(float(s) for s in grid.spacing)
    verts, faces, _normals, _values = marching_cubes(grid.values, level=0.0, spacing=spacing)
    verts_world = verts + grid.origin[None, :]
    return np.asarray(verts_world, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _triangle_segments(
    verts: npt.NDArray[np.float64],
    faces: npt.NDArray[np.int64],
    g_values: npt.NDArray[np.float64],
    eps: float = 1e-9,
) -> List[npt.NDArray[np.float64]]:
    """Compute intersection segments of triangles with the plane ``g = 0``.

    For each triangle where the scalar values ``g`` at its vertices
    change sign, two edge intersection points are found via linear
    interpolation. Each returned segment is an array of shape ``(2, 3)``.
    """

    segments: List[npt.NDArray[np.float64]] = []
    for tri in faces:
        idx = tri
        g = g_values[idx]

        # Treat near-zero as exactly zero to reduce numerical noise.
        g = np.where(np.abs(g) < eps, 0.0, g)
        if np.all(g >= 0.0) or np.all(g <= 0.0):
            continue

        v = verts[idx]
        edge_indices = [(0, 1), (1, 2), (2, 0)]
        points: list[np.ndarray] = []
        for a, b in edge_indices:
            ga, gb = g[a], g[b]
            if ga * gb < 0.0:
                t = ga / (ga - gb)
                p = v[a] + t * (v[b] - v[a])
                points.append(p)

        if len(points) == 2:
            segments.append(np.vstack(points))
    return segments


def _stitch_segments(segments: List[npt.NDArray[np.float64]], tol: float = 1e-5) -> List[npt.NDArray[np.float64]]:
    """Stitch small segments into polylines using endpoint proximity.

    This is a simple O(N²) algorithm adequate for modest meshes used in
    slicing tests. Endpoints within ``tol`` Euclidean distance are
    snapped together.
    """

    if not segments:
        return []

    unused = list(range(len(segments)))
    used = set()
    polylines: List[list[np.ndarray]] = []

    while unused:
        idx = unused.pop()
        if idx in used:
            continue
        used.add(idx)
        seg = segments[idx]
        poly = [seg[0], seg[1]]

        extended = True
        while extended:
            extended = False
            i = 0
            while i < len(unused):
                j = unused[i]
                s = segments[j]
                if np.linalg.norm(poly[-1] - s[0]) < tol:
                    poly.append(s[1])
                    used.add(j)
                    unused.pop(i)
                    extended = True
                    continue
                if np.linalg.norm(poly[-1] - s[1]) < tol:
                    poly.append(s[0])
                    used.add(j)
                    unused.pop(i)
                    extended = True
                    continue
                if np.linalg.norm(poly[0] - s[1]) < tol:
                    poly.insert(0, s[0])
                    used.add(j)
                    unused.pop(i)
                    extended = True
                    continue
                if np.linalg.norm(poly[0] - s[0]) < tol:
                    poly.insert(0, s[1])
                    used.add(j)
                    unused.pop(i)
                    extended = True
                    continue
                i += 1

        polylines.append(poly)

    return [np.vstack(p) for p in polylines]


def extract_contours_for_layer(
    grid: SDFGrid,
    layer: Layer,
    f_layer: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
) -> List[npt.NDArray[np.float64]]:
    """Extract intersection contours of ``φ=0`` with a given layer surface.

    Parameters
    ----------
    grid:
        Voxel SDF grid for :math:`\\varphi(x)`.
    layer:
        :class:`Layer` whose :attr:`Layer.level` specifies the value
        :math:`c` in the layer equation :math:`f_{layer}(x) = c`.
    f_layer:
        Callable implementing the scalar field :math:`f_{layer}(x)`.

    Returns
    -------
    list of numpy.ndarray
        Each entry is an array of shape ``(N_i, 3)`` representing a
        polyline contour in world coordinates.
    """

    verts, faces = _extract_zero_isosurface(grid)
    g_values = f_layer(verts) - float(layer.level)
    segments = _triangle_segments(verts, faces, g_values)
    return _stitch_segments(segments)

