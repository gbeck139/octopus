"""
`pysdf`-backed implementation of :class:`cxzb_slicer.sdf.provider.SDFProvider`.

The :mod:`pysdf` library constructs a triangle-mesh-based signed distance
function with a simple callable interface::

    from pysdf import SDF
    sdf = SDF(vertices, faces)
    distances = sdf(points)  # (N,) array, negative inside

This module wraps that callable in the abstract :class:`SDFProvider`
interface so it can be used interchangeably with other backends such as
voxel grids or Open3D raycasting scenes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt
import trimesh
from pysdf import SDF as _PySDF

from .provider import SDFProvider


@dataclass(slots=True)
class PySDFProvider(SDFProvider):
    """`pysdf`-based signed distance field provider.

    Parameters
    ----------
    vertices, faces:
        Triangle mesh geometry defining the solid. ``vertices`` is an
        array of shape ``(V, 3)`` and ``faces`` has shape ``(F, 3)``
        with integer indices into ``vertices``.
    bounds_min, bounds_max:
        Optional precomputed axis-aligned bounding box corners. If not
        provided they are computed from ``vertices``.

    Notes
    -----
    The signed distance is evaluated via the underlying :class:`pysdf.SDF`
    callable, which implements

    .. math::

        \\varphi(x) = s(x)\\,\\min_{y \\in \\partial\\Omega} \\lVert x - y \\rVert.
    """

    vertices: npt.NDArray[np.float64]
    faces: npt.NDArray[np.int64]
    _sdf: _PySDF
    bounds_min: npt.NDArray[np.float64]
    bounds_max: npt.NDArray[np.float64]

    def __init__(
        self,
        vertices: npt.NDArray[np.float64],
        faces: npt.NDArray[np.int64],
        bounds_min: npt.NDArray[np.float64] | None = None,
        bounds_max: npt.NDArray[np.float64] | None = None,
    ) -> None:
        v = np.asarray(vertices, dtype=np.float64)
        f = np.asarray(faces, dtype=np.int64)
        self.vertices = v
        self.faces = f
        self._sdf = _PySDF(v, f)

        if bounds_min is None or bounds_max is None:
            vmin = v.min(axis=0)
            vmax = v.max(axis=0)
        else:
            vmin = np.asarray(bounds_min, dtype=np.float64)
            vmax = np.asarray(bounds_max, dtype=np.float64)

        self.bounds_min = vmin
        self.bounds_max = vmax

    @classmethod
    def from_mesh(cls, mesh: trimesh.Trimesh) -> "PySDFProvider":
        """Construct a :class:`PySDFProvider` from a :class:`trimesh.Trimesh`.

        The mesh's vertex positions and faces are passed directly to
        :class:`pysdf.SDF`. The bounding box is taken from
        :attr:`mesh.bounds`.
        """

        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        bounds_min, bounds_max = np.asarray(mesh.bounds[0]), np.asarray(mesh.bounds[1])
        return cls(vertices=vertices, faces=faces, bounds_min=bounds_min, bounds_max=bounds_max)

    # ------------------------------------------------------------------
    # SDFProvider implementation
    # ------------------------------------------------------------------

    def signed_distance(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the signed distance :math:`\\varphi(x)` at query points.

        This is a fully vectorised wrapper around the underlying
        :class:`pysdf.SDF` callable. Input is converted to a contiguous
        ``float64`` array of shape ``(N, 3)``.
        """

        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
        return np.asarray(self._sdf(pts), dtype=np.float64)

    def bounding_box(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the axis-aligned bounding box of the mesh."""

        return self.bounds_min.copy(), self.bounds_max.copy()

