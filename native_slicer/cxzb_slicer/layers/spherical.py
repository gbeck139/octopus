"""
Spherical layer surfaces of the form

.. math::

    f_{\\text{sph}}(x) = \\lVert x - x_0 \\rVert = c,

where :math:`x_0` is the sphere centre. Iso-surfaces are concentric
spheres used for radial layering.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from cxzb_slicer.core.types import Layer, SlicerConfig, TOOLPATH_DTYPE
from cxzb_slicer.layers.generator import LayerGenerator
from cxzb_slicer.sdf.provider import SDFProvider


def spherical_level_function(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Evaluate :math:`f_{\\text{sph}}(x) = \\lVert x - x_0 \\rVert`."""

    pts = np.asarray(points, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    d = pts - c[None, :]
    return np.linalg.norm(d, axis=1)


def spherical_normal(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Return radial unit normals for spherical layers.

    The gradient of :math:`f_{\\text{sph}}(x) = \\lVert x - x_0 \\rVert`
    is

    .. math::

        \\nabla f_{\\text{sph}} = \\frac{x - x_0}{\\lVert x - x_0 \\rVert},

    with the convention that at :math:`x = x_0` the direction is
    :math:`(0, 0, 1)`.
    """

    pts = np.asarray(points, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")

    d = pts - c[None, :]
    lengths = np.linalg.norm(d, axis=1, keepdims=True)
    eps = 1e-9
    mask = lengths[:, 0] > eps

    n = np.zeros_like(d)
    n[mask] = d[mask] / lengths[mask]
    n[~mask, 2] = 1.0  # define centre direction as +Z
    return n


class SphericalLayerGenerator(LayerGenerator):
    """Spherical layer generator centred at ``center`` (or mesh midpoint)."""

    def __init__(self, center: Optional[np.ndarray] = None) -> None:
        self._center = None if center is None else np.asarray(center, dtype=np.float64)

    def generate_surfaces(self, sdf: SDFProvider, config: SlicerConfig) -> List[Layer]:
        bounds_min, bounds_max = sdf.bounding_box()
        if self._center is None:
            center = 0.5 * (bounds_min + bounds_max)
        else:
            center = self._center

        # Estimate radial extent from bounding box corners.
        corners = np.array(
            [
                [bounds_min[0], bounds_min[1], bounds_min[2]],
                [bounds_min[0], bounds_min[1], bounds_max[2]],
                [bounds_min[0], bounds_max[1], bounds_min[2]],
                [bounds_min[0], bounds_max[1], bounds_max[2]],
                [bounds_max[0], bounds_min[1], bounds_min[2]],
                [bounds_max[0], bounds_min[1], bounds_max[2]],
                [bounds_max[0], bounds_max[1], bounds_min[2]],
                [bounds_max[0], bounds_max[1], bounds_max[2]],
            ],
            dtype=np.float64,
        )

        radii = spherical_level_function(corners, center)
        r_min = float(np.min(radii))
        r_max = float(np.max(radii))

        dr = float(config.layer_height)
        if dr <= 0.0:
            raise ValueError("config.layer_height must be positive.")

        # Start slightly above zero to avoid degenerate radius at centre.
        start = max(dr, r_min)
        n_layers = int(np.floor((r_max - start) / dr)) + 1
        levels = start + dr * np.arange(n_layers, dtype=np.float64)

        layers: List[Layer] = []
        for idx, c_val in enumerate(levels):
            layers.append(
                Layer(
                    index=idx,
                    surface_id="spherical",
                    level=float(c_val),
                    points=np.empty(0, dtype=TOOLPATH_DTYPE),
                )
            )
        return layers

