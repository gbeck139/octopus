"""
Planar layer surfaces of the form :math:`z = c`.

The implicit function for planar layers is

.. math::

    f_{\\text{planar}}(x, y, z) = z,

so that the layer surface satisfies :math:`f_{\\text{planar}}(x) = c`.
The gradient is constant,

.. math::

    \\nabla f_{\\text{planar}} = (0, 0, 1),

which provides the layer normal used later for toolpath orientation.
"""

from __future__ import annotations

from typing import List

import numpy as np

from cxzb_slicer.core.types import Layer, SlicerConfig, TOOLPATH_DTYPE
from cxzb_slicer.layers.generator import LayerGenerator
from cxzb_slicer.sdf.provider import SDFProvider


def planar_level_function(points: np.ndarray) -> np.ndarray:
    """Return :math:`f_{\\text{planar}}(x, y, z) = z` for each point."""

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    return pts[:, 2]


def planar_normal(points: np.ndarray) -> np.ndarray:
    """Return the unit normal :math:`(0, 0, 1)` for all points."""

    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    out = np.zeros((n, 3), dtype=np.float64)
    out[:, 2] = 1.0
    return out


class PlanarLayerGenerator(LayerGenerator):
    """Standard planar layers ``z = c`` spaced by ``config.layer_height``."""

    def generate_surfaces(self, sdf: SDFProvider, config: SlicerConfig) -> List[Layer]:
        bounds_min, bounds_max = sdf.bounding_box()
        z_min = float(bounds_min[2])
        z_max = float(bounds_max[2])

        dz = float(config.layer_height)
        if dz <= 0.0:
            raise ValueError("config.layer_height must be positive.")

        n_layers = int(np.floor((z_max - z_min) / dz)) + 1
        levels = z_min + dz * np.arange(n_layers, dtype=np.float64)

        layers: List[Layer] = []
        for idx, c in enumerate(levels):
            layers.append(
                Layer(
                    index=idx,
                    surface_id="planar",
                    level=float(c),
                    points=np.empty(0, dtype=TOOLPATH_DTYPE),
                )
            )
        return layers

