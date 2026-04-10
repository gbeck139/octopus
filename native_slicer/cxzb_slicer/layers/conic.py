"""
Conic layer surfaces of the form

.. math::

    f_{\\text{cone}}(x, y, z) = z - r\\tan\\alpha = c,

where :math:`r = \\sqrt{x^2 + y^2}` and :math:`\\alpha` is the cone
half-angle. Each iso-surface :math:`f_{\\text{cone}}(x) = c_k` is a
cone whose apex lies on the Z-axis.
"""

from __future__ import annotations

from typing import List

import numpy as np

from cxzb_slicer.core.types import Layer, SlicerConfig, TOOLPATH_DTYPE
from cxzb_slicer.layers.generator import LayerGenerator
from cxzb_slicer.sdf.provider import SDFProvider


def conic_level_function(points: np.ndarray, alpha: float) -> np.ndarray:
    """Evaluate :math:`f_{\\text{cone}}(x, y, z) = z - r\\tan\\alpha`.

    Parameters
    ----------
    points:
        Array of shape ``(N, 3)`` with Cartesian coordinates.
    alpha:
        Cone half-angle :math:`\\alpha` in radians.
    """

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x * x + y * y)
    return z - r * np.tan(alpha)


def conic_normal(points: np.ndarray, alpha: float) -> np.ndarray:
    """Return the unit normal corresponding to :math:`f_{\\text{cone}}`.

    The gradient of the conic layer function is

    .. math::

        \\nabla f_{\\text{cone}} =
            \\left(-\\frac{x}{r}\\tan\\alpha,
                  -\\frac{y}{r}\\tan\\alpha,
                  1\\right),

    where :math:`r = \\sqrt{x^2 + y^2}`. At :math:`r = 0` the direction
    is defined as the vertical vector :math:`(0, 0, 1)`. The resulting
    vectors are normalised to unit length.
    """

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")

    x, y, _ = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x * x + y * y)
    eps = 1e-9

    nx = np.zeros_like(x)
    ny = np.zeros_like(y)
    nz = np.ones_like(x)

    mask = r > eps
    scale = -np.tan(alpha) / np.where(mask, r, 1.0)
    nx[mask] = x[mask] * scale[mask]
    ny[mask] = y[mask] * scale[mask]

    n = np.stack((nx, ny, nz), axis=-1)
    lengths = np.linalg.norm(n, axis=1, keepdims=True)
    lengths = np.clip(lengths, eps, None)
    return n / lengths


class ConicLayerGenerator(LayerGenerator):
    """Conic layer generator parameterised by half-angle ``alpha``."""

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def generate_surfaces(self, sdf: SDFProvider, config: SlicerConfig) -> List[Layer]:
        bounds_min, bounds_max = sdf.bounding_box()
        # Approximate f-cone range over the bounding box using its corners.
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

        f_vals = conic_level_function(corners, self.alpha)
        f_min = float(f_vals.min())
        f_max = float(f_vals.max())

        dz = float(config.layer_height)
        if dz <= 0.0:
            raise ValueError("config.layer_height must be positive.")

        # Since ∂f/∂z = 1, stepping c by layer_height yields approximately
        # uniform vertical spacing between level sets.
        n_layers = int(np.floor((f_max - f_min) / dz)) + 1
        levels = f_min + dz * np.arange(n_layers, dtype=np.float64)

        layers: List[Layer] = []
        for idx, c in enumerate(levels):
            layers.append(
                Layer(
                    index=idx,
                    surface_id="conic",
                    level=float(c),
                    points=np.empty(0, dtype=TOOLPATH_DTYPE),
                )
            )
        return layers

