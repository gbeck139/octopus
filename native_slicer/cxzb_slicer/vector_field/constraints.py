"""
Constraint providers for vector field solving.

The vector field solver expects a per-vertex **guidance field** ``g`` and
an optional boolean mask of **Dirichlet constraints** which enforce
``k[i] = g[i]`` at selected vertices.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import numpy.typing as npt

from cxzb_slicer.core.types import SlicerConfig

nparray_f64 = npt.NDArray[np.float64]
nparray_i64 = npt.NDArray[np.int64]


def _normalize(v: nparray_f64, eps: float = 1e-12) -> nparray_f64:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.clip(n, eps, None)
    return v / n


class ConstraintProvider(ABC):
    """Abstract base class for producing guidance and Dirichlet masks."""

    @abstractmethod
    def guidance_and_mask(
        self,
        verts: nparray_f64,
        faces: nparray_i64,
        config: SlicerConfig,
    ) -> Tuple[nparray_f64, npt.NDArray[np.bool_]]:
        """Return (guidance g, fixed_mask) for the given mesh."""


@dataclass(slots=True)
class SphereRadialConstraints(ConstraintProvider):
    """Radial guidance field g(x)=normalize(x-center), used for tests."""

    center: nparray_f64 = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    fix_caps: bool = False
    cap_fraction: float = 0.05

    def guidance_and_mask(
        self,
        verts: nparray_f64,
        faces: nparray_i64,  # noqa: ARG002
        config: SlicerConfig,  # noqa: ARG002
    ) -> Tuple[nparray_f64, npt.NDArray[np.bool_]]:
        v = np.asarray(verts, dtype=np.float64)
        c = np.asarray(self.center, dtype=np.float64)
        g = _normalize(v - c[None, :])

        fixed = np.zeros((v.shape[0],), dtype=bool)
        if self.fix_caps:
            z = v[:, 2]
            zmin = float(np.min(z))
            zmax = float(np.max(z))
            band = self.cap_fraction * (zmax - zmin)
            fixed = (z <= zmin + band) | (z >= zmax - band)

        return g, fixed


@dataclass(slots=True)
class BasicOverhangConstraints(ConstraintProvider):
    """Minimal mesh constraints stub.

    This implementation is intentionally conservative for now:

    - Provides vertical guidance on top/bottom caps (z percentiles).
    - Elsewhere uses a constant downward direction.

    A more complete version will incorporate face normals and an overhang
    threshold to set guidance directions on steep regions.
    """

    cap_fraction: float = 0.05

    def guidance_and_mask(
        self,
        verts: nparray_f64,
        faces: nparray_i64,  # noqa: ARG002
        config: SlicerConfig,  # noqa: ARG002
    ) -> Tuple[nparray_f64, npt.NDArray[np.bool_]]:
        v = np.asarray(verts, dtype=np.float64)
        g = np.zeros_like(v)

        # Default: downward.
        g[:, 2] = -1.0

        z = v[:, 2]
        zmin = float(np.min(z))
        zmax = float(np.max(z))
        band = self.cap_fraction * (zmax - zmin)
        bottom = z <= zmin + band
        top = z >= zmax - band

        # Enforce vertical at caps for adhesion/finish.
        g[bottom, :] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        g[top, :] = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        fixed = bottom | top
        return g, fixed

