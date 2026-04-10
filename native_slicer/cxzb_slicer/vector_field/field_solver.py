"""
Vector field solver for smooth tool orientation.

The research document proposes solving a Poisson-style alignment problem
with a (connection) Laplacian. For a practical baseline, we implement a
real-valued per-vertex vector field solve using the robust mesh Laplacian
and mass matrix from :mod:`robust_laplacian`:

.. math::

    (L + s M) k = s M g,

where

- :math:`L` is a (positive semi-definite) Laplace matrix,
- :math:`M` is the lumped mass matrix,
- :math:`g` is a guidance direction field (unit vectors),
- :math:`s > 0` controls the smoothness/alignment trade-off,
- :math:`k` is the solved direction field (unit vectors after normalization).

Finally, direction vectors :math:`k = (k_x, k_y, k_z)` convert to machine
axis angles using the convention from the research note:

.. math::

    B = \\arccos(k_z), \\qquad C = \\operatorname{atan2}(-k_x, k_y).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt
import robust_laplacian
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from cxzb_slicer.core.types import SlicerConfig

nparray_f64 = npt.NDArray[np.float64]
nparray_i64 = npt.NDArray[np.int64]


def _normalize(v: nparray_f64, eps: float = 1e-12) -> nparray_f64:
    """Normalize vectors row-wise."""

    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.clip(n, eps, None)
    return v / n


def vector_to_angles(k: nparray_f64) -> Tuple[nparray_f64, nparray_f64]:
    """Convert direction vectors to (B, C) angles.

    Implements:

    .. math::

        B = \\arccos(k_z), \\qquad C = \\operatorname{atan2}(-k_x, k_y).
    """

    kk = np.asarray(k, dtype=np.float64)
    if kk.ndim != 2 or kk.shape[1] != 3:
        raise ValueError(f"k must have shape (N, 3), got {kk.shape}")

    kz = np.clip(kk[:, 2], -1.0, 1.0)
    b = np.arccos(kz)
    c = np.arctan2(-kk[:, 0], kk[:, 1])
    return b, c


class VectorFieldSolver(ABC):
    """Abstract base class for vector field solvers."""

    @abstractmethod
    def solve(
        self,
        verts: nparray_f64,
        faces: nparray_i64,
        guidance: nparray_f64,
        fixed_mask: npt.NDArray[np.bool_],
        config: SlicerConfig,
        s: float,
    ) -> nparray_f64:
        """Solve for a smooth direction field k at mesh vertices."""


@dataclass(slots=True)
class PoissonVectorFieldSolver(VectorFieldSolver):
    """Poisson-style alignment solver using robust mesh Laplacian."""

    mollify_factor: float = 1e-5

    def solve(
        self,
        verts: nparray_f64,
        faces: nparray_i64,
        guidance: nparray_f64,
        fixed_mask: npt.NDArray[np.bool_],
        config: SlicerConfig,  # noqa: ARG002 (reserved for later constraints/limits)
        s: float,
    ) -> nparray_f64:
        v = np.asarray(verts, dtype=np.float64)
        f = np.asarray(faces, dtype=np.int64)
        g = _normalize(np.asarray(guidance, dtype=np.float64))
        mask = np.asarray(fixed_mask, dtype=bool)

        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"verts must have shape (V, 3), got {v.shape}")
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError(f"faces must have shape (F, 3), got {f.shape}")
        if g.shape != v.shape:
            raise ValueError(f"guidance must have shape {v.shape}, got {g.shape}")
        if mask.shape != (v.shape[0],):
            raise ValueError(f"fixed_mask must have shape ({v.shape[0]},), got {mask.shape}")

        if float(s) <= 0.0:
            raise ValueError("s must be positive.")

        L, M = robust_laplacian.mesh_laplacian(v, f, mollify_factor=self.mollify_factor)
        # System matrix
        A = L + float(s) * M

        # RHS is s M g (component-wise).
        rhs = float(s) * (M @ g)

        if np.any(mask):
            # Apply Dirichlet constraints by row replacement.
            # This is simplest in LIL format and fine for test-scale meshes.
            A_lil = A.tolil(copy=True)
            for i in np.flatnonzero(mask):
                A_lil.rows[i] = [i]
                A_lil.data[i] = [1.0]
            A = A_lil.tocsr()
            rhs = rhs.copy()
            rhs[mask, :] = g[mask, :]

        # Solve three independent sparse systems.
        kx = spla.spsolve(A, rhs[:, 0])
        ky = spla.spsolve(A, rhs[:, 1])
        kz = spla.spsolve(A, rhs[:, 2])

        k = np.stack((kx, ky, kz), axis=1).astype(np.float64, copy=False)
        k = _normalize(k)
        return k

