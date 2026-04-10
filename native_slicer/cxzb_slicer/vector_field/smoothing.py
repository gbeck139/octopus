"""
Vector field smoothing and motor-limit enforcement.

This module implements:

- Implicit diffusion smoothing of a per-vertex direction field:

  .. math::

      (M + \\tau L) k = M k^0

  solved component-wise and re-normalized.

- Along-path angular velocity enforcement for B/C axes. For consecutive
  toolpath points separated by arc length ``Δs`` at feed ``v``, enforce:

  .. math::

      |ΔB| \\le \\omega_B \\frac{Δs}{v},\\quad
      |ΔC| \\le \\omega_C \\frac{Δs}{v}.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt
import robust_laplacian
import scipy.sparse.linalg as spla

from cxzb_slicer.kinematics.singularity import unwrap_c_axis

nparray_f64 = npt.NDArray[np.float64]
nparray_i64 = npt.NDArray[np.int64]


def _normalize(v: nparray_f64, eps: float = 1e-12) -> nparray_f64:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.clip(n, eps, None)
    return v / n


class FieldSmoother(ABC):
    """Abstract base class for field smoothing."""

    @abstractmethod
    def smooth(self, verts: nparray_f64, faces: nparray_i64, k0: nparray_f64, tau: float) -> nparray_f64:
        """Return a smoothed field."""


@dataclass(slots=True)
class ImplicitDiffusionSmoother(FieldSmoother):
    """Implicit diffusion smoothing using robust mesh Laplacian."""

    mollify_factor: float = 1e-5

    def smooth(self, verts: nparray_f64, faces: nparray_i64, k0: nparray_f64, tau: float) -> nparray_f64:
        v = np.asarray(verts, dtype=np.float64)
        f = np.asarray(faces, dtype=np.int64)
        k = _normalize(np.asarray(k0, dtype=np.float64))

        if float(tau) <= 0.0:
            return k

        L, M = robust_laplacian.mesh_laplacian(v, f, mollify_factor=self.mollify_factor)
        A = M + float(tau) * L
        rhs = M @ k

        kx = spla.spsolve(A, rhs[:, 0])
        ky = spla.spsolve(A, rhs[:, 1])
        kz = spla.spsolve(A, rhs[:, 2])
        kout = np.stack((kx, ky, kz), axis=1).astype(np.float64, copy=False)
        return _normalize(kout)


def enforce_angular_velocity_limits_along_path(
    positions: nparray_f64,
    b: nparray_f64,
    c: nparray_f64,
    feed_mm_s: float,
    max_b_omega_deg_s: float,
    max_c_omega_deg_s: float,
) -> Tuple[nparray_f64, nparray_f64]:
    """Clamp per-segment ΔB and ΔC according to motor angular-velocity limits.

    This operates on a 1D sequence (a toolpath polyline). `c` is unwrapped
    before limiting.
    """

    pos = np.asarray(positions, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64).copy()
    cc = unwrap_c_axis(np.asarray(c, dtype=np.float64))

    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {pos.shape}")
    if bb.shape != (pos.shape[0],) or cc.shape != (pos.shape[0],):
        raise ValueError("b and c must be shape (N,)")

    v = float(feed_mm_s)
    if v <= 0.0:
        raise ValueError("feed_mm_s must be positive.")

    w_b = np.deg2rad(float(max_b_omega_deg_s))
    w_c = np.deg2rad(float(max_c_omega_deg_s))

    ds = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    # Avoid division by zero: if ds==0, set allowed delta to 0.
    dt = ds / v
    max_db = w_b * dt
    max_dc = w_c * dt

    for i in range(1, pos.shape[0]):
        db = bb[i] - bb[i - 1]
        dc = cc[i] - cc[i - 1]

        lim_db = max_db[i - 1]
        lim_dc = max_dc[i - 1]

        if lim_db <= 0.0:
            bb[i] = bb[i - 1]
        else:
            bb[i] = bb[i - 1] + np.clip(db, -lim_db, lim_db)

        if lim_dc <= 0.0:
            cc[i] = cc[i - 1]
        else:
            cc[i] = cc[i - 1] + np.clip(dc, -lim_dc, lim_dc)

    return bb, cc

