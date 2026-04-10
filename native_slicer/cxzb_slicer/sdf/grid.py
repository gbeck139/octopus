"""
Voxelised signed distance field (SDF) grid and interpolation utilities.

Given an abstract :class:`~cxzb_slicer.sdf.provider.SDFProvider` that
evaluates the signed distance :math:`\\varphi(x)` at arbitrary points,
this module constructs a regular Cartesian grid representation
``SDFGrid`` and provides trilinear interpolation for sub-voxel queries
as well as gradients via finite differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt

from .provider import SDFProvider
from cxzb_slicer.core.types import SlicerConfig


nparray_f64 = npt.NDArray[np.float64]


@dataclass(slots=True)
class SDFGrid:
    """Regular Cartesian grid storing samples of an SDF.

    Attributes
    ----------
    values:
        3D array of shape ``(Nx, Ny, Nz)`` containing samples of the
        signed distance field :math:`\\varphi(x)` at grid points.
    origin:
        World-space coordinates of the voxel at index ``(0, 0, 0)``.
    spacing:
        Grid spacing ``(dx, dy, dz)`` in world units.

    The continuous SDF is approximated by trilinear interpolation between
    these samples. For a query point with fractional index coordinates
    :math:`(i + t_x, j + t_y, k + t_z)`, the interpolant is

    .. math::

        \\varphi(x) = \\sum_{a,b,c\\in\\{0,1\\}}
            w_{abc} \\, \\varphi_{i+a, j+b, k+c},

    where the weights are

    .. math::

        w_{abc} =
            (1-a - (-1)^a t_x)\\,
            (1-b - (-1)^b t_y)\\,
            (1-c - (-1)^c t_z).
    """

    values: nparray_f64
    origin: nparray_f64
    spacing: nparray_f64

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return the grid resolution ``(Nx, Ny, Nz)``."""

        return tuple(int(s) for s in self.values.shape)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_sdf(
        cls,
        provider: SDFProvider,
        resolution: int = 64,
        padding: float = 0.0,
        config: SlicerConfig | None = None,
    ) -> "SDFGrid":
        """Sample an :class:`SDFProvider` into a regular grid.

        Parameters
        ----------
        provider:
            SDF backend used to evaluate :math:`\\varphi(x)`.
        resolution:
            Number of voxels per dimension, resulting in a cubic grid of
            shape ``(resolution, resolution, resolution)``.
        padding:
            Extra margin (in world units) added around the provider's
            bounding box before constructing the grid.
        config:
            Optional :class:`~cxzb_slicer.core.types.SlicerConfig`. It is
            accepted for future extensions but not used directly here.
        """

        del config  # currently unused but kept for interface stability

        bounds_min, bounds_max = provider.bounding_box()
        bounds_min = np.asarray(bounds_min, dtype=np.float64) - float(padding)
        bounds_max = np.asarray(bounds_max, dtype=np.float64) + float(padding)

        # Build a cubic domain with uniform spacing so that the largest
        # side of the bounding box maps to the full index range.
        extent = bounds_max - bounds_min
        max_extent = float(np.max(extent))
        if max_extent <= 0.0:
            raise ValueError("Invalid SDF bounding box: zero or negative extent.")

        n = int(resolution)
        if n < 2:
            raise ValueError("resolution must be at least 2.")

        h = max_extent / float(n - 1)
        spacing = np.array([h, h, h], dtype=np.float64)

        # Expand the smaller dimensions so the final domain is cubic.
        domain_max = bounds_min + spacing * float(n - 1)

        origin = bounds_min.copy()

        # Generate grid coordinates.
        i = np.arange(n, dtype=np.float64)
        coords = origin[:, None] + spacing[:, None] * i[None, :]
        xs, ys, zs = coords[0], coords[1], coords[2]
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)

        phi = provider.signed_distance(pts).reshape((n, n, n))
        return cls(values=phi, origin=origin, spacing=spacing)

    # ------------------------------------------------------------------
    # Trilinear interpolation and gradients
    # ------------------------------------------------------------------

    def _fractional_indices(self, points: nparray_f64) -> Tuple[nparray_f64, nparray_f64, nparray_f64]:
        """Return fractional indices (ix, iy, iz) for world-space points."""

        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {pts.shape}")

        rel = (pts - self.origin[None, :]) / self.spacing[None, :]
        return rel[:, 0], rel[:, 1], rel[:, 2]

    def _trilinear_sample(self, volume: nparray_f64, points: nparray_f64) -> nparray_f64:
        """Evaluate a scalar volume at arbitrary points via trilinear interpolation."""

        ix, iy, iz = self._fractional_indices(points)

        nx, ny, nz = volume.shape
        # Clamp indices to interior so that i1, j1, k1 are valid.
        i0 = np.clip(np.floor(ix).astype(np.int64), 0, nx - 2)
        j0 = np.clip(np.floor(iy).astype(np.int64), 0, ny - 2)
        k0 = np.clip(np.floor(iz).astype(np.int64), 0, nz - 2)

        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        tx = ix - i0
        ty = iy - j0
        tz = iz - k0

        v000 = volume[i0, j0, k0]
        v100 = volume[i1, j0, k0]
        v010 = volume[i0, j1, k0]
        v110 = volume[i1, j1, k0]
        v001 = volume[i0, j0, k1]
        v101 = volume[i1, j0, k1]
        v011 = volume[i0, j1, k1]
        v111 = volume[i1, j1, k1]

        c00 = v000 * (1.0 - tx) + v100 * tx
        c10 = v010 * (1.0 - tx) + v110 * tx
        c01 = v001 * (1.0 - tx) + v101 * tx
        c11 = v011 * (1.0 - tx) + v111 * tx

        c0 = c00 * (1.0 - ty) + c10 * ty
        c1 = c01 * (1.0 - ty) + c11 * ty

        return c0 * (1.0 - tz) + c1 * tz

    def sample(self, points: nparray_f64) -> nparray_f64:
        """Evaluate :math:`\\varphi(x)` at arbitrary query points.

        This uses trilinear interpolation of :attr:`values` according to
        the formula detailed in the class docstring.
        """

        return self._trilinear_sample(self.values, points)

    def gradient(self, points: nparray_f64) -> nparray_f64:
        """Approximate :math:`\\nabla\\varphi(x)` at arbitrary query points.

        The gradient components are first approximated on the grid using
        central differences,

        .. math::

            \\partial_x \\varphi_{i,j,k} \\approx
                \\frac{\\varphi_{i+1,j,k} - \\varphi_{i-1,j,k}}{2\\,\\Delta x},

        with forward/backward differences at the domain boundaries.
        Each component volume is then evaluated at the query points using
        the same trilinear interpolation as :meth:`sample`.
        """

        dx, dy, dz = self.spacing
        phi = self.values

        gx = np.empty_like(phi)
        gy = np.empty_like(phi)
        gz = np.empty_like(phi)

        # Central differences in the interior, forward/backward at edges.
        gx[1:-1, :, :] = (phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * dx)
        gx[0, :, :] = (phi[1, :, :] - phi[0, :, :]) / dx
        gx[-1, :, :] = (phi[-1, :, :] - phi[-2, :, :]) / dx

        gy[:, 1:-1, :] = (phi[:, 2:, :] - phi[:, :-2, :]) / (2.0 * dy)
        gy[:, 0, :] = (phi[:, 1, :] - phi[:, 0, :]) / dy
        gy[:, -1, :] = (phi[:, -1, :] - phi[:, -2, :]) / dy

        gz[:, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2.0 * dz)
        gz[:, :, 0] = (phi[:, :, 1] - phi[:, :, 0]) / dz
        gz[:, :, -1] = (phi[:, :, -1] - phi[:, :, -2]) / dz

        pts = np.asarray(points, dtype=np.float64)
        gx_s = self._trilinear_sample(gx, pts)
        gy_s = self._trilinear_sample(gy, pts)
        gz_s = self._trilinear_sample(gz, pts)

        return np.stack((gx_s, gy_s, gz_s), axis=-1)

