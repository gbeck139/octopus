"""Tests for SDFProvider implementations and SDFGrid."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt

from cxzb_slicer.sdf.grid import SDFGrid
from cxzb_slicer.sdf.provider import SDFProvider


class UnitSphereSDF(SDFProvider):
    """Analytic SDF for a unit sphere centred at the origin."""

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        r = np.linalg.norm(pts, axis=1)
        return r - 1.0

    def bounding_box(self):
        r = 1.0
        return np.array([-r, -r, -r], dtype=np.float64), np.array([r, r, r], dtype=np.float64)


def test_sdfgrid_sample_matches_analytic_sphere() -> None:
    """SDFGrid.sample approximates the analytic unit sphere SDF."""

    provider = UnitSphereSDF()
    grid = SDFGrid.from_sdf(provider, resolution=32, padding=0.1, config=None)

    rng = np.random.default_rng(123)
    points = rng.uniform(-1.2, 1.2, size=(200, 3))

    phi_analytic = provider.signed_distance(points)
    phi_grid = grid.sample(points)

    # Allow an error proportional to grid spacing.
    h = float(grid.spacing[0])
    npt.assert_allclose(phi_grid, phi_analytic, atol=3.0 * h)


def test_sdfgrid_gradient_matches_sphere_normals() -> None:
    """SDFGrid.gradient approximates x/‖x‖ on the unit sphere surface."""

    provider = UnitSphereSDF()
    grid = SDFGrid.from_sdf(provider, resolution=40, padding=0.1, config=None)

    rng = np.random.default_rng(456)
    # Sample points near the unit sphere.
    dirs = rng.normal(size=(100, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    points = dirs  # radius ≈ 1

    grad = grid.gradient(points)
    grad_norm = grad / np.linalg.norm(grad, axis=1, keepdims=True)

    dots = np.einsum("ij,ij->i", grad_norm, dirs)
    # Gradients should be closely aligned with analytic normals.
    assert np.all(dots > 0.95)


def test_sdfgrid_domain_covers_bounding_box_with_padding() -> None:
    """Grid origin/spacing yields a domain that covers the SDF bounding box."""

    provider = UnitSphereSDF()
    padding = 0.2
    grid = SDFGrid.from_sdf(provider, resolution=16, padding=padding, config=None)

    bmin, bmax = provider.bounding_box()
    origin = grid.origin
    extent = grid.spacing * (np.array(grid.shape, dtype=np.float64) - 1.0)
    domain_max = origin + extent

    # The analytic bounding box plus padding should lie inside [origin, domain_max].
    npt.assert_array_less(origin - 1e-8, bmin - padding)
    npt.assert_array_less(bmax + padding, domain_max + 1e-8)

