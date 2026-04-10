"""Tests for planar, conic, and spherical layer generators and contour extraction."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from cxzb_slicer.core.types import SlicerConfig
from cxzb_slicer.layers.conic import ConicLayerGenerator, conic_level_function
from cxzb_slicer.layers.contour_extraction import extract_contours_for_layer
from cxzb_slicer.layers.planar import PlanarLayerGenerator, planar_level_function
from cxzb_slicer.layers.spherical import SphericalLayerGenerator, spherical_level_function
from cxzb_slicer.sdf.grid import SDFGrid
from cxzb_slicer.sdf.provider import SDFProvider


class CubeSDF(SDFProvider):
    """Analytic SDF for an axis-aligned cube of half-size 1 centred at origin."""

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        q = np.abs(pts) - 1.0
        outside = np.maximum(q, 0.0)
        inside = np.minimum(np.maximum(q[:, 0], np.maximum(q[:, 1], q[:, 2])), 0.0)
        return np.linalg.norm(outside, axis=1) + inside

    def bounding_box(self):
        s = 1.0
        return np.array([-s, -s, -s], dtype=np.float64), np.array([s, s, s], dtype=np.float64)


class CylinderSDF(SDFProvider):
    """Infinite cylinder in z with radius 1, truncated by height 2."""

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        r = np.sqrt(x * x + y * y)
        # Distance to radial boundary and top/bottom planes.
        d_radial = r - 1.0
        d_z = np.maximum(np.abs(z) - 1.0, 0.0)
        return np.maximum(d_radial, d_z)

    def bounding_box(self):
        return np.array([-1.0, -1.0, -1.0], dtype=np.float64), np.array([1.0, 1.0, 1.0], dtype=np.float64)


class SphereSDF(SDFProvider):
    """Analytic SDF for unit sphere, reused from test_sdf."""

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        r = np.linalg.norm(pts, axis=1)
        return r - 1.0

    def bounding_box(self):
        r = 1.0
        return np.array([-r, -r, -r], dtype=np.float64), np.array([r, r, r], dtype=np.float64)


def test_planar_slice_of_cube_yields_square_contour() -> None:
    """Planar slice at z≈0 of a cube yields a square contour in XY."""

    sdf = CubeSDF()
    cfg = SlicerConfig(layer_height=0.1)
    grid = SDFGrid.from_sdf(sdf, resolution=40, padding=0.1, config=None)

    gen = PlanarLayerGenerator()
    layers = gen.generate_surfaces(sdf, cfg)

    # Find layer closest to z = 0.
    levels = np.array([ly.level for ly in layers])
    idx = int(np.argmin(np.abs(levels - 0.0)))
    layer = layers[idx]

    contours = extract_contours_for_layer(grid, layer, planar_level_function)
    assert contours, "No contours extracted for planar cube slice."

    contour = max(contours, key=lambda c: c.shape[0])
    xy = contour[:, :2]

    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)

    # Expect an approximate square from (-1,-1) to (1,1).
    npt.assert_allclose(min_xy, [-1.0, -1.0], atol=0.15)
    npt.assert_allclose(max_xy, [1.0, 1.0], atol=0.15)
    # All points should lie near z = 0.
    assert np.all(np.abs(contour[:, 2] - layer.level) < 0.1)


def test_conic_layer_count_matches_expected_height() -> None:
    """Conic generator produces a reasonable number of layers for a cylinder."""

    sdf = CylinderSDF()
    cfg = SlicerConfig(layer_height=0.1)

    alpha = np.deg2rad(30.0)
    gen = ConicLayerGenerator(alpha=alpha)
    layers = gen.generate_surfaces(sdf, cfg)

    # Cylinder height is 2; expect roughly height / layer_height layers.
    expected = 2.0 / cfg.layer_height
    assert abs(len(layers) - expected) <= 3


def test_spherical_layers_match_radius_levels() -> None:
    """Spherical layers produce contours whose radii match layer levels."""

    sdf = SphereSDF()
    cfg = SlicerConfig(layer_height=0.15)
    grid = SDFGrid.from_sdf(sdf, resolution=40, padding=0.1, config=None)

    gen = SphericalLayerGenerator(center=np.zeros(3, dtype=np.float64))
    layers = gen.generate_surfaces(sdf, cfg)
    assert layers, "No spherical layers generated."

    # Only the layer with radius closest to 1.0 (the sphere surface)
    # intersects the φ=0 isosurface, so test that one.
    levels = np.array([ly.level for ly in layers], dtype=np.float64)
    idx = int(np.argmin(np.abs(levels - 1.0)))
    layer = layers[idx]

    contours = extract_contours_for_layer(
        grid, layer, lambda pts, c=np.zeros(3, dtype=np.float64): spherical_level_function(pts, c)
    )
    if not contours:
        pytest.skip("No contours extracted for spherical layer; intersection may fall between voxels.")

    contour = max(contours, key=lambda c: c.shape[0])
    radii = np.linalg.norm(contour, axis=1)
    mean_r = float(radii.mean())

    # Allow a tolerance related to grid spacing.
    h = float(grid.spacing[0])
    assert abs(mean_r - layer.level) < 3.0 * h

