"""Tests for toolpath generation (Phase 3)."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt

from cxzb_slicer.core.types import Layer, SlicerConfig, TOOLPATH_DTYPE
from cxzb_slicer.layers.contour_extraction import extract_contours_for_layer
from cxzb_slicer.layers.planar import PlanarLayerGenerator, planar_level_function
from cxzb_slicer.sdf.grid import SDFGrid
from cxzb_slicer.sdf.provider import SDFProvider
from cxzb_slicer.toolpath.contour_to_path import DefaultContourPathConverter
from cxzb_slicer.toolpath.infill import RectilinearInfill
from cxzb_slicer.toolpath.path_optimizer import NearestNeighborPathOptimizer, merge_optimized_paths


class CubeSDF(SDFProvider):
    """Analytic SDF for a cube of half-size 1 at origin."""

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        q = np.abs(pts) - 1.0
        outside = np.maximum(q, 0.0)
        inside = np.minimum(np.maximum(q[:, 0], np.maximum(q[:, 1], q[:, 2])), 0.0)
        return np.linalg.norm(outside, axis=1) + inside

    def bounding_box(self):
        s = 1.0
        return np.array([-s, -s, -s], dtype=np.float64), np.array([s, s, s], dtype=np.float64)


def _point_in_polygon_even_odd(points_xy: np.ndarray, poly_xy: np.ndarray) -> np.ndarray:
    """Vectorized even-odd point-in-polygon for a single closed polygon."""

    pts = np.asarray(points_xy, dtype=np.float64)
    poly = np.asarray(poly_xy, dtype=np.float64)
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])

    x = pts[:, 0]
    y = pts[:, 1]

    x0 = poly[:-1, 0]
    y0 = poly[:-1, 1]
    x1 = poly[1:, 0]
    y1 = poly[1:, 1]

    # For each edge, test if ray crosses it.
    y_between = ((y0[None, :] > y[:, None]) != (y1[None, :] > y[:, None]))
    x_intersect = x0[None, :] + (y[:, None] - y0[None, :]) * (x1[None, :] - x0[None, :]) / (
        (y1 - y0)[None, :] + 1e-12
    )
    crosses = y_between & (x[:, None] < x_intersect)
    return (np.sum(crosses, axis=1) % 2) == 1


def test_contour_to_path_cube_planar_layer() -> None:
    """Contour-to-path produces closed extrude loops with correct normals/indices."""

    sdf = CubeSDF()
    cfg = SlicerConfig(layer_height=0.1)
    grid = SDFGrid.from_sdf(sdf, resolution=40, padding=0.1, config=None)

    gen = PlanarLayerGenerator()
    layers = gen.generate_surfaces(sdf, cfg)
    levels = np.array([ly.level for ly in layers], dtype=np.float64)
    idx = int(np.argmin(np.abs(levels - 0.0)))
    layer = layers[idx]

    contours = extract_contours_for_layer(grid, layer, planar_level_function)
    assert contours

    tp = DefaultContourPathConverter().contours_to_toolpath(contours, layer=layer, normals=None, config=cfg)
    assert tp.dtype == TOOLPATH_DTYPE
    assert len(tp) > 0
    assert np.all(tp["layer_index"] == layer.index)

    # Normals are unit and vertical.
    n = np.stack([tp["nx"], tp["ny"], tp["nz"]], axis=1)
    npt.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-8)
    npt.assert_allclose(n[:, 0], 0.0, atol=1e-8)
    npt.assert_allclose(n[:, 1], 0.0, atol=1e-8)
    npt.assert_allclose(n[:, 2], 1.0, atol=1e-8)

    # At least one extrude move exists; first/last point close for the first contour.
    assert np.any(tp["move_type"] == "extrude")
    xyz = np.stack([tp["x"], tp["y"], tp["z"]], axis=1)
    assert np.linalg.norm(xyz[0] - xyz[-1]) < 0.5 or np.any(tp["move_type"] == "travel")


def test_rectilinear_infill_points_inside_cube_slice() -> None:
    """Rectilinear infill should lie inside the cube contour polygon."""

    sdf = CubeSDF()
    cfg = SlicerConfig(layer_height=0.1, extrusion_width=0.2)
    grid = SDFGrid.from_sdf(sdf, resolution=50, padding=0.1, config=None)

    layer = Layer(index=0, surface_id="planar", level=0.0)
    contours = extract_contours_for_layer(grid, layer, planar_level_function)
    assert contours

    # Use the largest contour as boundary.
    boundary = max(contours, key=lambda c: c.shape[0])

    infill = RectilinearInfill().generate_infill(contours, layer_index=layer.index, config=cfg, angle=0.0)
    assert infill.dtype == TOOLPATH_DTYPE
    assert len(infill) > 0

    # Endpoints of clipped segments lie on the boundary; validate midpoints instead.
    pts_xy = np.stack([infill["x"], infill["y"]], axis=1)
    n_pairs = (len(pts_xy) // 2) * 2
    mid_xy = 0.5 * (pts_xy[:n_pairs:2] + pts_xy[1:n_pairs:2])

    inside = _point_in_polygon_even_odd(mid_xy, boundary[:, :2])
    assert np.mean(inside) > 0.95  # allow small boundary/interp noise

    # Scanline spacing roughly equals extrusion width.
    ys = infill["y"][infill["move_type"] == "extrude"]
    unique_y = np.unique(np.round(ys, 3))
    if unique_y.size >= 2:
        dy = np.diff(unique_y)
        assert np.isclose(np.median(dy), cfg.extrusion_width, atol=0.05)


def test_nearest_neighbor_optimizer_reduces_travel_distance() -> None:
    """Nearest-neighbor optimizer should not increase travel between paths."""

    def make_path(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        p = np.zeros(2, dtype=TOOLPATH_DTYPE)
        p["x"] = [start[0], end[0]]
        p["y"] = [start[1], end[1]]
        p["z"] = [start[2], end[2]]
        p["nx"] = 0.0
        p["ny"] = 0.0
        p["nz"] = 1.0
        p["tilt_angle"] = 0.0
        p["layer_index"] = 0
        p["move_type"] = "extrude"
        return p

    paths = [
        make_path(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        make_path(np.array([10.0, 0.0, 0.0]), np.array([11.0, 0.0, 0.0])),
        make_path(np.array([5.0, 5.0, 0.0]), np.array([6.0, 5.0, 0.0])),
    ]

    def travel_distance(seq):
        d = 0.0
        for a, b in zip(seq[:-1], seq[1:]):
            a_end = np.array([a["x"][-1], a["y"][-1], a["z"][-1]])
            b_start = np.array([b["x"][0], b["y"][0], b["z"][0]])
            d += float(np.linalg.norm(a_end - b_start))
        return d

    naive = travel_distance(paths)
    opt = NearestNeighborPathOptimizer().optimize_layer_paths(paths)
    optimized = travel_distance(opt)

    assert optimized <= naive + 1e-9
    merged = merge_optimized_paths(opt)
    assert merged.dtype == TOOLPATH_DTYPE

