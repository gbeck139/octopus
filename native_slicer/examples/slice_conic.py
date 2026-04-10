"""
Demo: conic layers on a cube with tilted orientations.

This example is similar to ``slice_cube.py`` but uses conic layers
defined by

.. math::

    f_{cone}(x,y,z) = z - r \\tan\\alpha,

so that layer normals tilt outward with radius. This produces a clear
visual difference in the viewer: arrows no longer point straight up.
"""

from __future__ import annotations

import numpy as np

from cxzb_slicer.core.types import SlicerConfig, TOOLPATH_DTYPE
from cxzb_slicer.extrusion.compensator import TiltExtrusionCompensator
from cxzb_slicer.extrusion.retraction import SimpleRetractionPlanner
from cxzb_slicer.gcode.validator import BasicGCodeValidator, populate_kinematics_fields
from cxzb_slicer.gcode.writer import DuetGCodeWriter
from cxzb_slicer.layers.conic import ConicLayerGenerator, conic_level_function, conic_normal
from cxzb_slicer.layers.contour_extraction import extract_contours_for_layer
from cxzb_slicer.sdf.grid import SDFGrid
from cxzb_slicer.sdf.provider import SDFProvider
from cxzb_slicer.toolpath.infill import RectilinearInfill
from cxzb_slicer.toolpath.path_optimizer import NearestNeighborPathOptimizer, build_layer_toolpath
from cxzb_slicer.visualization import SlicerViewer


class CubeSDF(SDFProvider):
    """Analytic SDF for an axis-aligned cube of half-size 20mm centred at origin."""

    def __init__(self, half_size: float = 20.0) -> None:
        self.half_size = float(half_size)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        q = np.abs(pts) - self.half_size
        outside = np.maximum(q, 0.0)
        inside = np.minimum(np.maximum(q[:, 0], np.maximum(q[:, 1], q[:, 2])), 0.0)
        return np.linalg.norm(outside, axis=1) + inside

    def bounding_box(self):
        s = self.half_size
        return np.array([-s, -s, 0.0], dtype=np.float64), np.array([s, s, 2 * s], dtype=np.float64)


def main() -> None:
    cfg = SlicerConfig(layer_height=0.4, extrusion_width=0.45, filament_diameter=1.75, nozzle_diameter=0.4)

    sdf = CubeSDF(half_size=20.0)
    grid = SDFGrid.from_sdf(sdf, resolution=64, padding=1.0, config=cfg)

    alpha = np.deg2rad(30.0)
    layer_gen = ConicLayerGenerator(alpha=alpha)
    layers = layer_gen.generate_surfaces(sdf, cfg)

    # Focus on layers where the cone intersects a mid-height region.
    layers = [ly for ly in layers if 5.0 <= ly.level <= 30.0]

    all_toolpaths = []
    for layer in layers:
        contours = extract_contours_for_layer(
            grid, layer, lambda pts, a=alpha: conic_level_function(pts, a)
        )
        if not contours:
            continue

        tp_layer = build_layer_toolpath(
            contours=contours,
            layer=layer,
            normals=None,
            config=cfg,
            infill_generator=RectilinearInfill(),
            optimizer=NearestNeighborPathOptimizer(),
            infill_angle=0.0,
        )
        if len(tp_layer) == 0:
            continue

        # Override normals with conic normals for visualization.
        pts = np.stack([tp_layer["x"], tp_layer["y"], tp_layer["z"]], axis=1).astype(np.float64)
        n = conic_normal(pts, alpha)
        tp_layer["nx"] = n[:, 0]
        tp_layer["ny"] = n[:, 1]
        tp_layer["nz"] = n[:, 2]
        tp_layer["tilt_angle"] = np.arccos(np.clip(tp_layer["nz"].astype(np.float64), -1.0, 1.0))

        all_toolpaths.append(tp_layer.astype(TOOLPATH_DTYPE, copy=False))

    if not all_toolpaths:
        raise RuntimeError("No toolpaths generated for conic demo.")

    toolpath = np.concatenate(all_toolpaths, axis=0).astype(TOOLPATH_DTYPE, copy=False)

    # Populate kinematics and extrusion.
    toolpath = populate_kinematics_fields(toolpath, cfg)
    toolpath["e"] = TiltExtrusionCompensator().compute_e_per_point(toolpath, cfg)
    toolpath = SimpleRetractionPlanner().apply(toolpath, cfg)

    viewer = SlicerViewer(title="CX-ZB Conic Layers Demo")
    viewer.show_sdf_isosurface(grid, level=0.0, color="white", opacity=0.15)

    mid_layer = layers[len(layers) // 2]
    mid_contours = extract_contours_for_layer(
        grid, mid_layer, lambda pts, a=alpha: conic_level_function(pts, a)
    )
    viewer.show_layer_contours(mid_contours, color="cyan")

    viewer.show_toolpath(toolpath, color_by="tilt_angle", cmap="plasma")
    viewer.show_orientation_glyphs(toolpath, step=40, scale=3.0, color="red")
    viewer.show()

    gcode = DuetGCodeWriter().write(toolpath, cfg)
    errors = BasicGCodeValidator(position_tol_mm=0.1).validate(toolpath, gcode, cfg)
    if errors:
        print("Validation issues (conic):")
        for e in errors:
            print(" -", e)
    else:
        print("Conic G-code validated successfully.")


if __name__ == "__main__":
    main()

