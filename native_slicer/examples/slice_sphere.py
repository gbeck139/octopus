"""
Demo: spherical layers on a unit sphere with radial orientations.

Shows non-planar layering and normals that follow the sphere radius.
"""

from __future__ import annotations

import numpy as np

from cxzb_slicer.core.types import Layer, SlicerConfig, TOOLPATH_DTYPE
from cxzb_slicer.extrusion.compensator import TiltExtrusionCompensator
from cxzb_slicer.extrusion.retraction import SimpleRetractionPlanner
from cxzb_slicer.gcode.validator import BasicGCodeValidator, populate_kinematics_fields
from cxzb_slicer.gcode.writer import DuetGCodeWriter
from cxzb_slicer.layers.contour_extraction import extract_contours_for_layer
from cxzb_slicer.layers.spherical import (
    SphericalLayerGenerator,
    spherical_level_function,
    spherical_normal,
)
from cxzb_slicer.sdf.grid import SDFGrid
from cxzb_slicer.sdf.provider import SDFProvider
from cxzb_slicer.toolpath.infill import RectilinearInfill
from cxzb_slicer.toolpath.path_optimizer import NearestNeighborPathOptimizer, build_layer_toolpath
from cxzb_slicer.visualization import SlicerViewer


class SphereSDF(SDFProvider):
    """Analytic SDF for a unit sphere, centred at origin."""

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        r = np.linalg.norm(pts, axis=1)
        return r - 1.0

    def bounding_box(self):
        r = 1.0
        return np.array([-r, -r, -r], dtype=np.float64), np.array([r, r, r], dtype=np.float64)


def main() -> None:
    cfg = SlicerConfig(layer_height=0.2, extrusion_width=0.25, filament_diameter=1.75, nozzle_diameter=0.4)

    sdf = SphereSDF()
    grid = SDFGrid.from_sdf(sdf, resolution=64, padding=0.2, config=cfg)

    center = np.zeros(3, dtype=np.float64)
    layer_gen = SphericalLayerGenerator(center=center)
    layers = layer_gen.generate_surfaces(sdf, cfg)
    # Don't filter further; generator already picks a sensible radius.

    all_toolpaths = []
    for layer in layers:
        contours = extract_contours_for_layer(
            grid, layer, lambda pts, c=center: spherical_level_function(pts, c)
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

        pts = np.stack([tp_layer["x"], tp_layer["y"], tp_layer["z"]], axis=1).astype(np.float64)
        n = spherical_normal(pts, center)
        tp_layer["nx"] = n[:, 0]
        tp_layer["ny"] = n[:, 1]
        tp_layer["nz"] = n[:, 2]
        tp_layer["tilt_angle"] = np.arccos(np.clip(tp_layer["nz"].astype(np.float64), -1.0, 1.0))

        all_toolpaths.append(tp_layer.astype(TOOLPATH_DTYPE, copy=False))

    if not all_toolpaths:
        print("No spherical contours found at this grid resolution; try increasing resolution or padding.")
        return

    toolpath = np.concatenate(all_toolpaths, axis=0).astype(TOOLPATH_DTYPE, copy=False)

    toolpath = populate_kinematics_fields(toolpath, cfg)
    toolpath["e"] = TiltExtrusionCompensator().compute_e_per_point(toolpath, cfg)
    toolpath = SimpleRetractionPlanner().apply(toolpath, cfg)

    viewer = SlicerViewer(title="CX-ZB Spherical Layers Demo")
    viewer.show_sdf_isosurface(grid, level=0.0, color="white", opacity=0.15)

    mid_layer = layers[len(layers) // 2]
    mid_contours = extract_contours_for_layer(
        grid, mid_layer, lambda pts, c=center: spherical_level_function(pts, c)
    )
    viewer.show_layer_contours(mid_contours, color="cyan")

    viewer.show_toolpath(toolpath, color_by="tilt_angle", cmap="viridis")
    viewer.show_orientation_glyphs(toolpath, step=20, scale=0.2, color="red")
    viewer.show()

    gcode = DuetGCodeWriter().write(toolpath, cfg)
    errors = BasicGCodeValidator(position_tol_mm=0.05).validate(toolpath, gcode, cfg)
    if errors:
        print("Validation issues (sphere):")
        for e in errors:
            print(" -", e)
    else:
        print("Spherical G-code validated successfully.")


if __name__ == "__main__":
    main()

