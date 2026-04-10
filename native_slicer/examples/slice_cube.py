"""
End-to-end example: slice a cube, generate toolpaths, and visualize.

Pipeline:

1. Analytic cube SDF -> :class:`SDFGrid`.
2. Planar layers -> contours via implicit-implicit intersection.
3. Contours -> toolpaths (with rectilinear infill and path optimization).
4. Populate kinematics fields (B/C, X_m/Z_m).
5. Compute extrusion & simple retraction.
6. Visualize mesh, contours, and toolpaths using :class:`SlicerViewer`.
7. Optionally write G-code.
"""

from __future__ import annotations

import numpy as np

from cxzb_slicer.core.types import Layer, SlicerConfig, TOOLPATH_DTYPE
from cxzb_slicer.extrusion.compensator import TiltExtrusionCompensator
from cxzb_slicer.extrusion.retraction import SimpleRetractionPlanner
from cxzb_slicer.gcode.validator import BasicGCodeValidator, populate_kinematics_fields
from cxzb_slicer.gcode.writer import DuetGCodeWriter
from cxzb_slicer.layers.contour_extraction import extract_contours_for_layer
from cxzb_slicer.layers.planar import PlanarLayerGenerator, planar_level_function
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

    # Generate planar layers.
    layer_gen = PlanarLayerGenerator()
    layers = layer_gen.generate_surfaces(sdf, cfg)

    # For demo, use a subset of layers.
    layers = [ly for ly in layers if 1.0 <= ly.level <= 30.0]

    all_toolpaths = []
    for layer in layers:
        contours = extract_contours_for_layer(grid, layer, planar_level_function)
        if not contours:
            continue

        normals = None  # planar generator implies (0,0,1)
        tp_layer = build_layer_toolpath(
            contours=contours,
            layer=layer,
            normals=normals,
            config=cfg,
            infill_generator=RectilinearInfill(),
            optimizer=NearestNeighborPathOptimizer(),
            infill_angle=0.0,
        )
        if len(tp_layer) == 0:
            continue
        all_toolpaths.append(tp_layer)

    if not all_toolpaths:
        raise RuntimeError("No toolpaths generated.")

    toolpath = np.concatenate(all_toolpaths, axis=0).astype(TOOLPATH_DTYPE, copy=False)

    # Populate kinematics and extrusion.
    toolpath = populate_kinematics_fields(toolpath, cfg)
    toolpath["e"] = TiltExtrusionCompensator().compute_e_per_point(toolpath, cfg)
    toolpath = SimpleRetractionPlanner().apply(toolpath, cfg)

    # Visualize.
    viewer = SlicerViewer(title="CX-ZB Slice Cube")
    viewer.show_sdf_isosurface(grid, level=0.0, color="white", opacity=0.2)

    # Show contours for the middle layer.
    mid_layer = layers[len(layers) // 2]
    mid_contours = extract_contours_for_layer(grid, mid_layer, planar_level_function)
    viewer.show_layer_contours(mid_contours, color="cyan")

    # Show toolpath colored by tilt angle with orientation glyphs.
    viewer.show_toolpath(toolpath, color_by="tilt_angle", cmap="viridis")
    viewer.show_orientation_glyphs(toolpath, step=50, scale=3.0, color="red")
    viewer.show()

    # Optionally, write and validate G-code.
    gcode = DuetGCodeWriter().write(toolpath, cfg)
    errors = BasicGCodeValidator(position_tol_mm=0.05).validate(toolpath, gcode, cfg)
    if errors:
        print("Validation issues:")
        for e in errors:
            print(" -", e)
    else:
        print("G-code validated successfully.")


if __name__ == "__main__":
    main()

