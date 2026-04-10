"""
PyVista-based visualization for CX-ZB slicer outputs.

This viewer focuses on three core objects:

- The object surface (mesh or SDF isosurface).
- Layer contours.
- Toolpaths with per-point attributes (tilt, feed, move_type, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import numpy.typing as npt
import pyvista as pv

from cxzb_slicer.core.types import TOOLPATH_DTYPE
from cxzb_slicer.sdf.grid import SDFGrid

nparray_f64 = npt.NDArray[np.float64]


@dataclass
class SlicerViewer:
    """High-level viewer for mesh, contours, and toolpaths."""

    title: str = "CX-ZB Slicer Viewer"

    def __post_init__(self) -> None:
        self.plotter = pv.Plotter(title=self.title)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def show_sdf_isosurface(
        self,
        grid: SDFGrid,
        level: float = 0.0,
        color: str = "white",
        opacity: float = 0.3,
    ) -> None:
        """Show an isosurface of φ(x)=level from an :class:`SDFGrid`."""

        values = grid.values - float(level)
        # Use ImageData with point-centered scalars for contouring.
        vol = pv.ImageData()
        vol.dimensions = np.array(values.shape, dtype=int)
        vol.origin = grid.origin + 0.5 * grid.spacing
        vol.spacing = grid.spacing
        vol["phi"] = values.ravel(order="F")
        surf = vol.contour([0.0], scalars="phi")
        self.plotter.add_mesh(surf, color=color, opacity=opacity, show_edges=False)

    def show_mesh(self, mesh: pv.PolyData, color: str = "lightgray") -> None:
        """Show an existing :class:`pyvista.PolyData` mesh."""

        self.plotter.add_mesh(mesh, color=color, show_edges=True, opacity=0.6)

    # ------------------------------------------------------------------
    # Contours & toolpaths
    # ------------------------------------------------------------------

    def show_layer_contours(
        self,
        contours: Sequence[nparray_f64],
        color: str = "cyan",
        line_width: float = 2.0,
    ) -> None:
        """Show layer contours as polylines in world coordinates."""

        for c in contours:
            pts = np.asarray(c, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
                continue
            poly = pv.lines_from_points(pts, close=False)
            self.plotter.add_mesh(poly, color=color, line_width=line_width)

    def show_toolpath(
        self,
        toolpath: np.ndarray,
        color_by: str = "tilt_angle",
        cmap: str = "viridis",
        line_width: float = 3.0,
    ) -> None:
        """Show a toolpath colored by a scalar field from ``TOOLPATH_DTYPE``."""

        tp = np.asarray(toolpath, dtype=TOOLPATH_DTYPE)
        if len(tp) == 0:
            return

        pts = np.stack([tp["x"], tp["y"], tp["z"]], axis=1).astype(np.float64)
        poly = pv.lines_from_points(pts, close=False)

        scalars = None
        if color_by in tp.dtype.names:
            scalars = tp[color_by].astype(np.float64)
        elif color_by == "move_type":
            # Map move types to integers: extrude=2, travel=1, retract=0.
            mv = tp["move_type"].astype(str)
            scalars = np.where(mv == "extrude", 2.0, np.where(mv == "travel", 1.0, 0.0))

        if scalars is not None:
            poly["scalars"] = scalars
            self.plotter.add_mesh(poly, scalars="scalars", line_width=line_width, cmap=cmap)
        else:
            self.plotter.add_mesh(poly, color="yellow", line_width=line_width)

    def show_orientation_glyphs(
        self,
        toolpath: np.ndarray,
        step: int = 10,
        scale: float = 5.0,
        color: str = "red",
    ) -> None:
        """Show orientation glyphs (arrows) along the toolpath normals."""

        tp = np.asarray(toolpath, dtype=TOOLPATH_DTYPE)
        if len(tp) == 0:
            return

        idx = np.arange(0, len(tp), max(int(step), 1))
        pts = np.stack([tp["x"], tp["y"], tp["z"]], axis=1).astype(np.float64)[idx]
        dirs = np.stack([tp["nx"], tp["ny"], tp["nz"]], axis=1).astype(np.float64)[idx]

        cloud = pv.PolyData(pts)
        cloud["vectors"] = dirs
        arrows = cloud.glyph(orient="vectors", scale=False, factor=scale)
        self.plotter.add_mesh(arrows, color=color)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def show(self) -> None:
        """Render the current scene."""

        self.plotter.show()

