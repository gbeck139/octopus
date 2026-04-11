"""Headless renderer for 4-axis G-code.

Dumps a fixed set of canned views as PNGs so an automated agent can
"see" the slicer output without launching the interactive visualizer.

Reuses GCodeVisualizer from better_visualizer.py for parsing and scene
construction, then renders with pv.Plotter(off_screen=True).

Usage:
    python render_gcode.py --gcode path/to/file.gcode --out tools/vision_cache/foo/latest
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from better_visualizer import GCodeVisualizer  # noqa: E402


WINDOW_SIZE = (1024, 768)
FILMSTRIP_FRAMES = 6


def _render(poly, out_path, camera, scalars, cmap, clim, title):
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    plotter.set_background("black")

    show_sbar = scalars is not None
    plotter.add_mesh(
        poly,
        scalars=scalars,
        cmap=cmap if show_sbar else None,
        color=None if show_sbar else "lime",
        clim=clim,
        line_width=3,
        render_lines_as_tubes=True,
        show_scalar_bar=show_sbar,
        scalar_bar_args={"title": title} if show_sbar else None,
    )
    plotter.add_text(title, position="upper_left", font_size=10, color="white")

    if camera == "iso":
        plotter.view_isometric()
    elif camera == "top":
        plotter.view_xy()
    elif camera == "front":
        plotter.view_yz()
    elif camera == "side":
        plotter.view_xz()
    plotter.reset_camera()

    plotter.screenshot(str(out_path))
    plotter.close()


def render_all(gcode_path, out_dir):
    viz = GCodeVisualizer(gcode_path)
    if len(viz.points) < 2:
        print(f"nothing to render: {gcode_path}")
        return

    viz.create_scene()
    poly = viz.poly
    extrude = poly.threshold(value=[0.9, 1.1], scalars="MoveType", preference="cell")
    if extrude.n_cells == 0:
        print("no extrusion moves found")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    z_data = extrude.cell_data["ZHeight"]
    b_data = extrude.cell_data["Tilt"]
    z_clim = [float(z_data.min()), float(z_data.max())]
    b_clim = [float(b_data.min()), float(b_data.max())]

    views = [
        ("iso.png",   "iso",   "ZHeight", "viridis",  z_clim, "Isometric (Z-height)"),
        ("top.png",   "top",   "ZHeight", "viridis",  z_clim, "Top (Z-height)"),
        ("front.png", "front", "ZHeight", "viridis",  z_clim, "Front (Z-height)"),
        ("side.png",  "side",  "ZHeight", "viridis",  z_clim, "Side (Z-height)"),
        ("tilt.png",  "iso",   "Tilt",    "coolwarm", b_clim, "Isometric (B-tilt)"),
        ("xz_section.png", "side", "Tilt", "coolwarm", b_clim, "Side (B-tilt)"),
    ]

    for fname, camera, scalars, cmap, clim, title in views:
        _render(extrude, out_dir / fname, camera, scalars, cmap, clim, title)
        print(f"  wrote {fname}")

    num_cells = extrude.n_cells
    frames = np.linspace(num_cells // FILMSTRIP_FRAMES, num_cells, FILMSTRIP_FRAMES, dtype=int)
    for i, cutoff in enumerate(frames):
        sub = extrude.threshold(value=[0, int(cutoff)], scalars="Index", preference="cell")
        if sub.n_cells == 0:
            continue
        _render(
            sub,
            out_dir / f"filmstrip_{i:02d}.png",
            "iso",
            "ZHeight",
            "viridis",
            z_clim,
            f"Filmstrip {i + 1}/{FILMSTRIP_FRAMES} ({cutoff}/{num_cells} moves)",
        )
        print(f"  wrote filmstrip_{i:02d}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcode", required=True)
    parser.add_argument("--out", required=True, help="Output directory for PNGs")
    args = parser.parse_args()

    if not os.path.exists(args.gcode):
        print(f"not found: {args.gcode}", file=sys.stderr)
        sys.exit(1)

    render_all(args.gcode, args.out)


if __name__ == "__main__":
    main()
