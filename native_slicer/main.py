"""
Native CX-ZB 4-axis slicer CLI.

Runs the full implicit slicing pipeline: load STL, compute SDF, generate
layer surfaces, extract contours, build toolpaths, apply inverse kinematics,
compute extrusion, and write G-code. Does not require PrusaSlicer.
"""

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

INPUT_MODELS_DIR = os.path.join(base_dir, "input_models")
OUTPUT_GCODE_DIR = os.path.join(base_dir, "output_gcode")


def run_slicer_pipeline(stl_path: str, model_name: str, config,
                        sdf_resolution: int = 64):
    """Run the full native slicing pipeline."""
    import numpy as np
    import trimesh
    from cxzb_slicer.sdf.pysdf_provider import PySDFProvider
    from cxzb_slicer.sdf.grid import SDFGrid
    from cxzb_slicer.layers.planar import PlanarLayerGenerator, planar_level_function
    from cxzb_slicer.layers.contour_extraction import extract_contours_for_layer
    from cxzb_slicer.toolpath import DefaultContourPathConverter
    from cxzb_slicer.toolpath.contour_to_path import merge_layer_toolpaths
    from cxzb_slicer.kinematics import CXZBKinematics, unwrap_c_axis
    from cxzb_slicer.extrusion import TiltExtrusionCompensator, SimpleRetractionPlanner
    from cxzb_slicer.gcode import DuetGCodeWriter

    # --- Load mesh ---
    print("\nLoading mesh...\n", flush=True)
    mesh = trimesh.load(stl_path)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a single mesh, got {type(mesh)}")
    print(f"Mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces", flush=True)

    # --- Build SDF ---
    print("\nBuilding signed distance field...\n", flush=True)
    sdf_provider = PySDFProvider.from_mesh(mesh)
    sdf_grid = SDFGrid.from_sdf(sdf_provider, resolution=sdf_resolution)

    # --- Generate layers ---
    print("\nGenerating layer surfaces...\n", flush=True)
    layer_gen = PlanarLayerGenerator()
    layers = layer_gen.generate_surfaces(sdf_provider, config)
    print(f"Generated {len(layers)} layers (height={config.layer_height}mm)", flush=True)

    # --- Extract contours and build toolpaths ---
    print("\nExtracting contours and building toolpaths...\n", flush=True)
    converter = DefaultContourPathConverter()
    layer_toolpaths = []

    for layer in layers:
        contours = extract_contours_for_layer(sdf_grid, layer, planar_level_function)
        if not contours:
            continue
        tp = converter.contours_to_toolpath(contours, layer, normals=None, config=config)
        if len(tp) > 0:
            layer_toolpaths.append(tp)

    if not layer_toolpaths:
        print("Warning: no toolpath points generated", flush=True)
        return

    toolpath = merge_layer_toolpaths(layer_toolpaths)
    print(f"Toolpath: {len(toolpath)} points across {len(layer_toolpaths)} layers", flush=True)

    # --- Inverse kinematics ---
    print("\nApplying inverse kinematics...\n", flush=True)
    ik = CXZBKinematics(config.tool_offset_L)
    x_m, z_m, c = ik.world_to_machine(
        toolpath["x"], toolpath["y"], toolpath["z"], toolpath["b"]
    )
    toolpath["x_m"] = x_m
    toolpath["z_m"] = z_m
    toolpath["c"] = unwrap_c_axis(c)

    # --- Extrusion ---
    print("\nComputing extrusion...\n", flush=True)
    compensator = TiltExtrusionCompensator()
    toolpath["e"] = compensator.compute_e_per_point(toolpath, config)
    retraction = SimpleRetractionPlanner()
    toolpath = retraction.apply(toolpath, config)

    # --- Write G-code ---
    os.makedirs(OUTPUT_GCODE_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_GCODE_DIR, f"{model_name}.gcode")

    print(f"\nWriting G-code to {output_path}...\n", flush=True)
    writer = DuetGCodeWriter()
    gcode = writer.write(toolpath, config)
    with open(output_path, "w") as f:
        f.write(gcode)

    print(f"\nDone! Output G-code: {output_path}\n", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Native CX-ZB 4-axis slicer (implicit slicing, no PrusaSlicer required)"
    )
    parser.add_argument("--stl", required=True, help="Path to input STL file")
    parser.add_argument("--model", required=True, help="Model name (used for output filenames)")
    parser.add_argument("--layer-height", type=float, default=0.2,
                        help="Layer height in mm (default: 0.2)")
    parser.add_argument("--nozzle-diameter", type=float, default=0.4,
                        help="Nozzle diameter in mm (default: 0.4)")
    parser.add_argument("--tool-offset", type=float, default=50.0,
                        help="Pivot-to-nozzle distance L in mm (default: 50.0)")
    parser.add_argument("--sdf-resolution", type=int, default=64,
                        help="SDF voxel grid resolution (default: 64)")
    args = parser.parse_args()

    from cxzb_slicer.core.types import SlicerConfig
    config = SlicerConfig(
        layer_height=args.layer_height,
        nozzle_diameter=args.nozzle_diameter,
        tool_offset_L=args.tool_offset,
    )

    run_slicer_pipeline(args.stl, args.model, config, sdf_resolution=args.sdf_resolution)


if __name__ == "__main__":
    main()
