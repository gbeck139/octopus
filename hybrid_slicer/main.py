import numpy as np
import pyvista as pv
import subprocess
import os
import sys
import shutil
import argparse

import analyze
import deform
import reform

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

INPUT_MODELS_DIR = os.path.join(base_dir, "input_models")
OUTPUT_MODELS_DIR = os.path.join(base_dir, "output_models")
INPUT_GCODE_DIR = os.path.join(base_dir, "input_gcode")
OUTPUT_GCODE_DIR = os.path.join(base_dir, "output_gcode")
PRUSA_CONFIG_DIR = os.path.join(base_dir, "prusa_slicer")


def run_slicer_pipeline(stl_path_input: str, MODEL_NAME: str, slicer_path: str,
                        max_overhang: float = 45.0,
                        rotation_multiplier: float = 1.5,
                        smoothing_iterations: int = 30,
                        grid_resolution: float = 1.0,
                        max_pos_rotation: float = 45.0,
                        max_neg_rotation: float = -45.0):
    """
    Full hybrid slicer pipeline:
    1. Load STL
    2. Analyze surface to compute per-vertex rotation field
    3. Deform mesh using rotation field
    4. Slice with PrusaSlicer
    5. Back-transform G-code
    """

    os.makedirs(INPUT_MODELS_DIR, exist_ok=True)
    local_stl_path = os.path.join(INPUT_MODELS_DIR, f"{MODEL_NAME}.stl")
    stl_path_input = os.path.abspath(stl_path_input)
    if os.path.abspath(local_stl_path) != stl_path_input:
        shutil.copyfile(stl_path_input, local_stl_path)

    # --- Step 1: Load and prepare mesh ---
    print("\nLoading mesh...\n", flush=True)
    mesh = deform.load_mesh(MODEL_NAME)

    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()

    # Subdivide to target density (same as deform.py will do, but we need
    # the subdivided mesh for analysis before deform gets it)
    TARGET_POINT_COUNT = 1000000
    while mesh.n_points < TARGET_POINT_COUNT:
        mesh = mesh.subdivide(1, subfilter='linear')

    # Center the mesh (matching deform.py's centering)
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    mesh.points -= np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, zmin])

    # --- Step 2: Analyze surface ---
    print("\nAnalyzing surface geometry...\n", flush=True)
    rotation_field = analyze.compute_rotation_field(
        mesh,
        max_overhang=max_overhang,
        rotation_multiplier=rotation_multiplier,
        smoothing_iterations=smoothing_iterations,
        max_pos_rotation=max_pos_rotation,
        max_neg_rotation=max_neg_rotation,
    )

    nonzero_count = np.count_nonzero(rotation_field)
    print(f"Rotation field: {nonzero_count}/{len(rotation_field)} vertices need rotation", flush=True)
    print(f"  Range: [{np.rad2deg(rotation_field.min()):.1f}, {np.rad2deg(rotation_field.max()):.1f}] deg", flush=True)

    # --- Step 3: Deform mesh ---
    print("\nDeforming model with hybrid rotation field...\n", flush=True)

    # Pass the already-prepared mesh directly (same vertices as analysis used)
    deformed_mesh, transform_params = deform.deform_mesh(
        mesh,
        scale=1,
        rotation_field=rotation_field,
        grid_resolution=grid_resolution,
        pre_prepared=True,
    )
    deform.save_deformed_mesh(deformed_mesh, transform_params, MODEL_NAME)

    # --- Step 4: Slice with PrusaSlicer ---
    stl_path = os.path.join(OUTPUT_MODELS_DIR, f"{MODEL_NAME}_deformed.stl")
    output_gcode = os.path.join(INPUT_GCODE_DIR, f"{MODEL_NAME}_deformed.gcode")
    ini_path = os.path.join(PRUSA_CONFIG_DIR, "my_printer_config.ini")

    os.makedirs(INPUT_GCODE_DIR, exist_ok=True)

    print("\n***PRUSA*** planar slicer is running...\n", flush=True)
    subprocess.run([
        slicer_path,
        "--load", ini_path,
        "--ensure-on-bed",
        "--export-gcode",
        stl_path,
        "--output", output_gcode
    ], check=True)

    print(f"G-code exported to {output_gcode}", flush=True)
    print("\n***PRUSA*** planar slicing finished\n", flush=True)

    # --- Step 5: Reform G-code ---
    print("\nReforming G-code with hybrid back-transform...\n", flush=True)
    reform.load_gcode_and_undeform(MODEL_NAME, transform_params)

    os.makedirs(OUTPUT_GCODE_DIR, exist_ok=True)

    output_path = os.path.join(OUTPUT_GCODE_DIR, f"{MODEL_NAME}_reformed.gcode")
    print(f"\nDone! Output G-code: {output_path}\n", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Cartesian/Radial Non-Planar Slicer"
    )
    parser.add_argument("--stl", required=True, help="Path to input STL file")
    parser.add_argument("--model", required=True, help="Model name (used for output filenames)")
    parser.add_argument("--prusa", required=True, help="Path to PrusaSlicer executable")
    parser.add_argument("--max-overhang", type=float, default=45.0,
                        help="Overhang angle threshold in degrees (default: 45)")
    parser.add_argument("--rotation-multiplier", type=float, default=1.5,
                        help="Scale factor for computed rotations (default: 1.5)")
    parser.add_argument("--smoothing-iterations", type=int, default=30,
                        help="Laplacian smoothing passes (default: 30)")
    parser.add_argument("--grid-resolution", type=float, default=1.0,
                        help="XY grid spacing in mm for rotation lookup (default: 1.0)")
    parser.add_argument("--max-pos-rotation", type=float, default=45.0,
                        help="Maximum positive rotation in degrees (default: 45)")
    parser.add_argument("--max-neg-rotation", type=float, default=-45.0,
                        help="Maximum negative rotation in degrees (default: -45)")
    args = parser.parse_args()

    run_slicer_pipeline(
        args.stl, args.model, args.prusa,
        max_overhang=args.max_overhang,
        rotation_multiplier=args.rotation_multiplier,
        smoothing_iterations=args.smoothing_iterations,
        grid_resolution=args.grid_resolution,
        max_pos_rotation=args.max_pos_rotation,
        max_neg_rotation=args.max_neg_rotation,
    )


if __name__ == "__main__":
    main()
