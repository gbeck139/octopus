import subprocess
import os
import sys
import shutil
import argparse

import deform
import reform
import merge

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


def run_slicer_pipeline(stl_path_input, model_name, slicer_path, z_split, cone_angle):
    """
    Full conic slicing pipeline:
    1. Load STL mesh
    2. Split at z_split into lower and upper portions
    3. Deform upper portion with conic transformation
    4. Planar slice both portions with PrusaSlicer
    5. Back-transform upper G-code to 4-axis coordinates
    6. Merge both G-codes into final output
    """
    os.makedirs(INPUT_MODELS_DIR, exist_ok=True)

    # Copy input STL to input_models/
    local_stl_path = os.path.join(INPUT_MODELS_DIR, f"{model_name}.stl")
    shutil.copyfile(stl_path_input, local_stl_path)

    # --- Stage 1: Deform ---
    print("\n=== Stage 1: Loading and splitting mesh ===\n", flush=True)

    mesh = deform.load_mesh(model_name)
    lower_mesh, upper_mesh, mesh_center_xy = deform.split_mesh(mesh, z_split)

    print(f"Lower mesh: {lower_mesh.n_points} points, {lower_mesh.n_cells} faces")
    print(f"Upper mesh: {upper_mesh.n_points} points, {upper_mesh.n_cells} faces")

    print(f"\nDeforming upper mesh with cone angle {cone_angle}°...\n", flush=True)

    upper_deformed, transform_params = deform.deform_upper_mesh(upper_mesh, cone_angle)
    # Store z_split in transform params for the merge step
    transform_params["z_split"] = float(z_split)
    deform.save_meshes(lower_mesh, upper_deformed, transform_params, model_name)

    # --- Stage 2: Planar slice both portions with PrusaSlicer ---
    ini_path = os.path.join(PRUSA_CONFIG_DIR, "my_printer_config.ini")
    os.makedirs(INPUT_GCODE_DIR, exist_ok=True)

    # Slice lower mesh
    lower_stl_path = os.path.join(OUTPUT_MODELS_DIR, f"{model_name}_lower.stl")
    lower_gcode_path = os.path.join(INPUT_GCODE_DIR, f"{model_name}_lower.gcode")

    # Both meshes are pre-positioned at bed center (150,150) in their STL files.
    # Use --dont-arrange so PrusaSlicer preserves their exact positions.
    # This ensures both halves share the same XY reference on the bed.

    print("\n=== Stage 2a: Planar slicing lower portion ===\n", flush=True)
    subprocess.run([
        slicer_path,
        "--load", ini_path,
        "--dont-arrange",
        "--export-gcode",
        lower_stl_path,
        "--output", lower_gcode_path,
    ], check=True)
    print(f"Lower G-code exported to {lower_gcode_path}", flush=True)

    # Slice upper deformed mesh
    upper_stl_path = os.path.join(OUTPUT_MODELS_DIR, f"{model_name}_upper_deformed.stl")
    upper_gcode_path = os.path.join(INPUT_GCODE_DIR, f"{model_name}_upper_deformed.gcode")

    print("\n=== Stage 2b: Planar slicing upper deformed portion ===\n", flush=True)
    result = subprocess.run([
        slicer_path,
        "--load", ini_path,
        "--dont-arrange",
        "--first-layer-height", "0.4",
        "--export-gcode",
        upper_stl_path,
        "--output", upper_gcode_path,
    ])
    if not os.path.exists(upper_gcode_path):
        if result.returncode != 0:
            print(f"Warning: PrusaSlicer exited with code {result.returncode}", flush=True)
        raise RuntimeError(f"PrusaSlicer failed to generate {upper_gcode_path}")
    print(f"Upper G-code exported to {upper_gcode_path}", flush=True)

    # --- Stage 3a: Convert lower planar G-code to CXZB format ---
    print("\n=== Stage 3a: Converting lower G-code to CXZB ===\n", flush=True)
    planar_cxzb_path = reform.convert_planar_to_cxzb(model_name)

    # --- Stage 3b: Back-transform upper G-code ---
    print("\n=== Stage 3b: Back-transforming upper G-code to 4-axis ===\n", flush=True)
    conic_gcode_path = reform.load_gcode_and_undeform(model_name, transform_params)

    # --- Stage 4: Merge ---
    print("\n=== Stage 4: Merging G-code files ===\n", flush=True)
    merged_path = merge.merge(planar_cxzb_path, conic_gcode_path, model_name)

    print(f"\n=== Done! Final G-code: {merged_path} ===\n", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Conic slicer for 4-axis printer. Splits mesh at a Z height, "
                    "applies conic deformation to upper portion, slices both, "
                    "and merges into 4-axis G-code."
    )
    parser.add_argument("--stl", required=True, help="Path to input STL file")
    parser.add_argument("--model", required=True, help="Model name for output files")
    parser.add_argument("--prusa", required=True, help="Path to PrusaSlicer executable")
    parser.add_argument("--z-split", required=True, type=float,
                        help="Z height at which to split the mesh")
    parser.add_argument("--cone-angle", required=True, type=float,
                        help="Cone half-angle in degrees (controls steepness)")
    args = parser.parse_args()

    run_slicer_pipeline(args.stl, args.model, args.prusa, args.z_split, args.cone_angle)


if __name__ == "__main__":
    main()
