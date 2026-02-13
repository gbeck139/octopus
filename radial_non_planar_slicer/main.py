import numpy as np
import pyvista as pv
import networkx as nx
from pygcode import Line
import time
#from scipy.spatial.transform import Rotation as R
import deform
import reform
import subprocess
import os
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
OUTPUT_MODELS_DIR = os.path.join(base_dir, "output_models")
INPUT_GCODE_DIR = os.path.join(base_dir, "input_gcode")
OUTPUT_GCODE_DIR = os.path.join(base_dir, "output_gcode")
PRUSA_CONFIG_DIR = os.path.join(base_dir, "prusa_slicer")

#TODO separate reform and deform and make command with arguments

#MODEL_NAME = '3DBenchy'

def run_slicer_pipeline(MODEL_NAME: str, slicer_path: str):

    # Save the original cwd
    #original_cwd = os.getcwd()

    # Change cwd to the folder where main.py is
    #os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("\nDeforming model...\n", flush=True)

    # Deform
    mesh = deform.load_mesh(MODEL_NAME)
    deformed_mesh, transform_params = deform.deform_mesh(mesh, scale=1)
    deform.save_deformed_mesh(deformed_mesh, transform_params, MODEL_NAME)
    #deform.plot_deformed_mesh(deformed_mesh)

    # Get paths for PrusaSlicer call

    #slicer_path = r"/home/grant/PrusaSlicer.AppImage"
    #stl_path = rf"radial_non_planar_slicer/output_models/{MODEL_NAME}_deformed.stl"
    #output_gcode = rf"radial_non_planar_slicer/input_gcode/{MODEL_NAME}_deformed.gcode"
    #ini_path = r"radial_non_planar_slicer/prusa_slicer/my_printer_config.ini"

    #slicer_path = r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe"
    #stl_path = rf"output_models/{MODEL_NAME}_deformed.stl"
    #output_gcode = rf"input_gcode/{MODEL_NAME}_deformed.gcode"
    #ini_path = r"prusa_slicer/my_printer_config.ini"

    stl_path = os.path.join(OUTPUT_MODELS_DIR, f"{MODEL_NAME}_deformed.stl")
    output_gcode = os.path.join(INPUT_GCODE_DIR, f"{MODEL_NAME}_deformed.gcode")
    ini_path = os.path.join(PRUSA_CONFIG_DIR, "my_printer_config.ini")

    os.makedirs(INPUT_GCODE_DIR, exist_ok=True)


    # Make sure output folder exists
    #os.makedirs(os.path.dirname(output_gcode), exist_ok=True)

    print("\n***PRUSA*** planar slicer is running...\n", flush=True)

    # Run slicing
    subprocess.run([
        slicer_path,
        "--load", ini_path,          # merges your printer/material/settings INI with PrusaSlicer's default settings
        "--ensure-on-bed",
        "--export-gcode",            # tells it to slice and export
        stl_path,
        "--output", output_gcode
    ], check=True)


    print(f"G-code exported to {output_gcode}", flush=True)

    print("\n***PRUSA*** planar slicing finished\n", flush=True)

    # Reform
    print("\nReforming model...\n", flush=True)
    reform.load_gcode_and_undeform(MODEL_NAME, transform_params)

    # Restore the original cwd afterwards (optional but safe)
    #os.chdir(original_cwd)

def main():
    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prusa", required=True)
    args = parser.parse_args()

    run_slicer_pipeline(args.model, args.prusa)


if __name__ == "__main__":
    main()