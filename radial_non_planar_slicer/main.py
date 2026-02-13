import numpy as np
import pyvista as pv
import networkx as nx
from pygcode import Line
import time
from scipy.spatial.transform import Rotation as R
import deform
import reform
import subprocess
import os
import sys
import shutil


#TODO separate reform and deform and make command with arguments

MODEL_NAME = '3DBenchy'

# Determine the base directory for the application
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # Running as script
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Function to find PrusaSlicer executable
def find_prusa_slicer():
    """Find PrusaSlicer executable in multiple possible locations"""
    possible_paths = [
        # Check dist folder first (for bundled distribution)
        os.path.join(BASE_DIR, 'PrusaSlicer', 'PrusaSlicer.AppImage'),
        os.path.join(BASE_DIR, 'PrusaSlicer', 'prusa-slicer'),
        os.path.join(BASE_DIR, 'PrusaSlicer', 'prusa-slicer.exe'),
        # Check system PATH
        'prusa-slicer',
        'prusa-slicer-console',
        'PrusaSlicer',
        # Original absolute path as fallback
        r"/home/grant/PrusaSlicer.AppImage",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
        # Try to find in PATH using cross-platform shutil.which
        try:
            which_result = shutil.which(path)
            if which_result:
                return which_result
        except (OSError, ValueError):
            # Ignore errors from shutil.which
            pass
    
    raise FileNotFoundError("PrusaSlicer not found. Please place PrusaSlicer in the 'PrusaSlicer' folder next to the executable or install it system-wide.")

mesh = deform.load_mesh(MODEL_NAME)
deformed_mesh, transform_params = deform.deform_mesh(mesh, scale=1)
deform.save_deformed_mesh(deformed_mesh, transform_params, MODEL_NAME)
deform.plot_deformed_mesh(deformed_mesh)

# Find PrusaSlicer executable
slicer_path = find_prusa_slicer()
print(f"Using PrusaSlicer at: {slicer_path}")

stl_path = rf"radial_non_planar_slicer/output_models/{MODEL_NAME}_deformed.stl"
output_gcode = rf"radial_non_planar_slicer/input_gcode/{MODEL_NAME}_deformed.gcode"
ini_path = r"radial_non_planar_slicer/prusa_slicer/my_printer_config.ini"

# Make sure output folder exists
os.makedirs(os.path.dirname(output_gcode), exist_ok=True)

print("\n***PRUSA*** planar slicer is running...\n")

# Run slicing
subprocess.run([
    slicer_path,
    "--load", ini_path,          # merges your printer/material/settings INI with PrusaSlicer's default settings
    "--export-gcode",            # tells it to slice and export
    stl_path,
    "--output", output_gcode
], check=True)


print(f"G-code exported to {output_gcode}")

print("\n***PRUSA*** planar slicing finished\n")

reform.load_gcode_and_undeform(MODEL_NAME, transform_params)
