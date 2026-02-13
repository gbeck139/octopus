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
import utils


#TODO separate reform and deform and make command with arguments

MODEL_NAME = '3DBenchy'

mesh = deform.load_mesh(MODEL_NAME)
deformed_mesh, transform_params = deform.deform_mesh(mesh, scale=1)
deform.save_deformed_mesh(deformed_mesh, transform_params, MODEL_NAME)
deform.plot_deformed_mesh(deformed_mesh)

# Determine paths
base_path = utils.get_base_path()

# Check if running in dev or bundled
if os.path.exists(os.path.join(base_path, 'PrusaSlicer')):
    # Bundled or structured
    slicer_dir = os.path.join(base_path, 'PrusaSlicer')
    # Try console version first, then GUI
    if os.path.exists(os.path.join(slicer_dir, 'prusa-slicer-console.exe')):
        slicer_path = os.path.join(slicer_dir, 'prusa-slicer-console.exe')
    else:
        slicer_path = os.path.join(slicer_dir, 'prusa-slicer.exe')
        
    stl_path = utils.get_resource_path(f"output_models/{MODEL_NAME}_deformed.stl")
    output_gcode = utils.get_resource_path(f"input_gcode/{MODEL_NAME}_deformed.gcode")
    ini_path = utils.get_resource_path(f"prusa_slicer/my_printer_config.ini")
else:
    # Dev mode fallback (original paths)
    # Trying to find Prusa Slicer on system or relative path
    # For now, let's assume user provides path or it's in a standard location
    # But better to just default to "PrusaSlicer" folder in current dir
     slicer_path = os.path.abspath(r"PrusaSlicer/prusa-slicer-console.exe") # You must put PrusaSlicer here
     if not os.path.exists(slicer_path):
         # Try standard install locations or ask user?
         # For this script to work, we need a slicer path.
         # Let's defaults to what user might have or error out
     # Check environment variable
     if 'PRUSA_SLICER_PATH' in os.environ:
         slicer_path = os.environ['PRUSA_SLICER_PATH']
     else:
         # Default fallback
         slicer_path = r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe"
         
     stl_path = os.path.abspath(f"radial_non_planar_slicer/output_models/{MODEL_NAME}_deformed.stl")
     output_gcode = os.path.abspath(f"radial_non_planar_slicer/input_gcode/{MODEL_NAME}_deformed.gcode")
     ini_path = os.path.abspath(f"radial_non_planar_slicer/prusa_slicer/my_printer_config.ini")

# Make sure output folder exists
os.makedirs(os.path.dirname(output_gcode), exist_ok=True)
os.makedirs(os.path.dirname(stl_path), exist_ok=True) 

print("\n***PRUSA*** planar slicer is running...\n")

if not os.path.exists(slicer_path):
    print(f"ERROR: Prusa Slicer not found at {slicer_path}")
    print("Please set PRUSA_SLICER_PATH environment variable or place PrusaSlicer in the dist folder.")
    exit(1)
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
