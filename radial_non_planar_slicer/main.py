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


#TODO separate reform and deform and make command with arguments

MODEL_NAME = '3DBenchy'

# Deform

mesh = deform.load_mesh(MODEL_NAME)
deformed_mesh, transform_params = deform.deform_mesh(mesh, scale=1)
deform.save_deformed_mesh(deformed_mesh, transform_params, MODEL_NAME)
deform.plot_deformed_mesh(deformed_mesh)

# Get paths for PrusaSlicer call

slicer_path = r"/home/grant/PrusaSlicer.AppImage"
stl_path = rf"radial_non_planar_slicer/output_models/{MODEL_NAME}_deformed.stl"
output_gcode = rf"radial_non_planar_slicer/input_gcode/{MODEL_NAME}_deformed.gcode"
ini_path = r"radial_non_planar_slicer/prusa_slicer/my_printer_config.ini"

#slicer_path = r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe"
#stl_path = rf"output_models/{MODEL_NAME}_deformed.stl"
#output_gcode = rf"input_gcode/{MODEL_NAME}_deformed.gcode"
#ini_path = r"prusa_slicer/my_printer_config.ini"

# Make sure output folder exists
os.makedirs(os.path.dirname(output_gcode), exist_ok=True)

print("\n***PRUSA*** planar slicer is running...\n")

# Run slicing
subprocess.run([
    slicer_path,
    "--load", ini_path,          # merges your printer/material/settings INI with PrusaSlicer's default settings
    "--ensure-on-bed",
    "--export-gcode",            # tells it to slice and export
    stl_path,
    "--output", output_gcode
], check=True)


print(f"G-code exported to {output_gcode}")

print("\n***PRUSA*** planar slicing finished\n")

# Reform

reform.load_gcode_and_undeform(MODEL_NAME, transform_params)
