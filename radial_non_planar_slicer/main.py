import deform
import reform
import subprocess
import sys
import os

def resource_path(relative_path):
    """ Finds the file whether running as a script or a bundled .exe """
    if hasattr(sys, '_MEIPASS'):
        # If running in PyInstaller bundle, look in the temp folder
        return os.path.join(sys._MEIPASS, relative_path)
    # If running normally, look in the current folder
    return os.path.join(os.path.abspath("."), relative_path)

# NOW apply it to your variables:
slicer_path = resource_path("prusa_slicer/PrusaSlicer.AppImage")
ini_path = resource_path("prusa_slicer/my_printer_config.ini")






#TODO separate reform and deform and make command with arguments

MODEL_NAME = '3DBenchy'

mesh = deform.load_mesh(MODEL_NAME)
deformed_mesh, transform_params = deform.deform_mesh(mesh, scale=1)
deform.save_deformed_mesh(deformed_mesh, transform_params, MODEL_NAME)
deform.plot_deformed_mesh(deformed_mesh)

# TODO: currently absolute paths. Change to relative paths. 
slicer_path = r"/home/grant/PrusaSlicer.AppImage"
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
