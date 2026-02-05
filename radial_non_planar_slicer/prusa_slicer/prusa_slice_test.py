import subprocess
import os

model_name = "dogbone_mini_flat"

# TODO: currently absolute paths. Change to relative paths. 
slicer_path = r"/home/grant/PrusaSlicer.AppImage"
stl_path = rf"radial_non_planar_slicer/output_models/{model_name}_deformed.stl"
output_gcode = rf"radial_non_planar_slicer/output_gcode/{model_name}_deformed.gcode"
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