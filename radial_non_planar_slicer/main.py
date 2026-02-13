import deform
import reform
import subprocess
import sys
import os

import deform
import reform
import subprocess
import sys
import os
import stat

def resource_path(relative_path):
    """ Finds the file whether running as a script or a bundled .exe """
    if hasattr(sys, '_MEIPASS'):
        # If running in PyInstaller bundle, look in the temp folder
        return os.path.join(sys._MEIPASS, relative_path)
    
    # If running normally, look relative to this script's directory
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

# 1. Setup Resource Paths (Bundled files)
#    These files must be included in your spec file's datas=[] list.
slicer_bin_path = resource_path(os.path.join("prusa_slicer", "PrusaSlicer.AppImage"))
config_ini_path = resource_path(os.path.join("prusa_slicer", "my_printer_config.ini"))

# Ensure AppImage is executable (Linux specific)
if sys.platform != "win32" and os.path.exists(slicer_bin_path):
    st = os.stat(slicer_bin_path)
    os.chmod(slicer_bin_path, st.st_mode | stat.S_IEXEC)

# 2. Setup Working Directories (User data)
#    When frozen, sys.executable is the exe path. Use its directory for input/output.
#    When script, use script directory.
if getattr(sys, 'frozen', False):
    app_dir = os.path.dirname(sys.executable)
else:
    app_dir = os.path.dirname(os.path.abspath(__file__))

input_models_dir = os.path.join(app_dir, "input_models")
output_models_dir = os.path.join(app_dir, "output_models") # JSONs and Deformed STLs
input_gcode_dir = os.path.join(app_dir, "input_gcode")     # Output from PrusaSlicer
output_gcode_dir = os.path.join(app_dir, "output_gcode")   # Final Result

# Ensure directories exist
for d in [input_models_dir, output_models_dir, input_gcode_dir, output_gcode_dir]:
    os.makedirs(d, exist_ok=True)


#TODO separate reform and deform and make command with arguments

MODEL_NAME = '3DBenchy'

# Construct Full Paths
target_stl_file = os.path.join(input_models_dir, f"{MODEL_NAME}.stl")
deformed_stl_file = os.path.join(output_models_dir, f"{MODEL_NAME}_deformed.stl")
transform_json_file = os.path.join(output_models_dir, f"{MODEL_NAME}_transform.json")
slicer_output_gcode = os.path.join(input_gcode_dir, f"{MODEL_NAME}_deformed.gcode")
final_reformed_gcode = os.path.join(output_gcode_dir, f"{MODEL_NAME}_reformed.gcode")

print(f"Processing Model: {MODEL_NAME}")
print(f"Input STL: {target_stl_file}")

# --- DEFORM ---
if not os.path.exists(target_stl_file):
    print(f"Error: Input file not found: {target_stl_file}")
    sys.exit(1)

mesh = deform.load_mesh(target_stl_file)
deformed_mesh, transform_params = deform.deform_mesh(mesh, scale=1)
deform.save_deformed_mesh(deformed_mesh, transform_params, deformed_stl_file, transform_json_file)

# Only plot if we are not frozen (optional, usually gui stuff blocks valid automation)
# deform.plot_deformed_mesh(deformed_mesh) 

print("\n***PRUSA*** planar slicer is running...\n")

# --- SLICE ---
# Run slicing
if not os.path.exists(slicer_bin_path):
    print(f"Error: Slicer executable not found at {slicer_bin_path}")
    sys.exit(1)

subprocess.run([
    slicer_bin_path,
    "--load", config_ini_path,   # merges your printer/material/settings INI with PrusaSlicer's default settings
    "--export-gcode",            # tells it to slice and export
    deformed_stl_file,
    "--output", slicer_output_gcode
], check=True)


print(f"G-code exported to {slicer_output_gcode}")

print("\n***PRUSA*** planar slicing finished\n")

# --- REFORM ---
reform.load_gcode_and_undeform(transform_json_file, slicer_output_gcode, final_reformed_gcode, transform_params)
print(f"Done! Final G-code at: {final_reformed_gcode}")
