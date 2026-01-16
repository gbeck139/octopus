import json

from cura_engine_wrapper import CuraEngineWrapper
from cura_mapping import build_cura_settings

# --- helpers ---
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# --- load configs ---
printer_cfg  = load_json(r"C:\Users\canca\Documents\3D Printing Slicer App\octopus\radial_non_planar_slicer\cura\profiles\printer.json")
material_cfg = load_json(r"C:\Users\canca\Documents\3D Printing Slicer App\octopus\radial_non_planar_slicer\cura\profiles\material.json")
process_cfg  = load_json(r"C:\Users\canca\Documents\3D Printing Slicer App\octopus\radial_non_planar_slicer\cura\profiles\process.json")

planar_cfg    = process_cfg["planar"]
nonplanar_cfg = process_cfg["non_planar"]

# --- init cura engine ---
cura = CuraEngineWrapper(
    cura_path=r"C:\Program Files\UltiMaker Cura 5.10.2\CuraEngine.exe",  # or absolute path, see section 3
    machine_def=r"C:\Users\canca\Documents\3D Printing Slicer App\octopus\radial_non_planar_slicer\cura\research_printer.def.json"
)

# --- build cura settings ---
cura_settings = build_cura_settings(planar_cfg, material_cfg)

# --- slice ---
cura.slice(
    stl_path=r"C:\Users\canca\Downloads\Cute Simple Halloween Pumpkin  - 6711982\files\pumpkin.stl",
    output_gcode="deformed_planar.gcode",
    cura_settings=cura_settings
)

print("Planar slicing complete.")
