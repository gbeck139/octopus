import json

from cura_engine_wrapper import CuraEngineWrapper
from cura_mapping import build_cura_settings

# --- helpers ---
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# --- load configs ---
printer_cfg  = load_json("profiles/printer.json")
material_cfg = load_json("profiles/material.json")
process_cfg  = load_json("profiles/process.json")

planar_cfg    = process_cfg["planar"]
nonplanar_cfg = process_cfg["non_planar"]

# --- init cura engine ---
cura = CuraEngineWrapper(
    cura_path=r"C:\Program Files\UltiMaker Cura 5.10.2\CuraEngine.exe",  # or absolute path, see section 3
    machine_def="research_printer.def.json"
)

# --- build cura settings ---
cura_settings = build_cura_settings(planar_cfg, material_cfg)

# --- slice ---
cura.slice(
    stl_path="model.stl",
    output_gcode="planar.gcode",
    cura_settings=cura_settings
)

print("Planar slicing complete.")
