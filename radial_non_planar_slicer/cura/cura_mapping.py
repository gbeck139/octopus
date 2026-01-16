PLANAR_TO_CURA = {
    "layer_height": "layer_height",
    "wall_count": "wall_line_count",
    "infill_density": "infill_sparse_density",
    "print_speed": "speed_print"
}

MATERIAL_TO_CURA = {
    "print_temperature": "material_print_temperature",
    "bed_temperature": "material_bed_temperature"
}

def build_cura_settings(planar, material):
    cura_settings = {}

    for my_key, cura_key in PLANAR_TO_CURA.items():
        if my_key in planar:
            cura_settings[cura_key] = planar[my_key]

    for my_key, cura_key in MATERIAL_TO_CURA.items():
        if my_key in material:
            cura_settings[cura_key] = material[my_key]

    return cura_settings
