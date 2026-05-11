DEFAULT_CONFIG = {
    "model": {
        "scale": 1.0,
        "rotX": 0.0,
        "rotY": 0.0,
        "rotZ": 0.0
    },

    "deformation": {
        "angle_base": 15.0,
        "angle_factor": 30.0,
        "target_point_count": 1000000
    },

    "printer": {
        "bed_center_x": 150.0,
        "bed_center_y": 150.0,
        "nozzle_offset": 43.0
    },

    "safety": {
        "min_safe_z": -50.0,
        "max_safe_z": 200.0,
        "max_safe_r": 1000.0
    },

    "print": {
        "layer_height": 0.2,
        "nozzle_temperature": 200,
        "bed_temperature": 60,
        "print_speed": 60,
        "travel_speed": 150,
        "infill_density": 15,
        "filament_diameter": 1.75
    }
}