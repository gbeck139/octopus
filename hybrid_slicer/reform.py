import numpy as np
import json
from pygcode import Line
import pyvista as pv
import matplotlib.pyplot as plt
import os
import sys

if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

INPUT_MODELS_DIR = os.path.join(base_dir, "input_models")
OUTPUT_MODELS_DIR = os.path.join(base_dir, "output_models")
INPUT_GCODE_DIR = os.path.join(base_dir, "input_gcode")
OUTPUT_GCODE_DIR = os.path.join(base_dir, "output_gcode")
PRUSA_CONFIG_DIR = os.path.join(base_dir, "prusa_slicer")


def lookup_rotation_from_3d_grid(grid_data, positions):
    """
    Trilinear interpolation of rotation values from a 3D grid.

    Args:
        grid_data: dict with x/y/z_min, x/y/z_max, nx, ny, nz, values
        positions: (N, 3) array of XYZ positions in deformed coordinates

    Returns:
        rotations: (N,) array of interpolated rotation values in radians
    """
    values = np.array(grid_data["values"])  # (nx, ny, nz)
    x_min, x_max = grid_data["x_min"], grid_data["x_max"]
    y_min, y_max = grid_data["y_min"], grid_data["y_max"]
    z_min, z_max = grid_data["z_min"], grid_data["z_max"]
    nx, ny, nz = grid_data["nx"], grid_data["ny"], grid_data["nz"]

    # Compute fractional grid indices
    fx = (positions[:, 0] - x_min) / (x_max - x_min) * (nx - 1)
    fy = (positions[:, 1] - y_min) / (y_max - y_min) * (ny - 1)
    fz = (positions[:, 2] - z_min) / (z_max - z_min) * (nz - 1)

    fx = np.clip(fx, 0, nx - 1.001)
    fy = np.clip(fy, 0, ny - 1.001)
    fz = np.clip(fz, 0, nz - 1.001)

    ix0 = np.floor(fx).astype(int)
    iy0 = np.floor(fy).astype(int)
    iz0 = np.floor(fz).astype(int)
    ix1 = np.minimum(ix0 + 1, nx - 1)
    iy1 = np.minimum(iy0 + 1, ny - 1)
    iz1 = np.minimum(iz0 + 1, nz - 1)

    sx = fx - ix0
    sy = fy - iy0
    sz = fz - iz0

    # Trilinear interpolation (8 corners)
    result = (
        values[ix0, iy0, iz0] * (1-sx)*(1-sy)*(1-sz) +
        values[ix1, iy0, iz0] * sx*(1-sy)*(1-sz) +
        values[ix0, iy1, iz0] * (1-sx)*sy*(1-sz) +
        values[ix1, iy1, iz0] * sx*sy*(1-sz) +
        values[ix0, iy0, iz1] * (1-sx)*(1-sy)*sz +
        values[ix1, iy0, iz1] * sx*(1-sy)*sz +
        values[ix0, iy1, iz1] * (1-sx)*sy*sz +
        values[ix1, iy1, iz1] * sx*sy*sz
    )

    return result


def load_gcode_and_undeform(MODEL_NAME, transform_params=None):

    if transform_params is None:
        try:
            json_path = os.path.join(OUTPUT_MODELS_DIR, f"{MODEL_NAME}_transform.json")
            with open(json_path, 'r') as f:
                transform_params = json.load(f)
        except FileNotFoundError:
            print(f"Error: Transform parameters not found for {MODEL_NAME}")
            return

    max_radius = transform_params["max_radius"]
    offsets_applied = np.array(transform_params["offsets_applied"])
    mode = transform_params.get("mode", "radial")

    # Set up rotation lookup based on mode
    if mode == "hybrid":
        rotation_grid = transform_params["rotation_grid"]
        use_grid = True
    else:
        # Legacy radial mode
        angle_base = transform_params["angle_base"]
        angle_factor = transform_params["angle_factor"]
        transition_z = transform_params.get("transition_z", 0.0)
        blend_height = transform_params.get("blend_height", 0.0)
        ROTATION = lambda radius: np.deg2rad(angle_base + angle_factor * (radius / max_radius))
        use_grid = False

        def compute_blend(z):
            z = np.asarray(z, dtype=float)
            if blend_height > 0:
                return np.clip((z - transition_z) / blend_height, 0.0, 1.0)
            else:
                return np.where(z >= transition_z, 1.0, 0.0)

    # Read gcode
    pos = np.array([0., 0., 20.])
    feed = 0
    gcode_points = []
    gcode_input_path = os.path.join(INPUT_GCODE_DIR, f"{MODEL_NAME}_deformed.gcode")
    with open(gcode_input_path, 'r') as fh:
        for line_text in fh.readlines():

            # Skip comment lines and non-standard commands
            line_stripped = line_text.strip()
            if not line_stripped or line_stripped.startswith(';') or line_stripped.startswith('EXCLUDE_OBJECT') or line_stripped.startswith('SET_') or line_stripped.startswith('START_') or line_stripped.startswith('END_') or line_stripped.startswith('G93') or line_stripped.startswith('G94'):
                continue

            line = Line(line_text)

            if not line.block.gcodes:
                continue

            extrusion = None
            move_command_seen = False

            for gcode in sorted(line.block.gcodes):
                if gcode.word == "G01" or gcode.word == "G00":
                    move_command_seen = True
                    prev_pos = pos.copy()

                    if gcode.X is not None:
                        pos[0] = gcode.X
                    if gcode.Y is not None:
                        pos[1] = gcode.Y
                    if gcode.Z is not None:
                        pos[2] = gcode.Z

                if gcode.word.letter == "F":
                    feed = gcode.word.value

            if not move_command_seen:
                continue

            for param in line.block.modal_params:
                if param.letter == "E":
                    extrusion = param.value

            # Segment moves
            delta_pos = pos - prev_pos
            distance = np.linalg.norm(delta_pos)
            if distance > 0 and gcode.word == "G01":
                seg_size = .1  # mm
                num_segments = -(-distance // seg_size)  # ceiling division
                seg_distance = distance / num_segments

                time_to_complete_move = (1 / feed) * seg_distance
                if time_to_complete_move == 0:
                    inv_time_feed = None
                else:
                    inv_time_feed = 1 / time_to_complete_move

                for i in range(int(num_segments)):
                    gcode_points.append({
                        "position": (prev_pos + delta_pos * (i + 1) / num_segments) + offsets_applied,
                        "command": gcode.word,
                        "extrusion": extrusion / num_segments if extrusion is not None else None,
                        "inv_time_feed": inv_time_feed,
                        "move_length": seg_distance,
                        "start_position": prev_pos,
                        "end_position": pos,
                        "unsegmented_move_length": distance,
                        "feedrate": feed
                    })
            else:
                if extrusion is None and distance == 0:
                    continue

                gcode_points.append({
                    "position": pos.copy() + offsets_applied,
                    "command": gcode.word,
                    "extrusion": extrusion,
                    "inv_time_feed": None,
                    "move_length": 0,
                    "feedrate": feed
                })

    # Untransform gcode
    positions = np.array([point["position"] for point in gcode_points])

    # Hardcoded center for Creality K1 Max (300x300mm bed)
    center_x = 150.0
    center_y = 150.0
    center_offset = np.array([center_x, center_y, 0])
    positions -= center_offset

    distances_to_center = np.linalg.norm(positions[:, :2], axis=1)

    if use_grid:
        # --- Hybrid mode: look up rotations from the 3D grid ---
        # The 3D grid maps (x, y, z_deformed) -> rotation, so we get the
        # correct rotation at every point regardless of Z height.
        # No iterative Z recovery or blend needed.
        effective_rotations = lookup_rotation_from_3d_grid(rotation_grid, positions)

        # Force rotation=0 for points far from the model (skirt/purge lines).
        # These are outside the mesh footprint and should not be rotated.
        far_mask = distances_to_center > max_radius * 1.5
        effective_rotations[far_mask] = 0.0

        print(f"Hybrid 3D grid lookup: {np.sum(effective_rotations == 0)} zero, "
              f"{np.sum(effective_rotations != 0)} nonzero, "
              f"range [{np.rad2deg(effective_rotations.min()):.1f}, "
              f"{np.rad2deg(effective_rotations.max()):.1f}] deg")
    else:
        # --- Legacy radial mode ---
        full_rotations = ROTATION(distances_to_center)

        # Iterative solve for original Z to determine effective rotations
        z_deformed = positions[:, 2].copy()
        z_est = z_deformed.copy()
        for _ in range(10):
            blend = compute_blend(z_est)
            eff_rot = full_rotations * blend
            cos_eff = np.cos(eff_rot)
            cos_eff = np.maximum(cos_eff, 0.1)
            z_est = (z_deformed - np.tan(eff_rot) * distances_to_center) * cos_eff

        blend = compute_blend(z_est)
        effective_rotations = full_rotations * blend

    # Undo the upward translation
    translate_upwards = np.hstack([
        np.zeros((len(positions), 2)),
        (np.tan(effective_rotations) * distances_to_center).reshape(-1, 1)
    ])
    new_positions = positions - translate_upwards

    # Undo Z-scaling
    cos_rotations = np.cos(effective_rotations)
    cos_rotations = np.maximum(cos_rotations, 0.1)
    new_positions[:, 2] *= cos_rotations

    # --- SAFETY FILTERING ---
    NOZZLE_OFFSET = 43  # mm
    MIN_SAFE_Z = -50.0
    MAX_SAFE_Z = 200.0
    MAX_SAFE_R = 1000.0

    valid_mask = []
    z_comp_min = 1000

    for i, pos in enumerate(new_positions):
        r = np.linalg.norm(pos[:2])
        z = pos[2]
        rotation = effective_rotations[i]

        r_comp = r + np.sin(rotation) * NOZZLE_OFFSET
        z_comp = z + (np.cos(rotation) - 1) * NOZZLE_OFFSET

        if z_comp < z_comp_min:
            z_comp_min = z_comp

        if z_comp < MIN_SAFE_Z or z_comp > MAX_SAFE_Z or r_comp > MAX_SAFE_R:
            valid_mask.append(False)
        else:
            valid_mask.append(True)

    print(f"Minimum machine Z required: {z_comp_min:.2f} mm")

    valid_mask = np.array(valid_mask, dtype=bool)

    new_positions = new_positions[valid_mask]
    gcode_points = [point for i, point in enumerate(gcode_points) if valid_mask[i]]
    effective_rotations = effective_rotations[valid_mask]

    print(f"Safety Filter: Removed {len(valid_mask) - sum(valid_mask)} unsafe points.")

    # Cap travel move height to just above the part
    max_z = 0
    for i, point in enumerate(gcode_points):
        if point["command"] == "G01":
            max_z = max(max_z, new_positions[i][2])
    for i, point in enumerate(gcode_points):
        if point["command"] == "G00":
            if new_positions[i][2] > max_z:
                new_positions[i][2] = max_z + 2.0

    # Rescale extrusion by change in move_length
    prev_pos = np.array([0., 0., 0.])
    for i, point in enumerate(gcode_points):
        if point["extrusion"] is not None and point["move_length"] != 0:
            extrusion_scale = np.linalg.norm(new_positions[i] - prev_pos) / point["move_length"]
            point["extrusion"] *= extrusion_scale
        prev_pos = new_positions[i]

    # Rescale extrusion to compensate for rotation deformation
    extrusion_scales = np.cos(effective_rotations)
    for i, point in enumerate(gcode_points):
        if point["extrusion"] is not None:
            point["extrusion"] *= extrusion_scales[i]

    NOZZLE_OFFSET = 43  # mm

    prev_theta = 0
    theta_accum = 0

    # Save transformed gcode
    os.makedirs(OUTPUT_GCODE_DIR, exist_ok=True)
    gcode_output_path = os.path.join(OUTPUT_GCODE_DIR, f"{MODEL_NAME}_reformed.gcode")
    with open(gcode_output_path, 'w') as fh:
        # Write header
        fh.write("; --- INITIALIZATION ---\n")
        fh.write("; Hybrid Cartesian/Radial Non-Planar Slicer\n")
        fh.write("G21              ; Establish metric units (millimeters)\n")
        fh.write("G90              ; Use absolute coordinates for all axes\n")
        fh.write("G94              ; Set feedrate mode to units per minute (mm/min)\n")
        fh.write("; M203 C3600 X5000 Z1000 B5000 E300 ; Set max feedrate limits (Currently disabled)\n")
        fh.write("M106 S128        ; Enable cooling fan at 50% power (PWM 128/255)\n")
        fh.write("; --- THERMAL MANAGEMENT ---\n")
        fh.write("M104 S200        ; Start heating nozzle to 200°C (Non-blocking)\n")
        fh.write("M109 S200        ; Wait for nozzle to reach target temperature before proceeding\n")
        fh.write("; --- HOMING & INITIAL POSITIONING ---\n")
        fh.write("G28              ; Execute homing sequence for all axes\n")
        fh.write("G0 C0 X0 Z20 B-15.0 F600 ; Rapid move to safe start clearance and orientation\n")
        fh.write("; --- EXTRUDER PREPARATION ---\n")
        fh.write("G92 E0           ; Zero out the current extruder position\n")
        fh.write("G1 E10 F200      ; Perform 10mm purge to prime the nozzle\n")
        fh.write("G92 E0           ; Reset extruder position after priming\n")
        fh.write("M83              ; Switch to relative extrusion mode (Required for RRF/standard logic)\n")
        current_feed_mode = "G94"
        for i, point in enumerate(gcode_points):
            position = new_positions[i]

            if position is None:
                continue

            if np.all(np.isnan(position)):
                continue

            # Convert to polar coordinates
            r = np.linalg.norm(position[:2])
            theta = np.arctan2(position[1], position[0])
            z = position[2]

            rotation = effective_rotations[i]

            # Compensate for nozzle offset
            r += np.sin(rotation) * NOZZLE_OFFSET
            z += (np.cos(rotation) - 1) * NOZZLE_OFFSET

            # Continuous rotation tracking
            delta_theta = theta - prev_theta
            if delta_theta > np.pi:
                delta_theta -= 2 * np.pi
            if delta_theta < -np.pi:
                delta_theta += 2 * np.pi

            theta_accum += delta_theta

            string = f"{point['command']} C{np.rad2deg(theta_accum):.5f} X{r:.5f} Z{z:.5f} B{-np.rad2deg(rotation):.5f}"

            if point["extrusion"] is not None:
                string += f" E{point['extrusion']:.4f}"

            if point["inv_time_feed"] is not None:
                if current_feed_mode != "G93":
                    fh.write("G93\n")
                    current_feed_mode = "G93"
                string += f" F{(point['inv_time_feed']):.4f}"
            else:
                if current_feed_mode != "G94":
                    fh.write("G94\n")
                    current_feed_mode = "G94"

                f_val = point.get('feedrate')
                if f_val is None or f_val == 0:
                    f_val = 50000 if point['command'] == 'G00' else 2400

                string += f" F{f_val:.4f}"

            fh.write(string + "\n")

            prev_theta = theta

        # Write footer
        fh.write("M104 S0          ; Disable nozzle heater (Allow to cool)\n")
        fh.write("M106 S0          ; Turn off component cooling fan\n")
        fh.write("M84              ; Cut power to stepper motors (Enables manual movement/prevents overheating)\n")


if __name__ == "__main__":
    MODEL_NAME = '3DBenchy'
    load_gcode_and_undeform(MODEL_NAME)
