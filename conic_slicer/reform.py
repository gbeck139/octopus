import numpy as np
import json
from pygcode import Line
from scipy.spatial import cKDTree
import pyvista as pv
import os
import sys

if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

OUTPUT_MODELS_DIR = os.path.join(base_dir, "output_models")
INPUT_GCODE_DIR = os.path.join(base_dir, "input_gcode")
OUTPUT_GCODE_DIR = os.path.join(base_dir, "output_gcode")

NOZZLE_OFFSET = 43  # mm


def load_gcode_and_undeform(model_name, transform_params=None):
    """
    Back-transform the planar G-code of the deformed upper mesh into 4-axis
    conic G-code (CXZB coordinates).

    Reads the planar G-code produced by PrusaSlicer from the deformed upper mesh,
    reverses the conic deformation, converts to polar coordinates, applies nozzle
    offset compensation, and writes the final 4-axis G-code.
    """
    if transform_params is None:
        json_path = os.path.join(OUTPUT_MODELS_DIR, f"{model_name}_transform.json")
        with open(json_path, 'r') as f:
            transform_params = json.load(f)

    cone_angle_deg = transform_params["cone_angle_deg"]
    z_offset = transform_params.get("z_offset", 0.0)

    cone_angle_rad = np.deg2rad(cone_angle_deg)
    cos_angle = max(np.cos(cone_angle_rad), 0.1)

    # Parse G-code
    pos = np.array([0., 0., 20.])
    feed = 0
    gcode_points = []

    gcode_input_path = os.path.join(INPUT_GCODE_DIR, f"{model_name}_upper_deformed.gcode")
    with open(gcode_input_path, 'r') as fh:
        for line_text in fh.readlines():
            line_stripped = line_text.strip()
            if (not line_stripped or line_stripped.startswith(';')
                    or line_stripped.startswith('EXCLUDE_OBJECT')
                    or line_stripped.startswith('SET_')
                    or line_stripped.startswith('START_')
                    or line_stripped.startswith('END_')
                    or line_stripped.startswith('G93')
                    or line_stripped.startswith('G94')):
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

            # Segment moves for smoother transformation
            delta_pos = pos - prev_pos
            distance = np.linalg.norm(delta_pos)
            if distance > 0 and gcode.word == "G01":
                seg_size = 0.1  # mm
                num_segments = int(-(-distance // seg_size))  # ceiling division
                seg_distance = distance / num_segments

                time_to_complete_move = (1 / feed) * seg_distance if feed > 0 else 0
                inv_time_feed = 1 / time_to_complete_move if time_to_complete_move > 0 else None

                for i in range(num_segments):
                    gcode_points.append({
                        "position": prev_pos + delta_pos * (i + 1) / num_segments,
                        "command": gcode.word,
                        "extrusion": extrusion / num_segments if extrusion is not None else None,
                        "inv_time_feed": inv_time_feed,
                        "move_length": seg_distance,
                        "feedrate": feed,
                    })
            else:
                if extrusion is None and distance == 0:
                    continue

                gcode_points.append({
                    "position": pos.copy(),
                    "command": gcode.word,
                    "extrusion": extrusion,
                    "inv_time_feed": None,
                    "move_length": 0,
                    "feedrate": feed,
                })

    if not gcode_points:
        print("Warning: No G-code points extracted from upper deformed G-code.")
        return

    # Reverse the conic deformation
    positions = np.array([point["position"] for point in gcode_points])

    # Center: PrusaSlicer centers the model on a 300x300mm bed
    center_x = 150.0
    center_y = 150.0
    center_offset = np.array([center_x, center_y, 0])
    positions -= center_offset

    # Read wave params
    wave_type = transform_params.get("wave_type", "none")
    wave_amplitude = transform_params.get("wave_amplitude", 0.0)
    wave_length = transform_params.get("wave_length", 1.0)

    # For wave correction: find the mesh-surface r for each G-code point.
    # G-code toolpath positions are offset from the mesh surface (nozzle width),
    # which causes a wave phase error. Using the nearest mesh vertex's r eliminates this.
    if wave_type != "none":
        mesh_path = os.path.join(OUTPUT_MODELS_DIR, f"{model_name}_upper_deformed.stl")
        deformed_mesh = pv.read(mesh_path)
        mesh_points = deformed_mesh.points - np.array([center_x, center_y, 0.0])
        tree = cKDTree(mesh_points)
        _, nearest_idx = tree.query(positions)  # lookup in z_saved space (before z_offset)
        mesh_r = np.linalg.norm(mesh_points[nearest_idx, :2], axis=1)
        print(f"Wave r-correction: loaded {len(mesh_points)} mesh vertices for KDTree lookup")

    # Add back the Z offset that was subtracted in deform to set z_min=0.
    # This restores positions to the deformed coordinate space.
    positions[:, 2] += z_offset

    distances_to_center = np.linalg.norm(positions[:, :2], axis=1)

    # Reverse Z-lift: z = z_deformed - tan(angle) * r
    translate_upwards = np.zeros((len(positions), 3))
    translate_upwards[:, 2] = np.tan(cone_angle_rad) * distances_to_center

    # Reverse wave modulation using mesh-surface r (not G-code r)
    if wave_type == "sine":
        translate_upwards[:, 2] += wave_amplitude * np.sin(2 * np.pi * mesh_r / wave_length)
    elif wave_type == "sawtooth":
        translate_upwards[:, 2] += wave_amplitude * (2 * (mesh_r % wave_length) / wave_length - 1)
    elif wave_type == "sine_azimuthal":
        n_lobes = wave_length
        mesh_theta = np.arctan2(mesh_points[nearest_idx, 1], mesh_points[nearest_idx, 0])
        translate_upwards[:, 2] += wave_amplitude * np.sin(n_lobes * mesh_theta)
    elif wave_type == "sine_curvature":
        weights_path = os.path.join(OUTPUT_MODELS_DIR, f"{model_name}_curvature_weights.npy")
        curvature_weights = np.load(weights_path)
        point_curv_weights = curvature_weights[nearest_idx]
        translate_upwards[:, 2] += wave_amplitude * point_curv_weights * np.sin(2 * np.pi * mesh_r / wave_length)
    elif wave_type == "sine_normal":
        weights_path = os.path.join(OUTPUT_MODELS_DIR, f"{model_name}_normal_weights.npy")
        normal_weights = np.load(weights_path)
        point_weights = normal_weights[nearest_idx]
        translate_upwards[:, 2] += wave_amplitude * point_weights * np.sin(2 * np.pi * mesh_r / wave_length)

    new_positions = positions - translate_upwards

    # Reverse Z-scaling
    new_positions[:, 2] *= cos_angle

    # Compute per-point surface angle from derivative of deformation
    dz_dr = np.tan(cone_angle_rad) * np.ones(len(new_positions))

    if wave_type == "sine":
        dz_dr += wave_amplitude * (2 * np.pi / wave_length) * np.cos(2 * np.pi * mesh_r / wave_length)
    elif wave_type == "sawtooth":
        dz_dr += wave_amplitude * 2.0 / wave_length
    elif wave_type == "sine_curvature":
        dz_dr += wave_amplitude * point_curv_weights * (2 * np.pi / wave_length) * np.cos(2 * np.pi * mesh_r / wave_length)
    elif wave_type == "sine_azimuthal":
        pass  # azimuthal wave doesn't change the radial slope
    elif wave_type == "sine_normal":
        dz_dr += wave_amplitude * point_weights * (2 * np.pi / wave_length) * np.cos(2 * np.pi * mesh_r / wave_length)

    rotations = np.arctan(dz_dr)

    # Safety filtering
    MIN_SAFE_Z = -50.0
    MAX_SAFE_Z = 200.0
    MAX_SAFE_R = 1000.0

    valid_mask = np.ones(len(new_positions), dtype=bool)
    z_comp_min = 1000

    for i, pos_pt in enumerate(new_positions):
        r = np.linalg.norm(pos_pt[:2])
        z = pos_pt[2]
        rotation = rotations[i]

        r_comp = r + np.sin(rotation) * NOZZLE_OFFSET
        z_comp = z + (np.cos(rotation) - 1) * NOZZLE_OFFSET

        if z_comp < z_comp_min:
            z_comp_min = z_comp

        if z_comp < MIN_SAFE_Z or z_comp > MAX_SAFE_Z or r_comp > MAX_SAFE_R:
            valid_mask[i] = False

    print(f"Minimum machine Z required: {z_comp_min:.2f} mm")

    new_positions = new_positions[valid_mask]
    gcode_points = [pt for i, pt in enumerate(gcode_points) if valid_mask[i]]
    rotations = rotations[valid_mask]

    print(f"Safety Filter: Removed {len(valid_mask) - sum(valid_mask)} unsafe points.")

    # Cap travel move height
    max_z = 0
    for i, point in enumerate(gcode_points):
        if point["command"] == "G01":
            max_z = max(max_z, new_positions[i][2])
    for i, point in enumerate(gcode_points):
        if point["command"] == "G00":
            if new_positions[i][2] > max_z:
                new_positions[i] = None

    # Rescale extrusion by change in move length
    prev_pos_arr = np.array([0., 0., 0.])
    for i, point in enumerate(gcode_points):
        if new_positions[i] is None:
            continue
        if point["extrusion"] is not None and point["move_length"] != 0:
            extrusion_scale = np.linalg.norm(new_positions[i] - prev_pos_arr) / point["move_length"]
            point["extrusion"] *= extrusion_scale
        prev_pos_arr = new_positions[i]

    # Rescale extrusion for cone deformation (thickness changes)
    for i, point in enumerate(gcode_points):
        if new_positions[i] is None:
            continue
        if point["extrusion"] is not None:
            point["extrusion"] *= cos_angle

    # Write 4-axis G-code
    os.makedirs(OUTPUT_GCODE_DIR, exist_ok=True)
    gcode_output_path = os.path.join(OUTPUT_GCODE_DIR, f"{model_name}_conic.gcode")

    prev_theta = 0
    theta_accum = 0
    current_feed_mode = "G94"

    with open(gcode_output_path, 'w') as fh:
        for i, point in enumerate(gcode_points):
            position = new_positions[i]

            if position is None:
                continue
            if np.all(np.isnan(position)):
                continue

            r = np.linalg.norm(position[:2])
            theta = np.arctan2(position[1], position[0])
            z = position[2]
            rotation = rotations[i]

            # Nozzle offset compensation
            r += np.sin(rotation) * NOZZLE_OFFSET
            z += (np.cos(rotation) - 1) * NOZZLE_OFFSET

            # Unwrap C-axis for continuous rotation
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
                string += f" F{point['inv_time_feed']:.4f}"
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

    print(f"Conic G-code written to {gcode_output_path}")
    return gcode_output_path


def convert_planar_to_cxzb(model_name):
    """
    Convert standard XY planar G-code (from PrusaSlicer) to CXZB 4-axis format
    with B=0 (no tilt). This is needed for the lower planar portion so the
    4-axis printer can execute it and so it can be merged with conic G-code.
    """
    gcode_input_path = os.path.join(INPUT_GCODE_DIR, f"{model_name}_lower.gcode")
    gcode_output_path = os.path.join(OUTPUT_GCODE_DIR, f"{model_name}_planar.gcode")

    pos = np.array([0., 0., 20.])
    feed = 0
    gcode_points = []

    with open(gcode_input_path, 'r') as fh:
        for line_text in fh.readlines():
            line_stripped = line_text.strip()
            if (not line_stripped or line_stripped.startswith(';')
                    or line_stripped.startswith('EXCLUDE_OBJECT')
                    or line_stripped.startswith('SET_')
                    or line_stripped.startswith('START_')
                    or line_stripped.startswith('END_')
                    or line_stripped.startswith('G93')
                    or line_stripped.startswith('G94')):
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

            # Segment moves for smoother polar conversion
            delta_pos = pos - prev_pos
            distance = np.linalg.norm(delta_pos)
            if distance > 0 and gcode.word == "G01":
                seg_size = 0.1  # mm
                num_segments = int(-(-distance // seg_size))
                seg_distance = distance / num_segments

                time_to_complete_move = (1 / feed) * seg_distance if feed > 0 else 0
                inv_time_feed = 1 / time_to_complete_move if time_to_complete_move > 0 else None

                for i in range(num_segments):
                    gcode_points.append({
                        "position": prev_pos + delta_pos * (i + 1) / num_segments,
                        "command": gcode.word,
                        "extrusion": extrusion / num_segments if extrusion is not None else None,
                        "inv_time_feed": inv_time_feed,
                        "move_length": seg_distance,
                        "feedrate": feed,
                    })
            else:
                if extrusion is None and distance == 0:
                    continue

                gcode_points.append({
                    "position": pos.copy(),
                    "command": gcode.word,
                    "extrusion": extrusion,
                    "inv_time_feed": None,
                    "move_length": 0,
                    "feedrate": feed,
                })

    if not gcode_points:
        print("Warning: No G-code points extracted from lower planar G-code.")
        return None

    positions = np.array([point["position"] for point in gcode_points])

    # Center: PrusaSlicer centers the model on a 300x300mm bed
    center_x = 150.0
    center_y = 150.0
    center_offset = np.array([center_x, center_y, 0])
    positions -= center_offset

    # Write CXZB G-code with B=0 (no tilt)
    os.makedirs(OUTPUT_GCODE_DIR, exist_ok=True)

    prev_theta = 0
    theta_accum = 0
    current_feed_mode = "G94"

    with open(gcode_output_path, 'w') as fh:
        for i, point in enumerate(gcode_points):
            position = positions[i]

            if np.any(np.isnan(position)):
                continue

            r = np.linalg.norm(position[:2])
            theta = np.arctan2(position[1], position[0])
            z = position[2]

            # Unwrap C-axis for continuous rotation
            delta_theta = theta - prev_theta
            if delta_theta > np.pi:
                delta_theta -= 2 * np.pi
            if delta_theta < -np.pi:
                delta_theta += 2 * np.pi
            theta_accum += delta_theta

            # B=0 (no tilt), no nozzle offset compensation needed
            string = f"{point['command']} C{np.rad2deg(theta_accum):.5f} X{r:.5f} Z{z:.5f} B0.00000"

            if point["extrusion"] is not None:
                string += f" E{point['extrusion']:.4f}"

            if point["inv_time_feed"] is not None:
                if current_feed_mode != "G93":
                    fh.write("G93\n")
                    current_feed_mode = "G93"
                string += f" F{point['inv_time_feed']:.4f}"
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

    print(f"Planar CXZB G-code written to {gcode_output_path}")
    return gcode_output_path


if __name__ == "__main__":
    MODEL_NAME = "test"
    load_gcode_and_undeform(MODEL_NAME)
