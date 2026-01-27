import numpy as np
import json
from pygcode import Line
import pyvista as pv
import matplotlib.pyplot as plt

def load_gcode_and_undeform(MODEL_NAME, transform_params=None):
    
    if transform_params is None:
        try:
             with open(f'radial_non_planar_slicer/output_models/{MODEL_NAME}_transform.json', 'r') as f:
                transform_params = json.load(f)
        except FileNotFoundError:
            print(f"Error: Transform parameters not found for {MODEL_NAME}")
            return

    max_radius = transform_params["max_radius"]
    angle_base = transform_params["angle_base"]
    angle_factor = transform_params["angle_factor"]
    offsets_applied = np.array(transform_params["offsets_applied"])
    
    ROTATION = lambda radius: np.deg2rad(angle_base + angle_factor * (radius / max_radius))

    # read gcode
    pos = np.array([0., 0., 20.])
    feed = 0
    gcode_points = []
    i = 0
    with open(f'radial_non_planar_slicer/input_gcode/{MODEL_NAME}_deformed.gcode', 'r') as fh:
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

            # extract position and feedrate
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

            # extract extrusion
            for param in line.block.modal_params:
                if param.letter == "E":
                    extrusion = param.value

            """
            
            FOR NEXT MEETING
            
            """
            # segment moves
            # prevents G0 (rapid moves) from hitting the part
            # makes G1 (feed moves) less jittery
            delta_pos = pos - prev_pos
            distance = np.linalg.norm(delta_pos)
            if distance > 0 and gcode.word == "G01":
                seg_size = 1 # mm
                num_segments = -(-distance // seg_size) # ceiling division
                seg_distance = distance/num_segments

                # calculate inverse time feed
                time_to_complete_move = (1/feed) * seg_distance # min/mm * mm = min
                if time_to_complete_move == 0:
                    inv_time_feed = None
                else:
                    inv_time_feed = 1/time_to_complete_move # 1/min

                for i in range(int(num_segments)):
                    gcode_points.append({
                        "position": (prev_pos + delta_pos * (i+1) / num_segments) + offsets_applied,
                        "command": gcode.word,
                        "extrusion": extrusion/num_segments if extrusion is not None else None,
                        # check for zero division or small nums
                        "inv_time_feed": inv_time_feed,
                        "move_length": seg_distance,
                        "start_position": prev_pos,
                        "end_position": pos,
                        "unsegmented_move_length": distance,
                        "feedrate": feed
                    })
            else:
                # Filter out null moves (no motion, no extrusion) to prevent stuttering
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

            """
            
            FOR NEXT MEETING: END
            
            """

    # untransform gcode
    positions = np.array([point["position"] for point in gcode_points])
    
    # Hardcoded center for Creality K1 Max (300x300mm bed)
    # We use this instead of bounding box to avoid issues with purge lines/skirts
    center_x = 150.0
    center_y = 150.0
    center_offset = np.array([center_x, center_y, 0])
    positions -= center_offset
    
    distances_to_center = np.linalg.norm(positions[:, :2], axis=1)
    rotations = ROTATION(distances_to_center) 
    translate_upwards = np.hstack([np.zeros((len(positions), 2)), np.tan(rotations.reshape(-1, 1)) * distances_to_center.reshape(-1, 1)])
    new_positions = positions - translate_upwards

    # Compensate for the Z-scaling applied in deform.py
    # NOTE: user reported "hill" artifacts / bowing at the bottom.
    # While mathematically strictly correct for surface normals, applying the cos(theta)
    # scaling to planar-sliced G-code (which has stair-steps) involves mapping flat disks to domes.
    # Disabling this scaling often results in a visually "flatter" restoration for solid blocks,
    # effectively treating the transform as a simple vertical shear.
    
    # cos_rotations = np.cos(rotations)
    # cos_rotations = np.maximum(cos_rotations, 0.1)
    # new_positions[:, 2] *= cos_rotations

    # --- SAFETY FILTERING ---
    # Filter out points that are geometrically impossible or dangerous (e.g. infinite wrapping of origin)
    # This keeps visualizations clean and print safe.
    NOZZLE_OFFSET = 43 # mm
    # Relaxed safety limits to allow for 4-axis dip
    MIN_SAFE_Z = -50.0 
    MAX_SAFE_Z = 200.0
    MAX_SAFE_R = 1000.0

    valid_mask = []
    
    z_comp_min = 1000
    
    for i, pos in enumerate(new_positions):
        # Replicate the kinematic calculation to check safety
        r = np.linalg.norm(pos[:2])
        z = pos[2]
        
        # We need the rotation for this specific point to check Z-dip
        # Rotation was calculated above based on 'distances_to_center', which corresponds to these indices
        rotation = rotations[i]

        # Apply nozzle compensation
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
    
    # Apply filter to both the numpy array and the list
    new_positions = new_positions[valid_mask]
    # Rebuild gcode_points list using itertools or list comp
    gcode_points = [point for i, point in enumerate(gcode_points) if valid_mask[i]]
    # Recalculate rotations array to keep it in sync for later use if needed (though we mainly used it for filtering)
    rotations = rotations[valid_mask]
    
    print(f"Safety Filter: Removed {len(valid_mask) - sum(valid_mask)} unsafe points.")

    # NOTE: reminder to use for our printer
    # cap travel move height to be just above the part and to not travel over the origin
    max_z = 0
    for i, point in enumerate(gcode_points):
        if point["command"] == "G01":
            max_z = max(max_z, new_positions[i][2])
    for i, point in enumerate(gcode_points):
        if point["command"] == "G00":
            if new_positions[i][2] > max_z:
                new_positions[i] = None


    # rescale extrusion by change in move_length
    prev_pos = np.array([0., 0., 0.])
    for i, point in enumerate(gcode_points):
        if point["extrusion"] is not None and point["move_length"] != 0:
            extrusion_scale = np.linalg.norm(new_positions[i] - prev_pos) / point["move_length"]
            point["extrusion"] *= extrusion_scale
        prev_pos = new_positions[i]

    # rescale extrusion to compensate for rotation deformation
    distances_to_center = np.linalg.norm(new_positions[:, :2], axis=1)
    extrusion_scales = np.cos(ROTATION(distances_to_center))
    for i, point in enumerate(gcode_points):
        if point["extrusion"] is not None:
            point["extrusion"] *= extrusion_scales[i]


    NOZZLE_OFFSET = 43 # mm (Defined earlier now)

    prev_theta = 0
    theta_accum = 0

    # save transformed gcode
    with open(f'radial_non_planar_slicer/output_gcode/{MODEL_NAME}_reformed.gcode', 'w') as fh:
        # write header
        fh.write("; --- INITIALIZATION ---\n")
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

            # if position[2] < 0:
            #     continue

            #################################################################################################
            ### If you want to print on another type of 4 axis printer, you will need to change this code ###
            #################################################################################################
            # convert to polar coordinates
            r = np.linalg.norm(position[:2])
            theta = np.arctan2(position[1], position[0])
            z = position[2]

            rotation = ROTATION(r) * 1

            # compensate for nozzle offset
            r += np.sin(rotation) * NOZZLE_OFFSET
            z += (np.cos(rotation) - 1) * NOZZLE_OFFSET

            # NOTE: change for non-continuous rotation
            delta_theta = theta - prev_theta
            if delta_theta > np.pi:
                delta_theta -= 2*np.pi
            if delta_theta < -np.pi:
                delta_theta += 2*np.pi

            theta_accum += delta_theta

            string = f"{point['command']} C{np.rad2deg(theta_accum):.5f} X{r:.5f} Z{z:.5f} B{-np.rad2deg(rotation):.5f}"
            #################################################################################################
            ### If you want to print on another type of 4 axis printer, you will need to change this code ###
            #################################################################################################


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
                
                # Use original feedrate if available, otherwise safe default for rapids/retracts
                f_val = point.get('feedrate')
                if f_val is None or f_val == 0:
                     f_val = 50000 if point['command'] == 'G00' else 2400 # 2400mm/min = 40mm/s for G1/Retracts

                string += f" F{f_val:.4f}"

            fh.write(string + "\n")

            # update previous values
            prev_theta = theta

        # write footer
        fh.write("M104 S0          ; Disable nozzle heater (Allow to cool)\n")
        fh.write("M106 S0          ; Turn off component cooling fan\n")
        fh.write("M84              ; Cut power to stepper motors (Enables manual movement/prevents overheating)\n")


    # get where z > 0
    z = new_positions[:, 2]
    point_cloud = pv.PolyData(new_positions[z > 0])
    point_cloud.plot(scalars=np.arange(len(new_positions[z > 0]))%2000, point_size=3, render_points_as_spheres=True) # doesnt work in google colab? uncomment to view if not in google colab

    # plot in matplotlib
    g01_points = np.array([new_positions[i] for i, point in enumerate(gcode_points) if point["command"] == "G01"])
    original_points = np.array([point["position"] for point in gcode_points if point["command"] == "G01"])

    plt.figure(figsize=(12, 12))
    c = original_points[:, 2][::-1] % 5 / 5
    plt.scatter(g01_points[:, 0], g01_points[:, 2], s=1, c=c)
    plt.gca().set_aspect('equal')
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Scatter Plot of G-code Points")
    plt.show()


if __name__ == "__main__":
    MODEL_NAME = '3DBenchy'  # Change as needed
    load_gcode_and_undeform(MODEL_NAME)