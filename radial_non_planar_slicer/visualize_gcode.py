import numpy as np
import matplotlib.pyplot as plt
import re
import os

# Configuration
SHOW_TRAVEL_MOVES = True # Set to True to see G0 moves (travel), False for only G1 (extrusion)

def visualize_gcode(model_name, nozzle_offset=43):
    filepath = f'radial_non_planar_slicer/output_gcode/{model_name}_reformed.gcode'
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    points = []
    
    # Regex to find coordinates. 
    # Looks for X, Z, C, B followed by a number (float or int)
    # G-code example: G1 C123.45 X10.0 Z5.0 B-15.0
    
    print(f"Reading {filepath}...")
    
    with open(filepath, 'r') as f:
        for line in f:
            if not line.startswith('G0') and not line.startswith('G1'):
                continue
                
            # Extract values using regex
            x_match = re.search(r'X([-\d\.]+)', line)
            z_match = re.search(r'Z([-\d\.]+)', line)
            c_match = re.search(r'C([-\d\.]+)', line)
            b_match = re.search(r'B([-\d\.]+)', line)
            e_match = re.search(r'E([-\d\.]+)', line)
            
            # Filter based on SHOW_TRAVEL_MOVES
            is_extrusion = False
            if e_match:
                if float(e_match.group(1)) > 0:
                    is_extrusion = True
            
            if not SHOW_TRAVEL_MOVES and not is_extrusion:
                continue
            
            if x_match and z_match and c_match and b_match:
                x_gcode = float(x_match.group(1))
                z_gcode = float(z_match.group(1))
                c_gcode = float(c_match.group(1))
                b_gcode = float(b_match.group(1))
                
                # 1. Recover Rotation (B is negative degrees of rotation)
                # In reform.py: B{-np.rad2deg(rotation)}
                # So: rotation = -B_rad
                rotation_rad = np.deg2rad(-b_gcode)
                
                # 2. Reverse Nozzle Offset
                # In reform.py:
                # r += np.sin(rotation) * NOZZLE_OFFSET
                # z += (np.cos(rotation) - 1) * NOZZLE_OFFSET
                
                r_model = x_gcode - np.sin(rotation_rad) * nozzle_offset
                z_model = z_gcode - (np.cos(rotation_rad) - 1) * nozzle_offset
                
                # 3. Polar to Cartesian
                c_rad = np.deg2rad(c_gcode)
                
                x = r_model * np.cos(c_rad)
                y = r_model * np.sin(c_rad)
                z = z_model
                
                points.append([x, y, z])

    if not points:
        print("No points found. Check G-code format.")
        return

    points = np.array(points)
    
    print(f"X Range: {points[:,0].min():.2f} to {points[:,0].max():.2f}")
    print(f"Y Range: {points[:,1].min():.2f} to {points[:,1].max():.2f}")
    print(f"Z Range: {points[:,2].min():.2f} to {points[:,2].max():.2f}")
    
    # DEBUG: Analyze B-values (Tilt)
    b_values = []
    with open(filepath, 'r') as f:
        for line in f:
            b_match = re.search(r'B([-\d\.]+)', line)
            if b_match:
                b_values.append(float(b_match.group(1)))
    
    if b_values:
        b_arr = np.array(b_values)
        print(f"DEBUG: B-axis (Tilt) Range: {b_arr.min():.2f} to {b_arr.max():.2f}")
        print(f"DEBUG: Average B: {b_arr.mean():.2f}")

    # Filter out crazy outliers for visualization
    # Assuming the part is within a reasonable volume (e.g. +/- 500mm)
    mask = (np.abs(points[:,2]) < 500) & (np.abs(points[:,0]) < 500) & (np.abs(points[:,1]) < 500)
    filtered_points = points[mask]
    
    if len(filtered_points) < len(points):
        print(f"Filtered {len(points) - len(filtered_points)} outlier points.")
        points = filtered_points

    # Plotting
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    # Using a subset of points if too many to speed up rendering
    step = 1 if len(points) < 10000 else len(points) // 10000
    
    # Color by Z height for better visualization
    sc = ax.scatter(points[::step, 0], points[::step, 1], points[::step, 2], 
                    c=points[::step, 2], cmap='viridis', s=1)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Reconstructed Toolpath: {model_name}')
    
    # Set equal aspect ratio for realistic view
    # Matplotlib 3D doesn't have a simple 'equal' aspect, so we fake it by setting limits
    max_range = np.array([points[:,0].max()-points[:,0].min(), 
                          points[:,1].max()-points[:,1].min(), 
                          points[:,2].max()-points[:,2].min()]).max() / 2.0

    mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
    mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
    mid_z = (points[:,2].max()+points[:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Force aspect ratio to be equal
    ax.set_box_aspect([1,1,1])

    plt.colorbar(sc, label='Z Height')
    plt.show()

if __name__ == "__main__":
    # Default model name from main.py context
    MODEL_NAME = '3DBenchy'  # Change as needed
    visualize_gcode(MODEL_NAME)
