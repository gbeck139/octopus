import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import json
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

# Grid resolution for sampling the rotation field (mm)
GRID_RESOLUTION = 1.0


def load_mesh(MODEL_NAME):
    mesh_path = os.path.join(INPUT_MODELS_DIR, f"{MODEL_NAME}.stl")
    mesh = pv.read(mesh_path)
    return mesh


def sample_rotation_to_grid(mesh_points, rotation_field, grid_resolution=GRID_RESOLUTION):
    """
    Sample a per-vertex rotation field onto a regular 2D XY grid using
    nearest-neighbor interpolation. This grid is saved in the transform JSON
    so reform.py can look up rotations at arbitrary G-code XY positions.

    Returns:
        grid_data: dict with keys 'x_min', 'y_min', 'x_max', 'y_max',
                   'resolution', 'nx', 'ny', 'values' (flattened row-major list)
    """
    xy = mesh_points[:, :2]
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)

    # Add a small margin
    margin = grid_resolution * 2
    x_min -= margin
    y_min -= margin
    x_max += margin
    y_max += margin

    nx = int(np.ceil((x_max - x_min) / grid_resolution)) + 1
    ny = int(np.ceil((y_max - y_min) / grid_resolution)) + 1

    # Create grid coordinates
    gx = np.linspace(x_min, x_max, nx)
    gy = np.linspace(y_min, y_max, ny)
    grid_xx, grid_yy = np.meshgrid(gx, gy)  # shape (ny, nx)
    grid_points = np.column_stack([grid_xx.ravel(), grid_yy.ravel()])

    # Nearest-neighbor: for each grid point, find closest mesh vertex
    from scipy.spatial import cKDTree
    tree = cKDTree(xy)
    _, indices = tree.query(grid_points)
    grid_values = rotation_field[indices].reshape(ny, nx)

    return {
        "x_min": float(x_min),
        "y_min": float(y_min),
        "x_max": float(x_max),
        "y_max": float(y_max),
        "resolution": float(grid_resolution),
        "nx": int(nx),
        "ny": int(ny),
        "values": grid_values.tolist()
    }


def deform_mesh(mesh, scale=1.0, rotation_field=None,
                grid_resolution=GRID_RESOLUTION,
                pre_prepared=False,
                # Legacy radial params (used when rotation_field is None)
                angle_base=15, angle_factor=30, transition_z=0.0, blend_height=0.0):
    """
    Deform the mesh for hybrid non-planar slicing.

    When rotation_field is provided (per-vertex numpy array), uses it directly
    instead of the radius-based ROTATION function. This enables spatially-varying
    hybrid cartesian/radial behavior.

    When rotation_field is None, falls back to the original radial behavior.

    When pre_prepared is True, the mesh is assumed to already be triangulated,
    subdivided, scaled, and centered (used when analyze.py already prepared it).
    """

    if not pre_prepared:
        # Ensure mesh is triangulated
        if not mesh.is_all_triangles:
            mesh = mesh.triangulate()

        # Subdivide if the mesh is too coarse
        TARGET_POINT_COUNT = 1000000
        while mesh.n_points < TARGET_POINT_COUNT:
            mesh = mesh.subdivide(1, subfilter='linear')

        mesh.points *= scale

        # Center around the middle of the bounding box, set bottom to z=0
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        mesh.points -= np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, zmin])

    mesh.field_data["faces"] = mesh.faces.reshape(-1, 4)[:, 1:]
    distances_to_center = np.linalg.norm(mesh.points[:, :2], axis=1)
    max_radius = np.max(distances_to_center)

    if rotation_field is not None:
        # --- Hybrid mode: use the provided per-vertex rotation field ---
        effective_rotations = rotation_field

        # Sample rotation field to a 2D grid for reform.py
        rotation_grid = sample_rotation_to_grid(
            mesh.points, rotation_field, grid_resolution
        )
    else:
        # --- Legacy radial mode ---
        ROTATION = lambda radius: np.deg2rad(angle_base + angle_factor * (radius / max_radius))
        rotations = ROTATION(distances_to_center)

        # Compute blend factor for Cartesian-to-radial transition
        original_z = mesh.points[:, 2].copy()
        if blend_height > 0:
            blend = np.clip((original_z - transition_z) / blend_height, 0.0, 1.0)
        else:
            blend = np.where(original_z >= transition_z, 1.0, 0.0)
        effective_rotations = rotations * blend

        rotation_grid = None

    # Scale Z to preserve thickness perpendicular to the surface
    cos_rotations = np.cos(effective_rotations)
    cos_rotations = np.maximum(cos_rotations, 0.1)
    mesh.points[:, 2] /= cos_rotations

    # Create delta vector for each point (upward translation)
    translate_upwards = np.hstack([
        np.zeros((len(mesh.points), 2)),
        (np.tan(effective_rotations) * distances_to_center).reshape(-1, 1)
    ])
    mesh.points = mesh.points + translate_upwards

    # Ensure bottom is at z=0 after deformation
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    if rotation_field is not None:
        # Hybrid mode: the rotation field is constrained so no vertex goes
        # below z=0, so the flat bottom stays at z=0. Only center XY.
        # Any remaining tiny negative-z vertices get clipped as safety.
        offsets_applied = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, 0.0])
        mesh.points -= offsets_applied
        mesh.points[:, 2] = np.maximum(mesh.points[:, 2], 0.0)
    else:
        # Legacy radial mode: shift everything so min-z = 0 (original behavior)
        offsets_applied = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, zmin])
        mesh.points -= offsets_applied

    transform_params = {
        "max_radius": float(max_radius),
        "offsets_applied": offsets_applied.tolist(),
        "mode": "hybrid" if rotation_field is not None else "radial",
    }

    if rotation_grid is not None:
        # The cartesian zone ceiling: G-code points below this deformed Z
        # should use rotation=0 (they're in the flat bottom region).
        # 2mm bottom zone + 1mm margin for safety.
        transform_params["rotation_grid"] = rotation_grid
        transform_params["cartesian_z_ceiling"] = 3.0
    else:
        # Legacy radial params
        transform_params["angle_base"] = float(angle_base)
        transform_params["angle_factor"] = float(angle_factor)
        transform_params["transition_z"] = float(transition_z)
        transform_params["blend_height"] = float(blend_height)

    return mesh, transform_params


def save_deformed_mesh(deformed_mesh, transform_params, MODEL_NAME):
    os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)

    stl_path = os.path.join(OUTPUT_MODELS_DIR, f"{MODEL_NAME}_deformed.stl")
    json_path = os.path.join(OUTPUT_MODELS_DIR, f"{MODEL_NAME}_transform.json")

    deformed_mesh.save(stl_path)

    with open(json_path, 'w') as f:
        json.dump(transform_params, f, indent=4)


def plot_deformed_mesh(deformed_mesh):
    plt.figure(figsize=(12, 12))
    plt.scatter(deformed_mesh.points[:, 0], deformed_mesh.points[:, 2], s=1)
    plt.gca().set_aspect('equal')
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Scatter Plot of Deformed Mesh")
    plt.show()


if __name__ == "__main__":
    MODEL_NAME = '3DBenchy'
    mesh = load_mesh(MODEL_NAME)
    deformed_mesh, transform_params = deform_mesh(mesh, scale=1)
    save_deformed_mesh(deformed_mesh, transform_params, MODEL_NAME)
    plot_deformed_mesh(deformed_mesh)
