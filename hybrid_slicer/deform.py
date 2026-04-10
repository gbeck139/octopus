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


def sample_rotation_to_3d_grid(deformed_points, rotation_field, grid_resolution=GRID_RESOLUTION):
    """
    Sample per-vertex rotation field onto a 3D grid using the DEFORMED mesh
    coordinates. This is sampled after deformation so the grid maps
    (x, y, z_deformed) -> rotation, allowing reform to look up the exact
    rotation at any G-code point position.

    Using a 3D grid (instead of 2D) correctly handles cases where different
    Z heights at the same XY have different rotations (e.g. hull side = 0°
    vs stern overhang = 45° above it).
    """
    from scipy.spatial import cKDTree

    mins = deformed_points.min(axis=0)
    maxs = deformed_points.max(axis=0)

    margin = grid_resolution * 2
    mins = mins - margin
    maxs = maxs + margin

    nx = int(np.ceil((maxs[0] - mins[0]) / grid_resolution)) + 1
    ny = int(np.ceil((maxs[1] - mins[1]) / grid_resolution)) + 1
    nz = int(np.ceil((maxs[2] - mins[2]) / grid_resolution)) + 1

    gx = np.linspace(mins[0], maxs[0], nx)
    gy = np.linspace(mins[1], maxs[1], ny)
    gz = np.linspace(mins[2], maxs[2], nz)
    grid_xx, grid_yy, grid_zz = np.meshgrid(gx, gy, gz, indexing='ij')
    grid_points = np.column_stack([grid_xx.ravel(), grid_yy.ravel(), grid_zz.ravel()])

    # For each 3D grid cell, find the nearest deformed mesh vertex
    tree = cKDTree(deformed_points)
    _, indices = tree.query(grid_points)
    grid_values = rotation_field[indices].reshape(nx, ny, nz)

    return {
        "x_min": float(mins[0]), "y_min": float(mins[1]), "z_min": float(mins[2]),
        "x_max": float(maxs[0]), "y_max": float(maxs[1]), "z_max": float(maxs[2]),
        "resolution": float(grid_resolution),
        "nx": int(nx), "ny": int(ny), "nz": int(nz),
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
        # Grid will be sampled after deformation (needs deformed coordinates)
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

    # Sample 3D grid AFTER deformation, BEFORE re-centering.
    # The grid maps (x, y, z_deformed) -> rotation, so reform can look up
    # the correct rotation at any G-code point in deformed space.
    if rotation_field is not None:
        print(f"Sampling 3D rotation grid...", flush=True)
        rotation_grid = sample_rotation_to_3d_grid(
            mesh.points, rotation_field, grid_resolution
        )
        print(f"  Grid shape: {rotation_grid['nx']}x{rotation_grid['ny']}x{rotation_grid['nz']}", flush=True)
    else:
        rotation_grid = None

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
        transform_params["rotation_grid"] = rotation_grid
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
