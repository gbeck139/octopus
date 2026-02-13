import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import json
import os

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


def load_mesh(MODEL_NAME):
    # Load the mesh
    #mesh = pv.read(f'radial_non_planar_slicer/input_models/{MODEL_NAME}.stl')
    #mesh = pv.read(f'input_models/{MODEL_NAME}.stl')
    mesh_path = os.path.join(INPUT_MODELS_DIR, f"{MODEL_NAME}.stl")
    mesh = pv.read(mesh_path)

    return mesh


def deform_mesh(mesh, scale=1.0, angle_base=15, angle_factor=30):
    """
    Deform the mesh for radial non-planar slicing.
    
    :param mesh: The input mesh to be deformed.
    :param scale: Scaling factor for the mesh size.
    :param angle_base: Base rotation angle in degrees at r=0.
    :param angle_factor: Angle increase factor in degrees.
    :return: Deformed mesh, transform_params dict.
    """ 

    # Ensure mesh is triangulated
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()

    # Subdivide if the mesh is too coarse (e.g. a simple box)
    # This prevents flat surfaces from staying flat when they should curve
    TARGET_POINT_COUNT = 1000000
    while mesh.n_points < TARGET_POINT_COUNT:
        mesh = mesh.subdivide(1, subfilter='linear')

    mesh.field_data["faces"] = mesh.faces.reshape(-1, 4)[:, 1:]
    mesh.points *= scale

    # center around the middle of the bounding box, and set bottom to z=0
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    mesh.points -= np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, zmin])
    # mesh.points -= np.array([0, 0, 0]) # optionally offset the part from the center
    distances_to_center = np.linalg.norm(mesh.points[:, :2], axis=1)
    max_radius = np.max(distances_to_center)

    # define rotation as a function of radius
    # TODO make this user definable
    ROTATION = lambda radius: np.deg2rad(angle_base + angle_factor * (radius / max_radius)) # Use for propeller and tree
    # ROTATION = lambda radius: np.full_like(radius, np.deg2rad(-40)) # Fixed rotation inwards
    # ROTATION = lambda radius: np.deg2rad(-40 + 30 * (1 - (radius / max_radius)) ** 2) # Use for bridge

    rotations = ROTATION(distances_to_center) # calculate rotations for all points

    # Scale Z to preserve thickness perpendicular to the surface
    # Without this, the part gets thinner as the slope increases (vertical shear vs bending)
    cos_rotations = np.cos(rotations)
    # Clamp to avoid extreme stretching or division by zero close to 90 degrees
    cos_rotations = np.maximum(cos_rotations, 0.1) 
    mesh.points[:, 2] /= cos_rotations

    # create delta vector for each point
    translate_upwards = np.hstack([np.zeros((len(mesh.points), 2)), np.tan(rotations.reshape(-1, 1)) * distances_to_center.reshape(-1, 1)])
    # apply deformation
    mesh.points = mesh.points + translate_upwards

    # ensure bottom is at z=0 after deformation
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    offsets_applied = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, zmin])
    mesh.points -= offsets_applied
    
    transform_params = {
        "max_radius": float(max_radius),
        "angle_base": float(angle_base),
        "angle_factor": float(angle_factor),
        "offsets_applied": offsets_applied.tolist()
    }

    return mesh, transform_params


def save_deformed_mesh(deformed_mesh, transform_params, MODEL_NAME):
    # save the mesh
    #deformed_mesh.save(f'radial_non_planar_slicer/output_models/{MODEL_NAME}_deformed.stl')
    #deformed_mesh.save(f'output_models/{MODEL_NAME}_deformed.stl')

    # save transform params
    #with open(f'radial_non_planar_slicer/output_models/{MODEL_NAME}_transform.json', 'w') as f:
    #with open(f'output_models/{MODEL_NAME}_transform.json', 'w') as f:
    #    json.dump(transform_params, f, indent=4)

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
    MODEL_NAME = '3DBenchy'  # Change as needed
    mesh = load_mesh(MODEL_NAME)
    deformed_mesh, transform_params = deform_mesh(mesh, scale=1)
    save_deformed_mesh(deformed_mesh, transform_params, MODEL_NAME)
    plot_deformed_mesh(deformed_mesh)