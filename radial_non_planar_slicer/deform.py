import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


import json
import os

def load_mesh(MODEL_NAME):
    # Load the mesh
    mesh = pv.read(f'radial_non_planar_slicer/input_models/{MODEL_NAME}.stl')
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

    # create delta vector for each point
    translate_upwards = np.hstack([np.zeros((len(mesh.points), 2)), np.tan(ROTATION(distances_to_center).reshape(-1, 1)) * distances_to_center.reshape(-1, 1)])
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
    deformed_mesh.save(f'radial_non_planar_slicer/output_models/{MODEL_NAME}_deformed.stl')
    
    # save transform params
    with open(f'radial_non_planar_slicer/output_models/{MODEL_NAME}_transform.json', 'w') as f:
        json.dump(transform_params, f, indent=4)



def plot_deformed_mesh(deformed_mesh):
    plt.figure(figsize=(12, 12))
    plt.scatter(deformed_mesh.points[:, 0], deformed_mesh.points[:, 2], s=1)
    plt.gca().set_aspect('equal')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot of Deformed Mesh")
    plt.show()


if __name__ == "__main__":
    MODEL_NAME = '3DBenchy'
    mesh = load_mesh(MODEL_NAME)
    deformed_mesh, transform_params = deform_mesh(mesh)
    save_deformed_mesh(deformed_mesh, transform_params, MODEL_NAME)
    plot_deformed_mesh(deformed_mesh)