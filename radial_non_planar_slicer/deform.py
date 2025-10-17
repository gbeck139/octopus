import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt



def load_mesh(MODEL_NAME):
    # Load the mesh
    mesh = pv.read(f'input_models/{MODEL_NAME}.stl')
    return mesh


def deform_mesh(mesh):
    # extract faces    
    mesh.field_data["faces"] = mesh.faces.reshape(-1, 4)[:, 1:] # assume all triangles

    # scale mesh
    mesh.points *= 1

    # center around the middle of the bounding box
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    mesh.points -= np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, zmin])
    # mesh.points -= np.array([0, 0, 0]) # optionally offset the part from the center

    # mesh.points = mesh.points[:10]

    # max radius of part
    max_radius = np.max(np.linalg.norm(mesh.points[:, :2], axis=1))

    # define rotation as a function of radius

    # TODO make this user definable

    ROTATION = lambda radius: np.deg2rad(15 + 30 * (radius / max_radius)) # Use for propeller and tree
    # ROTATION = lambda radius: np.full_like(radius, np.deg2rad(-40)) # Fixed rotation inwards
    # ROTATION = lambda radius: np.deg2rad(-40 + 30 * (1 - (radius / max_radius)) ** 2) # Use for bridge

    # rotate points around max diameter ring
    distances_to_center = np.linalg.norm(mesh.points[:, :2], axis=1)
    translate_upwards = np.hstack([np.zeros((len(mesh.points), 2)), np.tan(ROTATION(distances_to_center).reshape(-1, 1)) * distances_to_center.reshape(-1, 1)])

    mesh.points = mesh.points + translate_upwards

    # make bottom of part z=0 and center in bound box. remember the offsets for later
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    offsets_applied = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, zmin])
    mesh.points -= offsets_applied

    return mesh, ROTATION, offsets_applied


def save_deformed_mesh(deformed_mesh, MODEL_NAME):
    # save the mesh
    deformed_mesh.save(f'output_models/{MODEL_NAME}_deformed.stl')


def plot_deformed_mesh(deformed_mesh):
    plt.figure(figsize=(12, 12))
    plt.scatter(deformed_mesh.points[:, 0], deformed_mesh.points[:, 2], s=1)
    plt.gca().set_aspect('equal')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot of G-code Points")
    plt.show()