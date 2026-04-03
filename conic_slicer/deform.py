import numpy as np
import pyvista as pv
import json
import os
import sys

if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

INPUT_MODELS_DIR = os.path.join(base_dir, "input_models")
OUTPUT_MODELS_DIR = os.path.join(base_dir, "output_models")


def load_mesh(model_name):
    """Load an STL mesh from input_models/."""
    mesh_path = os.path.join(INPUT_MODELS_DIR, f"{model_name}.stl")
    mesh = pv.read(mesh_path)
    return mesh


def split_mesh(mesh, z_split):
    """
    Split a mesh at a horizontal Z plane into lower and upper portions.

    Uses pyvista's clip method. The split plane is at z=z_split, normal pointing up.
    Both halves are centered on the full mesh's XY center so they align when
    PrusaSlicer re-centers them on the bed.

    Returns (lower_mesh, upper_mesh, mesh_center_xy) where mesh_center_xy is
    the XY center used, needed for consistent coordinate transforms later.
    """
    # Ensure triangulated
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()

    # Compute the full mesh's XY center BEFORE splitting — both halves will
    # be centered on this same point so they stay aligned.
    xmin, xmax, ymin, ymax, _, _ = mesh.bounds
    mesh_center_xy = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, 0.0])

    # Center the full mesh on this point before splitting
    mesh = mesh.copy()
    mesh.points -= mesh_center_xy

    # clip() with invert=False keeps the side the normal points to (above the plane)
    upper = mesh.clip(normal='z', origin=(0, 0, z_split), invert=False)
    lower = mesh.clip(normal='z', origin=(0, 0, z_split), invert=True)

    # Ensure both are triangulated after clipping
    if not upper.is_all_triangles:
        upper = upper.triangulate()
    if not lower.is_all_triangles:
        lower = lower.triangulate()

    # Cap open edges from clipping to produce watertight meshes for PrusaSlicer
    upper = upper.fill_holes(hole_size=1e6)
    lower = lower.fill_holes(hole_size=1e6)
    if not upper.is_all_triangles:
        upper = upper.triangulate()
    if not lower.is_all_triangles:
        lower = lower.triangulate()

    return lower, upper, mesh_center_xy


def deform_upper_mesh(upper_mesh, cone_angle_deg):
    """
    Apply conic deformation to the upper mesh portion.

    The conic deformation lifts each vertex by tan(cone_angle) * r, where r is
    the radial distance from the mesh center in XY. Z is also scaled by 1/cos(cone_angle)
    to preserve perpendicular thickness.

    Expects the mesh to already be XY-centered (done by split_mesh).

    Args:
        upper_mesh: The upper portion of the split mesh (already XY-centered).
        cone_angle_deg: Cone half-angle in degrees. Controls how steep the cone is.

    Returns:
        (deformed_mesh, transform_params) where transform_params is a dict
        with all parameters needed to reverse the transformation.
    """
    mesh = upper_mesh.copy()

    # Ensure triangulated
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()

    # Subdivide if mesh is too coarse
    TARGET_POINT_COUNT = 500000
    while mesh.n_points < TARGET_POINT_COUNT:
        mesh = mesh.subdivide(1, subfilter='linear')

    cone_angle_rad = np.deg2rad(cone_angle_deg)

    # Mesh is already XY-centered by split_mesh — compute radial distances
    distances_to_center = np.linalg.norm(mesh.points[:, :2], axis=1)

    # Scale Z to preserve perpendicular thickness
    cos_angle = np.cos(cone_angle_rad)
    cos_angle = max(cos_angle, 0.1)  # clamp to avoid extreme stretching
    mesh.points[:, 2] /= cos_angle

    # Apply conic deformation: lift z by tan(angle) * r
    translate_upwards = np.zeros((len(mesh.points), 3))
    translate_upwards[:, 2] = np.tan(cone_angle_rad) * distances_to_center
    mesh.points += translate_upwards

    # Clip the cone tip to create a flat base that PrusaSlicer can extrude on.
    # Without this, the cone tip at r≈0 has nearly zero cross-section and
    # PrusaSlicer refuses to slice ("no extrusions in first layer").
    TIP_CLIP_HEIGHT = 2.0  # mm above the cone tip to clip
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    clip_z = zmin + TIP_CLIP_HEIGHT
    mesh = mesh.clip(normal='z', origin=(0, 0, clip_z), invert=False)
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    # Cap the open edges from clipping to make the mesh watertight
    mesh = mesh.fill_holes(hole_size=1e6)
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()

    # Set bottom to z=0 for PrusaSlicer. Do NOT re-center XY — both halves
    # must share the same XY origin (set by split_mesh) so they align.
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    z_offset = zmin
    mesh.points[:, 2] -= z_offset

    transform_params = {
        "cone_angle_deg": float(cone_angle_deg),
        "tip_clip_height": float(TIP_CLIP_HEIGHT),
        "z_offset": float(z_offset),
    }

    return mesh, transform_params


def save_meshes(lower_mesh, upper_deformed_mesh, transform_params, model_name):
    """Save the lower mesh, deformed upper mesh, and transform parameters.

    Both meshes are shifted so their XY origin sits at bed center (150,150).
    This way PrusaSlicer doesn't need to auto-center, and both halves stay
    perfectly aligned on the bed.
    """
    os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)

    BED_CENTER = np.array([150.0, 150.0, 0.0])

    lower_path = os.path.join(OUTPUT_MODELS_DIR, f"{model_name}_lower.stl")
    upper_path = os.path.join(OUTPUT_MODELS_DIR, f"{model_name}_upper_deformed.stl")
    json_path = os.path.join(OUTPUT_MODELS_DIR, f"{model_name}_transform.json")

    # Shift both meshes to bed center so PrusaSlicer preserves their position
    lower_save = lower_mesh.copy()
    lower_save.points += BED_CENTER
    lower_save.save(lower_path)

    upper_save = upper_deformed_mesh.copy()
    upper_save.points += BED_CENTER
    upper_save.save(upper_path)

    with open(json_path, 'w') as f:
        json.dump(transform_params, f, indent=4)

    print(f"Lower mesh saved to {lower_path}")
    print(f"Upper deformed mesh saved to {upper_path}")
    print(f"Transform params saved to {json_path}")


if __name__ == "__main__":
    MODEL_NAME = "test"
    Z_SPLIT = 10.0
    CONE_ANGLE = 30.0

    mesh = load_mesh(MODEL_NAME)
    lower, upper, center = split_mesh(mesh, Z_SPLIT)
    upper_deformed, params = deform_upper_mesh(upper, CONE_ANGLE)
    save_meshes(lower, upper_deformed, params, MODEL_NAME)
