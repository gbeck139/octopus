import numpy as np
import pyvista as pv

# --- Parameters ---
MAX_OVERHANG = 45.0          # degrees: below this, rotation = 0 (cartesian)
ROTATION_MULTIPLIER = 1.5    # scale factor for computed rotations
SMOOTHING_ITERATIONS = 30    # Laplacian smoothing passes
MAX_POS_ROTATION = 45.0      # degrees: max positive tilt
MAX_NEG_ROTATION = -45.0     # degrees: max negative tilt
CENTER_DEAD_ZONE = 1.0       # mm: force rotation=0 within this radius


def compute_vertex_normals(mesh):
    """
    Compute area-weighted per-vertex normals from face normals.
    """
    faces = mesh.faces.reshape(-1, 4)[:, 1:]  # (n_faces, 3) triangle indices
    points = mesh.points

    # Compute face normals and areas
    v0 = points[faces[:, 0]]
    v1 = points[faces[:, 1]]
    v2 = points[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    face_areas = np.linalg.norm(cross, axis=1, keepdims=True)
    face_normals = cross / np.maximum(face_areas, 1e-12)

    # Accumulate area-weighted normals per vertex
    vertex_normals = np.zeros_like(points)
    for j in range(3):
        np.add.at(vertex_normals, faces[:, j], cross)  # cross already has area magnitude

    # Normalize
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals /= np.maximum(norms, 1e-12)

    return vertex_normals


def compute_rotation_direction(mesh):
    """
    Determine rotation sign per vertex based on radial direction.

    For the radial slicer, positive rotation tilts the nozzle outward.
    We compute: sign = dot(overhang_direction_xy, radial_direction_xy).
    If the overhang faces outward, we tilt outward (positive).
    If the overhang faces inward, we tilt inward (negative).
    """
    vertex_normals = compute_vertex_normals(mesh)
    points = mesh.points

    # Radial direction in XY from center
    radial_xy = points[:, :2].copy()
    radial_norms = np.linalg.norm(radial_xy, axis=1, keepdims=True)
    radial_xy /= np.maximum(radial_norms, 1e-12)

    # Overhang direction is the XY component of the vertex normal projected downward
    # A face normal pointing outward-and-down means the overhang faces outward
    overhang_xy = vertex_normals[:, :2].copy()
    overhang_norms = np.linalg.norm(overhang_xy, axis=1, keepdims=True)
    overhang_xy /= np.maximum(overhang_norms, 1e-12)

    # Sign: positive if overhang faces outward (same direction as radial)
    sign = np.sign(np.sum(radial_xy * overhang_xy, axis=1))

    # Where overhang is purely vertical (no XY component), default to positive
    sign[sign == 0] = 1.0

    return sign


def laplacian_smooth(mesh, field, iterations):
    """
    Iterative Laplacian smoothing of a scalar field on a triangle mesh.
    Each iteration replaces each vertex value with the average of its neighbors.
    Uses sparse matrix multiplication for speed.
    """
    from scipy.sparse import csr_matrix

    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    n_verts = mesh.n_points

    # Build sparse adjacency matrix
    # Each edge (i,j) from each triangle face
    rows = np.concatenate([faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 1], faces[:, 2], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 2], faces[:, 0], faces[:, 1]])
    data = np.ones(len(rows), dtype=np.float64)
    adj = csr_matrix((data, (rows, cols)), shape=(n_verts, n_verts))

    # Normalize rows to get averaging matrix (each row sums to 1)
    row_sums = np.array(adj.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0  # avoid division by zero for isolated vertices
    inv_sums = 1.0 / row_sums
    # Scale each row by its inverse sum
    avg_matrix = csr_matrix((inv_sums[adj.nonzero()[0]] * adj.data, adj.indices, adj.indptr), shape=(n_verts, n_verts))

    smoothed = field.copy()
    for _ in range(iterations):
        smoothed = avg_matrix @ smoothed

    return smoothed


def compute_rotation_field(mesh,
                           max_overhang=MAX_OVERHANG,
                           rotation_multiplier=ROTATION_MULTIPLIER,
                           smoothing_iterations=SMOOTHING_ITERATIONS,
                           max_pos_rotation=MAX_POS_ROTATION,
                           max_neg_rotation=MAX_NEG_ROTATION,
                           center_dead_zone=CENTER_DEAD_ZONE):
    """
    Compute a per-vertex rotation field from mesh geometry.

    Flat surfaces (overhang < max_overhang) get rotation = 0 (cartesian).
    Overhang surfaces get nonzero rotation proportional to overhang severity.

    Returns:
        rotation_field: numpy array of shape (n_vertices,) with rotation in radians
    """
    max_overhang_rad = np.deg2rad(max_overhang)
    max_pos_rad = np.deg2rad(max_pos_rotation)
    max_neg_rad = np.deg2rad(max_neg_rotation)

    # Step 1: Compute per-vertex normals
    vertex_normals = compute_vertex_normals(mesh)

    # Step 2: Compute overhang angle per vertex
    # overhang_angle = angle between vertex normal and [0,0,1]
    up = np.array([0.0, 0.0, 1.0])
    cos_angles = np.clip(np.dot(vertex_normals, up), -1.0, 1.0)
    overhang_angles = np.arccos(cos_angles)  # 0 = facing up, pi = facing down

    # Step 3: Compute rotation magnitude
    # Faces with normal angle < 90+max_overhang from vertical are safe
    # (normal at 90deg = vertical wall, at 90+45=135deg = 45deg overhang)
    overhang_threshold = np.deg2rad(90.0 + max_overhang)
    needs_rotation = overhang_angles > overhang_threshold

    # Rotation magnitude = how far past the threshold
    rotation_magnitude = np.zeros(mesh.n_points)
    rotation_magnitude[needs_rotation] = (
        overhang_angles[needs_rotation] - overhang_threshold
    ) * rotation_multiplier

    # Step 4: Determine sign from radial direction
    sign = compute_rotation_direction(mesh)

    # Apply sign
    rotation_field = rotation_magnitude * sign

    # Step 5: Force rotation = 0 near center (avoid singularity)
    distances = np.linalg.norm(mesh.points[:, :2], axis=1)
    center_mask = distances < center_dead_zone
    rotation_field[center_mask] = 0.0

    # Step 6: Force rotation = 0 for bottom vertices (bed contact)
    # Use a generous zone so the first few layers are fully cartesian,
    # giving PrusaSlicer a solid first layer.
    BOTTOM_ZONE_HEIGHT = 2.0  # mm
    z_vals = mesh.points[:, 2]
    bottom_threshold = z_vals.min() + BOTTOM_ZONE_HEIGHT
    bottom_mask = z_vals < bottom_threshold
    rotation_field[bottom_mask] = 0.0

    # Step 6b: Add a blend zone above the bottom to ramp rotation gradually.
    # Without this, vertices just above the bottom zone can have large rotation
    # and get pushed below z=0 during deformation (tan(rot)*r > z).
    BLEND_ZONE_HEIGHT = 5.0  # mm above bottom zone
    blend_top = bottom_threshold + BLEND_ZONE_HEIGHT
    blend_mask = (z_vals >= bottom_threshold) & (z_vals < blend_top)
    blend_factor = (z_vals[blend_mask] - bottom_threshold) / BLEND_ZONE_HEIGHT
    rotation_field[blend_mask] *= blend_factor

    # Step 7: Laplacian smoothing
    if smoothing_iterations > 0:
        rotation_field = laplacian_smooth(mesh, rotation_field, smoothing_iterations)

    # Step 8: Re-zero bottom after smoothing (smoothing bleeds nonzero values back in)
    rotation_field[bottom_mask] = 0.0

    # Step 9: Physical constraint -- ensure no vertex deforms below z=0.
    # Deformed z = z/cos(rot) + tan(rot)*r. For this to be >= 0:
    # rot must satisfy: z/cos(rot) + tan(rot)*r >= 0
    # For negative rotation (tilting inward), tan(rot) is negative, so
    # z/cos(rot) - |tan(rot)|*r >= 0 => |tan(rot)| <= z/(r*cos(rot))
    # Simpler conservative bound: |rot| <= arctan(z / r) for negative rot
    for _ in range(3):  # iterate since clamping changes the field
        deformed_z = z_vals / np.maximum(np.cos(rotation_field), 0.1) + \
                     np.tan(rotation_field) * distances
        violators = deformed_z < 0
        if not np.any(violators):
            break
        # Scale down violating rotations
        rotation_field[violators] *= 0.5

    # Step 10: Clamp to limits
    rotation_field = np.clip(rotation_field, max_neg_rad, max_pos_rad)

    return rotation_field


if __name__ == "__main__":
    # Quick visualization test
    import os
    import sys

    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    INPUT_MODELS_DIR = os.path.join(base_dir, "input_models")

    MODEL_NAME = "3DBenchy"
    mesh_path = os.path.join(INPUT_MODELS_DIR, f"{MODEL_NAME}.stl")
    mesh = pv.read(mesh_path)

    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()

    TARGET_POINT_COUNT = 100000
    while mesh.n_points < TARGET_POINT_COUNT:
        mesh = mesh.subdivide(1, subfilter='linear')

    # Center the mesh
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    mesh.points -= np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, zmin])

    rotation_field = compute_rotation_field(mesh)

    print(f"Rotation field stats:")
    print(f"  Min: {np.rad2deg(rotation_field.min()):.1f} deg")
    print(f"  Max: {np.rad2deg(rotation_field.max()):.1f} deg")
    print(f"  Mean: {np.rad2deg(rotation_field.mean()):.1f} deg")
    print(f"  Nonzero: {np.count_nonzero(rotation_field)} / {len(rotation_field)}")

    # Plot
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars=np.rad2deg(rotation_field),
                     cmap='coolwarm', clim=[-45, 45],
                     scalar_bar_args={'title': 'Rotation (deg)'})
    plotter.show()
