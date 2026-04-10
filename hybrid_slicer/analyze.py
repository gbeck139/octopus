import numpy as np
import pyvista as pv

# --- Parameters ---
MAX_OVERHANG = 25.0          # degrees: overhangs past this (from vertical) get rotation
ROTATION_MULTIPLIER = 1.5    # scale factor for computed rotations
SMOOTHING_ITERATIONS = 30    # Laplacian smoothing passes (legacy, see COLUMN_SMOOTH_SIGMA)
COLUMN_SMOOTH_SIGMA = 5.0    # Gaussian sigma (mm) for smoothing the per-column rotation map
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
                           center_dead_zone=CENTER_DEAD_ZONE,
                           column_smooth_sigma=COLUMN_SMOOTH_SIGMA):
    """
    Compute a per-vertex rotation field from mesh geometry.

    The rotation is computed per XY column (consistent across Z heights)
    so that PrusaSlicer generates clean horizontal layers. The magnitude
    at each XY is determined by the worst overhang in that column.

    Returns:
        rotation_field: numpy array of shape (n_vertices,) with rotation in radians
    """
    from scipy.ndimage import gaussian_filter

    max_pos_rad = np.deg2rad(max_pos_rotation)
    max_neg_rad = np.deg2rad(max_neg_rotation)

    xy = mesh.points[:, :2]
    z_vals = mesh.points[:, 2]
    distances = np.linalg.norm(xy, axis=1)

    # Step 1: Compute per-vertex normals
    vertex_normals = compute_vertex_normals(mesh)

    # Step 2: Compute overhang angle per vertex (0 = up, pi = down)
    up = np.array([0.0, 0.0, 1.0])
    cos_angles = np.clip(np.dot(vertex_normals, up), -1.0, 1.0)
    overhang_angles = np.arccos(cos_angles)

    # Step 3: Compute rotation magnitude per vertex
    # Normal at 90° = vertical wall, at (90+max_overhang)° = threshold
    overhang_threshold = np.deg2rad(90.0 + max_overhang)
    needs_rotation = overhang_angles > overhang_threshold
    rotation_magnitude = np.zeros(mesh.n_points)
    rotation_magnitude[needs_rotation] = (
        overhang_angles[needs_rotation] - overhang_threshold
    ) * rotation_multiplier

    # Step 4: Determine sign from radial direction
    sign = compute_rotation_direction(mesh)
    per_vertex_rotation = rotation_magnitude * sign

    # Step 5: Project to per-XY column.
    # For each XY position, use the vertex with the largest |rotation|.
    # This ensures consistent rotation within vertical columns, giving
    # PrusaSlicer clean horizontal layers (like the radial slicer).
    COLUMN_GRID_RES = 1.0  # mm
    x_min, y_min = xy.min(axis=0) - COLUMN_GRID_RES * 2
    x_max, y_max = xy.max(axis=0) + COLUMN_GRID_RES * 2
    col_nx = int(np.ceil((x_max - x_min) / COLUMN_GRID_RES)) + 1
    col_ny = int(np.ceil((y_max - y_min) / COLUMN_GRID_RES)) + 1

    ix = np.clip(np.round((xy[:, 0] - x_min) / (x_max - x_min) * (col_nx - 1)).astype(int), 0, col_nx - 1)
    iy = np.clip(np.round((xy[:, 1] - y_min) / (y_max - y_min) * (col_ny - 1)).astype(int), 0, col_ny - 1)
    cell_idx = iy * col_nx + ix

    # Find the dominant signed rotation per column (vertex with max |rotation|)
    order = np.argsort(-np.abs(per_vertex_rotation))
    sorted_cells = cell_idx[order]
    sorted_rotations = per_vertex_rotation[order]
    _, first_idx = np.unique(sorted_cells, return_index=True)

    column_rotation = np.zeros(col_ny * col_nx)
    column_rotation[sorted_cells[first_idx]] = sorted_rotations[first_idx]
    column_grid = column_rotation.reshape(col_ny, col_nx)

    # Step 6: Gaussian smooth the 2D column map for gradual transitions.
    # This spreads overhang rotation to nearby columns, similar to how
    # the radial slicer applies rotation to the entire radius.
    if column_smooth_sigma > 0:
        column_grid = gaussian_filter(column_grid, sigma=column_smooth_sigma)

    # Step 7: Expand back to per-vertex with Z-dependent blend
    base_rotation = column_grid.ravel()[cell_idx]

    BOTTOM_ZONE_HEIGHT = 2.0  # mm: fully cartesian
    BLEND_ZONE_HEIGHT = 5.0   # mm: linear ramp from 0 to full rotation
    z_min = z_vals.min()
    bottom_threshold = z_min + BOTTOM_ZONE_HEIGHT
    blend_top = bottom_threshold + BLEND_ZONE_HEIGHT

    blend = np.ones(mesh.n_points)
    blend[z_vals < bottom_threshold] = 0.0
    blend_mask = (z_vals >= bottom_threshold) & (z_vals < blend_top)
    blend[blend_mask] = (z_vals[blend_mask] - bottom_threshold) / BLEND_ZONE_HEIGHT

    rotation_field = base_rotation * blend

    # Step 8: Force zero near center (avoid singularity)
    center_mask = distances < center_dead_zone
    rotation_field[center_mask] = 0.0

    # Step 9: Physical constraint — no vertex deforms below z=0
    for _ in range(3):
        deformed_z = z_vals / np.maximum(np.cos(rotation_field), 0.1) + \
                     np.tan(rotation_field) * distances
        violators = deformed_z < 0
        if not np.any(violators):
            break
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
