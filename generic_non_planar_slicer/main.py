def iterative_self_intersection_removal(mesh, max_rounds=5, min_cells=50):
    """Repeatedly remove triangles involved in self-intersections, with validation and fallback."""
    
    # Initial validation
    if mesh.n_points == 0 or mesh.n_cells == 0:
        print("Empty input mesh")
        return mesh
    
    # Save original mesh for fallback
    original_mesh = mesh.copy(deep=True)
    current_best = mesh
    
    for round_num in range(max_rounds):
        try:
            points = np.asarray(mesh.points)
            faces = mesh.faces.reshape(-1, 4)[:, 1:]
            
            if len(faces) < min_cells:
                print(f"Mesh too small after {round_num} rounds, reverting to best result.")
                return current_best
                
            if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                print("Invalid points detected, reverting to best result.")
                return current_best
        # Build triangle centers for KD-tree
        centers = np.mean(points[faces], axis=1)
        tree = KDTree(centers)
        radius = np.max(points.max(axis=0) - points.min(axis=0)) * 0.005
        pairs = list(tree.query_pairs(radius))
        intersection_counts = np.zeros(len(faces), dtype=int)
        for i, j in pairs:
            intersection_counts[i] += 1
            intersection_counts[j] += 1
        problem_triangles = np.argsort(intersection_counts)[::-1]
        intersection_threshold = np.mean(intersection_counts) + np.std(intersection_counts)
        triangles_to_remove = set(i for i in problem_triangles if intersection_counts[i] > intersection_threshold)
        if not triangles_to_remove:
            print(f"No more intersecting triangles after {round_num} rounds.")
            break
        keep_triangles = np.ones(len(faces), dtype=bool)
        keep_triangles[list(triangles_to_remove)] = False
        cells = np.insert(faces[keep_triangles], 0, 3, axis=1)
        mesh = pv.PolyData(points, cells)
        mesh = mesh.clean(tolerance=1e-7)
        export_mesh_diagnostics(mesh, filename_prefix=f"self_intersection_round_{round_num}")
    return mesh
import networkx as nx
import numpy as np
import pyvista as pv
import tetgen
from scipy.optimize import minimize, least_squares
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import open3d as o3d
import time
import os
import pickle
import base64

pv.global_theme.notebook = True

def encode_object(obj):
    return base64.b64encode(pickle.dumps(obj)).decode('utf-8')

def decode_object(encoded_str):
    return pickle.loads(base64.b64decode(encoded_str))

up_vector = np.array([0, 0, 1])

# Load mesh
model_name = "propeller"
# Build absolute mesh path relative to this script, so the script works when
# invoked from any current working directory.
mesh_path = os.path.join(os.path.dirname(__file__), 'input_models', f'{model_name}.stl')
# Try Open3D first, but fall back to PyVista if Open3D/ASSIMP cannot read the file.
try:
    o3d_mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(o3d_mesh.triangles) == 0 or len(o3d_mesh.vertices) == 0:
        raise RuntimeError("Open3D failed to load mesh or returned empty geometry")
    verts = np.asarray(o3d_mesh.vertices)
    tris = np.asarray(o3d_mesh.triangles)
except Exception:
    import pyvista as pv
    pvmesh = pv.read(mesh_path)
    verts = np.asarray(pvmesh.points)
    tris = pvmesh.faces.reshape(-1, 4)[:, 1:]

import json
def export_mesh_diagnostics(mesh, filename_prefix):
    """Export detailed mesh diagnostics including problematic regions for manual repair."""
    # Initial validation
    if mesh.n_points == 0 or mesh.n_cells == 0:
        print(f"Cannot export diagnostics for {filename_prefix} - empty mesh")
        return
        
    points = np.asarray(mesh.points)
    if points is None or len(points) == 0:
        print(f"Cannot export diagnostics for {filename_prefix} - no points")
        return
        
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        print(f"Cannot export diagnostics for {filename_prefix} - invalid points")
        return
        
    try:
        faces = mesh.faces.reshape(-1, 4)[:, 1:]
    except Exception as e:
        print(f"Cannot reshape faces for {filename_prefix}: {e}")
        return
        
    if len(faces) == 0:
        print(f"Cannot export diagnostics for {filename_prefix} - no faces")
        return

    # Analyze edges
    edge_counts = {}
    edge_to_faces = {}
    face_angles = {}  # Track angles for each face
    face_quality = {}  # Track quality metrics for each face
    
    for fi, tri in enumerate(faces):
        # Edge analysis
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            e = tuple(sorted((int(a), int(b))))
            edge_counts[e] = edge_counts.get(e, 0) + 1
            edge_to_faces.setdefault(e, []).append(fi)
        
        # Face quality analysis
        v0, v1, v2 = points[tri]
        e1 = np.linalg.norm(v1 - v0)
        e2 = np.linalg.norm(v2 - v1)
        e3 = np.linalg.norm(v0 - v2)
        
        # Calculate angles
        angles = []
        for i in range(3):
            p1 = points[tri[i]]
            p2 = points[tri[(i+1)%3]]
            p3 = points[tri[(i+2)%3]]
            v1 = p2 - p1
            v2 = p3 - p1
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angles.append(angle)
        face_angles[fi] = np.degrees(angles)
        
        # Calculate face quality metrics
        area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2
        perimeter = e1 + e2 + e3
        aspect_ratio = max(e1, e2, e3) / min(e1, e2, e3)
        
        face_quality[fi] = {
            "area": float(area),
            "aspect_ratio": float(aspect_ratio),
            "min_angle": float(np.min(angles)),
            "max_angle": float(np.max(angles))
        }
    
    # Identify problematic regions
    non_manifold_edges = [e for e, c in edge_counts.items() if c > 2]
    boundary_edges = [e for e, c in edge_counts.items() if c != 2]
    
    # Find problematic faces (those with bad quality metrics)
    bad_aspect_ratio = [fi for fi, q in face_quality.items() if q["aspect_ratio"] > 10]
    bad_angles = [fi for fi, angles in face_angles.items() if max(angles) > 170 or min(angles) < 10]
    tiny_faces = [fi for fi, q in face_quality.items() if q["area"] < 1e-10]
    
    # Group connected problematic regions
    def find_connected_faces(face_indices):
        """Find groups of connected problematic faces"""
        groups = []
        remaining = set(face_indices)
        while remaining:
            current = remaining.pop()
            group = {current}
            stack = [current]
            while stack:
                face = stack.pop()
                # Find neighbors through shared edges
                for v1, v2 in [(faces[face][0], faces[face][1]), 
                              (faces[face][1], faces[face][2]),
                              (faces[face][2], faces[face][0])]:
                    e = tuple(sorted((int(v1), int(v2))))
                    for neighbor in edge_to_faces.get(e, []):
                        if neighbor in remaining:
                            group.add(neighbor)
                            stack.append(neighbor)
                            remaining.remove(neighbor)
            groups.append(list(group))
        return groups
    
    problem_regions = {
        "non_manifold_regions": find_connected_faces([edge_to_faces[e][0] for e in non_manifold_edges[:100]]),
        "boundary_regions": find_connected_faces([edge_to_faces[e][0] for e in boundary_edges[:100]]),
        "bad_quality_regions": find_connected_faces(bad_aspect_ratio + bad_angles + tiny_faces)
    }
    
    diagnostics = {
        "mesh_stats": {
            "num_points": int(points.shape[0]),
            "num_faces": int(faces.shape[0]),
            "num_non_manifold_edges": len(non_manifold_edges),
            "num_boundary_edges": len(boundary_edges),
            "bbox": {
                "min": points.min(axis=0).tolist(),
                "max": points.max(axis=0).tolist()
            }
        },
        "problematic_elements": {
            "non_manifold_edges": non_manifold_edges[:100],
            "boundary_edges": boundary_edges[:100],
            "bad_aspect_ratio_faces": bad_aspect_ratio[:100],
            "bad_angle_faces": bad_angles[:100],
            "tiny_faces": tiny_faces[:100]
        },
        "problem_regions": problem_regions,
        "quality_metrics": {
            "worst_aspect_ratios": sorted(
                [(fi, q["aspect_ratio"]) for fi, q in face_quality.items()],
                key=lambda x: x[1], reverse=True
            )[:20],
            "smallest_faces": sorted(
                [(fi, q["area"]) for fi, q in face_quality.items()],
                key=lambda x: x[1]
            )[:20]
        },
        "repair_hints": {
            "non_manifold_edges": {str(e): edge_to_faces[e] for e in non_manifold_edges[:20]},
            "boundary_faces": {str(e): edge_to_faces[e] for e in boundary_edges[:20]}
        }
    }
    
    # Save detailed diagnostics
    with open(os.path.join(os.path.dirname(__file__), 'output_models', f'{filename_prefix}_diagnostics.json'), 'w') as f:
        json.dump(diagnostics, f, indent=2)
        
    def safe_save(mesh_obj, base_path):
        """Safely save mesh in VTK format based on its type"""
        try:
            # Try VTP first (PolyData)
            mesh_obj.save(f"{base_path}.vtp")
        except ValueError:
            try:
                # Try VTU second (UnstructuredGrid)
                mesh_obj.save(f"{base_path}.vtu")
            except ValueError:
                # Fallback to VTK
                mesh_obj.save(f"{base_path}.vtk")
        
        # Also try STL for visualization compatibility
        try:
            mesh_obj.save(f"{base_path}.stl")
        except Exception:
            print(f"Warning: Could not save STL format for {base_path}")
    
    # Export the full mesh
    mesh_path = os.path.join(os.path.dirname(__file__), 'output_models', f'{filename_prefix}_diagnostics')
    safe_save(mesh, mesh_path)
    
    # Also export problematic regions separately
    for region_type, region_groups in problem_regions.items():
        for i, group in enumerate(region_groups[:5]):  # Export first 5 groups of each type
            problem_faces = np.array(group)
            sub_mesh = mesh.extract_cells(problem_faces)
            export_path = os.path.join(os.path.dirname(__file__), 
                         'output_models', 
                         f'{filename_prefix}_{region_type}_group_{i}')
            safe_save(sub_mesh, export_path)

try:
    print("Running early pymeshfix repair on loaded surface...")
    import pymeshfix

    verts_in = np.asarray(verts, dtype=np.float64)
    tris_in = np.asarray(tris, dtype=np.int32)

    if len(tris_in) > 0 and tris_in.max() < len(verts_in):
        mf = pymeshfix.MeshFix(verts_in, tris_in)
        mf.repair()
        fixed_v, fixed_f = mf.v, mf.f

        if fixed_f is not None and len(fixed_f) > 0:
            print(f"pymeshfix repair: produced {len(fixed_v)} vertices and {len(fixed_f)} faces")
            verts = fixed_v
            tris = fixed_f
        else:
            print("pymeshfix repair produced no faces")
    else:
        print("Invalid input mesh for pymeshfix")
except Exception as _e:
    print(f"Early pymeshfix step skipped or failed: {_e}")

# Scale the mesh to help with numerical stability
scale_factor = 10.0  # Scale up by 10x
verts = verts * scale_factor

# Convert to tetrahedral mesh
print("Attempting tetrahedralization...")
input_tet = tetgen.TetGen(verts, tris)

# Set tetrahedralization parameters with progressive refinement strategy
tetra_args = {
    'switches': 'pq3.0a0.3YQ',  # More relaxed quality bounds + split + quiet
    'minratio': 3.0,           # Further increased maximum radius-edge ratio
    'mindihedral': 2.0,        # Even more reduced minimum dihedral angle for better convergence
    'verbose': 1,              # Output detailed information
    'maxvolume': None,         # No volume constraint
    'nobisect': True,          # Prevent splitting of boundary facets
    'quality': True,           # Enforce quality mesh generation
    'steinerleft': 100000      # Allow more Steiner points for better quality
}

# Add comprehensive validation
def get_neighboring_vertices(vertices, faces):
    """Build map of vertices that share edges"""
    neighbors = {}
    for face in faces:
        for i in range(3):
            v1, v2 = int(face[i]), int(face[(i+1) % 3])
            if v1 not in neighbors:
                neighbors[v1] = set()
            if v2 not in neighbors:
                neighbors[v2] = set()
            neighbors[v1].add(v2)
            neighbors[v2].add(v1)
    return neighbors

def find_and_fix_non_manifold_edges(vertices, faces, max_iterations=5):
    """Iteratively find and fix non-manifold edges by grouping connected non-manifold
    regions, creating centroids for each group and remapping the group's vertices
    to the centroid. This is conservative (we only merge when edges are part of a
    non-manifold set) and runs a few iterations to reduce remaining non-manifold edges.

    Returns cleaned (vertices, faces) with unused vertices removed.
    """
    if len(faces) == 0:
        return vertices, faces

    original_vertices = vertices.copy()

    for iteration in range(max_iterations):
        # Recompute edge adjacency counts
        edge_count = {}
        neighbors = get_neighboring_vertices(vertices, faces)
        for v1, v_neighbors in neighbors.items():
            for v2 in v_neighbors:
                edge = tuple(sorted((v1, v2)))
                edge_count[edge] = edge_count.get(edge, 0) + 1

        non_manifold_edges = [edge for edge, c in edge_count.items() if c > 2]
        if not non_manifold_edges:
            # no non-manifold edges remaining
            break

        print(f"find_and_fix_non_manifold_edges: iteration {iteration+1}, found {len(non_manifold_edges)} non-manifold edges")

        # Group connected non-manifold edges so we can process clusters together
        remaining = set(non_manifold_edges)
        groups = []
        while remaining:
            stack = [remaining.pop()]
            group = set()
            while stack:
                e = stack.pop()
                group.add(e)
                # find any remaining edges that share a vertex
                to_remove = []
                for other in list(remaining):
                    if set(e) & set(other):
                        stack.append(other)
                        to_remove.append(other)
                for r in to_remove:
                    remaining.remove(r)
            groups.append(group)

        # For each group, create a centroid vertex and remap group's vertices to it
        vertex_map = np.arange(len(vertices))
        new_vertices = vertices.tolist()
        for group in groups:
            verts_in_group = set()
            for e in group:
                verts_in_group.update(e)
            verts_in_group = sorted(list(verts_in_group))
            # compute centroid in original coordinate space for stability
            centroid = np.mean([original_vertices[v] for v in verts_in_group], axis=0)
            new_idx = len(new_vertices)
            new_vertices.append(centroid)
            for v in verts_in_group:
                vertex_map[v] = new_idx

        # Apply mapping to faces and drop degenerate ones
        new_faces = []
        for face in faces:
            mapped = [vertex_map[v] if v < len(vertex_map) else v for v in face]
            if len(set(mapped)) == 3:
                new_faces.append(mapped)

        vertices = np.array(new_vertices)
        faces = np.array(new_faces, dtype=int)

        # Continue to next iteration to see if non-manifold count reduced

    # final compacting: remove unreferenced vertices and reindex
    if len(faces) == 0:
        return vertices, faces

    used = np.unique(faces.flatten())
    remap = {old: new for new, old in enumerate(used)}
    compact_vertices = vertices[used]
    compact_faces = np.array([[remap[v] for v in face] for face in faces], dtype=int)

    return compact_vertices, compact_faces

def check_face_orientation(vertices, faces):
    """Check that all triangles have consistent orientation"""
    # Calculate face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]] 
    v2 = vertices[faces[:, 2]]
    norms = np.cross(v1 - v0, v2 - v0)
    # Check sign consistency along connected faces
    neighbors = get_neighboring_vertices(vertices, faces)
    consistent = True
    for i, face1 in enumerate(faces):
        for j, face2 in enumerate(faces[i+1:]):
            if len(set(face1) & set(face2)) == 2: # Faces share edge
                if np.dot(norms[i], norms[j]) < 0:
                    consistent = False
                    break
        if not consistent:
            break
    return consistent

def check_face_quality(vertices, faces):
    """Check geometric quality of faces"""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Edge lengths
    e1 = np.linalg.norm(v1 - v0, axis=1)
    e2 = np.linalg.norm(v2 - v1, axis=1)
    e3 = np.linalg.norm(v0 - v2, axis=1)
    
    # Area from cross product
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    
    # Aspect ratio (ratio of longest to shortest edge)
    max_edge = np.maximum.reduce([e1, e2, e3])
    min_edge = np.minimum.reduce([e1, e2, e3])
    aspect_ratio = max_edge / min_edge
    
    # Quality checks
    min_area = 1e-8
    max_aspect = 100
    min_edge_len = 1e-5
    
    bad_faces = np.where(
        (area < min_area) | 
        (aspect_ratio > max_aspect) |
        (min_edge < min_edge_len)
    )[0]
    
    return bad_faces

def subdivide_bad_triangles(vertices, faces):
    """Subdivide triangles with poor quality"""
    bad_faces = check_face_quality(vertices, faces)
    if len(bad_faces) == 0:
        return vertices, faces
        
    new_vertices = vertices.copy()
    new_faces = []
    
    for i, face in enumerate(faces):
        if i in bad_faces:
            # Calculate midpoints
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            # Add new vertices at edge midpoints
            m01 = 0.5 * (v0 + v1)
            m12 = 0.5 * (v1 + v2) 
            m20 = 0.5 * (v2 + v0)
            
            # Add new vertices
            new_verts = [m01, m12, m20]
            start_idx = len(new_vertices)
            new_vertices = np.vstack([new_vertices] + new_verts)
            
            # Create 4 new triangles
            new_faces.extend([
                [face[0], start_idx, start_idx+2],
                [start_idx, face[1], start_idx+1],
                [start_idx+2, start_idx+1, face[2]],
                [start_idx, start_idx+1, start_idx+2],
            ])
        else:
            new_faces.append(face)
            
    return new_vertices, np.array(new_faces)
def repair_degenerate_triangles(vertices, faces):
    """Fix degenerate or poor quality triangles using advanced repair strategies."""
    def triangle_quality(tri_verts):
        """Calculate comprehensive triangle quality metrics"""
        edges = np.roll(tri_verts, -1, axis=0) - tri_verts
        edge_lengths = np.linalg.norm(edges, axis=1)
        area = np.linalg.norm(np.cross(edges[0], edges[1])) / 2
        
        # Avoid division by zero
        min_edge = np.min(edge_lengths)
        if min_edge < 1e-10:
            return 0, 0, True  # Quality 0, area 0, is_degenerate True
            
        # Calculate aspect ratio and minimum height
        perimeter = np.sum(edge_lengths)
        aspect_ratio = np.max(edge_lengths) / min_edge
        heights = [area * 2 / length for length in edge_lengths]
        min_height = min(heights)
        
        # Combined quality score (0 = worst, 1 = best)
        quality = min(1.0, 
                     4 * np.sqrt(3) * area / (perimeter * np.max(edge_lengths)))
                     
        is_degenerate = (area < 1e-10 or aspect_ratio > 100 or min_height < 1e-8)
        
        return quality, area, is_degenerate

    def find_valid_vertex_position(v0, v1, v2, target_area=None):
        """Find a valid position for v2 that creates a non-degenerate triangle"""
        if target_area is None:
            # Use average edge length to estimate desired area
            avg_edge = np.linalg.norm(v1 - v0)
            target_area = avg_edge * avg_edge * np.sqrt(3) / 4
            
        # Vector from v0 to v1
        edge = v1 - v0
        edge_length = np.linalg.norm(edge)
        
        if edge_length < 1e-10:
            # Points too close, create an equilateral triangle
            edge = np.array([1e-8, 0, 0])
            edge_length = 1e-8
            
        # Create an orthogonal vector
        if abs(edge[2]) < abs(edge[0]):
            ortho = np.cross(edge, [0, 0, 1])
        else:
            ortho = np.cross(edge, [1, 0, 0])
            
        ortho = ortho / np.linalg.norm(ortho)
        
        # Calculate height needed for target area
        height = 2 * target_area / edge_length
        
        # Move v2 to create proper triangle
        new_v2 = v0 + edge/2 + ortho * height
        
        # Verify the new triangle
        quality, area, is_degenerate = triangle_quality(np.array([v0, v1, new_v2]))
        
        if is_degenerate:
            # If still degenerate, try different height
            new_v2 = v0 + edge/2 + ortho * height * 2
            
        return new_v2

    repaired_faces = []
    new_vertices = vertices.copy()
    vertices_added = 0
    faces_repaired = 0
    
    for face_idx, face in enumerate(faces):
        v0, v1, v2 = vertices[face]
        quality, area, is_degenerate = triangle_quality(np.array([v0, v1, v2]))
        
        if is_degenerate:
            faces_repaired += 1
            
            # Strategy 1: Try fixing by adjusting one vertex
            new_v2 = find_valid_vertex_position(v0, v1, v2)
            
            # Add new vertex
            new_vert_idx = len(new_vertices)
            new_vertices = np.vstack([new_vertices, new_v2])
            vertices_added += 1
            
            # Replace the degenerate triangle
            repaired_faces.append([face[0], face[1], new_vert_idx])
            
        else:
            repaired_faces.append(face)
    
    if faces_repaired > 0:
        print(f"Fixed {faces_repaired} degenerate triangles")
    
    return new_vertices, np.array(repaired_faces)

def subdivide_bad_triangles(vertices, faces):
    """Subdivide triangles with poor quality, avoiding creation of degenerate triangles"""
    bad_faces = check_face_quality(vertices, faces)
    if len(bad_faces) == 0:
        return vertices, faces
        
    new_vertices = vertices.copy()
    new_faces = []
    
    def safe_midpoint(v1, v2, jitter=1e-8):
        """Calculate midpoint with small random jitter to avoid degeneracy"""
        mid = 0.5 * (v1 + v2)
        if np.allclose(v1, v2, atol=1e-10):
            # Add jitter in random direction if points are too close
            jitter_vec = np.random.normal(0, jitter, 3)
            jitter_vec = jitter_vec / np.linalg.norm(jitter_vec)
            mid += jitter_vec
        return mid
    
    for i, face in enumerate(faces):
        if i in bad_faces:
            # Calculate midpoints with jitter
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            # Add new vertices at edge midpoints
            m01 = safe_midpoint(v0, v1)
            m12 = safe_midpoint(v1, v2)
            m20 = safe_midpoint(v2, v0)
            
            # Verify the new triangles won't be degenerate
            new_verts = [m01, m12, m20]
            start_idx = len(new_vertices)
            new_vertices = np.vstack([new_vertices] + new_verts)
            
            # Create 4 new triangles with careful orientation
            candidate_faces = [
                [face[0], start_idx, start_idx+2],
                [start_idx, face[1], start_idx+1],
                [start_idx+2, start_idx+1, face[2]],
                [start_idx, start_idx+1, start_idx+2],
            ]
            
            # Verify each new triangle
            for new_face in candidate_faces:
                tri_verts = np.array([new_vertices[idx] for idx in new_face])
                # Calculate normal and ensure proper orientation
                normal = np.cross(tri_verts[1] - tri_verts[0], 
                                tri_verts[2] - tri_verts[0])
                if np.dot(normal, [0, 0, 1]) < 0:  # If normal points down
                    new_face = [new_face[0], new_face[2], new_face[1]]
                new_faces.append(new_face)
        else:
            new_faces.append(face)
    
    return new_vertices, np.array(new_faces)
def fix_self_intersections_enhanced(vertices, faces):
    """Enhanced self-intersection removal with validation and fallback"""
    from scipy.spatial import KDTree
    
    # Validation
    if len(vertices) == 0 or len(faces) == 0:
        print("Empty input mesh")
        return vertices, faces
        
    # Save original state for fallback
    original_vertices = vertices.copy()
    original_faces = faces.copy()
    
    def triangle_aabb(tri_verts):
        """Calculate axis-aligned bounding box for a triangle"""
        return np.min(tri_verts, axis=0), np.max(tri_verts, axis=0)
    
    def check_triangle_intersection(tri1_verts, tri2_verts):
        """Quick check if two triangles might intersect using AABB"""
        min1, max1 = triangle_aabb(tri1_verts)
        min2, max2 = triangle_aabb(tri2_verts)
        return np.all(min1 <= max2) and np.all(min2 <= max1)
    
    # Calculate triangle centroids for KD-tree
    centroids = np.mean([vertices[faces[:, 0]], 
                        vertices[faces[:, 1]], 
                        vertices[faces[:, 2]]], axis=0)
    
    # Build KD-tree for fast neighbor queries
    tree = KDTree(centroids)
    
    # Find potential intersecting triangle pairs
    radius = np.mean([np.linalg.norm(vertices[f[1]] - vertices[f[0]]) for f in faces])
    pairs = tree.query_pairs(radius)
    
    # Track triangles to remove
    triangles_to_remove = set()
    
    # Check each potential pair
    for t1, t2 in pairs:
        if len(set(faces[t1]) & set(faces[t2])) == 0:  # Non-adjacent triangles
            tri1_verts = vertices[faces[t1]]
            tri2_verts = vertices[faces[t2]]
            if check_triangle_intersection(tri1_verts, tri2_verts):
                # Remove triangle with worse aspect ratio
                ar1 = triangle_aspect_ratio(tri1_verts)
                ar2 = triangle_aspect_ratio(tri2_verts)
                if ar1 > ar2:
                    triangles_to_remove.add(t1)
                else:
                    triangles_to_remove.add(t2)
    
    # Remove problematic triangles
    if triangles_to_remove:
        mask = np.ones(len(faces), dtype=bool)
        mask[list(triangles_to_remove)] = False
        faces = faces[mask]
        print(f"Removed {len(triangles_to_remove)} intersecting triangles")
    
    return vertices, faces

def triangle_aspect_ratio(tri_verts):
    """Calculate triangle aspect ratio"""
    edges = np.array([
        np.linalg.norm(tri_verts[1] - tri_verts[0]),
        np.linalg.norm(tri_verts[2] - tri_verts[1]),
        np.linalg.norm(tri_verts[0] - tri_verts[2])
    ])
    return np.max(edges) / np.min(edges)

def fill_holes(vertices, faces):
    """Fill holes in the mesh by identifying boundary edges and creating triangles.
    Uses a conservative filling approach that checks for non-manifold edges.
    """
    def edge_manifold_test(edge, edge_counts):
        """Check if adding a face would make edge non-manifold"""
        count = edge_counts.get(edge, 0)
        return count < 2

    def face_manifold_test(v0, v1, v2, edge_counts):
        """Check if adding a new face would create non-manifold edges"""
        e1 = tuple(sorted([v0, v1]))
        e2 = tuple(sorted([v1, v2]))
        e3 = tuple(sorted([v2, v0]))
        return all(edge_manifold_test(e, edge_counts) for e in [e1, e2, e3])
    
    def compute_hole_quality(hole_verts):
        """Compute quality metrics for hole to prioritize simpler holes"""
        if len(hole_verts) < 3:
            return float('inf')
            
        # Get hole properties
        diameter = max(np.linalg.norm(vertices[v1] - vertices[v2]) 
                      for i, v1 in enumerate(hole_verts)
                      for v2 in hole_verts[i+1:])
        
        # Calculate planarity score
        centroid = np.mean([vertices[v] for v in hole_verts], axis=0)
        normal = np.zeros(3)
        for i in range(len(hole_verts)):
            v1 = vertices[hole_verts[i]]
            v2 = vertices[hole_verts[(i+1)%len(hole_verts)]]
            normal += np.cross(v1 - centroid, v2 - centroid)
        planarity = np.linalg.norm(normal) / (len(hole_verts) * diameter * diameter)
        
        # Consider hole size and shape
        perimeter = sum(np.linalg.norm(vertices[hole_verts[(i+1)%len(hole_verts)]] - vertices[v])
                       for i, v in enumerate(hole_verts))
        area = abs(np.cross(vertices[hole_verts[1]] - vertices[hole_verts[0]],
                           vertices[hole_verts[2]] - vertices[hole_verts[0]])).sum() / 2
                           
        # Combined score (lower is better)
        score = (1 - planarity) * perimeter / (area + 1e-10) * len(hole_verts)
        return score

    # Build initial edge counts
    edge_counts = {}
    for face in faces:
        for i in range(3):
            edge = tuple(sorted([int(face[i]), int(face[(i+1)%3])]))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    # Find boundary edges
    boundary_edges = []
    for face in faces:
        for i in range(3):
            edge = tuple(sorted([int(face[i]), int(face[(i+1)%3])]))
            if edge_counts.get(edge, 0) == 1:
                boundary_edges.append(edge)

    if not boundary_edges:
        return vertices, faces

    print(f"Found {len(boundary_edges)} boundary edges")

    # Group boundary edges into holes
    holes = []
    remaining_edges = set(map(tuple, boundary_edges))
    while remaining_edges:
        current_hole = []
        start_edge = remaining_edges.pop()
        current_hole.append(start_edge)
        current_vertex = start_edge[1]
        
        while len(current_hole) < len(boundary_edges):
            found_next = False
            for edge in remaining_edges:
                if edge[0] == current_vertex:
                    current_hole.append(edge)
                    remaining_edges.remove(edge)
                    current_vertex = edge[1]
                    found_next = True
                    break
                elif edge[1] == current_vertex:
                    current_hole.append(tuple(reversed(edge)))
                    remaining_edges.remove(edge)
                    current_vertex = edge[0]
                    found_next = True
                    break
            if not found_next:
                break
                
        if len(current_hole) >= 3:
            # Extract unique vertices forming the hole
            hole_verts = []
            for edge in current_hole:
                if edge[0] not in hole_verts:
                    hole_verts.append(edge[0])
                if edge[1] not in hole_verts:
                    hole_verts.append(edge[1])
            holes.append(hole_verts)

    # Sort holes by quality (fill best holes first)
    holes.sort(key=compute_hole_quality)

    new_faces = list(faces)
    local_edge_counts = edge_counts.copy()

    def try_fill_hole(hole_verts):
        """Attempt to fill a hole conservatively"""
        if len(hole_verts) < 3:
            return False
            
        # Try fan triangulation from best vertex
        best_count = float('inf')
        best_tris = None
        
        for start_idx in range(len(hole_verts)):
            v0 = hole_verts[start_idx]
            candidate_tris = []
            valid = True
            
            for i in range(1, len(hole_verts)-1):
                v1 = hole_verts[(start_idx+i)%len(hole_verts)]
                v2 = hole_verts[(start_idx+i+1)%len(hole_verts)]
                
                # Check if adding this triangle would create non-manifold edges
                if not face_manifold_test(v0, v1, v2, local_edge_counts):
                    valid = False
                    break
                    
                candidate_tris.append([v0, v1, v2])
                
            if valid and len(candidate_tris) < best_count:
                best_count = len(candidate_tris)
                best_tris = candidate_tris
                
        if best_tris:
            # Update edge counts and add triangles
            for tri in best_tris:
                new_faces.append(tri)
                for i in range(3):
                    edge = tuple(sorted([tri[i], tri[(i+1)%3]]))
                    local_edge_counts[edge] = local_edge_counts.get(edge, 0) + 1
            return True
            
        return False

    # Try to fill each hole
    holes_filled = 0
    for hole_verts in holes:
        if try_fill_hole(hole_verts):
            holes_filled += 1
            
    print(f"Successfully filled {holes_filled} holes")
    return vertices, np.array(new_faces)


def fix_non_manifold_by_face_pruning(vertices, faces, max_remove_ratio=0.02):
    """Attempt to fix remaining non-manifold edges by pruning faces adjacent to
    the worst non-manifold edges. This is a destructive but effective fallback:
    for each non-manifold edge, remove the face(s) with the worst quality until
    the edge has <= 2 adjacent faces. We cap the number of removed faces to a
    small fraction (max_remove_ratio) of the total faces to avoid over-pruning.
    """
    if len(faces) == 0:
        return vertices, faces

    # Build edge -> faces map
    edge_to_faces = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            e = tuple(sorted((int(face[i]), int(face[(i+1)%3]))))
            edge_to_faces.setdefault(e, []).append(fi)

    # Identify problematic edges (>2 adjacent faces)
    bad_edges = [e for e, flist in edge_to_faces.items() if len(flist) > 2]
    if not bad_edges:
        return vertices, faces

    # Precompute face quality (aspect ratio) so we can remove worst faces first
    def face_aspect_ratio(face):
        v0, v1, v2 = vertices[face]
        e1 = np.linalg.norm(v1 - v0)
        e2 = np.linalg.norm(v2 - v1)
        e3 = np.linalg.norm(v0 - v2)
        min_e = max(min(e1, e2, e3), 1e-12)
        return max(e1, e2, e3) / min_e

    face_scores = np.array([face_aspect_ratio(f) for f in faces])

    faces_to_remove = set()
    for edge in bad_edges:
        adjacent = list(edge_to_faces[edge])
        # Sort by worst quality first (higher aspect ratio)
        adjacent_sorted = sorted(adjacent, key=lambda fi: face_scores[fi], reverse=True)
        # Remove faces until adjacency <= 2
        while len(adjacent_sorted) > 2 and len(faces_to_remove) < int(len(faces) * max_remove_ratio):
            rem = adjacent_sorted.pop(0)
            faces_to_remove.add(rem)

    if not faces_to_remove:
        return vertices, faces

    mask = np.ones(len(faces), dtype=bool)
    mask[list(faces_to_remove)] = False
    new_faces = faces[mask]

    # compact vertices
    used = np.unique(new_faces.flatten())
    remap = {old: new for new, old in enumerate(used)}
    compact_vertices = vertices[used]
    compact_faces = np.array([[remap[v] for v in face] for face in new_faces], dtype=int)

    print(f"Pruned {len(faces_to_remove)} faces to reduce non-manifold edges")
    return compact_vertices, compact_faces


def keep_largest_component(vertices, faces):
    """Keep only the largest connected component (by number of faces).
    This is a fallback to remove small disconnected or problematic components
    that often cause non-manifold edge counts to persist.
    """
    faces_pv = np.column_stack((np.full(len(faces), 3), faces)).flatten()
    mesh = pv.PolyData(vertices, faces_pv)
    if mesh.n_cells == 0:
        return vertices, faces
    conn = mesh.connectivity()
    if conn is None or 'RegionId' not in conn.cell_data:
        return vertices, faces
    labels = conn.cell_data['RegionId']
    # find region with most cells
    unique, counts = np.unique(labels, return_counts=True)
    largest_region = unique[np.argmax(counts)]
    sel = np.where(labels == largest_region)[0]
    if len(sel) == 0:
        return vertices, faces
    try:
        new_mesh = mesh.extract_cells(sel)
        if new_mesh is None or new_mesh.n_cells == 0:
            return vertices, faces
        new_faces = new_mesh.faces.reshape(-1, 4)[:, 1:]
        new_vertices = new_mesh.points
        return new_vertices, new_faces
    except Exception:
        # If extraction fails for any reason, fall back to returning the input
        return vertices, faces


def split_non_manifold_edges_by_vertex_duplication(vertices, faces):
    """For each non-manifold edge (edge shared by >2 faces), duplicate one of the
    edge vertices for excess faces so that each edge is shared by at most two faces.
    This resolves non-manifold edges by separating sheets rather than deleting geometry.
    """
    edge_to_faces = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            e = tuple(sorted((int(face[i]), int(face[(i+1)%3]))))
            edge_to_faces.setdefault(e, []).append(fi)

    vertices = vertices.tolist()
    faces = faces.copy().tolist()
    removed = 0

    for edge, flist in list(edge_to_faces.items()):
        if len(flist) <= 2:
            continue
        # keep first two faces attached to this edge; for others, duplicate one vertex
        keep = set(flist[:2])
        for fi in flist[2:]:
            if fi in keep:
                continue
            face = faces[fi]
            # decide which vertex to duplicate: choose the vertex with higher valence
            v_candidates = list(edge)
            # create a duplicate of the first vertex
            dup_idx = len(vertices)
            vertices.append(np.array(vertices[v_candidates[0]]))
            # replace occurrences of the original vertex in this face with duplicated index
            faces[fi] = [dup_idx if v == v_candidates[0] else v for v in face]
            removed += 1

    if removed == 0:
        return np.array(vertices), np.array(faces, dtype=int)

    # compact vertices and reindex
    faces_arr = np.array(faces, dtype=int)
    used = np.unique(faces_arr.flatten())
    remap = {old: new for new, old in enumerate(used)}
    compact_vertices = np.array(vertices)[used]
    compact_faces = np.array([[remap[v] for v in face] for face in faces_arr], dtype=int)

    print(f"Duplicated vertices for {removed} faces to split non-manifold edges")
    return compact_vertices, compact_faces


def aggressive_split_non_manifold_edges(vertices, faces, max_rounds=5):
    """Repeatedly split non-manifold edges by duplicating vertices for excess faces.
    This is more aggressive and iterative than the simple one-shot splitter and will
    continue until no non-manifold edges remain or max_rounds is reached.
    Returns (vertices, faces) possibly modified.
    """
    for round_i in range(max_rounds):
        # Build edge -> faces map
        edge_to_faces = {}
        for fi, face in enumerate(faces):
            for i in range(3):
                e = tuple(sorted((int(face[i]), int(face[(i+1)%3]))))
                edge_to_faces.setdefault(e, []).append(fi)

        bad_edges = [e for e, flist in edge_to_faces.items() if len(flist) > 2]
        if not bad_edges:
            break

        vertices = vertices.tolist()
        faces = faces.copy().tolist()
        changed = False

        # For each bad edge, duplicate vertices for excess faces
        for edge in bad_edges:
            flist = edge_to_faces[edge]
            # keep first two faces, duplicate for remaining
            kept = set(flist[:2])
            v0, v1 = edge
            for fi in flist[2:]:
                if fi in kept:
                    continue
                face = faces[fi]
                # choose which endpoint to duplicate by valence heuristic
                # compute valence (how many times vertex appears in faces)
                v0_count = sum([1 for f in faces if v0 in f])
                v1_count = sum([1 for f in faces if v1 in f])
                dup_v = v0 if v0_count >= v1_count else v1

                dup_idx = len(vertices)
                vertices.append(np.array(vertices[dup_v]))
                # replace dup_v in this face with dup_idx
                faces[fi] = [dup_idx if v == dup_v else v for v in face]
                changed = True

        # Compact and reindex if changed
        if not changed:
            break

        faces_arr = np.array(faces, dtype=int)
        used = np.unique(faces_arr.flatten())
        remap = {old: new for new, old in enumerate(used)}
        compact_vertices = np.array(vertices)[used]
        compact_faces = np.array([[remap[v] for v in face] for face in faces_arr], dtype=int)

        vertices = compact_vertices
        faces = compact_faces

        print(f"aggressive_split_non_manifold_edges: round {round_i+1} completed, remaining bad edges will be recomputed")

    return np.array(vertices), np.array(faces, dtype=int)


def targeted_prune_edges(vertices, faces, edge_list, max_remove_per_edge=10):
    """For each edge in edge_list (edges given as tuples), remove worst-quality
    adjacent faces until the edge has at most 2 adjacent faces or we've removed
    max_remove_per_edge faces for that edge.
    Returns compacted (vertices, faces).
    """
    if len(faces) == 0:
        return vertices, faces

    faces = faces.copy()
    vertices = vertices.copy()

    # Function to compute face quality metrics
    def face_metrics(face):
        """Return multiple quality metrics for a face"""
        v0, v1, v2 = vertices[face]
        e1 = np.linalg.norm(v1 - v0)
        e2 = np.linalg.norm(v2 - v1)
        e3 = np.linalg.norm(v0 - v2)
        min_e = max(min(e1, e2, e3), 1e-12)
        max_e = max(e1, e2, e3)
        area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2
        
        # Calculate geometric quality metrics
        aspect_ratio = max_e / min_e
        edge_ratio = max_e / min_e  
        area_metric = np.exp(-area * 100) if area > 0 else float('inf')
        
        # Combined quality score (weighted sum, higher is worse)
        quality_score = (
            0.4 * aspect_ratio +
            0.3 * edge_ratio + 
            0.3 * area_metric
        )
        
        # Return all metrics for multi-pass evaluation
        return quality_score, min_e, area

    # Build edge->faces map
    edge_to_faces = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            e = tuple(sorted((int(face[i]), int(face[(i+1)%3]))))
            edge_to_faces.setdefault(e, []).append(fi)

    faces_to_remove = set()
    # Track edges that weren't fully fixed for second pass
    still_bad_edges = []

    # First pass: conservative removal of single worst face per edge
    for edge in edge_list:
        flist = edge_to_faces.get(edge, [])
        if len(flist) <= 2:
            continue
        # compute quality for adjacent faces
        scored = [(fi, *face_metrics(faces[fi])) for fi in flist]
        # Sort by overall quality score (worst first) 
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Only remove single worst face in first pass
        if len(flist) > 2:
            worst_face = scored[0][0]
            if worst_face not in faces_to_remove:
                faces_to_remove.add(worst_face)
            
            # If edge still has >2 faces, track for second pass
            if len(flist) - 1 > 2:
                still_bad_edges.append((edge, flist))

    # Second pass: aggressive pruning keeping only best 2 faces
    if still_bad_edges:
        print(f"First pass left {len(still_bad_edges)} edges, attempting second pass...")
        for edge, flist in still_bad_edges:
            remaining = [f for f in flist if f not in faces_to_remove]
            if len(remaining) <= 2:
                continue
                
            # Re-score remaining faces using quality metrics
            scored = [(fi, *face_metrics(faces[fi])) for fi in remaining]
            # Sort by quality - LOWER scores are better quality
            scored.sort(key=lambda x: x[1])
            
            # Keep best two faces, remove all others
            faces_to_remove.update(fi for fi, _, _, _ in scored[2:])

    if faces_to_remove:
        mask = np.ones(len(faces), dtype=bool)
        mask[list(faces_to_remove)] = False
        new_faces = faces[mask]
        used = np.unique(new_faces.flatten())
        remap = {old: new for new, old in enumerate(used)}
        compact_vertices = vertices[used]
        compact_faces = np.array([[remap[v] for v in face] for face in new_faces], dtype=int)
        print(f"Pruned {len(faces_to_remove)} faces across {len(edge_list)} edges")
        return compact_vertices, compact_faces

    return vertices, faces

def repair_mesh(mesh):
    """Complete mesh repair pipeline with comprehensive diagnostics"""
    vertices = np.asarray(mesh.points)
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    
    print("Starting mesh repair pipeline...")
    print(f"Initial mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Export initial state diagnostics
    initial_mesh = pv.PolyData(vertices, np.column_stack((np.full(len(faces), 3), faces)).flatten())
    export_mesh_diagnostics(initial_mesh, "repair_initial_state")
    
    # Initial cleanup of non-manifold edges
    vertices, faces = find_and_fix_non_manifold_edges(vertices, faces)
    print(f"After non-manifold edge fix: {len(vertices)} vertices, {len(faces)} faces")
    current_mesh = pv.PolyData(vertices, np.column_stack((np.full(len(faces), 3), faces)).flatten())
    export_mesh_diagnostics(current_mesh, "repair_after_non_manifold_fix")
    
    # Check orientation and fix if needed
    if not check_face_orientation(vertices, faces):
        print("Fixing face orientations...")
        faces = np.flip(faces, axis=1)
        current_mesh = pv.PolyData(vertices, np.column_stack((np.full(len(faces), 3), faces)).flatten())
        export_mesh_diagnostics(current_mesh, "repair_after_orientation_fix")
    
    # Fix self-intersections
    print("Removing self-intersections...")
    vertices, faces = fix_self_intersections_enhanced(vertices, faces)
    print(f"After intersection removal: {len(vertices)} vertices, {len(faces)} faces")
    current_mesh = pv.PolyData(vertices, np.column_stack((np.full(len(faces), 3), faces)).flatten())
    export_mesh_diagnostics(current_mesh, "repair_after_intersection_removal")
    
    # Fill holes
    print("Filling holes...")
    vertices, faces = fill_holes(vertices, faces)
    print(f"After hole filling: {len(vertices)} vertices, {len(faces)} faces")
    current_mesh = pv.PolyData(vertices, np.column_stack((np.full(len(faces), 3), faces)).flatten())
    export_mesh_diagnostics(current_mesh, "repair_after_hole_filling")

    # Quick check for remaining non-manifold edges; attempt iterative pruning if needed
    for prune_iter, ratio in enumerate([0.005, 0.01, 0.02, 0.05, 0.1]):
        # Build edge counts
        edge_count = {}
        for face in faces:
            for i in range(3):
                e = tuple(sorted((int(face[i]), int(face[(i+1)%3]))))
                edge_count[e] = edge_count.get(e, 0) + 1
        remaining_non_manifold = [e for e, c in edge_count.items() if c > 2]
        if not remaining_non_manifold:
            break

        print(f"Remaining non-manifold edges after fill_holes: {len(remaining_non_manifold)}; attempt {prune_iter+1} pruning with ratio={ratio}")
        vertices, faces = fix_non_manifold_by_face_pruning(vertices, faces, max_remove_ratio=ratio)
        print(f"After pruning pass {prune_iter+1}: {len(vertices)} vertices, {len(faces)} faces")
    # Final aggressive fallback if there are still a few non-manifold edges
    edge_count = {}
    for face in faces:
        for i in range(3):
            e = tuple(sorted((int(face[i]), int(face[(i+1)%3]))))
            edge_count[e] = edge_count.get(e, 0) + 1
    remaining_non_manifold = [e for e, c in edge_count.items() if c > 2]
    if remaining_non_manifold:
        print(f"Final aggressive prune for remaining {len(remaining_non_manifold)} non-manifold edges")
        vertices, faces = fix_non_manifold_by_face_pruning(vertices, faces, max_remove_ratio=0.5)
        print(f"After final aggressive prune: {len(vertices)} vertices, {len(faces)} faces")
        
    # Carefully subdivide bad triangles with validation
    try:
        original_vertices, original_faces = vertices.copy(), faces.copy()
        vertices, faces = subdivide_bad_triangles(vertices, faces)
        
        # Validate the result
        if len(faces) > 0 and len(faces) >= len(original_faces) * 0.8:
            print(f"Subdivision successful: {len(vertices)} vertices, {len(faces)} faces")
        else:
            print("Subdivision produced too few faces, reverting to original")
            vertices, faces = original_vertices, original_faces
    except Exception as e:
        print(f"Subdivision failed: {e}, keeping original mesh")
        vertices, faces = original_vertices, original_faces
    
    print(f"After subdivision processing: {len(vertices)} vertices, {len(faces)} faces")
    
    # If a few non-manifold edges still exist, try splitting them by duplicating vertices
    edge_count = {}
    for face in faces:
        for i in range(3):
            e = tuple(sorted((int(face[i]), int(face[(i+1)%3]))))
            edge_count[e] = edge_count.get(e, 0) + 1
    remaining_non_manifold = [e for e, c in edge_count.items() if c > 2]
    if remaining_non_manifold:
        print(f"Attempting to split remaining {len(remaining_non_manifold)} non-manifold edges by vertex duplication")
        vertices, faces = split_non_manifold_edges_by_vertex_duplication(vertices, faces)
        print(f"After splitting non-manifold edges: {len(vertices)} vertices, {len(faces)} faces")

    # As a last resort, keep only the largest connected component to remove
    # small isolated pieces that contribute to non-manifold edges
    vertices, faces = keep_largest_component(vertices, faces)

    # Create new mesh
    faces_pv = np.column_stack((np.full(len(faces), 3), faces))
    repaired_mesh = pv.PolyData(vertices, faces_pv.flatten())
    
    # Export diagnostics before final cleanup
    export_mesh_diagnostics(repaired_mesh, "repair_before_final_cleanup")
    
    # Final cleanup pass
    repaired_mesh = repaired_mesh.clean(
        tolerance=1e-6,
        lines_to_points=False,
        polys_to_lines=False,
        strips_to_polys=True
    )
    
    print(f"Final mesh: {repaired_mesh.n_points} points, {repaired_mesh.n_cells} cells")
    
    # Export final state diagnostics
    export_mesh_diagnostics(repaired_mesh, "repair_final_state")
    
    return repaired_mesh

def validate_mesh_for_tetra(mesh):
    """Validate mesh before attempting tetrahedralization with comprehensive checks."""
    # Basic validation
    try:
        if mesh.n_points == 0 or mesh.n_cells == 0:
            raise RuntimeError("Empty mesh")
            
        points = mesh.points
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            raise RuntimeError("Mesh contains NaN or infinite values")
            
        # Get faces safely
        try:
            cells = mesh.faces.reshape(-1, 4)[:, 1:]
        except Exception as e:
            raise RuntimeError(f"Cannot get faces: {e}")
            
        if len(cells) == 0:
            raise RuntimeError("No cells in mesh")
    
    if mesh.number_of_points < 4:
        raise RuntimeError("Not enough points for tetrahedralization")
    
    if not mesh.is_all_triangles:
        raise RuntimeError("Mesh must be all triangles")
        
    # Check for basic mesh validity
    points = mesh.points
    cells = mesh.faces.reshape(-1, 4)[:, 1:]
    
    # Check for NaN/Inf values
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        raise RuntimeError("Mesh contains NaN or infinite values")
        
    # Check point indices are valid
    if np.any(cells >= len(points)) or np.any(cells < 0):
        raise RuntimeError("Invalid point indices in triangles")
        
    # Check for degenerate triangles
    min_area = 1e-10
    for tri in cells:
        v0, v1, v2 = points[tri]
        area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2
        if area < min_area:
            raise RuntimeError(f"Mesh contains degenerate triangle with area {area}")
            
    # Check for manifold edges
    edge_counts = {}
    for tri in cells:
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for v1, v2 in edges:
            edge = tuple(sorted([v1, v2]))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            
    non_manifold_edges = [e for e, count in edge_counts.items() if count > 2]
    if non_manifold_edges:
        raise RuntimeError(f"Mesh contains {len(non_manifold_edges)} non-manifold edges")
        
    # Check for self-intersections
    if check_self_intersections(mesh):
        raise RuntimeError("Mesh contains self-intersections")
        
    # Check mesh is closed
    boundary_edges = [e for e, count in edge_counts.items() if count != 2]
    if boundary_edges:
        raise RuntimeError(f"Mesh is not closed: contains {len(boundary_edges)} boundary edges")
    
    return True

def check_self_intersections(mesh):
    """Check if mesh has self-intersections using Open3D with validation."""
    try:
        if mesh.n_points == 0 or mesh.n_cells == 0:
            print("Cannot check intersections - empty mesh")
            return True
            
        points = mesh.points
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            print("Cannot check intersections - invalid points")
            return True
        
        # Try to get faces
        try:
            faces = mesh.faces.reshape(-1, 4)[:, 1:]
        except Exception as e:
            print(f"Cannot check intersections - face error: {e}")
            return True
            
        if len(faces) == 0:
            print("Cannot check intersections - no faces")
            return True
            
        # Create Open3D mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(points)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # Check intersection
        intersecting = o3d_mesh.is_self_intersecting()
        print(f"Self-intersections present: {intersecting}")
        return intersecting
        
    except Exception as e:
        print(f"Error checking self-intersections: {e}")
        return True

# Format faces array for PyVista (add count at the start of each face)
faces_pv = np.column_stack((np.full(len(tris), 3), tris)).flatten()

# Create PyVista grid for tetgen
input_mesh = pv.PolyData(verts, faces_pv)

# Apply enhanced mesh repair pipeline

# Aggressive vertex splitting fallback BEFORE main repair pipeline
print("Aggressive vertex splitting fallback before main repair pipeline...")
verts = np.asarray(input_mesh.points)
faces = input_mesh.faces.reshape(-1, 4)[:, 1:]
verts, faces = aggressive_split_non_manifold_edges(verts, faces, max_rounds=10)
faces_pv = np.column_stack((np.full(len(faces), 3), faces)).flatten()
input_mesh = pv.PolyData(verts, faces_pv)
input_mesh = input_mesh.clean(tolerance=1e-6)

print("Starting enhanced mesh repair process...")
input_mesh = repair_mesh(input_mesh)

# Additional cleanup with conservative tolerances

def safe_export_mesh_diagnostics(mesh, prefix):
    """Safely export mesh diagnostics with validation"""
    print(f"Attempting to export mesh diagnostics for {prefix}...")
    try:
        if mesh.n_points == 0 or mesh.n_cells == 0:
            print(f"Skipping diagnostics export for {prefix} - empty mesh")
            return
            
        if np.any(np.isnan(mesh.points)) or np.any(np.isinf(mesh.points)):
            print(f"Skipping diagnostics export for {prefix} - invalid points")
            return
            
        export_mesh_diagnostics(mesh, filename_prefix=prefix)
    except Exception as e:
        print(f"Failed to export diagnostics for {prefix}: {e}")

# Export diagnostics before final validation with safety checks
print("Exporting mesh diagnostics before final validation...")
safe_export_mesh_diagnostics(input_mesh, "pre_validation")

# Final validation with robust error handling
print("Step 9: Final aggressive cleanup with enhanced error handling...")

def safe_clean_mesh(mesh, min_cells_ratio=0.8):
    """Safely clean mesh with fallback options if cleaning fails"""
    original_cells = mesh.n_cells
    best_mesh = None
    best_cell_count = 0
    
    # Try different cleaning parameters
    for tolerance in [1e-6, 1e-5, 1e-4]:
        for merge_tol in [None, 1e-6, 1e-5]:
            try:
                temp_mesh = mesh.clean(
                    tolerance=tolerance,
                    merge_tolerance=merge_tol,
                    lines_to_points=False,
                    polys_to_lines=False,
                    strips_to_polys=False,
                    inplace=False
                )
                
                # Validate result
                if (temp_mesh.n_cells > 0 and
                    temp_mesh.n_cells >= best_cell_count and
                    temp_mesh.n_cells >= original_cells * min_cells_ratio):
                    
                    # Additional validation
                    pts = temp_mesh.points
                    if not (np.isnan(pts).any() or np.isinf(pts).any()):
                        best_mesh = temp_mesh
                        best_cell_count = temp_mesh.n_cells
                        print(f"Found valid cleaned mesh with {best_cell_count} cells")
                        
            except Exception as e:
                print(f"Clean attempt failed: {e}")
                continue
                
    return best_mesh if best_mesh is not None else mesh

# Try progressive cleanup with validation
def conservative_clean_mesh(mesh):
    """Clean mesh with progressive tolerances and strict validation"""
    print("Starting conservative mesh cleaning...")
    if mesh.n_cells == 0:
        print("Input mesh has no cells, cannot clean")
        return mesh
        
    best_mesh = None
    best_cells = 0
    original_cells = mesh.n_cells
    tolerances = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    
    for tolerance in tolerances:
        try:
            print(f"Attempting clean with tolerance {tolerance}")
            temp = mesh.clean(
                tolerance=tolerance,
                lines_to_points=False,
                polys_to_lines=False,
                inplace=False
            )
            if temp.n_cells == 0:
                print(f"Clean with tolerance {tolerance} produced empty mesh")
                continue
                
            # Validate cells and points
            if temp.n_cells >= original_cells * 0.8 and temp.n_points > 0:
                points = temp.points
                if not (np.isnan(points).any() or np.isinf(points).any()):
                    if temp.n_cells > best_cells:
                        best_mesh = temp
                        best_cells = temp.n_cells
                        print(f"Found better mesh with {best_cells} cells at tolerance {tolerance}")
                        
                        if best_cells >= original_cells * 0.95:
                            print("Found excellent quality mesh, stopping early")
                            break
            else:
                print(f"Clean resulted in too few cells: {temp.n_cells}")
                
        except Exception as e:
            print(f"Clean failed with tolerance {tolerance}: {e}")
            continue
            
    if best_mesh is not None:
        # Remove invalid points from best mesh
        points = best_mesh.points
        nan_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
        try:
            cleaned, _ = best_mesh.remove_points(~nan_mask)
            if cleaned.n_cells > 0:
                best_mesh = cleaned
                print("Successfully removed invalid points")
        except Exception as e:
            print(f"Point removal failed: {e}")
        
        return best_mesh
    else:
        print("No valid cleaned mesh found")
        return mesh

# Try the new conservative cleaning approach
cleaned_mesh = conservative_clean_mesh(input_mesh)
if cleaned_mesh is not input_mesh:
    print(f"Cleanup produced mesh with {cleaned_mesh.n_cells} cells")
    input_mesh = cleaned_mesh.triangulate()
else:
    print("Using original mesh")

# Export diagnostics after cleaning
print("Exporting mesh diagnostics after cleaning...")
export_mesh_diagnostics(input_mesh, filename_prefix="post_cleaning")


# Final best-effort: iterative self-intersection removal
print("Final best-effort: iterative self-intersection removal...")
input_mesh = iterative_self_intersection_removal(input_mesh, max_rounds=5, min_cells=50)

# Validate mesh before attempting tetrahedralization
print("Validating mesh for tetrahedralization...")
try:
    validate_mesh_for_tetra(input_mesh)
    print("Mesh validation successful")
except Exception as e:
    print(f"Mesh validation failed: {e}")
    export_mesh_diagnostics(input_mesh, filename_prefix="failed_validation")

# Convert to Open3D for advanced repair
print("Step 3: Converting to Open3D for repair...")
o3d_mesh = o3d.geometry.TriangleMesh()
o3d_mesh.vertices = o3d.utility.Vector3dVector(input_mesh.points)
o3d_mesh.triangles = o3d.utility.Vector3iVector(input_mesh.faces.reshape(-1, 4)[:, 1:])

# Pre-clean the mesh before detailed repair
print("Step 4: Applying Open3D mesh cleanup...")
o3d_mesh.compute_vertex_normals()
o3d_mesh.compute_triangle_normals()
o3d_mesh.remove_degenerate_triangles()
o3d_mesh.remove_duplicated_triangles()
o3d_mesh.remove_duplicated_vertices()
o3d_mesh.remove_unreferenced_vertices()

# Additional manifold edge checks
print("Step 5: Checking mesh manifolds...")
edges = o3d_mesh.get_non_manifold_edges()
if len(edges) > 0:
    print(f"Found {len(edges)} non-manifold edges")
    o3d_mesh = o3d_mesh.remove_vertices_by_mask(edges)
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_unreferenced_vertices()

# Verify vertex normals are consistent
o3d_mesh.compute_vertex_normals()
o3d_mesh.normalize_normals()

print("Step 4: Aggressive mesh repair using Open3D...")
o3d_mesh.remove_duplicated_vertices()
o3d_mesh.remove_duplicated_triangles()
o3d_mesh.remove_degenerate_triangles()
o3d_mesh.compute_vertex_normals()
o3d_mesh.compute_triangle_normals()

# Additional repair steps
print("Step 5: Advanced repair steps...")

def fix_self_intersections(mesh, max_iters=10):
    """
    Fix self-intersections by detecting intersecting triangle pairs and 
    subdividing them into smaller triangles that don't intersect.
    """
    print("Starting intersection removal process...")
    
    # Initial validation
    if mesh.n_points == 0 or mesh.n_cells == 0:
        print("Empty input mesh")
        return mesh
        
    # Save original mesh as fallback
    original_mesh = mesh.copy(deep=True)
    
    # Convert to numpy for efficient operations
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    if len(vertices) == 0 or len(triangles) == 0:
        print("Empty mesh detected")
        return mesh
        
    # Calculate bounding box and scale for tolerances
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    mesh_scale = np.linalg.norm(bbox_max - bbox_min)
    eps = mesh_scale * 1e-5
    
    def triangle_aabb(tri_verts):
        """Calculate axis-aligned bounding box for a triangle"""
        return np.min(tri_verts, axis=0), np.max(tri_verts, axis=0)
    
    def aabb_intersect(box1_min, box1_max, box2_min, box2_max):
        """Check if two AABBs intersect"""
        return np.all(box1_min <= box2_max) and np.all(box2_min <= box1_max)
    
    def tri_area(v0, v1, v2):
        """Calculate triangle area"""
        return np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2
    
    def split_triangle(v0, v1, v2):
        """Split a triangle into two by bisecting longest edge"""
        e1 = np.linalg.norm(v1 - v0)
        e2 = np.linalg.norm(v2 - v1)
        e3 = np.linalg.norm(v0 - v2)
        
        if e1 >= e2 and e1 >= e3:
            # Split edge between v0 and v1
            vm = (v0 + v1) / 2
            return [(vm, v1, v2), (v0, vm, v2)]
        elif e2 >= e1 and e2 >= e3:
            # Split edge between v1 and v2
            vm = (v1 + v2) / 2
            return [(v0, vm, v2), (v0, v1, vm)]
        else:
            # Split edge between v2 and v0
            vm = (v2 + v0) / 2
            return [(v0, v1, vm), (vm, v1, v2)]
    
    print("Phase 1: Initial mesh cleanup...")
    # Remove clearly degenerate triangles
    good_triangles = []
    for tri in triangles:
        try:
            v0, v1, v2 = vertices[tri]
            area = tri_area(v0, v1, v2)
            if area > eps * eps:
                good_triangles.append(tri)
        except IndexError:
            print(f"Warning: Invalid triangle indices {tri}, skipping")
            continue
    
    if not good_triangles:
        print("No valid triangles found")
        return mesh
        
    triangles = np.array(good_triangles)
    print(f"Found {len(triangles)} valid triangles")
    
    print("Phase 2: Processing intersecting triangle pairs...")
    # Process potentially intersecting triangles
    processed_vertices = vertices.tolist()
    final_triangles = []
    
    for i in range(len(triangles)):
        tri1 = triangles[i]
        tri1_verts = vertices[tri1]
        box1_min, box1_max = triangle_aabb(tri1_verts)
        
        needs_split = False
        
        # Check against all other triangles
        for j in range(i + 1, len(triangles)):
            tri2 = triangles[j]
            
            # Skip if triangles share vertices
            if len(set(tri1) & set(tri2)) > 0:
                continue
                
            tri2_verts = vertices[tri2]
            box2_min, box2_max = triangle_aabb(tri2_verts)
            
            # Quick AABB test
            if aabb_intersect(box1_min, box1_max, box2_min, box2_max):
                # Triangles might intersect - mark for splitting
                needs_split = True
                break
        
        if needs_split:
            # Split triangle into smaller triangles
            v0, v1, v2 = tri1_verts
            split_tris = split_triangle(v0, v1, v2)
            
            # Add new vertices and triangles
            for new_tri in split_tris:
                new_verts = []
                new_tri_indices = []
                
                for v in new_tri:
                    # Add vertex if not too close to existing ones
                    too_close = False
                    for idx, existing_v in enumerate(processed_vertices):
                        if np.linalg.norm(v - np.array(existing_v)) < eps:
                            new_tri_indices.append(idx)
                            too_close = True
                            break
                    
                    if not too_close:
                        processed_vertices.append(v)
                        new_tri_indices.append(len(processed_vertices) - 1)
                
                if len(set(new_tri_indices)) == 3:
                    final_triangles.append(new_tri_indices)
        else:
            final_triangles.append(tri1.tolist())
    
    print(f"Created {len(processed_vertices)} vertices and {len(final_triangles)} triangles")
    
    # Create new mesh
    clean_mesh = o3d.geometry.TriangleMesh()
    clean_mesh.vertices = o3d.utility.Vector3dVector(np.array(processed_vertices))
    clean_mesh.triangles = o3d.utility.Vector3iVector(np.array(final_triangles))
    
    print("Phase 3: Final cleanup...")
    # Remove any remaining issues
    clean_mesh.remove_degenerate_triangles()
    clean_mesh.remove_duplicated_triangles()
    clean_mesh.remove_duplicated_vertices()
    clean_mesh.remove_unreferenced_vertices()
    clean_mesh.compute_vertex_normals()
    
    # Ensure proper orientation
    clean_mesh.compute_vertex_normals()
    clean_mesh.orient_triangles()
    
    # Verify final mesh
    result_verts = np.asarray(clean_mesh.vertices)
    result_tris = np.asarray(clean_mesh.triangles)
    
    if len(result_verts) < 3 or len(result_tris) < 1:
        print("Warning: Repair resulted in degenerate mesh, returning original")
        return mesh
        
    print(f"Final mesh: {len(result_verts)} vertices, {len(result_tris)} triangles")
    
    return clean_mesh

# Convert to point cloud for outlier removal
print("Converting to point cloud for outlier removal...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d_mesh.vertices

# Remove outliers from point cloud
radius = float(np.mean(o3d_mesh.get_axis_aligned_bounding_box().get_extent()) * 0.005)  # Adaptive radius
print(f"Using radius {radius} for outlier removal...")
pcd, _ = pcd.remove_radius_outlier(nb_points=30, radius=radius)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

# Update mesh vertices
o3d_mesh.vertices = pcd.points

# Fix self-intersections
o3d_mesh = fix_self_intersections(o3d_mesh)

# Clean up manifold edges and boundaries
o3d_mesh.compute_vertex_normals()
o3d_mesh.compute_triangle_normals()

# Convert back to PyVista for additional repairs
cleaned_verts = np.asarray(o3d_mesh.vertices)
cleaned_tris = np.asarray(o3d_mesh.triangles)
faces_pv = np.column_stack((np.full(len(cleaned_tris), 3), cleaned_tris)).flatten()
input_mesh = pv.PolyData(cleaned_verts, faces_pv)

print("Step 6: Aggressive vertex splitting and cleanup...")
vertices = np.asarray(input_mesh.points)
faces = input_mesh.faces.reshape(-1, 4)[:, 1:]
vertices, faces = aggressive_split_non_manifold_edges(vertices, faces, max_rounds=10)
faces_pv = np.column_stack((np.full(len(faces), 3), faces)).flatten()
input_mesh = pv.PolyData(vertices, faces_pv)

# Progressive cleaning with decreasing tolerance - more conservative approach
print("Progressive cleaning with adaptive tolerance - conservative approach...")
best_mesh = None
best_cell_count = 0
tolerances = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]

for tolerance in tolerances:
    try:
        # Create temporary mesh for this iteration
        temp_mesh = input_mesh.clean(
            tolerance=tolerance,
            lines_to_points=False,
            polys_to_lines=False,
            strips_to_polys=False,
            inplace=False
        )
        
        # Validate the cleaned mesh
        if (temp_mesh.n_cells > 0 and 
            temp_mesh.n_points > 0 and 
            temp_mesh.n_cells > best_cell_count * 0.8):  # Allow some cell loss but not too much
            
            # Additional validation
            points = np.asarray(temp_mesh.points)
            if not (np.isnan(points).any() or np.isinf(points).any()):
                best_mesh = temp_mesh
                best_cell_count = temp_mesh.n_cells
                print(f"Found valid mesh at tolerance {tolerance} with {best_cell_count} cells")
                
                # Early exit if we have a good result
                if best_cell_count > input_mesh.n_cells * 0.9:
                    print("Found good quality mesh, stopping early")
                    break
        else:
            print(f"Skipping tolerance {tolerance} - would result in too few cells")
            
    except Exception as e:
        print(f"Clean failed with tolerance {tolerance}: {e}")
        continue

if best_mesh is not None:
    print(f"Using best mesh found with {best_mesh.n_cells} cells")
    input_mesh = best_mesh.triangulate()
else:
    print("All cleanup attempts failed, keeping original mesh")

# Export diagnostics after vertex splitting
print("Exporting diagnostics after aggressive vertex splitting...")
export_mesh_diagnostics(input_mesh, filename_prefix="post_vertex_splitting")

# Fill holes with adaptive parameters
print("Step 7: Advanced hole filling...")
for hole_size in [2000, 1000, 500, 250]:
    try:
        input_mesh.fill_holes(hole_size=hole_size, inplace=True)
        print(f"Hole filling succeeded with size {hole_size}")
        break
    except Exception as e:
        print(f"Hole filling failed with size {hole_size}: {e}")
        continue

# Additional smoothing and remeshing
print("Step 8: Final smoothing and cleanup...")
input_mesh = input_mesh.clean(tolerance=1e-4)
input_mesh = input_mesh.triangulate()

# Final validation
print("Step 9: Final aggressive cleanup...")

# Apply progressive geometric cleanup with strict validation
print("Applying final progressive cleanup with enhanced validation...")
original_mesh = input_mesh.copy(deep=True)
best_mesh = None
best_cell_count = 0

def validate_cleaned_mesh(mesh, original_mesh):
    """Validate a cleaned mesh against original"""
    if mesh.n_points == 0 or mesh.n_cells == 0:
        return False, "Empty mesh"
    
    points = mesh.points
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return False, "Invalid points"
    
    if mesh.n_cells < original_mesh.n_cells * 0.8:
        return False, f"Too few cells ({mesh.n_cells} vs {original_mesh.n_cells})"
        
    try:
        mesh.faces.reshape(-1, 4)[:, 1:]
    except Exception as e:
        return False, f"Invalid faces: {e}"
        
    return True, "Valid mesh"

# Try progressive cleanup with strict validation
tolerances = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
for tolerance in tolerances:
    try:
        temp_mesh = input_mesh.clean(
            tolerance=tolerance,
            lines_to_points=False,
            polys_to_lines=False,
            inplace=False
        )
        valid, reason = validate_cleaned_mesh(temp_mesh, original_mesh)
        if valid:
            if temp_mesh.n_cells > best_cell_count:
                best_mesh = temp_mesh.copy(deep=True)
                best_cell_count = temp_mesh.n_cells
                print(f"Found better mesh at tolerance {tolerance} with {best_cell_count} cells")
                
                if best_cell_count >= original_mesh.n_cells * 0.95:
                    print("Found excellent quality mesh, stopping early")
                    break
        else:
            print(f"Clean at tolerance {tolerance} produced invalid mesh: {reason}")
            
    except Exception as e:
        print(f"Cleanup failed with tolerance {tolerance}: {e}")
        continue

if best_mesh is not None:
    print(f"Using best cleaned mesh with {best_mesh.n_cells} cells")
    input_mesh = best_mesh
else:
    print("All cleanup attempts failed or produced invalid meshes, keeping original")
    input_mesh = original_mesh

# Ensure we're working with triangles
input_mesh = input_mesh.triangulate()

# Remove NaN and infinite points
points = input_mesh.points
nan_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
try:
    input_mesh, _ = input_mesh.remove_points(~nan_mask)  # Unpack the mesh from the tuple
except Exception as e:
    print(f"Point removal failed: {e}, keeping original points")
    # If point removal fails, keep the original points but warn the user
    print("Warning: Mesh may contain invalid points")

# Extract triangles and points
triangles = input_mesh.faces.reshape(-1, 4)[:, 1:]  # Skip first index which is count
points = input_mesh.points

# Build triangle centers for KD-tree
centers = np.mean(points[triangles], axis=1)
tree = KDTree(centers)

# Find triangle intersections within certain radius and count them
radius = np.max(points.max(axis=0) - points.min(axis=0)) * 0.005  # 0.5% of bounding box
pairs = list(tree.query_pairs(radius))

# Count intersections per triangle
intersection_counts = np.zeros(len(triangles), dtype=int)
for i, j in pairs:
    intersection_counts[i] += 1
    intersection_counts[j] += 1

# Sort triangles by intersection count (most intersections first)
problem_triangles = np.argsort(intersection_counts)[::-1]
intersection_threshold = np.mean(intersection_counts) + np.std(intersection_counts)

# Remove triangles with more than threshold intersections
triangles_to_remove = set(i for i in problem_triangles if intersection_counts[i] > intersection_threshold)

# Create mask of triangles to keep
keep_triangles = np.ones(len(triangles), dtype=bool)
keep_triangles[list(triangles_to_remove)] = False

# Create new mesh without intersecting triangles
cells = np.insert(triangles[keep_triangles], 0, 3, axis=1)  # Add count back
input_mesh = pv.PolyData(points, cells)

# Ensure mesh is clean
input_mesh = input_mesh.triangulate()
input_mesh = input_mesh.clean(tolerance=1e-7)

# Try MeshFix / pymeshfix as a robust final repair step to get a closed manifold
try:
    print("Attempting final repair with pymeshfix.MeshFix...")
    import pymeshfix
    pts = np.asarray(input_mesh.points)
    fcs = input_mesh.faces.reshape(-1, 4)[:, 1:]
    mf = pymeshfix.MeshFix(pts, fcs)
    mf.repair()
    fixed_pts, fixed_faces = mf.v, mf.f
    if fixed_faces is not None and len(fixed_faces) > 0:
        cells = np.column_stack((np.full(len(fixed_faces), 3), fixed_faces)).flatten()
        input_mesh = pv.PolyData(fixed_pts, cells)
        input_mesh = input_mesh.clean(tolerance=1e-7)
        input_mesh = input_mesh.triangulate()
        print("pymeshfix repair applied")
except Exception as e:
    print(f"pymeshfix repair failed or not available: {e}")

print("Step 10: Validating final mesh...")
# Check for self-intersections
n_intersections = len(tree.query_pairs(radius))
print(f"Self-intersections present: {n_intersections > 0}")
if n_intersections > 0:
    print("Warning: Self-intersections still present after repair")

print("Final mesh quality:")
print(f"Points: {input_mesh.n_points}")
print(f"Cells: {input_mesh.n_cells}")
print(f"Volume: {input_mesh.volume}")

print("Validating mesh for tetrahedralization...")

# Apply larger jitter for numerical stability
print("Step 11: Applying coordinate jitter for numerical stability...")
points = input_mesh.points
bbox_diag = np.sqrt(np.sum((points.max(axis=0) - points.min(axis=0)) ** 2))
jitter_scale = bbox_diag * 1e-3  # 0.1% of bounding box diagonal
jitter = np.random.normal(0, jitter_scale, points.shape)
points += jitter
input_mesh.points = points
print(f"Added random jitter with scale {jitter_scale:.2e}")

# Fix zero-volume elements
bbox = input_mesh.bounds
diag = np.linalg.norm([bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4]])
if diag == 0:
    diag = 1.0
cell_centers = input_mesh.cell_centers().points
for i in range(input_mesh.n_cells):
    tri = input_mesh.faces.reshape(-1, 4)[i, 1:]
    v0, v1, v2 = input_mesh.points[tri]
    vol = np.abs(np.cross(v1-v0, v2-v0)).sum()
    if vol < diag * 1e-12:  # Zero-volume element
        mid = cell_centers[i]
        v = np.random.normal(scale=diag*1e-6, size=3)
        v = v - np.dot(v, np.cross(v1-v0, v2-v0)) * np.cross(v1-v0, v2-v0)
        v = v / np.linalg.norm(v) * diag * 1e-6
        input_mesh.points[tri[1]] = mid + v

# More cleanup after volume fix
input_mesh = input_mesh.clean(tolerance=1e-6)
input_mesh = input_mesh.triangulate()

print("Step 10: Validating final mesh...")
intersecting = check_self_intersections(input_mesh)
if intersecting:
    print("Warning: Self-intersections still present after repair")
else:
    print("Mesh validation successful - no self-intersections detected")

print(f"Final mesh quality:")
print(f"Points: {input_mesh.n_points}")
print(f"Cells: {input_mesh.n_cells}")
print(f"Volume: {input_mesh.volume}")

# Validate mesh before attempting tetrahedralization
print("Validating mesh for tetrahedralization...")
validate_mesh_for_tetra(input_mesh)

print("Step 11: Applying coordinate jitter for numerical stability...")
print("Step 10: Applying small coordinate jitter for numerical stability...")
points = np.asarray(input_mesh.points)
faces = input_mesh.faces.reshape(-1, 4)[:, 1:]
bbox = input_mesh.bounds
diag = np.linalg.norm([bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4]])
if diag == 0:
    diag = 1.0
sigma = diag * 1e-3  # maximum jitter while preserving geometric features
np.random.seed(42)  # deterministic jitter
points += np.random.normal(scale=sigma, size=points.shape)
print(f"Added random jitter with scale {sigma:.2e}")

# Update surface mesh and TetGen input
input_mesh = pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)).flatten())
input_tet = tetgen.TetGen(points, faces)

# Try tetrahedralization with parameters
try:
    input_tet.tetrahedralize(**tetra_args)
except RuntimeError as e:
    print("First tetrahedralization attempt failed, trying repair...\n", e)
    try:
        print("Attempting advanced repair process...")
        # Extract current points and faces if available, otherwise fall back to the
        # PyVista input_mesh we built earlier. TetGen may not have produced node/face
        # attributes when tetrahedralize() failed with an exception.
        if hasattr(input_tet, 'node') and hasattr(input_tet, 'face'):
            pts = input_tet.node
            faces = input_tet.face
        else:
            print("TetGen did not produce node/face data; falling back to current PyVista surface for repair")
            pts = input_mesh.points
            faces = input_mesh.faces.reshape(-1, 4)[:, 1:]

        # Use PyVista's robust cleaning pipeline to repair the surface before retrying.
        repair_pv = pv.PolyData(pts, np.column_stack((np.full(len(faces), 3), faces)).flatten())
        repair_pv = repair_pv.clean(tolerance=1e-6)
        repair_pv = repair_pv.extract_surface().triangulate()
        repair_pv = repair_pv.clean(tolerance=1e-6)

        clean_verts = np.asarray(repair_pv.points)
        clean_faces = repair_pv.faces.reshape(-1, 4)[:, 1:]

        # Create new TetGen instance with cleaned mesh and retry with more relaxed params
        print("Creating new TetGen instance with cleaned mesh...")
        input_tet = tetgen.TetGen(clean_verts, clean_faces)
        print("Retrying tetrahedralization with repaired mesh...")
        tetra_args['switches'] = 'pq1.8a0.3'  # Even more relaxed quality bounds
        tetra_args['minratio'] = 3.0          # Further relaxed
        tetra_args['mindihedral'] = 3.0       # More forgiving
        input_tet.tetrahedralize(**tetra_args)
    except Exception as e2:
        # re-raise with more context
        raise RuntimeError("Failed to tetrahedralize even after attempting advanced repair") from e2# Get the grid and scale back to original size
input_tet = input_tet.grid
input_tet.points = input_tet.points / scale_factor  # Scale back to original size

# rotate
# input_tet = input_tet.rotate_x(-90) # b axis mount

# scale
# input_tet = input_tet.scale(1.5)

# make origin center bottom of bounding box
# PART_OFFSET = np.array([0., 10., 0.]) # z mount
# PART_OFFSET = np.array([-13., -10., 0.]) # bunny
# PART_OFFSET = np.array([60., 0., 0.]) # benchy
# PART_OFFSET = np.array([0., 10., 0.]) # benchy upsidedown tilted
# PART_OFFSET = np.array([0., 10., 0.]) # squirtle
# PART_OFFSET = np.array([-44., 0., 0.]) # b axis mount
# PART_OFFSET = np.array([50., 20., 0.]) # mew
PART_OFFSET = np.array([0., 0., 0.])
x_min, x_max, y_min, y_max, z_min, z_max = input_tet.bounds
input_tet.points -= np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]) + PART_OFFSET


# find neighbours
cell_neighbour_dict = {neighbour_type: {face: [] for face in range(input_tet.number_of_cells)} for neighbour_type in ["point", "edge", "face"]}
for neighbour_type in ["point", "edge", "face"]:
    cell_neighbours = []
    for cell_index in range(input_tet.number_of_cells):
        neighbours = input_tet.cell_neighbors(cell_index, f"{neighbour_type}s")
        for neighbour in neighbours:
            if neighbour > cell_index:
                cell_neighbours.append((cell_index, neighbour))
    for face_1, face_2 in np.array(cell_neighbours):
        cell_neighbour_dict[neighbour_type][face_1].append(face_2)
        cell_neighbour_dict[neighbour_type][face_2].append(face_1)

    input_tet.field_data[f"cell_{neighbour_type}_neighbours"] = np.array(cell_neighbours)

cell_neighbour_graph = nx.Graph()
cell_centers = input_tet.cell_centers().points
for edge in input_tet.field_data["cell_point_neighbours"]: # use point neighbours for best accuracy
    distance = np.linalg.norm(cell_centers[edge[0]] - cell_centers[edge[1]])
    cell_neighbour_graph.add_weighted_edges_from([(edge[0], edge[1], distance)])

def update_tet_attributes(tet):
    '''
    Calculate face normals, face centers, cell centers, and overhang angles for each cell in the tetrahedral mesh.
    '''

    surface_mesh = tet.extract_surface()
    cell_to_face = decode_object(tet.field_data["cell_to_face"])

    # put general data in field_data for easy access
    cells = tet.cells.reshape(-1, 5)[:, 1:] # assume all cells have 4 vertices
    tet.add_field_data(cells, "cells")
    cell_vertices = tet.points
    tet.add_field_data(cell_vertices, "cell_vertices")
    faces = surface_mesh.faces.reshape(-1, 4)[:, 1:] # assume all faces have 3 vertices
    tet.add_field_data(faces, "faces")
    face_vertices = surface_mesh.points
    tet.add_field_data(face_vertices, "face_vertices")

    tet.cell_data['face_normal'] = np.full((tet.number_of_cells, 3), np.nan)
    surface_mesh_face_normals = surface_mesh.face_normals
    for cell_index, face_indices in cell_to_face.items():
        face_normals = surface_mesh_face_normals[face_indices]
        # get the normal facing the most down
        most_down_normal_index = np.argmin(face_normals[:, 2])
        tet.cell_data['face_normal'][cell_index] = face_normals[most_down_normal_index]
    tet.cell_data['face_normal'] =  tet.cell_data['face_normal'] / np.linalg.norm(tet.cell_data['face_normal'], axis=1)[:, None]

    tet.cell_data['face_center'] = np.empty((tet.number_of_cells, 3))
    tet.cell_data['face_center'][:,:] = np.nan
    surface_mesh_cell_centers = surface_mesh.cell_centers().points
    for cell_index, face_indices in cell_to_face.items():
        face_centers = surface_mesh_cell_centers[face_indices]
        # get the normal facing the most down
        most_down_center_index = np.argmin(face_centers[:, 2])
        tet.cell_data['face_center'][cell_index] = face_centers[most_down_center_index]

    tet.cell_data["cell_center"] = tet.cell_centers().points

    # calculate bottom cells
    bottom_cell_threshold = np.nanmin(tet.cell_data['face_center'][:, 2])+0.3
    bottom_cells_mask = tet.cell_data['face_center'][:, 2] < bottom_cell_threshold
    tet.cell_data['is_bottom'] = bottom_cells_mask
    bottom_cells = np.where(bottom_cells_mask)[0]

    face_normals = tet.cell_data['face_normal'].copy()
    face_normals[bottom_cells_mask] = np.nan # make bottom faces not angled
    overhang_angle = np.arccos(np.dot(face_normals, up_vector))
    tet.cell_data['overhang_angle'] = overhang_angle

    overhang_direction = face_normals[:, :2].copy()
    overhang_direction /= np.linalg.norm(overhang_direction, axis=1)[:, None]
    tet.cell_data['overhang_direction'] = overhang_direction

    # calculate if cell will print in air by seeing if any cell centers along path to base are higher
    IN_AIR_THRESHOLD = 1
    tet.cell_data['in_air'] = np.full(tet.number_of_cells, False)

    _, paths_to_bottom = nx.multi_source_dijkstra(cell_neighbour_graph, set(bottom_cells))

    # put it in cell data
    tet.cell_data['path_to_bottom'] = np.full((tet.number_of_cells, np.max([len(x) for x in paths_to_bottom.values()])), -1)
    for cell_index, path_to_bottom in paths_to_bottom.items():
        tet.cell_data['path_to_bottom'][cell_index, :len(path_to_bottom)] = path_to_bottom

    # calculate if cell is in air
    for cell_index in range(tet.number_of_cells):
        path_to_bottom = paths_to_bottom[cell_index]
        if len(path_to_bottom) > 1:
            cell_heights = tet.cell_data['cell_center'][path_to_bottom, 2]
            if np.any(cell_heights > tet.cell_data['cell_center'][cell_index, 2] + IN_AIR_THRESHOLD):
                tet.cell_data['in_air'][cell_index] = True

    return tet

def calculate_tet_attributes(tet):
    '''
    Calculate shared vertices between cells, cell to face & face to cell relations, and bottom cells of the tetrahedral mesh.
    '''

    surface_mesh = tet.extract_surface()

    # put general data in field_data for easy access
    cells = tet.cells.reshape(-1, 5)[:, 1:] # assume all cells have 4 vertices
    tet.add_field_data(cells, "cells")
    cell_vertices = tet.points
    tet.add_field_data(cell_vertices, "cell_vertices")
    faces = surface_mesh.faces.reshape(-1, 4)[:, 1:] # assume all faces have 3 vertices
    tet.add_field_data(faces, "faces")
    face_vertices = surface_mesh.points
    tet.add_field_data(face_vertices, "face_vertices")

    # calculate shared vertices
    shared_vertices = []
    for cell_1, cell_2 in tet.field_data["cell_point_neighbours"]:
        shared_vertices_these_faces = np.intersect1d(cells[cell_1], cells[cell_2])
        for vertex in shared_vertices_these_faces:
            shared_vertices.append({
                    "cell_1_index": cell_1,
                    "cell_2_index": cell_2,
                    "cell_1_vertex_index": np.where(cells[cell_1] == vertex)[0][0],
                    "cell_2_vertex_index": np.where(cells[cell_2] == vertex)[0][0],
                })

    # calculate cell to face & face to cell relations
    cell_to_face = {}
    face_to_cell = {face_index: [] for face_index in range(len(faces))}
    cell_to_face_vertices = {}
    face_to_cell_vertices = {}
    for cell_vertex_index, cell_vertex in enumerate(tet.field_data["cell_vertices"].reshape(-1, 3)):
        face_vertex_index = np.where((face_vertices == cell_vertex).all(axis=1))[0]
        if len(face_vertex_index) == 1:
            cell_to_face_vertices[cell_vertex_index] = face_vertex_index[0]
            face_to_cell_vertices[face_vertex_index[0]] = cell_vertex_index

    for cell_index, cell in enumerate(tet.field_data["cells"]):
        face_vertex_indices = [cell_to_face_vertices[cell_vertex_index] for cell_vertex_index in cell if cell_vertex_index in cell_to_face_vertices]
        if len(face_vertex_indices) >= 3:
            extracted = surface_mesh.extract_points(face_vertex_indices, adjacent_cells=False)
            if extracted.number_of_cells >= 1:
                cell_to_face[cell_index] = list(extracted.cell_data['vtkOriginalCellIds'])
                for face_index in extracted.cell_data['vtkOriginalCellIds']:
                    face_to_cell[face_index].append(cell_index)

    tet.add_field_data(encode_object(cell_to_face), "cell_to_face")
    tet.add_field_data(encode_object(face_to_cell), "face_to_cell")

    # calculate has_face attribute
    tet.cell_data['has_face'] = np.zeros(tet.number_of_cells)
    for cell_index, face_indices in cell_to_face.items():
        tet.cell_data['has_face'][cell_index] = 1

    tet = update_tet_attributes(tet)

    # calculate bottom cells
    bottom_cells_mask = tet.cell_data['is_bottom']
    bottom_cells = np.where(bottom_cells_mask)[0]

    tet.cell_data['overhang_angle'][bottom_cells] = np.nan

    return tet, bottom_cells_mask, bottom_cells


bottom_cells_mask = None
bottom_cells = None
input_tet, bottom_cells_mask, bottom_cells = calculate_tet_attributes(input_tet)

# find bottom cell groups that are connected
bottom_cell_graph = nx.Graph()
for cell_index in bottom_cells:
    bottom_cell_graph.add_node(cell_index)
cell_point_neighbour_dict = cell_neighbour_dict["point"]
for cell_index in bottom_cells:
    for neighbour in cell_point_neighbour_dict[cell_index]:
        if neighbour in bottom_cells:
            bottom_cell_graph.add_edge(cell_index, neighbour)

bottom_cell_groups = [list(x) for x in list(nx.connected_components(bottom_cell_graph))]

undeformed_tet = input_tet.copy()


# In[6]:


def planeFit(points):
    """
    p, n = planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    """
    import numpy as np
    from numpy.linalg import svd
    points = np.reshape(points, (np.shape(points)[0], -1))
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:,np.newaxis]
    M = np.dot(x, x.T)
    return ctr, svd(M)[0][:,-1]

def calculate_path_length_to_base_gradient(tet, MAX_OVERHANG, INITIAL_ROTATION_FIELD_SMOOTHING, SET_INITIAL_ROTATION_TO_ZERO):
    '''
    Calculate the path length to base gradient for each cell in the tetrahedral mesh with respect to the radial direction. This is used to determine the optimal rotation direction for each cell.

    returns: path_length_to_base_gradient. A scalar for each cell in the tetrahedral mesh. This is the gradient in the radial direction of the path length to the closest bottom cell.
    '''

    # calculate initial rotation direction for each face
    path_length_to_base_gradient = np.zeros((tet.number_of_cells)) # this is a scalar with respect to the radial direction. ie the vector pointing to the cell center

    # find the path length for every overhang cell to a bottom cell
    cell_distance_to_bottom = np.empty((tet.number_of_cells))
    cell_distance_to_bottom[:] = np.nan
    distances_to_bottom, paths_to_bottom = nx.multi_source_dijkstra(cell_neighbour_graph, set(bottom_cells))# set([x[0] for x in tet.field_data["bottom_cell_groups"]]))
    closest_bottom_cell_indices = np.zeros((tet.number_of_cells), dtype=int)
    for cell_index in range(tet.number_of_cells):
        face_normal = tet.cell_data["face_normal"][cell_index]

        cell_is_overhang = np.arccos(np.dot(face_normal, [0,0,1])) > np.deg2rad(90+MAX_OVERHANG)
        if cell_is_overhang and cell_index not in bottom_cells:
            closest_bottom_cell_indices[cell_index] = paths_to_bottom[cell_index][0]
            cell_distance_to_bottom[cell_index] = distances_to_bottom[cell_index]

    tet.cell_data["cell_distance_to_bottom"] = cell_distance_to_bottom

    # calculate the gradient of path length to base for each cell
    for cell_index in range(tet.number_of_cells):
        if not np.isnan(cell_distance_to_bottom[cell_index]):
            local_cells = cell_neighbour_dict["edge"][cell_index]
            local_cells = np.hstack((local_cells, cell_index))
            # add neighbours neighbours
            # local_cells = neighbours.copy()
            # for neighbour in neighbours:
            #     local_cells.extend(cell_neighbour_dict["point"][neighbour])
            # local_cells = np.array(list(set(local_cells)))

            local_cell_path_lengths = [cell_distance_to_bottom[local_cell] for local_cell in local_cells]
            local_cell_path_lengths = np.array(local_cell_path_lengths)

            # remove neighbours with path length of nan
            local_cells = np.array(local_cells)[~np.isnan(local_cell_path_lengths)]
            local_cell_path_lengths = local_cell_path_lengths[~np.isnan(local_cell_path_lengths)]

            # if there are less than 3 neighbours with path length, roll to the closest bottom cell
            if len(local_cell_path_lengths) < 3:
                location_to_roll_to = tet.cell_data["cell_center"][closest_bottom_cell_indices[cell_index], :2]

                direction_to_bottom = location_to_roll_to - tet.cell_data["cell_center"][cell_index, :2]
                direction_to_bottom /= np.linalg.norm(direction_to_bottom)

                cell_center = tet.cell_data["cell_center"][cell_index, :2].copy()
                cell_center /= np.linalg.norm(cell_center)

                optimal_rotation_direction = np.dot(cell_center, direction_to_bottom) / np.abs(np.dot(cell_center, direction_to_bottom))
                if np.isnan(optimal_rotation_direction):
                    optimal_rotation_direction = 0

                path_length_to_base_gradient[cell_index] = optimal_rotation_direction

            # if there are 3 or more neighbours with path length, calculate the gradient in the radial direction
            # and use that as the optimal rotation direction
            else:
                points = np.hstack((tet.cell_data["cell_center"][local_cells, :2], local_cell_path_lengths[:, None]))
                _, plane_normal = planeFit(points.T)

                cell_center_direction_normalized = tet.cell_data["cell_center"][cell_index, :2] / np.linalg.norm(tet.cell_data["cell_center"][cell_index, :2])
                gradient_in_radial_direction = np.dot(cell_center_direction_normalized, plane_normal[:2])

                # if the gradient is nan, use the average of the neighbours
                if np.isnan(gradient_in_radial_direction):
                    gradient_in_radial_direction = np.mean(path_length_to_base_gradient[local_cells][~np.isnan(path_length_to_base_gradient[local_cells])])
                    if np.isnan(gradient_in_radial_direction):
                        gradient_in_radial_direction = 0

                path_length_to_base_gradient[cell_index] = gradient_in_radial_direction

    # smooth path_length_to_base_gradient with neighbours
    # not needed because we do neighbour difference minimization in the optimization step?
    if INITIAL_ROTATION_FIELD_SMOOTHING != 0:
        for i in range(INITIAL_ROTATION_FIELD_SMOOTHING):
            smoothed_path_length_to_base_gradient = np.zeros((tet.number_of_cells))
            for cell_index in range(tet.number_of_cells):
                if path_length_to_base_gradient[cell_index] != 0:
                    neighbours = cell_neighbour_dict["point"][cell_index]
                    local_cells = neighbours.copy()
                    for neighbour in neighbours:
                        local_cells.extend(cell_neighbour_dict["point"][neighbour])
                    local_cells = np.array(list(set(local_cells)))
                    local_cells = local_cells[path_length_to_base_gradient[local_cells]!=0]
                    smoothed_path_length_to_base_gradient[cell_index] = np.mean(path_length_to_base_gradient[local_cells])

        path_length_to_base_gradient = smoothed_path_length_to_base_gradient

    # replace 0 with nan
    if not SET_INITIAL_ROTATION_TO_ZERO:
        path_length_to_base_gradient[path_length_to_base_gradient == 0] = np.nan
    tet.cell_data["path_length_to_base_gradient"] = path_length_to_base_gradient # very sexy

    return path_length_to_base_gradient


# In[ ]:


def calculate_initial_rotation_field(tet, MAX_OVERHANG, ROTATION_MULTIPLIER, STEEP_OVERHANG_COMPENSATION, INITIAL_ROTATION_FIELD_SMOOTHING, SET_INITIAL_ROTATION_TO_ZERO, MAX_POS_ROTATION, MAX_NEG_ROTATION):
    '''
    Calculate the initial rotation field for each cell in the tetrahedral mesh to make overhangs less than MAX_OVERHANG.
    The direction of rotation ensures the part is printable.
    '''

    # create initial rotation field rotating faces to be in safe printing angle
    initial_rotation_field = np.full((tet.number_of_cells), np.nan)
    initial_rotation_field = np.abs(np.deg2rad(90+MAX_OVERHANG) - tet.cell_data['overhang_angle'])

    path_length_to_base_gradient = calculate_path_length_to_base_gradient(tet, MAX_OVERHANG, INITIAL_ROTATION_FIELD_SMOOTHING, SET_INITIAL_ROTATION_TO_ZERO)

    # if path_length_to_base_gradient is different to the cell's overhang direction, it needs to be rotated an additional amount (its overhang angle) to make it go the right way
    # Put behind a flag because it is normally not needed, and buggy/finnicky
    # Can try enable it for models with very steep overhangs (>90 degrees) (not common)
    if STEEP_OVERHANG_COMPENSATION:
        initial_rotation_field[tet.cell_data["in_air"]] += 2 * (np.deg2rad(180) - tet.cell_data['overhang_angle'][tet.cell_data["in_air"]])

    # # Apply the path_length_to_base_gradient (optimal overhang direction) to the initial rotation field
    initial_rotation_field *= path_length_to_base_gradient

    # apply rotation multiplier
    initial_rotation_field = np.clip(initial_rotation_field*ROTATION_MULTIPLIER, -np.deg2rad(360), np.deg2rad(360))

    # clip to max rotation
    initial_rotation_field = np.clip(initial_rotation_field, MAX_NEG_ROTATION, MAX_POS_ROTATION)

    tet.cell_data["initial_rotation_field"] = initial_rotation_field

    return initial_rotation_field

from scipy.sparse import lil_matrix

def calculate_rotation_matrices(tet, rotation_field):
    '''
    Calculate the rotation matrices for each cell in the tetrahedral mesh given the scalar
    rotation field that gives a rotation for each cell. Cells are rotated around the axis
    perpendicular to the radial direction and the z-axis.
    '''

    # create rotation matrix from theta around axis
    tangential_vectors = np.cross( np.array([0, 0, 1]), tet.cell_data["cell_center"][:, :2])
    # normalize
    tangential_vectors /= np.linalg.norm(tangential_vectors, axis=1)[:, None]
    # replace nan with [1,0,0]
    tangential_vectors[np.isnan(tangential_vectors).any(axis=1)] = [1, 0, 0]

    rotation_matrices = R.from_rotvec(rotation_field[:, None] * tangential_vectors).as_matrix()

    return rotation_matrices

def calculate_unique_vertices_rotated(tet, rotation_field):
    '''
    Calculate the vertices of a tetrahedral mesh after rotating each cell by the rotation field.
    Vertices are unique: they are not shared between cells.
    '''

    rotation_matrices = calculate_rotation_matrices(tet, rotation_field)

    # rotate each face by the rotation field around its center
    unique_vertices = np.zeros((tet.number_of_cells, 4, 3))
    for cell_index, cell in enumerate(tet.field_data["cells"]):
        unique_vertices[cell_index] = tet.field_data["cell_vertices"][cell]

    cell_centers = tet.cell_data["cell_center"]

    unique_vertices_rotated = cell_centers.reshape(-1, 1, 3, 1) + rotation_matrices.reshape(-1, 1, 3, 3) @ (unique_vertices.reshape(-1, 4, 3, 1) - cell_centers.reshape(-1, 1, 3, 1))
    # unique_vertices_rotated = rotation_matrices.reshape(-1, 1, 3, 3) @ unique_vertices.reshape(-1, 4, 3, 1)

    return unique_vertices_rotated

def apply_rotation_field_unique_vertices(tet, rotation_field):
    '''
    Apply the rotation field to the tetrahedral mesh and return a new tetrahedral mesh.
    Vertices are unique: they are not shared between cells.
    '''

    unique_vertices_rotated = calculate_unique_vertices_rotated(tet, rotation_field)

    unique_cells = np.zeros((tet.number_of_cells, 5), dtype=int)
    unique_cells[:, 0] = 4
    unique_cells[:, 1:] = np.arange(tet.number_of_cells*4).reshape(-1, 4)

    new_tet = pv.UnstructuredGrid(unique_cells.flatten(), np.full(tet.number_of_cells, pv.CellType.TETRA), unique_vertices_rotated.reshape(-1, 3))

    return new_tet

def apply_rotation_field(tet, rotation_field):
    '''
    Apply the rotation field to the tetrahedral mesh and return a new tetrahedral mesh.
    Vertices are shared between cells, so the surface is closed and smooth.
    '''

    new_vertices = np.zeros((tet.number_of_points, 3))
    vertices_count = np.zeros((tet.number_of_points))
    for cell in tet.field_data["cells"]:
        vertices_count[cell] += 1

    unique_vertices_rotated = calculate_unique_vertices_rotated(tet, rotation_field)

    for cell_index, vertices in enumerate(unique_vertices_rotated):
        for i, vertex in enumerate(vertices):
            new_vertices[tet.field_data["cells"][cell_index, i]] += vertex.T[0] / vertices_count[tet.field_data["cells"][cell_index][i]]

    new_tet = pv.UnstructuredGrid(tet.cells, np.full(tet.number_of_cells, pv.CellType.TETRA), new_vertices)

    return new_tet


def optimize_rotations(tet, NEIGHBOUR_LOSS_WEIGHT, MAX_OVERHANG, ROTATION_MULTIPLIER, ITERATIONS, SAVE_GIF, STEEP_OVERHANG_COMPENSATION, INITIAL_ROTATION_FIELD_SMOOTHING, SET_INITIAL_ROTATION_TO_ZERO, MAX_POS_ROTATION, MAX_NEG_ROTATION):
    '''
    Optimize the rotation field for each cell in the tetrahedral mesh to make overhangs less
    than MAX_OVERHANG while keeping the rotation field smooth.
    '''

    imgs = []
    plotter = pv.Plotter(off_screen=True)
    if SAVE_GIF:
        plotter.open_gif(f'gifs/{model_name}_optimize_rotations.gif')

    initial_rotation_field = calculate_initial_rotation_field(tet, MAX_OVERHANG, ROTATION_MULTIPLIER, STEEP_OVERHANG_COMPENSATION, INITIAL_ROTATION_FIELD_SMOOTHING, SET_INITIAL_ROTATION_TO_ZERO, MAX_POS_ROTATION, MAX_NEG_ROTATION)
    num_cells_with_initial_rotation = np.sum(~np.isnan(initial_rotation_field))

    def save_gif(rotation_field):
        new_tet = apply_rotation_field_unique_vertices(tet, rotation_field)
        new_tet.cell_data["rotation_field"] = rotation_field
        mesh_actor = plotter.add_mesh(new_tet,  clim=[-np.pi/4, np.pi/4], scalars="rotation_field", lighting=False)
        plotter.write_frame()
        plotter.remove_actor(mesh_actor)

    def objective_function(rotation_field):
        '''
        Objective function to minimize the neighbour losses and initial rotation losses.
        '''
        if SAVE_GIF:
            save_gif(rotation_field)

        # Compute neighbour losses using vectorized operations
        cell_face_neighbours = tet.field_data["cell_face_neighbours"]
        neighbour_differences = rotation_field[cell_face_neighbours[:, 0]] - rotation_field[cell_face_neighbours[:, 1]]
        neighbour_losses = NEIGHBOUR_LOSS_WEIGHT * neighbour_differences**2

        # Compute the initial rotation losses
        overhanging_mask = tet.cell_data['overhang_angle'] > np.deg2rad(90 + MAX_OVERHANG)
        valid_cell_indices = np.where(~np.isnan(initial_rotation_field))[0]#np.where(overhanging_mask)[0]
        initial_rotation_losses = (rotation_field[valid_cell_indices] - initial_rotation_field[valid_cell_indices])**2

        # Return the concatenated losses
        return np.concatenate((neighbour_losses, initial_rotation_losses))


    def objective_jacobian(rotation_field):
        start_time = time.time()
        # Initialize the sparse matrix with LIL format for efficient row-wise operations
        cell_face_neighbours = tet.field_data["cell_face_neighbours"]
        jac = lil_matrix((len(cell_face_neighbours) + num_cells_with_initial_rotation, tet.number_of_cells), dtype=np.float32)

        # Vectorized computation for neighbour loss derivatives
        cell_1 = cell_face_neighbours[:, 0]
        cell_2 = cell_face_neighbours[:, 1]

        # Compute the differences
        differences = rotation_field[cell_1] - rotation_field[cell_2]

        # Fill in the Jacobian for the first derivative of the neighbour loss function
        jac[range(len(cell_face_neighbours)), cell_1] = 2 * NEIGHBOUR_LOSS_WEIGHT * differences
        jac[range(len(cell_face_neighbours)), cell_2] = -2 * NEIGHBOUR_LOSS_WEIGHT * differences

        # Vectorized computation for initial rotation loss derivatives
        overhanging_mask = tet.cell_data['overhang_angle'] > np.deg2rad(90 + MAX_OVERHANG)
        valid_cell_indices = np.where(~np.isnan(initial_rotation_field))[0]#np.where(overhanging_mask)[0]

        # Fill in the Jacobian for the first derivative of the initial rotation loss function
        jac[len(cell_face_neighbours) + np.arange(len(valid_cell_indices)), valid_cell_indices] = \
            2 * (rotation_field[valid_cell_indices] - initial_rotation_field[valid_cell_indices])

        # print("Jacobian time:", time.time() - start_time)
        # Convert the LIL matrix to CSR format for efficient computations in further steps
        return jac.tocsr()

    def jac_sparsity():
        cell_face_neighbours = tet.field_data["cell_face_neighbours"]
        sparsity = lil_matrix((len(cell_face_neighbours) + num_cells_with_initial_rotation, tet.number_of_cells), dtype=np.int8)

        for i, (cell_1, cell_2) in enumerate(cell_face_neighbours):
            sparsity[i, cell_1] = 1
            sparsity[i, cell_2] = 1

        valid_cell_indices = np.where(~np.isnan(initial_rotation_field))[0]#np.where(overhanging_mask)[0]
        i = 0
        for cell_index, initial_rotation in enumerate(initial_rotation_field):
            if cell_index in valid_cell_indices:
                sparsity[len(cell_face_neighbours) + i, cell_index] = 1
                i += 1

        return sparsity.tocsr()
    
    if SAVE_GIF:
        plotter.close()

    # If the optimizer was not run or failed to produce 'result', fall back to the
    # initial rotation field computed earlier. This ensures the function always
    # returns a valid rotation field array.
    return initial_rotation_field

NEIGHBOUR_LOSS_WEIGHT = 20 # the larger the weight, the more the rotation field will be smoothed
MAX_OVERHANG = 30          # the maximum overhang angle in degrees
ROTATION_MULTIPLIER = 2   # the larger the multiplier, the more the rotation field will be rotated
SET_INITIAL_ROTATION_TO_ZERO = False # reduces influence of initial rotation field on non-overhanging tetrahedrons. good when initial rotation field is noisy
INITIAL_ROTATION_FIELD_SMOOTHING = 30
MAX_POS_ROTATION = np.deg2rad(3600) # normally set to 360 unless you get collisions
MAX_NEG_ROTATION = np.deg2rad(-3600) # normally set to 360 unless you get collisions
ITERATIONS = 100
SAVE_GIF = True
STEEP_OVERHANG_COMPENSATION = True

rotation_field = optimize_rotations(
    undeformed_tet,
    NEIGHBOUR_LOSS_WEIGHT,
    MAX_OVERHANG,
    ROTATION_MULTIPLIER,
    ITERATIONS,
    SAVE_GIF,
    STEEP_OVERHANG_COMPENSATION,
    INITIAL_ROTATION_FIELD_SMOOTHING,
    SET_INITIAL_ROTATION_TO_ZERO,
    MAX_POS_ROTATION,
    MAX_NEG_ROTATION
)
# rotation_field = calculate_initial_rotation_field(tet, MAX_OVERHANG, ROTATION_MULTIPLIER)
undeformed_tet_with_rotated_tetrahedrons = apply_rotation_field_unique_vertices(undeformed_tet, rotation_field)
undeformed_tet_with_rotated_tetrahedrons.cell_data["rotation_field"] = rotation_field
# new_tet.extract_cells(np.where(rotation_field != 0)[0]).plot()
undeformed_tet_with_rotated_tetrahedrons.plot(scalars="rotation_field")


# default
# ```
# NEIGHBOUR_LOSS_WEIGHT = 20 # the larger the weight, the more the rotation field will be smoothed
# MAX_OVERHANG = 30          # the maximum overhang angle in degrees
# ROTATION_MULTIPLIER = 2   # the larger the multiplier, the more the rotation field will be rotated
# SET_INITIAL_ROTATION_TO_ZERO = False # reduces influence of initial rotation field on non-overhanging tetrahedrons. good when initial rotation field is noisy
# ```
# 
# benchy upsidedown tilted
# scale: 1.5
# Iteration 1:
# ```
# NEIGHBOUR_LOSS_WEIGHT = 100 # the larger the weight, the more the rotation field will be smoothed
# MAX_OVERHANG = 5          # the maximum overhang angle in degrees
# ROTATION_MULTIPLIER = 1   # the larger the multiplier, the more the rotation field will be rotated
# SET_INITIAL_ROTATION_TO_ZERO = True # reduces influence of initial rotation field on non-overhanging tetrahedrons. good when initial rotation field is noisy
# INITIAL_ROTATION_FIELD_SMOOTHING = 30
# ```
# 
# Iteration 2:
# ```
# NEIGHBOUR_LOSS_WEIGHT = 50 # the larger the weight, the more the rotation field will be smoothed
# MAX_OVERHANG = 5          # the maximum overhang angle in degrees
# ROTATION_MULTIPLIER = 1   # the larger the multiplier, the more the rotation field will be rotated
# SET_INITIAL_ROTATION_TO_ZERO = True # reduces influence of initial rotation field on non-overhanging tetrahedrons. good when initial rotation field is noisy
# INITIAL_ROTATION_FIELD_SMOOTHING = 30
# ```
# 
# Iteration 3,4,5:
# ```
# NEIGHBOUR_LOSS_WEIGHT = 50 # the larger the weight, the more the rotation field will be smoothed
# MAX_OVERHANG = 50          # the maximum overhang angle in degrees
# ROTATION_MULTIPLIER = 3   # the larger the multiplier, the more the rotation field will be rotated
# SET_INITIAL_ROTATION_TO_ZERO = True # reduces influence of initial rotation field on non-overhanging tetrahedrons. good when initial rotation field is noisy
# INITIAL_ROTATION_FIELD_SMOOTHING = 30
# ```

# In[ ]:


# view the initial rotation field we are trying to optimize towards
# tet.extract_cells(tet.cell_data['overhang_angle'] > np.deg2rad(90 + MAX_OVERHANG)).plot(scalars="initial_rotation_field")
# tet.cell_data['overhang_angle'] > np.deg2rad(90 + MAX_OVERHANG)
# undeformed_tet.plot(scalars="initial_rotation_field")
# undeformed_tet.plot(scalars="in_air")
undeformed_tet.plot(scalars="cell_distance_to_bottom", cpos=[-0.5, -1, -1])
# undeformed_tet.plot(scalars="overhang_angle")
# undeformed_tet.plot(scalars="path_length_to_base_gradient")
# undeformed_tet.plot(scalars=new_tet1.cell_data['rotation_field'])

def show_path_to_base(tet, cell_index, plotter=pv.Plotter()):
    path_to_bottom = tet.cell_data['path_to_bottom'][cell_index]
    first_negative_index = np.where(path_to_bottom == -1)[0][0]
    path_to_bottom = path_to_bottom[:first_negative_index]
    print(path_to_bottom)

    for i in range(len(path_to_bottom)-1):
        path_to_base = pv.Line(tet.cell_data["cell_center"][path_to_bottom[i]], tet.cell_data["cell_center"][path_to_bottom[i+1]])
        plotter.add_mesh(path_to_base, color="red")

    plotter.add_mesh(tet, opacity=0.2)
    plotter.show()

# show_path_to_base(tet, np.where(tet.cell_data['has_face'].astype(bool) & (tet.cell_data["overhang_angle"] > 3))[0][7])

def show_path_to_base_gradient_calculation(tet, cell_indices):
    plotter = pv.Plotter()
    scalar = np.full(tet.number_of_cells, 0.0)

    for cell_index in cell_indices:
        local_cells = cell_neighbour_dict["edge"][cell_index]
        local_cells = np.hstack((local_cells, cell_index))
        local_cells_with_path_lengths = [x for x in local_cells if not np.isnan(tet.cell_data['cell_distance_to_bottom'][x])]
        path_lengths = tet.cell_data['cell_distance_to_bottom'][local_cells_with_path_lengths]

        cell_centers = tet.cell_data["cell_center"][local_cells_with_path_lengths]
        cell_centers_z_is_path_length = cell_centers.copy()
        cell_centers_z_is_path_length[:, 2] = path_lengths - np.min(path_lengths) + np.min(cell_centers[:, 2])
        points = pv.PolyData(cell_centers_z_is_path_length)
        glyph = points.glyph(geom=pv.Sphere(theta_resolution=8, phi_resolution=8, radius=0.1))

        if len(cell_centers_z_is_path_length) < 3:
            continue

        p, n = planeFit(cell_centers_z_is_path_length.T)
        plane = pv.Plane(center=p, direction=n, i_size=4, j_size=4)
        normal_arrow = pv.Arrow(start=p, direction=n, scale=1.5)

        # reflect arrow across xy plane
        reflected_n = -n - 2 * (np.dot(-n, up_vector)) * up_vector

        # extract radial component
        cell_center_direction_normalized = tet.cell_data["cell_center"][cell_index, :2] / np.linalg.norm(tet.cell_data["cell_center"][cell_index, :2])
        gradient_in_radial_direction = np.dot(cell_center_direction_normalized, reflected_n[:2]) * cell_center_direction_normalized
        nozzle_arrow = pv.Arrow(start=tet.cell_data["cell_center"][cell_index], direction=np.hstack((gradient_in_radial_direction, reflected_n[2])), scale=1)


        # plotter.add_mesh(glyph, color="blue", opacity=0.5, )
        # plotter.add_mesh(plane, color="green", opacity=0.2)
        # plotter.add_mesh(normal_arrow, color="orange")
        plotter.add_mesh(nozzle_arrow, color="red")
    scalar[local_cells_with_path_lengths] = 0.5
    scalar[cell_index] = 1
    plotter.add_mesh(tet, opacity=0.2, scalars=scalar, cmap="binary")
    plotter.show()

# show_path_to_base_gradient_calculation(undeformed_tet, [np.where(tet.cell_data['has_face'].astype(bool) & (tet.cell_data["overhang_angle"] > 3))[0][1]])
# show_path_to_base_gradient_calculation(undeformed_tet, np.where(tet.cell_data['has_face'].astype(bool) & (tet.cell_data["overhang_angle"] > np.deg2rad(90 + MAX_OVERHANG)))[0])

def show_dijkstras(tet, cell_index):
    plotter = pv.Plotter()#window_size=[3840, 2160])
    lines = []
    for neighbour in tet.field_data["cell_face_neighbours"]:
        lines += [tet.cell_data["cell_center"][neighbour[0]], tet.cell_data["cell_center"][neighbour[1]]]
    mesh = pv.line_segments_from_points(lines)
    plotter.add_mesh(mesh, color="grey", opacity=0.4)

    points = pv.PolyData(tet.cell_data["cell_center"])
    glyph = points.glyph(geom=pv.Sphere(theta_resolution=8, phi_resolution=8, radius=0.1))
    plotter.add_mesh(glyph, color="red", opacity=0.2)

    if cell_index is None:
        plotter.show()
        return

    plotter.camera_position = "xz"

    # run dijkstra's algorithm and visualize
    plotter.open_gif(f'gifs/{model_name}_dijkstra.gif')
    distances, paths = nx.single_source_dijkstra(cell_neighbour_graph, cell_index)
    dijkstra_actors = []
    for i in np.arange(0, tet.cell_data['cell_distance_to_bottom'][cell_index], 0.2):
        nodes_in_range = [node for node, distance in distances.items() if distance < i]
        if len(nodes_in_range) == 0:
            continue
        if set(nodes_in_range) & set(bottom_cells):
            break
        points = pv.PolyData(tet.cell_data["cell_center"][nodes_in_range])
        glyph = points.glyph(geom=pv.Sphere(theta_resolution=8, phi_resolution=8, radius=0.2))
        actor = plotter.add_mesh(glyph, color="blue", opacity=0.4)
        dijkstra_actors.append(actor)
        plotter.write_frame()


    path_to_bottom = tet.cell_data['path_to_bottom'][cell_index]
    first_negative_index = np.where(path_to_bottom == -1)[0][0]
    path_to_bottom = path_to_bottom[:first_negative_index]
    print(path_to_bottom)

    for i in range(len(path_to_bottom)-1):
        path_to_base = pv.Line(tet.cell_data["cell_center"][path_to_bottom[i]], tet.cell_data["cell_center"][path_to_bottom[i+1]])
        plotter.add_mesh(path_to_base, color="blue", line_width=5)

    for actor in dijkstra_actors:
        plotter.remove_actor(actor)

    plotter.write_frame()

    plotter.show()
    plotter.close()


# show_dijkstras(undeformed_tet, np.where(tet.cell_data['has_face'].astype(bool) & (tet.cell_data["overhang_angle"] > 3))[0][1])


# In[ ]:


N = np.eye(4) - 1/4 * np.ones((4, 4)) # the N matrix centers the vertices of a tetrahedron around the origin

save_gif_i = 0

def calculate_deformation(tet, rotation_field, ITERATIONS, SAVE_GIF):
    '''
    Try to find the optimal deformation of the tetrahedral mesh to make cells have the same rotation as
    the given rotation field.

    Our parameters are the vertices of the deformed mesh.
    '''

    new_vertices = tet.points.copy()

    params = new_vertices.flatten()

    rotation_matrices = calculate_rotation_matrices(tet, rotation_field)

    # Extract old vertices for all cells
    old_vertices = tet.field_data["cell_vertices"][tet.field_data["cells"]]
    # Apply the transformation for all cells
    old_vertices_transformed = np.einsum('ijk,ikl->ijl', rotation_matrices, (N @ old_vertices).transpose(0, 2, 1))

    plotter = pv.Plotter(off_screen=True)

    if SAVE_GIF:
        plotter.open_gif(f'gifs/{model_name}_calculate_deformation.gif')

    def save_gif(new_vertices):
        global save_gif_i
        save_gif_i += 1

        if save_gif_i % 10 != 0:
            return

        new_tet = pv.UnstructuredGrid(tet.cells, np.full(tet.number_of_cells, pv.CellType.TETRA), new_vertices)
        mesh_actor = plotter.add_mesh(new_tet)
        plotter.write_frame()
        plotter.remove_actor(mesh_actor)


    def objective_function(params):
        start_time = time.time()

        new_vertices = params[:tet.number_of_points * 3].reshape(-1, 3)

        if SAVE_GIF:
            save_gif(new_vertices)

        # Apply transformation for the new vertices
        new_vertices_transformed = (N @ new_vertices[tet.field_data["cells"]]).transpose(0, 2, 1)

        # Calculate position compatibility loss using vectorized operations
        position_losses = np.linalg.norm(new_vertices_transformed - old_vertices_transformed, axis=(1, 2))**2

        # print(f"Objective function took {time.time() - start_time} seconds")
        return position_losses

    def objective_jacobian(params):
        start_time = time.time()

        # Initialize Jacobian matrix
        J = lil_matrix((tet.number_of_cells, len(params)), dtype=np.float32)

        # Extract parameters
        new_vertices = params[:tet.number_of_points * 3].reshape(-1, 3)

        # Extract old vertices for all cells
        old_vertices = tet.field_data["cell_vertices"][tet.field_data["cells"]]

        # Apply the transformation for old and new vertices
        new_vertices_transformed = (N @ new_vertices[tet.field_data["cells"]]).transpose(0, 2, 1)

        # Compute the difference between transformed new and old vertices
        diff = new_vertices_transformed - old_vertices_transformed  # shape: (num_cells, 3, num_vertices_per_cell)

        # Reshape diff for easier broadcasting
        diff = diff.transpose(0, 2, 1)  # shape: (num_cells, num_vertices_per_cell, 3)

        # Now, for each cell, update the corresponding rows in the Jacobian
        cell_indices = np.repeat(np.arange(tet.number_of_cells), len(tet.field_data["cells"][0]))  # Cell indices repeated per vertex
        vertex_indices = np.ravel(tet.field_data["cells"])  # Flatten the cell-to-vertex mapping

        # For each component x, y, z in the vertex, update the Jacobian
        for dim in range(3):
            J[cell_indices, vertex_indices * 3 + dim] = 2 * diff[:, :, dim].ravel()

        # print(f"Objective jacobian took {time.time() - start_time} seconds")
        return J.tocsr()

    def jac_sparsity():
        sparsity = lil_matrix((tet.number_of_cells, len(params)), dtype=np.int8)

        cell_indices = np.repeat(np.arange(tet.number_of_cells), len(tet.field_data["cells"][0]))
        vertex_indices = np.ravel(tet.field_data["cells"])

        for dim in range(3):
            sparsity[cell_indices, vertex_indices * 3 + dim] = 1

        return sparsity.tocsr()

    result = least_squares(objective_function,
                    params,
                    max_nfev=ITERATIONS,
                    verbose=2,
                    jac=objective_jacobian,
                    jac_sparsity=jac_sparsity(),
                    method='trf',
                    x_scale='jac',
                    )

    plotter.close()

    return result.x[:tet.number_of_points*3].reshape(-1, 3)

ITERATIONS = 1000
SAVE_GIF = True
new_vertices = calculate_deformation(undeformed_tet, rotation_field, ITERATIONS, SAVE_GIF)
deformed_tet = pv.UnstructuredGrid(undeformed_tet.cells, np.full(undeformed_tet.number_of_cells, pv.CellType.TETRA), new_vertices)
deformed_tet.plot()

for key in undeformed_tet.field_data.keys():
    deformed_tet.field_data[key] = undeformed_tet.field_data[key]
for key in undeformed_tet.cell_data.keys():
    deformed_tet.cell_data[key] = undeformed_tet.cell_data[key]
deformed_tet = update_tet_attributes(deformed_tet)


# Run below to do another iteration

# In[10]:


undeformed_tet = deformed_tet.copy()


# Run below when finished deforming to save mesh as STL

# In[12]:


# make origin center bottom of bounding box
x_min, x_max, y_min, y_max, z_min, z_max = deformed_tet.bounds
offsets_applied = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min])
deformed_tet.points -= offsets_applied

deformed_tet.extract_surface().save(f'output_models/{model_name}_deformed_tet.stl')


# In[15]:


# save to pickle
with open(f'pickle_files/deformed_{model_name}.pkl', 'wb') as f:
    pickle.dump(deformed_tet, f)


# # Now, go and slice the stl file in Cura!
# 
# Settings:
# - Make the printer origin at the center of the buildplate
# - Dont use any pre/post scripts, z hop, etc. The config I use is provided in the github repo
# - Autoplace the model at the center by clicking "Arrange All Models"

# In[16]:


deformed_tet = pickle.load(open(f'pickle_files/deformed_{model_name}.pkl', 'rb'))


# In[17]:


def tetrahedron_volume(p1, p2, p3, p4):
    '''
    Calculate the volume of the tetrahedron formed by four points
    '''

    mat = np.vstack([p2 - p1, p3 - p1, p4 - p1])
    return np.abs(np.linalg.det(mat)) / 6

def calc_barycentric_coordinates(tet_a, tet_b, tet_c, tet_d, point):
    '''
    Calculate the barycentric coordinates of a point in a tetrahedron. This is used to interpolate
    parameters from the vertices of the tetrahedron to a point within the tetrhedron.
    '''

    total_volume = tetrahedron_volume(tet_a, tet_b, tet_c, tet_d)

    if total_volume == 0:
        raise ValueError("The points do not form a valid tetrahedron (zero volume).")

    # Calculate the sub-volumes for each face
    vol_a = tetrahedron_volume(point, tet_b, tet_c, tet_d)
    vol_b = tetrahedron_volume(point, tet_a, tet_c, tet_d)
    vol_c = tetrahedron_volume(point, tet_a, tet_b, tet_d)
    vol_d = tetrahedron_volume(point, tet_a, tet_b, tet_c)

    # Calculate barycentric coordinates as the ratio of sub-volumes to total volume
    lambda_a = vol_a / total_volume
    lambda_b = vol_b / total_volume
    lambda_c = vol_c / total_volume
    lambda_d = vol_d / total_volume

    # The barycentric coordinates should sum to 1
    return np.array([lambda_a, lambda_b, lambda_c, lambda_d])

def project_point_onto_plane(plane_x_axis, plane_y_axis, point):
    projected_x = np.sum(plane_x_axis * point, axis=1)
    projected_y = np.sum(plane_y_axis * point, axis=1)

    return np.array([projected_x, projected_y]).T


# In[18]:


deformed_tet, _, _ = calculate_tet_attributes(deformed_tet)


# In[19]:


from pygcode import Line
import time

SEG_SIZE = 0.6 # mm
MAX_ROTATION = 30 # degrees
MIN_ROTATION = -130 # degrees
NOZZLE_OFFSET = 42 # mm actuallt 41.5

# find how each vertex in tet has been transformed
vertex_transformations = deformed_tet.points - input_tet.points

# calculate tangential vectors (axis of rotation) for each cell
tangential_vectors = np.cross( np.array([0, 0, 1]), input_tet.cell_data["cell_center"][:, :2])
# normalize
tangential_vectors /= np.linalg.norm(tangential_vectors, axis=1)[:, None]
# replace nan with [1,0,0]
tangential_vectors[np.isnan(tangential_vectors).any(axis=1)] = [1, 0, 0]

# calculate rotation for each vertex and cell
num_cells_per_vertex = np.zeros((input_tet.number_of_points))
for cell_index, cell in enumerate(input_tet.field_data["cells"]):
    num_cells_per_vertex[cell] += 1
vertex_rotations = np.zeros((deformed_tet.number_of_points))
cell_rotations = np.zeros((deformed_tet.number_of_cells))
for cell_index, cell in enumerate(deformed_tet.field_data["cells"]):
    new_vertices = deformed_tet.field_data["cell_vertices"][cell]
    new_cell_center = deformed_tet.cell_data["cell_center"][cell_index]
    old_vertices = input_tet.field_data["cell_vertices"][cell]
    old_cell_center = input_tet.cell_data["cell_center"][cell_index]

    # center points
    new_vertices -= new_cell_center
    old_vertices -= old_cell_center

    # project on to radial plane
    plane_x_vector = old_cell_center[:2] / np.linalg.norm(old_cell_center[:2])
    plane_x_vector = np.array([plane_x_vector[0], plane_x_vector[1], 0])
    plane_y_vector = np.array([0,0,1])

    new_vertices_projected = project_point_onto_plane(plane_x_vector, plane_y_vector, new_vertices)
    old_vertices_projected = project_point_onto_plane(plane_x_vector, plane_y_vector, old_vertices)

    # find rotation between the two sets of points using the kabsch algorithm
    covariance_matrix = np.dot(new_vertices_projected.T, old_vertices_projected)
    U, _, Vt = np.linalg.svd(covariance_matrix)
    rotation_matrix = np.dot(U, Vt)

    # get rotation angle from matrix 2x2
    rotation = -np.arccos(min(max(rotation_matrix[0, 0], -1), 1))
    if rotation_matrix[1, 0] < 0:
        rotation = -rotation

    rotation = max(min(rotation, np.deg2rad(MAX_ROTATION)), np.deg2rad(MIN_ROTATION))

    cell_rotations[cell_index] = rotation

    for vertex_index in cell:
        vertex_rotations[vertex_index] += rotation / num_cells_per_vertex[vertex_index]

# calculate z squish scale for each cell (ratio of z length after rotation to z length before rotation)
tet_rotation_matrices = calculate_rotation_matrices(input_tet, cell_rotations)
z_squish_scales = np.full((deformed_tet.number_of_cells), np.nan)
for cell_index, cell in enumerate(deformed_tet.field_data["cells"]):
    warped_vertices = deformed_tet.field_data["cell_vertices"][cell]
    unwarped_vertices = input_tet.field_data["cell_vertices"][cell]

    # rotate new vertices to align with old vertices
    unwarped_vertices_rotated = (tet_rotation_matrices[cell_index].reshape(1, 3, 3) @ unwarped_vertices.reshape(4, 3, 1)).reshape(4, 3)

    # calculate z squish scale
    # z_squish_scales[cell_index] = (unwarped_vertices_rotated[:, 2].max() - unwarped_vertices_rotated[:, 2].min()) / (warped_vertices[:, 2].max() - warped_vertices[:, 2].min())
    z_squish_scales[cell_index] = tetrahedron_volume(*unwarped_vertices) / tetrahedron_volume(*warped_vertices)
    # z_squish_scales[cell_index] = min(z_squish_scales[cell_index], 5) # cap z squish scale


# read gcode
pos = np.array([0., 0., 20.])
feed = 5000
gcode_points = []
with open(f'input_gcode/{model_name}_deformed_tet.gcode', 'r') as fh:
    for line_text in fh.readlines():
        line = Line(line_text)

        if not line.block.gcodes:
            continue

        for gcode in sorted(line.block.gcodes):
            if gcode.word == "G01" or gcode.word == "G00":
                prev_pos = pos.copy()

                if gcode.X is not None:
                    pos[0] = gcode.X
                if gcode.Y is not None:
                    pos[1] = gcode.Y
                if gcode.Z is not None:
                    pos[2] = gcode.Z

                inv_time_feed = None
                # extract feed
                for word in line.block.words:
                    if word.letter == "F":
                        feed = word.value

                # extract extrusion
                extrusion = None
                for param in line.block.modal_params:
                    if param.letter == "E":
                        extrusion = param.value

                # segment moves
                # makes G1 (feed moves) less jittery
                delta_pos = pos - prev_pos
                distance = np.linalg.norm(delta_pos)
                if distance > 0:
                    num_segments = -(-distance // SEG_SIZE) # hacky round up
                    seg_distance = distance/num_segments

                    # calculate inverse time feed
                    time_to_complete_move = (1/feed) * seg_distance # min/mm * mm = min
                    if time_to_complete_move == 0:
                        inv_time_feed = None
                    else:
                        inv_time_feed = 1/time_to_complete_move # 1/min

                    for i in range(int(num_segments)):
                        gcode_points.append({
                            "position": (prev_pos + delta_pos * (i+1) / num_segments),
                            "command": gcode.word,
                            "extrusion": extrusion/num_segments if extrusion is not None else None,
                            "inv_time_feed": inv_time_feed,
                            "move_length": seg_distance,
                            "start_position": prev_pos,
                            "end_position": pos,
                            "unsegmented_move_length": distance,
                            "after_retract": False,
                            "feed": feed
                        })
                else:
                    # calculate inverse time feed
                    time_to_complete_move = (1/feed) * distance # min/mm * mm = min
                    if time_to_complete_move == 0:
                        inv_time_feed = None
                    else:
                        inv_time_feed = 1/time_to_complete_move # 1/min

                    gcode_points.append({
                        "position": pos.copy(),
                        "command": gcode.word,
                        "extrusion": extrusion,
                        "inv_time_feed": inv_time_feed,
                        "move_length": distance,
                        "unsegmented_move_length": distance,
                        "after_retract": False,
                        "feed": feed
                    })

                # # add G0 in same spot after retraction (so we can use it for zhop later)
                # if gcode.word == "G01" and extrusion is not None and extrusion < 0:
                #     gcode_points.append({
                #         "position": pos.copy(),
                #         "command": "G00",
                #         "extrusion": None,
                #         "inv_time_feed": None,
                #         "move_length": 0,
                #         "after_retract": True
                #     })

# calculate containging cell for each gcode point
gcode_points_containing_cells = deformed_tet.find_containing_cell([point["position"] for point in gcode_points])

# for cells with no containing cell, find the closest cell
gcode_points_closest_cells = deformed_tet.find_closest_cell([point["position"] for point in gcode_points])
# gcode_points_containing_cells[gcode_points_containing_cells == -1] = gcode_points_closest_cells[gcode_points_containing_cells == -1]

# transform gcode points to original mesh's shape
new_gcode_points = []
prev_new_position = None
travelling_over_air = False
travelling = False
prev_position = None
prev_rotation = 0
prev_travelling = False
prev_command = "G00"
ROTATION_AVERAGING_ALPHA = 0.2 # exponential moving average alpha for rotation
RETRACTION_LENGTH = 1.0
ROTATION_MAX_DELTA = np.deg2rad(1)
MAX_EXTRUSION_MULTIPLIER = 10
lost_vertices = []
highest_printed_point = 0
for cell_index, (gcode_point, containing_cell_index) in enumerate(zip(gcode_points, gcode_points_containing_cells)):
    position = gcode_point["position"]
    command = gcode_point["command"]
    inv_time_feed = gcode_point["inv_time_feed"]
    extrusion = gcode_point["extrusion"]

    def barycentric_interpolate_to_get_new_position_and_rotation(position, containing_cell_index, command, cell_index):
        if command == "G00" and containing_cell_index == -1: # Strict on travel moves being inside a tet
            return None, None
        if command == "G01" and containing_cell_index == -1: # Slightly more relaxed on printing moves
            containing_cell_index = gcode_points_closest_cells[cell_index]

        # get barycentric coordinates of pos in containing cell
        vertiex_indices = deformed_tet.field_data["cells"][containing_cell_index]
        cell_vertices = deformed_tet.field_data["cell_vertices"][vertiex_indices]
        barycentric_coordinates = calc_barycentric_coordinates(cell_vertices[0], cell_vertices[1], cell_vertices[2], cell_vertices[3], position)

        if np.sum(barycentric_coordinates) > 1.01:
            return None, None

        # calculate the new position of the point using the barycentric coordinates to weigh the vertex transformations
        # multiply barycentric coordinates row-wise with vertex transformations
        transformation = vertex_transformations[vertiex_indices] * barycentric_coordinates[:, None]

        # sum columns
        transformation = np.sum(transformation, axis=0)
        # apply to pos
        new_position = position - transformation

        # do the same for rotation
        rotation = np.sum(vertex_rotations[vertiex_indices] * barycentric_coordinates)

        return new_position, rotation

    dont_smooth_rotation = False
    new_position, rotation = barycentric_interpolate_to_get_new_position_and_rotation(position, containing_cell_index, command, cell_index)
    if new_position is None:
        if command == "G01":
            lost_vertices.append(position)
            continue
        elif command == "G00" and not travelling_over_air and prev_new_position is not None:
            new_position = np.array([prev_new_position[0], prev_new_position[1], highest_printed_point]) # z hop over gap
            rotation = max(min(prev_rotation, np.deg2rad(45)), np.deg2rad(-45)) # set rotation to a max of 45 because if rotation is very large, the extruder can "hang below" the nozzle and hit the part
            dont_smooth_rotation = True # force rotation immediately
            travelling_over_air = True
        elif travelling_over_air:
            continue
        else:
            continue
    else:
        if travelling_over_air:
            new_position[2] = highest_printed_point # finish z hop over gap
            rotation = max(min(rotation, np.deg2rad(45)), np.deg2rad(-45)) # set rotation to 0 because if rotation is very large, the extruder can "hang below" the nozzle and hit the part
            dont_smooth_rotation = True # force rotation immediately
        travelling_over_air = False

    extrusion_multiplier = 1
    if extrusion is not None and extrusion != RETRACTION_LENGTH and extrusion != -RETRACTION_LENGTH:

        # scale extrusion by z_squish_scale
        extrusion_multiplier = extrusion_multiplier * z_squish_scales[containing_cell_index]
        extrusion = extrusion * min(extrusion_multiplier, MAX_EXTRUSION_MULTIPLIER)
    elif extrusion == -RETRACTION_LENGTH:
        travelling = True
    elif extrusion == RETRACTION_LENGTH:
        travelling = False
    if prev_rotation is not None and not dont_smooth_rotation:
        rotation = ROTATION_AVERAGING_ALPHA * rotation + (1 - ROTATION_AVERAGING_ALPHA) * prev_rotation

    # if rotation delta between points is too high, add intermediate interpolation points to prevent nozzle from hitting part as rotating
    if prev_rotation is not None and prev_new_position is not None and np.abs(rotation - prev_rotation) > ROTATION_MAX_DELTA:
        delta_rotation = rotation - prev_rotation
        num_interpolations = int(np.abs(delta_rotation) / ROTATION_MAX_DELTA) + 1
        delta_pos = new_position - prev_new_position
        for i in range(num_interpolations):
            new_gcode_points.append({
                "position": prev_new_position + (delta_pos * ((i+1) / num_interpolations)),
                "original_position": position,
                "rotation": prev_rotation + (delta_rotation * ((i+1) / num_interpolations)),
                "command": prev_command,
                "extrusion": extrusion/num_interpolations if extrusion is not None else None,
                "inv_time_feed": inv_time_feed * num_interpolations if inv_time_feed is not None else None,
                "extrusion_multiplier": extrusion_multiplier,
                "feed": gcode_point["feed"],
                "travelling": prev_travelling
            })
    else:
        new_gcode_points.append({
            "position": new_position,
            "original_position": position,
            "rotation": rotation,
            "command": command,
            "extrusion": extrusion,
            "inv_time_feed": inv_time_feed,
            "extrusion_multiplier": extrusion_multiplier,
            "feed": gcode_point["feed"],
            "travelling": travelling
        })

    prev_rotation = rotation
    prev_new_position = new_position.copy()
    prev_travelling = travelling
    prev_command = command

    if command == "G01" and extrusion is not None and extrusion > 0 and (highest_printed_point != 0 or new_position[2] < 1):
        highest_printed_point = max(highest_printed_point, new_position[2])


print(f"Lost {len(lost_vertices)} vertices")


# In[20]:


prev_r = 0
prev_theta = 0
prev_z = 20

theta_accum = 0

# save transformed gcode
with open(f'output_gcode/{model_name}.gcode', 'w') as fh:
    # write header
    fh.write("G94 ; mm/min feed  \n")
    fh.write("G28 ; home \n")
    fh.write("M83 ; relative extrusion \n")
    fh.write("G1 E10 ; prime extruder \n")
    fh.write("G94 ; mm/min feed \n")
    fh.write("G90 ; absolute positioning \n")
    fh.write(f"G0 C{prev_theta} X{prev_r} Z{prev_z} B0 ; go to start \n")
    fh.write("G93 ; inverse time feed \n")

    for i, point in enumerate(new_gcode_points):
        position = point["position"]
        rotation = point["rotation"]

        if np.all(np.isnan(position)):
            continue

        if position[2] < 0:
            continue

        z_hop = 0
        if point["travelling"]:
            z_hop = 1

        # convert to polar coordinates
        r = np.linalg.norm(position[:2])
        theta = np.arctan2(position[1], position[0])
        z = position[2]

        # compensate for nozzle offset
        r += -np.sin(rotation) * (NOZZLE_OFFSET + z_hop)
        z += (np.cos(rotation) - 1) * (NOZZLE_OFFSET + z_hop) + z_hop

        delta_theta = theta - prev_theta
        if delta_theta > np.pi:
            delta_theta -= 2*np.pi
        if delta_theta < -np.pi:
            delta_theta += 2*np.pi

        theta_accum += delta_theta

        string = f"{point['command']} C{np.rad2deg(theta_accum):.5f} X{r:.5f} Z{z:.5f} B{np.rad2deg(rotation):.5f}" # polar printer
        # string = f"{point['command']} X{position[0]:.5f} Y{position[1]:.5f} Z{position[2]} B{np.rad2deg(rotation):.5f}" # cartesian printer (3 axis)

        if point["extrusion"] is not None:
            string += f" E{point['extrusion']:.4f}"

        no_feed_value = False
        if point["inv_time_feed"] is not None:
            string += f" F{(point['inv_time_feed']):.4f}"
        else:
            string += f" F20000"
            fh.write(f"G94\n")
            no_feed_value = True

        fh.write(string + "\n")

        if no_feed_value:
            fh.write(f"G93\n") # back to inv feed

        # update previous values
        prev_r = r
        prev_theta = theta
        prev_z = z


# In[21]:


# plot new_gcode_points using pyvista
temp = np.array([point["position"] for point in new_gcode_points])
#temp = np.array(a)
temp = pv.PolyData(temp)
temp.cell_data["rotation"] = np.array([np.rad2deg(point["rotation"]) for point in new_gcode_points])
temp.cell_data["travelling"] = np.array([point["travelling"] for point in new_gcode_points])
temp.cell_data["delta_rotation"] = np.clip(np.array([1] + [np.rad2deg(new_gcode_points[i+1]["rotation"] - new_gcode_points[i]["rotation"]) for i in range(len(new_gcode_points)-1)]), -10, 10)
temp.cell_data["command"] = np.array([point["command"] for point in new_gcode_points])
temp.cell_data["feed"] = np.array([min(point["feed"], 10000) for point in new_gcode_points])
temp.cell_data["original_z"] = np.array([point["original_position"][2] for point in new_gcode_points])
temp.cell_data["original_z_bands"] = temp.cell_data["original_z"] % 1
temp.cell_data["extrusion_multiplier"] = np.array([min(point["extrusion_multiplier"], 15) if point["extrusion_multiplier"] is not None else np.nan for point in new_gcode_points])

temp.extract_cells(np.array([point["command"] for point in new_gcode_points]) == "G01").plot(scalars="original_z_bands", cpos=[-0.5, -1, 0.5], point_size=10)
# .extract_cells(np.array([point["command"] for point in new_gcode_points]) == "G01")


# In[22]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the gcode
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
cmap = plt.get_cmap('viridis')
rotation_normalized = temp.cell_data["rotation"] / np.max(np.abs(temp.cell_data["rotation"]),axis=0)
colors = cmap(rotation_normalized)
# for i in np.arange(len(temp.points)-1):
#     ax.plot(
#         [temp.points[i,0],temp.points[i+1,0]],
#         [temp.points[i,1],temp.points[i+1,1]],
#         [temp.points[i,2],temp.points[i+1,2]],
#         c=colors[i],
#         markersize=0.8, linewidth=0.9, marker='.', alpha=0.5)
# plot G00 moves in different color
g00_indices = np.where(temp.cell_data["command"] == "G00")[0]
g01_indices = np.where(temp.cell_data["command"] == "G01")[0]
ax.plot(temp.points[g00_indices,0], temp.points[g00_indices,1], temp.points[g00_indices,2], markersize=0.4, linewidth=0.3, marker=".", alpha=0.5, color="red")
ax.plot(temp.points[g01_indices,0], temp.points[g01_indices,1], temp.points[g01_indices,2], markersize=0.4, linewidth=0.3, marker=".", alpha=0.5, color="blue")
ax.set_box_aspect((np.ptp(temp.points[:,0]), np.ptp(temp.points[:,1]), np.ptp(temp.points[:,2])))
plt.show()

