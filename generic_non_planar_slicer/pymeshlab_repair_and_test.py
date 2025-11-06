#!/usr/bin/env python3
"""Apply a sequence of PyMeshLab filters to attempt to repair the propeller mesh,
save output, and attempt TetGen tetrahedralization. The script is robust: it
iterates a candidate filter list and continues when filters are unavailable.
"""
import os
import sys
import traceback
import numpy as np
import pyvista as pv
import tetgen

base_dir = os.path.dirname(__file__)
mesh_name = 'propeller'
input_path = os.path.join(base_dir, 'input_models', f'{mesh_name}.stl')
repaired_path = os.path.join(base_dir, 'output_models', f'{mesh_name}_pymeshlab_repaired.stl')

print('Loading original mesh:', input_path)
pvmesh = pv.read(input_path)
verts = np.asarray(pvmesh.points)
faces = pvmesh.faces.reshape(-1, 4)[:, 1:]
print('Original vertices, faces:', verts.shape, faces.shape)

try:
    import pymeshlab
    print('pymeshlab version:', getattr(pymeshlab, '__version__', 'unknown'))
except Exception as e:
    print('Failed to import pymeshlab:', e)
    sys.exit(1)

ms = pymeshlab.MeshSet()
ms.load_new_mesh(input_path)
print('Mesh loaded into MeshSet with', ms.current_mesh().vertex_number(), 'verts and', ms.current_mesh().face_number(), 'faces')

# Candidate filter names (MeshLab filter names) - we'll try each one and skip failures
candidate_filters = [
    'mergeclosevertices',           # merge close vertices
    'remove_duplicate_faces',
    'remove_duplicate_vertices',
    'remove_zero_area_faces',
    'remove_unreferenced_vertices',
    'remove_isolated_pieces_by_face_number',
    'repair_non_manifold_edges',
    'repair_non_manifold_vertices',
    'remove_self_intersections',
    'close_holes',
    'triangulate',
    'reorient_faces_towards_positive_vertex_order',
]

print('\nApplying candidate filters (skipping those that error)...')
for fname in candidate_filters:
    try:
        print('Applying filter:', fname)
        # Try to call with no params first
        ms.apply_filter(fname)
        print('  OK')
    except Exception as e:
        # Print smaller traceback but continue
        print('  Filter failed or not available:', fname, '->', str(e).split('\n')[0])

# After filters, do some cleanup passes using commonly available filters with safe params
try:
    print('\nApplying additional cleanup: remove duplicate vertices/faces and close small holes')
    try:
        ms.apply_filter('remove_duplicate_vertices')
    except Exception:
        pass
    try:
        ms.apply_filter('remove_duplicate_faces')
    except Exception:
        pass
    try:
        ms.apply_filter('close_holes', maxholesize=1000)
    except Exception:
        # fallback name
        try:
            ms.apply_filter('close_holes', maxholesize=2000)
        except Exception:
            pass
except Exception:
    traceback.print_exc()

curmesh = ms.current_mesh()
print('After MeshLab filters: verts, faces =', curmesh.vertex_number(), curmesh.face_number())

# Save repaired mesh
print('Saving repaired mesh to:', repaired_path)
ms.save_current_mesh(repaired_path, save_face_color=False)

# Load saved mesh to ensure it's valid
pv_repaired = pv.read(repaired_path)
rv = np.asarray(pv_repaired.points)
rf = pv_repaired.faces.reshape(-1, 4)[:, 1:]
print('Saved repaired mesh stats:', rv.shape, rf.shape)

# Try TetGen test
print('\nAttempting TetGen on repaired mesh...')
try:
    scale = 10.0
    tg = tetgen.TetGen(rv * scale, rf)
    tetra_args = {'switches': 'pq1.6a0.2', 'plc': 1, 'verbose': 1}
    tg.tetrahedralize(**tetra_args)
    grid = tg.grid
    print('TetGen grid points:', grid.number_of_points)
    print('TetGen grid cells:', grid.number_of_cells)
    print('TetGen test succeeded')
except Exception as e:
    print('TetGen test failed:', e)
    # Print a short traceback
    traceback.print_exc()
    sys.exit(2)

print('\nFinished: pymeshlab repair + TetGen test succeeded')
