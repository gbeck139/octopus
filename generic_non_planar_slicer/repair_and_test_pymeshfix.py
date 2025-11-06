#!/usr/bin/env python3
"""Repair propeller.stl using PyMeshFix and attempt a TetGen tetrahedralization test.
Produces output_models/propeller_pymeshfix_repaired.stl
Prints success/failure and statistics.
"""
import os
import sys
import numpy as np
import pyvista as pv
import tetgen

mesh_name = 'propeller'
base_dir = os.path.dirname(__file__)
input_path = os.path.join(base_dir, 'input_models', f'{mesh_name}.stl')
repaired_path = os.path.join(base_dir, 'output_models', f'{mesh_name}_pymeshfix_repaired.stl')

print('Loading original mesh:', input_path)
# load with PyVista for robust reading
pvmesh = pv.read(input_path)
verts = np.asarray(pvmesh.points)
faces = pvmesh.faces.reshape(-1, 4)[:, 1:]
print('Original vertices, faces:', verts.shape, faces.shape)

try:
    import pymeshfix
    from pymeshfix import MeshFix
    print('Using pymeshfix MeshFix')
    # Try trimesh path if available
    try:
        import trimesh
        tmesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mf = MeshFix(tmesh)
        mf.repair(verbose=False)
        fixed = mf.mesh
        # trimesh mesh -> pyvista
        fixed_verts = np.asarray(fixed.vertices)
        fixed_faces = np.asarray(fixed.faces)
    except Exception:
        # Fallback: pass raw arrays
        mf = MeshFix(verts, faces)
        mf.repair(verbose=False)
        # MeshFix stores arrays differently across versions; try common names
        fixed_verts = getattr(mf, 'v', None) or getattr(mf, 'points', None) or getattr(mf, 'vertices', None)
        fixed_faces = getattr(mf, 'f', None) or getattr(mf, 'faces', None)
        if fixed_verts is None or fixed_faces is None:
            # As last resort, try mf.mesh
            mesh_obj = getattr(mf, 'mesh', None)
            if mesh_obj is not None:
                try:
                    fixed_verts = np.asarray(mesh_obj.vertices)
                    fixed_faces = np.asarray(mesh_obj.faces)
                except Exception:
                    pass

    if fixed_verts is None or fixed_faces is None:
        raise RuntimeError('Could not extract repaired mesh arrays from MeshFix')

    # Ensure faces are Nx3
    fixed_faces = np.asarray(fixed_faces)
    if fixed_faces.shape[1] != 3:
        # try reshape for trimesh style
        fixed_faces = fixed_faces.reshape(-1, 3)

    print('Fixed vertices, faces:', fixed_verts.shape, fixed_faces.shape)

    # Save repaired mesh
    out_pv = pv.PolyData(fixed_verts, np.column_stack((np.full(len(fixed_faces), 3), fixed_faces)).flatten())
    out_pv.save(repaired_path)
    print('Wrote repaired mesh to:', repaired_path)

except Exception as e:
    print('pymeshfix path failed:', e)
    print('Falling back to conservative PyVista cleanup and writing to repaired path')
    pvmesh = pvmesh.clean(tolerance=1e-6)
    pvmesh = pvmesh.extract_surface().triangulate()
    pvmesh.save(repaired_path)
    fixed_verts = np.asarray(pvmesh.points)
    fixed_faces = pvmesh.faces.reshape(-1, 4)[:, 1:]

# Quick TetGen test
print('Attempting quick TetGen test on repaired mesh...')
try:
    # Scale up slightly to help numerics
    scale = 10.0
    tg = tetgen.TetGen(fixed_verts * scale, fixed_faces)
    tetra_args = {'switches': 'pq1.8a0.3', 'plc': 1, 'verbose': 1}
    tg.tetrahedralize(**tetra_args)
    print('TetGen reported success; extracting grid...')
    grid = tg.grid
    print('TetGen grid points:', grid.number_of_points)
    print('TetGen grid cells:', grid.number_of_cells)
    print('TetGen test succeeded')
except Exception as e:
    print('TetGen test failed on repaired mesh:', e)
    # exit with non-zero so caller can inspect logs
    sys.exit(2)

print('Repair + test script finished successfully')
