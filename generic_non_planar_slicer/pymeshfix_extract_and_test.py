#!/usr/bin/env python3
"""Run MeshFix on the propeller using the array API, extract mf.v/mf.f, save, and test TetGen.
"""
import os
import sys
import numpy as np
import pyvista as pv
import tetgen

base_dir = os.path.dirname(__file__)
mesh_name = 'propeller'
input_path = os.path.join(base_dir, 'input_models', f'{mesh_name}.stl')
repaired_path = os.path.join(base_dir, 'output_models', f'{mesh_name}_pymeshfix_extracted.stl')

print('Loading original mesh:', input_path)
pvmesh = pv.read(input_path)
verts = np.asarray(pvmesh.points)
faces = pvmesh.faces.reshape(-1, 4)[:, 1:]
print('Original vertices, faces:', verts.shape, faces.shape)

from pymeshfix import MeshFix

print('Running MeshFix with array API...')
mf = MeshFix(verts, faces)
mf.repair(verbose=True)

# After repair, MeshFix exposes numpy arrays on mf.v and mf.f (observed)
if hasattr(mf, 'v') and hasattr(mf, 'f'):
    v = np.asarray(mf.v)
    f = np.asarray(mf.f)
    print('Extracted v,f shapes:', v.shape, f.shape)
else:
    # try points/faces
    v = np.asarray(getattr(mf, 'points', getattr(mf, 'vertices', None)))
    f = np.asarray(getattr(mf, 'faces', getattr(mf, 'f', None)))
    print('Fallback extracted shapes (may be None):', None if v is None else v.shape, None if f is None else f.shape)

if v is None or f is None:
    print('Failed to extract arrays from MeshFix object')
    sys.exit(2)

# Save repaired mesh
out_pv = pv.PolyData(v, np.column_stack((np.full(len(f), 3), f)).flatten())
out_pv.save(repaired_path)
print('Wrote repaired mesh to:', repaired_path)

# Quick TetGen test
print('Attempting TetGen test...')
try:
    scale = 10.0
    tg = tetgen.TetGen(v * scale, f)
    tetra_args = {'switches': 'pq1.8a0.3', 'plc': 1, 'verbose': 1}
    tg.tetrahedralize(**tetra_args)
    grid = tg.grid
    print('TetGen grid points:', grid.number_of_points)
    print('TetGen grid cells:', grid.number_of_cells)
    print('TetGen test succeeded')
except Exception as e:
    print('TetGen test failed:', e)
    sys.exit(3)

print('Done')
