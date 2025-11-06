#!/usr/bin/env python3
import os
import numpy as np
import pyvista as pv
import tetgen

base_dir = os.path.dirname(__file__)
input_path = os.path.join(base_dir, 'output_models', 'propeller_remeshed_poisson.stl')
jitter_path = os.path.join(base_dir, 'output_models', 'propeller_remeshed_poisson_jittered.stl')

print('Loading', input_path)
mesh = pv.read(input_path)
pts = mesh.points.copy()
# scale jitter by bbox diagonal
bbox = mesh.bounds
diag = np.linalg.norm([bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4]])
if diag == 0:
    diag = 1.0
sigma = diag * 1e-7  # very small jitter
np.random.seed(42)
pts += np.random.normal(scale=sigma, size=pts.shape)
mesh.points = pts
mesh.save(jitter_path)
print('Saved jittered mesh to', jitter_path, 'sigma=', sigma)

# Try TetGen
faces = mesh.faces.reshape(-1,4)[:,1:]
verts = np.asarray(mesh.points)
try:
    tg = tetgen.TetGen(verts*10.0, faces)
    tg.tetrahedralize(**{'switches':'pq1.6a0.2','plc':1,'verbose':1})
    grid = tg.grid
    print('TetGen succeeded on jittered mesh: points, cells =', grid.number_of_points, grid.number_of_cells)
except Exception as e:
    print('TetGen failed on jittered mesh:', e)
    raise
