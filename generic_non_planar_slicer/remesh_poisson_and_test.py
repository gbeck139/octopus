#!/usr/bin/env python3
"""Remesh the propeller using Open3D Poisson reconstruction from a downsampled
point cloud, clean the mesh, save it, and attempt a TetGen tetrahedralization test.
"""
import os
import sys
import numpy as np
import pyvista as pv
import open3d as o3d
import tetgen

base_dir = os.path.dirname(__file__)
mesh_name = 'propeller'
input_path = os.path.join(base_dir, 'input_models', f'{mesh_name}.stl')
remeshed_path = os.path.join(base_dir, 'output_models', f'{mesh_name}_remeshed_poisson.stl')

print('Loading original mesh:', input_path)
pvmesh = pv.read(input_path)
verts = np.asarray(pvmesh.points)
faces = pvmesh.faces.reshape(-1, 4)[:, 1:]
print('Original vertices, faces:', verts.shape, faces.shape)

# Convert to Open3D mesh
o3d_mesh = o3d.geometry.TriangleMesh()
o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

# Pre-clean
print('Pre-cleaning Open3D mesh...')
o3d_mesh.remove_duplicated_vertices()
o3d_mesh.remove_duplicated_triangles()
o3d_mesh.remove_degenerate_triangles()
o3d_mesh.compute_vertex_normals()

# Create point cloud from mesh and downsample
print('Sampling points from mesh (uniform sampling) and downsampling...')
pcd = o3d_mesh.sample_points_uniformly(number_of_points=200000)
print('Sampled points:', np.asarray(pcd.points).shape)

# Voxel downsample to remove dense coincident points
voxel_size = max(o3d_mesh.get_max_bound() - o3d_mesh.get_min_bound()) * 0.001
print('Voxel size for downsample:', voxel_size)
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
print('After voxel downsample:', np.asarray(pcd.points).shape)

# Estimate normals
print('Estimating normals for point cloud...')
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=50))
pcd.orient_normals_consistent_tangent_plane(100)

# Poisson reconstruction
print('Running Poisson reconstruction (depth=8)...')
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Warning) as cm:
    mesh_recon, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

print('Reconstructed mesh:', np.asarray(mesh_recon.vertices).shape, np.asarray(mesh_recon.triangles).shape)

# Crop the reconstructed mesh to the original bounding box to remove outer artifacts
bbox = o3d_mesh.get_axis_aligned_bounding_box()
mesh_crop = mesh_recon.crop(bbox)
print('Cropped mesh:', np.asarray(mesh_crop.vertices).shape, np.asarray(mesh_crop.triangles).shape)

# Clean
print('Cleaning reconstructed mesh...')
mesh_crop.remove_degenerate_triangles()
mesh_crop.remove_duplicated_triangles()
mesh_crop.remove_duplicated_vertices()
mesh_crop.remove_non_manifold_edges()
mesh_crop.compute_vertex_normals()

# Save remeshed mesh via PyVista for consistency
rv = np.asarray(mesh_crop.vertices)
rf = np.asarray(mesh_crop.triangles)
print('Final remeshed verts, faces:', rv.shape, rf.shape)
if len(rv) == 0 or len(rf) == 0:
    print('Remeshing produced empty mesh — aborting')
    sys.exit(2)

out_pv = pv.PolyData(rv, np.column_stack((np.full(len(rf), 3), rf)).flatten())
out_pv.save(remeshed_path)
print('Saved remeshed mesh to:', remeshed_path)

# Quick TetGen test
print('Attempting TetGen on remeshed mesh...')
try:
    scale = 10.0
    tg = tetgen.TetGen(rv * scale, rf)
    tetra_args = {'switches': 'pq1.6a0.2', 'plc': 1, 'verbose': 1}
    tg.tetrahedralize(**tetra_args)
    grid = tg.grid
    print('TetGen grid points:', grid.number_of_points)
    print('TetGen grid cells:', grid.number_of_cells)
    print('TetGen test succeeded on remeshed mesh')
except Exception as e:
    print('TetGen test failed:', e)
    sys.exit(3)

print('Remesh + TetGen test complete')
