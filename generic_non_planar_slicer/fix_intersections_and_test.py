#!/usr/bin/env python3
"""Detect segment->facet intersections with VTK's OBBTree, remove offending triangles,
fill holes, clean mesh, and test TetGen tetrahedralization.
"""
import os
import sys
import numpy as np
import pyvista as pv
import tetgen
import vtk

base_dir = os.path.dirname(__file__)
input_path = os.path.join(base_dir, 'output_models', 'propeller_remeshed_poisson.stl')
fixed_path = os.path.join(base_dir, 'output_models', 'propeller_remeshed_fixed.stl')

print('Loading remeshed mesh:', input_path)
mesh = pv.read(input_path)
print('Verts, cells:', mesh.n_points, mesh.n_cells)

# Build vtk OBBTree for intersection queries
print('Building VTK polydata and OBBTree...')
# Prepare faces and points
faces = mesh.faces.reshape(-1, 4)[:, 1:]
points = mesh.points
# Construct vtkPolyData from points and faces (avoid PyVista internals)
vtk_points = vtk.vtkPoints()
for p in mesh.points:
    vtk_points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))

vtk_cells = vtk.vtkCellArray()
for tri in faces:
    triangle = vtk.vtkTriangle()
    triangle.GetPointIds().SetId(0, int(tri[0]))
    triangle.GetPointIds().SetId(1, int(tri[1]))
    triangle.GetPointIds().SetId(2, int(tri[2]))
    vtk_cells.InsertNextCell(triangle)

vtk_poly = vtk.vtkPolyData()
vtk_poly.SetPoints(vtk_points)
vtk_poly.SetPolys(vtk_cells)
vtk_poly.Modified()

obb = vtk.vtkOBBTree()
obb.SetDataSet(vtk_poly)
obb.BuildLocator()

faces = mesh.faces.reshape(-1, 4)[:, 1:]
points = mesh.points

problem_triangles = set()

# Helper: get neighbor face indices sharing a vertex
vertex_to_faces = {}
for fi, tri in enumerate(faces):
    for v in tri:
        vertex_to_faces.setdefault(int(v), []).append(fi)

# For progress
total = len(faces)
print('Scanning triangles for edge intersections (this may take a while)...')

for fi, tri in enumerate(faces):
    # triangle vertices
    a = points[tri[0]]
    b = points[tri[1]]
    c = points[tri[2]]
    edges = [(a, b), (b, c), (c, a)]
    for p0, p1 in edges:
        # Create VTK points to receive intersections
        points_out = vtk.vtkPoints()
        cell_ids = vtk.vtkIdList()
        # Use tolerance small
        code = obb.IntersectWithLine(p0, p1, points_out, cell_ids)
        if code == 0:
            continue
        # iterate returned cell ids
        for k in range(cell_ids.GetNumberOfIds()):
            other_cell = cell_ids.GetId(k)
            # skip if intersection is with own triangle
            if other_cell == fi:
                continue
            # skip if other_cell shares a vertex with this tri (adjacent)
            other_tri = faces[other_cell]
            if set(tri) & set(other_tri):
                continue
            # Check intersection point is not exactly at endpoints (shared vertices)
            # get point coordinate
            ipt = np.array(points_out.GetPoint(k))
            if np.allclose(ipt, p0) or np.allclose(ipt, p1):
                continue
            # Mark both triangles as problematic
            problem_triangles.add(fi)
            problem_triangles.add(other_cell)
    # progress print every 2000
    if fi % 2000 == 0 and fi > 0:
        print(f'Scanned {fi}/{total}, problem triangles so far: {len(problem_triangles)}')

print('Total problematic triangles found:', len(problem_triangles))
if len(problem_triangles) == 0:
    print('No segment->facet intersections detected by OBBTree. Exiting (nothing to fix).')
    sys.exit(0)

# Remove problematic triangles
mask = np.ones(len(faces), dtype=bool)
mask[list(problem_triangles)] = False
kept_faces = faces[mask]
print('Keeping', kept_faces.shape[0], 'faces out of', faces.shape[0])

# Build new mesh
faces_pv = np.column_stack((np.full(len(kept_faces), 3), kept_faces)).flatten()
new_mesh = pv.PolyData(mesh.points, faces_pv)

# Clean and fill holes
print('Cleaning and filling holes...')
new_mesh = new_mesh.clean(tolerance=1e-6)
new_mesh = new_mesh.extract_surface().triangulate()
# attempt to fill holes (large hole_size may close large openings created by removals)
try:
    new_mesh.fill_holes(hole_size=5000, inplace=True)
except Exception:
    # Some versions return new mesh
    try:
        new_mesh = new_mesh.fill_holes(hole_size=5000)
    except Exception:
        pass

new_mesh = new_mesh.clean(tolerance=1e-6)

print('Final mesh verts, cells:', new_mesh.n_points, new_mesh.n_cells)
print('Saving fixed mesh to:', fixed_path)
new_mesh.save(fixed_path)

# Try TetGen
print('Attempting TetGen tetrahedralization on fixed mesh...')
try:
    verts = np.asarray(new_mesh.points)
    faces = new_mesh.faces.reshape(-1, 4)[:, 1:]
    tg = tetgen.TetGen(verts * 10.0, faces)
    tetra_args = {'switches': 'pq1.6a0.2', 'plc': 1, 'verbose': 1}
    tg.tetrahedralize(**tetra_args)
    grid = tg.grid
    print('TetGen success: points, cells =', grid.number_of_points, grid.number_of_cells)
except Exception as e:
    print('TetGen failed after automated intersection removal:', e)
    sys.exit(2)

print('Automated fix + TetGen succeeded')
