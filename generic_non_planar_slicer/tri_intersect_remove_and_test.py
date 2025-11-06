#!/usr/bin/env python3
"""Detect triangle-triangle intersections using KD-tree broadphase + exact tests,
remove offending triangles, fill holes, clean, and attempt TetGen.
"""
import os
import sys
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
import tetgen

base_dir = os.path.dirname(__file__)
input_path = os.path.join(base_dir, 'output_models', 'propeller_remeshed_poisson.stl')
fixed_path = os.path.join(base_dir, 'output_models', 'propeller_remeshed_tri_fixed.stl')

print('Loading remeshed mesh:', input_path)
mesh = pv.read(input_path)
faces = mesh.faces.reshape(-1, 4)[:, 1:]
points = mesh.points
print('Verts, faces:', mesh.n_points, faces.shape[0])

# Precompute centroids and radii
tri_verts = points[faces]  # (n,3,3)
centroids = tri_verts.mean(axis=1)
radii = np.linalg.norm(tri_verts - centroids[:, None, :], axis=2).max(axis=1)

# Build KD-tree on centroids
tree = cKDTree(centroids)

# Helpers
EPS = 1e-9

def seg_tri_intersect(p0, p1, tri):
    # segment p0->p1 intersects triangle tri (3x3) (non-coplanar test)
    v0, v1, v2 = tri
    u = v1 - v0
    v = v2 - v0
    n = np.cross(u, v)
    dir = p1 - p0
    denom = np.dot(n, dir)
    if abs(denom) < 1e-12:
        return False
    t = np.dot(n, v0 - p0) / denom
    if t < -EPS or t > 1 + EPS:
        return False
    P = p0 + t * dir
    # barycentric
    w = P - v0
    uu = np.dot(u, u)
    uv = np.dot(u, v)
    vv = np.dot(v, v)
    wu = np.dot(w, u)
    wv = np.dot(w, v)
    D = uv * uv - uu * vv
    if abs(D) < 1e-15:
        return False
    s = (uv * wv - vv * wu) / D
    if s < -EPS or s > 1 + EPS:
        return False
    t2 = (uv * wu - uu * wv) / D
    if t2 < -EPS or (s + t2) > 1 + EPS:
        return False
    return True

# 2D helpers for coplanar case
def point_in_tri_2d(pt, tri2d):
    # barycentric in 2D
    a, b, c = tri2d
    v0 = c - a
    v1 = b - a
    v2 = pt - a
    den = v0[0]*v1[1] - v1[0]*v0[1]
    if abs(den) < 1e-12:
        return False
    u = (v2[0]*v1[1] - v1[0]*v2[1]) / den
    v = (v0[0]*v2[1] - v2[0]*v0[1]) / den
    return (u >= -EPS) and (v >= -EPS) and (u + v <= 1 + EPS)

def seg_seg_intersect_2d(p1, p2, q1, q2):
    # 2D segment intersection
    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    o1 = orient(p1,p2,q1)
    o2 = orient(p1,p2,q2)
    o3 = orient(q1,q2,p1)
    o4 = orient(q1,q2,p2)
    if (o1*o2 < 0) and (o3*o4 < 0):
        return True
    return False

def coplanar_tri_intersect(tri1, tri2):
    # project to best 2D plane
    normal = np.cross(tri1[1]-tri1[0], tri1[2]-tri1[0])
    ax = np.argmax(np.abs(normal))
    # drop axis ax
    idx = [0,1,2]
    idx.remove(ax)
    t1 = tri1[:, idx]
    t2 = tri2[:, idx]
    # check edges intersection
    edges1 = [(t1[0], t1[1]), (t1[1], t1[2]), (t1[2], t1[0])]
    edges2 = [(t2[0], t2[1]), (t2[1], t2[2]), (t2[2], t2[0])]
    for e1 in edges1:
        for e2 in edges2:
            if seg_seg_intersect_2d(e1[0], e1[1], e2[0], e2[1]):
                return True
    # check containment
    if point_in_tri_2d(t1[0], t2) or point_in_tri_2d(t2[0], t1):
        return True
    return False

# Candidate detection and exact test
n = len(faces)
problem = set()
for i in range(n):
    # query neighbors within radius sum
    neigh_idx = tree.query_ball_point(centroids[i], radii[i]*2.0)
    for j in neigh_idx:
        if j <= i:
            continue
        # skip shared vertices
        if set(faces[i]) & set(faces[j]):
            continue
        # AABB quick reject
        mins_i = tri_verts[i].min(axis=0)
        maxs_i = tri_verts[i].max(axis=0)
        mins_j = tri_verts[j].min(axis=0)
        maxs_j = tri_verts[j].max(axis=0)
        if np.any(maxs_i < mins_j - 1e-9) or np.any(maxs_j < mins_i - 1e-9):
            continue
        tri_i = tri_verts[i]
        tri_j = tri_verts[j]
        # plane normals
        n1 = np.cross(tri_i[1]-tri_i[0], tri_i[2]-tri_i[0])
        n2 = np.cross(tri_j[1]-tri_j[0], tri_j[2]-tri_j[0])
        n1n = np.linalg.norm(n1)
        n2n = np.linalg.norm(n2)
        if n1n < 1e-12 or n2n < 1e-12:
            continue
        n1u = n1 / n1n
        n2u = n2 / n2n
        if abs(np.dot(n1u, n2u)) > 0.9999:
            # nearly coplanar -> coplanar test
            # check plane distance
            d = abs(np.dot(n1u, tri_j[0]-tri_i[0]))
            if d < 1e-6:
                if coplanar_tri_intersect(tri_i, tri_j):
                    problem.add(i); problem.add(j)
                    continue
        # non-coplanar: check edges of i vs tri j and edges j vs tri i
        edges_i = [(tri_i[0], tri_i[1]), (tri_i[1], tri_i[2]), (tri_i[2], tri_i[0])]
        edges_j = [(tri_j[0], tri_j[1]), (tri_j[1], tri_j[2]), (tri_j[2], tri_j[0])]
        inter = False
        for (p0,p1) in edges_i:
            if seg_tri_intersect(p0, p1, tri_j):
                inter = True; break
        if not inter:
            for (p0,p1) in edges_j:
                if seg_tri_intersect(p0, p1, tri_i):
                    inter = True; break
        if inter:
            problem.add(i); problem.add(j)
    if i % 2000 == 0 and i > 0:
        print(f'Processed {i}/{n}, problems so far: {len(problem)}')

print('Total problematic triangles found:', len(problem))
if len(problem) == 0:
    print('No triangle-triangle intersections found.')
    sys.exit(0)

# Remove problem triangles and rebuild
mask = np.ones(n, dtype=bool)
mask[list(problem)] = False
kept_faces = faces[mask]
faces_pv = np.column_stack((np.full(len(kept_faces), 3), kept_faces)).flatten()
new_mesh = pv.PolyData(points, faces_pv)
new_mesh = new_mesh.clean(tolerance=1e-6)
new_mesh = new_mesh.extract_surface().triangulate()
try:
    new_mesh.fill_holes(hole_size=5000, inplace=True)
except Exception:
    try:
        new_mesh = new_mesh.fill_holes(hole_size=5000)
    except Exception:
        pass
new_mesh = new_mesh.clean(tolerance=1e-6)
print('After removal: verts, faces =', new_mesh.n_points, new_mesh.n_cells)
new_mesh.save(fixed_path)
print('Saved fixed mesh to', fixed_path)

# attempt TetGen
print('Attempting TetGen...')
try:
    verts2 = np.asarray(new_mesh.points)
    faces2 = new_mesh.faces.reshape(-1,4)[:,1:]
    tg = tetgen.TetGen(verts2*10.0, faces2)
    tg.tetrahedralize(**{'switches':'pq1.6a0.2','plc':1,'verbose':1})
    grid = tg.grid
    print('TetGen succeeded: points, cells =', grid.number_of_points, grid.number_of_cells)
except Exception as e:
    print('TetGen still failed:', e)
    sys.exit(2)

print('Done: triangle-intersection removal + TetGen succeeded')
