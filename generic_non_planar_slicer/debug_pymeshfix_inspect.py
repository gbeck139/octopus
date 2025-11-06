#!/usr/bin/env python3
"""Diagnostic script to probe pymeshfix MeshFix internals on the propeller mesh.
Tries multiple MeshFix usage patterns and prints attributes/types so we can extract repaired arrays.
"""
import os
import sys
import traceback
import numpy as np
import pyvista as pv

base_dir = os.path.dirname(__file__)
mesh_name = 'propeller'
input_path = os.path.join(base_dir, 'input_models', f'{mesh_name}.stl')
repaired_out = os.path.join(base_dir, 'output_models', f'{mesh_name}_pymeshfix_inspect_repaired.stl')

print('Loading:', input_path)
pvmesh = pv.read(input_path)
verts = np.asarray(pvmesh.points)
faces = pvmesh.faces.reshape(-1, 4)[:, 1:]
print('Loaded verts, faces:', verts.shape, faces.shape)

try:
    from pymeshfix import MeshFix
    import pymeshfix
    print('pymeshfix version:', getattr(pymeshfix, '__version__', 'unknown'))
except Exception as e:
    print('Failed to import pymeshfix:', e)
    sys.exit(1)

# Helper to print summary
def summarize(name, obj):
    try:
        print('\n===', name, 'summary ===')
        print('type:', type(obj))
        if isinstance(obj, np.ndarray):
            print('ndarray shape:', obj.shape, 'dtype:', obj.dtype)
        else:
            # try length and some attrs
            if hasattr(obj, 'vertices') and hasattr(obj, 'faces'):
                try:
                    import trimesh
                    print('trimesh: vertices shape', np.asarray(obj.vertices).shape, 'faces shape', np.asarray(obj.faces).shape)
                except Exception:
                    pass
            if hasattr(obj, '__dict__'):
                keys = list(obj.__dict__.keys())
                print('attrs:', keys)
    except Exception as e:
        print('Error summarizing', name, e)

# Try 1: trimesh path
try:
    import trimesh
    print('\nAttempting trimesh path...')
    tmesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    summarize('trimesh before', tmesh)
    mf = MeshFix(tmesh)
    print('Created MeshFix(trimesh) ->', type(mf))
    summarize('MeshFix object (after init)', mf)
    print('Calling mf.repair(verbose=True)')
    mf.repair(verbose=True)
    print('repair done')
    summarize('mf.mesh', getattr(mf, 'mesh', None))
    # probe various attributes
    for attr in ['v','f','points','vertices','faces','mesh']:
        val = getattr(mf, attr, None)
        print(f"mf.{attr} ->", type(val))
        if isinstance(val, (list, tuple, np.ndarray)):
            try:
                arr = np.asarray(val)
                print('  shape:', arr.shape)
            except Exception as e:
                print('  could not convert to array:', e)
    # if mesh available
    repaired = getattr(mf, 'mesh', None)
    if repaired is not None and hasattr(repaired, 'vertices'):
        r_verts = np.asarray(repaired.vertices)
        r_faces = np.asarray(repaired.faces)
        print('Repaired via trimesh path arrays:', r_verts.shape, r_faces.shape)
        # write via pyvista
        out_pv = pv.PolyData(r_verts, np.column_stack((np.full(len(r_faces), 3), r_faces)).flatten())
        out_pv.save(repaired_out)
        print('Wrote repaired mesh to', repaired_out)
        sys.exit(0)
    else:
        print('No mf.mesh found after repair (trimesh path)')
except Exception:
    print('Trimesh path failed:')
    traceback.print_exc()

# Try 2: MeshFix on arrays (older/newer APIs differ)
try:
    print('\nAttempting MeshFix with raw arrays...')
    mf2 = MeshFix(verts, faces)
    print('Created MeshFix(arrays) ->', type(mf2))
    summarize('mf2', mf2)
    print('Calling mf2.repair(verbose=True)')
    mf2.repair(verbose=True)
    print('repair done')
    for attr in ['v','f','points','vertices','faces','mesh']:
        val = getattr(mf2, attr, None)
        print(f"mf2.{attr} ->", type(val))
        try:
            if isinstance(val, (list, tuple, np.ndarray)):
                arr = np.asarray(val)
                print('  shape:', arr.shape)
        except Exception as e:
            print('  could not convert to array:', e)
    # try mesh attribute
    repaired2 = getattr(mf2, 'mesh', None)
    if repaired2 is not None:
        try:
            rv = np.asarray(getattr(repaired2, 'vertices', None) or getattr(repaired2, 'v', None) or getattr(repaired2, 'points', None))
            rf = np.asarray(getattr(repaired2, 'faces', None) or getattr(repaired2, 'f', None))
            print('Repaired2 mesh arrays shapes:', rv.shape, rf.shape)
            out_pv = pv.PolyData(rv, np.column_stack((np.full(len(rf), 3), rf)).flatten())
            out_pv.save(repaired_out)
            print('Wrote repaired mesh to', repaired_out)
            sys.exit(0)
        except Exception:
            traceback.print_exc()
    # fallback: maybe mf2.v and mf2.f
    v = getattr(mf2, 'v', None) or getattr(mf2, 'points', None) or getattr(mf2, 'vertices', None)
    f = getattr(mf2, 'f', None) or getattr(mf2, 'faces', None)
    print('mf2 v type', type(v), 'f type', type(f))
    try:
        v = np.asarray(v)
        f = np.asarray(f)
        print('converted shapes', v.shape, f.shape)
        if f.ndim == 2 and f.shape[1] == 3:
            out_pv = pv.PolyData(v, np.column_stack((np.full(len(f), 3), f)).flatten())
            out_pv.save(repaired_out)
            print('Wrote repaired mesh to', repaired_out)
            sys.exit(0)
    except Exception:
        traceback.print_exc()

except Exception:
    print('MeshFix(arrays) path failed:')
    traceback.print_exc()

print('\nAll MeshFix usage attempts failed to produce a clearly extractable repaired mesh.')
print('Saved nothing. You can inspect the original and the previous repaired file at output_models/')
sys.exit(3)
