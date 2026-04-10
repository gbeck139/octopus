"""Tests for Phase 4 vector field solver."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import trimesh

from cxzb_slicer.core.types import SlicerConfig
from cxzb_slicer.vector_field.constraints import SphereRadialConstraints
from cxzb_slicer.vector_field.field_solver import PoissonVectorFieldSolver, vector_to_angles
from cxzb_slicer.vector_field.smoothing import enforce_angular_velocity_limits_along_path


def test_vector_field_on_sphere_is_radial_like() -> None:
    """On a sphere with radial guidance, k should align with position vectors."""

    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    cfg = SlicerConfig()
    constraints = SphereRadialConstraints(center=np.zeros(3, dtype=np.float64), fix_caps=False)
    g, fixed = constraints.guidance_and_mask(verts, faces, cfg)

    solver = PoissonVectorFieldSolver()
    k = solver.solve(verts, faces, guidance=g, fixed_mask=fixed, config=cfg, s=1e3)

    vhat = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    dots = np.einsum("ij,ij->i", k, vhat)
    assert float(np.mean(dots)) > 0.95


def test_vector_to_angles_and_limits() -> None:
    """Angle extraction works and angular velocity limiter enforces bounds."""

    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    cfg = SlicerConfig(max_b_omega_deg_s=10.0, max_c_omega_deg_s=10.0)
    g, fixed = SphereRadialConstraints().guidance_and_mask(verts, faces, cfg)
    k = PoissonVectorFieldSolver().solve(verts, faces, guidance=g, fixed_mask=fixed, config=cfg, s=1e2)

    b, c = vector_to_angles(k)
    assert b.shape == (verts.shape[0],)
    assert c.shape == (verts.shape[0],)

    # B is a tilt magnitude from vertical in [0, pi].
    assert np.all((b >= 0.0) & (b <= np.pi))

    # Create a simple path order (by z then x).
    order = np.lexsort((verts[:, 0], verts[:, 2]))
    pos = verts[order]
    b_path = b[order]
    c_path = c[order]

    b2, c2 = enforce_angular_velocity_limits_along_path(
        positions=pos,
        b=b_path,
        c=c_path,
        feed_mm_s=50.0,
        max_b_omega_deg_s=cfg.max_b_omega_deg_s,
        max_c_omega_deg_s=cfg.max_c_omega_deg_s,
    )

    # Verify per-segment bounds.
    ds = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    dt = ds / 50.0
    max_db = np.deg2rad(cfg.max_b_omega_deg_s) * dt + 1e-12
    max_dc = np.deg2rad(cfg.max_c_omega_deg_s) * dt + 1e-12

    db = np.abs(np.diff(b2))
    dc = np.abs(np.diff(c2))
    assert np.all(db <= max_db + 1e-9)
    assert np.all(dc <= max_dc + 1e-9)

