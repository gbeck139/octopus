"""Tests for extrusion compensation and nominal extrusion."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt

from cxzb_slicer.core.types import SlicerConfig, TOOLPATH_DTYPE
from cxzb_slicer.extrusion.compensator import TiltExtrusionCompensator, nominal_extrusion_mm_filament, tilt_compensate


def test_tilt_compensation_formula_and_clamp() -> None:
    """E_adj = E_nom / max(cosβ, 0.1) vectorized."""

    e_nom = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    beta = np.array([0.0, np.deg2rad(60.0), np.deg2rad(89.0)], dtype=np.float64)

    e_adj = tilt_compensate(e_nom, beta, min_cos_beta=0.1)
    expected = e_nom / np.maximum(np.cos(beta), 0.1)
    npt.assert_allclose(e_adj, expected)


def test_nominal_extrusion_volume_conservation_beta0() -> None:
    """For β=0, nominal extrusion equals bead volume / filament area."""

    cfg = SlicerConfig(layer_height=0.2, extrusion_width=0.4, filament_diameter=1.75)
    ds = np.array([10.0], dtype=np.float64)  # 10mm segment

    e_nom = nominal_extrusion_mm_filament(ds, cfg)

    v = cfg.extrusion_width * cfg.layer_height * ds[0]
    a_f = np.pi * (0.5 * cfg.filament_diameter) ** 2
    expected = v / a_f
    npt.assert_allclose(e_nom[0], expected)


def test_compensator_sets_extrude_only() -> None:
    """Compensator returns zero E for travel moves and first point."""

    cfg = SlicerConfig()
    tp = np.zeros(3, dtype=TOOLPATH_DTYPE)
    tp["x"] = [0.0, 1.0, 2.0]
    tp["y"] = 0.0
    tp["z"] = 0.0
    tp["move_type"] = ["extrude", "travel", "extrude"]
    tp["tilt_angle"] = 0.0

    e = TiltExtrusionCompensator().compute_e_per_point(tp, cfg)
    assert e.shape == (3,)
    assert e[0] == 0.0
    assert e[1] == 0.0
    assert e[2] > 0.0

