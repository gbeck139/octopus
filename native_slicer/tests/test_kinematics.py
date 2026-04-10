"""Tests for CX-ZB kinematics and singularity handling."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from cxzb_slicer.kinematics import (
    CXZBKinematics,
    apply_forbidden_cone,
    detect_singularity,
    machine_to_world,
    tcp_compensation,
    unwrap_c_axis,
    world_to_machine,
)


@pytest.mark.parametrize("L", [10.0, 50.0])
def test_roundtrip_world_machine_world_random_points(L: float) -> None:
    """World → machine → world round-trip is numerically stable.

    For random world-frame TCP positions and B-axis tilts, verify that
    :func:`world_to_machine` and :func:`machine_to_world` form an
    approximate inverse pair up to numerical precision.
    """

    rng = np.random.default_rng(seed=1234)
    n = 1000

    # Sample a cylindrical work volume around the C-axis.
    r = rng.uniform(5.0, 150.0, size=n)
    phi = rng.uniform(-np.pi, np.pi, size=n)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = rng.uniform(0.0, 200.0, size=n)

    # Avoid the exact B=0 singularity by staying comfortably within limits.
    beta = rng.uniform(np.deg2rad(-60.0), np.deg2rad(60.0), size=n)

    x_m, z_m, c = world_to_machine(x, y, z, beta, L)
    x_rt, y_rt, z_rt = machine_to_world(x_m, z_m, c, beta, L)

    npt.assert_allclose(x_rt, x, rtol=1e-12, atol=1e-9)
    npt.assert_allclose(y_rt, y, rtol=1e-12, atol=1e-9)
    npt.assert_allclose(z_rt, z, rtol=1e-12, atol=1e-9)


def test_tcp_compensation_matches_closed_form() -> None:
    """RTCP offsets satisfy ΔX = L·sinβ and ΔZ = L·(cosβ − 1)."""

    L = 42.0
    beta = np.linspace(-np.deg2rad(80.0), np.deg2rad(80.0), 200)

    dx, dz = tcp_compensation(beta, L)

    dx_expected = L * np.sin(beta)
    dz_expected = L * (np.cos(beta) - 1.0)

    npt.assert_allclose(dx, dx_expected, rtol=1e-13, atol=1e-12)
    npt.assert_allclose(dz, dz_expected, rtol=1e-13, atol=1e-12)


def test_singularity_detection_and_forbidden_cone() -> None:
    """B ≈ 0 is detected as singular and projected outside the forbidden cone."""

    b_min = np.deg2rad(5.0)
    beta = np.array([-0.5 * b_min, 0.0, 0.5 * b_min, 2.0 * b_min])

    singular_mask = detect_singularity(beta, b_min)
    assert singular_mask.tolist() == [True, True, True, False]

    beta_projected = apply_forbidden_cone(beta, b_min)
    assert np.all(np.abs(beta_projected[:3]) >= b_min - 1e-12)
    # Values already outside the cone should remain unchanged.
    npt.assert_allclose(beta_projected[-1], beta[-1])


def test_c_axis_unwrap_continuity() -> None:
    """C-axis unwrapping removes 2π jumps and preserves local continuity."""

    raw_deg = np.array([0.0, 90.0, 170.0, -170.0, -90.0, 0.0])
    raw = np.deg2rad(raw_deg)

    unwrapped = unwrap_c_axis(raw)
    # Successive differences should all be smaller than π in magnitude.
    diffs = np.diff(unwrapped)
    assert np.all(np.abs(diffs) < np.pi)


def test_class_interface_matches_functional_implementation() -> None:
    """CXZBKinematics class delegates correctly to functional API."""

    L = 30.0
    kin = CXZBKinematics(tool_offset_L=L)

    x = np.array([10.0, 20.0, -15.0])
    y = np.array([5.0, -10.0, 25.0])
    z = np.array([0.0, 50.0, 100.0])
    beta = np.deg2rad(np.array([10.0, -20.0, 30.0]))

    x_m_f, z_m_f, c_f = world_to_machine(x, y, z, beta, L)
    x_m_c, z_m_c, c_c = kin.world_to_machine(x, y, z, beta)

    npt.assert_allclose(x_m_c, x_m_f)
    npt.assert_allclose(z_m_c, z_m_f)
    npt.assert_allclose(c_c, c_f)

