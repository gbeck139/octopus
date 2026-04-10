"""
CX-ZB inverse kinematics with RTCP compensation.

The kinematic model follows the research note *Building a native 4-axis
slicer for CX-ZB kinematics* and is characterised by:

- C-axis: rotation of the bed around the world Z-axis by angle ``C``.
- B-axis: tilt of the head around the world Y-axis by angle ``β``.
- X/Z: linear gantry axes in the machine frame.
- Tool offset: distance ``L`` from the B-axis pivot to the nozzle tip
  along the negative Z-axis when ``β = 0``.

The core closed-form equations implemented here are:

.. math::

    C &= \\operatorname{atan2}(-y_w, x_w),\\\\
    r &= \\sqrt{x_w^2 + y_w^2},\\\\
    X_m &= r + L\\sin \\beta,\\\\
    Z_m &= z_w + L\\cos \\beta,

for inverse kinematics (world to machine), and

.. math::

    x_w &= (X_m - L\\sin \\beta)\\cos C,\\\\
    y_w &= -(X_m - L\\sin \\beta)\\sin C,\\\\
    z_w &= Z_m - L\\cos \\beta,

for forward kinematics (machine to world). The RTCP displacement of the
tool centre point (TCP) relative to the pivot is

.. math::

    \\Delta X = L\\sin \\beta,\\quad
    \\Delta Z = L(\\cos \\beta - 1),

which is used by :func:`tcp_compensation`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

ArrayLike = npt.ArrayLike


def _as_float_array(*arrays: ArrayLike) -> list[npt.NDArray[np.float64]]:
    """Convert inputs to broadcastable ``float64`` NumPy arrays."""

    converted = [np.asarray(a, dtype=np.float64) for a in arrays]
    broadcasted = np.broadcast_arrays(*converted)
    return [np.asarray(a, dtype=np.float64) for a in broadcasted]


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class AbstractCXZBKinematics(ABC):
    """Abstract base class for CX-ZB kinematics solvers.

    Concrete implementations must provide RTCP-aware mappings between
    world-frame TCP coordinates and machine coordinates.
    """

    @abstractmethod
    def world_to_machine(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        beta: ArrayLike,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Map world-frame TCP coordinates to machine coordinates.

        Implements the closed-form inverse kinematics

        .. math::

            C &= \\operatorname{atan2}(-y_w, x_w),\\\\
            r &= \\sqrt{x_w^2 + y_w^2},\\\\
            X_m &= r + L\\sin \\beta,\\\\
            Z_m &= z_w + L\\cos \\beta.
        """

    @abstractmethod
    def machine_to_world(
        self,
        x_m: ArrayLike,
        z_m: ArrayLike,
        c_angle: ArrayLike,
        beta: ArrayLike,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Map machine coordinates back to world-frame TCP coordinates.

        Implements the forward kinematics

        .. math::

            x_w &= (X_m - L\\sin \\beta)\\cos C,\\\\
            y_w &= -(X_m - L\\sin \\beta)\\sin C,\\\\
            z_w &= Z_m - L\\cos \\beta.
        """

    @abstractmethod
    def tcp_compensation(
        self,
        beta: ArrayLike,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return RTCP compensation offsets ``(ΔX, ΔZ)`` for a tilt ``β``."""


class CXZBKinematics(AbstractCXZBKinematics):
    """Concrete CX-ZB kinematics implementation parameterised by ``L``."""

    def __init__(self, tool_offset_L: float) -> None:
        self.tool_offset_L = float(tool_offset_L)

    def world_to_machine(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        beta: ArrayLike,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return world_to_machine(x, y, z, beta, self.tool_offset_L)

    def machine_to_world(
        self,
        x_m: ArrayLike,
        z_m: ArrayLike,
        c_angle: ArrayLike,
        beta: ArrayLike,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return machine_to_world(x_m, z_m, c_angle, beta, self.tool_offset_L)

    def tcp_compensation(
        self,
        beta: ArrayLike,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return tcp_compensation(beta, self.tool_offset_L)


# ---------------------------------------------------------------------------
# Functional interface
# ---------------------------------------------------------------------------

def world_to_machine(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    beta: ArrayLike,
    L: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Inverse kinematics: world to machine coordinates.

    Given world-frame TCP coordinates ``(x_w, y_w, z_w)`` and a desired
    B-axis tilt ``β``, compute machine coordinates ``(X_m, Z_m, C)`` for
    the CX-ZB kinematic with tool offset ``L``.

    The formulas implemented are (see the research document, Eq. 71–75):

    .. math::

        C  &= \\operatorname{atan2}(-y_w, x_w),\\\\
        r  &= \\sqrt{x_w^2 + y_w^2},\\\\
        X_m &= r + L\\sin \\beta,\\\\
        Z_m &= z_w + L\\cos \\beta.

    All inputs are broadcast using NumPy's standard broadcasting rules,
    and the outputs are ``float64`` arrays of the broadcast shape.
    """

    x_arr, y_arr, z_arr, beta_arr = _as_float_array(x, y, z, beta)
    L_f = float(L)

    c_angle = np.arctan2(-y_arr, x_arr)
    r = np.sqrt(x_arr**2 + y_arr**2)
    x_m = r + L_f * np.sin(beta_arr)
    z_m = z_arr + L_f * np.cos(beta_arr)

    return x_m, z_m, c_angle


def machine_to_world(
    x_m: ArrayLike,
    z_m: ArrayLike,
    c_angle: ArrayLike,
    beta: ArrayLike,
    L: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Forward kinematics: machine to world coordinates.

    Given machine coordinates ``(X_m, Z_m, C)`` and B-axis tilt ``β``,
    recover the world-frame TCP coordinates ``(x_w, y_w, z_w)`` using
    the closed-form forward kinematics

    .. math::

        x_w &= (X_m - L\\sin \\beta)\\cos C,\\\\
        y_w &= -(X_m - L\\sin \\beta)\\sin C,\\\\
        z_w &= Z_m - L\\cos \\beta.

    This is the exact inverse of :func:`world_to_machine` and is used
    primarily for verification and simulation.
    """

    x_m_arr, z_m_arr, c_arr, beta_arr = _as_float_array(x_m, z_m, c_angle, beta)
    L_f = float(L)

    r_eff = x_m_arr - L_f * np.sin(beta_arr)
    cos_c = np.cos(c_arr)
    sin_c = np.sin(c_arr)

    x_w = r_eff * cos_c
    y_w = -r_eff * sin_c
    z_w = z_m_arr - L_f * np.cos(beta_arr)

    return x_w, y_w, z_w


def tcp_compensation(
    beta: ArrayLike,
    L: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return RTCP gantry compensation ``(ΔX, ΔZ)`` for tilt ``β``.

    When the B-axis tilts by angle ``β`` around the Y-axis while the
    pivot remains fixed, the nozzle tip would move by

    .. math::

        \\Delta X_{tcp} &= -L\\sin \\beta,\\\\
        \\Delta Z_{tcp} &= L(1 - \\cos \\beta),

    relative to its position at ``β = 0``. To keep the TCP stationary,
    the gantry must move in the opposite direction, yielding the
    compensation

    .. math::

        \\Delta X &= L\\sin \\beta,\\\\
        \\Delta Z &= L(\\cos \\beta - 1).

    This function returns these compensation values as NumPy arrays.
    """

    (beta_arr,) = _as_float_array(beta)
    L_f = float(L)

    delta_x = L_f * np.sin(beta_arr)
    delta_z = L_f * (np.cos(beta_arr) - 1.0)
    return delta_x, delta_z

