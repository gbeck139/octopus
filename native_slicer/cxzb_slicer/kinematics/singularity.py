"""
Singularity handling utilities for CX-ZB rotary axes.

The CX-ZB kinematic has a well-known singularity when the B-axis angle
approaches zero: the tool direction becomes vertical and the C-axis
rotation no longer changes the orientation (any C value is equivalent).
The research document notes that the Jacobian magnitude

.. math::

    \\left\\lVert \\partial k / \\partial C \\right\\rVert = |\\sin B|

vanishes at :math:`B = 0`, implying unbounded C-axis velocity to realise
finite orientation changes. Two practical mitigations are implemented:

1. **Forbidden cone:** enforce a minimum tilt :math:`|B| \\geq B_{min}` by
   projecting near-vertical orientations onto the cone boundary.
2. **C-axis unwrapping:** maintain a continuous C-axis trajectory by
   removing :math:`2\\pi` jumps using :func:`numpy.unwrap`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

ArrayLike = npt.ArrayLike


class AbstractSingularityHandler(ABC):
    """Abstract interface for singularity handling strategies."""

    @abstractmethod
    def unwrap_c_axis(self, c_angles: ArrayLike) -> npt.NDArray[np.float64]:
        """Return a continuous version of the C-axis angle sequence."""

    @abstractmethod
    def apply_forbidden_cone(
        self,
        beta: ArrayLike,
        b_min: float,
    ) -> npt.NDArray[np.float64]:
        """Project B-axis angles away from the singular region near ``0``."""


def unwrap_c_axis(c_angles: ArrayLike, period: float = 2.0 * np.pi) -> npt.NDArray[np.float64]:
    """Unwrap C-axis angles to produce a continuous trajectory.

    Parameters
    ----------
    c_angles:
        Sequence of C-axis angles in radians.
    period:
        Angular period, default :math:`2\\pi`. Passed to
        :func:`numpy.unwrap` via the ``discont`` parameter.

    Returns
    -------
    numpy.ndarray
        Unwrapped angles where jumps greater than ``period / 2`` have
        been corrected by adding or subtracting integer multiples of
        ``period``.
    """

    c_arr = np.asarray(c_angles, dtype=np.float64)
    return np.unwrap(c_arr, discont=period / 2.0)


def apply_forbidden_cone(
    beta: ArrayLike,
    b_min: float,
) -> npt.NDArray[np.float64]:
    """Enforce a minimum tilt ``|B| ≥ B_min`` to avoid the pole singularity.

    The B-axis singularity arises when :math:`B \\approx 0`, where the
    C-axis becomes undefined. A simple and robust mitigation is the
    *forbidden cone*: project any angle with :math:`|B| < B_{min}` onto
    the boundary of the cone while preserving its sign:

    .. math::

        B' = \\operatorname{sign}(B)\\,B_{min}, \\quad
        \\text{for } |B| < B_{min}.

    Parameters
    ----------
    beta:
        Input B-axis angles in radians.
    b_min:
        Minimum magnitude in radians. Typical values are 2–5 degrees
        (:math:`\\approx 0.035`–:math:`0.087` rad).
    """

    beta_arr = np.asarray(beta, dtype=np.float64)
    abs_beta = np.abs(beta_arr)
    sign = np.where(beta_arr >= 0.0, 1.0, -1.0)

    # For exactly zero, arbitrarily pick the positive side of the cone.
    sign = np.where(abs_beta == 0.0, 1.0, sign)

    projected = sign * float(b_min)
    result = np.where(abs_beta < float(b_min), projected, beta_arr)
    return result


def detect_singularity(beta: ArrayLike, b_min: float) -> npt.NDArray[np.bool_]:
    """Return a boolean mask of points inside the forbidden cone.

    This helper is primarily for diagnostics and testing: it identifies
    where the B-axis angle lies in the near-vertical region
    :math:`|B| < B_{min}` where the C-axis pole singularity occurs.
    """

    beta_arr = np.asarray(beta, dtype=np.float64)
    return np.abs(beta_arr) < float(b_min)

