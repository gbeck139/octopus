"""
Extrusion computation and tilt-angle compensation.

Nominal extrusion is computed from deposited bead volume:

.. math::

    V = w\\,h\\,ds,

where :math:`w` is extrusion width, :math:`h` is layer height, and
:math:`ds` is segment length. Converting volume to **mm of filament**
uses the filament cross-section area

.. math::

    A_f = \\pi \\left(\\frac{d_f}{2}\\right)^2,

so the nominal filament length is

.. math::

    E_{nom} = \\frac{V}{A_f}.

Tilt compensation follows the research note:

.. math::

    E_{adj} = \\frac{E_{nom}}{\\max(\\cos\\beta,\\ c_{min})},

with :math:`c_{min}=0.1` by default to avoid runaway extrusion for
extreme tilts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from cxzb_slicer.core.types import SlicerConfig

nparray_f64 = npt.NDArray[np.float64]


def nominal_extrusion_mm_filament(ds_mm: nparray_f64, config: SlicerConfig) -> nparray_f64:
    """Compute nominal extrusion in mm of filament for segment lengths ``ds_mm``.

    Implements:

    .. math::

        E_{nom} = \\frac{w\\,h\\,ds}{\\pi (d_f/2)^2}.
    """

    ds = np.asarray(ds_mm, dtype=np.float64)
    w = float(config.extrusion_width)
    h = float(config.layer_height)
    d = float(config.filament_diameter)

    if d <= 0.0:
        raise ValueError("config.filament_diameter must be positive.")

    a_f = np.pi * (0.5 * d) ** 2
    v = w * h * ds
    return v / a_f


def tilt_compensate(e_nom: nparray_f64, beta: nparray_f64, min_cos_beta: float = 0.1) -> nparray_f64:
    """Apply tilt compensation to nominal extrusion.

    Implements:

    .. math::

        E_{adj} = \\frac{E_{nom}}{\\max(\\cos\\beta,\\ c_{min})}.
    """

    e = np.asarray(e_nom, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    c = np.maximum(np.cos(b), float(min_cos_beta))
    return e / c


class ExtrusionCompensator(ABC):
    """Abstract base class for extrusion computation."""

    @abstractmethod
    def compute_e_per_point(self, toolpath: np.ndarray, config: SlicerConfig) -> nparray_f64:
        """Return per-point relative extrusion values (M83 semantics)."""


@dataclass(slots=True)
class TiltExtrusionCompensator(ExtrusionCompensator):
    """Nominal extrusion + tilt compensation for structured toolpaths."""

    def compute_e_per_point(self, toolpath: np.ndarray, config: SlicerConfig) -> nparray_f64:
        tp = np.asarray(toolpath)
        if tp.ndim != 1:
            raise ValueError("toolpath must be a 1D structured array.")

        x = np.asarray(tp["x"], dtype=np.float64)
        y = np.asarray(tp["y"], dtype=np.float64)
        z = np.asarray(tp["z"], dtype=np.float64)

        ds = np.zeros_like(x)
        if len(tp) >= 2:
            dx = np.diff(x)
            dy = np.diff(y)
            dz = np.diff(z)
            ds[1:] = np.sqrt(dx * dx + dy * dy + dz * dz)

        # Use B if present; otherwise derive from stored tilt_angle (or normals).
        beta = np.asarray(tp["b"], dtype=np.float64)
        if np.allclose(beta, 0.0):
            # tilt_angle = arccos(nz) stored in dtype
            beta = np.asarray(tp["tilt_angle"], dtype=np.float64)

        e_nom = nominal_extrusion_mm_filament(ds, config)
        e_adj = tilt_compensate(e_nom, beta, min_cos_beta=float(config.min_cos_beta))

        move_type = np.asarray(tp["move_type"])
        is_extrude = move_type == "extrude"

        e = np.zeros_like(e_adj)
        e[is_extrude] = e_adj[is_extrude]
        # First point has no segment.
        if len(e) > 0:
            e[0] = 0.0
        return e

