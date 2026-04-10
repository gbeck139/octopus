"""
Core data structures for the CX-ZB 4-axis slicer.

This module defines:

- ``TOOLPATH_DTYPE``: a NumPy structured dtype used for fully vectorized
  toolpath operations.
- ``Layer``: a lightweight container for a single layer's toolpath.
- ``SlicerConfig``: global configuration and machine parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol

import numpy as np
import numpy.typing as npt

MoveType = Literal["extrude", "travel", "retract"]


# ---------------------------------------------------------------------------
# Toolpath representation
# ---------------------------------------------------------------------------

TOOLPATH_DTYPE: np.dtype = np.dtype(
    [
        # World-frame TCP position (workpiece coordinates)
        ("x", "f8"),
        ("y", "f8"),
        ("z", "f8"),
        # Machine coordinates after IK (gantry + rotary axes)
        ("x_m", "f8"),
        ("z_m", "f8"),
        ("c", "f8"),  # C-axis angle (radians)
        ("b", "f8"),  # B-axis angle (radians, tilt around Y)
        # Extrusion and feed
        ("e", "f8"),  # Extrusion amount (relative, typically mm of filament)
        ("f", "f8"),  # Feedrate (mm/min)
        # Layer normal in world frame (from ∇f_layer)
        ("nx", "f8"),
        ("ny", "f8"),
        ("nz", "f8"),
        # Auxiliary metadata
        ("tilt_angle", "f8"),       # |β| = arccos(nz)
        ("extrusion_width", "f8"),  # Local bead width
        ("layer_index", "i4"),      # Index of the layer that owns this point
        ("move_type", "U8"),        # One of {"extrude", "travel", "retract"}
    ]
)


class ToolpathLike(Protocol):
    """Protocol for any object that exposes a structured toolpath array."""

    @property
    def points(self) -> np.ndarray:
        """Return the underlying NumPy structured array with ``TOOLPATH_DTYPE``."""


@dataclass(slots=True)
class Layer:
    """A single logical printing layer.

    Attributes
    ----------
    index:
        Zero-based layer index.
    surface_id:
        Identifier of the generating implicit surface family
        (e.g. ``\"planar\"``, ``\"conic\"``, ``\"spherical\"``).
    level:
        Level-set parameter ``c`` such that the layer surface satisfies
        :math:`f_{layer}(x, y, z) = c`.
    points:
        Toolpath points for this layer as a NumPy structured array with
        ``TOOLPATH_DTYPE``. All kinematics and extrusion computations
        are performed in a fully vectorized manner over this array.
    metadata:
        Optional free-form dictionary for storing implementation-specific
        information (e.g. debug statistics, quality metrics).
    """

    index: int
    surface_id: str
    level: float
    points: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=TOOLPATH_DTYPE)
    )
    metadata: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SlicerConfig:
    """Global configuration for the CX-ZB slicer and machine.

    All machine- and process-specific parameters are collected here so that
    individual modules (kinematics, vector field, extrusion, G-code writer)
    can remain free of hardcoded constants. The most important RTCP-related
    quantity is the tool offset :math:`L`, defined as the distance from the
    B-axis pivot to the nozzle tip along the tool axis when :math:`\\beta = 0`.

    Parameters
    ----------
    tool_offset_L:
        Pivot-to-nozzle distance :math:`L` in millimetres. This appears in
        the RTCP formulas

        .. math::

            \\Delta X = L \\sin \\beta,\\quad
            \\Delta Z = L(\\cos \\beta - 1),

        and in the world-to-machine mapping

        .. math::

            X_m = r + L \\sin \\beta,\\quad
            Z_m = z_w + L \\cos \\beta.
    b_min, b_max:
        Minimum and maximum B-axis angles (radians).
    c_min, c_max:
        Soft bounds on C-axis rotation (radians). The singularity handler
        is responsible for unwrapping C beyond this range when needed.
    x_min, x_max, z_min, z_max:
        Linear axis travel limits in millimetres.
    max_feed_mm_s:
        Maximum nominal TCP feedrate in mm/s.
    max_b_omega_deg_s, max_c_omega_deg_s:
        Maximum angular velocities for B and C axes in degrees/s, used for
        validating toolpaths and enforcing motor limits.
    layer_height:
        Nominal layer height for planar slicing in millimetres.
    extrusion_width:
        Nominal extrusion width in millimetres.
    filament_diameter:
        Filament diameter in millimetres (e.g. 1.75).
    nozzle_diameter:
        Nozzle orifice diameter in millimetres.
    retract_length_mm:
        Retraction length in mm of filament for travel moves.
    retract_feed_mm_s:
        Retraction/unretraction speed in mm/s (converted to mm/min in G-code).
    print_feed_mm_s, travel_feed_mm_s:
        Default print and travel feedrates in mm/s when toolpath does not
        explicitly specify a feedrate.
    min_cos_beta:
        Minimum allowed value of :math:`\\cos\\beta` when applying tilt-angle
        extrusion compensation :math:`E_{adj} = E_{nom}/\\cos\\beta`.
    nozzle_temp_c, bed_temp_c:
        Optional set temperatures in °C used by the G-code preamble.
    """

    # RTCP geometry
    tool_offset_L: float = 50.0

    # Axis limits (radians for rotary, mm for linear)
    b_min: float = np.deg2rad(-75.0)
    b_max: float = np.deg2rad(75.0)
    c_min: float = -np.inf
    c_max: float = np.inf

    x_min: float = 0.0
    x_max: float = 300.0
    z_min: float = 0.0
    z_max: float = 300.0

    # Motion limits
    max_feed_mm_s: float = 150.0
    max_b_omega_deg_s: float = 30.0
    max_c_omega_deg_s: float = 60.0

    # Process parameters
    layer_height: float = 0.2
    extrusion_width: float = 0.45
    filament_diameter: float = 1.75
    nozzle_diameter: float = 0.4

    # Extrusion planning & retraction
    retract_length_mm: float = 0.8
    retract_feed_mm_s: float = 35.0
    min_cos_beta: float = 0.1

    # Default feeds (mm/s)
    print_feed_mm_s: float = 40.0
    travel_feed_mm_s: float = 120.0

    # Optional temperatures (°C)
    nozzle_temp_c: Optional[float] = None
    bed_temp_c: Optional[float] = None

    # Optional profile name or identifier
    profile_name: Optional[str] = None

    def as_dict(self) -> dict[str, object]:
        """Return a plain ``dict`` view of the configuration."""

        return {
            "tool_offset_L": self.tool_offset_L,
            "b_min": self.b_min,
            "b_max": self.b_max,
            "c_min": self.c_min,
            "c_max": self.c_max,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "z_min": self.z_min,
            "z_max": self.z_max,
            "max_feed_mm_s": self.max_feed_mm_s,
            "max_b_omega_deg_s": self.max_b_omega_deg_s,
            "max_c_omega_deg_s": self.max_c_omega_deg_s,
            "layer_height": self.layer_height,
            "extrusion_width": self.extrusion_width,
            "filament_diameter": self.filament_diameter,
            "nozzle_diameter": self.nozzle_diameter,
            "retract_length_mm": self.retract_length_mm,
            "retract_feed_mm_s": self.retract_feed_mm_s,
            "min_cos_beta": self.min_cos_beta,
            "print_feed_mm_s": self.print_feed_mm_s,
            "travel_feed_mm_s": self.travel_feed_mm_s,
            "nozzle_temp_c": self.nozzle_temp_c,
            "bed_temp_c": self.bed_temp_c,
            "profile_name": self.profile_name,
        }

