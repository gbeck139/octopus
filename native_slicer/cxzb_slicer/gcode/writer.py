"""
G-code writers for Duet (RepRapFirmware) and Marlin dialects.

Primary target: **RepRapFirmware (Duet)** using relative extrusion (`M83`).

Line format (per move):

.. code-block:: text

    G1 X{x_m:.3f} Z{z_m:.3f} C{c_deg:.3f} B{b_deg:.3f} E{e:.5f} F{f_mm_min:.0f}

Notes
-----
- The machine has **no Y axis** in output.
- Toolpath angles are stored internally in radians and emitted in degrees.
- Feedrates are emitted as **mm/min**. The toolpath field ``f`` is treated
  as mm/min if non-zero; otherwise defaults from :class:`SlicerConfig`
  are used (mm/s converted to mm/min).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

from cxzb_slicer.core.types import SlicerConfig, TOOLPATH_DTYPE


class GCodeWriter(ABC):
    """Abstract base class for G-code writers."""

    @abstractmethod
    def write(self, toolpath: np.ndarray, config: SlicerConfig) -> str:
        """Render the structured toolpath to a G-code string."""


def _format_g1(
    x_m: float | None,
    z_m: float | None,
    c_deg: float | None,
    b_deg: float | None,
    e: float | None,
    f_mm_min: float | None,
) -> str:
    parts: List[str] = ["G1"]
    if x_m is not None:
        parts.append(f"X{x_m:.3f}")
    if z_m is not None:
        parts.append(f"Z{z_m:.3f}")
    if c_deg is not None:
        parts.append(f"C{c_deg:.3f}")
    if b_deg is not None:
        parts.append(f"B{b_deg:.3f}")
    if e is not None:
        parts.append(f"E{e:.5f}")
    if f_mm_min is not None:
        parts.append(f"F{f_mm_min:.0f}")
    return " ".join(parts)


@dataclass(slots=True)
class DuetGCodeWriter(GCodeWriter):
    """RepRapFirmware (Duet) writer."""

    def write(self, toolpath: np.ndarray, config: SlicerConfig) -> str:
        tp = np.asarray(toolpath, dtype=TOOLPATH_DTYPE)
        lines: List[str] = []

        # Preamble
        lines.append("G28")
        lines.append("G90")
        lines.append("M83")
        if config.bed_temp_c is not None:
            lines.append(f"M140 S{float(config.bed_temp_c):.0f}")
            lines.append(f"M190 S{float(config.bed_temp_c):.0f}")
        if config.nozzle_temp_c is not None:
            lines.append(f"M104 S{float(config.nozzle_temp_c):.0f}")
            lines.append(f"M109 S{float(config.nozzle_temp_c):.0f}")
        lines.append("G92 E0")

        # Defaults in mm/min
        f_print = float(config.print_feed_mm_s) * 60.0
        f_travel = float(config.travel_feed_mm_s) * 60.0
        f_retract = float(config.retract_feed_mm_s) * 60.0

        for i in range(len(tp)):
            mt = str(tp["move_type"][i])
            x_m = float(tp["x_m"][i])
            z_m = float(tp["z_m"][i])
            c_deg = float(np.rad2deg(tp["c"][i]))
            b_deg = float(np.rad2deg(tp["b"][i]))
            e = float(tp["e"][i])

            f = float(tp["f"][i])
            if f <= 0.0:
                if mt == "travel":
                    f = f_travel
                elif mt == "retract":
                    f = f_retract
                else:
                    f = f_print

            lines.append(_format_g1(x_m=x_m, z_m=z_m, c_deg=c_deg, b_deg=b_deg, e=e, f_mm_min=f))

        # Postamble
        lines.append(_format_g1(x_m=None, z_m=None, c_deg=None, b_deg=0.0, e=-float(config.retract_length_mm), f_mm_min=f_retract))
        lines.append("G28 X Z")
        lines.append("M104 S0")
        lines.append("M140 S0")
        return "\n".join(lines) + "\n"


@dataclass(slots=True)
class MarlinGCodeWriter(GCodeWriter):
    """Marlin dialect writer (secondary)."""

    def write(self, toolpath: np.ndarray, config: SlicerConfig) -> str:
        # Marlin can also use M83, so we mirror Duet formatting.
        return DuetGCodeWriter().write(toolpath, config)

