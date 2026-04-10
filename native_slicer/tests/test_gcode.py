"""Tests for G-code writer and validator."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt

from cxzb_slicer.core.types import SlicerConfig, TOOLPATH_DTYPE
from cxzb_slicer.extrusion.compensator import TiltExtrusionCompensator
from cxzb_slicer.extrusion.retraction import SimpleRetractionPlanner
from cxzb_slicer.gcode.validator import BasicGCodeValidator, populate_kinematics_fields
from cxzb_slicer.gcode.writer import DuetGCodeWriter


def test_duet_writer_formats_degrees_and_relative_e() -> None:
    cfg = SlicerConfig(print_feed_mm_s=50.0, travel_feed_mm_s=100.0)
    tp = np.zeros(3, dtype=TOOLPATH_DTYPE)
    tp["x"] = [10.0, 20.0, 30.0]
    tp["y"] = [0.0, 0.0, 0.0]
    tp["z"] = [5.0, 5.0, 5.0]
    tp["nx"] = 0.0
    tp["ny"] = 0.0
    tp["nz"] = 1.0
    tp["move_type"] = ["extrude", "travel", "extrude"]

    tp = populate_kinematics_fields(tp, cfg)
    tp["e"] = TiltExtrusionCompensator().compute_e_per_point(tp, cfg)
    tp = SimpleRetractionPlanner().apply(tp, cfg)

    gcode = DuetGCodeWriter().write(tp, cfg)
    assert "M83" in gcode
    assert "G1 X" in gcode
    # Ensure degrees formatting (B/C appear with decimals)
    assert " B" in gcode and " C" in gcode


def test_validator_fk_roundtrip_passes_on_synthetic_path() -> None:
    cfg = SlicerConfig(tool_offset_L=50.0)
    tp = np.zeros(5, dtype=TOOLPATH_DTYPE)
    tp["x"] = np.linspace(20.0, 40.0, 5)
    tp["y"] = np.linspace(10.0, 15.0, 5)
    tp["z"] = np.linspace(2.0, 2.0, 5)
    tp["nx"] = 0.0
    tp["ny"] = 0.0
    tp["nz"] = 1.0
    tp["move_type"] = "extrude"

    tp = populate_kinematics_fields(tp, cfg)
    tp["e"] = TiltExtrusionCompensator().compute_e_per_point(tp, cfg)

    gcode = DuetGCodeWriter().write(tp, cfg)
    # Writer formats X/Z to 0.001mm and angles to 0.001deg, so allow ~1e-3mm FK mismatch.
    errors = BasicGCodeValidator(position_tol_mm=1e-3).validate(tp, gcode, cfg)
    assert errors == []

