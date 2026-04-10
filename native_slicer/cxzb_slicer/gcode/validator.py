"""
Post-process validation of emitted G-code and toolpath consistency.

This module provides:

- Parsing of `G1` lines to numeric values.
- Basic checks against machine limits.
- Forward-kinematics (FK) round-trip verification against the intended
  world toolpath.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from cxzb_slicer.core.types import SlicerConfig, TOOLPATH_DTYPE
from cxzb_slicer.kinematics.ik_solver import machine_to_world, world_to_machine
from cxzb_slicer.vector_field.field_solver import vector_to_angles


def parse_g1_lines(gcode: str) -> List[Dict[str, float]]:
    """Parse G1 lines into dicts of axis values.

    Returns a list of dicts containing any of: X,Z,B,C,E,F.
    """

    moves: List[Dict[str, float]] = []
    for line in gcode.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if not line.startswith("G1"):
            continue
        parts = line.split()
        d: Dict[str, float] = {}
        for p in parts[1:]:
            if len(p) < 2:
                continue
            ax = p[0].upper()
            if ax in {"X", "Z", "B", "C", "E", "F"}:
                try:
                    d[ax] = float(p[1:])
                except ValueError:
                    continue
        moves.append(d)
    return moves


def populate_kinematics_fields(toolpath: np.ndarray, config: SlicerConfig) -> np.ndarray:
    """Populate b,c,x_m,z_m fields from world coords and normals.

    Uses the convention from the research note:

    - k = normalize([nx, ny, nz])
    - (b, c) = vector_to_angles(k)
    - (x_m, z_m, c_angle) = world_to_machine(x, y, z, beta=b, L)
    - store c = c_angle (bed rotation)
    """

    tp = np.asarray(toolpath, dtype=TOOLPATH_DTYPE).copy()
    k = np.stack([tp["nx"], tp["ny"], tp["nz"]], axis=1).astype(np.float64, copy=False)
    n = np.linalg.norm(k, axis=1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    k = k / n

    b, c = vector_to_angles(k)
    tp["b"] = b

    x_m, z_m, c_angle = world_to_machine(
        tp["x"].astype(np.float64),
        tp["y"].astype(np.float64),
        tp["z"].astype(np.float64),
        b,
        float(config.tool_offset_L),
    )
    tp["x_m"] = x_m
    tp["z_m"] = z_m
    tp["c"] = c_angle
    return tp


class GCodeValidator(ABC):
    """Abstract base class for validators."""

    @abstractmethod
    def validate(self, toolpath: np.ndarray, gcode: str, config: SlicerConfig) -> List[str]:
        """Return a list of human-readable validation errors."""


@dataclass(slots=True)
class BasicGCodeValidator(GCodeValidator):
    """Basic validator for limits and FK round-trip."""

    position_tol_mm: float = 0.2

    def validate(self, toolpath: np.ndarray, gcode: str, config: SlicerConfig) -> List[str]:
        tp = np.asarray(toolpath, dtype=TOOLPATH_DTYPE)
        moves = parse_g1_lines(gcode)
        errors: List[str] = []

        # Axis limits check (X/Z and B).
        for i, m in enumerate(moves):
            if "X" in m and (m["X"] < config.x_min - 1e-9 or m["X"] > config.x_max + 1e-9):
                errors.append(f"G1[{i}] X out of bounds: {m['X']}")
            if "Z" in m and (m["Z"] < config.z_min - 1e-9 or m["Z"] > config.z_max + 1e-9):
                errors.append(f"G1[{i}] Z out of bounds: {m['Z']}")
            if "B" in m:
                b_rad = np.deg2rad(m["B"])
                if b_rad < config.b_min - 1e-9 or b_rad > config.b_max + 1e-9:
                    errors.append(f"G1[{i}] B out of bounds (deg): {m['B']}")

        # FK round-trip vs toolpath world points.
        # We align by index for moves that have X/Z/B/C and a corresponding tp row.
        n = min(len(tp), len(moves))
        if n == 0:
            return errors

        x_m = np.array([moves[i].get("X", np.nan) for i in range(n)], dtype=np.float64)
        z_m = np.array([moves[i].get("Z", np.nan) for i in range(n)], dtype=np.float64)
        c_deg = np.array([moves[i].get("C", np.nan) for i in range(n)], dtype=np.float64)
        b_deg = np.array([moves[i].get("B", np.nan) for i in range(n)], dtype=np.float64)

        ok = ~np.isnan(x_m) & ~np.isnan(z_m) & ~np.isnan(c_deg) & ~np.isnan(b_deg)
        if np.any(ok):
            xw, yw, zw = machine_to_world(
                x_m[ok],
                z_m[ok],
                np.deg2rad(c_deg[ok]),
                np.deg2rad(b_deg[ok]),
                float(config.tool_offset_L),
            )
            world_fk = np.stack([xw, yw, zw], axis=1)
            world_ref = np.stack([tp["x"][:n][ok], tp["y"][:n][ok], tp["z"][:n][ok]], axis=1).astype(np.float64)
            d = np.linalg.norm(world_fk - world_ref, axis=1)
            tol = float(self.position_tol_mm)
            if np.any(d > tol):
                worst = float(np.max(d))
                errors.append(f"FK round-trip mismatch: max error {worst:.6f} mm (tol {tol:.6f})")

        return errors

