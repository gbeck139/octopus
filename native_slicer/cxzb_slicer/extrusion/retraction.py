"""
Retraction planning for travel moves.

This module inserts retract/unretract events around segments marked as
``move_type == 'travel'`` in the structured toolpath. With Duet-style
relative extrusion (`M83`), retractions are represented as toolpath rows
with an `e` value of `-retract_length_mm` (retract) and
`+retract_length_mm` (unretract).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from cxzb_slicer.core.types import SlicerConfig, TOOLPATH_DTYPE


class RetractionPlanner(ABC):
    """Abstract base class for retraction insertion."""

    @abstractmethod
    def apply(self, toolpath: np.ndarray, config: SlicerConfig) -> np.ndarray:
        """Return a new toolpath with retract/unretract moves inserted."""


@dataclass(slots=True)
class SimpleRetractionPlanner(RetractionPlanner):
    """Insert one retract before travel and one unretract after travel."""

    def apply(self, toolpath: np.ndarray, config: SlicerConfig) -> np.ndarray:
        tp = np.asarray(toolpath, dtype=TOOLPATH_DTYPE)
        if len(tp) == 0:
            return tp

        move_type = tp["move_type"]
        is_travel = move_type == "travel"
        is_extrude = move_type == "extrude"

        # Identify boundaries of travel blocks.
        prev_travel = np.concatenate([[False], is_travel[:-1]])
        next_travel = np.concatenate([is_travel[1:], [False]])

        travel_start = is_travel & ~prev_travel
        travel_end = is_travel & ~next_travel

        # Retract before the first point of a travel block if previous move was extrude.
        retract_at = travel_start & np.concatenate([[False], is_extrude[:-1]])
        # Unretract after the last point of a travel block if next move is extrude.
        unretract_at = travel_end & np.concatenate([is_extrude[1:], [False]])

        n_new = int(np.sum(retract_at) + np.sum(unretract_at))
        if n_new == 0:
            return tp

        out = np.empty(len(tp) + n_new, dtype=TOOLPATH_DTYPE)

        retract_len = float(config.retract_length_mm)
        retract_f_mm_min = float(config.retract_feed_mm_s) * 60.0

        j = 0
        for i in range(len(tp)):
            if retract_at[i]:
                out[j] = tp[i]
                out[j]["move_type"] = "retract"
                out[j]["e"] = -retract_len
                out[j]["f"] = retract_f_mm_min
                j += 1

            out[j] = tp[i]
            j += 1

            if unretract_at[i]:
                out[j] = tp[i]
                out[j]["move_type"] = "retract"
                out[j]["e"] = +retract_len
                out[j]["f"] = retract_f_mm_min
                j += 1

        return out

