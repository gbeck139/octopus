"""
Configuration helpers and abstract interfaces for the CX-ZB slicer.

This module builds on :class:`cxzb_slicer.core.types.SlicerConfig` and
provides a small abstract base class so that alternative configuration
backends (e.g. TOML files, GUI frontends) can be swapped without touching
the rest of the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, Mapping

from .types import SlicerConfig


class AbstractSlicerConfig(ABC):
    """Abstract configuration interface for the slicer.

    The concrete :class:`SlicerConfig` dataclass implements this interface
    implicitly. Modules should depend on :class:`AbstractSlicerConfig`
    rather than the concrete type whenever possible.
    """

    @property
    @abstractmethod
    def tool_offset_L(self) -> float:  # pragma: no cover - trivial delegation
        """Pivot-to-nozzle distance :math:`L` in millimetres."""

    @property
    @abstractmethod
    def b_min(self) -> float:  # pragma: no cover - trivial delegation
        """Minimum B-axis angle (radians)."""

    @property
    @abstractmethod
    def b_max(self) -> float:  # pragma: no cover - trivial delegation
        """Maximum B-axis angle (radians)."""

    # The remaining properties follow the same pattern but are not
    # individually listed here to keep the interface concise.


def create_default_config(profile_name: str | None = None) -> SlicerConfig:
    """Return a default :class:`SlicerConfig` instance.

    The returned configuration uses conservative machine limits and a
    nominal tool offset :math:`L = 50\\,\\text{mm}` suitable for testing
    the RTCP equations

    .. math::

        \\Delta X = L \\sin \\beta,\\quad
        \\Delta Z = L(\\cos \\beta - 1).

    Parameters
    ----------
    profile_name:
        Optional descriptive name for the configuration profile.
    """

    cfg = SlicerConfig()
    if profile_name is not None:
        cfg.profile_name = profile_name
    return cfg


def config_from_mapping(mapping: Mapping[str, Any]) -> SlicerConfig:
    """Create a :class:`SlicerConfig` from a generic mapping.

    Any keys present in ``mapping`` that correspond to fields in
    :class:`SlicerConfig` are used to override the defaults. Extra keys
    are ignored, making this function tolerant of superset configuration
    files (e.g. shared across multiple machines).
    """

    cfg = SlicerConfig()
    for key, value in mapping.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def updated_config(cfg: SlicerConfig, **overrides: Any) -> SlicerConfig:
    """Return a copy of ``cfg`` with selected fields overridden.

    This is a convenience wrapper around :func:`dataclasses.replace`
    and is useful when exploring parameter sweeps in tests or examples.
    """

    return replace(cfg, **overrides)

