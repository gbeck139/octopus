"""
Abstract base class for non-planar layer surface generators.

A layer surface is defined implicitly by a scalar field
:math:`f_{layer}(x, y, z)` and a level parameter :math:`c` such that the
surface is the zero-set of :math:`g(x) = f_{layer}(x) - c`. The slicer
intersects this surface with the object boundary :math:`\\varphi(x) = 0`
to obtain contours.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from cxzb_slicer.core.types import Layer, SlicerConfig
from cxzb_slicer.sdf.provider import SDFProvider


class LayerGenerator(ABC):
    """Abstract base class for all layer surface generators.

    Implementations receive an :class:`SDFProvider` and global
    :class:`SlicerConfig` and must return a list of :class:`Layer`
    instances whose :attr:`Layer.level` encodes the value :math:`c` in
    the implicit equation

    .. math::

        f_{layer}(x, y, z) = c.
    """

    @abstractmethod
    def generate_surfaces(self, sdf: SDFProvider, config: SlicerConfig) -> List[Layer]:
        """Generate a sequence of layer surfaces for the given object."""

