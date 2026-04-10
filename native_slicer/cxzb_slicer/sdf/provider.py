"""
Abstract interface for signed distance field (SDF) providers.

A signed distance function :math:`\\varphi(x)` returns the shortest
distance from a point :math:`x \\in \\mathbb{R}^3` to the surface
boundary :math:`\\partial\\Omega` of a solid, with negative values
inside and positive values outside:

.. math::

    \\varphi(x) = s(x)\\,\\min_{y \\in \\partial\\Omega} \\lVert x - y \\rVert,

where :math:`s(x) \\in \\{-1, +1\\}` encodes the inside/outside sign.
Away from sharp features the SDF approximately satisfies the Eikonal
equation :math:`\\lVert \\nabla\\varphi \\rVert \\approx 1`.

This module defines :class:`SDFProvider`, an abstract base class used
throughout the slicer. Concrete implementations include a thin wrapper
around the :mod:`pysdf` library and a voxel-grid-backed SDF.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, Tuple

import numpy as np
import numpy.typing as npt


class SDFProvider(ABC):
    """Abstract base class for all SDF backends.

    Implementations must be fully vectorised: :meth:`signed_distance`
    accepts an :math:`(N, 3)` array of query points and returns an
    :math:`(N,)` array of signed distances.
    """

    @abstractmethod
    def signed_distance(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the signed distance :math:`\\varphi(x)` at query points.

        Parameters
        ----------
        points:
            Array of shape ``(N, 3)`` containing world-space sample
            locations :math:`x`.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(N,)`` with signed distances
            :math:`\\varphi(x)`.
        """

    @abstractmethod
    def bounding_box(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return axis-aligned bounding box of the represented solid.

        Returns
        -------
        (min_corner, max_corner):
            Two arrays of shape ``(3,)`` giving the minimum and maximum
            world coordinates of the bounding box that tightly encloses
            the solid.
        """


class SupportsSDF(Protocol):
    """Protocol for objects that can expose an :class:`SDFProvider`."""

    def as_sdf(self) -> SDFProvider:
        """Return an :class:`SDFProvider` view of this object."""

