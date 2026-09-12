r"""Domains a fixed-basis operator surrogate can be built over.

One, because one is what has a consumer. A POD basis needs a backend and
a metric, and a caller who already holds both needs nothing from this
module -- they satisfy :class:`MetricSpaceProtocol` by having two
methods, and it is structural, so neither side imports the other.

What :class:`UniformGridDomain` adds is the quadrature: trapezoid
weights over a tensor-product grid, which is what makes the inner
product approximate an integral rather than a sum. That is the whole
content, and it is why a time axis or an image needs no PDE solver to
serve as a domain here.
"""

from __future__ import annotations

from typing import Generic, Optional, Sequence

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.inner_product import (
    DiagonalInnerProduct,
    InnerProductProtocol,
)


class UniformGridDomain(Generic[Array]):
    r"""A tensor-product grid with trapezoid quadrature weights.

    The non-PDE case, and the one that shows this package needs no
    solver: a time axis is a grid in one dimension, an image is a grid
    in two. The weights are what earn the name -- they make
    :math:`\langle f, g \rangle_M` approximate :math:`\int f g` rather
    than a sum that happens to weight every point equally, which is the
    difference between a POD basis optimal in the field norm and one
    biased toward wherever the grid is fine.

    Sites are enumerated in C order over the axes, matching
    ``numpy.meshgrid(..., indexing="ij")`` flattened, so a caller
    building snapshots with ``field.ravel()`` needs no permutation.

    Despite the name the axes need not be uniformly spaced; the
    trapezoid weights handle a graded axis correctly, and a graded axis
    is exactly where the weighting matters most.

    Parameters
    ----------
    axes : sequence of Array
        One strictly increasing coordinate vector per dimension.
    bkd : Backend[Array]
        Computational backend.
    inner_product : InnerProductProtocol[Array], optional
        Override the trapezoid weights, for a measure that is not
        Lebesgue.
    """

    def __init__(
        self,
        axes: Sequence[Array],
        bkd: Backend[Array],
        inner_product: Optional[InnerProductProtocol[Array]] = None,
    ) -> None:
        if len(axes) == 0:
            raise ValueError("axes must not be empty")
        self._axes = [
            np.asarray(bkd.to_numpy(axis), dtype=float) for axis in axes
        ]
        for index, axis in enumerate(self._axes):
            if axis.ndim != 1 or axis.size < 2:
                raise ValueError(
                    f"axis {index} must be 1D with at least two points, "
                    f"got shape {axis.shape}"
                )
            if not bool(np.all(np.diff(axis) > 0.0)):
                raise ValueError(
                    f"axis {index} must be strictly increasing"
                )
        self._bkd = bkd
        mesh = np.meshgrid(*self._axes, indexing="ij")
        self._points = bkd.asarray(np.vstack([m.ravel() for m in mesh]))
        self._inner_product = (
            DiagonalInnerProduct(self._trapezoid_weights(), bkd)
            if inner_product is None
            else inner_product
        )
        nsites = int(self._points.shape[1])
        if self._inner_product.nstates() != nsites:
            raise ValueError(
                f"inner_product is defined on "
                f"{self._inner_product.nstates()} states but the grid "
                f"has {nsites} points"
            )

    def _trapezoid_weights(self) -> Array:
        """Return the tensor product of the per-axis trapezoid weights.

        Each interior point collects half of the interval on either
        side, so the weights sum to the measure of the box and reduce to
        the uniform spacing times the point count when the axis is
        equispaced.
        """
        per_axis = []
        for axis in self._axes:
            spacing = np.diff(axis)
            weights = np.zeros(axis.size)
            weights[:-1] += spacing / 2.0
            weights[1:] += spacing / 2.0
            per_axis.append(weights)
        total = per_axis[0]
        for weights in per_axis[1:]:
            total = np.multiply.outer(total, weights)
        return self._bkd.asarray(np.ascontiguousarray(total.ravel()))

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Number of grid dimensions."""
        return len(self._axes)

    def nsites(self) -> int:
        """Total number of grid points."""
        return int(self._points.shape[1])

    def sample_points(self) -> Array:
        """Return the grid points in C order. Shape: (ndim, nsites)."""
        return self._points

    def inner_product(self) -> InnerProductProtocol[Array]:
        """Return the metric, trapezoid weights unless overridden."""
        return self._inner_product

    def __repr__(self) -> str:
        shape = "x".join(str(axis.size) for axis in self._axes)
        return f"{self.__class__.__name__}(grid={shape})"
