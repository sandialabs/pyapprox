"""Dynamic wrapper for piecewise polynomial bases with node generators.

This module provides:
- NodeGenerator: Abstract base for generating interpolation nodes
- EquidistantNodeGenerator: Equidistant nodes on an interval
- DynamicPiecewiseBasis: Wrapper that adds set_nterms() to fixed-node bases

These enable piecewise polynomial bases to be used with sparse grids,
which require dynamic node count adjustment via set_nterms().
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, Tuple, Type

from pyapprox.typing.util.backends.protocols import Array, Backend


class NodeGenerator(ABC, Generic[Array]):
    """Abstract base for generating interpolation nodes dynamically.

    Node generators create nodes for piecewise polynomial bases,
    allowing the number of nodes to be changed via set_nterms().

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    bounds : Tuple[float, float]
        Domain bounds (lower, upper).
    """

    def __init__(self, bkd: Backend[Array], bounds: Tuple[float, float]):
        self._bkd = bkd
        self._bounds = bounds

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    @abstractmethod
    def __call__(self, nnodes: int) -> Array:
        """Generate nodes.

        Parameters
        ----------
        nnodes : int
            Number of nodes to generate.

        Returns
        -------
        Array
            Node positions with shape (nnodes,).
        """
        raise NotImplementedError


class EquidistantNodeGenerator(NodeGenerator[Array]):
    """Generate equidistant nodes on an interval.

    Creates nnodes equally-spaced points on [bounds[0], bounds[1]].

    Example
    -------
    >>> bkd = NumpyBkd()
    >>> gen = EquidistantNodeGenerator(bkd, (-1.0, 1.0))
    >>> nodes = gen(5)  # [-1, -0.5, 0, 0.5, 1]
    """

    def __call__(self, nnodes: int) -> Array:
        """Generate equidistant nodes.

        Parameters
        ----------
        nnodes : int
            Number of nodes.

        Returns
        -------
        Array
            Equidistant nodes with shape (nnodes,).
        """
        return self._bkd.linspace(self._bounds[0], self._bounds[1], nnodes)


class DynamicPiecewiseBasis(Generic[Array]):
    """Wrapper providing set_nterms() for piecewise polynomial bases.

    Wraps existing fixed-node piecewise classes (PiecewiseQuadratic, etc.)
    with a node generator to enable set_nterms() for sparse grid usage.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    basis_class : Type
        Piecewise basis class (e.g., PiecewiseQuadratic, PiecewiseLinear).
    node_generator : NodeGenerator[Array]
        Generator for creating nodes dynamically.

    Example
    -------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.surrogates.affine.univariate.piecewisepoly import (
    ...     PiecewiseQuadratic,
    ... )
    >>> bkd = NumpyBkd()
    >>> node_gen = EquidistantNodeGenerator(bkd, (-1.0, 1.0))
    >>> basis = DynamicPiecewiseBasis(bkd, PiecewiseQuadratic, node_gen)
    >>> basis.set_nterms(5)  # Now has 5 nodes
    >>> pts, wts = basis.quadrature_rule()
    """

    def __init__(
        self,
        bkd: Backend[Array],
        basis_class: Type[Any],
        node_generator: NodeGenerator[Array],
    ):
        self._bkd = bkd
        self._basis_class = basis_class
        self._node_gen = node_generator
        self._basis: Optional[Any] = None
        self._nterms = 0

    def set_nterms(self, nterms: int) -> None:
        """Set number of basis terms (nodes).

        Creates a new internal basis with the specified number of nodes.

        Parameters
        ----------
        nterms : int
            Number of nodes/terms.
        """
        nodes = self._node_gen(nterms)
        self._basis = self._basis_class(nodes, self._bkd)
        self._nterms = nterms

    def nterms(self) -> int:
        """Return current number of terms (nodes)."""
        return self._nterms

    def __call__(self, samples: Array) -> Array:
        """Evaluate basis functions at given points.

        Parameters
        ----------
        samples : Array
            Points to evaluate at, shape (npts,).

        Returns
        -------
        Array
            Basis values, shape (npts, nterms).

        Raises
        ------
        ValueError
            If set_nterms has not been called.
        """
        if self._basis is None:
            raise ValueError("Must call set_nterms before evaluation")
        result: Array = self._basis(samples)
        return result

    def quadrature_rule(self) -> Tuple[Array, Array]:
        """Return quadrature points and weights.

        Returns
        -------
        Tuple[Array, Array]
            (points, weights) where points has shape (nterms,)
            and weights has shape (nterms,).

        Raises
        ------
        ValueError
            If set_nterms has not been called.
        """
        if self._basis is None:
            raise ValueError("Must call set_nterms before quadrature_rule")
        result: Tuple[Array, Array] = self._basis.quadrature_rule()
        return result

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd
