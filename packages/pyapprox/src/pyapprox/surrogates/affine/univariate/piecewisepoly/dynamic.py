"""Dynamic wrapper for piecewise polynomial bases with node generators.

This module provides:
- NodeGenerator: Abstract base for generating interpolation nodes
- EquidistantNodeGenerator: Equidistant nodes on an interval
- DynamicPiecewiseBasis: Wrapper that adds set_nterms() to fixed-node bases

These enable piecewise polynomial bases to be used with sparse grids,
which require dynamic node count adjustment via set_nterms().
"""

from abc import ABC, abstractmethod
from typing import Callable, Generic, Optional, Tuple

from pyapprox.surrogates.affine.univariate.piecewisepoly.protocols import (
    PiecewisePolynomialProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.protocols.quadrature import UnivariateQuadratureRuleProtocol


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
    basis_class : Callable[[Array, Backend[Array]], PiecewisePolynomialProtocol[Array]]
        Builds the piecewise basis from nodes (e.g., PiecewiseQuadratic,
        PiecewiseLinear).
    node_generator : NodeGenerator[Array]
        Generator for creating nodes dynamically.
    quadrature : UnivariateQuadratureRuleProtocol[Array], optional
        Rule whose weights ``quadrature_rule`` returns, for example a
        ``PiecewiseMeasureQuadratureRule`` with the same basis class and
        node generator, for probability weights under a marginal. Its nodes
        must be this basis's nodes; a mismatch raises. Default None: the
        basis's own Lebesgue weights.

    Example
    -------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.surrogates.affine.univariate.piecewisepoly import (
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
        basis_class: Callable[
            [Array, Backend[Array]], PiecewisePolynomialProtocol[Array]
        ],
        node_generator: NodeGenerator[Array],
        quadrature: Optional[UnivariateQuadratureRuleProtocol[Array]] = None,
    ):
        self._bkd = bkd
        self._basis_class = basis_class
        self._node_gen = node_generator
        self._quadrature = quadrature
        self._basis: Optional[PiecewisePolynomialProtocol[Array]] = None
        self._nterms = 0

    def set_nterms(self, nterms: int) -> None:
        """Set number of basis terms (nodes).

        Creates a new internal basis with the specified number of nodes.
        For nterms=1, uses a constant basis (no internal basis created).

        Parameters
        ----------
        nterms : int
            Number of nodes/terms.
        """
        self._nterms = nterms
        if nterms == 1:
            # 1-point case: constant basis, no piecewise polynomial needed
            self._basis = None
        else:
            nodes = self._node_gen(nterms)
            self._basis = self._basis_class(nodes, self._bkd)

    def nterms(self) -> int:
        """Return current number of terms (nodes)."""
        return self._nterms

    def __call__(self, samples: Array) -> Array:
        """Evaluate basis functions at given points.

        Parameters
        ----------
        samples : Array
            Points to evaluate at, shape (1, nsamples).

        Returns
        -------
        Array
            Basis values, shape (nsamples, nterms).

        Raises
        ------
        ValueError
            If set_nterms has not been called or samples has wrong shape.
        """
        if self._nterms == 0:
            raise ValueError("Must call set_nterms before evaluation")

        if samples.ndim != 2 or samples.shape[0] != 1:
            raise ValueError(
                f"Expected samples shape (1, nsamples), got {samples.shape}"
            )

        samples_1d = samples[0]

        # Handle 1-point case: constant basis function = 1 everywhere
        if self._nterms == 1:
            npts = samples_1d.shape[0]
            return self._bkd.ones((npts, 1))

        if self._basis is None:
            raise RuntimeError("Must call set_nterms() first")
        result: Array = self._basis(samples_1d)
        return result

    def quadrature_rule(self) -> Tuple[Array, Array]:
        """Return quadrature points and weights.

        With an injected ``quadrature``, its rule for ``nterms`` points.
        Otherwise the basis's Lebesgue rule; for the 1-point case
        (constant basis), the midpoint with weight equal to the domain
        width.

        Returns
        -------
        Tuple[Array, Array]
            (points, weights) where points has shape (1, nterms)
            and weights has shape (nterms, 1). Matches LagrangeBasis1D format.

        Raises
        ------
        ValueError
            If set_nterms has not been called.
        """
        if self._nterms == 0:
            raise ValueError("Must call set_nterms before quadrature_rule")

        if self._quadrature is not None:
            points, weights = self._quadrature(self._nterms)
            points = self._bkd.reshape(points, (1, -1))
            if not self._bkd.allclose(
                points, self.get_samples(self._nterms), rtol=1e-12, atol=1e-14
            ):
                raise ValueError(
                    "the injected quadrature's nodes are not this basis's "
                    "nodes; build both from the same node generator"
                )
            return points, self._bkd.reshape(weights, (-1, 1))

        # Handle 1-point case: constant basis with weight = domain width
        if self._nterms == 1:
            a, b = self._node_gen._bounds
            points = self._bkd.reshape(self._bkd.asarray([(a + b) / 2.0]), (1, 1))
            weights = self._bkd.reshape(self._bkd.asarray([b - a]), (1, 1))
            return points, weights

        if self._basis is None:
            raise RuntimeError("Must call set_nterms() first")
        pts, wts = self._basis.quadrature_rule()
        # Reshape to match LagrangeBasis1D: points (1, n), weights (n, 1)
        return self._bkd.reshape(pts, (1, -1)), self._bkd.reshape(wts, (-1, 1))

    def get_samples(self, nterms: int) -> Array:
        """Return interpolation nodes for the given number of terms.

        Parameters
        ----------
        nterms : int
            Number of interpolation nodes.

        Returns
        -------
        Array
            Node locations with shape (1, nterms).
        """
        if nterms == 1:
            # 1-point case: midpoint
            a, b = self._node_gen._bounds
            return self._bkd.reshape(self._bkd.asarray([(a + b) / 2.0]), (1, 1))
        nodes = self._node_gen(nterms)
        return self._bkd.reshape(nodes, (1, nterms))

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd
