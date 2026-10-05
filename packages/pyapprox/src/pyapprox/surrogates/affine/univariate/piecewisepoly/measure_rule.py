r"""Piecewise polynomial quadrature for a probability measure.

The piecewise polynomial rules integrate against the Lebesgue measure. For an
expectation under a marginal with density :math:`p` on :math:`[a, b]`, the
weights here are the integrals of the basis functions against the density,

.. math::

    w_i = \frac{\int_a^b \phi_i(x)\, p(x)\, dx}{\int_a^b p(x)\, dx},

so the rule integrates the piecewise interpolant of :math:`f` exactly under
the (truncated, renormalized) measure, and keeps the basis's interpolation
order however fast :math:`p` varies.

Each :math:`w_i` is computed by Gauss-Legendre on every interval between
consecutive nodes. Every piecewise polynomial basis function is a
polynomial on every such interval, whatever its degree: splitting at every
node also splits at the kink of a linear hat and at the element boundaries
of a higher-degree basis. Only the density is left to resolve, and points
are added until an embedded estimate meets the tolerance.
"""

from typing import Callable, Dict, Generic, Tuple

import numpy as np

from pyapprox.probability.protocols.distribution import MarginalProtocol
from pyapprox.surrogates.affine.univariate.piecewisepoly.dynamic import (
    NodeGenerator,
)
from pyapprox.surrogates.affine.univariate.piecewisepoly.protocols import (
    PiecewisePolynomialProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class PiecewiseMeasureQuadratureRule(Generic[Array]):
    r"""Piecewise polynomial rule with probability weights.

    Satisfies ``UnivariateQuadratureRuleProtocol``: ``rule(npoints)``
    returns the generator's nodes, shape (1, npoints), and weights summing
    to 1, shape (npoints, 1), as ``gauss_quadrature_rule`` does. One point
    gives the interval's midpoint with weight 1.

    The accuracy of the weights is checked every time they are built:

    - **Embedded estimate.** The weights are computed with ``n`` and
      ``n + 2`` Gauss-Legendre points per interval, starting from
      ``points_per_interval``. While the largest change, relative to the
      largest weight, exceeds ``rtol``, two more points are added. If more
      than ``max_points_per_interval`` would be needed, a ``ValueError`` is
      raised: the density is not smooth on some interval (a kink not at a
      node) or is too peaked for the intervals.
    - **Mass.** Before renormalizing, the weights must sum to
      ``F(b) - F(a)`` from the marginal's CDF, within ``rtol``.

    The weights are exact (to rounding) when the density is a polynomial on
    each interval, as for a uniform or an integer-parameter Beta density,
    and converge with more points otherwise. For a density smooth on each
    interval, ``d + 1`` points per interval already keep the weights' error
    below a degree-``d`` rule's own; the adaptive loop does not rely on it.

    Parameters
    ----------
    marginal : MarginalProtocol[Array]
        The distribution whose density weights the rule.
    basis_class : Callable[[Array, Backend[Array]], PiecewisePolynomialProtocol[Array]]
        Builds the piecewise basis from nodes, for example
        ``PiecewiseQuadratic``. Any such constructor works; its node-count
        requirements (odd for quadratic, ``3k + 1`` for cubic) apply.
    node_generator : NodeGenerator[Array]
        Generates the nodes, for example ``EquidistantNodeGenerator`` on an
        interval. For an unbounded marginal the interval should hold all but
        a small mass, for example from ``get_bounds_from_marginal``; the
        weights are renormalized to it.
    rtol : float
        Tolerance of both checks. Default ``1e-10``.
    points_per_interval : int
        Gauss-Legendre points per interval to start from. Default 2.
    max_points_per_interval : int
        Largest count tried before raising. Default 64.
    store : bool
        If True, cache the rule of each node count, as
        ``ClenshawCurtisQuadratureRule`` does. Default True, unlike that
        rule, because the weights here come from an adaptive loop and a
        sparse grid asks for the same count from many subspaces.
    """

    def __init__(
        self,
        marginal: MarginalProtocol[Array],
        basis_class: Callable[
            [Array, Backend[Array]], PiecewisePolynomialProtocol[Array]
        ],
        node_generator: NodeGenerator[Array],
        rtol: float = 1e-10,
        points_per_interval: int = 2,
        max_points_per_interval: int = 64,
        store: bool = True,
    ) -> None:
        if rtol <= 0.0:
            raise ValueError(f"rtol must be positive, got {rtol}")
        if not 1 <= points_per_interval <= max_points_per_interval:
            raise ValueError(
                f"points_per_interval ({points_per_interval}) must lie in "
                f"[1, max_points_per_interval={max_points_per_interval}]"
            )
        self._marginal = marginal
        self._basis_class = basis_class
        self._node_generator = node_generator
        self._bkd = node_generator.bkd()
        self._rtol = rtol
        self._start = points_per_interval
        self._max = max_points_per_interval
        self._store = store
        self._cached_rules: Dict[int, Tuple[Array, Array]] = {}

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def _unnormalized_weights(self, nodes: Array, npoints: int) -> Array:
        """``int phi_i p`` by ``npoints`` Gauss-Legendre points per interval."""
        bkd = self._bkd
        unit_nodes, unit_weights = np.polynomial.legendre.leggauss(npoints)
        lefts, rights = nodes[:-1], nodes[1:]
        half = 0.5 * (rights - lefts)
        points = bkd.flatten(
            lefts[:, None] + half[:, None] * (bkd.asarray(unit_nodes)[None, :] + 1.0)
        )
        weights = bkd.flatten(half[:, None] * bkd.asarray(unit_weights)[None, :])
        density = self._marginal.pdf(bkd.reshape(points, (1, -1)))[0]
        values = self._basis_class(nodes, bkd)(points)
        return bkd.dot(values.T, weights * density)

    def _converged_weights(self, nodes: Array) -> Array:
        """Weights to ``rtol`` by the embedded estimate, not normalized."""
        bkd = self._bkd
        count = self._start
        previous = self._unnormalized_weights(nodes, count)
        while count + 2 <= self._max:
            current = self._unnormalized_weights(nodes, count + 2)
            change = bkd.to_float(bkd.max(bkd.abs(current - previous)))
            if change <= self._rtol * bkd.to_float(bkd.max(bkd.abs(current))):
                return current
            previous, count = current, count + 2
        raise ValueError(
            f"the weights did not reach rtol={self._rtol} within "
            f"{self._max} points per interval; the density may have a kink "
            "inside an interval (place a node there) or be too peaked for the "
            "intervals (use more nodes), or raise max_points_per_interval"
        )

    def _check_mass(self, weights: Array, lower: float, upper: float) -> float:
        """The weights' total, checked against the CDF."""
        bkd = self._bkd
        total = bkd.to_float(bkd.sum(weights))
        ends = self._marginal.cdf(bkd.asarray([[lower, upper]]))[0]
        mass = bkd.to_float(ends[1] - ends[0])
        if abs(total - mass) > self._rtol * mass:
            raise ValueError(
                f"the weights integrate the density to {total}, but the CDF "
                f"gives mass {mass} on [{lower}, {upper}]; the density and the "
                "CDF disagree, or the density is not resolved"
            )
        return total

    def __call__(self, npoints: int) -> Tuple[Array, Array]:
        """Nodes (1, npoints) and probability weights (npoints, 1)."""
        if npoints < 1:
            raise ValueError(f"npoints must be positive, got {npoints}")
        if self._store and npoints in self._cached_rules:
            return self._cached_rules[npoints]
        rule = self._compute(npoints)
        if self._store:
            self._cached_rules[npoints] = rule
        return rule

    def _compute(self, npoints: int) -> Tuple[Array, Array]:
        bkd = self._bkd
        if npoints == 1:
            ends = self._node_generator(2)
            return bkd.reshape(0.5 * (ends[0] + ends[1]), (1, 1)), bkd.ones((1, 1))
        nodes = self._node_generator(npoints)
        weights = self._converged_weights(nodes)
        total = self._check_mass(
            weights, bkd.to_float(nodes[0]), bkd.to_float(nodes[-1])
        )
        return bkd.reshape(nodes, (1, -1)), bkd.reshape(weights / total, (-1, 1))
