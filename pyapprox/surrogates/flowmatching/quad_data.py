"""Quadrature data container and builders for flow matching.

``FlowMatchingQuadData`` is a flat container holding pre-assembled
quadrature points and weights for the CFM loss integral.

``build_flow_matching_quad_data`` constructs this container from
pluggable 1-D quadrature rules for the time axis and the source
distribution axis, plus a forward map ``x0 -> x1``.  This decouples
the choice of time discretisation (Gauss-Legendre, uniform, ODE step
schedule, …) from the rest of the pipeline.
"""

from typing import Callable, Generic, Optional, Tuple

from pyapprox.util.backends.protocols import Array, Backend

QuadRule1D = Callable[[int], Tuple[Array, Array]]


class FlowMatchingQuadData(Generic[Array]):
    """Container for flow matching quadrature data.

    Stores the quadrature points (t, x0, x1) and weights needed to
    approximate the CFM loss integral. Optionally includes conditioning
    variables c.

    Parameters
    ----------
    t : Array
        Time quadrature points, shape ``(1, n_quad)``.
    x0 : Array
        Source sample quadrature points, shape ``(d, n_quad)``.
    x1 : Array
        Target sample quadrature points, shape ``(d, n_quad)``.
    weights : Array
        Quadrature weights, shape ``(n_quad,)``.
    bkd : Backend[Array]
        Computational backend.
    c : Array, optional
        Conditioning variables, shape ``(m, n_quad)``.
    """

    def __init__(
        self,
        t: Array,
        x0: Array,
        x1: Array,
        weights: Array,
        bkd: Backend[Array],
        c: Optional[Array] = None,
    ) -> None:
        self._t = t
        self._x0 = x0
        self._x1 = x1
        self._weights = weights
        self._bkd = bkd
        self._c = c
        self._validate()

    def _validate(self) -> None:
        """Check shape consistency of all fields."""
        t_shape = self._t.shape
        x0_shape = self._x0.shape
        x1_shape = self._x1.shape
        w_shape = self._weights.shape

        if len(t_shape) != 2 or t_shape[0] != 1:
            raise ValueError(f"t must have shape (1, n_quad), got {t_shape}")
        n_quad = t_shape[1]

        if len(x0_shape) != 2 or x0_shape[1] != n_quad:
            raise ValueError(f"x0 must have shape (d, {n_quad}), got {x0_shape}")
        if len(x1_shape) != 2 or x1_shape[1] != n_quad:
            raise ValueError(f"x1 must have shape (d, {n_quad}), got {x1_shape}")
        if x0_shape[0] != x1_shape[0]:
            raise ValueError(
                f"x0 and x1 must have same first dimension, "
                f"got {x0_shape[0]} and {x1_shape[0]}"
            )
        if len(w_shape) != 1 or w_shape[0] != n_quad:
            raise ValueError(f"weights must have shape ({n_quad},), got {w_shape}")
        if self._c is not None:
            c_shape = self._c.shape
            if len(c_shape) != 2 or c_shape[1] != n_quad:
                raise ValueError(f"c must have shape (m, {n_quad}), got {c_shape}")

    def t(self) -> Array:
        """Time quadrature points, shape ``(1, n_quad)``."""
        return self._t

    def x0(self) -> Array:
        """Source quadrature points, shape ``(d, n_quad)``."""
        return self._x0

    def x1(self) -> Array:
        """Target quadrature points, shape ``(d, n_quad)``."""
        return self._x1

    def weights(self) -> Array:
        """Quadrature weights, shape ``(n_quad,)``."""
        return self._weights

    def c(self) -> Optional[Array]:
        """Conditioning variables, shape ``(m, n_quad)`` or None."""
        return self._c

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def n_quad(self) -> int:
        """Number of quadrature points."""
        return self._t.shape[1]

    def d(self) -> int:
        """Spatial dimension (number of state variables)."""
        return self._x0.shape[0]

    def m(self) -> int:
        """Conditioning dimension, or 0 if no conditioning."""
        if self._c is None:
            return 0
        return self._c.shape[0]


# ------------------------------------------------------------------ #
#  Built-in 1-D quadrature rule factories                             #
# ------------------------------------------------------------------ #

def gauss_legendre_rule(bkd: Backend[Array]) -> QuadRule1D:
    """Gauss-Legendre quadrature on [0, 1].

    Returns a callable ``(n) -> (nodes, weights)`` where nodes has
    shape ``(n,)`` and weights has shape ``(n,)``.
    """
    from pyapprox.probability import UniformMarginal
    from pyapprox.surrogates.affine.univariate import create_basis_1d

    basis = create_basis_1d(UniformMarginal(0.0, 1.0, bkd), bkd)

    def rule(n: int) -> Tuple[Array, Array]:
        basis.set_nterms(n)
        pts, wts = basis.gauss_quadrature_rule(n)
        return pts[0, :], bkd.flatten(wts)

    return rule


def gauss_hermite_rule(bkd: Backend[Array]) -> QuadRule1D:
    """Gauss-Hermite quadrature for N(0, 1).

    Returns a callable ``(n) -> (nodes, weights)`` where nodes has
    shape ``(n,)`` and weights has shape ``(n,)``.
    """
    from pyapprox.probability import GaussianMarginal
    from pyapprox.surrogates.affine.univariate import create_basis_1d

    basis = create_basis_1d(GaussianMarginal(0.0, 1.0, bkd), bkd)

    def rule(n: int) -> Tuple[Array, Array]:
        basis.set_nterms(n)
        pts, wts = basis.gauss_quadrature_rule(n)
        return pts[0, :], bkd.flatten(wts)

    return rule


def uniform_rule(
    lo: float, hi: float, bkd: Backend[Array]
) -> QuadRule1D:
    """Equispaced nodes on [lo, hi] with trapezoidal weights.

    Returns a callable ``(n) -> (nodes, weights)`` where nodes has
    shape ``(n,)`` and weights has shape ``(n,)``.
    """

    def rule(n: int) -> Tuple[Array, Array]:
        nodes = bkd.linspace(lo, hi, n)
        h = (hi - lo) / (n - 1) if n > 1 else (hi - lo)
        wts = bkd.full((n,), h)
        # Trapezoidal: half weight at endpoints
        if n > 1:
            wts[0] = h / 2.0
            wts[n - 1] = h / 2.0
        return nodes, wts

    return rule


def fixed_nodes_rule(
    nodes: Array, weights: Array
) -> QuadRule1D:
    """Rule from pre-computed nodes and weights.

    The ``n`` argument to the returned callable is ignored; the same
    nodes and weights are always returned.
    """

    def rule(n: int) -> Tuple[Array, Array]:
        return nodes, weights

    return rule


def mc_rule(bkd: Backend[Array], seed: int = 0) -> QuadRule1D:
    """Monte Carlo quadrature for N(0, 1).

    Returns a callable ``(n) -> (nodes, weights)`` where nodes are
    i.i.d. draws from N(0, 1) and weights are uniform 1/n.
    """
    import numpy as np

    def rule(n: int) -> Tuple[Array, Array]:
        rng = np.random.RandomState(seed)
        nodes = bkd.asarray(rng.randn(n))
        wts = bkd.full((n,), 1.0 / n)
        return nodes, wts

    return rule


# ------------------------------------------------------------------ #
#  Builders                                                           #
# ------------------------------------------------------------------ #

def build_flow_matching_quad_data(
    t_rule: QuadRule1D,
    x0_rule: QuadRule1D,
    forward_map: Callable[[Array], Array],
    n_t: int,
    n_x: int,
    bkd: Backend[Array],
) -> FlowMatchingQuadData[Array]:
    """Build ``FlowMatchingQuadData`` from pluggable 1-D rules.

    Constructs a tensor product of ``t_rule(n_t)`` and ``x0_rule(n_x)``,
    applies ``forward_map`` to obtain ``x1``, and returns the flat
    quad data container.  Uses **deterministic coupling**: x1 = F(x0).

    Parameters
    ----------
    t_rule : QuadRule1D
        Callable ``(n) -> (nodes, weights)`` for the time axis.
    x0_rule : QuadRule1D
        Callable ``(n) -> (nodes, weights)`` for the source axis.
    forward_map : callable
        Maps source samples to target: ``x1 = forward_map(x0)``
        where both have shape ``(1, n_quad)``.
    n_t : int
        Number of time quadrature points.
    n_x : int
        Number of source quadrature points.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    FlowMatchingQuadData
        Flat tensor product quad data with ``n_t * n_x`` points.
    """
    t_nodes, t_wts = t_rule(n_t)
    x_nodes, x_wts = x0_rule(n_x)

    # Tensor product via meshgrid
    t_grid, x_grid = bkd.meshgrid(t_nodes, x_nodes, indexing="ij")
    t_flat = bkd.reshape(bkd.flatten(t_grid), (1, -1))
    x0_flat = bkd.reshape(bkd.flatten(x_grid), (1, -1))

    # Outer product of weights
    w_flat = bkd.flatten(
        bkd.reshape(t_wts, (-1, 1)) * x_wts
    )

    x1_flat = forward_map(x0_flat)

    return FlowMatchingQuadData(
        t=t_flat, x0=x0_flat, x1=x1_flat, weights=w_flat, bkd=bkd,
    )


# ------------------------------------------------------------------ #
#  Pair rules: (x0, x1) quadrature over p0(x0) * p1(x1)              #
# ------------------------------------------------------------------ #

"""Callable that returns (x0, x1, weights) arrays for n_pairs points.

x0 shape ``(1, n)``, x1 shape ``(1, n)``, weights shape ``(n,)``.
The rule integrates over the **product** measure p0(x0) * p1(x1),
which can be achieved by any 2D quadrature (tensor product of two
1D rules, MC pairs, Sobol, etc.).
"""
PairRule = Callable[[int], Tuple[Array, Array, Array]]


def tensor_product_pair_rule(
    x0_rule: QuadRule1D,
    x1_rule: QuadRule1D,
    forward_map: Callable[[Array], Array],
    bkd: Backend[Array],
) -> PairRule:
    """Tensor-product (x0, x1) rule with independent quadrature.

    Builds ``n_x0 × n_x1`` pairs from separate 1D rules for x0 and
    x1.  The x1 rule produces source-distribution nodes that are
    mapped through ``forward_map`` to target values.

    The returned callable takes a single ``n`` argument interpreted
    as the 1D resolution: ``n_x0 = n_x1 = n``, giving ``n²`` pairs.

    Parameters
    ----------
    x0_rule : QuadRule1D
        1D rule for x0 (source distribution).
    x1_rule : QuadRule1D
        1D rule for xi1 (source distribution seed for x1).
    forward_map : callable
        ``x1 = forward_map(xi1)`` with shape ``(1, n) -> (1, n)``.
    bkd : Backend[Array]
        Computational backend.
    """

    def rule(n: int) -> Tuple[Array, Array, Array]:
        x0_nodes, x0_wts = x0_rule(n)
        xi1_nodes, xi1_wts = x1_rule(n)
        x1_nodes = bkd.flatten(
            forward_map(bkd.reshape(xi1_nodes, (1, -1)))
        )

        # Tensor product: n × n pairs
        x0_flat = bkd.reshape(bkd.repeat(x0_nodes, n), (1, -1))
        x1_flat = bkd.reshape(bkd.tile(x1_nodes, (n,)), (1, -1))
        w_flat = bkd.repeat(x0_wts, n) * bkd.tile(xi1_wts, (n,))
        return x0_flat, x1_flat, w_flat

    return rule


def mc_pair_rule(
    forward_map: Callable[[Array], Array],
    bkd: Backend[Array],
    seed: int = 0,
) -> PairRule:
    """Monte Carlo (x0, x1) rule with independent draws.

    Draws x0 ~ N(0,1) and xi1 ~ N(0,1) independently, maps
    x1 = forward_map(xi1).  Weights are uniform 1/n.

    Parameters
    ----------
    forward_map : callable
        ``x1 = forward_map(xi1)`` with shape ``(1, n) -> (1, n)``.
    bkd : Backend[Array]
        Computational backend.
    seed : int
        Random seed.
    """
    import numpy as np

    def rule(n: int) -> Tuple[Array, Array, Array]:
        rng = np.random.RandomState(seed)
        x0 = bkd.asarray(rng.randn(1, n))
        xi1 = bkd.asarray(rng.randn(1, n))
        x1 = forward_map(xi1)
        w = bkd.full((n,), 1.0 / n)
        return x0, x1, w

    return rule


def pushforward_pair_rule(
    x0_rule: QuadRule1D,
    forward_map: Callable[[Array], Array],
    bkd: Backend[Array],
) -> PairRule:
    """Paired (x0, x1) rule with pushforward coupling x1 = F(x0).

    This is the original approach: a single 1D rule for x0, with
    x1 = forward_map(x0).  No independent averaging.

    Parameters
    ----------
    x0_rule : QuadRule1D
        1D rule for x0.
    forward_map : callable
        ``x1 = forward_map(x0)`` with shape ``(1, n) -> (1, n)``.
    bkd : Backend[Array]
        Computational backend.
    """

    def rule(n: int) -> Tuple[Array, Array, Array]:
        x0_nodes, x0_wts = x0_rule(n)
        x0_flat = bkd.reshape(x0_nodes, (1, -1))
        x1_flat = forward_map(x0_flat)
        return x0_flat, x1_flat, x0_wts

    return rule


def build_independent_quad_data(
    t_rule: QuadRule1D,
    pair_rule: PairRule,
    n_t: int,
    n_pairs: int,
    bkd: Backend[Array],
) -> FlowMatchingQuadData[Array]:
    """Build ``FlowMatchingQuadData`` as t × (x0, x1) tensor product.

    The time axis uses a structured 1D rule.  The spatial axis uses
    a ``PairRule`` that returns independent (x0, x1, weights) for
    ``n_pairs`` points — this can be a tensor-product of two 1D rules,
    Monte Carlo draws, or any other 2D quadrature over p0(x0) * p1(x1).

    Parameters
    ----------
    t_rule : QuadRule1D
        Callable ``(n) -> (nodes, weights)`` for the time axis.
    pair_rule : PairRule
        Callable ``(n) -> (x0, x1, weights)`` for the spatial axis.
        ``x0`` shape ``(1, n_pairs)``, ``x1`` shape ``(1, n_pairs)``,
        ``weights`` shape ``(n_pairs,)``.
    n_t : int
        Number of time quadrature points.
    n_pairs : int
        Argument passed to ``pair_rule``.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    FlowMatchingQuadData
        Flat tensor product with ``n_t * n_actual_pairs`` points,
        where ``n_actual_pairs`` is the number of pairs returned by
        the pair rule (may differ from ``n_pairs`` for tensor-product
        pair rules where ``n_pairs`` is the 1D resolution).
    """
    t_nodes, t_wts = t_rule(n_t)
    x0_pairs, x1_pairs, pair_wts = pair_rule(n_pairs)
    n_p = x0_pairs.shape[1]
    n_total = n_t * n_p

    # t × pairs tensor product
    t_flat = bkd.reshape(bkd.repeat(t_nodes, n_p), (1, n_total))
    x0_flat = bkd.tile(x0_pairs, (1, n_t))
    x1_flat = bkd.tile(x1_pairs, (1, n_t))
    w_flat = bkd.tile(pair_wts, (n_t,)) * bkd.repeat(t_wts, n_p)

    return FlowMatchingQuadData(
        t=t_flat, x0=x0_flat, x1=x1_flat, weights=w_flat, bkd=bkd,
    )
