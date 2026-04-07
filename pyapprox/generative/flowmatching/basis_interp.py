"""Basis interpolators for time-evolving flow matching bases.

Maps time values to ``StieltjesBasisState`` objects.

``IdentityInterpolator`` performs exact lookup at training time nodes
and raises when queried at other times.  It is only valid when the ODE
solver uses exactly the same t schedule as the training quadrature nodes.

``RecurrenceInterpolator`` interpolates the three-term recurrence
coefficients via a pluggable 1-D interpolation strategy and can
evaluate at arbitrary t in [0, 1].  This is the correct choice for
ODE integration at arbitrary time steps.
"""

from typing import Generic, Optional, Protocol, runtime_checkable

from pyapprox.generative.flowmatching.basis_state import StieltjesBasisState
from pyapprox.util.backends.protocols import Array, Backend


# ------------------------------------------------------------------ #
#  1-D scalar interpolation protocol and implementations              #
# ------------------------------------------------------------------ #

@runtime_checkable
class ScalarInterp1DProtocol(Protocol[Array]):
    """Protocol for 1-D scalar interpolation strategies.

    Implementations take a set of nodes and corresponding scalar values
    via ``fit``, then evaluate the interpolant at new query points via
    ``__call__``.
    """

    def fit(self, nodes: Array, values: Array) -> None:
        """Fit the interpolant.

        Parameters
        ----------
        nodes : Array
            Interpolation nodes, shape ``(n,)``.
        values : Array
            Scalar values at nodes, shape ``(n,)``.
        """
        ...

    def __call__(self, t_query: Array) -> Array:
        """Evaluate interpolant at query points.

        Parameters
        ----------
        t_query : Array
            Query points, shape ``(m,)``.

        Returns
        -------
        Array
            Interpolated values, shape ``(m,)``.
        """
        ...


class LagrangeInterp1D(Generic[Array]):
    """Lagrange polynomial interpolation of scalar sequences.

    Uses the barycentric Lagrange formula for numerical stability.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._nodes: Optional[Array] = None
        self._values: Optional[Array] = None
        self._bary_weights: Optional[Array] = None

    def fit(self, nodes: Array, values: Array) -> None:
        """Fit the interpolant.

        Parameters
        ----------
        nodes : Array
            Interpolation nodes, shape ``(n,)``.
        values : Array
            Scalar values at nodes, shape ``(n,)``.
        """
        from pyapprox.surrogates.affine.univariate.lagrange import (
            compute_barycentric_weights,
        )

        self._nodes = nodes
        self._values = values
        self._bary_weights = compute_barycentric_weights(nodes, self._bkd)

    def __call__(self, t_query: Array) -> Array:
        """Evaluate interpolant at query points.

        Parameters
        ----------
        t_query : Array
            Query points, shape ``(m,)``.

        Returns
        -------
        Array
            Interpolated values, shape ``(m,)``.
        """
        if self._nodes is None or self._values is None:
            raise RuntimeError("LagrangeInterp1D has not been fitted")

        from pyapprox.surrogates.affine.univariate.lagrange import (
            univariate_lagrange_polynomial,
        )

        # basis_vals shape: (m, n) where n = len(nodes)
        basis_vals = univariate_lagrange_polynomial(
            self._nodes, t_query, self._bkd, self._bary_weights,
        )
        # dot with values: (m, n) @ (n,) -> (m,)
        return basis_vals @ self._values


# ------------------------------------------------------------------ #
#  Basis state interpolators                                          #
# ------------------------------------------------------------------ #

class IdentityInterpolator(Generic[Array]):
    """Exact-lookup interpolator for basis states.

    Stores basis states at discrete time values and retrieves them
    by exact match. No interpolation is performed between time nodes.

    For ODE integration at arbitrary t, use ``RecurrenceInterpolator``
    instead.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    tol : float
        Tolerance for matching t values.
    """

    def __init__(
        self, bkd: Backend[Array], tol: float = 1e-12
    ) -> None:
        self._bkd = bkd
        self._tol = tol
        self._t_values: Optional[Array] = None
        self._states: Optional[list[StieltjesBasisState[Array]]] = None

    def fit(
        self,
        t_values: Array,
        states: list[StieltjesBasisState[Array]],
    ) -> None:
        """Store (t, state) pairs.

        Parameters
        ----------
        t_values : Array
            Sorted time values, shape ``(n_t,)``.
        states : list[StieltjesBasisState]
            One state per time value.
        """
        if len(t_values) != len(states):
            raise ValueError(
                f"t_values has {len(t_values)} entries but "
                f"states has {len(states)}"
            )
        self._t_values = t_values
        self._states = states

    def __call__(self, t_scalar: float) -> StieltjesBasisState[Array]:
        """Evaluate basis state at t (exact match required).

        Parameters
        ----------
        t_scalar : float
            Time value. Must match a training node within tolerance.

        Returns
        -------
        StieltjesBasisState
            Basis state at the matched training node.

        Raises
        ------
        ValueError
            If no t value within tolerance is found.
        """
        if self._t_values is None or self._states is None:
            raise RuntimeError("IdentityInterpolator has not been fitted")
        diffs = self._bkd.abs(self._t_values - t_scalar)
        idx = int(self._bkd.to_float(self._bkd.argmin(diffs)))
        min_diff = self._bkd.to_float(diffs[idx])
        if min_diff > self._tol:
            raise ValueError(
                f"t={t_scalar} not found in training set "
                f"(nearest diff={min_diff:.2e}). "
                f"IdentityInterpolator requires exact t matches. "
                f"Use RecurrenceInterpolator for arbitrary t evaluation."
            )
        return self._states[idx]

    def t_values(self) -> Array:
        """Return the fitted time values, shape ``(n_t,)``."""
        if self._t_values is None:
            raise RuntimeError("IdentityInterpolator has not been fitted")
        return self._t_values

    def n_states(self) -> int:
        """Number of fitted states."""
        if self._states is None:
            return 0
        return len(self._states)


class RecurrenceInterpolator(Generic[Array]):
    """Interpolates recurrence coefficients between training time nodes.

    At training times ``t_k``, the Stieltjes/Lanczos algorithm produces
    recurrence coefficients ``(alpha_n(t_k), beta_n(t_k))`` for each
    polynomial degree ``n``.  This class interpolates these coefficients
    at arbitrary ``t`` using a pluggable 1-D interpolation strategy,
    yielding a smooth basis evolution.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    strategy_factory : callable, optional
        Factory ``() -> ScalarInterp1DProtocol`` that creates a fresh
        1-D interpolation strategy for each coefficient sequence.
        Defaults to ``LagrangeInterp1D``.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        strategy_factory: Optional[object] = None,
    ) -> None:
        self._bkd = bkd
        if strategy_factory is None:
            self._strategy_factory = lambda: LagrangeInterp1D(bkd)
        else:
            self._strategy_factory = strategy_factory  # type: ignore[assignment]
        self._t_values: Optional[Array] = None
        self._states: Optional[list[StieltjesBasisState[Array]]] = None
        # One interpolation strategy per coefficient sequence
        self._interps: Optional[list[object]] = None

    def fit(
        self,
        t_values: Array,
        states: list[StieltjesBasisState[Array]],
    ) -> None:
        """Store training (t, state) pairs and fit interpolation strategies.

        Parameters
        ----------
        t_values : Array
            Sorted time values, shape ``(n_t,)``.
        states : list[StieltjesBasisState]
            One state per time value.
        """
        if len(t_values) != len(states):
            raise ValueError(
                f"t_values has {len(t_values)} entries but "
                f"states has {len(states)}"
            )
        self._t_values = t_values
        self._states = states

        bkd = self._bkd
        n_t = len(states)
        nterms = states[0].n_basis()

        # Stack recurrence coefficients: (n_t, nterms, 2)
        rcoef_stack = bkd.stack([s.rcoefs() for s in states], axis=0)

        # Fit one interpolation strategy per (degree, alpha/beta) pair
        self._interps = []
        for n in range(nterms):
            for col in range(2):
                interp = self._strategy_factory()
                # values at training nodes for this coefficient sequence
                vals = rcoef_stack[:, n, col]
                interp.fit(t_values, vals)
                self._interps.append(interp)

    def __call__(self, t_scalar: float) -> StieltjesBasisState[Array]:
        """Evaluate interpolated basis state at arbitrary t.

        Parameters
        ----------
        t_scalar : float
            Time value in [0, 1].

        Returns
        -------
        StieltjesBasisState
            Basis state with interpolated recurrence coefficients.
        """
        if self._t_values is None or self._interps is None:
            raise RuntimeError("RecurrenceInterpolator has not been fitted")

        bkd = self._bkd

        # Check if t_scalar matches a training node exactly
        diffs = bkd.abs(self._t_values - t_scalar)
        idx = int(bkd.to_float(bkd.argmin(diffs)))
        if bkd.to_float(diffs[idx]) < 1e-12:
            return self._states[idx]  # type: ignore[index]

        nterms = self._states[0].n_basis()  # type: ignore[index]
        t_arr = bkd.asarray([t_scalar])

        # Evaluate each interpolation strategy
        rcoefs = bkd.zeros((nterms, 2))
        ii = 0
        for n in range(nterms):
            for col in range(2):
                val = self._interps[ii](t_arr)
                rcoefs[n, col] = val[0]
                ii += 1

        return StieltjesBasisState(rcoefs, bkd)

    def t_values(self) -> Array:
        """Return the fitted time values, shape ``(n_t,)``."""
        if self._t_values is None:
            raise RuntimeError("RecurrenceInterpolator has not been fitted")
        return self._t_values

    def n_states(self) -> int:
        """Number of fitted training states."""
        if self._states is None:
            return 0
        return len(self._states)
