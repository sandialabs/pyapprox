"""Basis interpolators for time-evolving flow matching bases.

Maps time values to ``StieltjesBasisState`` objects.

``IdentityInterpolator`` performs exact lookup at training time nodes
and raises when queried at other times.  The ODE solver must use
exactly the same t schedule as the training quadrature nodes.
"""

from typing import Generic, Optional

from pyapprox.generative.flowmatching.basis_state import StieltjesBasisState
from pyapprox.util.backends.protocols import Array, Backend


class IdentityInterpolator(Generic[Array]):
    """Exact-lookup interpolator for basis states.

    Stores basis states at discrete time values and retrieves them
    by exact match. No interpolation is performed between time nodes.

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
                f"Evaluation is only valid at training time nodes."
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
