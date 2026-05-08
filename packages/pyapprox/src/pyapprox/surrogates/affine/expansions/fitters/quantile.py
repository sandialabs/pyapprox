"""Quantile regression fitter wrapping existing solver infrastructure."""

from typing import Any, Generic, Optional

from pyapprox.optimization.linear import QuantileRegressionSolver
from pyapprox.surrogates.affine.expansions.fitters.results import (
    DirectSolverResult,
)
from pyapprox.surrogates.affine.protocols import BasisExpansionProtocol
from pyapprox.util.backends.protocols import Array, Backend


class QuantileFitter(Generic[Array]):
    """Quantile regression fitter wrapping QuantileRegressionSolver.

    Solves: min_c Σ_i ρ_τ(y_i - Φ_i c)

    where ρ_τ(u) = u(τ - I(u < 0)) is the check function (pinball loss).

    Only supports nqoi=1 (single quantity of interest).

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    quantile : float
        Target quantile τ in [0, 1]. Default: 0.5 (median).
    options : dict, optional
        Options passed to scipy.optimize.linprog.

    Raises
    ------
    ValueError
        If quantile not in [0, 1].
    """

    def __init__(
        self,
        bkd: Backend[Array],
        quantile: float = 0.5,
        options: Optional[dict[str, Any]] = None,
    ):
        if not 0 <= quantile <= 1:
            raise ValueError(f"quantile must be in [0, 1], got {quantile}")
        self._bkd = bkd
        self._quantile = quantile
        self._options = options
        self._solver = QuantileRegressionSolver(bkd, quantile, options)

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def quantile(self) -> float:
        """Return target quantile."""
        return self._quantile

    def fit(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
    ) -> DirectSolverResult[Array, BasisExpansionProtocol[Array]]:
        """Fit via quantile regression: min_c Σ_i ρ_τ(y_i - Φ_i c)

        Parameters
        ----------
        expansion : BasisExpansionProtocol
            Must have basis_matrix() and with_params() methods.
        samples : Array
            Input samples. Shape: (nvars, nsamples)
        values : Array
            Target values. Shape: (1, nsamples) or (nsamples,).
            Only nqoi=1 supported.

        Returns
        -------
        DirectSolverResult
            Result containing fitted expansion.

        Raises
        ------
        ValueError
            If nqoi > 1.
        """
        bkd = self._bkd

        # Handle 1D values
        if values.ndim == 1:
            values = bkd.reshape(values, (1, -1))

        # Validate single QoI
        if values.shape[0] != 1:
            raise ValueError(
                f"QuantileFitter only supports nqoi=1, got {values.shape[0]}"
            )

        # Get basis matrix: (nsamples, nterms)
        Phi = expansion.basis_matrix(samples)

        # Solve using existing solver
        # Solver expects: basis_matrix (nsamples, nterms), values (nsamples, 1)
        params = self._solver.solve(Phi, values.T)  # (nterms, 1)

        # Create fitted expansion (immutable pattern)
        fitted_expansion = expansion.with_params(params)

        return DirectSolverResult(
            surrogate=fitted_expansion,
            params=params,
        )
