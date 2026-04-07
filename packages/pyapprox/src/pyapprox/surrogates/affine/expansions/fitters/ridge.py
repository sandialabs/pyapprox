"""Ridge fitter wrapping existing solver infrastructure."""

from typing import Generic

from pyapprox.optimization.linear import RidgeRegressionSolver
from pyapprox.surrogates.affine.expansions.fitters.results import (
    DirectSolverResult,
)
from pyapprox.surrogates.affine.protocols import BasisExpansionProtocol
from pyapprox.util.backends.protocols import Array, Backend


class RidgeFitter(Generic[Array]):
    """Ridge regression (L2-regularized) fitter.

    Wraps RidgeRegressionSolver. Requires expansion with `basis_matrix()`
    and `with_params()` methods.

    Solves: min_c ||Phi c - y||_2^2 + alpha ||c||_2^2

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    alpha : float
        L2 regularization strength. Must be positive.

    Raises
    ------
    ValueError
        If alpha <= 0.
    """

    def __init__(self, bkd: Backend[Array], alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self._bkd = bkd
        self._alpha = alpha
        self._solver = RidgeRegressionSolver(bkd, alpha=alpha)

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def alpha(self) -> float:
        """Return regularization strength."""
        return self._alpha

    def fit(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
    ) -> DirectSolverResult[Array, BasisExpansionProtocol[Array]]:
        """Fit via ridge regression: min_c ||Phi c - y||_2^2 + alpha ||c||_2^2

        Parameters
        ----------
        expansion : BasisExpansionProtocol
            Must have basis_matrix() and with_params() methods.
        samples : Array
            Input samples. Shape: (nvars, nsamples)
        values : Array
            Target values. Shape: (nqoi, nsamples) or (nsamples,) for nqoi=1.

        Returns
        -------
        DirectSolverResult
            Result containing fitted expansion.
        """
        # Handle 1D values
        if values.ndim == 1:
            values = self._bkd.reshape(values, (1, -1))

        # Get basis matrix: (nsamples, nterms)
        Phi = expansion.basis_matrix(samples)

        # Solve using existing solver
        # Solver expects: basis_matrix (nsamples, nterms), values (nsamples, nqoi)
        params = self._solver.solve(Phi, values.T)  # (nterms, nqoi)

        # Create fitted expansion (immutable pattern)
        fitted_expansion = expansion.with_params(params)

        return DirectSolverResult(
            surrogate=fitted_expansion,
            params=params,
        )
