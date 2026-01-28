"""Least squares fitter wrapping existing solver infrastructure."""

from typing import Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.optimization.linear import LeastSquaresSolver
from pyapprox.typing.surrogates.affine.protocols import BasisExpansionProtocol
from pyapprox.typing.optimization.fitting.results import DirectSolverResult


class LeastSquaresFitter(Generic[Array]):
    """Least squares fitter wrapping existing LeastSquaresSolver.

    Requires expansion with `basis_matrix()` and `with_params()` methods.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    rcond : float, optional
        Cutoff for small singular values. Default: machine precision.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        rcond: Optional[float] = None,
    ):
        self._bkd = bkd
        self._rcond = rcond
        self._solver = LeastSquaresSolver(bkd, rcond=rcond)

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def fit(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
    ) -> DirectSolverResult[Array, BasisExpansionProtocol[Array]]:
        """Fit via least squares: min_c ||Phi c - y||_2^2

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
