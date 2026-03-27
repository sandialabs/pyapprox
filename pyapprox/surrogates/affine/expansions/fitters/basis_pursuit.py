"""Basis Pursuit fitter wrapping existing solver infrastructure."""

from typing import Any, Generic, Optional

from pyapprox.optimization.linear.sparse import BasisPursuitSolver
from pyapprox.surrogates.affine.expansions.fitters.results import (
    SparseResult,
)
from pyapprox.surrogates.affine.protocols import BasisExpansionProtocol
from pyapprox.util.backends.protocols import Array, Backend


class BasisPursuitFitter(Generic[Array]):
    """Basis Pursuit fitter wrapping BasisPursuitSolver.

    Solves: min_c ||c||_1
            subject to: Φc = y

    For sparse recovery when data is exactly reproducible.

    Only supports nqoi=1 (single quantity of interest).

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    options : dict, optional
        Options passed to scipy.optimize.linprog.

    """

    def __init__(
        self,
        bkd: Backend[Array],
        options: Optional[dict[str, Any]] = None,
    ):
        self._bkd = bkd
        self._options = options
        self._solver = BasisPursuitSolver(bkd, options)

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def fit(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
    ) -> SparseResult[Array, BasisExpansionProtocol[Array]]:
        """Fit via Basis Pursuit: min ||c||_1 s.t. Φc = y

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
        SparseResult
            Result containing fitted expansion and sparsity information.

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
                f"BasisPursuitFitter only supports nqoi=1, got {values.shape[0]}"
            )

        # Get basis matrix: (nsamples, nterms)
        Phi = expansion.basis_matrix(samples)

        # Solve using existing solver
        # Solver expects: basis_matrix (nsamples, nterms), values (nsamples, 1)
        params = self._solver.solve(Phi, values.T)  # (nterms, 1)

        # Compute sparsity information
        threshold = 1e-10
        bkd.abs(params[:, 0]) > threshold
        support_list = []
        for ii in range(params.shape[0]):
            if float(bkd.abs(params[ii, 0])) > threshold:
                support_list.append(ii)
        support = bkd.asarray(support_list)
        n_nonzero = len(support_list)

        # Create fitted expansion (immutable pattern)
        fitted_expansion = expansion.with_params(params)

        return SparseResult(
            surrogate=fitted_expansion,
            params=params,
            n_nonzero=n_nonzero,
            support=support,
            regularization_strength=0.0,  # BP has no regularization
        )
