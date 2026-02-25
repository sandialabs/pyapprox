"""Basis Pursuit Denoising (BPDN) fitter wrapping existing solver."""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.optimization.linear.sparse import BasisPursuitDenoisingSolver
from pyapprox.surrogates.affine.protocols import BasisExpansionProtocol
from pyapprox.surrogates.affine.expansions.fitters.results import (
    SparseResult,
)


class BPDNFitter(Generic[Array]):
    """Basis Pursuit Denoising (BPDN) fitter.

    Wraps BasisPursuitDenoisingSolver. Requires expansion with `basis_matrix()`
    and `with_params()` methods.

    Solves: min_c (1/2)||Phi c - y||_2^2 + lambda ||c||_1

    This is the penalized (Lagrangian) form of L1-regularized least squares,
    also known as LASSO regression.

    Only supports nqoi=1 (single quantity of interest).

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    penalty : float
        L1 penalty parameter (lambda). Must be positive.
    max_iter : int
        Maximum iterations for coordinate descent. Default: 1000.
    tol : float
        Convergence tolerance. Default: 1e-4.

    Raises
    ------
    ValueError
        If penalty <= 0.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        penalty: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ):
        if penalty <= 0:
            raise ValueError(f"penalty must be positive, got {penalty}")
        self._bkd = bkd
        self._penalty = penalty
        self._max_iter = max_iter
        self._tol = tol
        self._solver = BasisPursuitDenoisingSolver(
            bkd, penalty=penalty, max_iter=max_iter, tol=tol
        )

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def penalty(self) -> float:
        """Return L1 penalty parameter (lambda)."""
        return self._penalty

    def fit(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
    ) -> SparseResult[Array, BasisExpansionProtocol[Array]]:
        """Fit via BPDN: min_c (1/2)||Phi c - y||_2^2 + lambda ||c||_1

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
                f"BPDNFitter only supports nqoi=1, got {values.shape[0]}"
            )

        # Get basis matrix: (nsamples, nterms)
        Phi = expansion.basis_matrix(samples)

        # Solve using existing solver
        # Solver expects: basis_matrix (nsamples, nterms), values (nsamples, 1)
        params = self._solver.solve(Phi, values.T)  # (nterms, 1)

        # Compute sparsity information
        tol = 1e-10
        abs_params = bkd.abs(params[:, 0])
        nonzero_mask = abs_params > tol
        support = bkd.where(nonzero_mask)[0]
        n_nonzero = support.shape[0]

        # Create fitted expansion (immutable pattern)
        fitted_expansion = expansion.with_params(params)

        return SparseResult(
            surrogate=fitted_expansion,
            params=params,
            n_nonzero=n_nonzero,
            support=support,
            regularization_strength=self._penalty,
        )
