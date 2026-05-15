"""Orthogonal Matching Pursuit (OMP) fitter for sparse recovery."""

from typing import Generic, List, Literal, Optional, Union, overload

from pyapprox.optimization.linear.sparse import OMPSolver, OMPTerminationFlag
from pyapprox.surrogates.affine.expansions.fitters.results import (
    DirectSolverResult,
    OMPResult,
)
from pyapprox.surrogates.affine.protocols import BasisExpansionProtocol
from pyapprox.util.backends.protocols import Array, Backend


class OMPFitter(Generic[Array]):
    """Orthogonal Matching Pursuit fitter for sparse recovery.

    Greedy algorithm that iteratively selects basis terms most correlated
    with the residual. Requires expansion with `basis_matrix()` and
    `with_params()` methods.

    Supports any nqoi. For nqoi=1 the default returns an ``OMPResult``
    with full OMP diagnostics (support, selection order, residual
    history, termination flag). For nqoi>1 the default returns a
    ``DirectSolverResult`` (params + surrogate only) to avoid storing
    per-QoI metadata. Pass ``return_diagnostics=True`` to force an
    ``OMPResult`` (nqoi=1 only) or ``return_diagnostics=False`` to
    force a lightweight ``DirectSolverResult``.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    max_nonzeros : int
        Maximum number of non-zero coefficients. Default: 10.
    rtol : float
        Relative tolerance for residual. Default: 1e-3.
        Algorithm stops when ||residual|| / ||y|| < rtol.

    Raises
    ------
    ValueError
        If max_nonzeros < 1.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        max_nonzeros: int = 10,
        rtol: float = 1e-3,
    ):
        if max_nonzeros < 1:
            raise ValueError(f"max_nonzeros must be >= 1, got {max_nonzeros}")
        self._bkd = bkd
        self._max_nonzeros = max_nonzeros
        self._rtol = rtol

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def max_nonzeros(self) -> int:
        """Return maximum number of non-zero coefficients."""
        return self._max_nonzeros

    def rtol(self) -> float:
        """Return relative tolerance for residual."""
        return self._rtol

    @overload
    def fit(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
        return_diagnostics: Literal[True],
    ) -> OMPResult[Array, BasisExpansionProtocol[Array]]: ...

    @overload
    def fit(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
        return_diagnostics: Literal[False],
    ) -> DirectSolverResult[Array, BasisExpansionProtocol[Array]]: ...

    @overload
    def fit(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
        return_diagnostics: None = ...,
    ) -> Union[
        OMPResult[Array, BasisExpansionProtocol[Array]],
        DirectSolverResult[Array, BasisExpansionProtocol[Array]],
    ]: ...

    def fit(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
        return_diagnostics: Optional[bool] = None,
    ) -> Union[
        OMPResult[Array, BasisExpansionProtocol[Array]],
        DirectSolverResult[Array, BasisExpansionProtocol[Array]],
    ]:
        """Fit via Orthogonal Matching Pursuit.

        Parameters
        ----------
        expansion : BasisExpansionProtocol
            Must have basis_matrix() and with_params() methods.
        samples : Array
            Input samples. Shape: (nvars, nsamples)
        values : Array
            Target values. Shape: (nqoi, nsamples) or (nsamples,) for nqoi=1.
        return_diagnostics : bool, optional
            If True, return OMPResult with full diagnostics (nqoi=1 only).
            If False, return DirectSolverResult (params + surrogate only).
            If None (default), True for nqoi=1, False for nqoi>1.

        Returns
        -------
        OMPResult or DirectSolverResult
            OMPResult when diagnostics requested, DirectSolverResult otherwise.

        Raises
        ------
        ValueError
            If return_diagnostics=True and nqoi > 1.
        """
        bkd = self._bkd

        # Handle 1D values
        if values.ndim == 1:
            values = bkd.reshape(values, (1, -1))

        nqoi = values.shape[0]

        # Auto-select: diagnostics for nqoi=1, lightweight for nqoi>1
        if return_diagnostics is None:
            return_diagnostics = nqoi == 1

        if return_diagnostics and nqoi != 1:
            raise ValueError(
                "return_diagnostics=True requires nqoi=1, "
                f"got nqoi={nqoi}"
            )

        # Get basis matrix: (nsamples, nterms)
        Phi = expansion.basis_matrix(samples)

        if not return_diagnostics:
            # Delegate to OMPSolver which handles any nqoi via
            # SingleQoiSolverMixin looping
            solver = OMPSolver(
                bkd,
                max_nonzeros=self._max_nonzeros,
                rtol=self._rtol,
            )
            coef = solver.solve(Phi, values.T)
            # Create fitted expansion (immutable pattern)
            fitted_expansion = expansion.with_params(coef)
            return DirectSolverResult(
                surrogate=fitted_expansion,
                params=coef,
            )

        return self._fit_with_diagnostics(expansion, Phi, values, bkd)

    def _fit_with_diagnostics(
        self,
        expansion: BasisExpansionProtocol[Array],
        Phi: Array,
        values: Array,
        bkd: Backend[Array],
    ) -> OMPResult[Array, BasisExpansionProtocol[Array]]:
        """Run OMP for nqoi=1 and collect full diagnostics."""
        nsamples, nterms = Phi.shape

        # Transpose values for solver: (nsamples, 1)
        y = values.T

        # Initialize
        coef = bkd.zeros((nterms, 1))
        residual = bkd.copy(y)
        active_indices: List[int] = []
        selection_order: List[int] = []
        residual_history: List[float] = []
        initial_norm = float(bkd.norm(y))
        termination_flag: Optional[OMPTerminationFlag] = None

        # Precompute column norms for correlation normalization
        col_norms = bkd.norm(Phi, axis=0)
        # Avoid division by zero
        col_norms = bkd.where(col_norms > 1e-14, col_norms, bkd.ones_like(col_norms))

        # Store the active coefficients for final assignment
        active_coef: Optional[Array] = None

        for iteration in range(self._max_nonzeros):
            # Find column most correlated with residual
            correlations = bkd.abs(bkd.dot(Phi.T, residual)[:, 0]) / col_norms

            # Zero out already selected columns
            for idx in active_indices:
                correlations = bkd.where(
                    bkd.arange(nterms) == idx,
                    bkd.zeros_like(correlations),
                    correlations,
                )

            best_idx = bkd.to_int(bkd.argmax(correlations))

            # Check if column is linearly dependent (correlation ~0)
            if bkd.to_float(correlations[best_idx]) < 1e-14:
                termination_flag = OMPTerminationFlag.COLUMNS_DEPENDENT
                break

            active_indices.append(best_idx)
            selection_order.append(best_idx)

            # Solve least squares on active columns
            active_matrix = Phi[:, active_indices]
            active_coef = bkd.lstsq(active_matrix, y, rcond=None)

            # Update residual
            residual = y - bkd.dot(active_matrix, active_coef)

            # Record residual norm
            residual_norm = float(bkd.norm(residual))
            residual_history.append(residual_norm)

            # Check convergence
            if initial_norm > 1e-14:
                rel_residual = residual_norm / initial_norm
                if rel_residual < self._rtol:
                    termination_flag = OMPTerminationFlag.RESIDUAL_TOLERANCE
                    break
        else:
            termination_flag = OMPTerminationFlag.MAX_NONZEROS

        # Build full coefficient array
        if active_coef is not None:
            for ii, idx in enumerate(active_indices):
                coef[idx, 0] = active_coef[ii, 0]

        # Compute support (same as selection_order for OMP)
        support = bkd.asarray(active_indices)

        # Create fitted expansion (immutable pattern)
        fitted_expansion = expansion.with_params(coef)

        return OMPResult(
            surrogate=fitted_expansion,
            params=coef,
            n_nonzero=len(active_indices),
            support=support,
            selection_order=bkd.asarray(selection_order),
            residual_history=bkd.asarray(residual_history),
            termination_flag=termination_flag,
        )
