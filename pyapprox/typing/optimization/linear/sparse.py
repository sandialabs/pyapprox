"""Sparse solvers for basis expansion fitting.

This module provides solvers for sparse coefficient recovery:
- OMPSolver: Orthogonal Matching Pursuit (greedy)
- BasisPursuitSolver: L1 minimization via linear programming
"""

from typing import Generic, Optional, Tuple
from enum import IntEnum

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.optimization.linear.base import (
    LinearSystemSolver,
    SingleQoiSolverMixin,
)


class OMPTerminationFlag(IntEnum):
    """Termination conditions for OMP algorithm."""

    RESIDUAL_TOLERANCE = 0  # Residual below tolerance
    MAX_NONZEROS = 1  # Maximum sparsity reached
    COLUMNS_DEPENDENT = 2  # Selected columns not independent


class OMPSolver(
    SingleQoiSolverMixin, LinearSystemSolver[Array], Generic[Array]
):
    """Orthogonal Matching Pursuit solver for sparse solutions.

    Greedy algorithm that iteratively:
    1. Finds the column most correlated with residual
    2. Adds it to the active set
    3. Solves least squares on active columns
    4. Updates residual

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    max_nonzeros : int
        Maximum number of non-zero coefficients. Default: 10.
    rtol : float
        Relative tolerance for residual. Default: 1e-3.
    verbosity : int
        Verbosity level. Default: 0.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        max_nonzeros: int = 10,
        rtol: float = 1e-3,
        verbosity: int = 0,
    ):
        super().__init__(bkd)
        self._max_nonzeros = max_nonzeros
        self._rtol = rtol
        self._verbosity = verbosity
        self._termination_flag: Optional[OMPTerminationFlag] = None

    def set_max_nonzeros(self, max_nonzeros: int) -> None:
        """Set maximum number of non-zero coefficients.

        Parameters
        ----------
        max_nonzeros : int
            Maximum sparsity level.
        """
        self._max_nonzeros = max_nonzeros

    def set_rtol(self, rtol: float) -> None:
        """Set relative tolerance for residual.

        Parameters
        ----------
        rtol : float
            Relative tolerance.
        """
        self._rtol = rtol

    @property
    def termination_flag(self) -> Optional[OMPTerminationFlag]:
        """Return termination flag from last solve."""
        return self._termination_flag

    def _solve(self, basis_matrix: Array, values: Array) -> Array:
        """Solve using Orthogonal Matching Pursuit.

        Parameters
        ----------
        basis_matrix : Array
            Basis matrix Φ. Shape: (nsamples, nterms)
        values : Array
            Target values y. Shape: (nsamples, 1)

        Returns
        -------
        Array
            Sparse coefficients c. Shape: (nterms, 1)
        """
        self._validate_single_qoi(values)

        nsamples, nterms = basis_matrix.shape
        bkd = self._bkd

        # Initialize
        coef = bkd.zeros((nterms, 1))
        residual = bkd.copy(values)
        active_indices: list = []
        initial_norm = bkd.norm(values)

        # Precompute column norms for correlation
        col_norms = bkd.norm(basis_matrix, axis=0)
        # Avoid division by zero
        col_norms = bkd.where(
            col_norms > 1e-14, col_norms, bkd.ones_like(col_norms)
        )

        for iteration in range(self._max_nonzeros):
            # Find column most correlated with residual
            correlations = bkd.abs(
                bkd.dot(basis_matrix.T, residual)[:, 0]
            ) / col_norms
            # Zero out already selected columns
            for idx in active_indices:
                correlations[idx] = 0.0

            best_idx = int(bkd.argmax(correlations))

            # Check if column is linearly dependent
            if correlations[best_idx] < 1e-14:
                self._termination_flag = OMPTerminationFlag.COLUMNS_DEPENDENT
                break

            active_indices.append(best_idx)

            # Solve least squares on active columns
            active_matrix = basis_matrix[:, active_indices]
            active_coef = bkd.lstsq(active_matrix, values, rcond=None)

            # Update residual
            residual = values - bkd.dot(active_matrix, active_coef)

            # Check convergence
            rel_residual = bkd.norm(residual) / initial_norm
            if rel_residual < self._rtol:
                self._termination_flag = OMPTerminationFlag.RESIDUAL_TOLERANCE
                # Set active coefficients
                for ii, idx in enumerate(active_indices):
                    coef[idx, 0] = active_coef[ii, 0]
                break

            if self._verbosity > 0:
                print(
                    f"OMP iter {iteration}: selected {best_idx}, "
                    f"rel_residual = {float(rel_residual):.2e}"
                )
        else:
            self._termination_flag = OMPTerminationFlag.MAX_NONZEROS

        # Set active coefficients
        for ii, idx in enumerate(active_indices):
            coef[idx, 0] = active_coef[ii, 0]

        return coef


class BasisPursuitSolver(
    SingleQoiSolverMixin, LinearSystemSolver[Array], Generic[Array]
):
    """Basis Pursuit solver for sparse recovery via L1 minimization.

    Solves: min_c ||c||_1
            subject to: Φc = y

    Uses linear programming formulation.

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
        options: Optional[dict] = None,
    ):
        super().__init__(bkd)
        self._options = options or {}

    def set_options(self, options: dict) -> None:
        """Set solver options.

        Parameters
        ----------
        options : dict
            Options for scipy.optimize.linprog.
        """
        self._options = options

    def _solve(self, basis_matrix: Array, values: Array) -> Array:
        """Solve using Basis Pursuit (L1 minimization).

        Parameters
        ----------
        basis_matrix : Array
            Basis matrix Φ. Shape: (nsamples, nterms)
        values : Array
            Target values y. Shape: (nsamples, 1)

        Returns
        -------
        Array
            Sparse coefficients c. Shape: (nterms, 1)
        """
        self._validate_single_qoi(values)

        # Import scipy for LP solver
        from scipy.optimize import linprog
        from scipy import sparse

        bkd = self._bkd
        A = bkd.to_numpy(basis_matrix)
        b = bkd.to_numpy(values)[:, 0]
        nterms = A.shape[1]

        # Reformulate: c = u - v where u, v >= 0
        # min 1^T u + 1^T v
        # s.t. A(u - v) = b, u >= 0, v >= 0
        c_obj = [1.0] * (2 * nterms)  # coefficients of objective

        # Equality constraints: [A, -A] @ [u; v] = b
        A_eq = sparse.hstack([sparse.csr_matrix(A), sparse.csr_matrix(-A)])
        b_eq = b

        result = linprog(
            c_obj,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(0, None),
            method="highs",
            options=self._options,
        )

        if not result.success:
            raise RuntimeError(f"Basis Pursuit failed: {result.message}")

        # Extract solution: c = u - v
        u = result.x[:nterms]
        v = result.x[nterms:]
        coef = u - v

        return bkd.asarray(coef.reshape(-1, 1))


class BasisPursuitDenoisingSolver(
    SingleQoiSolverMixin, LinearSystemSolver[Array], Generic[Array]
):
    """Basis Pursuit Denoising (BPDN) solver.

    Solves: min_c (1/2)||Φc - y||_2^2 + λ||c||_1

    Also known as LASSO regression.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    penalty : float
        L1 penalty parameter λ.
    max_iter : int
        Maximum iterations for coordinate descent. Default: 1000.
    tol : float
        Convergence tolerance. Default: 1e-4.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        penalty: float,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ):
        super().__init__(bkd)
        self._penalty = penalty
        self._max_iter = max_iter
        self._tol = tol

    def set_penalty(self, penalty: float) -> None:
        """Set L1 penalty.

        Parameters
        ----------
        penalty : float
            L1 penalty parameter.
        """
        self._penalty = penalty

    def _soft_threshold(self, x: Array, threshold: float) -> Array:
        """Apply soft thresholding operator.

        S(x, t) = sign(x) * max(|x| - t, 0)

        Parameters
        ----------
        x : Array
            Input array.
        threshold : float
            Threshold value.

        Returns
        -------
        Array
            Soft-thresholded array.
        """
        bkd = self._bkd
        return bkd.sign(x) * bkd.where(
            bkd.abs(x) > threshold,
            bkd.abs(x) - threshold,
            bkd.zeros_like(x),
        )

    def _solve(self, basis_matrix: Array, values: Array) -> Array:
        """Solve using coordinate descent for LASSO.

        Parameters
        ----------
        basis_matrix : Array
            Basis matrix Φ. Shape: (nsamples, nterms)
        values : Array
            Target values y. Shape: (nsamples, 1)

        Returns
        -------
        Array
            Sparse coefficients c. Shape: (nterms, 1)
        """
        self._validate_single_qoi(values)

        bkd = self._bkd
        A = basis_matrix
        y = values[:, 0]
        nterms = A.shape[1]

        # Precompute column norms squared
        col_norms_sq = bkd.sum(A ** 2, axis=0)

        # Initialize coefficients
        coef = bkd.zeros((nterms,))
        residual = y - bkd.dot(A, coef)

        for iteration in range(self._max_iter):
            coef_old = bkd.copy(coef)

            # Coordinate descent
            for jj in range(nterms):
                if col_norms_sq[jj] < 1e-14:
                    continue

                # Compute partial residual
                coef_j = coef[jj]
                residual = residual + A[:, jj] * coef_j

                # Compute correlation
                rho = bkd.dot(A[:, jj], residual)

                # Soft threshold update
                coef[jj] = float(
                    self._soft_threshold(
                        bkd.asarray([rho / col_norms_sq[jj]]),
                        self._penalty / col_norms_sq[jj],
                    )[0]
                )

                # Update residual
                residual = residual - A[:, jj] * coef[jj]

            # Check convergence
            diff = bkd.norm(coef - coef_old)
            if diff < self._tol:
                break

        return bkd.reshape(coef, (-1, 1))
