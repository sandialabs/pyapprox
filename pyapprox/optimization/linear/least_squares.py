"""Least squares solvers for basis expansion fitting.

This module provides least squares based solvers:
- LeastSquaresSolver: Standard least squares
- RidgeRegressionSolver: L2 regularized least squares
- LinearlyConstrainedLstSqSolver: Least squares with equality constraints
"""

from typing import Generic, Optional

from pyapprox.optimization.linear.base import LinearSystemSolver
from pyapprox.util.backends.protocols import Array, Backend


class LeastSquaresSolver(LinearSystemSolver[Array], Generic[Array]):
    """Least squares solver for fitting basis expansions.

    Solves: min_c ||Φc - y||_2^2

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    rcond : float, optional
        Cutoff for small singular values. Default: None (machine precision).
    """

    def __init__(self, bkd: Backend[Array], rcond: Optional[float] = None):
        super().__init__(bkd)
        self._rcond = rcond

    def _solve(self, basis_matrix: Array, values: Array) -> Array:
        """Solve least squares problem.

        Parameters
        ----------
        basis_matrix : Array
            Basis matrix Φ. Shape: (nsamples, nterms)
        values : Array
            Target values y. Shape: (nsamples, nqoi)

        Returns
        -------
        Array
            Coefficients c. Shape: (nterms, nqoi)
        """
        return self._bkd.lstsq(basis_matrix, values, rcond=self._rcond)


class RidgeRegressionSolver(LinearSystemSolver[Array], Generic[Array]):
    """Ridge regression (L2 regularized) solver.

    Solves: min_c ||Φc - y||_2^2 + α||c||_2^2

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    alpha : float
        Regularization strength.
    """

    def __init__(self, bkd: Backend[Array], alpha: float):
        super().__init__(bkd)
        self._alpha = alpha

    def set_regularization(self, alpha: float) -> None:
        """Set regularization strength.

        Parameters
        ----------
        alpha : float
            Regularization parameter.
        """
        self._alpha = alpha

    def _solve(self, basis_matrix: Array, values: Array) -> Array:
        """Solve ridge regression problem.

        Parameters
        ----------
        basis_matrix : Array
            Basis matrix Φ. Shape: (nsamples, nterms)
        values : Array
            Target values y. Shape: (nsamples, nqoi)

        Returns
        -------
        Array
            Coefficients c. Shape: (nterms, nqoi)
        """
        nterms = basis_matrix.shape[1]
        # (Φ^T Φ + αI) c = Φ^T y
        gram = self._bkd.dot(basis_matrix.T, basis_matrix)
        gram = gram + self._alpha * self._bkd.eye(nterms)
        rhs = self._bkd.dot(basis_matrix.T, values)
        return self._bkd.solve(gram, rhs)


class LinearlyConstrainedLstSqSolver(LinearSystemSolver[Array], Generic[Array]):
    """Least squares solver with linear equality constraints.

    Solves: min_c ||Ac - b||_2^2
            subject to: Cc = d

    Uses the null-space method:
    1. Compute null space Z of C via SVD
    2. Find particular solution c_p satisfying C c_p = d (via lstsq)
    3. Solve unconstrained lstsq: min ||A Z alpha - (b - A c_p)||^2
    4. c = c_p + Z alpha

    Handles both p <= n (standard) and p > n (overconstrained equality)
    cases. When rank(C) = n, the constraints fully determine c and
    the objective is irrelevant.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    constraint_matrix : Array
        Constraint matrix C. Shape: (nconstraints, nterms)
    constraint_vector : Array
        Constraint values d. Shape: (nconstraints,) or (nconstraints, 1)
    rcond : float, optional
        Cutoff for small singular values.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        constraint_matrix: Array,
        constraint_vector: Array,
        rcond: Optional[float] = None,
    ):
        super().__init__(bkd)
        self._constraint_matrix = constraint_matrix
        self._constraint_vector = constraint_vector
        if constraint_vector.ndim == 1:
            self._constraint_vector = bkd.reshape(constraint_vector, (-1, 1))
        self._rcond = rcond

    def set_constraints(
        self,
        constraint_matrix: Array,
        constraint_vector: Array,
    ) -> None:
        """Set linear equality constraints Cx = d.

        Parameters
        ----------
        constraint_matrix : Array
            Constraint matrix C. Shape: (nconstraints, nterms)
        constraint_vector : Array
            Constraint values d. Shape: (nconstraints,)
        """
        self._constraint_matrix = constraint_matrix
        self._constraint_vector = constraint_vector
        if constraint_vector.ndim == 1:
            self._constraint_vector = self._bkd.reshape(constraint_vector, (-1, 1))

    def _solve(self, basis_matrix: Array, values: Array) -> Array:
        """Solve constrained least squares via null-space method.

        min ||A theta - b||^2  s.t.  C theta = d

        1. Compute null space Z of C via SVD
        2. Find particular solution theta_p satisfying C theta_p = d
        3. Solve unconstrained lstsq: min ||A Z alpha - (b - A theta_p)||^2
        4. theta = theta_p + Z alpha

        No KKT or saddle-point system is formed, avoiding ill-
        conditioned augmented matrices entirely.

        Parameters
        ----------
        basis_matrix : Array
            Basis matrix Φ. Shape: (nsamples, nterms)
        values : Array
            Target values y. Shape: (nsamples, nqoi)

        Returns
        -------
        Array
            Coefficients c. Shape: (nterms, nqoi)
        """
        C = self._constraint_matrix
        d = self._constraint_vector
        A = basis_matrix
        b = values
        bkd = self._bkd

        # Step 1: Null space of C via SVD of C^T
        # C is (p, n). SVD of C^T = U S V^T gives null space as
        # columns of U beyond the rank.
        U, S, _Vh = bkd.svd(C.T, full_matrices=True)
        tol = 1e-12 * bkd.to_float(S[0]) if S.shape[0] > 0 else 1e-12
        rank_C = int(bkd.to_float(
            bkd.sum(bkd.asarray([1.0 if bkd.to_float(s) > tol else 0.0
                                  for s in S]))
        ))
        Z = U[:, rank_C:]  # (n, n - rank_C)

        # Step 2: Particular solution satisfying C theta_p = d
        theta_p = bkd.lstsq(C, d, rcond=self._rcond)

        # Step 3: Unconstrained lstsq in null-space coordinates
        AZ = bkd.dot(A, Z)
        rhs = b - bkd.dot(A, theta_p)
        alpha = bkd.lstsq(AZ, rhs, rcond=self._rcond)

        # Step 4: Reconstruct full solution
        return theta_p + bkd.dot(Z, alpha)
