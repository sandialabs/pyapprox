"""Least squares solvers for basis expansion fitting.

This module provides least squares based solvers:
- LeastSquaresSolver: Standard least squares
- RidgeRegressionSolver: L2 regularized least squares
- LinearlyConstrainedLstSqSolver: Least squares with equality constraints
"""

from typing import Generic, Optional

from pyapprox.surrogates.affine.solvers.base import LinearSystemSolver
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

    Solves: min_c ||Φc - y||_2^2
            subject to: Cc = d

    Uses the method of Lagrange multipliers:
    x = (A^T A)^-1 (A^T y - C^T (C(A^T A)^-1 C^T)^-1 (C(A^T A)^-1 A^T y - d))

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
        """Solve constrained least squares problem.

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
        y = values

        # Compute (A^T A)^-1
        ATA = self._bkd.dot(A.T, A)
        ATA_inv = self._bkd.inv(ATA)

        # Compute A^T y
        ATy = self._bkd.dot(A.T, y)

        # Compute (A^T A)^-1 A^T y
        ATA_inv_ATy = self._bkd.dot(ATA_inv, ATy)

        # Compute C (A^T A)^-1 C^T
        C_ATA_inv = self._bkd.dot(C, ATA_inv)
        C_ATA_inv_CT = self._bkd.dot(C_ATA_inv, C.T)

        # Compute C (A^T A)^-1 A^T y - d
        C_ATA_inv_ATy = self._bkd.dot(C, ATA_inv_ATy)
        residual = C_ATA_inv_ATy - d

        # Solve for Lagrange multipliers
        lagrange = self._bkd.solve(C_ATA_inv_CT, residual)

        # Compute solution
        coef = ATA_inv_ATy - self._bkd.dot(C_ATA_inv.T, lagrange)

        return coef
