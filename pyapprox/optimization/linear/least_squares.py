"""Least squares solvers for basis expansion fitting.

This module provides least squares based solvers:
- LeastSquaresSolver: Standard least squares
- RidgeRegressionSolver: L2 regularized least squares
- LinearlyConstrainedLstSqSolver: Least squares with equality constraints
"""

from typing import Generic, Optional

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.optimization.linear.base import LinearSystemSolver


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

    Uses the KKT system for numerical stability:
    [A^T A,  C^T] [theta ]   [A^T y]
    [C,      0  ] [lambda] = [d    ]

    Solved via lstsq for robustness to near-singular cases.

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
            self._constraint_vector = self._bkd.reshape(
                constraint_vector, (-1, 1)
            )

    def _solve(self, basis_matrix: Array, values: Array) -> Array:
        """Solve constrained least squares problem via KKT system.

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
        bkd = self._bkd

        nterms = A.shape[1]
        nconstraints = C.shape[0]

        # Build KKT matrix:
        # [A^T A,  C^T] [theta ]   [A^T y]
        # [C,      0  ] [lambda] = [d    ]
        ATA = bkd.dot(A.T, A)
        ATy = bkd.dot(A.T, y)

        zeros = bkd.zeros((nconstraints, nconstraints))

        # Build augmented KKT matrix
        top_row = bkd.concatenate([ATA, C.T], axis=1)
        bottom_row = bkd.concatenate([C, zeros], axis=1)
        KKT = bkd.concatenate([top_row, bottom_row], axis=0)

        # Build augmented RHS
        rhs = bkd.concatenate([ATy, d], axis=0)

        # Solve full system using lstsq for robustness
        solution = bkd.lstsq(KKT, rhs, rcond=self._rcond)

        # Extract coefficients (first nterms rows)
        coef = solution[:nterms, :]

        return coef
