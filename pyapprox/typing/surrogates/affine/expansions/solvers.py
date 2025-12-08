"""Linear system solvers for basis expansion fitting.

This module provides solvers for the linear system: Φc = y, where Φ is the
basis matrix, c are the coefficients, and y are the target values.
"""

from typing import Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend


class LeastSquaresSolver(Generic[Array]):
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
        self._bkd = bkd
        self._rcond = rcond

    def solve(self, basis_matrix: Array, values: Array) -> Array:
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


class WeightedLeastSquaresSolver(Generic[Array]):
    """Weighted least squares solver.

    Solves: min_c ||W^{1/2}(Φc - y)||_2^2

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    weights : Array
        Sample weights. Shape: (nsamples,)
    rcond : float, optional
        Cutoff for small singular values.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        weights: Array,
        rcond: float = None,
    ):
        self._bkd = bkd
        self._weights = weights
        self._rcond = rcond

    def solve(self, basis_matrix: Array, values: Array) -> Array:
        """Solve weighted least squares problem.

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
        sqrt_w = self._bkd.sqrt(self._weights)
        weighted_basis = basis_matrix * self._bkd.reshape(sqrt_w, (-1, 1))
        weighted_values = values * self._bkd.reshape(sqrt_w, (-1, 1))
        return self._bkd.lstsq(weighted_basis, weighted_values, rcond=self._rcond)


class RidgeRegressionSolver(Generic[Array]):
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
        self._bkd = bkd
        self._alpha = alpha

    def solve(self, basis_matrix: Array, values: Array) -> Array:
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
