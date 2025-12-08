"""Base classes for linear system solvers.

This module provides abstract base classes and mixins for solvers
that find coefficients for basis expansions.
"""

from abc import ABC, abstractmethod
from typing import Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend


class LinearSystemSolver(ABC, Generic[Array]):
    """Abstract base class for linear system solvers.

    Solves systems of the form: find c such that Φc ≈ y,
    where Φ is the basis matrix, c are coefficients, and y are targets.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd
        self._weights: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def set_weights(self, weights: Array) -> None:
        """Set sample weights for weighted solving.

        Parameters
        ----------
        weights : Array
            Sample weights. Shape: (nsamples,)
        """
        self._weights = weights

    def solve(self, basis_matrix: Array, values: Array) -> Array:
        """Solve for coefficients.

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
        # Ensure values is 2D
        if values.ndim == 1:
            values = self._bkd.reshape(values, (-1, 1))

        # Apply weights if set
        if self._weights is not None:
            basis_matrix, values = self._apply_weights(basis_matrix, values)

        return self._solve(basis_matrix, values)

    def _apply_weights(
        self, basis_matrix: Array, values: Array
    ) -> tuple[Array, Array]:
        """Apply sample weights to basis matrix and values.

        Parameters
        ----------
        basis_matrix : Array
            Basis matrix. Shape: (nsamples, nterms)
        values : Array
            Target values. Shape: (nsamples, nqoi)

        Returns
        -------
        tuple[Array, Array]
            Weighted (basis_matrix, values).
        """
        sqrt_w = self._bkd.sqrt(self._weights)
        sqrt_w_col = self._bkd.reshape(sqrt_w, (-1, 1))
        return basis_matrix * sqrt_w_col, values * sqrt_w_col

    @abstractmethod
    def _solve(self, basis_matrix: Array, values: Array) -> Array:
        """Subclass implementation of solve.

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
        ...


class SingleQoiSolverMixin:
    """Mixin that enforces single QoI constraint.

    Some solvers (e.g., sparse, quantile) only support single QoI.
    """

    def _validate_single_qoi(self, values: Array) -> None:
        """Validate that values has single QoI.

        Parameters
        ----------
        values : Array
            Target values. Shape: (nsamples, nqoi)

        Raises
        ------
        ValueError
            If nqoi > 1.
        """
        if values.ndim > 1 and values.shape[1] > 1:
            raise ValueError(
                f"Solver only supports single QoI, got {values.shape[1]}"
            )
