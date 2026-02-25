"""Protocols for linear system solvers.

This module defines protocols for solvers that find coefficients
for basis expansions by solving linear systems: Φc = y.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class LinearSystemSolverProtocol(Protocol, Generic[Array]):
    """Protocol for linear system solvers.

    Solvers implement methods to find coefficients c that minimize
    some objective involving the residual Φc - y.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

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
        ...


@runtime_checkable
class WeightedSolverProtocol(Protocol, Generic[Array]):  # type: ignore[misc]
    """Protocol for solvers that support sample weights."""

    def set_weights(self, weights: Array) -> None:
        """Set sample weights.

        Parameters
        ----------
        weights : Array
            Sample weights. Shape: (nsamples,)
        """
        ...


@runtime_checkable
class SparseSolverProtocol(Protocol, Generic[Array]):  # type: ignore[misc]
    """Protocol for sparse solvers with sparsity control."""

    def set_max_nonzeros(self, max_nonzeros: int) -> None:
        """Set maximum number of non-zero coefficients.

        Parameters
        ----------
        max_nonzeros : int
            Maximum sparsity level.
        """
        ...


@runtime_checkable
class RegularizedSolverProtocol(Protocol, Generic[Array]):  # type: ignore[misc]
    """Protocol for regularized solvers."""

    def set_regularization(self, alpha: float) -> None:
        """Set regularization strength.

        Parameters
        ----------
        alpha : float
            Regularization parameter.
        """
        ...


@runtime_checkable
class QuantileSolverProtocol(Protocol, Generic[Array]):  # type: ignore[misc]
    """Protocol for quantile regression solvers."""

    def set_quantile(self, quantile: float) -> None:
        """Set target quantile.

        Parameters
        ----------
        quantile : float
            Quantile level in [0, 1].
        """
        ...


@runtime_checkable
class ConstrainedSolverProtocol(Protocol, Generic[Array]):  # type: ignore[misc]
    """Protocol for solvers with linear constraints."""

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
        ...
