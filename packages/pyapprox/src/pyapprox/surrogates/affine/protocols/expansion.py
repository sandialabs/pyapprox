"""Protocols for basis expansions.

This module defines protocols for basis expansions that express functions
as linear combinations of basis functions: f(x) ≈ Σ_i c_i φ_i(x).
"""

from typing import Generic, Protocol, Self, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class BasisExpansionProtocol(Protocol, Generic[Array]):
    """Protocol for basis expansions.

    A basis expansion represents a function as a linear combination of
    basis functions: f(x) = Σ_i c_i φ_i(x).
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nvars(self) -> int:
        """Return the number of input variables."""
        ...

    def nterms(self) -> int:
        """Return the number of basis terms."""
        ...

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        ...

    def get_coefficients(self) -> Array:
        """Return coefficients. Shape: (nterms, nqoi)."""
        ...

    def set_coefficients(self, coef: Array) -> None:
        """Set coefficients. Shape: (nterms, nqoi)."""
        ...

    def __call__(self, samples: Array) -> Array:
        """Evaluate expansion at samples.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Values at samples. Shape: (nqoi, nsamples)
        """
        ...

    def basis_matrix(self, samples: Array) -> Array:
        """Compute basis matrix (design matrix) Phi(samples).

        Parameters
        ----------
        samples : Array
            Input samples. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Basis matrix. Shape: (nsamples, nterms)
        """
        ...

    def with_params(self, params: Array) -> Self:
        """Return NEW instance with parameters set. Original unchanged.

        Parameters
        ----------
        params : Array
            Coefficient values. Shape: (nterms, nqoi)

        Returns
        -------
        Self
            New expansion with coefficients set.
        """
        ...


@runtime_checkable
class BasisExpansionHasJacobianProtocol(Protocol, Generic[Array]):
    """Protocol for expansions that support Jacobian computation."""

    def jacobian_batch(self, samples: Array) -> Array:
        """Compute Jacobians of expansion at samples.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples). Must be 2D.

        Returns
        -------
        Array
            Jacobians. Shape: (nsamples, nqoi, nvars)

        Raises
        ------
        ValueError
            If samples is not 2D with shape (nvars, nsamples).
        """
        ...


@runtime_checkable
class BasisExpansionHasHessianProtocol(Protocol, Generic[Array]):
    """Protocol for expansions that support Hessian computation."""

    def hessian_batch(self, samples: Array) -> Array:
        """Compute Hessians of expansion at samples.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples). Must be 2D.

        Returns
        -------
        Array
            Hessians. Shape: (nsamples, nvars, nvars)

        Raises
        ------
        ValueError
            If nqoi != 1 (Hessian only supported for scalar-valued functions).
        """
        ...


@runtime_checkable
class FittableBasisExpansionProtocol(Protocol, Generic[Array]):  # type: ignore[misc]
    """Protocol for expansions that can be fitted to data."""

    def fit(self, samples: Array, values: Array) -> None:
        """Fit expansion to data.

        Parameters
        ----------
        samples : Array
            Training sample points. Shape: (nvars, nsamples)
        values : Array
            Training values. Shape: (nqoi, nsamples)
        """
        ...


@runtime_checkable
class PCEStatisticsProtocol(Protocol, Generic[Array]):
    """Protocol for objects that support PCE statistics.

    PCE statistics exploit orthonormality to compute moments analytically
    from coefficients without sampling.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def get_coefficients(self) -> Array:
        """Return coefficients. Shape: (nterms, nqoi)."""
        ...

    def get_indices(self) -> Array:
        """Return multi-indices. Shape: (nvars, nterms)."""
        ...

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        ...

    def nvars(self) -> int:
        """Return the number of input variables."""
        ...


@runtime_checkable
class LinearSystemSolverProtocol(Protocol, Generic[Array]):
    """Protocol for linear system solvers used in expansion fitting."""

    def solve(self, basis_matrix: Array, values: Array) -> Array:
        """Solve the linear system: basis_matrix @ coef = values.

        Parameters
        ----------
        basis_matrix : Array
            Basis matrix. Shape: (nsamples, nterms)
        values : Array
            Target values. Shape: (nsamples, nqoi)

        Returns
        -------
        Array
            Coefficients. Shape: (nterms, nqoi)
        """
        ...
