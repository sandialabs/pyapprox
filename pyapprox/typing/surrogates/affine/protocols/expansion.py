"""Protocols for basis expansions.

This module defines protocols for basis expansions that express functions
as linear combinations of basis functions: f(x) ≈ Σ_i c_i φ_i(x).
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend


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
            Values at samples. Shape: (nsamples, nqoi)
        """
        ...


@runtime_checkable
class BasisExpansionHasJacobianProtocol(Protocol, Generic[Array]):
    """Protocol for expansions that support Jacobian computation."""

    def jacobians(self, samples: Array) -> Array:
        """Compute Jacobians of expansion at samples.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Jacobians. Shape: (nsamples, nqoi, nvars)
        """
        ...


@runtime_checkable
class BasisExpansionHasHessianProtocol(Protocol, Generic[Array]):
    """Protocol for expansions that support Hessian computation."""

    def hessians(self, samples: Array) -> Array:
        """Compute Hessians of expansion at samples.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Hessians. Shape: (nsamples, nqoi, nvars, nvars)
        """
        ...


@runtime_checkable
class FittableBasisExpansionProtocol(Protocol, Generic[Array]):
    """Protocol for expansions that can be fitted to data."""

    def fit(self, samples: Array, values: Array) -> None:
        """Fit expansion to data.

        Parameters
        ----------
        samples : Array
            Training sample points. Shape: (nvars, nsamples)
        values : Array
            Training values. Shape: (nsamples, nqoi)
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
