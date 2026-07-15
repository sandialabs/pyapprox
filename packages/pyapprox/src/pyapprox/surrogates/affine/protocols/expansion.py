"""Protocols for basis expansions.

This module defines protocols for basis expansions that express functions
as linear combinations of basis functions: f(x) ≈ Σ_i c_i φ_i(x).
"""

from typing import Generic, Protocol, Self, runtime_checkable

from pyapprox.interface.functions.derivatives import Derivatives
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

    def nparams(self) -> int:
        """Return the total number of parameters (nterms * nqoi)."""
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

    def derivatives(self) -> Derivatives[Array]:
        """Return the derivative bundle.

        Capability w.r.t. inputs is declared through the bundle fields
        (``jacobian_batch`` etc.); absent capability is ``None``.
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
class BasisExpansionHasParamJacobianProtocol(Protocol, Generic[Array]):
    """Protocol for expansions providing their own parameter Jacobian.

    Expansions with a nonlinear (or otherwise custom) parameterization
    implement ``jacobian_wrt_params``; linear expansions without it get
    the generic linear formula derived from ``basis_matrix``. Parameter-
    family derivatives are public named methods (they carry the
    "wrt params" context in the name), so this structural gate — not the
    Derivatives bundle, which covers derivatives w.r.t. inputs — is how
    consumers detect them.
    """

    def jacobian_wrt_params(self, samples: Array) -> Array:
        """Compute Jacobian w.r.t. active parameters.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Jacobians. Shape: (nsamples, nqoi, nactive_params)
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
