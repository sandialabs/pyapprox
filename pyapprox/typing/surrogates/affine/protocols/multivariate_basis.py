"""Protocols for multivariate basis functions.

This module defines the protocol hierarchy for multivariate bases:

Protocol Hierarchy:
    BasisProtocol (base) - evaluation only
        ↓
    BasisWithJacobianProtocol - adds first derivatives
        ↓
    BasisWithJacobianAndHessianProtocol - adds second derivatives

"Has" protocols define single capabilities:
    - BasisHasJacobianProtocol
    - BasisHasHessianProtocol

"With" protocols compose capabilities:
    - BasisWithJacobianProtocol = BasisProtocol + HasJacobian
    - BasisWithJacobianAndHessianProtocol = WithJacobian + HasHessian

Multi-index protocols:
    - MultiIndexBasisProtocol - for bases with multi-index structure
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class BasisProtocol(Protocol, Generic[Array]):
    """Core protocol for multivariate basis functions.

    All multivariate bases must implement this protocol. This is the minimal
    interface required for evaluation.

    Methods
    -------
    bkd() -> Backend[Array]
        Return the computational backend.
    nterms() -> int
        Return the number of basis terms.
    nvars() -> int
        Return the number of input variables.
    __call__(samples: Array) -> Array
        Evaluate basis at sample points.

    Notes
    -----
    Input samples shape: (nvars, nsamples)
    Output shape: (nsamples, nterms)
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nterms(self) -> int:
        """Return the number of basis terms."""
        ...

    def nvars(self) -> int:
        """Return the number of input variables."""
        ...

    def __call__(self, samples: Array) -> Array:
        """Evaluate basis functions at sample points.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Basis values. Shape: (nsamples, nterms)
        """
        ...


@runtime_checkable
class BasisHasJacobianProtocol(Protocol, Generic[Array]):
    """Protocol for multivariate bases that support first derivatives.

    Methods
    -------
    jacobian_batch(samples: Array) -> Array
        Evaluate first derivatives at sample points.

    Notes
    -----
    Input shape: (nvars, nsamples) - must be 2D, raises ValueError for 1D input.
    Output shape: (nsamples, nterms, nvars)
    jacobian_batch[i, j, k] = d(basis_j)/d(x_k) evaluated at sample i
    """

    def jacobian_batch(self, samples: Array) -> Array:
        """Evaluate first derivatives of basis functions.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples). Must be 2D.

        Returns
        -------
        Array
            Jacobians of each basis term. Shape: (nsamples, nterms, nvars)

        Raises
        ------
        ValueError
            If samples is not 2D with shape (nvars, nsamples).
        """
        ...


@runtime_checkable
class BasisHasHessianProtocol(Protocol, Generic[Array]):
    """Protocol for multivariate bases that support second derivatives.

    Methods
    -------
    hessian_batch(samples: Array) -> Array
        Evaluate second derivatives at sample points.

    Notes
    -----
    Input shape: (nvars, nsamples) - must be 2D, raises ValueError for 1D input.
    Output shape: (nsamples, nterms, nvars, nvars)
    hessian_batch[i, j, k, l] = d^2(basis_j)/d(x_k)d(x_l) evaluated at sample i
    """

    def hessian_batch(self, samples: Array) -> Array:
        """Evaluate second derivatives of basis functions.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples). Must be 2D.

        Returns
        -------
        Array
            Hessians of each basis term. Shape: (nsamples, nterms, nvars, nvars)

        Raises
        ------
        ValueError
            If samples is not 2D with shape (nvars, nsamples).
        """
        ...


# Composed protocols (combine base + "Has" protocols)


class BasisWithJacobianProtocol(
    BasisProtocol[Array],
    BasisHasJacobianProtocol[Array],
    Protocol,
    Generic[Array],
):
    """Protocol for multivariate bases with evaluation and first derivatives.

    This is the typical protocol for differentiable bases built from
    differentiable univariate components.
    """

    pass


class BasisWithJacobianAndHessianProtocol(
    BasisWithJacobianProtocol[Array],
    BasisHasHessianProtocol[Array],
    Protocol,
    Generic[Array],
):
    """Protocol for multivariate bases with full derivative support.

    This is the complete derivative support protocol for twice-differentiable
    bases built from twice-differentiable univariate components.
    """

    pass


# Multi-index protocols


@runtime_checkable
class MultiIndexBasisProtocol(Protocol, Generic[Array]):
    """Protocol for bases with multi-index structure.

    Multi-index bases are tensor products of univariate bases, where each
    basis term corresponds to a multi-index specifying the degree in each
    variable.

    Methods
    -------
    get_indices() -> Array
        Return the multi-indices defining the basis.
    set_indices(indices: Array) -> None
        Set the multi-indices defining the basis.

    Notes
    -----
    Indices shape: (nvars, nterms)
    Each column is a multi-index specifying degrees for each variable.
    """

    def get_indices(self) -> Array:
        """Return the multi-indices defining the basis.

        Returns
        -------
        Array
            Multi-indices. Shape: (nvars, nterms)
        """
        ...

    def set_indices(self, indices: Array) -> None:
        """Set the multi-indices defining the basis.

        Parameters
        ----------
        indices : Array
            Multi-indices. Shape: (nvars, nterms)
        """
        ...


@runtime_checkable
class TensorProductBasisProtocol(Protocol, Generic[Array]):
    """Protocol for tensor product bases.

    Tensor product bases are constructed from univariate bases by taking
    products. This protocol provides access to the underlying univariate bases.

    Methods
    -------
    get_univariate_bases() -> list
        Return the univariate bases for each variable.
    """

    def get_univariate_bases(self) -> list:
        """Return the univariate bases.

        Returns
        -------
        list
            List of univariate basis objects, one per variable.
        """
        ...


# Combined protocols for common patterns


class MultiIndexBasisWithJacobianProtocol(
    BasisWithJacobianProtocol[Array],
    MultiIndexBasisProtocol[Array],
    Protocol,
    Generic[Array],
):
    """Multi-index basis with Jacobian support."""

    pass


class MultiIndexBasisWithJacobianAndHessianProtocol(
    BasisWithJacobianAndHessianProtocol[Array],
    MultiIndexBasisProtocol[Array],
    Protocol,
    Generic[Array],
):
    """Multi-index basis with full derivative support."""

    pass
