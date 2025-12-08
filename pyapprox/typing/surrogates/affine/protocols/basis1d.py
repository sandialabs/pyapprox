"""Protocols for univariate (1D) basis functions.

This module defines the protocol hierarchy for univariate bases:

Protocol Hierarchy:
    Basis1DProtocol (base) - evaluation only
        ↓
    Basis1DWithJacobianProtocol - adds first derivatives
        ↓
    Basis1DWithJacobianAndHessianProtocol - adds second derivatives

"Has" protocols define single capabilities:
    - Basis1DHasJacobianProtocol
    - Basis1DHasHessianProtocol
    - Basis1DHasDerivativesProtocol (arbitrary order)

"With" protocols compose capabilities:
    - Basis1DWithJacobianProtocol = Basis1DProtocol + HasJacobian
    - Basis1DWithJacobianAndHessianProtocol = WithJacobian + HasHessian

Specialized protocols:
    - OrthonormalPolynomial1DProtocol - for orthonormal polynomial bases
"""

from typing import Generic, Tuple, Protocol, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class Basis1DProtocol(Protocol, Generic[Array]):
    """Core protocol for univariate (1D) basis functions.

    All 1D bases must implement this protocol. This is the minimal
    interface required for evaluation.

    Methods
    -------
    bkd() -> Backend[Array]
        Return the computational backend.
    set_nterms(nterms: int) -> None
        Set the number of basis terms to evaluate.
    nterms() -> int
        Return the current number of terms.
    __call__(samples: Array) -> Array
        Evaluate basis at sample points.

    Notes
    -----
    Input samples shape: (1, nsamples)
    Output shape: (nsamples, nterms)
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def set_nterms(self, nterms: int) -> None:
        """Set the number of basis terms."""
        ...

    def nterms(self) -> int:
        """Return the number of basis terms."""
        ...

    def __call__(self, samples: Array) -> Array:
        """Evaluate basis functions at sample points.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples)

        Returns
        -------
        Array
            Basis values. Shape: (nsamples, nterms)
        """
        ...


@runtime_checkable
class Basis1DHasJacobianProtocol(Protocol, Generic[Array]):
    """Protocol for 1D bases that support first derivatives.

    Methods
    -------
    jacobians(samples: Array) -> Array
        Evaluate first derivatives at sample points.

    Notes
    -----
    Output shape: (nsamples, nterms) - derivative w.r.t. the single variable.
    """

    def jacobians(self, samples: Array) -> Array:
        """Evaluate first derivatives of basis functions.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples)

        Returns
        -------
        Array
            First derivatives. Shape: (nsamples, nterms)
        """
        ...


@runtime_checkable
class Basis1DHasHessianProtocol(Protocol, Generic[Array]):
    """Protocol for 1D bases that support second derivatives.

    Methods
    -------
    hessians(samples: Array) -> Array
        Evaluate second derivatives at sample points.

    Notes
    -----
    Output shape: (nsamples, nterms) - second derivative w.r.t. the single variable.
    """

    def hessians(self, samples: Array) -> Array:
        """Evaluate second derivatives of basis functions.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples)

        Returns
        -------
        Array
            Second derivatives. Shape: (nsamples, nterms)
        """
        ...


@runtime_checkable
class Basis1DHasDerivativesProtocol(Protocol, Generic[Array]):
    """Protocol for 1D bases that support arbitrary order derivatives.

    Methods
    -------
    derivatives(samples: Array, order: int) -> Array
        Evaluate derivatives of specified order.

    Notes
    -----
    This is for bases that can efficiently compute any derivative order.
    """

    def derivatives(self, samples: Array, order: int) -> Array:
        """Evaluate derivatives of specified order.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples)
        order : int
            Derivative order. 0 = values, 1 = first deriv, etc.

        Returns
        -------
        Array
            Derivatives of specified order. Shape: (nsamples, nterms)
        """
        ...


# Composed protocols (combine base + "Has" protocols)


class Basis1DWithJacobianProtocol(
    Basis1DProtocol[Array],
    Basis1DHasJacobianProtocol[Array],
    Protocol,
    Generic[Array],
):
    """Protocol for 1D bases with evaluation and first derivatives.

    This is the typical protocol for differentiable bases like polynomials.
    Piecewise constant/linear bases may not implement this.
    """

    pass


class Basis1DWithJacobianAndHessianProtocol(
    Basis1DWithJacobianProtocol[Array],
    Basis1DHasHessianProtocol[Array],
    Protocol,
    Generic[Array],
):
    """Protocol for 1D bases with evaluation, first and second derivatives.

    This is the full derivative support protocol for twice-differentiable bases.
    """

    pass


# Specialized protocols


@runtime_checkable
class OrthonormalPolynomial1DProtocol(Protocol, Generic[Array]):
    """Protocol for orthonormal polynomial bases.

    Orthonormal polynomials have additional capabilities:
    - Gauss quadrature rule computation
    - Recursion coefficient access

    These properties enable PCE statistics computation.
    """

    def gauss_quadrature_rule(self, npoints: int) -> Tuple[Array, Array]:
        """Compute Gaussian quadrature rule for this polynomial family.

        Parameters
        ----------
        npoints : int
            Number of quadrature points.

        Returns
        -------
        points : Array
            Quadrature points. Shape: (1, npoints)
        weights : Array
            Quadrature weights. Shape: (npoints,)
        """
        ...

    def recursion_coefficients(self) -> Array:
        """Return the three-term recursion coefficients.

        Returns
        -------
        Array
            Recursion coefficients. Shape: (nterms, 2)
            Column 0: alpha_n coefficients
            Column 1: beta_n coefficients
        """
        ...


@runtime_checkable
class Basis1DHasQuadratureProtocol(Protocol, Generic[Array]):
    """Protocol for 1D bases with quadrature rule support."""

    def quadrature_rule(self, npoints: int) -> Tuple[Array, Array]:
        """Return quadrature points and weights.

        Parameters
        ----------
        npoints : int
            Number of quadrature points.

        Returns
        -------
        points : Array
            Quadrature points. Shape: (1, npoints)
        weights : Array
            Quadrature weights. Shape: (npoints,)
        """
        ...
