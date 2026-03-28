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
    - Basis1DHasQuadratureProtocol

"With" protocols compose capabilities:
    - Basis1DWithJacobianProtocol = Basis1DProtocol + HasJacobian
    - Basis1DWithJacobianAndHessianProtocol = WithJacobian + HasHessian

Interpolation protocols:
    - InterpolationBasis1DProtocol - for tensor product interpolation

Specialized protocols:
    - OrthonormalPolynomial1DProtocol - for orthonormal polynomial bases
"""

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


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
    jacobian_batch(samples: Array) -> Array
        Evaluate first derivatives at sample points.

    Notes
    -----
    Input shape: (1, nsamples) - must be 2D, raises ValueError for 1D input.
    Output shape: (nsamples, nterms) - derivative w.r.t. the single variable.
    """

    def jacobian_batch(self, samples: Array) -> Array:
        """Evaluate first derivatives of basis functions.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples). Must be 2D.

        Returns
        -------
        Array
            First derivatives. Shape: (nsamples, nterms)

        Raises
        ------
        ValueError
            If samples is not 2D with shape (1, nsamples).
        """
        ...


@runtime_checkable
class Basis1DHasHessianProtocol(Protocol, Generic[Array]):
    """Protocol for 1D bases that support second derivatives.

    Methods
    -------
    hessian_batch(samples: Array) -> Array
        Evaluate second derivatives at sample points.

    Notes
    -----
    Input shape: (1, nsamples) - must be 2D, raises ValueError for 1D input.
    Output shape: (nsamples, nterms) - second derivative w.r.t. the single variable.
    """

    def hessian_batch(self, samples: Array) -> Array:
        """Evaluate second derivatives of basis functions.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples). Must be 2D.

        Returns
        -------
        Array
            Second derivatives. Shape: (nsamples, nterms)

        Raises
        ------
        ValueError
            If samples is not 2D with shape (1, nsamples).
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


# Interpolation protocols


@runtime_checkable
class InterpolationBasis1DProtocol(Protocol, Generic[Array]):
    """Protocol for 1D bases suitable for tensor product interpolation.

    This protocol defines the requirements for a 1D basis that can be used
    in tensor product interpolation. It requires:
    - Evaluation of basis functions at sample points
    - A method to get interpolation nodes (sample locations)

    Derivative support (jacobian_batch, hessian_batch) is checked at runtime
    via isinstance checks against Basis1DHasJacobianProtocol and
    Basis1DHasHessianProtocol.

    This protocol is NOT satisfied by orthogonal polynomial bases (Legendre,
    Hermite, etc.) directly. Use LagrangeBasis1D or piecewise polynomial
    bases for tensor product interpolation.

    Methods
    -------
    bkd() -> Backend[Array]
        Return the computational backend.
    set_nterms(nterms: int) -> None
        Set the number of basis terms (interpolation points).
    nterms() -> int
        Return the current number of terms.
    __call__(samples: Array) -> Array
        Evaluate basis at sample points.
    get_samples(nterms: int) -> Array
        Return interpolation nodes for the given number of terms.

    Notes
    -----
    Input samples shape: (1, nsamples)
    Output shape: (nsamples, nterms)
    Interpolation nodes shape: (1, nterms)

    Examples
    --------
    LagrangeBasis1D satisfies this protocol:

    >>> isinstance(lagrange_basis, InterpolationBasis1DProtocol)
    True

    Orthogonal polynomial bases do NOT satisfy this protocol:

    >>> isinstance(legendre_poly, InterpolationBasis1DProtocol)
    False
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def set_nterms(self, nterms: int) -> None:
        """Set the number of basis terms (interpolation points)."""
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

    def get_samples(self, nterms: int) -> Array:
        """Return interpolation nodes for the given number of terms.

        Parameters
        ----------
        nterms : int
            Number of interpolation points.

        Returns
        -------
        Array
            Interpolation nodes. Shape: (1, nterms)
        """
        ...

    def quadrature_rule(self) -> Tuple[Array, Array]:
        """Return quadrature points and weights for current nterms.

        Must call `set_nterms()` before using this method.

        Returns
        -------
        points : Array
            Quadrature points. Shape: (1, nterms)
        weights : Array
            Quadrature weights. Shape: (nterms, 1)

        Raises
        ------
        ValueError
            If `set_nterms()` has not been called.
        """
        ...


# Specialized protocols


@runtime_checkable
class OrthonormalPolynomial1DProtocol(Basis1DProtocol[Array], Protocol, Generic[Array]):
    """Protocol for orthonormal polynomial bases.

    Extends Basis1DProtocol with additional capabilities:
    - First and second derivatives (jacobian_batch, hessian_batch)
    - Gauss quadrature rule computation
    - Recursion coefficient access

    These properties enable PCE statistics computation and
    derivative-based operations in TransformedBasis1D.
    """

    def derivatives(self, samples: Array, order: int) -> Array:
        """Evaluate derivatives of specified order.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples)
        order : int
            Derivative order. 0 = values, 1 = first derivative, etc.

        Returns
        -------
        Array
            Derivatives. Shape: (nsamples, nterms)
        """
        ...

    def jacobian_batch(self, samples: Array) -> Array:
        """Evaluate first derivatives of basis functions.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples). Must be 2D.

        Returns
        -------
        Array
            First derivatives. Shape: (nsamples, nterms)
        """
        ...

    def hessian_batch(self, samples: Array) -> Array:
        """Evaluate second derivatives of basis functions.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples). Must be 2D.

        Returns
        -------
        Array
            Second derivatives. Shape: (nsamples, nterms)
        """
        ...

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
class Basis1DHasQuadratureProtocol(Protocol, Generic[Array]):  # type: ignore[misc]
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


@runtime_checkable
class PhysicalDomainBasis1DProtocol(Protocol, Generic[Array]):
    """Protocol for 1D bases that accept samples in physical domain.

    This protocol is for bases that accept samples directly from the
    marginal distribution's support (physical/user domain), rather than
    requiring transformation to canonical domain first.

    Implementations include:
    - TransformedBasis1D: Wraps a canonical-domain polynomial with a transform
    - NativeBasis1D: Wraps a physical-domain polynomial (e.g., discrete numeric)

    This protocol provides a consistent interface for PCE and sparse grids,
    ensuring samples can be used directly without manual transformation.

    Methods
    -------
    bkd() -> Backend[Array]
        Return the computational backend.
    set_nterms(nterms: int) -> None
        Set the number of basis terms.
    nterms() -> int
        Return the current number of terms.
    __call__(samples: Array) -> Array
        Evaluate basis at physical domain sample points.
    gauss_quadrature_rule(npoints: int) -> Tuple[Array, Array]
        Return quadrature points (in physical domain) and weights.

    Notes
    -----
    Input samples shape: (1, nsamples) in physical domain
    Output shape: (nsamples, nterms)
    Quadrature points shape: (1, npoints) in physical domain
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
        """Evaluate basis functions at physical domain sample points.

        Parameters
        ----------
        samples : Array
            Sample points in physical domain. Shape: (1, nsamples)

        Returns
        -------
        Array
            Basis values. Shape: (nsamples, nterms)
        """
        ...

    def gauss_quadrature_rule(self, npoints: int) -> Tuple[Array, Array]:
        """Compute Gaussian quadrature rule in physical domain.

        Parameters
        ----------
        npoints : int
            Number of quadrature points.

        Returns
        -------
        points : Array
            Quadrature points in physical domain. Shape: (1, npoints)
        weights : Array
            Quadrature weights. Shape: (npoints, 1)
        """
        ...
