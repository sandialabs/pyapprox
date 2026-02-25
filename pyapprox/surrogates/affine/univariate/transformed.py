"""Physical-domain basis wrappers for univariate polynomials.

This module provides wrapper classes that ensure polynomial bases accept
samples in the physical domain (user domain), regardless of whether the
underlying polynomial operates in canonical or physical domain.

Classes
-------
TransformedBasis1D
    Wraps a canonical-domain polynomial with a domain transform.
    Transforms samples from physical to canonical before evaluation.

NativeBasis1D
    Wraps a physical-domain polynomial (no transform needed).
    Used for discrete polynomials that operate directly on support points.

These wrappers implement PhysicalDomainBasis1DProtocol and ensure consistent
behavior for PCE and sparse grid construction.

Example
-------
>>> from pyapprox.util.backends.numpy import NumpyBkd
>>> from pyapprox.surrogates.affine.univariate.globalpoly import (
...     LegendrePolynomial1D,
... )
>>> from pyapprox.surrogates.affine.univariate.transforms import (
...     BoundedAffineTransform1D,
... )
>>> bkd = NumpyBkd()
>>> poly = LegendrePolynomial1D(bkd)
>>> transform = BoundedAffineTransform1D(bkd, lb=0.0, ub=1.0)
>>> basis = TransformedBasis1D(poly, transform)
>>> basis.set_nterms(5)
>>> samples = bkd.asarray([[0.0, 0.25, 0.5, 0.75, 1.0]])  # Physical domain [0,1]
>>> values = basis(samples)  # Transformed to [-1,1] internally
"""

from typing import Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.affine.univariate.transforms import (
    Univariate1DTransformProtocol,
)


class TransformedBasis1D(Generic[Array]):
    """Wrapper for canonical-domain polynomials with domain transform.

    This class wraps a polynomial that operates in canonical domain and
    applies a transform to map physical-domain samples to canonical domain
    before evaluation. Quadrature points are mapped back to physical domain.

    Parameters
    ----------
    polynomial : OrthonormalPolynomial1D[Array]
        The underlying polynomial (expects canonical domain samples).
    transform : Univariate1DTransformProtocol[Array]
        Transform mapping physical domain to canonical domain.

    Raises
    ------
    ValueError
        If the polynomial already operates in physical domain.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.surrogates.affine.univariate.globalpoly import (
    ...     LegendrePolynomial1D,
    ... )
    >>> from pyapprox.surrogates.affine.univariate.transforms import (
    ...     BoundedAffineTransform1D,
    ... )
    >>> bkd = NumpyBkd()
    >>> poly = LegendrePolynomial1D(bkd)
    >>> transform = BoundedAffineTransform1D(bkd, lb=0.0, ub=1.0)
    >>> basis = TransformedBasis1D(poly, transform)
    >>> basis.set_nterms(5)
    >>> # Samples in physical domain [0, 1]
    >>> samples = bkd.asarray([[0.0, 0.5, 1.0]])
    >>> values = basis(samples)  # Shape: (3, 5)
    """

    def __init__(
        self,
        polynomial,  # OrthonormalPolynomial1D[Array]
        transform: Univariate1DTransformProtocol[Array],
    ) -> None:
        if hasattr(polynomial, "operates_in_physical_domain"):
            if polynomial.operates_in_physical_domain():
                raise ValueError(
                    f"TransformedBasis1D requires a canonical-domain polynomial, "
                    f"but {type(polynomial).__name__} operates in physical domain. "
                    f"Use NativeBasis1D instead."
                )
        self._polynomial = polynomial
        self._transform = transform

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._polynomial.bkd()

    def polynomial(self):
        """Return the underlying polynomial."""
        return self._polynomial

    def transform(self) -> Univariate1DTransformProtocol[Array]:
        """Return the domain transform."""
        return self._transform

    def set_nterms(self, nterms: int) -> None:
        """Set the number of basis terms."""
        self._polynomial.set_nterms(nterms)

    def nterms(self) -> int:
        """Return the number of basis terms."""
        return self._polynomial.nterms()

    def __call__(self, samples: Array) -> Array:
        """Evaluate basis functions at physical domain sample points.

        Transforms samples to canonical domain, evaluates the polynomial,
        and returns the values.

        Parameters
        ----------
        samples : Array
            Sample points in physical domain. Shape: (1, nsamples)

        Returns
        -------
        Array
            Basis values. Shape: (nsamples, nterms)
        """
        canonical = self._transform.map_to_canonical(samples)
        return self._polynomial(canonical)

    def jacobian_batch(self, samples: Array) -> Array:
        """Evaluate first derivatives at physical domain sample points.

        Transforms samples to canonical domain, evaluates the polynomial
        derivatives, and applies the chain rule for the domain transform.

        Parameters
        ----------
        samples : Array
            Sample points in physical domain. Shape: (1, nsamples). Must be 2D.

        Returns
        -------
        Array
            First derivatives. Shape: (nsamples, nterms)
        """
        canonical = self._transform.map_to_canonical(samples)
        # Chain rule: d/dx_phys = d/dx_can * dx_can/dx_phys
        # jacobian_factor = dx_can/dx_phys = 1/scale
        return self._polynomial.jacobian_batch(canonical) * self._transform.jacobian_factor()

    def hessian_batch(self, samples: Array) -> Array:
        """Evaluate second derivatives at physical domain sample points.

        Transforms samples to canonical domain, evaluates the polynomial
        second derivatives, and applies the chain rule twice.

        Parameters
        ----------
        samples : Array
            Sample points in physical domain. Shape: (1, nsamples). Must be 2D.

        Returns
        -------
        Array
            Second derivatives. Shape: (nsamples, nterms)
        """
        canonical = self._transform.map_to_canonical(samples)
        # Chain rule twice: d²/dx_phys² = d²/dx_can² * (dx_can/dx_phys)²
        jac_factor = self._transform.jacobian_factor()
        return self._polynomial.hessian_batch(canonical) * (jac_factor ** 2)

    def derivatives(self, samples: Array, order: int) -> Array:
        """Evaluate derivatives of specified order at physical domain samples.

        Parameters
        ----------
        samples : Array
            Sample points in physical domain. Shape: (1, nsamples)
        order : int
            Derivative order. 0 = values, 1 = first derivative, etc.

        Returns
        -------
        Array
            Derivatives. Shape: (nsamples, nterms)
        """
        if order == 0:
            return self(samples)
        canonical = self._transform.map_to_canonical(samples)
        jac_factor = self._transform.jacobian_factor()
        return self._polynomial.derivatives(canonical, order) * (jac_factor ** order)

    def gauss_quadrature_rule(self, npoints: int) -> Tuple[Array, Array]:
        """Compute Gaussian quadrature rule in physical domain.

        Gets quadrature from the underlying polynomial (in canonical domain)
        and maps the points back to physical domain.

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
        canonical_pts, weights = self._polynomial.gauss_quadrature_rule(npoints)
        physical_pts = self._transform.map_from_canonical(canonical_pts)
        return physical_pts, weights

    def __repr__(self) -> str:
        return (
            f"TransformedBasis1D("
            f"polynomial={self._polynomial!r}, "
            f"transform={self._transform!r})"
        )


class NativeBasis1D(Generic[Array]):
    """Wrapper for physical-domain polynomials (no transform needed).

    This class wraps a polynomial that already operates in physical domain
    (e.g., discrete numeric polynomials). No transformation is applied.

    Parameters
    ----------
    polynomial : OrthonormalPolynomial1D[Array]
        The underlying polynomial (already operates in physical domain).

    Raises
    ------
    ValueError
        If the polynomial does not operate in physical domain.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.surrogates.affine.univariate.globalpoly.numeric import (
    ...     DiscreteNumericOrthonormalPolynomial1D,
    ... )
    >>> bkd = NumpyBkd()
    >>> xk = bkd.asarray([0.0, 1.0, 2.0])
    >>> pk = bkd.asarray([0.25, 0.5, 0.25])
    >>> poly = DiscreteNumericOrthonormalPolynomial1D(bkd, xk, pk)
    >>> basis = NativeBasis1D(poly)
    >>> basis.set_nterms(3)
    >>> # Samples are actual support values
    >>> samples = bkd.asarray([[0.0, 1.0, 2.0]])
    >>> values = basis(samples)  # Shape: (3, 3)
    """

    def __init__(self, polynomial) -> None:  # OrthonormalPolynomial1D[Array]
        if hasattr(polynomial, "operates_in_physical_domain"):
            if not polynomial.operates_in_physical_domain():
                raise ValueError(
                    f"NativeBasis1D requires a physical-domain polynomial, "
                    f"but {type(polynomial).__name__} operates in canonical domain. "
                    f"Use TransformedBasis1D instead."
                )
        self._polynomial = polynomial

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._polynomial.bkd()

    def polynomial(self):
        """Return the underlying polynomial."""
        return self._polynomial

    def set_nterms(self, nterms: int) -> None:
        """Set the number of basis terms."""
        self._polynomial.set_nterms(nterms)

    def nterms(self) -> int:
        """Return the number of basis terms."""
        return self._polynomial.nterms()

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
        return self._polynomial(samples)

    def jacobian_batch(self, samples: Array) -> Array:
        """Evaluate first derivatives at physical domain sample points.

        Parameters
        ----------
        samples : Array
            Sample points in physical domain. Shape: (1, nsamples). Must be 2D.

        Returns
        -------
        Array
            First derivatives. Shape: (nsamples, nterms)
        """
        return self._polynomial.jacobian_batch(samples)

    def hessian_batch(self, samples: Array) -> Array:
        """Evaluate second derivatives at physical domain sample points.

        Parameters
        ----------
        samples : Array
            Sample points in physical domain. Shape: (1, nsamples). Must be 2D.

        Returns
        -------
        Array
            Second derivatives. Shape: (nsamples, nterms)
        """
        return self._polynomial.hessian_batch(samples)

    def derivatives(self, samples: Array, order: int) -> Array:
        """Evaluate derivatives of specified order at physical domain samples.

        Parameters
        ----------
        samples : Array
            Sample points in physical domain. Shape: (1, nsamples)
        order : int
            Derivative order. 0 = values, 1 = first derivative, etc.

        Returns
        -------
        Array
            Derivatives. Shape: (nsamples, nterms)
        """
        return self._polynomial.derivatives(samples, order)

    def gauss_quadrature_rule(self, npoints: int) -> Tuple[Array, Array]:
        """Compute Gaussian quadrature rule in physical domain.

        For physical-domain polynomials, quadrature points are already
        in physical domain.

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
        return self._polynomial.gauss_quadrature_rule(npoints)

    def __repr__(self) -> str:
        return f"NativeBasis1D(polynomial={self._polynomial!r})"


__all__ = [
    "TransformedBasis1D",
    "NativeBasis1D",
]
