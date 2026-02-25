"""Base class for orthonormal polynomial 1D basis functions.

This module provides the abstract base class for orthonormal polynomial
families. Concrete implementations (Jacobi, Hermite, etc.) inherit from
OrthonormalPolynomial1D and implement _get_recursion_coefficients.
"""

from abc import abstractmethod
from typing import Generic, Tuple, Optional

from pyapprox.util.backends.protocols import Array, Backend


def evaluate_orthonormal_polynomial_1d(
    rcoefs: Array, bkd: Backend[Array], samples: Array
) -> Array:
    """Evaluate orthonormal polynomial using recursion coefficients.

    Parameters
    ----------
    rcoefs : Array
        Recursion coefficients. Shape: (nterms, 2)
        Column 0: alpha coefficients
        Column 1: beta coefficients
    bkd : Backend[Array]
        Computational backend.
    samples : Array
        Sample points. Shape: (1, nsamples)

    Returns
    -------
    Array
        Polynomial values. Shape: (nsamples, nterms)
    """
    # samples passed in is 2D array with shape [1, nsamples], squeeze to 1D
    samples_1d = samples[0]
    nsamples = samples_1d.shape[0]
    nterms = rcoefs.shape[0]

    vals = [bkd.full((nsamples,), 1.0 / rcoefs[0, 1])]

    if nterms > 1:
        vals.append(1 / rcoefs[1, 1] * ((samples_1d - rcoefs[0, 0]) * vals[0]))

    for jj in range(2, nterms):
        vals.append(
            1.0
            / rcoefs[jj, 1]
            * (
                (samples_1d - rcoefs[jj - 1, 0]) * vals[jj - 1]
                - rcoefs[jj - 1, 1] * vals[jj - 2]
            )
        )
    return bkd.stack(vals, axis=1)


def evaluate_orthonormal_polynomial_derivatives_1d(
    rcoefs: Array,
    bkd: Backend[Array],
    samples: Array,
    order: int,
    return_all: bool = False,
) -> Array:
    """Evaluate derivatives of orthonormal polynomial.

    Parameters
    ----------
    rcoefs : Array
        Recursion coefficients. Shape: (nterms, 2)
    bkd : Backend[Array]
        Computational backend.
    samples : Array
        Sample points. Shape: (1, nsamples)
    order : int
        Derivative order. 0 = values, 1 = first derivative, etc.
    return_all : bool, optional
        If True, return all derivatives up to order. Default False.

    Returns
    -------
    Array
        If return_all: Shape (nsamples, nterms * (order+1))
        Otherwise: Shape (nsamples, nterms)
    """
    if order < 0:
        raise ValueError(f"derivative order {order} must be >= 0")

    vals = evaluate_orthonormal_polynomial_1d(rcoefs, bkd, samples)

    # samples passed in is 2D array with shape [1, nsamples], squeeze to 1D
    samples_1d = samples[0]
    nsamples = samples_1d.shape[0]

    nterms = rcoefs.shape[0]
    a = rcoefs[:, 0]
    b = rcoefs[:, 1]

    result = [vals]
    vals_T = vals.T

    for _order in range(1, order + 1):
        can_derivs = []
        for jj in range(_order):
            can_derivs.append(bkd.full((nsamples,), 0.0))

        # Use log-gamma to avoid overflow issues
        can_derivs.append(
            bkd.full(
                (nsamples,),
                bkd.exp(
                    bkd.gammaln(bkd.asarray(_order + 1))
                    - 0.5 * bkd.sum(bkd.log(b[: _order + 1] ** 2))
                ),
            )
        )

        for jj in range(_order + 1, nterms):
            can_derivs.append(
                (
                    (samples_1d - a[jj - 1]) * can_derivs[jj - 1]
                    - b[jj - 1] * can_derivs[jj - 2]
                    + _order * vals_T[jj - 1]
                )
                / b[jj]
            )

        derivs = bkd.stack(can_derivs, axis=1)
        vals_T = can_derivs
        result.append(derivs)

    if return_all:
        return bkd.hstack(result)
    return result[-1]


class OrthonormalPolynomial1D(Generic[Array]):
    """Base class for orthonormal polynomial 1D basis functions.

    Orthonormal polynomials are defined by their recursion coefficients.
    Concrete subclasses implement _get_recursion_coefficients for specific
    polynomial families.

    This base class operates in the canonical domain of the polynomial.
    Transformations to/from physical domain are handled separately.

    Subclasses should set the class attribute `_operates_in_physical_domain`
    to True if they operate directly in the physical domain (e.g., discrete
    polynomials on actual support points). By default, polynomials operate
    in canonical domain and require a transform wrapper for physical domain
    samples.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Attributes
    ----------
    _rcoefs : Array or None
        Recursion coefficients. Shape: (nterms, 2)
    _prob_meas : bool
        If True, polynomials are orthonormal w.r.t. probability measure.
    _operates_in_physical_domain : bool
        Class attribute. If True, polynomial expects physical domain samples.
        If False (default), polynomial expects canonical domain samples.
    """

    # Class attribute: override in subclasses that operate in physical domain
    _operates_in_physical_domain: bool = False

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd
        self._rcoefs: Optional[Array] = None
        self._prob_meas: bool = True

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def operates_in_physical_domain(self) -> bool:
        """Return True if this polynomial operates in physical domain.

        Polynomials that return True expect samples directly from the
        marginal distribution's support (physical domain). Polynomials
        that return False (default) expect samples in canonical domain
        and require a transform wrapper when used with physical samples.

        Returns
        -------
        bool
            True if polynomial operates in physical domain, False otherwise.
        """
        return self._operates_in_physical_domain

    def _ncoefs(self) -> int:
        """Return number of computed recursion coefficients."""
        if self._rcoefs is None:
            raise ValueError("Recursion coefficients have not been set")
        return self._rcoefs.shape[0]

    @abstractmethod
    def _get_recursion_coefficients(self, ncoefs: int) -> Array:
        """Compute recursion coefficients for this polynomial family.

        Parameters
        ----------
        ncoefs : int
            Number of coefficients to compute.

        Returns
        -------
        Array
            Recursion coefficients. Shape: (ncoefs, 2)
        """
        raise NotImplementedError

    def set_nterms(self, nterms: int) -> None:
        """Set the number of basis terms.

        Computes and stores recursion coefficients for nterms polynomials.

        Parameters
        ----------
        nterms : int
            Number of polynomial terms.
        """
        if self._rcoefs is None or self._ncoefs() < nterms:
            self._rcoefs = self._bkd.array(
                self._get_recursion_coefficients(nterms)
            )
        elif self._ncoefs() >= nterms:
            self._rcoefs = self._rcoefs[:nterms, :]

    def nterms(self) -> int:
        """Return the number of basis terms."""
        if self._rcoefs is None:
            return 0
        return self._rcoefs.shape[0]

    def recursion_coefficients(self) -> Array:
        """Return the recursion coefficients.

        Returns
        -------
        Array
            Recursion coefficients. Shape: (nterms, 2)
        """
        if self._rcoefs is None:
            raise ValueError("Must set nterms before accessing coefficients")
        return self._rcoefs

    def __call__(self, samples: Array) -> Array:
        """Evaluate basis functions at sample points.

        Parameters
        ----------
        samples : Array
            Sample points in canonical domain. Shape: (1, nsamples)

        Returns
        -------
        Array
            Polynomial values. Shape: (nsamples, nterms)
        """
        if self._rcoefs is None:
            raise ValueError("Must set nterms before evaluation")
        return evaluate_orthonormal_polynomial_1d(
            self._rcoefs, self._bkd, samples
        )

    def jacobian_batch(self, samples: Array) -> Array:
        """Evaluate first derivatives of basis functions.

        Parameters
        ----------
        samples : Array
            Sample points in canonical domain. Shape: (1, nsamples). Must be 2D.

        Returns
        -------
        Array
            First derivatives. Shape: (nsamples, nterms)

        Raises
        ------
        ValueError
            If samples is not 2D with shape (1, nsamples).
        """
        return evaluate_orthonormal_polynomial_derivatives_1d(
            self._rcoefs, self._bkd, samples, order=1
        )

    def hessian_batch(self, samples: Array) -> Array:
        """Evaluate second derivatives of basis functions.

        Parameters
        ----------
        samples : Array
            Sample points in canonical domain. Shape: (1, nsamples). Must be 2D.

        Returns
        -------
        Array
            Second derivatives. Shape: (nsamples, nterms)

        Raises
        ------
        ValueError
            If samples is not 2D with shape (1, nsamples).
        """
        return evaluate_orthonormal_polynomial_derivatives_1d(
            self._rcoefs, self._bkd, samples, order=2
        )

    def derivatives(self, samples: Array, order: int) -> Array:
        """Evaluate derivatives of specified order.

        Parameters
        ----------
        samples : Array
            Sample points in canonical domain. Shape: (1, nsamples)
        order : int
            Derivative order. 0 = values, 1 = first derivative, etc.

        Returns
        -------
        Array
            Derivatives. Shape: (nsamples, nterms)
        """
        if order == 0:
            return self(samples)
        return evaluate_orthonormal_polynomial_derivatives_1d(
            self._rcoefs, self._bkd, samples, order=order
        )

    def gauss_quadrature_rule(self, npoints: int) -> Tuple[Array, Array]:
        """Compute Gaussian quadrature rule.

        Parameters
        ----------
        npoints : int
            Number of quadrature points.

        Returns
        -------
        points : Array
            Quadrature points in canonical domain. Shape: (1, npoints)
        weights : Array
            Quadrature weights. Shape: (npoints, 1)
        """
        if self._rcoefs is None:
            raise ValueError("Must set nterms before computing quadrature")
        if npoints > self._ncoefs():
            raise ValueError(
                f"npoints={npoints} > ncoefs={self._ncoefs()}. "
                "Call set_nterms with a larger value first."
            )
        return self._gauss_quadrature_from_rcoefs(npoints, self._rcoefs)

    def _gauss_quadrature_from_rcoefs(
        self, npoints: int, rcoefs: Array
    ) -> Tuple[Array, Array]:
        """Compute Gaussian quadrature from recursion coefficients.

        Forms the Jacobi matrix and computes eigenvalues/eigenvectors.

        Parameters
        ----------
        npoints : int
            Number of quadrature points.
        rcoefs : Array
            Recursion coefficients.

        Returns
        -------
        points : Array
            Quadrature points. Shape: (1, npoints)
        weights : Array
            Quadrature weights. Shape: (npoints, 1)
        """
        a = rcoefs[:, 0]
        b = rcoefs[:, 1]

        # Form Jacobi matrix
        J = (
            self._bkd.diag(a[:npoints], 0)
            + self._bkd.diag(b[1:npoints], 1)
            + self._bkd.diag(b[1:npoints], -1)
        )
        x, eigvecs = self._bkd.eigh(J)

        if self._prob_meas:
            w = b[0] * eigvecs[0, :] ** 2
        else:
            vals = self(x[None, :])[:, :npoints]
            w = 1.0 / self._bkd.sum(vals**2, axis=1)

        return x[None, :], w[:, None]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nterms={self.nterms()})"
