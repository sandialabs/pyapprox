"""Monomial basis functions for 1D approximation.

This module provides a simple monomial basis {1, x, x², x³, ...} for univariate
function approximation.
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class MonomialBasis1D(Generic[Array]):
    """Univariate monomial basis.

    The monomial basis consists of powers of x: {1, x, x², x³, ..., x^(n-1)}
    where n is the number of terms.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> basis = MonomialBasis1D(bkd)
    >>> basis.set_nterms(3)
    >>> samples = bkd.asarray([[0.0, 0.5, 1.0]])
    >>> basis(samples)  # Returns [[1, 0, 0], [1, 0.5, 0.25], [1, 1, 1]]
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd
        self._nterms = 0

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def set_nterms(self, nterms: int) -> None:
        """Set the number of basis terms.

        Parameters
        ----------
        nterms : int
            Number of monomials to use (degree + 1).
        """
        self._nterms = nterms

    def nterms(self) -> int:
        """Return the number of basis terms."""
        return self._nterms

    def __call__(self, samples: Array) -> Array:
        """Evaluate monomial basis at sample points.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples). Must be 2D.

        Returns
        -------
        Array
            Basis values. Shape: (nsamples, nterms)

        Raises
        ------
        ValueError
            If samples is not 2D with shape (1, nsamples).
        """
        powers = self._bkd.reshape(self._bkd.arange(self._nterms), (1, -1))
        return samples.T ** powers

    def jacobian_batch(self, samples: Array) -> Array:
        """Evaluate first derivatives of monomial basis.

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
        return self._derivatives(samples, 1)

    def hessian_batch(self, samples: Array) -> Array:
        """Evaluate second derivatives of monomial basis.

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
        return self._derivatives(samples, 2)

    def _derivatives(self, samples: Array, order: int) -> Array:
        """Evaluate derivatives of specified order.

        For monomials x^n, derivatives are:
            - 0th order: x^n
            - 1st order: n * x^(n-1)
            - 2nd order: n * (n-1) * x^(n-2)

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples)
        order : int
            Derivative order.

        Returns
        -------
        Array
            Derivatives. Shape: (nsamples, nterms)
        """
        if order < 0:
            raise ValueError(f"Derivative order {order} must be >= 0")
        if order == 0:
            return self(samples)

        # Powers after differentiation
        powers = self._bkd.hstack(
            (
                self._bkd.zeros((order,)),
                self._bkd.arange(self._nterms - order),
            )
        )

        # Coefficients from differentiation
        # For order=1: [0, 1, 2, 3, ...] (first order coefficients)
        # For order=2: [0, 0, 2, 6, 12, ...] (second order coefficients)
        if order == 1:
            consts = self._bkd.arange(self._nterms, dtype=samples.dtype)
        else:
            # For higher order, compute n * (n-1) * ... * (n-order+1)
            consts = self._bkd.ones(self._nterms, dtype=samples.dtype)
            for k in range(order):
                consts = consts * self._bkd.maximum(
                    self._bkd.arange(self._nterms) - k,
                    self._bkd.zeros(self._nterms),
                )

        return (samples.T ** powers[None, :]) * consts[None, :]

    def derivatives(self, samples: Array, order: int) -> Array:
        """Evaluate derivatives of specified order.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples). Must be 2D.
        order : int
            Derivative order. 0 = values, 1 = first, 2 = second, etc.

        Returns
        -------
        Array
            Derivatives. Shape: (nsamples, nterms)
        """
        return self._derivatives(samples, order)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MonomialBasis1D(nterms={self.nterms()})"
