"""Laguerre orthonormal polynomials.

Laguerre polynomials are orthogonal with respect to the weight function
x^rho * exp(-x) on [0, infinity), which corresponds to the Gamma distribution.
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.affine.univariate.globalpoly.orthopoly_base import (
    OrthonormalPolynomial1D,
)


def laguerre_recurrence(
    nterms: int,
    rho: float,
    bkd: Backend[Array],
    probability: bool = True,
) -> Array:
    """Compute recursion coefficients for Laguerre polynomials.

    Laguerre polynomials are orthogonal with respect to:
        w(x) = x^rho * exp(-x) on [0, infinity)

    This corresponds to a Gamma(rho+1, 1) distribution.

    Parameters
    ----------
    nterms : int
        Number of terms (recursion coefficients).
    rho : float
        Shape parameter (rho > -1).
    bkd : Backend[Array]
        Computational backend.
    probability : bool
        If True, normalize for probability measure. Default: True.

    Returns
    -------
    Array
        Recursion coefficients. Shape: (nterms, 2)
        Column 0: a_n coefficients
        Column 1: b_n coefficients
    """
    if rho <= -1:
        raise ValueError(f"rho must be > -1, got {rho}")

    ab = bkd.zeros((nterms, 2))

    # a_n = 2*n + 1 + rho for n >= 0
    for nn in range(nterms):
        ab[nn, 0] = 2 * nn + 1 + rho

    # b_0 = sqrt(Gamma(1 + rho)) for probability measure
    # b_n = sqrt(n * (n + rho)) for n >= 1
    if probability:
        ab[0, 1] = 1.0
    else:
        # Use gammaln for numerical stability
        ab[0, 1] = bkd.sqrt(bkd.exp(bkd.gammaln(bkd.asarray([1.0 + rho]))))[0]

    for nn in range(1, nterms):
        ab[nn, 1] = bkd.sqrt(bkd.asarray([nn * (nn + rho)]))[0]

    return ab


class LaguerrePolynomial1D(OrthonormalPolynomial1D[Array], Generic[Array]):
    """Laguerre orthonormal polynomial basis.

    Orthogonal with respect to the Gamma distribution weight:
        w(x) = x^rho * exp(-x) on [0, infinity)

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    rho : float
        Shape parameter (rho > -1). Default: 0.0.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> poly = LaguerrePolynomial1D(bkd, rho=0.0)
    >>> poly.set_nterms(5)
    >>> x = bkd.linspace(0.0, 10.0, 100)
    >>> values = poly(bkd.reshape(x, (1, -1)))
    """

    def __init__(
        self,
        bkd: Backend[Array],
        rho: float = 0.0,
    ):
        if rho <= -1:
            raise ValueError(f"rho must be > -1, got {rho}")
        self._rho = rho
        super().__init__(bkd)

    @property
    def rho(self) -> float:
        """Return the shape parameter."""
        return self._rho

    def _get_recursion_coefficients(self, nterms: int) -> Array:
        """Compute Laguerre recursion coefficients.

        Parameters
        ----------
        nterms : int
            Number of coefficients needed.

        Returns
        -------
        Array
            Recursion coefficients. Shape: (nterms, 2)
        """
        return laguerre_recurrence(nterms, self._rho, self._bkd, probability=True)

    def __repr__(self) -> str:
        return f"LaguerrePolynomial1D(rho={self._rho}, nterms={self.nterms()})"
