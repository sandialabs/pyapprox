"""Hermite polynomial family.

Hermite polynomials are orthonormal on (-inf, inf) with weight function:
    w(x) = x^(2*rho) * exp(-x^2)

Special case:
    - Probabilists' Hermite: rho=0, probability=True
      Orthonormal w.r.t. standard normal distribution
"""

from typing import Generic

from scipy import special as sp

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.affine.univariate.globalpoly.orthopoly_base import (
    OrthonormalPolynomial1D,
)


def hermite_recurrence(
    ncoefs: int,
    rho: Array,
    probability: bool,
    bkd: Backend[Array],
) -> Array:
    """Compute recursion coefficients for Hermite polynomials.

    Parameters
    ----------
    ncoefs : int
        Number of coefficients to compute.
    rho : Array
        Hermite parameter. Use rho=0 for probabilists' Hermite.
    probability : bool
        If True, normalize for probability measure (standard normal).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Recursion coefficients. Shape: (ncoefs, 2)
    """
    if ncoefs < 1:
        return bkd.ones((0, 2))

    ab = bkd.zeros((ncoefs, 2))
    ab[0, 1] = bkd.asarray(sp.gamma(bkd.to_numpy(rho + 0.5)))

    if rho == 0 and probability:
        ab[1:, 1] = bkd.arange(1.0, ncoefs)
    else:
        ab[1:, 1] = 0.5 * bkd.arange(1.0, ncoefs)

    ab[bkd.arange(ncoefs) % 2 == 1, 1] += rho

    ab[:, 1] = bkd.sqrt(ab[:, 1])

    if probability:
        ab[0, 1] = 1.0

    return ab


class HermitePolynomial1D(OrthonormalPolynomial1D[Array], Generic[Array]):
    """Hermite polynomials orthonormal on (-inf, inf).

    Weight function: w(x) = x^(2*rho) * exp(-x^2)

    For the standard normal distribution, use rho=0 and prob_meas=True.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    rho : float, optional
        Hermite parameter. Default 0.0 for probabilists' Hermite.
    prob_meas : bool, optional
        If True, use probability measure normalization. Default True.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        rho: float = 0.0,
        prob_meas: bool = True,
    ):
        super().__init__(bkd)
        self._rho = rho
        self._prob_meas = prob_meas

    def _get_recursion_coefficients(self, ncoefs: int) -> Array:
        return hermite_recurrence(
            ncoefs,
            rho=self._bkd.asarray(self._rho),
            probability=self._prob_meas,
            bkd=self._bkd,
        )

    def __repr__(self) -> str:
        return f"HermitePolynomial1D(rho={self._rho}, nterms={self.nterms()})"
