"""Jacobi polynomial family: Jacobi, Legendre, Chebyshev.

These polynomials are orthonormal on [-1, 1] with weight function:
    w(x) = (1-x)^alpha * (1+x)^beta

Special cases:
    - Legendre: alpha=beta=0 (uniform weight)
    - Chebyshev 1st kind: alpha=beta=-0.5
    - Chebyshev 2nd kind: alpha=beta=0.5
"""

import math
from typing import Generic, Tuple

from scipy import special as sp

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.univariate.globalpoly.orthopoly_base import (
    OrthonormalPolynomial1D,
)


def jacobi_recurrence(
    ncoefs: int,
    alpha: Array,
    beta: Array,
    probability: bool,
    bkd: Backend[Array],
) -> Array:
    """Compute recursion coefficients for Jacobi polynomials.

    Jacobi polynomials are orthonormal w.r.t. the weight function:
        w(x) = (1-x)^alpha * (1+x)^beta  on [-1, 1]

    For Beta(a,b) distribution on [0,1], use alpha=b-1, beta=a-1.

    Parameters
    ----------
    ncoefs : int
        Number of coefficients to compute.
    alpha : Array
        First Jacobi parameter.
    beta : Array
        Second Jacobi parameter.
    probability : bool
        If True, normalize for probability measure (set b[0] = 1.0).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Recursion coefficients. Shape: (ncoefs, 2) or (0, 2) if ncoefs == 0.

    Notes
    -----
    For ncoefs=1:
        - a[0] = (beta - alpha) / (alpha + beta + 2) is the 1-point quadrature node
        - b[0] = 1 (probability) or integral of weight function (physics measure)

    For Legendre (alpha=beta=0): a[0]=0, b[0]=1, so 1-pt quad is x=0, w=1.
    """
    if ncoefs == 0:
        return bkd.ones((0, 2))

    ab = bkd.ones((ncoefs, 2)) * bkd.array([beta**2.0 - alpha**2.0, 1.0])

    # First coefficient (always needed for ncoefs >= 1)
    ab[0, 0] = (beta - alpha) / (alpha + beta + 2.0)
    ab[0, 1] = bkd.exp(
        (alpha + beta + 1.0) * math.log(2.0)
        + bkd.asarray(
            sp.gammaln(bkd.to_numpy(alpha + 1.0))
            + sp.gammaln(bkd.to_numpy(beta + 1.0))
            - sp.gammaln(bkd.to_numpy(alpha + beta + 2.0))
        )
    )

    if ncoefs > 1:
        ab[1, 0] /= (2.0 + alpha + beta) * (4.0 + alpha + beta)
        ab[1, 1] = (
            4.0
            * (alpha + 1.0)
            * (beta + 1.0)
            / ((alpha + beta + 2.0) ** 2 * (alpha + beta + 3.0))
        )

    if ncoefs > 2:
        inds = bkd.arange(2, ncoefs)
        ab[2:, 0] /= (2.0 * inds + alpha + beta) * (2 * inds + alpha + beta + 2.0)
        ab[2:, 1] = (
            4 * inds * (inds + alpha) * (inds + beta) * (inds + alpha + beta)
        )
        ab[2:, 1] /= (
            (2.0 * inds + alpha + beta) ** 2
            * (2.0 * inds + alpha + beta + 1.0)
            * (2.0 * inds + alpha + beta - 1)
        )

    ab[:, 1] = bkd.sqrt(ab[:, 1])

    if probability:
        ab[0, 1] = 1.0

    return ab


class JacobiPolynomial1D(OrthonormalPolynomial1D[Array], Generic[Array]):
    """Jacobi polynomials orthonormal on [-1, 1].

    Weight function: w(x) = (1-x)^alpha * (1+x)^beta

    Parameters
    ----------
    alpha : float
        First Jacobi parameter. For Beta(a,b) distribution, use alpha=b-1.
    beta : float
        Second Jacobi parameter. For Beta(a,b) distribution, use beta=a-1.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, alpha: float, beta: float, bkd: Backend[Array]):
        super().__init__(bkd)
        self._alpha = alpha
        self._beta = beta

    def _get_recursion_coefficients(self, ncoefs: int) -> Array:
        return jacobi_recurrence(
            ncoefs,
            alpha=self._bkd.asarray(self._alpha),
            beta=self._bkd.asarray(self._beta),
            probability=self._prob_meas,
            bkd=self._bkd,
        )

    def gauss_lobatto_quadrature_rule(
        self, npoints: int
    ) -> Tuple[Array, Array]:
        """Compute Gauss-Lobatto quadrature rule.

        Gauss-Lobatto quadrature includes the endpoints [-1, 1].

        Parameters
        ----------
        npoints : int
            Number of quadrature points (must be >= 3).

        Returns
        -------
        points : Array
            Quadrature points. Shape: (1, npoints)
        weights : Array
            Quadrature weights. Shape: (npoints, 1)
        """
        if npoints == 1:
            return self.gauss_quadrature_rule(npoints)
        if npoints < 3:
            raise ValueError("Gauss-Lobatto requires at least 3 points")
        if self._rcoefs is None:
            raise ValueError("Must set nterms before computing quadrature")
        if npoints > self._ncoefs():
            raise ValueError(
                f"npoints={npoints} > ncoefs={self._ncoefs()}. "
                "Call set_nterms with a larger value first."
            )

        N = npoints - 2
        rcoefs = self._bkd.copy(self._rcoefs[:npoints])

        # Correct first b coefficient
        rcoefs[0, 1] = math.exp(
            (self._alpha + self._beta + 1.0) * math.log(2.0)
            + self._bkd.gammaln(self._bkd.asarray(self._alpha) + 1.0)
            + self._bkd.gammaln(self._bkd.asarray(self._beta) + 1.0)
            - self._bkd.gammaln(
                self._bkd.asarray(self._alpha + self._beta) + 2.0
            )
        )

        # Modify last coefficients for Lobatto rule
        rcoefs[npoints - 1, 0] = (self._alpha - self._beta) / (
            2 * N + self._alpha + self._beta + 2
        )
        rcoefs[npoints - 1, 1] = math.sqrt(
            4
            * (N + self._alpha + 1)
            * (N + self._beta + 1)
            * (N + self._alpha + self._beta + 1)
            / (
                (2 * N + self._alpha + self._beta + 1)
                * (2 * N + self._alpha + self._beta + 2) ** 2
            )
        )
        return self._gauss_quadrature_from_rcoefs(npoints, rcoefs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(alpha={self._alpha}, "
            f"beta={self._beta}, nterms={self.nterms()})"
        )


class LegendrePolynomial1D(JacobiPolynomial1D[Array], Generic[Array]):
    """Legendre polynomials orthonormal on [-1, 1].

    Legendre polynomials are Jacobi polynomials with alpha=beta=0,
    corresponding to uniform weight on [-1, 1].

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]):
        super().__init__(0.0, 0.0, bkd)

    def __repr__(self) -> str:
        return f"LegendrePolynomial1D(nterms={self.nterms()})"


class Chebyshev1stKindPolynomial1D(JacobiPolynomial1D[Array], Generic[Array]):
    """Chebyshev polynomials of the first kind.

    Orthogonal w.r.t. weight w(x) = 1/sqrt(1-x^2) on [-1, 1].
    These are Jacobi polynomials with alpha=beta=-0.5.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]):
        super().__init__(-0.5, -0.5, bkd)

    def __call__(self, samples: Array) -> Array:
        """Evaluate Chebyshev polynomials.

        Note: Applies scaling factor to match standard normalization.
        """
        vals = super().__call__(samples)
        vals[:, 1:] /= 2**0.5
        return vals

    def jacobian_batch(self, samples: Array) -> Array:
        """Evaluate first derivatives of Chebyshev polynomials.

        Note: Applies same scaling factor as evaluation.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples). Must be 2D.

        Returns
        -------
        Array
            First derivatives. Shape: (nsamples, nterms)
        """
        jacs = super().jacobian_batch(samples)
        jacs[:, 1:] /= 2**0.5
        return jacs

    def hessian_batch(self, samples: Array) -> Array:
        """Evaluate second derivatives of Chebyshev polynomials.

        Note: Applies same scaling factor as evaluation.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples). Must be 2D.

        Returns
        -------
        Array
            Second derivatives. Shape: (nsamples, nterms)
        """
        hess = super().hessian_batch(samples)
        hess[:, 1:] /= 2**0.5
        return hess

    def gauss_quadrature_rule(self, npoints: int) -> Tuple[Array, Array]:
        """Compute Gauss-Chebyshev quadrature.

        Note: Weights include factor of pi for the Chebyshev weight function.
        """
        quad_x, quad_w = super().gauss_quadrature_rule(npoints)
        return quad_x, quad_w * math.pi

    def __repr__(self) -> str:
        return f"Chebyshev1stKindPolynomial1D(nterms={self.nterms()})"


class Chebyshev2ndKindPolynomial1D(JacobiPolynomial1D[Array], Generic[Array]):
    """Chebyshev polynomials of the second kind.

    Orthogonal w.r.t. weight w(x) = sqrt(1-x^2) on [-1, 1].
    These are Jacobi polynomials with alpha=beta=0.5.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]):
        super().__init__(0.5, 0.5, bkd)

    def gauss_quadrature_rule(self, npoints: int) -> Tuple[Array, Array]:
        """Compute Gauss-Chebyshev quadrature of the second kind.

        Note: Weights include factor of pi/2.
        """
        quad_x, quad_w = super().gauss_quadrature_rule(npoints)
        return quad_x, quad_w * math.pi / 2

    def __repr__(self) -> str:
        return f"Chebyshev2ndKindPolynomial1D(nterms={self.nterms()})"
