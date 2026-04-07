"""Discrete orthonormal polynomials.

This module provides polynomials orthogonal with respect to discrete
probability distributions:
- Krawtchouk: Binomial distribution
- Hahn: Hypergeometric distribution
- Charlier: Poisson distribution
- DiscreteChebyshev: Uniform discrete distribution
"""

from typing import Generic

from pyapprox.surrogates.affine.univariate.globalpoly.orthopoly_base import (
    OrthonormalPolynomial1D,
)
from pyapprox.util.backends.protocols import Array, Backend


def krawtchouk_recurrence(
    nterms: int,
    n_trials: int,
    p: float,
    bkd: Backend[Array],
) -> Array:
    """Compute recursion coefficients for Krawtchouk polynomials.

    Krawtchouk polynomials are orthogonal with respect to the
    Binomial(n_trials, p) distribution.

    Parameters
    ----------
    nterms : int
        Number of terms (recursion coefficients).
    n_trials : int
        Number of trials in binomial distribution.
    p : float
        Success probability (0 < p < 1).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Recursion coefficients. Shape: (nterms, 2)
    """
    if not 0 < p < 1:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if nterms > n_trials + 1:
        raise ValueError(
            f"nterms ({nterms}) cannot exceed n_trials + 1 ({n_trials + 1})"
        )

    ab = bkd.zeros((nterms, 2))

    # a_k = p*(n-k) + k*(1-p)
    for kk in range(nterms):
        ab[kk, 0] = p * (n_trials - kk) + kk * (1 - p)

    # b_0 = 1 (probability measure)
    # b_k = sqrt(p*(1-p)*k*(n-k+1)) for k >= 1
    ab[0, 1] = 1.0
    for kk in range(1, nterms):
        ab[kk, 1] = bkd.sqrt(bkd.asarray([p * (1 - p) * kk * (n_trials - kk + 1)]))[0]

    return ab


class KrawtchoukPolynomial1D(OrthonormalPolynomial1D[Array], Generic[Array]):
    """Krawtchouk orthonormal polynomial basis.

    Orthogonal with respect to Binomial(n_trials, p) distribution.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    n_trials : int
        Number of trials in binomial distribution.
    p : float
        Success probability (0 < p < 1).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        n_trials: int,
        p: float,
    ):
        if not 0 < p < 1:
            raise ValueError(f"p must be in (0, 1), got {p}")
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}")
        self._n_trials = n_trials
        self._p = p
        super().__init__(bkd)

    @property
    def n_trials(self) -> int:
        """Return number of trials."""
        return self._n_trials

    @property
    def p(self) -> float:
        """Return success probability."""
        return self._p

    def _get_recursion_coefficients(self, nterms: int) -> Array:
        return krawtchouk_recurrence(nterms, self._n_trials, self._p, self._bkd)

    def __repr__(self) -> str:
        return (
            f"KrawtchoukPolynomial1D(n_trials={self._n_trials}, "
            f"p={self._p}, nterms={self.nterms()})"
        )


def hahn_recurrence(
    nterms: int,
    N: int,
    alpha: float,
    beta: float,
    bkd: Backend[Array],
) -> Array:
    """Compute recursion coefficients for Hahn polynomials.

    Hahn polynomials are orthogonal with respect to the
    Hypergeometric distribution.

    Parameters
    ----------
    nterms : int
        Number of terms (recursion coefficients).
    N : int
        Population size parameter.
    alpha : float
        First shape parameter (alpha > -1 or alpha < -N).
    beta : float
        Second shape parameter (beta > -1 or beta < -N).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Recursion coefficients. Shape: (nterms, 2)
    """
    if nterms > N + 1:
        raise ValueError(f"nterms ({nterms}) cannot exceed N + 1 ({N + 1})")

    ab = bkd.zeros((nterms, 2))
    ab[0, 1] = 1.0  # probability measure

    for nn in range(nterms):
        if nn == 0:
            # a_0
            num = (alpha + 1) * (alpha + beta + 2) + (beta + 1) * N
            denom = alpha + beta + 2
            ab[0, 0] = num / denom
        else:
            # a_n for n >= 1
            n = nn
            t1 = (alpha + beta + 2 * n) * (alpha + beta + 2 * n + 2)
            if abs(t1) < 1e-14:
                ab[n, 0] = 0.0
            else:
                num1 = (alpha + n + 1) * (alpha + beta + n + 1) * (N - n)
                num2 = (beta + n + 1) * n * (alpha + beta + N + n + 1)
                ab[n, 0] = (num1 + num2) / t1

        if nn > 0:
            # b_n for n >= 1
            n = nn
            t1 = (alpha + beta + 2 * n - 1) * (alpha + beta + 2 * n) ** 2
            t1 *= alpha + beta + 2 * n + 1
            if abs(t1) < 1e-14:
                ab[n, 1] = 0.0
            else:
                num = n * (alpha + n) * (beta + n) * (alpha + beta + n)
                num *= (N - n + 1) * (alpha + beta + N + n)
                ab[n, 1] = bkd.sqrt(bkd.asarray([num / t1]))[0]

    return ab


class HahnPolynomial1D(OrthonormalPolynomial1D[Array], Generic[Array]):
    """Hahn orthonormal polynomial basis.

    Orthogonal with respect to Hypergeometric distribution.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    N : int
        Population size parameter.
    alpha : float
        First shape parameter.
    beta : float
        Second shape parameter.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        N: int,
        alpha: float,
        beta: float,
    ):
        self._N = N
        self._alpha = alpha
        self._beta = beta
        super().__init__(bkd)

    @property
    def N(self) -> int:
        """Return population size parameter."""
        return self._N

    @property
    def alpha(self) -> float:
        """Return first shape parameter."""
        return self._alpha

    @property
    def beta(self) -> float:
        """Return second shape parameter."""
        return self._beta

    def _get_recursion_coefficients(self, nterms: int) -> Array:
        return hahn_recurrence(nterms, self._N, self._alpha, self._beta, self._bkd)

    def __repr__(self) -> str:
        return (
            f"HahnPolynomial1D(N={self._N}, alpha={self._alpha}, "
            f"beta={self._beta}, nterms={self.nterms()})"
        )


def charlier_recurrence(
    nterms: int,
    mu: float,
    bkd: Backend[Array],
) -> Array:
    """Compute recursion coefficients for Charlier polynomials.

    Charlier polynomials are orthogonal with respect to the
    Poisson(mu) distribution.

    Parameters
    ----------
    nterms : int
        Number of terms (recursion coefficients).
    mu : float
        Poisson rate parameter (mu > 0).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Recursion coefficients. Shape: (nterms, 2)
    """
    if mu <= 0:
        raise ValueError(f"mu must be > 0, got {mu}")

    ab = bkd.zeros((nterms, 2))

    # a_k = mu + k
    for kk in range(nterms):
        ab[kk, 0] = mu + kk

    # b_0 = 1 (probability measure)
    # b_k = sqrt(mu * k) for k >= 1
    ab[0, 1] = 1.0
    for kk in range(1, nterms):
        ab[kk, 1] = bkd.sqrt(bkd.asarray([mu * kk]))[0]

    return ab


class CharlierPolynomial1D(OrthonormalPolynomial1D[Array], Generic[Array]):
    """Charlier orthonormal polynomial basis.

    Orthogonal with respect to Poisson(mu) distribution.

    This polynomial operates in physical domain - it expects samples
    directly from the Poisson support {0, 1, 2, ...}.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    mu : float
        Poisson rate parameter (mu > 0).
    """

    # Operates in physical domain (Poisson support {0, 1, 2, ...})
    _operates_in_physical_domain = True

    def __init__(
        self,
        bkd: Backend[Array],
        mu: float,
    ):
        if mu <= 0:
            raise ValueError(f"mu must be > 0, got {mu}")
        self._mu = mu
        super().__init__(bkd)

    @property
    def mu(self) -> float:
        """Return Poisson rate parameter."""
        return self._mu

    def _get_recursion_coefficients(self, nterms: int) -> Array:
        return charlier_recurrence(nterms, self._mu, self._bkd)

    def __repr__(self) -> str:
        return f"CharlierPolynomial1D(mu={self._mu}, nterms={self.nterms()})"


def discrete_chebyshev_recurrence(
    nterms: int,
    N: int,
    bkd: Backend[Array],
) -> Array:
    """Compute recursion coefficients for discrete Chebyshev polynomials.

    Discrete Chebyshev polynomials are orthogonal with respect to
    the uniform discrete distribution on N points {0, 1, ..., N-1}.

    Parameters
    ----------
    nterms : int
        Number of terms (recursion coefficients).
    N : int
        Number of discrete points.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Recursion coefficients. Shape: (nterms, 2)
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    if nterms > N:
        raise ValueError(f"nterms ({nterms}) cannot exceed N ({N})")

    ab = bkd.zeros((nterms, 2))

    # a_k = (N-1)/2 (constant for symmetric measure)
    for kk in range(nterms):
        ab[kk, 0] = 0.5 * N * (1.0 - 1.0 / N)

    # b_k^2 = 0.25 * N^2 * (1 - (k/N)^2) / (4 - 1/k^2) for k >= 1
    for kk in range(1, nterms):
        ab[kk, 1] = 0.25 * N * N * (1 - (kk * 1.0 / N) ** 2) / (4 - 1.0 / kk**2)
    ab[:, 1] = bkd.sqrt(ab[:, 1])

    # b_0 = 1 (probability measure)
    ab[0, 1] = 1.0

    return ab


class DiscreteChebyshevPolynomial1D(OrthonormalPolynomial1D[Array], Generic[Array]):
    """Discrete Chebyshev orthonormal polynomial basis.

    Orthogonal with respect to uniform discrete distribution
    on N points {0, 1, ..., N-1}.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    N : int
        Number of discrete points.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        N: int,
    ):
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        self._N = N
        super().__init__(bkd)

    @property
    def N(self) -> int:
        """Return number of discrete points."""
        return self._N

    def _get_recursion_coefficients(self, nterms: int) -> Array:
        return discrete_chebyshev_recurrence(nterms, self._N, self._bkd)

    def __repr__(self) -> str:
        return f"DiscreteChebyshevPolynomial1D(N={self._N}, nterms={self.nterms()})"
