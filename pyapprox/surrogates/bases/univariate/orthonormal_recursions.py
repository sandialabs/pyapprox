import numpy as np
from scipy import special as sp

from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


def jacobi_recurrence(
    N: int,
    alpha: float = 0.0,
    beta: float = 0.0,
    probability: bool = True,
    bkd: LinAlgMixin = NumpyLinAlgMixin,
) -> Array:
    r"""
    Compute the recursion coefficients of Jacobi polynomials which are
    orthonormal with respect to the Beta random variables

    Parameters
    ----------
    alpha : float
        The first parameter of the Jacobi polynomials. For the Beta
        distribution with parameters :math:`\hat{\alpha},\hat{\beta}` we have
        :math:`\alpha=\hat{\beta}-1`

    beta : float
        The second parameter of the Jacobi polynomials
        For the Beta distribution
        with parameters :math:`\hat{\alpha},\hat{\beta}` we have
        :math:`\beta=\hat{\alpha}-1`

    Returns
    -------
    ab : bkd.ndarray (Nterms,2)
        The recursion coefficients of the Nterms orthonormal polynomials
    """

    if N < 1:
        return bkd.ones((0, 2))

    ab = bkd.ones((N, 2)) * bkd.array([beta**2.0 - alpha**2.0, 1.0])

    # Special cases
    ab[0, 0] = (beta - alpha) / (alpha + beta + 2.0)
    ab[0, 1] = bkd.exp(
        (alpha + beta + 1.0) * bkd.log(2.0)
        + sp.gammaln(alpha + 1.0)
        + sp.gammaln(beta + 1.0)
        - sp.gammaln(alpha + beta + 2.0)
    )

    if N > 1:
        ab[1, 0] /= (2.0 + alpha + beta) * (4.0 + alpha + beta)
        ab[1, 1] = (
            4.0
            * (alpha + 1.0)
            * (beta + 1.0)
            / ((alpha + beta + 2.0) ** 2 * (alpha + beta + 3.0))
        )

    inds = bkd.arange(2.0, N)
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


def hermite_recurrence(
    Nterms: int,
    rho: float = 0.0,
    probability: bool = True,
    bkd: LinAlgMixin = NumpyLinAlgMixin,
) -> Array:
    r"""
    Compute the recursion coefficients of for the Hermite
    polynomial family.

    .. math:: x^{2\rho}\exp(-x^2)

    Parameters
    ----------
    rho : float
        The parameter of the hermite polynomials. The special case of
    :math:`\rho=0` and probability=True returns the probablists
    Hermite polynomial

    Returns
    -------
    ab : bkd.ndarray (Nterms,2)
        The recursion coefficients of the Nterms orthonormal polynomials
    """

    if Nterms < 1:
        return bkd.ones((0, 2))

    ab = bkd.zeros((Nterms, 2))
    ab[0, 1] = sp.gamma(rho + 0.5)  # = bkd.sqrt(bkd.pi) for rho=0

    if rho == 0 and probability:
        ab[1:, 1] = bkd.arange(1.0, Nterms)
    else:
        ab[1:, 1] = 0.5 * bkd.arange(1.0, Nterms)

    ab[bkd.arange(Nterms) % 2 == 1, 1] += rho

    ab[:, 1] = bkd.sqrt(ab[:, 1])

    if probability:
        ab[0, 1] = 1.0

    return ab


def charlier_recurrence(
    N: int,
    a: float,
    bkd: LinAlgMixin = NumpyLinAlgMixin,
) -> Array:
    r"""
    Compute the recursion coefficients of the polynomials which are
    orthonormal with respect to the Poisson distribution.

    Parameters
    ----------
    N : integer
        The number of polynomial terms requested

    a: float
        The rate parameter of the Poisson distribution

    Returns
    -------
    ab : bkd.ndarray (N,2)
        The recursion coefficients of the N orthonormal polynomials

    Notes
    -----
    Note as rate gets smaller the number of terms that can be accurately
    computed will decrease because the problem gets more ill conditioned.
    This is caused because the number of masses with significant weights
    gets smaller as rate does
    """

    if N < 1:
        return bkd.ones((0, 2))

    ab = bkd.zeros((N, 2))
    ab[0, 0] = a
    ab[0, 1] = 1
    for i in range(1, N):
        ab[i, 0] = a + i
        ab[i, 1] = a * i

    # orthonormal
    ab[:, 1] = bkd.sqrt(ab[:, 1])

    return ab


def krawtchouk_recurrence(
    Nterms: int,
    Ntrials: int,
    p: float,
    bkd: LinAlgMixin = NumpyLinAlgMixin,
) -> Array:
    r"""
    Compute the recursion coefficients of the polynomials which are
    orthonormal with respect to the binomial probability mass function

    .. math:: {N \choose k} p^k (1-p)^{(n-k)}

    which is the probability of k successes from N trials.

    Parameters
    ----------
    Nterms : integer
        The number of polynomial terms requested

    Ntrials : integer
        The number of trials

    p : float
        The probability of success :math:`p\in(0,1)`

    Returns
    -------
    ab : bkd.ndarray (Nterms,2)
        The recursion coefficients of the Nterms orthonormal polynomials
    """

    assert Nterms <= Ntrials
    assert p > 0 and p < 1

    if Nterms < 1:
        return bkd.ones((0, 2))

    ab = bkd.array(
        [
            [
                p * (Ntrials - n) + n * (1 - p),
                p * (1 - p) * n * (Ntrials - n + 1),
            ]
            for n in range(Nterms)
        ]
    )

    ab[:, 1] = bkd.sqrt(ab[:, 1])

    ab[0, 1] = 1.0

    # the probability flag does not apply here
    # (beta0 comes out 0 in the three term recurrence), instead we set it
    # to 1, the norm of the p0 polynomial

    return ab


def hahn_recurrence(
    Nterms: int,
    N: int,
    alphaPoly: float,
    betaPoly: float,
    bkd: LinAlgMixin = NumpyLinAlgMixin,
) -> Array:
    r"""
    Compute the recursion coefficients of the polynomials which are
    orthonormal with respect to the hypergeometric probability mass function

    .. math:: w(x)=\frac{{n \choose x}{M-n \choose N-x}}{{ M \choose N}}.

    for

    .. math:: \max(0, M-(M-n)) \le x \le \min(n, N)

    which describes the probability of x successes in N draws, without
    replacement, from a finite population of size M that contains exactly
    n successes.

    Parameters
    ----------
    Nterms : integer
        The number of polynomial terms requested

    N : integer
        The number of draws

    alphaPoly : integer
         :math:`-n+1`

    betPoly : integer
         :math:`-M-1+n`

    Returns
    -------
    ab : bkd.ndarray (Nterms,2)
        The recursion coefficients of the Nterms orthonormal polynomials
    """
    assert Nterms <= N

    if Nterms < 1:
        return bkd.ones((0, 2))

    An = bkd.zeros(Nterms)
    Cn = bkd.zeros(Nterms)
    for n in range(Nterms):
        numA = (alphaPoly + n + 1) * (N - n) * (n + alphaPoly + betaPoly + 1)
        numC = n * (betaPoly + n) * (N + alphaPoly + betaPoly + n + 1)
        denA = (alphaPoly + betaPoly + 2 * n + 1) * (
            alphaPoly + betaPoly + 2 * n + 2
        )
        denC = (alphaPoly + betaPoly + 2 * n + 1) * (
            alphaPoly + betaPoly + 2 * n
        )
        An[n] = numA / denA
        Cn[n] = numC / denC

    if Nterms == 1:
        return bkd.array([[An[0] + Cn[0], 1]])

    ab = bkd.array(
        [[An[0] + Cn[0], 1]]
        + [[An[n] + Cn[n], An[n - 1] * Cn[n]] for n in range(1, Nterms)]
    )

    ab[:, 1] = bkd.sqrt(ab[:, 1])

    ab[0, 1] = 1.0

    return ab


def discrete_chebyshev_recurrence(
    N: int, Ntrials: int, bkd: LinAlgMixin = NumpyLinAlgMixin
) -> Array:
    r"""
    Compute the recursion coefficients of the polynomials which are
    orthonormal with respect to the probability measure

    .. math:: w(x) = \frac{\delta_i(x)}{M}

    where :math:`\delta_i(x)` is the dirac delta function which is one when
    :math:`x=i`, for :math:`i=1,\ldots,M` and zero otherwise

    Parameters
    ----------
    N : integer
        The number of polynomial terms requested

    Ntrials : integer
        The number of probability masses (M)

    Returns
    -------
    ab : bkd.ndarray (N,2)
        The recursion coefficients of the N orthonormal polynomials
    """
    assert N <= Ntrials

    if N < 1:
        return bkd.ones((0, 2))

    ab = bkd.zeros((N, 2))
    ab[:, 0] = 0.5 * Ntrials * (1.0 - 1.0 / Ntrials)
    ab[0, 1] = Ntrials
    for i in range(1, N):
        ab[i, 1] = (
            0.25
            * Ntrials**2
            * (1 - (i * 1.0 / Ntrials) ** 2)
            / (4 - 1.0 / i**2)
        )

    ab[:, 1] = bkd.sqrt(ab[:, 1])

    ab[0, 1] = 1.0

    return ab


def convert_orthonormal_recurence_to_three_term_recurence(
    recursion_coefs: Array, bkd: LinAlgMixin = NumpyLinAlgMixin
) -> Array:
    r"""
    Convert two term recursion coefficients

    .. math:: b_{n+1} p_{n+1} = (x - a_n) p_n - \sqrt{b_n} p_{n-1}

    into the equivalent
    three recursion coefficients

    .. math:: p_{n+1} = \tilde{a}_{n+1}x - \tilde{b_n}p_n - \tilde{c}_n p_{n-1}

    Parameters
    ----------
    recursion_coefs : bkd.ndarray (num_recursion_coeffs,2)
       The two term recursion coefficients
       :math:`a_n,b_n`

    Returns
    -------
    abc : bkd.ndarray (num_recursion_coeffs,3)
       The three term recursion coefficients
       :math:`\tilde{a}_n,\tilde{b}_n,\tilde{c}_n`
    """

    num_terms = recursion_coefs.shape[0]
    abc = bkd.zeros((num_terms, 3))
    abc[:, 0] = 1.0 / recursion_coefs[:, 1]
    abc[1:, 1] = recursion_coefs[:-1, 0] / recursion_coefs[1:, 1]
    abc[1:, 2] = recursion_coefs[:-1, 1] / recursion_coefs[1:, 1]
    return abc


def laguerre_recurrence(
    N: int,
    rho: float,
    probability: bool = True,
    bkd: LinAlgMixin = NumpyLinAlgMixin,
) -> Array:
    r"""
    Compute the recursion coefficients of Laguerre polynomials which are
    orthonormal with respect to the Gamma random variables with PDF
    :math:`x^{\rho}\exp(-x)`

    Parameters
    ----------
    rho : float
        The first parameter of the Laguerre polynomials

    Returns
    -------
    ab : bkd.ndarray (Nterms,2)
        The recursion coefficients of the Nterms orthonormal polynomials
    """
    ab = bkd.zeros((N, 2))

    nu = 1 + rho
    indices = bkd.arange(1, N)
    ab[0, 1] = sp.gamma(nu)
    ab[1:, 1] = (indices + rho) * indices
    ab[:, 1] = bkd.sqrt(ab[:, 1])

    ab[0, 0] = nu
    ab[1:, 0] = 2 * indices + nu

    if probability:
        ab[0, 1] = 1.0

    return ab
