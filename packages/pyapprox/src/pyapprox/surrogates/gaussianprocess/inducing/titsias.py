"""Titsias-optimal variational parameters for sparse GP.

Computes the closed-form optimal q(u) from Titsias (2009) in whitened
form, for use as initialization or as a test reference.
"""

from typing import Tuple

from pyapprox.util.backends.protocols import Array, Backend


def titsias_optimal_whitened_q_u(
    K_uu: Array,
    K_uf: Array,
    y: Array,
    noise_var: Array,
    L_uu: Array,
    bkd: Backend[Array],
) -> Tuple[Array, Array]:
    """Compute Titsias-optimal q(u) in whitened form.

    Given inducing points Z with kernel matrices K_uu, K_uf, and
    observations y with noise variance sigma^2, the Titsias (2009)
    optimal variational distribution is q(u) = N(m*, S*) where:

        Sigma = (K_uu + sigma^{-2} K_uf K_fu)^{-1}
        m*    = sigma^{-2} K_uu Sigma K_uf y
        S*    = K_uu Sigma K_uu

    The whitened parameters satisfy m* = L_uu m_tilde*, S* = L_uu S_tilde* L_uu^T,
    and L_tilde* = chol(S_tilde*).

    The caller is responsible for adding any prior-regularization
    nugget to K_uu BEFORE calling this function. K_uu must equal
    L_uu @ L_uu.T to floating-point precision; this is the prior
    covariance of u under whatever model the caller has chosen.

    K_uf is the cross-covariance between inducing and training
    inputs and is NEVER regularized, even when Z = X.

    A small fixed jitter is applied internally to intermediate
    matrices before Cholesky for numerical stability. This jitter
    is not a modeling choice and should be much smaller than any
    reasonable prior regularization.

    Parameters
    ----------
    K_uu : Array, shape (M, M)
        Prior covariance at inducing points (including any nugget that
        the model treats as part of its prior).
    K_uf : Array, shape (M, N)
        Cross-covariance between inducing and training points (no nugget).
    y : Array, shape (N,)
        Training targets (single output, 1D).
    noise_var : Array, shape (1,) or scalar
        Observation noise variance sigma^2.
    L_uu : Array, shape (M, M)
        Lower Cholesky of K_uu.
    bkd : Backend[Array]
        Backend for numerical operations.

    Returns
    -------
    m_tilde_star : Array, shape (M,)
        Whitened optimal mean, ready to set on a
        GaussianVariationalDistribution.
    L_tilde_star : Array, shape (M, M)
        Whitened optimal Cholesky factor (lower triangular), ready to
        set on a GaussianVariationalDistribution.
    """
    _jitter = 1e-14
    M = K_uu.shape[0]
    K_fu = K_uf.T

    # A = K_uu + sigma^{-2} K_uf K_fu
    A = K_uu + (1.0 / noise_var) * bkd.dot(K_uf, K_fu)
    L_A = bkd.cholesky(A + bkd.eye(M) * _jitter)

    # Sigma = A^{-1}, applied via Cholesky solves
    # m* = sigma^{-2} K_uu A^{-1} K_uf y
    K_uf_y = bkd.dot(K_uf, bkd.reshape(y, (y.shape[0], 1)))  # (M, 1)
    A_inv_Kuf_y = bkd.solve_triangular(
        L_A.T,
        bkd.solve_triangular(L_A, K_uf_y, lower=True),
        lower=False,
    )
    m_star = (1.0 / noise_var) * bkd.dot(K_uu, A_inv_Kuf_y)[:, 0]

    # S* = K_uu A^{-1} K_uu
    A_inv_Kuu = bkd.solve_triangular(
        L_A.T,
        bkd.solve_triangular(L_A, K_uu, lower=True),
        lower=False,
    )
    S_star = bkd.dot(K_uu, A_inv_Kuu)

    # Whiten: m_tilde* = L_uu^{-1} m*
    m_tilde_star = bkd.solve_triangular(
        L_uu, bkd.reshape(m_star, (M, 1)), lower=True,
    )[:, 0]

    # S_tilde* = L_uu^{-1} S* L_uu^{-T}
    S_tilde = bkd.solve_triangular(L_uu, S_star, lower=True)
    S_tilde = bkd.solve_triangular(L_uu, S_tilde.T, lower=True).T
    # Symmetrize for numerical stability
    S_tilde = 0.5 * (S_tilde + S_tilde.T)

    L_tilde_star = bkd.cholesky(S_tilde + bkd.eye(M) * _jitter)

    return m_tilde_star, L_tilde_star
