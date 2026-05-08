"""Gaussian variational distribution q(u) = N(m, LL^T) with whitened parameterization.

Follows Hensman et al. (2013): the optimizer sees whitened parameters
m_tilde and L_tilde, where the un-whitened moments are recovered as
m = L_uu @ m_tilde and S = L_uu @ L_tilde @ L_tilde^T @ L_uu^T.
"""

from typing import Generic, Optional

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import (
    CholeskyHyperParameter,
    HyperParameter,
    HyperParameterList,
)


class GaussianVariationalDistribution(Generic[Array]):
    """Explicit q(u) = N(m, LL^T) with whitened parameterization.

    Parameters
    ----------
    num_inducing : int
        Number of inducing points M.
    bkd : Backend[Array]
        Backend for numerical operations.
    mean_init : Optional[Array]
        Initial whitened mean m_tilde, shape (M,). Defaults to zeros.
    chol_init : Optional[Array]
        Initial whitened Cholesky L_tilde, shape (M, M) lower triangular.
        Defaults to identity.
    """

    def __init__(
        self,
        num_inducing: int,
        bkd: Backend[Array],
        mean_init: Optional[Array] = None,
        chol_init: Optional[Array] = None,
    ) -> None:
        self._bkd = bkd
        self._M = num_inducing

        if mean_init is None:
            mean_init = bkd.zeros((num_inducing,))
        if chol_init is None:
            chol_init = bkd.eye(num_inducing)

        self._mean_param = HyperParameter(
            "q_mean",
            num_inducing,
            mean_init,
            bkd.tile(bkd.array([[-1e6, 1e6]]), (num_inducing, 1)),
            bkd=bkd,
        )

        n_tril = num_inducing * (num_inducing + 1) // 2
        self._chol_param = CholeskyHyperParameter(
            "q_chol",
            num_inducing,
            chol_init,
            bkd.tile(bkd.array([[-1e6, 1e6]]), (n_tril, 1)),
            bkd=bkd,
        )

        self._hyp_list = HyperParameterList(
            [self._mean_param, self._chol_param]
        )

    def hyp_list(self) -> HyperParameterList[Array]:
        return self._hyp_list

    def num_inducing(self) -> int:
        return self._M

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def whitened_mean(self) -> Array:
        """Return whitened mean m_tilde, shape (M,)."""
        return self._mean_param.get_values()

    def whitened_cholesky(self) -> Array:
        """Return whitened Cholesky L_tilde, shape (M, M) lower triangular."""
        return self._chol_param.factor()

    def mean(self, L_uu: Array) -> Array:
        """Un-whitened mean: m = L_uu @ m_tilde, shape (M,)."""
        return self._bkd.dot(L_uu, self.whitened_mean())

    def cholesky(self, L_uu: Array) -> Array:
        """Un-whitened Cholesky: L = L_uu @ L_tilde, shape (M, M)."""
        return self._bkd.dot(L_uu, self.whitened_cholesky())

    def covariance(self, L_uu: Array) -> Array:
        """Un-whitened covariance: S = L @ L^T, shape (M, M)."""
        L = self.cholesky(L_uu)
        return self._bkd.dot(L, L.T)

    def kl_divergence_to_prior(self) -> Array:
        """Whitened KL: KL[N(m_tilde, L_tilde L_tilde^T) || N(0, I)].

        = 0.5 * (||m_tilde||^2 + tr(L_tilde L_tilde^T) - M - log|L_tilde L_tilde^T|)
        """
        bkd = self._bkd
        m = self.whitened_mean()
        L = self.whitened_cholesky()
        M = self._M

        m_sq = bkd.sum(m * m)
        trace_S = bkd.sum(L * L)
        log_det_S = 2.0 * bkd.sum(bkd.log(bkd.abs(bkd.get_diagonal(L))))

        return 0.5 * (m_sq + trace_S - M - log_det_S)

    def sample(
        self,
        L_uu: Array,
        n_samples: int,
        eps: Optional[Array] = None,
    ) -> Array:
        """Reparameterized samples from q(u): u = L_uu @ (m_tilde + L_tilde @ eps).

        Parameters
        ----------
        L_uu : Array
            Cholesky of K_uu, shape (M, M).
        n_samples : int
            Number of samples.
        eps : Optional[Array]
            Standard normal noise, shape (n_samples, M). If None, drawn randomly.

        Returns
        -------
        Array
            Samples, shape (n_samples, M).
        """
        bkd = self._bkd
        M = self._M
        if eps is None:
            eps = bkd.array(np.random.randn(n_samples, M))

        m = self.whitened_mean()
        L = self.whitened_cholesky()

        # (n_samples, M) = eps @ L^T + m[None, :]
        whitened = bkd.dot(eps, L.T) + m[None, :]
        # (n_samples, M) = whitened @ L_uu^T
        return bkd.dot(whitened, L_uu.T)

    def __repr__(self) -> str:
        return f"GaussianVariationalDistribution(M={self._M})"
