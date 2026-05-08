"""DGP layer: sparse variational GP with explicit q(u) (Hensman 2013).

Each DGPLayer is a single-output SVGP layer with whitened variational
distribution. Unlike VariationalGaussianProcess (collapsed Titsias ELBO),
this layer maintains an explicit variational distribution q(u) that the
optimizer updates directly — required for doubly-stochastic VI in Deep GPs.
"""

import copy
from typing import Generic, List, Optional, Tuple

from pyapprox.surrogates.gaussianprocess.inducing.inducing_points import (
    InducingPoints,
)
from pyapprox.surrogates.gaussianprocess.inducing.variational_distribution import (
    GaussianVariationalDistribution,
)
from pyapprox.surrogates.gaussianprocess.likelihoods.gaussian import (
    GaussianLikelihood,
)
from pyapprox.surrogates.gaussianprocess.mean_functions import (
    MeanFunction,
    ZeroMean,
)
from pyapprox.surrogates.kernels.protocols import Kernel
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


class DGPLayer(Generic[Array]):
    """Single-output SVGP layer with explicit q(u) for Deep GP composition.

    Implements the Hensman (2013) whitened parameterization:
        q(u) = N(L_uu m_tilde, L_uu L_tilde L_tilde^T L_uu^T)

    The posterior predictive at input h is:
        mu(h)  = alpha^T m_tilde + mean(h)
        var(h) = diag(K_ff) - colsum(alpha^2) + colsum((L_tilde^T alpha)^2)

    where alpha = L_uu^{-1} K_uf and K_uf = kernel(Z, h).

    Parameters
    ----------
    kernel : Kernel[Array]
        Covariance kernel.
    mean_function : MeanFunction[Array]
        Prior mean function.
    inducing_points : InducingPoints[Array]
        Inducing point locations Z, shape (nvars, M).
    variational_dist : GaussianVariationalDistribution[Array]
        Whitened variational distribution q(u).
    bkd : Backend[Array]
        Backend for numerical operations.
    likelihood : Optional[GaussianLikelihood[Array]]
        Observation likelihood (only for observed/leaf layers).
    nugget : float
        Jitter for K_uu Cholesky stability.
    """

    def __init__(
        self,
        kernel: Kernel[Array],
        mean_function: MeanFunction[Array],
        inducing_points: InducingPoints[Array],
        variational_dist: GaussianVariationalDistribution[Array],
        bkd: Backend[Array],
        likelihood: Optional[GaussianLikelihood[Array]] = None,
        nugget: float = 1e-6,
    ) -> None:
        if not isinstance(kernel, Kernel):
            raise TypeError(
                f"kernel must be a Kernel instance, got {type(kernel).__name__}"
            )
        if not isinstance(mean_function, MeanFunction):
            raise TypeError(
                f"mean_function must satisfy MeanFunction protocol, "
                f"got {type(mean_function).__name__}"
            )
        if inducing_points.num_inducing() != variational_dist.num_inducing():
            raise ValueError(
                f"inducing_points has M={inducing_points.num_inducing()} but "
                f"variational_dist has M={variational_dist.num_inducing()}"
            )
        self._kernel = kernel
        self._mean = mean_function
        self._inducing_points = inducing_points
        self._variational_dist = variational_dist
        self._bkd = bkd
        self._likelihood = likelihood
        self._nugget = nugget

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def kernel(self) -> Kernel[Array]:
        return self._kernel

    def mean_function(self) -> MeanFunction[Array]:
        return self._mean

    def inducing_points(self) -> InducingPoints[Array]:
        return self._inducing_points

    def variational_dist(self) -> GaussianVariationalDistribution[Array]:
        return self._variational_dist

    def likelihood(self) -> Optional[GaussianLikelihood[Array]]:
        return self._likelihood

    def nvars(self) -> int:
        return self._inducing_points.nvars()

    def hyp_list(self) -> HyperParameterList[Array]:
        hyps: List = (
            self._kernel.hyp_list().hyperparameters()
            + self._mean.hyp_list().hyperparameters()
            + self._inducing_points.hyp_list().hyperparameters()
            + self._variational_dist.hyp_list().hyperparameters()
        )
        if self._likelihood is not None:
            hyps += self._likelihood.hyp_list().hyperparameters()
        return HyperParameterList(hyps)

    def _compute_L_uu(self) -> Array:
        """Cholesky of K_uu + nugget*I."""
        bkd = self._bkd
        Z = self._inducing_points.get_samples()
        K_uu = self._kernel(Z, Z)
        K_uu = K_uu + bkd.eye(K_uu.shape[0]) * self._nugget
        return bkd.cholesky(K_uu)

    def predict_marginal(self, h: Array) -> Tuple[Array, Array]:
        """SVGP posterior mean and variance at layer inputs h.

        Parameters
        ----------
        h : Array
            Layer inputs, shape (nvars, N).

        Returns
        -------
        mean : Array
            Posterior mean, shape (1, N).
        var : Array
            Posterior marginal variance, shape (1, N).
        """
        bkd = self._bkd
        Z = self._inducing_points.get_samples()
        L_uu = self._compute_L_uu()

        K_uf = self._kernel(Z, h)

        # alpha = L_uu^{-1} K_uf, shape (M, N)
        alpha = bkd.solve_triangular(L_uu, K_uf, lower=True)

        m_tilde = self._variational_dist.whitened_mean()
        L_tilde = self._variational_dist.whitened_cholesky()

        # mean = alpha^T m_tilde + mean_function(h)
        mean = bkd.dot(alpha.T, m_tilde) + self._mean(h)[0, :]
        mean = bkd.reshape(mean, (1, h.shape[1]))

        # var = diag(K_ff) - colsum(alpha^2) + colsum((L_tilde^T alpha)^2)
        K_ff_diag = self._kernel.diag(h)
        # colsum(alpha^2) via einsum
        q_diag = bkd.einsum("ij,ij->j", alpha, alpha)
        # L_tilde^T @ alpha, shape (M, N)
        Lt_alpha = bkd.dot(L_tilde.T, alpha)
        s_diag = bkd.einsum("ij,ij->j", Lt_alpha, Lt_alpha)

        var = K_ff_diag - q_diag + s_diag
        var = var * (var >= 0.0)
        var = bkd.reshape(var, (1, h.shape[1]))

        return mean, var

    def sample(self, h: Array, n_samples: int, eps: Optional[Array] = None) -> Array:
        """Reparameterized samples from the layer posterior at inputs h.

        Parameters
        ----------
        h : Array
            Layer inputs, shape (nvars, N).
        n_samples : int
            Number of samples S.
        eps : Optional[Array]
            Standard normal noise for inducing values, shape (n_samples, M).

        Returns
        -------
        Array
            Samples, shape (n_samples, 1, N).
        """
        bkd = self._bkd
        Z = self._inducing_points.get_samples()
        L_uu = self._compute_L_uu()
        K_uf = self._kernel(Z, h)

        # alpha = L_uu^{-1} K_uf, shape (M, N)
        alpha = bkd.solve_triangular(L_uu, K_uf, lower=True)

        # u_samples shape (n_samples, M) — un-whitened inducing samples
        u_samples = self._variational_dist.sample(L_uu, n_samples, eps=eps)

        # L_uu^{-1} u_samples^T, shape (M, n_samples)
        v = bkd.solve_triangular(L_uu, u_samples.T, lower=True)

        # f_mean_offset = alpha^T @ v, shape (N, n_samples)
        f_mean_offset = bkd.dot(alpha.T, v)

        # Add mean function contribution
        mean_val = self._mean(h)  # (1, N)
        # f = f_mean_offset + mean(h), shape (N, n_samples)
        f = f_mean_offset + mean_val.T

        # Transpose to (n_samples, 1, N)
        return bkd.reshape(f.T, (n_samples, 1, h.shape[1]))

    def kl_to_prior(self) -> Array:
        """KL divergence KL[q(u) || p(u)] (scalar)."""
        return self._variational_dist.kl_divergence_to_prior()

    def _clone_unfitted(self) -> "DGPLayer[Array]":
        return copy.deepcopy(self)

    def _copy_fitted_state_from(self, other: "DGPLayer[Array]") -> None:
        self._kernel.hyp_list().set_values(
            other._kernel.hyp_list().get_values()
        )
        self._mean.hyp_list().set_values(
            other._mean.hyp_list().get_values()
        )
        self._inducing_points.hyp_list().set_values(
            other._inducing_points.hyp_list().get_values()
        )
        self._variational_dist.hyp_list().set_values(
            other._variational_dist.hyp_list().get_values()
        )
        if self._likelihood is not None and other._likelihood is not None:
            self._likelihood.hyp_list().set_values(
                other._likelihood.hyp_list().get_values()
            )

    def __repr__(self) -> str:
        M = self._inducing_points.num_inducing()
        nvars = self.nvars()
        has_lik = self._likelihood is not None
        return (
            f"DGPLayer(nvars={nvars}, M={M}, "
            f"kernel={self._kernel.__class__.__name__}, "
            f"has_likelihood={has_lik})"
        )
