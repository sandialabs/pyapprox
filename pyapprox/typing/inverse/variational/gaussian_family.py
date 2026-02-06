"""
Gaussian variational family with diagonal covariance.

Provides a variational family parameterized by mean and standard deviation
vectors, suitable for variational inference with Gaussian posteriors.
"""

from typing import Any, Generic, List, Optional, Union

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter import (
    HyperParameter,
    HyperParameterList,
)
from pyapprox.typing.util.hyperparameter.log_hyperparameter import (
    LogHyperParameter,
)
from pyapprox.typing.probability.gaussian.diagonal import (
    DiagonalMultivariateGaussian,
)


class GaussianVariationalFamily(Generic[Array]):
    """Diagonal Gaussian variational family.

    Parameterized by mean vector and standard deviation vector.
    Internally stores mean as ``HyperParameter`` and stdev as
    ``LogHyperParameter`` (ensuring positivity).

    Parameters
    ----------
    nvars : int
        Number of latent variables.
    bkd : Backend[Array]
        Computational backend.
    mean_init : list or None
        Initial mean values. Defaults to zeros.
    stdev_init : list or None
        Initial standard deviation values. Defaults to ones.
    """

    def __init__(
        self,
        nvars: int,
        bkd: Backend[Array],
        mean_init: Optional[Union[List[float], Array]] = None,
        stdev_init: Optional[Union[List[float], Array]] = None,
    ) -> None:
        self._nvars = nvars
        self._bkd = bkd

        if mean_init is None:
            mean_init = [0.0] * nvars
        if stdev_init is None:
            stdev_init = [1.0] * nvars

        mean_hyp = HyperParameter(
            "mean", nvars, bkd.asarray(mean_init),
            (-1e6, 1e6), bkd,
        )
        stdev_hyp = LogHyperParameter(
            "stdev", nvars, bkd.asarray(stdev_init),
            (1e-8, 1e6), bkd,
        )
        self._hyp_list = HyperParameterList([mean_hyp, stdev_hyp])

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def hyp_list(self) -> HyperParameterList:
        return self._hyp_list

    def _get_mean_stdev(self) -> tuple:
        """Extract mean and stdev from hyp_list values.

        LogHyperParameter stores log-transformed values internally,
        so get_values() returns log(stdev). We exponentiate to recover stdev.
        """
        values = self._hyp_list.get_values()
        mean = values[:self._nvars]
        log_stdev = values[self._nvars:]
        stdev = self._bkd.exp(log_stdev)
        return mean, stdev

    def reparameterize(
        self, base_samples: Array, params: Optional[Array] = None
    ) -> Array:
        """Reparameterize base samples to variational samples.

        Parameters
        ----------
        base_samples : Array
            Standard normal samples, shape ``(nvars, nsamples)``.
        params : Array or None
            Per-sample unconstrained active params, shape
            ``(nactive_params, nsamples)``.  If None, use hyp_list values.

        Returns
        -------
        Array
            Variational samples, shape ``(nvars, nsamples)``.
        """
        if params is None:
            mean, stdev = self._get_mean_stdev()
            return mean[:, None] + stdev[:, None] * base_samples
        else:
            # Per-sample params: first nvars are means, rest are log-stdevs
            mean = params[:self._nvars, :]
            log_stdev = params[self._nvars:, :]
            stdev = self._bkd.exp(log_stdev)
            return mean + stdev * base_samples

    def logpdf(
        self, samples: Array, params: Optional[Array] = None
    ) -> Array:
        """Evaluate log-pdf of the variational distribution.

        Parameters
        ----------
        samples : Array
            Points at which to evaluate, shape ``(nvars, nsamples)``.
        params : Array or None
            Per-sample unconstrained active params, shape
            ``(nactive_params, nsamples)``.

        Returns
        -------
        Array
            Log-pdf values, shape ``(1, nsamples)``.
        """
        if params is None:
            mean, stdev = self._get_mean_stdev()
            mean_col = self._bkd.reshape(mean, (self._nvars, 1))
            var = stdev ** 2
            dist = DiagonalMultivariateGaussian(mean_col, var, self._bkd)
            return dist.logpdf(samples)
        else:
            mean = params[:self._nvars, :]  # (nvars, N)
            log_stdev = params[self._nvars:, :]  # (nvars, N)
            stdev = self._bkd.exp(log_stdev)
            var = stdev ** 2
            # Vectorized log-pdf: sum over nvars of
            #   -0.5*log(2*pi) - log(sigma) - 0.5*((x - mu)/sigma)^2
            log_2pi = float(np.log(2.0 * np.pi))
            z = (samples - mean) / stdev
            logp_per_dim = -0.5 * log_2pi - log_stdev - 0.5 * z ** 2
            # Sum over nvars dimension (axis=0), keep as (1, N)
            return self._bkd.reshape(
                self._bkd.sum(logp_per_dim, axis=0), (1, samples.shape[1])
            )

    def kl_divergence(
        self, prior: Any, params: Optional[Array] = None
    ) -> Any:
        """Compute KL(q || prior) for diagonal Gaussian.

        Parameters
        ----------
        prior : DiagonalMultivariateGaussian
            The prior distribution.
        params : Array or None
            Per-sample unconstrained active params, shape
            ``(nactive_params, nsamples)``.

        Returns
        -------
        float or Array
            Scalar KL when params=None. Shape ``(1, nsamples)`` when
            params provided.
        """
        if params is None:
            mean, stdev = self._get_mean_stdev()
            mean_col = self._bkd.reshape(mean, (self._nvars, 1))
            var = stdev ** 2
            q = DiagonalMultivariateGaussian(mean_col, var, self._bkd)
            return q.kl_divergence(prior)
        else:
            # Per-sample KL: (1, N)
            # prior params as (nvars, 1) for broadcasting
            prior_mean = prior.mean()  # (nvars, 1)
            prior_var = prior.variances()  # (nvars,)
            prior_var_col = self._bkd.reshape(prior_var, (self._nvars, 1))

            mu1 = params[:self._nvars, :]  # (nvars, N)
            log_sigma1 = params[self._nvars:, :]  # (nvars, N)
            sigma1 = self._bkd.exp(log_sigma1)
            var1 = sigma1 ** 2  # (nvars, N)

            # KL = 0.5 * sum_d(var1/var0 + (mu0-mu1)^2/var0 - 1
            #                   + log(var0) - log(var1))
            kl_per_dim = (
                var1 / prior_var_col
                + (prior_mean - mu1) ** 2 / prior_var_col
                - 1.0
                + self._bkd.log(prior_var_col)
                - self._bkd.log(var1)
            )
            # Sum over nvars (axis=0) → (N,), reshape to (1, N)
            kl_sum = self._bkd.sum(kl_per_dim, axis=0)
            return self._bkd.reshape(0.5 * kl_sum, (1, params.shape[1]))

    def base_distribution(self) -> DiagonalMultivariateGaussian:
        """Return the base distribution (standard normal).

        This is informational — the API user draws samples from this
        distribution and passes them to ``ELBOObjective``.

        Returns
        -------
        DiagonalMultivariateGaussian
            Standard normal with ``nvars`` dimensions.
        """
        return DiagonalMultivariateGaussian(
            self._bkd.zeros((self._nvars, 1)),
            self._bkd.ones((self._nvars,)),
            self._bkd,
        )

    def __repr__(self) -> str:
        return (
            f"GaussianVariationalFamily(nvars={self._nvars}, "
            f"nparams={self._hyp_list.nparams()})"
        )
