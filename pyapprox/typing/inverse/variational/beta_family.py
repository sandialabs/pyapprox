"""
Beta variational family with independent marginals.

Provides a variational family parameterized by per-dimension alpha and beta
shape parameters, suitable for variational inference on [0, 1]^d.
"""

from typing import Any, Generic, List, Optional, Union

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter import HyperParameterList
from pyapprox.typing.util.hyperparameter.log_hyperparameter import (
    LogHyperParameter,
)
from pyapprox.typing.probability.univariate.beta import BetaMarginal
from pyapprox.typing.probability.univariate.uniform import UniformMarginal
from pyapprox.typing.probability.joint.independent import IndependentJoint


class BetaVariationalFamily(Generic[Array]):
    """Independent Beta variational family.

    Parameterized by per-dimension alpha and beta shape parameters.
    Internally stores log(alpha) and log(beta) as ``LogHyperParameter``
    (ensuring positivity).

    Parameters
    ----------
    nvars : int
        Number of latent variables.
    bkd : Backend[Array]
        Computational backend.
    alpha_init : list or None
        Initial alpha values per dimension. Defaults to 2.0 for all.
    beta_init : list or None
        Initial beta values per dimension. Defaults to 2.0 for all.
    """

    def __init__(
        self,
        nvars: int,
        bkd: Backend[Array],
        alpha_init: Optional[Union[List[float], Array]] = None,
        beta_init: Optional[Union[List[float], Array]] = None,
    ) -> None:
        self._nvars = nvars
        self._bkd = bkd

        if alpha_init is None:
            alpha_init = [2.0] * nvars
        if beta_init is None:
            beta_init = [2.0] * nvars

        alpha_hyp = LogHyperParameter(
            "alpha", nvars, bkd.asarray(alpha_init),
            (1e-10, 1e10), bkd,
        )
        beta_hyp = LogHyperParameter(
            "beta", nvars, bkd.asarray(beta_init),
            (1e-10, 1e10), bkd,
        )
        self._hyp_list = HyperParameterList([alpha_hyp, beta_hyp])

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def hyp_list(self) -> HyperParameterList:
        return self._hyp_list

    def _get_alpha_beta(self) -> tuple:
        """Extract alpha and beta from hyp_list values.

        LogHyperParameter stores log-transformed values internally,
        so get_values() returns log(alpha), log(beta). We exponentiate.
        """
        values = self._hyp_list.get_values()
        log_alpha = values[:self._nvars]
        log_beta = values[self._nvars:]
        alpha = self._bkd.exp(log_alpha)
        beta = self._bkd.exp(log_beta)
        return alpha, beta

    def _build_marginals(self) -> List[BetaMarginal]:
        """Build BetaMarginal instances from current hyp_list values."""
        alpha, beta = self._get_alpha_beta()
        alpha_np = self._bkd.to_numpy(alpha)
        beta_np = self._bkd.to_numpy(beta)
        return [
            BetaMarginal(float(alpha_np[i]), float(beta_np[i]), self._bkd)
            for i in range(self._nvars)
        ]

    def reparameterize(
        self, base_samples: Array, params: Optional[Array] = None
    ) -> Array:
        """Reparameterize uniform base samples to Beta samples.

        Parameters
        ----------
        base_samples : Array
            Uniform(0,1) samples, shape ``(nvars, nsamples)``.
        params : Array or None
            Per-sample unconstrained active params, shape
            ``(nactive_params, nsamples)``. If None, use hyp_list values.

        Returns
        -------
        Array
            Variational samples in [0, 1], shape ``(nvars, nsamples)``.
        """
        if params is not None:
            raise NotImplementedError(
                "Per-sample params for BetaVariationalFamily not yet supported"
            )
        marginals = self._build_marginals()
        parts = []
        for i in range(self._nvars):
            row = self._bkd.reshape(base_samples[i], (1, -1))
            parts.append(marginals[i].invcdf(row)[0])
        return self._bkd.stack(parts, axis=0)

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
        if params is not None:
            # Vectorized Beta log-pdf with per-sample params
            alpha = self._bkd.exp(params[:self._nvars, :])  # (nvars, N)
            beta = self._bkd.exp(params[self._nvars:, :])  # (nvars, N)
            log_beta_func = (
                self._bkd.gammaln(alpha)
                + self._bkd.gammaln(beta)
                - self._bkd.gammaln(alpha + beta)
            )
            logp_per_dim = (
                (alpha - 1.0) * self._bkd.log(samples)
                + (beta - 1.0) * self._bkd.log(1.0 - samples)
                - log_beta_func
            )
            return self._bkd.reshape(
                self._bkd.sum(logp_per_dim, axis=0),
                (1, samples.shape[1]),
            )
        marginals = self._build_marginals()
        joint = IndependentJoint(marginals, self._bkd)
        return joint.logpdf(samples)

    def kl_divergence(
        self, prior: Any, params: Optional[Array] = None
    ) -> Any:
        """Compute KL(q || prior) for independent Beta marginals.

        Uses the analytical Beta-Beta KL divergence formula:
            KL(Beta(a1,b1) || Beta(a2,b2)) =
                log B(a2,b2) - log B(a1,b1)
                + (a1-a2)*psi(a1) + (b1-b2)*psi(b1)
                + (a2-a1+b2-b1)*psi(a1+b1)

        For independent marginals, total KL = sum of marginal KLs.

        Parameters
        ----------
        prior : BetaVariationalFamily
            The prior distribution. Must be a BetaVariationalFamily.
        params : Array or None
            Per-sample unconstrained active params, shape
            ``(nactive_params, nsamples)``.

        Returns
        -------
        float or Array
            Scalar KL when params=None. Shape ``(1, nsamples)`` when
            params provided.

        Raises
        ------
        NotImplementedError
            If prior is not a BetaVariationalFamily.
        """
        if not isinstance(prior, BetaVariationalFamily):
            raise NotImplementedError(
                "Analytical KL only supported for BetaVariationalFamily prior, "
                f"got {type(prior).__name__}"
            )

        if params is not None:
            # Per-sample KL
            alpha1 = self._bkd.exp(params[:self._nvars, :])  # (nvars, N)
            beta1 = self._bkd.exp(params[self._nvars:, :])  # (nvars, N)

            alpha2, beta2 = prior._get_alpha_beta()
            alpha2 = self._bkd.reshape(alpha2, (self._nvars, 1))
            beta2 = self._bkd.reshape(beta2, (self._nvars, 1))

            kl_per_dim = (
                self._bkd.gammaln(alpha2) + self._bkd.gammaln(beta2)
                - self._bkd.gammaln(alpha2 + beta2)
                - self._bkd.gammaln(alpha1) - self._bkd.gammaln(beta1)
                + self._bkd.gammaln(alpha1 + beta1)
                + (alpha1 - alpha2) * self._bkd.digamma(alpha1)
                + (beta1 - beta2) * self._bkd.digamma(beta1)
                + (alpha2 - alpha1 + beta2 - beta1)
                * self._bkd.digamma(alpha1 + beta1)
            )
            kl_sum = self._bkd.sum(kl_per_dim, axis=0)
            return self._bkd.reshape(kl_sum, (1, params.shape[1]))

        # Scalar KL using hyp_list values
        alpha1, beta1 = self._get_alpha_beta()
        alpha2, beta2 = prior._get_alpha_beta()

        kl_per_dim = (
            self._bkd.gammaln(alpha2) + self._bkd.gammaln(beta2)
            - self._bkd.gammaln(alpha2 + beta2)
            - self._bkd.gammaln(alpha1) - self._bkd.gammaln(beta1)
            + self._bkd.gammaln(alpha1 + beta1)
            + (alpha1 - alpha2) * self._bkd.digamma(alpha1)
            + (beta1 - beta2) * self._bkd.digamma(beta1)
            + (alpha2 - alpha1 + beta2 - beta1)
            * self._bkd.digamma(alpha1 + beta1)
        )
        return self._bkd.sum(kl_per_dim)

    def base_distribution(self) -> IndependentJoint:
        """Return the base distribution (independent Uniform(0,1)).

        This is informational -- the API user draws samples from this
        distribution and passes them to ``ELBOObjective``.

        Returns
        -------
        IndependentJoint
            Independent Uniform(0,1) marginals with ``nvars`` dimensions.
        """
        marginals = [
            UniformMarginal(0.0, 1.0, self._bkd)
            for _ in range(self._nvars)
        ]
        return IndependentJoint(marginals, self._bkd)

    def __repr__(self) -> str:
        return (
            f"BetaVariationalFamily(nvars={self._nvars}, "
            f"nparams={self._hyp_list.nparams()})"
        )
