"""
Conjugate Gaussian OED utilities for prediction.

Provides analytical formulas for expected deviations when computing
prediction OED with linear Gaussian models and conjugate priors.
"""

# TODO: currently analytical only consists of analytical
# expressions for prediction based oed. Inference for parameters
# with KL is in kl_diagnostics elsewhere in the package

import math
from abc import ABC, abstractmethod
from typing import Generic, Optional

from scipy import stats

from pyapprox.inverse.conjugate.gaussian import DenseGaussianConjugatePosterior
from pyapprox.inverse.pushforward.gaussian import GaussianPushforward
from pyapprox.probability.risk import LogNormalAnalyticalRiskMeasures
from pyapprox.util.backends.protocols import Array, Backend


def _compute_expected_kl_divergence_pushforward(
    prior_pushforward_mean: Array,
    prior_pushforward_cov: Array,
    posterior_pushforward_cov: Array,
    nu_vec_pushforward: Array,
    Cmat_pushforward: Array,
    bkd: Backend[Array],
) -> float:
    """
    Compute expected KL divergence between pushforward distributions.

    Parameters
    ----------
    prior_pushforward_mean : Array
        Mean of prior pushforward. Shape: (nqoi, 1)
    prior_pushforward_cov : Array
        Covariance of prior pushforward. Shape: (nqoi, nqoi)
    posterior_pushforward_cov : Array
        Covariance of posterior pushforward. Shape: (nqoi, nqoi)
    nu_vec_pushforward : Array
        Expected posterior pushforward mean. Shape: (nqoi, 1)
    Cmat_pushforward : Array
        Covariance of posterior pushforward mean. Shape: (nqoi, nqoi)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    float
        E_data[KL(posterior_pushforward || prior_pushforward)]
    """
    prior_pushforward_hessian = bkd.inv(prior_pushforward_cov)
    nqoi = posterior_pushforward_cov.shape[0]

    kl_div = bkd.trace(prior_pushforward_hessian @ posterior_pushforward_cov) - nqoi

    _, log_det_prior = bkd.slogdet(prior_pushforward_cov)
    _, log_det_post = bkd.slogdet(posterior_pushforward_cov)
    kl_div = kl_div + float(log_det_prior) - float(log_det_post)

    kl_div = kl_div + bkd.trace(prior_pushforward_hessian @ Cmat_pushforward)
    xi = prior_pushforward_mean - nu_vec_pushforward
    kl_div = kl_div + float((xi.T @ prior_pushforward_hessian @ xi)[0, 0])
    kl_div = 0.5 * kl_div
    return float(kl_div)


class ConjugateGaussianOEDPredictionUtilityBase(ABC, Generic[Array]):
    """
    Base class for conjugate Gaussian OED prediction utilities.

    Computes expected deviation/divergence of the pushforward of the
    posterior through a linear prediction model.

    Parameters
    ----------
    prior_mean : Array
        Prior mean. Shape: (nvars, 1)
    prior_cov : Array
        Prior covariance. Shape: (nvars, nvars)
    qoi_mat : Array
        QoI prediction matrix. Shape: (nqoi, nvars)
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        prior_mean: Array,
        prior_cov: Array,
        qoi_mat: Array,
        bkd: Backend[Array],
    ) -> None:
        self._bkd = bkd
        self._prior_mean = prior_mean
        self._prior_cov = prior_cov
        self._prior_cov_inv = bkd.inv(prior_cov)
        self._qoi_mat = qoi_mat

        # Prior pushforward
        self._prior_pushforward = GaussianPushforward(
            qoi_mat, prior_mean, prior_cov, bkd
        )

        # State
        self._obs_mat: Optional[Array] = None
        self._noise_cov: Optional[Array] = None
        self._posterior: Optional[DenseGaussianConjugatePosterior[Array]] = None
        self._utility: Optional[float] = None

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def set_observation_matrix(self, obs_mat: Array) -> None:
        """
        Set the observation matrix.

        Parameters
        ----------
        obs_mat : Array
            Observation matrix. Shape: (nobs, nvars)
        """
        if obs_mat.shape[1] != self._prior_mean.shape[0]:
            raise ValueError("obs matrix has the wrong number of columns")
        self._obs_mat = obs_mat

    def set_noise_covariance(self, noise_cov: Array) -> None:
        """
        Set the noise covariance and compute utility.

        Parameters
        ----------
        noise_cov : Array
            Noise covariance. Shape: (nobs, nobs)
        """
        self._noise_cov = noise_cov
        self._compute()

    def _compute_expected_posterior_stats(self) -> None:
        """Compute expected posterior statistics."""
        if self._obs_mat is None:
            raise ValueError("must call set_observation_matrix()")
        if self._noise_cov is None:
            raise ValueError("must call set_noise_covariance()")

        self._posterior = DenseGaussianConjugatePosterior(
            self._obs_mat,
            self._prior_mean,
            self._prior_cov,
            self._noise_cov,
            self._bkd,
        )
        # Value of obs does not matter for expected stats
        dummy_obs = self._bkd.ones((self._obs_mat.shape[0], 1))
        self._posterior.compute(dummy_obs)

        # Get expected posterior statistics
        self._nu_vec = self._posterior._nu_vec
        self._Cmat = self._posterior._Cmat

    @abstractmethod
    def _compute_utility(self) -> float:
        """Compute the utility value."""
        raise NotImplementedError

    def _compute(self) -> None:
        """Compute expected posterior stats and utility."""
        self._compute_expected_posterior_stats()

        # Posterior pushforward
        self._post_pushforward = GaussianPushforward(
            self._qoi_mat,
            self._posterior.posterior_mean(),
            self._posterior.posterior_covariance(),
            self._bkd,
        )

        self._utility = self._compute_utility()

    def value(self) -> float:
        """
        Return the computed utility.

        Returns
        -------
        float
            The expected deviation/divergence value.
        """
        if self._utility is None:
            raise ValueError("must call set_noise_covariance()")
        return self._utility


class ConjugateGaussianOEDExpectedStdDev(
    ConjugateGaussianOEDPredictionUtilityBase[Array]
):
    """
    Expected standard deviation of posterior pushforward.

    For a scalar QoI, returns sqrt(Var[QoI | data]).
    """

    def _compute_utility(self) -> float:
        return float(self._bkd.sqrt(self._post_pushforward.covariance()[0, 0]))


class ConjugateGaussianOEDExpectedEntropicDev(
    ConjugateGaussianOEDPredictionUtilityBase[Array]
):
    """
    Expected entropic deviation of posterior pushforward.

    Parameters
    ----------
    prior_mean : Array
        Prior mean. Shape: (nvars, 1)
    prior_cov : Array
        Prior covariance. Shape: (nvars, nvars)
    qoi_mat : Array
        QoI prediction matrix. Shape: (nqoi, nvars)
    lamda : float
        Risk aversion parameter for entropic deviation.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        prior_mean: Array,
        prior_cov: Array,
        qoi_mat: Array,
        lamda: float,
        bkd: Backend[Array],
    ) -> None:
        self._lamda = lamda
        super().__init__(prior_mean, prior_cov, qoi_mat, bkd)

    def _compute_utility(self) -> float:
        # Entropic deviation = lamda * variance / 2
        return self._lamda * float(self._post_pushforward.covariance()[0, 0]) / 2.0


class ConjugateGaussianOEDExpectedAVaRDev(
    ConjugateGaussianOEDPredictionUtilityBase[Array]
):
    """
    Expected AVaR deviation of posterior pushforward.

    Parameters
    ----------
    prior_mean : Array
        Prior mean. Shape: (nvars, 1)
    prior_cov : Array
        Prior covariance. Shape: (nvars, nvars)
    qoi_mat : Array
        QoI prediction matrix. Shape: (nqoi, nvars)
    beta : float
        AVaR quantile level.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        prior_mean: Array,
        prior_cov: Array,
        qoi_mat: Array,
        beta: float,
        bkd: Backend[Array],
    ) -> None:
        self._beta = beta
        super().__init__(prior_mean, prior_cov, qoi_mat, bkd)

    def _compute_utility(self) -> float:
        # AVaR deviation = sigma * phi(Phi^{-1}(beta)) / (1 - beta)
        sigma = float(self._bkd.sqrt(self._post_pushforward.covariance()[0, 0]))
        return float(
            sigma
            * stats.norm.pdf(stats.norm.ppf(self._beta))
            / (1.0 - self._beta)
        )


class ConjugateGaussianOEDExpectedKLDivergence(
    ConjugateGaussianOEDPredictionUtilityBase[Array]
):
    """
    Expected KL divergence between posterior and prior pushforwards.
    """

    def _compute_utility(self) -> float:
        return _compute_expected_kl_divergence_pushforward(
            self._prior_pushforward.mean(),
            self._prior_pushforward.covariance(),
            self._post_pushforward.covariance(),
            self._qoi_mat @ self._nu_vec,
            self._qoi_mat @ self._Cmat @ self._qoi_mat.T,
            self._bkd,
        )


class ConjugateGaussianOEDAVaROfExpectedStdDev(
    ConjugateGaussianOEDPredictionUtilityBase[Array]
):
    """
    AVaR of per-prediction standard deviations.

    For a linear Gaussian model with npred prediction QoIs, computes
    AVaR_{beta}({sigma_1, ..., sigma_npred}) where sigma_j is the
    posterior standard deviation of QoI j.

    Parameters
    ----------
    prior_mean : Array
        Prior mean. Shape: (nvars, 1)
    prior_cov : Array
        Prior covariance. Shape: (nvars, nvars)
    qoi_mat : Array
        QoI prediction matrix. Shape: (npred, nvars)
    beta : float
        AVaR quantile level.
    bkd : Backend[Array]
        Computational backend.
    delta : float, optional
        Smoothing parameter for AVaR. Default: 100000
    """

    def __init__(
        self,
        prior_mean: Array,
        prior_cov: Array,
        qoi_mat: Array,
        beta: float,
        bkd: Backend[Array],
        delta: float = 100000,
    ) -> None:
        self._beta = beta
        self._delta = delta
        super().__init__(prior_mean, prior_cov, qoi_mat, bkd)

    def _compute_utility(self) -> float:
        from pyapprox.risk.avar import SampleAverageSmoothedAVaR

        npred = self._qoi_mat.shape[0]
        post_cov = self._post_pushforward.covariance()

        # Compute per-prediction standard deviations
        sigmas = []
        for j in range(npred):
            sigma_j = float(self._bkd.sqrt(post_cov[j, j]))
            sigmas.append(sigma_j)

        # Apply AVaR to the vector of sigmas with uniform weights
        sigma_arr = self._bkd.reshape(
            self._bkd.asarray(sigmas), (1, npred)
        )
        weights = self._bkd.full((1, npred), 1.0 / npred)

        avar = SampleAverageSmoothedAVaR(self._beta, self._bkd, delta=self._delta)
        result = avar(sigma_arr, weights)
        return float(self._bkd.to_numpy(result)[0, 0])


class ConjugateGaussianOEDForLogNormalExpectedStdDev(
    ConjugateGaussianOEDPredictionUtilityBase[Array]
):
    """
    Expected standard deviation when QoI is lognormal.

    For a scalar QoI that is lognormal (QoI = exp(linear_QoI)),
    computes the expected standard deviation.
    """

    def _lognormal_mean(self, mu: float, sigma: float) -> float:
        """Compute mean of lognormal with underlying normal N(mu, sigma^2)."""
        return math.exp(mu + sigma**2 / 2.0)

    def _compute_utility(self) -> float:
        tau_hat = self._qoi_mat @ self._nu_vec
        sigma_hat_sq = self._qoi_mat @ self._Cmat @ self._qoi_mat.T

        tmp = float(self._bkd.exp(self._post_pushforward.covariance()[0, 0]))
        factor = (tmp - 1.0) * tmp

        return math.sqrt(factor) * self._lognormal_mean(
            float(tau_hat[0, 0]),
            math.sqrt(float(sigma_hat_sq[0, 0])),
        )


class ConjugateGaussianOEDForLogNormalAVaRStdDev(
    ConjugateGaussianOEDPredictionUtilityBase[Array]
):
    """
    AVaR of standard deviation when QoI is lognormal.

    Parameters
    ----------
    prior_mean : Array
        Prior mean. Shape: (nvars, 1)
    prior_cov : Array
        Prior covariance. Shape: (nvars, nvars)
    qoi_mat : Array
        QoI prediction matrix. Shape: (nqoi, nvars)
    beta : float
        AVaR quantile level.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        prior_mean: Array,
        prior_cov: Array,
        qoi_mat: Array,
        beta: float,
        bkd: Backend[Array],
    ) -> None:
        self._beta = beta
        super().__init__(prior_mean, prior_cov, qoi_mat, bkd)

    def _compute_utility(self) -> float:
        tau_hat = self._qoi_mat @ self._nu_vec
        sigma_hat_sq = self._qoi_mat @ self._Cmat @ self._qoi_mat.T

        tmp = float(self._bkd.exp(self._post_pushforward.covariance()[0, 0]))
        factor = (tmp - 1.0) * tmp

        # Use LogNormalAnalyticalRiskMeasures for AVaR
        risk_measures = LogNormalAnalyticalRiskMeasures(
            float(self._bkd.to_numpy(tau_hat)[0, 0]),
            math.sqrt(float(self._bkd.to_numpy(sigma_hat_sq)[0, 0])),
        )
        return math.sqrt(factor) * risk_measures.AVaR(self._beta)
