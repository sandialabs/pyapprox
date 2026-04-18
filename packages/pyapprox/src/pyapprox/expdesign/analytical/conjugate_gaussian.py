"""
Conjugate Gaussian OED utilities for prediction.

Provides analytical formulas for expected deviations when computing
prediction OED with linear Gaussian models and conjugate priors.
"""

# TODO: currently analytical only consists of analytical
# expressions for prediction based oed. Inference for parameters
# with KL is in kl_diagnostics elsewhere in the package

import itertools
import math
from abc import ABC, abstractmethod
from typing import Generic, List, Optional

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


class ConjugateGaussianOEDDataMeanQoIMeanStdDev(
    ConjugateGaussianOEDPredictionUtilityBase[Array]
):
    """
    Expected standard deviation of posterior pushforward.

    For a scalar QoI, returns sqrt(Var[QoI | data]).
    """

    def _compute_utility(self) -> float:
        return float(self._bkd.sqrt(self._post_pushforward.covariance()[0, 0]))


class ConjugateGaussianOEDDataMeanQoIMeanEntropicDev(
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


class ConjugateGaussianOEDDataAVaRQoIMeanAVaRDev(
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


class ConjugateGaussianOEDExpectedPushforwardKLDivergence(
    ConjugateGaussianOEDPredictionUtilityBase[Array]
):
    """
    E_y[KL(posterior_pushforward || prior_pushforward)].

    Computes expected KL divergence between the pushforward of the
    posterior and prior through the QoI prediction matrix Q.  This
    differs from the parameter-space expected information gain (EIG)
    whenever Q is rank-deficient (npred < nparams).
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


class ConjugateGaussianOEDExpectedInformationGain(Generic[Array]):
    """
    E_y[KL(posterior || prior)] in parameter space.

    For conjugate Gaussian the posterior covariance is data-independent:

        Σ_post = (Σ_prior⁻¹ + Hᵀ Σ_noise⁻¹ H)⁻¹

    so the expected information gain simplifies to

        EIG = ½ (log|Σ_prior| − log|Σ_post|)

    This matches what the NMC KL-OED objective estimates.

    Parameters
    ----------
    prior_cov : Array
        Prior covariance. Shape: (nvars, nvars)
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        prior_cov: Array,
        bkd: Backend[Array],
    ) -> None:
        self._bkd = bkd
        self._prior_cov = prior_cov
        self._obs_mat: Optional[Array] = None
        self._noise_cov: Optional[Array] = None
        self._value: Optional[float] = None

    def set_observation_matrix(self, obs_mat: Array) -> None:
        self._obs_mat = obs_mat

    def set_noise_covariance(self, noise_cov: Array) -> None:
        self._noise_cov = noise_cov
        self._compute()

    def _compute(self) -> None:
        if self._obs_mat is None:
            raise ValueError("must call set_observation_matrix()")
        if self._noise_cov is None:
            raise ValueError("must call set_noise_covariance()")
        bkd = self._bkd
        noise_cov_inv = bkd.inv(self._noise_cov)
        prior_cov_inv = bkd.inv(self._prior_cov)
        post_cov_inv = prior_cov_inv + bkd.dot(
            self._obs_mat.T, bkd.dot(noise_cov_inv, self._obs_mat)
        )
        post_cov = bkd.inv(post_cov_inv)
        _, log_det_prior = bkd.slogdet(self._prior_cov)
        _, log_det_post = bkd.slogdet(post_cov)
        self._value = 0.5 * float(log_det_prior - log_det_post)

    def value(self) -> float:
        if self._value is None:
            raise ValueError("must call set_noise_covariance()")
        return self._value


class ConjugateGaussianOEDDataMeanQoIAVaRStdDev(
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


class ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev(
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


class ConjugateGaussianOEDForLogNormalDataAVaRQoIMeanStdDev(
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


class ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev(
    ConjugateGaussianOEDPredictionUtilityBase[Array]
):
    """
    E_y[AVaR_alpha over vector lognormal Std(W_j|y)] for diagnostics.

    Non-differentiable version (returns float via value()) for use in
    the diagnostics registry. For gradient-based optimization, use
    ``LogNormalDataMeanQoIAVaRStdDevObjective`` instead.

    Uses the general formula that handles arbitrary (non-equal) posterior
    variances across QoI locations. The ordering of D_j = K_j * exp(tau_j)
    depends on the random posterior slope mu_1^*. The formula enumerates
    all C(Q,2) crossing thresholds on mu_1^* where pairs of D values swap
    order, partitions the real line into intervals of fixed ordering, and
    integrates the AVaR tail contribution in each interval.

    Parameters
    ----------
    prior_mean : Array
        Prior mean. Shape: (nvars, 1)
    prior_cov : Array
        Prior covariance. Shape: (nvars, nvars)
    qoi_mat : Array
        QoI prediction matrix. Shape: (npred, nvars). Must be degree-1
        basis (2 columns: [1, x_j]).
    alpha : float
        AVaR level in [0, 1).
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        prior_mean: Array,
        prior_cov: Array,
        qoi_mat: Array,
        alpha: float,
        bkd: Backend[Array],
    ) -> None:
        self._alpha = alpha
        super().__init__(prior_mean, prior_cov, qoi_mat, bkd)

    def _compute_utility(self) -> float:
        bkd = self._bkd
        npred = self._qoi_mat.shape[0]
        m = math.ceil(npred * (1 - self._alpha))

        Cmat = self._Cmat
        nu_vec = self._nu_vec
        if self._posterior is None:
            raise ValueError("must call set_noise_covariance()")
        Sigma_star = self._posterior.posterior_covariance()

        # Per-QoI statistics
        x_vals: List[float] = []
        nu_j_vals: List[float] = []
        sigma_tau_j_sq_vals: List[float] = []
        sigma_post_j_sq_vals: List[float] = []
        K_vals: List[float] = []
        log_K_vals: List[float] = []

        for j in range(npred):
            psi_j = self._qoi_mat[j : j + 1]
            x_j = float(bkd.to_numpy(self._qoi_mat[j, 1]))
            nu_j = float(bkd.to_numpy((psi_j @ nu_vec)[0, 0]))
            sigma_tau_j_sq = float(
                bkd.to_numpy((psi_j @ Cmat @ psi_j.T)[0, 0])
            )
            sigma_post_j_sq = float(
                bkd.to_numpy((psi_j @ Sigma_star @ psi_j.T)[0, 0])
            )
            K_j = (
                math.exp(sigma_post_j_sq / 2)
                * math.sqrt(math.exp(sigma_post_j_sq) - 1)
            )
            x_vals.append(x_j)
            nu_j_vals.append(nu_j)
            sigma_tau_j_sq_vals.append(sigma_tau_j_sq)
            sigma_post_j_sq_vals.append(sigma_post_j_sq)
            K_vals.append(K_j)
            log_K_vals.append(math.log(K_j))

        # alpha=0 special case: all terms, no ordering needed
        if self._alpha == 0.0:
            total = 0.0
            for j in range(npred):
                total += K_vals[j] * math.exp(
                    nu_j_vals[j] + sigma_tau_j_sq_vals[j] / 2
                )
            return total / m

        # Distribution of mu_1^*: marginal from (mu_0^*, mu_1^*) ~ N(nu, C)
        c_11 = float(bkd.to_numpy(Cmat[1, 1]))
        nu_1 = float(bkd.to_numpy(nu_vec[1, 0]))
        std_mu1 = math.sqrt(c_11)

        # Compute all crossing thresholds:
        # D_i > D_j iff K_i*exp(tau_i) > K_j*exp(tau_j)
        # iff log(K_i) + tau_i > log(K_j) + tau_j
        # iff (x_i - x_j)*mu_1^* > log(K_j/K_i) + (nu_j - nu_i)
        #   [at prior mean level, not needed]
        # Actually: log(D_j) = log(K_j) + tau_j and tau_j = mu_0^* + x_j*mu_1^*
        # So D_i > D_j iff (x_i - x_j)*mu_1^* > log(K_j) - log(K_i)
        #   [conditioned on mu_0^* cancelling]
        # Wait: tau_i - tau_j = (x_i - x_j)*mu_1^*, and log(D_i) - log(D_j)
        #   = log(K_i/K_j) + (x_i - x_j)*mu_1^*
        # So D_i > D_j iff (x_i - x_j)*mu_1^* > log(K_j/K_i)
        # Threshold: mu_1^* = log(K_j/K_i) / (x_i - x_j)  when x_i != x_j
        thresholds: List[float] = []
        for i, j in itertools.combinations(range(npred), 2):
            dx = x_vals[i] - x_vals[j]
            if abs(dx) < 1e-15:
                continue
            t = (log_K_vals[j] - log_K_vals[i]) / dx
            thresholds.append(t)

        thresholds = sorted(set(thresholds))

        # Build intervals: (-inf, t_0), (t_0, t_1), ..., (t_{n-1}, +inf)
        # For each interval, pick a representative point, determine D ordering,
        # identify top-m indices, integrate contribution.

        if not thresholds:
            # All x_vals equal or npred=1 — ordering is fixed everywhere
            span = 1.0
        else:
            span = thresholds[-1] - thresholds[0] if len(thresholds) > 1 else 1.0

        sentinel_lo = (thresholds[0] - 10 * span - 1) if thresholds else nu_1
        sentinel_hi = (thresholds[-1] + 10 * span + 1) if thresholds else nu_1

        # Interval boundaries (including -inf and +inf sentinels)
        boundaries: List[Optional[float]] = (
            [None] + [float(t) for t in thresholds] + [None]
        )

        total = 0.0
        for k in range(len(boundaries) - 1):
            t_lo = boundaries[k]
            t_hi = boundaries[k + 1]

            # Representative point for determining D ordering
            if t_lo is None and t_hi is None:
                rep = nu_1
            elif t_lo is None:
                rep = sentinel_lo
            elif t_hi is None:
                rep = sentinel_hi
            else:
                rep = (t_lo + t_hi) / 2

            # Compute D_j ranking at representative point
            log_D_at_rep = [
                log_K_vals[j] + x_vals[j] * rep for j in range(npred)
            ]
            # Sort descending by log_D -> top-m are AVaR tail
            ranked = sorted(range(npred), key=lambda j: log_D_at_rep[j], reverse=True)
            tail_indices = ranked[:m]

            # Integrate each tail component's contribution over this interval
            # E[K_j * exp(tau_j) * 1(mu_1^* in [t_lo, t_hi])]
            # = K_j * E[exp(nu_0^* + x_j*mu_1^*) * 1(mu_1^* in [t_lo, t_hi])]
            #
            # (tau_j, mu_1^*) is jointly Gaussian. Marginals:
            #   tau_j ~ N(nu_j, sigma_tau_j^2)
            #   mu_1^* ~ N(nu_1, c_11)
            #   Cov(tau_j, mu_1^*) = psi_j^T C e_1 = c_01 + x_j*c_11
            #     where e_1 selects column 1 of C
            #
            # E[exp(tau_j) * 1(mu_1^* in [a,b])]
            #   = exp(nu_j + sigma_tau_j^2/2) * [Phi(d_j(b)) - Phi(d_j(a))]
            # where d_j(t) = (t - nu_1 - cov_j/c_11 * ... )
            # Actually, conditioning on mu_1^* and integrating:
            # E[exp(tau_j) * 1(a < mu_1^* < b)]
            #   = exp(nu_j + sigma_tau_j^2/2)
            #     * [Phi((b - nu_1)/std_mu1 - rho_j*sigma_tau_j/std_mu1)  -- NO
            #
            # Let me use the MGF approach directly.
            # Let Z = (tau_j, mu_1^*) ~ N(mu_Z, Sigma_Z) where
            #   mu_Z = (nu_j, nu_1)
            #   Sigma_Z = [[sigma_tau_j^2, cov_j1], [cov_j1, c_11]]
            # E[exp(tau_j) * 1(a < mu_1^* < b)]
            #   = integral over mu1 of
            #       exp(E[tau_j|mu1] + Var[tau_j|mu1]/2) * phi_mu1(mu1) dmu1
            #     from a to b
            # where E[tau_j|mu1] = nu_j + (cov_j1/c_11)*(mu1 - nu_1)
            #       Var[tau_j|mu1] = sigma_tau_j^2 - cov_j1^2/c_11
            # = exp(nu_j + (sigma_tau_j^2 - cov_j1^2/c_11)/2)
            #   * integral_a^b exp((cov_j1/c_11)*(mu1-nu_1)) * phi(mu1; nu_1, c_11) dmu1
            # = exp(nu_j + (sigma_tau_j^2 - cov_j1^2/c_11)/2)
            #   * exp(cov_j1^2/(2*c_11))
            #   * [Phi((b - nu_1 - cov_j1)/std_mu1) - Phi((a - nu_1 - cov_j1)/std_mu1)]
            #   (completing the square in the exponent)
            # = exp(nu_j + sigma_tau_j^2/2)
            #   * [Phi((b - nu_1 - cov_j1)/std_mu1) - Phi((a - nu_1 - cov_j1)/std_mu1)]

            c_01 = float(bkd.to_numpy(Cmat[0, 1]))

            for j in tail_indices:
                cov_j1 = c_01 + x_vals[j] * c_11
                base = K_vals[j] * math.exp(
                    nu_j_vals[j] + sigma_tau_j_sq_vals[j] / 2
                )
                shift = cov_j1 / std_mu1

                if t_lo is None:
                    phi_lo = 0.0
                else:
                    phi_lo = float(stats.norm.cdf((t_lo - nu_1) / std_mu1 - shift))
                if t_hi is None:
                    phi_hi = 1.0
                else:
                    phi_hi = float(stats.norm.cdf((t_hi - nu_1) / std_mu1 - shift))

                total += base * (phi_hi - phi_lo)

        return total / m


class ConjugateGaussianOEDForLogNormalDataMeanStdDevQoIMeanStdDev(
    ConjugateGaussianOEDPredictionUtilityBase[Array]
):
    """
    E_y[Std(W|y)] + c * Std_y[Std(W|y)] — safety margin utility.

    For scalar lognormal QoI W = exp(psi^T theta), computes:

        U4(w) = K * exp(nu + s^2/2) * (1 + c * sqrt(exp(s^2) - 1))

    where K = sqrt((exp(sigma_post^2) - 1) * exp(sigma_post^2)),
    nu = psi^T E[mu_*], s^2 = psi^T Cov[mu_*] psi, and sigma_post^2
    is the posterior pushforward variance.

    Parameters
    ----------
    prior_mean : Array
        Prior mean. Shape: (nvars, 1)
    prior_cov : Array
        Prior covariance. Shape: (nvars, nvars)
    qoi_mat : Array
        QoI prediction matrix. Shape: (1, nvars) — scalar QoI only.
    safety_factor : float
        The coefficient c >= 0.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        prior_mean: Array,
        prior_cov: Array,
        qoi_mat: Array,
        safety_factor: float,
        bkd: Backend[Array],
    ) -> None:
        self._safety_factor = safety_factor
        super().__init__(prior_mean, prior_cov, qoi_mat, bkd)

    def _compute_utility(self) -> float:
        bkd = self._bkd

        # Posterior pushforward variance
        sigma_post_sq = float(bkd.to_numpy(
            self._post_pushforward.covariance()[0, 0]
        ))
        exp_s_post = math.exp(sigma_post_sq)
        K = math.sqrt((exp_s_post - 1.0) * exp_s_post)

        # E[tau] and Var[tau] = psi^T C psi
        tau_hat = float(bkd.to_numpy(
            (self._qoi_mat @ self._nu_vec)[0, 0]
        ))
        s_sq = float(bkd.to_numpy(
            (self._qoi_mat @ self._Cmat @ self._qoi_mat.T)[0, 0]
        ))

        exp_mean = math.exp(tau_hat + s_sq / 2.0)
        return K * exp_mean * (
            1.0 + self._safety_factor * math.sqrt(math.exp(s_sq) - 1.0)
        )
