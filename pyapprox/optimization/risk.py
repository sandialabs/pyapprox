from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from scipy import stats, optimize
from scipy.special import (
    erfinv,
    gamma as gamma_fn,
    gammainc,
    beta as beta_fn,
    betainc,
)

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.linalg import (
    inverse_from_cholesky_factor,
    log_determinant_from_cholesky_factor,
)


class RiskMeasure(ABC):
    """
    Compute the value at risk of a variable Y using a set of samples.
    """

    def __init__(
        self,
        sort: bool = True,
        backend: BackendMixin = NumpyMixin,
    ):
        self._sort = sort
        self._bkd = backend

    def set_samples(self, samples: Array, quadw: Array = None):
        if samples.ndim != 2 or samples.shape[0] != 1:
            raise ValueError("samples must be a 2D array with 1 row")
        if quadw is None:
            quadw = self._bkd.full((samples.shape[1],), 1 / samples.shape[1])
        if quadw.ndim != 1:
            raise ValueError("samples must be a 1D array")
        if samples.shape[1] != quadw.shape[0]:
            raise ValueError(f"{quadw.shape=} does not match {samples.shape=}")
        if self._sort:
            idx = self._bkd.argsort(samples[0])
            self._samples = samples[0, idx]
            self._quadw = quadw[idx]
        else:
            self._samples = samples[0]
            self._quadw = quadw

    def __call__(self) -> float:
        if not hasattr(self, "_samples"):
            raise RuntimeError("must call set_samples first")
        return self._value()

    @abstractmethod
    def _value(self) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)


class SafetyMarginRiskMeasure(RiskMeasure):
    def __init__(
        self,
        strength: float,
        backend: BackendMixin = NumpyMixin,
    ):
        self._strength = strength
        super().__init__(sort=False, backend=backend)

    def _value(self) -> float:
        mean = self._samples @ self._quadw
        std = self._bkd.sqrt((self._samples**2) @ self._quadw - mean**2)
        return mean + self._strength * std

    def __repr__(self) -> str:
        return "{0}(strength={1})".format(
            self.__class__.__name__, self._strength
        )


class ValueAtRiskMeasure(RiskMeasure):
    def __init__(
        self,
        beta: float,
        sort: bool = True,
        return_all: bool = True,
        backend: BackendMixin = NumpyMixin,
    ):
        self.set_beta(beta)
        self._return_all = return_all
        super().__init__(sort, backend)

    def set_beta(self, beta: float):
        if beta < 0 or beta >= 1:
            raise ValueError("beta must be in [0, 1)")
        self._beta = beta

    def __repr__(self) -> str:
        return "{0}(q={1})".format(self.__class__.__name__, self._beta)


class ValueAtRisk(ValueAtRiskMeasure):
    def _value(self) -> Tuple[Array, int]:
        weights_sum = self._bkd.sum(self._quadw)
        ecdf = self._bkd.cumsum(self._quadw) / weights_sum
        idx = self._bkd.arange(self._samples.shape[0])[ecdf >= self._beta][0]
        if self._return_all:
            return self._samples[idx], idx
        return self._samples[idx]


class AverageValueAtRisk(ValueAtRiskMeasure):
    def _value(self) -> Tuple[Array, Array]:
        VaR = ValueAtRisk(self._beta, True, backend=self._bkd)
        VaR.set_samples(self._samples[None, :], self._quadw)
        var, idx = VaR()
        cvar = (
            var
            + 1.0
            / (1.0 - self._beta)
            * (self._samples[idx + 1 :] - var)
            @ self._quadw[idx + 1 :]
        )
        if self._return_all:
            return cvar, var
        return cvar

    def optimize(self):
        """
        Compute AVaR by solving linear program.

        Warning
        -------
        When beta is very close to corresponding to a sample
        e.g. beta=4/6, nsamples = 6
        then VaR computed may be the next largest sample after the true AVaR
        E.g. beta=4/6-1e-8 will work but beta=4/6 will not
        """
        nsamples = self._samples.shape[0]
        c = np.hstack((np.array([1]), 1 / (1 - self._beta) * self._quadw))
        A_ineq = -np.hstack((np.ones((nsamples, 1)), np.eye(nsamples)))
        b_ineq = -self._samples
        bounds = [(-np.inf, np.inf)] + [(0, np.inf)] * nsamples
        lin_res = optimize.linprog(
            c,
            A_ineq,
            b_ineq,
            bounds=bounds,
            options={
                "dual_feasibility_tolerance": 1e-10,
                "primal_feasibility_tolerance": 1e-10,
                "ipm_optimality_tolerance": 1e-12,
            },
        )
        return lin_res.fun, lin_res.x[0]


class EntropicRisk(RiskMeasure):
    r"""
    Evaluate the Entropic Risk Measure.

    The entrropic risk measure is not positively homogeneous,
    i.e. R[t*X] != r*R[X].
    Thus a stakeholder may be guided by the risk measure toprefer an out-
    come X over Y when the outcome is measured in dollars:
    i.e. R[X] < R[Y], and yet the same decision-
    maker may prefer Y over X when the profit is measured
    in cents: R[100 X] > R[100 Y ]
    """

    def __init__(
        self,
        beta: float,
        backend: BackendMixin = NumpyMixin,
    ):
        self._beta = beta
        super().__init__(False, backend)

    def _value(self) -> float:
        return (
            self._bkd.log(
                self._bkd.exp(self._beta * self._samples) @ self._quadw
            )
            / self._beta
        )

    def __repr__(self) -> str:
        return "{0}(strength={1})".format(self.__class__.__name__, self._beta)


class UtilitySSD(RiskMeasure):
    r"""
    Compute the conditional expectation of :math:`Y`
    (sutility form of second order stochastic dominance)

    .. math::
      \mathbb{E}\left[\max(0,\eta-Y)\right]

    where \math:`\eta\in Y' in the domain of :math:`Y'

    The conditional expectation is convex non-negative and non-decreasing.
    """

    def __init__(
        self,
        backend: BackendMixin = NumpyMixin,
    ):
        super().__init__(False, backend)

    def set_eta(self, eta: Array):
        if eta.ndim != 1:
            raise ValueError("eta must be a 1D array")
        self._eta = eta

    def _value(self) -> float:
        return (
            self._bkd.maximum(0, self._eta[:, None] - self._samples[None, :])
            @ self._quadw
        )


class DisutilitySSD(UtilitySSD):
    r"""
    Compute the conditional expectation of :math:`-Y`
    (disutility form of second order stochastic dominance)

    .. math::
      \mathbb{E}\left[\max(0,Y-\eta)\right]

    where \math:`\eta\in Y' in the domain of :math:`Y'

    The conditional expectation is convex non-negative and non-decreasing.
    """

    def _value(self) -> float:
        return (
            self._bkd.maximum(0, self._eta[:, None] + self._samples[None, :])
            @ self._quadw
        )


class BetaAnalyticalRiskMeasures:
    def __init__(
        self,
        a: float,
        b: float,
        loc: float = 0,
        scale: float = 1,
    ):
        self._a = a
        self._b = b
        self._loc = loc
        self._scale = float(scale)
        self._marginal = stats.beta(a=a, b=b, loc=loc, scale=scale)

    def AVaR(self, beta: float) -> float:
        qq = self._marginal.ppf(beta)
        qq = (qq - self._loc) / self._scale
        cvar01 = (
            (1 - betainc(1 + self._a, self._b, qq))
            * gamma_fn(1 + self._a)
            * gamma_fn(self._b)
            / gamma_fn(1 + self._a + self._b)
        ) / (beta_fn(self._a, self._b) * (1 - beta))
        return cvar01 * self._scale + self._loc

    def hellinger_divergence(self, a2, b2):
        if self._loc != 0 or self._scale != 1:
            raise RuntimeError("can only compute for variable on [0, 1]")
        return 2 * (
            1
            - beta_fn((self._a + a2) / 2, (self._b + b2) / 2)
            / np.sqrt(beta_fn(self._a, self._b) * beta_fn(a2, b2))
        )


class GaussianAnalyticalRiskMeasures:
    def __init__(
        self,
        mu: float,
        sigma: float,
    ):
        self._mu = mu
        self._sigma = sigma

    def AVaR(self, beta: float) -> float:
        """
        Compute the average value at risk of a univariate Gaussian variable
        See https://doi.org/10.1007/s10479-019-03373-1
        Calculating AVaR and bPOE for common probability
        distributions with application to portfolio optimization
        and density estimation.
        """
        return self._mu + self._sigma * stats.norm.pdf(
            stats.norm.ppf(beta)
        ) / (1 - beta)

    def entropic(self, beta: float) -> float:
        return np.exp(self._mu + self._sigma**2 / 2)

    def kl_divergence(self, mu2: float, sigma2: float) -> float:
        return multivariate_gaussian_kl_divergence(
            np.array([[self._mu]]),
            np.array([[self._sigma**2]]),
            np.array([[mu2]]),
            np.array([[sigma2**2]]),
        )


class ChiSquaredAnalyticalRiskMeasures:
    def __init__(
        self,
        k: int,
    ):
        self._k = k

    def _upper_gammainc(self, a: float, b: float):
        return gamma_fn(a) * (1 - gammainc(a, b))

    def AVaR(self, beta: float) -> float:
        """
        Compute the average value at risk of a univariate Chi-squared
        variable
        """

        VaR = stats.chi2.ppf(beta, self._k)
        avar = (
            2
            * self._upper_gammainc(1 + self._k / 2, VaR / 2)
            / gamma_fn(self._k / 2)
            / (1 - beta)
        )
        return avar


class LogNormalAnalyticalRiskMeasures:
    def __init__(
        self,
        mu: float,
        sigma: float,
    ):
        self._mu = mu
        self._sigma = sigma
        self._marginal = stats.lognorm(scale=np.exp(mu), s=sigma)

    def mean(self) -> float:
        """
        Compute the mean of a univariate lognormal variable
        """
        return np.exp(self._mu + self._sigma**2 / 2)

    def std(self) -> float:
        """
        Compute the standard deviation of a univariate lognormal variable
        """
        return self._marginal.std()

    def VaR(self, beta: float) -> float:
        return self._marginal.ppf(beta)

    def AVaR(self, beta: float) -> float:
        """
        Compute the average value at risk of a univariate lognormal
        variable
        """
        mean = self.mean()
        if beta == 0:
            return mean
        quantile = np.exp(
            self._mu + self._sigma * np.sqrt(2) * erfinv(2 * beta - 1)
        )
        return (
            mean
            * stats.norm.cdf(
                (self._mu + self._sigma**2 - np.log(quantile)) / self._sigma
            )
            / (1 - beta)
        )

    def _expectation_lte_eta(self, eta: float) -> float:
        vals = np.zeros((eta.shape[0],))
        idx = np.where(eta > 0)[0]
        vals[idx] = (
            self.mean()
            * stats.norm.cdf(
                (np.log(eta[idx]) - self._mu - self._sigma**2) / self._sigma
            )
            / self._marginal.cdf(eta[idx])
        )
        return vals

    def _expectation_gte_eta(self, eta: float) -> float:
        vals = np.full((eta.shape[0],), self.mean())
        idx = np.where(eta > 0)[0]
        vals[idx] = (
            self.mean()
            * stats.norm.cdf(
                -(np.log(eta[idx]) - self._mu - self._sigma**2) / self._sigma
            )
            / (1 - self._marginal.cdf(eta[idx]))
        )
        return vals

    def utility_SSD(self, eta: float) -> float:
        return self._marginal.cdf(eta) * (eta - self._expectation_lte_eta(eta))

    def disutility_SSD(self, eta: float) -> float:
        return (1 - self._marginal.cdf(-eta)) * (
            eta + self._expectation_gte_eta(-eta)
        )

    def kl_divergence(self, mu2: float, sigma2: float) -> float:
        return multivariate_gaussian_kl_divergence(
            np.array([[self._mu]]),
            np.array([[self._sigma**2]]),
            np.array([[mu2]]),
            np.array([[sigma2**2]]),
        )

    def __repr__(self) -> str:
        return "{0}(mean={1}, stdev={2})".format(
            self.__class__.__name__, self._mu, self._sigma
        )


def multivariate_gaussian_kl_divergence(
    mean1: Array,
    cov1: Array,
    mean2: Array,
    cov2: Array,
    bkd: BackendMixin = NumpyMixin,
) -> float:
    r"""
    Compute KL( N(mean1, cov1) || N(mean2, cov2) )

    :math:`\int p_1(x)\log\left(\frac{p_1(x)}{p_2(x)}\right)dx`

    :math:`p_2(x)` must dominate :math:`p_1(x)`, e.g. for Bayesian inference
    the :math:`p_2(x)` is the posterior and :math:`p_1(x)` is the prior
    """
    if mean1.ndim != 2 or mean2.ndim != 2:
        raise ValueError("means must have shape (nvars, 1)")
    nvars = mean1.shape[0]
    cov2_inv = bkd.inv(cov2)
    val = bkd.log(bkd.det(cov2) / bkd.det(cov1)) - float(nvars)
    val += bkd.trace(cov2_inv @ cov1)
    val += ((mean2 - mean1).T @ (cov2_inv @ (mean2 - mean1))).squeeze()
    return 0.5 * val


# Useful thesis with derivations of KL and Renyi divergences for a number
# of canonical distributions
# Manuel Gil. 2011. ON RÉNYI DIVERGENCE MEASURES FOR CONTINUOUS ALPHABET
# SOURCES. https://mast.queensu.ca/~communications/Papers/gil-msc11.pdf


class FDivergence(ABC):
    r"""
    Compute f divergence between two densities

    .. math:: \int_\Gamma f\left(\frac{p(z)}{q(z)}\right)q(x)\,dx
    """

    def __init__(
        self,
        density1: callable,
        density2: callable,
        quad_rule_tuple: Tuple[Array, Array],
        backend: BackendMixin = NumpyMixin,
    ):
        r"""
        Parameters
        ----------
        density1 : callable
            The density p(z)

        density2 : callable
            The density q(z)

        quad_rule : tuple
            x,w - quadrature points and weights
            x : np.ndarray (num_vars,num_samples)
            w : np.ndarray (num_samples)
        """
        self._quad_rule_tuple = quad_rule_tuple
        self._density1 = density1
        self._density2 = density2
        self._bkd = backend

    @abstractmethod
    def _divergence_function(self, ratios: Array) -> Array:
        raise NotImplementedError

    def __call__(self) -> float:
        quadx, quadw = self._quad_rule_tuple
        assert quadw.ndim == 1

        d1_vals, d2_vals = self._density1(quadx), self._density2(quadx)
        II = self._bkd.where(d2_vals > 1e-15)[0]
        ratios = self._bkd.zeros(d2_vals.shape) + 1e-15
        ratios[II] = d1_vals[II] / d2_vals[II]
        if not np.all(np.isfinite(ratios)):
            print(d1_vals[II], d2_vals[II])
            msg = "Densities are not absolutely continuous. "
            msg += "Ensure that density2(z)=0 implies density1(z)=0"
            raise Exception(msg)

        divergence_integrand = self._divergence_function(ratios) * d2_vals

        return divergence_integrand @ quadw


class KLDivergence(FDivergence):
    def _divergence_function(self, ratios: Array) -> Array:
        return ratios * self._bkd.log(ratios)


class TVDivergence(FDivergence):
    # Total variation
    def _divergence_function(self, ratios: Array) -> Array:
        return 0.5 * self._bkd.abs(ratios - 1)


class HellingerDivergence(FDivergence):
    # Squared hellinger int (p(z)**0.5-q(z)**0.5)**2 dz
    def _divergence_function(self, ratios: Array) -> Array:
        # Note some formulations use 0.5 times above integral. We do not
        # do that here
        return (self._bkd.sqrt(ratios) - 1) ** 2


class ExactKLDivergence(ABC):
    def __init__(self, backend: BackendMixin = NumpyMixin):
        self._bkd = backend

    @abstractmethod
    def __call__(self) -> float:
        raise NotImplementedError


class IndependentGaussianExactKLDivergence(ExactKLDivergence):
    def __init__(
        self,
        nvars: int,
        backend: BackendMixin = NumpyMixin,
    ):
        super().__init__(backend)
        self._nvars = nvars

    def _check_vector(self, mean: Array):
        if mean.shape != (self._nvars, 1):
            raise ValueError(
                "vector has the wrong shape. Was {0} should be {1}".format(
                    mean.shape, (self._nvars, 1)
                )
            )

    def set_left_distribution(self, mean1: Array, diag1: Array):
        self._check_vector(mean1)
        self._check_vector(diag1)
        self._mean1 = mean1
        self._diag1 = diag1[:, 0]
        self._cov1_log_det = self._bkd.sum(self._bkd.log(self._diag1))

    def set_right_distribution(self, mean2: Array, diag2: Array):
        self._check_vector(mean2)
        self._check_vector(diag2)
        self._mean2 = mean2
        self._diag2 = diag2[:, 0]
        self._cov2_log_det = self._bkd.sum(self._bkd.log(self._diag2))
        self._diag2_inv = 1 / self._diag2

    def __call__(self) -> float:
        val = 0.5 * (
            self._bkd.sum(self._bkd.log(self._diag2))
            - self._bkd.sum(self._bkd.log(self._diag1))
            - self._nvars
            + self._bkd.sum(self._diag1 * self._diag2_inv)
            + (self._mean2 - self._mean1).T
            @ (self._diag2_inv[:, None] * (self._mean2 - self._mean1))
        )
        return val


class CholeskyBasedGaussianExactKLDivergence(ExactKLDivergence):
    def __init__(
        self,
        nvars: int,
        backend: BackendMixin = NumpyMixin,
    ):
        super().__init__(backend)
        self._nvars = nvars

    def _check_mean(self, mean: Array):
        if mean.shape != (self._nvars, 1):
            raise ValueError("mean has the wrong shape")

    def _check_chol_factor(self, chol: Array):
        if chol.shape != (self._nvars, self._nvars):
            raise ValueError("cholesky factor has the wrong shape")

    def set_left_distribution(self, mean1: Array, chol1: Array):
        self._check_mean(mean1)
        self._check_chol_factor(chol1)
        self._mean1 = mean1
        self._chol1 = chol1
        self._cov1_log_det = log_determinant_from_cholesky_factor(
            self._chol1, self._bkd
        )

    def set_right_distribution(self, mean2: Array, chol2: Array):
        self._check_mean(mean2)
        self._check_chol_factor(chol2)
        self._mean2 = mean2
        self._chol2 = chol2
        self._cov2_inv = inverse_from_cholesky_factor(self._chol2, self._bkd)
        self._cov2_log_det = log_determinant_from_cholesky_factor(
            self._chol2, self._bkd
        )

    def __call__(self) -> float:
        val = 0.5 * (
            self._cov2_log_det
            - self._cov1_log_det
            - self._nvars
            # + self._bkd.trace(
            #     cov2_inv @ (self._chol1 @ self._chol1.T)
            # )  # replace with sum of hadamard prod below
            + self._bkd.sum(self._cov2_inv * (self._chol1 @ self._chol1.T))
            + (self._mean2 - self._mean1).T
            @ (self._cov2_inv @ (self._mean2 - self._mean1))
        )
        return val


#  See https://link.springer.com/article/10.1007/s10287-014-0225-7
#  Port over from new code if needed
# def cvar_importance_sampling_biasing_density(pdf, function, beta, VaR, tau, x):
#
# def generate_samples_from_cvar_importance_sampling_biasing_density(
#     function, beta, VaR, generate_candidate_samples, nsamples
# ):
