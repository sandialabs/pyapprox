from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from scipy import stats, optimize
from scipy.special import erfinv, gamma as gamma_fn, gammainc

from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


class ValueAtRiskMixin(ABC):
    """
    Compute the value at risk of a variable Y using a set of samples.
    """

    def __init__(
        self,
        beta: float,
        sort: bool = True,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self.set_beta(beta)
        self._sort = sort
        self._bkd = backend

    def set_beta(self, beta: float):
        if beta < 0 or beta >= 1:
            raise ValueError("beta must be in [0, 1)")
        self._beta = beta

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

    def __call__(self):
        if not hasattr(self, "_samples"):
            raise RuntimeError("must call set_samples first")
        return self._value()

    @abstractmethod
    def _value(self):
        raise NotImplementedError


class ValueAtRisk(ValueAtRiskMixin):
    def _value(self) -> Tuple[Array, int]:
        weights_sum = self._bkd.sum(self._quadw)
        ecdf = self._quadw.cumsum() / weights_sum
        idx = self._bkd.arange(self._samples.shape[0])[ecdf >= self._beta][0]
        return self._samples[idx], idx


class AverageValueAtRisk(ValueAtRiskMixin):
    def _value(self) -> Tuple[Array, Array]:
        VaR = ValueAtRisk(self._beta, False, self._bkd)
        VaR.set_samples(self._samples[None, :], self._quadw)
        var, idx = VaR()
        cvar = (
            var
            + 1.0
            / (1 - self._beta)
            * ((self._samples[idx + 1 :] - var))
            @ self._quadw[idx + 1 :]
        )
        return cvar, var

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


class AnalyticalAVaR:
    @staticmethod
    def gaussian(mu: float, sigma: float, beta: float) -> float:
        """
        Compute the average value at risk of a univariate Gaussian variable
        See https://doi.org/10.1007/s10479-019-03373-1
        Calculating AVaR and bPOE for common probability
        distributions with application to portfolio optimization
        and density estimation.
        """
        return mu + sigma * stats.norm.pdf(stats.norm.ppf(beta)) / (1 - beta)

    @staticmethod
    def _upper_gammainc(a, b):
        return gamma_fn(a) * (1 - gammainc(a, b))

    @staticmethod
    def chi_squared(k: int, beta: float) -> float:
        """
        Compute the average value at risk of a univariate Chi-squared
        variable
        """

        VaR = stats.chi2.ppf(beta, k)
        cvar = (
            2
            * AnalyticalAVaR._upper_gammainc(1 + k / 2, VaR / 2)
            / gamma_fn(k / 2)
            / (1 - beta)
        )
        return cvar

    @staticmethod
    def lognormal_mean(mu, sigma_sq):
        """
        Compute the mean of a univariate lognormal variable
        """
        return np.exp(mu + sigma_sq / 2)

    @staticmethod
    def lognormal(mu: float, sigma_sq: float, beta: float) -> float:
        """
        Compute the average value at risk of a univariate lognormal
        variable
        """
        mean = AnalyticalAVaR.lognormal_mean(mu, sigma_sq)
        if beta == 0:
            return mean
        if sigma_sq < 0 and sigma_sq > -1e-16:
            sigma_sq = 0
        sigma = np.sqrt(sigma_sq)
        quantile = np.exp(mu + sigma * np.sqrt(2) * erfinv(2 * beta - 1))
        return (
            mean
            * stats.norm.cdf((mu + sigma_sq - np.log(quantile)) / sigma)
            / (1 - beta)
        )
