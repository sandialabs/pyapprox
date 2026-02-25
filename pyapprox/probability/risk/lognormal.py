"""
Analytical risk measures for lognormal distributions.

Provides closed-form formulas for various risk measures of univariate
lognormal random variables.
"""

import numpy as np
from scipy import stats
from scipy.special import erfinv


class LogNormalAnalyticalRiskMeasures:
    """
    Analytical risk measures for univariate lognormal distributions.

    A lognormal random variable Y = exp(X) where X ~ N(mu, sigma^2).

    Parameters
    ----------
    mu : float
        Mean of the underlying normal distribution (not the lognormal mean).
    sigma : float
        Standard deviation of the underlying normal distribution.

    Examples
    --------
    >>> risk = LogNormalAnalyticalRiskMeasures(mu=0.0, sigma=1.0)
    >>> risk.mean()  # doctest: +ELLIPSIS
    1.6487...
    >>> risk.std()  # doctest: +ELLIPSIS
    2.1612...
    """

    def __init__(self, mu: float, sigma: float) -> None:
        self._mu = mu
        self._sigma = sigma
        self._marginal = stats.lognorm(scale=np.exp(mu), s=sigma)

    def mean(self) -> float:
        """
        Compute the mean of the lognormal distribution.

        Returns
        -------
        float
            E[exp(X)] = exp(mu + sigma^2/2)
        """
        return np.exp(self._mu + self._sigma**2 / 2)

    def std(self) -> float:
        """
        Compute the standard deviation of the lognormal distribution.

        Returns
        -------
        float
            Standard deviation of the lognormal variable.
        """
        return self._marginal.std()

    def variance(self) -> float:
        """
        Compute the variance of the lognormal distribution.

        Returns
        -------
        float
            Variance of the lognormal variable.
        """
        return self._marginal.var()

    def VaR(self, beta: float) -> float:
        """
        Compute the Value at Risk at level beta.

        Parameters
        ----------
        beta : float
            Risk level in [0, 1).

        Returns
        -------
        float
            The beta-quantile of the lognormal distribution.
        """
        return self._marginal.ppf(beta)

    def AVaR(self, beta: float) -> float:
        """
        Compute the Average Value at Risk (CVaR) at level beta.

        For a lognormal distribution with underlying normal N(mu, sigma^2):
            AVaR_beta = E[Y] * Phi((mu + sigma^2 - log(q_beta))/sigma) / (1-beta)

        where q_beta = exp(mu + sigma * Phi^{-1}(beta)) is the beta-quantile.

        Parameters
        ----------
        beta : float
            Risk level in [0, 1).

        Returns
        -------
        float
            Average Value at Risk.
        """
        mean = self.mean()
        if beta == 0:
            return mean
        quantile = np.exp(self._mu + self._sigma * np.sqrt(2) * erfinv(2 * beta - 1))
        return (
            mean
            * stats.norm.cdf(
                (self._mu + self._sigma**2 - np.log(quantile)) / self._sigma
            )
            / (1 - beta)
        )

    def _expectation_lte_eta(self, eta: np.ndarray) -> np.ndarray:
        """
        Compute E[Y | Y <= eta] * P(Y <= eta).

        Parameters
        ----------
        eta : ndarray
            Threshold values.

        Returns
        -------
        ndarray
            Conditional expectation contribution.
        """
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

    def _expectation_gte_eta(self, eta: np.ndarray) -> np.ndarray:
        """
        Compute E[Y | Y >= eta] * P(Y >= eta).

        Parameters
        ----------
        eta : ndarray
            Threshold values.

        Returns
        -------
        ndarray
            Conditional expectation contribution.
        """
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

    def utility_SSD(self, eta: np.ndarray) -> np.ndarray:
        """
        Compute the utility form of second-order stochastic dominance.

        Parameters
        ----------
        eta : ndarray
            Threshold values.

        Returns
        -------
        ndarray
            SSD utility values.
        """
        return self._marginal.cdf(eta) * (eta - self._expectation_lte_eta(eta))

    def disutility_SSD(self, eta: np.ndarray) -> np.ndarray:
        """
        Compute the disutility form of second-order stochastic dominance.

        Parameters
        ----------
        eta : ndarray
            Threshold values.

        Returns
        -------
        ndarray
            SSD disutility values.
        """
        return (1 - self._marginal.cdf(-eta)) * (eta + self._expectation_gte_eta(-eta))

    def kl_divergence(self, mu2: float, sigma2: float) -> float:
        """
        Compute KL divergence to another lognormal distribution.

        The KL divergence between two lognormal distributions equals the
        KL divergence between the underlying normal distributions.

        Parameters
        ----------
        mu2 : float
            Mean of the underlying normal of the second distribution.
        sigma2 : float
            Standard deviation of the underlying normal of the second distribution.

        Returns
        -------
        float
            KL(self || LogNormal(mu2, sigma2))
        """
        # KL(N(mu1, s1^2) || N(mu2, s2^2)) =
        #   log(s2/s1) + (s1^2 + (mu1-mu2)^2)/(2*s2^2) - 0.5
        return (
            np.log(sigma2 / self._sigma)
            + (self._sigma**2 + (self._mu - mu2) ** 2) / (2 * sigma2**2)
            - 0.5
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(mu={self._mu}, sigma={self._sigma})"
