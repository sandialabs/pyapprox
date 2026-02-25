"""
Analytical risk measures for Gaussian distributions.

Provides closed-form formulas for various risk measures of univariate
Gaussian random variables.
"""

from scipy import stats


class GaussianAnalyticalRiskMeasures:
    """
    Analytical risk measures for univariate Gaussian distributions.

    Provides closed-form formulas for mean, variance, standard deviation,
    entropic risk, and Average Value at Risk (AVaR/CVaR) for Gaussian
    random variables.

    Parameters
    ----------
    mu : float
        Mean of the Gaussian distribution.
    sigma : float
        Standard deviation of the Gaussian distribution.

    Examples
    --------
    >>> risk = GaussianAnalyticalRiskMeasures(mu=0.0, sigma=1.0)
    >>> risk.mean()
    0.0
    >>> risk.variance()
    1.0
    >>> risk.AVaR(0.5)  # doctest: +ELLIPSIS
    0.79788...
    """

    def __init__(self, mu: float, sigma: float) -> None:
        self._mu = mu
        self._sigma = sigma

    def mean(self) -> float:
        """Return the mean of the distribution."""
        return self._mu

    def variance(self) -> float:
        """Return the variance of the distribution."""
        return self._sigma**2

    def stdev(self) -> float:
        """Return the standard deviation of the distribution."""
        return self._sigma

    def mean_plus_stddev(self, alpha: float) -> float:
        """
        Compute mean plus alpha times standard deviation.

        Parameters
        ----------
        alpha : float
            Coefficient for standard deviation.

        Returns
        -------
        float
            mu + alpha * sigma
        """
        return self._mu + alpha * self._sigma

    def entropic(self, alpha: float) -> float:
        """
        Compute the entropic risk measure.

        The entropic risk measure for a Gaussian is:
            (1/alpha) * log(E[exp(alpha * X)]) = mu + alpha * sigma^2 / 2

        Parameters
        ----------
        alpha : float
            Risk aversion parameter (must be > 0).

        Returns
        -------
        float
            Entropic risk value.

        Notes
        -----
        The entropic risk measure is NOT positively homogeneous, meaning
        R(t*X) != t*R(X). This can lead to scale-dependent preferences.
        """
        return self._mu + alpha * self._sigma**2 / 2.0

    def AVaR(self, beta: float) -> float:
        """
        Compute the Average Value at Risk (CVaR).

        For a Gaussian distribution:
            AVaR_beta = mu + sigma * phi(Phi^{-1}(beta)) / (1 - beta)

        where phi is the standard normal PDF and Phi^{-1} is the inverse CDF.

        Parameters
        ----------
        beta : float
            Risk level in [0, 1). AVaR_beta is the expected value of the
            random variable given that it exceeds the beta-quantile.

        Returns
        -------
        float
            Average Value at Risk.

        References
        ----------
        https://doi.org/10.1007/s10479-019-03373-1
        "Calculating AVaR and bPOE for common probability distributions
        with application to portfolio optimization and density estimation"
        """
        return self._mu + self._sigma * stats.norm.pdf(stats.norm.ppf(beta)) / (
            1.0 - beta
        )
