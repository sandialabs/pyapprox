"""
Standalone tests for LogNormal analytical risk measures.

PERMANENT - no legacy imports.
"""

import numpy as np
import pytest
from scipy import stats

from pyapprox.probability.risk import LogNormalAnalyticalRiskMeasures


class TestLogNormalAnalyticalRiskMeasures:
    """Tests for LogNormalAnalyticalRiskMeasures."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mu = 0.5
        self.sigma = 1.0
        self.risk = LogNormalAnalyticalRiskMeasures(self.mu, self.sigma)

    def test_mean_formula(self):
        """Test mean returns exp(mu + sigma^2/2)."""
        expected = np.exp(self.mu + self.sigma**2 / 2)
        assert self.risk.mean() == pytest.approx(expected, abs=1e-12)

    def test_mean_matches_scipy(self):
        """Test mean matches scipy lognorm distribution."""
        rv = stats.lognorm(scale=np.exp(self.mu), s=self.sigma)
        assert self.risk.mean() == pytest.approx(rv.mean(), abs=1e-12)

    def test_std_matches_scipy(self):
        """Test std matches scipy lognorm distribution."""
        rv = stats.lognorm(scale=np.exp(self.mu), s=self.sigma)
        assert self.risk.std() == pytest.approx(rv.std(), abs=1e-12)

    def test_variance_matches_scipy(self):
        """Test variance matches scipy lognorm distribution."""
        rv = stats.lognorm(scale=np.exp(self.mu), s=self.sigma)
        assert self.risk.variance() == pytest.approx(rv.var(), abs=1e-12)

    def test_var_matches_scipy_ppf(self):
        """Test VaR matches scipy quantile function."""
        rv = stats.lognorm(scale=np.exp(self.mu), s=self.sigma)
        for beta in [0.1, 0.5, 0.9, 0.95]:
            assert self.risk.VaR(beta) == pytest.approx(rv.ppf(beta), abs=1e-12)

    def test_avar_at_beta_zero(self):
        """Test AVaR at beta=0 equals mean."""
        assert self.risk.AVaR(0) == pytest.approx(self.risk.mean(), abs=1e-12)

    def test_avar_greater_than_mean(self):
        """Test AVaR is always >= mean for beta > 0."""
        for beta in [0.1, 0.5, 0.9, 0.95]:
            avar = self.risk.AVaR(beta)
            assert avar >= self.risk.mean()

    def test_avar_increases_with_beta(self):
        """Test AVaR is increasing in beta."""
        betas = [0.1, 0.3, 0.5, 0.7, 0.9]
        avars = [self.risk.AVaR(b) for b in betas]
        for i in range(len(avars) - 1):
            assert avars[i] < avars[i + 1]

    def test_avar_monte_carlo_approximation(self):
        """Test AVaR matches Monte Carlo approximation."""
        np.random.seed(42)
        rv = stats.lognorm(scale=np.exp(self.mu), s=self.sigma)
        nsamples = int(1e5)
        samples = rv.rvs(nsamples)

        beta = 0.75
        quantile = np.percentile(samples, beta * 100)
        tail_samples = samples[samples >= quantile]
        mc_avar = np.mean(tail_samples)

        analytical_avar = self.risk.AVaR(beta)
        assert analytical_avar == pytest.approx(mc_avar, abs=0.1)

    def test_kl_divergence_same_distribution(self):
        """Test KL divergence to self is zero."""
        kl = self.risk.kl_divergence(self.mu, self.sigma)
        assert kl == pytest.approx(0.0, abs=1e-12)

    def test_kl_divergence_different_distribution(self):
        """Test KL divergence to different distribution is positive."""
        mu2, sigma2 = 1.0, 2.0
        kl = self.risk.kl_divergence(mu2, sigma2)
        assert kl > 0.0

    def test_kl_divergence_formula(self):
        """Test KL divergence matches analytical formula for normals."""
        # KL between lognormals = KL between underlying normals
        mu2, sigma2 = 1.0, 2.0
        # KL(N(mu1, s1^2) || N(mu2, s2^2)) =
        #   log(s2/s1) + (s1^2 + (mu1-mu2)^2)/(2*s2^2) - 0.5
        expected = (
            np.log(sigma2 / self.sigma)
            + (self.sigma**2 + (self.mu - mu2) ** 2) / (2 * sigma2**2)
            - 0.5
        )
        assert self.risk.kl_divergence(mu2, sigma2) == pytest.approx(
            expected, abs=1e-12
        )

    def test_utility_ssd_at_zero(self):
        """Test utility SSD at eta=0."""
        eta = np.array([0.0])
        result = self.risk.utility_SSD(eta)
        # At eta=0, CDF(0)=0 for lognormal, so result should be 0
        assert result[0] == pytest.approx(0.0, abs=1e-12)

    def test_utility_ssd_nonnegative(self):
        """Test utility SSD is non-negative for positive eta."""
        eta = np.array([0.5, 1.0, 2.0, 5.0])
        result = self.risk.utility_SSD(eta)
        for val in result:
            assert val >= 0.0

    def test_disutility_ssd_formula(self):
        """Test disutility SSD formula consistency."""
        eta = np.array([0.5, 1.0, 2.0])
        result = self.risk.disutility_SSD(eta)
        # Just check it returns values without error and they are finite
        for val in result:
            assert np.isfinite(val)

    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.risk)
        assert "LogNormalAnalyticalRiskMeasures" in repr_str
        assert str(self.mu) in repr_str
        assert str(self.sigma) in repr_str


class TestLogNormalAnalyticalRiskMeasuresStandardParams:
    """Test with standard parameters mu=0, sigma=1."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.risk = LogNormalAnalyticalRiskMeasures(0.0, 1.0)

    def test_mean_standard(self):
        """Test mean for standard lognormal."""
        # E[exp(Z)] where Z ~ N(0,1) = exp(0.5)
        expected = np.exp(0.5)
        assert self.risk.mean() == pytest.approx(expected, abs=1e-12)

    def test_median_is_one(self):
        """Test median (VaR at 0.5) is exp(mu)=1 for mu=0."""
        assert self.risk.VaR(0.5) == pytest.approx(1.0, abs=1e-12)
