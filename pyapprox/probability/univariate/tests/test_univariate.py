"""
Tests for univariate distributions.
"""

import pytest
from scipy import stats

from pyapprox.probability.univariate import (
    GaussianMarginal,
    ScipyContinuousMarginal,
    ScipyDiscreteMarginal,
)
from pyapprox.util.backends.numpy import NumpyBkd


class TestGaussianMarginal:
    """Tests for GaussianMarginal."""

    @pytest.fixture(autouse=True)
    def _setup(self, bkd) -> None:
        self._bkd = bkd
        self.mean = 2.0
        self.stdev = 0.5
        self.dist = GaussianMarginal(self.mean, self.stdev, self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns 1."""
        assert self.dist.nvars() == 1

    def test_pdf_at_mean(self) -> None:
        """Test PDF is maximized at mean."""
        samples = self._bkd.asarray([[self.mean - 1, self.mean, self.mean + 1]])
        pdf_vals = self.dist(samples)
        # PDF at mean should be greater than at other points
        assert float(pdf_vals[0, 1]) > float(pdf_vals[0, 0])
        assert float(pdf_vals[0, 1]) > float(pdf_vals[0, 2])

    def test_cdf_at_mean(self) -> None:
        """Test CDF at mean is 0.5."""
        cdf_val = self.dist.cdf(self._bkd.asarray([[self.mean]]))[0, 0]
        assert self._bkd.allclose(
            self._bkd.asarray([float(cdf_val)]),
            self._bkd.asarray([0.5]),
            atol=1e-10,
        )

    def test_cdf_monotonic(self) -> None:
        """Test CDF is monotonically increasing."""
        samples = self._bkd.asarray([[0.0, 1.0, 2.0, 3.0, 4.0]])
        cdf_vals = self.dist.cdf(samples)
        diffs = cdf_vals[0, 1:] - cdf_vals[0, :-1]
        assert self._bkd.all_bool(diffs >= 0)

    def test_invcdf_cdf_inverse(self) -> None:
        """Test invcdf(cdf(x)) = x."""
        samples = self._bkd.asarray([[1.0, 2.0, 3.0]])
        recovered = self.dist.invcdf(self.dist.cdf(samples))
        assert self._bkd.allclose(recovered, samples, rtol=1e-6)

    def test_cdf_invcdf_inverse(self) -> None:
        """Test cdf(invcdf(p)) = p."""
        probs = self._bkd.asarray([[0.1, 0.5, 0.9]])
        recovered = self.dist.cdf(self.dist.invcdf(probs))
        assert self._bkd.allclose(recovered, probs, rtol=1e-6)

    def test_rvs_shape(self) -> None:
        """Test rvs returns correct shape."""
        samples = self.dist.rvs(100)
        assert samples.shape == (1, 100)

    def test_logpdf_exp_equals_pdf(self) -> None:
        """Test exp(logpdf) = pdf."""
        samples = self._bkd.asarray([[1.0, 2.0, 3.0]])
        pdf_vals = self.dist(samples)
        logpdf_vals = self.dist.logpdf(samples)
        assert self._bkd.allclose(self._bkd.exp(logpdf_vals), pdf_vals, rtol=1e-6)

    def test_mean_variance(self) -> None:
        """Test mean and variance."""
        assert self._bkd.allclose(
            self._bkd.asarray([self.dist.mean_value()]),
            self._bkd.asarray([self.mean]),
            atol=1e-10,
        )
        assert self._bkd.allclose(
            self._bkd.asarray([self.dist.variance()]),
            self._bkd.asarray([self.stdev**2]),
            atol=1e-10,
        )
        assert self._bkd.allclose(
            self._bkd.asarray([self.dist.std()]),
            self._bkd.asarray([self.stdev]),
            atol=1e-10,
        )

    def test_is_bounded(self) -> None:
        """Test Gaussian is unbounded."""
        assert not self.dist.is_bounded()

    def test_interval(self) -> None:
        """Test interval contains correct probability."""
        alpha = 0.95
        interval = self.dist.interval(alpha)
        # interval returns shape (1, 2) with [lower, upper] bounds
        assert interval.shape == (1, 2)
        assert float(interval[0, 0]) < self.mean
        assert float(interval[0, 1]) > self.mean

    def test_invcdf_jacobian(self) -> None:
        """Test invcdf Jacobian = 1/pdf at quantile."""
        probs = self._bkd.asarray([[0.25, 0.5, 0.75]])
        jacobian = self.dist.invcdf_jacobian(probs)
        quantiles = self.dist.invcdf(probs)
        pdf_at_quantiles = self.dist(quantiles)
        assert self._bkd.allclose(jacobian, 1.0 / pdf_at_quantiles, rtol=1e-6)

    def test_equality(self) -> None:
        """Test equality comparison."""
        dist2 = GaussianMarginal(self.mean, self.stdev, self._bkd)
        dist3 = GaussianMarginal(0.0, 1.0, self._bkd)
        assert self.dist == dist2
        assert self.dist != dist3


class TestScipyContinuousMarginal:
    """Tests for ScipyContinuousMarginal."""

    @pytest.fixture(autouse=True)
    def _setup(self, bkd) -> None:
        self._bkd = bkd
        # Beta distribution on [0, 1]
        self.scipy_rv = stats.beta(a=2, b=5)
        self.dist = ScipyContinuousMarginal(self.scipy_rv, self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns 1."""
        assert self.dist.nvars() == 1

    def test_pdf(self) -> None:
        """Test PDF matches scipy."""
        samples = self._bkd.asarray([[0.1, 0.3, 0.5, 0.7]])
        pdf_vals = self.dist(samples)
        expected = self._bkd.asarray([self.scipy_rv.pdf([0.1, 0.3, 0.5, 0.7])])
        assert self._bkd.allclose(pdf_vals, expected, rtol=1e-6)

    def test_cdf(self) -> None:
        """Test CDF matches scipy."""
        samples = self._bkd.asarray([[0.1, 0.3, 0.5, 0.7]])
        cdf_vals = self.dist.cdf(samples)
        expected = self._bkd.asarray([self.scipy_rv.cdf([0.1, 0.3, 0.5, 0.7])])
        assert self._bkd.allclose(cdf_vals, expected, rtol=1e-6)

    def test_invcdf(self) -> None:
        """Test invcdf matches scipy ppf."""
        probs = self._bkd.asarray([[0.1, 0.5, 0.9]])
        invcdf_vals = self.dist.invcdf(probs)
        expected = self._bkd.asarray([self.scipy_rv.ppf([0.1, 0.5, 0.9])])
        assert self._bkd.allclose(invcdf_vals, expected, rtol=1e-6)

    def test_rvs_shape(self) -> None:
        """Test rvs returns correct shape."""
        samples = self.dist.rvs(100)
        assert samples.shape == (1, 100)

    def test_is_bounded(self) -> None:
        """Test beta is bounded."""
        assert self.dist.is_bounded()

    def test_unbounded_distribution(self) -> None:
        """Test unbounded distribution detection."""
        normal_rv = stats.norm(loc=0, scale=1)
        normal_dist = ScipyContinuousMarginal(normal_rv, self._bkd)
        assert not normal_dist.is_bounded()

    def test_invalid_distribution(self) -> None:
        """Test that discrete distributions raise error."""
        with pytest.raises(ValueError):
            discrete_rv = stats.binom(n=10, p=0.5)
            ScipyContinuousMarginal(discrete_rv, self._bkd)

    def test_distribution_info(self) -> None:
        """Test distribution info extraction."""
        assert self.dist.name == "beta"
        assert "a" in self.dist.shapes
        assert "b" in self.dist.shapes


class TestScipyDiscreteMarginal:
    """Tests for ScipyDiscreteMarginal."""

    @pytest.fixture(autouse=True)
    def _setup(self, bkd) -> None:
        self._bkd = bkd
        # Binomial distribution
        self.scipy_rv = stats.binom(n=10, p=0.3)
        self.dist = ScipyDiscreteMarginal(self.scipy_rv, self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns 1."""
        assert self.dist.nvars() == 1

    def test_pmf(self) -> None:
        """Test PMF matches scipy."""
        samples = self._bkd.asarray([[0.0, 3.0, 5.0, 10.0]])
        pmf_vals = self.dist(samples)
        expected = self._bkd.asarray([self.scipy_rv.pmf([0, 3, 5, 10])])
        assert self._bkd.allclose(pmf_vals, expected, rtol=1e-6)

    def test_cdf(self) -> None:
        """Test CDF matches scipy."""
        samples = self._bkd.asarray([[0.0, 3.0, 5.0, 10.0]])
        cdf_vals = self.dist.cdf(samples)
        expected = self._bkd.asarray([self.scipy_rv.cdf([0, 3, 5, 10])])
        assert self._bkd.allclose(cdf_vals, expected, rtol=1e-6)

    def test_invcdf(self) -> None:
        """Test invcdf matches scipy ppf."""
        probs = self._bkd.asarray([[0.1, 0.5, 0.9]])
        invcdf_vals = self.dist.invcdf(probs)
        expected = self._bkd.asarray([self.scipy_rv.ppf([0.1, 0.5, 0.9])])
        assert self._bkd.allclose(invcdf_vals, expected, rtol=1e-6)

    def test_rvs_shape(self) -> None:
        """Test rvs returns correct shape."""
        samples = self.dist.rvs(100)
        assert samples.shape == (1, 100)

    def test_is_bounded(self) -> None:
        """Test binomial is bounded."""
        assert self.dist.is_bounded()

    def test_unbounded_distribution(self) -> None:
        """Test unbounded distribution detection."""
        poisson_rv = stats.poisson(mu=5)
        poisson_dist = ScipyDiscreteMarginal(poisson_rv, self._bkd)
        assert not poisson_dist.is_bounded()

    def test_invalid_distribution(self) -> None:
        """Test that continuous distributions raise error."""
        with pytest.raises(ValueError):
            continuous_rv = stats.norm(loc=0, scale=1)
            ScipyDiscreteMarginal(continuous_rv, self._bkd)


class TestProtocolCompliance:
    """Test that distributions satisfy protocol interfaces."""

    def test_gaussian_marginal_protocol(self) -> None:
        """Test GaussianMarginal has required methods."""
        bkd = NumpyBkd()
        dist = GaussianMarginal(0, 1, bkd)

        # MarginalProtocol methods
        assert callable(getattr(dist, "bkd", None))
        assert callable(getattr(dist, "nvars", None))
        assert callable(getattr(dist, "rvs", None))
        assert callable(getattr(dist, "logpdf", None))
        assert callable(getattr(dist, "cdf", None))
        assert callable(getattr(dist, "invcdf", None))

        # MarginalWithJacobianProtocol methods
        assert callable(getattr(dist, "invcdf_jacobian", None))

    def test_scipy_continuous_protocol(self) -> None:
        """Test ScipyContinuousMarginal has required methods."""
        bkd = NumpyBkd()
        dist = ScipyContinuousMarginal(stats.norm(0, 1), bkd)

        assert callable(getattr(dist, "bkd", None))
        assert callable(getattr(dist, "nvars", None))
        assert callable(getattr(dist, "rvs", None))
        assert callable(getattr(dist, "logpdf", None))
        assert callable(getattr(dist, "cdf", None))
        assert callable(getattr(dist, "invcdf", None))

    def test_scipy_discrete_protocol(self) -> None:
        """Test ScipyDiscreteMarginal has required methods."""
        bkd = NumpyBkd()
        dist = ScipyDiscreteMarginal(stats.binom(10, 0.5), bkd)

        assert callable(getattr(dist, "bkd", None))
        assert callable(getattr(dist, "nvars", None))
        assert callable(getattr(dist, "rvs", None))
        assert callable(getattr(dist, "logpdf", None))
        assert callable(getattr(dist, "cdf", None))
        assert callable(getattr(dist, "invcdf", None))
