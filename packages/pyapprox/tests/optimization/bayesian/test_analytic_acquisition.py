"""Tests for analytic acquisition functions."""


from pyapprox.optimization.bayesian.acquisition.analytic import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from pyapprox.optimization.bayesian.protocols import AcquisitionContext


class _MockSurrogate:
    """Mock surrogate with fixed mu and sigma for testing."""

    def __init__(self, mu, sigma, bkd):
        self._mu = mu
        self._sigma = sigma
        self._bkd = bkd

    def predict(self, X):
        n = X.shape[1]
        return self._bkd.tile(
            self._bkd.reshape(self._mu, (1, 1)), (1, n)
        )

    def predict_std(self, X):
        n = X.shape[1]
        return self._bkd.tile(
            self._bkd.reshape(self._sigma, (1, 1)), (1, n)
        )

    def predict_covariance(self, X):
        n = X.shape[1]
        return self._bkd.eye(n) * self._sigma * self._sigma

    def is_fitted(self):
        return True


class TestExpectedImprovement:
    def test_ei_positive(self, bkd) -> None:
        """EI > 0 when there is potential for improvement."""
        mu = bkd.array([0.5])
        sigma = bkd.array([0.3])
        surr = _MockSurrogate(mu, sigma, bkd)
        best_value = bkd.array([1.0])  # maximization space
        ctx = AcquisitionContext(
            surrogate=surr,
            best_value=best_value,
            bkd=bkd,
            pending_X=None,
            minimize=False,
        )
        ei = ExpectedImprovement()
        X = bkd.array([[0.0]])  # dummy point
        result = ei.evaluate(X, ctx)
        assert float(bkd.to_numpy(result)[0]) > 0.0

    def test_ei_zero_sigma(self, bkd) -> None:
        """EI = 0 when sigma = 0."""
        mu = bkd.array([0.5])
        sigma = bkd.array([0.0])
        surr = _MockSurrogate(mu, sigma, bkd)
        best_value = bkd.array([1.0])
        ctx = AcquisitionContext(
            surrogate=surr,
            best_value=best_value,
            bkd=bkd,
            pending_X=None,
            minimize=False,
        )
        ei = ExpectedImprovement()
        X = bkd.array([[0.0]])
        result = ei.evaluate(X, ctx)
        bkd.assert_allclose(result, bkd.array([0.0]), atol=1e-10)

    def test_ei_minimize(self, bkd) -> None:
        """EI works in minimization mode."""
        # mu = 2.0, best observed y_min = 1.0
        # best_value = 1.0 (raw min)
        mu = bkd.array([2.0])
        sigma = bkd.array([1.0])
        surr = _MockSurrogate(mu, sigma, bkd)
        best_value = bkd.array([1.0])
        ctx = AcquisitionContext(
            surrogate=surr,
            best_value=best_value,
            bkd=bkd,
            pending_X=None,
            minimize=True,
        )
        ei = ExpectedImprovement()
        X = bkd.array([[0.0]])
        result = ei.evaluate(X, ctx)
        # Should still be >= 0
        assert float(bkd.to_numpy(result)[0]) >= 0.0

    def test_ei_batch(self, bkd) -> None:
        """EI handles multiple points."""
        mu = bkd.array([0.5])
        sigma = bkd.array([0.3])
        surr = _MockSurrogate(mu, sigma, bkd)
        best_value = bkd.array([1.0])
        ctx = AcquisitionContext(
            surrogate=surr,
            best_value=best_value,
            bkd=bkd,
            pending_X=None,
            minimize=False,
        )
        ei = ExpectedImprovement()
        X = bkd.array([[0.0, 0.5, 1.0]])
        result = ei.evaluate(X, ctx)
        assert result.shape == (3,)


class TestUpperConfidenceBound:
    def test_ucb_maximize(self, bkd) -> None:
        """UCB = mu + beta * sigma for maximization."""
        mu = bkd.array([1.0])
        sigma = bkd.array([0.5])
        surr = _MockSurrogate(mu, sigma, bkd)
        best_value = bkd.array([1.0])
        ctx = AcquisitionContext(
            surrogate=surr,
            best_value=best_value,
            bkd=bkd,
            pending_X=None,
            minimize=False,
        )
        ucb = UpperConfidenceBound(beta=2.0)
        X = bkd.array([[0.0]])
        result = ucb.evaluate(X, ctx)
        expected = bkd.array([1.0 + 2.0 * 0.5])
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_ucb_minimize(self, bkd) -> None:
        """UCB = -mu + beta * sigma for minimization."""
        mu = bkd.array([1.0])
        sigma = bkd.array([0.5])
        surr = _MockSurrogate(mu, sigma, bkd)
        best_value = bkd.array([1.0])  # raw min(y)
        ctx = AcquisitionContext(
            surrogate=surr,
            best_value=best_value,
            bkd=bkd,
            pending_X=None,
            minimize=True,
        )
        ucb = UpperConfidenceBound(beta=2.0)
        X = bkd.array([[0.0]])
        result = ucb.evaluate(X, ctx)
        # UCB for minimize: -mu + beta*sigma = -1.0 + 2.0*0.5 = 0.0
        expected = bkd.array([-1.0 + 2.0 * 0.5])
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_ucb_custom_beta(self, bkd) -> None:
        """UCB with custom beta."""
        mu = bkd.array([0.0])
        sigma = bkd.array([1.0])
        surr = _MockSurrogate(mu, sigma, bkd)
        best_value = bkd.array([0.0])
        ctx = AcquisitionContext(
            surrogate=surr,
            best_value=best_value,
            bkd=bkd,
            pending_X=None,
            minimize=False,
        )
        ucb = UpperConfidenceBound(beta=3.0)
        X = bkd.array([[0.0]])
        result = ucb.evaluate(X, ctx)
        expected = bkd.array([3.0])
        bkd.assert_allclose(result, expected, rtol=1e-12)


class TestProbabilityOfImprovement:
    def test_pi_positive(self, bkd) -> None:
        """PI > 0 when improvement is possible."""
        mu = bkd.array([0.5])
        sigma = bkd.array([0.3])
        surr = _MockSurrogate(mu, sigma, bkd)
        best_value = bkd.array([1.0])
        ctx = AcquisitionContext(
            surrogate=surr,
            best_value=best_value,
            bkd=bkd,
            pending_X=None,
            minimize=False,
        )
        pi = ProbabilityOfImprovement()
        X = bkd.array([[0.0]])
        result = pi.evaluate(X, ctx)
        val = float(bkd.to_numpy(result)[0])
        assert 0.0 < val <= 1.0

    def test_pi_zero_sigma(self, bkd) -> None:
        """PI = 0 when sigma = 0."""
        mu = bkd.array([0.5])
        sigma = bkd.array([0.0])
        surr = _MockSurrogate(mu, sigma, bkd)
        best_value = bkd.array([1.0])
        ctx = AcquisitionContext(
            surrogate=surr,
            best_value=best_value,
            bkd=bkd,
            pending_X=None,
            minimize=False,
        )
        pi = ProbabilityOfImprovement()
        X = bkd.array([[0.0]])
        result = pi.evaluate(X, ctx)
        bkd.assert_allclose(result, bkd.array([0.0]), atol=1e-10)

    def test_pi_high_when_mu_near_best(self, bkd) -> None:
        """PI is high when mu is near best with large sigma."""
        mu = bkd.array([0.9])
        sigma = bkd.array([1.0])
        surr = _MockSurrogate(mu, sigma, bkd)
        best_value = bkd.array([1.0])
        ctx = AcquisitionContext(
            surrogate=surr,
            best_value=best_value,
            bkd=bkd,
            pending_X=None,
            minimize=False,
        )
        pi = ProbabilityOfImprovement()
        X = bkd.array([[0.0]])
        result = pi.evaluate(X, ctx)
        val = float(bkd.to_numpy(result)[0])
        assert val > 0.4  # should be around 0.54
