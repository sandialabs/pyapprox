"""Tests for AcquisitionOptimizer."""

import numpy as np

from pyapprox.optimization.bayesian.acquisition.analytic import (
    ExpectedImprovement,
)
from pyapprox.optimization.bayesian.acquisition_optimizer import (
    AcquisitionOptimizer,
)
from pyapprox.optimization.bayesian.domain.box import BoxDomain
from pyapprox.optimization.bayesian.protocols import AcquisitionContext
from pyapprox.optimization.minimize.scipy.slsqp import (
    ScipySLSQPOptimizer,
)


class _MockSurrogateForMinimize:
    """Mock surrogate for minimization: mu(x) = (x - 0.7)^2, sigma(x) = 0.2.

    Minimum of mu is at x=0.7 with mu=0. Used with minimize=True.
    """

    def __init__(self, bkd):
        self._bkd = bkd

    def predict(self, X):
        x = X[0]  # (n,)
        mu = (x - 0.7) ** 2
        return self._bkd.reshape(mu, (1, mu.shape[0]))

    def predict_std(self, X):
        n = X.shape[1]
        return self._bkd.full((1, n), 0.2)

    def predict_covariance(self, X):
        n = X.shape[1]
        return self._bkd.eye(n) * 0.04

    def is_fitted(self):
        return True


class TestAcquisitionOptimizer:
    def test_maximize_finds_good_point(self, bkd) -> None:
        """Optimizer finds point near where acquisition is highest."""
        np.random.seed(42)
        surr = _MockSurrogateForMinimize(bkd)
        domain = BoxDomain(bkd.array([[0.0, 1.0]]), bkd)
        # minimize=True: best_value = min(y_observed)
        # Suppose we observed y_min = 0.1
        best_value = bkd.array([0.1])
        ctx = AcquisitionContext(
            surrogate=surr,
            best_value=best_value,
            bkd=bkd,
            pending_X=None,
            minimize=True,
        )

        ei = ExpectedImprovement()
        optimizer = ScipySLSQPOptimizer(maxiter=100)
        acq_opt = AcquisitionOptimizer(
            optimizer, bkd, n_restarts=5, n_raw_candidates=100
        )

        result = acq_opt.maximize(ei, ctx, domain)
        assert result.shape == (1, 1)
        # EI should be highest near x=0.7 where mu is lowest
        result_val = float(bkd.to_numpy(result)[0, 0])
        assert abs(result_val - 0.7) < 0.2

    def test_maximize_respects_bounds(self, bkd) -> None:
        """Optimizer result is within domain bounds."""
        np.random.seed(42)
        surr = _MockSurrogateForMinimize(bkd)
        domain = BoxDomain(bkd.array([[0.2, 0.8]]), bkd)
        best_value = bkd.array([0.1])
        ctx = AcquisitionContext(
            surrogate=surr,
            best_value=best_value,
            bkd=bkd,
            pending_X=None,
            minimize=True,
        )

        ei = ExpectedImprovement()
        optimizer = ScipySLSQPOptimizer(maxiter=100)
        acq_opt = AcquisitionOptimizer(
            optimizer, bkd, n_restarts=5, n_raw_candidates=100
        )

        result = acq_opt.maximize(ei, ctx, domain)
        result_val = float(bkd.to_numpy(result)[0, 0])
        assert 0.2 - 1e-8 <= result_val <= 0.8 + 1e-8
