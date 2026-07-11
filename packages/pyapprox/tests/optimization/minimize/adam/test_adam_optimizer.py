"""Regression tests for AdamOptimizer against the public bind/minimize
API, using legacy-style producers. Written BEFORE the Derivatives-bundle
consumer rewrite so the rewrite is protected."""

import pytest

from tests._helpers.optimizer_fixtures import (
    QuadraticNoDerivatives,
    QuadraticWithJacobian,
)

from pyapprox.optimization.minimize.adam.adam_optimizer import AdamOptimizer


class TestAdamOptimizer:
    def test_converges_with_analytic_jacobian(self, bkd):
        objective = QuadraticWithJacobian(bkd, [1.0, -0.5])
        bounds = bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]])
        optimizer = AdamOptimizer(lr=0.1, maxiter=1000).bind(
            objective, bounds
        )
        result = optimizer.minimize(bkd.asarray([[0.0], [0.0]]))
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[1.0], [-0.5]]), atol=1e-3
        )
        assert objective.njacobian_calls > 0

    def test_rejects_objective_without_jacobian(self, bkd):
        objective = QuadraticNoDerivatives(bkd, [1.0, -0.5])
        bounds = bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]])
        with pytest.raises(TypeError, match="jacobian"):
            AdamOptimizer().bind(objective, bounds)
