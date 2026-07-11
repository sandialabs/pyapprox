"""Regression tests for ScipySLSQPOptimizer against the public
bind/minimize API, using legacy-style producers. Written BEFORE the
Derivatives-bundle consumer rewrite so the rewrite is protected."""

from tests._helpers.optimizer_fixtures import (
    QuadraticNoDerivatives,
    QuadraticWithJacobian,
    SumConstraint,
    SumConstraintWithJacobian,
)

from pyapprox.optimization.minimize.scipy.slsqp import ScipySLSQPOptimizer


class TestScipySLSQPOptimizer:
    def test_converges_with_analytic_jacobian(self, bkd):
        objective = QuadraticWithJacobian(bkd, [1.0, -0.5])
        bounds = bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]])
        optimizer = ScipySLSQPOptimizer(ftol=1e-12).bind(objective, bounds)
        result = optimizer.minimize(bkd.asarray([[0.0], [0.0]]))
        assert result.success()
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[1.0], [-0.5]]), atol=1e-6
        )
        assert objective.njacobian_calls > 0

    def test_converges_without_jacobian(self, bkd):
        objective = QuadraticNoDerivatives(bkd, [1.0, -0.5])
        bounds = bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]])
        optimizer = ScipySLSQPOptimizer().bind(objective, bounds)
        result = optimizer.minimize(bkd.asarray([[0.0], [0.0]]))
        assert result.success()
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[1.0], [-0.5]]), atol=1e-5
        )

    def test_inequality_constraint_with_jacobian(self, bkd):
        # minimize sum(x^2) s.t. x0 + x1 >= 1 -> optimum (0.5, 0.5)
        objective = QuadraticWithJacobian(bkd, [0.0, 0.0])
        constraint = SumConstraintWithJacobian(bkd, 2, 1.0, float("inf"))
        bounds = bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]])
        optimizer = ScipySLSQPOptimizer().bind(
            objective, bounds, [constraint]
        )
        result = optimizer.minimize(bkd.asarray([[2.0], [2.0]]))
        assert result.success()
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[0.5], [0.5]]), atol=1e-6
        )
        assert constraint.njacobian_calls > 0

    def test_inequality_constraint_without_jacobian(self, bkd):
        objective = QuadraticWithJacobian(bkd, [0.0, 0.0])
        constraint = SumConstraint(bkd, 2, 1.0, float("inf"))
        bounds = bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]])
        optimizer = ScipySLSQPOptimizer().bind(
            objective, bounds, [constraint]
        )
        result = optimizer.minimize(bkd.asarray([[2.0], [2.0]]))
        assert result.success()
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[0.5], [0.5]]), atol=1e-5
        )

    def test_equality_constraint(self, bkd):
        # minimize sum(x^2) s.t. x0 + x1 == 2 -> optimum (1, 1)
        objective = QuadraticWithJacobian(bkd, [0.0, 0.0])
        constraint = SumConstraintWithJacobian(bkd, 2, 2.0, 2.0)
        bounds = bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]])
        optimizer = ScipySLSQPOptimizer().bind(
            objective, bounds, [constraint]
        )
        result = optimizer.minimize(bkd.asarray([[2.0], [0.5]]))
        assert result.success()
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[1.0], [1.0]]), atol=1e-6
        )
