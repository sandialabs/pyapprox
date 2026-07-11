"""Regression tests for LBFGSBOptimizer against the public bind/minimize
API, using legacy-style producers. Written BEFORE the Derivatives-bundle
consumer rewrite so the rewrite is protected."""

from tests._helpers.optimizer_fixtures import (
    QuadraticNoDerivatives,
    QuadraticWithJacobian,
)

from pyapprox.optimization.minimize.scipy.lbfgsb import LBFGSBOptimizer


class TestLBFGSBOptimizer:
    def test_converges_with_analytic_jacobian(self, bkd):
        objective = QuadraticWithJacobian(bkd, [1.0, -0.5])
        bounds = bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]])
        optimizer = LBFGSBOptimizer(gtol=1e-10).bind(objective, bounds)
        result = optimizer.minimize(bkd.asarray([[0.0], [0.0]]))
        assert result.success()
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[1.0], [-0.5]]), atol=1e-6
        )
        # the analytic jacobian path must actually be wired
        assert objective.njacobian_calls > 0

    def test_converges_without_jacobian(self, bkd):
        objective = QuadraticNoDerivatives(bkd, [1.0, -0.5])
        bounds = bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]])
        optimizer = LBFGSBOptimizer().bind(objective, bounds)
        result = optimizer.minimize(bkd.asarray([[0.0], [0.0]]))
        assert result.success()
        # scipy's own finite differencing supplies the gradient
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[1.0], [-0.5]]), atol=1e-5
        )

    def test_bounds_are_respected(self, bkd):
        objective = QuadraticWithJacobian(bkd, [0.0, 0.0])
        bounds = bkd.asarray([[1.0, 5.0], [-5.0, 5.0]])
        optimizer = LBFGSBOptimizer().bind(objective, bounds)
        result = optimizer.minimize(bkd.asarray([[2.0], [1.0]]))
        # unconstrained optimum (0, 0) is infeasible in x0: clamps to 1
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[1.0], [0.0]]), atol=1e-6
        )

    def test_minimize_before_bind_raises(self, numpy_bkd):
        optimizer = LBFGSBOptimizer()
        try:
            optimizer.minimize(numpy_bkd.asarray([[0.0]]))
        except RuntimeError as error:
            assert "bind" in str(error)
        else:
            raise AssertionError("expected RuntimeError")
