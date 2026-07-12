"""Regression tests for ScipyTrustConstrOptimizer against the public
bind/minimize API, using legacy-style producers. Written BEFORE the
Derivatives-bundle consumer rewrite so the rewrite is protected."""

from tests._helpers.optimizer_fixtures import (
    QuadraticNoDerivatives,
    QuadraticWithJacobian,
    QuadraticWithJacobianAndHVP,
    SumConstraint,
    SumConstraintWithJacobian,
    SumConstraintWithJacobianAndWHVP,
)

from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)


class TestScipyTrustConstrOptimizer:
    def test_converges_with_jacobian_and_hvp(self, bkd):
        objective = QuadraticWithJacobianAndHVP(bkd, [1.0, -0.5])
        bounds = bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]])
        optimizer = ScipyTrustConstrOptimizer(gtol=1e-10).bind(
            objective, bounds
        )
        result = optimizer.minimize(bkd.asarray([[0.0], [0.0]]))
        assert result.success()
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[1.0], [-0.5]]), atol=1e-6
        )
        # both second-order paths must actually be wired
        assert objective.njacobian_calls > 0
        assert objective.nhvp_calls > 0

    def test_hvp_probe_vectors_are_coerced_to_double(self, bkd):
        # scipy probes hessp with an int8 vector during setup; the
        # optimizer stack must coerce it before it reaches the producer
        objective = QuadraticWithJacobianAndHVP(bkd, [1.0, -0.5])
        bounds = bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]])
        optimizer = ScipyTrustConstrOptimizer().bind(objective, bounds)
        optimizer.minimize(bkd.asarray([[0.0], [0.0]]))
        assert objective.nhvp_calls > 0
        assert objective.foreign_vec_dtypes == []

    def test_converges_without_derivatives(self, bkd):
        objective = QuadraticNoDerivatives(bkd, [1.0, -0.5])
        bounds = bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]])
        optimizer = ScipyTrustConstrOptimizer().bind(objective, bounds)
        result = optimizer.minimize(bkd.asarray([[0.0], [0.0]]))
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[1.0], [-0.5]]), atol=1e-4
        )

    def test_constraint_with_jacobian_and_whvp(self, bkd):
        # minimize sum(x^2) s.t. x0 + x1 >= 1 -> optimum (0.5, 0.5)
        objective = QuadraticWithJacobianAndHVP(bkd, [0.0, 0.0])
        constraint = SumConstraintWithJacobianAndWHVP(
            bkd, 2, 1.0, float("inf")
        )
        bounds = bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]])
        optimizer = ScipyTrustConstrOptimizer(gtol=1e-10).bind(
            objective, bounds, [constraint]
        )
        result = optimizer.minimize(bkd.asarray([[2.0], [2.0]]))
        # barrier method leaves ~1e-5 slack on the active constraint
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[0.5], [0.5]]), atol=1e-4
        )
        assert constraint.njacobian_calls > 0
        assert constraint.nwhvp_calls > 0
        assert constraint.foreign_dtypes == []

    def test_constraint_with_jacobian_only(self, bkd):
        objective = QuadraticWithJacobian(bkd, [0.0, 0.0])
        constraint = SumConstraintWithJacobian(bkd, 2, 1.0, float("inf"))
        bounds = bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]])
        optimizer = ScipyTrustConstrOptimizer().bind(
            objective, bounds, [constraint]
        )
        result = optimizer.minimize(bkd.asarray([[2.0], [2.0]]))
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[0.5], [0.5]]), atol=1e-5
        )

    def test_constraint_without_jacobian(self, bkd):
        # regression: the pre-bundle factory passed a callable returning
        # the string '2-point' when the constraint had no jacobian; scipy
        # expects the string itself
        objective = QuadraticWithJacobian(bkd, [0.0, 0.0])
        constraint = SumConstraint(bkd, 2, 1.0, float("inf"))
        bounds = bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]])
        optimizer = ScipyTrustConstrOptimizer().bind(
            objective, bounds, [constraint]
        )
        result = optimizer.minimize(bkd.asarray([[2.0], [2.0]]))
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[0.5], [0.5]]), atol=1e-4
        )
