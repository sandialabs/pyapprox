import numpy as np
import pytest

from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.optimization.minimize.benchmarks.evutushenko import (
    EvtushenkoNonLinearConstraint,
    EvtushenkoObjective,
)
from pyapprox.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)
from pyapprox.optimization.minimize.scipy.slsqp import (
    ScipySLSQPOptimizer,
)


class TestScipySLSQPOptimizer:

    def test_optimizer_with_quadratic_objective_and_linear_constraints(
        self, bkd,
    ) -> None:
        """Test SLSQP with a valid 2D objective and linear equality constraint."""
        c = bkd.asarray([1.0, -1.0])

        def value_function(x):
            return bkd.stack(
                [x[0] ** 2 + x[1] ** 2 + c[0] * x[0] + c[1] * x[1]], axis=1
            )

        def jacobian_function(x):
            return bkd.stack([2 * x[0] + c[0], 2 * x[1] + c[1]], axis=1)

        function = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=2,
            fun=value_function,
            jacobian=jacobian_function,
            bkd=bkd,
        )

        init_guess = bkd.asarray([[0.5], [0.5]])

        A = bkd.asarray([[1.0, 1.0]])
        b = bkd.asarray([1.0])

        linear_constraint = PyApproxLinearConstraint(A, b, b, bkd)

        optimizer = ScipySLSQPOptimizer(
            objective=function,
            bounds=bkd.array([[-np.inf, np.inf], [-np.inf, np.inf]]),
            constraints=[linear_constraint],
            maxiter=100,
        )

        result = optimizer.minimize(init_guess)

        assert result.success()

        expected_optima = bkd.array([0.0, 1.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-8)

        expected_fun = value_function(expected_optima)
        bkd.assert_allclose(result.fun(), float(expected_fun[0, 0]), atol=1e-8)

        bkd.assert_allclose(
            A @ result.optima()[:, 0] - b, bkd.zeros(b.shape), atol=1e-8
        )

    def test_optimizer_with_evtushenko_objective_and_constraints(self, bkd) -> None:
        """Test SLSQP with the Evtushenko objective and nonlinear+linear
        constraints."""
        objective = EvtushenkoObjective(backend=bkd)
        nonlinear_constraint = EvtushenkoNonLinearConstraint(backend=bkd)

        linear_con = PyApproxLinearConstraint(
            bkd.ones((1, 3)), bkd.asarray([1.0]), bkd.asarray([1.0]), bkd
        )

        init_guess = bkd.asarray([[0.1], [0.7], [0.2]])

        optimizer = ScipySLSQPOptimizer(
            objective=objective,
            bounds=bkd.array([[0, np.inf], [0, np.inf], [0, np.inf]]),
            constraints=[nonlinear_constraint, linear_con],
            maxiter=200,
        )

        result = optimizer.minimize(init_guess)

        assert result.success()

        nonlinear_constraint_value = nonlinear_constraint(result.optima())
        assert (
            bkd.all_bool(nonlinear_constraint_value >= nonlinear_constraint.lb())
            and bkd.all_bool(nonlinear_constraint_value <= nonlinear_constraint.ub())
        )

        linear_constraint_value = bkd.ones((1, 3)) @ result.optima()
        bkd.assert_allclose(linear_constraint_value, bkd.ones((1, 1)), atol=1e-8)

        expected_optima = bkd.array([0.0, 0.0, 1.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=4e-6)

        expected_fun = objective(expected_optima)
        bkd.assert_allclose(result.fun(), expected_fun.item(), atol=1e-8)

    def test_deferred_binding(self, bkd) -> None:
        """Test optimizer constructed without objective/bounds."""
        def value_function(x):
            return bkd.stack([x[0] ** 2 + x[1] ** 2], axis=1)

        def jacobian_function(x):
            return bkd.stack([2 * x[0], 2 * x[1]], axis=1)

        function = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=2,
            fun=value_function,
            jacobian=jacobian_function,
            bkd=bkd,
        )

        bounds = bkd.array([[-5.0, 5.0], [-5.0, 5.0]])
        init_guess = bkd.asarray([[1.0], [1.0]])

        optimizer = ScipySLSQPOptimizer(maxiter=100)
        assert not optimizer.is_bound()

        with pytest.raises(RuntimeError):
            optimizer.minimize(init_guess)

        optimizer.bind(function, bounds)
        assert optimizer.is_bound()
        result = optimizer.minimize(init_guess)

        assert result.success()
        expected_optima = bkd.array([0.0, 0.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-6)

    def test_rebinding(self, bkd) -> None:
        """Test that optimizer can be rebound to different objectives."""
        def f1(x):
            return bkd.stack([x[0] ** 2], axis=1)

        obj1 = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=1,
            fun=f1,
            jacobian=lambda x: bkd.stack([2 * x[0]], axis=1),
            bkd=bkd,
        )

        def f2(x):
            return bkd.stack([(x[0] - 1) ** 2], axis=1)

        obj2 = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=1,
            fun=f2,
            jacobian=lambda x: bkd.stack([2 * (x[0] - 1)], axis=1),
            bkd=bkd,
        )

        bounds = bkd.array([[-5.0, 5.0]])
        init_guess = bkd.asarray([[2.0]])

        optimizer = ScipySLSQPOptimizer(maxiter=100)

        optimizer.bind(obj1, bounds)
        result1 = optimizer.minimize(init_guess)
        bkd.assert_allclose(result1.optima(), bkd.array([[0.0]]), atol=1e-6)

        optimizer.bind(obj2, bounds)
        result2 = optimizer.minimize(init_guess)
        bkd.assert_allclose(result2.optima(), bkd.array([[1.0]]), atol=1e-5)

    def test_copy(self, bkd) -> None:
        """Test copy() returns unbound optimizer with same options."""
        def value_function(x):
            return bkd.stack([x[0] ** 2], axis=1)

        obj = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=1,
            fun=value_function,
            jacobian=lambda x: bkd.stack([2 * x[0]], axis=1),
            bkd=bkd,
        )

        bounds = bkd.array([[-5.0, 5.0]])

        optimizer = ScipySLSQPOptimizer(maxiter=500, ftol=1e-8, disp=False)
        optimizer.bind(obj, bounds)
        assert optimizer.is_bound()

        copy_opt = optimizer.copy()
        assert not copy_opt.is_bound()
        assert copy_opt._maxiter == 500
        assert copy_opt._ftol == 1e-8

        copy_opt.bind(obj, bounds)
        result = copy_opt.minimize(bkd.asarray([[2.0]]))
        bkd.assert_allclose(result.optima(), bkd.array([[0.0]]), atol=1e-6)

    def test_backward_compatibility(self, bkd) -> None:
        """Test existing API with objective/bounds in constructor."""
        def value_function(x):
            return bkd.stack([x[0] ** 2 + x[1] ** 2], axis=1)

        function = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=2,
            fun=value_function,
            jacobian=lambda x: bkd.stack([2 * x[0], 2 * x[1]], axis=1),
            bkd=bkd,
        )

        bounds = bkd.array([[-5.0, 5.0], [-5.0, 5.0]])
        init_guess = bkd.asarray([[1.0], [1.0]])

        optimizer = ScipySLSQPOptimizer(objective=function, bounds=bounds, maxiter=100)

        assert optimizer.is_bound()

        result = optimizer.minimize(init_guess)
        assert result.success()
        expected_optima = bkd.array([0.0, 0.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-6)

    def test_inequality_constraints(self, bkd) -> None:
        """Test SLSQP with inequality constraints (not equality)."""
        # Minimize x^2 + y^2 subject to x + y >= 1
        def value_function(x):
            return bkd.stack([x[0] ** 2 + x[1] ** 2], axis=1)

        function = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=2,
            fun=value_function,
            jacobian=lambda x: bkd.stack([2 * x[0], 2 * x[1]], axis=1),
            bkd=bkd,
        )

        # Linear inequality constraint: x + y >= 1 (lb=1, ub=inf)
        A = bkd.asarray([[1.0, 1.0]])
        lb = bkd.asarray([1.0])
        ub = bkd.asarray([np.inf])
        ineq_constraint = PyApproxLinearConstraint(A, lb, ub, bkd)

        init_guess = bkd.asarray([[1.0], [1.0]])

        optimizer = ScipySLSQPOptimizer(
            objective=function,
            bounds=bkd.array([[-np.inf, np.inf], [-np.inf, np.inf]]),
            constraints=[ineq_constraint],
            maxiter=100,
        )

        result = optimizer.minimize(init_guess)
        assert result.success()

        # Optimal is x=y=0.5 (on the constraint boundary)
        expected_optima = bkd.array([0.5, 0.5])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-6)

    def test_invalid_objective_raises_typeerror(self, bkd) -> None:
        """Test that binding an invalid objective raises TypeError."""
        bounds = bkd.array([[-5.0, 5.0]])

        optimizer = ScipySLSQPOptimizer(maxiter=100)

        invalid_objective = lambda x: x**2  # noqa: E731

        with pytest.raises(TypeError) as context:
            optimizer.bind(invalid_objective, bounds)

        assert "ObjectiveProtocol" in str(context.value)
        assert "Missing" in str(context.value)

    def test_invalid_constraint_raises_typeerror(self, bkd) -> None:
        """Test that binding an invalid constraint raises TypeError."""
        def value_function(x):
            return bkd.stack([x[0] ** 2], axis=1)

        objective = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=1,
            fun=value_function,
            jacobian=lambda x: bkd.stack([2 * x[0]], axis=1),
            bkd=bkd,
        )

        bounds = bkd.array([[-5.0, 5.0]])

        optimizer = ScipySLSQPOptimizer(maxiter=100)

        invalid_constraint = lambda x: x  # noqa: E731

        with pytest.raises(TypeError) as context:
            optimizer.bind(objective, bounds, constraints=[invalid_constraint])

        assert "NonlinearConstraintProtocol" in str(context.value)
        assert "missing methods" in str(context.value)
