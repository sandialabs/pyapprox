import numpy as np
import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("pyrol"):
    pytest.skip("pyrol not installed", allow_module_level=True)

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianAndHVPFromCallable,
)
from pyapprox.optimization.minimize.benchmarks.evutushenko import (
    EvtushenkoNonLinearConstraint,
    EvtushenkoObjective,
)
from pyapprox.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)
from pyapprox.optimization.minimize.rol.rol_optimizer import (
    ROLOptimizer,
)


class TestROLOptimizer:

    def test_optimizer_with_quadratic_objective_and_linear_constraints(
        self, bkd,
    ) -> None:
        c = bkd.asarray([1.0, -1.0])

        def value_function(x):
            return bkd.stack(
                [x[0] ** 2 + x[1] ** 2 + c[0] * x[0] + c[1] * x[1]], axis=1
            )

        def jacobian_function(x):
            return bkd.stack([2 * x[0] + c[0], 2 * x[1] + c[1]], axis=1)

        def hvp_function(x, v):
            return bkd.stack([2 * v[0], 2 * v[1]], axis=0)

        function = FunctionWithJacobianAndHVPFromCallable(
            nvars=2,
            fun=value_function,
            jacobian=jacobian_function,
            hvp=hvp_function,
            bkd=bkd,
        )

        init_guess = bkd.asarray([[0.5], [0.5]])

        derivative_checker = DerivativeChecker(function)
        errors = derivative_checker.check_derivatives(init_guess)
        assert derivative_checker.error_ratio(errors[0]) <= 1e-6

        A = bkd.asarray([[1.0, 1.0]])
        b = bkd.asarray([1.0])

        linear_constraint = PyApproxLinearConstraint(A, b, b, bkd)

        optimizer = ROLOptimizer(
            objective=function,
            bounds=bkd.array([[-np.inf, np.inf], [-np.inf, np.inf]]),
            constraints=[linear_constraint],
            verbosity=0,
        )

        result = optimizer.minimize(init_guess)

        assert result.success()

        expected_optima = bkd.array([0.0, 1.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-6)

        expected_fun = value_function(expected_optima)
        bkd.assert_allclose(result.fun(), float(expected_fun[0, 0]), atol=1e-6)

        bkd.assert_allclose(
            A @ result.optima()[:, 0] - b, bkd.zeros(b.shape), atol=1e-6
        )

    def test_optimizer_with_evtushenko_objective_and_constraints(
        self, bkd,
    ) -> None:
        objective = EvtushenkoObjective(backend=bkd)
        nonlinear_constraint = EvtushenkoNonLinearConstraint(backend=bkd)

        linear_con = PyApproxLinearConstraint(
            bkd.ones((1, 3)), bkd.asarray([1.0]), bkd.asarray([1.0]), bkd
        )

        init_guess = bkd.asarray([[0.1], [0.7], [0.2]])

        objective_derivative_checker = DerivativeChecker(objective)
        errors = objective_derivative_checker.check_derivatives(
            init_guess, verbosity=0,
        )
        assert objective_derivative_checker.error_ratio(errors[0]) <= 1e-6

        constraint_derivative_checker = DerivativeChecker(nonlinear_constraint)
        errors = constraint_derivative_checker.check_derivatives(
            init_guess, verbosity=0,
        )
        assert constraint_derivative_checker.error_ratio(errors[0]) <= 2e-6

        optimizer = ROLOptimizer(
            objective=objective,
            bounds=bkd.array([[0, np.inf], [0, np.inf], [0, np.inf]]),
            constraints=[nonlinear_constraint, linear_con],
            verbosity=0,
        )

        result = optimizer.minimize(init_guess)

        assert result.success()

        nonlinear_constraint_value = nonlinear_constraint(result.optima())
        assert bkd.all_bool(
            nonlinear_constraint_value >= nonlinear_constraint.lb()
        ) and bkd.all_bool(
            nonlinear_constraint_value <= nonlinear_constraint.ub()
        )

        linear_constraint_value = bkd.ones((1, 3)) @ result.optima()
        assert linear_constraint_value[0, 0] == pytest.approx(1.0, abs=1e-6)

        expected_optima = bkd.array([0.0, 0.0, 1.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=4e-4)

        expected_fun = objective(expected_optima)
        bkd.assert_allclose(result.fun(), expected_fun.item(), atol=1e-4)

    def test_deferred_binding(self, bkd) -> None:
        def value_function(x):
            return bkd.stack([x[0] ** 2 + x[1] ** 2], axis=1)

        def jacobian_function(x):
            return bkd.stack([2 * x[0], 2 * x[1]], axis=1)

        function = FunctionWithJacobianAndHVPFromCallable(
            nvars=2,
            fun=value_function,
            jacobian=jacobian_function,
            hvp=lambda x, v: bkd.stack([2 * v[0], 2 * v[1]], axis=0),
            bkd=bkd,
        )

        bounds = bkd.array([[-5.0, 5.0], [-5.0, 5.0]])
        init_guess = bkd.asarray([[1.0], [1.0]])

        optimizer = ROLOptimizer(verbosity=0)
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
        def f1(x):
            return bkd.stack([x[0] ** 2], axis=1)

        obj1 = FunctionWithJacobianAndHVPFromCallable(
            nvars=1,
            fun=f1,
            jacobian=lambda x: bkd.stack([2 * x[0]], axis=1),
            hvp=lambda x, v: bkd.stack([2 * v[0]], axis=0),
            bkd=bkd,
        )

        def f2(x):
            return bkd.stack([(x[0] - 1) ** 2], axis=1)

        obj2 = FunctionWithJacobianAndHVPFromCallable(
            nvars=1,
            fun=f2,
            jacobian=lambda x: bkd.stack([2 * (x[0] - 1)], axis=1),
            hvp=lambda x, v: bkd.stack([2 * v[0]], axis=0),
            bkd=bkd,
        )

        bounds = bkd.array([[-5.0, 5.0]])
        init_guess = bkd.asarray([[2.0]])

        optimizer = ROLOptimizer(verbosity=0)

        optimizer.bind(obj1, bounds)
        result1 = optimizer.minimize(init_guess)
        bkd.assert_allclose(result1.optima(), bkd.array([[0.0]]), atol=1e-6)

        optimizer.bind(obj2, bounds)
        result2 = optimizer.minimize(init_guess)
        bkd.assert_allclose(result2.optima(), bkd.array([[1.0]]), atol=1e-5)

    def test_copy(self, bkd) -> None:
        def value_function(x):
            return bkd.stack([x[0] ** 2], axis=1)

        obj = FunctionWithJacobianAndHVPFromCallable(
            nvars=1,
            fun=value_function,
            jacobian=lambda x: bkd.stack([2 * x[0]], axis=1),
            hvp=lambda x, v: bkd.stack([2 * v[0]], axis=0),
            bkd=bkd,
        )

        bounds = bkd.array([[-5.0, 5.0]])

        optimizer = ROLOptimizer(verbosity=0)
        optimizer.bind(obj, bounds)
        assert optimizer.is_bound()

        copy_opt = optimizer.copy()
        assert not copy_opt.is_bound()
        assert copy_opt._verbosity == 0

        copy_opt.bind(obj, bounds)
        result = copy_opt.minimize(bkd.asarray([[2.0]]))
        bkd.assert_allclose(result.optima(), bkd.array([[0.0]]), atol=1e-6)

    def test_backward_compatibility(self, bkd) -> None:
        def value_function(x):
            return bkd.stack([x[0] ** 2 + x[1] ** 2], axis=1)

        function = FunctionWithJacobianAndHVPFromCallable(
            nvars=2,
            fun=value_function,
            jacobian=lambda x: bkd.stack([2 * x[0], 2 * x[1]], axis=1),
            hvp=lambda x, v: bkd.stack([2 * v[0], 2 * v[1]], axis=0),
            bkd=bkd,
        )

        bounds = bkd.array([[-5.0, 5.0], [-5.0, 5.0]])
        init_guess = bkd.asarray([[1.0], [1.0]])

        optimizer = ROLOptimizer(objective=function, bounds=bounds, verbosity=0)

        assert optimizer.is_bound()

        result = optimizer.minimize(init_guess)
        assert result.success()
        expected_optima = bkd.array([0.0, 0.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-6)

    def test_invalid_objective_raises_typeerror(self, bkd) -> None:
        bounds = bkd.array([[-5.0, 5.0]])

        optimizer = ROLOptimizer(verbosity=0)

        invalid_objective = lambda x: x**2  # noqa: E731

        with pytest.raises(TypeError) as context:
            optimizer.bind(invalid_objective, bounds)

        assert "ObjectiveProtocol" in str(context.value)
        assert "Missing" in str(context.value)

    def test_invalid_constraint_raises_typeerror(self, bkd) -> None:
        def value_function(x):
            return bkd.stack([x[0] ** 2], axis=1)

        objective = FunctionWithJacobianAndHVPFromCallable(
            nvars=1,
            fun=value_function,
            jacobian=lambda x: bkd.stack([2 * x[0]], axis=1),
            hvp=lambda x, v: bkd.stack([2 * v[0]], axis=0),
            bkd=bkd,
        )

        bounds = bkd.array([[-5.0, 5.0]])

        optimizer = ROLOptimizer(verbosity=0)

        invalid_constraint = lambda x: x  # noqa: E731

        with pytest.raises(TypeError) as context:
            optimizer.bind(objective, bounds, constraints=[invalid_constraint])

        assert "NonlinearConstraintProtocol" in str(context.value)
        assert "missing methods" in str(context.value)
