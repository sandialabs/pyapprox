import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)
from pyapprox.optimization.minimize.scipy.slsqp import (
    ScipySLSQPOptimizer,
)
from pyapprox.optimization.minimize.benchmarks.evutushenko import (
    EvtushenkoObjective,
    EvtushenkoNonLinearConstraint,
)


class TestScipySLSQPOptimizer(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_optimizer_with_quadratic_objective_and_linear_constraints(
        self,
    ) -> None:
        """Test SLSQP with a valid 2D objective and linear equality constraint."""
        bkd = self.bkd()

        c = bkd.asarray([1.0, -1.0])

        def value_function(x: Array) -> Array:
            return bkd.stack(
                [x[0] ** 2 + x[1] ** 2 + c[0] * x[0] + c[1] * x[1]], axis=1
            )

        def jacobian_function(x: Array) -> Array:
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

        self.assertTrue(result.success())

        expected_optima = bkd.array([0.0, 1.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-8)

        expected_fun = value_function(expected_optima)
        bkd.assert_allclose(
            result.fun(), float(expected_fun[0, 0]), atol=1e-8
        )

        bkd.assert_allclose(
            A @ result.optima()[:, 0] - b, bkd.zeros(b.shape), atol=1e-8
        )

    def test_optimizer_with_evtushenko_objective_and_constraints(self) -> None:
        """Test SLSQP with the Evtushenko objective and nonlinear+linear
        constraints."""
        bkd = self.bkd()

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

        self.assertTrue(result.success())

        nonlinear_constraint_value = nonlinear_constraint(result.optima())
        self.assertTrue(
            bkd.all_bool(
                nonlinear_constraint_value >= nonlinear_constraint.lb()
            )
            and bkd.all_bool(
                nonlinear_constraint_value <= nonlinear_constraint.ub()
            )
        )

        linear_constraint_value = bkd.ones((1, 3)) @ result.optima()
        bkd.assert_allclose(
            linear_constraint_value, bkd.ones((1, 1)), atol=1e-8
        )

        expected_optima = bkd.array([0.0, 0.0, 1.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=4e-6)

        expected_fun = objective(expected_optima)
        bkd.assert_allclose(
            result.fun(), expected_fun.item(), atol=1e-8
        )

    def test_deferred_binding(self) -> None:
        """Test optimizer constructed without objective/bounds."""
        bkd = self.bkd()

        def value_function(x: Array) -> Array:
            return bkd.stack([x[0] ** 2 + x[1] ** 2], axis=1)

        def jacobian_function(x: Array) -> Array:
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
        self.assertFalse(optimizer.is_bound())

        with self.assertRaises(RuntimeError):
            optimizer.minimize(init_guess)

        optimizer.bind(function, bounds)
        self.assertTrue(optimizer.is_bound())
        result = optimizer.minimize(init_guess)

        self.assertTrue(result.success())
        expected_optima = bkd.array([0.0, 0.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-6)

    def test_rebinding(self) -> None:
        """Test that optimizer can be rebound to different objectives."""
        bkd = self.bkd()

        def f1(x: Array) -> Array:
            return bkd.stack([x[0] ** 2], axis=1)

        obj1 = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=1,
            fun=f1,
            jacobian=lambda x: bkd.stack([2 * x[0]], axis=1),
            bkd=bkd,
        )

        def f2(x: Array) -> Array:
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

    def test_copy(self) -> None:
        """Test copy() returns unbound optimizer with same options."""
        bkd = self.bkd()

        def value_function(x: Array) -> Array:
            return bkd.stack([x[0] ** 2], axis=1)

        obj = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=1,
            fun=value_function,
            jacobian=lambda x: bkd.stack([2 * x[0]], axis=1),
            bkd=bkd,
        )

        bounds = bkd.array([[-5.0, 5.0]])

        optimizer = ScipySLSQPOptimizer(
            maxiter=500, ftol=1e-8, disp=False
        )
        optimizer.bind(obj, bounds)
        self.assertTrue(optimizer.is_bound())

        copy_opt = optimizer.copy()
        self.assertFalse(copy_opt.is_bound())
        self.assertEqual(copy_opt._maxiter, 500)
        self.assertEqual(copy_opt._ftol, 1e-8)

        copy_opt.bind(obj, bounds)
        result = copy_opt.minimize(bkd.asarray([[2.0]]))
        bkd.assert_allclose(result.optima(), bkd.array([[0.0]]), atol=1e-6)

    def test_backward_compatibility(self) -> None:
        """Test existing API with objective/bounds in constructor."""
        bkd = self.bkd()

        def value_function(x: Array) -> Array:
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

        optimizer = ScipySLSQPOptimizer(
            objective=function, bounds=bounds, maxiter=100
        )

        self.assertTrue(optimizer.is_bound())

        result = optimizer.minimize(init_guess)
        self.assertTrue(result.success())
        expected_optima = bkd.array([0.0, 0.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-6)

    def test_inequality_constraints(self) -> None:
        """Test SLSQP with inequality constraints (not equality)."""
        bkd = self.bkd()

        # Minimize x^2 + y^2 subject to x + y >= 1
        def value_function(x: Array) -> Array:
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
        self.assertTrue(result.success())

        # Optimal is x=y=0.5 (on the constraint boundary)
        expected_optima = bkd.array([0.5, 0.5])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-6)

    def test_invalid_objective_raises_typeerror(self) -> None:
        """Test that binding an invalid objective raises TypeError."""
        bkd = self.bkd()
        bounds = bkd.array([[-5.0, 5.0]])

        optimizer = ScipySLSQPOptimizer(maxiter=100)

        invalid_objective = lambda x: x**2  # noqa: E731

        with self.assertRaises(TypeError) as context:
            optimizer.bind(invalid_objective, bounds)

        self.assertIn("ObjectiveProtocol", str(context.exception))
        self.assertIn("Missing", str(context.exception))

    def test_invalid_constraint_raises_typeerror(self) -> None:
        """Test that binding an invalid constraint raises TypeError."""
        bkd = self.bkd()

        def value_function(x: Array) -> Array:
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

        with self.assertRaises(TypeError) as context:
            optimizer.bind(objective, bounds, constraints=[invalid_constraint])

        self.assertIn("NonlinearConstraintProtocol", str(context.exception))
        self.assertIn("missing methods", str(context.exception))


class TestScipySLSQPOptimizerNumpy(
    TestScipySLSQPOptimizer[NDArray[Any]]
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestScipySLSQPOptimizerTorch(
    TestScipySLSQPOptimizer[torch.Tensor]
):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


def load_tests(
    loader: unittest.TestLoader, tests, pattern: str
) -> unittest.TestSuite:
    test_suite = unittest.TestSuite()
    for test_class in [
        TestScipySLSQPOptimizerNumpy,
        TestScipySLSQPOptimizerTorch,
    ]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
