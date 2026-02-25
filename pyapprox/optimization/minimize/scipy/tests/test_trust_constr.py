import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

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
from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestScipyTrustConstrOptimizer(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific
        backend.
        """
        raise NotImplementedError("Derived classes must implement this method.")

    def test_optimizer_with_quadratic_objective_and_linear_constraints(
        self,
    ) -> None:
        """
        Test the ScipyTrustConstrOptimizer class with a valid 2D objective and
        constraints.
        """
        bkd = self.bkd()

        # Define constants for the linear-quadratic function
        c = bkd.asarray([1.0, -1.0])  # Linear coefficients

        # Define the linear-quadratic function
        def value_function(x: Array) -> Array:
            return bkd.stack(
                [x[0] ** 2 + x[1] ** 2 + c[0] * x[0] + c[1] * x[1]], axis=1
            )

        # Define the Jacobian of the linear-quadratic function
        def jacobian_function(x: Array) -> Array:
            return bkd.stack([2 * x[0] + c[0], 2 * x[1] + c[1]], axis=1)

        # Define the Hessian-vector product of the linear-quadratic function
        def hvp_function(x: Array, v: Array) -> Array:
            return bkd.stack([2 * v[0], 2 * v[1]], axis=0)

        # Wrap the function using FunctionWithJacobianAndHVPFromCallable
        function = FunctionWithJacobianAndHVPFromCallable(
            nvars=2,
            fun=value_function,
            jacobian=jacobian_function,
            hvp=hvp_function,
            bkd=bkd,
        )

        # Define initial guess
        init_guess = bkd.asarray([[0.5], [0.5]])

        # Check derivatives
        derivative_checker = DerivativeChecker(function)
        errors = derivative_checker.check_derivatives(init_guess)
        self.assertLessEqual(derivative_checker.error_ratio(errors[0]), 1e-6)

        # Define coefficient matrix and constraint vector for the linear constraint
        A = bkd.asarray([[1.0, 1.0]])  # Coefficient matrix
        b = bkd.asarray([1.0])  # Constraint vector

        # Create a generic linear constraint
        linear_constraint = PyApproxLinearConstraint(A, b, b, bkd)

        # Initialize the optimizer
        optimizer = ScipyTrustConstrOptimizer(
            objective=function,
            bounds=bkd.array([[-np.inf, np.inf], [-np.inf, np.inf]]),
            constraints=[linear_constraint],
            verbosity=0,
            maxiter=2,
            gtol=1e-6,
        )

        # Perform optimization
        result = optimizer.minimize(init_guess)

        # Assert that the optimization was successful
        self.assertTrue(result.success())

        # Derive the analytical solution
        expected_optima = bkd.array([0.0, 1.0])[:, None]

        # Assert that the result matches the expected values
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-8)

        # Assert that the objective value matches the expected value
        expected_fun = value_function(expected_optima)
        self.bkd().assert_allclose(result.fun(), float(expected_fun[0, 0]), atol=1e-8)

        # Assert the constraint is satisfied
        self.bkd().assert_allclose(
            A @ result.optima()[:, 0] - b, self._bkd.zeros(b.shape), atol=1e-8
        )

    def test_optimizer_with_evtushenko_objective_and_constraints(self) -> None:
        """
        Test the ScipyTrustConstrOptimizer class with the Evtushenko objective,
        nonlinear constraints, and a linear constraint.
        """
        bkd = self.bkd()

        # Define the Evtushenko objective
        objective = EvtushenkoObjective(backend=bkd)

        # Define the Evtushenko nonlinear constraint
        nonlinear_constraint = EvtushenkoNonLinearConstraint(backend=bkd)

        # Define the linear constraint
        linear_con = PyApproxLinearConstraint(
            bkd.ones((1, 3)), bkd.asarray([1.0]), bkd.asarray([1.0]), bkd
        )

        # Define initial guess
        init_guess = bkd.asarray([[0.1], [0.7], [0.2]])

        objective_derivative_checker = DerivativeChecker(objective)
        errors = objective_derivative_checker.check_derivatives(init_guess, verbosity=0)
        self.assertLessEqual(objective_derivative_checker.error_ratio(errors[0]), 1e-6)

        constraint_derivative_checker = DerivativeChecker(nonlinear_constraint)
        errors = constraint_derivative_checker.check_derivatives(
            init_guess, verbosity=0
        )
        # Use 2e-6 tolerance to account for numerical precision in derivative
        # approximation
        self.assertLessEqual(constraint_derivative_checker.error_ratio(errors[0]), 2e-6)

        # Initialize the optimizer
        optimizer = ScipyTrustConstrOptimizer(
            objective=objective,
            bounds=bkd.array([[0, np.inf], [0, np.inf], [0, np.inf]]),
            constraints=[nonlinear_constraint, linear_con],
            verbosity=0,
            maxiter=100,
            gtol=1e-15,
        )

        # Perform optimization
        result = optimizer.minimize(init_guess)

        # Assert that the optimization was successful
        self.assertTrue(result.success())

        # Assert that the constraints are satisfied
        nonlinear_constraint_value = nonlinear_constraint(result.optima())
        self.assertTrue(
            bkd.all_bool(nonlinear_constraint_value >= nonlinear_constraint.lb())
            and bkd.all_bool(nonlinear_constraint_value <= nonlinear_constraint.ub())
        )

        linear_constraint_value = bkd.ones((1, 3)) @ result.optima()
        self.assertAlmostEqual(linear_constraint_value[0, 0], 1.0, places=8)

        # Assert that the result matches the expected optima
        expected_optima = bkd.array([0.0, 0.0, 1.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=4e-6)

        # Assert that the objective value matches the expected value
        expected_fun = objective(expected_optima)
        self.bkd().assert_allclose(result.fun(), expected_fun.item(), atol=1e-8)

        # Check that function, jacobian and whvp of objective are
        # called.
        self.assertTrue(result.get_raw_result().nfev > 0)
        self.assertTrue(result.get_raw_result().njev > 0)
        self.assertTrue(result.get_raw_result().nhev > 0)

        # Check that function, jacobian and whvp of constraints are
        # called.
        self.assertTrue(any(bkd.asarray(result.get_raw_result().constr_nfev) > 0))
        self.assertTrue(any(bkd.asarray(result.get_raw_result().constr_njev) > 0))
        self.assertTrue(any(bkd.asarray(result.get_raw_result().constr_nhev) > 0))

    def test_deferred_binding(self) -> None:
        """Test optimizer constructed without objective/bounds."""
        bkd = self.bkd()

        # Define a simple quadratic objective
        def value_function(x: Array) -> Array:
            return bkd.stack([x[0] ** 2 + x[1] ** 2], axis=1)

        def jacobian_function(x: Array) -> Array:
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

        # Create optimizer without objective/bounds
        optimizer = ScipyTrustConstrOptimizer(verbosity=0, maxiter=100)
        self.assertFalse(optimizer.is_bound())

        # Should raise RuntimeError if minimizing without binding
        with self.assertRaises(RuntimeError):
            optimizer.minimize(init_guess)

        # Bind and optimize
        optimizer.bind(function, bounds)
        self.assertTrue(optimizer.is_bound())
        result = optimizer.minimize(init_guess)

        # Check result
        self.assertTrue(result.success())
        expected_optima = bkd.array([0.0, 0.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-6)

    def test_rebinding(self) -> None:
        """Test that optimizer can be rebound to different objectives."""
        bkd = self.bkd()

        # First objective: f1(x) = x^2
        def f1(x: Array) -> Array:
            return bkd.stack([x[0] ** 2], axis=1)

        obj1 = FunctionWithJacobianAndHVPFromCallable(
            nvars=1,
            fun=f1,
            jacobian=lambda x: bkd.stack([2 * x[0]], axis=1),
            hvp=lambda x, v: bkd.stack([2 * v[0]], axis=0),
            bkd=bkd,
        )

        # Second objective: f2(x) = (x - 1)^2
        def f2(x: Array) -> Array:
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

        # Create optimizer
        optimizer = ScipyTrustConstrOptimizer(maxiter=100)

        # Bind to first objective
        optimizer.bind(obj1, bounds)
        result1 = optimizer.minimize(init_guess)
        bkd.assert_allclose(result1.optima(), bkd.array([[0.0]]), atol=1e-6)

        # Re-bind to second objective
        optimizer.bind(obj2, bounds)
        result2 = optimizer.minimize(init_guess)
        bkd.assert_allclose(result2.optima(), bkd.array([[1.0]]), atol=1e-5)

    def test_copy(self) -> None:
        """Test copy() returns unbound optimizer with same options."""
        bkd = self.bkd()

        # Create objective
        def value_function(x: Array) -> Array:
            return bkd.stack([x[0] ** 2], axis=1)

        obj = FunctionWithJacobianAndHVPFromCallable(
            nvars=1,
            fun=value_function,
            jacobian=lambda x: bkd.stack([2 * x[0]], axis=1),
            hvp=lambda x, v: bkd.stack([2 * v[0]], axis=0),
            bkd=bkd,
        )

        bounds = bkd.array([[-5.0, 5.0]])

        # Create and bind optimizer
        optimizer = ScipyTrustConstrOptimizer(maxiter=500, gtol=1e-8, verbosity=0)
        optimizer.bind(obj, bounds)
        self.assertTrue(optimizer.is_bound())

        # Copy should be unbound with same options
        copy_opt = optimizer.copy()
        self.assertFalse(copy_opt.is_bound())
        self.assertEqual(copy_opt._maxiter, 500)
        self.assertEqual(copy_opt._gtol, 1e-8)

        # Copy can be bound and used independently
        copy_opt.bind(obj, bounds)
        result = copy_opt.minimize(bkd.asarray([[2.0]]))
        bkd.assert_allclose(result.optima(), bkd.array([[0.0]]), atol=1e-6)

    def test_backward_compatibility(self) -> None:
        """Test existing API with objective/bounds in constructor still works."""
        bkd = self.bkd()

        def value_function(x: Array) -> Array:
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

        # Create optimizer with objective/bounds in constructor (old API)
        optimizer = ScipyTrustConstrOptimizer(
            objective=function, bounds=bounds, maxiter=100
        )

        # Should be bound immediately
        self.assertTrue(optimizer.is_bound())

        # Should work without explicit bind()
        result = optimizer.minimize(init_guess)
        self.assertTrue(result.success())
        expected_optima = bkd.array([0.0, 0.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-6)

    def test_invalid_objective_raises_typeerror(self) -> None:
        """Test that binding an invalid objective raises TypeError."""
        bkd = self.bkd()
        bounds = bkd.array([[-5.0, 5.0]])

        optimizer = ScipyTrustConstrOptimizer(maxiter=100)

        # Plain function is not a valid objective (missing bkd, nvars, nqoi)
        invalid_objective = lambda x: x**2  # noqa: E731

        with self.assertRaises(TypeError) as context:
            optimizer.bind(invalid_objective, bounds)

        # Error message should mention missing methods
        self.assertIn("ObjectiveProtocol", str(context.exception))
        self.assertIn("Missing", str(context.exception))

    def test_invalid_constraint_raises_typeerror(self) -> None:
        """Test that binding an invalid constraint raises TypeError."""
        bkd = self.bkd()

        def value_function(x: Array) -> Array:
            return bkd.stack([x[0] ** 2], axis=1)

        objective = FunctionWithJacobianAndHVPFromCallable(
            nvars=1,
            fun=value_function,
            jacobian=lambda x: bkd.stack([2 * x[0]], axis=1),
            hvp=lambda x, v: bkd.stack([2 * v[0]], axis=0),
            bkd=bkd,
        )

        bounds = bkd.array([[-5.0, 5.0]])

        optimizer = ScipyTrustConstrOptimizer(maxiter=100)

        # Plain function is not a valid constraint (missing lb, ub, etc)
        invalid_constraint = lambda x: x  # noqa: E731

        with self.assertRaises(TypeError) as context:
            optimizer.bind(objective, bounds, constraints=[invalid_constraint])

        # Error message should mention missing methods
        self.assertIn("NonlinearConstraintProtocol", str(context.exception))
        self.assertIn("missing methods", str(context.exception))


class TestScipyTrustConstrOptimizerNumpy(TestScipyTrustConstrOptimizer[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestScipyTrustConstrOptimizerTorch(TestScipyTrustConstrOptimizer[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


# Custom test loader to exclude the base class
def load_tests(loader: unittest.TestLoader, tests, pattern: str) -> unittest.TestSuite:
    """
    Custom test loader to exclude the base class
    ContinuousScipyRandomVariable1D.
    """
    test_suite = unittest.TestSuite()
    for test_class in [
        TestScipyTrustConstrOptimizerNumpy,
        TestScipyTrustConstrOptimizerTorch,
    ]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


# Main block to explicitly run tests using the custom loader
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
