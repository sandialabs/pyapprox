import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianAndHVPFromCallable,
)
from pyapprox.typing.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)
from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.typing.optimization.minimize.benchmarks.evutushenko import (
    EvtushenkoObjective,
    EvtushenkoNonLinearConstraint,
)


class TestScipyTrustConstrOptimizer(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific
        backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

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
        self.assertLessEqual(derivative_checker.error_ratio(errors[0]), 1e-7)

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
        self.bkd().assert_allclose(
            result.fun(), float(expected_fun[0, 0]), atol=1e-8
        )

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
        errors = objective_derivative_checker.check_derivatives(
            init_guess, verbosity=0
        )
        self.assertLessEqual(
            objective_derivative_checker.error_ratio(errors[0]), 1e-7
        )

        constraint_derivative_checker = DerivativeChecker(nonlinear_constraint)
        errors = constraint_derivative_checker.check_derivatives(
            init_guess, verbosity=0
        )
        self.assertLessEqual(
            constraint_derivative_checker.error_ratio(errors[0]), 1e-6
        )

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
            bkd.all_bool(
                nonlinear_constraint_value >= nonlinear_constraint.lb()
            )
            and bkd.all_bool(
                nonlinear_constraint_value <= nonlinear_constraint.ub()
            )
        )

        linear_constraint_value = bkd.ones((1, 3)) @ result.optima()
        self.assertAlmostEqual(linear_constraint_value[0, 0], 1.0, places=8)

        # Assert that the result matches the expected optima
        expected_optima = bkd.array([0.0, 0.0, 1.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=4e-6)

        # Assert that the objective value matches the expected value
        expected_fun = objective(expected_optima)
        self.bkd().assert_allclose(
            result.fun(), expected_fun.item(), atol=1e-8
        )

        # Check that function, jacobian and whvp of objective are
        # called.
        self.assertTrue(result.get_raw_result().nfev > 0)
        self.assertTrue(result.get_raw_result().njev > 0)
        self.assertTrue(result.get_raw_result().nhev > 0)

        # Check that function, jacobian and whvp of constraints are
        # called.
        self.assertTrue(
            any(bkd.asarray(result.get_raw_result().constr_nfev) > 0)
        )
        self.assertTrue(
            any(bkd.asarray(result.get_raw_result().constr_njev) > 0)
        )
        self.assertTrue(
            any(bkd.asarray(result.get_raw_result().constr_nhev) > 0)
        )


class TestScipyTrustConstrOptimizerNumpy(
    TestScipyTrustConstrOptimizer[NDArray[Any]]
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestScipyTrustConstrOptimizerTorch(
    TestScipyTrustConstrOptimizer[torch.Tensor]
):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


# Custom test loader to exclude the base class
def load_tests(
    loader: unittest.TestLoader, tests, pattern: str
) -> unittest.TestSuite:
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
