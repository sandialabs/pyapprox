import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.util.numpy import NumpyBkd
from pyapprox.typing.util.abstracttestcase import AbstractTestCase
from pyapprox.typing.interface.functions.hessian import (
    FunctionWithJacobianAndHVPFromCallable,
)
from pyapprox.typing.optimization.linear_constraint import (
    PyApproxLinearConstraint,
)
from pyapprox.typing.optimization.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.typing.interface.derivative_checks.derivative_checker import (
    DerivativeChecker,
)


class TestScipyTrustConstrOptimizer(Generic[Array], AbstractTestCase):
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
            return bkd.asarray(
                [x[0] ** 2 + x[1] ** 2 + c[0] * x[0] + c[1] * x[1]]
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
        self.assertTrue(
            derivative_checker.error_ratios_satisfied(errors[0], 1e-7)
        )

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
        print(result.get_raw_result())

        # Assert that the optimization was successful
        self.assertTrue(result.success())

        # Derive the analytical solution
        expected_optima = bkd.array([0.0, 1.0])[:, None]

        # Assert that the result matches the expected values
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-8)

        # Assert that the objective value matches the expected value
        expected_fun = value_function(expected_optima)
        self.assertAlmostEqual(result.fun(), expected_fun, places=6)

        # Assert the constraint is satisfied
        self.assertAlmostEqual(A @ result.optima() - b, 0.0, places=8)


class TestScipyTrustConstrOptimizerNumpy(
    TestScipyTrustConstrOptimizer[NDArray[Any]], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
