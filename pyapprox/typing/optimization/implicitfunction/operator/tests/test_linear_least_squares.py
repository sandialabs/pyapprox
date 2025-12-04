import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.optimization.implicitfunction.benchmarks.linear_state_equation import (
    LinearStateEquation,
)
from pyapprox.typing.optimization.implicitfunction.functionals.mean_squared_error import (
    MSEFunctional,
)
from pyapprox.typing.optimization.implicitfunction.functionals.tikhonov_mean_squared_error import (
    TikhonovMSEFunctional,
)
from pyapprox.typing.optimization.implicitfunction.operator.operator_with_hvp import (
    AdjointOperatorWithJacobianAndHVP,
)
from pyapprox.typing.optimization.implicitfunction.operator.check_derivatives import (
    ImplicitFunctionDerivativeChecker,
)


class TestLinearLeastSquares(Generic[Array], unittest.TestCase):
    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific
        backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_linear_least_squares(self) -> None:
        """
        Test the value and derivative computation for linear least squares problems.
        """
        # Problem setup
        np.random.seed(1)
        nstates, nparams = 3, 2
        bkd = self.bkd()
        Amat = bkd.asarray(np.random.normal(0, 1, (nstates, nparams)))
        param = bkd.asarray(np.random.normal(0, 1, (nparams, 1)))
        obs = Amat @ param
        init_state = bkd.full((nstates, 1), 1.0)

        # Create state equation and functional
        state_eq = LinearStateEquation(Amat, obs, bkd)
        functional = MSEFunctional(nstates, nparams, bkd)
        functional.set_observations(obs)

        # Create adjoint operator
        adjoint_operator = AdjointOperatorWithJacobianAndHVP(
            state_eq, functional
        )

        # Test value computation
        value = adjoint_operator(init_state, param)
        state = state_eq.solve(init_state, param)
        expected_value = functional(state, param)
        bkd.assert_allclose(value, expected_value)

        # Test derivative computation
        derivative_checker = ImplicitFunctionDerivativeChecker(
            adjoint_operator
        )
        tols = derivative_checker.get_derivative_tolerances(1e-7)
        derivative_checker.check_derivatives(
            init_state, param, tols, verbosity=0
        )

    def test_tikhonov_linear_least_squares(self) -> None:
        """
        Test Tikhonov regularization for linear least squares problems.
        """
        # Problem setup
        np.random.seed(1)
        nobs, nvars = 3, 2
        bkd = self.bkd()
        Amat = bkd.asarray(np.random.randn(nobs, nvars))  # Random matrix
        param = bkd.asarray(np.random.randn(nvars, 1))  # Random parameters
        obs = Amat @ param  # Observations
        init_state = bkd.zeros((nobs, 1))  # Initial state

        # Create state equation and functional
        state_eq = LinearStateEquation(Amat, obs, bkd)
        functional = TikhonovMSEFunctional(nobs, nvars, bkd)
        functional.set_observations(obs)

        # Create adjoint operator
        adjoint_operator = AdjointOperatorWithJacobianAndHVP(
            state_eq, functional
        )

        # Test derivative computation
        derivative_checker = ImplicitFunctionDerivativeChecker(
            adjoint_operator
        )
        tols = derivative_checker.get_derivative_tolerances(1e-8)
        tols[[2, 3, 4]] = 2.0e-7
        derivative_checker.check_derivatives(
            init_state, param, tols, verbosity=0
        )


# Derived test class for NumPy backend
class TestLinearLeastSquaresNumpy(TestLinearLeastSquares[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestLinearLeastSquaresTorch(TestLinearLeastSquares[torch.Tensor]):
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
        TestLinearLeastSquaresNumpy,
        TestLinearLeastSquaresTorch,
    ]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


# Main block to explicitly run tests using the custom loader
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
