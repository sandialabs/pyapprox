import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.optimization.implicitfunction.benchmarks.wildeys_nonlinear_state_equation import (  # noqa: E501
    NonLinearCoupledStateEquations,
)
from pyapprox.optimization.implicitfunction.functionals.subset_of_states import (
    SubsetOfStatesAdjointFunctional,
)
from pyapprox.optimization.implicitfunction.operator.check_derivatives import (
    ImplicitFunctionDerivativeChecker,
)
from pyapprox.optimization.implicitfunction.operator.sensitivities import (
    VectorAdjointOperatorWithJacobian,
)
from pyapprox.optimization.rootfinding.newton import NewtonSolverOptions
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestSensitivities(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific
        backend.
        """
        raise NotImplementedError("Derived classes must implement this method.")

    def test_nonlinear_coupled_residual_vector_functional(self):
        """
        Test sensitivities for nonlinear coupled residual equations with a vector
        functional.
        """
        bkd = self.bkd()

        # Create state equation
        state_eq = NonLinearCoupledStateEquations(
            self.bkd(), NewtonSolverOptions(rtol=1e-10, atol=1e-10)
        )

        # Define parameters and initial state
        param = bkd.array([0.8, 1.1])[:, None]
        init_state = bkd.array([-1.0, -1.0])[:, None]

        # Create functional
        functional = SubsetOfStatesAdjointFunctional(
            state_eq.nstates(), state_eq.nparams(), bkd.arange(2), bkd
        )

        # Create adjoint operator
        adjoint_op = VectorAdjointOperatorWithJacobian(state_eq, functional)

        # Check derivatives
        derivative_checker = ImplicitFunctionDerivativeChecker(adjoint_op)
        tols = derivative_checker.get_derivative_tolerances(1e-6)

        # Reduce finite difference step sizes for Newton convergence
        fd_eps = self.bkd().flip(self.bkd().logspace(-13, -1, 12))
        derivative_checker.check_derivatives(
            init_state, param, tols, fd_eps=fd_eps, verbosity=0
        )


# Derived test class for NumPy backend
class TestSensitivitiesNumpy(TestSensitivities[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestSensitivitiesTorch(TestSensitivities[torch.Tensor]):
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
        TestSensitivitiesNumpy,
        TestSensitivitiesTorch,
    ]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


# Main block to explicitly run tests using the custom loader
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
