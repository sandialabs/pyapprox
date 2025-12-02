from typing import Generic, Any
import unittest
from pyapprox.util.print_wrapper import *
import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.util.numpy import NumpyBkd
from pyapprox.typing.util.torch import TorchBkd
from pyapprox.typing.optimization.implicitfunction.benchmarks.wildeys_nonlinear_state_equation import (
    NonLinearCoupledStateEquations,
)
from pyapprox.typing.optimization.implicitfunction.functionals.weighted_sum import (
    WeightedSumFunctional,
)
from pyapprox.typing.optimization.implicitfunction.operator.operator_with_hvp import (
    AdjointOperatorWithJacobianAndHVP,
)
from pyapprox.typing.optimization.implicitfunction.operator.check_derivatives import (
    ImplicitFunctionDerivativeChecker,
)
from pyapprox.typing.util.abstracttestcase import AbstractTestCase
from pyapprox.typing.optimization.rootfinding.newton import NewtonSolverOptions


class TestNonLinearCoupledEquations(Generic[Array], AbstractTestCase):
    def setUp(self) -> None:
        np.random.seed(1)

    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific
        backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_state_equation_solution(self) -> None:
        """
        Test the solution of the nonlinear coupled equations residual.
        """
        state_eq = NonLinearCoupledStateEquations(
            self.bkd(), NewtonSolverOptions(rtol=1e-10, atol=1e-10)
        )
        state_eq._apow = 2
        state_eq._bpow = 3
        param = self.bkd().array([0.8, 1.1])[:, None]
        init_iterate = self.bkd().array([-1.0, -1.0])[:, None]
        state = state_eq.solve(init_iterate, param)

        a, b = param
        exact_state = self.bkd().stack(
            [
                -self.bkd().sqrt((b + 1) * (b**2 - b + 1) / (a**2 * b**3 + 1)),
                -self.bkd().sqrt(-(a - 1) * (a + 1) / (a**2 * b**3 + 1)),
            ]
        )
        self.bkd().assert_allclose(state, exact_state)

    def test_functional_and_adjoint_operator(self) -> None:
        """
        Test the functional and adjoint operator for nonlinear coupled equations.
        """
        state_eq = NonLinearCoupledStateEquations(
            self.bkd(), NewtonSolverOptions(rtol=1e-10, atol=1e-10)
        )
        state_eq._apow = 2
        state_eq._bpow = 3
        param = self.bkd().array([0.8, 1.1])[:, None]
        init_iterate = self.bkd().array([-1.0, -1.0])[:, None]

        weights = self.bkd().ones((2, 1))
        functional = WeightedSumFunctional(weights, 2, self.bkd())
        adjoint_op = AdjointOperatorWithJacobianAndHVP(state_eq, functional)
        derivative_checker = ImplicitFunctionDerivativeChecker(adjoint_op)
        tols = derivative_checker.get_derivative_tolerances(1e-6)
        # Reduce finite difference step sizes for Newton convergence
        fd_eps = self.bkd().flip(self.bkd().logspace(-13, -1, 12))
        derivative_checker.check_derivatives(
            init_iterate, param, tols, fd_eps=fd_eps, verbosity=0
        )


# Derived test class for NumPy backend
class TestNonLinearCoupledEquationsNumpy(
    TestNonLinearCoupledEquations[NDArray[Any]], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestNonLinearCoupledEquationsTorch(
    TestNonLinearCoupledEquations[torch.Tensor], unittest.TestCase
):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
