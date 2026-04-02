import numpy as np
import pytest

from pyapprox.optimization.implicitfunction.benchmarks.wildeys_nonlinear_state_equation import (  # noqa: E501
    NonLinearCoupledStateEquations,
)
from pyapprox.optimization.implicitfunction.functionals.weighted_sum import (
    WeightedSumFunctional,
)
from pyapprox.optimization.implicitfunction.operator.check_derivatives import (
    ImplicitFunctionDerivativeChecker,
)
from pyapprox.optimization.implicitfunction.operator.operator_with_hvp import (
    AdjointOperatorWithJacobianAndHVP,
)
from pyapprox.util.rootfinding.newton import NewtonSolverOptions


class TestNonLinearCoupledEquations:

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(1)

    def test_state_equation_solution(self, bkd) -> None:
        """
        Test the solution of the nonlinear coupled equations residual.
        """
        state_eq = NonLinearCoupledStateEquations(
            bkd, NewtonSolverOptions(rtol=1e-10, atol=1e-10)
        )
        state_eq._apow = 2
        state_eq._bpow = 3
        param = bkd.array([0.8, 1.1])[:, None]
        init_iterate = bkd.array([-1.0, -1.0])[:, None]
        state = state_eq.solve(init_iterate, param)

        a, b = param
        exact_state = bkd.stack(
            [
                -bkd.sqrt((b + 1) * (b**2 - b + 1) / (a**2 * b**3 + 1)),
                -bkd.sqrt(-(a - 1) * (a + 1) / (a**2 * b**3 + 1)),
            ]
        )
        bkd.assert_allclose(state, exact_state)

    def test_functional_and_adjoint_operator(self, bkd) -> None:
        """
        Test the functional and adjoint operator for nonlinear coupled equations.
        """
        state_eq = NonLinearCoupledStateEquations(
            bkd, NewtonSolverOptions(rtol=1e-10, atol=1e-10)
        )
        state_eq._apow = 2
        state_eq._bpow = 3
        param = bkd.array([0.8, 1.1])[:, None]
        init_iterate = bkd.array([-1.0, -1.0])[:, None]

        weights = bkd.ones((2, 1))
        functional = WeightedSumFunctional(weights, 2, bkd)
        adjoint_op = AdjointOperatorWithJacobianAndHVP(state_eq, functional)
        derivative_checker = ImplicitFunctionDerivativeChecker(adjoint_op)
        tols = derivative_checker.get_derivative_tolerances(1e-6)
        # Reduce finite difference step sizes for Newton convergence
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 12))
        derivative_checker.check_derivatives(
            init_iterate, param, tols, fd_eps=fd_eps, verbosity=0
        )
