import numpy as np

from pyapprox.benchmarks.functions.algebraic.linear_state_equation import (
    LinearStateEquation,
)
from pyapprox.optimization.implicitfunction.functionals.mean_squared_error import (
    MSEFunctional,
)
from pyapprox.optimization.implicitfunction.functionals.tikhonov_mean_squared_error import (  # noqa: E501
    TikhonovMSEFunctional,
)
from pyapprox.optimization.implicitfunction.operator.check_derivatives import (
    ImplicitFunctionDerivativeChecker,
)
from pyapprox.optimization.implicitfunction.operator.operator_with_hvp import (
    AdjointOperatorWithJacobianAndHVP,
)


class TestLinearLeastSquares:

    def test_linear_least_squares(self, bkd) -> None:
        """
        Test the value and derivative computation for linear least squares problems.
        """
        # Problem setup
        np.random.seed(1)
        nstates, nparams = 3, 2
        Amat = bkd.asarray(np.random.normal(0, 1, (nstates, nparams)))
        param = bkd.asarray(np.random.normal(0, 1, (nparams, 1)))
        obs = Amat @ param
        init_state = bkd.full((nstates, 1), 1.0)

        # Create state equation and functional
        state_eq = LinearStateEquation(Amat, obs, bkd)
        functional = MSEFunctional(nstates, nparams, bkd)
        functional.set_observations(obs)

        # Create adjoint operator
        adjoint_operator = AdjointOperatorWithJacobianAndHVP(state_eq, functional)

        # Test value computation
        value = adjoint_operator(init_state, param)
        state = state_eq.solve(init_state, param)
        expected_value = functional(state, param)
        bkd.assert_allclose(value, expected_value)

        # Test derivative computation
        derivative_checker = ImplicitFunctionDerivativeChecker(adjoint_operator)
        tols = derivative_checker.get_derivative_tolerances(2e-6)
        derivative_checker.check_derivatives(init_state, param, tols, verbosity=0)

    def test_tikhonov_linear_least_squares(self, bkd) -> None:
        """
        Test Tikhonov regularization for linear least squares problems.
        """
        # Problem setup
        np.random.seed(1)
        nobs, nvars = 3, 2
        Amat = bkd.asarray(np.random.randn(nobs, nvars))  # Random matrix
        param = bkd.asarray(np.random.randn(nvars, 1))  # Random parameters
        obs = Amat @ param  # Observations
        init_state = bkd.zeros((nobs, 1))  # Initial state

        # Create state equation and functional
        state_eq = LinearStateEquation(Amat, obs, bkd)
        functional = TikhonovMSEFunctional(nobs, nvars, bkd)
        functional.set_observations(obs)

        # Create adjoint operator
        adjoint_operator = AdjointOperatorWithJacobianAndHVP(state_eq, functional)

        # Test derivative computation
        derivative_checker = ImplicitFunctionDerivativeChecker(adjoint_operator)
        tols = derivative_checker.get_derivative_tolerances(5e-6)
        tols[[2, 3, 4]] = 5e-6
        derivative_checker.check_derivatives(init_state, param, tols, verbosity=0)
