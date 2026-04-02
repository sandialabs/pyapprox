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
from pyapprox.util.rootfinding.newton import NewtonSolverOptions


class TestSensitivities:

    def test_nonlinear_coupled_residual_vector_functional(self, bkd):
        """
        Test sensitivities for nonlinear coupled residual equations with a vector
        functional.
        """
        # Create state equation
        state_eq = NonLinearCoupledStateEquations(
            bkd, NewtonSolverOptions(rtol=1e-10, atol=1e-10)
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
        tols = derivative_checker.get_derivative_tolerances(2e-6)

        # Reduce finite difference step sizes for Newton convergence
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 12))
        derivative_checker.check_derivatives(
            init_state, param, tols, fd_eps=fd_eps, verbosity=0
        )
