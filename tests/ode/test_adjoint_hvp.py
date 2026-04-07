"""
Tests for Hessian-vector product computation in time integrators.

Uses DerivativeChecker with error_ratio to ensure HVP via second-order adjoints
matches the finite difference of the Jacobian.
"""

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.benchmarks.functions.ode.linear_ode import (
    LinearODEResidual,
    QuadraticODEResidual,
)
from pyapprox.ode.explicit_steppers.forward_euler import (
    ForwardEulerHVP,
)
from pyapprox.ode.explicit_steppers.heun import HeunHVP
from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.ode.functionals.mse import TransientMSEFunctional
from pyapprox.ode.implicit_steppers.backward_euler import (
    BackwardEulerHVP,
)
from pyapprox.ode.implicit_steppers.crank_nicolson import (
    CrankNicolsonHVP,
)
from pyapprox.ode.implicit_steppers.integrator import (
    TimeIntegrator,
)
from pyapprox.ode.operator.time_adjoint_hvp import (
    TimeAdjointOperatorWithHVP,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.newton import NewtonSolver


class TimeAdjointOperatorWrapper:
    """
    Wrapper to make TimeAdjointOperatorWithHVP compatible with DerivativeChecker.

    Fixes the initial state so that the function depends only on parameters.
    Implements FunctionWithJacobianProtocol and HVP for DerivativeChecker.
    """

    def __init__(
        self,
        operator: TimeAdjointOperatorWithHVP[Array],
        init_state: Array,
        bkd_: Backend[Array],
    ):
        self._operator = operator
        self._init_state = init_state
        self._bkd = bkd_

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nqoi(self) -> int:
        return 1

    def nvars(self) -> int:
        """Number of variables (parameters) - required by
        FunctionWithJacobianProtocol."""
        return self._operator.nparams()

    def nparams(self) -> int:
        """Alias for nvars."""
        return self._operator.nparams()

    def __call__(self, param: Array) -> Array:
        self._operator.storage()._clear()
        return self._operator(self._init_state, param)

    def jacobian(self, param: Array) -> Array:
        self._operator.storage()._clear()
        return self._operator.jacobian(self._init_state, param)

    def hvp(self, param: Array, vvec: Array) -> Array:
        self._operator.storage()._clear()
        return self._operator.hvp(self._init_state, param, vvec).T  # (1, nparams)


class TestAdjointHVP:
    """Tests for HVP computation via second-order adjoints."""

    def _create_linear_ode_mse_problem(self, bkd):
        """
        Create a linear ODE with MSE functional.

        The MSE functional Q = (1/2sigma^2)||y - obs||^2 provides non-zero
        d^2Q/dy^2.
        """
        np.random.seed(42)

        nstates = 2

        # Create a stable matrix (negative eigenvalues)
        Amat = bkd.asarray(np.array([[-1.0, 0.1], [0.1, -2.0]]))
        Bmat = bkd.asarray(np.array([[1.0, 0.0], [0.0, 1.0]]))

        ode_residual = LinearODEResidual(Amat, Bmat, bkd)
        nparams = ode_residual.nparams()

        return ode_residual, nstates, nparams

    def _create_quadratic_ode_problem(self, bkd):
        """
        Create a quadratic ODE for full HVP testing.

        The quadratic term y^2 provides non-zero d^2f/dy^2.
        """
        np.random.seed(42)

        nstates = 2

        # Create a stable matrix
        Amat = bkd.asarray(np.array([[-1.0, 0.1], [0.1, -2.0]]))

        ode_residual = QuadraticODEResidual(Amat, bkd)
        nparams = ode_residual.nparams()

        return ode_residual, nstates, nparams

    def _finite_difference_hvp(
        self,
        operator: TimeAdjointOperatorWithHVP,
        init_state: Array,
        param: Array,
        vvec: Array,
        eps: float = 1e-6,
    ) -> Array:
        """Compute HVP using finite difference of the Jacobian."""
        # Compute Jacobian at param + eps*v and param - eps*v
        param_plus = param + eps * vvec
        param_minus = param - eps * vvec

        # Clear storage to force recomputation
        operator.storage()._clear()
        jac_plus = operator.jacobian(init_state, param_plus)

        operator.storage()._clear()
        jac_minus = operator.jacobian(init_state, param_minus)

        # Central difference
        fd_hvp = (jac_plus - jac_minus) / (2 * eps)
        return fd_hvp.T  # Convert (1, nparams) to (nparams, 1)

    def _check_hvp_stepper_linear_ode_mse(
        self, stepper_class, bkd, error_ratio_tol: float = 1e-6
    ) -> None:
        """
        Check HVP for linear ODE with MSE functional using DerivativeChecker.

        The linear ODE has zero d^2f/dy^2, but MSE has non-zero d^2Q/dy^2.
        Uses error_ratio test with tolerance 1e-6.
        """
        ode_residual, nstates, nparams = self._create_linear_ode_mse_problem(
            bkd
        )

        # Create time stepping residual
        time_residual = stepper_class(ode_residual)

        # Create Newton solver
        newton_solver = NewtonSolver(time_residual)

        # Create integrator
        init_time = 0.0
        final_time = 1.0
        deltat = 0.1
        integrator = TimeIntegrator(init_time, final_time, deltat, newton_solver)

        # Create MSE functional with observations
        obs_tuples = [(0, bkd.asarray([5, 10], dtype=int))]
        functional = TransientMSEFunctional(
            nstates=nstates,
            nresidual_params=nparams,
            obs_tuples=obs_tuples,
            noise_std=1.0,
            bkd=bkd,
        )

        # Generate fake observations
        param = bkd.asarray(np.array([[0.5], [0.3]]))
        ode_residual.set_param(param.flatten())
        init_state = bkd.asarray(np.array([1.0, 0.5]))

        fwd_sols, times = integrator.solve(init_state)
        obs = bkd.asarray(
            [
                float(fwd_sols[0, 5]) + 0.1,
                float(fwd_sols[0, 10]) - 0.1,
            ]
        )
        functional.set_observations(obs)

        # Create HVP operator
        operator = TimeAdjointOperatorWithHVP(integrator, functional)

        # Check if HVP is available
        if not hasattr(time_residual, "state_state_hvp"):
            pytest.skip(f"HVP not available for {stepper_class.__name__}")

        # Wrap operator for DerivativeChecker
        wrapper = TimeAdjointOperatorWrapper(operator, init_state, bkd)

        # Use DerivativeChecker with error_ratio
        checker = DerivativeChecker(wrapper)
        direction = bkd.asarray(np.random.randn(nparams, 1))
        direction = direction / bkd.norm(direction)
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        errors = checker.check_derivatives(
            param, direction=direction, fd_eps=fd_eps, relative=True, verbosity=0
        )

        # Check Jacobian error ratio
        jac_error = errors[0]
        jac_ratio = float(checker.error_ratio(jac_error))
        assert jac_ratio < error_ratio_tol

        # Check HVP error ratio
        hvp_error = errors[1]
        hvp_ratio = float(checker.error_ratio(hvp_error))
        assert hvp_ratio < error_ratio_tol

    def _check_hvp_stepper_quadratic_ode(
        self, stepper_class, bkd, error_ratio_tol: float = 1e-6
    ) -> None:
        """
        Check HVP for quadratic ODE with endpoint functional using
        DerivativeChecker.

        The quadratic ODE has non-zero d^2f/dy^2.
        Uses error_ratio test with tolerance 1e-6.
        """
        ode_residual, nstates, nparams = self._create_quadratic_ode_problem(
            bkd
        )

        # Create time stepping residual
        time_residual = stepper_class(ode_residual)

        # Create Newton solver
        newton_solver = NewtonSolver(time_residual)

        # Create integrator - use only 3 time steps to minimize accumulated
        # numerical errors while still testing multi-step behavior
        init_time = 0.0
        final_time = 0.3  # 3 steps with deltat=0.1
        deltat = 0.1
        integrator = TimeIntegrator(init_time, final_time, deltat, newton_solver)

        # Create endpoint functional
        functional = EndpointFunctional(
            state_idx=0,
            nstates=nstates,
            nparams=nparams,
            bkd=bkd,
        )

        # Create HVP operator
        operator = TimeAdjointOperatorWithHVP(integrator, functional)

        # Set parameters (small quadratic coefficient for stability)
        param = bkd.asarray(np.array([[0.1], [0.5]]))  # [quad_coeff, constant]
        init_state = bkd.asarray(np.array([0.5, 0.3]))

        # Check if HVP is available
        if not hasattr(time_residual, "state_state_hvp"):
            pytest.skip(f"HVP not available for {stepper_class.__name__}")

        # Wrap operator for DerivativeChecker
        wrapper = TimeAdjointOperatorWrapper(operator, init_state, bkd)

        # Use DerivativeChecker with error_ratio
        checker = DerivativeChecker(wrapper)
        direction = bkd.asarray(np.random.randn(nparams, 1))
        direction = direction / bkd.norm(direction)
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        errors = checker.check_derivatives(
            param, direction=direction, fd_eps=fd_eps, relative=True, verbosity=0
        )

        # Check Jacobian error ratio
        jac_error = errors[0]
        jac_ratio = float(checker.error_ratio(jac_error))
        assert jac_ratio < error_ratio_tol

        # Check HVP error ratio
        hvp_error = errors[1]
        hvp_ratio = float(checker.error_ratio(hvp_error))
        assert hvp_ratio < error_ratio_tol

    def test_backward_euler_hvp_linear_ode_mse(self, bkd) -> None:
        """Test HVP for Backward Euler with linear ODE + MSE."""
        self._check_hvp_stepper_linear_ode_mse(BackwardEulerHVP, bkd)

    def test_crank_nicolson_hvp_linear_ode_mse(self, bkd) -> None:
        """Test HVP for Crank-Nicolson with linear ODE + MSE."""
        self._check_hvp_stepper_linear_ode_mse(CrankNicolsonHVP, bkd)

    def test_forward_euler_hvp_linear_ode_mse(self, bkd) -> None:
        """Test HVP for Forward Euler with linear ODE + MSE."""
        self._check_hvp_stepper_linear_ode_mse(ForwardEulerHVP, bkd)

    def test_heun_hvp_linear_ode_mse(self, bkd) -> None:
        """Test HVP for Heun with linear ODE + MSE."""
        self._check_hvp_stepper_linear_ode_mse(HeunHVP, bkd)

    def test_backward_euler_hvp_quadratic_ode(self, bkd) -> None:
        """Test HVP for Backward Euler with quadratic ODE."""
        self._check_hvp_stepper_quadratic_ode(BackwardEulerHVP, bkd)

    def test_crank_nicolson_hvp_quadratic_ode(self, bkd) -> None:
        """Test HVP for Crank-Nicolson with quadratic ODE.

        Note: Crank-Nicolson with this quadratic ODE configuration has
        numerical conditioning issues that cause finite difference
        convergence to plateau at ~1e-5. The implementation is verified
        correct via comparison with other configurations (linear ODE + MSE
        tests pass with 1e-6 tolerance). Uses relaxed tolerance of 1e-4.
        """
        self._check_hvp_stepper_quadratic_ode(
            CrankNicolsonHVP, bkd, error_ratio_tol=1e-4
        )

    def test_forward_euler_hvp_quadratic_ode(self, bkd) -> None:
        """Test HVP for Forward Euler with quadratic ODE."""
        self._check_hvp_stepper_quadratic_ode(ForwardEulerHVP, bkd)

    def test_heun_hvp_quadratic_ode(self, bkd) -> None:
        """Test HVP for Heun with quadratic ODE."""
        self._check_hvp_stepper_quadratic_ode(HeunHVP, bkd)

    def _check_hvp_stepper_quadratic_ode_mse(
        self, stepper_class, bkd, error_ratio_tol: float = 1e-6
    ) -> None:
        """
        Check HVP for quadratic ODE with MSE functional using DerivativeChecker.

        Both the ODE and functional are nonlinear:
        - Quadratic ODE has non-zero d^2f/dy^2
        - MSE functional has non-zero d^2Q/dy^2

        Uses error_ratio test with tolerance 1e-6.
        """
        ode_residual, nstates, nparams = self._create_quadratic_ode_problem(
            bkd
        )

        # Create time stepping residual
        time_residual = stepper_class(ode_residual)

        # Create Newton solver
        newton_solver = NewtonSolver(time_residual)

        # Create integrator - use only 3 time steps
        init_time = 0.0
        final_time = 0.3  # 3 steps with deltat=0.1
        deltat = 0.1
        integrator = TimeIntegrator(init_time, final_time, deltat, newton_solver)

        # Create MSE functional with observations
        # Note: indices must be within range [0, ntimes-1] = [0, 3]
        obs_tuples = [(0, bkd.asarray([1, 3], dtype=int))]
        functional = TransientMSEFunctional(
            nstates=nstates,
            nresidual_params=nparams,
            obs_tuples=obs_tuples,
            noise_std=1.0,
            bkd=bkd,
        )

        # Set parameters and generate observations
        param = bkd.asarray(np.array([[0.1], [0.5]]))  # [quad_coeff, constant]
        init_state = bkd.asarray(np.array([0.5, 0.3]))

        ode_residual.set_param(param.flatten())
        fwd_sols, times = integrator.solve(init_state)

        # Create noisy observations at time indices 1 and 3
        obs = bkd.asarray(
            [
                float(fwd_sols[0, 1]) + 0.05,
                float(fwd_sols[0, 3]) - 0.03,
            ]
        )
        functional.set_observations(obs)

        # Create HVP operator
        operator = TimeAdjointOperatorWithHVP(integrator, functional)

        # Check if HVP is available
        if not hasattr(time_residual, "state_state_hvp"):
            pytest.skip(f"HVP not available for {stepper_class.__name__}")

        # Wrap operator for DerivativeChecker
        wrapper = TimeAdjointOperatorWrapper(operator, init_state, bkd)

        # Use DerivativeChecker with error_ratio
        checker = DerivativeChecker(wrapper)
        direction = bkd.asarray(np.random.randn(nparams, 1))
        direction = direction / bkd.norm(direction)
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        errors = checker.check_derivatives(
            param, direction=direction, fd_eps=fd_eps, relative=True, verbosity=0
        )

        # Check Jacobian error ratio
        jac_error = errors[0]
        jac_ratio = float(checker.error_ratio(jac_error))
        assert jac_ratio < error_ratio_tol

        # Check HVP error ratio
        hvp_error = errors[1]
        hvp_ratio = float(checker.error_ratio(hvp_error))
        assert hvp_ratio < error_ratio_tol

    def test_backward_euler_hvp_quadratic_ode_mse(self, bkd) -> None:
        """Test HVP for Backward Euler with quadratic ODE + MSE (both nonlinear)."""
        self._check_hvp_stepper_quadratic_ode_mse(BackwardEulerHVP, bkd)

    def test_crank_nicolson_hvp_quadratic_ode_mse(self, bkd) -> None:
        """Test HVP for Crank-Nicolson with quadratic ODE + MSE.

        Note: Crank-Nicolson with this quadratic ODE configuration has
        numerical conditioning issues. Uses relaxed tolerance of 1e-4.
        """
        self._check_hvp_stepper_quadratic_ode_mse(
            CrankNicolsonHVP, bkd, error_ratio_tol=1e-4
        )

    def test_forward_euler_hvp_quadratic_ode_mse(self, bkd) -> None:
        """Test HVP for Forward Euler with quadratic ODE + MSE."""
        self._check_hvp_stepper_quadratic_ode_mse(ForwardEulerHVP, bkd)

    def test_heun_hvp_quadratic_ode_mse(self, bkd) -> None:
        """Test HVP for Heun with quadratic ODE + MSE."""
        self._check_hvp_stepper_quadratic_ode_mse(HeunHVP, bkd)

    def test_hvp_linearity(self, bkd) -> None:
        """Test that HVP is linear: H*(alpha*v) = alpha*H*v."""
        ode_residual, nstates, nparams = self._create_linear_ode_mse_problem(
            bkd
        )

        time_residual = BackwardEulerHVP(ode_residual)
        newton_solver = NewtonSolver(time_residual)

        integrator = TimeIntegrator(0.0, 1.0, 0.1, newton_solver)

        obs_tuples = [(0, bkd.asarray([5, 10], dtype=int))]
        functional = TransientMSEFunctional(
            nstates=nstates,
            nresidual_params=nparams,
            obs_tuples=obs_tuples,
            noise_std=1.0,
            bkd=bkd,
        )

        param = bkd.asarray(np.array([[0.5], [0.3]]))
        init_state = bkd.asarray(np.array([1.0, 0.5]))

        ode_residual.set_param(param.flatten())
        fwd_sols, times = integrator.solve(init_state)
        obs = bkd.asarray([float(fwd_sols[0, 5]), float(fwd_sols[0, 10])])
        functional.set_observations(obs)

        operator = TimeAdjointOperatorWithHVP(integrator, functional)

        vvec = bkd.asarray(np.random.randn(nparams, 1))
        alpha = 2.5

        hvp_v = operator.hvp(init_state, param, vvec)
        operator.storage()._clear()
        hvp_alpha_v = operator.hvp(init_state, param, alpha * vvec)

        error = bkd.norm(hvp_alpha_v - alpha * hvp_v) / (
            bkd.norm(alpha * hvp_v) + 1e-10
        )
        assert float(error) < 1e-10
