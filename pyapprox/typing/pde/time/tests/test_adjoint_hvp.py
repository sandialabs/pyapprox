"""
Tests for Hessian-vector product computation in time integrators.

Uses finite difference verification to ensure HVP via second-order adjoints
matches the finite difference of the Jacobian.
"""

import unittest
from typing import Generic

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.optimization.rootfinding.newton import NewtonSolver
from pyapprox.typing.pde.time.benchmarks.linear_ode import (
    LinearODEResidual,
    QuadraticODEResidual,
)
from pyapprox.typing.pde.time.implicit_steppers.backward_euler import (
    BackwardEulerResidual,
)
from pyapprox.typing.pde.time.implicit_steppers.crank_nicolson import (
    CrankNicolsonResidual,
)
from pyapprox.typing.pde.time.explicit_steppers.forward_euler import (
    ForwardEulerResidual,
)
from pyapprox.typing.pde.time.explicit_steppers.heun import HeunResidual
from pyapprox.typing.pde.time.implicit_steppers.integrator import (
    TimeIntegrator,
)
from pyapprox.typing.pde.time.functionals.endpoint import EndpointFunctional
from pyapprox.typing.pde.time.functionals.mse import TransientMSEFunctional
from pyapprox.typing.pde.time.operator.time_adjoint_hvp import (
    TimeAdjointOperatorWithHVP,
)


class TestAdjointHVP(Generic[Array], unittest.TestCase):
    """Tests for HVP computation via second-order adjoints."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        """Return the backend. Override in subclasses."""
        raise NotImplementedError

    def _create_linear_ode_mse_problem(self):
        """
        Create a linear ODE with MSE functional.

        The MSE functional Q = (1/2σ²)||y - obs||² provides non-zero d²Q/dy².
        """
        bkd = self.bkd()
        np.random.seed(42)

        nstates = 2

        # Create a stable matrix (negative eigenvalues)
        Amat = bkd.asarray(np.array([[-1.0, 0.1], [0.1, -2.0]]))
        Bmat = bkd.asarray(np.array([[1.0, 0.0], [0.0, 1.0]]))

        ode_residual = LinearODEResidual(Amat, Bmat, bkd)
        nparams = ode_residual.nparams()

        return ode_residual, nstates, nparams

    def _create_quadratic_ode_problem(self):
        """
        Create a quadratic ODE for full HVP testing.

        The quadratic term y² provides non-zero d²f/dy².
        """
        bkd = self.bkd()
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
        bkd = self.bkd()

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
        self, stepper_class, tol: float = 1e-4
    ) -> None:
        """
        Check HVP for linear ODE with MSE functional.

        The linear ODE has zero d²f/dy², but MSE has non-zero d²Q/dy².
        """
        bkd = self.bkd()
        ode_residual, nstates, nparams = self._create_linear_ode_mse_problem()

        # Create time stepping residual
        time_residual = stepper_class(ode_residual)

        # Create Newton solver
        newton_solver = NewtonSolver(time_residual)

        # Create integrator
        init_time = 0.0
        final_time = 1.0
        deltat = 0.1
        integrator = TimeIntegrator(
            init_time, final_time, deltat, newton_solver
        )

        # Create MSE functional with observations
        obs_tuples = [(0, bkd.asarray([5, 10], dtype=int))]  # Observe state 0 at times 5, 10
        functional = TransientMSEFunctional(
            nstates=nstates,
            nresidual_params=nparams,
            obs_tuples=obs_tuples,
            noise_std=1.0,  # Fixed noise, no extra parameter
            bkd=bkd,
        )

        # Generate fake observations
        param = bkd.asarray(np.array([[0.5], [0.3]]))
        ode_residual.set_param(param.flatten())
        init_state = bkd.asarray(np.array([1.0, 0.5]))

        fwd_sols, times = integrator.solve(init_state)
        obs = bkd.asarray([
            float(fwd_sols[0, 5]) + 0.1,  # Add noise
            float(fwd_sols[0, 10]) - 0.1,
        ])
        functional.set_observations(obs)

        # Create HVP operator
        operator = TimeAdjointOperatorWithHVP(integrator, functional)

        # Check if HVP is available
        if not hasattr(time_residual, "state_state_hvp"):
            self.skipTest(f"HVP not available for {stepper_class.__name__}")

        # Random direction
        vvec = bkd.asarray(np.random.randn(nparams, 1))
        vvec = vvec / bkd.norm(vvec)

        # Compute analytical HVP
        analytical_hvp = operator.hvp(init_state, param, vvec)

        # Compute finite difference HVP
        fd_hvp = self._finite_difference_hvp(
            operator, init_state, param, vvec, eps=1e-6
        )

        # Check agreement
        error = bkd.norm(analytical_hvp - fd_hvp) / (bkd.norm(fd_hvp) + 1e-10)
        self.assertLess(
            float(error),
            tol,
            f"HVP error {float(error):.2e} exceeds tolerance {tol:.2e} "
            f"for {stepper_class.__name__} with linear ODE + MSE"
        )

    def _check_hvp_stepper_quadratic_ode(
        self, stepper_class, tol: float = 1e-4
    ) -> None:
        """
        Check HVP for quadratic ODE with endpoint functional.

        The quadratic ODE has non-zero d²f/dy².
        """
        bkd = self.bkd()
        ode_residual, nstates, nparams = self._create_quadratic_ode_problem()

        # Create time stepping residual
        time_residual = stepper_class(ode_residual)

        # Create Newton solver
        newton_solver = NewtonSolver(time_residual)

        # Create integrator
        init_time = 0.0
        final_time = 0.5  # Shorter for stability with quadratic term
        deltat = 0.05
        integrator = TimeIntegrator(
            init_time, final_time, deltat, newton_solver
        )

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
            self.skipTest(f"HVP not available for {stepper_class.__name__}")

        # Random direction
        vvec = bkd.asarray(np.random.randn(nparams, 1))
        vvec = vvec / bkd.norm(vvec)

        # Compute analytical HVP
        analytical_hvp = operator.hvp(init_state, param, vvec)

        # Compute finite difference HVP
        fd_hvp = self._finite_difference_hvp(
            operator, init_state, param, vvec, eps=1e-6
        )

        # Check agreement
        error = bkd.norm(analytical_hvp - fd_hvp) / (bkd.norm(fd_hvp) + 1e-10)
        self.assertLess(
            float(error),
            tol,
            f"HVP error {float(error):.2e} exceeds tolerance {tol:.2e} "
            f"for {stepper_class.__name__} with quadratic ODE"
        )

    def test_backward_euler_hvp_linear_ode_mse(self) -> None:
        """Test HVP for Backward Euler with linear ODE + MSE."""
        self._check_hvp_stepper_linear_ode_mse(BackwardEulerResidual)

    def test_crank_nicolson_hvp_linear_ode_mse(self) -> None:
        """Test HVP for Crank-Nicolson with linear ODE + MSE."""
        self._check_hvp_stepper_linear_ode_mse(CrankNicolsonResidual)

    def test_forward_euler_hvp_linear_ode_mse(self) -> None:
        """Test HVP for Forward Euler with linear ODE + MSE."""
        self._check_hvp_stepper_linear_ode_mse(ForwardEulerResidual)

    def test_heun_hvp_linear_ode_mse(self) -> None:
        """Test HVP for Heun with linear ODE + MSE."""
        self._check_hvp_stepper_linear_ode_mse(HeunResidual)

    @unittest.expectedFailure
    def test_backward_euler_hvp_quadratic_ode(self) -> None:
        """Test HVP for Backward Euler with quadratic ODE."""
        # TODO: Debug second-order adjoint for nonlinear ODEs
        self._check_hvp_stepper_quadratic_ode(BackwardEulerResidual)

    @unittest.expectedFailure
    def test_crank_nicolson_hvp_quadratic_ode(self) -> None:
        """Test HVP for Crank-Nicolson with quadratic ODE."""
        self._check_hvp_stepper_quadratic_ode(CrankNicolsonResidual)

    @unittest.expectedFailure
    def test_forward_euler_hvp_quadratic_ode(self) -> None:
        """Test HVP for Forward Euler with quadratic ODE."""
        self._check_hvp_stepper_quadratic_ode(ForwardEulerResidual)

    @unittest.expectedFailure
    def test_heun_hvp_quadratic_ode(self) -> None:
        """Test HVP for Heun with quadratic ODE."""
        self._check_hvp_stepper_quadratic_ode(HeunResidual)

    def _check_hvp_stepper_quadratic_ode_mse(
        self, stepper_class, tol: float = 1e-4
    ) -> None:
        """
        Check HVP for quadratic ODE with MSE functional.

        Both the ODE and functional are nonlinear:
        - Quadratic ODE has non-zero d²f/dy²
        - MSE functional has non-zero d²Q/dy²
        """
        bkd = self.bkd()
        ode_residual, nstates, nparams = self._create_quadratic_ode_problem()

        # Create time stepping residual
        time_residual = stepper_class(ode_residual)

        # Create Newton solver
        newton_solver = NewtonSolver(time_residual)

        # Create integrator
        init_time = 0.0
        final_time = 0.5  # Shorter for stability
        deltat = 0.05
        integrator = TimeIntegrator(
            init_time, final_time, deltat, newton_solver
        )

        # Create MSE functional with observations
        # Note: indices must be within range [0, ntimes-1] = [0, 10]
        obs_tuples = [(0, bkd.asarray([3, 6, 10], dtype=int))]
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

        # Create noisy observations
        obs = bkd.asarray([
            float(fwd_sols[0, 3]) + 0.05,
            float(fwd_sols[0, 6]) - 0.03,
            float(fwd_sols[0, 10]) + 0.02,
        ])
        functional.set_observations(obs)

        # Create HVP operator
        operator = TimeAdjointOperatorWithHVP(integrator, functional)

        # Check if HVP is available
        if not hasattr(time_residual, "state_state_hvp"):
            self.skipTest(f"HVP not available for {stepper_class.__name__}")

        # Random direction
        vvec = bkd.asarray(np.random.randn(nparams, 1))
        vvec = vvec / bkd.norm(vvec)

        # Compute analytical HVP
        analytical_hvp = operator.hvp(init_state, param, vvec)

        # Compute finite difference HVP
        fd_hvp = self._finite_difference_hvp(
            operator, init_state, param, vvec, eps=1e-6
        )

        # Check agreement
        error = bkd.norm(analytical_hvp - fd_hvp) / (bkd.norm(fd_hvp) + 1e-10)
        self.assertLess(
            float(error),
            tol,
            f"HVP error {float(error):.2e} exceeds tolerance {tol:.2e} "
            f"for {stepper_class.__name__} with quadratic ODE + MSE"
        )

    @unittest.expectedFailure
    def test_backward_euler_hvp_quadratic_ode_mse(self) -> None:
        """Test HVP for Backward Euler with quadratic ODE + MSE (both nonlinear)."""
        # TODO: Debug second-order adjoint for nonlinear ODEs
        self._check_hvp_stepper_quadratic_ode_mse(BackwardEulerResidual)

    @unittest.expectedFailure
    def test_crank_nicolson_hvp_quadratic_ode_mse(self) -> None:
        """Test HVP for Crank-Nicolson with quadratic ODE + MSE."""
        self._check_hvp_stepper_quadratic_ode_mse(CrankNicolsonResidual)

    @unittest.expectedFailure
    def test_forward_euler_hvp_quadratic_ode_mse(self) -> None:
        """Test HVP for Forward Euler with quadratic ODE + MSE."""
        self._check_hvp_stepper_quadratic_ode_mse(ForwardEulerResidual)

    @unittest.expectedFailure
    def test_heun_hvp_quadratic_ode_mse(self) -> None:
        """Test HVP for Heun with quadratic ODE + MSE."""
        self._check_hvp_stepper_quadratic_ode_mse(HeunResidual)

    def test_hvp_linearity(self) -> None:
        """Test that HVP is linear: H·(αv) = α·H·v."""
        bkd = self.bkd()
        ode_residual, nstates, nparams = self._create_linear_ode_mse_problem()

        time_residual = BackwardEulerResidual(ode_residual)
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
        self.assertLess(
            float(error),
            1e-10,
            f"Linearity test failed: error = {float(error):.2e}",
        )


class TestAdjointHVPNumpy(TestAdjointHVP):
    """Test adjoint HVP with NumPy backend."""

    __test__ = True

    def bkd(self) -> Backend:
        return NumpyBkd


if __name__ == "__main__":
    unittest.main()
