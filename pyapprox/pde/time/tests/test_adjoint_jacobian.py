"""
Tests for adjoint-based gradient computation in time integrators.

Uses finite difference verification to ensure adjoint gradients are correct.
"""

import unittest
from typing import Generic

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.test_utils import load_tests
from pyapprox.optimization.rootfinding.newton import NewtonSolver
from pyapprox.pde.time.benchmarks.linear_ode import LinearODEResidual
from pyapprox.pde.time.implicit_steppers.backward_euler import (
    BackwardEulerResidual,
)
from pyapprox.pde.time.implicit_steppers.crank_nicolson import (
    CrankNicolsonResidual,
)
from pyapprox.pde.time.explicit_steppers.forward_euler import (
    ForwardEulerResidual,
)
from pyapprox.pde.time.explicit_steppers.heun import HeunResidual
from pyapprox.pde.time.implicit_steppers.integrator import (
    TimeIntegrator,
)
from pyapprox.pde.time.functionals.endpoint import EndpointFunctional


class TestAdjointJacobian(Generic[Array], unittest.TestCase):
    """Tests for adjoint-based gradient computation."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        """Return the backend. Override in subclasses."""
        raise NotImplementedError

    def _create_linear_ode_problem(self):
        """Create a simple linear ODE problem for testing."""
        bkd = self.bkd()
        np.random.seed(42)

        nstates = 2
        nparams = 2

        # Create a stable matrix (negative eigenvalues)
        Amat = bkd.asarray(np.array([[-1.0, 0.1], [0.1, -2.0]]))
        Bmat = bkd.asarray(np.array([[1.0, 0.0], [0.0, 1.0]]))

        ode_residual = LinearODEResidual(Amat, Bmat, bkd)

        return ode_residual, nstates, nparams

    def _finite_difference_gradient(
        self,
        integrator: TimeIntegrator,
        init_state: Array,
        param: Array,
        eps: float = 1e-6,
    ) -> Array:
        """Compute gradient using finite differences."""
        bkd = self.bkd()
        nparams = param.shape[0]
        fd_grad = bkd.zeros((1, nparams))
        fd_grad = bkd.copy(fd_grad)

        for ii in range(nparams):
            param_plus = bkd.copy(param)
            param_minus = bkd.copy(param)
            param_plus[ii, 0] += eps
            param_minus[ii, 0] -= eps

            # Forward solve with perturbed parameters
            integrator.time_residual().native_residual.set_param(param_plus.flatten())
            fwd_sols_plus, times = integrator.solve(init_state)
            qoi_plus = integrator._functional(fwd_sols_plus, param_plus)

            integrator.time_residual().native_residual.set_param(param_minus.flatten())
            fwd_sols_minus, times = integrator.solve(init_state)
            qoi_minus = integrator._functional(fwd_sols_minus, param_minus)

            fd_grad[0, ii] = float((qoi_plus - qoi_minus) / (2 * eps))

        return fd_grad

    def _check_gradient_stepper(
        self, stepper_class, tol: float = 1e-5
    ) -> None:
        """Check gradient for a given stepper class."""
        bkd = self.bkd()
        ode_residual, nstates, nparams = self._create_linear_ode_problem()

        # Create time stepping residual (backend is extracted from ode_residual)
        time_residual = stepper_class(ode_residual)

        # Create Newton solver with time residual
        newton_solver = NewtonSolver(time_residual)

        # Create integrator
        init_time = 0.0
        final_time = 1.0
        deltat = 0.1
        integrator = TimeIntegrator(
            init_time, final_time, deltat, newton_solver
        )

        # Create functional
        functional = EndpointFunctional(
            state_idx=0, nstates=nstates, nparams=nparams, bkd=bkd
        )
        integrator.set_functional(functional)

        # Set parameters
        param = bkd.asarray(np.array([[0.5], [0.3]]))
        ode_residual.set_param(param.flatten())

        # Initial state
        init_state = bkd.asarray(np.array([1.0, 0.5]))

        # Forward solve
        fwd_sols, times = integrator.solve(init_state)

        # Compute adjoint gradient
        adjoint_grad = integrator.gradient(fwd_sols, times, param)

        # Compute finite difference gradient
        fd_grad = self._finite_difference_gradient(
            integrator, init_state, param, eps=1e-6
        )

        # Check gradient
        error = bkd.norm(adjoint_grad - fd_grad) / (bkd.norm(fd_grad) + 1e-10)
        self.assertLess(
            float(error),
            tol,
            f"Gradient error {float(error):.2e} exceeds tolerance {tol:.2e} "
            f"for {stepper_class.__name__}"
        )

    def test_backward_euler_gradient(self) -> None:
        """Test adjoint gradient for Backward Euler."""
        self._check_gradient_stepper(BackwardEulerResidual, tol=1e-4)

    def test_crank_nicolson_gradient(self) -> None:
        """Test adjoint gradient for Crank-Nicolson."""
        self._check_gradient_stepper(CrankNicolsonResidual, tol=1e-4)

    def test_forward_euler_gradient(self) -> None:
        """Test adjoint gradient for Forward Euler."""
        self._check_gradient_stepper(ForwardEulerResidual, tol=1e-4)

    def test_heun_gradient(self) -> None:
        """Test adjoint gradient for Heun."""
        self._check_gradient_stepper(HeunResidual, tol=1e-4)


class TestAdjointJacobianNumpy(TestAdjointJacobian):
    """Test adjoint Jacobian with NumPy backend."""

    __test__ = True

    def bkd(self) -> Backend:
        return NumpyBkd


if __name__ == "__main__":
    unittest.main()
