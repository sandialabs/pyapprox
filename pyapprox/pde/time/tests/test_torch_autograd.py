"""
Tests comparing analytical Jacobian and HVP with PyTorch autograd.

This module verifies that the analytical derivatives implemented in the
time adjoint module match those computed by PyTorch autograd to within 1e-8.

The approach is to create an autograd-compatible forward simulation that
computes Q(p) end-to-end using torch operations, then compare:
1. Jacobian dQ/dp from adjoint method vs torch.autograd.grad
2. HVP H·v from second-order adjoints vs torch.autograd.functional.hvp

Note: The existing ODE residual classes break the autograd graph by using
float() to extract values. This test module creates autograd-friendly
versions for comparison.
"""


import pytest
import torch

from pyapprox.benchmarks.functions.ode.linear_ode import (
    QuadraticODEResidual,
)
from pyapprox.pde.time.explicit_steppers.forward_euler import (
    ForwardEulerHVP,
)
from pyapprox.pde.time.explicit_steppers.heun import HeunHVP
from pyapprox.pde.time.functionals.endpoint import EndpointFunctional
from pyapprox.pde.time.implicit_steppers.backward_euler import (
    BackwardEulerHVP,
)
from pyapprox.pde.time.implicit_steppers.crank_nicolson import (
    CrankNicolsonHVP,
)
from pyapprox.pde.time.implicit_steppers.integrator import TimeIntegrator
from pyapprox.pde.time.operator.time_adjoint_hvp import (
    TimeAdjointOperatorWithHVP,
)
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.rootfinding.newton import NewtonSolver

# =============================================================================
# Autograd-compatible forward simulation functions
# =============================================================================
# These implement the same time stepping as the adjoint module but maintain
# the autograd computation graph throughout.


def forward_euler_step_autograd(y, param, Amat, dt):
    """Forward Euler step: y_n = y_{n-1} + dt * f(y_{n-1})."""
    # f(y) = A*y + p[0]*y^2 + p[1] for quadratic ODE
    p0, p1 = param[0], param[1]
    f = Amat @ y + p0 * y**2 + p1
    return y + dt * f


def backward_euler_step_autograd(y, param, Amat, dt, max_iter=20, tol=1e-12):
    """Backward Euler step via Newton iteration (differentiable)."""
    # Initial guess
    y_new = y.clone()

    p0, p1 = param[0], param[1]

    for _ in range(max_iter):
        # f(y_new) = A*y_new + p0*y_new^2 + p1
        f_new = Amat @ y_new + p0 * y_new**2 + p1
        # Residual: R = y_new - y - dt*f(y_new)
        R = y_new - y - dt * f_new
        # Jacobian: dR/dy = I - dt*J where J = A + 2*p0*diag(y_new)
        J = Amat + 2.0 * p0 * torch.diag(y_new)
        dR = torch.eye(len(y), dtype=y.dtype) - dt * J
        # Newton update
        delta = torch.linalg.solve(dR, R)
        y_new = y_new - delta
        if torch.max(torch.abs(delta)) < tol:
            break

    return y_new


def heun_step_autograd(y, param, Amat, dt):
    """Heun (RK2) step: predictor-corrector."""
    p0, p1 = param[0], param[1]

    # Predictor: k1 = f(y)
    k1 = Amat @ y + p0 * y**2 + p1
    y_pred = y + dt * k1

    # Corrector: k2 = f(y_pred)
    k2 = Amat @ y_pred + p0 * y_pred**2 + p1

    return y + 0.5 * dt * (k1 + k2)


def crank_nicolson_step_autograd(y, param, Amat, dt, max_iter=20, tol=1e-12):
    """Crank-Nicolson step via Newton iteration (differentiable)."""
    y_new = y.clone()

    p0, p1 = param[0], param[1]

    for _ in range(max_iter):
        # f(y) = A*y + p0*y^2 + p1
        f_old = Amat @ y + p0 * y**2 + p1
        f_new = Amat @ y_new + p0 * y_new**2 + p1

        # Residual: R = y_new - y - (dt/2)*(f(y) + f(y_new))
        R = y_new - y - 0.5 * dt * (f_old + f_new)

        # Jacobian: dR/dy_new = I - (dt/2)*J_new
        J_new = Amat + 2.0 * p0 * torch.diag(y_new)
        dR = torch.eye(len(y), dtype=y.dtype) - 0.5 * dt * J_new

        delta = torch.linalg.solve(dR, R)
        y_new = y_new - delta
        if torch.max(torch.abs(delta)) < tol:
            break

    return y_new


def integrate_autograd(y0, param, Amat, dt, nsteps, stepper_name):
    """Integrate ODE using autograd-friendly stepper."""
    y = y0.clone()

    step_fn = {
        "forward_euler": forward_euler_step_autograd,
        "backward_euler": backward_euler_step_autograd,
        "heun": heun_step_autograd,
        "crank_nicolson": crank_nicolson_step_autograd,
    }[stepper_name]

    for _ in range(nsteps):
        y = step_fn(y, param, Amat, dt)

    return y


def functional_autograd(y_final, state_idx=0):
    """Endpoint functional Q = y[state_idx]."""
    return y_final[state_idx]


class TestTorchAutogradComparison:
    """Compare analytical derivatives with PyTorch autograd.

    Uses autograd-friendly simulation functions to verify that the
    adjoint-based Jacobian and HVP match torch.autograd results.
    """

    def setup_method(self):
        """Set up torch with float64 precision."""
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(42)
        _bkd = TorchBkd()
        self._tol = 1e-8
        self._Amat = torch.tensor([[-0.5, 0.1], [0.2, -0.3]], dtype=torch.float64)
        self._dt = 0.1
        self._nsteps = 3

    def _stepper_name(self, stepper_class):
        """Get stepper name for autograd integration."""
        name_map = {
            ForwardEulerHVP: "forward_euler",
            BackwardEulerHVP: "backward_euler",
            HeunHVP: "heun",
            CrankNicolsonHVP: "crank_nicolson",
        }
        return name_map[stepper_class]

    def _create_quadratic_ode_operator(self, bkd, stepper_class) :
        """Create operator with quadratic ODE (has non-zero HVP)."""
        nstates = 2
        nparams = 2

        Amat = bkd.asarray([[-0.5, 0.1], [0.2, -0.3]])
        ode_residual = QuadraticODEResidual(Amat, bkd)

        time_residual = stepper_class(ode_residual)
        newton_solver = NewtonSolver(time_residual)
        integrator = TimeIntegrator(
            init_time=0.0,
            final_time=self._nsteps * self._dt,
            deltat=self._dt,
            newton_solver=newton_solver,
        )
        functional = EndpointFunctional(
            state_idx=0,
            nstates=nstates,
            nparams=nparams,
            bkd=bkd,
        )
        return TimeAdjointOperatorWithHVP(integrator, functional), ode_residual

    def _compute_autograd_jacobian(self, stepper_name, init_state, param):
        """Compute Jacobian using torch.autograd.grad with autograd-friendly sim."""
        param_flat = param.flatten().clone().detach().requires_grad_(True)

        def forward_fn(p):
            y_final = integrate_autograd(
                init_state, p, self._Amat, self._dt, self._nsteps, stepper_name
            )
            return functional_autograd(y_final)

        Q = forward_fn(param_flat)
        grad = torch.autograd.grad(Q, param_flat)[0]
        return grad.detach()

    def _compute_autograd_hvp(self, stepper_name, init_state, param, vvec):
        """Compute HVP using torch.autograd.functional.hvp."""
        param_flat = param.flatten()
        vvec_flat = vvec.flatten()

        def forward_fn(p):
            y_final = integrate_autograd(
                init_state, p, self._Amat, self._dt, self._nsteps, stepper_name
            )
            return functional_autograd(y_final)

        _, hvp = torch.autograd.functional.hvp(forward_fn, param_flat, vvec_flat)
        return hvp.detach()

    def _check_jacobian_stepper(self, bkd, stepper_class) :
        """Check Jacobian matches autograd for given stepper."""
        stepper_name = self._stepper_name(stepper_class)

        operator, ode_residual = self._create_quadratic_ode_operator(bkd, stepper_class)

        param = bkd.asarray([[0.1], [0.5]])
        init_state = bkd.asarray([1.0, 0.5])

        # Compute analytical Jacobian via adjoint
        ode_residual.set_param(param)
        operator.storage()._clear()
        analytical_jac = operator.jacobian(init_state, param)

        # Compute autograd Jacobian
        autograd_jac = self._compute_autograd_jacobian(stepper_name, init_state, param)

        # Compare
        analytical_flat = analytical_jac.flatten()
        autograd_flat = autograd_jac.flatten()

        error = float(torch.max(torch.abs(analytical_flat - autograd_flat)))
        norm = float(torch.max(torch.abs(autograd_flat))) + 1e-12
        rel_error = error / norm

        assert rel_error < self._tol

    def _check_hvp_stepper(self, bkd, stepper_class) :
        """Check HVP matches autograd for given stepper with quadratic ODE."""
        stepper_name = self._stepper_name(stepper_class)

        operator, ode_residual = self._create_quadratic_ode_operator(bkd, stepper_class)

        param = bkd.asarray([[0.1], [0.5]])
        init_state = bkd.asarray([1.0, 0.5])

        # Random direction for HVP
        vvec = bkd.asarray([[0.7], [-0.3]])
        vvec = vvec / torch.norm(vvec)

        # Check HVP is available
        time_residual = operator._integrator._newton_solver._residual
        if not hasattr(time_residual, "state_state_hvp"):
            pytest.skip(f"HVP not available for {stepper_class.__name__}")

        # Compute analytical HVP via second-order adjoints
        ode_residual.set_param(param)
        operator.storage()._clear()
        analytical_hvp = operator.hvp(init_state, param, vvec)

        # Compute autograd HVP
        autograd_hvp = self._compute_autograd_hvp(stepper_name, init_state, param, vvec)

        # Compare
        analytical_flat = analytical_hvp.flatten()
        autograd_flat = autograd_hvp.flatten()

        error = float(torch.max(torch.abs(analytical_flat - autograd_flat)))
        norm = float(torch.max(torch.abs(autograd_flat))) + 1e-12
        rel_error = error / norm

        assert rel_error < self._tol

    # =========================================================================
    # Jacobian tests
    # =========================================================================

    def test_backward_euler_jacobian(self, torch_bkd):
        """Test Backward Euler Jacobian matches autograd."""
        bkd = torch_bkd
        self._check_jacobian_stepper(bkd, BackwardEulerHVP)

    def test_crank_nicolson_jacobian(self, torch_bkd):
        """Test Crank-Nicolson Jacobian matches autograd."""
        bkd = torch_bkd
        self._check_jacobian_stepper(bkd, CrankNicolsonHVP)

    def test_forward_euler_jacobian(self, torch_bkd):
        """Test Forward Euler Jacobian matches autograd."""
        bkd = torch_bkd
        self._check_jacobian_stepper(bkd, ForwardEulerHVP)

    def test_heun_jacobian(self, torch_bkd):
        """Test Heun Jacobian matches autograd."""
        bkd = torch_bkd
        self._check_jacobian_stepper(bkd, HeunHVP)

    # =========================================================================
    # HVP tests
    # =========================================================================

    def test_backward_euler_hvp(self, torch_bkd):
        """Test Backward Euler HVP matches autograd."""
        bkd = torch_bkd
        self._check_hvp_stepper(bkd, BackwardEulerHVP)

    def test_crank_nicolson_hvp(self, torch_bkd):
        """Test Crank-Nicolson HVP matches autograd."""
        bkd = torch_bkd
        self._check_hvp_stepper(bkd, CrankNicolsonHVP)

    def test_forward_euler_hvp(self, torch_bkd):
        """Test Forward Euler HVP matches autograd."""
        bkd = torch_bkd
        self._check_hvp_stepper(bkd, ForwardEulerHVP)

    def test_heun_hvp(self, torch_bkd):
        """Test Heun HVP matches autograd."""
        bkd = torch_bkd
        self._check_hvp_stepper(bkd, HeunHVP)

    # =========================================================================
    # Additional validation tests
    # =========================================================================

    def test_hvp_multiple_directions(self, torch_bkd):
        """Test HVP with multiple random directions."""
        bkd = torch_bkd
        stepper_class = BackwardEulerHVP
        stepper_name = self._stepper_name(stepper_class)

        operator, ode_residual = self._create_quadratic_ode_operator(bkd, stepper_class)

        param = bkd.asarray([[0.1], [0.5]])
        init_state = bkd.asarray([1.0, 0.5])

        # Test multiple directions
        for seed in range(5):
            torch.manual_seed(seed)
            vvec = torch.randn(2, 1, dtype=torch.float64)
            vvec = vvec / torch.norm(vvec)

            # Compute analytical HVP
            ode_residual.set_param(param)
            operator.storage()._clear()
            analytical_hvp = operator.hvp(init_state, param, vvec)

            # Compute autograd HVP
            autograd_hvp = self._compute_autograd_hvp(
                stepper_name, init_state, param, vvec
            )

            # Compare
            error = float(torch.max(torch.abs(analytical_hvp.flatten() - autograd_hvp)))
            norm = float(torch.max(torch.abs(autograd_hvp))) + 1e-12
            rel_error = error / norm

            assert rel_error < self._tol

    def test_jacobian_matches_hvp_with_basis_vectors(self, torch_bkd):
        """Verify H·e_i equals i-th column of Hessian (via FD of Jacobian)."""
        bkd = torch_bkd
        stepper_class = BackwardEulerHVP

        operator, ode_residual = self._create_quadratic_ode_operator(bkd, stepper_class)

        param = bkd.asarray([[0.1], [0.5]])
        init_state = bkd.asarray([1.0, 0.5])
        nparams = 2

        # Compute full Hessian via finite difference of Jacobian
        eps = 1e-6
        hessian_fd = torch.zeros(nparams, nparams, dtype=torch.float64)

        for ii in range(nparams):
            param_plus = param.clone()
            param_minus = param.clone()
            param_plus[ii, 0] += eps
            param_minus[ii, 0] -= eps

            ode_residual.set_param(param_plus)
            operator.storage()._clear()
            jac_plus = operator.jacobian(init_state, param_plus)

            ode_residual.set_param(param_minus)
            operator.storage()._clear()
            jac_minus = operator.jacobian(init_state, param_minus)

            hessian_fd[:, ii] = (jac_plus - jac_minus).flatten() / (2 * eps)

        # Compute HVP for each basis vector
        for ii in range(nparams):
            ei = torch.zeros(nparams, 1, dtype=torch.float64)
            ei[ii, 0] = 1.0

            ode_residual.set_param(param)
            operator.storage()._clear()
            hvp_ei = operator.hvp(init_state, param, ei)

            # Compare with column of Hessian
            error = float(torch.max(torch.abs(hvp_ei.flatten() - hessian_fd[:, ii])))
            norm = float(torch.max(torch.abs(hessian_fd[:, ii]))) + 1e-12
            rel_error = error / norm

            # Use slightly relaxed tolerance for FD comparison
            assert rel_error < 1e-5
