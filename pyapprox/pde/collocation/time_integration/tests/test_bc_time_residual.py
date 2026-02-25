"""Tests for BCEnforcingTimeResidual adapter."""

import math
import unittest
from typing import Generic

from numpy.typing import NDArray
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.mesh import (
    create_uniform_mesh_1d,
    TransformedMesh1D,
)
from pyapprox.pde.collocation.boundary import (
    zero_dirichlet_bc,
    constant_dirichlet_bc,
)
from pyapprox.pde.collocation.physics import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.time_integration import (
    PhysicsToODEResidualAdapter,
    BCEnforcingTimeResidual,
)
from pyapprox.pde.time.implicit_steppers.backward_euler import (
    BackwardEulerResidual,
)


class TestBCEnforcingTimeResidual(Generic[Array], unittest.TestCase):
    """Base test class for BCEnforcingTimeResidual."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def _setup_diffusion_problem(self, npts=15):
        """Create a 1D diffusion problem with Dirichlet BCs."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        D = 0.5
        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=D)

        left_idx = umesh.boundary_indices(0)
        right_idx = umesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        adapter = PhysicsToODEResidualAdapter(physics, bkd)
        stepper = BackwardEulerResidual(adapter)
        bc_residual = BCEnforcingTimeResidual(stepper, physics, bkd)

        nodes = basis.nodes()
        state = bkd.sin(math.pi * nodes)

        return {
            "bkd": bkd,
            "physics": physics,
            "adapter": adapter,
            "stepper": stepper,
            "bc_residual": bc_residual,
            "state": state,
            "npts": npts,
            "nodes": nodes,
            "left_idx": left_idx,
            "right_idx": right_idx,
            "D": D,
        }

    def test_bc_residual_matches_manual(self):
        """Compare BCEnforcingTimeResidual against manual BE+BC computation.

        Reproduces the logic from CollocationModel._backward_euler_step.
        """
        s = self._setup_diffusion_problem()
        bkd = s["bkd"]
        physics = s["physics"]
        adapter = s["adapter"]
        state = s["state"]
        npts = s["npts"]

        deltat = 0.05
        t_n = 0.1
        t_np1 = t_n + deltat
        prev_state = bkd.copy(state)

        # Perturb state for current step
        y = prev_state + 0.01 * bkd.cos(math.pi * s["nodes"])

        # --- Manual backward Euler residual with BCs ---
        adapter.set_time(t_np1)
        f_y = adapter(y)
        mass = physics.mass_matrix()
        manual_residual = physics.apply_mass_matrix(y - prev_state) - deltat * f_y
        jac_f = adapter.jacobian(y)
        manual_jacobian = mass - deltat * jac_f
        manual_residual, manual_jacobian = physics.apply_boundary_conditions(
            manual_residual, manual_jacobian, y, t_np1
        )

        # --- BCEnforcingTimeResidual ---
        s["bc_residual"].set_time(t_n, deltat, prev_state)
        bc_residual_val = s["bc_residual"](y)
        bc_jacobian_val = s["bc_residual"].jacobian(y)

        bkd.assert_allclose(bc_residual_val, manual_residual, atol=1e-12)
        bkd.assert_allclose(bc_jacobian_val, manual_jacobian, atol=1e-12)

    def test_bc_rows_are_identity_in_jacobian(self):
        """Verify that boundary rows in the Jacobian are identity rows."""
        s = self._setup_diffusion_problem()
        bkd = s["bkd"]
        npts = s["npts"]

        deltat = 0.05
        prev_state = s["state"]
        y = bkd.copy(prev_state)

        s["bc_residual"].set_time(0.0, deltat, prev_state)
        jac = s["bc_residual"].jacobian(y)

        # Check boundary rows are identity
        left_idx = int(s["left_idx"][0])
        right_idx = int(s["right_idx"][0])

        for idx in [left_idx, right_idx]:
            expected_row = bkd.zeros((npts,))
            expected_row = bkd.copy(expected_row)
            expected_row[idx] = 1.0
            bkd.assert_allclose(jac[idx, :], expected_row, atol=1e-14)

    def test_bc_residual_at_boundaries(self):
        """Verify residual at boundary DOFs is u - g(t)."""
        s = self._setup_diffusion_problem()
        bkd = s["bkd"]

        deltat = 0.05
        prev_state = s["state"]
        y = bkd.copy(prev_state)

        s["bc_residual"].set_time(0.0, deltat, prev_state)
        res = s["bc_residual"](y)

        # For zero Dirichlet BCs: residual at boundary = u[boundary] - 0
        left_idx = int(s["left_idx"][0])
        right_idx = int(s["right_idx"][0])

        bkd.assert_allclose(
            bkd.asarray([res[left_idx]]),
            bkd.asarray([y[left_idx]]),
            atol=1e-14,
        )
        bkd.assert_allclose(
            bkd.asarray([res[right_idx]]),
            bkd.asarray([y[right_idx]]),
            atol=1e-14,
        )

    def test_dynamic_binding_no_param_jacobian(self):
        """Verify param_jacobian is NOT exposed for non-parameterized physics."""
        s = self._setup_diffusion_problem()
        self.assertFalse(hasattr(s["bc_residual"], "param_jacobian"))
        self.assertFalse(hasattr(s["bc_residual"], "adjoint_diag_jacobian"))

    def test_sensitivity_protocol_delegation(self):
        """Test that sensitivity methods delegate correctly."""
        s = self._setup_diffusion_problem()
        bc_res = s["bc_residual"]

        # is_explicit should return False for backward Euler
        self.assertFalse(bc_res.is_explicit())
        # has_prev_state_hessian should return False for backward Euler
        self.assertFalse(bc_res.has_prev_state_hessian())
        # native_residual should be the adapter
        self.assertIs(bc_res.native_residual, s["adapter"])

    def test_linsolve_with_bcs(self):
        """Test that linsolve uses the BC-modified Jacobian."""
        s = self._setup_diffusion_problem()
        bkd = s["bkd"]

        deltat = 0.05
        prev_state = s["state"]
        y = bkd.copy(prev_state)

        s["bc_residual"].set_time(0.0, deltat, prev_state)
        res = s["bc_residual"](y)
        delta = s["bc_residual"].linsolve(y, res)

        # delta should satisfy: J * delta = res
        jac = s["bc_residual"].jacobian(y)
        bkd.assert_allclose(bkd.dot(jac, delta), res, atol=1e-10)

    def test_newton_convergence(self):
        """Test that Newton iteration converges with BC-enforcing residual."""
        s = self._setup_diffusion_problem()
        bkd = s["bkd"]

        deltat = 0.01
        t_n = 0.0
        prev_state = s["state"]
        y = bkd.copy(prev_state)

        bc_res = s["bc_residual"]
        bc_res.set_time(t_n, deltat, prev_state)

        # Manual Newton iteration
        for _ in range(20):
            res = bc_res(y)
            res_norm = float(bkd.norm(res))
            if res_norm < 1e-10:
                break
            jac = bc_res.jacobian(y)
            delta = bkd.solve(jac, -res)
            y = y + delta

        self.assertLess(res_norm, 1e-10)


class TestBCEnforcingTimeResidualNumpy(
    TestBCEnforcingTimeResidual[NDArray]
):
    __test__ = True

    def bkd(self):
        return NumpyBkd()
