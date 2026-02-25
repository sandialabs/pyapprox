"""Tests for variable Lame parameter support in LinearElasticityPhysics.

Verifies:
1. set_mu/set_lamda setters with validation
2. Variable-Lame Jacobian correctness via DerivativeChecker
3. Manufactured solution with spatially varying lambda(x), mu(x)
4. Residual sensitivity methods via finite differences
5. Dual backend support (NumPy and PyTorch)
"""

import math
import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.pde.collocation.manufactured_solutions import (
    ManufacturedLinearElasticityEquations,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh2D,
    create_uniform_mesh_2d,
)
from pyapprox.pde.collocation.physics import LinearElasticityPhysics
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class PhysicsDerivativeWrapper(Generic[Array]):
    """Wrapper to adapt physics interface for DerivativeChecker."""

    def __init__(self, physics, time=0.0):
        self._physics = physics
        self._time = time
        self._backend = physics._bkd

    def bkd(self):
        return self._backend

    def nvars(self):
        return self._physics.nstates()

    def nqoi(self):
        return self._physics.nstates()

    def __call__(self, samples):
        if samples.ndim == 2:
            return self._backend.stack(
                [
                    self._physics.residual(samples[:, i], self._time)
                    for i in range(samples.shape[1])
                ],
                axis=1,
            )
        return self._physics.residual(samples, self._time).reshape(-1, 1)

    def jacobian(self, sample):
        if sample.ndim == 2:
            sample = sample[:, 0]
        return self._physics.jacobian(sample, self._time)


def _setup_2d_physics(bkd, npts_1d=6, lamda=1.0, mu=1.0, forcing=None):
    """Create 2D linear elasticity physics on [-1,1]^2."""
    mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)
    basis = ChebyshevBasis2D(mesh, bkd)
    mesh_obj = create_uniform_mesh_2d((npts_1d, npts_1d), (-1.0, 1.0, -1.0, 1.0), bkd)
    physics = LinearElasticityPhysics(basis, bkd, lamda=lamda, mu=mu, forcing=forcing)
    return physics, basis, mesh_obj


def _get_nodes(basis, bkd):
    """Get 2D physical coordinates. Shape: (2, npts)."""
    nodes_x = basis.nodes_x()
    nodes_y = basis.nodes_y()
    xx, yy = bkd.meshgrid(nodes_x, nodes_y, indexing="xy")
    return bkd.stack([xx.flatten(), yy.flatten()], axis=0)


def _set_dirichlet_all_sides(physics, mesh_obj, bkd, npts):
    """Apply homogeneous Dirichlet BCs on all 4 sides for both components."""
    bcs = []
    for side in range(4):
        boundary_idx = mesh_obj.boundary_indices(side)
        bc_u = zero_dirichlet_bc(bkd, boundary_idx)
        bcs.append(bc_u)
        boundary_idx_v = bkd.asarray([idx + npts for idx in boundary_idx])
        bc_v = zero_dirichlet_bc(bkd, boundary_idx_v)
        bcs.append(bc_v)
    physics.set_boundary_conditions(bcs)


def _interior_indices(mesh_obj, npts):
    """Get interior DOF indices (both u and v components)."""
    boundary_set = set()
    for side in range(4):
        for idx in mesh_obj.boundary_indices(side):
            boundary_set.add(idx)
            boundary_set.add(idx + npts)
    return [i for i in range(2 * npts) if i not in boundary_set]


class TestVariableLame(Generic[Array], unittest.TestCase):
    """Test variable Lame parameter support in LinearElasticityPhysics."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Setter tests
    # ------------------------------------------------------------------

    def test_set_mu_positive_scalar(self):
        """set_mu with positive scalar updates array and preserves value."""
        bkd = self.bkd()
        physics, basis, _ = _setup_2d_physics(bkd)
        physics.set_mu(2.5)
        bkd.assert_allclose(
            physics._mu_array,
            bkd.full((basis.npts(),), 2.5),
        )
        self.assertEqual(physics._mu_value, 2.5)

    def test_set_mu_positive_array(self):
        """set_mu with positive array updates array and clears value."""
        bkd = self.bkd()
        physics, basis, _ = _setup_2d_physics(bkd)
        npts = basis.npts()
        mu_arr = bkd.full((npts,), 3.0)
        physics.set_mu(mu_arr)
        bkd.assert_allclose(physics._mu_array, mu_arr)
        self.assertIsNone(physics._mu_value)

    def test_set_mu_nonpositive_raises(self):
        """set_mu raises ValueError for non-positive mu."""
        bkd = self.bkd()
        physics, _, _ = _setup_2d_physics(bkd)
        with self.assertRaises(ValueError):
            physics.set_mu(0.0)
        with self.assertRaises(ValueError):
            physics.set_mu(-1.0)

    def test_set_lamda_nonnegative(self):
        """set_lamda accepts zero and positive values."""
        bkd = self.bkd()
        physics, basis, _ = _setup_2d_physics(bkd)
        npts = basis.npts()
        physics.set_lamda(0.0)
        bkd.assert_allclose(physics._lambda_array, bkd.zeros((npts,)))
        physics.set_lamda(1.5)
        bkd.assert_allclose(physics._lambda_array, bkd.full((npts,), 1.5))

    def test_set_lamda_negative_raises(self):
        """set_lamda raises ValueError for negative lambda."""
        bkd = self.bkd()
        physics, _, _ = _setup_2d_physics(bkd)
        with self.assertRaises(ValueError):
            physics.set_lamda(-0.1)

    def test_set_scalar_preserves_constant_path(self):
        """set_mu/set_lamda with scalars keep _*_value non-None."""
        bkd = self.bkd()
        physics, _, _ = _setup_2d_physics(bkd)
        physics.set_mu(2.0)
        physics.set_lamda(3.0)
        self.assertEqual(physics._mu_value, 2.0)
        self.assertEqual(physics._lambda_value, 3.0)

    # ------------------------------------------------------------------
    # Variable Lame Jacobian
    # ------------------------------------------------------------------

    def test_variable_lame_jacobian(self):
        """DerivativeChecker validates variable-Lame Jacobian."""
        bkd = self.bkd()
        physics, basis, _ = _setup_2d_physics(bkd, npts_1d=6)
        nodes = _get_nodes(basis, bkd)
        basis.npts()

        # Spatially varying Lame parameters
        x = nodes[0, :]
        y = nodes[1, :]
        mu_field = 1.0 + 0.5 * bkd.sin(math.pi * x) * bkd.cos(math.pi * y)
        lam_field = 2.0 + 0.3 * bkd.cos(math.pi * x) * bkd.sin(math.pi * y)
        physics.set_mu(mu_field)
        physics.set_lamda(lam_field)

        wrapper = PhysicsDerivativeWrapper(physics)
        nstates = physics.nstates()
        np.random.seed(42)
        state_np = np.random.randn(nstates) * 0.1
        sample = bkd.asarray(state_np)[:, None]

        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, verbosity=0)
        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-5)

    def test_variable_matches_uniform(self):
        """Variable arrays of constant value match scalar constructor."""
        bkd = self.bkd()
        lam_val, mu_val = 1.5, 0.8

        # Constant via constructor
        physics_c, basis_c, _ = _setup_2d_physics(
            bkd, npts_1d=6, lamda=lam_val, mu=mu_val
        )
        # Constant via setters with arrays
        physics_v, basis_v, _ = _setup_2d_physics(bkd, npts_1d=6)
        npts = basis_v.npts()
        physics_v.set_mu(bkd.full((npts,), mu_val))
        physics_v.set_lamda(bkd.full((npts,), lam_val))

        nstates = physics_c.nstates()
        np.random.seed(123)
        state = bkd.asarray(np.random.randn(nstates) * 0.1)

        res_c = physics_c.residual(state, 0.0)
        res_v = physics_v.residual(state, 0.0)
        bkd.assert_allclose(res_v, res_c, rtol=1e-12)

        jac_c = physics_c.jacobian(state, 0.0)
        jac_v = physics_v.jacobian(state, 0.0)
        bkd.assert_allclose(jac_v, jac_c, rtol=1e-10)

    # ------------------------------------------------------------------
    # Manufactured solution with variable Lame
    # ------------------------------------------------------------------

    def test_manufactured_variable_lame(self):
        """Residual = 0 at exact solution with variable lambda(x), mu(x)."""
        bkd = self.bkd()
        npts_1d = 10
        physics, basis, mesh_obj = _setup_2d_physics(bkd, npts_1d=npts_1d)
        nodes = _get_nodes(basis, bkd)
        npts = basis.npts()

        # Variable Lame: polynomial in x, y so derivatives are exact
        lambda_str = "2 + x + y"
        mu_str = "1 + 0.5*x*y"

        man_sol = ManufacturedLinearElasticityEquations(
            sol_strs=[
                "(1 - x**2)*(1 - y**2)",
                "(1 - x**2)*(1 - y**2)*x",
            ],
            nvars=2,
            lambda_str=lambda_str,
            mu_str=mu_str,
            bkd=bkd,
            oned=True,
        )

        # Set variable Lame on physics
        lam_vals = man_sol.functions["lambda"](nodes)
        mu_vals = man_sol.functions["mu"](nodes)
        physics.set_lamda(lam_vals)
        physics.set_mu(mu_vals)

        # Get exact solution and forcing
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)
        u_exact_flat = bkd.concatenate([u_exact[:, 0], u_exact[:, 1]])
        forcing_flat = bkd.concatenate([forcing[:, 0], forcing[:, 1]])

        # Set forcing on physics (recreate with forcing)
        physics_f = LinearElasticityPhysics(
            basis, bkd, lamda=1.0, mu=1.0, forcing=lambda t: forcing_flat
        )
        physics_f.set_lamda(lam_vals)
        physics_f.set_mu(mu_vals)

        _set_dirichlet_all_sides(physics_f, mesh_obj, bkd, npts)

        residual = physics_f.residual(u_exact_flat, 0.0)
        jacobian = physics_f.jacobian(u_exact_flat, 0.0)
        residual_bc, _ = physics_f.apply_boundary_conditions(
            residual, jacobian, u_exact_flat, 0.0
        )

        interior_idx = _interior_indices(mesh_obj, npts)
        interior_res = bkd.asarray([residual_bc[i] for i in interior_idx])
        bkd.assert_allclose(
            interior_res,
            bkd.zeros(interior_res.shape),
            atol=1e-8,
        )

    # ------------------------------------------------------------------
    # Residual sensitivity FD tests
    # ------------------------------------------------------------------

    def test_residual_mu_sensitivity_fd(self):
        """Finite-difference validation of residual_mu_sensitivity."""
        bkd = self.bkd()
        physics, basis, _ = _setup_2d_physics(bkd, npts_1d=6)
        nodes = _get_nodes(basis, bkd)
        npts = basis.npts()
        nstates = physics.nstates()

        # Set variable mu
        x = nodes[0, :]
        nodes[1, :]
        mu0 = 1.0 + 0.5 * bkd.sin(math.pi * x)
        physics.set_mu(mu0)

        # Random state
        np.random.seed(42)
        state = bkd.asarray(np.random.randn(nstates) * 0.1)

        # Random perturbation
        np.random.seed(43)
        delta_mu = bkd.asarray(np.random.randn(npts) * 0.01)

        # Analytical sensitivity
        sens_analytical = physics.residual_mu_sensitivity(state, 0.0, delta_mu)

        # Finite difference
        eps = 1e-7
        physics.set_mu(mu0 + eps * delta_mu)
        res_plus = physics.residual(state, 0.0)
        physics.set_mu(mu0 - eps * delta_mu)
        res_minus = physics.residual(state, 0.0)
        sens_fd = (res_plus - res_minus) / (2.0 * eps)

        # Restore
        physics.set_mu(mu0)

        bkd.assert_allclose(sens_analytical, sens_fd, atol=1e-5)

    def test_residual_lamda_sensitivity_fd(self):
        """Finite-difference validation of residual_lamda_sensitivity."""
        bkd = self.bkd()
        physics, basis, _ = _setup_2d_physics(bkd, npts_1d=6)
        nodes = _get_nodes(basis, bkd)
        npts = basis.npts()
        nstates = physics.nstates()

        # Set variable lambda
        nodes[0, :]
        y = nodes[1, :]
        lam0 = 2.0 + 0.3 * bkd.cos(math.pi * y)
        physics.set_lamda(lam0)

        # Random state
        np.random.seed(42)
        state = bkd.asarray(np.random.randn(nstates) * 0.1)

        # Random perturbation
        np.random.seed(44)
        delta_lam = bkd.asarray(np.random.randn(npts) * 0.01)

        # Analytical sensitivity
        sens_analytical = physics.residual_lamda_sensitivity(state, 0.0, delta_lam)

        # Finite difference
        eps = 1e-7
        physics.set_lamda(lam0 + eps * delta_lam)
        res_plus = physics.residual(state, 0.0)
        physics.set_lamda(lam0 - eps * delta_lam)
        res_minus = physics.residual(state, 0.0)
        sens_fd = (res_plus - res_minus) / (2.0 * eps)

        # Restore
        physics.set_lamda(lam0)

        bkd.assert_allclose(sens_analytical, sens_fd, atol=1e-5)


class TestVariableLameNumpy(TestVariableLame[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestVariableLameTorch(TestVariableLame[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
