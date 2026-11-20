"""Tests for variable Lame parameter support in LinearElasticityPhysics.

Verifies:
1. Lame parameters held as NodalFields, validated when read
2. Variable-Lame Jacobian correctness via DerivativeChecker
3. Residual sensitivity methods via finite differences
4. Dual backend support (NumPy and PyTorch)

Solves with spatially varying lambda(x), mu(x) are verified by the
collocation linear-elasticity catalog.
"""

import math
from typing import Generic

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import (
    TransformedMesh2D,
    create_uniform_mesh_2d,
)
from pyapprox.pde.collocation.physics import LinearElasticityPhysics
from pyapprox.pde.collocation.physics.coefficients import NodalField
from pyapprox.util.backends.protocols import Array


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

    def derivatives(self):
        return Derivatives.first_order(jacobian=self.jacobian)

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


class TestVariableLame:
    """Test variable Lame parameter support in LinearElasticityPhysics."""

    # ------------------------------------------------------------------
    # Lame coefficients and their validation
    # ------------------------------------------------------------------

    def test_constants_become_fields(self, bkd):
        """Constant Lame parameters are held as rewritable NodalFields."""
        physics, basis, _ = _setup_2d_physics(bkd, lamda=1.5, mu=2.5)
        npts = basis.npts()
        assert isinstance(physics.mu_function(), NodalField)
        assert isinstance(physics.lamda_function(), NodalField)
        lam, mu = physics.lame_values(0.0)
        bkd.assert_allclose(mu, bkd.full((npts,), 2.5))
        bkd.assert_allclose(lam, bkd.full((npts,), 1.5))
        physics.mu_function().set_values(bkd.full((npts,), 3.0))
        bkd.assert_allclose(physics.lame_values(0.0)[1], bkd.full((npts,), 3.0))

    def test_nonpositive_mu_raises_on_read(self, bkd):
        """A non-positive mu is refused when the physics reads it."""
        for value in (0.0, -1.0):
            physics, basis, _ = _setup_2d_physics(bkd)
            physics.mu_function().set_values(bkd.full((basis.npts(),), value))
            state = bkd.zeros((physics.nstates(),))
            with pytest.raises(ValueError, match="mu must be positive"):
                physics.residual(state, 0.0)

    def test_lamda_nonnegative(self, bkd):
        """A zero lambda is accepted; a negative one is refused on read."""
        physics, basis, _ = _setup_2d_physics(bkd)
        npts = basis.npts()
        physics.lamda_function().set_values(bkd.zeros((npts,)))
        bkd.assert_allclose(physics.lame_values(0.0)[0], bkd.zeros((npts,)))
        physics.lamda_function().set_values(bkd.full((npts,), -0.1))
        with pytest.raises(ValueError, match="lamda must be non-negative"):
            physics.jacobian(bkd.zeros((physics.nstates(),)), 0.0)

    # ------------------------------------------------------------------
    # Variable Lame Jacobian
    # ------------------------------------------------------------------

    def test_variable_lame_jacobian(self, bkd):
        """DerivativeChecker validates variable-Lame Jacobian."""
        physics, basis, _ = _setup_2d_physics(bkd, npts_1d=6)
        nodes = _get_nodes(basis, bkd)
        basis.npts()

        # Spatially varying Lame parameters
        x = nodes[0, :]
        y = nodes[1, :]
        mu_field = 1.0 + 0.5 * bkd.sin(math.pi * x) * bkd.cos(math.pi * y)
        lam_field = 2.0 + 0.3 * bkd.cos(math.pi * x) * bkd.sin(math.pi * y)
        physics.mu_function().set_values(mu_field)
        physics.lamda_function().set_values(lam_field)

        wrapper = PhysicsDerivativeWrapper(physics)
        nstates = physics.nstates()
        np.random.seed(42)
        state_np = np.random.randn(nstates) * 0.1
        sample = bkd.asarray(state_np)[:, None]

        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, verbosity=0)
        assert checker.error_ratio(errors[0]) <= 1e-5

    def test_variable_matches_uniform(self, bkd):
        """Variable arrays of constant value match scalar constructor."""
        lam_val, mu_val = 1.5, 0.8

        # Constant via constructor
        physics_c, basis_c, _ = _setup_2d_physics(
            bkd, npts_1d=6, lamda=lam_val, mu=mu_val
        )
        # Constant via the fields' set_values with arrays
        physics_v, basis_v, _ = _setup_2d_physics(bkd, npts_1d=6)
        npts = basis_v.npts()
        physics_v.mu_function().set_values(bkd.full((npts,), mu_val))
        physics_v.lamda_function().set_values(bkd.full((npts,), lam_val))

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
    # Residual sensitivity FD tests
    # ------------------------------------------------------------------

    def test_residual_mu_sensitivity_fd(self, bkd):
        """Finite-difference validation of the mu assembly's action."""
        physics, basis, _ = _setup_2d_physics(bkd, npts_1d=6)
        nodes = _get_nodes(basis, bkd)
        npts = basis.npts()
        nstates = physics.nstates()

        # Set variable mu
        x = nodes[0, :]
        nodes[1, :]
        mu0 = 1.0 + 0.5 * bkd.sin(math.pi * x)
        physics.mu_function().set_values(mu0)

        # Random state
        np.random.seed(42)
        state = bkd.asarray(np.random.randn(nstates) * 0.1)

        # Random perturbation
        np.random.seed(43)
        delta_mu = bkd.asarray(np.random.randn(npts) * 0.01)

        # Analytical sensitivity via the full-matrix assembly
        sens_analytical = physics.residual_mu_jacobian(state) @ delta_mu

        # Finite difference
        eps = 1e-7
        physics.mu_function().set_values(mu0 + eps * delta_mu)
        res_plus = physics.residual(state, 0.0)
        physics.mu_function().set_values(mu0 - eps * delta_mu)
        res_minus = physics.residual(state, 0.0)
        sens_fd = (res_plus - res_minus) / (2.0 * eps)

        # Restore
        physics.mu_function().set_values(mu0)

        bkd.assert_allclose(sens_analytical, sens_fd, atol=1e-5)

    def test_residual_lamda_sensitivity_fd(self, bkd):
        """Finite-difference validation of the lambda assembly's action."""
        physics, basis, _ = _setup_2d_physics(bkd, npts_1d=6)
        nodes = _get_nodes(basis, bkd)
        npts = basis.npts()
        nstates = physics.nstates()

        # Set variable lambda
        nodes[0, :]
        y = nodes[1, :]
        lam0 = 2.0 + 0.3 * bkd.cos(math.pi * y)
        physics.lamda_function().set_values(lam0)

        # Random state
        np.random.seed(42)
        state = bkd.asarray(np.random.randn(nstates) * 0.1)

        # Random perturbation
        np.random.seed(44)
        delta_lam = bkd.asarray(np.random.randn(npts) * 0.01)

        # Analytical sensitivity via the full-matrix assembly
        sens_analytical = physics.residual_lamda_jacobian(state) @ delta_lam

        # Finite difference
        eps = 1e-7
        physics.lamda_function().set_values(lam0 + eps * delta_lam)
        res_plus = physics.residual(state, 0.0)
        physics.lamda_function().set_values(lam0 - eps * delta_lam)
        res_minus = physics.residual(state, 0.0)
        sens_fd = (res_plus - res_minus) / (2.0 * eps)

        # Restore
        physics.lamda_function().set_values(lam0)

        bkd.assert_allclose(sens_analytical, sens_fd, atol=1e-5)
