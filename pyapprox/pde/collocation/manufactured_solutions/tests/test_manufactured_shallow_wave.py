"""Parameterized tests for Shallow Wave manufactured solutions with physics.

Verifies:
1. Residual = 0 at exact solution (using polynomial solutions for machine precision)
2. Jacobian correctness via DerivativeChecker
3. Multiple test cases via parameterization

Sign Convention Note:
---------------------
The ManufacturedShallowWave class computes forcing for the hyperbolic form:
    du/dt + dF/dx = S (source term)
    => forcing = dF/dx + S

For 1D momentum equation:
    d(hu)/dt + d(hu²/h + 0.5*g*h²)/dx = -g*h*db/dx + f
    => manufactured forcing = d(flux)/dx + g*h*db/dx

The ShallowWavePhysics computes residual for:
    du/dt = residual(u, t)
    => residual = -d(flux)/dx - g*h*db/dx + forcing

For steady state (du/dt = 0), the physics gives:
    0 = -d(flux)/dx - g*h*db/dx + f
    => f = d(flux)/dx + g*h*db/dx

This matches the manufactured solution forcing, so no sign negation is needed.
"""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import constant_dirichlet_bc
from pyapprox.pde.collocation.manufactured_solutions import (
    ManufacturedShallowWave,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    create_uniform_mesh_1d,
)
from pyapprox.pde.collocation.physics import ShallowWavePhysics
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class PhysicsDerivativeWrapper(Generic[Array]):
    """Wrapper to adapt physics interface for DerivativeChecker.

    DerivativeChecker expects:
    - bkd() method
    - nvars() method
    - nqoi() method
    - __call__(samples) for batch evaluation returning (nqoi, nsamples)
    - jacobian(sample) for single sample returning (nqoi, nvars)
    """

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
        # samples shape: (nvars, nsamples), return (nqoi, nsamples)
        if samples.ndim == 2:
            return self._backend.stack(
                [
                    self._physics.residual(samples[:, i], self._time)
                    for i in range(samples.shape[1])
                ],
                axis=1,
            )
        # Single sample: return (nqoi, 1)
        return self._physics.residual(samples, self._time).reshape(-1, 1)

    def jacobian(self, sample):
        # sample shape: (nvars, 1), return (nqoi, nvars)
        if sample.ndim == 2:
            sample = sample[:, 0]
        return self._physics.jacobian(sample, self._time)


class TestManufacturedShallowWave1D(Generic[Array], unittest.TestCase):
    """Test 1D Shallow Wave physics with manufactured solutions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_steady_shallow_wave_residual(self):
        """Test steady shallow wave residual with manufactured solution."""
        bkd = self.bkd()
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        # Polynomial solution ensuring positive depth
        # h = 2 + 0.3*(1 - x²) ensures h > 0 everywhere
        # uh = 0.5*(1 - x²) gives zero momentum at boundaries
        man_sol = ManufacturedShallowWave(
            nvars=1,
            depth_str="2 + 0.3*(1 - x**2)",
            mom_strs=["0.5*(1 - x**2)"],
            bed_str="0.1*x",
            bkd=bkd,
            oned=True,
        )

        # Get manufactured solution values
        sol = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))
        bed = man_sol.functions["bed"](nodes.reshape(1, -1))

        h_exact = sol[:, 0]
        hu_exact = sol[:, 1]

        # Create physics with manufactured forcing for all components
        all_forcing = bkd.hstack([forcing[:, 0], forcing[:, 1]])
        physics = ShallowWavePhysics(
            basis, bkd, bed=bed.flatten(), g=9.81, forcing=lambda t: all_forcing
        )

        # Set Dirichlet BCs at boundaries for both h and hu
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)

        bc_h_left = constant_dirichlet_bc(bkd, left_idx, float(h_exact[int(left_idx)]))
        bc_h_right = constant_dirichlet_bc(
            bkd, right_idx, float(h_exact[int(right_idx)])
        )
        bc_hu_left = constant_dirichlet_bc(
            bkd, left_idx + npts, float(hu_exact[int(left_idx)])
        )
        bc_hu_right = constant_dirichlet_bc(
            bkd, right_idx + npts, float(hu_exact[int(right_idx)])
        )
        physics.set_boundary_conditions(
            [bc_h_left, bc_h_right, bc_hu_left, bc_hu_right]
        )

        # Create state and compute residual
        state_exact = bkd.hstack([h_exact, hu_exact])
        residual = physics.residual(state_exact, 0.0)

        # Apply BCs
        jac = physics.jacobian(state_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, jac, state_exact, 0.0
        )

        # Check all interior residuals (both continuity and momentum)
        boundary_indices = {0, npts - 1, npts, 2 * npts - 1}
        interior = [i for i in range(2 * npts) if i not in boundary_indices]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior])

        bkd.assert_allclose(
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-8
        )

    def test_steady_shallow_wave_jacobian(self):
        """Test Shallow Wave Jacobian via derivative checker."""
        bkd = self.bkd()
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        man_sol = ManufacturedShallowWave(
            nvars=1,
            depth_str="2 + 0.3*(1 - x**2)",
            mom_strs=["0.5*(1 - x**2)"],
            bed_str="0.1*x",
            bkd=bkd,
            oned=True,
        )

        sol = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))
        bed = man_sol.functions["bed"](nodes.reshape(1, -1))

        all_forcing = bkd.hstack([forcing[:, 0], forcing[:, 1]])
        physics = ShallowWavePhysics(
            basis, bkd, bed=bed.flatten(), g=9.81, forcing=lambda t: all_forcing
        )

        state_exact = bkd.hstack([sol[:, 0], sol[:, 1]])
        wrapper = PhysicsDerivativeWrapper(physics)
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(state_exact.reshape(-1, 1))
        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-5)

    def test_flat_bed(self):
        """Test with flat bed topography."""
        bkd = self.bkd()
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        man_sol = ManufacturedShallowWave(
            nvars=1,
            depth_str="2 + 0.3*(1 - x**2)",
            mom_strs=["0.5*(1 - x**2)"],
            bed_str="0.0",  # Flat bed
            bkd=bkd,
            oned=True,
        )

        sol = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))
        bed = man_sol.functions["bed"](nodes.reshape(1, -1))

        h_exact = sol[:, 0]
        hu_exact = sol[:, 1]

        all_forcing = bkd.hstack([forcing[:, 0], forcing[:, 1]])
        physics = ShallowWavePhysics(
            basis, bkd, bed=bed.flatten(), g=9.81, forcing=lambda t: all_forcing
        )

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_h_left = constant_dirichlet_bc(bkd, left_idx, float(h_exact[int(left_idx)]))
        bc_h_right = constant_dirichlet_bc(
            bkd, right_idx, float(h_exact[int(right_idx)])
        )
        bc_hu_left = constant_dirichlet_bc(
            bkd, left_idx + npts, float(hu_exact[int(left_idx)])
        )
        bc_hu_right = constant_dirichlet_bc(
            bkd, right_idx + npts, float(hu_exact[int(right_idx)])
        )
        physics.set_boundary_conditions(
            [bc_h_left, bc_h_right, bc_hu_left, bc_hu_right]
        )

        state_exact = bkd.hstack([h_exact, hu_exact])
        residual = physics.residual(state_exact, 0.0)
        jac = physics.jacobian(state_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, jac, state_exact, 0.0
        )

        # Check all interior residuals (both continuity and momentum)
        boundary_indices = {0, npts - 1, npts, 2 * npts - 1}
        interior = [i for i in range(2 * npts) if i not in boundary_indices]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior])

        bkd.assert_allclose(
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-8
        )

    def test_quiescent_state(self):
        """Test quiescent state: h + b = const, u = 0 gives zero residual."""
        bkd = self.bkd()
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        # Quiescent state: h + b = constant, u = 0
        # This is a physical equilibrium that should give zero residual
        surface_level = 2.0
        bed = 0.5 * nodes  # Sloped bed
        h = surface_level - bed  # Depth that gives flat surface
        hu = bkd.zeros_like(h)  # Zero momentum

        physics = ShallowWavePhysics(basis, bkd, bed=bed, g=9.81)

        state = bkd.hstack([h, hu])
        residual = physics.residual(state, 0.0)

        # Both continuity and momentum residuals should be zero
        bkd.assert_allclose(residual, bkd.zeros_like(residual), atol=1e-10)


class TestShallowWave1DParameterized(ParametrizedTestCase):
    """Parameterized 1D Shallow Wave residual tests."""

    def bkd(self):
        return NumpyBkd()

    @parametrize(
        "name,depth_str,mom_str,bed_str,npts",
        [
            ("quadratic_sloped", "2 + 0.3*(1 - x**2)", "0.5*(1 - x**2)", "0.1*x", 20),
            ("quadratic_flat", "2 + 0.3*(1 - x**2)", "0.5*(1 - x**2)", "0.0", 20),
            (
                "quadratic_curved_bed",
                "2 + 0.3*(1 - x**2)",
                "0.5*(1 - x**2)",
                "0.05*x**2",
                20,
            ),
            ("quartic_depth", "2 + 0.2*(1 - x**2)**2", "0.3*(1 - x**2)", "0.1*x", 25),
            ("higher_depth", "3 + 0.5*(1 - x**2)", "0.8*(1 - x**2)", "0.1*x", 20),
        ],
    )
    def test_shallow_wave_1d_momentum_residual(
        self, name, depth_str, mom_str, bed_str, npts
    ):
        """Test 1D Shallow Wave momentum residual for parameterized cases."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        man_sol = ManufacturedShallowWave(
            nvars=1,
            depth_str=depth_str,
            mom_strs=[mom_str],
            bed_str=bed_str,
            bkd=bkd,
            oned=True,
        )

        sol = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))
        bed = man_sol.functions["bed"](nodes.reshape(1, -1))

        h_exact = sol[:, 0]
        hu_exact = sol[:, 1]

        all_forcing = bkd.hstack([forcing[:, 0], forcing[:, 1]])
        physics = ShallowWavePhysics(
            basis, bkd, bed=bed.flatten(), g=9.81, forcing=lambda t: all_forcing
        )

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_h_left = constant_dirichlet_bc(bkd, left_idx, float(h_exact[int(left_idx)]))
        bc_h_right = constant_dirichlet_bc(
            bkd, right_idx, float(h_exact[int(right_idx)])
        )
        bc_hu_left = constant_dirichlet_bc(
            bkd, left_idx + npts, float(hu_exact[int(left_idx)])
        )
        bc_hu_right = constant_dirichlet_bc(
            bkd, right_idx + npts, float(hu_exact[int(right_idx)])
        )
        physics.set_boundary_conditions(
            [bc_h_left, bc_h_right, bc_hu_left, bc_hu_right]
        )

        state_exact = bkd.hstack([h_exact, hu_exact])
        residual = physics.residual(state_exact, 0.0)
        jac = physics.jacobian(state_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, jac, state_exact, 0.0
        )

        # Check all interior residuals (both continuity and momentum)
        boundary_indices = {0, npts - 1, npts, 2 * npts - 1}
        interior = [i for i in range(2 * npts) if i not in boundary_indices]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior])

        bkd.assert_allclose(
            interior_residual,
            bkd.zeros(interior_residual.shape),
            atol=1e-7,
        )

    @parametrize(
        "name,depth_str,mom_str,bed_str,npts",
        [
            ("jacobian_basic", "2 + 0.3*(1 - x**2)", "0.5*(1 - x**2)", "0.1*x", 12),
            ("jacobian_flat_bed", "2 + 0.3*(1 - x**2)", "0.5*(1 - x**2)", "0.0", 12),
            (
                "jacobian_high_depth",
                "3 + 0.5*(1 - x**2)",
                "0.3*(1 - x**2)",
                "0.1*x",
                12,
            ),
        ],
    )
    def test_shallow_wave_1d_jacobian(self, name, depth_str, mom_str, bed_str, npts):
        """Test 1D Shallow Wave Jacobian via DerivativeChecker."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        man_sol = ManufacturedShallowWave(
            nvars=1,
            depth_str=depth_str,
            mom_strs=[mom_str],
            bed_str=bed_str,
            bkd=bkd,
            oned=True,
        )

        sol = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))
        bed = man_sol.functions["bed"](nodes.reshape(1, -1))

        all_forcing = bkd.hstack([forcing[:, 0], forcing[:, 1]])
        physics = ShallowWavePhysics(
            basis, bkd, bed=bed.flatten(), g=9.81, forcing=lambda t: all_forcing
        )

        state_exact = bkd.hstack([sol[:, 0], sol[:, 1]])
        wrapper = PhysicsDerivativeWrapper(physics)
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(state_exact.reshape(-1, 1))
        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-5)


# Concrete backend implementations
class TestManufacturedShallowWave1DNumpy(TestManufacturedShallowWave1D[NDArray[Any]]):
    """NumPy backend tests for 1D Shallow Wave."""

    __test__ = True

    def bkd(self):
        return NumpyBkd()


class TestManufacturedShallowWave1DTorch(TestManufacturedShallowWave1D[torch.Tensor]):
    """PyTorch backend tests for 1D Shallow Wave."""

    __test__ = True

    def bkd(self):
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


if __name__ == "__main__":
    unittest.main()
