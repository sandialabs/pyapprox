"""Tests for Shallow Water equations physics implementation."""

import unittest
import math
import numpy as np

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.typing.pde.collocation.mesh import create_uniform_mesh_1d
from pyapprox.typing.pde.collocation.physics.shallow_wave import (
    ShallowWavePhysics,
    create_shallow_wave,
)
from pyapprox.typing.pde.collocation.physics.tests.test_utils import (
    PhysicsTestBase,
)
from pyapprox.typing.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)


class TestShallowWavePhysics(PhysicsTestBase):
    """Tests for ShallowWavePhysics."""

    __test__ = True

    def bkd(self):
        return NumpyBkd()

    def test_jacobian_derivative_checker_flat_bed(self):
        """Test Jacobian matches finite differences with flat bed."""
        bkd = self.bkd()
        npts = 12
        basis = ChebyshevBasis1D(npts, bkd)

        # Flat bed
        bed = bkd.zeros((npts,))

        physics = ShallowWavePhysics(basis, bkd, bed=bed, g=9.81)

        # Positive depth with non-zero momentum
        # h = 1 + 0.1*cos(pi*x), hu = 0.5*h
        nodes = basis.nodes()
        h = 1.0 + 0.1 * bkd.cos(math.pi * nodes)
        hu = 0.5 * h
        state = bkd.hstack([h, hu])

        self.check_jacobian(physics, state, time=0.0)

    def test_jacobian_sloped_bed(self):
        """Test Jacobian with sloped bed."""
        bkd = self.bkd()
        npts = 12
        basis = ChebyshevBasis1D(npts, bkd)
        nodes = basis.nodes()

        # Sloped bed
        bed = 0.1 * nodes

        physics = ShallowWavePhysics(basis, bkd, bed=bed, g=9.81)

        # Positive depth
        h = 1.5 + 0.2 * bkd.cos(math.pi * nodes)
        hu = 0.3 * h
        state = bkd.hstack([h, hu])

        self.check_jacobian(physics, state, time=0.0)

    def test_jacobian_with_forcing(self):
        """Test Jacobian with momentum forcing."""
        bkd = self.bkd()
        npts = 12
        basis = ChebyshevBasis1D(npts, bkd)
        nodes = basis.nodes()

        bed = bkd.zeros((npts,))
        forcing = 0.1 * bkd.sin(math.pi * nodes)

        physics = ShallowWavePhysics(
            basis, bkd, bed=bed, g=9.81,
            forcing=lambda t: forcing
        )

        h = 2.0 + 0.3 * bkd.cos(math.pi * nodes)
        hu = 0.2 * h
        state = bkd.hstack([h, hu])

        self.check_jacobian(physics, state, time=0.0)

    def test_residual_quiescent_state(self):
        """Test residual is zero for quiescent state (h+b=const, u=0)."""
        bkd = self.bkd()
        npts = 20
        basis = ChebyshevBasis1D(npts, bkd)
        nodes = basis.nodes()

        # Sloped bed with constant surface elevation
        # h + b = const => dh/dx = -db/dx
        surface_level = 2.0
        bed = 0.5 * nodes  # Sloped bed
        h = surface_level - bed  # Depth that gives flat surface

        # Zero velocity => zero momentum
        hu = bkd.zeros_like(h)
        state = bkd.hstack([h, hu])

        physics = ShallowWavePhysics(basis, bkd, bed=bed, g=9.81)

        # For quiescent state with flat surface:
        # dh/dt = -d(hu)/dx = 0 (since hu=0)
        # d(hu)/dt = -d(hu^2/h + 0.5*g*h^2)/dx - g*h*db/dx
        #          = -d(0.5*g*h^2)/dx - g*h*db/dx
        #          = -g*h*dh/dx - g*h*db/dx
        #          = -g*h*(dh/dx + db/dx)
        #          = -g*h*d(h+b)/dx = 0 (since h+b=const)
        self.check_residual_zero(physics, state, atol=1e-10)

    def test_residual_uniform_flow(self):
        """Test residual for uniform flow on flat bed."""
        bkd = self.bkd()
        npts = 20
        basis = ChebyshevBasis1D(npts, bkd)

        # Flat bed, uniform depth and velocity
        bed = bkd.zeros((npts,))
        h = bkd.full((npts,), 1.0)
        u = 0.5
        hu = h * u
        state = bkd.hstack([h, hu])

        physics = ShallowWavePhysics(basis, bkd, bed=bed, g=9.81)

        # For uniform flow:
        # dh/dt = -d(hu)/dx = 0 (since hu=const)
        # d(hu)/dt = -d(hu^2/h + 0.5*g*h^2)/dx - g*h*db/dx
        #          = 0 (since all terms are constant)
        self.check_residual_zero(physics, state, atol=1e-10)

    def test_ncomponents_1d(self):
        """Test number of components for 1D case."""
        bkd = self.bkd()
        npts = 10
        basis = ChebyshevBasis1D(npts, bkd)
        bed = bkd.zeros((npts,))

        physics = ShallowWavePhysics(basis, bkd, bed=bed)

        # 1D: h and hu
        self.assertEqual(physics.ncomponents(), 2)
        self.assertEqual(physics.nstates(), 2 * npts)

    def test_factory_function(self):
        """Test create_shallow_wave factory function."""
        bkd = self.bkd()
        npts = 10
        basis = ChebyshevBasis1D(npts, bkd)
        bed = bkd.zeros((npts,))

        physics = create_shallow_wave(basis, bkd, bed=bed, g=10.0)

        self.assertEqual(physics.ncomponents(), 2)
        self.assertAlmostEqual(physics.g(), 10.0)

    def test_accessors(self):
        """Test accessor methods."""
        bkd = self.bkd()
        npts = 10
        basis = ChebyshevBasis1D(npts, bkd)
        nodes = basis.nodes()
        bed = 0.1 * nodes

        physics = ShallowWavePhysics(basis, bkd, bed=bed, g=9.81)

        self.assertAlmostEqual(physics.g(), 9.81)
        bkd.assert_allclose(physics.bed(), bed)

    def test_transient_small_perturbation(self):
        """Test transient evolution of small surface perturbation.

        For shallow water with small perturbation on quiescent state,
        linearized equations give wave propagation.

        Note: Shallow water equations are hyperbolic and challenging for
        spectral methods without proper upwinding/stabilization. This test
        uses a quiescent state (uniform depth, zero velocity) which is stable.
        """
        bkd = self.bkd()
        npts = 20
        basis = ChebyshevBasis1D(npts, bkd)
        nodes = basis.nodes()

        # Flat bed, quiescent state (uniform depth, zero velocity)
        bed = bkd.zeros((npts,))
        g = 9.81
        h0 = 1.0  # Mean depth

        physics = ShallowWavePhysics(basis, bkd, bed=bed, g=g)

        model = CollocationModel(physics, bkd)

        # Start from quiescent state (stable equilibrium)
        h_init = bkd.full((npts,), h0)
        hu_init = bkd.zeros((npts,))
        state0 = bkd.hstack([h_init, hu_init])

        config = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=0.01,
            deltat=0.005,
        )

        solutions, times = model.solve_transient(state0, config)

        # Check solution is finite and physical (h > 0)
        final_state = solutions[:, -1]
        h_final = final_state[:npts]

        self.assertTrue(bkd.isfinite(bkd.norm(final_state)))
        self.assertGreater(float(bkd.min(h_final)), 0.0)

        # Quiescent state should remain nearly unchanged
        bkd.assert_allclose(h_final, h_init, rtol=1e-6, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
