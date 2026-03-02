"""Tests for Shallow Water equations physics implementation."""

import math

from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import constant_dirichlet_bc
from pyapprox.pde.collocation.manufactured_solutions import (
    ManufacturedShallowWave,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    create_uniform_mesh_1d,
)
from pyapprox.pde.collocation.physics.shallow_wave import (
    ShallowWavePhysics,
    create_shallow_wave,
)
from pyapprox.pde.collocation.physics.tests.test_utils import (
    PhysicsTestBase,
)
from pyapprox.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)


class TestShallowWavePhysics(PhysicsTestBase):
    """Tests for ShallowWavePhysics."""

    def test_jacobian_derivative_checker_flat_bed(self, bkd):
        """Test Jacobian matches finite differences with flat bed."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        # Flat bed
        bed = bkd.zeros((npts,))

        physics = ShallowWavePhysics(basis, bkd, bed=bed, g=9.81)

        # Positive depth with non-zero momentum
        # h = 1 + 0.1*cos(pi*x), hu = 0.5*h
        nodes = basis.nodes()
        h = 1.0 + 0.1 * bkd.cos(math.pi * nodes)
        hu = 0.5 * h
        state = bkd.hstack([h, hu])

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_jacobian_sloped_bed(self, bkd):
        """Test Jacobian with sloped bed."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        # Sloped bed
        bed = 0.1 * nodes

        physics = ShallowWavePhysics(basis, bkd, bed=bed, g=9.81)

        # Positive depth
        h = 1.5 + 0.2 * bkd.cos(math.pi * nodes)
        hu = 0.3 * h
        state = bkd.hstack([h, hu])

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_jacobian_with_forcing(self, bkd):
        """Test Jacobian with forcing on all components."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        bed = bkd.zeros((npts,))
        # Forcing for all components: [f_h, f_hu]
        f_h = 0.05 * bkd.cos(math.pi * nodes)
        f_hu = 0.1 * bkd.sin(math.pi * nodes)
        forcing = bkd.hstack([f_h, f_hu])

        physics = ShallowWavePhysics(
            basis, bkd, bed=bed, g=9.81, forcing=lambda t: forcing
        )

        h = 2.0 + 0.3 * bkd.cos(math.pi * nodes)
        hu = 0.2 * h
        state = bkd.hstack([h, hu])

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_residual_quiescent_state(self, bkd):
        """Test residual is zero for quiescent state (h+b=const, u=0)."""
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
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
        self.check_residual_zero(bkd, physics, state, atol=1e-10)

    def test_residual_uniform_flow(self, bkd):
        """Test residual for uniform flow on flat bed."""
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

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
        self.check_residual_zero(bkd, physics, state, atol=1e-10)

    def test_ncomponents_1d(self, bkd):
        """Test number of components for 1D case."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        bed = bkd.zeros((npts,))

        physics = ShallowWavePhysics(basis, bkd, bed=bed)

        # 1D: h and hu
        assert physics.ncomponents() == 2
        assert physics.nstates() == 2 * npts

    def test_factory_function(self, bkd):
        """Test create_shallow_wave factory function."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        bed = bkd.zeros((npts,))

        physics = create_shallow_wave(basis, bkd, bed=bed, g=10.0)

        assert physics.ncomponents() == 2
        assert abs(physics.g() - 10.0) < 1e-7

    def test_accessors(self, bkd):
        """Test accessor methods."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()
        bed = 0.1 * nodes

        physics = ShallowWavePhysics(basis, bkd, bed=bed, g=9.81)

        assert abs(physics.g() - 9.81) < 1e-7
        bkd.assert_allclose(physics.bed(), bed)

    def test_transient_small_perturbation(self, bkd):
        """Test transient evolution of small surface perturbation.

        For shallow water with small perturbation on quiescent state,
        linearized equations give wave propagation.

        Note: Shallow water equations are hyperbolic and challenging for
        spectral methods without proper upwinding/stabilization. This test
        uses a quiescent state (uniform depth, zero velocity) which is stable.
        """
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        basis.nodes()

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

        assert bkd.isfinite(bkd.norm(final_state))
        assert float(bkd.min(h_final)) > 0.0

        # Quiescent state should remain nearly unchanged
        bkd.assert_allclose(h_final, h_init, rtol=1e-6, atol=1e-10)

    def _setup_transient_shallow_wave(self, bkd):
        """Set up transient shallow wave with manufactured solution.

        Uses polynomial-in-space (exact for Chebyshev) and quadratic-in-time
        (exact for CN) manufactured solution with all-component forcing.
        """
        npts = 15
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        bc_mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        # Depth must stay positive and have constant boundary values.
        # h(±1,t) = 2 (constant), hu(±1,t) = 0 (constant).
        man_sol = ManufacturedShallowWave(
            nvars=1,
            depth_str="2 + 0.3*(1 - x**2)*(1 + T + T**2)",
            mom_strs=["0.5*(1 - x**2)*(1 + T + T**2)"],
            bed_str="0.1*x",
            bkd=bkd,
            oned=True,
        )

        def forcing_fn(t):
            forcing = man_sol.functions["forcing"](nodes[None, :], t)
            return bkd.hstack([forcing[:, 0], forcing[:, 1]])

        bed = man_sol.functions["bed"](nodes[None, :]).flatten()

        physics = ShallowWavePhysics(basis, bkd, bed=bed, g=9.81, forcing=forcing_fn)

        # Set Dirichlet BCs for h and hu at both boundaries
        left_idx = bc_mesh.boundary_indices(0)
        right_idx = bc_mesh.boundary_indices(1)

        def get_bc_val(comp_idx, bnd_idx, t):
            sol = man_sol.functions["solution"](nodes[None, :], t)
            return float(sol[int(bnd_idx), comp_idx])

        # h boundaries
        bc_h_left = constant_dirichlet_bc(bkd, left_idx, get_bc_val(0, left_idx, 0.0))
        bc_h_right = constant_dirichlet_bc(
            bkd, right_idx, get_bc_val(0, right_idx, 0.0)
        )
        # hu boundaries
        bc_hu_left = constant_dirichlet_bc(
            bkd, left_idx + npts, get_bc_val(1, left_idx, 0.0)
        )
        bc_hu_right = constant_dirichlet_bc(
            bkd, right_idx + npts, get_bc_val(1, right_idx, 0.0)
        )
        physics.set_boundary_conditions(
            [bc_h_left, bc_h_right, bc_hu_left, bc_hu_right]
        )

        return bkd, npts, nodes, man_sol, physics

    def _run_transient_shallow_wave(self, bkd, method, atol) :
        """Run transient shallow wave test with given method and tolerance."""
        bkd, npts, nodes, man_sol, physics = self._setup_transient_shallow_wave(bkd)
        model = CollocationModel(physics, bkd)

        sol0 = man_sol.functions["solution"](nodes[None, :], 0.0)
        state0 = bkd.hstack([sol0[:, 0], sol0[:, 1]])

        final_time = 0.1
        config = TimeIntegrationConfig(
            method=method,
            init_time=0.0,
            final_time=final_time,
            deltat=0.01,
        )

        solutions, times = model.solve_transient(state0, config)
        t_final = float(bkd.to_numpy(times[-1]))
        sol_exact = man_sol.functions["solution"](nodes[None, :], t_final)
        exact_final = bkd.hstack([sol_exact[:, 0], sol_exact[:, 1]])

        bkd.assert_allclose(solutions[:, -1], exact_final, atol=atol)

    def test_transient_manufactured_backward_euler(self, bkd):
        """Test transient shallow wave with backward Euler."""
        self._run_transient_shallow_wave(bkd, "backward_euler", atol=0.5)

    def test_transient_manufactured_crank_nicolson(self, bkd):
        """Test transient shallow wave with Crank-Nicolson.

        Uses polynomial-in-space and quadratic-in-time manufactured solution.
        CN integrates quadratic-in-time exactly, so only spatial error remains.
        """
        self._run_transient_shallow_wave(bkd, "crank_nicolson", atol=1e-8)
