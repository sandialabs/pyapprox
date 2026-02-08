"""Tests for Shallow Ice physics implementation."""

import unittest
import math
import numpy as np

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.typing.pde.collocation.mesh import (
    create_uniform_mesh_1d,
    TransformedMesh1D,
)
from pyapprox.typing.pde.collocation.boundary import (
    constant_dirichlet_bc,
)
from pyapprox.typing.pde.collocation.physics.shallow_ice import (
    ShallowIcePhysics,
    create_shallow_ice,
)
from pyapprox.typing.pde.collocation.physics.tests.test_utils import (
    PhysicsTestBase,
    PhysicsNewtonResidual,
)
from pyapprox.typing.optimization.rootfinding.newton import NewtonSolver
from pyapprox.typing.pde.collocation.manufactured_solutions.shallow_ice import (
    ManufacturedShallowIce,
)
from pyapprox.typing.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)


class TestShallowIcePhysics(PhysicsTestBase):
    """Tests for ShallowIcePhysics."""

    __test__ = True

    def bkd(self):
        return NumpyBkd()

    def test_jacobian_derivative_checker(self):
        """Test Jacobian matches finite differences using DerivativeChecker."""
        bkd = self.bkd()
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        # Flat bed
        bed = bkd.zeros((npts,))

        physics = ShallowIcePhysics(
            basis, bkd, bed=bed, friction=1e6, A=1e-16, rho=917.0
        )

        # Positive ice thickness (physically meaningful)
        state = 100.0 + 50.0 * (1.0 - nodes ** 2)

        self.check_jacobian(physics, state, time=0.0)

    def test_jacobian_sloped_bed(self):
        """Test Jacobian with sloped bed topography."""
        bkd = self.bkd()
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        # Sloped bed
        bed = 100.0 * nodes

        physics = ShallowIcePhysics(
            basis, bkd, bed=bed, friction=1e6, A=1e-16, rho=917.0
        )

        # Positive ice thickness
        state = 150.0 + 30.0 * (1.0 - nodes ** 2)

        self.check_jacobian(physics, state, time=0.0)

    def test_residual_at_manufactured_solution(self):
        """Verify residual is near zero at manufactured solution.

        Uses normalized parameters (A=1, rho=1) to avoid numerical issues
        with the highly nonlinear shallow ice equation. With glaciological
        parameters (A~1e-16, rho~917), the forcing terms are O(1e10) which
        causes numerical issues with spectral methods.
        """
        bkd = self.bkd()
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        # Normalized parameters for numerical stability
        A = 1.0
        rho = 1.0

        # Manufactured solution with smooth positive ice thickness
        man_sol = ManufacturedShallowIce(
            sol_str="2 + 0.5*(1 - x**2)",
            nvars=1,
            bed_str="0.1*x",  # small sloped bed
            friction_str="1.0",
            A=A,
            rho=rho,
            bkd=bkd,
            oned=True,
        )

        # Get forcing from manufactured solution
        # Manufactured: dH/dt - div(D*grad(s)) = f → f = -div(D*grad(s))
        # Physics:      residual = div(D*grad(s)) + f
        # For residual = 0: f = -div(D*grad(s)) which is what manufactured computes
        forcing = man_sol.functions["forcing"](nodes[None, :])
        if forcing.ndim == 2:
            forcing = forcing[:, 0]
        # NO negation needed - sign conventions match

        # Get bed elevation
        bed = man_sol.functions["bed"](nodes[None, :])
        if bed.ndim == 2:
            bed = bed[:, 0]

        physics = ShallowIcePhysics(
            basis, bkd, bed=bed, friction=1.0, A=A, rho=rho,
            forcing=lambda t: forcing
        )

        # Get exact solution
        exact_solution = man_sol.functions["solution"](nodes[None, :])
        if exact_solution.ndim == 2:
            exact_solution = exact_solution[:, 0]

        # Check residual is near zero at manufactured solution
        self.check_residual_zero(physics, exact_solution, atol=1e-6)

    def test_solve_steady_from_small_perturbation(self):
        """Verify Newton converges from small perturbation.

        Uses normalized parameters for numerical stability.
        Tests that NewtonSolver converges when starting close to solution.
        """
        bkd = self.bkd()
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        # Normalized parameters for numerical stability
        A = 1.0
        rho = 1.0

        # Manufactured solution
        man_sol = ManufacturedShallowIce(
            sol_str="2 + 0.5*(1 - x**2)",
            nvars=1,
            bed_str="0.1*x",
            friction_str="1.0",
            A=A,
            rho=rho,
            bkd=bkd,
            oned=True,
        )

        # Get forcing (no negation needed - sign conventions match)
        forcing = man_sol.functions["forcing"](nodes[None, :])
        if forcing.ndim == 2:
            forcing = forcing[:, 0]

        # Get bed elevation
        bed = man_sol.functions["bed"](nodes[None, :])
        if bed.ndim == 2:
            bed = bed[:, 0]

        physics = ShallowIcePhysics(
            basis, bkd, bed=bed, friction=1.0, A=A, rho=rho,
            forcing=lambda t: forcing
        )

        # Set boundary conditions
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)

        exact_solution = man_sol.functions["solution"](nodes[None, :])
        if exact_solution.ndim == 2:
            exact_solution = exact_solution[:, 0]

        bc_left = constant_dirichlet_bc(bkd, left_idx, float(exact_solution[left_idx]))
        bc_right = constant_dirichlet_bc(bkd, right_idx, float(exact_solution[right_idx]))
        physics.set_boundary_conditions([bc_left, bc_right])

        # Small perturbation from exact solution
        np.random.seed(42)
        perturbation = 0.01 * exact_solution * np.random.randn(npts)
        initial_guess = exact_solution + bkd.array(perturbation)

        # Solve with NewtonSolver
        residual_wrapper = PhysicsNewtonResidual(physics, time=0.0)
        solver = NewtonSolver(residual_wrapper)
        solver.set_options(maxiters=50, atol=1e-10, rtol=1e-10)
        solution = solver.solve(initial_guess)

        bkd.assert_allclose(solution, exact_solution, atol=1e-6)

    def test_factory_function(self):
        """Test create_shallow_ice factory function."""
        bkd = self.bkd()
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        bed = bkd.zeros((npts,))

        physics = create_shallow_ice(
            basis, bkd, bed=bed, friction=1e6, A=1e-16, rho=917.0
        )

        self.assertEqual(physics.ncomponents(), 1)
        self.assertEqual(physics.nstates(), npts)

    def _setup_transient_shallow_ice(self):
        """Set up transient shallow ice with manufactured solution.

        Uses polynomial-in-space (exact for Chebyshev) and quadratic-in-time
        (exact for CN). Boundary values are constant: H(±1,t)=2.
        Normalized parameters (A=1, rho=1) for numerical stability.
        """
        bkd = self.bkd()
        npts = 15
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        bc_mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        A = 1.0
        rho = 1.0

        man_sol = ManufacturedShallowIce(
            sol_str="2 + 0.5*(1 - x**2)*(1 + T + T**2)",
            nvars=1,
            bed_str="0.1*x",
            friction_str="1.0",
            A=A,
            rho=rho,
            bkd=bkd,
            oned=True,
        )

        def forcing_fn(t):
            return man_sol.functions["forcing"](nodes[None, :], t)

        bed = man_sol.functions["bed"](nodes[None, :])
        if bed.ndim == 2:
            bed = bed[:, 0]

        physics = ShallowIcePhysics(
            basis, bkd, bed=bed, friction=1.0, A=A, rho=rho,
            forcing=forcing_fn
        )

        exact_at_0 = man_sol.functions["solution"](nodes[None, :], 0.0)
        left_idx = bc_mesh.boundary_indices(0)
        right_idx = bc_mesh.boundary_indices(1)
        bc_left = constant_dirichlet_bc(
            bkd, left_idx, float(exact_at_0[int(left_idx)])
        )
        bc_right = constant_dirichlet_bc(
            bkd, right_idx, float(exact_at_0[int(right_idx)])
        )
        physics.set_boundary_conditions([bc_left, bc_right])

        return bkd, npts, nodes, man_sol, physics

    def _run_transient_shallow_ice(self, method, atol):
        """Run transient shallow ice test with given method and tolerance."""
        bkd, npts, nodes, man_sol, physics = (
            self._setup_transient_shallow_ice()
        )
        model = CollocationModel(physics, bkd)

        state0 = man_sol.functions["solution"](nodes[None, :], 0.0)

        final_time = 0.1
        config = TimeIntegrationConfig(
            method=method,
            init_time=0.0,
            final_time=final_time,
            deltat=0.01,
        )

        solutions, times = model.solve_transient(state0, config)
        t_final = float(bkd.to_numpy(times[-1]))
        exact_final = man_sol.functions["solution"](nodes[None, :], t_final)

        bkd.assert_allclose(solutions[:, -1], exact_final, atol=atol)

    def test_transient_manufactured_backward_euler(self):
        """Test transient shallow ice with backward Euler."""
        self._run_transient_shallow_ice("backward_euler", atol=0.1)

    def test_transient_manufactured_crank_nicolson(self):
        """Test transient shallow ice with Crank-Nicolson.

        Uses polynomial-in-space and quadratic-in-time manufactured solution.
        CN integrates quadratic-in-time exactly, so only spatial error remains.
        """
        self._run_transient_shallow_ice("crank_nicolson", atol=1e-8)


if __name__ == "__main__":
    unittest.main()
