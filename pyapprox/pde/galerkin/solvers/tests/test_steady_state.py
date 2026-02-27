"""Tests for SteadyStateSolver."""


import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.pde.galerkin.physics import LinearAdvectionDiffusionReaction
from pyapprox.pde.galerkin.solvers import SteadyStateSolver


class TestSteadyStateSolverBase:
    """Base test class for SteadyStateSolver."""
    def test_linear_solve_with_forcing(self, numpy_bkd) -> None:
        """Test linear steady-state solve with forcing term.

        Uses reaction term to make the problem non-singular (pure diffusion
        with natural BCs has a singular stiffness matrix).
        """
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=20, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        # Constant forcing
        def forcing(x):
            return np.ones(x.shape[1])

        # Add reaction term to make problem well-posed (non-singular)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis,
            diffusivity=1.0,
            reaction=1.0,  # Makes stiffness matrix non-singular
            forcing=forcing,
            bkd=bkd,
        )

        solver = SteadyStateSolver(physics, tol=1e-12)
        result = solver.solve_linear()

        assert result.converged
        assert result.residual_norm < 1e-10

    def test_newton_solve_converges(self, numpy_bkd) -> None:
        """Test Newton solve converges for linear problem.

        Uses reaction term to make the problem non-singular.
        """
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        def forcing(x):
            return np.sin(np.pi * x[0])

        # Add reaction term to make problem well-posed
        physics = LinearAdvectionDiffusionReaction(
            basis=basis,
            diffusivity=0.1,
            reaction=0.5,  # Makes stiffness matrix non-singular
            forcing=forcing,
            bkd=bkd,
        )

        solver = SteadyStateSolver(physics, tol=1e-10)

        # Start from zero (use float64 for consistency with skfem)
        u_guess = bkd.asarray(np.zeros(physics.nstates(), dtype=np.float64))
        result = solver.solve(u_guess)

        assert result.converged
        assert result.residual_norm < 1e-10
        # For linear problem, should converge in 1 iteration
        assert result.iterations == 1

    def test_solver_result_attributes(self, numpy_bkd) -> None:
        """Test SolverResult has expected attributes."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        # Add reaction term to make problem well-posed
        physics = LinearAdvectionDiffusionReaction(
            basis=basis,
            diffusivity=1.0,
            reaction=1.0,
            forcing=lambda x: np.ones(x.shape[1]),
            bkd=bkd,
        )

        solver = SteadyStateSolver(physics)
        result = solver.solve_linear()

        # Check all attributes exist
        assert hasattr(result, "solution")
        assert hasattr(result, "converged")
        assert hasattr(result, "iterations")
        assert hasattr(result, "residual_norm")
        assert hasattr(result, "message")

    def test_2d_steady_state(self, numpy_bkd) -> None:
        """Test steady-state solve in 2D.

        Uses reaction term to make the problem non-singular.
        """
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5, ny=5, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd
        )
        basis = LagrangeBasis(mesh, degree=1)

        # Add reaction term to make problem well-posed
        physics = LinearAdvectionDiffusionReaction(
            basis=basis,
            diffusivity=0.1,
            reaction=0.5,
            forcing=lambda x: np.ones(x.shape[1]),
            bkd=bkd,
        )

        solver = SteadyStateSolver(physics, tol=1e-10)
        result = solver.solve_linear()

        assert result.converged
        assert result.residual_norm < 1e-8

    def test_solver_with_zero_forcing(self, numpy_bkd) -> None:
        """Test that zero forcing with reaction gives zero solution.

        With reaction term r*u, zero forcing leads to u=0 as the unique solution.
        """
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        # No forcing but with reaction to make problem non-singular
        physics = LinearAdvectionDiffusionReaction(
            basis=basis,
            diffusivity=1.0,
            reaction=1.0,  # Makes stiffness matrix non-singular
            bkd=bkd,
        )

        solver = SteadyStateSolver(physics)

        # Use float64 for consistency with skfem
        u_guess = bkd.asarray(np.zeros(physics.nstates(), dtype=np.float64))
        result = solver.solve(u_guess)

        # Solution should be zero (or very close)
        u_np = bkd.to_numpy(result.solution)
        assert np.linalg.norm(u_np) < 1e-10


# Try to import torch for dual-backend testing
