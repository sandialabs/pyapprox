"""Tests for SteadyStateSolver."""


import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.pde.constitutive.coefficient_functions import TimeIndependent
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.pde.galerkin.physics import LinearAdvectionDiffusionReaction
from pyapprox.pde.galerkin.solvers import SteadyStateSolver
from pyapprox.util.backends.numpy import NumpyBkd


class TestSteadyStateSolverBase:
    """Base test class for SteadyStateSolver."""
    def test_linear_solve_with_forcing(self, numpy_bkd: NumpyBkd) -> None:
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
            forcing=TimeIndependent(forcing),
            bkd=bkd,
        )

        solver = SteadyStateSolver(compose_galerkin_system(physics).steady(), tol=1e-12)
        result = solver.solve()

        assert result.converged
        assert result.residual_norm < 1e-10

    def test_newton_solve_converges(self, numpy_bkd: NumpyBkd) -> None:
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
            forcing=TimeIndependent(forcing),
            bkd=bkd,
        )

        solver = SteadyStateSolver(compose_galerkin_system(physics).steady(), tol=1e-10)

        # Start from zero (use float64 for consistency with skfem)
        u_guess = bkd.asarray(np.zeros(physics.nstates(), dtype=np.float64))
        result = solver.solve(u_guess)

        assert result.converged
        assert result.residual_norm < 1e-10
        # For linear problem, should converge in 1 iteration
        assert result.iterations == 1

    def test_solver_result_attributes(self, numpy_bkd: NumpyBkd) -> None:
        """Test the result has the expected attributes."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        # Add reaction term to make problem well-posed
        physics = LinearAdvectionDiffusionReaction(
            basis=basis,
            diffusivity=1.0,
            reaction=1.0,
            forcing=TimeIndependent(lambda x: np.ones(x.shape[1])),
            bkd=bkd,
        )

        solver = SteadyStateSolver(compose_galerkin_system(physics).steady())
        result = solver.solve()

        # Check all attributes exist
        assert hasattr(result, "solution")
        assert hasattr(result, "converged")
        assert hasattr(result, "iterations")
        assert hasattr(result, "residual_norm")
        assert hasattr(result, "message")

    def test_2d_steady_state(self, numpy_bkd: NumpyBkd) -> None:
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
            forcing=TimeIndependent(lambda x: np.ones(x.shape[1])),
            bkd=bkd,
        )

        solver = SteadyStateSolver(compose_galerkin_system(physics).steady(), tol=1e-10)
        result = solver.solve()

        assert result.converged
        assert result.residual_norm < 1e-8

    def test_solver_with_zero_forcing(self, numpy_bkd: NumpyBkd) -> None:
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

        solver = SteadyStateSolver(compose_galerkin_system(physics).steady())

        # Use float64 for consistency with skfem
        u_guess = bkd.asarray(np.zeros(physics.nstates(), dtype=np.float64))
        result = solver.solve(u_guess)

        # Solution should be zero (or very close)
        u_np = bkd.to_numpy(result.solution)
        assert np.linalg.norm(u_np) < 1e-10


# Try to import torch for dual-backend testing


_NSTATES = 20


def _affine_system() -> tuple[np.ndarray, np.ndarray]:
    """A 20 x 20 SPD system scaled by 1e12: after an exact solve its
    residual stays near 1e-3 (round-off), never below 1e-10."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(_NSTATES, _NSTATES))
    return (
        1e12 * (A @ A.T + _NSTATES * np.eye(_NSTATES)),
        1e12 * rng.normal(size=_NSTATES),
    )


class _AffineSteadyOperator:
    """``F(u) = b - K u`` from ``_affine_system``, declared linear or
    not."""

    def __init__(self, bkd: NumpyBkd, linear: bool) -> None:
        self._bkd = bkd
        self._linear = linear
        K, b = _affine_system()
        self._K = bkd.asarray(K)
        self._b = bkd.asarray(b)

    def bkd(self) -> NumpyBkd:
        return self._bkd

    def nstates(self) -> int:
        return _NSTATES

    def steady_residual(self, state: np.ndarray) -> np.ndarray:
        return self._b - self._K @ state

    def steady_jacobian(self, state: np.ndarray) -> np.ndarray:
        return -self._K

    def is_linear(self) -> bool:
        return self._linear


class TestSteadyStateSolverStopping:
    def test_declared_linear_takes_one_step(self, numpy_bkd: NumpyBkd) -> None:
        """One exact step, accepted without the absolute test; the
        residual norm is still reported."""
        bkd = numpy_bkd
        result = SteadyStateSolver(_AffineSteadyOperator(bkd, True)).solve(
            bkd.zeros((_NSTATES,))
        )
        assert result.converged
        assert result.iterations == 1
        assert result.residual_norm is not None
        K, b = _affine_system()
        bkd.assert_allclose(
            result.solution, bkd.asarray(np.linalg.solve(K, b)), rtol=1e-10
        )

    def test_rtol_reaches_round_off(self, numpy_bkd: NumpyBkd) -> None:
        """Not declared linear: Newton with only the absolute tolerance
        cannot stop at round-off of order 1e-3; rtol can."""
        bkd = numpy_bkd
        operator = _AffineSteadyOperator(bkd, False)
        guess = bkd.zeros((_NSTATES,))
        assert not SteadyStateSolver(operator, tol=1e-10, max_iter=5).solve(
            guess
        ).converged
        result = SteadyStateSolver(
            operator, tol=1e-10, max_iter=5, rtol=1e-12
        ).solve(guess)
        assert result.converged
