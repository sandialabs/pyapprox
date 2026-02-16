"""Tests for SteadyStateSolver."""

import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.typing.pde.galerkin.basis import LagrangeBasis
from pyapprox.typing.pde.galerkin.physics import LinearAdvectionDiffusionReaction
from pyapprox.typing.pde.galerkin.solvers import SteadyStateSolver


class TestSteadyStateSolverBase(Generic[Array], unittest.TestCase):
    """Base test class for SteadyStateSolver."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self.bkd_inst = self.bkd()

    def test_linear_solve_with_forcing(self) -> None:
        """Test linear steady-state solve with forcing term.

        Uses reaction term to make the problem non-singular (pure diffusion
        with natural BCs has a singular stiffness matrix).
        """
        mesh = StructuredMesh1D(nx=20, bounds=(0.0, 1.0), bkd=self.bkd_inst)
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
            bkd=self.bkd_inst,
        )

        solver = SteadyStateSolver(physics, tol=1e-12)
        result = solver.solve_linear()

        self.assertTrue(result.converged)
        self.assertLess(result.residual_norm, 1e-10)

    def test_newton_solve_converges(self) -> None:
        """Test Newton solve converges for linear problem.

        Uses reaction term to make the problem non-singular.
        """
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        def forcing(x):
            return np.sin(np.pi * x[0])

        # Add reaction term to make problem well-posed
        physics = LinearAdvectionDiffusionReaction(
            basis=basis,
            diffusivity=0.1,
            reaction=0.5,  # Makes stiffness matrix non-singular
            forcing=forcing,
            bkd=self.bkd_inst,
        )

        solver = SteadyStateSolver(physics, tol=1e-10)

        # Start from zero (use float64 for consistency with skfem)
        u_guess = self.bkd_inst.asarray(np.zeros(physics.nstates(), dtype=np.float64))
        result = solver.solve(u_guess)

        self.assertTrue(result.converged)
        self.assertLess(result.residual_norm, 1e-10)
        # For linear problem, should converge in 1 iteration
        self.assertEqual(result.iterations, 1)

    def test_solver_result_attributes(self) -> None:
        """Test SolverResult has expected attributes."""
        mesh = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        # Add reaction term to make problem well-posed
        physics = LinearAdvectionDiffusionReaction(
            basis=basis,
            diffusivity=1.0,
            reaction=1.0,
            forcing=lambda x: np.ones(x.shape[1]),
            bkd=self.bkd_inst,
        )

        solver = SteadyStateSolver(physics)
        result = solver.solve_linear()

        # Check all attributes exist
        self.assertTrue(hasattr(result, 'solution'))
        self.assertTrue(hasattr(result, 'converged'))
        self.assertTrue(hasattr(result, 'iterations'))
        self.assertTrue(hasattr(result, 'residual_norm'))
        self.assertTrue(hasattr(result, 'message'))

    def test_2d_steady_state(self) -> None:
        """Test steady-state solve in 2D.

        Uses reaction term to make the problem non-singular.
        """
        mesh = StructuredMesh2D(
            nx=5, ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst
        )
        basis = LagrangeBasis(mesh, degree=1)

        # Add reaction term to make problem well-posed
        physics = LinearAdvectionDiffusionReaction(
            basis=basis,
            diffusivity=0.1,
            reaction=0.5,
            forcing=lambda x: np.ones(x.shape[1]),
            bkd=self.bkd_inst,
        )

        solver = SteadyStateSolver(physics, tol=1e-10)
        result = solver.solve_linear()

        self.assertTrue(result.converged)
        self.assertLess(result.residual_norm, 1e-8)

    def test_solver_with_zero_forcing(self) -> None:
        """Test that zero forcing with reaction gives zero solution.

        With reaction term r*u, zero forcing leads to u=0 as the unique solution.
        """
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        # No forcing but with reaction to make problem non-singular
        physics = LinearAdvectionDiffusionReaction(
            basis=basis,
            diffusivity=1.0,
            reaction=1.0,  # Makes stiffness matrix non-singular
            bkd=self.bkd_inst,
        )

        solver = SteadyStateSolver(physics)

        # Use float64 for consistency with skfem
        u_guess = self.bkd_inst.asarray(np.zeros(physics.nstates(), dtype=np.float64))
        result = solver.solve(u_guess)

        # Solution should be zero (or very close)
        u_np = self.bkd_inst.to_numpy(result.solution)
        self.assertLess(np.linalg.norm(u_np), 1e-10)


class TestSteadyStateSolverNumpy(TestSteadyStateSolverBase[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Try to import torch for dual-backend testing
try:
    import torch
    from pyapprox.typing.util.backends.torch import TorchBkd

    class TestSteadyStateSolverTorch(TestSteadyStateSolverBase[torch.Tensor]):
        """PyTorch backend tests."""

        __test__ = True

        def setUp(self) -> None:
            self._bkd = TorchBkd()
            super().setUp()

        def bkd(self) -> Backend[torch.Tensor]:
            return self._bkd

        @unittest.skip("sparse solve not available on CPU with TorchBkd")
        def test_linear_solve_with_forcing(self) -> None:
            pass

        @unittest.skip("sparse solve not available on CPU with TorchBkd")
        def test_newton_solve_converges(self) -> None:
            pass

        @unittest.skip("sparse solve not available on CPU with TorchBkd")
        def test_solver_result_attributes(self) -> None:
            pass

        @unittest.skip("sparse solve not available on CPU with TorchBkd")
        def test_2d_steady_state(self) -> None:
            pass

except ImportError:
    pass


from pyapprox.typing.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
