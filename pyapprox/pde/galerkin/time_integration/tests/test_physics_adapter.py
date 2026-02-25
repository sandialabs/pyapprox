"""Tests for GalerkinPhysicsODEAdapter.

Tests that the adapter correctly translates GalerkinPhysics to ODEResidualProtocol
and works with the time steppers in typing.pde.time.
"""

import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.sparse_utils import solve_maybe_sparse
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.physics import LinearAdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration import GalerkinPhysicsODEAdapter
from pyapprox.pde.time.implicit_steppers import BackwardEulerResidual


class TestPhysicsAdapterBase(Generic[Array], unittest.TestCase):
    """Base test class for GalerkinPhysicsODEAdapter."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self.bkd_inst = self.bkd()

        # Create simple 1D physics for testing
        self.mesh = StructuredMesh1D(
            nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst
        )
        self.basis = LagrangeBasis(self.mesh, degree=1)
        self.physics = LinearAdvectionDiffusionReaction(
            basis=self.basis, diffusivity=0.01, bkd=self.bkd_inst
        )
        self.adapter = GalerkinPhysicsODEAdapter(self.physics)

    def test_adapter_has_required_methods(self) -> None:
        """Test adapter exposes required ODEResidualProtocol methods."""
        self.assertTrue(callable(getattr(self.adapter, "bkd")))
        self.assertTrue(callable(getattr(self.adapter, "__call__")))
        self.assertTrue(callable(getattr(self.adapter, "set_time")))
        self.assertTrue(callable(getattr(self.adapter, "jacobian")))
        self.assertTrue(callable(getattr(self.adapter, "mass_matrix")))

    def test_residual_call(self) -> None:
        """Test calling adapter returns residual with correct shape."""
        u0 = self.physics.initial_condition(lambda x: np.sin(np.pi * x[0]))
        self.adapter.set_time(0.0)
        residual = self.adapter(u0)
        self.assertEqual(residual.shape, (self.physics.nstates(),))

    def test_jacobian_shape(self) -> None:
        """Test Jacobian has correct shape."""
        u0 = self.bkd_inst.asarray(np.zeros(self.physics.nstates()))
        self.adapter.set_time(0.0)
        jac = self.adapter.jacobian(u0)
        self.assertEqual(jac.shape, (self.physics.nstates(), self.physics.nstates()))

    def test_mass_matrix_shape(self) -> None:
        """Test mass matrix has correct shape."""
        M = self.adapter.mass_matrix(self.physics.nstates())
        self.assertEqual(M.shape, (self.physics.nstates(), self.physics.nstates()))

    def test_mass_matrix_cached(self) -> None:
        """Test mass matrix is cached."""
        M1 = self.adapter.mass_matrix(self.physics.nstates())
        M2 = self.adapter.mass_matrix(self.physics.nstates())
        # Check they are the same object (cached)
        self.assertIs(M1, M2)

    def test_set_time(self) -> None:
        """Test set_time updates internal time."""
        self.adapter.set_time(1.5)
        self.assertEqual(self.adapter._time, 1.5)

    def test_bkd_returns_backend(self) -> None:
        """Test bkd returns correct backend."""
        self.assertIs(self.adapter.bkd(), self.bkd_inst)

    def test_with_backward_euler(self) -> None:
        """Test adapter works with BackwardEulerResidual."""
        # Create time stepper
        stepper = BackwardEulerResidual(self.adapter)

        # Set up initial condition
        u0 = self.physics.initial_condition(lambda x: np.sin(np.pi * x[0]))

        # Set time stepping context
        dt = 0.01
        stepper.set_time(0.0, dt, u0)

        # Evaluate residual (this tests the interface compatibility)
        res = stepper(u0)
        self.assertEqual(res.shape, (self.physics.nstates(),))

        # Evaluate Jacobian
        jac = stepper.jacobian(u0)
        self.assertEqual(jac.shape, (self.physics.nstates(), self.physics.nstates()))

    def test_time_stepping_single_step(self) -> None:
        """Test taking a single time step with Newton's method."""
        # Create time stepper
        stepper = BackwardEulerResidual(self.adapter)

        # Initial condition
        u0 = self.physics.initial_condition(lambda x: np.sin(np.pi * x[0]))

        # Set time stepping context
        dt = 0.001
        stepper.set_time(0.0, dt, u0)

        # Simple Newton iteration for one time step
        u_new = self.bkd_inst.copy(u0)
        for _ in range(5):  # Newton iterations
            res = stepper(u_new)
            jac = stepper.jacobian(u_new)
            du = solve_maybe_sparse(self.bkd_inst, jac, -res)
            u_new = u_new + du

        # Check solution is different from initial
        u0_np = self.bkd_inst.to_numpy(u0)
        u_new_np = self.bkd_inst.to_numpy(u_new)
        self.assertTrue(np.linalg.norm(u_new_np - u0_np) > 1e-10)

    def test_newton_convergence(self) -> None:
        """Test Newton iteration converges for a single time step."""
        stepper = BackwardEulerResidual(self.adapter)

        # Initial condition: sine wave
        u = self.physics.initial_condition(lambda x: np.sin(np.pi * x[0]))

        # Set up single time step
        dt = 0.001
        stepper.set_time(0.0, dt, u)

        # Track residual norms during Newton iteration
        u_new = self.bkd_inst.copy(u)
        residual_norms = []

        for _ in range(10):  # Newton iterations
            res = stepper(u_new)
            res_np = self.bkd_inst.to_numpy(res)
            residual_norms.append(np.linalg.norm(res_np))

            jac = stepper.jacobian(u_new)
            du = solve_maybe_sparse(self.bkd_inst, jac, -res)
            u_new = u_new + du

        # Newton should converge - final residual should be much smaller
        self.assertLess(residual_norms[-1], 1e-10)
        # And should be much smaller than initial
        self.assertLess(residual_norms[-1], residual_norms[0] * 1e-6)


class TestPhysicsAdapterNumpy(TestPhysicsAdapterBase[NDArray[Any]]):
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
    from pyapprox.util.backends.torch import TorchBkd

    class TestPhysicsAdapterTorch(TestPhysicsAdapterBase[torch.Tensor]):
        """PyTorch backend tests.

        Tests requiring sparse solves are skipped because
        torch.sparse.spsolve is not available on CPU.
        """

        __test__ = True

        def setUp(self) -> None:
            self._bkd = TorchBkd()
            super().setUp()

        def bkd(self) -> Backend[torch.Tensor]:
            return self._bkd

        @unittest.skip("sparse solve not available on CPU with TorchBkd")
        def test_time_stepping_single_step(self) -> None:
            pass

        @unittest.skip("sparse solve not available on CPU with TorchBkd")
        def test_newton_convergence(self) -> None:
            pass

except ImportError:
    pass


from pyapprox.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
