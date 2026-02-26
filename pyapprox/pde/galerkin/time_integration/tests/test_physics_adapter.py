"""Tests for GalerkinPhysicsODEAdapter.

Tests that the adapter correctly translates GalerkinPhysics to ODEResidualProtocol
and works with the time steppers in typing.pde.time.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import LinearAdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration import GalerkinPhysicsODEAdapter
from pyapprox.pde.sparse_utils import solve_maybe_sparse
from pyapprox.pde.time.implicit_steppers import BackwardEulerResidual
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array


class TestPhysicsAdapterBase:
    """Base test class for GalerkinPhysicsODEAdapter."""

    def _setup(self, bkd):
        # Create simple 1D physics for testing
        self.mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        self.basis = LagrangeBasis(self.mesh, degree=1)
        self.physics = LinearAdvectionDiffusionReaction(
            basis=self.basis, diffusivity=0.01, bkd=bkd
        )
        self.adapter = GalerkinPhysicsODEAdapter(self.physics)

    def test_adapter_has_required_methods(self, numpy_bkd) -> None:
        """Test adapter exposes required ODEResidualProtocol methods."""
        bkd = numpy_bkd
        self._setup(bkd)
        assert callable(getattr(self.adapter, "bkd"))
        assert callable(getattr(self.adapter, "__call__"))
        assert callable(getattr(self.adapter, "set_time"))
        assert callable(getattr(self.adapter, "jacobian"))
        assert callable(getattr(self.adapter, "mass_matrix"))

    def test_residual_call(self, numpy_bkd) -> None:
        """Test calling adapter returns residual with correct shape."""
        bkd = numpy_bkd
        self._setup(bkd)
        u0 = self.physics.initial_condition(lambda x: np.sin(np.pi * x[0]))
        self.adapter.set_time(0.0)
        residual = self.adapter(u0)
        assert residual.shape == (self.physics.nstates(),)

    def test_jacobian_shape(self, numpy_bkd) -> None:
        """Test Jacobian has correct shape."""
        bkd = numpy_bkd
        self._setup(bkd)
        u0 = bkd.asarray(np.zeros(self.physics.nstates()))
        self.adapter.set_time(0.0)
        jac = self.adapter.jacobian(u0)
        assert jac.shape == (self.physics.nstates(), self.physics.nstates())

    def test_mass_matrix_shape(self, numpy_bkd) -> None:
        """Test mass matrix has correct shape."""
        bkd = numpy_bkd
        self._setup(bkd)
        M = self.adapter.mass_matrix(self.physics.nstates())
        assert M.shape == (self.physics.nstates(), self.physics.nstates())

    def test_mass_matrix_cached(self, numpy_bkd) -> None:
        """Test mass matrix is cached."""
        bkd = numpy_bkd
        self._setup(bkd)
        M1 = self.adapter.mass_matrix(self.physics.nstates())
        M2 = self.adapter.mass_matrix(self.physics.nstates())
        # Check they are the same object (cached)
        assert M1 is M2

    def test_set_time(self, numpy_bkd) -> None:
        """Test set_time updates internal time."""
        bkd = numpy_bkd
        self._setup(bkd)
        self.adapter.set_time(1.5)
        assert self.adapter._time == 1.5

    def test_bkd_returns_backend(self, numpy_bkd) -> None:
        """Test bkd returns correct backend."""
        bkd = numpy_bkd
        self._setup(bkd)
        assert self.adapter.bkd() is bkd

    def test_with_backward_euler(self, numpy_bkd) -> None:
        """Test adapter works with BackwardEulerResidual."""
        bkd = numpy_bkd
        self._setup(bkd)
        # Create time stepper
        stepper = BackwardEulerResidual(self.adapter)

        # Set up initial condition
        u0 = self.physics.initial_condition(lambda x: np.sin(np.pi * x[0]))

        # Set time stepping context
        dt = 0.01
        stepper.set_time(0.0, dt, u0)

        # Evaluate residual (this tests the interface compatibility)
        res = stepper(u0)
        assert res.shape == (self.physics.nstates(),)

        # Evaluate Jacobian
        jac = stepper.jacobian(u0)
        assert jac.shape == (self.physics.nstates(), self.physics.nstates())

    def test_time_stepping_single_step(self, numpy_bkd) -> None:
        """Test taking a single time step with Newton's method."""
        bkd = numpy_bkd
        self._setup(bkd)
        # Create time stepper
        stepper = BackwardEulerResidual(self.adapter)

        # Initial condition
        u0 = self.physics.initial_condition(lambda x: np.sin(np.pi * x[0]))

        # Set time stepping context
        dt = 0.001
        stepper.set_time(0.0, dt, u0)

        # Simple Newton iteration for one time step
        u_new = bkd.copy(u0)
        for _ in range(5):  # Newton iterations
            res = stepper(u_new)
            jac = stepper.jacobian(u_new)
            du = solve_maybe_sparse(bkd, jac, -res)
            u_new = u_new + du

        # Check solution is different from initial
        u0_np = bkd.to_numpy(u0)
        u_new_np = bkd.to_numpy(u_new)
        assert np.linalg.norm(u_new_np - u0_np) > 1e-10

    def test_newton_convergence(self, numpy_bkd) -> None:
        """Test Newton iteration converges for a single time step."""
        bkd = numpy_bkd
        self._setup(bkd)
        stepper = BackwardEulerResidual(self.adapter)

        # Initial condition: sine wave
        u = self.physics.initial_condition(lambda x: np.sin(np.pi * x[0]))

        # Set up single time step
        dt = 0.001
        stepper.set_time(0.0, dt, u)

        # Track residual norms during Newton iteration
        u_new = bkd.copy(u)
        residual_norms = []

        for _ in range(10):  # Newton iterations
            res = stepper(u_new)
            res_np = bkd.to_numpy(res)
            residual_norms.append(np.linalg.norm(res_np))

            jac = stepper.jacobian(u_new)
            du = solve_maybe_sparse(bkd, jac, -res)
            u_new = u_new + du

        # Newton should converge - final residual should be much smaller
        assert residual_norms[-1] < 1e-10
        # And should be much smaller than initial
        assert residual_norms[-1] < residual_norms[0] * 1e-6


