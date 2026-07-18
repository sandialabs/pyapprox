"""Tests for GalerkinPhysicsToODEResidualAdapter.

Tests that the adapter correctly translates GalerkinPhysics to ODEResidualProtocol
and works with the time steppers in typing.pde.time.
"""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)


import numpy as np
from pyapprox.ode.implicit_steppers import BackwardEulerHVP
from pyapprox.ode.step_context import StepContext
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.util.linalg.sparse_dispatch import solve_maybe_sparse

from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import LinearAdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration import (
    GalerkinPhysicsToODEResidualAdapter,
    GalerkinPhysicsToODEResidualWithParamJacobianAdapter,
    GalerkinPhysicsToODEResidualWithSetParamAdapter,
    create_galerkin_physics_ode_residual,
)


class TestPhysicsAdapterBase:
    """Base test class for GalerkinPhysicsToODEResidualAdapter."""

    def _setup(self, bkd):
        # Create simple 1D physics for testing
        self.mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        self.basis = LagrangeBasis(self.mesh, degree=1)
        self.physics = LinearAdvectionDiffusionReaction(
            basis=self.basis, diffusivity=0.01, bkd=bkd
        )
        self.adapter = GalerkinPhysicsToODEResidualAdapter(self.physics)

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
        n = self.physics.nstates()
        M = self.adapter.mass_matrix().as_matrix()
        assert M.shape == (n, n)

    def test_mass_matrix_cached(self, numpy_bkd) -> None:
        """Test mass matrix is cached."""
        bkd = numpy_bkd
        self._setup(bkd)
        M1 = self.adapter.mass_matrix()
        M2 = self.adapter.mass_matrix()
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
        """Test adapter works with BackwardEulerHVP."""
        bkd = numpy_bkd
        self._setup(bkd)
        # Create time stepper
        stepper = BackwardEulerHVP(self.adapter)

        # Set up initial condition
        u0 = self.physics.initial_condition(lambda x: np.sin(np.pi * x[0]))

        # Set time stepping context
        dt = 0.01
        stepper.bind(StepContext(t_prev=0.0, deltat=dt, y_prev=u0))

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
        stepper = BackwardEulerHVP(self.adapter)

        # Initial condition
        u0 = self.physics.initial_condition(lambda x: np.sin(np.pi * x[0]))

        # Set time stepping context
        dt = 0.001
        stepper.bind(StepContext(t_prev=0.0, deltat=dt, y_prev=u0))

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
        stepper = BackwardEulerHVP(self.adapter)

        # Initial condition: sine wave
        u = self.physics.initial_condition(lambda x: np.sin(np.pi * x[0]))

        # Set up single time step
        dt = 0.001
        stepper.bind(StepContext(t_prev=0.0, deltat=dt, y_prev=u))

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




class TestGalerkinAdapterFactoryTiers:
    """Factory selects the fixed adapter tier from the bundle."""

    def _make_physics(self, bkd):
        mesh = StructuredMesh1D(nx=8, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        return LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, bkd=bkd
        )

    def test_base_tier_without_parameterization(self, numpy_bkd):
        physics = self._make_physics(numpy_bkd)
        adapter = create_galerkin_physics_ode_residual(physics)
        assert type(adapter) is GalerkinPhysicsToODEResidualAdapter
        assert not hasattr(adapter, "param_jacobian")
        assert not hasattr(adapter, "nparams")

    def test_set_param_tier_for_eval_only(self, numpy_bkd):
        bkd = numpy_bkd
        physics = self._make_physics(bkd)

        class EvalOnlyParameterization:
            def nparams(self):
                return 1

            def apply(self, phys, params_1d):
                pass

            def param_derivatives(self):
                return ParamDerivatives.none()

        adapter = create_galerkin_physics_ode_residual(
            physics, EvalOnlyParameterization()
        )
        assert type(adapter) is GalerkinPhysicsToODEResidualWithSetParamAdapter
        assert not hasattr(adapter, "param_jacobian")
        adapter.set_param(bkd.array([0.5]))
        assert adapter.nparams() == 1

    def test_param_jacobian_tier_for_first_order(self, numpy_bkd):
        bkd = numpy_bkd
        physics = self._make_physics(bkd)
        nstates = physics.nstates()

        def _jac(phys, state, time, params_1d):
            return bkd.zeros((nstates, 1))

        def _init_jac(phys, params_1d):
            return bkd.zeros((nstates, 1))

        class FirstOrderParameterization:
            def nparams(self):
                return 1

            def apply(self, phys, params_1d):
                pass

            def param_derivatives(self):
                return ParamDerivatives.first_order(_jac, _init_jac)

        adapter = create_galerkin_physics_ode_residual(
            physics, FirstOrderParameterization()
        )
        assert isinstance(
            adapter, GalerkinPhysicsToODEResidualWithParamJacobianAdapter
        )
        adapter.set_param(bkd.array([0.5]))
        assert adapter.param_jacobian(bkd.zeros((nstates,))).shape == (
            nstates, 1,
        )

    def test_set_param_rejects_2d(self, numpy_bkd):
        bkd = numpy_bkd
        physics = self._make_physics(bkd)

        class EvalOnlyParameterization:
            def nparams(self):
                return 1

            def apply(self, phys, params_1d):
                pass

            def param_derivatives(self):
                return ParamDerivatives.none()

        adapter = create_galerkin_physics_ode_residual(
            physics, EvalOnlyParameterization()
        )
        with pytest.raises(ValueError, match="must be 1D"):
            adapter.set_param(bkd.array([[0.5]]))
