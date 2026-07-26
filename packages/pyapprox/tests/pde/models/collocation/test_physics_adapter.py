"""Tests for the parameterized collocation adapter tiers and factory."""

import math

import pytest
from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.ode.implicit_steppers.backward_euler import BackwardEulerHVP
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.operator.check_derivatives import (
    TimeAdjointDerivativeChecker,
)
from pyapprox.ode.operator.time_adjoint_hvp import (
    TimeAdjointOperatorWithHVP,
)
from pyapprox.pde.boundary import BCDofClassification
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.mesh import TransformedMesh1D
from pyapprox.pde.collocation.physics import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.time_integration import (
    CollocationPhysicsToODEResidualAdapter,
)
from pyapprox.pde.field_maps.basis_expansion import (
    BasisExpansion,
)
from pyapprox.pde.models.collocation import (
    CollocationPhysicsToODEResidualWithHVPAdapter,
    CollocationPhysicsToODEResidualWithParamJacobianAdapter,
    CollocationPhysicsToODEResidualWithSetParamAdapter,
    create_collocation_physics_ode_residual,
)
from pyapprox.pde.parameterizations.derivatives import (
    ParamDerivatives,
)
from pyapprox.pde.parameterizations.diffusion import (
    create_diffusion_parameterization,
)
from pyapprox.pde.parameterizations.fields import ConstantInTimeField
from pyapprox.util.rootfinding.newton import NewtonSolver

from tests._helpers.adjoint_checks import NoHVPQuadraticFieldMap


class _ToyCurvaturePhysics:
    """Minimal PhysicsProtocol with genuine state curvature.

    f(y, t) = -(1+t) * c ⊙ y^2 + g. Nonlinear in y and explicitly
    time-dependent so adapter mis-wiring (wrong time, stale params,
    swapped arguments) produces nonzero FD mismatches.
    """

    def __init__(self, bkd, basis, gvec):
        self._bkd = bkd
        self._basis = basis
        self._gvec = gvec
        self._c = bkd.ones((basis.npts(),))

    def set_coefficient(self, c):
        self._c = c

    def bkd(self):
        return self._bkd

    def basis(self):
        return self._basis

    def nstates(self):
        return self._basis.npts()

    def ncomponents(self):
        return 1

    def residual(self, state, time):
        return -(1.0 + time) * self._c * state**2 + self._gvec

    def jacobian(self, state, time):
        return self._bkd.diag(-2.0 * (1.0 + time) * self._c * state)

    def boundary_conditions(self):
        return []

    def bc_dof_classification(self):
        return BCDofClassification([], [])

    def apply_boundary_conditions(self, residual, jacobian, state, time):
        return residual, jacobian

    def mass_matrix(self):
        return self._bkd.eye(self.nstates())

    def apply_mass_matrix(self, vec):
        return vec

    def state_state_hvp(self, state, adj_state, wvec, time):
        # lam^T d2f/dy2 w with diagonal d2f/dy_i2 = -2(1+t) c_i
        return -2.0 * (1.0 + time) * self._c * adj_state * wvec


class _ToyCurvatureParameterization:
    """Analytic derivatives of R = -(1+t) exp(Phi p) y^2 + g.

    The coefficient field c = exp(Phi p) makes R non-quadratic in p, so
    all three parameter HVP blocks are nonzero.
    """

    def __init__(self, physics, bkd, phi):
        self._physics = physics
        self._bkd = bkd
        self._phi = phi

    def _cfield(self, params_1d):
        return self._bkd.exp(self._phi @ params_1d)

    def nparams(self):
        return self._phi.shape[1]

    def physics(self):
        return self._physics

    def apply(self, params_1d):
        self._physics.set_coefficient(self._cfield(params_1d))

    def param_derivatives(self):
        return ParamDerivatives.second_order(
            self._param_jacobian,
            self._initial_param_jacobian,
            self._param_param_hvp,
            self._state_param_hvp,
            self._param_state_hvp,
        )

    def _param_jacobian(self, state, time, params_1d):
        # dR/dp_j = -(1+t) phi_j ⊙ c ⊙ y^2
        base = -(1.0 + time) * self._cfield(params_1d) * state**2
        return base[:, None] * self._phi

    def _initial_param_jacobian(self, params_1d):
        return self._bkd.zeros((self._phi.shape[0], self.nparams()))

    def _param_param_hvp(self, state, time, params_1d, adj_state, vec):
        # lam^T d2R/dp2 v = Phi^T (lam ⊙ base ⊙ (Phi v))
        base = -(1.0 + time) * self._cfield(params_1d) * state**2
        return self._phi.T @ (adj_state * base * (self._phi @ vec))

    def _state_param_hvp(self, state, time, params_1d, adj_state, vec):
        # (d2R/dydp . v)^T lam, state-shaped
        dcv = -2.0 * (1.0 + time) * self._cfield(params_1d) * state
        return dcv * (self._phi @ vec) * adj_state

    def _param_state_hvp(self, state, time, params_1d, adj_state, wvec):
        # lam^T d2R/dpdy w, param-shaped
        dcv = -2.0 * (1.0 + time) * self._cfield(params_1d) * state
        return self._phi.T @ (adj_state * dcv * wvec)


def _make_hvp_tier_setup(bkd):
    """Build the HVP-tier toy adapter and a nontrivial evaluation point."""
    npts = 6
    mesh = TransformedMesh1D(npts, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    nodes = basis.nodes()
    phi = bkd.stack([bkd.ones((npts,)), nodes], axis=1)
    physics = _ToyCurvaturePhysics(bkd, basis, bkd.cos(nodes))
    adapter = create_collocation_physics_ode_residual(
        physics, bkd, _ToyCurvatureParameterization(physics, bkd, phi)
    )
    params_2d = bkd.array([0.4, -0.7])[:, None]
    state = bkd.cos(0.5 * math.pi * nodes) + 1.2
    adj = bkd.array([1.0, -0.5, 0.25, 2.0, -1.0, 0.7])
    adapter.set_time(0.3)
    adapter.set_param(params_2d[:, 0])
    return adapter, state, adj, params_2d


def _assert_fd_ratio(bkd, errors, tol):
    """Assert the FD-sweep min/max error ratio shows convergence.

    tol is explicit at each call site so individual tests tighten it as
    far as their derivative block allows.
    """
    ratio = float(bkd.min(errors) / bkd.max(errors))
    assert ratio <= tol, f"FD error ratio {ratio:.2e} exceeds {tol:.2e}"


class TestCollocationAdapterFactoryTiers:
    """Factory selects the fixed adapter tier from the bundle."""

    def test_factory_selects_param_jacobian_tier(self, bkd):
        """First-order bundle selects the WithParamJacobian tier."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        phi0 = bkd.ones((npts,))
        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)

        # Curvature without a declared hvp forces a first-order bundle
        # (linear maps declare hvp = 0 exactly and select the HVP tier).
        fm = NoHVPQuadraticFieldMap(bkd, bkd.full((npts,), 1.0), phi0[:, None])
        param = create_diffusion_parameterization(physics, bkd, basis, fm)

        adapter = create_collocation_physics_ode_residual(physics, bkd, param)
        assert isinstance(
            adapter, CollocationPhysicsToODEResidualWithParamJacobianAdapter
        )
        # not the HVP tier: the bundle is first order
        assert not isinstance(adapter, CollocationPhysicsToODEResidualWithHVPAdapter)

        # Test that it works
        adapter.set_param(bkd.array([0.5]))
        nodes = basis.nodes()
        state = bkd.sin(math.pi * nodes)
        adapter.set_time(0.0)

        param_jac = adapter.param_jacobian(state)
        assert param_jac.shape == (npts, 1)

    def test_factory_base_tier_for_basic_physics(self, bkd):
        """No parameterization selects the base tier."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
        adapter = create_collocation_physics_ode_residual(physics, bkd)

        assert type(adapter) is CollocationPhysicsToODEResidualAdapter
        # Base tier declares no parameter capability
        assert not hasattr(adapter, "param_jacobian")
        assert not hasattr(adapter, "nparams")

    def test_factory_set_param_tier_for_eval_only(self, bkd):
        """Eval-only bundle selects the WithSetParam tier."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)

        class EvalOnlyParameterization:
            def __init__(self, physics):
                self._physics = physics

            def nparams(self):
                return 1

            def physics(self):
                return self._physics

            def apply(self, params_1d):
                field = bkd.full((npts,), 1.0) + params_1d[0]
                self._physics.set_diffusion(ConstantInTimeField(field))

            def param_derivatives(self):
                return ParamDerivatives.none()

        adapter = create_collocation_physics_ode_residual(
            physics, bkd, EvalOnlyParameterization(physics)
        )
        assert type(adapter) is CollocationPhysicsToODEResidualWithSetParamAdapter
        assert not hasattr(adapter, "param_jacobian")
        adapter.set_param(bkd.array([0.5]))
        assert adapter.nparams() == 1

    def test_factory_selects_hvp_tier(self, bkd):
        """Second-order bundle + state_state_hvp physics selects HVP tier."""
        adapter, _, _, _ = _make_hvp_tier_setup(bkd)
        assert isinstance(adapter, CollocationPhysicsToODEResidualWithHVPAdapter)

    def test_hvp_tier_all_ode_derivatives_fd(self, bkd):
        """FD-validate every HVP-tier adapter method against the toy model.

        Drives TimeAdjointDerivativeChecker.check_all_ode_derivatives —
        the maintained orchestrator for exactly the six ODE-residual
        checks (state jacobian, param jacobian, and all four HVP blocks)
        — over the toy adapter. The toy is nonlinear in y, non-quadratic
        in p, and time-dependent, so every block is NONZERO and adapter
        mis-wiring (swapped arguments, stale params, wrong time) fails
        the FD sweeps. Building the operator stack also proves the
        fixed-tier adapter is accepted by the HVP stepper/integrator
        narrowing chain.
        """
        adapter, state, adj, params_2d = _make_hvp_tier_setup(bkd)
        time_residual = BackwardEulerHVP(adapter)
        newton_solver = NewtonSolver(time_residual)
        integrator = TimeIntegrator(0.0, 0.1, 0.05, newton_solver)
        functional = EndpointFunctional(
            state_idx=0,
            nstates=state.shape[0],
            nparams=params_2d.shape[0],
            bkd=bkd,
        )
        operator = TimeAdjointOperatorWithHVP(integrator, functional)
        checker = TimeAdjointDerivativeChecker(operator)

        errors = checker.check_all_ode_derivatives(
            state, params_2d, adj_state=adj, time=0.3
        )
        # order: jacobian, param_jacobian, state_state, state_param,
        # param_state, param_param. state_state is diagonal/exact so its
        # sweep converges much deeper than the parameter blocks.
        tols = [1e-5, 1e-5, 1e-10, 1e-5, 1e-5, 1e-5]
        for block_errors, tol in zip(errors, tols):
            _assert_fd_ratio(bkd, block_errors, tol=tol)

    def test_param_jacobian_before_set_param_raises(self, bkd):
        """Parameter derivatives require set_param first."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
        fm = BasisExpansion(bkd, 1.0, [bkd.ones((npts,))])
        param = create_diffusion_parameterization(physics, bkd, basis, fm)

        adapter = create_collocation_physics_ode_residual(physics, bkd, param)
        assert isinstance(
            adapter, CollocationPhysicsToODEResidualWithParamJacobianAdapter
        )
        state = bkd.zeros((npts,))
        with pytest.raises(RuntimeError, match="set_param"):
            adapter.param_jacobian(state)
