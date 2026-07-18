"""Tests for time integration module."""

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
from pyapprox.pde.field_maps.basis_expansion import (
    BasisExpansion,
)
from pyapprox.pde.parameterizations.derivatives import (
    ParamDerivatives,
)
from pyapprox.pde.parameterizations.diffusion import (
    create_diffusion_parameterization,
)
from pyapprox.util.rootfinding.newton import NewtonSolver

from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import (
    constant_dirichlet_bc,
    zero_dirichlet_bc,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    create_uniform_mesh_1d,
)
from pyapprox.pde.collocation.physics import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.time_integration import (
    CollocationModel,
    PhysicsToODEResidualAdapter,
    PhysicsToODEResidualWithHVPAdapter,
    PhysicsToODEResidualWithParamJacobianAdapter,
    PhysicsToODEResidualWithSetParamAdapter,
    TimeIntegrationConfig,
    create_physics_ode_residual,
)


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

    def __init__(self, bkd, phi):
        self._bkd = bkd
        self._phi = phi

    def _cfield(self, params_1d):
        return self._bkd.exp(self._phi @ params_1d)

    def nparams(self):
        return self._phi.shape[1]

    def apply(self, phys, params_1d):
        phys.set_coefficient(self._cfield(params_1d))

    def param_derivatives(self):
        return ParamDerivatives.second_order(
            self._param_jacobian,
            self._initial_param_jacobian,
            self._param_param_hvp,
            self._state_param_hvp,
            self._param_state_hvp,
        )

    def _param_jacobian(self, phys, state, time, params_1d):
        # dR/dp_j = -(1+t) phi_j ⊙ c ⊙ y^2
        base = -(1.0 + time) * self._cfield(params_1d) * state**2
        return base[:, None] * self._phi

    def _initial_param_jacobian(self, phys, params_1d):
        return self._bkd.zeros((self._phi.shape[0], self.nparams()))

    def _param_param_hvp(self, phys, state, time, params_1d, adj_state, vec):
        # lam^T d2R/dp2 v = Phi^T (lam ⊙ base ⊙ (Phi v))
        base = -(1.0 + time) * self._cfield(params_1d) * state**2
        return self._phi.T @ (adj_state * base * (self._phi @ vec))

    def _state_param_hvp(self, phys, state, time, params_1d, adj_state, vec):
        # (d2R/dydp . v)^T lam, state-shaped
        dcv = -2.0 * (1.0 + time) * self._cfield(params_1d) * state
        return dcv * (self._phi @ vec) * adj_state

    def _param_state_hvp(self, phys, state, time, params_1d, adj_state, wvec):
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
    adapter = create_physics_ode_residual(
        physics, bkd, _ToyCurvatureParameterization(bkd, phi)
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


class TestPhysicsToODEResidualAdapter:
    """Base test class for PhysicsToODEResidualAdapter."""

    def test_adapter_basic_interface(self, bkd):
        """Test that adapter provides ODEResidual interface."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
        adapter = PhysicsToODEResidualAdapter(physics, bkd)

        # Check interface methods exist
        assert callable(adapter.bkd)
        assert callable(adapter.set_time)
        assert callable(adapter.jacobian)
        assert callable(adapter.mass_matrix)

        # __call__ should work
        state = bkd.zeros((npts,))
        adapter.set_time(0.0)
        result = adapter(state)
        assert result.shape == (npts,)

    def test_adapter_residual_consistency(self, bkd):
        """Test that adapter residual matches physics residual."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
        adapter = PhysicsToODEResidualAdapter(physics, bkd)

        nodes = basis.nodes()
        state = bkd.sin(math.pi * nodes)
        time = 0.5

        # Physics residual
        physics_res = physics.residual(state, time)

        # Adapter residual (without BCs)
        adapter.set_time(time)
        adapter_res = adapter(state)

        # Should match when no BCs
        bkd.assert_allclose(adapter_res, physics_res, atol=1e-14)

    def test_adapter_jacobian_consistency(self, bkd):
        """Test that adapter Jacobian matches physics Jacobian."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
        adapter = PhysicsToODEResidualAdapter(physics, bkd)

        nodes = basis.nodes()
        state = bkd.sin(math.pi * nodes)
        time = 0.5

        # Physics Jacobian
        physics_jac = physics.jacobian(state, time)

        # Adapter Jacobian (without BCs)
        adapter.set_time(time)
        adapter_jac = adapter.jacobian(state)

        bkd.assert_allclose(adapter_jac, physics_jac, atol=1e-14)

    def test_adapter_mass_matrix(self, bkd):
        """Test that adapter returns identity mass matrix."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
        adapter = PhysicsToODEResidualAdapter(physics, bkd)

        mass = adapter.mass_matrix()
        assert mass.is_identity()
        bkd.assert_allclose(mass.as_matrix(), bkd.eye(npts), atol=1e-14)

    def test_adapter_with_boundary_conditions(self, bkd):
        """Test that BCs are applied via physics.apply_boundary_conditions.

        The adapter returns the raw physics Jacobian. Boundary conditions
        are applied by CollocationModel._apply_boundary_conditions, which
        calls physics.apply_boundary_conditions on the Newton system.
        """
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)

        # Set BCs: u(-1) = 0, u(1) = 1
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = constant_dirichlet_bc(bkd, right_idx, 1.0)
        physics.set_boundary_conditions([bc_left, bc_right])

        # State that satisfies BCs
        nodes = basis.nodes()
        state = 0.5 * (nodes + 1.0)  # Linear from 0 to 1

        # Get raw Jacobian from physics, then apply BCs
        residual = physics.residual(state, 0.0)
        jacobian = physics.jacobian(state, 0.0)
        _, jacobian_with_bc = physics.apply_boundary_conditions(
            residual, jacobian, state, 0.0
        )

        # Boundary rows should be identity-like after applying BCs
        bkd.assert_allclose(jacobian_with_bc[0, :], bkd.eye(npts)[0, :], atol=1e-14)
        bkd.assert_allclose(jacobian_with_bc[-1, :], bkd.eye(npts)[-1, :], atol=1e-14)

    def test_factory_selects_param_jacobian_tier(self, bkd):
        """First-order bundle selects the WithParamJacobian tier."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        phi0 = bkd.ones((npts,))
        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)

        fm = BasisExpansion(bkd, 1.0, [phi0])
        param = create_diffusion_parameterization(bkd, basis, fm)

        adapter = create_physics_ode_residual(physics, bkd, param)
        assert isinstance(
            adapter, PhysicsToODEResidualWithParamJacobianAdapter
        )
        # not the HVP tier: bundle is first-order and physics has no
        # state_state_hvp
        assert not isinstance(adapter, PhysicsToODEResidualWithHVPAdapter)

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
        adapter = create_physics_ode_residual(physics, bkd)

        assert type(adapter) is PhysicsToODEResidualAdapter
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
            def nparams(self):
                return 1

            def apply(self, phys, params_1d):
                field = bkd.full((npts,), 1.0) + params_1d[0]
                phys.set_diffusion(lambda t, _f=field: _f)

            def param_derivatives(self):
                return ParamDerivatives.none()

        adapter = create_physics_ode_residual(
            physics, bkd, EvalOnlyParameterization()
        )
        assert type(adapter) is PhysicsToODEResidualWithSetParamAdapter
        assert not hasattr(adapter, "param_jacobian")
        adapter.set_param(bkd.array([0.5]))
        assert adapter.nparams() == 1

    def test_factory_selects_hvp_tier(self, bkd):
        """Second-order bundle + state_state_hvp physics selects HVP tier."""
        adapter, _, _, _ = _make_hvp_tier_setup(bkd)
        assert isinstance(adapter, PhysicsToODEResidualWithHVPAdapter)

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
        param = create_diffusion_parameterization(bkd, basis, fm)

        adapter = create_physics_ode_residual(physics, bkd, param)
        assert isinstance(
            adapter, PhysicsToODEResidualWithParamJacobianAdapter
        )
        state = bkd.zeros((npts,))
        with pytest.raises(RuntimeError, match="set_param"):
            adapter.param_jacobian(state)


class TestCollocationModel:
    """Base test class for CollocationModel."""

    def test_model_creation(self, bkd):
        """Test basic model creation."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
        model = CollocationModel(physics, bkd)

        assert model.nstates() == npts
        assert model.physics() is physics
        assert model.bkd() is bkd

    def test_solve_steady_poisson(self, bkd):
        """Test steady-state solve for Poisson equation.

        Solve: D * laplacian(u) + f = 0 with u(-1) = 0, u(1) = 0

        For u = sin(pi*x), laplacian(u) = -pi^2 * sin(pi*x)
        So D * laplacian(u) = -D * pi^2 * sin(pi*x)
        For residual = 0, need f = D * pi^2 * sin(pi*x)
        """
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        D = 1.0
        nodes = basis.nodes()

        # Forcing f such that D*laplacian(u) + f = 0 for u = sin(pi*x)
        def forcing(t):
            return (math.pi**2) * bkd.sin(math.pi * nodes)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=D, forcing=forcing)

        # BCs: u(-1) = 0, u(1) = 0 (consistent with sin(pi*x))
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)

        # Initial guess - use exact solution for better convergence
        initial_guess = bkd.sin(math.pi * nodes) * 0.5

        # Solve
        u_steady = model.solve_steady(initial_guess, tol=1e-10, maxiter=50)

        # Compare to exact solution
        u_exact = bkd.sin(math.pi * nodes)
        bkd.assert_allclose(u_steady, u_exact, atol=1e-6)

    def test_solve_transient_decay(self, bkd):
        """Test transient solve for exponential decay.

        Solve: du/dt = -r * u with u(0) = 1
        Exact solution: u(t) = exp(-r * t)

        For backward Euler with small time step, should converge to exact.
        """
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        r = 2.0
        physics = AdvectionDiffusionReaction(basis, bkd, reaction=-r)

        model = CollocationModel(physics, bkd)

        # Initial condition
        u0 = bkd.ones((npts,))

        # Time integration config with smaller time step for accuracy
        config = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=1.0,
            deltat=0.01,  # Smaller dt for better accuracy
        )

        # Solve
        solutions, times = model.solve_transient(u0, config)

        # Compare to exact at final time
        u_exact_final = math.exp(-r * 1.0) * bkd.ones((npts,))
        bkd.assert_allclose(solutions[:, -1], u_exact_final, rtol=0.05)

    def test_transient_forward_euler(self, bkd):
        """Test Forward Euler time stepping."""
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        r = 1.0
        physics = AdvectionDiffusionReaction(basis, bkd, reaction=-r)
        model = CollocationModel(physics, bkd)

        u0 = bkd.ones((npts,))
        config = TimeIntegrationConfig(
            method="forward_euler",
            final_time=0.5,
            deltat=0.01,  # Small dt for stability
        )

        solutions, times = model.solve_transient(u0, config)

        u_exact_final = math.exp(-r * 0.5) * bkd.ones((npts,))
        bkd.assert_allclose(solutions[:, -1], u_exact_final, rtol=0.02)

    def test_transient_heun(self, bkd):
        """Test Heun's method (RK2) time stepping."""
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        r = 1.0
        physics = AdvectionDiffusionReaction(basis, bkd, reaction=-r)
        model = CollocationModel(physics, bkd)

        u0 = bkd.ones((npts,))
        config = TimeIntegrationConfig(
            method="heun",
            final_time=0.5,
            deltat=0.05,
        )

        solutions, times = model.solve_transient(u0, config)

        u_exact_final = math.exp(-r * 0.5) * bkd.ones((npts,))
        # Heun should be more accurate than Forward Euler
        bkd.assert_allclose(solutions[:, -1], u_exact_final, rtol=0.01)

    def test_transient_crank_nicolson(self, bkd):
        """Test Crank-Nicolson time stepping."""
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        r = 1.0
        physics = AdvectionDiffusionReaction(basis, bkd, reaction=-r)
        model = CollocationModel(physics, bkd)

        u0 = bkd.ones((npts,))
        config = TimeIntegrationConfig(
            method="crank_nicolson",
            final_time=0.5,
            deltat=0.1,
        )

        solutions, times = model.solve_transient(u0, config)

        u_exact_final = math.exp(-r * 0.5) * bkd.ones((npts,))
        bkd.assert_allclose(solutions[:, -1], u_exact_final, rtol=0.01)

    def test_transient_diffusion(self, bkd):
        """Test transient diffusion equation.

        Solve: du/dt = D * laplacian(u) with BCs and IC.
        """
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        D = 0.1
        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=D)

        # BCs: u(-1) = 0, u(1) = 0
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)

        # Initial condition: sin(pi*x)
        nodes = basis.nodes()
        u0 = bkd.sin(math.pi * nodes)

        config = TimeIntegrationConfig(
            method="backward_euler",
            final_time=0.5,
            deltat=0.01,
        )

        solutions, times = model.solve_transient(u0, config)

        # Exact solution: exp(-D * pi^2 * t) * sin(pi*x)
        u_exact_final = math.exp(-D * math.pi**2 * 0.5) * bkd.sin(math.pi * nodes)

        # Should be reasonably close
        # Use atol for boundary points which are near zero
        bkd.assert_allclose(solutions[:, -1], u_exact_final, rtol=0.05, atol=1e-10)

    def test_time_output_shape(self, bkd):
        """Test that output shapes are correct."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
        model = CollocationModel(physics, bkd)

        u0 = bkd.zeros((npts,))
        config = TimeIntegrationConfig(
            method="backward_euler",
            final_time=0.5,
            deltat=0.1,
        )

        solutions, times = model.solve_transient(u0, config)

        # Should have 6 time points: 0, 0.1, 0.2, 0.3, 0.4, 0.5
        assert times.shape[0] == 6
        assert solutions.shape == (npts, 6)


# NumPy backend tests
