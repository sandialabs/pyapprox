"""Tests for steady-state forward model classes."""

import math

import numpy as np
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.interface.functions.protocols import (
    FunctionProtocol,
)
from pyapprox.optimization.implicitfunction.functionals.subset_of_states import (
    SubsetOfStatesAdjointFunctional,
)
from pyapprox.optimization.implicitfunction.operator.check_derivatives import (
    ImplicitFunctionDerivativeChecker,
)
from pyapprox.optimization.implicitfunction.operator.sensitivities import (
    VectorAdjointOperatorWithJacobian,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import (
    zero_dirichlet_bc,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    create_uniform_mesh_1d,
)
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.field_maps.basis_expansion import (
    BasisExpansion,
)
from pyapprox.pde.models.collocation import create_collocation_model
from pyapprox.pde.models.collocation.steady import (
    CollocationStateEquationWithJacobianAdapter,
    SteadyForwardModel,
)
from pyapprox.pde.parameterizations.derivatives import (
    ParamDerivatives,
)
from pyapprox.pde.parameterizations.diffusion import (
    create_diffusion_parameterization,
)
from pyapprox.pde.parameterizations.fields import ConstantInTimeField


def _create_parameterized_diffusion_problem(bkd, npts=20):
    """Create a parameterized diffusion problem for testing.

    Uses base AdvectionDiffusionReaction + DiffusionParameterization.

    Problem: -div(D(x) * grad(u)) = f with u(-1) = 0, u(1) = 0
    D(x) = D_base + p0 * phi0(x) + p1 * phi1(x)
    phi0(x) = 1 (constant), phi1(x) = x (linear)

    Returns
    -------
    physics, parameterization, init_state
    """
    mesh = TransformedMesh1D(npts, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    mesh_obj = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
    nodes = basis.nodes()

    phi0 = bkd.ones((npts,))
    phi1 = nodes

    def forcing(t):
        return (math.pi**2) * bkd.sin(math.pi * nodes)

    physics = AdvectionDiffusionReaction(
        basis,
        bkd,
        diffusion=2.0,
        forcing=forcing,
    )

    left_idx = mesh_obj.boundary_indices(0)
    right_idx = mesh_obj.boundary_indices(1)
    bc_left = zero_dirichlet_bc(bkd, left_idx)
    bc_right = zero_dirichlet_bc(bkd, right_idx)
    physics.set_boundary_conditions([bc_left, bc_right])

    fm = BasisExpansion(bkd, 2.0, [phi0, phi1])
    param = create_diffusion_parameterization(physics, bkd, fm)
    init_state = bkd.zeros((npts,))
    return physics, param, init_state


class TestCollocationStateEquationWithJacobianAdapter:
    def test_solve_matches_collocation_model(self, bkd):
        """Adapter solve matches direct CollocationModel.solve_steady."""
        physics, param, init_state_1d = _create_parameterized_diffusion_problem(bkd)

        model = create_collocation_model(physics, bkd, parameterization=param)
        adapter = CollocationStateEquationWithJacobianAdapter(
            model, bkd, parameterization=param
        )

        param_1d = bkd.array([0.5, 0.1])
        param_2d = param_1d[:, None]
        init_state_2d = init_state_1d[:, None]

        # Direct solve
        param.apply(param_1d)
        u_direct = model.solve_steady(init_state_1d)

        # Adapter solve
        u_adapter = adapter.solve(init_state_2d, param_2d)

        bkd.assert_allclose(u_adapter[:, 0], u_direct, rtol=1e-10)

    def test_residual_at_solution_is_zero(self, bkd):
        """Residual should be near-zero at the converged solution."""
        physics, param, init_state_1d = _create_parameterized_diffusion_problem(bkd)

        model = create_collocation_model(physics, bkd, parameterization=param)
        adapter = CollocationStateEquationWithJacobianAdapter(
            model, bkd, parameterization=param
        )

        param_2d = bkd.array([0.3, -0.1])[:, None]
        init_state_2d = init_state_1d[:, None]

        sol = adapter.solve(init_state_2d, param_2d)
        residual = adapter(sol, param_2d)

        bkd.assert_allclose(residual, bkd.zeros_like(residual), atol=1e-8)

    def test_derivative_checker_state_jacobian(self, bkd):
        """DerivativeChecker validates state Jacobian via FD."""
        physics, param, init_state_1d = _create_parameterized_diffusion_problem(bkd)

        model = create_collocation_model(physics, bkd, parameterization=param)
        adapter = CollocationStateEquationWithJacobianAdapter(
            model, bkd, parameterization=param
        )

        param_2d = bkd.array([0.3, 0.1])[:, None]
        init_state_2d = init_state_1d[:, None]

        # Solve to get a meaningful state
        sol = adapter.solve(init_state_2d, param_2d)

        # Use FunctionWithJacobianFromCallable to check state Jacobian
        wrapper = FunctionWithJacobianFromCallable(
            nqoi=adapter.nstates(),
            nvars=adapter.nstates(),
            fun=lambda state: adapter(state, param_2d),
            jacobian=lambda state: adapter.state_jacobian(state, param_2d),
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sol, direction=None, relative=True)[0]
        # min/max error ratio should be small
        assert float(bkd.min(errors) / bkd.max(errors)) <= 1e-5

    def test_derivative_checker_param_jacobian(self, bkd):
        """DerivativeChecker validates parameter Jacobian via FD."""
        physics, param, init_state_1d = _create_parameterized_diffusion_problem(bkd)

        model = create_collocation_model(physics, bkd, parameterization=param)
        adapter = CollocationStateEquationWithJacobianAdapter(
            model, bkd, parameterization=param
        )

        param_2d = bkd.array([0.3, 0.1])[:, None]
        init_state_2d = init_state_1d[:, None]

        sol = adapter.solve(init_state_2d, param_2d)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=adapter.nstates(),
            nvars=adapter.nparams(),
            fun=lambda p: adapter(sol, p),
            jacobian=lambda p: adapter.param_jacobian(sol, p),
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(param_2d, direction=None, relative=True)[0]
        assert float(bkd.min(errors) / bkd.max(errors)) <= 1e-5

    def test_param_jacobian_bc_rows_zeroed(self, bkd):
        """Boundary rows of param_jacobian should be zero."""
        physics, param, init_state_1d = _create_parameterized_diffusion_problem(bkd)

        model = create_collocation_model(physics, bkd, parameterization=param)
        adapter = CollocationStateEquationWithJacobianAdapter(
            model, bkd, parameterization=param
        )

        param_2d = bkd.array([0.3, 0.1])[:, None]
        sol = adapter.solve(init_state_1d[:, None], param_2d)

        pjac = adapter.param_jacobian(sol, param_2d)

        # Check that BC rows are zero
        for idx in adapter._bc_indices:
            bkd.assert_allclose(
                pjac[idx, :],
                bkd.zeros((adapter.nparams(),)),
                atol=1e-15,
            )


class TestSteadyForwardModel:
    def test_call_matches_direct_solve(self, bkd):
        """__call__ with identity functional matches solve_steady result."""
        physics, param, init_state = _create_parameterized_diffusion_problem(bkd)

        forward_model = SteadyForwardModel(
            physics, bkd, init_state, parameterization=param
        )

        param_1d = bkd.array([0.3, 0.1])
        samples = param_1d[:, None]  # (nvars, 1)

        # Direct solve
        param.apply(param_1d)
        model = create_collocation_model(physics, bkd, parameterization=param)
        u_direct = model.solve_steady(init_state)

        # Forward model
        qoi = forward_model(samples)

        # With identity functional, QoI = full solution
        bkd.assert_allclose(qoi[:, 0], u_direct, rtol=1e-8)

    def test_call_works(self, bkd):
        """SteadyForwardModel __call__ works."""
        physics, param, init_state = _create_parameterized_diffusion_problem(bkd)
        fwd = SteadyForwardModel(physics, bkd, init_state, parameterization=param)
        samples = bkd.array([0.3, 0.1])[:, None]
        result = fwd(samples)
        assert result.shape[0] == fwd.nqoi()
        assert result.shape[1] == 1

    def test_call_multiple_samples(self, bkd):
        """__call__ handles multiple parameter samples correctly."""
        physics, param, init_state = _create_parameterized_diffusion_problem(bkd)

        forward_model = SteadyForwardModel(
            physics, bkd, init_state, parameterization=param
        )

        np.random.seed(42)
        nsamples = 3
        param_array = np.random.uniform(-0.3, 0.3, (2, nsamples))
        samples = bkd.asarray(param_array)

        result = forward_model(samples)
        assert result.shape[0] == forward_model.nqoi()
        assert result.shape[1] == nsamples

        # Verify each sample individually
        for ii in range(nsamples):
            single_result = forward_model(samples[:, ii : ii + 1])
            bkd.assert_allclose(result[:, ii : ii + 1], single_result, rtol=1e-10)

    def test_nvars_nqoi(self, bkd):
        """nvars and nqoi are correct for default functional."""
        physics, param, init_state = _create_parameterized_diffusion_problem(bkd)

        forward_model = SteadyForwardModel(
            physics, bkd, init_state, parameterization=param
        )

        assert forward_model.nvars() == 2
        assert forward_model.nqoi() == physics.nstates()

    def test_jacobian_works(self, bkd):
        """SteadyForwardModel jacobian works."""
        physics, param, init_state = _create_parameterized_diffusion_problem(bkd)
        fwd = SteadyForwardModel(physics, bkd, init_state, parameterization=param)
        fwd_jac = fwd.derivatives().jacobian
        assert fwd_jac is not None
        sample = bkd.array([0.3, 0.1])[:, None]
        jac = fwd_jac(sample)
        assert jac.shape == (fwd.nqoi(), fwd.nvars())

    def test_jacobian_with_identity_functional(self, bkd):
        """Jacobian with identity functional passes DerivativeChecker."""
        physics, param, init_state = _create_parameterized_diffusion_problem(bkd)

        forward_model = SteadyForwardModel(
            physics, bkd, init_state, parameterization=param
        )

        param_2d = bkd.array([0.3, 0.1])[:, None]

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=forward_model.nqoi(),
            nvars=forward_model.nvars(),
            fun=forward_model,
            jacobian=forward_model.derivatives().jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(param_2d, direction=None, relative=True)[0]
        assert float(bkd.min(errors) / bkd.max(errors)) <= 1e-5

    def test_jacobian_with_subset_functional(self, bkd):
        """Jacobian with subset functional (2 of nstates) via DerivativeChecker."""
        physics, param, init_state = _create_parameterized_diffusion_problem(bkd)
        nstates = physics.nstates()
        nparams = param.nparams()

        # Pick 3 interior indices as QoI
        subset = bkd.asarray(np.array([3, 7, 12]))
        functional = SubsetOfStatesAdjointFunctional(nstates, nparams, subset, bkd)

        forward_model = SteadyForwardModel(
            physics,
            bkd,
            init_state,
            functional=functional,
            parameterization=param,
        )
        assert forward_model.nqoi() == 3

        param_2d = bkd.array([0.3, 0.1])[:, None]

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=forward_model.nqoi(),
            nvars=forward_model.nvars(),
            fun=forward_model,
            jacobian=forward_model.derivatives().jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(param_2d, direction=None, relative=True)[0]
        assert float(bkd.min(errors) / bkd.max(errors)) <= 1e-5

    def test_implicit_function_derivative_checker(self, bkd):
        """Full ImplicitFunctionDerivativeChecker validates all derivatives."""
        physics, param, init_state = _create_parameterized_diffusion_problem(bkd)
        nstates = physics.nstates()
        nparams = param.nparams()

        subset = bkd.asarray(np.array([3, 7, 12]))
        functional = SubsetOfStatesAdjointFunctional(nstates, nparams, subset, bkd)

        model = create_collocation_model(physics, bkd, parameterization=param)
        state_eq = CollocationStateEquationWithJacobianAdapter(
            model, bkd, parameterization=param
        )
        adjoint_op = VectorAdjointOperatorWithJacobian(state_eq, functional)

        init_state_2d = init_state[:, None]
        param_2d = bkd.array([0.3, 0.1])[:, None]

        derivative_checker = ImplicitFunctionDerivativeChecker(adjoint_op)
        tols = derivative_checker.get_derivative_tolerances(1e-5)
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 12))
        derivative_checker.check_derivatives(
            init_state_2d, param_2d, tols, fd_eps=fd_eps, verbosity=0
        )

    def test_no_jacobian_with_eval_only_param(self, bkd):
        """Eval-only parameterization -> bundle jacobian is None.

        Field maps must provide jacobian (protocol-required), so the
        evaluation-only negative path lives at the parameterization
        layer: a parameterization without param_jacobian yields a
        forward model whose bundle declares no jacobian.
        """
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        mesh_obj = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        def forcing(t):
            return (math.pi**2) * bkd.sin(math.pi * nodes)

        physics = AdvectionDiffusionReaction(
            basis,
            bkd,
            diffusion=1.0,
            forcing=forcing,
        )
        left_idx = mesh_obj.boundary_indices(0)
        right_idx = mesh_obj.boundary_indices(1)
        physics.set_boundary_conditions(
            [
                zero_dirichlet_bc(bkd, left_idx),
                zero_dirichlet_bc(bkd, right_idx),
            ]
        )

        class EvalOnlyParameterization:
            def __init__(self, physics):
                self._physics = physics

            def nparams(self):
                return 2

            def physics(self):
                return self._physics

            def apply(self, params_1d):
                phi0 = bkd.ones((npts,))
                field = (
                    bkd.full((npts,), 1.0) + params_1d[0] * phi0 + params_1d[1] * nodes
                )
                self._physics.set_diffusion(ConstantInTimeField(field))

            def owned_coefficients(self):
                return ("mock_coef",)

            def param_derivatives(self):
                return ParamDerivatives.none()

        dp = EvalOnlyParameterization(physics)

        init_state = bkd.zeros((npts,))
        fwd = SteadyForwardModel(physics, bkd, init_state, parameterization=dp)
        assert fwd.derivatives().jacobian is None
        # __call__ still works
        samples = bkd.array([0.3, 0.1])[:, None]
        result = fwd(samples)
        assert result.shape[0] == fwd.nqoi()
        assert result.shape[1] == 1

    def test_no_hvp_with_linear_param(self, bkd):
        """SteadyForwardModel's Derivatives are jacobian-only (it has
        no HVP path), independent of the parameterization's tier."""
        physics, param, init_state = _create_parameterized_diffusion_problem(bkd)
        fwd = SteadyForwardModel(physics, bkd, init_state, parameterization=param)
        assert fwd.derivatives().hvp is None

    def test_protocol_isinstance_with_jacobian(self, bkd):
        """Forward model with jacobian satisfies FunctionProtocol."""
        physics, param, init_state = _create_parameterized_diffusion_problem(bkd)
        fwd = SteadyForwardModel(physics, bkd, init_state, parameterization=param)
        assert isinstance(fwd, FunctionProtocol)
        derivs = fwd.derivatives()
        assert derivs.jacobian is not None
        assert derivs.hvp is None

    def test_torch_autograd_jacobian(self, torch_bkd):
        """Torch autograd.functional.jacobian matches fwd.jacobian."""
        import torch

        bkd = torch_bkd
        physics, param, init_state = _create_parameterized_diffusion_problem(bkd)
        fwd = SteadyForwardModel(physics, bkd, init_state, parameterization=param)
        sample = torch.tensor([0.3, 0.1], dtype=torch.float64)

        def fwd_call(p):
            return fwd(p[:, None])[:, 0]

        autograd_jac = torch.autograd.functional.jacobian(fwd_call, sample)
        analytical_jac = fwd.derivatives().jacobian(sample[:, None])
        bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-8, atol=1e-12)

    def test_protocol_isinstance_eval_only(self, bkd):
        """Eval-only forward model satisfies only FunctionProtocol."""
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        mesh_obj = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        def forcing(t):
            return (math.pi**2) * bkd.sin(math.pi * nodes)

        physics = AdvectionDiffusionReaction(
            basis,
            bkd,
            diffusion=1.0,
            forcing=forcing,
        )
        left_idx = mesh_obj.boundary_indices(0)
        right_idx = mesh_obj.boundary_indices(1)
        physics.set_boundary_conditions(
            [
                zero_dirichlet_bc(bkd, left_idx),
                zero_dirichlet_bc(bkd, right_idx),
            ]
        )

        class EvalOnlyParameterization:
            def __init__(self, physics):
                self._physics = physics

            def nparams(self):
                return 1

            def physics(self):
                return self._physics

            def apply(self, params_1d):
                field = bkd.full((npts,), 1.0) + params_1d[0] * bkd.ones((npts,))
                self._physics.set_diffusion(ConstantInTimeField(field))

            def owned_coefficients(self):
                return ("mock_coef",)

            def param_derivatives(self):
                return ParamDerivatives.none()

        dp = EvalOnlyParameterization(physics)

        init_state = bkd.zeros((npts,))
        fwd = SteadyForwardModel(physics, bkd, init_state, parameterization=dp)
        assert isinstance(fwd, FunctionProtocol)
        # eval-only parameterization: the bundle declares no capability
        assert fwd.derivatives().jacobian is None


class TestSteadyForwardModelTiers:
    """Construction-time derivative-tier selection."""

    def _scalar_functional(self, bkd, npts, nparams):
        from pyapprox.optimization.implicitfunction.functionals.weighted_sum import (
            WeightedSumFunctional,
        )

        weights = bkd.copy(bkd.zeros((npts, 1)))
        weights[npts // 2] = 1.0
        return WeightedSumFunctional(weights, nparams, bkd)

    def _scalar_model(self, bkd, npts=16):
        physics, param, init_state = (
            _create_parameterized_diffusion_problem(bkd, npts)
        )
        functional = self._scalar_functional(bkd, npts, param.nparams())
        fwd = SteadyForwardModel(
            physics,
            bkd,
            init_state,
            functional=functional,
            parameterization=param,
        )
        return fwd, param

    def test_hvp_tier_selected_for_scalar_qoi(self, bkd):
        """Second-order bundle + hvp physics + Dirichlet-only BCs +
        scalar QoI exposes Derivatives.second_order."""
        fwd, _ = self._scalar_model(bkd)
        derivs = fwd.derivatives()
        assert derivs.jacobian is not None
        assert derivs.hvp is not None

    def test_jacobian_and_hvp_derivative_checker(self, numpy_bkd):
        """DerivativeChecker validates the model's jacobian and hvp,
        plus the FD-noise-immune symmetry identity at 1e-12."""
        bkd = numpy_bkd
        fwd, _ = self._scalar_model(bkd)
        derivs = fwd.derivatives()
        assert derivs.hvp is not None
        sample = bkd.asarray(np.array([[0.3], [-0.2]]))

        checker = DerivativeChecker(fwd)
        errors = checker.check_derivatives(sample, verbosity=0)
        assert checker.error_ratio(errors[0]) <= 1e-6
        # Measured V-shape (decays to 1.4e-7 then roundoff climbs):
        # the scalar-QoI HVP FD floor sits above the jacobian's,
        # nudging the clean ratio marginally past 1e-6 — noise, not a
        # plateau.
        assert bkd.to_float(bkd.min(errors[1])) <= 1e-6
        assert checker.error_ratio(errors[1]) <= 5e-6

        vvec = bkd.asarray(np.array([[0.7], [0.4]]))
        uvec = bkd.asarray(np.array([[-0.5], [0.9]]))
        h_v = derivs.hvp(sample, vvec)
        h_u = derivs.hvp(sample, uvec)
        bkd.assert_allclose(
            bkd.sum(h_v * uvec), bkd.sum(h_u * vvec), rtol=1e-12
        )

    def test_vector_qoi_stays_first_order(self, bkd):
        """The default all-states functional keeps the Jacobian tier
        (no warning: the downgrade is functional-driven)."""
        physics, param, init_state = (
            _create_parameterized_diffusion_problem(bkd)
        )
        fwd = SteadyForwardModel(
            physics, bkd, init_state, parameterization=param
        )
        derivs = fwd.derivatives()
        assert derivs.jacobian is not None
        assert derivs.hvp is None

    def test_flux_bc_downgrades_with_warning(self, bkd):
        """A coefficient-dependent flux BC row forces the Jacobian
        tier with a UserWarning."""
        import pytest
        from pyapprox.pde.collocation.boundary.robin import (
            flux_neumann_bc,
        )

        npts = 16
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        mesh_obj = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        def forcing(t):
            return (math.pi**2) * bkd.sin(math.pi * nodes)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=2.0, forcing=forcing
        )
        physics.set_boundary_conditions(
            [
                zero_dirichlet_bc(bkd, mesh_obj.boundary_indices(0)),
                flux_neumann_bc(
                    bkd,
                    mesh_obj.boundary_indices(1),
                    bkd.asarray(np.array([[1.0]])),
                    physics,
                    values=0.0,
                ),
            ]
        )
        fm = BasisExpansion(
            bkd, 2.0, [bkd.ones((npts,)), nodes]
        )
        param = create_diffusion_parameterization(physics, bkd, fm)
        functional = self._scalar_functional(bkd, npts, param.nparams())
        with pytest.warns(UserWarning, match="downgraded"):
            fwd = SteadyForwardModel(
                physics,
                bkd,
                bkd.zeros((npts,)),
                functional=functional,
                parameterization=param,
            )
        derivs = fwd.derivatives()
        assert derivs.jacobian is not None
        assert derivs.hvp is None

    def test_inexact_wrapper_smoke(self, numpy_bkd):
        """The modernized model stays FunctionProtocol-consumable by
        the OUU InexactWrapper."""
        bkd = numpy_bkd
        from pyapprox.optimization.minimize.inexact.fixed import (
            FixedSampleStrategy,
        )
        from pyapprox.optimization.minimize.inexact.wrapper import (
            InexactWrapper,
        )
        from pyapprox.risk import SampleAverageMean

        fwd, param = self._scalar_model(bkd)
        quad_samples = bkd.asarray(np.array([[-0.2, 0.0, 0.2]]))
        quad_weights = bkd.asarray(
            np.array([1.0 / 4, 1.0 / 2, 1.0 / 4])
        )
        wrapper = InexactWrapper(
            model=fwd,
            stat=SampleAverageMean(bkd),
            strategy=FixedSampleStrategy(quad_samples, quad_weights, bkd),
            design_indices=[1],
            bkd=bkd,
        )
        design_sample = bkd.asarray(np.array([[0.1]]))
        value = wrapper(design_sample)
        assert value.shape[0] == 1
        assert math.isfinite(bkd.to_float(value[0, 0]))
