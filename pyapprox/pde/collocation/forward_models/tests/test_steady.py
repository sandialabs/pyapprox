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
    FunctionWithJacobianAndHVPProtocol,
    FunctionWithJacobianProtocol,
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
from pyapprox.pde.collocation.forward_models.steady import (
    CollocationStateEquationAdapter,
    SteadyForwardModel,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    create_uniform_mesh_1d,
)
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.time_integration.collocation_model import (
    CollocationModel,
)
from pyapprox.pde.field_maps.basis_expansion import (
    BasisExpansion,
)
from pyapprox.pde.parameterizations.diffusion import (
    DiffusionParameterization,
    create_diffusion_parameterization,
)


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
    param = create_diffusion_parameterization(bkd, basis, fm)
    init_state = bkd.zeros((npts,))
    return physics, param, init_state


class TestCollocationStateEquationAdapter:
    def test_solve_matches_collocation_model(self, bkd):
        """Adapter solve matches direct CollocationModel.solve_steady."""
        physics, param, init_state_1d = _create_parameterized_diffusion_problem(bkd)

        model = CollocationModel(physics, bkd, parameterization=param)
        adapter = CollocationStateEquationAdapter(model, bkd, parameterization=param)

        param_1d = bkd.array([0.5, 0.1])
        param_2d = param_1d[:, None]
        init_state_2d = init_state_1d[:, None]

        # Direct solve
        param.apply(physics, param_1d)
        u_direct = model.solve_steady(init_state_1d)

        # Adapter solve
        u_adapter = adapter.solve(init_state_2d, param_2d)

        bkd.assert_allclose(u_adapter[:, 0], u_direct, rtol=1e-10)

    def test_residual_at_solution_is_zero(self, bkd):
        """Residual should be near-zero at the converged solution."""
        physics, param, init_state_1d = _create_parameterized_diffusion_problem(bkd)

        model = CollocationModel(physics, bkd, parameterization=param)
        adapter = CollocationStateEquationAdapter(model, bkd, parameterization=param)

        param_2d = bkd.array([0.3, -0.1])[:, None]
        init_state_2d = init_state_1d[:, None]

        sol = adapter.solve(init_state_2d, param_2d)
        residual = adapter(sol, param_2d)

        bkd.assert_allclose(residual, bkd.zeros_like(residual), atol=1e-8)

    def test_derivative_checker_state_jacobian(self, bkd):
        """DerivativeChecker validates state Jacobian via FD."""
        physics, param, init_state_1d = _create_parameterized_diffusion_problem(bkd)

        model = CollocationModel(physics, bkd, parameterization=param)
        adapter = CollocationStateEquationAdapter(model, bkd, parameterization=param)

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

        model = CollocationModel(physics, bkd, parameterization=param)
        adapter = CollocationStateEquationAdapter(model, bkd, parameterization=param)

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

        model = CollocationModel(physics, bkd, parameterization=param)
        adapter = CollocationStateEquationAdapter(model, bkd, parameterization=param)

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
        param.apply(physics, param_1d)
        model = CollocationModel(physics, bkd, parameterization=param)
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
        assert hasattr(fwd, "jacobian")
        sample = bkd.array([0.3, 0.1])[:, None]
        jac = fwd.jacobian(sample)
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
            jacobian=forward_model.jacobian,
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
            jacobian=forward_model.jacobian,
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

        model = CollocationModel(physics, bkd, parameterization=param)
        state_eq = CollocationStateEquationAdapter(model, bkd, parameterization=param)
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
        """Eval-only field map -> hasattr(fwd, 'jacobian') is False."""
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

        class EvalOnlyFieldMap:
            def nvars(self):
                return 2

            def __call__(self, params_1d):
                phi0 = bkd.ones((npts,))
                return (
                    bkd.full((npts,), 1.0) + params_1d[0] * phi0 + params_1d[1] * nodes
                )

        D_mats = [basis.derivative_matrix()]
        fm = EvalOnlyFieldMap()
        dp = DiffusionParameterization(fm, D_mats, bkd)

        init_state = bkd.zeros((npts,))
        fwd = SteadyForwardModel(physics, bkd, init_state, parameterization=dp)
        assert not hasattr(fwd, "jacobian")
        # __call__ still works
        samples = bkd.array([0.3, 0.1])[:, None]
        result = fwd(samples)
        assert result.shape[0] == fwd.nqoi()
        assert result.shape[1] == 1

    def test_no_hvp_with_linear_param(self, bkd):
        """Linear BasisExpansion has no HVP -> hasattr(fwd, 'hvp') is False."""
        physics, param, init_state = _create_parameterized_diffusion_problem(bkd)
        fwd = SteadyForwardModel(physics, bkd, init_state, parameterization=param)
        assert not hasattr(fwd, "hvp")

    def test_protocol_isinstance_with_jacobian(self, bkd):
        """Forward model with jacobian satisfies FunctionWithJacobianProtocol."""
        physics, param, init_state = _create_parameterized_diffusion_problem(bkd)
        fwd = SteadyForwardModel(physics, bkd, init_state, parameterization=param)
        assert isinstance(fwd, FunctionProtocol)
        assert isinstance(fwd, FunctionWithJacobianProtocol)
        assert not isinstance(fwd, FunctionWithJacobianAndHVPProtocol)

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
        analytical_jac = fwd.jacobian(sample[:, None])
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

        class EvalOnlyFieldMap:
            def nvars(self):
                return 1

            def __call__(self, params_1d):
                return bkd.full((npts,), 1.0) + params_1d[0] * bkd.ones((npts,))

        D_mats = [basis.derivative_matrix()]
        fm = EvalOnlyFieldMap()
        dp = DiffusionParameterization(fm, D_mats, bkd)

        init_state = bkd.zeros((npts,))
        fwd = SteadyForwardModel(physics, bkd, init_state, parameterization=dp)
        assert isinstance(fwd, FunctionProtocol)
        assert not isinstance(fwd, FunctionWithJacobianProtocol)
