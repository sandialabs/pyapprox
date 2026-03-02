"""Tests for physics parameterization implementations."""

import math

import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.pde.collocation.forward_models.steady import (
    SteadyForwardModel,
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
from pyapprox.pde.field_maps.kle_factory import (
    create_lognormal_kle_field_map,
)
from pyapprox.pde.field_maps.scalar import (
    ScalarAmplitude,
)
from pyapprox.pde.parameterizations.composite import (
    CompositeParameterization,
)
from pyapprox.pde.parameterizations.diffusion import (
    DiffusionParameterization,
    create_diffusion_parameterization,
)
from pyapprox.pde.parameterizations.forcing import (
    ForcingParameterization,
)
from pyapprox.pde.parameterizations.protocol import (
    ParameterizationProtocol,
)
from pyapprox.pde.parameterizations.reaction import (
    ReactionParameterization,
)


def _create_diffusion_physics_and_basis(bkd, npts=20):
    """Create base ADR physics with BCs for testing parameterizations."""
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
    bc_left = zero_dirichlet_bc(bkd, left_idx)
    bc_right = zero_dirichlet_bc(bkd, right_idx)
    physics.set_boundary_conditions([bc_left, bc_right])

    return physics, basis, nodes


class TestParameterizations:
    def test_diffusion_isinstance(self, bkd) -> None:
        """DiffusionParameterization satisfies ParameterizationProtocol."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm = BasisExpansion(bkd, 1.0, [phi0])
        dp = create_diffusion_parameterization(bkd, basis, fm)
        assert isinstance(dp, ParameterizationProtocol)

    def test_diffusion_init_type_error(self, bkd) -> None:
        """DiffusionParameterization raises TypeError for non-FieldMap."""
        with pytest.raises(TypeError):
            DiffusionParameterization("not_a_field_map", [], bkd)

    def test_diffusion_apply_sets_field(self, bkd) -> None:
        """DiffusionParameterization.apply sets diffusion on physics."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        phi1 = nodes
        fm = BasisExpansion(bkd, 1.0, [phi0, phi1])
        dp = create_diffusion_parameterization(bkd, basis, fm)

        params = bkd.array([0.5, -0.3])
        dp.apply(physics, params)

        # After apply, diffusion should be 1.0 + 0.5*1 + (-0.3)*nodes
        expected_diff = bkd.full((npts,), 1.0) + 0.5 * phi0 + (-0.3) * phi1
        actual_diff = physics._get_diffusion(0.0)
        bkd.assert_allclose(actual_diff, expected_diff, rtol=1e-12)

    def test_diffusion_param_jacobian_fd(self, bkd) -> None:
        """DiffusionParameterization.param_jacobian matches FD."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        num_kle_terms = 2
        mesh_coords = ((nodes + 1.0) / 2.0)[None, :]
        mean_log = bkd.zeros((npts,))
        fm = create_lognormal_kle_field_map(
            mesh_coords,
            mean_log,
            bkd,
            num_kle_terms=num_kle_terms,
            sigma=0.3,
        )
        dp = create_diffusion_parameterization(bkd, basis, fm)

        # Get a non-trivial state by solving with some parameters
        state = bkd.sin(math.pi * nodes)
        time = 0.0

        def residual_of_params(samples):
            results = []
            for i in range(samples.shape[1]):
                p = samples[:, i]
                dp.apply(physics, p)
                res = physics.residual(state, time)
                results.append(res)
            return bkd.stack(results, axis=1)

        def jac_of_params(sample):
            p = sample[:, 0]
            dp.apply(physics, p)
            return dp.param_jacobian(physics, state, time, p)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=npts,
            nvars=dp.nparams(),
            fun=residual_of_params,
            jacobian=jac_of_params,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        params = bkd.array([0.1, -0.1])[:, None]
        errors = checker.check_derivatives(params)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        assert ratio <= 1e-5

    def test_diffusion_param_jacobian_autograd(self, torch_bkd) -> None:
        """Torch autograd matches DiffusionParameterization.param_jacobian."""
        import torch

        bkd = torch_bkd
        # Use physics WITHOUT BCs so residual is pure PDE operator
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        def forcing(t):
            return (math.pi**2) * bkd.sin(math.pi * nodes)

        physics = AdvectionDiffusionReaction(
            basis,
            bkd,
            diffusion=1.0,
            forcing=forcing,
        )

        phi0 = bkd.ones((npts,))
        phi1 = nodes
        fm = BasisExpansion(bkd, 1.0, [phi0, phi1])
        dp = create_diffusion_parameterization(bkd, basis, fm)

        # Use a state that's non-zero everywhere to avoid near-zero issues
        state = bkd.cos(0.5 * math.pi * nodes) + 1.0
        time = 0.0

        params = torch.tensor([0.3, -0.1], dtype=torch.float64)

        def torch_residual(p):
            dp.apply(physics, p)
            return physics.residual(state, time)

        autograd_jac = torch.autograd.functional.jacobian(torch_residual, params)
        dp.apply(physics, params)
        analytical_jac = dp.param_jacobian(physics, state, time, params)
        bkd.assert_allclose(analytical_jac, autograd_jac, atol=1e-12)

    def test_diffusion_initial_param_jacobian_zeros(self, bkd) -> None:
        """DiffusionParameterization.initial_param_jacobian returns zeros."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm = BasisExpansion(bkd, 1.0, [phi0])
        dp = create_diffusion_parameterization(bkd, basis, fm)
        params = bkd.array([0.5])
        result = dp.initial_param_jacobian(physics, params)
        expected = bkd.zeros((npts, 1))
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_forcing_apply_and_jacobian(self, bkd) -> None:
        """ForcingParameterization.apply and param_jacobian work correctly."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        base_forcing = bkd.sin(math.pi * nodes)
        fm = ScalarAmplitude(bkd, base_forcing)
        fp = ForcingParameterization(fm, bkd)

        assert isinstance(fp, ParameterizationProtocol)
        assert fp.nparams() == 1

        state = bkd.sin(math.pi * nodes)
        time = 0.0

        def residual_of_params(samples):
            results = []
            for i in range(samples.shape[1]):
                p = samples[:, i]
                fp.apply(physics, p)
                res = physics.residual(state, time)
                results.append(res)
            return bkd.stack(results, axis=1)

        def jac_of_params(sample):
            p = sample[:, 0]
            fp.apply(physics, p)
            return fp.param_jacobian(physics, state, time, p)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=npts,
            nvars=1,
            fun=residual_of_params,
            jacobian=jac_of_params,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        params = bkd.array([2.0])[:, None]
        errors = checker.check_derivatives(params)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        assert ratio <= 1e-5

    def test_reaction_apply_and_jacobian(self, bkd) -> None:
        """ReactionParameterization.apply and param_jacobian work correctly."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm = BasisExpansion(bkd, 0.0, [phi0])
        rp = ReactionParameterization(fm, bkd)

        assert isinstance(rp, ParameterizationProtocol)
        assert rp.nparams() == 1

        state = bkd.sin(math.pi * nodes)
        time = 0.0

        def residual_of_params(samples):
            results = []
            for i in range(samples.shape[1]):
                p = samples[:, i]
                rp.apply(physics, p)
                res = physics.residual(state, time)
                results.append(res)
            return bkd.stack(results, axis=1)

        def jac_of_params(sample):
            p = sample[:, 0]
            rp.apply(physics, p)
            return rp.param_jacobian(physics, state, time, p)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=npts,
            nvars=1,
            fun=residual_of_params,
            jacobian=jac_of_params,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        params = bkd.array([-0.5])[:, None]
        errors = checker.check_derivatives(params)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        assert ratio <= 1e-5

    def test_composite_isinstance(self, bkd) -> None:
        """CompositeParameterization satisfies ParameterizationProtocol."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm_d = BasisExpansion(bkd, 1.0, [phi0])
        dp = create_diffusion_parameterization(bkd, basis, fm_d)

        base_forcing = bkd.sin(math.pi * nodes)
        fm_f = ScalarAmplitude(bkd, base_forcing)
        fp = ForcingParameterization(fm_f, bkd)

        comp = CompositeParameterization([dp, fp], bkd)
        assert isinstance(comp, ParameterizationProtocol)

    def test_composite_init_type_error(self, bkd) -> None:
        """CompositeParameterization raises TypeError for non-protocol part."""
        with pytest.raises(TypeError):
            CompositeParameterization(["not_a_param"], bkd)

    def test_composite_nparams(self, bkd) -> None:
        """CompositeParameterization.nparams is sum of parts."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        phi1 = nodes
        fm_d = BasisExpansion(bkd, 1.0, [phi0, phi1])
        dp = create_diffusion_parameterization(bkd, basis, fm_d)

        base_forcing = bkd.sin(math.pi * nodes)
        fm_f = ScalarAmplitude(bkd, base_forcing)
        fp = ForcingParameterization(fm_f, bkd)

        comp = CompositeParameterization([dp, fp], bkd)
        assert comp.nparams() == 3

    def test_composite_param_jacobian_fd(self, bkd) -> None:
        """CompositeParameterization.param_jacobian matches FD."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        num_kle_terms = 2
        mesh_coords = ((nodes + 1.0) / 2.0)[None, :]
        mean_log = bkd.zeros((npts,))
        fm_d = create_lognormal_kle_field_map(
            mesh_coords,
            mean_log,
            bkd,
            num_kle_terms=num_kle_terms,
            sigma=0.3,
        )
        dp = create_diffusion_parameterization(bkd, basis, fm_d)

        base_forcing = bkd.sin(math.pi * nodes)
        fm_f = ScalarAmplitude(bkd, base_forcing)
        fp = ForcingParameterization(fm_f, bkd)

        comp = CompositeParameterization([dp, fp], bkd)
        state = bkd.sin(math.pi * nodes)
        time = 0.0

        def residual_of_params(samples):
            results = []
            for i in range(samples.shape[1]):
                p = samples[:, i]
                comp.apply(physics, p)
                res = physics.residual(state, time)
                results.append(res)
            return bkd.stack(results, axis=1)

        def jac_of_params(sample):
            p = sample[:, 0]
            comp.apply(physics, p)
            return comp.param_jacobian(physics, state, time, p)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=npts,
            nvars=comp.nparams(),
            fun=residual_of_params,
            jacobian=jac_of_params,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        params = bkd.array([0.1, -0.1, 2.0])[:, None]
        errors = checker.check_derivatives(params)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        assert ratio <= 1e-5

    def test_composite_initial_param_jacobian(self, bkd) -> None:
        """CompositeParameterization.initial_param_jacobian block structure."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm_d = BasisExpansion(bkd, 1.0, [phi0])
        dp = create_diffusion_parameterization(bkd, basis, fm_d)

        base_forcing = bkd.ones((npts,))
        fm_f = ScalarAmplitude(bkd, base_forcing)
        fp = ForcingParameterization(fm_f, bkd)

        comp = CompositeParameterization([dp, fp], bkd)
        params = bkd.array([0.5, 1.0])
        result = comp.initial_param_jacobian(physics, params)
        expected = bkd.zeros((npts, 2))
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_composite_dynamic_binding_all_differentiable(self, bkd) -> None:
        """Composite with all-differentiable parts HAS param_jacobian."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm = BasisExpansion(bkd, 1.0, [phi0])
        dp = create_diffusion_parameterization(bkd, basis, fm)
        comp = CompositeParameterization([dp], bkd)
        assert hasattr(comp, "param_jacobian")

    def test_composite_dynamic_binding_eval_only(self, bkd) -> None:
        """Composite with eval-only part does NOT have param_jacobian."""

        class EvalOnlyFieldMap:
            def nvars(self) -> int:
                return 1

            def __call__(self, params_1d):
                return params_1d

        class EvalOnlyParam:
            def nparams(self) -> int:
                return 1

            def apply(self, physics, params_1d):
                pass

        comp = CompositeParameterization([EvalOnlyParam()], bkd)
        assert not hasattr(comp, "param_jacobian")

    def test_composite_append_removes_param_jacobian(self, bkd) -> None:
        """Appending non-differentiable part removes param_jacobian."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm = BasisExpansion(bkd, 1.0, [phi0])
        dp = create_diffusion_parameterization(bkd, basis, fm)

        comp = CompositeParameterization([dp], bkd)
        assert hasattr(comp, "param_jacobian")

        class EvalOnlyParam:
            def nparams(self) -> int:
                return 1

            def apply(self, physics, params_1d):
                pass

        comp.append(EvalOnlyParam())
        assert not hasattr(comp, "param_jacobian")

    def test_composite_append_type_error(self, bkd) -> None:
        """CompositeParameterization.append raises TypeError for non-protocol."""
        comp = CompositeParameterization([], bkd)
        with pytest.raises(TypeError):
            comp.append("not_a_param")

    def test_forcing_init_type_error(self, bkd) -> None:
        """ForcingParameterization raises TypeError for non-FieldMap."""
        with pytest.raises(TypeError):
            ForcingParameterization("not_a_field_map", bkd)

    def test_reaction_init_type_error(self, bkd) -> None:
        """ReactionParameterization raises TypeError for non-FieldMap."""
        with pytest.raises(TypeError):
            ReactionParameterization("not_a_field_map", bkd)


class TestCompositeWithSteadyForwardModel:
    """Integration test: CompositeParameterization with SteadyForwardModel."""
    def _create_composite_forward_model(self, bkd) :
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        num_kle_terms = 2
        mesh_coords = ((nodes + 1.0) / 2.0)[None, :]
        mean_log = bkd.zeros((npts,))
        fm_d = create_lognormal_kle_field_map(
            mesh_coords,
            mean_log,
            bkd,
            num_kle_terms=num_kle_terms,
            sigma=0.3,
        )
        dp = create_diffusion_parameterization(bkd, basis, fm_d)

        base_forcing = bkd.sin(math.pi * nodes)
        fm_f = ScalarAmplitude(bkd, base_forcing)
        fp = ForcingParameterization(fm_f, bkd)

        comp = CompositeParameterization([dp, fp], bkd)
        init_state = bkd.zeros((npts,))
        fwd = SteadyForwardModel(physics, bkd, init_state, parameterization=comp)
        return fwd

    def test_nvars_is_sum(self, bkd) -> None:
        """Forward model nvars = nkle + 1 (diffusion + forcing)."""
        fwd = self._create_composite_forward_model(bkd)
        assert fwd.nvars() == 3

    def test_call_works(self, bkd) -> None:
        """Forward model __call__ works with CompositeParameterization."""
        fwd = self._create_composite_forward_model(bkd)
        samples = bkd.array([0.1, -0.1, 2.0])[:, None]
        result = fwd(samples)
        assert result.shape[0] == fwd.nqoi()
        assert result.shape[1] == 1

    def test_jacobian_derivative_checker(self, bkd) -> None:
        """Forward model Jacobian passes DerivativeChecker."""
        fwd = self._create_composite_forward_model(bkd)
        wrapper = FunctionWithJacobianFromCallable(
            nqoi=fwd.nqoi(),
            nvars=fwd.nvars(),
            fun=fwd,
            jacobian=fwd.jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = bkd.array([0.1, -0.1, 2.0])[:, None]
        errors = checker.check_derivatives(sample)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        assert ratio <= 1e-5
