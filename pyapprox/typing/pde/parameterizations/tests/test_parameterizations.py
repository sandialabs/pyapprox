"""Tests for physics parameterization implementations."""

import math
import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.typing.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.typing.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.typing.pde.collocation.mesh import (
    create_uniform_mesh_1d,
    TransformedMesh1D,
)
from pyapprox.typing.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.typing.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.typing.pde.collocation.forward_models.steady import (
    SteadyForwardModel,
)
from pyapprox.typing.pde.field_maps.basis_expansion import (
    BasisExpansion,
)
from pyapprox.typing.pde.field_maps.scalar import (
    ScalarAmplitude,
)
from pyapprox.typing.pde.field_maps.protocol import (
    FieldMapProtocol,
)
from pyapprox.typing.pde.parameterizations.protocol import (
    ParameterizationProtocol,
)
from pyapprox.typing.pde.parameterizations.diffusion import (
    DiffusionParameterization,
    create_diffusion_parameterization,
)
from pyapprox.typing.pde.parameterizations.forcing import (
    ForcingParameterization,
)
from pyapprox.typing.pde.parameterizations.reaction import (
    ReactionParameterization,
)
from pyapprox.typing.pde.parameterizations.composite import (
    CompositeParameterization,
)
from pyapprox.typing.pde.field_maps.kle_factory import (
    create_lognormal_kle_field_map,
)


def _create_diffusion_physics_and_basis(bkd, npts=20):
    """Create base ADR physics with BCs for testing parameterizations."""
    mesh = TransformedMesh1D(npts, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    mesh_obj = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
    nodes = basis.nodes()

    forcing = lambda t: (math.pi ** 2) * bkd.sin(math.pi * nodes)

    physics = AdvectionDiffusionReaction(
        basis, bkd, diffusion=1.0, forcing=forcing,
    )

    left_idx = mesh_obj.boundary_indices(0)
    right_idx = mesh_obj.boundary_indices(1)
    bc_left = zero_dirichlet_bc(bkd, left_idx)
    bc_right = zero_dirichlet_bc(bkd, right_idx)
    physics.set_boundary_conditions([bc_left, bc_right])

    return physics, basis, nodes


class TestParameterizations(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_diffusion_isinstance(self) -> None:
        """DiffusionParameterization satisfies ParameterizationProtocol."""
        bkd = self._bkd
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm = BasisExpansion(bkd, 1.0, [phi0])
        dp = create_diffusion_parameterization(bkd, basis, fm)
        self.assertTrue(isinstance(dp, ParameterizationProtocol))

    def test_diffusion_init_type_error(self) -> None:
        """DiffusionParameterization raises TypeError for non-FieldMap."""
        bkd = self._bkd
        with self.assertRaises(TypeError):
            DiffusionParameterization("not_a_field_map", [], bkd)

    def test_diffusion_apply_sets_field(self) -> None:
        """DiffusionParameterization.apply sets diffusion on physics."""
        bkd = self._bkd
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

    def test_diffusion_param_jacobian_fd(self) -> None:
        """DiffusionParameterization.param_jacobian matches FD."""
        bkd = self._bkd
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        num_kle_terms = 2
        mesh_coords = ((nodes + 1.0) / 2.0)[None, :]
        mean_log = bkd.zeros((npts,))
        fm = create_lognormal_kle_field_map(
            mesh_coords, mean_log, bkd,
            num_kle_terms=num_kle_terms, sigma=0.3,
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
        self.assertLessEqual(ratio, 1e-5)

    def test_diffusion_initial_param_jacobian_zeros(self) -> None:
        """DiffusionParameterization.initial_param_jacobian returns zeros."""
        bkd = self._bkd
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm = BasisExpansion(bkd, 1.0, [phi0])
        dp = create_diffusion_parameterization(bkd, basis, fm)
        params = bkd.array([0.5])
        result = dp.initial_param_jacobian(physics, params)
        expected = bkd.zeros((npts, 1))
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_forcing_apply_and_jacobian(self) -> None:
        """ForcingParameterization.apply and param_jacobian work correctly."""
        bkd = self._bkd
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        base_forcing = bkd.sin(math.pi * nodes)
        fm = ScalarAmplitude(bkd, base_forcing)
        fp = ForcingParameterization(fm, bkd)

        self.assertTrue(isinstance(fp, ParameterizationProtocol))
        self.assertEqual(fp.nparams(), 1)

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
        self.assertLessEqual(ratio, 1e-5)

    def test_reaction_apply_and_jacobian(self) -> None:
        """ReactionParameterization.apply and param_jacobian work correctly."""
        bkd = self._bkd
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm = BasisExpansion(bkd, 0.0, [phi0])
        rp = ReactionParameterization(fm, bkd)

        self.assertTrue(isinstance(rp, ParameterizationProtocol))
        self.assertEqual(rp.nparams(), 1)

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
        self.assertLessEqual(ratio, 1e-5)

    def test_composite_isinstance(self) -> None:
        """CompositeParameterization satisfies ParameterizationProtocol."""
        bkd = self._bkd
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm_d = BasisExpansion(bkd, 1.0, [phi0])
        dp = create_diffusion_parameterization(bkd, basis, fm_d)

        base_forcing = bkd.sin(math.pi * nodes)
        fm_f = ScalarAmplitude(bkd, base_forcing)
        fp = ForcingParameterization(fm_f, bkd)

        comp = CompositeParameterization([dp, fp], bkd)
        self.assertTrue(isinstance(comp, ParameterizationProtocol))

    def test_composite_init_type_error(self) -> None:
        """CompositeParameterization raises TypeError for non-protocol part."""
        bkd = self._bkd
        with self.assertRaises(TypeError):
            CompositeParameterization(["not_a_param"], bkd)

    def test_composite_nparams(self) -> None:
        """CompositeParameterization.nparams is sum of parts."""
        bkd = self._bkd
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
        self.assertEqual(comp.nparams(), 3)  # 2 diffusion + 1 forcing

    def test_composite_param_jacobian_fd(self) -> None:
        """CompositeParameterization.param_jacobian matches FD."""
        bkd = self._bkd
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        num_kle_terms = 2
        mesh_coords = ((nodes + 1.0) / 2.0)[None, :]
        mean_log = bkd.zeros((npts,))
        fm_d = create_lognormal_kle_field_map(
            mesh_coords, mean_log, bkd,
            num_kle_terms=num_kle_terms, sigma=0.3,
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
        self.assertLessEqual(ratio, 1e-5)

    def test_composite_initial_param_jacobian(self) -> None:
        """CompositeParameterization.initial_param_jacobian block structure."""
        bkd = self._bkd
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

    def test_composite_dynamic_binding_all_differentiable(self) -> None:
        """Composite with all-differentiable parts HAS param_jacobian."""
        bkd = self._bkd
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm = BasisExpansion(bkd, 1.0, [phi0])
        dp = create_diffusion_parameterization(bkd, basis, fm)
        comp = CompositeParameterization([dp], bkd)
        self.assertTrue(hasattr(comp, "param_jacobian"))

    def test_composite_dynamic_binding_eval_only(self) -> None:
        """Composite with eval-only part does NOT have param_jacobian."""
        bkd = self._bkd

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
        self.assertFalse(hasattr(comp, "param_jacobian"))

    def test_composite_append_removes_param_jacobian(self) -> None:
        """Appending non-differentiable part removes param_jacobian."""
        bkd = self._bkd
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm = BasisExpansion(bkd, 1.0, [phi0])
        dp = create_diffusion_parameterization(bkd, basis, fm)

        comp = CompositeParameterization([dp], bkd)
        self.assertTrue(hasattr(comp, "param_jacobian"))

        class EvalOnlyParam:
            def nparams(self) -> int:
                return 1

            def apply(self, physics, params_1d):
                pass

        comp.append(EvalOnlyParam())
        self.assertFalse(hasattr(comp, "param_jacobian"))

    def test_composite_append_type_error(self) -> None:
        """CompositeParameterization.append raises TypeError for non-protocol."""
        bkd = self._bkd
        comp = CompositeParameterization([], bkd)
        with self.assertRaises(TypeError):
            comp.append("not_a_param")

    def test_forcing_init_type_error(self) -> None:
        """ForcingParameterization raises TypeError for non-FieldMap."""
        bkd = self._bkd
        with self.assertRaises(TypeError):
            ForcingParameterization("not_a_field_map", bkd)

    def test_reaction_init_type_error(self) -> None:
        """ReactionParameterization raises TypeError for non-FieldMap."""
        bkd = self._bkd
        with self.assertRaises(TypeError):
            ReactionParameterization("not_a_field_map", bkd)


class TestParameterizationsNumpy(TestParameterizations[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestParameterizationsTorch(TestParameterizations[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def test_diffusion_param_jacobian_autograd(self) -> None:
        """Torch autograd matches DiffusionParameterization.param_jacobian."""
        bkd = self._bkd
        # Use physics WITHOUT BCs so residual is pure PDE operator
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()
        forcing = lambda t: (math.pi ** 2) * bkd.sin(math.pi * nodes)
        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, forcing=forcing,
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

        autograd_jac = torch.autograd.functional.jacobian(
            torch_residual, params
        )
        dp.apply(physics, params)
        analytical_jac = dp.param_jacobian(physics, state, time, params)
        bkd.assert_allclose(analytical_jac, autograd_jac, atol=1e-12)


class TestCompositeWithSteadyForwardModel(Generic[Array], unittest.TestCase):
    """Integration test: CompositeParameterization with SteadyForwardModel."""
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _create_composite_forward_model(self):
        bkd = self._bkd
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        num_kle_terms = 2
        mesh_coords = ((nodes + 1.0) / 2.0)[None, :]
        mean_log = bkd.zeros((npts,))
        fm_d = create_lognormal_kle_field_map(
            mesh_coords, mean_log, bkd,
            num_kle_terms=num_kle_terms, sigma=0.3,
        )
        dp = create_diffusion_parameterization(bkd, basis, fm_d)

        base_forcing = bkd.sin(math.pi * nodes)
        fm_f = ScalarAmplitude(bkd, base_forcing)
        fp = ForcingParameterization(fm_f, bkd)

        comp = CompositeParameterization([dp, fp], bkd)
        init_state = bkd.zeros((npts,))
        fwd = SteadyForwardModel(
            physics, bkd, init_state, parameterization=comp
        )
        return fwd

    def test_nvars_is_sum(self) -> None:
        """Forward model nvars = nkle + 1 (diffusion + forcing)."""
        fwd = self._create_composite_forward_model()
        self.assertEqual(fwd.nvars(), 3)  # 2 KLE + 1 forcing

    def test_call_works(self) -> None:
        """Forward model __call__ works with CompositeParameterization."""
        bkd = self._bkd
        fwd = self._create_composite_forward_model()
        samples = bkd.array([0.1, -0.1, 2.0])[:, None]
        result = fwd(samples)
        self.assertEqual(result.shape[0], fwd.nqoi())
        self.assertEqual(result.shape[1], 1)

    def test_jacobian_derivative_checker(self) -> None:
        """Forward model Jacobian passes DerivativeChecker."""
        bkd = self._bkd
        fwd = self._create_composite_forward_model()
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
        self.assertLessEqual(ratio, 1e-5)


class TestCompositeWithSteadyForwardModelNumpy(
    TestCompositeWithSteadyForwardModel[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCompositeWithSteadyForwardModelTorch(
    TestCompositeWithSteadyForwardModel[torch.Tensor]
):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()
