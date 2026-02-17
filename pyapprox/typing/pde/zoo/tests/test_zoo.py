"""Tests for zoo factory functions."""

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
from pyapprox.typing.interface.functions.protocols import (
    FunctionProtocol,
    FunctionWithJacobianProtocol,
    FunctionWithJacobianAndHVPProtocol,
)
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.typing.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.typing.pde.time.config import TimeIntegrationConfig
from pyapprox.typing.pde.zoo.diffusion import (
    create_steady_diffusion_1d,
    create_transient_diffusion_1d,
)
from pyapprox.typing.pde.field_maps.kle_factory import (
    create_lognormal_kle_field_map,
)


def _make_kle_field_map(bkd, nodes, num_kle_terms=2):
    """Helper: create lognormal KLE field map on Chebyshev nodes."""
    npts = nodes.shape[0]
    mesh_coords = ((nodes + 1.0) / 2.0)[None, :]  # map [-1,1] -> [0,1]
    mean_log = bkd.zeros((npts,))
    return create_lognormal_kle_field_map(
        mesh_coords, mean_log, bkd,
        num_kle_terms=num_kle_terms, sigma=0.3,
    )


class TestSteadyDiffusionZoo(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _create_model(self):
        bkd = self._bkd
        npts = 20
        from pyapprox.typing.pde.collocation.basis import ChebyshevBasis1D
        from pyapprox.typing.pde.collocation.mesh import TransformedMesh1D
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        field_map = _make_kle_field_map(bkd, nodes)
        forcing = lambda t: (math.pi ** 2) * bkd.sin(math.pi * nodes)

        return create_steady_diffusion_1d(
            bkd=bkd,
            npts=npts,
            domain=(-1.0, 1.0),
            forcing=forcing,
            field_map=field_map,
        )

    def test_factory_produces_valid_model(self):
        """Zoo factory produces a working forward model."""
        fwd = self._create_model()
        bkd = self._bkd
        samples = bkd.zeros((2, 1))
        result = fwd(samples)
        self.assertEqual(result.shape[0], fwd.nqoi())
        self.assertEqual(result.shape[1], 1)

    def test_has_jacobian(self):
        """Zoo model has jacobian."""
        fwd = self._create_model()
        self.assertTrue(hasattr(fwd, "jacobian"))

    def test_isinstance_function_protocol(self):
        """Zoo model satisfies FunctionProtocol."""
        fwd = self._create_model()
        self.assertTrue(isinstance(fwd, FunctionProtocol))

    def test_isinstance_jacobian_protocol(self):
        """Zoo model satisfies FunctionWithJacobianProtocol."""
        fwd = self._create_model()
        self.assertTrue(isinstance(fwd, FunctionWithJacobianProtocol))

    def test_jacobian_derivative_checker(self):
        """Zoo model Jacobian passes DerivativeChecker."""
        bkd = self._bkd
        fwd = self._create_model()
        wrapper = FunctionWithJacobianFromCallable(
            nqoi=fwd.nqoi(),
            nvars=fwd.nvars(),
            fun=fwd,
            jacobian=fwd.jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = bkd.array([0.1, -0.1])[:, None]
        errors = checker.check_derivatives(sample)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        self.assertLessEqual(ratio, 1e-5)

    def test_matches_manual_construction(self):
        """Zoo factory matches manually constructed forward model."""
        bkd = self._bkd
        npts = 20
        from pyapprox.typing.pde.collocation.basis import ChebyshevBasis1D
        from pyapprox.typing.pde.collocation.mesh import (
            TransformedMesh1D,
            create_uniform_mesh_1d,
        )
        from pyapprox.typing.pde.collocation.boundary import zero_dirichlet_bc
        from pyapprox.typing.pde.collocation.physics.advection_diffusion import (
            AdvectionDiffusionReaction,
        )
        from pyapprox.typing.pde.parameterizations.diffusion import (
            create_diffusion_parameterization,
        )
        from pyapprox.typing.pde.collocation.forward_models.steady import (
            SteadyForwardModel,
        )

        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        mesh_obj = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        field_map = _make_kle_field_map(bkd, nodes)
        forcing = lambda t: (math.pi ** 2) * bkd.sin(math.pi * nodes)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, forcing=forcing,
        )
        left_idx = mesh_obj.boundary_indices(0)
        right_idx = mesh_obj.boundary_indices(1)
        physics.set_boundary_conditions([
            zero_dirichlet_bc(bkd, left_idx),
            zero_dirichlet_bc(bkd, right_idx),
        ])
        param = create_diffusion_parameterization(bkd, basis, field_map)
        init_state = bkd.zeros((npts,))
        fwd_manual = SteadyForwardModel(
            physics, bkd, init_state, parameterization=param
        )

        fwd_zoo = self._create_model()

        samples = bkd.zeros((2, 1))
        bkd.assert_allclose(
            fwd_zoo(samples), fwd_manual(samples), rtol=1e-10, atol=1e-14
        )
        bkd.assert_allclose(
            fwd_zoo.jacobian(samples), fwd_manual.jacobian(samples),
            rtol=1e-10, atol=1e-14,
        )

    def test_cannot_specify_both_field_map_and_basis_funs(self):
        """Raises ValueError if both field_map and basis_funs given."""
        bkd = self._bkd
        npts = 10
        from pyapprox.typing.pde.collocation.basis import ChebyshevBasis1D
        from pyapprox.typing.pde.collocation.mesh import TransformedMesh1D
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        field_map = _make_kle_field_map(bkd, nodes)

        with self.assertRaises(ValueError):
            create_steady_diffusion_1d(
                bkd=bkd,
                npts=npts,
                domain=(-1.0, 1.0),
                forcing=lambda t: bkd.zeros((npts,)),
                field_map=field_map,
                basis_funs=[bkd.ones((npts,))],
            )


class TestTransientDiffusionZoo(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _create_model(self):
        bkd = self._bkd
        npts = 15
        from pyapprox.typing.pde.collocation.basis import ChebyshevBasis1D
        from pyapprox.typing.pde.collocation.mesh import TransformedMesh1D
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        field_map = _make_kle_field_map(bkd, nodes)

        time_config = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=0.1,
            deltat=0.02,
            newton_tol=1e-10,
            newton_maxiter=20,
        )

        return create_transient_diffusion_1d(
            bkd=bkd,
            npts=npts,
            domain=(-1.0, 1.0),
            init_state_func=lambda nodes: bkd.sin(math.pi * nodes),
            time_config=time_config,
            field_map=field_map,
        )

    def test_factory_produces_valid_model(self):
        """Zoo factory produces a working transient forward model."""
        fwd = self._create_model()
        bkd = self._bkd
        samples = bkd.zeros((2, 1))
        result = fwd(samples)
        self.assertEqual(result.shape[0], fwd.nqoi())
        self.assertEqual(result.shape[1], 1)

    def test_has_jacobian(self):
        """Zoo transient model has jacobian."""
        fwd = self._create_model()
        self.assertTrue(hasattr(fwd, "jacobian"))

    def test_isinstance_protocols(self):
        """Zoo transient model satisfies expected protocols."""
        fwd = self._create_model()
        self.assertTrue(isinstance(fwd, FunctionProtocol))
        self.assertTrue(isinstance(fwd, FunctionWithJacobianProtocol))

    def test_jacobian_derivative_checker(self):
        """Zoo transient model Jacobian passes DerivativeChecker."""
        bkd = self._bkd
        fwd = self._create_model()
        wrapper = FunctionWithJacobianFromCallable(
            nqoi=fwd.nqoi(),
            nvars=fwd.nvars(),
            fun=fwd,
            jacobian=fwd.jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = bkd.array([0.1, -0.1])[:, None]
        errors = checker.check_derivatives(
            sample, direction=None, relative=True
        )[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        self.assertLessEqual(ratio, 1e-5)


class TestDiffusionPositivityValidation(Generic[Array], unittest.TestCase):
    """Tests for strict positivity validation in DiffusionParameterization."""
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _make_param_and_physics(self, npts):
        bkd = self._bkd
        from pyapprox.typing.pde.collocation.basis import ChebyshevBasis1D
        from pyapprox.typing.pde.collocation.mesh import TransformedMesh1D
        from pyapprox.typing.pde.collocation.physics.advection_diffusion import (
            AdvectionDiffusionReaction,
        )
        from pyapprox.typing.pde.field_maps.basis_expansion import (
            BasisExpansion,
        )
        from pyapprox.typing.pde.parameterizations.diffusion import (
            create_diffusion_parameterization,
        )
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
        # Single basis function = ones, so field = base + p[0]*ones
        fm = BasisExpansion(bkd, 0.0, [bkd.ones((npts,))])
        param = create_diffusion_parameterization(bkd, basis, fm)
        return param, physics

    def test_nonpositive_diffusion_raises(self):
        """ValueError raised when parameterized diffusion is non-positive."""
        bkd = self._bkd
        npts = 5
        param, physics = self._make_param_and_physics(npts)
        # field = 0.0 + (-0.1)*ones = -0.1 everywhere
        with self.assertRaises(ValueError) as ctx:
            param.apply(physics, bkd.array([-0.1]))
        self.assertIn("positive", str(ctx.exception))

    def test_zero_diffusion_raises(self):
        """ValueError raised when parameterized diffusion is zero."""
        bkd = self._bkd
        npts = 5
        param, physics = self._make_param_and_physics(npts)
        with self.assertRaises(ValueError):
            param.apply(physics, bkd.array([0.0]))

    def test_positive_diffusion_succeeds(self):
        """No error when parameterized diffusion is positive."""
        bkd = self._bkd
        npts = 5
        param, physics = self._make_param_and_physics(npts)
        # field = 0.0 + 1.5*ones = 1.5 everywhere
        param.apply(physics, bkd.array([1.5]))  # Should not raise


class TestSteadyDiffusionZooNumpy(TestSteadyDiffusionZoo[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSteadyDiffusionZooTorch(TestSteadyDiffusionZoo[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestTransientDiffusionZooNumpy(TestTransientDiffusionZoo[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTransientDiffusionZooTorch(TestTransientDiffusionZoo[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestDiffusionPositivityNumpy(TestDiffusionPositivityValidation[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDiffusionPositivityTorch(TestDiffusionPositivityValidation[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()
