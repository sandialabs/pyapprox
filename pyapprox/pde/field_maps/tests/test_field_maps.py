"""Tests for field map implementations."""

import math

import pytest
import numpy as np

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.pde.field_maps.basis_expansion import (
    BasisExpansion,
)
from pyapprox.pde.field_maps.kle_factory import (
    create_lognormal_kle_field_map,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import (
    MeshKLEFieldMap,
)
from pyapprox.pde.field_maps.protocol import (
    FieldMapProtocol,
)
from pyapprox.pde.field_maps.scalar import (
    ScalarAmplitude,
)
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
)
class TestFieldMaps:
    def test_basis_expansion_isinstance(self, bkd) -> None:
        """BasisExpansion satisfies FieldMapProtocol."""
        npts = 5
        phi0 = bkd.ones((npts,))
        phi1 = bkd.array([0.0, 0.25, 0.5, 0.75, 1.0])
        fm = BasisExpansion(bkd, 1.0, [phi0, phi1])
        assert isinstance(fm, FieldMapProtocol)

    def test_basis_expansion_call(self, bkd) -> None:
        """BasisExpansion returns correct field."""
        npts = 5
        phi0 = bkd.ones((npts,))
        phi1 = bkd.array([0.0, 0.25, 0.5, 0.75, 1.0])
        fm = BasisExpansion(bkd, 2.0, [phi0, phi1])
        params = bkd.array([0.5, -1.0])
        result = fm(params)
        # expected: 2.0 + 0.5*1 + (-1.0)*[0, 0.25, 0.5, 0.75, 1.0]
        expected = bkd.array([2.5, 2.25, 2.0, 1.75, 1.5])
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_basis_expansion_nvars(self, bkd) -> None:
        """BasisExpansion.nvars matches number of basis functions."""
        npts = 5
        phi0 = bkd.ones((npts,))
        phi1 = bkd.array([0.0, 0.25, 0.5, 0.75, 1.0])
        phi2 = bkd.array([0.0, 0.0625, 0.25, 0.5625, 1.0])
        fm = BasisExpansion(bkd, 1.0, [phi0, phi1, phi2])
        assert fm.nvars() == 3

    def test_basis_expansion_jacobian_fd(self, bkd) -> None:
        """BasisExpansion.jacobian matches FD via DerivativeChecker."""
        npts = 10
        phi0 = bkd.ones((npts,))
        phi1 = bkd.linspace(0.0, 1.0, npts)
        fm = BasisExpansion(bkd, 1.0, [phi0, phi1])

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=npts,
            nvars=fm.nvars(),
            fun=lambda samples: bkd.stack(
                [fm(samples[:, i]) for i in range(samples.shape[1])], axis=1
            ),
            jacobian=lambda sample: fm.jacobian(sample[:, 0]),
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        params = bkd.array([0.3, -0.5])[:, None]
        errors = checker.check_derivatives(params)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        assert ratio <= 1e-5

    def test_basis_expansion_jacobian_cached(self, bkd) -> None:
        """BasisExpansion.jacobian returns same object each call."""
        npts = 5
        phi0 = bkd.ones((npts,))
        fm = BasisExpansion(bkd, 1.0, [phi0])
        params = bkd.array([0.5])
        j1 = fm.jacobian(params)
        j2 = fm.jacobian(params)
        assert j1 is j2

    def test_scalar_amplitude_call(self, bkd) -> None:
        """ScalarAmplitude returns p[0] * base_field."""
        base = bkd.array([1.0, 2.0, 3.0])
        fm = ScalarAmplitude(bkd, base)
        params = bkd.array([2.5])
        result = fm(params)
        expected = bkd.array([2.5, 5.0, 7.5])
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_scalar_amplitude_nvars(self, bkd) -> None:
        """ScalarAmplitude.nvars is always 1."""
        base = bkd.array([1.0, 2.0])
        fm = ScalarAmplitude(bkd, base)
        assert fm.nvars() == 1

    def test_scalar_amplitude_jacobian_cached(self, bkd) -> None:
        """ScalarAmplitude.jacobian returns same object each call."""
        base = bkd.array([1.0, 2.0, 3.0])
        fm = ScalarAmplitude(bkd, base)
        params = bkd.array([1.0])
        j1 = fm.jacobian(params)
        j2 = fm.jacobian(params)
        assert j1 is j2

    def test_scalar_amplitude_jacobian_fd(self, bkd) -> None:
        """ScalarAmplitude.jacobian matches FD."""
        npts = 5
        base = bkd.linspace(1.0, 3.0, npts)
        fm = ScalarAmplitude(bkd, base)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=npts,
            nvars=1,
            fun=lambda samples: bkd.stack(
                [fm(samples[:, i]) for i in range(samples.shape[1])], axis=1
            ),
            jacobian=lambda sample: fm.jacobian(sample[:, 0]),
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        params = bkd.array([1.5])[:, None]
        errors = checker.check_derivatives(params)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        assert ratio <= 1e-5

    # --- MeshKLEFieldMap tests ---

    def test_mesh_kle_field_map_isinstance(self, bkd) -> None:
        """MeshKLEFieldMap satisfies FieldMapProtocol."""
        _npts, _nterms = 4, 2
        mean = bkd.array([1.0, 2.0, 3.0, 4.0])
        W = bkd.array([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        fm = MeshKLEFieldMap(bkd, mean, W)
        assert isinstance(fm, FieldMapProtocol)

    def test_mesh_kle_field_map_call(self, bkd) -> None:
        """MeshKLEFieldMap returns mean + W @ params."""
        mean = bkd.array([1.0, 2.0, 3.0, 4.0])
        W = bkd.array([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        fm = MeshKLEFieldMap(bkd, mean, W)
        params = bkd.array([1.0, -1.0])
        result = fm(params)
        # expected: [1,2,3,4] + [2*1+0*(-1), 0*1+1*(-1), 0, 0] = [3, 1, 3, 4]
        expected = bkd.array([3.0, 1.0, 3.0, 4.0])
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_mesh_kle_field_map_nvars(self, bkd) -> None:
        """MeshKLEFieldMap.nvars matches number of columns in W."""
        W = bkd.ones((5, 3))
        mean = bkd.zeros((5,))
        fm = MeshKLEFieldMap(bkd, mean, W)
        assert fm.nvars() == 3

    def test_mesh_kle_field_map_jacobian_is_W(self, bkd) -> None:
        """MeshKLEFieldMap.jacobian returns W regardless of params."""
        W = bkd.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mean = bkd.zeros((3,))
        fm = MeshKLEFieldMap(bkd, mean, W)
        params = bkd.array([0.5, -0.3])
        jac = fm.jacobian(params)
        bkd.assert_allclose(jac, W, rtol=1e-12)

    def test_mesh_kle_field_map_jacobian_fd(self, bkd) -> None:
        """MeshKLEFieldMap.jacobian matches FD via DerivativeChecker."""
        npts, _nterms = 5, 2
        W = bkd.array(
            [
                [1.0, 0.5],
                [0.3, -0.2],
                [0.0, 1.0],
                [-0.5, 0.1],
                [0.8, 0.4],
            ]
        )
        mean = bkd.linspace(0.0, 1.0, npts)
        fm = MeshKLEFieldMap(bkd, mean, W)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=npts,
            nvars=fm.nvars(),
            fun=lambda samples: bkd.stack(
                [fm(samples[:, i]) for i in range(samples.shape[1])], axis=1
            ),
            jacobian=lambda sample: fm.jacobian(sample[:, 0]),
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        params = bkd.array([0.5, -0.3])[:, None]
        errors = checker.check_derivatives(params)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        assert ratio <= 1e-5

    def test_mesh_kle_field_map_zero_params_gives_mean(self, bkd) -> None:
        """MeshKLEFieldMap with zero params returns mean field."""
        mean = bkd.array([1.0, 2.0, 3.0])
        W = bkd.ones((3, 2))
        fm = MeshKLEFieldMap(bkd, mean, W)
        params = bkd.zeros((2,))
        bkd.assert_allclose(fm(params), mean, rtol=1e-12)

    # --- TransformedFieldMap tests ---

    def test_transformed_call(self, bkd) -> None:
        """TransformedFieldMap applies transform correctly."""
        npts = 4
        phi0 = bkd.ones((npts,))
        inner = BasisExpansion(bkd, 0.0, [phi0])
        tfm = TransformedFieldMap(
            inner,
            transform=lambda x: bkd.exp(x),
            transform_deriv=lambda x: bkd.exp(x),
            bkd=bkd,
        )
        params = bkd.array([1.0])
        result = tfm(params)
        # inner(params) = 0.0 + 1.0*ones = ones -> exp(1) = e
        expected = bkd.full((npts,), math.e)
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_transformed_jacobian_fd(self, bkd) -> None:
        """TransformedFieldMap.jacobian matches FD."""
        npts = 8
        phi0 = bkd.ones((npts,))
        phi1 = bkd.linspace(0.0, 1.0, npts)
        inner = BasisExpansion(bkd, 0.5, [phi0, phi1])
        tfm = TransformedFieldMap(
            inner,
            transform=lambda x: bkd.exp(x),
            transform_deriv=lambda x: bkd.exp(x),
            bkd=bkd,
        )

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=npts,
            nvars=tfm.nvars(),
            fun=lambda samples: bkd.stack(
                [tfm(samples[:, i]) for i in range(samples.shape[1])], axis=1
            ),
            jacobian=lambda sample: tfm.jacobian(sample[:, 0]),
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        params = bkd.array([0.2, -0.3])[:, None]
        errors = checker.check_derivatives(params)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        assert ratio <= 1e-5

    def test_transformed_no_jacobian_without_inner_jacobian(self, bkd) -> None:
        """TransformedFieldMap has no jacobian if inner lacks it."""

        class EvalOnlyFieldMap:
            def nvars(self) -> int:
                return 1

            def __call__(self, params_1d):
                return params_1d

        inner = EvalOnlyFieldMap()
        tfm = TransformedFieldMap(
            inner,
            transform=lambda x: x,
            transform_deriv=lambda x: bkd.ones(x.shape),
            bkd=bkd,
        )
        assert not hasattr(tfm, "jacobian")

    def test_transformed_init_type_error(self, bkd) -> None:
        """TransformedFieldMap raises TypeError for non-FieldMap inner."""
        with pytest.raises(TypeError):
            TransformedFieldMap(
                "not_a_field_map",
                transform=lambda x: x,
                transform_deriv=lambda x: x,
                bkd=bkd,
            )

    def test_transformed_isinstance(self, bkd) -> None:
        """TransformedFieldMap satisfies FieldMapProtocol."""
        npts = 3
        inner = BasisExpansion(bkd, 0.0, [bkd.ones((npts,))])
        tfm = TransformedFieldMap(
            inner,
            transform=lambda x: x,
            transform_deriv=lambda x: bkd.ones(x.shape),
            bkd=bkd,
        )
        assert isinstance(tfm, FieldMapProtocol)

    # --- Lognormal KLE factory tests ---

    def test_lognormal_kle_factory_shapes(self, bkd) -> None:
        """create_lognormal_kle_field_map produces correct shapes."""
        npts = 10
        num_kle_terms = 3
        mesh_coords = bkd.linspace(0.0, 1.0, npts)[None, :]  # (1, npts)
        mean_log = bkd.zeros((npts,))
        tfm = create_lognormal_kle_field_map(
            mesh_coords,
            mean_log,
            bkd,
            num_kle_terms=num_kle_terms,
            sigma=0.3,
        )
        assert tfm.nvars() == num_kle_terms
        assert isinstance(tfm, FieldMapProtocol)
        params = bkd.zeros((num_kle_terms,))
        result = tfm(params)
        assert result.shape[0] == npts

    def test_lognormal_kle_zero_params_gives_exp_mean(self, bkd) -> None:
        """Lognormal KLE with theta=0 gives exp(mean_log_field)."""
        npts = 8
        mesh_coords = bkd.linspace(0.0, 1.0, npts)[None, :]
        mean_log = bkd.array([0.5, 1.0, 0.2, -0.1, 0.3, 0.8, 0.0, -0.5])
        tfm = create_lognormal_kle_field_map(
            mesh_coords,
            mean_log,
            bkd,
            num_kle_terms=2,
            sigma=0.3,
        )
        params = bkd.zeros((2,))
        result = tfm(params)
        expected = bkd.exp(mean_log)
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_lognormal_kle_always_positive(self, bkd) -> None:
        """Lognormal KLE output is strictly positive for any params."""
        npts = 10
        mesh_coords = bkd.linspace(0.0, 1.0, npts)[None, :]
        mean_log = bkd.zeros((npts,))
        tfm = create_lognormal_kle_field_map(
            mesh_coords,
            mean_log,
            bkd,
            num_kle_terms=2,
            sigma=0.5,
        )
        np.random.seed(42)
        for _ in range(5):
            params = bkd.array(np.random.randn(2) * 3.0)
            result = tfm(params)
            min_val = float(bkd.min(result))
            assert min_val > 0.0

    def test_basis_expansion_autograd(self, torch_bkd) -> None:
        """Torch autograd Jacobian matches BasisExpansion.jacobian."""
        import torch

        bkd = torch_bkd
        npts = 6
        phi0 = bkd.array([1.0] * npts)
        phi1 = bkd.linspace(0.0, 1.0, npts)
        fm = BasisExpansion(bkd, 1.0, [phi0, phi1])

        params = torch.tensor([0.3, -0.5], dtype=torch.float64)

        def torch_fun(p):
            return fm(p)

        autograd_jac = torch.autograd.functional.jacobian(torch_fun, params)
        analytical_jac = fm.jacobian(params)
        bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-10)

    def test_transformed_autograd(self, torch_bkd) -> None:
        """Torch autograd Jacobian matches TransformedFieldMap.jacobian."""
        import torch

        bkd = torch_bkd
        npts = 5
        phi0 = bkd.array([1.0] * npts)
        phi1 = bkd.linspace(0.0, 1.0, npts)
        inner = BasisExpansion(bkd, 0.5, [phi0, phi1])
        tfm = TransformedFieldMap(
            inner,
            transform=lambda x: bkd.exp(x),
            transform_deriv=lambda x: bkd.exp(x),
            bkd=bkd,
        )

        params = torch.tensor([0.2, -0.3], dtype=torch.float64)

        def torch_fun(p):
            return tfm(p)

        autograd_jac = torch.autograd.functional.jacobian(torch_fun, params)
        analytical_jac = tfm.jacobian(params)
        bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-10)

    def test_mesh_kle_field_map_autograd(self, torch_bkd) -> None:
        """Torch autograd Jacobian matches MeshKLEFieldMap.jacobian."""
        import torch

        bkd = torch_bkd
        W = bkd.array([[1.0, 0.5], [0.3, -0.2], [0.0, 1.0]])
        mean = bkd.array([1.0, 2.0, 3.0])
        fm = MeshKLEFieldMap(bkd, mean, W)

        params = torch.tensor([0.5, -0.3], dtype=torch.float64)

        def torch_fun(p):
            return fm(p)

        autograd_jac = torch.autograd.functional.jacobian(torch_fun, params)
        analytical_jac = fm.jacobian(params)
        bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-10)

    def test_lognormal_kle_autograd(self, torch_bkd) -> None:
        """Torch autograd Jacobian matches lognormal KLE Jacobian."""
        import torch

        bkd = torch_bkd
        npts = 8
        num_kle_terms = 2
        mesh_coords = bkd.linspace(0.0, 1.0, npts)[None, :]
        mean_log = bkd.full((npts,), 0.5)
        tfm = create_lognormal_kle_field_map(
            mesh_coords,
            mean_log,
            bkd,
            num_kle_terms=num_kle_terms,
            sigma=0.3,
        )

        params = torch.tensor([0.3, -0.2], dtype=torch.float64)

        def torch_fun(p):
            return tfm(p)

        autograd_jac = torch.autograd.functional.jacobian(torch_fun, params)
        analytical_jac = tfm.jacobian(params)
        bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-10)

    def test_lognormal_kle_jacobian_fd(self, bkd) -> None:
        """Lognormal KLE Jacobian matches FD via DerivativeChecker."""
        npts = 10
        num_kle_terms = 2
        mesh_coords = bkd.linspace(0.0, 1.0, npts)[None, :]
        mean_log = bkd.full((npts,), 0.5)
        tfm = create_lognormal_kle_field_map(
            mesh_coords,
            mean_log,
            bkd,
            num_kle_terms=num_kle_terms,
            sigma=0.3,
        )

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=npts,
            nvars=num_kle_terms,
            fun=lambda samples: bkd.stack(
                [tfm(samples[:, i]) for i in range(samples.shape[1])], axis=1
            ),
            jacobian=lambda sample: tfm.jacobian(sample[:, 0]),
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        params = bkd.array([0.3, -0.2])[:, None]
        errors = checker.check_derivatives(params)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        assert ratio <= 1e-5
