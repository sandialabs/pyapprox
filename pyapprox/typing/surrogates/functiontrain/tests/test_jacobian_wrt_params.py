"""Tests for jacobian_wrt_params methods in FunctionTrain components.

Tests validate Jacobian computations using DerivativeChecker per CLAUDE.md convention.
Tests cover:
- ConstantExpansion (no trainable params)
- FunctionTrainCore (per-core Jacobian)
- FunctionTrain (full Jacobian via forward-backward sweep)
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.surrogates.affine.univariate import create_bases_1d
from pyapprox.typing.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.typing.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.typing.surrogates.affine.expansions import BasisExpansion
from pyapprox.typing.probability import UniformMarginal

from pyapprox.typing.surrogates.functiontrain.core import FunctionTrainCore
from pyapprox.typing.surrogates.functiontrain.functiontrain import FunctionTrain
from pyapprox.typing.surrogates.functiontrain.additive import (
    create_additive_functiontrain,
    ConstantExpansion,
)
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.typing.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)


class TestConstantExpansionJacobian(Generic[Array], unittest.TestCase):
    """Tests for ConstantExpansion.jacobian_wrt_params."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def test_jacobian_shape_nqoi_1(self) -> None:
        """Test jacobian shape for nqoi=1."""
        bkd = self._bkd
        const = ConstantExpansion(1.0, bkd, nqoi=1)

        nsamples = 5
        samples = bkd.asarray(np.random.randn(1, nsamples))
        jac = const.jacobian_wrt_params(samples)

        # Shape: (nsamples, nqoi, nparams=0)
        self.assertEqual(jac.shape, (nsamples, 1, 0))

    def test_jacobian_shape_nqoi_3(self) -> None:
        """Test jacobian shape for nqoi=3."""
        bkd = self._bkd
        const = ConstantExpansion(2.5, bkd, nqoi=3)

        nsamples = 7
        samples = bkd.asarray(np.random.randn(1, nsamples))
        jac = const.jacobian_wrt_params(samples)

        # Shape: (nsamples, nqoi, nparams=0)
        self.assertEqual(jac.shape, (nsamples, 3, 0))


class TestFunctionTrainCoreJacobian(Generic[Array], unittest.TestCase):
    """Tests for FunctionTrainCore.jacobian_wrt_params."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_univariate_expansion(
        self, max_level: int, nqoi: int = 1
    ) -> BasisExpansion[Array]:
        """Create a univariate polynomial expansion."""
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_jacobian_shape_single_expansion(self) -> None:
        """Test jacobian shape for core with single expansion."""
        bkd = self._bkd
        max_level = 2
        exp = self._create_univariate_expansion(max_level, nqoi=1)
        core = FunctionTrainCore([[exp]], bkd)

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        jac = core.jacobian_wrt_params(samples)

        # Shape: (r_left, r_right, nsamples, nqoi, nparams)
        r_left, r_right = core.ranks()
        nparams = core.nparams()
        self.assertEqual(jac.shape, (r_left, r_right, nsamples, 1, nparams))

    def test_jacobian_shape_2x2_with_constants(self) -> None:
        """Test jacobian shape for 2x2 core with some constants."""
        bkd = self._bkd
        max_level = 1
        exp = self._create_univariate_expansion(max_level, nqoi=1)
        const_0 = ConstantExpansion(0.0, bkd, nqoi=1)
        const_1 = ConstantExpansion(1.0, bkd, nqoi=1)

        # Middle core structure: [[1, 0], [f, 1]]
        core = FunctionTrainCore(
            [[const_1, const_0], [exp, const_1]],
            bkd,
        )

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        jac = core.jacobian_wrt_params(samples)

        # Only the expansion contributes params (constants have nparams=0)
        r_left, r_right = core.ranks()
        nparams = core.nparams()
        self.assertEqual(r_left, 2)
        self.assertEqual(r_right, 2)
        self.assertEqual(nparams, exp.nparams())
        self.assertEqual(jac.shape, (2, 2, nsamples, 1, nparams))

    def test_jacobian_zero_params_core(self) -> None:
        """Test jacobian for core with no trainable params."""
        bkd = self._bkd
        const_0 = ConstantExpansion(0.0, bkd, nqoi=1)
        const_1 = ConstantExpansion(1.0, bkd, nqoi=1)

        # Core with only constants - no trainable params
        core = FunctionTrainCore([[const_1, const_0]], bkd)
        self.assertEqual(core.nparams(), 0)

        nsamples = 5
        samples = bkd.asarray(np.random.randn(1, nsamples))
        jac = core.jacobian_wrt_params(samples)

        # Shape: (1, 2, nsamples, 1, 0)
        self.assertEqual(jac.shape, (1, 2, nsamples, 1, 0))

    def test_jacobian_derivative_checker(self) -> None:
        """Test core jacobian using DerivativeChecker."""
        bkd = self._bkd
        max_level = 2
        nqoi = 1
        exp = self._create_univariate_expansion(max_level, nqoi)
        # Set non-zero coefficients
        nterms = exp.nterms()
        exp.set_coefficients(bkd.asarray(np.random.randn(nterms, nqoi)))

        core = FunctionTrainCore([[exp]], bkd)
        nparams = core.nparams()

        # Fixed samples for evaluation
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 1)))

        # Create wrapper: params -> core output
        def fun(params: Array) -> Array:
            # params shape: (nparams, 1)
            new_core = core.with_params(params[:, 0])
            # core output: (r_left, r_right, nsamples, nqoi) = (1, 1, 1, 1)
            return new_core(samples)[0, 0].T  # (nsamples=1, nqoi) -> (nqoi, nsamples=1)

        def jacobian_func(params: Array) -> Array:
            # params shape: (nparams, 1)
            new_core = core.with_params(params[:, 0])
            # core jacobian: (r_left, r_right, nsamples, nqoi, nparams)
            jac = new_core.jacobian_wrt_params(samples)
            # Extract (nsamples, nqoi, nparams) then return (nqoi, nparams) for single sample
            return jac[0, 0, 0, :, :]  # (nqoi, nparams)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=nqoi,
            nvars=nparams,
            fun=fun,
            jacobian=jacobian_func,
            bkd=bkd,
        )

        checker = DerivativeChecker(function_obj)
        params = core._flatten_params()
        sample_params = bkd.reshape(params, (nparams, 1))
        errors = checker.check_derivatives(sample_params, verbosity=0)

        error_ratio = checker.error_ratio(errors[0])
        self.assertLess(float(error_ratio), 1e-6)


class TestFunctionTrainJacobian(Generic[Array], unittest.TestCase):
    """Tests for FunctionTrain.jacobian_wrt_params."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_univariate_expansion(
        self, max_level: int, nqoi: int = 1
    ) -> BasisExpansion[Array]:
        """Create a univariate polynomial expansion."""
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def _create_additive_ft(
        self, nvars: int = 3, max_level: int = 2, nqoi: int = 1
    ) -> FunctionTrain[Array]:
        """Create an additive FunctionTrain for testing."""
        bkd = self._bkd
        univariate_bases = [
            self._create_univariate_expansion(max_level, nqoi)
            for _ in range(nvars)
        ]
        return create_additive_functiontrain(univariate_bases, bkd, nqoi)

    def test_jacobian_shape(self) -> None:
        """Test jacobian output shape."""
        bkd = self._bkd
        nvars = 3
        ft = self._create_additive_ft(nvars=nvars, max_level=2, nqoi=1)

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        jac = ft.jacobian_wrt_params(samples)

        # Shape: (nsamples, nqoi, nparams)
        self.assertEqual(jac.shape, (nsamples, 1, ft.nparams()))

    def test_jacobian_shape_multi_qoi(self) -> None:
        """Test jacobian shape with multiple QoIs."""
        bkd = self._bkd
        nvars = 3
        nqoi = 2
        ft = self._create_additive_ft(nvars=nvars, max_level=2, nqoi=nqoi)

        nsamples = 4
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        jac = ft.jacobian_wrt_params(samples)

        # Shape: (nsamples, nqoi, nparams)
        self.assertEqual(jac.shape, (nsamples, nqoi, ft.nparams()))

    def test_jacobian_single_var_ft(self) -> None:
        """Test jacobian for single variable FT (nvars=1 edge case)."""
        bkd = self._bkd
        max_level = 2
        nqoi = 1
        exp = self._create_univariate_expansion(max_level, nqoi)

        # Create single-core FT (nvars=1)
        core = FunctionTrainCore([[exp]], bkd)
        ft = FunctionTrain([core], bkd, nqoi)

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        jac = ft.jacobian_wrt_params(samples)

        # Shape: (nsamples, nqoi, nparams)
        self.assertEqual(jac.shape, (nsamples, nqoi, ft.nparams()))

    def test_jacobian_zero_params_ft(self) -> None:
        """Test jacobian for FT with no trainable params (edge case)."""
        bkd = self._bkd
        const_1 = ConstantExpansion(1.0, bkd, nqoi=1)

        # Single core with only constant (nparams=0)
        core = FunctionTrainCore([[const_1]], bkd)
        ft = FunctionTrain([core], bkd, nqoi=1)
        self.assertEqual(ft.nparams(), 0)

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        jac = ft.jacobian_wrt_params(samples)

        # Shape: (nsamples, nqoi, 0)
        self.assertEqual(jac.shape, (nsamples, 1, 0))

    def test_jacobian_derivative_checker(self) -> None:
        """Test FunctionTrain jacobian using DerivativeChecker."""
        bkd = self._bkd
        nvars = 3
        max_level = 1
        nqoi = 1
        ft = self._create_additive_ft(nvars=nvars, max_level=max_level, nqoi=nqoi)

        # Set random parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        # Fixed samples for evaluation
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, 1)))

        # Create wrapper: params -> FT output
        def fun(params: Array) -> Array:
            # params shape: (nparams, 1)
            new_ft = ft.with_params(params[:, 0])
            return new_ft(samples)  # (nqoi, nsamples=1)

        def jacobian_func(params: Array) -> Array:
            # params shape: (nparams, 1)
            new_ft = ft.with_params(params[:, 0])
            # FT jacobian: (nsamples, nqoi, nparams)
            jac = new_ft.jacobian_wrt_params(samples)
            # Return (nqoi, nparams) for single sample
            return jac[0, :, :]  # (nqoi, nparams)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=nqoi,
            nvars=nparams,
            fun=fun,
            jacobian=jacobian_func,
            bkd=bkd,
        )

        checker = DerivativeChecker(function_obj)
        params = ft._flatten_params()
        sample_params = bkd.reshape(params, (nparams, 1))
        errors = checker.check_derivatives(sample_params, verbosity=0)

        error_ratio = checker.error_ratio(errors[0])
        self.assertLess(float(error_ratio), 1e-6)

    def test_jacobian_derivative_checker_multi_qoi(self) -> None:
        """Test jacobian with multiple QoIs using DerivativeChecker."""
        bkd = self._bkd
        nvars = 2
        max_level = 1
        nqoi = 2
        ft = self._create_additive_ft(nvars=nvars, max_level=max_level, nqoi=nqoi)

        # Set random parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        # Fixed samples for evaluation
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, 1)))

        # Create wrapper: params -> FT output
        def fun(params: Array) -> Array:
            new_ft = ft.with_params(params[:, 0])
            return new_ft(samples)  # (nqoi, nsamples=1)

        def jacobian_func(params: Array) -> Array:
            new_ft = ft.with_params(params[:, 0])
            jac = new_ft.jacobian_wrt_params(samples)
            return jac[0, :, :]  # (nqoi, nparams)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=nqoi,
            nvars=nparams,
            fun=fun,
            jacobian=jacobian_func,
            bkd=bkd,
        )

        checker = DerivativeChecker(function_obj)
        params = ft._flatten_params()
        sample_params = bkd.reshape(params, (nparams, 1))
        errors = checker.check_derivatives(sample_params, verbosity=0)

        error_ratio = checker.error_ratio(errors[0])
        self.assertLess(float(error_ratio), 1e-6)


# NumPy backend tests
class TestConstantExpansionJacobianNumpy(TestConstantExpansionJacobian[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFunctionTrainCoreJacobianNumpy(TestFunctionTrainCoreJacobian[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFunctionTrainJacobianNumpy(TestFunctionTrainJacobian[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestConstantExpansionJacobianTorch(TestConstantExpansionJacobian[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestFunctionTrainCoreJacobianTorch(TestFunctionTrainCoreJacobian[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestFunctionTrainJacobianTorch(TestFunctionTrainJacobian[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


# Torch-only test: verify autograd compatibility
class TestFunctionTrainJacobianAutograd(unittest.TestCase):
    """Test that FunctionTrain jacobian matches torch.autograd."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(42)

    def _create_univariate_expansion(
        self, max_level: int, nqoi: int = 1
    ) -> BasisExpansion[torch.Tensor]:
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def _create_additive_ft(
        self, nvars: int = 3, max_level: int = 2, nqoi: int = 1
    ) -> FunctionTrain[torch.Tensor]:
        bkd = self._bkd
        univariate_bases = [
            self._create_univariate_expansion(max_level, nqoi)
            for _ in range(nvars)
        ]
        return create_additive_functiontrain(univariate_bases, bkd, nqoi)

    def test_jacobian_matches_autograd(self) -> None:
        """Test that analytical jacobian matches torch autograd."""
        bkd = self._bkd
        nvars = 2
        max_level = 1
        nqoi = 1
        ft = self._create_additive_ft(nvars=nvars, max_level=max_level, nqoi=nqoi)

        # Set random parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        nsamples = 3
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        # Get analytical jacobian
        analytical_jac = ft.jacobian_wrt_params(samples)  # (nsamples, nqoi, nparams)

        # Get autograd jacobian
        def ft_output(p: torch.Tensor) -> torch.Tensor:
            """FT output given flat params."""
            ft_new = ft.with_params(p)
            return ft_new(samples).T  # (nsamples, nqoi)

        params_tensor = ft._flatten_params()
        autograd_jac = torch.autograd.functional.jacobian(ft_output, params_tensor)
        # autograd_jac shape: (nsamples, nqoi, nparams)

        bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-10)

    def test_jacobian_matches_autograd_multi_qoi(self) -> None:
        """Test analytical jacobian matches autograd with multiple QoIs."""
        bkd = self._bkd
        nvars = 2
        max_level = 1
        nqoi = 2
        ft = self._create_additive_ft(nvars=nvars, max_level=max_level, nqoi=nqoi)

        # Set random parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        nsamples = 3
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        # Get analytical jacobian
        analytical_jac = ft.jacobian_wrt_params(samples)  # (nsamples, nqoi, nparams)

        # Get autograd jacobian
        def ft_output(p: torch.Tensor) -> torch.Tensor:
            ft_new = ft.with_params(p)
            return ft_new(samples).T  # (nsamples, nqoi)

        params_tensor = ft._flatten_params()
        autograd_jac = torch.autograd.functional.jacobian(ft_output, params_tensor)

        bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
