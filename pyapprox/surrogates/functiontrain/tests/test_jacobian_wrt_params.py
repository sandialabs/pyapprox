"""Tests for jacobian_wrt_params methods in FunctionTrain components.

Tests validate Jacobian computations using DerivativeChecker per CLAUDE.md convention.
Tests cover:
- ConstantExpansion (no trainable params)
- FunctionTrainCore (per-core Jacobian)
- FunctionTrain (full Jacobian via forward-backward sweep)
"""

import numpy as np
import pytest
import torch

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.functiontrain.additive import (
    ConstantExpansion,
    create_additive_functiontrain,
)
from pyapprox.surrogates.functiontrain.core import FunctionTrainCore
from pyapprox.surrogates.functiontrain.functiontrain import FunctionTrain
from pyapprox.util.backends.torch import TorchBkd


class TestConstantExpansionJacobian:
    """Tests for ConstantExpansion.jacobian_wrt_params."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_jacobian_shape_nqoi_1(self, bkd) -> None:
        """Test jacobian shape for nqoi=1."""
        const = ConstantExpansion(1.0, bkd, nqoi=1)

        nsamples = 5
        samples = bkd.asarray(np.random.randn(1, nsamples))
        jac = const.jacobian_wrt_params(samples)

        # Shape: (nsamples, nqoi, nparams=0)
        assert jac.shape == (nsamples, 1, 0)

    def test_jacobian_shape_nqoi_3(self, bkd) -> None:
        """Test jacobian shape for nqoi=3."""
        const = ConstantExpansion(2.5, bkd, nqoi=3)

        nsamples = 7
        samples = bkd.asarray(np.random.randn(1, nsamples))
        jac = const.jacobian_wrt_params(samples)

        # Shape: (nsamples, nqoi, nparams=0)
        assert jac.shape == (nsamples, 3, 0)


class TestFunctionTrainCoreJacobian:
    """Tests for FunctionTrainCore.jacobian_wrt_params."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_univariate_expansion(self, bkd, max_level, nqoi=1):
        """Create a univariate polynomial expansion."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_jacobian_shape_single_expansion(self, bkd) -> None:
        """Test jacobian shape for core with single expansion."""
        max_level = 2
        exp = self._create_univariate_expansion(bkd, max_level, nqoi=1)
        core = FunctionTrainCore([[exp]], bkd)

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        jac = core.jacobian_wrt_params(samples)

        # Shape: (r_left, r_right, nsamples, nqoi, nparams)
        r_left, r_right = core.ranks()
        nparams = core.nparams()
        assert jac.shape == (r_left, r_right, nsamples, 1, nparams)

    def test_jacobian_shape_2x2_with_constants(self, bkd) -> None:
        """Test jacobian shape for 2x2 core with some constants."""
        max_level = 1
        exp = self._create_univariate_expansion(bkd, max_level, nqoi=1)
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
        assert r_left == 2
        assert r_right == 2
        assert nparams == exp.nparams()
        assert jac.shape == (2, 2, nsamples, 1, nparams)

    def test_jacobian_zero_params_core(self, bkd) -> None:
        """Test jacobian for core with no trainable params."""
        const_0 = ConstantExpansion(0.0, bkd, nqoi=1)
        const_1 = ConstantExpansion(1.0, bkd, nqoi=1)

        # Core with only constants - no trainable params
        core = FunctionTrainCore([[const_1, const_0]], bkd)
        assert core.nparams() == 0

        nsamples = 5
        samples = bkd.asarray(np.random.randn(1, nsamples))
        jac = core.jacobian_wrt_params(samples)

        # Shape: (1, 2, nsamples, 1, 0)
        assert jac.shape == (1, 2, nsamples, 1, 0)

    def test_jacobian_derivative_checker(self, bkd) -> None:
        """Test core jacobian using DerivativeChecker."""
        max_level = 2
        nqoi = 1
        exp = self._create_univariate_expansion(bkd, max_level, nqoi)
        # Set non-zero coefficients
        nterms = exp.nterms()
        exp.set_coefficients(bkd.asarray(np.random.randn(nterms, nqoi)))

        core = FunctionTrainCore([[exp]], bkd)
        nparams = core.nparams()

        # Fixed samples for evaluation
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 1)))

        # Create wrapper: params -> core output
        def fun(params):
            # params shape: (nparams, 1)
            new_core = core.with_params(params[:, 0])
            # core output: (r_left, r_right, nsamples, nqoi) = (1, 1, 1, 1)
            return new_core(samples)[0, 0].T  # (nsamples=1, nqoi) -> (nqoi, nsamples=1)

        def jacobian_func(params):
            # params shape: (nparams, 1)
            new_core = core.with_params(params[:, 0])
            # core jacobian: (r_left, r_right, nsamples, nqoi, nparams)
            jac = new_core.jacobian_wrt_params(samples)
            # Extract (nsamples, nqoi, nparams) then return (nqoi, nparams) for single
            # sample
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
        assert float(error_ratio) < 1e-6


class TestFunctionTrainJacobian:
    """Tests for FunctionTrain.jacobian_wrt_params."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_univariate_expansion(self, bkd, max_level, nqoi=1):
        """Create a univariate polynomial expansion."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def _create_additive_ft(self, bkd, nvars=3, max_level=2, nqoi=1):
        """Create an additive FunctionTrain for testing."""
        univariate_bases = [
            self._create_univariate_expansion(bkd, max_level, nqoi) for _ in range(nvars)
        ]
        return create_additive_functiontrain(univariate_bases, bkd, nqoi)

    def test_jacobian_shape(self, bkd) -> None:
        """Test jacobian output shape."""
        nvars = 3
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=2, nqoi=1)

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        jac = ft.jacobian_wrt_params(samples)

        # Shape: (nsamples, nqoi, nparams)
        assert jac.shape == (nsamples, 1, ft.nparams())

    def test_jacobian_shape_multi_qoi(self, bkd) -> None:
        """Test jacobian shape with multiple QoIs."""
        nvars = 3
        nqoi = 2
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=2, nqoi=nqoi)

        nsamples = 4
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        jac = ft.jacobian_wrt_params(samples)

        # Shape: (nsamples, nqoi, nparams)
        assert jac.shape == (nsamples, nqoi, ft.nparams())

    def test_jacobian_single_var_ft(self, bkd) -> None:
        """Test jacobian for single variable FT (nvars=1 edge case)."""
        max_level = 2
        nqoi = 1
        exp = self._create_univariate_expansion(bkd, max_level, nqoi)

        # Create single-core FT (nvars=1)
        core = FunctionTrainCore([[exp]], bkd)
        ft = FunctionTrain([core], bkd, nqoi)

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        jac = ft.jacobian_wrt_params(samples)

        # Shape: (nsamples, nqoi, nparams)
        assert jac.shape == (nsamples, nqoi, ft.nparams())

    def test_jacobian_zero_params_ft(self, bkd) -> None:
        """Test jacobian for FT with no trainable params (edge case)."""
        const_1 = ConstantExpansion(1.0, bkd, nqoi=1)

        # Single core with only constant (nparams=0)
        core = FunctionTrainCore([[const_1]], bkd)
        ft = FunctionTrain([core], bkd, nqoi=1)
        assert ft.nparams() == 0

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        jac = ft.jacobian_wrt_params(samples)

        # Shape: (nsamples, nqoi, 0)
        assert jac.shape == (nsamples, 1, 0)

    def test_jacobian_derivative_checker(self, bkd) -> None:
        """Test FunctionTrain jacobian using DerivativeChecker."""
        nvars = 3
        max_level = 1
        nqoi = 1
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=max_level, nqoi=nqoi)

        # Set random parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        # Fixed samples for evaluation
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, 1)))

        # Create wrapper: params -> FT output
        def fun(params):
            # params shape: (nparams, 1)
            new_ft = ft.with_params(params[:, 0])
            return new_ft(samples)  # (nqoi, nsamples=1)

        def jacobian_func(params):
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
        assert float(error_ratio) < 1e-6

    def test_jacobian_derivative_checker_multi_qoi(self, bkd) -> None:
        """Test jacobian with multiple QoIs using DerivativeChecker."""
        nvars = 2
        max_level = 1
        nqoi = 2
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=max_level, nqoi=nqoi)

        # Set random parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        # Fixed samples for evaluation
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, 1)))

        # Create wrapper: params -> FT output
        def fun(params):
            new_ft = ft.with_params(params[:, 0])
            return new_ft(samples)  # (nqoi, nsamples=1)

        def jacobian_func(params):
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
        assert float(error_ratio) < 1e-6


# Torch-only test: verify autograd compatibility
class TestFunctionTrainJacobianAutograd:
    """Test that FunctionTrain jacobian matches torch.autograd."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(42)

    def _create_univariate_expansion(self, max_level, nqoi=1):
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def _create_additive_ft(self, nvars=3, max_level=2, nqoi=1):
        bkd = self._bkd
        univariate_bases = [
            self._create_univariate_expansion(max_level, nqoi) for _ in range(nvars)
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
