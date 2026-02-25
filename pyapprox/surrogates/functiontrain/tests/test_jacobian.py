"""Tests for jacobian methods (w.r.t. inputs) in FunctionTrain components.

Tests validate Jacobian computations using DerivativeChecker per CLAUDE.md convention.
Tests cover:
- ConstantExpansion (zero jacobian for constants)
- FunctionTrainCore (per-core input Jacobian)
- FunctionTrain (full input Jacobian via forward-backward sweep)
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import load_tests  # noqa: F401

from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.probability import UniformMarginal

from pyapprox.surrogates.functiontrain.core import FunctionTrainCore
from pyapprox.surrogates.functiontrain.functiontrain import FunctionTrain
from pyapprox.surrogates.functiontrain.additive import (
    create_additive_functiontrain,
    ConstantExpansion,
)
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)


class TestConstantExpansionInputJacobian(Generic[Array], unittest.TestCase):
    """Tests for ConstantExpansion jacobian w.r.t. inputs."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def test_jacobian_batch_shape_nqoi_1(self) -> None:
        """Test jacobian_batch shape for nqoi=1."""
        bkd = self._bkd
        const = ConstantExpansion(1.0, bkd, nqoi=1)

        nsamples = 5
        samples = bkd.asarray(np.random.randn(1, nsamples))
        jac = const.jacobian_batch(samples)

        # Shape: (nsamples, nqoi, nvars=1)
        self.assertEqual(jac.shape, (nsamples, 1, 1))

    def test_jacobian_batch_shape_nqoi_3(self) -> None:
        """Test jacobian_batch shape for nqoi=3."""
        bkd = self._bkd
        const = ConstantExpansion(2.5, bkd, nqoi=3)

        nsamples = 7
        samples = bkd.asarray(np.random.randn(1, nsamples))
        jac = const.jacobian_batch(samples)

        # Shape: (nsamples, nqoi, nvars=1)
        self.assertEqual(jac.shape, (nsamples, 3, 1))

    def test_jacobian_batch_is_zero(self) -> None:
        """Test that jacobian_batch returns all zeros."""
        bkd = self._bkd
        const = ConstantExpansion(5.0, bkd, nqoi=2)

        nsamples = 4
        samples = bkd.asarray(np.random.randn(1, nsamples))
        jac = const.jacobian_batch(samples)

        expected = bkd.zeros((nsamples, 2, 1))
        bkd.assert_allclose(jac, expected)

    def test_jacobian_shape(self) -> None:
        """Test single-sample jacobian shape."""
        bkd = self._bkd
        const = ConstantExpansion(1.0, bkd, nqoi=2)

        sample = bkd.asarray(np.random.randn(1, 1))
        jac = const.jacobian(sample)

        # Shape: (nqoi, nvars=1)
        self.assertEqual(jac.shape, (2, 1))

    def test_jacobian_is_zero(self) -> None:
        """Test that single-sample jacobian returns zeros."""
        bkd = self._bkd
        const = ConstantExpansion(3.0, bkd, nqoi=1)

        sample = bkd.asarray(np.random.randn(1, 1))
        jac = const.jacobian(sample)

        expected = bkd.zeros((1, 1))
        bkd.assert_allclose(jac, expected)


class TestFunctionTrainCoreInputJacobian(Generic[Array], unittest.TestCase):
    """Tests for FunctionTrainCore.jacobian_wrt_input."""

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

    def test_supports_input_jacobian(self) -> None:
        """Test supports_input_jacobian method."""
        bkd = self._bkd
        max_level = 2
        exp = self._create_univariate_expansion(max_level, nqoi=1)
        core = FunctionTrainCore([[exp]], bkd)

        self.assertTrue(core.supports_input_jacobian())

    def test_supports_input_jacobian_with_constants(self) -> None:
        """Test supports_input_jacobian with ConstantExpansion."""
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

        self.assertTrue(core.supports_input_jacobian())

    def test_jacobian_wrt_input_shape(self) -> None:
        """Test jacobian_wrt_input output shape."""
        bkd = self._bkd
        max_level = 2
        exp = self._create_univariate_expansion(max_level, nqoi=1)
        core = FunctionTrainCore([[exp]], bkd)

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        jac = core.jacobian_wrt_input(samples)

        # Shape: (r_left, r_right, nsamples, nqoi)
        r_left, r_right = core.ranks()
        self.assertEqual(jac.shape, (r_left, r_right, nsamples, 1))

    def test_jacobian_wrt_input_shape_2x2(self) -> None:
        """Test jacobian_wrt_input shape for 2x2 core."""
        bkd = self._bkd
        max_level = 1
        exp = self._create_univariate_expansion(max_level, nqoi=1)
        const_0 = ConstantExpansion(0.0, bkd, nqoi=1)
        const_1 = ConstantExpansion(1.0, bkd, nqoi=1)

        core = FunctionTrainCore(
            [[const_1, const_0], [exp, const_1]],
            bkd,
        )

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        jac = core.jacobian_wrt_input(samples)

        self.assertEqual(jac.shape, (2, 2, nsamples, 1))


class TestFunctionTrainInputJacobian(Generic[Array], unittest.TestCase):
    """Tests for FunctionTrain.jacobian and jacobian_batch."""

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

    def test_jacobian_batch_shape(self) -> None:
        """Test jacobian_batch output shape."""
        bkd = self._bkd
        nvars = 3
        ft = self._create_additive_ft(nvars=nvars, max_level=2, nqoi=1)

        # Set random parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        jac = ft.jacobian_batch(samples)

        # Shape: (nsamples, nqoi, nvars)
        self.assertEqual(jac.shape, (nsamples, 1, nvars))

    def test_jacobian_batch_shape_multi_qoi(self) -> None:
        """Test jacobian_batch shape with multiple QoIs."""
        bkd = self._bkd
        nvars = 3
        nqoi = 2
        ft = self._create_additive_ft(nvars=nvars, max_level=2, nqoi=nqoi)

        # Set random parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        nsamples = 4
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        jac = ft.jacobian_batch(samples)

        # Shape: (nsamples, nqoi, nvars)
        self.assertEqual(jac.shape, (nsamples, nqoi, nvars))

    def test_jacobian_shape(self) -> None:
        """Test single-sample jacobian shape."""
        bkd = self._bkd
        nvars = 3
        nqoi = 1
        ft = self._create_additive_ft(nvars=nvars, max_level=2, nqoi=nqoi)

        # Set random parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        sample = bkd.asarray(np.random.uniform(-1, 1, (nvars, 1)))
        jac = ft.jacobian(sample)

        # Shape: (nqoi, nvars)
        self.assertEqual(jac.shape, (nqoi, nvars))

    def test_jacobian_single_var_ft(self) -> None:
        """Test jacobian for single variable FT (nvars=1 edge case)."""
        bkd = self._bkd
        max_level = 2
        nqoi = 1
        exp = self._create_univariate_expansion(max_level, nqoi)

        # Create single-core FT (nvars=1)
        core = FunctionTrainCore([[exp]], bkd)
        ft = FunctionTrain([core], bkd, nqoi)

        # Set random parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        jac = ft.jacobian_batch(samples)

        # Shape: (nsamples, nqoi, nvars=1)
        self.assertEqual(jac.shape, (nsamples, nqoi, 1))

    def test_jacobian_derivative_checker(self) -> None:
        """Test FunctionTrain input jacobian using DerivativeChecker."""
        bkd = self._bkd
        nvars = 3
        max_level = 2
        nqoi = 1
        ft = self._create_additive_ft(nvars=nvars, max_level=max_level, nqoi=nqoi)

        # Set random parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        # Create wrapper: x -> f(x)
        def fun(x: Array) -> Array:
            return ft(x)

        def jacobian_func(x: Array) -> Array:
            return ft.jacobian(x)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=nqoi,
            nvars=nvars,
            fun=fun,
            jacobian=jacobian_func,
            bkd=bkd,
        )

        checker = DerivativeChecker(function_obj)
        sample = bkd.asarray(np.random.uniform(-1, 1, (nvars, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        error_ratio = checker.error_ratio(errors[0])
        self.assertLess(float(error_ratio), 1e-6)

    def test_jacobian_derivative_checker_multi_qoi(self) -> None:
        """Test jacobian with multiple QoIs using DerivativeChecker."""
        bkd = self._bkd
        nvars = 2
        max_level = 2
        nqoi = 2
        ft = self._create_additive_ft(nvars=nvars, max_level=max_level, nqoi=nqoi)

        # Set random parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        # Create wrapper
        def fun(x: Array) -> Array:
            return ft(x)

        def jacobian_func(x: Array) -> Array:
            return ft.jacobian(x)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=nqoi,
            nvars=nvars,
            fun=fun,
            jacobian=jacobian_func,
            bkd=bkd,
        )

        checker = DerivativeChecker(function_obj)
        sample = bkd.asarray(np.random.uniform(-1, 1, (nvars, 1)))
        # Multi-QoI requires weights
        weights = bkd.asarray(np.random.randn(nqoi, 1))
        errors = checker.check_derivatives(sample, weights=weights, verbosity=0)

        error_ratio = checker.error_ratio(errors[0])
        self.assertLess(float(error_ratio), 1e-6)

    def test_jacobian_single_var_derivative_checker(self) -> None:
        """Test jacobian for nvars=1 using DerivativeChecker."""
        bkd = self._bkd
        max_level = 2
        nqoi = 1
        exp = self._create_univariate_expansion(max_level, nqoi)

        # Set random coefficients
        nterms = exp.nterms()
        exp.set_coefficients(bkd.asarray(np.random.randn(nterms, nqoi)))

        # Create single-core FT
        core = FunctionTrainCore([[exp]], bkd)
        ft = FunctionTrain([core], bkd, nqoi)

        def fun(x: Array) -> Array:
            return ft(x)

        def jacobian_func(x: Array) -> Array:
            return ft.jacobian(x)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=nqoi,
            nvars=1,
            fun=fun,
            jacobian=jacobian_func,
            bkd=bkd,
        )

        checker = DerivativeChecker(function_obj)
        sample = bkd.asarray(np.random.uniform(-1, 1, (1, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        error_ratio = checker.error_ratio(errors[0])
        self.assertLess(float(error_ratio), 1e-6)


# NumPy backend tests
class TestConstantExpansionInputJacobianNumpy(
    TestConstantExpansionInputJacobian[NDArray[Any]]
):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFunctionTrainCoreInputJacobianNumpy(
    TestFunctionTrainCoreInputJacobian[NDArray[Any]]
):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFunctionTrainInputJacobianNumpy(
    TestFunctionTrainInputJacobian[NDArray[Any]]
):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestConstantExpansionInputJacobianTorch(
    TestConstantExpansionInputJacobian[torch.Tensor]
):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestFunctionTrainCoreInputJacobianTorch(
    TestFunctionTrainCoreInputJacobian[torch.Tensor]
):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestFunctionTrainInputJacobianTorch(
    TestFunctionTrainInputJacobian[torch.Tensor]
):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


# Torch-only test: verify autograd compatibility
class TestFunctionTrainInputJacobianAutograd(unittest.TestCase):
    """Test that FunctionTrain input jacobian matches torch.autograd."""

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
        nvars = 3
        max_level = 2
        nqoi = 1
        ft = self._create_additive_ft(nvars=nvars, max_level=max_level, nqoi=nqoi)

        # Set random parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        nsamples = 3
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        # Get analytical jacobian
        analytical_jac = ft.jacobian_batch(samples)  # (nsamples, nqoi, nvars)

        # Get autograd jacobian for each sample
        for ii in range(nsamples):
            sample = samples[:, ii : ii + 1].clone().requires_grad_(True)

            def ft_output(x: torch.Tensor) -> torch.Tensor:
                return ft(x).flatten()  # (nqoi,)

            autograd_jac = torch.autograd.functional.jacobian(ft_output, sample)
            # autograd_jac shape: (nqoi, nvars, 1) -> squeeze to (nqoi, nvars)
            autograd_jac = autograd_jac.squeeze(-1)

            bkd.assert_allclose(
                analytical_jac[ii], autograd_jac, rtol=1e-10
            )

    def test_jacobian_matches_autograd_multi_qoi(self) -> None:
        """Test analytical jacobian matches autograd with multiple QoIs."""
        bkd = self._bkd
        nvars = 2
        max_level = 2
        nqoi = 2
        ft = self._create_additive_ft(nvars=nvars, max_level=max_level, nqoi=nqoi)

        # Set random parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        nsamples = 3
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        # Get analytical jacobian
        analytical_jac = ft.jacobian_batch(samples)  # (nsamples, nqoi, nvars)

        # Get autograd jacobian for each sample
        for ii in range(nsamples):
            sample = samples[:, ii : ii + 1].clone().requires_grad_(True)

            def ft_output(x: torch.Tensor) -> torch.Tensor:
                return ft(x).flatten()  # (nqoi,)

            autograd_jac = torch.autograd.functional.jacobian(ft_output, sample)
            # autograd_jac shape: (nqoi, nvars, 1) -> squeeze to (nqoi, nvars)
            autograd_jac = autograd_jac.squeeze(-1)

            bkd.assert_allclose(
                analytical_jac[ii], autograd_jac, rtol=1e-10
            )


if __name__ == "__main__":
    unittest.main()
