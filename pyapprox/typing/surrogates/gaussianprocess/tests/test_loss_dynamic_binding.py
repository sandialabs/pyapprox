"""
Tests for NegativeLogMarginalLikelihoodLoss dynamic method binding.

This module tests that the loss function correctly exposes/hides the hvp method
based on kernel capabilities, and that the optimizer correctly uses or skips
HVP based on its availability.

Design Notes
------------
All Matern kernel variants (RBF/nu=inf, Matern 5/2, Matern 3/2) now have
hvp_wrt_params support. To test dynamic binding for kernels without HVP,
we create a MockKernelNoHVP that wraps a Matern kernel but doesn't expose
the hvp_wrt_params method.

NOTE: HVP is currently DISABLED in NegativeLogMarginalLikelihoodLoss due to
a suspected bug. Benchmarks show that trust-constr with HVP sometimes takes
MORE iterations than without HVP, which should never happen for correct
Hessian-based optimization. See benchmark_hvp.py for details.
Many tests in this file are skipped until the bug is fixed.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter import HyperParameterList
from pyapprox.typing.surrogates.kernels import (
    SquaredExponentialKernel,
    Matern52Kernel,
    Matern32Kernel,
    Kernel,
)
from pyapprox.typing.surrogates.kernels.iid_gaussian_noise import IIDGaussianNoise
from pyapprox.typing.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.typing.surrogates.gaussianprocess.loss import (
    NegativeLogMarginalLikelihoodLoss
)
from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)


# HVP is currently disabled due to suspected bug - see loss.py
HVP_DISABLED_REASON = (
    "HVP is currently disabled in NegativeLogMarginalLikelihoodLoss due to "
    "suspected bug. See benchmark_hvp.py for details."
)


class MockKernelNoHVP(Kernel):
    """
    Mock kernel wrapper that explicitly excludes hvp_wrt_params.

    Used for testing dynamic method binding behavior when HVP is unavailable.
    """

    def __init__(self, inner_kernel: Kernel):
        super().__init__(inner_kernel.bkd())
        self._inner = inner_kernel
        self._nvars = inner_kernel.nvars()

    def nvars(self) -> int:
        return self._nvars

    def hyp_list(self) -> HyperParameterList:
        return self._inner.hyp_list()

    def __call__(self, X1: Array, X2: Array = None) -> Array:
        return self._inner(X1, X2)

    def diag(self, X1: Array) -> Array:
        return self._inner.diag(X1)

    def jacobian(self, X1: Array, X2: Array) -> Array:
        return self._inner.jacobian(X1, X2)

    def jacobian_wrt_params(self, samples: Array) -> Array:
        return self._inner.jacobian_wrt_params(samples)

    # Deliberately NOT implementing hvp_wrt_params to test dynamic binding


from pyapprox.typing.util.test_utils import load_tests


class TestLossDynamicBinding(Generic[Array], unittest.TestCase):
    """Test NLML loss dynamic method binding based on kernel capabilities."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)
        self._nvars = 2
        self._n_train = 10

        # Generate training data
        self._X_train = self._bkd.array(np.random.randn(self._nvars, self._n_train))
        self._y_train = self._bkd.array(np.random.randn(self._n_train, 1))

    def _create_rbf_kernel(self):
        """Create Squared Exponential kernel (nu=inf), which has hvp_wrt_params."""
        return SquaredExponentialKernel(
            lenscale=self._bkd.array([1.0, 1.0]),
            lenscale_bounds=(0.1, 10.0),
            nvars=self._nvars,
            bkd=self._bkd
        )

    def _create_matern52_kernel(self):
        """Create Matern 5/2 kernel, which now has hvp_wrt_params."""
        return Matern52Kernel(
            lenscale=self._bkd.array([1.0, 1.0]),
            lenscale_bounds=(0.1, 10.0),
            nvars=self._nvars,
            bkd=self._bkd
        )

    def _create_matern32_kernel(self):
        """Create Matern 3/2 kernel, which now has hvp_wrt_params."""
        return Matern32Kernel(
            lenscale=self._bkd.array([1.0, 1.0]),
            lenscale_bounds=(0.1, 10.0),
            nvars=self._nvars,
            bkd=self._bkd
        )

    def _create_kernel_without_hvp(self):
        """Create a kernel wrapper that explicitly lacks hvp_wrt_params."""
        inner = self._create_rbf_kernel()
        return MockKernelNoHVP(inner)

    def test_all_matern_kernels_have_hvp(self):
        """Test that all Matern kernel variants now have hvp_wrt_params."""
        rbf = self._create_rbf_kernel()
        matern52 = self._create_matern52_kernel()
        matern32 = self._create_matern32_kernel()

        self.assertTrue(
            hasattr(rbf, 'hvp_wrt_params'),
            "RBF kernel (nu=inf) should have hvp_wrt_params"
        )
        self.assertTrue(
            hasattr(matern52, 'hvp_wrt_params'),
            "Matern 5/2 kernel should have hvp_wrt_params"
        )
        self.assertTrue(
            hasattr(matern32, 'hvp_wrt_params'),
            "Matern 3/2 kernel should have hvp_wrt_params"
        )

    def test_mock_kernel_lacks_hvp(self):
        """Test that MockKernelNoHVP does NOT have hvp_wrt_params."""
        mock = self._create_kernel_without_hvp()
        self.assertFalse(
            hasattr(mock, 'hvp_wrt_params'),
            "MockKernelNoHVP should NOT have hvp_wrt_params"
        )

    @unittest.skip(HVP_DISABLED_REASON)
    def test_loss_hvp_with_rbf_kernel(self):
        """Test that loss has hvp when kernel has hvp_wrt_params (RBF)."""
        kernel = self._create_rbf_kernel()
        gp = ExactGaussianProcess(
            kernel=kernel,
            nvars=self._nvars,
            bkd=self._bkd,
            nugget=0.1
        )
        loss = NegativeLogMarginalLikelihoodLoss(gp, self._X_train, self._y_train)

        self.assertTrue(
            hasattr(loss, 'hvp'),
            "Loss should have hvp when kernel has hvp_wrt_params"
        )
        self.assertTrue(
            loss._supports_hvp,
            "Loss._supports_hvp should be True"
        )

    @unittest.skip(HVP_DISABLED_REASON)
    def test_loss_hvp_with_matern52_kernel(self):
        """Test that loss has hvp when kernel has hvp_wrt_params (Matern 5/2)."""
        kernel = self._create_matern52_kernel()
        gp = ExactGaussianProcess(
            kernel=kernel,
            nvars=self._nvars,
            bkd=self._bkd,
            nugget=0.1
        )
        loss = NegativeLogMarginalLikelihoodLoss(gp, self._X_train, self._y_train)

        self.assertTrue(
            hasattr(loss, 'hvp'),
            "Loss should have hvp when Matern 5/2 kernel is used"
        )
        self.assertTrue(
            loss._supports_hvp,
            "Loss._supports_hvp should be True for Matern 5/2"
        )

    @unittest.skip(HVP_DISABLED_REASON)
    def test_loss_hvp_with_matern32_kernel(self):
        """Test that loss has hvp when kernel has hvp_wrt_params (Matern 3/2)."""
        kernel = self._create_matern32_kernel()
        gp = ExactGaussianProcess(
            kernel=kernel,
            nvars=self._nvars,
            bkd=self._bkd,
            nugget=0.1
        )
        loss = NegativeLogMarginalLikelihoodLoss(gp, self._X_train, self._y_train)

        self.assertTrue(
            hasattr(loss, 'hvp'),
            "Loss should have hvp when Matern 3/2 kernel is used"
        )
        self.assertTrue(
            loss._supports_hvp,
            "Loss._supports_hvp should be True for Matern 3/2"
        )

    @unittest.skip(HVP_DISABLED_REASON)
    def test_composition_kernel_hvp_when_all_have_it(self):
        """Test that composed kernel has hvp_wrt_params when all components do."""
        # Product of RBF and Matern 5/2 - both have hvp_wrt_params
        k1 = self._create_rbf_kernel()
        k2 = self._create_matern52_kernel()
        product_kernel = k1 * k2

        # Product should have hvp_wrt_params
        self.assertTrue(
            hasattr(product_kernel, 'hvp_wrt_params'),
            "Product of RBF and Matern52 kernels should have hvp_wrt_params"
        )

        # Loss should have hvp
        gp = ExactGaussianProcess(
            kernel=product_kernel,
            nvars=self._nvars,
            bkd=self._bkd,
            nugget=0.1
        )
        loss = NegativeLogMarginalLikelihoodLoss(gp, self._X_train, self._y_train)
        self.assertTrue(
            hasattr(loss, 'hvp'),
            "Loss with product of kernels with HVP should have hvp"
        )

    def test_composition_kernel_no_hvp_when_one_missing(self):
        """Test that composed kernel lacks hvp_wrt_params when one component doesn't have it."""
        # Product of RBF (has hvp) and MockKernelNoHVP (no hvp)
        k_with = self._create_rbf_kernel()
        k_without = self._create_kernel_without_hvp()
        product_kernel = k_with * k_without

        # Product should NOT have hvp_wrt_params (AND logic)
        self.assertFalse(
            hasattr(product_kernel, 'hvp_wrt_params'),
            "Product kernel should NOT have hvp_wrt_params when one component lacks it"
        )

        # Loss should NOT have hvp
        gp = ExactGaussianProcess(
            kernel=product_kernel,
            nvars=self._nvars,
            bkd=self._bkd,
            nugget=0.1
        )
        loss = NegativeLogMarginalLikelihoodLoss(gp, self._X_train, self._y_train)
        self.assertFalse(
            hasattr(loss, 'hvp'),
            "Loss with mixed kernel capabilities should NOT have hvp"
        )

    def test_sum_kernel_hvp_when_all_have_it(self):
        """Test that sum kernel has hvp_wrt_params when all components do."""
        # RBF kernel + IIDGaussianNoise (both have hvp_wrt_params)
        k1 = self._create_rbf_kernel()
        noise = IIDGaussianNoise(
            noise_variance=0.1,
            variance_bounds=(1e-4, 1.0),
            bkd=self._bkd
        )

        # Both IIDGaussianNoise and RBF have hvp_wrt_params
        self.assertTrue(
            hasattr(noise, 'hvp_wrt_params'),
            "IIDGaussianNoise should have hvp_wrt_params"
        )

        sum_kernel = k1 + noise

        self.assertTrue(
            hasattr(sum_kernel, 'hvp_wrt_params'),
            "Sum of kernels with hvp_wrt_params should have hvp_wrt_params"
        )

    @unittest.skip(HVP_DISABLED_REASON)
    def test_optimizer_uses_hvp_with_rbf(self):
        """Test that optimizer uses HVP when loss has it (RBF kernel)."""
        kernel = self._create_rbf_kernel()
        gp = ExactGaussianProcess(
            kernel=kernel,
            nvars=self._nvars,
            bkd=self._bkd,
            nugget=0.1
        )
        loss = NegativeLogMarginalLikelihoodLoss(gp, self._X_train, self._y_train)

        # Confirm loss has hvp
        self.assertTrue(hasattr(loss, 'hvp'))

        # Create optimizer
        nparams = loss.nvars()
        bounds = self._bkd.array([[-5.0, 5.0]] * nparams)

        optimizer = ScipyTrustConstrOptimizer(
            objective=loss,
            bounds=bounds,
            constraints=[],
            verbosity=0,
            maxiter=3,  # Just a few iterations to check
            gtol=1e-12  # Strict tolerance so we get multiple iterations
        )

        # Get initial params
        init_params = self._bkd.reshape(
            gp.hyp_list().get_active_values(),
            (nparams, 1)
        )

        # Run optimization
        result = optimizer.minimize(init_params)

        # Check that HVP was evaluated (nhev > 0)
        scipy_result = result.get_raw_result()
        self.assertGreater(
            scipy_result.nhev, 0,
            f"Optimizer should use HVP when available, but nhev={scipy_result.nhev}"
        )
        # Also check jacobian was evaluated
        self.assertGreater(
            scipy_result.njev, 0,
            f"Optimizer should use jacobian, but njev={scipy_result.njev}"
        )

    @unittest.skip(HVP_DISABLED_REASON)
    def test_optimizer_uses_hvp_with_matern52(self):
        """Test that optimizer uses HVP when loss has it (Matern 5/2 kernel)."""
        kernel = self._create_matern52_kernel()
        gp = ExactGaussianProcess(
            kernel=kernel,
            nvars=self._nvars,
            bkd=self._bkd,
            nugget=0.1
        )
        loss = NegativeLogMarginalLikelihoodLoss(gp, self._X_train, self._y_train)

        # Confirm loss has hvp
        self.assertTrue(hasattr(loss, 'hvp'))

        # Create optimizer
        nparams = loss.nvars()
        bounds = self._bkd.array([[-5.0, 5.0]] * nparams)

        optimizer = ScipyTrustConstrOptimizer(
            objective=loss,
            bounds=bounds,
            constraints=[],
            verbosity=0,
            maxiter=3,
            gtol=1e-12
        )

        # Get initial params
        init_params = self._bkd.reshape(
            gp.hyp_list().get_active_values(),
            (nparams, 1)
        )

        # Run optimization
        result = optimizer.minimize(init_params)

        # Check that HVP was evaluated (nhev > 0)
        scipy_result = result.get_raw_result()
        self.assertGreater(
            scipy_result.nhev, 0,
            f"Optimizer should use HVP for Matern52, but nhev={scipy_result.nhev}"
        )

    def test_optimizer_skips_hvp_when_unavailable(self):
        """Test that optimizer does not use HVP when loss lacks it."""
        # Use RBF * MockKernelNoHVP, which lacks HVP due to MockKernelNoHVP
        k_with = self._create_rbf_kernel()
        k_without = self._create_kernel_without_hvp()
        kernel = k_with * k_without

        gp = ExactGaussianProcess(
            kernel=kernel,
            nvars=self._nvars,
            bkd=self._bkd,
            nugget=0.1
        )
        loss = NegativeLogMarginalLikelihoodLoss(gp, self._X_train, self._y_train)

        # Confirm loss does NOT have hvp
        self.assertFalse(hasattr(loss, 'hvp'))

        # Create optimizer
        nparams = loss.nvars()
        bounds = self._bkd.array([[-5.0, 5.0]] * nparams)

        optimizer = ScipyTrustConstrOptimizer(
            objective=loss,
            bounds=bounds,
            constraints=[],
            verbosity=0,
            maxiter=3,
            gtol=1e-12
        )

        # Get initial params
        init_params = self._bkd.reshape(
            gp.hyp_list().get_active_values(),
            (nparams, 1)
        )

        # Run optimization
        result = optimizer.minimize(init_params)

        # Check that HVP was NOT evaluated (nhev == 0)
        scipy_result = result.get_raw_result()
        self.assertEqual(
            scipy_result.nhev, 0,
            f"Optimizer should NOT use HVP when unavailable, but nhev={scipy_result.nhev}"
        )
        # Jacobian should still be evaluated
        self.assertGreater(
            scipy_result.njev, 0,
            f"Optimizer should still use jacobian, but njev={scipy_result.njev}"
        )


class TestLossDynamicBindingNumpy(TestLossDynamicBinding[NDArray[Any]]):
    """Test NLML loss dynamic binding with NumPy backend."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLossDynamicBindingTorch(TestLossDynamicBinding[torch.Tensor]):
    """Test NLML loss dynamic binding with PyTorch backend."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
