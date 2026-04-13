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

import numpy as np
import pytest

from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.loss import NegativeLogMarginalLikelihoodLoss
from pyapprox.surrogates.kernels import (
    Kernel,
    Matern32Kernel,
    Matern52Kernel,
    SquaredExponentialKernel,
)
from pyapprox.surrogates.kernels.iid_gaussian_noise import IIDGaussianNoise
from pyapprox.util.hyperparameter import HyperParameterList

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

    def __call__(self, X1, X2=None):
        return self._inner(X1, X2)

    def diag(self, X1):
        return self._inner.diag(X1)

    def jacobian(self, X1, X2):
        return self._inner.jacobian(X1, X2)

    def jacobian_wrt_params(self, samples):
        return self._inner.jacobian_wrt_params(samples)

    # Deliberately NOT implementing hvp_wrt_params to test dynamic binding


class TestLossDynamicBinding:
    """Test NLML loss dynamic method binding based on kernel capabilities."""

    def _setup(self, bkd):
        """Set up test data."""
        np.random.seed(42)
        nvars = 2
        n_train = 10

        X_train = bkd.array(np.random.randn(nvars, n_train))
        y_train = bkd.array(np.random.randn(1, n_train))

        return nvars, X_train, y_train

    def _create_rbf_kernel(self, bkd, nvars):
        """Create Squared Exponential kernel (nu=inf), which has hvp_wrt_params."""
        return SquaredExponentialKernel(
            lenscale=bkd.array([1.0, 1.0]),
            lenscale_bounds=(0.1, 10.0),
            nvars=nvars,
            bkd=bkd
        )

    def _create_matern52_kernel(self, bkd, nvars):
        """Create Matern 5/2 kernel, which now has hvp_wrt_params."""
        return Matern52Kernel(
            lenscale=bkd.array([1.0, 1.0]),
            lenscale_bounds=(0.1, 10.0),
            nvars=nvars,
            bkd=bkd
        )

    def _create_matern32_kernel(self, bkd, nvars):
        """Create Matern 3/2 kernel, which now has hvp_wrt_params."""
        return Matern32Kernel(
            lenscale=bkd.array([1.0, 1.0]),
            lenscale_bounds=(0.1, 10.0),
            nvars=nvars,
            bkd=bkd
        )

    def _create_kernel_without_hvp(self, bkd, nvars):
        """Create a kernel wrapper that explicitly lacks hvp_wrt_params."""
        inner = self._create_rbf_kernel(bkd, nvars)
        return MockKernelNoHVP(inner)

    def test_all_matern_kernels_have_hvp(self, bkd):
        """Test that all Matern kernel variants now have hvp_wrt_params."""
        nvars, _, _ = self._setup(bkd)

        rbf = self._create_rbf_kernel(bkd, nvars)
        matern52 = self._create_matern52_kernel(bkd, nvars)
        matern32 = self._create_matern32_kernel(bkd, nvars)

        assert hasattr(rbf, 'hvp_wrt_params'), \
            "RBF kernel (nu=inf) should have hvp_wrt_params"
        assert hasattr(matern52, 'hvp_wrt_params'), \
            "Matern 5/2 kernel should have hvp_wrt_params"
        assert hasattr(matern32, 'hvp_wrt_params'), \
            "Matern 3/2 kernel should have hvp_wrt_params"

    def test_mock_kernel_lacks_hvp(self, bkd):
        """Test that MockKernelNoHVP does NOT have hvp_wrt_params."""
        nvars, _, _ = self._setup(bkd)

        mock = self._create_kernel_without_hvp(bkd, nvars)
        assert not hasattr(mock, 'hvp_wrt_params'), \
            "MockKernelNoHVP should NOT have hvp_wrt_params"

    @pytest.mark.skip(reason=HVP_DISABLED_REASON)
    def test_loss_hvp_with_rbf_kernel(self, bkd):
        """Test that loss has hvp when kernel has hvp_wrt_params (RBF)."""
        nvars, X_train, y_train = self._setup(bkd)

        kernel = self._create_rbf_kernel(bkd, nvars)
        gp = ExactGaussianProcess(
            kernel=kernel,
            nvars=nvars,
            bkd=bkd,
            nugget=0.1
        )
        loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)

        assert hasattr(loss, 'hvp'), \
            "Loss should have hvp when kernel has hvp_wrt_params"
        assert loss._supports_hvp, \
            "Loss._supports_hvp should be True"

    @pytest.mark.skip(reason=HVP_DISABLED_REASON)
    def test_loss_hvp_with_matern52_kernel(self, bkd):
        """Test that loss has hvp when kernel has hvp_wrt_params (Matern 5/2)."""
        nvars, X_train, y_train = self._setup(bkd)

        kernel = self._create_matern52_kernel(bkd, nvars)
        gp = ExactGaussianProcess(
            kernel=kernel,
            nvars=nvars,
            bkd=bkd,
            nugget=0.1
        )
        loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)

        assert hasattr(loss, 'hvp'), \
            "Loss should have hvp when Matern 5/2 kernel is used"
        assert loss._supports_hvp, \
            "Loss._supports_hvp should be True for Matern 5/2"

    @pytest.mark.skip(reason=HVP_DISABLED_REASON)
    def test_loss_hvp_with_matern32_kernel(self, bkd):
        """Test that loss has hvp when kernel has hvp_wrt_params (Matern 3/2)."""
        nvars, X_train, y_train = self._setup(bkd)

        kernel = self._create_matern32_kernel(bkd, nvars)
        gp = ExactGaussianProcess(
            kernel=kernel,
            nvars=nvars,
            bkd=bkd,
            nugget=0.1
        )
        loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)

        assert hasattr(loss, 'hvp'), \
            "Loss should have hvp when Matern 3/2 kernel is used"
        assert loss._supports_hvp, \
            "Loss._supports_hvp should be True for Matern 3/2"

    @pytest.mark.skip(reason=HVP_DISABLED_REASON)
    def test_composition_kernel_hvp_when_all_have_it(self, bkd):
        """Test that composed kernel has hvp_wrt_params when all components do."""
        nvars, X_train, y_train = self._setup(bkd)

        # Product of RBF and Matern 5/2 - both have hvp_wrt_params
        k1 = self._create_rbf_kernel(bkd, nvars)
        k2 = self._create_matern52_kernel(bkd, nvars)
        product_kernel = k1 * k2

        # Product should have hvp_wrt_params
        assert hasattr(product_kernel, 'hvp_wrt_params'), \
            "Product of RBF and Matern52 kernels should have hvp_wrt_params"

        # Loss should have hvp
        gp = ExactGaussianProcess(
            kernel=product_kernel,
            nvars=nvars,
            bkd=bkd,
            nugget=0.1
        )
        loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)
        assert hasattr(loss, 'hvp'), \
            "Loss with product of kernels with HVP should have hvp"

    def test_composition_kernel_no_hvp_when_one_missing(self, bkd):
        """Test composed kernel lacks hvp_wrt_params
        when one component doesn't have it."""
        nvars, X_train, y_train = self._setup(bkd)

        # Product of RBF (has hvp) and MockKernelNoHVP (no hvp)
        k_with = self._create_rbf_kernel(bkd, nvars)
        k_without = self._create_kernel_without_hvp(bkd, nvars)
        product_kernel = k_with * k_without

        # Product should NOT have hvp_wrt_params (AND logic)
        assert not hasattr(product_kernel, 'hvp_wrt_params'), \
            "Product kernel should NOT have hvp_wrt_params when one component lacks it"

        # Loss should NOT have hvp
        gp = ExactGaussianProcess(
            kernel=product_kernel,
            nvars=nvars,
            bkd=bkd,
            nugget=0.1
        )
        loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)
        assert not hasattr(loss, 'hvp'), \
            "Loss with mixed kernel capabilities should NOT have hvp"

    def test_sum_kernel_hvp_when_all_have_it(self, bkd):
        """Test that sum kernel has hvp_wrt_params when all components do."""
        nvars, _, _ = self._setup(bkd)

        # RBF kernel + IIDGaussianNoise (both have hvp_wrt_params)
        k1 = self._create_rbf_kernel(bkd, nvars)
        noise = IIDGaussianNoise(
            noise_variance=0.1,
            variance_bounds=(1e-4, 1.0),
            bkd=bkd
        )

        # Both IIDGaussianNoise and RBF have hvp_wrt_params
        assert hasattr(noise, 'hvp_wrt_params'), \
            "IIDGaussianNoise should have hvp_wrt_params"

        sum_kernel = k1 + noise

        assert hasattr(sum_kernel, 'hvp_wrt_params'), \
            "Sum of kernels with hvp_wrt_params should have hvp_wrt_params"

    @pytest.mark.skip(reason=HVP_DISABLED_REASON)
    def test_optimizer_uses_hvp_with_rbf(self, bkd):
        """Test that optimizer uses HVP when loss has it (RBF kernel)."""
        nvars, X_train, y_train = self._setup(bkd)

        kernel = self._create_rbf_kernel(bkd, nvars)
        gp = ExactGaussianProcess(
            kernel=kernel,
            nvars=nvars,
            bkd=bkd,
            nugget=0.1
        )
        loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)

        # Confirm loss has hvp
        assert hasattr(loss, 'hvp')

        # Create optimizer
        nparams = loss.nvars()
        bounds = bkd.array([[-5.0, 5.0]] * nparams)

        optimizer = ScipyTrustConstrOptimizer(
            objective=loss,
            bounds=bounds,
            constraints=[],
            verbosity=0,
            maxiter=3,  # Just a few iterations to check
            gtol=1e-12  # Strict tolerance so we get multiple iterations
        )

        # Get initial params
        init_params = bkd.reshape(
            gp.hyp_list().get_active_values(),
            (nparams, 1)
        )

        # Run optimization
        result = optimizer.minimize(init_params)

        # Check that HVP was evaluated (nhev > 0)
        scipy_result = result.get_raw_result()
        assert scipy_result.nhev > 0, \
            f"Optimizer should use HVP when available, but nhev={scipy_result.nhev}"
        # Also check jacobian was evaluated
        assert scipy_result.njev > 0, \
            f"Optimizer should use jacobian, but njev={scipy_result.njev}"

    @pytest.mark.skip(reason=HVP_DISABLED_REASON)
    def test_optimizer_uses_hvp_with_matern52(self, bkd):
        """Test that optimizer uses HVP when loss has it (Matern 5/2 kernel)."""
        nvars, X_train, y_train = self._setup(bkd)

        kernel = self._create_matern52_kernel(bkd, nvars)
        gp = ExactGaussianProcess(
            kernel=kernel,
            nvars=nvars,
            bkd=bkd,
            nugget=0.1
        )
        loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)

        # Confirm loss has hvp
        assert hasattr(loss, 'hvp')

        # Create optimizer
        nparams = loss.nvars()
        bounds = bkd.array([[-5.0, 5.0]] * nparams)

        optimizer = ScipyTrustConstrOptimizer(
            objective=loss,
            bounds=bounds,
            constraints=[],
            verbosity=0,
            maxiter=3,
            gtol=1e-12
        )

        # Get initial params
        init_params = bkd.reshape(
            gp.hyp_list().get_active_values(),
            (nparams, 1)
        )

        # Run optimization
        result = optimizer.minimize(init_params)

        # Check that HVP was evaluated (nhev > 0)
        scipy_result = result.get_raw_result()
        assert scipy_result.nhev > 0, \
            f"Optimizer should use HVP for Matern52, but nhev={scipy_result.nhev}"

    def test_optimizer_skips_hvp_when_unavailable(self, bkd):
        """Test that optimizer does not use HVP when loss lacks it."""
        nvars, X_train, y_train = self._setup(bkd)

        # Use RBF * MockKernelNoHVP, which lacks HVP due to MockKernelNoHVP
        k_with = self._create_rbf_kernel(bkd, nvars)
        k_without = self._create_kernel_without_hvp(bkd, nvars)
        kernel = k_with * k_without

        gp = ExactGaussianProcess(
            kernel=kernel,
            nvars=nvars,
            bkd=bkd,
            nugget=0.1
        )
        loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)

        # Confirm loss does NOT have hvp
        assert not hasattr(loss, 'hvp')

        # Create optimizer
        nparams = loss.nvars()
        bounds = bkd.array([[-5.0, 5.0]] * nparams)

        optimizer = ScipyTrustConstrOptimizer(
            objective=loss,
            bounds=bounds,
            constraints=[],
            verbosity=0,
            maxiter=3,
            gtol=1e-12
        )

        # Get initial params
        init_params = bkd.reshape(
            gp.hyp_list().get_active_values(),
            (nparams, 1)
        )

        # Run optimization
        result = optimizer.minimize(init_params)

        # Check that HVP was NOT evaluated (nhev == 0)
        scipy_result = result.get_raw_result()
        assert scipy_result.nhev == 0, \
            "Optimizer should NOT use HVP when " \
            f"unavailable, but nhev={scipy_result.nhev}"
        # Jacobian should still be evaluated
        assert scipy_result.njev > 0, \
            f"Optimizer should still use jacobian, but njev={scipy_result.njev}"
