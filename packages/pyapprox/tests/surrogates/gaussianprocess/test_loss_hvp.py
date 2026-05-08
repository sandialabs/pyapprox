"""
Tests for NegativeLogMarginalLikelihoodLoss HVP with respect to hyperparameters.

Tests the adjoint method implementation for ultra-fast Hessian-vector products.

NOTE: These tests are currently SKIPPED because HVP is disabled in the loss
function due to a suspected bug. Benchmarks show that trust-constr with HVP
sometimes takes MORE iterations than without HVP, which should never happen
for correct Hessian-based optimization. See benchmark_hvp.py for details.
"""

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.surrogates.gaussianprocess import (
    ConstantMean,
    ExactGaussianProcess,
    NegativeLogMarginalLikelihoodLoss,
)
from pyapprox.surrogates.kernels import SquaredExponentialKernel

# HVP is currently disabled due to suspected bug - see loss.py
HVP_DISABLED_REASON = (
    "HVP is currently disabled in NegativeLogMarginalLikelihoodLoss due to "
    "suspected bug. Benchmarks show trust-constr with HVP takes MORE iterations "
    "than without HVP in some cases. See benchmark_hvp.py for details."
)


@pytest.mark.skip(reason=HVP_DISABLED_REASON)
class TestLossHVP:
    """Test NLML loss HVP with respect to hyperparameters."""

    def _setup(self, bkd):
        """Set up test environment."""
        np.random.seed(42)
        nvars = 2
        n_train = 15

        # Create kernel with two length scale hyperparameters
        # Use SquaredExponentialKernel (RBF) for simpler second derivatives
        length_scale = bkd.array([1.0, 1.0])
        kernel = SquaredExponentialKernel(
            lenscale=length_scale,
            lenscale_bounds=(0.1, 10.0),
            nvars=nvars,
            bkd=bkd,
        )

        # Create GP
        gp = ExactGaussianProcess(
            kernel=kernel, nvars=nvars, bkd=bkd, nugget=0.01
        )

        # Generate training data
        X_train = bkd.array(np.random.randn(nvars, n_train))
        y_train = bkd.array(np.random.randn(1, n_train))

        # Create loss function
        loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)

        # Get initial parameters
        params = gp.hyp_list().get_active_values()

        return gp, kernel, loss, params, nvars, n_train

    def test_hvp_shape(self, bkd):
        """Test that HVP returns correct shape."""
        _, _, loss, params, _, _ = self._setup(bkd)

        nactive = loss.nvars()
        direction = bkd.array(np.random.randn(nactive))

        hvp = loss.hvp(params, direction)

        # Should have shape (nactive, 1) following standard convention
        assert hvp.shape == (nactive, 1)

    def test_hvp_linearity(self, bkd):
        """Test that HVP is linear in direction: H(theta)*(aV) = a*H(theta)*V."""
        _, _, loss, params, _, _ = self._setup(bkd)

        nactive = loss.nvars()
        direction = bkd.array(np.random.randn(nactive))
        a = 2.5

        hvp1 = loss.hvp(params, direction)
        hvp2 = loss.hvp(params, direction * a)

        # hvp2 should be a * hvp1
        assert bkd.allclose(hvp2, hvp1 * a, rtol=1e-6, atol=1e-8)

    def test_hvp_zero_direction(self, bkd):
        """Test HVP with zero direction vector."""
        _, _, loss, params, _, _ = self._setup(bkd)

        nactive = loss.nvars()
        direction = bkd.zeros((nactive,))

        hvp = loss.hvp(params, direction)

        # Should be zero
        zero_hvp = bkd.zeros((nactive, 1))
        assert bkd.allclose(hvp, zero_hvp, atol=1e-12)

    def test_hvp_with_derivative_checker(self, bkd):
        """Test HVP using DerivativeChecker with finite differences."""
        _, _, loss, params, _, _ = self._setup(bkd)

        # Create derivative checker
        checker = DerivativeChecker(loss)

        # Test point (current hyperparameters)
        params_2d = bkd.reshape(params, (len(params), 1))

        # Random direction for checking
        direction = bkd.array(np.random.randn(len(params), 1))
        direction = direction / bkd.norm(direction)

        # Custom FD step sizes
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        # Check derivatives
        errors = checker.check_derivatives(
            params_2d, direction=direction, fd_eps=fd_eps, relative=True, verbosity=0
        )

        # Verify Jacobian is correct
        jac_error = errors[0]
        assert bkd.all_bool(bkd.isfinite(jac_error))
        jac_ratio = float(checker.error_ratio(jac_error))
        assert jac_ratio < 1e-6, f"Jacobian error ratio: {jac_ratio}"

        # Verify HVP is correct
        hvp_error = errors[1]
        assert bkd.all_bool(bkd.isfinite(hvp_error))
        hvp_ratio = float(checker.error_ratio(hvp_error))
        assert hvp_ratio < 1e-6, f"HVP error ratio: {hvp_ratio}"

    def test_hvp_coordinate_directions(self, bkd):
        """Test HVP in coordinate directions (unit vectors)."""
        _, _, loss, params, _, _ = self._setup(bkd)

        nactive = loss.nvars()

        for d in range(nactive):
            # Direction along axis d
            direction = bkd.zeros((nactive,))
            direction[d] = 1.0

            hvp = loss.hvp(params, direction)

            # HVP should have correct shape (nactive, 1)
            assert hvp.shape == (nactive, 1)

            # HVP[d, 0] should match the d-th diagonal element of Hessian
            # (We're not testing the value, just that it computes without error)
            assert bkd.all_bool(bkd.isfinite(hvp))

    def test_hvp_shape_mismatch_error(self, bkd):
        """Test that HVP raises error when direction size doesn't match."""
        _, _, loss, params, _, _ = self._setup(bkd)

        wrong_direction = bkd.array([1.0, 2.0, 3.0])  # Wrong size

        with pytest.raises(ValueError):
            loss.hvp(params, wrong_direction)

    def test_hvp_with_constant_mean(self, bkd):
        """Test HVP with ConstantMean function (adds 1 hyperparameter)."""
        np.random.seed(42)
        nvars = 2
        n_train = 15

        kernel = SquaredExponentialKernel(
            lenscale=bkd.array([1.0, 1.0]),
            lenscale_bounds=(0.1, 10.0),
            nvars=nvars,
            bkd=bkd,
        )

        # Create GP with constant mean
        mean = ConstantMean(0.5, (-2.0, 2.0), bkd)
        gp = ExactGaussianProcess(
            kernel=kernel,
            nvars=nvars,
            bkd=bkd,
            mean_function=mean,
            nugget=0.01,
        )

        X_train = bkd.array(np.random.randn(nvars, n_train))
        y_train = bkd.array(np.random.randn(n_train, 1))

        loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)

        # Should have 3 hyperparameters (2 length scales + 1 constant)
        nactive = loss.nvars()
        assert nactive == 3

        # Test HVP
        params = gp.hyp_list().get_active_values()
        direction = bkd.array(np.random.randn(nactive))

        hvp = loss.hvp(params, direction)

        # Should have correct shape (nactive, 1)
        assert hvp.shape == (nactive, 1)

        # Should be finite
        assert bkd.all_bool(bkd.isfinite(hvp))
