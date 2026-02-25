"""
Tests for NegativeLogMarginalLikelihoodLoss HVP with respect to hyperparameters.

Tests the adjoint method implementation for ultra-fast Hessian-vector products.

NOTE: These tests are currently SKIPPED because HVP is disabled in the loss
function due to a suspected bug. Benchmarks show that trust-constr with HVP
sometimes takes MORE iterations than without HVP, which should never happen
for correct Hessian-based optimization. See benchmark_hvp.py for details.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.protocols import Array
from pyapprox.surrogates.kernels import SquaredExponentialKernel
from pyapprox.surrogates.gaussianprocess import (
    ExactGaussianProcess,
    ConstantMean,
    NegativeLogMarginalLikelihoodLoss
)
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker
)


from pyapprox.util.test_utils import load_tests


# HVP is currently disabled due to suspected bug - see loss.py
HVP_DISABLED_REASON = (
    "HVP is currently disabled in NegativeLogMarginalLikelihoodLoss due to "
    "suspected bug. Benchmarks show trust-constr with HVP takes MORE iterations "
    "than without HVP in some cases. See benchmark_hvp.py for details."
)


@unittest.skip(HVP_DISABLED_REASON)
class TestLossHVP(Generic[Array], unittest.TestCase):
    """Test NLML loss HVP with respect to hyperparameters."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

        # Create a simple 2D GP with 2 length scale hyperparameters
        self._nvars = 2
        self._n_train = 15

        # Create kernel with two length scale hyperparameters
        # Use SquaredExponentialKernel (RBF) for simpler second derivatives
        length_scale = self._bkd.array([1.0, 1.0])
        self._kernel = SquaredExponentialKernel(
            lenscale=length_scale,
            lenscale_bounds=(0.1, 10.0),
            nvars=self._nvars,
            bkd=self._bkd
        )

        # Create GP
        self._gp = ExactGaussianProcess(
            kernel=self._kernel,
            nvars=self._nvars,
            bkd=self._bkd,
            nugget=0.01
        )

        # Generate training data
        X_train = self._bkd.array(np.random.randn(self._nvars, self._n_train))
        y_train = self._bkd.array(np.random.randn(1, self._n_train))

        # Create loss function
        self._loss = NegativeLogMarginalLikelihoodLoss(
            self._gp, X_train, y_train
        )

        # Get initial parameters
        self._params = self._gp.hyp_list().get_active_values()

    def test_hvp_shape(self):
        """Test that HVP returns correct shape."""
        nactive = self._loss.nvars()
        direction = self._bkd.array(np.random.randn(nactive))

        hvp = self._loss.hvp(self._params, direction)

        # Should have shape (nactive, 1) following standard convention
        self.assertEqual(hvp.shape, (nactive, 1))

    def test_hvp_linearity(self):
        """Test that HVP is linear in direction: H(θ)·(aV) = a·H(θ)·V."""
        nactive = self._loss.nvars()
        direction = self._bkd.array(np.random.randn(nactive))
        a = 2.5

        hvp1 = self._loss.hvp(self._params, direction)
        hvp2 = self._loss.hvp(self._params, direction * a)

        # hvp2 should be a * hvp1
        self.assertTrue(
            self._bkd.allclose(hvp2, hvp1 * a, rtol=1e-6, atol=1e-8)
        )

    def test_hvp_zero_direction(self):
        """Test HVP with zero direction vector."""
        nactive = self._loss.nvars()
        direction = self._bkd.zeros((nactive,))

        hvp = self._loss.hvp(self._params, direction)

        # Should be zero
        zero_hvp = self._bkd.zeros((nactive, 1))
        self.assertTrue(
            self._bkd.allclose(hvp, zero_hvp, atol=1e-12)
        )

    def test_hvp_with_derivative_checker(self):
        """Test HVP using DerivativeChecker with finite differences."""
        # Create derivative checker
        checker = DerivativeChecker(self._loss)

        # Test point (current hyperparameters)
        params_2d = self._bkd.reshape(self._params, (len(self._params), 1))

        # Random direction for checking
        direction = self._bkd.array(np.random.randn(len(self._params), 1))
        direction = direction / self._bkd.norm(direction)

        # Custom FD step sizes
        fd_eps = self._bkd.flip(self._bkd.logspace(-14, 0, 15))

        # Check derivatives
        errors = checker.check_derivatives(
            params_2d,
            direction=direction,
            fd_eps=fd_eps,
            relative=True,
            verbosity=0
        )

        # Verify Jacobian is correct
        jac_error = errors[0]
        self.assertTrue(
            self._bkd.all_bool(self._bkd.isfinite(jac_error))
        )
        jac_ratio = float(checker.error_ratio(jac_error))
        self.assertLess(jac_ratio, 1e-6, f"Jacobian error ratio: {jac_ratio}")

        # Verify HVP is correct
        hvp_error = errors[1]
        self.assertTrue(
            self._bkd.all_bool(self._bkd.isfinite(hvp_error))
        )
        hvp_ratio = float(checker.error_ratio(hvp_error))
        self.assertLess(hvp_ratio, 1e-6, f"HVP error ratio: {hvp_ratio}")

    def test_hvp_coordinate_directions(self):
        """Test HVP in coordinate directions (unit vectors)."""
        nactive = self._loss.nvars()

        for d in range(nactive):
            # Direction along axis d
            direction = self._bkd.zeros((nactive,))
            direction[d] = 1.0

            hvp = self._loss.hvp(self._params, direction)

            # HVP should have correct shape (nactive, 1)
            self.assertEqual(hvp.shape, (nactive, 1))

            # HVP[d, 0] should match the d-th diagonal element of Hessian
            # (We're not testing the value, just that it computes without error)
            self.assertTrue(self._bkd.all_bool(self._bkd.isfinite(hvp)))

    def test_hvp_shape_mismatch_error(self):
        """Test that HVP raises error when direction size doesn't match."""
        wrong_direction = self._bkd.array([1.0, 2.0, 3.0])  # Wrong size

        with self.assertRaises(ValueError):
            self._loss.hvp(self._params, wrong_direction)

    def test_hvp_with_constant_mean(self):
        """Test HVP with ConstantMean function (adds 1 hyperparameter)."""
        # Create GP with constant mean
        mean = ConstantMean(0.5, (-2.0, 2.0), self._bkd)
        gp = ExactGaussianProcess(
            kernel=self._kernel,
            nvars=self._nvars,
            bkd=self._bkd,
            mean_function=mean,
            nugget=0.01
        )

        X_train = self._bkd.array(np.random.randn(self._nvars, self._n_train))
        y_train = self._bkd.array(np.random.randn(self._n_train, 1))

        loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)

        # Should have 3 hyperparameters (2 length scales + 1 constant)
        nactive = loss.nvars()
        self.assertEqual(nactive, 3)

        # Test HVP
        params = gp.hyp_list().get_active_values()
        direction = self._bkd.array(np.random.randn(nactive))

        hvp = loss.hvp(params, direction)

        # Should have correct shape (nactive, 1)
        self.assertEqual(hvp.shape, (nactive, 1))

        # Should be finite
        self.assertTrue(self._bkd.all_bool(self._bkd.isfinite(hvp)))


class TestLossHVPNumpy(TestLossHVP[NDArray[Any]]):
    """Test NLML loss HVP with NumPy backend."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLossHVPTorch(TestLossHVP[torch.Tensor]):
    """Test NLML loss HVP with PyTorch backend."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
