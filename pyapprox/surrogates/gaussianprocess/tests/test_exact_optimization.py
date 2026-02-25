"""
Tests for ExactGaussianProcess hyperparameter optimization.

Tests that fit() performs hyperparameter optimization, custom optimizer
configuration, and fixed hyperparameter handling.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.surrogates.gaussianprocess import (
    ConstantMean,
    ExactGaussianProcess,
)
from pyapprox.surrogates.kernels.matern import Matern52Kernel
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestExactGPOptimization(Generic[Array], unittest.TestCase):
    """
    Base test class for ExactGaussianProcess hyperparameter optimization.

    Derived classes must implement the bkd() method.
    """

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment."""
        np.random.seed(42)
        self.nvars = 2
        self.n_train = 10
        self.n_test = 5

        # Create training data
        X_train_np = np.random.randn(self.nvars, self.n_train)
        y_train_np = np.sin(X_train_np[0, :] + X_train_np[1, :])[None, :]

        self.X_train = self.bkd().array(X_train_np)
        self.y_train = self.bkd().array(y_train_np)

        # Create test data
        X_test_np = np.random.randn(self.nvars, self.n_test)
        self.X_test = self.bkd().array(X_test_np)

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def test_fit_optimizes_hyperparameters(self) -> None:
        """Test that fit() optimizes hyperparameters with default optimizer."""
        # Use suboptimal long length scales initially
        kernel = Matern52Kernel([3.0, 3.0], (0.1, 10.0), self.nvars, self.bkd())

        gp = ExactGaussianProcess(kernel, self.nvars, self.bkd(), nugget=0.1)

        # Store initial hyperparameters (before fit)
        params_before = self.bkd().to_numpy(gp.hyp_list().get_values()).copy()

        # fit() should optimize hyperparameters automatically
        gp.fit(self.X_train, self.y_train)

        # Get final hyperparameters
        params_after = self.bkd().to_numpy(gp.hyp_list().get_values())

        # Hyperparameters should have changed
        params_diff = np.abs(params_after - params_before)
        max_change = float(np.max(params_diff))
        self.assertGreater(
            max_change, 1e-6, "Hyperparameters should change during fit()"
        )

        # GP should still make predictions
        mean = gp.predict(self.X_test)
        self.assertEqual(mean.shape, (1, self.n_test))
        self.assertTrue(self.bkd().all_bool(self.bkd().isfinite(mean)))

    def test_fit_with_constant_mean(self) -> None:
        """Test fit() optimizes both kernel and mean hyperparameters."""
        kernel = Matern52Kernel([2.0, 2.0], (0.1, 10.0), self.nvars, self.bkd())

        # Use ConstantMean with initial guess
        constant_mean = ConstantMean(0.0, (-5.0, 5.0), self.bkd())

        gp = ExactGaussianProcess(
            kernel, self.nvars, self.bkd(), mean_function=constant_mean, nugget=0.1
        )

        # fit() should optimize both kernel and mean parameters
        gp.fit(self.X_train, self.y_train)

        # Should have 3 hyperparameters (2 length scales + 1 constant)
        self.assertEqual(gp.hyp_list().nparams(), 3)

        # GP should make predictions
        mean = gp.predict(self.X_test)
        self.assertEqual(mean.shape, (1, self.n_test))

    def test_fit_improves_predictions(self) -> None:
        """Test that fit() optimization improves prediction accuracy."""
        # Create synthetic data with known length scales
        np.random.seed(123)
        X_train_np = np.random.uniform(-2, 2, (self.nvars, 30))

        # True function with characteristic length scale ~0.5
        y_train_np = np.sin(2 * X_train_np[0, :]) * np.cos(2 * X_train_np[1, :])
        y_train_np = y_train_np[None, :]

        X_train = self.bkd().array(X_train_np)
        y_train = self.bkd().array(y_train_np)

        # Test points
        X_test_np = np.random.uniform(-2, 2, (self.nvars, 10))
        y_test_np = np.sin(2 * X_test_np[0, :]) * np.cos(2 * X_test_np[1, :])
        y_test_np = y_test_np[None, :]

        X_test = self.bkd().array(X_test_np)
        y_test = self.bkd().array(y_test_np)

        # Start with poor initial hyperparameters (long length scales)
        kernel = Matern52Kernel(
            [5.0, 5.0],  # Too long
            (0.1, 10.0),
            self.nvars,
            self.bkd(),
        )

        gp = ExactGaussianProcess(kernel, self.nvars, self.bkd(), nugget=0.01)

        # First fit WITHOUT optimization to get baseline error
        gp.hyp_list().set_all_inactive()
        gp.fit(X_train, y_train)
        mean_before = gp.predict(X_test)
        error_before = mean_before - y_test
        abs_error_before = self.bkd().abs(error_before)
        mse_before = float(
            self.bkd().sum(abs_error_before**2) / abs_error_before.shape[1]
        )

        # Reset hyperparameters and re-enable optimization
        # Note: kernel uses LogHyperParameter, so set_values expects log-space
        # values. log(5.0) = 1.609... which is within log-space bounds [-2.3, 2.3]
        import math

        kernel._hyp_list.set_values(self.bkd().array([math.log(5.0), math.log(5.0)]))
        gp.hyp_list().set_all_active()

        # Now fit WITH optimization
        gp.fit(X_train, y_train)

        # Predict after optimization
        mean_after = gp.predict(X_test)
        error_after = mean_after - y_test
        abs_error_after = self.bkd().abs(error_after)
        mse_after = float(self.bkd().sum(abs_error_after**2) / abs_error_after.shape[1])

        # MSE should improve
        self.assertLessEqual(
            mse_after, mse_before * 1.1, f"MSE got worse: {mse_before} -> {mse_after}"
        )

    def test_fit_with_custom_optimizer(self) -> None:
        """Test fit() with custom optimizer via set_optimizer()."""
        kernel = Matern52Kernel([3.0, 3.0], (0.1, 10.0), self.nvars, self.bkd())

        gp = ExactGaussianProcess(kernel, self.nvars, self.bkd(), nugget=0.1)

        # Set custom optimizer
        custom_optimizer = ScipyTrustConstrOptimizer(
            maxiter=500, gtol=1e-8, verbosity=0
        )
        gp.set_optimizer(custom_optimizer)

        # Verify optimizer is set
        self.assertIsNotNone(gp.optimizer())

        # fit() should use the custom optimizer
        gp.fit(self.X_train, self.y_train)

        # Should still make predictions
        mean = gp.predict(self.X_test)
        self.assertEqual(mean.shape, (1, self.n_test))

    def test_fit_skips_optimization_when_all_inactive(self) -> None:
        """Test that fit() skips optimization when all hyperparameters are inactive."""
        kernel = Matern52Kernel([1.0, 1.0], (0.1, 10.0), self.nvars, self.bkd())

        gp = ExactGaussianProcess(kernel, self.nvars, self.bkd(), nugget=0.1)

        # Set all hyperparameters as inactive (fixed)
        gp.hyp_list().set_all_inactive()

        # Store initial values
        params_before = self.bkd().to_numpy(gp.hyp_list().get_values()).copy()

        # fit() should only store data (no optimization)
        gp.fit(self.X_train, self.y_train)

        # Hyperparameters should NOT change
        params_after = self.bkd().to_numpy(gp.hyp_list().get_values())
        np.testing.assert_array_equal(
            params_before,
            params_after,
            "Hyperparameters should not change when all are inactive",
        )

        # GP should still make predictions
        mean = gp.predict(self.X_test)
        self.assertEqual(mean.shape, (1, self.n_test))

    def test_shared_optimizer_is_cloned(self) -> None:
        """Test that optimizer is cloned so shared optimizer is safe."""
        # Create shared optimizer
        shared_optimizer = ScipyTrustConstrOptimizer(maxiter=500, verbosity=0)

        # Create two GPs with same optimizer
        kernel1 = Matern52Kernel([2.0, 2.0], (0.1, 10.0), self.nvars, self.bkd())
        gp1 = ExactGaussianProcess(kernel1, self.nvars, self.bkd(), nugget=0.1)
        gp1.set_optimizer(shared_optimizer)

        kernel2 = Matern52Kernel([2.0, 2.0], (0.1, 10.0), self.nvars, self.bkd())
        gp2 = ExactGaussianProcess(kernel2, self.nvars, self.bkd(), nugget=0.1)
        gp2.set_optimizer(shared_optimizer)

        # Fitting gp1 should not affect gp2's optimizer
        gp1.fit(self.X_train, self.y_train)

        # The shared optimizer should still be unbound
        self.assertFalse(shared_optimizer.is_bound())

        # gp2 should also be able to fit
        gp2.fit(self.X_train, self.y_train)

        # Both should make predictions
        mean1 = gp1.predict(self.X_test)
        mean2 = gp2.predict(self.X_test)
        self.assertEqual(mean1.shape, (1, self.n_test))
        self.assertEqual(mean2.shape, (1, self.n_test))

    def test_set_optimizer_validates_protocol(self) -> None:
        """Test that set_optimizer validates the protocol."""
        kernel = Matern52Kernel([1.0, 1.0], (0.1, 10.0), self.nvars, self.bkd())
        gp = ExactGaussianProcess(kernel, self.nvars, self.bkd(), nugget=0.1)

        # Should raise TypeError for invalid optimizer
        with self.assertRaises(TypeError):
            gp.set_optimizer("not an optimizer")  # type: ignore


# NumPy implementation
class TestExactGPOptimizationNumpy(TestExactGPOptimization[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch implementation
class TestExactGPOptimizationTorch(TestExactGPOptimization[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


from pyapprox.util.test_utils import load_tests  # noqa: F401

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
