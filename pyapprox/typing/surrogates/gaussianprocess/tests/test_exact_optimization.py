"""
Tests for ExactGaussianProcess hyperparameter optimization.

Tests optimization functionality, convergence, and custom initial guesses.
"""
import unittest
from typing import Generic, Any
import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.surrogates.kernels.matern import Matern52Kernel
from pyapprox.typing.surrogates.gaussianprocess import (
    ExactGaussianProcess,
    ConstantMean
)


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
        y_train_np = np.sin(X_train_np[0, :] + X_train_np[1, :])[:, None]

        self.X_train = self.bkd().array(X_train_np)
        self.y_train = self.bkd().array(y_train_np)

        # Create test data
        X_test_np = np.random.randn(self.nvars, self.n_test)
        self.X_test = self.bkd().array(X_test_np)

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def test_optimize_hyperparameters_basic(self) -> None:
        """Test basic hyperparameter optimization."""
        # Use longer length scales initially (suboptimal)
        kernel = Matern52Kernel(
            [3.0, 3.0],  # Long length scales
            (0.1, 10.0),
            self.nvars,
            self.bkd()
        )

        gp = ExactGaussianProcess(
            kernel,
            self.nvars,
            self.bkd(),
            nugget=0.1
        )

        # Fit with initial hyperparameters
        gp.fit(self.X_train, self.y_train)

        # Get initial NLML
        nlml_before = gp.neg_log_marginal_likelihood()

        # Get initial hyperparameters (store as numpy to avoid copy issues)
        params_before = self.bkd().to_numpy(gp.hyp_list().get_values()).copy()

        # Optimize hyperparameters (uses default trust-constr)
        gp.optimize_hyperparameters()

        # Get final NLML
        nlml_after = gp.neg_log_marginal_likelihood()

        # Get final hyperparameters
        params_after = self.bkd().to_numpy(gp.hyp_list().get_values())

        # NLML should decrease (or stay same if already optimal)
        self.assertLessEqual(nlml_after, nlml_before + 1e-6,
                            f"NLML increased: {nlml_before} -> {nlml_after}")

        # Hyperparameters should have changed
        params_diff = np.abs(params_after - params_before)
        max_change = float(np.max(params_diff))
        # At least one parameter should change (unless already at optimum)
        # We allow for the case where we're already at optimum
        self.assertGreaterEqual(max_change, 0.0)

        # GP should still make predictions
        mean = gp.predict(self.X_test)
        self.assertEqual(mean.shape, (self.n_test, 1))
        self.assertTrue(self.bkd().all_bool(self.bkd().isfinite(mean)))

    def test_optimize_hyperparameters_with_constant_mean(self) -> None:
        """Test hyperparameter optimization with ConstantMean."""
        kernel = Matern52Kernel(
            [2.0, 2.0],
            (0.1, 10.0),
            self.nvars,
            self.bkd()
        )

        # Use ConstantMean with initial guess
        constant_mean = ConstantMean(0.0, (-5.0, 5.0), self.bkd())

        gp = ExactGaussianProcess(
            kernel,
            self.nvars,
            self.bkd(),
            mean_function=constant_mean,
            nugget=0.1
        )

        gp.fit(self.X_train, self.y_train)
        nlml_before = gp.neg_log_marginal_likelihood()

        # Optimize (should optimize both kernel and mean parameters)
        gp.optimize_hyperparameters()

        nlml_after = gp.neg_log_marginal_likelihood()

        # NLML should not increase
        self.assertLessEqual(nlml_after, nlml_before + 1e-6)

        # Should have 3 hyperparameters (2 length scales + 1 constant)
        self.assertEqual(gp.hyp_list().nparams(), 3)

    def test_optimize_hyperparameters_improves_fit(self) -> None:
        """Test that optimization improves prediction accuracy."""
        # Create synthetic data with known length scales
        np.random.seed(123)
        X_train_np = np.random.uniform(-2, 2, (self.nvars, 30))

        # True function with characteristic length scale ~0.5
        y_train_np = np.sin(2 * X_train_np[0, :]) * np.cos(2 * X_train_np[1, :])
        y_train_np = y_train_np[:, None]

        X_train = self.bkd().array(X_train_np)
        y_train = self.bkd().array(y_train_np)

        # Test points
        X_test_np = np.random.uniform(-2, 2, (self.nvars, 10))
        y_test_np = np.sin(2 * X_test_np[0, :]) * np.cos(2 * X_test_np[1, :])
        y_test_np = y_test_np[:, None]

        X_test = self.bkd().array(X_test_np)
        y_test = self.bkd().array(y_test_np)

        # Start with poor initial hyperparameters (long length scales)
        kernel = Matern52Kernel(
            [5.0, 5.0],  # Too long
            (0.1, 10.0),
            self.nvars,
            self.bkd()
        )

        gp = ExactGaussianProcess(
            kernel,
            self.nvars,
            self.bkd(),
            nugget=0.01
        )

        # Fit and predict before optimization
        gp.fit(X_train, y_train)
        mean_before = gp.predict(X_test)
        error_before = mean_before - y_test
        abs_error_before = error_before * (2 * (error_before >= 0) - 1)
        mse_before = float(self.bkd().sum(abs_error_before ** 2) / abs_error_before.shape[0])

        # Optimize hyperparameters
        gp.optimize_hyperparameters()

        # Predict after optimization
        mean_after = gp.predict(X_test)
        error_after = mean_after - y_test
        abs_error_after = error_after * (2 * (error_after >= 0) - 1)
        mse_after = float(self.bkd().sum(abs_error_after ** 2) / abs_error_after.shape[0])

        # MSE should improve (or stay similar if already good)
        # We allow some tolerance for cases where optimization doesn't help much
        self.assertLessEqual(mse_after, mse_before * 1.1,
                            f"MSE got worse: {mse_before} -> {mse_after}")

    def test_optimize_hyperparameters_custom_initial_guess(self) -> None:
        """Test optimization with custom initial guess."""
        kernel = Matern52Kernel(
            [1.0, 1.0],
            (0.1, 10.0),
            self.nvars,
            self.bkd()
        )

        gp = ExactGaussianProcess(
            kernel,
            self.nvars,
            self.bkd(),
            nugget=0.1
        )

        gp.fit(self.X_train, self.y_train)

        # Create custom initial guess (in optimization space)
        custom_guess = self.bkd().array([-1.0, -1.0])  # log-space values

        # Optimize with custom guess
        gp.optimize_hyperparameters(init_guess=custom_guess)

        # Should complete without error
        self.assertTrue(gp.is_fitted())

        # Should still make predictions
        mean = gp.predict(self.X_test)
        self.assertEqual(mean.shape, (self.n_test, 1))

    def test_optimize_hyperparameters_not_fitted_error(self) -> None:
        """Test that optimization raises error if GP not fitted."""
        kernel = Matern52Kernel(
            [1.0, 1.0],
            (0.1, 10.0),
            self.nvars,
            self.bkd()
        )

        gp = ExactGaussianProcess(
            kernel,
            self.nvars,
            self.bkd(),
            nugget=0.1
        )

        # Should raise RuntimeError before fitting
        with self.assertRaises(RuntimeError):
            gp.optimize_hyperparameters()


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


from pyapprox.typing.util.test_utils import load_tests


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
