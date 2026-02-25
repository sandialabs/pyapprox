"""
Tests for Gaussian Process loss functions.

This module tests the NegativeLogMarginalLikelihoodLoss class, focusing
on gradient accuracy using DerivativeChecker.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.surrogates.gaussianprocess import (
    ConstantMean,
    ExactGaussianProcess,
    NegativeLogMarginalLikelihoodLoss,
    ZeroMean,
)
from pyapprox.surrogates.kernels.matern import Matern52Kernel
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestNLMLLoss(Generic[Array], unittest.TestCase):
    """
    Base test class for NegativeLogMarginalLikelihoodLoss.

    Derived classes must implement the bkd() method.
    """

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment."""
        np.random.seed(42)
        self.nvars = 2
        self.n_train = 10

        # Create training data
        X_train_np = np.random.randn(self.nvars, self.n_train)
        y_train_np = np.sin(X_train_np[0, :] + X_train_np[1, :])[
            None, :
        ]  # Shape: (1, n_train)

        self.X_train = self.bkd().array(X_train_np)
        self.y_train = self.bkd().array(y_train_np)

        # Create kernel with optimizable hyperparameters
        self.kernel = Matern52Kernel([1.0, 1.0], (0.1, 10.0), self.nvars, self.bkd())

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def test_initialization(self) -> None:
        """Test loss initialization."""
        gp = ExactGaussianProcess(self.kernel, self.nvars, self.bkd(), nugget=0.1)

        loss = NegativeLogMarginalLikelihoodLoss(gp, self.X_train, self.y_train)

        # Check nvars corresponds to number of hyperparameters
        self.assertGreater(loss.nvars(), 0)
        self.assertEqual(loss.nqoi(), 1)

    def test_loss_evaluation(self) -> None:
        """Test basic loss evaluation."""
        gp = ExactGaussianProcess(self.kernel, self.nvars, self.bkd(), nugget=0.1)

        loss = NegativeLogMarginalLikelihoodLoss(gp, self.X_train, self.y_train)

        # Get current hyperparameters
        params = gp.hyp_list().get_active_values()

        # Evaluate loss
        nll = loss(params)

        # Check output shape and properties
        self.assertEqual(nll.shape, (1, 1))
        self.assertTrue(self.bkd().all_bool(self.bkd().isfinite(nll)))

        # NLL should be positive for typical data
        self.assertGreater(float(nll[0, 0]), 0)

    def test_loss_changes_with_hyperparameters(self) -> None:
        """Test that loss changes when hyperparameters change."""
        gp = ExactGaussianProcess(self.kernel, self.nvars, self.bkd(), nugget=0.1)

        loss = NegativeLogMarginalLikelihoodLoss(gp, self.X_train, self.y_train)

        # Get current hyperparameters
        params1 = gp.hyp_list().get_active_values()
        nll1 = loss(params1)

        # Perturb hyperparameters
        params2 = params1 + self.bkd().array([0.1] * len(params1))
        nll2 = loss(params2)

        # Loss should be different
        self.assertNotEqual(float(nll1[0, 0]), float(nll2[0, 0]))

    def test_jacobian_shape(self) -> None:
        """Test Jacobian output shape."""
        gp = ExactGaussianProcess(self.kernel, self.nvars, self.bkd(), nugget=0.1)

        loss = NegativeLogMarginalLikelihoodLoss(gp, self.X_train, self.y_train)

        params = gp.hyp_list().get_active_values()
        grad = loss.jacobian(params)

        # Check gradient shape: (1, nactive)
        self.assertEqual(grad.shape, (1, loss.nvars()))
        self.assertTrue(self.bkd().all_bool(self.bkd().isfinite(grad)))

    def test_gradient_with_zero_mean(self) -> None:
        """Test gradient computation with ZeroMean function."""
        gp = ExactGaussianProcess(
            self.kernel,
            self.nvars,
            self.bkd(),
            mean_function=ZeroMean(self.bkd()),
            nugget=0.1,
        )

        loss = NegativeLogMarginalLikelihoodLoss(gp, self.X_train, self.y_train)

        # Create derivative checker
        checker = DerivativeChecker(loss)

        # Get current hyperparameters (in optimization space)
        params = gp.hyp_list().get_active_values()

        # Use logarithmically-spaced step sizes from 1 down to 1e-14
        fd_eps = self.bkd().flip(self.bkd().logspace(-14, 0, 15))

        # Check gradient accuracy
        errors = checker.check_derivatives(
            params[:, None],  # Shape: (nactive, 1)
            fd_eps=fd_eps,
            relative=True,
            verbosity=0,
        )

        # Get gradient error (first element of errors list)
        grad_error = errors[0]

        # All errors should be finite
        self.assertTrue(
            self.bkd().all_bool(self.bkd().isfinite(grad_error)),
            "Gradient errors contain non-finite values",
        )

        # Minimum error should be small
        min_error = float(self.bkd().min(grad_error))
        self.assertLess(
            min_error,
            1e-6,
            f"Minimum gradient relative error {min_error} exceeds threshold",
        )

        # Error ratio should indicate good convergence
        error_ratio = float(checker.error_ratio(grad_error))
        self.assertLess(
            error_ratio,
            1e-6,
            f"Error ratio {error_ratio:.2e} suggests poor convergence",
        )

    def test_gradient_with_constant_mean(self) -> None:
        """Test gradient computation with ConstantMean function."""
        # ConstantMean adds one more hyperparameter to optimize
        constant_mean = ConstantMean(0.5, (-10.0, 10.0), self.bkd())

        gp = ExactGaussianProcess(
            self.kernel, self.nvars, self.bkd(), mean_function=constant_mean, nugget=0.1
        )

        loss = NegativeLogMarginalLikelihoodLoss(gp, self.X_train, self.y_train)

        # Now we have kernel parameters + constant parameter
        self.assertGreater(loss.nvars(), self.nvars)

        # Create derivative checker
        checker = DerivativeChecker(loss)

        # Get current hyperparameters
        params = gp.hyp_list().get_active_values()

        # Use logarithmically-spaced step sizes
        fd_eps = self.bkd().flip(self.bkd().logspace(-14, 0, 15))

        # Check gradient accuracy
        errors = checker.check_derivatives(
            params[:, None], fd_eps=fd_eps, relative=True, verbosity=0
        )

        grad_error = errors[0]

        # All errors should be finite
        self.assertTrue(
            self.bkd().all_bool(self.bkd().isfinite(grad_error)),
            "Gradient errors contain non-finite values",
        )

        # Minimum error should be small
        min_error = float(self.bkd().min(grad_error))
        self.assertLess(
            min_error,
            1e-6,
            f"Minimum gradient relative error {min_error} exceeds threshold",
        )

        # Error ratio should indicate good convergence
        error_ratio = float(checker.error_ratio(grad_error))
        self.assertLess(
            error_ratio,
            1e-6,
            f"Error ratio {error_ratio:.2e} suggests poor convergence",
        )

    def test_gradient_different_noise_levels(self) -> None:
        """Test gradient with different noise variance levels."""
        for noise_var in [1e-6, 0.01, 0.1, 1.0]:
            with self.subTest(nugget=noise_var):
                gp = ExactGaussianProcess(
                    self.kernel, self.nvars, self.bkd(), nugget=noise_var
                )

                loss = NegativeLogMarginalLikelihoodLoss(gp, self.X_train, self.y_train)

                params = gp.hyp_list().get_active_values()
                grad = loss.jacobian(params)

                # Gradient should be finite
                self.assertTrue(self.bkd().all_bool(self.bkd().isfinite(grad)))

    def test_gradient_with_small_dataset(self) -> None:
        """Test gradient with small dataset (n=3 points)."""
        # Create small dataset
        X_small = self.X_train[:, :3]
        y_small = self.y_train[:, :3]  # Shape: (1, 3)

        gp = ExactGaussianProcess(self.kernel, self.nvars, self.bkd(), nugget=0.1)

        loss = NegativeLogMarginalLikelihoodLoss(gp, X_small, y_small)

        # Create derivative checker
        checker = DerivativeChecker(loss)
        params = gp.hyp_list().get_active_values()

        # Use logarithmically-spaced step sizes
        fd_eps = self.bkd().flip(self.bkd().logspace(-14, 0, 15))

        errors = checker.check_derivatives(
            params[:, None], fd_eps=fd_eps, relative=True, verbosity=0
        )

        grad_error = errors[0]

        # Should still have good gradient accuracy
        min_error = float(self.bkd().min(grad_error))
        self.assertLess(min_error, 1e-5)

        error_ratio = float(checker.error_ratio(grad_error))
        self.assertLess(error_ratio, 1e-6)

    def test_gradient_with_larger_dataset(self) -> None:
        """Test gradient with larger dataset (n=50 points)."""
        np.random.seed(123)
        X_large_np = np.random.randn(self.nvars, 50)
        X_large = self.bkd().array(X_large_np)
        y_large = self.bkd().array(
            np.sin(X_large_np[0, :] + X_large_np[1, :])[None, :]  # Shape: (1, 50)
        )

        gp = ExactGaussianProcess(self.kernel, self.nvars, self.bkd(), nugget=0.1)

        loss = NegativeLogMarginalLikelihoodLoss(gp, X_large, y_large)

        # Create derivative checker
        checker = DerivativeChecker(loss)
        params = gp.hyp_list().get_active_values()

        # Use logarithmically-spaced step sizes
        fd_eps = self.bkd().flip(self.bkd().logspace(-14, 0, 15))

        errors = checker.check_derivatives(
            params[:, None], fd_eps=fd_eps, relative=True, verbosity=0
        )

        grad_error = errors[0]

        # Should still have good gradient accuracy
        min_error = float(self.bkd().min(grad_error))
        self.assertLess(min_error, 1e-5)

        error_ratio = float(checker.error_ratio(grad_error))
        self.assertLess(error_ratio, 1e-6)

    def test_repr(self) -> None:
        """Test string representation."""
        gp = ExactGaussianProcess(self.kernel, self.nvars, self.bkd(), nugget=0.1)

        loss = NegativeLogMarginalLikelihoodLoss(gp, self.X_train, self.y_train)

        repr_str = repr(loss)
        self.assertIn("NegativeLogMarginalLikelihoodLoss", repr_str)
        self.assertIn("nvars", repr_str)
        self.assertIn("n_train", repr_str)


# NumPy implementation
class TestNLMLLossNumpy(TestNLMLLoss[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch implementation
class TestNLMLLossTorch(TestNLMLLoss[torch.Tensor]):
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
