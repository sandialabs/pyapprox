"""
Tests for Gaussian Process loss functions.

This module tests the NegativeLogMarginalLikelihoodLoss class, focusing
on gradient accuracy using DerivativeChecker.
"""

import numpy as np

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


class TestNLMLLoss:
    """
    Test class for NegativeLogMarginalLikelihoodLoss.
    """

    def test_initialization(self, bkd) -> None:
        """Test loss initialization."""
        np.random.seed(42)
        nvars = 2
        n_train = 10

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(X_train_np[0, :] + X_train_np[1, :])[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        kernel = Matern52Kernel([1.0, 1.0], (0.1, 10.0), nvars, bkd)

        gp = ExactGaussianProcess(kernel, nvars, bkd, nugget=0.1)

        loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)

        # Check nvars corresponds to number of hyperparameters
        assert loss.nvars() > 0
        assert loss.nqoi() == 1

    def test_loss_evaluation(self, bkd) -> None:
        """Test basic loss evaluation."""
        np.random.seed(42)
        nvars = 2
        n_train = 10

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(X_train_np[0, :] + X_train_np[1, :])[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        kernel = Matern52Kernel([1.0, 1.0], (0.1, 10.0), nvars, bkd)

        gp = ExactGaussianProcess(kernel, nvars, bkd, nugget=0.1)

        loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)

        # Get current hyperparameters
        params = gp.hyp_list().get_active_values()

        # Evaluate loss
        nll = loss(params)

        # Check output shape and properties
        assert nll.shape == (1, 1)
        assert bkd.all_bool(bkd.isfinite(nll))

        # NLL should be positive for typical data
        assert float(nll[0, 0]) > 0

    def test_loss_changes_with_hyperparameters(self, bkd) -> None:
        """Test that loss changes when hyperparameters change."""
        np.random.seed(42)
        nvars = 2
        n_train = 10

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(X_train_np[0, :] + X_train_np[1, :])[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        kernel = Matern52Kernel([1.0, 1.0], (0.1, 10.0), nvars, bkd)

        gp = ExactGaussianProcess(kernel, nvars, bkd, nugget=0.1)

        loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)

        # Get current hyperparameters
        params1 = gp.hyp_list().get_active_values()
        nll1 = loss(params1)

        # Perturb hyperparameters
        params2 = params1 + bkd.array([0.1] * len(params1))
        nll2 = loss(params2)

        # Loss should be different
        assert float(nll1[0, 0]) != float(nll2[0, 0])

    def test_jacobian_shape(self, bkd) -> None:
        """Test Jacobian output shape."""
        np.random.seed(42)
        nvars = 2
        n_train = 10

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(X_train_np[0, :] + X_train_np[1, :])[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        kernel = Matern52Kernel([1.0, 1.0], (0.1, 10.0), nvars, bkd)

        gp = ExactGaussianProcess(kernel, nvars, bkd, nugget=0.1)

        loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)

        params = gp.hyp_list().get_active_values()
        grad = loss.jacobian(params)

        # Check gradient shape: (1, nactive)
        assert grad.shape == (1, loss.nvars())
        assert bkd.all_bool(bkd.isfinite(grad))

    def test_gradient_with_zero_mean(self, bkd) -> None:
        """Test gradient computation with ZeroMean function."""
        np.random.seed(42)
        nvars = 2
        n_train = 10

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(X_train_np[0, :] + X_train_np[1, :])[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        kernel = Matern52Kernel([1.0, 1.0], (0.1, 10.0), nvars, bkd)

        gp = ExactGaussianProcess(
            kernel,
            nvars,
            bkd,
            mean_function=ZeroMean(bkd),
            nugget=0.1,
        )

        loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)

        # Create derivative checker
        checker = DerivativeChecker(loss)

        # Get current hyperparameters (in optimization space)
        params = gp.hyp_list().get_active_values()

        # Use logarithmically-spaced step sizes from 1 down to 1e-14
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

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
        assert bkd.all_bool(bkd.isfinite(grad_error)), \
            "Gradient errors contain non-finite values"

        # Minimum error should be small
        min_error = float(bkd.min(grad_error))
        assert min_error < 1e-6, \
            f"Minimum gradient relative error {min_error} exceeds threshold"

        # Error ratio should indicate good convergence
        error_ratio = float(checker.error_ratio(grad_error))
        assert error_ratio < 1e-6, \
            f"Error ratio {error_ratio:.2e} suggests poor convergence"

    def test_gradient_with_constant_mean(self, bkd) -> None:
        """Test gradient computation with ConstantMean function."""
        np.random.seed(42)
        nvars = 2
        n_train = 10

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(X_train_np[0, :] + X_train_np[1, :])[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        kernel = Matern52Kernel([1.0, 1.0], (0.1, 10.0), nvars, bkd)

        # ConstantMean adds one more hyperparameter to optimize
        constant_mean = ConstantMean(0.5, (-10.0, 10.0), bkd)

        gp = ExactGaussianProcess(
            kernel, nvars, bkd, mean_function=constant_mean, nugget=0.1
        )

        loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)

        # Now we have kernel parameters + constant parameter
        assert loss.nvars() > nvars

        # Create derivative checker
        checker = DerivativeChecker(loss)

        # Get current hyperparameters
        params = gp.hyp_list().get_active_values()

        # Use logarithmically-spaced step sizes
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        # Check gradient accuracy
        errors = checker.check_derivatives(
            params[:, None], fd_eps=fd_eps, relative=True, verbosity=0
        )

        grad_error = errors[0]

        # All errors should be finite
        assert bkd.all_bool(bkd.isfinite(grad_error)), \
            "Gradient errors contain non-finite values"

        # Minimum error should be small
        min_error = float(bkd.min(grad_error))
        assert min_error < 1e-6, \
            f"Minimum gradient relative error {min_error} exceeds threshold"

        # Error ratio should indicate good convergence
        error_ratio = float(checker.error_ratio(grad_error))
        assert error_ratio < 1e-6, \
            f"Error ratio {error_ratio:.2e} suggests poor convergence"

    def test_gradient_different_noise_levels(self, bkd) -> None:
        """Test gradient with different noise variance levels."""
        np.random.seed(42)
        nvars = 2
        n_train = 10

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(X_train_np[0, :] + X_train_np[1, :])[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        kernel = Matern52Kernel([1.0, 1.0], (0.1, 10.0), nvars, bkd)

        for noise_var in [1e-6, 0.01, 0.1, 1.0]:
            gp = ExactGaussianProcess(
                kernel, nvars, bkd, nugget=noise_var
            )

            loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)

            params = gp.hyp_list().get_active_values()
            grad = loss.jacobian(params)

            # Gradient should be finite
            assert bkd.all_bool(bkd.isfinite(grad))

    def test_gradient_with_small_dataset(self, bkd) -> None:
        """Test gradient with small dataset (n=3 points)."""
        np.random.seed(42)
        nvars = 2
        n_train = 10

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(X_train_np[0, :] + X_train_np[1, :])[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        kernel = Matern52Kernel([1.0, 1.0], (0.1, 10.0), nvars, bkd)

        # Create small dataset
        X_small = X_train[:, :3]
        y_small = y_train[:, :3]  # Shape: (1, 3)

        gp = ExactGaussianProcess(kernel, nvars, bkd, nugget=0.1)

        loss = NegativeLogMarginalLikelihoodLoss(gp, X_small, y_small)

        # Create derivative checker
        checker = DerivativeChecker(loss)
        params = gp.hyp_list().get_active_values()

        # Use logarithmically-spaced step sizes
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        errors = checker.check_derivatives(
            params[:, None], fd_eps=fd_eps, relative=True, verbosity=0
        )

        grad_error = errors[0]

        # Should still have good gradient accuracy
        min_error = float(bkd.min(grad_error))
        assert min_error < 1e-5

        error_ratio = float(checker.error_ratio(grad_error))
        assert error_ratio < 1e-6

    def test_gradient_with_larger_dataset(self, bkd) -> None:
        """Test gradient with larger dataset (n=50 points)."""
        np.random.seed(123)
        nvars = 2

        X_large_np = np.random.randn(nvars, 50)
        X_large = bkd.array(X_large_np)
        y_large = bkd.array(
            np.sin(X_large_np[0, :] + X_large_np[1, :])[None, :]  # Shape: (1, 50)
        )

        kernel = Matern52Kernel([1.0, 1.0], (0.1, 10.0), nvars, bkd)

        gp = ExactGaussianProcess(kernel, nvars, bkd, nugget=0.1)

        loss = NegativeLogMarginalLikelihoodLoss(gp, X_large, y_large)

        # Create derivative checker
        checker = DerivativeChecker(loss)
        params = gp.hyp_list().get_active_values()

        # Use logarithmically-spaced step sizes
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        errors = checker.check_derivatives(
            params[:, None], fd_eps=fd_eps, relative=True, verbosity=0
        )

        grad_error = errors[0]

        # Should still have good gradient accuracy
        min_error = float(bkd.min(grad_error))
        assert min_error < 1e-5

        error_ratio = float(checker.error_ratio(grad_error))
        assert error_ratio < 1e-6

    def test_repr(self, bkd) -> None:
        """Test string representation."""
        np.random.seed(42)
        nvars = 2
        n_train = 10

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(X_train_np[0, :] + X_train_np[1, :])[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        kernel = Matern52Kernel([1.0, 1.0], (0.1, 10.0), nvars, bkd)

        gp = ExactGaussianProcess(kernel, nvars, bkd, nugget=0.1)

        loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)

        repr_str = repr(loss)
        assert "NegativeLogMarginalLikelihoodLoss" in repr_str
        assert "nvars" in repr_str
        assert "n_train" in repr_str
