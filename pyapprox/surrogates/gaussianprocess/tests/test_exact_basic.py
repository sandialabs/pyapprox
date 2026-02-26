"""
Tests for basic ExactGaussianProcess functionality.

Tests initialization, fitting, prediction, mean functions, and plotting.
"""
import math

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from pyapprox.interface.functions.plot.plot1d import Plotter1D
from pyapprox.interface.functions.plot.plot2d_rectangular import (
    Plotter2DRectangularDomain,
)
from pyapprox.surrogates.gaussianprocess import (
    ConstantMean,
    ExactGaussianProcess,
    ZeroMean,
)
from pyapprox.surrogates.kernels.matern import (
    Matern52Kernel,
    MaternKernel,
)


class TestExactGPBasic:
    """
    Test class for basic ExactGaussianProcess functionality.
    """

    def test_initialization(self, bkd) -> None:
        """Test GP initialization."""
        np.random.seed(42)
        nvars = 2
        kernel = Matern52Kernel(
            [1.0, 1.0],
            (0.1, 10.0),
            nvars,
            bkd
        )

        gp = ExactGaussianProcess(
            kernel,
            nvars,
            bkd,
            nugget=0.1
        )

        assert gp.nvars() == nvars
        assert not gp.is_fitted()
        assert isinstance(gp.kernel(), MaternKernel)

    def test_fit_and_predict(self, bkd) -> None:
        """Test basic fit and predict."""
        np.random.seed(42)
        nvars = 2
        n_train = 10
        n_test = 5

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(
            X_train_np[0, :] + X_train_np[1, :]
        )[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        X_test_np = np.random.randn(nvars, n_test)
        X_test = bkd.array(X_test_np)

        kernel = Matern52Kernel(
            [1.0, 1.0],
            (0.1, 10.0),
            nvars,
            bkd
        )

        gp = ExactGaussianProcess(
            kernel,
            nvars,
            bkd,
            nugget=0.1
        )

        # Fit
        gp.fit(X_train, y_train)
        assert gp.is_fitted()

        # Predict
        mean = gp.predict(X_test)
        assert mean.shape == (1, n_test)

        # All predictions should be finite
        assert bkd.all_bool(bkd.isfinite(mean))

    def test_predict_std(self, bkd) -> None:
        """Test standard deviation prediction."""
        np.random.seed(42)
        nvars = 2
        n_train = 10
        n_test = 5

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(
            X_train_np[0, :] + X_train_np[1, :]
        )[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        X_test_np = np.random.randn(nvars, n_test)
        X_test = bkd.array(X_test_np)

        kernel = Matern52Kernel(
            [1.0, 1.0],
            (0.1, 10.0),
            nvars,
            bkd
        )

        gp = ExactGaussianProcess(
            kernel,
            nvars,
            bkd,
            nugget=0.1
        )

        gp.fit(X_train, y_train)

        # Predict standard deviation
        std = gp.predict_std(X_test)
        assert std.shape == (1, n_test)

        # All std should be positive
        assert bkd.all_bool(std > 0)

    def test_predict_covariance(self, bkd) -> None:
        """Test covariance prediction."""
        np.random.seed(42)
        nvars = 2
        n_train = 10
        n_test = 5

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(
            X_train_np[0, :] + X_train_np[1, :]
        )[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        X_test_np = np.random.randn(nvars, n_test)
        X_test = bkd.array(X_test_np)

        kernel = Matern52Kernel(
            [1.0, 1.0],
            (0.1, 10.0),
            nvars,
            bkd
        )

        gp = ExactGaussianProcess(
            kernel,
            nvars,
            bkd,
            nugget=0.1
        )

        gp.fit(X_train, y_train)

        # Predict covariance
        cov = gp.predict_covariance(X_test)
        assert cov.shape == (n_test, n_test)

        # Covariance should be symmetric
        bkd.assert_allclose(cov, cov.T, rtol=1e-6)

    def test_interpolation_low_noise(self, bkd) -> None:
        """Test that GP interpolates training data with low noise."""
        np.random.seed(42)
        nvars = 2
        n_train = 10

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(
            X_train_np[0, :] + X_train_np[1, :]
        )[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        kernel = Matern52Kernel(
            [1.0, 1.0],
            (0.1, 10.0),
            nvars,
            bkd
        )

        gp = ExactGaussianProcess(
            kernel,
            nvars,
            bkd,
            nugget=1e-8  # Very low noise
        )

        gp.fit(X_train, y_train)

        # Predict at training points
        mean = gp.predict(X_train)

        # Should be very close to training data
        bkd.assert_allclose(mean, y_train, rtol=1e-3, atol=1e-3)

    def test_uncertainty_at_training_points(self, bkd) -> None:
        """Test that uncertainty is low at training points."""
        np.random.seed(42)
        nvars = 2
        n_train = 10

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(
            X_train_np[0, :] + X_train_np[1, :]
        )[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        kernel = Matern52Kernel(
            [1.0, 1.0],
            (0.1, 10.0),
            nvars,
            bkd
        )

        gp = ExactGaussianProcess(
            kernel,
            nvars,
            bkd,
            nugget=0.01
        )

        gp.fit(X_train, y_train)

        # Predict std at training points
        std = gp.predict_std(X_train)

        # Std should be close to noise level (but not exactly due to kernel effects)
        expected_std = bkd.full((1, n_train), math.sqrt(0.01))
        bkd.assert_allclose(std, expected_std, rtol=0.3, atol=0.03)

    def test_neg_log_marginal_likelihood(self, bkd) -> None:
        """Test negative log marginal likelihood computation."""
        np.random.seed(42)
        nvars = 2
        n_train = 10

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(
            X_train_np[0, :] + X_train_np[1, :]
        )[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        kernel = Matern52Kernel(
            [1.0, 1.0],
            (0.1, 10.0),
            nvars,
            bkd
        )

        gp = ExactGaussianProcess(
            kernel,
            nvars,
            bkd,
            nugget=0.1
        )

        gp.fit(X_train, y_train)

        # Compute NLML
        nlml = gp.neg_log_marginal_likelihood()

        # Should be a positive finite value
        assert math.isfinite(nlml)
        assert nlml > 0

    def test_zero_mean_function(self, bkd) -> None:
        """Test with zero mean function."""
        np.random.seed(42)
        nvars = 2
        n_train = 10
        n_test = 5

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(
            X_train_np[0, :] + X_train_np[1, :]
        )[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        X_test_np = np.random.randn(nvars, n_test)
        X_test = bkd.array(X_test_np)

        kernel = Matern52Kernel(
            [1.0, 1.0],
            (0.1, 10.0),
            nvars,
            bkd
        )

        zero_mean = ZeroMean(bkd)

        gp = ExactGaussianProcess(
            kernel,
            nvars,
            bkd,
            mean_function=zero_mean,
            nugget=0.1
        )

        gp.fit(X_train, y_train)
        mean = gp.predict(X_test)

        assert mean.shape == (1, n_test)

    def test_constant_mean_function(self, bkd) -> None:
        """Test with constant mean function."""
        np.random.seed(42)
        nvars = 2
        n_train = 10
        n_test = 5

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(
            X_train_np[0, :] + X_train_np[1, :]
        )[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        X_test_np = np.random.randn(nvars, n_test)
        X_test = bkd.array(X_test_np)

        kernel = Matern52Kernel(
            [1.0, 1.0],
            (0.1, 10.0),
            nvars,
            bkd
        )

        constant_mean = ConstantMean(1.0, (-10.0, 10.0), bkd)

        gp = ExactGaussianProcess(
            kernel,
            nvars,
            bkd,
            mean_function=constant_mean,
            nugget=0.1
        )

        gp.fit(X_train, y_train)
        mean = gp.predict(X_test)

        assert mean.shape == (1, n_test)

    def test_call_alias(self, bkd) -> None:
        """Test that __call__ is alias for predict (both return (nqoi, n_test))."""
        np.random.seed(42)
        nvars = 2
        n_train = 10
        n_test = 5

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(
            X_train_np[0, :] + X_train_np[1, :]
        )[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        X_test_np = np.random.randn(nvars, n_test)
        X_test = bkd.array(X_test_np)

        kernel = Matern52Kernel(
            [1.0, 1.0],
            (0.1, 10.0),
            nvars,
            bkd
        )

        gp = ExactGaussianProcess(
            kernel,
            nvars,
            bkd,
            nugget=0.1
        )

        gp.fit(X_train, y_train)

        mean_predict = gp.predict(X_test)  # Shape: (nqoi, n_test)
        mean_call = gp(X_test)  # Shape: (nqoi, n_test) for FunctionProtocol

        # __call__ should return same as predict
        bkd.assert_allclose(mean_predict, mean_call)

    def test_error_before_fit(self, bkd) -> None:
        """Test that prediction raises error before fit."""
        np.random.seed(42)
        nvars = 2
        n_test = 5

        X_test_np = np.random.randn(nvars, n_test)
        X_test = bkd.array(X_test_np)

        kernel = Matern52Kernel(
            [1.0, 1.0],
            (0.1, 10.0),
            nvars,
            bkd
        )

        gp = ExactGaussianProcess(
            kernel,
            nvars,
            bkd,
            nugget=0.1
        )

        with pytest.raises(RuntimeError):
            gp.predict(X_test)

        with pytest.raises(RuntimeError):
            gp.predict_std(X_test)

        with pytest.raises(RuntimeError):
            gp.predict_covariance(X_test)

        with pytest.raises(RuntimeError):
            gp.neg_log_marginal_likelihood()

    def test_exact_interpolation_minimal_noise(self, bkd) -> None:
        """Test GP exactly interpolates noiseless data."""
        nvars = 2
        kernel = Matern52Kernel(
            [1.0, 1.0],
            (0.1, 10.0),
            nvars,
            bkd
        )

        # Use zero noise variance (no nugget term) to test exact interpolation
        gp = ExactGaussianProcess(
            kernel,
            nvars,
            bkd,
            nugget=1e-14  # Machine precision level noise
        )

        # Create clean training data (no noise added)
        X_train_clean = bkd.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        y_train_clean = bkd.array([[1.0, 2.0, 3.0]])  # Shape: (1, 3)

        gp.fit(X_train_clean, y_train_clean)

        # Predict at training points - should interpolate exactly
        mean = gp.predict(X_train_clean)

        # When noise is at machine precision level, error should be < 1e-14
        error = mean - y_train_clean
        abs_error = bkd.abs(error)
        max_abs_error = float(bkd.max(abs_error))
        assert max_abs_error < 1e-12, \
            f"Interpolation error {max_abs_error:.2e} exceeds threshold"

        # Uncertainty at training points should be very small
        std = gp.predict_std(X_train_clean)
        assert bkd.all_bool(std < 1e-5)

    def test_polynomial_approximation_accuracy(self, bkd) -> None:
        """Test that GP can accurately approximate a low-degree polynomial."""
        nvars = 2

        # f(x1, x2) = 1 + x1 + x2 + 0.5*x1^2 + 0.5*x2^2
        def polynomial(X: np.ndarray) -> np.ndarray:
            """Evaluate polynomial at input locations."""
            x1 = X[0, :]
            x2 = X[1, :]
            return 1.0 + x1 + x2 + 0.5 * x1**2 + 0.5 * x2**2

        # Create very dense training data in [-1, 1]^2 for accurate approximation
        n_train_1d = 9  # 9x9 = 81 training points
        x1_train = np.linspace(-1, 1, n_train_1d)
        x2_train = np.linspace(-1, 1, n_train_1d)
        X1_train, X2_train = np.meshgrid(x1_train, x2_train)
        X_train_np = np.vstack([X1_train.ravel(), X2_train.ravel()])
        y_train_np = polynomial(X_train_np)[None, :]  # Shape: (1, n_train)

        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        # Use Matern kernel with very long length scales to capture smooth polynomial
        kernel = Matern52Kernel(
            [5.0, 5.0],  # Very long length scales for smooth approximation
            (0.1, 10.0),
            nvars,
            bkd
        )

        gp = ExactGaussianProcess(
            kernel,
            nvars,
            bkd,
            nugget=1e-10  # Minimal noise
        )

        gp.fit(X_train, y_train)

        # Create test points (unseen locations, but within training domain)
        np.random.seed(123)
        X_test_np = np.random.uniform(-0.9, 0.9, (nvars, 15))
        y_test_np = polynomial(X_test_np)[None, :]  # Shape: (1, n_test)

        X_test = bkd.array(X_test_np)
        y_test = bkd.array(y_test_np)

        # Predict at test points
        mean = gp.predict(X_test)

        # Compute approximation error using backend operations only
        error = mean - y_test
        abs_error = bkd.abs(error)
        max_abs_error = float(bkd.max(abs_error))
        mean_abs_error = float(bkd.sum(abs_error) / abs_error.shape[1])

        # With dense training data and appropriate kernel, error should be very small
        # This demonstrates GP's ability to approximate smooth functions
        assert max_abs_error < 1e-3, \
            f"Poly approx max error " \
            f"{max_abs_error:.6e} exceeds threshold"

        # Mean error should be even smaller
        assert mean_abs_error < 1e-4, \
            f"Poly approx mean error " \
            f"{mean_abs_error:.6e} exceeds threshold"

    def test_plot_1d_gp_mean(self, bkd) -> None:
        """Test plotting 1D GP mean function using Plotter1D."""
        # Create 1D GP
        nvars_1d = 1
        kernel_1d = Matern52Kernel(
            [1.0],
            (0.1, 10.0),
            nvars_1d,
            bkd
        )

        gp = ExactGaussianProcess(
            kernel_1d,
            nvars_1d,
            bkd,
            nugget=0.01
        )

        # Create 1D training data
        X_train_1d = bkd.reshape(bkd.linspace(-2, 2, 10), (1, -1))
        y_train_1d = bkd.reshape(
            bkd.sin(X_train_1d[0, :]), (1, -1)
        )

        gp.fit(X_train_1d, y_train_1d)

        # Create plotter and plot
        plotter = Plotter1D(gp, plot_limits=[-2.5, 2.5])
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Plot GP mean
        plotter.plot(ax, qoi=0, npts_1d=100, label='GP Mean')

        # Plot training data
        ax.plot(
            bkd.to_numpy(X_train_1d[0, :]),
            bkd.to_numpy(y_train_1d[0, :]),
            'ro', label='Training Data'
        )

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('1D GP Mean Function')
        ax.legend()
        ax.grid(True)

        plt.close(fig)

        # Test passes if no exceptions raised
        assert True

    def test_plot_2d_gp_mean(self, bkd) -> None:
        """Test plotting 2D GP mean function using Plotter2DRectangularDomain."""
        np.random.seed(42)
        nvars = 2
        n_train = 10

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(
            X_train_np[0, :] + X_train_np[1, :]
        )[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        kernel = Matern52Kernel(
            [1.0, 1.0],
            (0.1, 10.0),
            nvars,
            bkd
        )

        # Use existing 2D setup
        gp = ExactGaussianProcess(
            kernel,
            nvars,
            bkd,
            nugget=0.01
        )

        gp.fit(X_train, y_train)

        # Create plotter and plot
        plotter = Plotter2DRectangularDomain(
            gp,
            plot_limits=[-2, 2, -2, 2]
        )

        # Test surface plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        plotter.plot(ax, qoi=0, npts_1d=30)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('GP Mean')
        ax.set_title('2D GP Mean Surface')
        plt.close(fig)

        # Test contour plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        plotter.plot_contours(ax, qoi=0, npts_1d=30, levels=15)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('2D GP Mean Contours')

        # Add training points to contour plot
        ax.plot(
            bkd.to_numpy(X_train[0, :]),
            bkd.to_numpy(X_train[1, :]),
            'ro', markersize=8, label='Training Data'
        )
        ax.legend()
        plt.close(fig)

        # Test passes if no exceptions raised
        assert True
