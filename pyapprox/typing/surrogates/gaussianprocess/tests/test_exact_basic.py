"""
Tests for basic ExactGaussianProcess functionality.

Tests initialization, fitting, prediction, mean functions, and plotting.
"""
import unittest
from typing import Generic, Any
import numpy as np
import torch
from numpy.typing import NDArray
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from pyapprox.typing.util.backends.protocols import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.surrogates.kernels.matern import MaternKernel
from pyapprox.typing.surrogates.gaussianprocess import (
    ExactGaussianProcess,
    ZeroMean,
    ConstantMean
)
from pyapprox.typing.interface.functions.plot.plot1d import Plotter1D
from pyapprox.typing.interface.functions.plot.plot2d_rectangular import (
    Plotter2DRectangularDomain
)


class TestExactGPBasic(Generic[Array], unittest.TestCase):
    """
    Base test class for basic ExactGaussianProcess functionality.

    Derived classes must implement the bkd() method.
    """

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

        # Create kernel
        self.kernel = MaternKernel(
            2.5,
            [1.0, 1.0],
            (0.1, 10.0),
            self.nvars,
            self.bkd()
        )

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def test_initialization(self) -> None:
        """Test GP initialization."""
        gp = ExactGaussianProcess(
            self.kernel,
            self.nvars,
            self.bkd(),
            noise_variance=0.1
        )

        self.assertEqual(gp.nvars(), self.nvars)
        self.assertFalse(gp.is_fitted())
        self.assertIsInstance(gp.kernel(), MaternKernel)

    def test_fit_and_predict(self) -> None:
        """Test basic fit and predict."""
        gp = ExactGaussianProcess(
            self.kernel,
            self.nvars,
            self.bkd(),
            noise_variance=0.1
        )

        # Fit
        gp.fit(self.X_train, self.y_train)
        self.assertTrue(gp.is_fitted())

        # Predict
        mean = gp.predict(self.X_test)
        self.assertEqual(mean.shape, (self.n_test, 1))

        # All predictions should be finite
        self.assertTrue(self.bkd().all_bool(self.bkd().isfinite(mean)))

    def test_predict_std(self) -> None:
        """Test standard deviation prediction."""
        gp = ExactGaussianProcess(
            self.kernel,
            self.nvars,
            self.bkd(),
            noise_variance=0.1
        )

        gp.fit(self.X_train, self.y_train)

        # Predict standard deviation
        std = gp.predict_std(self.X_test)
        self.assertEqual(std.shape, (self.n_test, 1))

        # All std should be positive
        self.assertTrue(self.bkd().all_bool(std > 0))

    def test_predict_covariance(self) -> None:
        """Test covariance prediction."""
        gp = ExactGaussianProcess(
            self.kernel,
            self.nvars,
            self.bkd(),
            noise_variance=0.1
        )

        gp.fit(self.X_train, self.y_train)

        # Predict covariance
        cov = gp.predict_covariance(self.X_test)
        self.assertEqual(cov.shape, (self.n_test, self.n_test))

        # Covariance should be symmetric
        self.bkd().assert_allclose(cov, cov.T, rtol=1e-6)

    def test_interpolation_low_noise(self) -> None:
        """Test that GP interpolates training data with low noise."""
        gp = ExactGaussianProcess(
            self.kernel,
            self.nvars,
            self.bkd(),
            noise_variance=1e-8  # Very low noise
        )

        gp.fit(self.X_train, self.y_train)

        # Predict at training points
        mean = gp.predict(self.X_train)

        # Should be very close to training data
        self.bkd().assert_allclose(mean, self.y_train, rtol=1e-3, atol=1e-3)

    def test_uncertainty_at_training_points(self) -> None:
        """Test that uncertainty is low at training points."""
        gp = ExactGaussianProcess(
            self.kernel,
            self.nvars,
            self.bkd(),
            noise_variance=0.01
        )

        gp.fit(self.X_train, self.y_train)

        # Predict std at training points
        std = gp.predict_std(self.X_train)

        # Std should be close to noise level (but not exactly due to kernel effects)
        expected_std = self.bkd().full((self.n_train, 1), np.sqrt(0.01))
        self.bkd().assert_allclose(std, expected_std, rtol=0.3, atol=0.03)

    def test_neg_log_marginal_likelihood(self) -> None:
        """Test negative log marginal likelihood computation."""
        gp = ExactGaussianProcess(
            self.kernel,
            self.nvars,
            self.bkd(),
            noise_variance=0.1
        )

        gp.fit(self.X_train, self.y_train)

        # Compute NLML
        nlml = gp.neg_log_marginal_likelihood()

        # Should be a positive finite value
        self.assertTrue(np.isfinite(nlml))
        self.assertGreater(nlml, 0)

    def test_zero_mean_function(self) -> None:
        """Test with zero mean function."""
        zero_mean = ZeroMean(self.bkd())

        gp = ExactGaussianProcess(
            self.kernel,
            self.nvars,
            self.bkd(),
            mean_function=zero_mean,
            noise_variance=0.1
        )

        gp.fit(self.X_train, self.y_train)
        mean = gp.predict(self.X_test)

        self.assertEqual(mean.shape, (self.n_test, 1))

    def test_constant_mean_function(self) -> None:
        """Test with constant mean function."""
        constant_mean = ConstantMean(1.0, (-10.0, 10.0), self.bkd())

        gp = ExactGaussianProcess(
            self.kernel,
            self.nvars,
            self.bkd(),
            mean_function=constant_mean,
            noise_variance=0.1
        )

        gp.fit(self.X_train, self.y_train)
        mean = gp.predict(self.X_test)

        self.assertEqual(mean.shape, (self.n_test, 1))

    def test_call_alias(self) -> None:
        """Test that __call__ returns transposed predict for FunctionProtocol compatibility."""
        gp = ExactGaussianProcess(
            self.kernel,
            self.nvars,
            self.bkd(),
            noise_variance=0.1
        )

        gp.fit(self.X_train, self.y_train)

        mean_predict = gp.predict(self.X_test)  # Shape: (n_test, nqoi)
        mean_call = gp(self.X_test)  # Shape: (nqoi, n_test) for FunctionProtocol

        # __call__ should return transpose of predict
        self.bkd().assert_allclose(mean_predict, mean_call.T)

    def test_error_before_fit(self) -> None:
        """Test that prediction raises error before fit."""
        gp = ExactGaussianProcess(
            self.kernel,
            self.nvars,
            self.bkd(),
            noise_variance=0.1
        )

        with self.assertRaises(RuntimeError):
            gp.predict(self.X_test)

        with self.assertRaises(RuntimeError):
            gp.predict_std(self.X_test)

        with self.assertRaises(RuntimeError):
            gp.predict_covariance(self.X_test)

        with self.assertRaises(RuntimeError):
            gp.neg_log_marginal_likelihood()

    def test_exact_interpolation_minimal_noise(self) -> None:
        """Test that GP exactly interpolates noiseless data with minimal noise variance."""
        # Use zero noise variance (no nugget term) to test exact interpolation
        gp = ExactGaussianProcess(
            self.kernel,
            self.nvars,
            self.bkd(),
            noise_variance=1e-14  # Machine precision level noise
        )

        # Create clean training data (no noise added)
        X_train_clean = self.bkd().array(np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]))
        y_train_clean = self.bkd().array(np.array([[1.0], [2.0], [3.0]]))

        gp.fit(X_train_clean, y_train_clean)

        # Predict at training points - should interpolate exactly
        mean = gp.predict(X_train_clean)

        # When noise is at machine precision level, error should be < 1e-14
        error = mean - y_train_clean
        abs_error = self.bkd().abs(error)
        max_abs_error = float(self.bkd().max(abs_error))
        self.assertLess(max_abs_error, 1e-12,
                       f"Interpolation error {max_abs_error:.2e} exceeds threshold")

        # Uncertainty at training points should be very small
        std = gp.predict_std(X_train_clean)
        self.assertTrue(self.bkd().all_bool(std < 1e-5))

    def test_polynomial_approximation_accuracy(self) -> None:
        """Test that GP can accurately approximate a low-degree polynomial."""
        # Define a simple quadratic polynomial: f(x1, x2) = 1 + x1 + x2 + 0.5*x1^2 + 0.5*x2^2
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
        y_train_np = polynomial(X_train_np)[:, None]

        X_train = self.bkd().array(X_train_np)
        y_train = self.bkd().array(y_train_np)

        # Use Matern kernel with very long length scales to capture smooth polynomial
        kernel = MaternKernel(
            2.5,  # nu
            [5.0, 5.0],  # Very long length scales for smooth approximation
            (0.1, 10.0),
            self.nvars,
            self.bkd()
        )

        gp = ExactGaussianProcess(
            kernel,
            self.nvars,
            self.bkd(),
            noise_variance=1e-10  # Minimal noise
        )

        gp.fit(X_train, y_train)

        # Create test points (unseen locations, but within training domain)
        np.random.seed(123)
        X_test_np = np.random.uniform(-0.9, 0.9, (self.nvars, 15))
        y_test_np = polynomial(X_test_np)[:, None]

        X_test = self.bkd().array(X_test_np)
        y_test = self.bkd().array(y_test_np)

        # Predict at test points
        mean = gp.predict(X_test)

        # Compute approximation error using backend operations only
        error = mean - y_test
        abs_error = self.bkd().abs(error)
        max_abs_error = float(self.bkd().max(abs_error))
        mean_abs_error = float(self.bkd().sum(abs_error) / abs_error.shape[0])

        # With dense training data and appropriate kernel, error should be very small
        # This demonstrates GP's ability to approximate smooth functions
        self.assertLess(max_abs_error, 1e-3,
                       f"Polynomial approximation max error {max_abs_error:.6e} exceeds threshold")

        # Mean error should be even smaller
        self.assertLess(mean_abs_error, 1e-4,
                       f"Polynomial approximation mean error {mean_abs_error:.6e} exceeds threshold")

    def test_plot_1d_gp_mean(self) -> None:
        """Test plotting 1D GP mean function using Plotter1D."""
        # Create 1D GP
        nvars_1d = 1
        kernel_1d = MaternKernel(
            2.5,
            [1.0],
            (0.1, 10.0),
            nvars_1d,
            self.bkd()
        )

        gp = ExactGaussianProcess(
            kernel_1d,
            nvars_1d,
            self.bkd(),
            noise_variance=0.01
        )

        # Create 1D training data
        X_train_1d = self.bkd().array(np.linspace(-2, 2, 10).reshape(1, -1))
        y_train_1d = self.bkd().array(np.sin(X_train_1d[0, :])[:, None])

        gp.fit(X_train_1d, y_train_1d)

        # Create plotter and plot
        plotter = Plotter1D(gp, plot_limits=[-2.5, 2.5])
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Plot GP mean
        plotter.plot(ax, qoi=0, npts_1d=100, label='GP Mean')

        # Plot training data
        ax.plot(
            self.bkd().to_numpy(X_train_1d[0, :]),
            self.bkd().to_numpy(y_train_1d[:, 0]),
            'ro', label='Training Data'
        )

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('1D GP Mean Function')
        ax.legend()
        ax.grid(True)

        plt.close(fig)

        # Test passes if no exceptions raised
        self.assertTrue(True)

    def test_plot_2d_gp_mean(self) -> None:
        """Test plotting 2D GP mean function using Plotter2DRectangularDomain."""
        # Use existing 2D setup
        gp = ExactGaussianProcess(
            self.kernel,
            self.nvars,
            self.bkd(),
            noise_variance=0.01
        )

        gp.fit(self.X_train, self.y_train)

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
            self.bkd().to_numpy(self.X_train[0, :]),
            self.bkd().to_numpy(self.X_train[1, :]),
            'ro', markersize=8, label='Training Data'
        )
        ax.legend()
        plt.close(fig)

        # Test passes if no exceptions raised
        self.assertTrue(True)


# NumPy implementation
class TestExactGPBasicNumpy(TestExactGPBasic[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch implementation
class TestExactGPBasicTorch(TestExactGPBasic[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


def load_tests(loader: unittest.TestLoader, tests, pattern: str) -> unittest.TestSuite:
    """Custom test loader to exclude base class."""
    test_suite = unittest.TestSuite()
    for test_class in [TestExactGPBasicNumpy, TestExactGPBasicTorch]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
