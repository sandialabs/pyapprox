"""
Tests for Gaussian Process with multi-output kernels.

This module tests multi-output GP prediction using IndependentMultiOutputKernel
and LinearCoregionalizationKernel with a single GP handling all outputs.
"""

import unittest
from typing import Generic, Any

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.surrogates.kernels.matern import MaternKernel
from pyapprox.typing.surrogates.kernels.scalings import PolynomialScaling
from pyapprox.typing.surrogates.kernels.iid_gaussian_noise import IIDGaussianNoise
from pyapprox.typing.surrogates.kernels.multioutput import (
    IndependentMultiOutputKernel,
    LinearCoregionalizationKernel,
)
from pyapprox.typing.surrogates.gaussianprocess.multioutput import MultiOutputGP


class TestMultiOutputGPWithIndependentKernel(Generic[Array], unittest.TestCase):
    """
    Test multi-output GP using IndependentMultiOutputKernel.

    Tests a single GP that uses a multi-output kernel to handle multiple
    outputs simultaneously, achieving < 1e-6 interpolation accuracy.
    """

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        """Set up test environment."""
        np.random.seed(42)
        self.nvars = 2
        self.noutputs = 2

    def _create_test_data(self, n_train: int):
        """Create synthetic multi-output test data."""
        # Create training inputs (same for all outputs)
        X_train_np = np.random.randn(self.nvars, n_train)
        X_train = self.bkd().array(X_train_np)

        # Create multi-output training data
        # Output 1: sin(x1 + x2)
        # Output 2: cos(x1 - x2)
        y1 = np.sin(X_train_np[0, :] + X_train_np[1, :])
        y2 = np.cos(X_train_np[0, :] - X_train_np[1, :])

        # Stack outputs: shape (n_train * noutputs, 1)
        y_train_stacked_np = np.concatenate([y1, y2])[:, np.newaxis]
        y_train_stacked = self.bkd().array(y_train_stacked_np)

        # Also return individual outputs for verification
        y_train_np = np.column_stack([y1, y2])
        y_train = self.bkd().array(y_train_np)

        return X_train, y_train_stacked, y_train

    def test_interpolation_accuracy_sufficient_points(self) -> None:
        """
        Test multi-output GP interpolates training data with < 1e-6 accuracy.

        Uses MultiOutputGP with IndependentMultiOutputKernel to handle
        all outputs simultaneously. Verifies each output correctly interpolates
        its specific function (sin and cos).
        """
        n_train = 50
        X_train, y_train_stacked, y_train_unstacked = self._create_test_data(n_train)

        # Create independent kernels for each output
        kernels = []
        for i in range(self.noutputs):
            matern = MaternKernel(
                2.5,
                [1.0] * self.nvars,
                (0.1, 10.0),
                self.nvars,
                self.bkd()
            )
            constant = PolynomialScaling([1.0], (0.1, 10.0), self.bkd(), nvars=self.nvars)
            noise = IIDGaussianNoise(1e-14, (1e-16, 1e-12), self.bkd())
            kernel = constant * matern + noise
            kernels.append(kernel)

        # Create multi-output kernel
        mo_kernel = IndependentMultiOutputKernel(kernels)

        # Create and fit GP
        gp = MultiOutputGP(mo_kernel, nugget=1e-14)
        X_list = [X_train] * self.noutputs
        gp.fit(X_list, y_train_stacked)

        # Predict at training points (should interpolate)
        y_pred_stacked = gp.predict(X_list)

        # Check overall interpolation accuracy
        error = self.bkd().abs(y_pred_stacked - y_train_stacked)
        max_error = self.bkd().max(error).item()

        # Should interpolate with very high accuracy
        self.assertLess(max_error, 1e-6,
                       f"Interpolation error {max_error:.2e} exceeds 1e-6")

        # Verify each output interpolates its specific function correctly
        X_train_np = self.bkd().to_numpy(X_train)

        for i in range(self.noutputs):
            start = i * n_train
            end = (i + 1) * n_train
            y_pred_i = y_pred_stacked[start:end, 0]
            y_true_i = y_train_unstacked[:, i]

            # Check interpolation error
            error_i = self.bkd().abs(y_pred_i - y_true_i)
            max_error_i = self.bkd().max(error_i).item()
            self.assertLess(max_error_i, 1e-6,
                           f"Output {i} error {max_error_i:.2e} exceeds 1e-6")

            # Verify correct function: output 0 = sin, output 1 = cos
            y_pred_i_np = self.bkd().to_numpy(y_pred_i)
            if i == 0:
                # Output 0: sin(x1 + x2)
                expected = np.sin(X_train_np[0, :] + X_train_np[1, :])
                func_error = np.abs(y_pred_i_np - expected)
                self.assertLess(np.max(func_error), 1e-6,
                               f"Output 0 does not match sin(x1+x2)")
            elif i == 1:
                # Output 1: cos(x1 - x2)
                expected = np.cos(X_train_np[0, :] - X_train_np[1, :])
                func_error = np.abs(y_pred_i_np - expected)
                self.assertLess(np.max(func_error), 1e-6,
                               f"Output 1 does not match cos(x1-x2)")

    def test_prediction_at_new_points(self) -> None:
        """
        Test multi-output GP prediction at new test points.

        Tests both predict() and predict_with_uncertainty() methods.
        Verifies predictions are reasonable approximations of true functions.
        """
        n_train = 30
        n_test = 10

        X_train, y_train_stacked, _ = self._create_test_data(n_train)

        # Create test data
        X_test_np = np.random.randn(self.nvars, n_test)
        X_test = self.bkd().array(X_test_np)

        # True test values for verification
        y_test_0 = np.sin(X_test_np[0, :] + X_test_np[1, :])
        y_test_1 = np.cos(X_test_np[0, :] - X_test_np[1, :])

        # Create multi-output kernel
        kernels = []
        for i in range(self.noutputs):
            matern = MaternKernel(
                2.5,
                [1.0] * self.nvars,
                (0.1, 10.0),
                self.nvars,
                self.bkd()
            )
            kernels.append(matern)

        mo_kernel = IndependentMultiOutputKernel(kernels)

        # Create and fit GP
        gp = MultiOutputGP(mo_kernel, nugget=1e-14)
        X_train_list = [X_train] * self.noutputs
        gp.fit(X_train_list, y_train_stacked)

        # Predict at test points
        X_test_list = [X_test] * self.noutputs
        y_pred = gp.predict(X_test_list)

        # Check shape
        expected_shape = (n_test * self.noutputs, 1)
        self.assertEqual(y_pred.shape, expected_shape)

        # Predictions should be in reasonable range
        y_pred_abs = self.bkd().abs(y_pred)
        max_pred = self.bkd().max(y_pred_abs).item()
        self.assertLess(max_pred, 10.0)

        # Verify predictions are reasonable approximations
        y_pred_np = self.bkd().to_numpy(y_pred)
        y_pred_0 = y_pred_np[0:n_test, 0]
        y_pred_1 = y_pred_np[n_test:2*n_test, 0]

        # Should approximate the true functions (not perfect, but reasonable)
        error_0 = np.abs(y_pred_0 - y_test_0)
        error_1 = np.abs(y_pred_1 - y_test_1)

        # With enough training data, predictions should be decent
        self.assertLess(np.mean(error_0), 0.5)
        self.assertLess(np.mean(error_1), 0.5)

        # Test predict_with_uncertainty
        y_pred_unc, y_std = gp.predict_with_uncertainty(X_test_list)

        # Check shapes
        self.assertEqual(y_pred_unc.shape, expected_shape)
        self.assertEqual(y_std.shape, expected_shape)

        # Predictions should match
        error_match = self.bkd().abs(y_pred - y_pred_unc)
        max_match_error = self.bkd().max(error_match).item()
        self.assertLess(max_match_error, 1e-10)

        # Uncertainties should be positive
        y_std_min = self.bkd().min(y_std).item()
        self.assertGreaterEqual(y_std_min, 0.0)

    def test_block_diagonal_structure_preserved(self) -> None:
        """
        Test that independent kernel maintains block-diagonal structure.

        Uses the kernel directly to verify off-diagonal blocks are zero.
        """
        n_train = 20
        X_train, _, _ = self._create_test_data(n_train)

        # Create simple kernels
        kernels = []
        for i in range(self.noutputs):
            matern = MaternKernel(
                2.5,
                [1.0] * self.nvars,
                (0.1, 10.0),
                self.nvars,
                self.bkd()
            )
            kernels.append(matern)

        mo_kernel = IndependentMultiOutputKernel(kernels)

        # Build kernel matrix
        X_list = [X_train] * self.noutputs
        K = mo_kernel(X_list, block_format=False)

        # Check that off-diagonal blocks are zero
        for i in range(self.noutputs):
            for j in range(self.noutputs):
                if i != j:
                    row_start = i * n_train
                    row_end = (i + 1) * n_train
                    col_start = j * n_train
                    col_end = (j + 1) * n_train

                    K_ij = K[row_start:row_end, col_start:col_end]
                    K_ij_abs = self.bkd().abs(K_ij)
                    max_val = self.bkd().max(K_ij_abs).item()
                    self.assertLess(max_val, 1e-10,
                                   f"Off-diagonal block [{i},{j}] not zero")


class TestMultiOutputGPWithLMCKernel(Generic[Array], unittest.TestCase):
    """
    Test multi-output GP using LinearCoregionalizationKernel.

    Tests a single GP with LMC kernel that captures correlations between outputs.
    """

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        """Set up test environment."""
        np.random.seed(42)
        self.nvars = 2
        self.noutputs = 2

    def _create_test_data(self, n_train: int):
        """Create synthetic multi-output test data."""
        X_train_np = np.random.randn(self.nvars, n_train)
        X_train = self.bkd().array(X_train_np)

        # Create correlated outputs
        y1 = np.sin(X_train_np[0, :] + X_train_np[1, :])
        y2 = np.cos(X_train_np[0, :] - X_train_np[1, :])

        # Stack outputs
        y_train_stacked_np = np.concatenate([y1, y2])[:, np.newaxis]
        y_train_stacked = self.bkd().array(y_train_stacked_np)

        return X_train, y_train_stacked

    def test_lmc_interpolation_accuracy(self) -> None:
        """
        Test LMC kernel achieves < 1e-6 interpolation accuracy.

        LMC kernel models output correlations while maintaining
        high interpolation accuracy. Uses MultiOutputGP.
        """
        n_train = 40
        X_train, y_train_stacked = self._create_test_data(n_train)

        # Create base kernels
        base_kernels = []
        for q in range(2):  # 2 components
            matern = MaternKernel(
                2.5,
                [1.0] * self.nvars,
                (0.1, 10.0),
                self.nvars,
                self.bkd()
            )
            constant = PolynomialScaling([1.0], (0.1, 10.0), self.bkd(), nvars=self.nvars)
            kernel = constant * matern
            base_kernels.append(kernel)

        # Create coregionalization matrices
        B1_np = np.array([[1.0, 0.5], [0.5, 1.0]])  # Positive correlation
        B2_np = np.array([[0.5, 0.0], [0.0, 0.5]])  # Independent component

        coreg_matrices = [
            self.bkd().array(B1_np),
            self.bkd().array(B2_np),
        ]

        # Create LMC kernel
        lmc_kernel = LinearCoregionalizationKernel(
            base_kernels,
            coreg_matrices,
            self.noutputs
        )

        # Create and fit GP
        gp = MultiOutputGP(lmc_kernel, nugget=1e-14)
        X_list = [X_train] * self.noutputs
        gp.fit(X_list, y_train_stacked)

        # Predict at training points
        y_pred = gp.predict(X_list)

        # Check interpolation accuracy
        error = self.bkd().abs(y_pred - y_train_stacked)
        max_error = self.bkd().max(error).item()

        self.assertLess(max_error, 1e-6,
                       f"LMC interpolation error {max_error:.2e} exceeds 1e-6")

    def test_lmc_captures_correlations(self) -> None:
        """Test that LMC kernel has non-zero off-diagonal blocks."""
        n_train = 20
        X_train, _ = self._create_test_data(n_train)

        # Create base kernel
        matern = MaternKernel(
            2.5,
            [1.0] * self.nvars,
            (0.1, 10.0),
            self.nvars,
            self.bkd()
        )

        # Coregionalization matrix with correlation
        B_np = np.array([[1.0, 0.7], [0.7, 1.0]])
        B = self.bkd().array(B_np)

        lmc_kernel = LinearCoregionalizationKernel(
            [matern],
            [B],
            self.noutputs
        )

        # Get kernel blocks
        X_list = [X_train] * self.noutputs
        K_blocks = lmc_kernel(X_list, block_format=True)

        # Off-diagonal blocks should be non-zero
        K_01 = K_blocks[0][1]
        K_01_abs = self.bkd().abs(K_01)
        max_val = self.bkd().max(K_01_abs).item()

        self.assertGreater(max_val, 0.01,
                          "LMC should have non-zero output correlations")


class TestMultiOutputGPWithIndependentKernelNumpy(
    TestMultiOutputGPWithIndependentKernel[NDArray[Any]]
):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        if not hasattr(self, '_bkd'):
            self._bkd = NumpyBkd()
        return self._bkd


class TestMultiOutputGPWithIndependentKernelTorch(
    TestMultiOutputGPWithIndependentKernel[torch.Tensor]
):
    """PyTorch backend tests."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestMultiOutputGPWithLMCKernelNumpy(
    TestMultiOutputGPWithLMCKernel[NDArray[Any]]
):
    """NumPy backend tests for LMC."""

    def bkd(self) -> NumpyBkd:
        if not hasattr(self, '_bkd'):
            self._bkd = NumpyBkd()
        return self._bkd


class TestMultiOutputGPWithLMCKernelTorch(
    TestMultiOutputGPWithLMCKernel[torch.Tensor]
):
    """PyTorch backend tests for LMC."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


def load_tests(
    loader: unittest.TestLoader, tests, pattern: str
) -> unittest.TestSuite:
    """Custom test loader to exclude base classes."""
    test_suite = unittest.TestSuite()
    for test_class in [
        TestMultiOutputGPWithIndependentKernelNumpy,
        TestMultiOutputGPWithIndependentKernelTorch,
        TestMultiOutputGPWithLMCKernelNumpy,
        TestMultiOutputGPWithLMCKernelTorch,
    ]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner()
    runner.run(suite)
