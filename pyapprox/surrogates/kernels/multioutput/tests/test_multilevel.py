"""
Tests for MultiLevelKernel with new scaling functions.
"""

import unittest
from typing import Any, Generic
import numpy as np
from numpy.typing import NDArray

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.surrogates.kernels.matern import Matern52Kernel
from pyapprox.surrogates.kernels.multioutput import (
    MultiLevelKernel,
)
from pyapprox.surrogates.kernels.scalings import (
    PolynomialScaling,
)


class TestMultiLevelKernel(Generic[Array], unittest.TestCase):
    """Base test class for MultiLevelKernel."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._nvars = 1

    def _create_constant_scaling(self, value: float) -> PolynomialScaling:
        """Helper to create constant scaling using PolynomialScaling with degree 0."""
        return PolynomialScaling([value], (0.1, 2.0), self._bkd, nvars=self._nvars)

    def _create_linear_scaling(self, c0: float, c1: float) -> PolynomialScaling:
        """Helper to create linear scaling using PolynomialScaling with degree 1."""
        return PolynomialScaling([c0, c1], (0.1, 2.0), self._bkd)

    def test_two_level_constant_scaling(self):
        """Test two-level kernel with constant scaling."""
        # Create two kernels
        k0 = Matern52Kernel([1.0], (0.1, 10.0), self._nvars, self._bkd)
        k1 = Matern52Kernel([0.5], (0.1, 10.0), self._nvars, self._bkd)

        # Constant scaling ρ_0 = 0.8
        rho_0 = self._create_constant_scaling(0.8)

        # Create multi-level kernel
        ml_kernel = MultiLevelKernel([k0, k1], [rho_0])

        # Test data
        X0 = self._bkd.array([[-1.0, 0.0, 1.0]])
        X1 = self._bkd.array([[-0.5, 0.5]])

        # Compute kernel matrix
        K = ml_kernel([X0, X1])

        # Verify shape
        self.assertEqual(K.shape, (5, 5))

        # Verify autoregressive structure:
        # K_00 = k_0(X0, X0)
        K_00 = k0(X0, X0)
        np.testing.assert_allclose(K[:3, :3], K_00, rtol=1e-10)

        # K_11 = ρ_0² * k_0(X1, X1) + k_1(X1, X1)
        K_11_expected = 0.8**2 * k0(X1, X1) + k1(X1, X1)
        np.testing.assert_allclose(K[3:, 3:], K_11_expected, rtol=1e-10)

        # K_10 = ρ_0 * k_0(X1, X0)
        K_10_expected = 0.8 * k0(X1, X0)
        np.testing.assert_allclose(K[3:, :3], K_10_expected, rtol=1e-10)

        # K_01 = K_10^T (symmetry)
        np.testing.assert_allclose(K[:3, 3:], K[3:, :3].T, rtol=1e-10)

    def test_three_level_constant_scaling(self):
        """Test three-level kernel with constant scalings."""
        # Create three kernels
        k0 = Matern52Kernel([1.0], (0.1, 10.0), self._nvars, self._bkd)
        k1 = Matern52Kernel([0.8], (0.1, 10.0), self._nvars, self._bkd)
        k2 = Matern52Kernel([0.5], (0.1, 10.0), self._nvars, self._bkd)

        # Constant scalings
        rho_0 = self._create_constant_scaling(0.9)
        rho_1 = self._create_constant_scaling(0.85)

        # Create multi-level kernel
        ml_kernel = MultiLevelKernel([k0, k1, k2], [rho_0, rho_1])

        # Test data (same points for all levels for simplicity)
        X = self._bkd.array([[0.0, 0.5]])
        X_list = [X, X, X]

        # Compute kernel matrix
        K = ml_kernel(X_list)

        # Verify shape
        self.assertEqual(K.shape, (6, 6))

        # Verify autoregressive structure for level 2:
        # f_0 ~ GP(0, k_0)
        # f_1 = ρ_0 * f_0 + δ_1 ~ GP(0, ρ_0² k_0 + k_1)
        # f_2 = ρ_1 * f_1 + δ_2 = ρ_0*ρ_1 * f_0 + ρ_1 * δ_1 + δ_2
        #
        # K_22 = (ρ_0*ρ_1)² k_0 + ρ_1² k_1 + k_2
        K_22_expected = (
            (0.9 * 0.85)**2 * k0(X, X) +
            0.85**2 * k1(X, X) +
            k2(X, X)
        )
        np.testing.assert_allclose(K[4:, 4:], K_22_expected, rtol=1e-10)

        # K_21 = Cov[f_2, f_1] = Cov[ρ_0*ρ_1*f_0 + ρ_1*δ_1, ρ_0*f_0 + δ_1]
        #      = ρ_0² * ρ_1 * k_0 + ρ_1 * k_1
        K_21_expected = 0.9**2 * 0.85 * k0(X, X) + 0.85 * k1(X, X)
        np.testing.assert_allclose(K[4:, 2:4], K_21_expected, rtol=1e-10)

        # K_20 = Cov[f_2, f_0] = Cov[ρ_0*ρ_1*f_0 + ρ_1*δ_1 + δ_2, f_0]
        #      = ρ_0 * ρ_1 * k_0
        K_20_expected = 0.9 * 0.85 * k0(X, X)
        np.testing.assert_allclose(K[4:, :2], K_20_expected, rtol=1e-10)

    def test_spatially_varying_scaling(self):
        """Test with spatially varying (linear) scaling."""
        # Create two kernels
        k0 = Matern52Kernel([1.0], (0.1, 10.0), self._nvars, self._bkd)
        k1 = Matern52Kernel([0.5], (0.1, 10.0), self._nvars, self._bkd)

        # Linear scaling ρ(x) = 0.9 + 0.1*x
        rho_0 = self._create_linear_scaling(0.9, 0.1)

        # Create multi-level kernel
        ml_kernel = MultiLevelKernel([k0, k1], [rho_0])

        # Test at specific points
        X0 = self._bkd.array([[-1.0, 0.0, 1.0]])
        X1 = self._bkd.array([[-1.0, 1.0]])

        # Compute kernel matrix
        K = ml_kernel([X0, X1])

        # Manually compute K_10 to verify spatially varying scaling
        # ρ(-1) = 0.8, ρ(0) = 0.9, ρ(1) = 1.0
        # Use eval_scaling to get scaling values, not kernel matrix
        rho_X1 = rho_0.eval_scaling(X1)  # Shape (2, 1): [[0.8], [1.0]]

        # K_10 = Cov[f_1, f_0] = Cov[ρ*f_0 + δ_1, f_0] = ρ(X1) * k0(X1, X0)
        # Note: scaling only applies on level 1 side, not level 0
        K_10_manual = rho_X1 * k0(X1, X0)

        np.testing.assert_allclose(K[3:, :3], K_10_manual, rtol=1e-10)

    def test_block_format(self):
        """Test block format output."""
        # Create simple two-level kernel
        k0 = Matern52Kernel([1.0], (0.1, 10.0), self._nvars, self._bkd)
        k1 = Matern52Kernel([0.5], (0.1, 10.0), self._nvars, self._bkd)
        rho_0 = self._create_constant_scaling(0.8)
        ml_kernel = MultiLevelKernel([k0, k1], [rho_0])

        X0 = self._bkd.array([[0.0, 0.5]])
        X1 = self._bkd.array([[0.25]])

        # Get block format
        blocks = ml_kernel([X0, X1], block_format=True)

        # Verify structure
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(blocks[0]), 2)
        self.assertEqual(len(blocks[1]), 2)

        # K_00 should be (2, 2)
        self.assertEqual(blocks[0][0].shape, (2, 2))

        # K_11 should be (1, 1)
        self.assertEqual(blocks[1][1].shape, (1, 1))

        # K_10 should be (1, 2)
        self.assertEqual(blocks[1][0].shape, (1, 2))

        # K_01 should be (2, 1) - transpose of K_10
        self.assertEqual(blocks[0][1].shape, (2, 1))

    def test_symmetry(self):
        """Test that kernel matrix is symmetric for self-covariance."""
        k0 = Matern52Kernel([1.0], (0.1, 10.0), self._nvars, self._bkd)
        k1 = Matern52Kernel([0.5], (0.1, 10.0), self._nvars, self._bkd)
        rho_0 = self._create_constant_scaling(0.85)
        ml_kernel = MultiLevelKernel([k0, k1], [rho_0])

        X = self._bkd.array(np.random.randn(1, 5))
        X_list = [X, X]

        K = ml_kernel(X_list)

        # Verify symmetry
        np.testing.assert_allclose(K, K.T, rtol=1e-10)

    def test_hyperparameter_list(self):
        """Test combined hyperparameter list."""
        k0 = Matern52Kernel([1.0], (0.1, 10.0), self._nvars, self._bkd)
        k1 = Matern52Kernel([0.5], (0.1, 10.0), self._nvars, self._bkd)
        rho_0 = self._create_linear_scaling(0.9, 0.1)
        ml_kernel = MultiLevelKernel([k0, k1], [rho_0])

        hyp_list = ml_kernel.hyp_list()

        # Should have:
        # - 1 lengthscale from k0
        # - 1 lengthscale from k1
        # - 2 coefficients from rho_0 (c0, c1)
        # Total: 4 hyperparameters
        self.assertEqual(len(hyp_list.hyperparameters()), 4)

    def test_validation_errors(self):
        """Test that validation catches errors."""
        k0 = Matern52Kernel([1.0], (0.1, 10.0), self._nvars, self._bkd)
        k1 = Matern52Kernel([0.5], (0.1, 10.0), self._nvars, self._bkd)
        rho_0 = self._create_constant_scaling(0.8)

        # Wrong number of scalings
        with self.assertRaises(ValueError):
            MultiLevelKernel([k0, k1], [])  # Need 1 scaling, got 0

        with self.assertRaises(ValueError):
            MultiLevelKernel([k0, k1], [rho_0, rho_0])  # Need 1, got 2

        # Empty kernel list
        with self.assertRaises(ValueError):
            MultiLevelKernel([], [])

    def test_cross_covariance(self):
        """Test cross-covariance with different test points."""
        k0 = Matern52Kernel([1.0], (0.1, 10.0), self._nvars, self._bkd)
        k1 = Matern52Kernel([0.5], (0.1, 10.0), self._nvars, self._bkd)
        rho_0 = self._create_constant_scaling(0.8)
        ml_kernel = MultiLevelKernel([k0, k1], [rho_0])

        X1_0 = self._bkd.array([[0.0, 0.5]])
        X1_1 = self._bkd.array([[0.25]])

        X2_0 = self._bkd.array([[-0.5, 0.0, 1.0]])
        X2_1 = self._bkd.array([[0.0, 0.5]])

        # Cross-covariance K(X1, X2)
        K_cross = ml_kernel([X1_0, X1_1], [X2_0, X2_1])

        # Verify shape: (2 + 1, 3 + 2) = (3, 5)
        self.assertEqual(K_cross.shape, (3, 5))

        # Verify some blocks manually
        K_00_cross = k0(X1_0, X2_0)
        np.testing.assert_allclose(K_cross[:2, :3], K_00_cross, rtol=1e-10)

    def test_precision_matrix_4model_sequential(self) -> None:
        """
        Test precision matrix for 4-model sequential multilevel: 0 -> 1 -> 2 -> 3.

        In a sequential autoregressive structure, each model is conditionally
        independent of non-adjacent models given the intermediate models.
        The precision matrix should be block tri-diagonal:

        - K_inv[0, 2] = 0 (model 0 independent of 2 given 1)
        - K_inv[0, 3] = 0 (model 0 independent of 3 given 1, 2)
        - K_inv[1, 3] = 0 (model 1 independent of 3 given 2)

        And their transposes should also be zero.
        """
        # Create 4-level sequential structure
        kernels = [
            Matern52Kernel([1.0], (0.1, 10.0), self._nvars, self._bkd),
            Matern52Kernel([0.7], (0.1, 10.0), self._nvars, self._bkd),
            Matern52Kernel([0.5], (0.1, 10.0), self._nvars, self._bkd),
            Matern52Kernel([0.3], (0.1, 10.0), self._nvars, self._bkd),
        ]

        # Use constant scalings (PolynomialScaling with degree 0)
        from pyapprox.surrogates.kernels.scalings import PolynomialScaling
        scalings = [
            PolynomialScaling([0.9], (0.5, 1.5), self._bkd, nvars=self._nvars),   # 0 -> 1
            PolynomialScaling([0.85], (0.5, 1.5), self._bkd, nvars=self._nvars),  # 1 -> 2
            PolynomialScaling([0.8], (0.5, 1.5), self._bkd, nvars=self._nvars),   # 2 -> 3
        ]

        ml_kernel = MultiLevelKernel(kernels, scalings)

        # Create test points (same for all levels for simplicity)
        np.random.seed(42)
        n_samples = 4
        X_np = np.random.randn(self._nvars, n_samples)
        X = self._bkd.array(X_np)
        X_list = [X, X, X, X]

        # Compute full covariance matrix
        K = ml_kernel(X_list, block_format=False)
        K_np = self._bkd.to_numpy(K)

        # Compute precision matrix (use numpy for linalg operations in tests)
        K_inv = np.linalg.inv(K_np)

        # Extract off-diagonal blocks that should be zero
        # Each level has n_samples rows/cols
        K_inv_02 = K_inv[0:n_samples, 2*n_samples:3*n_samples]
        K_inv_20 = K_inv[2*n_samples:3*n_samples, 0:n_samples]

        K_inv_03 = K_inv[0:n_samples, 3*n_samples:4*n_samples]
        K_inv_30 = K_inv[3*n_samples:4*n_samples, 0:n_samples]

        K_inv_13 = K_inv[n_samples:2*n_samples, 3*n_samples:4*n_samples]
        K_inv_31 = K_inv[3*n_samples:4*n_samples, n_samples:2*n_samples]

        # Verify conditional independence structure
        np.testing.assert_allclose(
            K_inv_02, 0.0, atol=1e-10,
            err_msg="Precision K_inv[0,2] should be zero (level 0 independent of 2 given 1)"
        )
        np.testing.assert_allclose(
            K_inv_20, 0.0, atol=1e-10,
            err_msg="Precision K_inv[2,0] should be zero (level 2 independent of 0 given 1)"
        )

        np.testing.assert_allclose(
            K_inv_03, 0.0, atol=1e-10,
            err_msg="Precision K_inv[0,3] should be zero (level 0 independent of 3 given 1,2)"
        )
        np.testing.assert_allclose(
            K_inv_30, 0.0, atol=1e-10,
            err_msg="Precision K_inv[3,0] should be zero (level 3 independent of 0 given 1,2)"
        )

        np.testing.assert_allclose(
            K_inv_13, 0.0, atol=1e-10,
            err_msg="Precision K_inv[1,3] should be zero (level 1 independent of 3 given 2)"
        )
        np.testing.assert_allclose(
            K_inv_31, 0.0, atol=1e-10,
            err_msg="Precision K_inv[3,1] should be zero (level 3 independent of 1 given 2)"
        )

        # Verify that adjacent blocks are NOT zero (sanity check)
        K_inv_01 = K_inv[0:n_samples, n_samples:2*n_samples]
        K_inv_12 = K_inv[n_samples:2*n_samples, 2*n_samples:3*n_samples]
        K_inv_23 = K_inv[2*n_samples:3*n_samples, 3*n_samples:4*n_samples]

        self.assertGreater(
            np.abs(K_inv_01).max(), 1e-5,
            "Adjacent block K_inv[0,1] should be non-zero"
        )
        self.assertGreater(
            np.abs(K_inv_12).max(), 1e-5,
            "Adjacent block K_inv[1,2] should be non-zero"
        )
        self.assertGreater(
            np.abs(K_inv_23).max(), 1e-5,
            "Adjacent block K_inv[2,3] should be non-zero"
        )



class TestMultiLevelKernelNumpy(TestMultiLevelKernel[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMultiLevelKernelOptimizeTorch(unittest.TestCase):
    """Torch-only test: ML optimization requires autograd."""

    def setUp(self) -> None:
        import torch
        torch.set_default_dtype(torch.float64)
        from pyapprox.util.backends.torch import TorchBkd
        self._bkd = TorchBkd()
        self._nvars = 2

    def test_optimize_hyperparameters(self):
        """Test hyperparameter optimization for multilevel kernel GP."""
        from pyapprox.surrogates.gaussianprocess.torch_multioutput import (
            TorchMultiOutputGP,
        )
        from pyapprox.surrogates.gaussianprocess.fitters import (
            MultiOutputGPMaximumLikelihoodFitter,
        )

        np.random.seed(42)
        bkd = self._bkd

        # Create a 2-level kernel
        k0 = Matern52Kernel([1.0], (0.1, 10.0), self._nvars, bkd)
        k1 = Matern52Kernel([0.5], (0.1, 10.0), self._nvars, bkd)
        rho_0 = PolynomialScaling(
            [0.8], (0.01, 5.0), bkd, nvars=self._nvars,
        )
        ml_kernel = MultiLevelKernel([k0, k1], [rho_0])

        # Generate synthetic data
        n_train = 10
        X_train = bkd.array(np.random.randn(self._nvars, n_train))
        X_list = [X_train, X_train]

        y0 = bkd.sin(X_train[0, :])
        y1 = 0.8 * y0 + 0.1 * bkd.array(np.random.randn(n_train))
        y_train = [
            bkd.reshape(y0, (1, -1)),
            bkd.reshape(y1, (1, -1)),
        ]

        gp = TorchMultiOutputGP(ml_kernel, nugget=1e-4)

        # Fit with initial hyperparameters to get baseline NLL
        gp._fit_internal(X_list, y_train)
        nll_before = float(bkd.to_numpy(
            gp.neg_log_marginal_likelihood()
        ))

        # Optimize via fitter
        fitter = MultiOutputGPMaximumLikelihoodFitter(bkd)
        result = fitter.fit(gp, X_list, y_train)

        nll_after = float(result.neg_log_marginal_likelihood())
        self.assertLess(
            nll_after, nll_before,
            f"NLL should decrease: {nll_before:.4f} -> {nll_after:.4f}"
        )


from pyapprox.util.test_utils import load_tests  # noqa: F401


if __name__ == "__main__":
    unittest.main()
