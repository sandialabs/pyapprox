"""
Tests for TorchMaternKernel with arbitrary nu.
"""

import unittest
import math

import torch

from pyapprox.surrogates.kernels.torch_matern import TorchMaternKernel
from pyapprox.surrogates.kernels import Matern52Kernel, Matern32Kernel
from pyapprox.util.backends.torch import TorchBkd


class TestTorchMaternKernel(unittest.TestCase):
    """Tests for TorchMaternKernel."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.bkd = TorchBkd()

    def test_kernel_shape(self):
        """Test kernel matrix has correct shape."""
        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0, 1.0],
            lenscale_bounds=(0.1, 10.0), nvars=2
        )

        X1 = torch.randn(2, 10)
        X2 = torch.randn(2, 5)

        K = kernel(X1, X2)
        self.assertEqual(K.shape, (10, 5))

    def test_kernel_symmetry(self):
        """Test kernel is symmetric when X1 = X2."""
        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0), nvars=1
        )

        X = torch.randn(1, 10)
        K = kernel(X, X)

        self.assertTrue(torch.allclose(K, K.T, rtol=1e-10))

    def test_kernel_positive_definite(self):
        """Test kernel matrix is positive definite."""
        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0), nvars=1
        )

        X = torch.linspace(-2, 2, 20).reshape(1, -1)
        K = kernel(X, X)

        # Add small nugget for numerical stability
        K_stable = K + 1e-10 * torch.eye(K.shape[0])

        # Check eigenvalues are positive
        eigenvalues = torch.linalg.eigvalsh(K_stable)
        self.assertTrue(torch.all(eigenvalues > 0))

    def test_diagonal_is_one(self):
        """Test diagonal elements are 1."""
        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0, 1.0],
            lenscale_bounds=(0.1, 10.0), nvars=2
        )

        X = torch.randn(2, 10)
        K = kernel(X, X)
        diag = torch.diag(K)

        self.assertTrue(torch.allclose(diag, torch.ones_like(diag), rtol=1e-6))

    def test_nu_25_matches_matern52(self):
        """Test nu=2.5 matches backend-agnostic Matern52Kernel."""
        torch_kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0), nvars=1
        )
        ref_kernel = Matern52Kernel(
            lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0),
            nvars=1,
            bkd=self.bkd
        )

        X = torch.linspace(-2, 2, 20).reshape(1, -1)

        K_torch = torch_kernel(X, X)
        K_ref = ref_kernel(X, X)

        self.assertTrue(torch.allclose(K_torch, K_ref, rtol=1e-5, atol=1e-6))

    def test_nu_15_matches_matern32(self):
        """Test nu=1.5 matches backend-agnostic Matern32Kernel."""
        torch_kernel = TorchMaternKernel(
            nu=1.5, lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0), nvars=1
        )
        ref_kernel = Matern32Kernel(
            lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0),
            nvars=1,
            bkd=self.bkd
        )

        X = torch.linspace(-2, 2, 20).reshape(1, -1)

        K_torch = torch_kernel(X, X)
        K_ref = ref_kernel(X, X)

        self.assertTrue(torch.allclose(K_torch, K_ref, rtol=1e-5, atol=1e-6))

    def test_arbitrary_nu(self):
        """Test kernel works with arbitrary nu values."""
        # Test various nu values that aren't 1.5 or 2.5
        for nu in [0.5, 1.0, 2.0, 2.3, 3.0, 4.5]:
            kernel = TorchMaternKernel(
                nu=nu, lenscale=[1.0],
                lenscale_bounds=(0.1, 10.0), nvars=1
            )

            X = torch.linspace(-2, 2, 10).reshape(1, -1)
            K = kernel(X, X)

            # Check basic properties
            self.assertEqual(K.shape, (10, 10))
            self.assertTrue(torch.allclose(K, K.T, rtol=1e-10))
            # For general nu, asymptotic approximation may not give exactly 1
            # on diagonal due to numerical stability epsilon
            self.assertTrue(torch.allclose(
                torch.diag(K), torch.ones(10), rtol=1e-3
            ))

    def test_large_nu_approximates_rbf(self):
        """Test large nu approximates RBF/squared exponential kernel."""
        torch_kernel = TorchMaternKernel(
            nu=150.0, lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0), nvars=1
        )

        X = torch.linspace(-2, 2, 20).reshape(1, -1)
        K = torch_kernel(X, X)

        # Compute RBF kernel manually
        distances = torch.cdist(X.T, X.T)
        K_rbf = torch.exp(-0.5 * distances**2)

        self.assertTrue(torch.allclose(K, K_rbf, rtol=1e-3, atol=1e-4))

    def test_length_scale_effect(self):
        """Test length scale affects correlation range."""
        X = torch.linspace(-2, 2, 20).reshape(1, -1)

        # Small length scale -> narrower correlation
        kernel_small = TorchMaternKernel(
            nu=2.5, lenscale=[0.5],
            lenscale_bounds=(0.1, 10.0), nvars=1
        )
        K_small = kernel_small(X, X)

        # Large length scale -> wider correlation
        kernel_large = TorchMaternKernel(
            nu=2.5, lenscale=[2.0],
            lenscale_bounds=(0.1, 10.0), nvars=1
        )
        K_large = kernel_large(X, X)

        # Off-diagonal elements should be larger for larger length scale
        off_diag_small = K_small[0, 5]
        off_diag_large = K_large[0, 5]

        self.assertGreater(float(off_diag_large), float(off_diag_small))

    def test_autograd_compatible(self):
        """Test kernel is compatible with autograd."""
        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0), nvars=1
        )

        # Create tensor and reshape, then require grad on the reshaped tensor
        X = torch.linspace(-2, 2, 10).reshape(1, -1).requires_grad_(True)
        K = kernel(X, X)

        # Should be able to compute gradients
        loss = K.sum()
        loss.backward()

        self.assertIsNotNone(X.grad)
        self.assertEqual(X.grad.shape, X.shape)

    def test_hyp_list(self):
        """Test hyperparameter list."""
        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0, 2.0],
            lenscale_bounds=(0.1, 10.0), nvars=2
        )

        hyp_list = kernel.hyp_list()
        self.assertEqual(hyp_list.nparams(), 2)

    def test_repr(self):
        """Test string representation."""
        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0), nvars=1
        )

        repr_str = repr(kernel)
        self.assertIn("GeneralMaternKernel", repr_str)
        self.assertIn("nu=2.5", repr_str)


if __name__ == "__main__":
    unittest.main()
