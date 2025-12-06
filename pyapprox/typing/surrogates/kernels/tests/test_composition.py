import unittest
from typing import Generic, Any

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.surrogates.kernels.composition import (
    ProductKernel,
    SumKernel,
)
from pyapprox.typing.surrogates.kernels.matern import MaternKernel


class TestProductKernel(Generic[Array], unittest.TestCase):
    """
    Base test class for ProductKernel.

    Derived classes must implement the bkd() method to provide the backend.
    """

    def setUp(self) -> None:
        """
        Set up test environment for ProductKernel.
        """
        np.random.seed(42)
        self.nvars = 2
        self.nsamples1 = 5
        self.nsamples2 = 4

        # Create two Matern kernels for composition
        self.kernel1 = MaternKernel(
            2.5,
            [1.0, 1.0],
            (0.1, 10.0),
            self.nvars,
            self.bkd()
        )
        self.kernel2 = MaternKernel(
            1.5,
            [0.5, 0.5],
            (0.1, 10.0),
            self.nvars,
            self.bkd()
        )

        # Create sample data
        self.X1 = self.bkd().array(np.random.randn(self.nvars, self.nsamples1))
        self.X2 = self.bkd().array(np.random.randn(self.nvars, self.nsamples2))

    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_initialization(self) -> None:
        """
        Test ProductKernel initialization.
        """
        product = ProductKernel(self.kernel1, self.kernel2)
        self.assertEqual(product.nvars(), self.nvars)
        self.assertIsNotNone(product.bkd())

    def test_backend_mismatch_error(self) -> None:
        """
        Test that ProductKernel raises error when backends don't match.
        """
        # Create kernel with different backend
        if isinstance(self.bkd(), NumpyBkd):
            other_bkd = TorchBkd()
        else:
            other_bkd = NumpyBkd()

        kernel_other = MaternKernel(
            2.5,
            [1.0, 1.0],
            (0.1, 10.0),
            self.nvars,
            other_bkd
        )

        with self.assertRaises(ValueError) as context:
            ProductKernel(self.kernel1, kernel_other)

        self.assertIn("same backend type", str(context.exception))

    def test_hyperparameter_list_combination(self) -> None:
        """
        Test that hyperparameter lists are combined correctly.
        """
        product = ProductKernel(self.kernel1, self.kernel2)
        hyp_list = product.hyp_list()

        # Should have parameters from both kernels
        nparams1 = self.kernel1.hyp_list().nparams()
        nparams2 = self.kernel2.hyp_list().nparams()
        self.assertEqual(hyp_list.nparams(), nparams1 + nparams2)

    def test_kernel_matrix_shape(self) -> None:
        """
        Test that ProductKernel produces correct output shape.
        """
        product = ProductKernel(self.kernel1, self.kernel2)

        # Test with two different inputs
        K = product(self.X1, self.X2)
        self.assertEqual(K.shape, (self.nsamples1, self.nsamples2))

        # Test with single input
        K_self = product(self.X1)
        self.assertEqual(K_self.shape, (self.nsamples1, self.nsamples1))

    def test_kernel_matrix_values(self) -> None:
        """
        Test that ProductKernel computes K1 * K2 correctly.
        """
        product = ProductKernel(self.kernel1, self.kernel2)

        K1 = self.kernel1(self.X1, self.X2)
        K2 = self.kernel2(self.X1, self.X2)
        K_product = product(self.X1, self.X2)

        expected = K1 * K2
        self.bkd().assert_allclose(K_product, expected)

    def test_diagonal(self) -> None:
        """
        Test diagonal computation for ProductKernel.
        """
        product = ProductKernel(self.kernel1, self.kernel2)

        diag1 = self.kernel1.diag(self.X1)
        diag2 = self.kernel2.diag(self.X1)
        diag_product = product.diag(self.X1)

        expected = diag1 * diag2
        self.bkd().assert_allclose(diag_product, expected)

    def test_jacobian(self) -> None:
        """
        Test Jacobian computation for ProductKernel.

        ProductKernel should satisfy product rule: d(K1*K2)/dx = dK1*K2 + K1*dK2
        """
        product = ProductKernel(self.kernel1, self.kernel2)

        jac = product.jacobian(self.X1, self.X2)

        # Check shape
        self.assertEqual(jac.shape, (self.nsamples1, self.nsamples2, self.nvars))

        # Check finiteness
        self.assertTrue(self.bkd().all_bool(self.bkd().isfinite(jac)))

        # Manually compute using product rule
        K1 = self.kernel1(self.X1, self.X2)
        K2 = self.kernel2(self.X1, self.X2)
        dK1 = self.kernel1.jacobian(self.X1, self.X2)
        dK2 = self.kernel2.jacobian(self.X1, self.X2)

        expected = dK1 * K2[..., None] + K1[..., None] * dK2
        self.bkd().assert_allclose(jac, expected)

    def test_param_jacobian(self) -> None:
        """
        Test parameter Jacobian for ProductKernel.

        The parameter Jacobian should stack derivatives from both kernels.
        """
        product = ProductKernel(self.kernel1, self.kernel2)

        jac = product.jacobian_wrt_params(self.X1)

        # Check shape
        nparams_total = self.kernel1.hyp_list().nparams() + self.kernel2.hyp_list().nparams()
        self.assertEqual(jac.shape, (self.nsamples1, self.nsamples1, nparams_total))

        # Check finiteness
        self.assertTrue(self.bkd().all_bool(self.bkd().isfinite(jac)))

    def test_operator_overloading(self) -> None:
        """
        Test that * operator creates ProductKernel.
        """
        product = self.kernel1 * self.kernel2
        self.assertIsInstance(product, ProductKernel)

        # Verify it produces same result as explicit construction
        product_explicit = ProductKernel(self.kernel1, self.kernel2)
        K1 = product(self.X1, self.X2)
        K2 = product_explicit(self.X1, self.X2)
        self.bkd().assert_allclose(K1, K2)

    def test_hvp_wrt_x1(self) -> None:
        """
        Test HVP computation for ProductKernel using product rule.
        """
        product = ProductKernel(self.kernel1, self.kernel2)

        # Single point for HVP
        X1_single = self.X1[:, 0:1]  # (nvars, 1)
        direction = self.bkd().array(np.random.randn(self.nvars, 1))
        direction = direction / self.bkd().norm(direction)
        direction_flat = self.bkd().reshape(direction, (self.nvars,))

        hvp = product.hvp_wrt_x1(X1_single, self.X2, direction_flat)

        # Check shape: (n1, n2, nvars)
        self.assertEqual(hvp.shape, (1, self.nsamples2, self.nvars))

        # Check finiteness
        self.assertTrue(self.bkd().all_bool(self.bkd().isfinite(hvp)))


class TestSumKernel(Generic[Array], unittest.TestCase):
    """
    Base test class for SumKernel.

    Derived classes must implement the bkd() method to provide the backend.
    """

    def setUp(self) -> None:
        """
        Set up test environment for SumKernel.
        """
        np.random.seed(42)
        self.nvars = 2
        self.nsamples1 = 5
        self.nsamples2 = 4

        # Create two Matern kernels for composition
        self.kernel1 = MaternKernel(
            2.5,
            [1.0, 1.0],
            (0.1, 10.0),
            self.nvars,
            self.bkd()
        )
        self.kernel2 = MaternKernel(
            1.5,
            [0.5, 0.5],
            (0.1, 10.0),
            self.nvars,
            self.bkd()
        )

        # Create sample data
        self.X1 = self.bkd().array(np.random.randn(self.nvars, self.nsamples1))
        self.X2 = self.bkd().array(np.random.randn(self.nvars, self.nsamples2))

    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_initialization(self) -> None:
        """
        Test SumKernel initialization.
        """
        sum_kernel = SumKernel(self.kernel1, self.kernel2)
        self.assertEqual(sum_kernel.nvars(), self.nvars)
        self.assertIsNotNone(sum_kernel.bkd())

    def test_backend_mismatch_error(self) -> None:
        """
        Test that SumKernel raises error when backends don't match.
        """
        # Create kernel with different backend
        if isinstance(self.bkd(), NumpyBkd):
            other_bkd = TorchBkd()
        else:
            other_bkd = NumpyBkd()

        kernel_other = MaternKernel(
            2.5,
            [1.0, 1.0],
            (0.1, 10.0),
            self.nvars,
            other_bkd
        )

        with self.assertRaises(ValueError) as context:
            SumKernel(self.kernel1, kernel_other)

        self.assertIn("same backend type", str(context.exception))

    def test_hyperparameter_list_combination(self) -> None:
        """
        Test that hyperparameter lists are combined correctly.
        """
        sum_kernel = SumKernel(self.kernel1, self.kernel2)
        hyp_list = sum_kernel.hyp_list()

        # Should have parameters from both kernels
        nparams1 = self.kernel1.hyp_list().nparams()
        nparams2 = self.kernel2.hyp_list().nparams()
        self.assertEqual(hyp_list.nparams(), nparams1 + nparams2)

    def test_kernel_matrix_shape(self) -> None:
        """
        Test that SumKernel produces correct output shape.
        """
        sum_kernel = SumKernel(self.kernel1, self.kernel2)

        # Test with two different inputs
        K = sum_kernel(self.X1, self.X2)
        self.assertEqual(K.shape, (self.nsamples1, self.nsamples2))

        # Test with single input
        K_self = sum_kernel(self.X1)
        self.assertEqual(K_self.shape, (self.nsamples1, self.nsamples1))

    def test_kernel_matrix_values(self) -> None:
        """
        Test that SumKernel computes K1 + K2 correctly.
        """
        sum_kernel = SumKernel(self.kernel1, self.kernel2)

        K1 = self.kernel1(self.X1, self.X2)
        K2 = self.kernel2(self.X1, self.X2)
        K_sum = sum_kernel(self.X1, self.X2)

        expected = K1 + K2
        self.bkd().assert_allclose(K_sum, expected)

    def test_diagonal(self) -> None:
        """
        Test diagonal computation for SumKernel.
        """
        sum_kernel = SumKernel(self.kernel1, self.kernel2)

        diag1 = self.kernel1.diag(self.X1)
        diag2 = self.kernel2.diag(self.X1)
        diag_sum = sum_kernel.diag(self.X1)

        expected = diag1 + diag2
        self.bkd().assert_allclose(diag_sum, expected)

    def test_jacobian(self) -> None:
        """
        Test Jacobian computation for SumKernel.

        SumKernel should satisfy sum rule: d(K1+K2)/dx = dK1 + dK2
        """
        sum_kernel = SumKernel(self.kernel1, self.kernel2)

        jac = sum_kernel.jacobian(self.X1, self.X2)

        # Check shape
        self.assertEqual(jac.shape, (self.nsamples1, self.nsamples2, self.nvars))

        # Check finiteness
        self.assertTrue(self.bkd().all_bool(self.bkd().isfinite(jac)))

        # Manually compute using sum rule
        dK1 = self.kernel1.jacobian(self.X1, self.X2)
        dK2 = self.kernel2.jacobian(self.X1, self.X2)

        expected = dK1 + dK2
        self.bkd().assert_allclose(jac, expected)

    def test_param_jacobian(self) -> None:
        """
        Test parameter Jacobian for SumKernel.

        The parameter Jacobian should stack derivatives from both kernels.
        """
        sum_kernel = SumKernel(self.kernel1, self.kernel2)

        jac = sum_kernel.jacobian_wrt_params(self.X1)

        # Check shape
        nparams_total = self.kernel1.hyp_list().nparams() + self.kernel2.hyp_list().nparams()
        self.assertEqual(jac.shape, (self.nsamples1, self.nsamples1, nparams_total))

        # Check finiteness
        self.assertTrue(self.bkd().all_bool(self.bkd().isfinite(jac)))

    def test_operator_overloading(self) -> None:
        """
        Test that + operator creates SumKernel.
        """
        sum_kernel = self.kernel1 + self.kernel2
        self.assertIsInstance(sum_kernel, SumKernel)

        # Verify it produces same result as explicit construction
        sum_explicit = SumKernel(self.kernel1, self.kernel2)
        K1 = sum_kernel(self.X1, self.X2)
        K2 = sum_explicit(self.X1, self.X2)
        self.bkd().assert_allclose(K1, K2)

    def test_hvp_wrt_x1(self) -> None:
        """
        Test HVP computation for SumKernel using sum rule.
        """
        sum_kernel = SumKernel(self.kernel1, self.kernel2)

        # Single point for HVP
        X1_single = self.X1[:, 0:1]  # (nvars, 1)
        direction = self.bkd().array(np.random.randn(self.nvars, 1))
        direction = direction / self.bkd().norm(direction)
        direction_flat = self.bkd().reshape(direction, (self.nvars,))

        hvp = sum_kernel.hvp_wrt_x1(X1_single, self.X2, direction_flat)

        # Check shape: (n1, n2, nvars)
        self.assertEqual(hvp.shape, (1, self.nsamples2, self.nvars))

        # Check finiteness
        self.assertTrue(self.bkd().all_bool(self.bkd().isfinite(hvp)))


class TestNestedComposition(Generic[Array], unittest.TestCase):
    """
    Test nested composition of kernels.
    """

    def setUp(self) -> None:
        """
        Set up test environment for nested compositions.
        """
        np.random.seed(42)
        self.nvars = 2
        self.nsamples = 5

        # Create three kernels
        self.k1 = MaternKernel(
            2.5, [1.0, 1.0], (0.1, 10.0), self.nvars, self.bkd()
        )
        self.k2 = MaternKernel(
            1.5, [0.5, 0.5], (0.1, 10.0), self.nvars, self.bkd()
        )
        self.k3 = MaternKernel(
            3/2, [2.0, 2.0], (0.1, 10.0), self.nvars, self.bkd()
        )

        self.X = self.bkd().array(np.random.randn(self.nvars, self.nsamples))

    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_nested_sum_product(self) -> None:
        """
        Test nested composition: (k1 + k2) * k3
        """
        nested = (self.k1 + self.k2) * self.k3

        # Compute manually
        K1 = self.k1(self.X, self.X)
        K2 = self.k2(self.X, self.X)
        K3 = self.k3(self.X, self.X)
        expected = (K1 + K2) * K3

        K_nested = nested(self.X, self.X)
        self.bkd().assert_allclose(K_nested, expected)

    def test_nested_product_sum(self) -> None:
        """
        Test nested composition: k1 * k2 + k3
        """
        nested = self.k1 * self.k2 + self.k3

        # Compute manually
        K1 = self.k1(self.X, self.X)
        K2 = self.k2(self.X, self.X)
        K3 = self.k3(self.X, self.X)
        expected = K1 * K2 + K3

        K_nested = nested(self.X, self.X)
        self.bkd().assert_allclose(K_nested, expected)

    def test_deeply_nested(self) -> None:
        """
        Test deeply nested composition: (k1 + k2) * (k2 + k3)
        """
        nested = (self.k1 + self.k2) * (self.k2 + self.k3)

        # Compute manually
        K1 = self.k1(self.X, self.X)
        K2 = self.k2(self.X, self.X)
        K3 = self.k3(self.X, self.X)
        expected = (K1 + K2) * (K2 + K3)

        K_nested = nested(self.X, self.X)
        self.bkd().assert_allclose(K_nested, expected)


# NumPy implementations
class TestProductKernelNumpy(TestProductKernel[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestSumKernelNumpy(TestSumKernel[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestNestedCompositionNumpy(TestNestedComposition[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch implementations
class TestProductKernelTorch(TestProductKernel[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestSumKernelTorch(TestSumKernel[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestNestedCompositionTorch(TestNestedComposition[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


# Custom test loader to exclude base classes
def load_tests(
    loader: unittest.TestLoader, tests, pattern: str
) -> unittest.TestSuite:
    """
    Custom test loader to exclude the base classes.
    """
    test_suite = unittest.TestSuite()
    for test_class in [
        TestProductKernelNumpy,
        TestProductKernelTorch,
        TestSumKernelNumpy,
        TestSumKernelTorch,
        TestNestedCompositionNumpy,
        TestNestedCompositionTorch,
    ]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner()
    runner.run(suite)
