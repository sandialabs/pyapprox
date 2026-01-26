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
    SeparableProductKernel,
)
from pyapprox.typing.surrogates.kernels.matern import (
    Matern52Kernel,
    Matern32Kernel,
    SquaredExponentialKernel,
)
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.typing.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianFromCallable,
)


class TestProductKernel(Generic[Array], unittest.TestCase):
    """
    Base test class for ProductKernel.

    Derived classes must implement the bkd() method to provide the backend.
    """

    __test__ = False

    def setUp(self) -> None:
        """
        Set up test environment for ProductKernel.
        """
        np.random.seed(42)
        self.nvars = 2
        self.nsamples1 = 5
        self.nsamples2 = 4

        # Create two Matern kernels for composition
        self.kernel1 = Matern52Kernel(
            [1.0, 1.0],
            (0.1, 10.0),
            self.nvars,
            self.bkd()
        )
        self.kernel2 = Matern32Kernel(
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

        kernel_other = Matern52Kernel(
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

    __test__ = False

    def setUp(self) -> None:
        """
        Set up test environment for SumKernel.
        """
        np.random.seed(42)
        self.nvars = 2
        self.nsamples1 = 5
        self.nsamples2 = 4

        # Create two Matern kernels for composition
        self.kernel1 = Matern52Kernel(
            [1.0, 1.0],
            (0.1, 10.0),
            self.nvars,
            self.bkd()
        )
        self.kernel2 = Matern32Kernel(
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

        kernel_other = Matern52Kernel(
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

    __test__ = False

    def setUp(self) -> None:
        """
        Set up test environment for nested compositions.
        """
        np.random.seed(42)
        self.nvars = 2
        self.nsamples = 5

        # Create three kernels
        self.k1 = Matern52Kernel(
            [1.0, 1.0], (0.1, 10.0), self.nvars, self.bkd()
        )
        self.k2 = Matern32Kernel(
            [0.5, 0.5], (0.1, 10.0), self.nvars, self.bkd()
        )
        self.k3 = Matern32Kernel(
            [2.0, 2.0], (0.1, 10.0), self.nvars, self.bkd()
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


class TestSeparableProductKernel(Generic[Array], unittest.TestCase):
    """
    Base test class for SeparableProductKernel.

    Tests the separable product kernel where each 1D kernel operates
    on a different dimension: k(x, y) = ∏_i k_i(x_i, y_i)
    """

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment."""
        np.random.seed(42)
        self._bkd = self.bkd()

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def test_factorization(self) -> None:
        """Verify k(x,y) = k1(x1,y1) * k2(x2,y2) for SeparableProductKernel."""
        bkd = self._bkd

        l1, l2 = 1.5, 2.0
        k1 = SquaredExponentialKernel([l1], (0.01, 100.0), 1, bkd)
        k2 = SquaredExponentialKernel([l2], (0.01, 100.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        # Test points
        x = bkd.array([[0.3], [0.7]])  # Point (0.3, 0.7)
        y = bkd.array([[0.8], [0.2]])  # Point (0.8, 0.2)

        # Full kernel evaluation
        k_full = kernel(x, y)

        # Factored evaluation
        k1_val = k1(bkd.array([[0.3]]), bkd.array([[0.8]]))
        k2_val = k2(bkd.array([[0.7]]), bkd.array([[0.2]]))
        k_factored = k1_val * k2_val

        bkd.assert_allclose(k_full, k_factored, rtol=1e-12)

    def test_batch(self) -> None:
        """Test SeparableProductKernel with batched inputs."""
        bkd = self._bkd

        k1 = SquaredExponentialKernel([1.0], (0.01, 100.0), 1, bkd)
        k2 = SquaredExponentialKernel([2.0], (0.01, 100.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        # Multiple test points
        X1 = bkd.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  # 3 points in 2D
        X2 = bkd.array([[0.7, 0.9], [0.8, 1.0]])  # 2 points in 2D

        K = kernel(X1, X2)

        # Should have shape (3, 2)
        self.assertEqual(K.shape, (3, 2))

        # Verify each element manually
        for i in range(3):
            for j in range(2):
                k1_val = k1(
                    bkd.array([[X1[0, i]]]),
                    bkd.array([[X2[0, j]]])
                )
                k2_val = k2(
                    bkd.array([[X1[1, i]]]),
                    bkd.array([[X2[1, j]]])
                )
                expected = k1_val[0, 0] * k2_val[0, 0]
                bkd.assert_allclose(
                    bkd.asarray([K[i, j]]),
                    bkd.asarray([expected]),
                    rtol=1e-12
                )

    def test_nvars(self) -> None:
        """Test nvars matches number of 1D kernels."""
        bkd = self._bkd

        k1 = SquaredExponentialKernel([1.0], (0.01, 100.0), 1, bkd)
        k2 = SquaredExponentialKernel([2.0], (0.01, 100.0), 1, bkd)
        k3 = SquaredExponentialKernel([0.5], (0.01, 100.0), 1, bkd)

        kernel_2d = SeparableProductKernel([k1, k2], bkd)
        kernel_3d = SeparableProductKernel([k1, k2, k3], bkd)

        self.assertEqual(kernel_2d.nvars(), 2)
        self.assertEqual(kernel_3d.nvars(), 3)

    def test_get_kernel_1d(self) -> None:
        """Test get_kernel_1d returns correct kernels."""
        bkd = self._bkd

        k1 = SquaredExponentialKernel([1.5], (0.01, 100.0), 1, bkd)
        k2 = SquaredExponentialKernel([2.5], (0.01, 100.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        self.assertIs(kernel.get_kernel_1d(0), k1)
        self.assertIs(kernel.get_kernel_1d(1), k2)

    def test_diag(self) -> None:
        """Test diagonal computation."""
        bkd = self._bkd

        k1 = SquaredExponentialKernel([1.0], (0.01, 100.0), 1, bkd)
        k2 = SquaredExponentialKernel([2.0], (0.01, 100.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        X = bkd.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        diag = kernel.diag(X)

        # Diagonal should be product of 1D diagonals
        # For RBF kernels, diag = 1 for all points
        expected = bkd.ones((3,))
        bkd.assert_allclose(diag, expected, rtol=1e-12)

    def test_jacobian_wrt_params(self) -> None:
        """Test jacobian_wrt_params for SeparableProductKernel using DerivativeChecker."""
        bkd = self._bkd

        k1 = SquaredExponentialKernel([1.0], (0.01, 100.0), 1, bkd)
        k2 = SquaredExponentialKernel([2.0], (0.01, 100.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        # Test points
        n = 5
        samples = bkd.array(np.random.rand(2, n))

        # Get jacobian and check shape/finiteness
        jac = kernel.jacobian_wrt_params(samples)
        nparams = kernel.hyp_list().nparams()
        self.assertEqual(jac.shape, (n, n, nparams))
        self.assertTrue(bkd.all_bool(bkd.isfinite(jac)))

        # Verify using DerivativeChecker
        vec = bkd.ones((n, 1))

        def fun(p):
            kernel.hyp_list().set_active_values(p[:, 0])
            return kernel(samples, samples) @ vec

        def jac_fn(p):
            kernel.hyp_list().set_active_values(p[:, 0])
            return bkd.einsum(
                "ijk,jl->ik",
                kernel.jacobian_wrt_params(samples),
                vec,
            )

        function_object = FunctionWithJacobianFromCallable(
            nqoi=n,
            nvars=nparams,
            fun=fun,
            jacobian=jac_fn,
            bkd=bkd,
        )
        checker = DerivativeChecker(function_object)
        sample = kernel.hyp_list().get_active_values()[:, None]
        errors = checker.check_derivatives(sample, verbosity=0)
        self.assertLess(checker.error_ratio(errors[0]), 1e-6)

    def test_hyperparameters_combined(self) -> None:
        """Test hyperparameters are combined from all 1D kernels."""
        bkd = self._bkd

        k1 = SquaredExponentialKernel([1.0], (0.01, 100.0), 1, bkd)
        k2 = SquaredExponentialKernel([2.0], (0.01, 100.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        # Each RBF kernel has 1 length scale parameter
        self.assertEqual(kernel.hyp_list().nparams(), 2)

    def test_invalid_nvars_raises_error(self) -> None:
        """Test that non-1D kernels raise ValueError."""
        bkd = self._bkd

        k1 = SquaredExponentialKernel([1.0], (0.01, 100.0), 1, bkd)
        k2_invalid = SquaredExponentialKernel([1.0, 2.0], (0.01, 100.0), 2, bkd)

        with self.assertRaises(ValueError) as context:
            SeparableProductKernel([k1, k2_invalid], bkd)

        self.assertIn("nvars=1", str(context.exception))


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


class TestSeparableProductKernelNumpy(TestSeparableProductKernel[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


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


class TestSeparableProductKernelTorch(TestSeparableProductKernel[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


from pyapprox.typing.util.test_utils import load_tests


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner()
    runner.run(suite)
