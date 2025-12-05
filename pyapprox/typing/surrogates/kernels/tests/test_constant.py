import unittest
from typing import Generic, Any

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.surrogates.kernels.constant import ConstantKernel
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.typing.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianFromCallable,
)


class TestConstantKernel(Generic[Array], unittest.TestCase):
    """
    Base test class for ConstantKernel.

    Derived classes must implement the bkd() method to provide the backend.
    """

    def setUp(self) -> None:
        """
        Set up test environment for ConstantKernel.
        """
        np.random.seed(42)
        self.nvars = 2
        self.nsamples1 = 5
        self.nsamples2 = 4
        self.constant_value = 2.5
        self.constant_bounds = (0.1, 10.0)

        # Create kernel
        self.kernel = ConstantKernel(
            self.constant_value,
            self.constant_bounds,
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
        Test ConstantKernel initialization.
        """
        kernel = ConstantKernel(
            self.constant_value,
            self.constant_bounds,
            self.bkd()
        )
        self.assertIsNotNone(kernel.bkd())
        self.assertEqual(kernel.nvars(), 0)  # ConstantKernel has no spatial dependence

    def test_hyperparameter_list(self) -> None:
        """
        Test that hyperparameter list is set up correctly.
        """
        hyp_list = self.kernel.hyp_list()
        self.assertEqual(hyp_list.nparams(), 1)  # Single constant parameter

    def test_kernel_matrix_shape(self) -> None:
        """
        Test that ConstantKernel produces correct output shape.
        """
        # Test with two different inputs
        K = self.kernel(self.X1, self.X2)
        self.assertEqual(K.shape, (self.nsamples1, self.nsamples2))

        # Test with single input
        K_self = self.kernel(self.X1)
        self.assertEqual(K_self.shape, (self.nsamples1, self.nsamples1))

    def test_kernel_matrix_values(self) -> None:
        """
        Test that ConstantKernel returns constant values everywhere.
        """
        K = self.kernel(self.X1, self.X2)

        # All entries should equal the constant value
        expected = self.bkd().full(
            (self.nsamples1, self.nsamples2),
            self.constant_value
        )
        self.bkd().assert_allclose(K, expected)

    def test_kernel_matrix_single_input(self) -> None:
        """
        Test that ConstantKernel returns constant values for single input.
        """
        K = self.kernel(self.X1)

        # All entries should equal the constant value
        expected = self.bkd().full(
            (self.nsamples1, self.nsamples1),
            self.constant_value
        )
        self.bkd().assert_allclose(K, expected)

    def test_diagonal(self) -> None:
        """
        Test diagonal computation for ConstantKernel.
        """
        diag = self.kernel.diag(self.X1)

        # Diagonal should be vector of constant values
        expected = self.bkd().full((self.nsamples1,), self.constant_value)
        self.bkd().assert_allclose(diag, expected)

    def test_jacobian_is_zero(self) -> None:
        """
        Test that Jacobian w.r.t. inputs is zero.

        ConstantKernel has no spatial dependence, so dK/dX = 0.
        """
        jac = self.kernel.jacobian(self.X1, self.X2)

        # Should be shape (n1, n2, nvars) and all zeros
        expected_shape = (self.nsamples1, self.nsamples2, self.nvars)
        self.assertEqual(jac.shape, expected_shape)

        expected = self.bkd().zeros(expected_shape)
        self.bkd().assert_allclose(jac, expected)

    def test_param_jacobian(self) -> None:
        """
        Test parameter Jacobian for ConstantKernel using DerivativeChecker.

        For log-parameterized constant: dK/d(log_c) = c
        """
        # Get active parameter values
        params = self.kernel.hyp_list().get_active_values()

        # Create wrapper function
        def kernel_func_params(p: Array) -> Array:
            # p has shape (nvars, 1), need to flatten to 1D for set_active_values
            p_flat = self.bkd().flatten(p)
            self.kernel.hyp_list().set_active_values(p_flat)
            K = self.kernel(self.X1, self.X1)
            K_flat = self.bkd().flatten(K)
            # Reshape to (nqoi, nsamp) where nsamp = 1
            return self.bkd().reshape(K_flat, (K_flat.shape[0], 1))

        def kernel_jac_params(p: Array) -> Array:
            # p has shape (nvars, 1), need to flatten to 1D for set_active_values
            p_flat = self.bkd().flatten(p)
            self.kernel.hyp_list().set_active_values(p_flat)
            jac = self.kernel.jacobian_wrt_params(self.X1)
            # jac has shape (n, n, nparams)
            # Flatten to (n*n, nparams) which equals (nqoi, nvars)
            n = self.X1.shape[1]
            jac_flat = self.bkd().reshape(jac, (n * n, jac.shape[2]))
            return jac_flat

        func_with_jac = FunctionWithJacobianFromCallable(
            self.nsamples1 * self.nsamples1,
            self.kernel.hyp_list().nactive_params(),
            kernel_func_params,
            kernel_jac_params,
            self.bkd()
        )

        checker = DerivativeChecker(func_with_jac)
        params_reshaped = self.bkd().reshape(params, (params.shape[0], 1))
        errors = checker.check_derivatives(params_reshaped)
        self.assertLessEqual(checker.error_ratio(errors[0]), 2e-7)

    def test_param_jacobian_values(self) -> None:
        """
        Test that parameter Jacobian has correct values.

        For log-parameterized constant: dK/d(log_c) = c
        All entries should equal the constant value.
        """
        jac = self.kernel.jacobian_wrt_params(self.X1)

        # Should be shape (n, n, 1) with all entries = constant_value
        expected_shape = (self.nsamples1, self.nsamples1, 1)
        self.assertEqual(jac.shape, expected_shape)

        expected = self.bkd().full(expected_shape, self.constant_value)
        self.bkd().assert_allclose(jac, expected)

    def test_hyperparameter_update(self) -> None:
        """
        Test that kernel values update when hyperparameters change.
        """
        # Get initial kernel matrix
        K_before = self.kernel(self.X1, self.X2)

        # Update constant value
        new_constant = 5.0
        # LogHyperParameter expects log values
        self.kernel.hyp_list().set_active_values(
            self.bkd().array([np.log(new_constant)])
        )

        # Get new kernel matrix
        K_after = self.kernel(self.X1, self.X2)

        # Values should have changed
        expected = self.bkd().full(
            (self.nsamples1, self.nsamples2),
            new_constant
        )
        self.bkd().assert_allclose(K_after, expected)

        # Should be different from before
        self.assertFalse(
            self.bkd().allclose(K_before, K_after)
        )

    def test_composition_with_other_kernel(self) -> None:
        """
        Test that ConstantKernel composes correctly with other kernels.
        """
        from pyapprox.typing.surrogates.kernels.matern import MaternKernel

        # Create a Matern kernel
        matern = MaternKernel(
            2.5,
            [1.0, 1.0],
            (0.1, 10.0),
            self.nvars,
            self.bkd()
        )

        # Product: scales the Matern kernel
        product = matern * self.kernel
        K_product = product(self.X1, self.X2)

        # Should equal Matern * constant
        K_matern = matern(self.X1, self.X2)
        expected = K_matern * self.constant_value
        self.bkd().assert_allclose(K_product, expected)

        # Sum: adds constant offset
        sum_kernel = matern + self.kernel
        K_sum = sum_kernel(self.X1, self.X2)

        # Should equal Matern + constant
        expected_sum = K_matern + self.constant_value
        self.bkd().assert_allclose(K_sum, expected_sum)

    def test_fixed_hyperparameter(self) -> None:
        """
        Test that fixed hyperparameters are not optimized.
        """
        fixed_kernel = ConstantKernel(
            self.constant_value,
            self.constant_bounds,
            self.bkd(),
            fixed=True
        )

        # Fixed kernels should have 0 active parameters
        self.assertEqual(fixed_kernel.hyp_list().nactive_params(), 0)

    def test_repr(self) -> None:
        """
        Test the string representation of ConstantKernel.
        """
        repr_str = repr(self.kernel)
        self.assertIn("ConstantKernel", repr_str)


# NumPy implementation
class TestConstantKernelNumpy(TestConstantKernel[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch implementation
class TestConstantKernelTorch(TestConstantKernel[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


# Custom test loader to exclude base class
def load_tests(
    loader: unittest.TestLoader, tests, pattern: str
) -> unittest.TestSuite:
    """
    Custom test loader to exclude the base class.
    """
    test_suite = unittest.TestSuite()
    for test_class in [
        TestConstantKernelNumpy,
        TestConstantKernelTorch,
    ]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner()
    runner.run(suite)
