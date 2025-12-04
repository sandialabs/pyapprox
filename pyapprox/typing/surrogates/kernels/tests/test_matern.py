import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.surrogates.kernels.matern import MaternKernel
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.typing.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianFromCallable,
)


class TestMaternKernel(Generic[Array], unittest.TestCase):
    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific
        backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def setUp(self) -> None:
        """
        Set up the test environment for MaternKernel.
        """
        np.random.seed(1)
        self.nu_values = [np.inf, 5 / 2, 3 / 2]
        self.lenscale = self.bkd().array(
            [1.0, 2.0]
        )  # Vector-valued length scale
        self.lenscale_bounds = (0.1, 10.0)
        self.nvars = 2
        self.X1 = self.bkd().array([[0.0, 0.5, 1.0], [0.0, 0.25, 0.75]])
        self.X2 = self.bkd().array([[0.0, 0.5, 1.0], [0.0, 0.25, 0.75]])

    def test_kernel_matrix(self) -> None:
        """
        Test the computation of the kernel matrix for supported nu values.
        """
        for nu in self.nu_values:
            kernel = MaternKernel(
                nu=nu,
                lenscale=self.lenscale,
                lenscale_bounds=self.lenscale_bounds,
                nvars=self.nvars,
                bkd=self.bkd(),
            )
            kernel_matrix = kernel(self.X1, self.X2)
            self.assertEqual(
                kernel_matrix.shape, (self.X1.shape[1], self.X2.shape[1])
            )
            self.assertTrue(
                self.bkd().all_bool(kernel_matrix >= 0)
            )  # Kernel values must be non-negative

    def test_diagonal(self) -> None:
        """
        Test the computation of the diagonal of the kernel matrix.
        """
        for nu in self.nu_values:
            kernel = MaternKernel(
                nu=nu,
                lenscale=self.lenscale,
                lenscale_bounds=self.lenscale_bounds,
                nvars=self.nvars,
                bkd=self.bkd(),
            )
            diag = kernel.diag(self.X1)
            self.assertEqual(diag.shape, (self.X1.shape[1],))
            self.bkd().assert_allclose(diag, self.bkd().ones(self.X1.shape[1]))

    def test_jacobian(self) -> None:
        """
        Test the computation of the Jacobian of the kernel with respect to
        input data.
        """
        for nu in self.nu_values:
            kernel = MaternKernel(
                nu=nu,
                lenscale=self.lenscale,
                lenscale_bounds=self.lenscale_bounds,
                nvars=self.nvars,
                bkd=self.bkd(),
            )
            jacobian = kernel.jacobian(self.X1, self.X2)
            self.assertEqual(
                jacobian.shape,
                (self.X1.shape[1], self.X2.shape[1], self.nvars),
            )
            self.assertTrue(
                self.bkd().all_bool(self.bkd().isfinite(jacobian))
            )  # Jacobian values must be finite

            # Wrap the function using FunctionWithJacobianFromCallable
            vec = self.bkd().ones((self.X2.shape[1], 1))
            function_object = FunctionWithJacobianFromCallable(
                nqoi=1,  # just computing jacobian for one sample
                nvars=self.nvars,
                fun=lambda x: kernel(
                    self.bkd().reshape(x, (self.nvars, -1)), self.X2
                )
                @ vec,
                jacobian=lambda x: self.bkd().einsum(
                    "ijk,jl->ik",
                    kernel.jacobian(
                        self.bkd().reshape(x, (self.nvars, -1)), self.X2
                    ),
                    vec,
                ),
                bkd=self.bkd(),
            )
            # Initialize DerivativeChecker
            checker = DerivativeChecker(function_object)
            # Check derivatives
            sample = self.bkd().flatten(self.X1[:, :1])[:, None]
            errors = checker.check_derivatives(sample)

            # Assert that the gradient errors are below a tolerance
            self.assertLessEqual(checker.error_ratio(errors[0]), 2e-7)

    def test_param_jacobian(self) -> None:
        """
        Test the computation of the Jacobian of the kernel with respect to
        hyperparameters.
        """
        for nu in self.nu_values:
            kernel = MaternKernel(
                nu=nu,
                lenscale=self.lenscale,
                lenscale_bounds=self.lenscale_bounds,
                nvars=self.nvars,
                bkd=self.bkd(),
            )
            jacobian_params = kernel.jacobian_wrt_params(self.X1)
            self.assertEqual(
                jacobian_params.shape,
                (self.X1.shape[1], self.X1.shape[1], self.nvars),
            )
            self.assertTrue(
                self.bkd().all_bool(self.bkd().isfinite(jacobian_params))
            )  # Jacobian values must be finite

            # Wrap the function using FunctionWithJacobianFromCallable
            vec = self.bkd().ones((self.X1.shape[1], 1))

            def fun(p):
                kernel.hyp_list().hyperparameters()[0].set_active_values(
                    p[:, 0]
                )
                return kernel(self.X1, self.X1) @ vec

            def jac(p):
                kernel.hyp_list().hyperparameters()[0].set_active_values(
                    p[:, 0]
                )
                return self.bkd().einsum(
                    "ijk,jl->ik",
                    kernel.jacobian_wrt_params(self.X1),
                    vec,
                )

            function_object = FunctionWithJacobianFromCallable(
                nqoi=self.X1.shape[1],
                nvars=self.nvars,
                fun=fun,
                jacobian=jac,
                bkd=self.bkd(),
            )
            print(kernel)
            # Initialize DerivativeChecker
            checker = DerivativeChecker(function_object)
            # Check derivatives
            sample = (
                kernel.hyp_list()
                .hyperparameters()[0]
                .get_active_values()[:, None]
            )
            errors = checker.check_derivatives(sample, verbosity=1)
            # Assert that the gradient errors are below a tolerance
            print(checker.error_ratio(errors[0]))
            self.assertLessEqual(checker.error_ratio(errors[0]), 1e-7)

    def test_repr(self) -> None:
        """
        Test the string representation of the MaternKernel.
        """
        kernel = MaternKernel(
            nu=3 / 2,
            lenscale=self.lenscale,
            lenscale_bounds=self.lenscale_bounds,
            nvars=self.nvars,
            bkd=self.bkd(),
        )
        repr_str = repr(kernel)
        self.assertIn("MaternKernel", repr_str)
        self.assertIn("nu=1.5", repr_str)
        self.assertIn("lenscale", repr_str)


# Derived test class for NumPy backend
class TestMaternKernelNumpy(TestMaternKernel[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestMaternKernelTorch(TestMaternKernel[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


# Custom test loader to exclude the base class
def load_tests(
    loader: unittest.TestLoader, tests, pattern: str
) -> unittest.TestSuite:
    """
    Custom test loader to exclude the base class Function1D.
    """
    test_suite = unittest.TestSuite()
    for test_class in [TestMaternKernelNumpy, TestMaternKernelTorch]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


# Main block to explicitly run tests using the custom loader
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner()
    runner.run(suite)
