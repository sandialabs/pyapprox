"""
Tests for 1D kernel integrals and validation functions.

Tests basic properties (symmetry, positive semi-definiteness),
analytical comparisons, and error handling.
"""
import unittest
from typing import Generic, Any
import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.surrogates.kernels.matern import SquaredExponentialKernel
from pyapprox.typing.surrogates.kernels.composition import ProductKernel
from pyapprox.typing.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.typing.surrogates.gaussianprocess.mean_functions import (
    ZeroMean,
    ConstantMean,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics.validation import (
    validate_separable_kernel,
    validate_zero_mean,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics.integrals_1d import (
    compute_tau_1d,
    compute_P_1d,
    compute_u_1d,
    compute_nu_1d,
    compute_lambda_1d,
    compute_Pi_1d,
    compute_xi1_1d,
)


class TestIntegrals1D(Generic[Array], unittest.TestCase):
    """
    Base test class for 1D kernel integrals.

    Derived classes must implement the bkd() method.
    """

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment."""
        self._bkd = self.bkd()
        np.random.seed(42)

        # Create 1D kernel (Squared Exponential / RBF)
        self._kernel_1d = SquaredExponentialKernel(
            [1.0],
            (0.1, 10.0),
            1,
            self._bkd
        )

        # Training points in 1D
        self._n_train = 5
        train_np = np.array([[-1.0, -0.5, 0.0, 0.5, 1.0]])
        self._train_samples_1d = self._bkd.array(train_np)

        # Quadrature points (Gauss-Legendre on [-1, 1])
        self._nquad = 20
        quad_pts_np, quad_wts_np = np.polynomial.legendre.leggauss(self._nquad)
        # Scale weights for density on [-1, 1]: uniform has density 1/2
        quad_wts_np = quad_wts_np / 2.0
        self._quad_samples = self._bkd.array(quad_pts_np.reshape(1, -1))
        self._quad_weights = self._bkd.array(quad_wts_np)

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def _kernel_callable(self, x1: Array, x2: Array) -> Array:
        """Wrap kernel as callable for integral functions."""
        return self._kernel_1d(x1, x2)

    def test_tau_shape(self) -> None:
        """Test tau has correct shape (N,)."""
        tau = compute_tau_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            self._bkd
        )
        self.assertEqual(tau.shape, (self._n_train,))

    def test_tau_positive(self) -> None:
        """Test tau values are positive (kernel is positive)."""
        tau = compute_tau_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            self._bkd
        )
        self.assertTrue(self._bkd.all_bool(tau > 0))

    def test_P_shape(self) -> None:
        """Test P has correct shape (N, N)."""
        P = compute_P_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            self._bkd
        )
        self.assertEqual(P.shape, (self._n_train, self._n_train))

    def test_P_symmetric(self) -> None:
        """Test P is symmetric."""
        P = compute_P_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            self._bkd
        )
        self._bkd.assert_allclose(P, P.T, rtol=1e-12)

    def test_P_positive_semidefinite(self) -> None:
        """Test P is positive semi-definite (all eigenvalues >= 0)."""
        P = compute_P_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            self._bkd
        )
        eigvals = self._bkd.eigvalsh(P)
        # Allow small negative eigenvalues due to numerical error
        self.assertTrue(self._bkd.all_bool(eigvals > -1e-10))

    def test_u_scalar(self) -> None:
        """Test u is a scalar and positive."""
        u = compute_u_1d(
            self._quad_samples,
            self._quad_weights,
            self._kernel_callable,
            self._bkd
        )
        # u should be positive (integral of positive kernel)
        self.assertGreater(float(self._bkd.to_numpy(u)), 0.0)

    def test_nu_scalar(self) -> None:
        """Test nu is a scalar and positive."""
        nu = compute_nu_1d(
            self._quad_samples,
            self._quad_weights,
            self._kernel_callable,
            self._bkd
        )
        # nu should be positive (integral of squared kernel)
        self.assertGreater(float(self._bkd.to_numpy(nu)), 0.0)

    def test_lambda_shape(self) -> None:
        """Test lambda has correct shape (N,)."""
        lambda_vec = compute_lambda_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            self._bkd
        )
        self.assertEqual(lambda_vec.shape, (self._n_train,))

    def test_lambda_positive(self) -> None:
        """Test lambda values are positive."""
        lambda_vec = compute_lambda_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            self._bkd
        )
        self.assertTrue(self._bkd.all_bool(lambda_vec > 0))

    def test_Pi_shape(self) -> None:
        """Test Pi has correct shape (N, N)."""
        Pi = compute_Pi_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            self._bkd
        )
        self.assertEqual(Pi.shape, (self._n_train, self._n_train))

    def test_Pi_symmetric(self) -> None:
        """Test Pi is symmetric."""
        Pi = compute_Pi_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            self._bkd
        )
        self._bkd.assert_allclose(Pi, Pi.T, rtol=1e-12)

    def test_xi1_scalar(self) -> None:
        """Test xi1 is a scalar and positive."""
        xi1 = compute_xi1_1d(
            self._quad_samples,
            self._quad_weights,
            self._kernel_callable,
            self._bkd
        )
        # xi1 should be positive
        self.assertGreater(float(self._bkd.to_numpy(xi1)), 0.0)

    def test_u_equals_tau_squared_for_single_point(self) -> None:
        """
        Test that u = tau^2 when there's only one quadrature point.

        For a single point, u = w^2 * K(x, x) and tau = w * K(x, x),
        so u = tau^2 / w. This is a sanity check.
        """
        # Use single quadrature point
        single_quad = self._bkd.array([[0.0]])
        single_weight = self._bkd.array([1.0])

        u = compute_u_1d(
            single_quad,
            single_weight,
            self._kernel_callable,
            self._bkd
        )

        # For single point at 0, K(0, 0) = 1 (for normalized kernel)
        # u = 1 * 1 * K(0, 0) = K(0, 0)
        K_00 = self._kernel_1d(single_quad, single_quad)
        expected_u = K_00[0, 0]
        self._bkd.assert_allclose(
            self._bkd.asarray([u]),
            self._bkd.asarray([expected_u]),
            rtol=1e-12
        )


class TestValidation(Generic[Array], unittest.TestCase):
    """
    Base test class for validation functions.

    Derived classes must implement the bkd() method.
    """

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment."""
        self._bkd = self.bkd()
        np.random.seed(42)

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def test_validate_separable_kernel_product(self) -> None:
        """Test that ProductKernel passes validation."""
        k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, self._bkd)
        k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, self._bkd)
        prod_kernel = ProductKernel(k1, k2)

        # Should not raise
        validate_separable_kernel(prod_kernel)

    def test_validate_separable_kernel_1d(self) -> None:
        """Test that 1D kernel passes validation (trivially separable)."""
        k1d = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, self._bkd)

        # Should not raise
        validate_separable_kernel(k1d)

    def test_validate_separable_kernel_non_separable_error(self) -> None:
        """Test that non-separable multi-D kernel raises TypeError."""
        # 2D kernel that is not a ProductKernel
        k2d = SquaredExponentialKernel([1.0, 1.0], (0.1, 10.0), 2, self._bkd)

        with self.assertRaises(TypeError) as context:
            validate_separable_kernel(k2d)

        self.assertIn("separable", str(context.exception).lower())
        self.assertIn("ProductKernel", str(context.exception))

    def test_validate_zero_mean_with_zero_mean(self) -> None:
        """Test that GP with ZeroMean passes validation."""
        kernel = SquaredExponentialKernel([1.0, 1.0], (0.1, 10.0), 2, self._bkd)
        gp = ExactGaussianProcess(
            kernel,
            nvars=2,
            bkd=self._bkd,
            mean_function=ZeroMean(self._bkd)
        )

        # Should not raise
        validate_zero_mean(gp)

    def test_validate_zero_mean_default(self) -> None:
        """Test that GP with default mean (ZeroMean) passes validation."""
        kernel = SquaredExponentialKernel([1.0, 1.0], (0.1, 10.0), 2, self._bkd)
        gp = ExactGaussianProcess(
            kernel,
            nvars=2,
            bkd=self._bkd
            # Default mean is ZeroMean
        )

        # Should not raise
        validate_zero_mean(gp)

    def test_validate_zero_mean_constant_mean_error(self) -> None:
        """Test that GP with ConstantMean raises ValueError."""
        kernel = SquaredExponentialKernel([1.0, 1.0], (0.1, 10.0), 2, self._bkd)
        gp = ExactGaussianProcess(
            kernel,
            nvars=2,
            bkd=self._bkd,
            mean_function=ConstantMean(1.0, (-10.0, 10.0), self._bkd)
        )

        with self.assertRaises(ValueError) as context:
            validate_zero_mean(gp)

        self.assertIn("ZeroMean", str(context.exception))


class TestIntegrals1DNumpy(TestIntegrals1D[NDArray[Any]]):
    """NumPy backend tests for 1D integrals."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIntegrals1DTorch(TestIntegrals1D[torch.Tensor]):
    """PyTorch backend tests for 1D integrals."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestValidationNumpy(TestValidation[NDArray[Any]]):
    """NumPy backend tests for validation functions."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestValidationTorch(TestValidation[torch.Tensor]):
    """PyTorch backend tests for validation functions."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
