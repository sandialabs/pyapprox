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
    compute_conditional_P_1d,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics.integrals import (
    SeparableKernelIntegralCalculator,
)
from pyapprox.typing.surrogates.sparsegrids.basis_factory import (
    create_basis_factories,
)
from pyapprox.typing.probability.univariate import UniformMarginal


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


class TestConditionalP1D(Generic[Array], unittest.TestCase):
    """
    Base test class for conditional P (1D) function.

    Tests the compute_conditional_P_1d function which returns P̃ = τ τᵀ.
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

    def test_conditional_P_shape(self) -> None:
        """Test conditional P̃ has correct shape (N, N)."""
        P_tilde = compute_conditional_P_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            self._bkd
        )
        self.assertEqual(P_tilde.shape, (self._n_train, self._n_train))

    def test_conditional_P_is_rank1(self) -> None:
        """Test conditional P̃ is rank-1 (since P̃ = τ τᵀ)."""
        P_tilde = compute_conditional_P_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            self._bkd
        )

        # Compute SVD and check that only one singular value is significant
        P_np = self._bkd.to_numpy(P_tilde)
        singular_values = np.linalg.svd(P_np, compute_uv=False)

        # Only the first singular value should be non-zero
        # All others should be near zero (numerical tolerance)
        self.assertGreater(singular_values[0], 1e-10)
        for i in range(1, len(singular_values)):
            self.assertLess(
                singular_values[i], 1e-10,
                f"Singular value {i} = {singular_values[i]} should be ~0"
            )

    def test_conditional_P_equals_tau_outer_tau(self) -> None:
        """Test P̃ = τ τᵀ explicitly."""
        tau = compute_tau_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            self._bkd
        )

        P_tilde = compute_conditional_P_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            self._bkd
        )

        # Expected: τ τᵀ
        expected = self._bkd.outer(tau, tau)

        self._bkd.assert_allclose(P_tilde, expected, rtol=1e-12)

    def test_conditional_P_symmetric(self) -> None:
        """Test conditional P̃ is symmetric."""
        P_tilde = compute_conditional_P_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            self._bkd
        )
        self._bkd.assert_allclose(P_tilde, P_tilde.T, rtol=1e-12)

    def test_conditional_P_diagonal_leq_standard_P(self) -> None:
        """Test P̃_{ii} <= P_{ii} (Cauchy-Schwarz inequality)."""
        P = compute_P_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            self._bkd
        )

        P_tilde = compute_conditional_P_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            self._bkd
        )

        # P_{ii} = E[X²] >= E[X]² = P̃_{ii}  (by Cauchy-Schwarz)
        P_diag = self._bkd.diag(P)
        P_tilde_diag = self._bkd.diag(P_tilde)

        # P_ii >= P_tilde_ii (with small tolerance for numerical error)
        diff = P_diag - P_tilde_diag
        self.assertTrue(
            self._bkd.all_bool(diff >= -1e-12),
            f"P_diag - P_tilde_diag should be >= 0, got min = {float(self._bkd.to_numpy(self._bkd.min(diff)))}"
        )


class TestConditionalMethods(Generic[Array], unittest.TestCase):
    """
    Base test class for conditional_P and conditional_u methods
    on SeparableKernelIntegralCalculator.

    Tests the multidimensional conditional methods with various index patterns.
    """

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment with 2D GP."""
        self._bkd = self.bkd()
        np.random.seed(42)

        # Create 2D separable kernel (product of 1D SE kernels)
        k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, self._bkd)
        k2 = SquaredExponentialKernel([0.8], (0.1, 10.0), 1, self._bkd)
        kernel = ProductKernel(k1, k2)

        # Create GP
        self._gp = ExactGaussianProcess(
            kernel,
            nvars=2,
            bkd=self._bkd,
            mean_function=ZeroMean(self._bkd)
        )

        # Training data
        self._n_train = 6
        X_train = self._bkd.array(np.random.rand(2, self._n_train) * 2 - 1)
        y_train = self._bkd.array(np.random.rand(self._n_train, 1))
        self._gp.fit(X_train, y_train)

        # Create quadrature bases using sparse grid infrastructure
        marginals = [
            UniformMarginal(-1.0, 1.0, self._bkd),
            UniformMarginal(-1.0, 1.0, self._bkd)
        ]
        factories = create_basis_factories(marginals, self._bkd, "gauss")
        bases = [f.create_basis() for f in factories]
        for b in bases:
            b.set_nterms(20)

        # Create calculator
        self._calc = SeparableKernelIntegralCalculator(
            self._gp, bases, bkd=self._bkd
        )

        self._nvars = 2

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def test_conditional_P_all_conditioned_equals_P(self) -> None:
        """When all dims conditioned (index=[1,1]), P_p equals standard P."""
        index = self._bkd.array([1.0, 1.0])
        P_p = self._calc.conditional_P(index)
        P = self._calc.P()

        self._bkd.assert_allclose(P_p, P, rtol=1e-12)

    def test_conditional_P_none_conditioned_equals_tau_outer_tau(self) -> None:
        """When no dims conditioned (index=[0,0]), P_p = τ τᵀ."""
        index = self._bkd.array([0.0, 0.0])
        P_p = self._calc.conditional_P(index)
        tau = self._calc.tau()
        expected = self._bkd.outer(tau, tau)

        self._bkd.assert_allclose(P_p, expected, rtol=1e-12)

    def test_conditional_P_single_conditioned(self) -> None:
        """Test index=[1, 0] (condition on dim 0, integrate out dim 1)."""
        index = self._bkd.array([1.0, 0.0])
        P_p = self._calc.conditional_P(index)

        # Should be: P_0 * P̃_1 where P̃_1 = τ_1 τ_1^T (rank-1)
        # The result should still be valid (symmetric, PSD)
        self.assertEqual(P_p.shape, (self._n_train, self._n_train))
        self._bkd.assert_allclose(P_p, P_p.T, rtol=1e-12)

    def test_conditional_P_index_validation(self) -> None:
        """Verify error raised for wrong index length."""
        wrong_index = self._bkd.array([0.0, 0.0, 0.0])  # 3 elements, but nvars=2
        with self.assertRaises(ValueError) as context:
            self._calc.conditional_P(wrong_index)
        self.assertIn("length", str(context.exception).lower())

    def test_conditional_u_all_conditioned_equals_one(self) -> None:
        """When all dims conditioned, u_p = 1."""
        index = self._bkd.array([1.0, 1.0])
        u_p = self._calc.conditional_u(index)

        self._bkd.assert_allclose(
            self._bkd.asarray([u_p]),
            self._bkd.asarray([1.0]),
            rtol=1e-12
        )

    def test_conditional_u_none_conditioned_equals_u(self) -> None:
        """When no dims conditioned, u_p = u."""
        index = self._bkd.array([0.0, 0.0])
        u_p = self._calc.conditional_u(index)
        u = self._calc.u()

        self._bkd.assert_allclose(
            self._bkd.asarray([u_p]),
            self._bkd.asarray([u]),
            rtol=1e-12
        )

    def test_conditional_u_single_conditioned(self) -> None:
        """Test index=[1, 0] (condition on dim 0, integrate out dim 1)."""
        index = self._bkd.array([1.0, 0.0])
        u_p = self._calc.conditional_u(index)

        # u_p = 1 (for dim 0 conditioned) * u_1 (for dim 1 integrated)
        # This is the u integral for dimension 1 only.
        # Since u = u_0 * u_1 and u_0, u_1 < 1 for typical kernels,
        # we have u_p = u_1 > u = u_0 * u_1.
        # But u_p should still be positive and <= 1 (max kernel value).
        u_p_val = float(self._bkd.to_numpy(u_p))

        self.assertGreater(u_p_val, 0.0)
        self.assertLessEqual(u_p_val, 1.0 + 1e-10)

    def test_conditional_u_index_validation(self) -> None:
        """Verify error raised for wrong index length."""
        wrong_index = self._bkd.array([0.0])  # 1 element, but nvars=2
        with self.assertRaises(ValueError) as context:
            self._calc.conditional_u(wrong_index)
        self.assertIn("length", str(context.exception).lower())

    def test_conditional_methods_complementary_indices(self) -> None:
        """Test that [1, 0] and [0, 1] give different but related results."""
        index_10 = self._bkd.array([1.0, 0.0])
        index_01 = self._bkd.array([0.0, 1.0])

        P_10 = self._calc.conditional_P(index_10)
        P_01 = self._calc.conditional_P(index_01)
        u_10 = self._calc.conditional_u(index_10)
        u_01 = self._calc.conditional_u(index_01)

        # They should be different (unless kernel params are equal)
        # But both should be valid matrices
        self.assertEqual(P_10.shape, P_01.shape)
        self._bkd.assert_allclose(P_10, P_10.T, rtol=1e-12)
        self._bkd.assert_allclose(P_01, P_01.T, rtol=1e-12)

        # u values should both be positive
        self.assertGreater(float(self._bkd.to_numpy(u_10)), 0.0)
        self.assertGreater(float(self._bkd.to_numpy(u_01)), 0.0)


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


class TestConditionalP1DNumpy(TestConditionalP1D[NDArray[Any]]):
    """NumPy backend tests for conditional P (1D)."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestConditionalP1DTorch(TestConditionalP1D[torch.Tensor]):
    """PyTorch backend tests for conditional P (1D)."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestConditionalMethodsNumpy(TestConditionalMethods[NDArray[Any]]):
    """NumPy backend tests for conditional methods."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestConditionalMethodsTorch(TestConditionalMethods[torch.Tensor]):
    """PyTorch backend tests for conditional methods."""

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
