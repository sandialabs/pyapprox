"""
Tests for 1D kernel integrals and validation functions.

Tests basic properties (symmetry, positive semi-definiteness),
analytical comparisons, and error handling.
"""

import numpy as np
import pytest

from pyapprox.probability.univariate import UniformMarginal
from pyapprox.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.mean_functions import (
    ConstantMean,
    ZeroMean,
)
from pyapprox.surrogates.gaussianprocess.statistics.integrals import (
    SeparableKernelIntegralCalculator,
)
from pyapprox.surrogates.gaussianprocess.statistics.integrals_1d import (
    compute_conditional_P_1d,
    compute_lambda_1d,
    compute_nu_1d,
    compute_P_1d,
    compute_Pi_1d,
    compute_tau_1d,
    compute_u_1d,
    compute_xi1_1d,
)
from pyapprox.surrogates.gaussianprocess.statistics.validation import (
    validate_separable_kernel,
    validate_zero_mean,
)
from pyapprox.surrogates.kernels.composition import (
    ProductKernel,
    SeparableProductKernel,
)
from pyapprox.surrogates.kernels.matern import (
    Matern52Kernel,
    SquaredExponentialKernel,
)
from pyapprox.surrogates.sparsegrids.basis_factory import (
    create_basis_factories,
)


class TestIntegrals1D:
    """
    Base test class for 1D kernel integrals.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, bkd):
        """Set up test environment."""
        np.random.seed(42)

        # Create 1D kernel (Squared Exponential / RBF)
        self._kernel_1d = SquaredExponentialKernel(
            [1.0],
            (0.1, 10.0),
            1,
            bkd
        )

        # Training points in 1D
        self._n_train = 5
        train_np = np.array([[-1.0, -0.5, 0.0, 0.5, 1.0]])
        self._train_samples_1d = bkd.array(train_np)

        # Quadrature points (Gauss-Legendre on [-1, 1])
        self._nquad = 20
        quad_pts_np, quad_wts_np = np.polynomial.legendre.leggauss(self._nquad)
        # Scale weights for density on [-1, 1]: uniform has density 1/2
        quad_wts_np = quad_wts_np / 2.0
        self._quad_samples = bkd.array(quad_pts_np.reshape(1, -1))
        self._quad_weights = bkd.array(quad_wts_np)

    def _kernel_callable(self, x1, x2):
        """Wrap kernel as callable for integral functions."""
        return self._kernel_1d(x1, x2)

    def test_tau_shape(self, bkd) -> None:
        """Test tau has correct shape (N,)."""
        tau = compute_tau_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            bkd
        )
        assert tau.shape == (self._n_train,)

    def test_tau_positive(self, bkd) -> None:
        """Test tau values are positive (kernel is positive)."""
        tau = compute_tau_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            bkd
        )
        assert bkd.all_bool(tau > 0)

    def test_P_shape(self, bkd) -> None:
        """Test P has correct shape (N, N)."""
        P = compute_P_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            bkd
        )
        assert P.shape == (self._n_train, self._n_train)

    def test_P_symmetric(self, bkd) -> None:
        """Test P is symmetric."""
        P = compute_P_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            bkd
        )
        bkd.assert_allclose(P, P.T, rtol=1e-12)

    def test_P_positive_semidefinite(self, bkd) -> None:
        """Test P is positive semi-definite (all eigenvalues >= 0)."""
        P = compute_P_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            bkd
        )
        eigvals = bkd.eigvalsh(P)
        # Allow small negative eigenvalues due to numerical error
        assert bkd.all_bool(eigvals > -1e-10)

    def test_u_scalar(self, bkd) -> None:
        """Test u is a scalar and positive."""
        u = compute_u_1d(
            self._quad_samples,
            self._quad_weights,
            self._kernel_callable,
            bkd
        )
        # u should be positive (integral of positive kernel)
        assert float(bkd.to_numpy(u)) > 0.0

    def test_nu_scalar(self, bkd) -> None:
        """Test nu is a scalar and positive."""
        nu = compute_nu_1d(
            self._quad_samples,
            self._quad_weights,
            self._kernel_callable,
            bkd
        )
        # nu should be positive (integral of squared kernel)
        assert float(bkd.to_numpy(nu)) > 0.0

    def test_lambda_shape(self, bkd) -> None:
        """Test lambda has correct shape (N,)."""
        lambda_vec = compute_lambda_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            bkd
        )
        assert lambda_vec.shape == (self._n_train,)

    def test_lambda_positive(self, bkd) -> None:
        """Test lambda values are positive."""
        lambda_vec = compute_lambda_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            bkd
        )
        assert bkd.all_bool(lambda_vec > 0)

    def test_Pi_shape(self, bkd) -> None:
        """Test Pi has correct shape (N, N)."""
        Pi = compute_Pi_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            bkd
        )
        assert Pi.shape == (self._n_train, self._n_train)

    def test_Pi_symmetric(self, bkd) -> None:
        """Test Pi is symmetric."""
        Pi = compute_Pi_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            bkd
        )
        bkd.assert_allclose(Pi, Pi.T, rtol=1e-12)

    def test_xi1_scalar(self, bkd) -> None:
        """Test xi1 is a scalar and positive."""
        xi1 = compute_xi1_1d(
            self._quad_samples,
            self._quad_weights,
            self._kernel_callable,
            bkd
        )
        # xi1 should be positive
        assert float(bkd.to_numpy(xi1)) > 0.0

    def test_u_equals_tau_squared_for_single_point(self, bkd) -> None:
        """
        Test that u = tau^2 when there's only one quadrature point.

        For a single point, u = w^2 * K(x, x) and tau = w * K(x, x),
        so u = tau^2 / w. This is a sanity check.
        """
        # Use single quadrature point
        single_quad = bkd.array([[0.0]])
        single_weight = bkd.array([1.0])

        u = compute_u_1d(
            single_quad,
            single_weight,
            self._kernel_callable,
            bkd
        )

        # For single point at 0, K(0, 0) = 1 (for normalized kernel)
        # u = 1 * 1 * K(0, 0) = K(0, 0)
        K_00 = self._kernel_1d(single_quad, single_quad)
        expected_u = K_00[0, 0]
        bkd.assert_allclose(
            bkd.asarray([u]),
            bkd.asarray([expected_u]),
            rtol=1e-12
        )


class TestConditionalP1D:
    """
    Base test class for conditional P (1D) function.

    Tests the compute_conditional_P_1d function which returns P_tilde = tau tau^T.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, bkd):
        """Set up test environment."""
        np.random.seed(42)

        # Create 1D kernel (Squared Exponential / RBF)
        self._kernel_1d = SquaredExponentialKernel(
            [1.0],
            (0.1, 10.0),
            1,
            bkd
        )

        # Training points in 1D
        self._n_train = 5
        train_np = np.array([[-1.0, -0.5, 0.0, 0.5, 1.0]])
        self._train_samples_1d = bkd.array(train_np)

        # Quadrature points (Gauss-Legendre on [-1, 1])
        self._nquad = 20
        quad_pts_np, quad_wts_np = np.polynomial.legendre.leggauss(self._nquad)
        # Scale weights for density on [-1, 1]: uniform has density 1/2
        quad_wts_np = quad_wts_np / 2.0
        self._quad_samples = bkd.array(quad_pts_np.reshape(1, -1))
        self._quad_weights = bkd.array(quad_wts_np)

    def _kernel_callable(self, x1, x2):
        """Wrap kernel as callable for integral functions."""
        return self._kernel_1d(x1, x2)

    def test_conditional_P_shape(self, bkd) -> None:
        """Test conditional P_tilde has correct shape (N, N)."""
        P_tilde = compute_conditional_P_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            bkd
        )
        assert P_tilde.shape == (self._n_train, self._n_train)

    def test_conditional_P_is_rank1(self, bkd) -> None:
        """Test conditional P_tilde is rank-1 (since P_tilde = tau tau^T)."""
        P_tilde = compute_conditional_P_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            bkd
        )

        # Compute SVD and check that only one singular value is significant
        P_np = bkd.to_numpy(P_tilde)
        singular_values = np.linalg.svd(P_np, compute_uv=False)

        # Only the first singular value should be non-zero
        # All others should be near zero (numerical tolerance)
        assert singular_values[0] > 1e-10
        for i in range(1, len(singular_values)):
            assert singular_values[i] < 1e-10, \
                f"Singular value {i} = {singular_values[i]} should be ~0"

    def test_conditional_P_equals_tau_outer_tau(self, bkd) -> None:
        """Test P_tilde = tau tau^T explicitly."""
        tau = compute_tau_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            bkd
        )

        P_tilde = compute_conditional_P_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            bkd
        )

        # Expected: tau tau^T
        expected = bkd.outer(tau, tau)

        bkd.assert_allclose(P_tilde, expected, rtol=1e-12)

    def test_conditional_P_symmetric(self, bkd) -> None:
        """Test conditional P_tilde is symmetric."""
        P_tilde = compute_conditional_P_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            bkd
        )
        bkd.assert_allclose(P_tilde, P_tilde.T, rtol=1e-12)

    def test_conditional_P_diagonal_leq_standard_P(self, bkd) -> None:
        """Test P_tilde_{ii} <= P_{ii} (Cauchy-Schwarz inequality)."""
        P = compute_P_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            bkd
        )

        P_tilde = compute_conditional_P_1d(
            self._quad_samples,
            self._quad_weights,
            self._train_samples_1d,
            self._kernel_callable,
            bkd
        )

        # P_{ii} = E[X^2] >= E[X]^2 = P_tilde_{ii}  (by Cauchy-Schwarz)
        P_diag = bkd.diag(P)
        P_tilde_diag = bkd.diag(P_tilde)

        # P_ii >= P_tilde_ii (with small tolerance for numerical error)
        diff = P_diag - P_tilde_diag
        assert bkd.all_bool(diff >= -1e-12), \
            ("P_diag - P_tilde_diag should be >= 0, "
             f"got min = {float(bkd.to_numpy(bkd.min(diff)))}")


class TestConditionalMethods:
    """
    Base test class for conditional_P and conditional_u methods
    on SeparableKernelIntegralCalculator.

    Tests the multidimensional conditional methods with various index patterns.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, bkd):
        """Set up test environment with 2D GP."""
        np.random.seed(42)

        # Create 2D separable kernel (product of 1D SE kernels)
        k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        k2 = SquaredExponentialKernel([0.8], (0.1, 10.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        # Create GP
        self._gp = ExactGaussianProcess(
            kernel,
            nvars=2,
            bkd=bkd,
            mean_function=ZeroMean(bkd)
        )

        # Training data
        self._n_train = 6
        X_train = bkd.array(np.random.rand(2, self._n_train) * 2 - 1)
        y_train = bkd.array(np.random.rand(self._n_train).reshape(1, -1))
        self._gp.hyp_list().set_all_inactive()
        self._gp.fit(X_train, y_train)

        # Create quadrature bases using sparse grid infrastructure
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd)
        ]
        factories = create_basis_factories(marginals, bkd, "gauss")
        bases = [f.create_basis() for f in factories]
        for b in bases:
            b.set_nterms(20)

        # Create calculator
        self._calc = SeparableKernelIntegralCalculator(
            self._gp, bases, marginals, bkd=bkd
        )

        self._nvars = 2

    def test_conditional_P_all_conditioned_equals_P(self, bkd) -> None:
        """When all dims conditioned (index=[1,1]), P_p equals standard P."""
        index = bkd.array([1.0, 1.0])
        P_p = self._calc.conditional_P(index)
        P = self._calc.P()

        bkd.assert_allclose(P_p, P, rtol=1e-12)

    def test_conditional_P_none_conditioned_equals_tau_outer_tau(self, bkd) -> None:
        """When no dims conditioned (index=[0,0]), P_p = tau tau^T."""
        index = bkd.array([0.0, 0.0])
        P_p = self._calc.conditional_P(index)
        tau = self._calc.tau_C()
        expected = bkd.outer(tau, tau)

        bkd.assert_allclose(P_p, expected, rtol=1e-12)

    def test_conditional_P_single_conditioned(self, bkd) -> None:
        """Test index=[1, 0] (condition on dim 0, integrate out dim 1)."""
        index = bkd.array([1.0, 0.0])
        P_p = self._calc.conditional_P(index)

        # Should be: P_0 * P_tilde_1 where P_tilde_1 = tau_1 tau_1^T (rank-1)
        # The result should still be valid (symmetric, PSD)
        assert P_p.shape == (self._n_train, self._n_train)
        bkd.assert_allclose(P_p, P_p.T, rtol=1e-12)

    def test_conditional_P_index_validation(self, bkd) -> None:
        """Verify error raised for wrong index length."""
        wrong_index = bkd.array([0.0, 0.0, 0.0])  # 3 elements, but nvars=2
        with pytest.raises(ValueError) as context:
            self._calc.conditional_P(wrong_index)
        assert "length" in str(context.value).lower()

    def test_conditional_u_all_conditioned_equals_one(self, bkd) -> None:
        """When all dims conditioned, u_p = 1."""
        index = bkd.array([1.0, 1.0])
        u_p = self._calc.conditional_u(index)

        bkd.assert_allclose(
            bkd.asarray([u_p]),
            bkd.asarray([1.0]),
            rtol=1e-12
        )

    def test_conditional_u_none_conditioned_equals_u(self, bkd) -> None:
        """When no dims conditioned, u_p = u."""
        index = bkd.array([0.0, 0.0])
        u_p = self._calc.conditional_u(index)
        u = self._calc.u()

        bkd.assert_allclose(
            bkd.asarray([u_p]),
            bkd.asarray([u]),
            rtol=1e-12
        )

    def test_conditional_u_single_conditioned(self, bkd) -> None:
        """Test index=[1, 0] (condition on dim 0, integrate out dim 1)."""
        index = bkd.array([1.0, 0.0])
        u_p = self._calc.conditional_u(index)

        # u_p = 1 (for dim 0 conditioned) * u_1 (for dim 1 integrated)
        # This is the u integral for dimension 1 only.
        # Since u = u_0 * u_1 and u_0, u_1 < 1 for typical kernels,
        # we have u_p = u_1 > u = u_0 * u_1.
        # But u_p should still be positive and <= 1 (max kernel value).
        u_p_val = float(bkd.to_numpy(u_p))

        assert u_p_val > 0.0
        assert u_p_val <= 1.0 + 1e-10

    def test_conditional_u_index_validation(self, bkd) -> None:
        """Verify error raised for wrong index length."""
        wrong_index = bkd.array([0.0])  # 1 element, but nvars=2
        with pytest.raises(ValueError) as context:
            self._calc.conditional_u(wrong_index)
        assert "length" in str(context.value).lower()

    def test_conditional_methods_complementary_indices(self, bkd) -> None:
        """Test that [1, 0] and [0, 1] give different but related results."""
        index_10 = bkd.array([1.0, 0.0])
        index_01 = bkd.array([0.0, 1.0])

        P_10 = self._calc.conditional_P(index_10)
        P_01 = self._calc.conditional_P(index_01)
        u_10 = self._calc.conditional_u(index_10)
        u_01 = self._calc.conditional_u(index_01)

        # They should be different (unless kernel params are equal)
        # But both should be valid matrices
        assert P_10.shape == P_01.shape
        bkd.assert_allclose(P_10, P_10.T, rtol=1e-12)
        bkd.assert_allclose(P_01, P_01.T, rtol=1e-12)

        # u values should both be positive
        assert float(bkd.to_numpy(u_10)) > 0.0
        assert float(bkd.to_numpy(u_01)) > 0.0


class TestValidation:
    """
    Base test class for validation functions.
    """

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_validate_separable_kernel_product(self, bkd) -> None:
        """Test that ProductKernel passes validation."""
        k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        prod_kernel = ProductKernel(k1, k2)

        # Should not raise
        validate_separable_kernel(prod_kernel)

    def test_validate_separable_kernel_1d(self, bkd) -> None:
        """Test that 1D kernel passes validation (trivially separable)."""
        k1d = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)

        # Should not raise
        validate_separable_kernel(k1d)

    def test_validate_separable_kernel_non_separable_error(self, bkd) -> None:
        """Test that non-separable multi-D kernel raises TypeError."""
        # Matern 5/2 is NOT separable (uses combined distance in polynomial)
        k2d = Matern52Kernel([1.0, 1.0], (0.1, 10.0), 2, bkd)

        with pytest.raises(TypeError) as context:
            validate_separable_kernel(k2d)

        assert "separable" in str(context.value).lower()

    def test_validate_zero_mean_with_zero_mean(self, bkd) -> None:
        """Test that GP with ZeroMean passes validation."""
        kernel = SquaredExponentialKernel([1.0, 1.0], (0.1, 10.0), 2, bkd)
        gp = ExactGaussianProcess(
            kernel,
            nvars=2,
            bkd=bkd,
            mean_function=ZeroMean(bkd)
        )

        # Should not raise
        validate_zero_mean(gp)

    def test_validate_zero_mean_default(self, bkd) -> None:
        """Test that GP with default mean (ZeroMean) passes validation."""
        kernel = SquaredExponentialKernel([1.0, 1.0], (0.1, 10.0), 2, bkd)
        gp = ExactGaussianProcess(
            kernel,
            nvars=2,
            bkd=bkd
            # Default mean is ZeroMean
        )

        # Should not raise
        validate_zero_mean(gp)

    def test_validate_zero_mean_constant_mean_error(self, bkd) -> None:
        """Test that GP with ConstantMean raises ValueError."""
        kernel = SquaredExponentialKernel([1.0, 1.0], (0.1, 10.0), 2, bkd)
        gp = ExactGaussianProcess(
            kernel,
            nvars=2,
            bkd=bkd,
            mean_function=ConstantMean(1.0, (-10.0, 10.0), bkd)
        )

        with pytest.raises(ValueError) as context:
            validate_zero_mean(gp)

        assert "ZeroMean" in str(context.value)
