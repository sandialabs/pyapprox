import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.surrogates.gaussianprocess.exact import (
    ExactGaussianProcess,
)
from pyapprox.surrogates.kernels.composition import (
    ProductKernel,
    SeparableProductKernel,
    SumKernel,
)
from pyapprox.surrogates.kernels.matern import (
    Matern32Kernel,
    Matern52Kernel,
    SquaredExponentialKernel,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from tests._helpers.markers import slow_test


class TestProductKernel:
    """
    Base test class for ProductKernel.
    """

    def _setup_data(self, bkd):
        np.random.seed(42)
        self.nvars = 2
        self.nsamples1 = 5
        self.nsamples2 = 4

        # Create two Matern kernels for composition
        self.kernel1 = Matern52Kernel([1.0, 1.0], (0.1, 10.0), self.nvars, bkd)
        self.kernel2 = Matern32Kernel([0.5, 0.5], (0.1, 10.0), self.nvars, bkd)

        # Create sample data
        self.X1 = bkd.array(np.random.randn(self.nvars, self.nsamples1))
        self.X2 = bkd.array(np.random.randn(self.nvars, self.nsamples2))

    def test_initialization(self, bkd) -> None:
        """
        Test ProductKernel initialization.
        """
        self._setup_data(bkd)
        product = ProductKernel(self.kernel1, self.kernel2)
        assert product.nvars() == self.nvars
        assert product.bkd() is not None

    def test_backend_mismatch_error(self, bkd) -> None:
        """
        Test that ProductKernel raises error when backends don't match.
        """
        self._setup_data(bkd)
        # Create kernel with different backend
        if isinstance(bkd, NumpyBkd):
            other_bkd = TorchBkd()
        else:
            other_bkd = NumpyBkd()

        kernel_other = Matern52Kernel([1.0, 1.0], (0.1, 10.0), self.nvars, other_bkd)

        with pytest.raises(ValueError) as context:
            ProductKernel(self.kernel1, kernel_other)

        assert "same backend type" in str(context.value)

    def test_hyperparameter_list_combination(self, bkd) -> None:
        """
        Test that hyperparameter lists are combined correctly.
        """
        self._setup_data(bkd)
        product = ProductKernel(self.kernel1, self.kernel2)
        hyp_list = product.hyp_list()

        # Should have parameters from both kernels
        nparams1 = self.kernel1.hyp_list().nparams()
        nparams2 = self.kernel2.hyp_list().nparams()
        assert hyp_list.nparams() == nparams1 + nparams2

    def test_kernel_matrix_shape(self, bkd) -> None:
        """
        Test that ProductKernel produces correct output shape.
        """
        self._setup_data(bkd)
        product = ProductKernel(self.kernel1, self.kernel2)

        # Test with two different inputs
        K = product(self.X1, self.X2)
        assert K.shape == (self.nsamples1, self.nsamples2)

        # Test with single input
        K_self = product(self.X1)
        assert K_self.shape == (self.nsamples1, self.nsamples1)

    def test_kernel_matrix_values(self, bkd) -> None:
        """
        Test that ProductKernel computes K1 * K2 correctly.
        """
        self._setup_data(bkd)
        product = ProductKernel(self.kernel1, self.kernel2)

        K1 = self.kernel1(self.X1, self.X2)
        K2 = self.kernel2(self.X1, self.X2)
        K_product = product(self.X1, self.X2)

        expected = K1 * K2
        bkd.assert_allclose(K_product, expected)

    def test_diagonal(self, bkd) -> None:
        """
        Test diagonal computation for ProductKernel.
        """
        self._setup_data(bkd)
        product = ProductKernel(self.kernel1, self.kernel2)

        diag1 = self.kernel1.diag(self.X1)
        diag2 = self.kernel2.diag(self.X1)
        diag_product = product.diag(self.X1)

        expected = diag1 * diag2
        bkd.assert_allclose(diag_product, expected)

    def test_jacobian(self, bkd) -> None:
        """
        Test Jacobian computation for ProductKernel.

        ProductKernel should satisfy product rule: d(K1*K2)/dx = dK1*K2 + K1*dK2
        """
        self._setup_data(bkd)
        product = ProductKernel(self.kernel1, self.kernel2)

        jac = product.jacobian(self.X1, self.X2)

        # Check shape
        assert jac.shape == (self.nsamples1, self.nsamples2, self.nvars)

        # Check finiteness
        assert bkd.all_bool(bkd.isfinite(jac))

        # Manually compute using product rule
        K1 = self.kernel1(self.X1, self.X2)
        K2 = self.kernel2(self.X1, self.X2)
        dK1 = self.kernel1.jacobian(self.X1, self.X2)
        dK2 = self.kernel2.jacobian(self.X1, self.X2)

        expected = dK1 * K2[..., None] + K1[..., None] * dK2
        bkd.assert_allclose(jac, expected)

    def test_param_jacobian(self, bkd) -> None:
        """
        Test parameter Jacobian for ProductKernel.

        The parameter Jacobian should stack derivatives from both kernels.
        """
        self._setup_data(bkd)
        product = ProductKernel(self.kernel1, self.kernel2)

        jac = product.jacobian_wrt_params(self.X1)

        # Check shape
        nparams_total = (
            self.kernel1.hyp_list().nparams() + self.kernel2.hyp_list().nparams()
        )
        assert jac.shape == (self.nsamples1, self.nsamples1, nparams_total)

        # Check finiteness
        assert bkd.all_bool(bkd.isfinite(jac))

    def test_operator_overloading(self, bkd) -> None:
        """
        Test that * operator creates ProductKernel.
        """
        self._setup_data(bkd)
        product = self.kernel1 * self.kernel2
        assert isinstance(product, ProductKernel)

        # Verify it produces same result as explicit construction
        product_explicit = ProductKernel(self.kernel1, self.kernel2)
        K1 = product(self.X1, self.X2)
        K2 = product_explicit(self.X1, self.X2)
        bkd.assert_allclose(K1, K2)

    def test_hvp_wrt_x1(self, bkd) -> None:
        """
        Test HVP computation for ProductKernel using product rule.
        """
        self._setup_data(bkd)
        product = ProductKernel(self.kernel1, self.kernel2)

        # Single point for HVP
        X1_single = self.X1[:, 0:1]  # (nvars, 1)
        direction = bkd.array(np.random.randn(self.nvars, 1))
        direction = direction / bkd.norm(direction)
        direction_flat = bkd.reshape(direction, (self.nvars,))

        hvp = product.hvp_wrt_x1(X1_single, self.X2, direction_flat)

        # Check shape: (n1, n2, nvars)
        assert hvp.shape == (1, self.nsamples2, self.nvars)

        # Check finiteness
        assert bkd.all_bool(bkd.isfinite(hvp))


class TestSumKernel:
    """
    Base test class for SumKernel.
    """

    def _setup_data(self, bkd):
        np.random.seed(42)
        self.nvars = 2
        self.nsamples1 = 5
        self.nsamples2 = 4

        # Create two Matern kernels for composition
        self.kernel1 = Matern52Kernel([1.0, 1.0], (0.1, 10.0), self.nvars, bkd)
        self.kernel2 = Matern32Kernel([0.5, 0.5], (0.1, 10.0), self.nvars, bkd)

        # Create sample data
        self.X1 = bkd.array(np.random.randn(self.nvars, self.nsamples1))
        self.X2 = bkd.array(np.random.randn(self.nvars, self.nsamples2))

    def test_initialization(self, bkd) -> None:
        """
        Test SumKernel initialization.
        """
        self._setup_data(bkd)
        sum_kernel = SumKernel(self.kernel1, self.kernel2)
        assert sum_kernel.nvars() == self.nvars
        assert sum_kernel.bkd() is not None

    def test_backend_mismatch_error(self, bkd) -> None:
        """
        Test that SumKernel raises error when backends don't match.
        """
        self._setup_data(bkd)
        # Create kernel with different backend
        if isinstance(bkd, NumpyBkd):
            other_bkd = TorchBkd()
        else:
            other_bkd = NumpyBkd()

        kernel_other = Matern52Kernel([1.0, 1.0], (0.1, 10.0), self.nvars, other_bkd)

        with pytest.raises(ValueError) as context:
            SumKernel(self.kernel1, kernel_other)

        assert "same backend type" in str(context.value)

    def test_hyperparameter_list_combination(self, bkd) -> None:
        """
        Test that hyperparameter lists are combined correctly.
        """
        self._setup_data(bkd)
        sum_kernel = SumKernel(self.kernel1, self.kernel2)
        hyp_list = sum_kernel.hyp_list()

        # Should have parameters from both kernels
        nparams1 = self.kernel1.hyp_list().nparams()
        nparams2 = self.kernel2.hyp_list().nparams()
        assert hyp_list.nparams() == nparams1 + nparams2

    def test_kernel_matrix_shape(self, bkd) -> None:
        """
        Test that SumKernel produces correct output shape.
        """
        self._setup_data(bkd)
        sum_kernel = SumKernel(self.kernel1, self.kernel2)

        # Test with two different inputs
        K = sum_kernel(self.X1, self.X2)
        assert K.shape == (self.nsamples1, self.nsamples2)

        # Test with single input
        K_self = sum_kernel(self.X1)
        assert K_self.shape == (self.nsamples1, self.nsamples1)

    def test_kernel_matrix_values(self, bkd) -> None:
        """
        Test that SumKernel computes K1 + K2 correctly.
        """
        self._setup_data(bkd)
        sum_kernel = SumKernel(self.kernel1, self.kernel2)

        K1 = self.kernel1(self.X1, self.X2)
        K2 = self.kernel2(self.X1, self.X2)
        K_sum = sum_kernel(self.X1, self.X2)

        expected = K1 + K2
        bkd.assert_allclose(K_sum, expected)

    def test_diagonal(self, bkd) -> None:
        """
        Test diagonal computation for SumKernel.
        """
        self._setup_data(bkd)
        sum_kernel = SumKernel(self.kernel1, self.kernel2)

        diag1 = self.kernel1.diag(self.X1)
        diag2 = self.kernel2.diag(self.X1)
        diag_sum = sum_kernel.diag(self.X1)

        expected = diag1 + diag2
        bkd.assert_allclose(diag_sum, expected)

    def test_jacobian(self, bkd) -> None:
        """
        Test Jacobian computation for SumKernel.

        SumKernel should satisfy sum rule: d(K1+K2)/dx = dK1 + dK2
        """
        self._setup_data(bkd)
        sum_kernel = SumKernel(self.kernel1, self.kernel2)

        jac = sum_kernel.jacobian(self.X1, self.X2)

        # Check shape
        assert jac.shape == (self.nsamples1, self.nsamples2, self.nvars)

        # Check finiteness
        assert bkd.all_bool(bkd.isfinite(jac))

        # Manually compute using sum rule
        dK1 = self.kernel1.jacobian(self.X1, self.X2)
        dK2 = self.kernel2.jacobian(self.X1, self.X2)

        expected = dK1 + dK2
        bkd.assert_allclose(jac, expected)

    def test_param_jacobian(self, bkd) -> None:
        """
        Test parameter Jacobian for SumKernel.

        The parameter Jacobian should stack derivatives from both kernels.
        """
        self._setup_data(bkd)
        sum_kernel = SumKernel(self.kernel1, self.kernel2)

        jac = sum_kernel.jacobian_wrt_params(self.X1)

        # Check shape
        nparams_total = (
            self.kernel1.hyp_list().nparams() + self.kernel2.hyp_list().nparams()
        )
        assert jac.shape == (self.nsamples1, self.nsamples1, nparams_total)

        # Check finiteness
        assert bkd.all_bool(bkd.isfinite(jac))

    def test_operator_overloading(self, bkd) -> None:
        """
        Test that + operator creates SumKernel.
        """
        self._setup_data(bkd)
        sum_kernel = self.kernel1 + self.kernel2
        assert isinstance(sum_kernel, SumKernel)

        # Verify it produces same result as explicit construction
        sum_explicit = SumKernel(self.kernel1, self.kernel2)
        K1 = sum_kernel(self.X1, self.X2)
        K2 = sum_explicit(self.X1, self.X2)
        bkd.assert_allclose(K1, K2)

    def test_hvp_wrt_x1(self, bkd) -> None:
        """
        Test HVP computation for SumKernel using sum rule.
        """
        self._setup_data(bkd)
        sum_kernel = SumKernel(self.kernel1, self.kernel2)

        # Single point for HVP
        X1_single = self.X1[:, 0:1]  # (nvars, 1)
        direction = bkd.array(np.random.randn(self.nvars, 1))
        direction = direction / bkd.norm(direction)
        direction_flat = bkd.reshape(direction, (self.nvars,))

        hvp = sum_kernel.hvp_wrt_x1(X1_single, self.X2, direction_flat)

        # Check shape: (n1, n2, nvars)
        assert hvp.shape == (1, self.nsamples2, self.nvars)

        # Check finiteness
        assert bkd.all_bool(bkd.isfinite(hvp))


class TestNestedComposition:
    """
    Test nested composition of kernels.
    """

    def _setup_data(self, bkd):
        np.random.seed(42)
        self.nvars = 2
        self.nsamples = 5

        # Create three kernels
        self.k1 = Matern52Kernel([1.0, 1.0], (0.1, 10.0), self.nvars, bkd)
        self.k2 = Matern32Kernel([0.5, 0.5], (0.1, 10.0), self.nvars, bkd)
        self.k3 = Matern32Kernel([2.0, 2.0], (0.1, 10.0), self.nvars, bkd)

        self.X = bkd.array(np.random.randn(self.nvars, self.nsamples))

    def test_nested_sum_product(self, bkd) -> None:
        """
        Test nested composition: (k1 + k2) * k3
        """
        self._setup_data(bkd)
        nested = (self.k1 + self.k2) * self.k3

        # Compute manually
        K1 = self.k1(self.X, self.X)
        K2 = self.k2(self.X, self.X)
        K3 = self.k3(self.X, self.X)
        expected = (K1 + K2) * K3

        K_nested = nested(self.X, self.X)
        bkd.assert_allclose(K_nested, expected)

    def test_nested_product_sum(self, bkd) -> None:
        """
        Test nested composition: k1 * k2 + k3
        """
        self._setup_data(bkd)
        nested = self.k1 * self.k2 + self.k3

        # Compute manually
        K1 = self.k1(self.X, self.X)
        K2 = self.k2(self.X, self.X)
        K3 = self.k3(self.X, self.X)
        expected = K1 * K2 + K3

        K_nested = nested(self.X, self.X)
        bkd.assert_allclose(K_nested, expected)

    def test_deeply_nested(self, bkd) -> None:
        """
        Test deeply nested composition: (k1 + k2) * (k2 + k3)
        """
        self._setup_data(bkd)
        nested = (self.k1 + self.k2) * (self.k2 + self.k3)

        # Compute manually
        K1 = self.k1(self.X, self.X)
        K2 = self.k2(self.X, self.X)
        K3 = self.k3(self.X, self.X)
        expected = (K1 + K2) * (K2 + K3)

        K_nested = nested(self.X, self.X)
        bkd.assert_allclose(K_nested, expected)


class TestSeparableProductKernel:
    """
    Base test class for SeparableProductKernel.

    Tests the separable product kernel where each 1D kernel operates
    on a different dimension: k(x, y) = prod_i k_i(x_i, y_i)
    """

    def test_factorization(self, bkd) -> None:
        """Verify k(x,y) = k1(x1,y1) * k2(x2,y2) for SeparableProductKernel."""
        np.random.seed(42)

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

    def test_batch(self, bkd) -> None:
        """Test SeparableProductKernel with batched inputs."""
        np.random.seed(42)

        k1 = SquaredExponentialKernel([1.0], (0.01, 100.0), 1, bkd)
        k2 = SquaredExponentialKernel([2.0], (0.01, 100.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        # Multiple test points
        X1 = bkd.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  # 3 points in 2D
        X2 = bkd.array([[0.7, 0.9], [0.8, 1.0]])  # 2 points in 2D

        K = kernel(X1, X2)

        # Should have shape (3, 2)
        assert K.shape == (3, 2)

        # Verify each element manually
        for i in range(3):
            for j in range(2):
                k1_val = k1(bkd.array([[X1[0, i]]]), bkd.array([[X2[0, j]]]))
                k2_val = k2(bkd.array([[X1[1, i]]]), bkd.array([[X2[1, j]]]))
                expected = k1_val[0, 0] * k2_val[0, 0]
                bkd.assert_allclose(
                    bkd.asarray([K[i, j]]), bkd.asarray([expected]), rtol=1e-12
                )

    def test_nvars(self, bkd) -> None:
        """Test nvars matches number of 1D kernels."""
        np.random.seed(42)

        k1 = SquaredExponentialKernel([1.0], (0.01, 100.0), 1, bkd)
        k2 = SquaredExponentialKernel([2.0], (0.01, 100.0), 1, bkd)
        k3 = SquaredExponentialKernel([0.5], (0.01, 100.0), 1, bkd)

        kernel_2d = SeparableProductKernel([k1, k2], bkd)
        kernel_3d = SeparableProductKernel([k1, k2, k3], bkd)

        assert kernel_2d.nvars() == 2
        assert kernel_3d.nvars() == 3

    def test_get_kernel_1d(self, bkd) -> None:
        """Test get_kernel_1d returns correct kernels."""
        np.random.seed(42)

        k1 = SquaredExponentialKernel([1.5], (0.01, 100.0), 1, bkd)
        k2 = SquaredExponentialKernel([2.5], (0.01, 100.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        assert kernel.get_kernel_1d(0) is k1
        assert kernel.get_kernel_1d(1) is k2

    def test_diag(self, bkd) -> None:
        """Test diagonal computation."""
        np.random.seed(42)

        k1 = SquaredExponentialKernel([1.0], (0.01, 100.0), 1, bkd)
        k2 = SquaredExponentialKernel([2.0], (0.01, 100.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        X = bkd.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        diag = kernel.diag(X)

        # Diagonal should be product of 1D diagonals
        # For RBF kernels, diag = 1 for all points
        expected = bkd.ones((3,))
        bkd.assert_allclose(diag, expected, rtol=1e-12)

    def test_jacobian_wrt_params(self, bkd) -> None:
        """Test jacobian_wrt_params for SeparableProductKernel using
        DerivativeChecker."""
        np.random.seed(42)

        k1 = SquaredExponentialKernel([1.0], (0.01, 100.0), 1, bkd)
        k2 = SquaredExponentialKernel([2.0], (0.01, 100.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        # Test points
        n = 5
        samples = bkd.array(np.random.rand(2, n))

        # Get jacobian and check shape/finiteness
        jac = kernel.jacobian_wrt_params(samples)
        nparams = kernel.hyp_list().nparams()
        assert jac.shape == (n, n, nparams)
        assert bkd.all_bool(bkd.isfinite(jac))

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
        assert checker.error_ratio(errors[0]) < 1e-6

    def test_jacobian_wrt_params_matches_se_kernel(self, bkd) -> None:
        """Verify SeparableProductKernel jacobian_wrt_params matches SE kernel."""
        np.random.seed(42)

        ls = [1.5, 2.5]
        bounds = (0.01, 100.0)

        # Multi-dim SE kernel
        se_kernel = SquaredExponentialKernel(ls, bounds, 2, bkd)

        # Equivalent separable product kernel from 1D SE kernels
        k1 = SquaredExponentialKernel([ls[0]], bounds, 1, bkd)
        k2 = SquaredExponentialKernel([ls[1]], bounds, 1, bkd)
        sep_kernel = SeparableProductKernel([k1, k2], bkd)

        samples = bkd.array(np.random.rand(2, 7))

        jac_se = se_kernel.jacobian_wrt_params(samples)
        jac_sep = sep_kernel.jacobian_wrt_params(samples)

        bkd.assert_allclose(jac_sep, jac_se, rtol=1e-12)

    @slow_test
    def test_optimal_hyperparameters_match_se_kernel(self, bkd) -> None:
        """Verify GP fitting with SeparableProductKernel matches SE kernel."""
        np.random.seed(42)

        nvars = 2
        ntrain = 30
        bounds = (0.1, 10.0)
        init_ls = [5.0, 5.0]

        # Training data
        X_train = bkd.array(np.random.uniform(-2, 2, (nvars, ntrain)))
        (
            np.sin(2 * np.random.RandomState(42).uniform(-2, 2, ntrain))
            * np.cos(np.random.RandomState(43).uniform(-2, 2, ntrain))
        )
        # Use deterministic function of training points
        X_np = bkd.to_numpy(X_train)
        y_np = np.sin(2 * X_np[0, :]) * np.cos(X_np[1, :])
        y_train = bkd.array(y_np[None, :])

        # GP with multi-dim SE kernel
        se_kernel = SquaredExponentialKernel(init_ls, bounds, nvars, bkd)
        gp_se = ExactGaussianProcess(se_kernel, nvars, bkd, nugget=1e-8)
        gp_se.fit(X_train, y_train)

        # GP with separable product kernel
        k1 = SquaredExponentialKernel([init_ls[0]], bounds, 1, bkd)
        k2 = SquaredExponentialKernel([init_ls[1]], bounds, 1, bkd)
        sep_kernel = SeparableProductKernel([k1, k2], bkd)
        gp_sep = ExactGaussianProcess(sep_kernel, nvars, bkd, nugget=1e-8)
        gp_sep.fit(X_train, y_train)

        # Compare optimal hyperparameters
        params_se = gp_se.hyp_list().get_values()
        params_sep = gp_sep.hyp_list().get_values()
        bkd.assert_allclose(params_sep, params_se, atol=1e-2)

        # Compare NLL values (should be very close)
        nll_se = gp_se.neg_log_marginal_likelihood()
        nll_sep = gp_sep.neg_log_marginal_likelihood()
        bkd.assert_allclose(
            bkd.asarray([nll_sep]),
            bkd.asarray([nll_se]),
            atol=1e-4,
        )

    def test_hyperparameters_combined(self, bkd) -> None:
        """Test hyperparameters are combined from all 1D kernels."""
        np.random.seed(42)

        k1 = SquaredExponentialKernel([1.0], (0.01, 100.0), 1, bkd)
        k2 = SquaredExponentialKernel([2.0], (0.01, 100.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        # Each RBF kernel has 1 length scale parameter
        assert kernel.hyp_list().nparams() == 2

    def test_invalid_nvars_raises_error(self, bkd) -> None:
        """Test that non-1D kernels raise ValueError."""
        np.random.seed(42)

        k1 = SquaredExponentialKernel([1.0], (0.01, 100.0), 1, bkd)
        k2_invalid = SquaredExponentialKernel([1.0, 2.0], (0.01, 100.0), 2, bkd)

        with pytest.raises(ValueError) as context:
            SeparableProductKernel([k1, k2_invalid], bkd)

        assert "nvars=1" in str(context.value)


class TestSeparableKernelProtocol:
    """
    Test that kernels correctly satisfy SeparableKernelProtocol.

    Tests protocol compliance for SeparableProductKernel and
    SquaredExponentialKernel, and verifies that non-separable
    kernels (M32, M52) do NOT satisfy the protocol.
    """

    def test_separable_product_kernel_satisfies_protocol(self, bkd) -> None:
        """SeparableProductKernel should satisfy SeparableKernelProtocol."""
        from pyapprox.surrogates.kernels.protocols import (
            SeparableKernelProtocol,
        )

        np.random.seed(42)
        k1 = SquaredExponentialKernel([1.0], (0.01, 100.0), 1, bkd)
        k2 = SquaredExponentialKernel([2.0], (0.01, 100.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        assert isinstance(kernel, SeparableKernelProtocol)

    def test_ard_se_kernel_satisfies_protocol(self, bkd) -> None:
        """SquaredExponentialKernel with ARD should satisfy SeparableKernelProtocol."""
        from pyapprox.surrogates.kernels.protocols import (
            SeparableKernelProtocol,
        )

        np.random.seed(42)
        kernel = SquaredExponentialKernel([1.0, 2.0], (0.01, 100.0), 2, bkd)

        assert isinstance(kernel, SeparableKernelProtocol)

    def test_matern52_does_not_satisfy_protocol(self, bkd) -> None:
        """Matern52Kernel should NOT satisfy SeparableKernelProtocol."""
        from pyapprox.surrogates.kernels.protocols import (
            SeparableKernelProtocol,
        )

        np.random.seed(42)
        kernel = Matern52Kernel([1.0, 2.0], (0.01, 100.0), 2, bkd)

        assert not isinstance(kernel, SeparableKernelProtocol)

    def test_matern32_does_not_satisfy_protocol(self, bkd) -> None:
        """Matern32Kernel should NOT satisfy SeparableKernelProtocol."""
        from pyapprox.surrogates.kernels.protocols import (
            SeparableKernelProtocol,
        )

        np.random.seed(42)
        kernel = Matern32Kernel([1.0, 2.0], (0.01, 100.0), 2, bkd)

        assert not isinstance(kernel, SeparableKernelProtocol)

    def test_se_kernel_get_kernel_1d_returns_correct_type(self, bkd) -> None:
        """get_kernel_1d should return SquaredExponentialKernel."""
        np.random.seed(42)
        kernel = SquaredExponentialKernel([1.0, 2.0, 3.0], (0.01, 100.0), 3, bkd)

        k0 = kernel.get_kernel_1d(0)
        k1 = kernel.get_kernel_1d(1)
        k2 = kernel.get_kernel_1d(2)

        # Should be same type
        assert isinstance(k0, SquaredExponentialKernel)
        assert isinstance(k1, SquaredExponentialKernel)
        assert isinstance(k2, SquaredExponentialKernel)

        # Should have nvars=1
        assert k0.nvars() == 1
        assert k1.nvars() == 1
        assert k2.nvars() == 1

    def test_se_kernel_get_kernel_1d_correct_length_scales(self, bkd) -> None:
        """get_kernel_1d should return kernel with correct length scale."""
        np.random.seed(42)
        ls = [1.5, 2.5, 3.5]
        kernel = SquaredExponentialKernel(ls, (0.01, 100.0), 3, bkd)

        for dim in range(3):
            k_1d = kernel.get_kernel_1d(dim)
            ls_1d = k_1d._log_lenscale.exp_values()
            bkd.assert_allclose(ls_1d, bkd.asarray([ls[dim]]), rtol=1e-12)

    def test_se_kernel_get_kernel_1d_factorization(self, bkd) -> None:
        """SE kernel should factor correctly: K(x,y) = prod_d K_d(x_d, y_d)."""
        np.random.seed(42)
        ls = [1.5, 2.5]
        kernel = SquaredExponentialKernel(ls, (0.01, 100.0), 2, bkd)

        # Test points
        x = bkd.asarray([[0.3], [0.7]])  # (2, 1)
        y = bkd.asarray([[0.8], [0.2]])  # (2, 1)

        # Full kernel evaluation
        K_full = kernel(x, y)

        # Factored evaluation using 1D kernels
        k0 = kernel.get_kernel_1d(0)
        k1 = kernel.get_kernel_1d(1)

        K0 = k0(bkd.asarray([[0.3]]), bkd.asarray([[0.8]]))
        K1 = k1(bkd.asarray([[0.7]]), bkd.asarray([[0.2]]))
        K_factored = K0 * K1

        bkd.assert_allclose(K_full, K_factored, rtol=1e-12)

    def test_se_kernel_get_kernel_1d_batch_factorization(self, bkd) -> None:
        """SE kernel factorization should work for batched inputs."""
        np.random.seed(42)
        ls = [1.0, 2.0]
        kernel = SquaredExponentialKernel(ls, (0.01, 100.0), 2, bkd)

        # Batched test points
        X1 = bkd.asarray([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  # (2, 3)
        X2 = bkd.asarray([[0.7, 0.9], [0.8, 1.0]])  # (2, 2)

        # Full kernel evaluation
        K_full = kernel(X1, X2)  # (3, 2)

        # Factored evaluation
        k0 = kernel.get_kernel_1d(0)
        k1 = kernel.get_kernel_1d(1)

        K0 = k0(bkd.reshape(X1[0, :], (1, -1)), bkd.reshape(X2[0, :], (1, -1)))
        K1 = k1(bkd.reshape(X1[1, :], (1, -1)), bkd.reshape(X2[1, :], (1, -1)))
        K_factored = K0 * K1

        bkd.assert_allclose(K_full, K_factored, rtol=1e-12)

    def test_se_kernel_get_kernel_1d_out_of_bounds(self, bkd) -> None:
        """get_kernel_1d should raise IndexError for invalid dim."""
        np.random.seed(42)
        kernel = SquaredExponentialKernel([1.0, 2.0], (0.01, 100.0), 2, bkd)

        with pytest.raises(IndexError):
            kernel.get_kernel_1d(-1)

        with pytest.raises(IndexError):
            kernel.get_kernel_1d(2)

        with pytest.raises(IndexError):
            kernel.get_kernel_1d(10)
