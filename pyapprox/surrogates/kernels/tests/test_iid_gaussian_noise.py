import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.surrogates.kernels.iid_gaussian_noise import IIDGaussianNoise
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd


class TestIIDGaussianNoise:
    """
    Base test class for IIDGaussianNoise.
    """

    def _setup_data(self, bkd):
        np.random.seed(42)
        self.nvars = 2
        self.nsamples1 = 5
        self.nsamples2 = 4
        self.noise_variance = 0.25
        self.variance_bounds = (0.01, 1.0)

        # Create kernel
        self.kernel = IIDGaussianNoise(
            self.noise_variance, self.variance_bounds, bkd
        )

        # Create sample data
        self.X1 = bkd.array(np.random.randn(self.nvars, self.nsamples1))
        self.X2 = bkd.array(np.random.randn(self.nvars, self.nsamples2))

    def test_initialization(self, bkd) -> None:
        """
        Test IIDGaussianNoise initialization.
        """
        self._setup_data(bkd)
        kernel = IIDGaussianNoise(self.noise_variance, self.variance_bounds, bkd)
        assert kernel.bkd() is not None
        assert kernel.nvars() == 0  # No spatial dependence

    def test_hyperparameter_list(self, bkd) -> None:
        """
        Test that hyperparameter list is set up correctly.
        """
        self._setup_data(bkd)
        hyp_list = self.kernel.hyp_list()
        assert hyp_list.nparams() == 1  # Single noise level parameter

    def test_kernel_matrix_shape_self(self, bkd) -> None:
        """
        Test that IIDGaussianNoise produces correct shape for self-covariance.
        """
        self._setup_data(bkd)
        K = self.kernel(self.X1)
        assert K.shape == (self.nsamples1, self.nsamples1)

    def test_kernel_matrix_shape_cross(self, bkd) -> None:
        """
        Test that IIDGaussianNoise produces correct shape for cross-covariance.
        """
        self._setup_data(bkd)
        K = self.kernel(self.X1, self.X2)
        assert K.shape == (self.nsamples1, self.nsamples2)

    def test_kernel_matrix_self_is_diagonal(self, bkd) -> None:
        """
        Test that self-covariance is a diagonal matrix.

        K(X, X) = sigma^2 * I
        """
        self._setup_data(bkd)
        K = self.kernel(self.X1)

        # Should be diagonal matrix with noise_variance on diagonal
        expected = bkd.eye(self.nsamples1) * self.noise_variance
        bkd.assert_allclose(K, expected)

    def test_kernel_matrix_cross_is_zeros(self, bkd) -> None:
        """
        Test that cross-covariance is zeros.

        K(X, X') = 0 for X != X'
        (no correlation between different inputs)
        """
        self._setup_data(bkd)
        K = self.kernel(self.X1, self.X2)

        # Should be all zeros
        expected = bkd.zeros((self.nsamples1, self.nsamples2))
        bkd.assert_allclose(K, expected)

    def test_diagonal(self, bkd) -> None:
        """
        Test diagonal computation for IIDGaussianNoise.
        """
        self._setup_data(bkd)
        diag = self.kernel.diag(self.X1)

        # Diagonal should be vector of noise_variance values
        expected = bkd.full((self.nsamples1,), self.noise_variance)
        bkd.assert_allclose(diag, expected)

    def test_jacobian_is_zero(self, bkd) -> None:
        """
        Test that Jacobian w.r.t. inputs is zero.

        IIDGaussianNoise has no spatial dependence, so dK/dX = 0.
        """
        self._setup_data(bkd)
        jac = self.kernel.jacobian(self.X1, self.X2)

        # Should be shape (n1, n2, nvars) and all zeros
        expected_shape = (self.nsamples1, self.nsamples2, self.nvars)
        assert jac.shape == expected_shape

        expected = bkd.zeros(expected_shape)
        bkd.assert_allclose(jac, expected)

    def test_param_jacobian(self, bkd) -> None:
        """
        Test parameter Jacobian for IIDGaussianNoise using DerivativeChecker.

        For log-parameterized noise: dK/d(log_sigma^2) = sigma^2 on diagonal
        """
        self._setup_data(bkd)
        # Get active parameter values
        params = self.kernel.hyp_list().get_active_values()

        # Create wrapper function
        def kernel_func_params(p):
            # p has shape (nvars, 1), need to flatten to 1D for set_active_values
            p_flat = bkd.flatten(p)
            self.kernel.hyp_list().set_active_values(p_flat)
            K = self.kernel(self.X1)
            K_flat = bkd.flatten(K)
            # Reshape to (nqoi, nsamp) where nsamp = 1
            return bkd.reshape(K_flat, (K_flat.shape[0], 1))

        def kernel_jac_params(p):
            # p has shape (nvars, 1), need to flatten to 1D for set_active_values
            p_flat = bkd.flatten(p)
            self.kernel.hyp_list().set_active_values(p_flat)
            jac = self.kernel.jacobian_wrt_params(self.X1)
            # jac has shape (n, n, nparams)
            # Flatten to (n*n, nparams) which equals (nqoi, nvars)
            n = self.X1.shape[1]
            jac_flat = bkd.reshape(jac, (n * n, jac.shape[2]))
            return jac_flat

        func_with_jac = FunctionWithJacobianFromCallable(
            self.nsamples1 * self.nsamples1,
            self.kernel.hyp_list().nactive_params(),
            kernel_func_params,
            kernel_jac_params,
            bkd,
        )

        checker = DerivativeChecker(func_with_jac)
        params_reshaped = bkd.reshape(params, (params.shape[0], 1))
        errors = checker.check_derivatives(params_reshaped)
        assert checker.error_ratio(errors[0]) <= 1e-6

    def test_param_jacobian_structure(self, bkd) -> None:
        """
        Test that parameter Jacobian has correct structure.

        For IIDGaussianNoise: dK/d(log_sigma^2) should be diagonal with sigma^2 on diagonal.
        """
        self._setup_data(bkd)
        jac = self.kernel.jacobian_wrt_params(self.X1)

        # Should be shape (n, n, 1)
        expected_shape = (self.nsamples1, self.nsamples1, 1)
        assert jac.shape == expected_shape

        # Check diagonal entries
        for i in range(self.nsamples1):
            bkd.assert_allclose(
                bkd.asarray([float(jac[i, i, 0])]),
                bkd.asarray([self.noise_variance]),
                rtol=1e-10,
            )

        # Check off-diagonal entries are zero
        for i in range(self.nsamples1):
            for j in range(self.nsamples1):
                if i != j:
                    bkd.assert_allclose(
                        bkd.asarray([float(jac[i, j, 0])]),
                        bkd.asarray([0.0]),
                        atol=1e-10,
                    )

    def test_hyperparameter_update(self, bkd) -> None:
        """
        Test that kernel values update when hyperparameters change.
        """
        self._setup_data(bkd)
        # Get initial kernel matrix
        K_before = self.kernel(self.X1)

        # Update noise level
        new_noise = 0.5
        # LogHyperParameter expects log values
        self.kernel.hyp_list().set_active_values(bkd.array([np.log(new_noise)]))

        # Get new kernel matrix
        K_after = self.kernel(self.X1)

        # Values should have changed
        expected = bkd.eye(self.nsamples1) * new_noise
        bkd.assert_allclose(K_after, expected)

        # Should be different from before
        assert not bkd.allclose(K_before, K_after)

    def test_composition_with_matern(self, bkd) -> None:
        """
        Test that IIDGaussianNoise composes correctly with Matern kernel.

        This is the typical use case: signal kernel + noise kernel
        """
        from pyapprox.surrogates.kernels.matern import Matern52Kernel

        self._setup_data(bkd)
        # Create a Matern kernel
        matern = Matern52Kernel([1.0, 1.0], (0.1, 10.0), self.nvars, bkd)

        # Sum: typical GP kernel
        gp_kernel = matern + self.kernel
        K = gp_kernel(self.X1)

        # Should equal Matern + noise on diagonal
        K_matern = matern(self.X1)
        expected = K_matern + bkd.eye(self.nsamples1) * self.noise_variance
        bkd.assert_allclose(K, expected)

    def test_cross_covariance_with_composition(self, bkd) -> None:
        """
        Test cross-covariance with composed kernel.

        When computing K(X1, X2) with X1 != X2, the noise term should be zero.
        """
        from pyapprox.surrogates.kernels.matern import Matern52Kernel

        self._setup_data(bkd)
        matern = Matern52Kernel([1.0, 1.0], (0.1, 10.0), self.nvars, bkd)

        # Sum: typical GP kernel
        gp_kernel = matern + self.kernel
        K_cross = gp_kernel(self.X1, self.X2)

        # Noise term should be zero for cross-covariance
        # So K(X1, X2) = K_matern(X1, X2) + 0
        K_matern_cross = matern(self.X1, self.X2)
        bkd.assert_allclose(K_cross, K_matern_cross)

    def test_fixed_hyperparameter(self, bkd) -> None:
        """
        Test that fixed hyperparameters are not optimized.
        """
        self._setup_data(bkd)
        fixed_kernel = IIDGaussianNoise(
            self.noise_variance, self.variance_bounds, bkd, fixed=True
        )

        # Fixed kernels should have 0 active parameters
        assert fixed_kernel.hyp_list().nactive_params() == 0

    def test_different_input_dimensions(self, bkd) -> None:
        """
        Test that IIDGaussianNoise works with different input dimensions.
        """
        self._setup_data(bkd)
        # Test with 1D input
        X_1d = bkd.array(np.random.randn(1, 10))
        K_1d = self.kernel(X_1d)
        assert K_1d.shape == (10, 10)

        # Test with 5D input
        X_5d = bkd.array(np.random.randn(5, 10))
        K_5d = self.kernel(X_5d)
        assert K_5d.shape == (10, 10)

        # Both should give same result (no spatial dependence)
        bkd.assert_allclose(K_1d, K_5d)

    def test_repr(self, bkd) -> None:
        """
        Test the string representation of IIDGaussianNoise.
        """
        self._setup_data(bkd)
        repr_str = repr(self.kernel)
        assert "IIDGaussianNoise" in repr_str

    def test_hvp_wrt_x1(self, bkd) -> None:
        """
        Test HVP returns zeros (noise doesn't depend on x).

        Since IIDGaussianNoise doesn't depend on spatial inputs x,
        its Hessian with respect to x is zero, and thus HVP should
        always return zeros.
        """
        self._setup_data(bkd)
        # Test points
        X1 = bkd.array(np.random.randn(self.nvars, 3))
        X2 = bkd.array(np.random.randn(self.nvars, 4))
        direction = bkd.array(
            np.random.randn(
                self.nvars,
            )
        )  # Shape (nvars,)

        # Compute HVP
        hvp = self.kernel.hvp_wrt_x1(X1, X2, direction)

        # Should be all zeros with shape (n1, n2, nvars)
        expected_shape = (X1.shape[1], X2.shape[1], self.nvars)
        assert hvp.shape == expected_shape

        # All values should be zero
        zeros = bkd.zeros(expected_shape)
        assert bkd.allclose(hvp, zeros, atol=1e-15)
