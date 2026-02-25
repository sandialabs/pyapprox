import unittest
from typing import Generic, Any

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.protocols import Backend, Array
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.surrogates.kernels.iid_gaussian_noise import IIDGaussianNoise
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianFromCallable,
)


class TestIIDGaussianNoise(Generic[Array], unittest.TestCase):
    """
    Base test class for IIDGaussianNoise.

    Derived classes must implement the bkd() method to provide the backend.
    """

    __test__ = False

    def setUp(self) -> None:
        """
        Set up test environment for IIDGaussianNoise.
        """
        np.random.seed(42)
        self.nvars = 2
        self.nsamples1 = 5
        self.nsamples2 = 4
        self.noise_variance = 0.25
        self.variance_bounds = (0.01, 1.0)

        # Create kernel
        self.kernel = IIDGaussianNoise(
            self.noise_variance,
            self.variance_bounds,
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
        Test IIDGaussianNoise initialization.
        """
        kernel = IIDGaussianNoise(
            self.noise_variance,
            self.variance_bounds,
            self.bkd()
        )
        self.assertIsNotNone(kernel.bkd())
        self.assertEqual(kernel.nvars(), 0)  # No spatial dependence

    def test_hyperparameter_list(self) -> None:
        """
        Test that hyperparameter list is set up correctly.
        """
        hyp_list = self.kernel.hyp_list()
        self.assertEqual(hyp_list.nparams(), 1)  # Single noise level parameter

    def test_kernel_matrix_shape_self(self) -> None:
        """
        Test that IIDGaussianNoise produces correct shape for self-covariance.
        """
        K = self.kernel(self.X1)
        self.assertEqual(K.shape, (self.nsamples1, self.nsamples1))

    def test_kernel_matrix_shape_cross(self) -> None:
        """
        Test that IIDGaussianNoise produces correct shape for cross-covariance.
        """
        K = self.kernel(self.X1, self.X2)
        self.assertEqual(K.shape, (self.nsamples1, self.nsamples2))

    def test_kernel_matrix_self_is_diagonal(self) -> None:
        """
        Test that self-covariance is a diagonal matrix.

        K(X, X) = σ² * I
        """
        K = self.kernel(self.X1)

        # Should be diagonal matrix with noise_variance on diagonal
        expected = self.bkd().eye(self.nsamples1) * self.noise_variance
        self.bkd().assert_allclose(K, expected)

    def test_kernel_matrix_cross_is_zeros(self) -> None:
        """
        Test that cross-covariance is zeros.

        K(X, X') = 0 for X ≠ X'
        (no correlation between different inputs)
        """
        K = self.kernel(self.X1, self.X2)

        # Should be all zeros
        expected = self.bkd().zeros((self.nsamples1, self.nsamples2))
        self.bkd().assert_allclose(K, expected)

    def test_diagonal(self) -> None:
        """
        Test diagonal computation for IIDGaussianNoise.
        """
        diag = self.kernel.diag(self.X1)

        # Diagonal should be vector of noise_variance values
        expected = self.bkd().full((self.nsamples1,), self.noise_variance)
        self.bkd().assert_allclose(diag, expected)

    def test_jacobian_is_zero(self) -> None:
        """
        Test that Jacobian w.r.t. inputs is zero.

        IIDGaussianNoise has no spatial dependence, so dK/dX = 0.
        """
        jac = self.kernel.jacobian(self.X1, self.X2)

        # Should be shape (n1, n2, nvars) and all zeros
        expected_shape = (self.nsamples1, self.nsamples2, self.nvars)
        self.assertEqual(jac.shape, expected_shape)

        expected = self.bkd().zeros(expected_shape)
        self.bkd().assert_allclose(jac, expected)

    def test_param_jacobian(self) -> None:
        """
        Test parameter Jacobian for IIDGaussianNoise using DerivativeChecker.

        For log-parameterized noise: dK/d(log_σ²) = σ² on diagonal
        """
        # Get active parameter values
        params = self.kernel.hyp_list().get_active_values()

        # Create wrapper function
        def kernel_func_params(p: Array) -> Array:
            # p has shape (nvars, 1), need to flatten to 1D for set_active_values
            p_flat = self.bkd().flatten(p)
            self.kernel.hyp_list().set_active_values(p_flat)
            K = self.kernel(self.X1)
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
        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-6)

    def test_param_jacobian_structure(self) -> None:
        """
        Test that parameter Jacobian has correct structure.

        For IIDGaussianNoise: dK/d(log_σ²) should be diagonal with σ² on diagonal.
        """
        jac = self.kernel.jacobian_wrt_params(self.X1)

        # Should be shape (n, n, 1)
        expected_shape = (self.nsamples1, self.nsamples1, 1)
        self.assertEqual(jac.shape, expected_shape)

        # Check diagonal entries
        for i in range(self.nsamples1):
            self.assertAlmostEqual(
                float(jac[i, i, 0]),
                self.noise_variance,
                places=10
            )

        # Check off-diagonal entries are zero
        for i in range(self.nsamples1):
            for j in range(self.nsamples1):
                if i != j:
                    self.assertAlmostEqual(
                        float(jac[i, j, 0]),
                        0.0,
                        places=10
                    )

    def test_hyperparameter_update(self) -> None:
        """
        Test that kernel values update when hyperparameters change.
        """
        # Get initial kernel matrix
        K_before = self.kernel(self.X1)

        # Update noise level
        new_noise = 0.5
        # LogHyperParameter expects log values
        self.kernel.hyp_list().set_active_values(
            self.bkd().array([np.log(new_noise)])
        )

        # Get new kernel matrix
        K_after = self.kernel(self.X1)

        # Values should have changed
        expected = self.bkd().eye(self.nsamples1) * new_noise
        self.bkd().assert_allclose(K_after, expected)

        # Should be different from before
        self.assertFalse(
            self.bkd().allclose(K_before, K_after)
        )

    def test_composition_with_matern(self) -> None:
        """
        Test that IIDGaussianNoise composes correctly with Matern kernel.

        This is the typical use case: signal kernel + noise kernel
        """
        from pyapprox.surrogates.kernels.matern import Matern52Kernel

        # Create a Matern kernel
        matern = Matern52Kernel(
            [1.0, 1.0],
            (0.1, 10.0),
            self.nvars,
            self.bkd()
        )

        # Sum: typical GP kernel
        gp_kernel = matern + self.kernel
        K = gp_kernel(self.X1)

        # Should equal Matern + noise on diagonal
        K_matern = matern(self.X1)
        expected = K_matern + self.bkd().eye(self.nsamples1) * self.noise_variance
        self.bkd().assert_allclose(K, expected)

    def test_cross_covariance_with_composition(self) -> None:
        """
        Test cross-covariance with composed kernel.

        When computing K(X1, X2) with X1 ≠ X2, the noise term should be zero.
        """
        from pyapprox.surrogates.kernels.matern import Matern52Kernel

        matern = Matern52Kernel(
            [1.0, 1.0],
            (0.1, 10.0),
            self.nvars,
            self.bkd()
        )

        # Sum: typical GP kernel
        gp_kernel = matern + self.kernel
        K_cross = gp_kernel(self.X1, self.X2)

        # Noise term should be zero for cross-covariance
        # So K(X1, X2) = K_matern(X1, X2) + 0
        K_matern_cross = matern(self.X1, self.X2)
        self.bkd().assert_allclose(K_cross, K_matern_cross)

    def test_fixed_hyperparameter(self) -> None:
        """
        Test that fixed hyperparameters are not optimized.
        """
        fixed_kernel = IIDGaussianNoise(
            self.noise_variance,
            self.variance_bounds,
            self.bkd(),
            fixed=True
        )

        # Fixed kernels should have 0 active parameters
        self.assertEqual(fixed_kernel.hyp_list().nactive_params(), 0)

    def test_different_input_dimensions(self) -> None:
        """
        Test that IIDGaussianNoise works with different input dimensions.
        """
        # Test with 1D input
        X_1d = self.bkd().array(np.random.randn(1, 10))
        K_1d = self.kernel(X_1d)
        self.assertEqual(K_1d.shape, (10, 10))

        # Test with 5D input
        X_5d = self.bkd().array(np.random.randn(5, 10))
        K_5d = self.kernel(X_5d)
        self.assertEqual(K_5d.shape, (10, 10))

        # Both should give same result (no spatial dependence)
        self.bkd().assert_allclose(K_1d, K_5d)

    def test_repr(self) -> None:
        """
        Test the string representation of IIDGaussianNoise.
        """
        repr_str = repr(self.kernel)
        self.assertIn("IIDGaussianNoise", repr_str)

    def test_hvp_wrt_x1(self) -> None:
        """
        Test HVP returns zeros (noise doesn't depend on x).

        Since IIDGaussianNoise doesn't depend on spatial inputs x,
        its Hessian with respect to x is zero, and thus HVP should
        always return zeros.
        """
        # Test points
        X1 = self.bkd().array(np.random.randn(self.nvars, 3))
        X2 = self.bkd().array(np.random.randn(self.nvars, 4))
        direction = self.bkd().array(np.random.randn(self.nvars,))  # Shape (nvars,)

        # Compute HVP
        hvp = self.kernel.hvp_wrt_x1(X1, X2, direction)

        # Should be all zeros with shape (n1, n2, nvars)
        expected_shape = (X1.shape[1], X2.shape[1], self.nvars)
        self.assertEqual(hvp.shape, expected_shape)

        # All values should be zero
        zeros = self.bkd().zeros(expected_shape)
        self.assertTrue(
            self.bkd().allclose(hvp, zeros, atol=1e-15)
        )


# NumPy implementation
class TestIIDGaussianNoiseNumpy(TestIIDGaussianNoise[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch implementation
class TestIIDGaussianNoiseTorch(TestIIDGaussianNoise[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


from pyapprox.util.test_utils import load_tests


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner()
    runner.run(suite)
