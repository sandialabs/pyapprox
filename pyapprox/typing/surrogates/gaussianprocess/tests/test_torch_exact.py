"""
Tests for TorchExactGaussianProcess with autograd derivatives.
"""

import unittest
import math

import torch

from pyapprox.typing.surrogates.kernels.torch_matern import TorchMaternKernel
from pyapprox.typing.surrogates.kernels import Matern52Kernel
from pyapprox.typing.surrogates.gaussianprocess.torch_exact import (
    TorchExactGaussianProcess
)
from pyapprox.typing.surrogates.gaussianprocess.exact import (
    ExactGaussianProcess
)
from pyapprox.typing.util.backends.torch import TorchBkd


class TestTorchExactGaussianProcess(unittest.TestCase):
    """Tests for TorchExactGaussianProcess."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(42)
        self.bkd = TorchBkd()

    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0), nvars=1
        )
        gp = TorchExactGaussianProcess(kernel, nvars=1)

        # Generate training data
        X_train = torch.linspace(-2, 2, 10).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]  # Shape: (1, n_train)

        gp.fit(X_train, y_train)
        self.assertTrue(gp.is_fitted())

        # Predict
        X_test = torch.linspace(-2, 2, 5).reshape(1, -1)
        mean = gp.predict(X_test)

        self.assertEqual(mean.shape, (1, 5))

    def test_predict_interpolates_training_data(self):
        """Test predictions at training points match training values."""
        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0), nvars=1
        )
        gp = TorchExactGaussianProcess(kernel, nvars=1, nugget=1e-10)

        X_train = torch.linspace(-2, 2, 10).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]  # Shape: (1, n_train)

        gp.fit(X_train, y_train)

        # Predict at training points
        mean = gp.predict(X_train)

        self.assertTrue(torch.allclose(mean, y_train, rtol=1e-4, atol=1e-4))

    def test_predict_std(self):
        """Test standard deviation prediction."""
        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0), nvars=1
        )
        gp = TorchExactGaussianProcess(kernel, nvars=1, nugget=1e-10)

        X_train = torch.linspace(-2, 2, 10).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]  # Shape: (1, n_train)

        gp.fit(X_train, y_train)

        X_test = torch.linspace(-2, 2, 5).reshape(1, -1)
        std = gp.predict_std(X_test)

        self.assertEqual(std.shape, (1, 5))
        self.assertTrue(torch.all(std >= 0))

    def test_std_near_zero_at_training_points(self):
        """Test std is near zero at training points."""
        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0), nvars=1
        )
        gp = TorchExactGaussianProcess(kernel, nvars=1, nugget=1e-10)

        X_train = torch.linspace(-2, 2, 10).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]  # Shape: (1, n_train)

        gp.fit(X_train, y_train)

        std = gp.predict_std(X_train)

        self.assertTrue(torch.all(std < 1e-4))

    def test_jacobian_shape(self):
        """Test Jacobian has correct shape."""
        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0, 1.0],
            lenscale_bounds=(0.1, 10.0), nvars=2
        )
        gp = TorchExactGaussianProcess(kernel, nvars=2)

        X_train = torch.randn(2, 20)
        y_train = torch.randn(1, 20)  # Shape: (nqoi, n_train)

        gp.fit(X_train, y_train)

        # Single sample
        X_test = torch.randn(2, 1, requires_grad=True)
        jac = gp.jacobian(X_test)
        self.assertEqual(jac.shape, (1, 2))  # (nqoi, nvars)

        # Multiple samples - use jacobian_batch
        X_test_multi = torch.randn(2, 5, requires_grad=True)
        jac_multi = gp.jacobian_batch(X_test_multi)
        self.assertEqual(jac_multi.shape, (5, 1, 2))  # (n_samples, nqoi, nvars)

    def test_jacobian_finite_difference(self):
        """Test Jacobian matches finite difference approximation."""
        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0), nvars=1
        )
        gp = TorchExactGaussianProcess(kernel, nvars=1)

        X_train = torch.linspace(-2, 2, 20).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]  # Shape: (1, n_train)

        gp.fit(X_train, y_train)

        x_test = torch.tensor([[0.5]], requires_grad=True)
        jac = gp.jacobian(x_test)

        # Finite difference
        eps = 1e-6
        x_plus = torch.tensor([[0.5 + eps]])
        x_minus = torch.tensor([[0.5 - eps]])

        f_plus = gp.predict(x_plus)[0, 0]  # Shape: (1, 1) -> scalar
        f_minus = gp.predict(x_minus)[0, 0]  # Shape: (1, 1) -> scalar

        jac_fd = (f_plus - f_minus) / (2 * eps)

        self.assertTrue(torch.allclose(
            jac[0, 0], jac_fd, rtol=1e-4, atol=1e-5
        ))

    def test_hvp_not_available(self):
        """Test that hvp method is not available (torch.cdist limitation)."""
        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0, 1.0],
            lenscale_bounds=(0.1, 10.0), nvars=2
        )
        gp = TorchExactGaussianProcess(kernel, nvars=2)

        # HVP should not be available due to torch.cdist limitation
        self.assertFalse(hasattr(gp, 'hvp'))

        # Jacobian should still be available
        self.assertTrue(hasattr(gp, 'jacobian'))

    def test_neg_log_marginal_likelihood(self):
        """Test NLML computation."""
        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0), nvars=1
        )
        gp = TorchExactGaussianProcess(kernel, nvars=1)

        X_train = torch.linspace(-2, 2, 10).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]  # Shape: (1, n_train)

        gp.fit(X_train, y_train)

        nlml = gp.neg_log_marginal_likelihood()

        # Should be a finite number
        self.assertFalse(math.isnan(nlml))
        self.assertFalse(math.isinf(nlml))

    def test_higher_half_integer_nu_gp(self):
        """Test GP with higher half-integer nu values (3.5, 4.5)."""
        # Test nu=3.5 which is a half-integer value
        for nu in [3.5, 4.5]:
            kernel = TorchMaternKernel(
                nu=nu, lenscale=[1.0],
                lenscale_bounds=(0.1, 10.0), nvars=1
            )
            gp = TorchExactGaussianProcess(kernel, nvars=1)

            X_train = torch.linspace(-2, 2, 20).reshape(1, -1)
            y_train = torch.sin(X_train[0])[None, :]  # Shape: (1, n_train)

            gp.fit(X_train, y_train)

            X_test = torch.linspace(-2, 2, 10).reshape(1, -1)
            mean = gp.predict(X_test)

            self.assertEqual(mean.shape, (1, 10))

            # Check predictions are reasonable
            y_true = torch.sin(X_test[0])[None, :]  # Shape: (1, n_test)
            error = torch.abs(mean - y_true).mean()
            self.assertLess(float(error), 0.5)

    def test_call_interface(self):
        """Test __call__ interface for compatibility."""
        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0), nvars=1
        )
        gp = TorchExactGaussianProcess(kernel, nvars=1)

        X_train = torch.linspace(-2, 2, 10).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]  # Shape: (1, n_train)

        gp.fit(X_train, y_train)

        X_test = torch.linspace(-2, 2, 5).reshape(1, -1)
        result = gp(X_test)

        # __call__ returns (nqoi, n_test)
        self.assertEqual(result.shape, (1, 5))

    def test_multidimensional_input(self):
        """Test GP with multi-dimensional inputs."""
        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0, 1.0, 1.0],
            lenscale_bounds=(0.1, 10.0), nvars=3
        )
        gp = TorchExactGaussianProcess(kernel, nvars=3)

        X_train = torch.randn(3, 30)
        y_train = torch.randn(1, 30)  # Shape: (nqoi, n_train)

        gp.fit(X_train, y_train)

        X_test = torch.randn(3, 10)
        mean = gp.predict(X_test)

        self.assertEqual(mean.shape, (1, 10))

    def test_repr(self):
        """Test string representation."""
        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0), nvars=1
        )
        gp = TorchExactGaussianProcess(kernel, nvars=1)

        repr_str = repr(gp)
        self.assertIn("TorchExactGaussianProcess", repr_str)
        self.assertIn("not fitted", repr_str)

        X_train = torch.linspace(-2, 2, 10).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]  # Shape: (1, n_train)
        gp.fit(X_train, y_train)

        repr_str = repr(gp)
        self.assertIn("fitted", repr_str)

    def test_matches_matern52_predictions(self):
        """Test TorchExactGP with nu=2.5 matches ExactGP with Matern52Kernel."""
        bkd = TorchBkd()

        # Create both GPs with same hyperparameters
        torch_kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0), nvars=1
        )
        ref_kernel = Matern52Kernel(
            lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0),
            nvars=1,
            bkd=bkd
        )

        torch_gp = TorchExactGaussianProcess(torch_kernel, nvars=1)
        ref_gp = ExactGaussianProcess(ref_kernel, nvars=1, bkd=bkd)

        # Fix hyperparameters on ref_gp to match TorchExactGP (no optimization)
        ref_gp.hyp_list().set_all_inactive()

        # Same training data
        X_train = torch.linspace(-2, 2, 15).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]  # Shape: (1, n_train)

        torch_gp.fit(X_train, y_train)
        ref_gp.fit(X_train, y_train)

        # Compare predictions
        X_test = torch.linspace(-2, 2, 10).reshape(1, -1)
        torch_pred = torch_gp.predict(X_test)
        ref_pred = ref_gp.predict(X_test)

        # Predictions should match to high precision
        self.assertTrue(torch.allclose(
            torch_pred, ref_pred, rtol=1e-6, atol=1e-6
        ))

    def test_matches_matern52_jacobian(self):
        """Test autograd Jacobian matches analytical Jacobian from Matern52."""
        bkd = TorchBkd()

        # Create both GPs
        torch_kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0), nvars=1
        )
        ref_kernel = Matern52Kernel(
            lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0),
            nvars=1,
            bkd=bkd
        )

        torch_gp = TorchExactGaussianProcess(torch_kernel, nvars=1)
        ref_gp = ExactGaussianProcess(ref_kernel, nvars=1, bkd=bkd)

        # Fix hyperparameters on ref_gp to match TorchExactGP (no optimization)
        ref_gp.hyp_list().set_all_inactive()

        # Same training data
        X_train = torch.linspace(-2, 2, 15).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]  # Shape: (1, n_train)

        torch_gp.fit(X_train, y_train)
        ref_gp.fit(X_train, y_train)

        # Compare Jacobians at single point
        X_test = torch.tensor([[0.5]])
        torch_jac = torch_gp.jacobian(X_test)
        ref_jac = ref_gp.jacobian(X_test)

        # Jacobians should match to high precision
        self.assertTrue(torch.allclose(
            torch_jac, ref_jac, rtol=1e-5, atol=1e-6
        ))

    def test_jacobian_derivative_checker(self):
        """Test Jacobian using DerivativeChecker with finite differences."""
        from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker
        )

        bkd = TorchBkd()

        # Create GP with 2D input
        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0, 1.0],
            lenscale_bounds=(0.1, 10.0), nvars=2
        )
        gp = TorchExactGaussianProcess(kernel, nvars=2)

        # Training data
        X_train = torch.randn(2, 15)
        y_train = torch.sin(X_train[0] + X_train[1])[None, :]  # Shape: (1, n_train)
        gp.fit(X_train, y_train)

        # Create derivative checker
        checker = DerivativeChecker(gp)

        # Test point for Jacobian evaluation
        x0 = bkd.array([[0.5], [0.5]])

        # Use logarithmically-spaced step sizes from 1 down to 1e-14
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        # Check Jacobian accuracy
        errors = checker.check_derivatives(
            x0,
            fd_eps=fd_eps,
            relative=True,
            verbosity=0
        )

        # Get Jacobian error (first element of errors list)
        jac_error = errors[0]

        # All errors should be finite
        self.assertTrue(bkd.all_bool(bkd.isfinite(jac_error)),
                       "Jacobian errors contain non-finite values")

        # Minimum error should be small, showing accuracy with optimal step size
        min_error = float(bkd.min(jac_error))
        self.assertLess(min_error, 1e-6,
                       f"Minimum Jacobian relative error {min_error} exceeds threshold")

        # Test that error ratio (min/max) is very small, indicating good convergence
        error_ratio = float(checker.error_ratio(jac_error))
        self.assertLess(error_ratio, 1e-6,
                       f"Error ratio {error_ratio:.2e} suggests poor convergence")

    def test_jacobian_derivative_checker_higher_nu(self):
        """Test Jacobian using DerivativeChecker for higher half-integer nu."""
        from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker
        )

        bkd = TorchBkd()

        # Test nu=3.5 (higher half-integer)
        kernel = TorchMaternKernel(
            nu=3.5, lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0), nvars=1
        )
        gp = TorchExactGaussianProcess(kernel, nvars=1)

        # Training data
        X_train = torch.linspace(-2, 2, 20).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]  # Shape: (1, n_train)
        gp.fit(X_train, y_train)

        # Create derivative checker
        checker = DerivativeChecker(gp)

        # Test point
        x0 = bkd.array([[0.5]])
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        # Check Jacobian accuracy
        errors = checker.check_derivatives(
            x0,
            fd_eps=fd_eps,
            relative=True,
            verbosity=0
        )

        jac_error = errors[0]
        min_error = float(bkd.min(jac_error))

        # Jacobian should be accurate
        self.assertLess(min_error, 1e-6,
                       f"Minimum Jacobian error {min_error} for nu=3.5 exceeds threshold")

        # Error ratio should be small, indicating good convergence
        error_ratio = float(checker.error_ratio(jac_error))
        self.assertLess(error_ratio, 1e-6,
                       f"Error ratio {error_ratio:.2e} for nu=3.5 suggests poor convergence")


if __name__ == "__main__":
    unittest.main()
