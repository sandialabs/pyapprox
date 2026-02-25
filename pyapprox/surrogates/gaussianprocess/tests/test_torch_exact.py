"""
Tests for TorchExactGaussianProcess with autograd derivatives.
"""

import math
import unittest

import torch

from pyapprox.surrogates.gaussianprocess.exact import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.torch_exact import (
    TorchExactGaussianProcess,
)
from pyapprox.surrogates.kernels import Matern52Kernel
from pyapprox.surrogates.kernels.torch_matern import TorchMaternKernel
from pyapprox.util.backends.torch import TorchBkd


def _make_gp(nu=2.5, nvars=1, lenscale=None, nugget=1e-6):
    """Helper to create a TorchExactGP with params set inactive."""
    if lenscale is None:
        lenscale = [1.0] * nvars
    kernel = TorchMaternKernel(
        nu=nu, lenscale=lenscale, lenscale_bounds=(0.1, 10.0), nvars=nvars
    )
    gp = TorchExactGaussianProcess(kernel, nvars=nvars, nugget=nugget)
    # TorchMaternKernel lacks jacobian_wrt_params, so fix all params
    gp.hyp_list().set_all_inactive()
    return gp


class TestTorchExactGaussianProcess(unittest.TestCase):
    """Tests for TorchExactGaussianProcess."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(42)
        self.bkd = TorchBkd()

    def test_inherits_from_exact_gp(self):
        """TorchExactGaussianProcess should inherit from ExactGaussianProcess."""
        gp = _make_gp()
        self.assertIsInstance(gp, ExactGaussianProcess)

    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        gp = _make_gp()

        X_train = torch.linspace(-2, 2, 10).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]

        gp.fit(X_train, y_train)
        self.assertTrue(gp.is_fitted())

        X_test = torch.linspace(-2, 2, 5).reshape(1, -1)
        mean = gp.predict(X_test)

        self.assertEqual(mean.shape, (1, 5))

    def test_predict_interpolates_training_data(self):
        """Test predictions at training points match training values."""
        gp = _make_gp(nugget=1e-10)

        X_train = torch.linspace(-2, 2, 10).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]

        gp.fit(X_train, y_train)

        mean = gp.predict(X_train)

        self.assertTrue(torch.allclose(mean, y_train, rtol=1e-4, atol=1e-4))

    def test_predict_std(self):
        """Test standard deviation prediction."""
        gp = _make_gp(nugget=1e-10)

        X_train = torch.linspace(-2, 2, 10).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]

        gp.fit(X_train, y_train)

        X_test = torch.linspace(-2, 2, 5).reshape(1, -1)
        std = gp.predict_std(X_test)

        self.assertEqual(std.shape, (1, 5))
        self.assertTrue(torch.all(std >= 0))

    def test_std_near_zero_at_training_points(self):
        """Test std is near zero at training points."""
        gp = _make_gp(nugget=1e-10)

        X_train = torch.linspace(-2, 2, 10).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]

        gp.fit(X_train, y_train)

        std = gp.predict_std(X_train)

        self.assertTrue(torch.all(std < 1e-4))

    def test_jacobian_shape(self):
        """Test Jacobian has correct shape."""
        gp = _make_gp(nvars=2, lenscale=[1.0, 1.0])

        X_train = torch.randn(2, 20)
        y_train = torch.randn(1, 20)

        gp.fit(X_train, y_train)

        # Single sample
        X_test = torch.randn(2, 1, requires_grad=True)
        jac = gp.jacobian(X_test)
        self.assertEqual(jac.shape, (1, 2))

        # Multiple samples - use jacobian_batch
        X_test_multi = torch.randn(2, 5, requires_grad=True)
        jac_multi = gp.jacobian_batch(X_test_multi)
        self.assertEqual(jac_multi.shape, (5, 1, 2))

    def test_jacobian_finite_difference(self):
        """Test Jacobian matches finite difference approximation."""
        gp = _make_gp()

        X_train = torch.linspace(-2, 2, 20).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]

        gp.fit(X_train, y_train)

        x_test = torch.tensor([[0.5]], requires_grad=True)
        jac = gp.jacobian(x_test)

        # Finite difference
        eps = 1e-6
        x_plus = torch.tensor([[0.5 + eps]])
        x_minus = torch.tensor([[0.5 - eps]])

        f_plus = gp.predict(x_plus)[0, 0]
        f_minus = gp.predict(x_minus)[0, 0]

        jac_fd = (f_plus - f_minus) / (2 * eps)

        self.assertTrue(torch.allclose(jac[0, 0], jac_fd, rtol=1e-4, atol=1e-5))

    def test_hvp_not_available(self):
        """Test that hvp method is not available (torch.cdist limitation)."""
        gp = _make_gp(nvars=2, lenscale=[1.0, 1.0])

        # HVP should not be available due to torch.cdist limitation
        self.assertFalse(hasattr(gp, "hvp"))

        # Jacobian should still be available
        self.assertTrue(hasattr(gp, "jacobian"))

    def test_neg_log_marginal_likelihood(self):
        """Test NLML computation."""
        gp = _make_gp()

        X_train = torch.linspace(-2, 2, 10).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]

        gp.fit(X_train, y_train)

        nlml = gp.neg_log_marginal_likelihood()

        # Should be a finite number
        self.assertFalse(math.isnan(nlml))
        self.assertFalse(math.isinf(nlml))

    def test_higher_half_integer_nu_gp(self):
        """Test GP with higher half-integer nu values (3.5, 4.5)."""
        for nu in [3.5, 4.5]:
            gp = _make_gp(nu=nu)

            X_train = torch.linspace(-2, 2, 20).reshape(1, -1)
            y_train = torch.sin(X_train[0])[None, :]

            gp.fit(X_train, y_train)

            X_test = torch.linspace(-2, 2, 10).reshape(1, -1)
            mean = gp.predict(X_test)

            self.assertEqual(mean.shape, (1, 10))

            # Check predictions are reasonable
            y_true = torch.sin(X_test[0])[None, :]
            error = torch.abs(mean - y_true).mean()
            self.assertLess(float(error), 0.5)

    def test_call_interface(self):
        """Test __call__ interface for compatibility."""
        gp = _make_gp()

        X_train = torch.linspace(-2, 2, 10).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]

        gp.fit(X_train, y_train)

        X_test = torch.linspace(-2, 2, 5).reshape(1, -1)
        result = gp(X_test)

        # __call__ returns (nqoi, n_test)
        self.assertEqual(result.shape, (1, 5))

    def test_multidimensional_input(self):
        """Test GP with multi-dimensional inputs."""
        gp = _make_gp(nvars=3, lenscale=[1.0, 1.0, 1.0])

        X_train = torch.randn(3, 30)
        y_train = torch.randn(1, 30)

        gp.fit(X_train, y_train)

        X_test = torch.randn(3, 10)
        mean = gp.predict(X_test)

        self.assertEqual(mean.shape, (1, 10))

    def test_repr(self):
        """Test string representation."""
        gp = _make_gp()

        repr_str = repr(gp)
        self.assertIn("TorchExactGaussianProcess", repr_str)
        self.assertIn("not fitted", repr_str)

        X_train = torch.linspace(-2, 2, 10).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]
        gp.fit(X_train, y_train)

        repr_str = repr(gp)
        self.assertIn("fitted", repr_str)

    def test_matches_matern52_predictions(self):
        """Test TorchExactGP with nu=2.5 matches ExactGP with Matern52Kernel."""
        bkd = TorchBkd()

        # Create both GPs with same hyperparameters
        torch_kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0], lenscale_bounds=(0.1, 10.0), nvars=1
        )
        ref_kernel = Matern52Kernel(
            lenscale=[1.0], lenscale_bounds=(0.1, 10.0), nvars=1, bkd=bkd
        )

        torch_gp = TorchExactGaussianProcess(torch_kernel, nvars=1)
        ref_gp = ExactGaussianProcess(ref_kernel, nvars=1, bkd=bkd)

        # Fix hyperparameters on both GPs (no optimization)
        torch_gp.hyp_list().set_all_inactive()
        ref_gp.hyp_list().set_all_inactive()

        # Same training data
        X_train = torch.linspace(-2, 2, 15).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]

        torch_gp.fit(X_train, y_train)
        ref_gp.fit(X_train, y_train)

        # Compare predictions
        X_test = torch.linspace(-2, 2, 10).reshape(1, -1)
        torch_pred = torch_gp.predict(X_test)
        ref_pred = ref_gp.predict(X_test)

        # Predictions should match to high precision
        self.assertTrue(torch.allclose(torch_pred, ref_pred, rtol=1e-6, atol=1e-6))

    def test_matches_matern52_jacobian(self):
        """Test autograd Jacobian matches analytical Jacobian from Matern52."""
        bkd = TorchBkd()

        # Create both GPs
        torch_kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0], lenscale_bounds=(0.1, 10.0), nvars=1
        )
        ref_kernel = Matern52Kernel(
            lenscale=[1.0], lenscale_bounds=(0.1, 10.0), nvars=1, bkd=bkd
        )

        torch_gp = TorchExactGaussianProcess(torch_kernel, nvars=1)
        ref_gp = ExactGaussianProcess(ref_kernel, nvars=1, bkd=bkd)

        # Fix hyperparameters on both GPs (no optimization)
        torch_gp.hyp_list().set_all_inactive()
        ref_gp.hyp_list().set_all_inactive()

        # Same training data
        X_train = torch.linspace(-2, 2, 15).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]

        torch_gp.fit(X_train, y_train)
        ref_gp.fit(X_train, y_train)

        # Compare Jacobians at single point
        X_test = torch.tensor([[0.5]])
        torch_jac = torch_gp.jacobian(X_test)
        ref_jac = ref_gp.jacobian(X_test)

        # Jacobians should match to high precision
        self.assertTrue(torch.allclose(torch_jac, ref_jac, rtol=1e-5, atol=1e-6))

    def test_jacobian_derivative_checker(self):
        """Test Jacobian using DerivativeChecker with finite differences."""
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        bkd = TorchBkd()

        # Create GP with 2D input
        gp = _make_gp(nvars=2, lenscale=[1.0, 1.0])

        # Training data
        X_train = torch.randn(2, 15)
        y_train = torch.sin(X_train[0] + X_train[1])[None, :]
        gp.fit(X_train, y_train)

        # Create derivative checker
        checker = DerivativeChecker(gp)

        # Test point for Jacobian evaluation
        x0 = bkd.array([[0.5], [0.5]])

        # Use logarithmically-spaced step sizes from 1 down to 1e-14
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        # Check Jacobian accuracy
        errors = checker.check_derivatives(
            x0, fd_eps=fd_eps, relative=True, verbosity=0
        )

        # Get Jacobian error (first element of errors list)
        jac_error = errors[0]

        # All errors should be finite
        self.assertTrue(
            bkd.all_bool(bkd.isfinite(jac_error)),
            "Jacobian errors contain non-finite values",
        )

        # Minimum error should be small, showing accuracy with optimal step size
        min_error = float(bkd.min(jac_error))
        self.assertLess(
            min_error,
            1e-6,
            f"Minimum Jacobian relative error {min_error} exceeds threshold",
        )

        # Test that error ratio (min/max) is very small, indicating good convergence
        error_ratio = float(checker.error_ratio(jac_error))
        self.assertLess(
            error_ratio,
            1e-6,
            f"Error ratio {error_ratio:.2e} suggests poor convergence",
        )

    def test_jacobian_derivative_checker_higher_nu(self):
        """Test Jacobian using DerivativeChecker for higher half-integer nu."""
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        bkd = TorchBkd()

        # Test nu=3.5 (higher half-integer)
        gp = _make_gp(nu=3.5)

        # Training data
        X_train = torch.linspace(-2, 2, 20).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]
        gp.fit(X_train, y_train)

        # Create derivative checker
        checker = DerivativeChecker(gp)

        # Test point
        x0 = bkd.array([[0.5]])
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        # Check Jacobian accuracy
        errors = checker.check_derivatives(
            x0, fd_eps=fd_eps, relative=True, verbosity=0
        )

        jac_error = errors[0]
        min_error = float(bkd.min(jac_error))

        # Jacobian should be accurate
        self.assertLess(
            min_error,
            1e-6,
            f"Minimum Jacobian error {min_error} for nu=3.5 exceeds threshold",
        )

        # Error ratio should be small, indicating good convergence
        error_ratio = float(checker.error_ratio(jac_error))
        self.assertLess(
            error_ratio,
            1e-6,
            f"Error ratio {error_ratio:.2e} for nu=3.5 suggests poor convergence",
        )

    def test_optimization_via_autograd(self):
        """Test hyperparameter optimization works via autograd (no analytical
        gradients)."""
        from pyapprox.surrogates.gaussianprocess.gp_loss import (
            GPNegativeLogMarginalLikelihoodLoss,
        )

        # TorchMaternKernel lacks jacobian_wrt_params, so optimization
        # must use autograd through loss.__call__()
        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[0.5], lenscale_bounds=(0.1, 10.0), nvars=1
        )
        gp = TorchExactGaussianProcess(kernel, nvars=1)
        # Leave params active so optimization runs

        X_train = torch.linspace(-2, 2, 20).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]

        # Create loss and configure via GP hook (as fit() would do)
        gp._fit_internal(X_train, y_train)
        loss = GPNegativeLogMarginalLikelihoodLoss(gp, (X_train, y_train))
        gp._configure_loss(loss)

        self.assertFalse(hasattr(kernel, "jacobian_wrt_params"))
        self.assertTrue(
            hasattr(loss, "jacobian"),
            "Autograd jacobian should be bound by _configure_loss",
        )

        # Compute loss and gradient
        params = gp.hyp_list().get_active_values()
        nll = loss(params)
        grad = loss.jacobian(params)

        self.assertEqual(nll.shape, (1, 1))
        self.assertEqual(grad.shape, (1, len(params)))

        # Gradient should be finite
        self.assertTrue(torch.all(torch.isfinite(grad)))

    def test_fit_optimizes_hyperparameters(self):
        """Test that fit() optimizes hyperparameters using autograd gradients."""
        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[0.3], lenscale_bounds=(0.1, 10.0), nvars=1
        )
        gp = TorchExactGaussianProcess(kernel, nvars=1)
        # Leave params active for optimization

        X_train = torch.linspace(-2, 2, 20).reshape(1, -1)
        y_train = torch.sin(X_train[0])[None, :]

        initial_lenscale = kernel.hyp_list().get_values().clone()

        gp.fit(X_train, y_train)

        final_lenscale = kernel.hyp_list().get_values()

        # Hyperparameters should have changed during optimization
        self.assertFalse(
            torch.allclose(initial_lenscale, final_lenscale),
            "Hyperparameters should change during optimization",
        )

        # Predictions should be reasonable
        X_test = torch.linspace(-2, 2, 10).reshape(1, -1)
        mean = gp.predict(X_test)
        y_true = torch.sin(X_test[0])[None, :]
        error = torch.abs(mean - y_true).mean()
        self.assertLess(
            float(error), 0.5, f"Mean prediction error {float(error)} too large"
        )


if __name__ == "__main__":
    unittest.main()
