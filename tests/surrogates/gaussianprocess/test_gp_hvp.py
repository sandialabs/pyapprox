"""
Tests for Gaussian Process Hessian-vector products with respect to inputs.
"""

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianAndHVPFromCallable,
)
from pyapprox.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.surrogates.kernels import (
    Matern32Kernel,
    Matern52Kernel,
    SquaredExponentialKernel,
)
from pyapprox.surrogates.kernels.iid_gaussian_noise import IIDGaussianNoise
from pyapprox.surrogates.kernels.scalings import PolynomialScaling


class TestGPHVP:
    """Test Gaussian Process HVP with respect to inputs."""

    def _setup_gp(self, bkd):
        """Set up a simple 2D GP for testing."""
        np.random.seed(42)
        nvars = 2
        n_train = 10

        # Create kernel
        length_scale = bkd.array([0.5, 0.5])
        kernel = Matern52Kernel(
            lenscale=length_scale,
            lenscale_bounds=(0.1, 10.0),
            nvars=nvars,
            bkd=bkd,
        )

        # Create GP
        gp = ExactGaussianProcess(
            kernel=kernel, nvars=nvars, bkd=bkd, nugget=0.01
        )

        # Generate training data
        X_train = bkd.array(np.random.randn(nvars, n_train))
        y_train = bkd.array(
            np.random.randn(1, n_train)
        )  # Shape: (1, n_train)

        # Fit GP
        gp.fit(X_train, y_train)

        return gp, kernel, nvars

    def test_hvp_shape(self, bkd):
        """Test that HVP returns correct shape."""
        gp, _, nvars = self._setup_gp(bkd)

        # Single sample
        x = bkd.array([[0.5], [0.5]])
        v = bkd.array([[1.0], [0.0]])

        hvp = gp.hvp(x, v)

        # Should have shape (nvars, 1)
        assert hvp.shape == (nvars, 1)

    def test_hvp_linearity(self, bkd):
        """Test that HVP is linear in direction: H(x)·(aV) = a·H(x)·V."""
        gp, _, _ = self._setup_gp(bkd)

        x = bkd.array([[0.5], [0.5]])
        v = bkd.array([[1.0], [0.5]])
        a = 2.5

        hvp1 = gp.hvp(x, v)
        hvp2 = gp.hvp(x, v * a)

        # hvp2 should be a * hvp1
        assert bkd.allclose(hvp2, hvp1 * a, rtol=1e-6, atol=1e-8)

    def test_hvp_with_derivative_checker(self, bkd):
        """Test HVP using DerivativeChecker with finite differences."""
        gp, _, nvars = self._setup_gp(bkd)

        # Create a function wrapper for the GP
        def value_function(x):
            # x shape: (nvars, 1)
            # Predict at x, return shape (1, 1)
            pred = gp.predict(x)
            return bkd.reshape(pred, (1, 1))

        def jacobian_function(x):
            # x shape: (nvars, 1)
            # GP jacobian returns shape (nqoi, nvars) = (1, nvars)
            # Need to return shape (nqoi, nvars) = (1, nvars)
            jac = gp.jacobian(x)
            return jac

        def hvp_function(x, v):
            # x, v shape: (nvars, 1)
            # HVP shape: (nvars, 1) -> flatten for function interface
            hvp = gp.hvp(x, v)
            # Return shape (nvars, 1)
            return hvp

        # Wrap the function
        function = FunctionWithJacobianAndHVPFromCallable(
            nvars=nvars,
            fun=value_function,
            jacobian=jacobian_function,
            hvp=hvp_function,
            bkd=bkd,
        )

        # Create derivative checker
        checker = DerivativeChecker(function)

        # Test point
        x0 = bkd.array([[0.5], [0.5]])

        # Custom FD step sizes
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        # Check derivatives
        errors = checker.check_derivatives(
            x0, fd_eps=fd_eps, relative=True, verbosity=0
        )

        # Verify Jacobian is correct
        jac_error = errors[0]
        assert bkd.all_bool(bkd.isfinite(jac_error))
        jac_ratio = float(checker.error_ratio(jac_error))
        assert jac_ratio < 1e-6, f"Jacobian error ratio: {jac_ratio}"

        # Verify HVP is correct
        hvp_error = errors[1]
        assert bkd.all_bool(bkd.isfinite(hvp_error))
        hvp_ratio = float(checker.error_ratio(hvp_error))
        assert hvp_ratio < 1e-6, f"HVP error ratio: {hvp_ratio}"

    def test_hvp_multiple_samples(self, bkd):
        """Test HVP with multiple samples using hvp_batch."""
        gp, _, nvars = self._setup_gp(bkd)

        # Multiple samples
        X = bkd.array(np.random.randn(nvars, 3))
        V = bkd.array(np.random.randn(nvars, 3))

        hvp = gp.hvp_batch(X, V)

        # Should have shape (n_samples, nvars) = (3, nvars)
        assert hvp.shape == (3, nvars)

        # Each row should match single-sample computation
        for i in range(3):
            x_i = X[:, i : i + 1]
            v_i = V[:, i : i + 1]
            hvp_i_single = gp.hvp(x_i, v_i)  # (nvars, 1)

            bkd.assert_allclose(
                hvp[i, :], hvp_i_single[:, 0], rtol=1e-10, atol=1e-12
            )

    def test_hvp_zero_direction(self, bkd):
        """Test HVP with zero direction vector."""
        gp, _, nvars = self._setup_gp(bkd)

        x = bkd.array([[0.5], [0.5]])
        v = bkd.zeros((nvars, 1))

        hvp = gp.hvp(x, v)

        # Should be zero
        zero_hvp = bkd.zeros((nvars, 1))
        assert bkd.allclose(hvp, zero_hvp, atol=1e-12)

    def test_hvp_coordinate_directions(self, bkd):
        """Test HVP in coordinate directions."""
        gp, _, nvars = self._setup_gp(bkd)

        x = bkd.array([[0.5], [0.5]])

        for d in range(nvars):
            # Direction along axis d
            v = bkd.zeros((nvars, 1))
            v[d, 0] = 1.0

            hvp = gp.hvp(x, v)

            # HVP should only have non-zero entry in dimension d
            # (approximately, due to cross-terms in Hessian)
            assert hvp.shape == (nvars, 1)

    def test_hvp_shape_mismatch_error(self, bkd):
        """Test that HVP raises error when shapes don't match."""
        gp, _, _ = self._setup_gp(bkd)

        x = bkd.array([[0.5], [0.5]])
        v_wrong = bkd.array([[1.0]])  # Only 1 variable

        with pytest.raises(ValueError):
            gp.hvp(x, v_wrong)

    def test_hvp_not_fitted_error(self, bkd):
        """Test that HVP raises error when GP not fitted."""
        gp, kernel, nvars = self._setup_gp(bkd)

        # Create unfitted GP
        gp_unfitted = ExactGaussianProcess(
            kernel=kernel, nvars=nvars, bkd=bkd
        )

        x = bkd.array([[0.5], [0.5]])
        v = bkd.array([[1.0], [0.0]])

        with pytest.raises(RuntimeError):
            gp_unfitted.hvp(x, v)


class TestGPHVPCompositionKernels:
    """
    Test GP HVP with composition kernels using derivative checker.

    Tests the full pipeline: scaling * matern + noise composition
    used in a GP, verified against finite differences.
    """

    def _setup(self, bkd):
        """Set up test data."""
        np.random.seed(42)
        nvars = 2
        n_train = 20

        X_train = bkd.array(np.random.randn(nvars, n_train))
        y_train = bkd.array(
            np.random.randn(1, n_train)
        )  # Shape: (1, n_train)

        return nvars, X_train, y_train

    def _create_matern_kernel(self, nu, nvars, bkd):
        """Create Matern kernel for given nu value."""
        if nu == 1.5:
            return Matern32Kernel(
                [1.0] * nvars, (0.1, 10.0), nvars, bkd
            )
        elif nu == 2.5:
            return Matern52Kernel(
                [1.0] * nvars, (0.1, 10.0), nvars, bkd
            )
        elif nu == np.inf:
            return SquaredExponentialKernel(
                [1.0] * nvars, (0.1, 10.0), nvars, bkd
            )
        else:
            raise ValueError(f"Unsupported nu value: {nu}")

    def _test_composition_hvp_for_nu(self, nu, bkd) -> None:
        """
        Test HVP for composition kernel with specific Matern nu.

        Parameters
        ----------
        nu : float
            Matern smoothness parameter (1.5, 2.5, or np.inf)
        """
        nvars, X_train, y_train = self._setup(bkd)

        # Create composition: scaling * matern + noise
        scaling = PolynomialScaling([0.8], (0.1, 2.0), bkd, nvars=nvars)
        matern = self._create_matern_kernel(nu, nvars, bkd)
        noise = IIDGaussianNoise(0.01, (0.001, 0.1), bkd)
        kernel = scaling * matern + noise

        # Create and fit GP (with fixed hyperparameters to skip optimization)
        gp = ExactGaussianProcess(kernel=kernel, nvars=nvars, bkd=bkd)
        gp.hyp_list().set_all_inactive()  # Skip optimization for HVP test
        gp.fit(X_train, y_train)

        # Test point and direction
        x_test = bkd.array(np.random.randn(nvars, 1))
        v_test = bkd.array(np.random.randn(nvars, 1))
        v_test = v_test / bkd.norm(v_test)

        # Compute HVP
        hvp_result = gp.hvp(x_test, v_test)
        assert hvp_result.shape == (nvars, 1)

        # Verify with derivative checker
        def mean_func(x_shaped):
            return gp.predict(x_shaped)

        def jac_func(x_shaped):
            return gp.jacobian(x_shaped)

        func_with_hvp = FunctionWithJacobianAndHVPFromCallable(
            nvars=nvars,
            fun=mean_func,
            jacobian=jac_func,
            hvp=lambda x, v: gp.hvp(x, v),
            bkd=bkd,
        )

        checker = DerivativeChecker(func_with_hvp)
        errors = checker.check_derivatives(x_test, direction=v_test, verbosity=0)

        # Verify Jacobian is correct
        jac_error = errors[0]
        assert bkd.all_bool(bkd.isfinite(jac_error))
        jac_ratio = float(checker.error_ratio(jac_error))
        assert jac_ratio < 2e-6, f"Jacobian error ratio: {jac_ratio}"

        # Verify HVP is correct
        hvp_error = errors[1]
        assert bkd.all_bool(bkd.isfinite(hvp_error))
        hvp_ratio = float(checker.error_ratio(hvp_error))
        assert hvp_ratio < 2e-6, f"HVP error ratio: {hvp_ratio}"

    def test_composition_hvp_matern_1_5(self, bkd) -> None:
        """Test composition HVP with Matern nu=1.5."""
        self._test_composition_hvp_for_nu(1.5, bkd)

    def test_composition_hvp_matern_2_5(self, bkd) -> None:
        """Test composition HVP with Matern nu=2.5."""
        self._test_composition_hvp_for_nu(2.5, bkd)

    def test_composition_hvp_matern_inf(self, bkd) -> None:
        """Test composition HVP with Matern nu=inf (RBF)."""
        self._test_composition_hvp_for_nu(np.inf, bkd)
