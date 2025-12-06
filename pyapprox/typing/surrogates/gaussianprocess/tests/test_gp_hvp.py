"""
Tests for Gaussian Process Hessian-vector products with respect to inputs.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.surrogates.kernels import (
    Matern32Kernel,
    Matern52Kernel,
    SquaredExponentialKernel,
)
from pyapprox.typing.surrogates.kernels.iid_gaussian_noise import IIDGaussianNoise
from pyapprox.typing.surrogates.kernels.scalings import PolynomialScaling
from pyapprox.typing.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker
)
from pyapprox.typing.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianAndHVPFromCallable
)


from pyapprox.typing.util.test_utils import load_tests


class TestGPHVP(Generic[Array], unittest.TestCase):
    """Test Gaussian Process HVP with respect to inputs."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

        # Create a simple 2D GP
        self._nvars = 2
        self._n_train = 10

        # Create kernel
        length_scale = self._bkd.array([0.5, 0.5])
        self._kernel = Matern52Kernel(
            lenscale=length_scale,
            lenscale_bounds=(0.1, 10.0),
            nvars=self._nvars,
            bkd=self._bkd
        )

        # Create GP
        self._gp = ExactGaussianProcess(
            kernel=self._kernel,
            nvars=self._nvars,
            bkd=self._bkd,
            nugget=0.01
        )

        # Generate training data
        X_train = self._bkd.array(np.random.randn(self._nvars, self._n_train))
        y_train = self._bkd.array(np.random.randn(self._n_train, 1))

        # Fit GP
        self._gp.fit(X_train, y_train)

    def test_hvp_shape(self):
        """Test that HVP returns correct shape."""
        # Single sample
        x = self._bkd.array([[0.5], [0.5]])
        v = self._bkd.array([[1.0], [0.0]])

        hvp = self._gp.hvp(x, v)

        # Should have shape (nvars, 1)
        self.assertEqual(hvp.shape, (self._nvars, 1))

    def test_hvp_linearity(self):
        """Test that HVP is linear in direction: H(x)·(aV) = a·H(x)·V."""
        x = self._bkd.array([[0.5], [0.5]])
        v = self._bkd.array([[1.0], [0.5]])
        a = 2.5

        hvp1 = self._gp.hvp(x, v)
        hvp2 = self._gp.hvp(x, v * a)

        # hvp2 should be a * hvp1
        self.assertTrue(
            self._bkd.allclose(hvp2, hvp1 * a, rtol=1e-6, atol=1e-8)
        )

    def test_hvp_with_derivative_checker(self):
        """Test HVP using DerivativeChecker with finite differences."""
        # Create a function wrapper for the GP
        def value_function(x):
            # x shape: (nvars, 1)
            # Predict at x, return shape (1, 1)
            pred = self._gp.predict(x)
            return self._bkd.reshape(pred, (1, 1))

        def jacobian_function(x):
            # x shape: (nvars, 1)
            # GP jacobian returns shape (nqoi, nvars) = (1, nvars)
            # Need to return shape (nqoi, nvars) = (1, nvars)
            jac = self._gp.jacobian(x)
            return jac

        def hvp_function(x, v):
            # x, v shape: (nvars, 1)
            # HVP shape: (nvars, 1) -> flatten for function interface
            hvp = self._gp.hvp(x, v)
            # Return shape (nvars, 1)
            return hvp

        # Wrap the function
        function = FunctionWithJacobianAndHVPFromCallable(
            nvars=self._nvars,
            fun=value_function,
            jacobian=jacobian_function,
            hvp=hvp_function,
            bkd=self._bkd
        )

        # Create derivative checker
        checker = DerivativeChecker(function)

        # Test point
        x0 = self._bkd.array([[0.5], [0.5]])

        # Custom FD step sizes
        fd_eps = self._bkd.flip(self._bkd.logspace(-14, 0, 15))

        # Check derivatives
        errors = checker.check_derivatives(
            x0,
            fd_eps=fd_eps,
            relative=True,
            verbosity=0
        )

        # Verify Jacobian is correct
        jac_error = errors[0]
        self.assertTrue(
            self._bkd.all_bool(self._bkd.isfinite(jac_error))
        )
        jac_ratio = float(checker.error_ratio(jac_error))
        self.assertLess(jac_ratio, 1e-6, f"Jacobian error ratio: {jac_ratio}")

        # Verify HVP is correct
        hvp_error = errors[1]
        self.assertTrue(
            self._bkd.all_bool(self._bkd.isfinite(hvp_error))
        )
        hvp_ratio = float(checker.error_ratio(hvp_error))
        self.assertLess(hvp_ratio, 1e-6, f"HVP error ratio: {hvp_ratio}")

    def test_hvp_multiple_samples(self):
        """Test HVP with multiple samples."""
        # Multiple samples
        X = self._bkd.array(np.random.randn(self._nvars, 3))
        V = self._bkd.array(np.random.randn(self._nvars, 3))

        hvp = self._gp.hvp(X, V)

        # Should have shape (nvars, 3)
        self.assertEqual(hvp.shape, (self._nvars, 3))

        # Each column should match single-sample computation
        for i in range(3):
            x_i = X[:, i:i+1]
            v_i = V[:, i:i+1]
            hvp_i_single = self._gp.hvp(x_i, v_i)

            self.assertTrue(
                self._bkd.allclose(
                    hvp[:, i:i+1],
                    hvp_i_single,
                    rtol=1e-10,
                    atol=1e-12
                )
            )

    def test_hvp_zero_direction(self):
        """Test HVP with zero direction vector."""
        x = self._bkd.array([[0.5], [0.5]])
        v = self._bkd.zeros((self._nvars, 1))

        hvp = self._gp.hvp(x, v)

        # Should be zero
        zero_hvp = self._bkd.zeros((self._nvars, 1))
        self.assertTrue(
            self._bkd.allclose(hvp, zero_hvp, atol=1e-12)
        )

    def test_hvp_coordinate_directions(self):
        """Test HVP in coordinate directions."""
        x = self._bkd.array([[0.5], [0.5]])

        for d in range(self._nvars):
            # Direction along axis d
            v = self._bkd.zeros((self._nvars, 1))
            v[d, 0] = 1.0

            hvp = self._gp.hvp(x, v)

            # HVP should only have non-zero entry in dimension d
            # (approximately, due to cross-terms in Hessian)
            self.assertEqual(hvp.shape, (self._nvars, 1))

    def test_hvp_shape_mismatch_error(self):
        """Test that HVP raises error when shapes don't match."""
        x = self._bkd.array([[0.5], [0.5]])
        v_wrong = self._bkd.array([[1.0]])  # Only 1 variable

        with self.assertRaises(ValueError):
            self._gp.hvp(x, v_wrong)

    def test_hvp_not_fitted_error(self):
        """Test that HVP raises error when GP not fitted."""
        # Create unfitted GP
        gp_unfitted = ExactGaussianProcess(
            kernel=self._kernel,
            nvars=self._nvars,
            bkd=self._bkd
        )

        x = self._bkd.array([[0.5], [0.5]])
        v = self._bkd.array([[1.0], [0.0]])

        with self.assertRaises(RuntimeError):
            gp_unfitted.hvp(x, v)


class TestGPHVPNumpy(TestGPHVP[NDArray[Any]]):
    """Test GP HVP with NumPy backend."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGPHVPTorch(TestGPHVP[torch.Tensor]):
    """Test GP HVP with PyTorch backend."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGPHVPCompositionKernels(Generic[Array], unittest.TestCase):
    """
    Test GP HVP with composition kernels using derivative checker.

    Tests the full pipeline: scaling * matern + noise composition
    used in a GP, verified against finite differences.
    """

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)
        self.nvars = 2
        self.n_train = 20

        # Create sample data
        self.X_train = self._bkd.array(np.random.randn(self.nvars, self.n_train))
        self.y_train = self._bkd.array(np.random.randn(self.n_train, 1))

    def _create_matern_kernel(self, nu: float):
        """Create Matern kernel for given nu value."""
        if nu == 1.5:
            return Matern32Kernel(
                [1.0]*self.nvars, (0.1, 10.0), self.nvars, self._bkd
            )
        elif nu == 2.5:
            return Matern52Kernel(
                [1.0]*self.nvars, (0.1, 10.0), self.nvars, self._bkd
            )
        elif nu == np.inf:
            return SquaredExponentialKernel(
                [1.0]*self.nvars, (0.1, 10.0), self.nvars, self._bkd
            )
        else:
            raise ValueError(f"Unsupported nu value: {nu}")

    def _test_composition_hvp_for_nu(self, nu: float) -> None:
        """
        Test HVP for composition kernel with specific Matern nu.

        Parameters
        ----------
        nu : float
            Matern smoothness parameter (1.5, 2.5, or np.inf)
        """
        # Create composition: scaling * matern + noise
        scaling = PolynomialScaling([0.8], (0.1, 2.0), self._bkd, nvars=self.nvars)
        matern = self._create_matern_kernel(nu)
        noise = IIDGaussianNoise(0.01, (0.001, 0.1), self._bkd)
        kernel = scaling * matern + noise

        # Create and fit GP
        gp = ExactGaussianProcess(kernel=kernel, nvars=self.nvars, bkd=self._bkd)
        gp.fit(self.X_train, self.y_train)

        # Test point and direction
        x_test = self._bkd.array(np.random.randn(self.nvars, 1))
        v_test = self._bkd.array(np.random.randn(self.nvars, 1))
        v_test = v_test / self._bkd.norm(v_test)

        # Compute HVP
        hvp_result = gp.hvp(x_test, v_test)
        self.assertEqual(hvp_result.shape, (self.nvars, 1))

        # Verify with derivative checker
        def mean_func(x_shaped):
            return gp.predict(x_shaped)

        def jac_func(x_shaped):
            return gp.jacobian(x_shaped)

        func_with_hvp = FunctionWithJacobianAndHVPFromCallable(
            nvars=self.nvars,
            fun=mean_func,
            jacobian=jac_func,
            hvp=lambda x, v: gp.hvp(x, v),
            bkd=self._bkd
        )

        checker = DerivativeChecker(func_with_hvp)
        errors = checker.check_derivatives(x_test, direction=v_test, verbosity=0)

        # Verify Jacobian is correct
        jac_error = errors[0]
        self.assertTrue(
            self._bkd.all_bool(self._bkd.isfinite(jac_error))
        )
        jac_ratio = float(checker.error_ratio(jac_error))
        self.assertLess(jac_ratio, 1e-6, f"Jacobian error ratio: {jac_ratio}")

        # Verify HVP is correct
        hvp_error = errors[1]
        self.assertTrue(
            self._bkd.all_bool(self._bkd.isfinite(hvp_error))
        )
        hvp_ratio = float(checker.error_ratio(hvp_error))
        self.assertLess(hvp_ratio, 1e-6, f"HVP error ratio: {hvp_ratio}")

    def test_composition_hvp_matern_1_5(self) -> None:
        """Test composition HVP with Matern nu=1.5."""
        self._test_composition_hvp_for_nu(1.5)

    def test_composition_hvp_matern_2_5(self) -> None:
        """Test composition HVP with Matern nu=2.5."""
        self._test_composition_hvp_for_nu(2.5)

    def test_composition_hvp_matern_inf(self) -> None:
        """Test composition HVP with Matern nu=inf (RBF)."""
        self._test_composition_hvp_for_nu(np.inf)


class TestGPHVPCompositionKernelsNumpy(TestGPHVPCompositionKernels[NDArray[Any]]):
    """Test GP HVP with composition kernels using NumPy backend."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGPHVPCompositionKernelsTorch(TestGPHVPCompositionKernels[torch.Tensor]):
    """Test GP HVP with composition kernels using PyTorch backend."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
