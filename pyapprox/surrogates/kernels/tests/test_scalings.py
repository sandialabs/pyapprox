"""
Tests for scaling functions.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.protocols import Array
from pyapprox.surrogates.kernels.scalings import PolynomialScaling


from pyapprox.util.test_utils import load_tests


class TestPolynomialScaling(Generic[Array], unittest.TestCase):
    """Base class for PolynomialScaling tests."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._nvars = 1
        self._bounds = (0.1, 2.0)

    def test_constant_scaling(self):
        """Test degree 0 (constant) scaling."""
        # ρ(x) = 0.8 (constant)
        scaling = PolynomialScaling([0.8], self._bounds, self._bkd, nvars=self._nvars)

        X = self._bkd.array([[-1.0, 0.0, 1.0]])
        rho = scaling.eval_scaling(X)

        # All values should be 0.8
        expected = self._bkd.ones((3, 1)) * 0.8
        self.assertTrue(
            self._bkd.allclose(rho, expected)
        )

    def test_constant_scaling_jacobian(self):
        """Test that constant scaling has zero spatial Jacobian."""
        scaling = PolynomialScaling([0.8], self._bounds, self._bkd, nvars=self._nvars)

        X = self._bkd.array([[-1.0, 0.0, 1.0]])
        jac_x = scaling.jacobian_scaling(X)

        # Spatial derivative should be zero
        expected = self._bkd.zeros((3, 1))
        self.assertTrue(
            self._bkd.allclose(jac_x, expected)
        )

    def test_constant_scaling_param_jacobian(self):
        """Test parameter Jacobian for constant scaling."""
        scaling = PolynomialScaling([0.8], self._bounds, self._bkd, nvars=self._nvars)

        X = self._bkd.array([[-1.0, 0.0, 1.0]])
        jac_params = scaling.jacobian_wrt_params(X)

        # ∂ρ/∂c0 = 1
        expected = self._bkd.ones((3, 1))
        self.assertTrue(
            self._bkd.allclose(jac_params, expected)
        )

    def test_linear_scaling_1d(self):
        """Test degree 1 (linear) scaling in 1D."""
        # ρ(x) = 0.9 + 0.1*x
        scaling = PolynomialScaling([0.9, 0.1], self._bounds, self._bkd)

        X = self._bkd.array([[-1.0, 0.0, 1.0]])
        rho = scaling.eval_scaling(X)

        # Expected: [0.8, 0.9, 1.0]
        expected = self._bkd.array([[0.8], [0.9], [1.0]])
        self.assertTrue(
            self._bkd.allclose(rho, expected, rtol=1e-6, atol=1e-7)
        )

    def test_linear_scaling_jacobian_1d(self):
        """Test spatial Jacobian for linear scaling in 1D."""
        # ρ(x) = 0.9 + 0.1*x
        scaling = PolynomialScaling([0.9, 0.1], self._bounds, self._bkd)

        X = self._bkd.array([[-1.0, 0.0, 1.0]])
        jac_x = scaling.jacobian_scaling(X)

        # ∂ρ/∂x = 0.1 (constant slope)
        expected = self._bkd.ones((3, 1)) * 0.1
        self.assertTrue(
            self._bkd.allclose(jac_x, expected, rtol=1e-6, atol=1e-7)
        )

    def test_linear_scaling_param_jacobian_1d(self):
        """Test parameter Jacobian for linear scaling in 1D."""
        # ρ(x) = 0.9 + 0.1*x
        scaling = PolynomialScaling([0.9, 0.1], self._bounds, self._bkd)

        X = self._bkd.array([[-1.0, 0.0, 1.0]])
        jac_params = scaling.jacobian_wrt_params(X)

        # ∂ρ/∂c0 = 1, ∂ρ/∂c1 = x
        expected = self._bkd.hstack([
            self._bkd.ones((3, 1)),
            self._bkd.array([[-1.0], [0.0], [1.0]])
        ])
        self.assertTrue(
            self._bkd.allclose(jac_params, expected, rtol=1e-10)
        )

    def test_linear_scaling_2d(self):
        """Test degree 1 (linear) scaling in 2D."""
        # ρ(x1, x2) = 1.0 + 0.5*x1 + 0.3*x2
        scaling = PolynomialScaling([1.0, 0.5, 0.3], self._bounds, self._bkd)

        # Test points: (0, 0), (1, 0), (0, 1), (1, 1)
        X = self._bkd.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        rho = scaling.eval_scaling(X)

        # Expected: [1.0, 1.5, 1.3, 1.8]
        expected = self._bkd.array([[1.0], [1.5], [1.3], [1.8]])
        self.assertTrue(
            self._bkd.allclose(rho, expected, rtol=1e-6, atol=1e-7)
        )

    def test_linear_scaling_jacobian_2d(self):
        """Test spatial Jacobian for linear scaling in 2D."""
        # ρ(x1, x2) = 1.0 + 0.5*x1 + 0.3*x2
        scaling = PolynomialScaling([1.0, 0.5, 0.3], self._bounds, self._bkd)

        X = self._bkd.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        jac_x = scaling.jacobian_scaling(X)

        # ∂ρ/∂x1 = 0.5, ∂ρ/∂x2 = 0.3 (constant gradients)
        expected = self._bkd.hstack([
            self._bkd.ones((4, 1)) * 0.5,
            self._bkd.ones((4, 1)) * 0.3
        ])
        self.assertTrue(
            self._bkd.allclose(jac_x, expected, rtol=1e-6, atol=1e-7)
        )

    def test_linear_scaling_param_jacobian_2d(self):
        """Test parameter Jacobian for linear scaling in 2D."""
        # ρ(x1, x2) = 1.0 + 0.5*x1 + 0.3*x2
        scaling = PolynomialScaling([1.0, 0.5, 0.3], self._bounds, self._bkd)

        X = self._bkd.array([[0.0, 1.0], [0.0, 0.0]])
        jac_params = scaling.jacobian_wrt_params(X)

        # ∂ρ/∂c0 = 1, ∂ρ/∂c1 = x1, ∂ρ/∂c2 = x2
        # For (0, 0): [1, 0, 0]
        # For (1, 0): [1, 1, 0]
        expected = self._bkd.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        self.assertTrue(
            self._bkd.allclose(jac_params, expected, rtol=1e-10)
        )

    def test_hyperparameter_bounds(self):
        """Test that hyperparameters respect bounds."""
        scaling = PolynomialScaling([0.8], self._bounds, self._bkd, nvars=self._nvars)

        hyp_list = scaling.hyp_list()
        self.assertEqual(hyp_list.nparams(), 1)

        # Check bounds
        bounds = hyp_list.get_bounds()
        self.assertEqual(len(bounds), 1)
        self.assertAlmostEqual(bounds[0][0], 0.1)
        self.assertAlmostEqual(bounds[0][1], 2.0)

    def test_hyperparameter_update(self):
        """Test updating hyperparameters."""
        scaling = PolynomialScaling([0.8, 0.2], self._bounds, self._bkd)

        # Update parameters
        hyp_list = scaling.hyp_list()
        new_values = self._bkd.array([0.9, 0.15])
        hyp_list.set_active_values(new_values)

        # Evaluate with new parameters
        X = self._bkd.array([[0.0, 1.0]])
        rho = scaling.eval_scaling(X)

        # Expected: [0.9, 1.05]
        expected = self._bkd.array([[0.9], [1.05]])
        self.assertTrue(
            self._bkd.allclose(rho, expected, rtol=1e-6, atol=1e-7)
        )

    def test_fixed_parameters(self):
        """Test fixed (non-trainable) parameters."""
        scaling = PolynomialScaling(
            [0.8], self._bounds, self._bkd, nvars=self._nvars, fixed=True
        )

        hyp_list = scaling.hyp_list()
        self.assertEqual(hyp_list.nactive_params(), 0)
        self.assertEqual(hyp_list.nparams(), 1)

    def test_invalid_degree_0_without_nvars(self):
        """Test that degree 0 requires nvars parameter."""
        with self.assertRaises(ValueError):
            PolynomialScaling([0.8], self._bounds, self._bkd)

    def test_degree_inferred_from_coefficients(self):
        """Test that nvars is correctly inferred for degree 1."""
        # 1D: 2 coefficients
        scaling_1d = PolynomialScaling([0.9, 0.1], self._bounds, self._bkd)
        self.assertEqual(scaling_1d.nvars(), 1)

        # 2D: 3 coefficients
        scaling_2d = PolynomialScaling([1.0, 0.5, 0.3], self._bounds, self._bkd)
        self.assertEqual(scaling_2d.nvars(), 2)

        # 3D: 4 coefficients
        scaling_3d = PolynomialScaling([1.0, 0.5, 0.3, 0.2], self._bounds, self._bkd)
        self.assertEqual(scaling_3d.nvars(), 3)

    def test_hvp_wrt_x1(self):
        """
        Test HVP returns zeros (scaling doesn't depend on x).

        Since PolynomialScaling of degree 0 (constant) doesn't depend on x,
        its Hessian is zero, and thus HVP should always return zeros.
        """
        # Test degree 0 (constant) scaling
        scaling = PolynomialScaling([0.8], self._bounds, self._bkd, nvars=2)

        # Test points
        X1 = self._bkd.array(np.random.randn(2, 3))
        X2 = self._bkd.array(np.random.randn(2, 4))
        direction = self._bkd.array(np.random.randn(2,))  # Shape (nvars,)

        # Compute HVP
        hvp = scaling.hvp_wrt_x1(X1, X2, direction)

        # Should be all zeros with shape (n1, n2, nvars)
        expected_shape = (X1.shape[1], X2.shape[1], 2)
        self.assertEqual(hvp.shape, expected_shape)

        # All values should be zero
        zeros = self._bkd.zeros(expected_shape)
        self.assertTrue(
            self._bkd.allclose(hvp, zeros, atol=1e-15)
        )


class TestPolynomialScalingNumpy(TestPolynomialScaling[NDArray[Any]]):
    """Test PolynomialScaling with NumPy backend."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPolynomialScalingTorch(TestPolynomialScaling[torch.Tensor]):
    """Test PolynomialScaling with PyTorch backend."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
