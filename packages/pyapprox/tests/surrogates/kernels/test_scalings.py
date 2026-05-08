"""
Tests for scaling functions.
"""

import numpy as np
import pytest

from pyapprox.surrogates.kernels.scalings import PolynomialScaling


class TestPolynomialScaling:
    """Base class for PolynomialScaling tests."""

    def test_constant_scaling(self, bkd):
        """Test degree 0 (constant) scaling."""
        _nvars = 1
        _bounds = (0.1, 2.0)
        # rho(x) = 0.8 (constant)
        scaling = PolynomialScaling([0.8], _bounds, bkd, nvars=_nvars)

        X = bkd.array([[-1.0, 0.0, 1.0]])
        rho = scaling.eval_scaling(X)

        # All values should be 0.8
        expected = bkd.ones((3, 1)) * 0.8
        assert bkd.allclose(rho, expected)

    def test_constant_scaling_jacobian(self, bkd):
        """Test that constant scaling has zero spatial Jacobian."""
        _nvars = 1
        _bounds = (0.1, 2.0)
        scaling = PolynomialScaling([0.8], _bounds, bkd, nvars=_nvars)

        X = bkd.array([[-1.0, 0.0, 1.0]])
        jac_x = scaling.jacobian_scaling(X)

        # Spatial derivative should be zero
        expected = bkd.zeros((3, 1))
        assert bkd.allclose(jac_x, expected)

    def test_constant_scaling_param_jacobian(self, bkd):
        """Test parameter Jacobian for constant scaling."""
        _nvars = 1
        _bounds = (0.1, 2.0)
        scaling = PolynomialScaling([0.8], _bounds, bkd, nvars=_nvars)

        X = bkd.array([[-1.0, 0.0, 1.0]])
        jac_params = scaling.jacobian_wrt_params(X)

        # d_rho/d_c0 = 1
        expected = bkd.ones((3, 1))
        assert bkd.allclose(jac_params, expected)

    def test_linear_scaling_1d(self, bkd):
        """Test degree 1 (linear) scaling in 1D."""
        _bounds = (0.1, 2.0)
        # rho(x) = 0.9 + 0.1*x
        scaling = PolynomialScaling([0.9, 0.1], _bounds, bkd)

        X = bkd.array([[-1.0, 0.0, 1.0]])
        rho = scaling.eval_scaling(X)

        # Expected: [0.8, 0.9, 1.0]
        expected = bkd.array([[0.8], [0.9], [1.0]])
        assert bkd.allclose(rho, expected, rtol=1e-6, atol=1e-7)

    def test_linear_scaling_jacobian_1d(self, bkd):
        """Test spatial Jacobian for linear scaling in 1D."""
        _bounds = (0.1, 2.0)
        # rho(x) = 0.9 + 0.1*x
        scaling = PolynomialScaling([0.9, 0.1], _bounds, bkd)

        X = bkd.array([[-1.0, 0.0, 1.0]])
        jac_x = scaling.jacobian_scaling(X)

        # d_rho/d_x = 0.1 (constant slope)
        expected = bkd.ones((3, 1)) * 0.1
        assert bkd.allclose(jac_x, expected, rtol=1e-6, atol=1e-7)

    def test_linear_scaling_param_jacobian_1d(self, bkd):
        """Test parameter Jacobian for linear scaling in 1D."""
        _bounds = (0.1, 2.0)
        # rho(x) = 0.9 + 0.1*x
        scaling = PolynomialScaling([0.9, 0.1], _bounds, bkd)

        X = bkd.array([[-1.0, 0.0, 1.0]])
        jac_params = scaling.jacobian_wrt_params(X)

        # d_rho/d_c0 = 1, d_rho/d_c1 = x
        expected = bkd.hstack(
            [bkd.ones((3, 1)), bkd.array([[-1.0], [0.0], [1.0]])]
        )
        assert bkd.allclose(jac_params, expected, rtol=1e-10)

    def test_linear_scaling_2d(self, bkd):
        """Test degree 1 (linear) scaling in 2D."""
        _bounds = (0.1, 2.0)
        # rho(x1, x2) = 1.0 + 0.5*x1 + 0.3*x2
        scaling = PolynomialScaling([1.0, 0.5, 0.3], _bounds, bkd)

        # Test points: (0, 0), (1, 0), (0, 1), (1, 1)
        X = bkd.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        rho = scaling.eval_scaling(X)

        # Expected: [1.0, 1.5, 1.3, 1.8]
        expected = bkd.array([[1.0], [1.5], [1.3], [1.8]])
        assert bkd.allclose(rho, expected, rtol=1e-6, atol=1e-7)

    def test_linear_scaling_jacobian_2d(self, bkd):
        """Test spatial Jacobian for linear scaling in 2D."""
        _bounds = (0.1, 2.0)
        # rho(x1, x2) = 1.0 + 0.5*x1 + 0.3*x2
        scaling = PolynomialScaling([1.0, 0.5, 0.3], _bounds, bkd)

        X = bkd.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        jac_x = scaling.jacobian_scaling(X)

        # d_rho/d_x1 = 0.5, d_rho/d_x2 = 0.3 (constant gradients)
        expected = bkd.hstack(
            [bkd.ones((4, 1)) * 0.5, bkd.ones((4, 1)) * 0.3]
        )
        assert bkd.allclose(jac_x, expected, rtol=1e-6, atol=1e-7)

    def test_linear_scaling_param_jacobian_2d(self, bkd):
        """Test parameter Jacobian for linear scaling in 2D."""
        _bounds = (0.1, 2.0)
        # rho(x1, x2) = 1.0 + 0.5*x1 + 0.3*x2
        scaling = PolynomialScaling([1.0, 0.5, 0.3], _bounds, bkd)

        X = bkd.array([[0.0, 1.0], [0.0, 0.0]])
        jac_params = scaling.jacobian_wrt_params(X)

        # d_rho/d_c0 = 1, d_rho/d_c1 = x1, d_rho/d_c2 = x2
        # For (0, 0): [1, 0, 0]
        # For (1, 0): [1, 1, 0]
        expected = bkd.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        assert bkd.allclose(jac_params, expected, rtol=1e-10)

    def test_hyperparameter_bounds(self, bkd):
        """Test that hyperparameters respect bounds."""
        _nvars = 1
        _bounds = (0.1, 2.0)
        scaling = PolynomialScaling([0.8], _bounds, bkd, nvars=_nvars)

        hyp_list = scaling.hyp_list()
        assert hyp_list.nparams() == 1

        # Check bounds
        bounds = hyp_list.get_bounds()
        assert len(bounds) == 1
        bkd.assert_allclose(bkd.asarray([bounds[0][0]]), bkd.asarray([0.1]), rtol=1e-10)
        bkd.assert_allclose(bkd.asarray([bounds[0][1]]), bkd.asarray([2.0]), rtol=1e-10)

    def test_hyperparameter_update(self, bkd):
        """Test updating hyperparameters."""
        _bounds = (0.1, 2.0)
        scaling = PolynomialScaling([0.8, 0.2], _bounds, bkd)

        # Update parameters
        hyp_list = scaling.hyp_list()
        new_values = bkd.array([0.9, 0.15])
        hyp_list.set_active_values(new_values)

        # Evaluate with new parameters
        X = bkd.array([[0.0, 1.0]])
        rho = scaling.eval_scaling(X)

        # Expected: [0.9, 1.05]
        expected = bkd.array([[0.9], [1.05]])
        assert bkd.allclose(rho, expected, rtol=1e-6, atol=1e-7)

    def test_fixed_parameters(self, bkd):
        """Test fixed (non-trainable) parameters."""
        _nvars = 1
        _bounds = (0.1, 2.0)
        scaling = PolynomialScaling(
            [0.8], _bounds, bkd, nvars=_nvars, fixed=True
        )

        hyp_list = scaling.hyp_list()
        assert hyp_list.nactive_params() == 0
        assert hyp_list.nparams() == 1

    def test_invalid_degree_0_without_nvars(self, bkd):
        """Test that degree 0 requires nvars parameter."""
        _bounds = (0.1, 2.0)
        with pytest.raises(ValueError):
            PolynomialScaling([0.8], _bounds, bkd)

    def test_degree_inferred_from_coefficients(self, bkd):
        """Test that nvars is correctly inferred for degree 1."""
        _bounds = (0.1, 2.0)
        # 1D: 2 coefficients
        scaling_1d = PolynomialScaling([0.9, 0.1], _bounds, bkd)
        assert scaling_1d.nvars() == 1

        # 2D: 3 coefficients
        scaling_2d = PolynomialScaling([1.0, 0.5, 0.3], _bounds, bkd)
        assert scaling_2d.nvars() == 2

        # 3D: 4 coefficients
        scaling_3d = PolynomialScaling([1.0, 0.5, 0.3, 0.2], _bounds, bkd)
        assert scaling_3d.nvars() == 3

    def test_hvp_wrt_x1(self, bkd):
        """
        Test HVP returns zeros (scaling doesn't depend on x).

        Since PolynomialScaling of degree 0 (constant) doesn't depend on x,
        its Hessian is zero, and thus HVP should always return zeros.
        """
        # Test degree 0 (constant) scaling
        scaling = PolynomialScaling([0.8], (0.1, 2.0), bkd, nvars=2)

        # Test points
        X1 = bkd.array(np.random.randn(2, 3))
        X2 = bkd.array(np.random.randn(2, 4))
        direction = bkd.array(
            np.random.randn(
                2,
            )
        )  # Shape (nvars,)

        # Compute HVP
        hvp = scaling.hvp_wrt_x1(X1, X2, direction)

        # Should be all zeros with shape (n1, n2, nvars)
        expected_shape = (X1.shape[1], X2.shape[1], 2)
        assert hvp.shape == expected_shape

        # All values should be zero
        zeros = bkd.zeros(expected_shape)
        assert bkd.allclose(hvp, zeros, atol=1e-15)
