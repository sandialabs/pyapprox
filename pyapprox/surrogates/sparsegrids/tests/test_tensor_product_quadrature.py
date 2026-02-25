"""Tests for tensor product interpolation and quadrature.

This module tests pure tensor product operations WITHOUT Smolyak combination,
to isolate and verify tensor product assembly works correctly for:
- Gauss-Lagrange interpolation/quadrature
- Leja-Lagrange interpolation/quadrature
- Piecewise polynomial interpolation
- Mixed basis types across dimensions

These tests verify:
1. Exact interpolation of polynomials within the degree range
2. Exact integration (mean computation) for polynomials
3. Quadrature weights sum to 1 (for probability measure)
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.surrogates.sparsegrids.tests.test_helpers import (
    create_tensor_product_pce,
    create_test_joint,
    create_test_tensor_product_subspace,
    create_test_tensor_product_subspace_mixed,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests, slow_test, slower_test  # noqa: F401

# =============================================================================
# Test Configurations
# =============================================================================

# Tensor product Gauss configs: (name, joint_config, npts_1d)
TENSOR_PRODUCT_GAUSS_CONFIGS = [
    ("2d_uniform_3x3", "2d_uniform", [3, 3]),
    ("2d_uniform_3x4", "2d_uniform", [3, 4]),
    ("2d_uniform_5x5", "2d_uniform", [5, 5]),
    ("2d_gaussian_3x3", "2d_gaussian", [3, 3]),
    ("2d_beta_3x3", "2d_beta", [3, 3]),
    ("2d_beta_4x4", "2d_beta", [4, 4]),
    ("2d_gamma_3x3", "2d_gamma", [3, 3]),
    ("2d_gamma_4x4", "2d_gamma", [4, 4]),
    ("2d_mixed_ug_3x3", "2d_mixed_ug", [3, 3]),
    ("3d_uniform_3x3x3", "3d_uniform", [3, 3, 3]),
]

TENSOR_PRODUCT_LEJA_CONFIGS = [
    ("2d_uniform_3x3", "2d_uniform", [3, 3]),
    ("2d_uniform_5x5", "2d_uniform", [5, 5]),
    ("2d_beta_3x3", "2d_beta", [3, 3]),
    ("3d_uniform_3x3x3", "3d_uniform", [3, 3, 3]),
]

# Piecewise configs - use bounded distributions only
TENSOR_PRODUCT_PIECEWISE_LINEAR_CONFIGS = [
    ("2d_uniform_5x5", "2d_uniform", [5, 5]),
    ("2d_uniform_7x7", "2d_uniform", [7, 7]),
    ("2d_beta_5x5", "2d_beta", [5, 5]),
    ("3d_uniform_5x5x5", "3d_uniform", [5, 5, 5]),
]

# Mixed basis configs: (name, joint_config, basis_types, npts_1d)
MIXED_TENSOR_PRODUCT_CONFIGS = [
    ("gauss_leja_uniform", "2d_uniform", ["gauss", "leja"], [3, 3]),
    ("gauss_gauss_beta", "2d_beta", ["gauss", "gauss"], [4, 4]),
]


# =============================================================================
# Gauss-Lagrange Tests
# =============================================================================


class TestTensorProductGauss(Generic[Array], ParametrizedTestCase, unittest.TestCase):
    """Test pure tensor product Gauss-Lagrange interpolation and quadrature."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    @parametrize(
        "name,joint_config,npts_1d",
        TENSOR_PRODUCT_GAUSS_CONFIGS,
    )
    @slower_test
    def test_interpolation_exact(self, name, joint_config, npts_1d):
        """Tensor product exactly interpolates tensor product PCE."""
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_tensor_product_pce(joint, npts_1d, nqoi=1, bkd=self._bkd)
        subspace = create_test_tensor_product_subspace(
            joint, npts_1d, self._bkd, "gauss"
        )

        # Interpolate PCE values
        samples = subspace.get_samples()
        values = pce(samples)
        subspace.set_values(values)

        # Test at random points
        np.random.seed(123)
        test_pts = joint.rvs(20)
        self._bkd.assert_allclose(subspace(test_pts), pce(test_pts), rtol=1e-10)

    @parametrize(
        "name,joint_config,npts_1d",
        TENSOR_PRODUCT_GAUSS_CONFIGS,
    )
    def test_integration_exact(self, name, joint_config, npts_1d):
        """Tensor product quadrature computes exact mean."""
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_tensor_product_pce(joint, npts_1d, nqoi=1, bkd=self._bkd)
        subspace = create_test_tensor_product_subspace(
            joint, npts_1d, self._bkd, "gauss"
        )

        # Interpolate PCE values
        samples = subspace.get_samples()
        values = pce(samples)
        subspace.set_values(values)

        # Expected mean is the constant term (index 0) of orthonormal PCE
        expected_mean = pce.get_coefficients()[0, :]
        computed_mean = subspace.integrate()
        self._bkd.assert_allclose(computed_mean, expected_mean, rtol=1e-10)

    @parametrize(
        "name,joint_config,npts_1d",
        TENSOR_PRODUCT_GAUSS_CONFIGS,
    )
    def test_weights_sum_to_one(self, name, joint_config, npts_1d):
        """Tensor product quadrature weights sum to 1 for probability measure."""
        joint = create_test_joint(joint_config, self._bkd)
        subspace = create_test_tensor_product_subspace(
            joint, npts_1d, self._bkd, "gauss"
        )

        # Integrate constant function f(x) = 1
        samples = subspace.get_samples()
        values = self._bkd.ones((1, samples.shape[1]))
        subspace.set_values(values)

        # Mean of constant 1 should be 1
        self._bkd.assert_allclose(
            subspace.integrate(),
            self._bkd.asarray([1.0]),
            rtol=1e-12,
        )


class TestTensorProductGaussNumpy(TestTensorProductGauss[NDArray[Any]]):
    """NumPy backend tests for tensor product Gauss quadrature."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTensorProductGaussTorch(TestTensorProductGauss[torch.Tensor]):
    """PyTorch backend tests for tensor product Gauss quadrature."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# =============================================================================
# Leja-Lagrange Tests
# =============================================================================


class TestTensorProductLeja(Generic[Array], ParametrizedTestCase, unittest.TestCase):
    """Test pure tensor product Leja-Lagrange interpolation and quadrature."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    @parametrize(
        "name,joint_config,npts_1d",
        TENSOR_PRODUCT_LEJA_CONFIGS,
    )
    def test_interpolation_exact(self, name, joint_config, npts_1d):
        """Tensor product exactly interpolates tensor product PCE."""
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_tensor_product_pce(joint, npts_1d, nqoi=1, bkd=self._bkd)
        subspace = create_test_tensor_product_subspace(
            joint, npts_1d, self._bkd, "leja"
        )

        # Interpolate PCE values
        samples = subspace.get_samples()
        values = pce(samples)
        subspace.set_values(values)

        # Test at random points
        np.random.seed(123)
        test_pts = joint.rvs(20)
        self._bkd.assert_allclose(subspace(test_pts), pce(test_pts), rtol=1e-10)

    @parametrize(
        "name,joint_config,npts_1d",
        TENSOR_PRODUCT_LEJA_CONFIGS,
    )
    @slow_test
    def test_integration_exact(self, name, joint_config, npts_1d):
        """Tensor product Leja quadrature computes exact mean."""
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_tensor_product_pce(joint, npts_1d, nqoi=1, bkd=self._bkd)
        subspace = create_test_tensor_product_subspace(
            joint, npts_1d, self._bkd, "leja"
        )

        # Interpolate PCE values
        samples = subspace.get_samples()
        values = pce(samples)
        subspace.set_values(values)

        # Expected mean is the constant term of orthonormal PCE
        expected_mean = pce.get_coefficients()[0, :]
        computed_mean = subspace.integrate()
        self._bkd.assert_allclose(computed_mean, expected_mean, rtol=1e-10)

    @parametrize(
        "name,joint_config,npts_1d",
        TENSOR_PRODUCT_LEJA_CONFIGS,
    )
    def test_weights_sum_to_one(self, name, joint_config, npts_1d):
        """Tensor product Leja weights sum to 1 for probability measure."""
        joint = create_test_joint(joint_config, self._bkd)
        subspace = create_test_tensor_product_subspace(
            joint, npts_1d, self._bkd, "leja"
        )

        # Integrate constant function f(x) = 1
        samples = subspace.get_samples()
        values = self._bkd.ones((1, samples.shape[1]))
        subspace.set_values(values)

        self._bkd.assert_allclose(
            subspace.integrate(),
            self._bkd.asarray([1.0]),
            rtol=1e-12,
        )


class TestTensorProductLejaNumpy(TestTensorProductLeja[NDArray[Any]]):
    """NumPy backend tests for tensor product Leja quadrature."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTensorProductLejaTorch(TestTensorProductLeja[torch.Tensor]):
    """PyTorch backend tests for tensor product Leja quadrature."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# =============================================================================
# Piecewise Polynomial Tests
# =============================================================================


class TestTensorProductPiecewise(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Test tensor product piecewise linear interpolation."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    @parametrize(
        "name,joint_config,npts_1d",
        TENSOR_PRODUCT_PIECEWISE_LINEAR_CONFIGS,
    )
    def test_interpolation_linear_exact(self, name, joint_config, npts_1d):
        """Tensor product piecewise linear exactly interpolates linear functions."""
        joint = create_test_joint(joint_config, self._bkd)
        subspace = create_test_tensor_product_subspace(
            joint, npts_1d, self._bkd, "piecewise_linear"
        )

        # Create linear function: f(x) = 1 + sum_i (i+1)*x_i
        nvars = joint.nvars()

        def linear_func(samples):
            result = self._bkd.ones((1, samples.shape[1]))
            for i in range(nvars):
                result = result + (i + 1) * samples[i : i + 1, :]
            return result

        # Interpolate
        samples = subspace.get_samples()
        values = linear_func(samples)
        subspace.set_values(values)

        # Test at random points
        np.random.seed(123)
        test_pts = joint.rvs(20)
        expected = linear_func(test_pts)
        computed = subspace(test_pts)
        self._bkd.assert_allclose(computed, expected, rtol=1e-10)

    @parametrize(
        "name,joint_config,npts_1d",
        TENSOR_PRODUCT_PIECEWISE_LINEAR_CONFIGS,
    )
    def test_weights_positive(self, name, joint_config, npts_1d):
        """Tensor product piecewise linear weights are non-negative."""
        joint = create_test_joint(joint_config, self._bkd)
        subspace = create_test_tensor_product_subspace(
            joint, npts_1d, self._bkd, "piecewise_linear"
        )

        weights = subspace.get_quadrature_weights()
        # All weights should be non-negative
        self.assertTrue(
            self._bkd.all_bool(weights >= 0),
            f"Found negative weights: {weights}",
        )


class TestTensorProductPiecewiseNumpy(TestTensorProductPiecewise[NDArray[Any]]):
    """NumPy backend tests for tensor product piecewise interpolation."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTensorProductPiecewiseTorch(TestTensorProductPiecewise[torch.Tensor]):
    """PyTorch backend tests for tensor product piecewise interpolation."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# =============================================================================
# Mixed Basis Tests
# =============================================================================


class TestMixedTensorProduct(Generic[Array], ParametrizedTestCase, unittest.TestCase):
    """Test tensor products with mixed basis types per dimension."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    @parametrize(
        "name,joint_config,basis_types,npts_1d",
        MIXED_TENSOR_PRODUCT_CONFIGS,
    )
    def test_interpolation_exact(self, name, joint_config, basis_types, npts_1d):
        """Mixed basis tensor product exactly interpolates tensor product PCE."""
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_tensor_product_pce(joint, npts_1d, nqoi=1, bkd=self._bkd)
        subspace = create_test_tensor_product_subspace_mixed(
            joint, basis_types, npts_1d, self._bkd
        )

        # Interpolate PCE values
        samples = subspace.get_samples()
        values = pce(samples)
        subspace.set_values(values)

        # Test at random points
        np.random.seed(123)
        test_pts = joint.rvs(20)
        self._bkd.assert_allclose(subspace(test_pts), pce(test_pts), rtol=1e-10)

    @parametrize(
        "name,joint_config,basis_types,npts_1d",
        MIXED_TENSOR_PRODUCT_CONFIGS,
    )
    def test_integration_exact(self, name, joint_config, basis_types, npts_1d):
        """Mixed basis tensor product computes exact mean."""
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_tensor_product_pce(joint, npts_1d, nqoi=1, bkd=self._bkd)
        subspace = create_test_tensor_product_subspace_mixed(
            joint, basis_types, npts_1d, self._bkd
        )

        # Interpolate PCE values
        samples = subspace.get_samples()
        values = pce(samples)
        subspace.set_values(values)

        # Expected mean is the constant term of orthonormal PCE
        expected_mean = pce.get_coefficients()[0, :]
        computed_mean = subspace.integrate()
        self._bkd.assert_allclose(computed_mean, expected_mean, rtol=1e-10)

    @parametrize(
        "name,joint_config,basis_types,npts_1d",
        MIXED_TENSOR_PRODUCT_CONFIGS,
    )
    def test_weights_sum_to_one(self, name, joint_config, basis_types, npts_1d):
        """Mixed basis tensor product weights sum to 1."""
        joint = create_test_joint(joint_config, self._bkd)
        subspace = create_test_tensor_product_subspace_mixed(
            joint, basis_types, npts_1d, self._bkd
        )

        # Integrate constant function f(x) = 1
        samples = subspace.get_samples()
        values = self._bkd.ones((1, samples.shape[1]))
        subspace.set_values(values)

        self._bkd.assert_allclose(
            subspace.integrate(),
            self._bkd.asarray([1.0]),
            rtol=1e-12,
        )


class TestMixedTensorProductNumpy(TestMixedTensorProduct[NDArray[Any]]):
    """NumPy backend tests for mixed tensor product."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMixedTensorProductTorch(TestMixedTensorProduct[torch.Tensor]):
    """PyTorch backend tests for mixed tensor product."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
