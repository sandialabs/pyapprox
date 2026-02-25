"""
Tests for adjoint-based OED criteria (C, A, I-optimal).

Tests cover:
- Value and Jacobian comparison with legacy code
- HVP verification via numerical finite differences
- Homoscedastic and heteroscedastic noise
- Least squares regression
- Dual-backend support (NumPy and PyTorch)
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.expdesign.local.criteria import (
    AOptimalCriterion,
    COptimalCriterion,
    IOptimalCriterion,
)
from pyapprox.expdesign.local.design_matrices import (
    LeastSquaresDesignMatrices,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array
from pyapprox.util.backends.torch import TorchBkd


class TestCOptimalCriterion(Generic[Array], unittest.TestCase):
    """Base test class for C-optimal criterion."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

        # Create design factors
        self._ndesign_pts = 10
        self._ndesign_vars = 4
        self._design_factors = self._bkd.asarray(
            np.random.randn(self._ndesign_pts, self._ndesign_vars)
        )
        self._noise_mult = self._bkd.asarray(0.5 + np.random.rand(self._ndesign_pts))

        # C-optimal vector
        self._vec = self._bkd.asarray(np.ones(self._ndesign_vars))

        # Random weights on simplex
        raw_weights = np.random.uniform(0, 1, (self._ndesign_pts, 1))
        self._weights = self._bkd.asarray(raw_weights / raw_weights.sum())

        # Direction for HVP tests
        self._direction = self._bkd.asarray(np.random.randn(self._ndesign_pts, 1))

    def test_value_shape(self):
        """Test that C-optimal value has correct shape."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = COptimalCriterion(dm, self._vec, self._bkd)

        val = crit(self._weights)
        self.assertEqual(val.shape, (1, 1))
        self.assertTrue(self._bkd.all_bool(self._bkd.isfinite(val)))

    def test_jacobian_shape(self):
        """Test that C-optimal Jacobian has correct shape."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = COptimalCriterion(dm, self._vec, self._bkd)

        jac = crit.jacobian(self._weights)
        self.assertEqual(jac.shape, (1, self._ndesign_pts))
        self.assertTrue(self._bkd.all_bool(self._bkd.isfinite(jac)))

    def test_hvp_shape(self):
        """Test that C-optimal HVP has correct shape."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = COptimalCriterion(dm, self._vec, self._bkd)

        hvp = crit.hvp(self._weights, self._direction)
        self.assertEqual(hvp.shape, (self._ndesign_pts, 1))
        self.assertTrue(self._bkd.all_bool(self._bkd.isfinite(hvp)))

    def test_hvp_available(self):
        """Test that HVP is available (following optional methods convention)."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = COptimalCriterion(dm, self._vec, self._bkd)
        self.assertTrue(hasattr(crit, "hvp"))

    def test_nvars_nqoi(self):
        """Test nvars and nqoi properties."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = COptimalCriterion(dm, self._vec, self._bkd)

        self.assertEqual(crit.nvars(), self._ndesign_pts)
        self.assertEqual(crit.nqoi(), 1)
        self.assertEqual(crit.ndesign_vars(), self._ndesign_vars)


class TestCOptimalCriterionNumpy(TestCOptimalCriterion[NDArray[Any]]):
    """NumPy backend tests for C-optimal criterion."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()

    def test_numerical_jacobian(self):
        """Verify Jacobian using finite differences."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = COptimalCriterion(dm, self._vec, self._bkd)

        analytical_jac = crit.jacobian(self._weights)

        eps = 1e-7
        numerical_jac = np.zeros((1, self._ndesign_pts))
        for k in range(self._ndesign_pts):
            weights_plus = self._weights.copy()
            weights_minus = self._weights.copy()
            weights_plus[k, 0] += eps
            weights_minus[k, 0] -= eps
            val_plus = crit(weights_plus)[0, 0]
            val_minus = crit(weights_minus)[0, 0]
            numerical_jac[0, k] = (val_plus - val_minus) / (2 * eps)

        numerical_jac = self._bkd.asarray(numerical_jac)
        self._bkd.assert_allclose(analytical_jac, numerical_jac, rtol=1e-5)

    def test_numerical_hvp(self):
        """Verify HVP using finite differences."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = COptimalCriterion(dm, self._vec, self._bkd)

        analytical_hvp = crit.hvp(self._weights, self._direction)

        eps = 1e-7
        jac_plus = crit.jacobian(self._weights + eps * self._direction)
        jac_minus = crit.jacobian(self._weights - eps * self._direction)
        numerical_hvp = (jac_plus - jac_minus).T / (2 * eps)

        self._bkd.assert_allclose(analytical_hvp, numerical_hvp, rtol=1e-5)


class TestCOptimalCriterionTorch(TestCOptimalCriterion[torch.Tensor]):
    """PyTorch backend tests for C-optimal criterion."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestAOptimalCriterion(Generic[Array], unittest.TestCase):
    """Base test class for A-optimal criterion."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

        self._ndesign_pts = 10
        self._ndesign_vars = 4
        self._design_factors = self._bkd.asarray(
            np.random.randn(self._ndesign_pts, self._ndesign_vars)
        )
        self._noise_mult = self._bkd.asarray(0.5 + np.random.rand(self._ndesign_pts))

        raw_weights = np.random.uniform(0, 1, (self._ndesign_pts, 1))
        self._weights = self._bkd.asarray(raw_weights / raw_weights.sum())

        self._direction = self._bkd.asarray(np.random.randn(self._ndesign_pts, 1))

    def test_value_shape(self):
        """Test that A-optimal value has correct shape."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = AOptimalCriterion(dm, self._bkd)

        val = crit(self._weights)
        self.assertEqual(val.shape, (1, 1))
        self.assertTrue(self._bkd.all_bool(self._bkd.isfinite(val)))

    def test_jacobian_shape(self):
        """Test that A-optimal Jacobian has correct shape."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = AOptimalCriterion(dm, self._bkd)

        jac = crit.jacobian(self._weights)
        self.assertEqual(jac.shape, (1, self._ndesign_pts))
        self.assertTrue(self._bkd.all_bool(self._bkd.isfinite(jac)))

    def test_hvp_shape(self):
        """Test that A-optimal HVP has correct shape."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = AOptimalCriterion(dm, self._bkd)

        hvp = crit.hvp(self._weights, self._direction)
        self.assertEqual(hvp.shape, (self._ndesign_pts, 1))
        self.assertTrue(self._bkd.all_bool(self._bkd.isfinite(hvp)))

    def test_nvars_nqoi(self):
        """Test nvars and nqoi properties."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = AOptimalCriterion(dm, self._bkd)

        self.assertEqual(crit.nvars(), self._ndesign_pts)
        self.assertEqual(crit.nqoi(), 1)


class TestAOptimalCriterionNumpy(TestAOptimalCriterion[NDArray[Any]]):
    """NumPy backend tests for A-optimal criterion."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()

    def test_numerical_jacobian(self):
        """Verify Jacobian using finite differences."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = AOptimalCriterion(dm, self._bkd)

        analytical_jac = crit.jacobian(self._weights)

        eps = 1e-7
        numerical_jac = np.zeros((1, self._ndesign_pts))
        for k in range(self._ndesign_pts):
            weights_plus = self._weights.copy()
            weights_minus = self._weights.copy()
            weights_plus[k, 0] += eps
            weights_minus[k, 0] -= eps
            val_plus = crit(weights_plus)[0, 0]
            val_minus = crit(weights_minus)[0, 0]
            numerical_jac[0, k] = (val_plus - val_minus) / (2 * eps)

        numerical_jac = self._bkd.asarray(numerical_jac)
        self._bkd.assert_allclose(analytical_jac, numerical_jac, rtol=1e-5)

    def test_numerical_hvp(self):
        """Verify HVP using finite differences."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = AOptimalCriterion(dm, self._bkd)

        analytical_hvp = crit.hvp(self._weights, self._direction)

        eps = 1e-7
        jac_plus = crit.jacobian(self._weights + eps * self._direction)
        jac_minus = crit.jacobian(self._weights - eps * self._direction)
        numerical_hvp = (jac_plus - jac_minus).T / (2 * eps)

        self._bkd.assert_allclose(analytical_hvp, numerical_hvp, rtol=1e-5)

    def test_trace_formula(self):
        """Verify A-optimal equals trace(Cov) for homoscedastic case."""
        dm = LeastSquaresDesignMatrices(self._design_factors, self._bkd)
        crit = AOptimalCriterion(dm, self._bkd)

        val = crit(self._weights)

        # Compute trace(M1^{-1}) directly
        M1 = dm.M1(self._weights)
        M1_inv = self._bkd.inv(M1)
        expected = self._bkd.trace(M1_inv)

        self._bkd.assert_allclose(val, self._bkd.reshape(expected, (1, 1)), rtol=1e-12)


class TestAOptimalCriterionTorch(TestAOptimalCriterion[torch.Tensor]):
    """PyTorch backend tests for A-optimal criterion."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestIOptimalCriterion(Generic[Array], unittest.TestCase):
    """Base test class for I-optimal criterion."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

        self._ndesign_pts = 10
        self._ndesign_vars = 4
        self._npred_pts = 8
        self._design_factors = self._bkd.asarray(
            np.random.randn(self._ndesign_pts, self._ndesign_vars)
        )
        self._pred_factors = self._bkd.asarray(
            np.random.randn(self._npred_pts, self._ndesign_vars)
        )
        self._noise_mult = self._bkd.asarray(0.5 + np.random.rand(self._ndesign_pts))

        raw_weights = np.random.uniform(0, 1, (self._ndesign_pts, 1))
        self._weights = self._bkd.asarray(raw_weights / raw_weights.sum())

        self._direction = self._bkd.asarray(np.random.randn(self._ndesign_pts, 1))

    def test_value_shape(self):
        """Test that I-optimal value has correct shape."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = IOptimalCriterion(dm, self._pred_factors, self._bkd)

        val = crit(self._weights)
        self.assertEqual(val.shape, (1, 1))
        self.assertTrue(self._bkd.all_bool(self._bkd.isfinite(val)))

    def test_jacobian_shape(self):
        """Test that I-optimal Jacobian has correct shape."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = IOptimalCriterion(dm, self._pred_factors, self._bkd)

        jac = crit.jacobian(self._weights)
        self.assertEqual(jac.shape, (1, self._ndesign_pts))
        self.assertTrue(self._bkd.all_bool(self._bkd.isfinite(jac)))

    def test_hvp_shape(self):
        """Test that I-optimal HVP has correct shape."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = IOptimalCriterion(dm, self._pred_factors, self._bkd)

        hvp = crit.hvp(self._weights, self._direction)
        self.assertEqual(hvp.shape, (self._ndesign_pts, 1))
        self.assertTrue(self._bkd.all_bool(self._bkd.isfinite(hvp)))

    def test_nvars_nqoi(self):
        """Test nvars and nqoi properties."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = IOptimalCriterion(dm, self._pred_factors, self._bkd)

        self.assertEqual(crit.nvars(), self._ndesign_pts)
        self.assertEqual(crit.nqoi(), 1)


class TestIOptimalCriterionNumpy(TestIOptimalCriterion[NDArray[Any]]):
    """NumPy backend tests for I-optimal criterion."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()

    def test_numerical_jacobian(self):
        """Verify Jacobian using finite differences."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = IOptimalCriterion(dm, self._pred_factors, self._bkd)

        analytical_jac = crit.jacobian(self._weights)

        eps = 1e-7
        numerical_jac = np.zeros((1, self._ndesign_pts))
        for k in range(self._ndesign_pts):
            weights_plus = self._weights.copy()
            weights_minus = self._weights.copy()
            weights_plus[k, 0] += eps
            weights_minus[k, 0] -= eps
            val_plus = crit(weights_plus)[0, 0]
            val_minus = crit(weights_minus)[0, 0]
            numerical_jac[0, k] = (val_plus - val_minus) / (2 * eps)

        numerical_jac = self._bkd.asarray(numerical_jac)
        self._bkd.assert_allclose(analytical_jac, numerical_jac, rtol=1e-5)

    def test_numerical_hvp(self):
        """Verify HVP using finite differences."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = IOptimalCriterion(dm, self._pred_factors, self._bkd)

        analytical_hvp = crit.hvp(self._weights, self._direction)

        eps = 1e-7
        jac_plus = crit.jacobian(self._weights + eps * self._direction)
        jac_minus = crit.jacobian(self._weights - eps * self._direction)
        numerical_hvp = (jac_plus - jac_minus).T / (2 * eps)

        self._bkd.assert_allclose(analytical_hvp, numerical_hvp, rtol=1e-5)

    def test_with_pred_weights(self):
        """Test I-optimal with custom prediction weights."""
        pred_weights = self._bkd.asarray(np.random.rand(self._npred_pts))
        pred_weights = pred_weights / self._bkd.sum(pred_weights)

        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = IOptimalCriterion(
            dm, self._pred_factors, self._bkd, pred_weights=pred_weights
        )

        val = crit(self._weights)
        self.assertEqual(val.shape, (1, 1))
        self.assertTrue(self._bkd.all_bool(self._bkd.isfinite(val)))


class TestIOptimalCriterionTorch(TestIOptimalCriterion[torch.Tensor]):
    """PyTorch backend tests for I-optimal criterion."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
