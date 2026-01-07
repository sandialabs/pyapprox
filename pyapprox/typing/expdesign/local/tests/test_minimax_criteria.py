"""
Tests for minimax OED criteria (G-optimal, R-optimal).

Tests cover:
- Value and Jacobian comparison with legacy code
- Multi-output (nqoi > 1) shape verification
- Numerical Jacobian verification
- Dual-backend support (NumPy and PyTorch)
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.test_utils import load_tests
from pyapprox.typing.expdesign.local.design_matrices import (
    LeastSquaresDesignMatrices,
)
from pyapprox.typing.expdesign.local.criteria import (
    GOptimalCriterion,
    ROptimalCriterion,
)


class TestGOptimalCriterion(Generic[Array], unittest.TestCase):
    """Base test class for G-optimal criterion."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

        # Create design factors
        self._ndesign_pts = 10
        self._ndesign_vars = 4
        self._npred_pts = 6
        self._design_factors = self._bkd.asarray(
            np.random.randn(self._ndesign_pts, self._ndesign_vars)
        )
        self._pred_factors = self._bkd.asarray(
            np.random.randn(self._npred_pts, self._ndesign_vars)
        )
        self._noise_mult = self._bkd.asarray(
            0.5 + np.random.rand(self._ndesign_pts)
        )

        # Random weights on simplex
        raw_weights = np.random.uniform(0, 1, (self._ndesign_pts, 1))
        self._weights = self._bkd.asarray(raw_weights / raw_weights.sum())

    def test_value_shape(self):
        """Test that G-optimal value has correct shape (npred_pts, 1)."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = GOptimalCriterion(dm, self._pred_factors, self._bkd)

        val = crit(self._weights)
        self.assertEqual(val.shape, (self._npred_pts, 1))
        self.assertTrue(self._bkd.all_bool(self._bkd.isfinite(val)))

    def test_jacobian_shape(self):
        """Test that G-optimal Jacobian has correct shape (npred_pts, nvars)."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = GOptimalCriterion(dm, self._pred_factors, self._bkd)

        jac = crit.jacobian(self._weights)
        self.assertEqual(jac.shape, (self._npred_pts, self._ndesign_pts))
        self.assertTrue(self._bkd.all_bool(self._bkd.isfinite(jac)))

    def test_nqoi(self):
        """Test that nqoi returns number of prediction points."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = GOptimalCriterion(dm, self._pred_factors, self._bkd)

        self.assertEqual(crit.nqoi(), self._npred_pts)

    def test_nvars(self):
        """Test that nvars returns number of design points."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = GOptimalCriterion(dm, self._pred_factors, self._bkd)

        self.assertEqual(crit.nvars(), self._ndesign_pts)

    def test_hvp_not_implemented(self):
        """Test that HVP is not implemented for G-optimal."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = GOptimalCriterion(dm, self._pred_factors, self._bkd)

        self.assertFalse(crit.hvp_implemented())

    def test_values_are_positive(self):
        """Test that prediction variances are positive."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = GOptimalCriterion(dm, self._pred_factors, self._bkd)

        val = crit(self._weights)
        self.assertTrue(self._bkd.all_bool(val > 0))


class TestGOptimalCriterionNumpy(TestGOptimalCriterion[NDArray[Any]]):
    """NumPy backend tests for G-optimal criterion."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()

    def test_legacy_comparison(self):
        """Compare with legacy implementation."""
        from pyapprox.expdesign.local import GOptimalLstSqCriterion
        from pyapprox.util.backends.numpy import NumpyMixin

        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        new_crit = GOptimalCriterion(dm, self._pred_factors, self._bkd)
        legacy_crit = GOptimalLstSqCriterion(
            self._design_factors,
            self._pred_factors,
            noise_mult=self._noise_mult,
            backend=NumpyMixin,
        )

        new_val = new_crit(self._weights)
        legacy_val = legacy_crit(self._weights)
        # Legacy returns (1, npred_pts), we return (npred_pts, 1)
        legacy_val_reshaped = legacy_val.T
        self._bkd.assert_allclose(new_val, legacy_val_reshaped, rtol=1e-10)

        new_jac = new_crit.jacobian(self._weights)
        legacy_jac = legacy_crit.jacobian(self._weights)
        self._bkd.assert_allclose(new_jac, legacy_jac, rtol=1e-10)

    def test_numerical_jacobian(self):
        """Verify Jacobian using finite differences."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = GOptimalCriterion(dm, self._pred_factors, self._bkd)

        analytical_jac = crit.jacobian(self._weights)

        eps = 1e-7
        numerical_jac = np.zeros((self._npred_pts, self._ndesign_pts))
        for k in range(self._ndesign_pts):
            weights_plus = self._weights.copy()
            weights_minus = self._weights.copy()
            weights_plus[k, 0] += eps
            weights_minus[k, 0] -= eps
            val_plus = crit(weights_plus)[:, 0]
            val_minus = crit(weights_minus)[:, 0]
            numerical_jac[:, k] = (val_plus - val_minus) / (2 * eps)

        numerical_jac = self._bkd.asarray(numerical_jac)
        # Use both rtol and atol since some values can be very small
        self._bkd.assert_allclose(
            analytical_jac, numerical_jac, rtol=1e-5, atol=1e-8
        )

    def test_homoscedastic_case(self):
        """Test G-optimal for homoscedastic noise."""
        dm = LeastSquaresDesignMatrices(self._design_factors, self._bkd)
        crit = GOptimalCriterion(dm, self._pred_factors, self._bkd)

        val = crit(self._weights)
        self.assertEqual(val.shape, (self._npred_pts, 1))
        self.assertTrue(self._bkd.all_bool(val > 0))


class TestGOptimalCriterionTorch(TestGOptimalCriterion[torch.Tensor]):
    """PyTorch backend tests for G-optimal criterion."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestROptimalCriterion(Generic[Array], unittest.TestCase):
    """Base test class for R-optimal criterion."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

        self._ndesign_pts = 10
        self._ndesign_vars = 4
        self._npred_pts = 6
        self._design_factors = self._bkd.asarray(
            np.random.randn(self._ndesign_pts, self._ndesign_vars)
        )
        self._pred_factors = self._bkd.asarray(
            np.random.randn(self._npred_pts, self._ndesign_vars)
        )
        self._noise_mult = self._bkd.asarray(
            0.5 + np.random.rand(self._ndesign_pts)
        )

        raw_weights = np.random.uniform(0, 1, (self._ndesign_pts, 1))
        self._weights = self._bkd.asarray(raw_weights / raw_weights.sum())

    def test_value_shape(self):
        """Test that R-optimal value has correct shape."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = ROptimalCriterion(dm, self._pred_factors, self._bkd)

        val = crit(self._weights)
        self.assertEqual(val.shape, (self._npred_pts, 1))

    def test_jacobian_shape(self):
        """Test that R-optimal Jacobian has correct shape."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = ROptimalCriterion(dm, self._pred_factors, self._bkd)

        jac = crit.jacobian(self._weights)
        self.assertEqual(jac.shape, (self._npred_pts, self._ndesign_pts))

    def test_identical_to_g_optimal(self):
        """Test that R-optimal produces same values as G-optimal."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        g_crit = GOptimalCriterion(dm, self._pred_factors, self._bkd)

        # Need separate design matrices instance for R-optimal
        dm2 = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        r_crit = ROptimalCriterion(dm2, self._pred_factors, self._bkd)

        g_val = g_crit(self._weights)
        r_val = r_crit(self._weights)
        self._bkd.assert_allclose(g_val, r_val, rtol=1e-12)

        g_jac = g_crit.jacobian(self._weights)
        r_jac = r_crit.jacobian(self._weights)
        self._bkd.assert_allclose(g_jac, r_jac, rtol=1e-12)


class TestROptimalCriterionNumpy(TestROptimalCriterion[NDArray[Any]]):
    """NumPy backend tests for R-optimal criterion."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()

    def test_legacy_comparison(self):
        """Compare with legacy implementation."""
        from pyapprox.expdesign.local import ROptimalLstSqCriterion
        from pyapprox.util.backends.numpy import NumpyMixin

        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        new_crit = ROptimalCriterion(dm, self._pred_factors, self._bkd)
        legacy_crit = ROptimalLstSqCriterion(
            self._design_factors,
            self._pred_factors,
            noise_mult=self._noise_mult,
            backend=NumpyMixin,
        )

        new_val = new_crit(self._weights)
        legacy_val = legacy_crit(self._weights)
        legacy_val_reshaped = legacy_val.T
        self._bkd.assert_allclose(new_val, legacy_val_reshaped, rtol=1e-10)

        new_jac = new_crit.jacobian(self._weights)
        legacy_jac = legacy_crit.jacobian(self._weights)
        self._bkd.assert_allclose(new_jac, legacy_jac, rtol=1e-10)


class TestROptimalCriterionTorch(TestROptimalCriterion[torch.Tensor]):
    """PyTorch backend tests for R-optimal criterion."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
