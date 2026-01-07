"""
Tests for D-optimal criterion.

Tests cover:
- Homoscedastic and heteroscedastic noise
- Least squares and quantile regression
- Value and Jacobian comparison with legacy code
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
    QuantileDesignMatrices,
)
from pyapprox.typing.expdesign.local.criteria import (
    DOptimalCriterion,
    DOptimalQuantileCriterion,
)


class TestDOptimalCriterion(Generic[Array], unittest.TestCase):
    """Base test class for D-optimal criterion."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

        # Create design factors
        self._ndesign_pts = 7
        self._ndesign_vars = 4
        self._design_factors = self._bkd.asarray(
            np.random.randn(self._ndesign_pts, self._ndesign_vars)
        )
        self._noise_mult = self._bkd.asarray(
            1 + np.abs(np.random.randn(self._ndesign_pts))
        )

        # Random weights on simplex
        raw_weights = np.random.uniform(0, 1, (self._ndesign_pts, 1))
        self._weights = self._bkd.asarray(raw_weights / raw_weights.sum())

    def test_homoscedastic_least_squares_value(self):
        """Test D-optimal value for homoscedastic least squares."""
        dm = LeastSquaresDesignMatrices(self._design_factors, self._bkd)
        crit = DOptimalCriterion(dm, self._bkd)

        val = crit(self._weights)

        # Check shape
        self.assertEqual(val.shape, (1, 1))

        # Check that value is finite
        self.assertTrue(self._bkd.all_bool(self._bkd.isfinite(val)))

    def test_homoscedastic_least_squares_jacobian(self):
        """Test D-optimal Jacobian for homoscedastic least squares."""
        dm = LeastSquaresDesignMatrices(self._design_factors, self._bkd)
        crit = DOptimalCriterion(dm, self._bkd)

        jac = crit.jacobian(self._weights)

        # Check shape
        self.assertEqual(jac.shape, (1, self._ndesign_pts))

        # Check that Jacobian is finite
        self.assertTrue(self._bkd.all_bool(self._bkd.isfinite(jac)))

    def test_heteroscedastic_least_squares_value(self):
        """Test D-optimal value for heteroscedastic least squares."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = DOptimalCriterion(dm, self._bkd)

        val = crit(self._weights)

        # Check shape
        self.assertEqual(val.shape, (1, 1))
        self.assertTrue(self._bkd.all_bool(self._bkd.isfinite(val)))

    def test_heteroscedastic_least_squares_jacobian(self):
        """Test D-optimal Jacobian for heteroscedastic least squares."""
        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = DOptimalCriterion(dm, self._bkd)

        jac = crit.jacobian(self._weights)

        self.assertEqual(jac.shape, (1, self._ndesign_pts))
        self.assertTrue(self._bkd.all_bool(self._bkd.isfinite(jac)))

    def test_quantile_homoscedastic_value(self):
        """Test D-optimal value for homoscedastic quantile regression."""
        dm = QuantileDesignMatrices(self._design_factors, self._bkd)
        crit = DOptimalQuantileCriterion(dm, self._bkd)

        val = crit(self._weights)

        self.assertEqual(val.shape, (1, 1))
        self.assertTrue(self._bkd.all_bool(self._bkd.isfinite(val)))

    def test_quantile_heteroscedastic_jacobian(self):
        """Test D-optimal Jacobian for heteroscedastic quantile regression."""
        dm = QuantileDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        crit = DOptimalQuantileCriterion(dm, self._bkd)

        jac = crit.jacobian(self._weights)

        self.assertEqual(jac.shape, (1, self._ndesign_pts))
        self.assertTrue(self._bkd.all_bool(self._bkd.isfinite(jac)))

    def test_nvars_nqoi(self):
        """Test nvars and nqoi properties."""
        dm = LeastSquaresDesignMatrices(self._design_factors, self._bkd)
        crit = DOptimalCriterion(dm, self._bkd)

        self.assertEqual(crit.nvars(), self._ndesign_pts)
        self.assertEqual(crit.nqoi(), 1)
        self.assertEqual(crit.ndesign_vars(), self._ndesign_vars)


class TestDOptimalCriterionNumpy(TestDOptimalCriterion[NDArray[Any]]):
    """NumPy backend tests for D-optimal criterion."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()

    def test_legacy_comparison_homoscedastic(self):
        """Compare with legacy implementation for homoscedastic case."""
        from pyapprox.expdesign.local import DOptimalLstSqCriterion
        from pyapprox.util.backends.numpy import NumpyMixin

        dm = LeastSquaresDesignMatrices(self._design_factors, self._bkd)
        new_crit = DOptimalCriterion(dm, self._bkd)
        legacy_crit = DOptimalLstSqCriterion(
            self._design_factors, None, backend=NumpyMixin
        )

        new_val = new_crit(self._weights)
        legacy_val = legacy_crit(self._weights)
        self._bkd.assert_allclose(new_val, legacy_val, rtol=1e-12)

        new_jac = new_crit.jacobian(self._weights)
        legacy_jac = legacy_crit.jacobian(self._weights)
        self._bkd.assert_allclose(new_jac, legacy_jac, rtol=1e-12)

    def test_legacy_comparison_heteroscedastic(self):
        """Compare with legacy implementation for heteroscedastic case."""
        from pyapprox.expdesign.local import DOptimalLstSqCriterion
        from pyapprox.util.backends.numpy import NumpyMixin

        dm = LeastSquaresDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        new_crit = DOptimalCriterion(dm, self._bkd)
        legacy_crit = DOptimalLstSqCriterion(
            self._design_factors, self._noise_mult, backend=NumpyMixin
        )

        new_val = new_crit(self._weights)
        legacy_val = legacy_crit(self._weights)
        self._bkd.assert_allclose(new_val, legacy_val, rtol=1e-12)

        new_jac = new_crit.jacobian(self._weights)
        legacy_jac = legacy_crit.jacobian(self._weights)
        self._bkd.assert_allclose(new_jac, legacy_jac, rtol=1e-12)

    def test_legacy_comparison_quantile(self):
        """Compare with legacy quantile regression implementation."""
        from pyapprox.expdesign.local import (
            DOptimalQuantileCriterion as LegacyQuantileCrit,
        )
        from pyapprox.util.backends.numpy import NumpyMixin

        dm = QuantileDesignMatrices(
            self._design_factors, self._bkd, self._noise_mult
        )
        new_crit = DOptimalQuantileCriterion(dm, self._bkd)
        legacy_crit = LegacyQuantileCrit(
            self._design_factors, self._noise_mult, backend=NumpyMixin
        )

        new_val = new_crit(self._weights)
        legacy_val = legacy_crit(self._weights)
        self._bkd.assert_allclose(new_val, legacy_val, rtol=1e-12)

        new_jac = new_crit.jacobian(self._weights)
        legacy_jac = legacy_crit.jacobian(self._weights)
        self._bkd.assert_allclose(new_jac, legacy_jac, rtol=1e-12)

    def test_numerical_jacobian(self):
        """Verify Jacobian using finite differences."""
        dm = LeastSquaresDesignMatrices(self._design_factors, self._bkd)
        crit = DOptimalCriterion(dm, self._bkd)

        # Compute analytical Jacobian
        analytical_jac = crit.jacobian(self._weights)

        # Compute numerical Jacobian
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


class TestDOptimalCriterionTorch(TestDOptimalCriterion[torch.Tensor]):
    """PyTorch backend tests for D-optimal criterion."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
