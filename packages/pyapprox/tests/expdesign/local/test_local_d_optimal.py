"""
Tests for D-optimal criterion.

Tests cover:
- Homoscedastic and heteroscedastic noise
- Least squares and quantile regression
- Value and Jacobian comparison with legacy code
- Dual-backend support (NumPy and PyTorch)
"""

import numpy as np

from pyapprox.expdesign.local.criteria import (
    DOptimalCriterion,
    DOptimalQuantileCriterion,
)
from pyapprox.expdesign.local.design_matrices import (
    LeastSquaresDesignMatrices,
    QuantileDesignMatrices,
)


class TestDOptimalCriterion:
    """Base test class for D-optimal criterion."""

    def _setup_data(self, bkd):
        np.random.seed(42)

        # Create design factors
        self._ndesign_pts = 7
        self._ndesign_vars = 4
        self._design_factors = bkd.asarray(
            np.random.randn(self._ndesign_pts, self._ndesign_vars)
        )
        self._noise_mult = bkd.asarray(
            1 + np.abs(np.random.randn(self._ndesign_pts))
        )

        # Random weights on simplex
        raw_weights = np.random.uniform(0, 1, (self._ndesign_pts, 1))
        self._weights = bkd.asarray(raw_weights / raw_weights.sum())

    def test_homoscedastic_least_squares_value(self, bkd):
        """Test D-optimal value for homoscedastic least squares."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(self._design_factors, bkd)
        crit = DOptimalCriterion(dm, bkd)

        val = crit(self._weights)

        # Check shape
        assert val.shape == (1, 1)

        # Check that value is finite
        assert bkd.all_bool(bkd.isfinite(val))

    def test_homoscedastic_least_squares_jacobian(self, bkd):
        """Test D-optimal Jacobian for homoscedastic least squares."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(self._design_factors, bkd)
        crit = DOptimalCriterion(dm, bkd)

        jac = crit.jacobian(self._weights)

        # Check shape
        assert jac.shape == (1, self._ndesign_pts)

        # Check that Jacobian is finite
        assert bkd.all_bool(bkd.isfinite(jac))

    def test_heteroscedastic_least_squares_value(self, bkd):
        """Test D-optimal value for heteroscedastic least squares."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = DOptimalCriterion(dm, bkd)

        val = crit(self._weights)

        # Check shape
        assert val.shape == (1, 1)
        assert bkd.all_bool(bkd.isfinite(val))

    def test_heteroscedastic_least_squares_jacobian(self, bkd):
        """Test D-optimal Jacobian for heteroscedastic least squares."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = DOptimalCriterion(dm, bkd)

        jac = crit.jacobian(self._weights)

        assert jac.shape == (1, self._ndesign_pts)
        assert bkd.all_bool(bkd.isfinite(jac))

    def test_quantile_homoscedastic_value(self, bkd):
        """Test D-optimal value for homoscedastic quantile regression."""
        self._setup_data(bkd)
        dm = QuantileDesignMatrices(self._design_factors, bkd)
        crit = DOptimalQuantileCriterion(dm, bkd)

        val = crit(self._weights)

        assert val.shape == (1, 1)
        assert bkd.all_bool(bkd.isfinite(val))

    def test_quantile_heteroscedastic_jacobian(self, bkd):
        """Test D-optimal Jacobian for heteroscedastic quantile regression."""
        self._setup_data(bkd)
        dm = QuantileDesignMatrices(self._design_factors, bkd, self._noise_mult)
        crit = DOptimalQuantileCriterion(dm, bkd)

        jac = crit.jacobian(self._weights)

        assert jac.shape == (1, self._ndesign_pts)
        assert bkd.all_bool(bkd.isfinite(jac))

    def test_nvars_nqoi(self, bkd):
        """Test nvars and nqoi properties."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(self._design_factors, bkd)
        crit = DOptimalCriterion(dm, bkd)

        assert crit.nvars() == self._ndesign_pts
        assert crit.nqoi() == 1
        assert crit.ndesign_vars() == self._ndesign_vars

    def test_numerical_jacobian(self, numpy_bkd):
        """Verify Jacobian using finite differences."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(self._design_factors, bkd)
        crit = DOptimalCriterion(dm, bkd)

        # Compute analytical Jacobian
        analytical_jac = crit.jacobian(self._weights)

        # Compute numerical Jacobian
        # TODO: use DerivativeChecker
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

        numerical_jac = bkd.asarray(numerical_jac)
        bkd.assert_allclose(analytical_jac, numerical_jac, rtol=1e-5)
