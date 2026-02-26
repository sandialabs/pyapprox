"""
Tests for minimax OED criteria (G-optimal, R-optimal).

Tests cover:
- Value and Jacobian comparison with legacy code
- Multi-output (nqoi > 1) shape verification
- Numerical Jacobian verification
- Dual-backend support (NumPy and PyTorch)
"""

import numpy as np

from pyapprox.expdesign.local.criteria import (
    GOptimalCriterion,
    ROptimalCriterion,
)
from pyapprox.expdesign.local.design_matrices import (
    LeastSquaresDesignMatrices,
)


class TestGOptimalCriterion:
    """Base test class for G-optimal criterion."""

    def _setup_data(self, bkd):
        np.random.seed(42)

        # Create design factors
        self._ndesign_pts = 10
        self._ndesign_vars = 4
        self._npred_pts = 6
        self._design_factors = bkd.asarray(
            np.random.randn(self._ndesign_pts, self._ndesign_vars)
        )
        self._pred_factors = bkd.asarray(
            np.random.randn(self._npred_pts, self._ndesign_vars)
        )
        self._noise_mult = bkd.asarray(0.5 + np.random.rand(self._ndesign_pts))

        # Random weights on simplex
        raw_weights = np.random.uniform(0, 1, (self._ndesign_pts, 1))
        self._weights = bkd.asarray(raw_weights / raw_weights.sum())

    def test_value_shape(self, bkd):
        """Test that G-optimal value has correct shape (npred_pts, 1)."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = GOptimalCriterion(dm, self._pred_factors, bkd)

        val = crit(self._weights)
        assert val.shape == (self._npred_pts, 1)
        assert bkd.all_bool(bkd.isfinite(val))

    def test_jacobian_shape(self, bkd):
        """Test that G-optimal Jacobian has correct shape (npred_pts, nvars)."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = GOptimalCriterion(dm, self._pred_factors, bkd)

        jac = crit.jacobian(self._weights)
        assert jac.shape == (self._npred_pts, self._ndesign_pts)
        assert bkd.all_bool(bkd.isfinite(jac))

    def test_nqoi(self, bkd):
        """Test that nqoi returns number of prediction points."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = GOptimalCriterion(dm, self._pred_factors, bkd)

        assert crit.nqoi() == self._npred_pts

    def test_nvars(self, bkd):
        """Test that nvars returns number of design points."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = GOptimalCriterion(dm, self._pred_factors, bkd)

        assert crit.nvars() == self._ndesign_pts

    def test_hvp_not_available(self, bkd):
        """Test that HVP is not available for G-optimal (optional methods
        convention)."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = GOptimalCriterion(dm, self._pred_factors, bkd)

        assert not hasattr(crit, "hvp")

    def test_values_are_positive(self, bkd):
        """Test that prediction variances are positive."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = GOptimalCriterion(dm, self._pred_factors, bkd)

        val = crit(self._weights)
        assert bkd.all_bool(val > 0)

    def test_numerical_jacobian(self, numpy_bkd):
        """Verify Jacobian using finite differences."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = GOptimalCriterion(dm, self._pred_factors, bkd)

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

        numerical_jac = bkd.asarray(numerical_jac)
        # Use both rtol and atol since some values can be very small
        bkd.assert_allclose(analytical_jac, numerical_jac, rtol=1e-5, atol=1e-8)

    def test_homoscedastic_case(self, numpy_bkd):
        """Test G-optimal for homoscedastic noise."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(self._design_factors, bkd)
        crit = GOptimalCriterion(dm, self._pred_factors, bkd)

        val = crit(self._weights)
        assert val.shape == (self._npred_pts, 1)
        assert bkd.all_bool(val > 0)


class TestROptimalCriterion:
    """Base test class for R-optimal criterion."""

    def _setup_data(self, bkd):
        np.random.seed(42)

        self._ndesign_pts = 10
        self._ndesign_vars = 4
        self._npred_pts = 6
        self._design_factors = bkd.asarray(
            np.random.randn(self._ndesign_pts, self._ndesign_vars)
        )
        self._pred_factors = bkd.asarray(
            np.random.randn(self._npred_pts, self._ndesign_vars)
        )
        self._noise_mult = bkd.asarray(0.5 + np.random.rand(self._ndesign_pts))

        raw_weights = np.random.uniform(0, 1, (self._ndesign_pts, 1))
        self._weights = bkd.asarray(raw_weights / raw_weights.sum())

    def test_value_shape(self, bkd):
        """Test that R-optimal value has correct shape."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = ROptimalCriterion(dm, self._pred_factors, bkd)

        val = crit(self._weights)
        assert val.shape == (self._npred_pts, 1)

    def test_jacobian_shape(self, bkd):
        """Test that R-optimal Jacobian has correct shape."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = ROptimalCriterion(dm, self._pred_factors, bkd)

        jac = crit.jacobian(self._weights)
        assert jac.shape == (self._npred_pts, self._ndesign_pts)

    def test_identical_to_g_optimal(self, bkd):
        """Test that R-optimal produces same values as G-optimal."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        g_crit = GOptimalCriterion(dm, self._pred_factors, bkd)

        # Need separate design matrices instance for R-optimal
        dm2 = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        r_crit = ROptimalCriterion(dm2, self._pred_factors, bkd)

        g_val = g_crit(self._weights)
        r_val = r_crit(self._weights)
        bkd.assert_allclose(g_val, r_val, rtol=1e-12)

        g_jac = g_crit.jacobian(self._weights)
        r_jac = r_crit.jacobian(self._weights)
        bkd.assert_allclose(g_jac, r_jac, rtol=1e-12)
