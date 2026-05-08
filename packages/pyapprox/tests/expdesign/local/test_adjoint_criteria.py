"""
Tests for adjoint-based OED criteria (C, A, I-optimal).

Tests cover:
- Value and Jacobian comparison with legacy code
- HVP verification via numerical finite differences
- Homoscedastic and heteroscedastic noise
- Least squares regression
- Dual-backend support (NumPy and PyTorch)
"""

import numpy as np

from pyapprox.expdesign.local.criteria import (
    AOptimalCriterion,
    COptimalCriterion,
    IOptimalCriterion,
)
from pyapprox.expdesign.local.design_matrices import (
    LeastSquaresDesignMatrices,
)

# TODO: most of this code is repetitive. Cant we just use consitent
# api of criterion to write tests once and use pytest paramterized tests
# TODO: always use DerivativeChecker for checking derivatives

class TestCOptimalCriterion:
    """Base test class for C-optimal criterion."""

    def _setup_data(self, bkd):
        np.random.seed(42)

        # Create design factors
        self._ndesign_pts = 10
        self._ndesign_vars = 4
        self._design_factors = bkd.asarray(
            np.random.randn(self._ndesign_pts, self._ndesign_vars)
        )
        self._noise_mult = bkd.asarray(0.5 + np.random.rand(self._ndesign_pts))

        # C-optimal vector
        self._vec = bkd.asarray(np.ones(self._ndesign_vars))

        # Random weights on simplex
        raw_weights = np.random.uniform(0, 1, (self._ndesign_pts, 1))
        self._weights = bkd.asarray(raw_weights / raw_weights.sum())

        # Direction for HVP tests
        self._direction = bkd.asarray(np.random.randn(self._ndesign_pts, 1))

    def test_value_shape(self, bkd):
        """Test that C-optimal value has correct shape."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = COptimalCriterion(dm, self._vec, bkd)

        val = crit(self._weights)
        assert val.shape == (1, 1)
        assert bkd.all_bool(bkd.isfinite(val))

    def test_jacobian_shape(self, bkd):
        """Test that C-optimal Jacobian has correct shape."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = COptimalCriterion(dm, self._vec, bkd)

        jac = crit.jacobian(self._weights)
        assert jac.shape == (1, self._ndesign_pts)
        assert bkd.all_bool(bkd.isfinite(jac))

    def test_hvp_shape(self, bkd):
        """Test that C-optimal HVP has correct shape."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = COptimalCriterion(dm, self._vec, bkd)

        hvp = crit.hvp(self._weights, self._direction)
        assert hvp.shape == (self._ndesign_pts, 1)
        assert bkd.all_bool(bkd.isfinite(hvp))

    def test_hvp_available(self, bkd):
        """Test that HVP is available (following optional methods convention)."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = COptimalCriterion(dm, self._vec, bkd)
        assert hasattr(crit, "hvp")

    def test_nvars_nqoi(self, bkd):
        """Test nvars and nqoi properties."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = COptimalCriterion(dm, self._vec, bkd)

        assert crit.nvars() == self._ndesign_pts
        assert crit.nqoi() == 1
        assert crit.ndesign_vars() == self._ndesign_vars

    def test_numerical_jacobian(self, numpy_bkd):
        """Verify Jacobian using finite differences."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = COptimalCriterion(dm, self._vec, bkd)

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

        numerical_jac = bkd.asarray(numerical_jac)
        bkd.assert_allclose(analytical_jac, numerical_jac, rtol=1e-5)

    def test_numerical_hvp(self, numpy_bkd):
        """Verify HVP using finite differences."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = COptimalCriterion(dm, self._vec, bkd)

        analytical_hvp = crit.hvp(self._weights, self._direction)

        eps = 1e-7
        jac_plus = crit.jacobian(self._weights + eps * self._direction)
        jac_minus = crit.jacobian(self._weights - eps * self._direction)
        numerical_hvp = (jac_plus - jac_minus).T / (2 * eps)

        bkd.assert_allclose(analytical_hvp, numerical_hvp, rtol=1e-5)


class TestAOptimalCriterion:
    """Base test class for A-optimal criterion."""

    def _setup_data(self, bkd):
        np.random.seed(42)

        self._ndesign_pts = 10
        self._ndesign_vars = 4
        self._design_factors = bkd.asarray(
            np.random.randn(self._ndesign_pts, self._ndesign_vars)
        )
        self._noise_mult = bkd.asarray(0.5 + np.random.rand(self._ndesign_pts))

        raw_weights = np.random.uniform(0, 1, (self._ndesign_pts, 1))
        self._weights = bkd.asarray(raw_weights / raw_weights.sum())

        self._direction = bkd.asarray(np.random.randn(self._ndesign_pts, 1))

    def test_value_shape(self, bkd):
        """Test that A-optimal value has correct shape."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = AOptimalCriterion(dm, bkd)

        val = crit(self._weights)
        assert val.shape == (1, 1)
        assert bkd.all_bool(bkd.isfinite(val))

    def test_jacobian_shape(self, bkd):
        """Test that A-optimal Jacobian has correct shape."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = AOptimalCriterion(dm, bkd)

        jac = crit.jacobian(self._weights)
        assert jac.shape == (1, self._ndesign_pts)
        assert bkd.all_bool(bkd.isfinite(jac))

    def test_hvp_shape(self, bkd):
        """Test that A-optimal HVP has correct shape."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = AOptimalCriterion(dm, bkd)

        hvp = crit.hvp(self._weights, self._direction)
        assert hvp.shape == (self._ndesign_pts, 1)
        assert bkd.all_bool(bkd.isfinite(hvp))

    def test_nvars_nqoi(self, bkd):
        """Test nvars and nqoi properties."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = AOptimalCriterion(dm, bkd)

        assert crit.nvars() == self._ndesign_pts
        assert crit.nqoi() == 1

    def test_numerical_jacobian(self, numpy_bkd):
        """Verify Jacobian using finite differences."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = AOptimalCriterion(dm, bkd)

        analytical_jac = crit.jacobian(self._weights)

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

    def test_numerical_hvp(self, numpy_bkd):
        """Verify HVP using finite differences."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = AOptimalCriterion(dm, bkd)

        analytical_hvp = crit.hvp(self._weights, self._direction)

        eps = 1e-7
        jac_plus = crit.jacobian(self._weights + eps * self._direction)
        jac_minus = crit.jacobian(self._weights - eps * self._direction)
        numerical_hvp = (jac_plus - jac_minus).T / (2 * eps)

        bkd.assert_allclose(analytical_hvp, numerical_hvp, rtol=1e-5)

    def test_trace_formula(self, numpy_bkd):
        """Verify A-optimal equals trace(Cov) for homoscedastic case."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(self._design_factors, bkd)
        crit = AOptimalCriterion(dm, bkd)

        val = crit(self._weights)

        # Compute trace(M1^{-1}) directly
        M1 = dm.M1(self._weights)
        M1_inv = bkd.inv(M1)
        expected = bkd.trace(M1_inv)

        bkd.assert_allclose(val, bkd.reshape(expected, (1, 1)), rtol=1e-12)


class TestIOptimalCriterion:
    """Base test class for I-optimal criterion."""

    def _setup_data(self, bkd):
        np.random.seed(42)

        self._ndesign_pts = 10
        self._ndesign_vars = 4
        self._npred_pts = 8
        self._design_factors = bkd.asarray(
            np.random.randn(self._ndesign_pts, self._ndesign_vars)
        )
        self._pred_factors = bkd.asarray(
            np.random.randn(self._npred_pts, self._ndesign_vars)
        )
        self._noise_mult = bkd.asarray(0.5 + np.random.rand(self._ndesign_pts))

        raw_weights = np.random.uniform(0, 1, (self._ndesign_pts, 1))
        self._weights = bkd.asarray(raw_weights / raw_weights.sum())

        self._direction = bkd.asarray(np.random.randn(self._ndesign_pts, 1))

    def test_value_shape(self, bkd):
        """Test that I-optimal value has correct shape."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = IOptimalCriterion(dm, self._pred_factors, bkd)

        val = crit(self._weights)
        assert val.shape == (1, 1)
        assert bkd.all_bool(bkd.isfinite(val))

    def test_jacobian_shape(self, bkd):
        """Test that I-optimal Jacobian has correct shape."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = IOptimalCriterion(dm, self._pred_factors, bkd)

        jac = crit.jacobian(self._weights)
        assert jac.shape == (1, self._ndesign_pts)
        assert bkd.all_bool(bkd.isfinite(jac))

    def test_hvp_shape(self, bkd):
        """Test that I-optimal HVP has correct shape."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = IOptimalCriterion(dm, self._pred_factors, bkd)

        hvp = crit.hvp(self._weights, self._direction)
        assert hvp.shape == (self._ndesign_pts, 1)
        assert bkd.all_bool(bkd.isfinite(hvp))

    def test_nvars_nqoi(self, bkd):
        """Test nvars and nqoi properties."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = IOptimalCriterion(dm, self._pred_factors, bkd)

        assert crit.nvars() == self._ndesign_pts
        assert crit.nqoi() == 1

    def test_numerical_jacobian(self, numpy_bkd):
        """Verify Jacobian using finite differences."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = IOptimalCriterion(dm, self._pred_factors, bkd)

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

        numerical_jac = bkd.asarray(numerical_jac)
        bkd.assert_allclose(analytical_jac, numerical_jac, rtol=1e-5)

    def test_numerical_hvp(self, numpy_bkd):
        """Verify HVP using finite differences."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = IOptimalCriterion(dm, self._pred_factors, bkd)

        analytical_hvp = crit.hvp(self._weights, self._direction)

        eps = 1e-7
        jac_plus = crit.jacobian(self._weights + eps * self._direction)
        jac_minus = crit.jacobian(self._weights - eps * self._direction)
        numerical_hvp = (jac_plus - jac_minus).T / (2 * eps)

        bkd.assert_allclose(analytical_hvp, numerical_hvp, rtol=1e-5)

    def test_with_pred_weights(self, numpy_bkd):
        """Test I-optimal with custom prediction weights."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        pred_weights = bkd.asarray(np.random.rand(self._npred_pts))
        pred_weights = pred_weights / bkd.sum(pred_weights)

        dm = LeastSquaresDesignMatrices(
            self._design_factors, bkd, self._noise_mult
        )
        crit = IOptimalCriterion(
            dm, self._pred_factors, bkd, pred_weights=pred_weights
        )

        val = crit(self._weights)
        assert val.shape == (1, 1)
        assert bkd.all_bool(bkd.isfinite(val))
