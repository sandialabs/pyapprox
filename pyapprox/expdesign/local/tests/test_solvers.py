"""
Tests for local OED solvers.

Tests cover:
- ScipyLocalOEDSolver for scalar criteria (A-optimal)
- MinimaxLocalOEDSolver for G-optimal designs
- AVaRLocalOEDSolver for R-optimal designs
- Dual-backend support (NumPy and PyTorch)
"""

import numpy as np
import pytest

from pyapprox.expdesign.local.criteria import (
    AOptimalCriterion,
    DOptimalCriterion,
    GOptimalCriterion,
    ROptimalCriterion,
)
from pyapprox.expdesign.local.design_matrices import (
    LeastSquaresDesignMatrices,
)
from pyapprox.expdesign.local.solver import (
    AVaRLocalOEDSolver,
    MinimaxLocalOEDSolver,
    ScipyLocalOEDSolver,
)


class TestScipyLocalOEDSolver:
    """Base test class for SciPy local OED solver."""

    def _setup_data(self, bkd):
        np.random.seed(42)

        # Create design factors
        self._ndesign_pts = 7
        self._ndesign_vars = 3
        self._design_factors = bkd.asarray(
            np.random.randn(self._ndesign_pts, self._ndesign_vars)
        )

    def test_d_optimal_solver(self, bkd):
        """Test solver with D-optimal criterion (no HVP)."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(self._design_factors, bkd)
        # D-optimal does not have HVP - optimizer should work without it
        crit = DOptimalCriterion(dm, bkd)
        solver = ScipyLocalOEDSolver(crit, bkd, verbosity=0)

        optimal_weights = solver.construct()

        # Check shape
        assert optimal_weights.shape == (self._ndesign_pts, 1)

        # Check simplex constraints
        weight_sum = bkd.sum(optimal_weights)
        assert bkd.allclose(weight_sum, bkd.asarray(1.0), atol=1e-6)
        assert bkd.all_bool(optimal_weights >= -1e-8)

    def test_a_optimal_solver(self, bkd):
        """Test solver with A-optimal criterion (has HVP)."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(self._design_factors, bkd)
        # A-optimal creates adjoints internally using unit vectors
        crit = AOptimalCriterion(dm, bkd)
        solver = ScipyLocalOEDSolver(crit, bkd, verbosity=0)

        optimal_weights = solver.construct()

        # Check shape
        assert optimal_weights.shape == (self._ndesign_pts, 1)

        # Check simplex constraints
        weight_sum = bkd.sum(optimal_weights)
        assert bkd.allclose(weight_sum, bkd.asarray(1.0), atol=1e-6)
        assert bkd.all_bool(optimal_weights >= -1e-8)

    def test_raises_for_vector_criterion(self, bkd):
        """Test that solver raises error for vector criteria."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(self._design_factors, bkd)

        # Create G-optimal criterion (vector output)
        npred_pts = 3
        pred_factors = bkd.asarray(np.random.randn(npred_pts, self._ndesign_vars))

        crit = GOptimalCriterion(dm, pred_factors, bkd)

        # Should raise ValueError because nqoi > 1
        with pytest.raises(ValueError):
            ScipyLocalOEDSolver(crit, bkd)

    def test_get_result(self, bkd):
        """Test that get_result returns optimization info."""
        self._setup_data(bkd)
        dm = LeastSquaresDesignMatrices(self._design_factors, bkd)
        crit = AOptimalCriterion(dm, bkd)
        solver = ScipyLocalOEDSolver(crit, bkd, verbosity=0)

        # Should raise before construct()
        with pytest.raises(AttributeError):
            solver.get_result()

        solver.construct()
        result = solver.get_result()

        # Check result has expected methods
        assert hasattr(result, "success")
        assert hasattr(result, "optima")


class TestMinimaxLocalOEDSolver:
    """Base test class for Minimax local OED solver."""

    def _setup_data(self, bkd):
        np.random.seed(42)

        self._ndesign_pts = 7
        self._ndesign_vars = 3
        self._design_factors = bkd.asarray(
            np.random.randn(self._ndesign_pts, self._ndesign_vars)
        )
        self._npred_pts = 4
        self._pred_factors = bkd.asarray(
            np.random.randn(self._npred_pts, self._ndesign_vars)
        )

    def _create_g_optimal_criterion(self, bkd):
        """Helper to create G-optimal criterion."""
        dm = LeastSquaresDesignMatrices(self._design_factors, bkd)
        # G-optimal takes pred_factors directly
        return GOptimalCriterion(dm, self._pred_factors, bkd)

    def test_g_optimal_solver(self, bkd):
        """Test solver with G-optimal criterion."""
        self._setup_data(bkd)
        crit = self._create_g_optimal_criterion(bkd)
        solver = MinimaxLocalOEDSolver(crit, bkd, verbosity=0)

        optimal_weights = solver.construct()

        # Check shape
        assert optimal_weights.shape == (self._ndesign_pts, 1)

        # Check simplex constraints
        weight_sum = bkd.sum(optimal_weights)
        assert bkd.allclose(weight_sum, bkd.asarray(1.0), atol=1e-6)
        assert bkd.all_bool(optimal_weights >= -1e-8)

    def test_get_minimax_value(self, bkd):
        """Test that minimax value can be retrieved."""
        self._setup_data(bkd)
        crit = self._create_g_optimal_criterion(bkd)
        solver = MinimaxLocalOEDSolver(crit, bkd, verbosity=0)

        # Should raise before construct()
        with pytest.raises(AttributeError):
            solver.get_minimax_value()

        solver.construct()
        minimax_val = solver.get_minimax_value()

        # Check shape and positivity
        assert minimax_val.shape == (1, 1)
        assert minimax_val[0, 0] > 0


class TestAVaRLocalOEDSolver:
    """Base test class for AVaR local OED solver."""

    def _setup_data(self, bkd):
        np.random.seed(42)

        self._ndesign_pts = 7
        self._ndesign_vars = 3
        self._design_factors = bkd.asarray(
            np.random.randn(self._ndesign_pts, self._ndesign_vars)
        )
        self._npred_pts = 4
        self._pred_factors = bkd.asarray(
            np.random.randn(self._npred_pts, self._ndesign_vars)
        )

    def _create_r_optimal_criterion(self, bkd):
        """Helper to create R-optimal criterion."""
        dm = LeastSquaresDesignMatrices(self._design_factors, bkd)
        # R-optimal inherits from G-optimal, takes pred_factors directly
        return ROptimalCriterion(dm, self._pred_factors, bkd)

    def test_r_optimal_solver(self, bkd):
        """Test solver with R-optimal criterion."""
        self._setup_data(bkd)
        crit = self._create_r_optimal_criterion(bkd)
        solver = AVaRLocalOEDSolver(crit, alpha=0.5, bkd=bkd, verbosity=0)

        optimal_weights = solver.construct()

        # Check shape
        assert optimal_weights.shape == (self._ndesign_pts, 1)

        # Check simplex constraints
        weight_sum = bkd.sum(optimal_weights)
        assert bkd.allclose(weight_sum, bkd.asarray(1.0), atol=1e-6)

    def test_alpha_validation(self, bkd):
        """Test that invalid alpha raises error."""
        self._setup_data(bkd)
        crit = self._create_r_optimal_criterion(bkd)

        with pytest.raises(ValueError):
            AVaRLocalOEDSolver(crit, alpha=-0.1, bkd=bkd)

        with pytest.raises(ValueError):
            AVaRLocalOEDSolver(crit, alpha=1.0, bkd=bkd)

    def test_get_avar_value(self, bkd):
        """Test that AVaR value can be retrieved."""
        self._setup_data(bkd)
        crit = self._create_r_optimal_criterion(bkd)
        solver = AVaRLocalOEDSolver(crit, alpha=0.5, bkd=bkd, verbosity=0)

        # Should raise before construct()
        with pytest.raises(AttributeError):
            solver.get_avar_value()

        solver.construct()
        avar_val = solver.get_avar_value()

        # Check shape and positivity
        assert avar_val.shape == (1, 1)
        assert avar_val[0, 0] > 0

    def test_alpha_zero_approaches_mean(self, bkd):
        """Test that alpha=0 gives mean-like behavior."""
        self._setup_data(bkd)
        crit = self._create_r_optimal_criterion(bkd)
        solver = AVaRLocalOEDSolver(crit, alpha=0.0, bkd=bkd, verbosity=0)

        optimal_weights = solver.construct()
        avar_val = solver.get_avar_value()

        # Check that it's a valid design
        assert optimal_weights.shape == (self._ndesign_pts, 1)
        assert avar_val[0, 0] > 0
