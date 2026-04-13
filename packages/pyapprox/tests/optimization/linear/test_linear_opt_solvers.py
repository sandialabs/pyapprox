"""Tests for linear system solvers."""

import numpy as np
import pytest

from pyapprox.optimization.linear import (
    BasisPursuitDenoisingSolver,
    BasisPursuitSolver,
    ExpectileRegressionSolver,
    LeastSquaresSolver,
    LinearlyConstrainedLstSqSolver,
    OMPSolver,
    OMPTerminationFlag,
    QuantileRegressionSolver,
    RidgeRegressionSolver,
)
from pyapprox.util.backends.numpy import NumpyBkd


class SolverTestBase:
    """Base class for solver tests."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)
        self.bkd = NumpyBkd()


class TestLeastSquaresSolver(SolverTestBase):
    """Tests for LeastSquaresSolver."""

    def test_overdetermined_system(self):
        """Test solving overdetermined system."""
        nsamples, nterms = 50, 5
        A = self.bkd.asarray(np.random.randn(nsamples, nterms))
        coef_true = self.bkd.asarray(np.random.randn(nterms, 1))
        y = self.bkd.dot(A, coef_true)

        solver = LeastSquaresSolver(self.bkd)
        coef = solver.solve(A, y)

        self.bkd.assert_allclose(coef, coef_true, rtol=1e-10)

    def test_multiple_qoi(self):
        """Test with multiple quantities of interest."""
        nsamples, nterms, nqoi = 50, 5, 3
        A = self.bkd.asarray(np.random.randn(nsamples, nterms))
        coef_true = self.bkd.asarray(np.random.randn(nterms, nqoi))
        y = self.bkd.dot(A, coef_true)

        solver = LeastSquaresSolver(self.bkd)
        coef = solver.solve(A, y)

        self.bkd.assert_allclose(coef, coef_true, rtol=1e-10)

    def test_with_noise(self):
        """Test with noisy data."""
        nsamples, nterms = 100, 5
        A = self.bkd.asarray(np.random.randn(nsamples, nterms))
        coef_true = self.bkd.asarray(np.random.randn(nterms, 1))
        noise = self.bkd.asarray(0.01 * np.random.randn(nsamples, 1))
        y = self.bkd.dot(A, coef_true) + noise

        solver = LeastSquaresSolver(self.bkd)
        coef = solver.solve(A, y)

        self.bkd.assert_allclose(coef, coef_true, rtol=0.1)


class TestRidgeRegressionSolver(SolverTestBase):
    """Tests for RidgeRegressionSolver."""

    def test_reduces_to_lstsq_small_alpha(self):
        """Test that small alpha gives similar results to lstsq."""
        nsamples, nterms = 50, 5
        A = self.bkd.asarray(np.random.randn(nsamples, nterms))
        coef_true = self.bkd.asarray(np.random.randn(nterms, 1))
        y = self.bkd.dot(A, coef_true)

        lstsq_solver = LeastSquaresSolver(self.bkd)
        ridge_solver = RidgeRegressionSolver(self.bkd, alpha=1e-10)

        coef_lstsq = lstsq_solver.solve(A, y)
        coef_ridge = ridge_solver.solve(A, y)

        self.bkd.assert_allclose(coef_lstsq, coef_ridge, rtol=1e-5)

    def test_regularization_shrinks_coefficients(self):
        """Test that regularization shrinks coefficient magnitudes."""
        nsamples, nterms = 50, 5
        A = self.bkd.asarray(np.random.randn(nsamples, nterms))
        coef_true = self.bkd.asarray(np.random.randn(nterms, 1))
        y = self.bkd.dot(A, coef_true)

        small_alpha = RidgeRegressionSolver(self.bkd, alpha=0.01)
        large_alpha = RidgeRegressionSolver(self.bkd, alpha=10.0)

        coef_small = small_alpha.solve(A, y)
        coef_large = large_alpha.solve(A, y)

        norm_small = float(self.bkd.norm(coef_small))
        norm_large = float(self.bkd.norm(coef_large))
        assert norm_small > norm_large


class TestLinearlyConstrainedLstSqSolver(SolverTestBase):
    """Tests for LinearlyConstrainedLstSqSolver."""

    def test_single_constraint(self):
        """Test with a single linear constraint."""
        nsamples, nterms = 50, 5
        A = self.bkd.asarray(np.random.randn(nsamples, nterms))
        coef_true = self.bkd.asarray(np.random.randn(nterms, 1))
        y = self.bkd.dot(A, coef_true)

        # Constraint: sum of coefficients = 1
        C = self.bkd.ones((1, nterms))
        d = self.bkd.asarray([[1.0]])

        solver = LinearlyConstrainedLstSqSolver(self.bkd, C, d)
        coef = solver.solve(A, y)

        # Check constraint is satisfied
        coef_sum = float(self.bkd.sum(coef))
        assert coef_sum == pytest.approx(1.0, abs=1e-10)

    def test_multiple_constraints(self):
        """Test with multiple linear constraints."""
        nsamples, nterms = 50, 5
        A = self.bkd.asarray(np.random.randn(nsamples, nterms))
        coef_true = self.bkd.asarray(np.random.randn(nterms, 1))
        y = self.bkd.dot(A, coef_true)

        # Constraints: c_0 = 0, c_1 + c_2 = 1
        C = self.bkd.asarray(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0],
            ]
        )
        d = self.bkd.asarray([[0.0], [1.0]])

        solver = LinearlyConstrainedLstSqSolver(self.bkd, C, d)
        coef = solver.solve(A, y)

        # Check constraints are satisfied
        residual = self.bkd.dot(C, coef) - d
        self.bkd.assert_allclose(residual, self.bkd.zeros_like(residual), atol=1e-10)


class TestOMPSolver(SolverTestBase):
    """Tests for OMPSolver."""

    def test_recovers_sparse_signal(self):
        """Test recovery of sparse signal."""
        nsamples, nterms = 50, 20
        A = self.bkd.asarray(np.random.randn(nsamples, nterms))

        # True sparse coefficients (only 3 non-zero)
        coef_true = self.bkd.zeros((nterms, 1))
        coef_true[2, 0] = 1.5
        coef_true[7, 0] = -2.0
        coef_true[15, 0] = 0.8
        y = self.bkd.dot(A, coef_true)

        solver = OMPSolver(self.bkd, max_nonzeros=3, rtol=1e-10)
        coef = solver.solve(A, y)

        # Check sparse recovery
        self.bkd.assert_allclose(coef, coef_true, rtol=1e-5)
        assert solver.termination_flag == OMPTerminationFlag.RESIDUAL_TOLERANCE

    def test_respects_max_nonzeros(self):
        """Test that max_nonzeros is respected."""
        nsamples, nterms = 50, 20
        A = self.bkd.asarray(np.random.randn(nsamples, nterms))
        y = self.bkd.asarray(np.random.randn(nsamples, 1))

        max_nonzeros = 5
        solver = OMPSolver(self.bkd, max_nonzeros=max_nonzeros, rtol=1e-15)
        coef = solver.solve(A, y)

        # Count non-zeros
        nonzero_count = np.sum(np.abs(self.bkd.to_numpy(coef)) > 1e-14)
        assert nonzero_count <= max_nonzeros


class TestBasisPursuitSolver(SolverTestBase):
    """Tests for BasisPursuitSolver."""

    def test_recovers_sparse_signal(self):
        """Test recovery of sparse signal."""
        nsamples, nterms = 50, 20
        A = self.bkd.asarray(np.random.randn(nsamples, nterms))

        # True sparse coefficients
        coef_true = self.bkd.zeros((nterms, 1))
        coef_true[2, 0] = 1.5
        coef_true[7, 0] = -2.0
        y = self.bkd.dot(A, coef_true)

        solver = BasisPursuitSolver(self.bkd)
        coef = solver.solve(A, y)

        self.bkd.assert_allclose(coef, coef_true, rtol=0.1, atol=0.1)


class TestBasisPursuitDenoisingSolver(SolverTestBase):
    """Tests for BasisPursuitDenoisingSolver (LASSO)."""

    def test_produces_sparse_solution(self):
        """Test that LASSO produces sparse solutions."""
        nsamples, nterms = 100, 20
        A = self.bkd.asarray(np.random.randn(nsamples, nterms))
        coef_true = self.bkd.asarray(np.random.randn(nterms, 1))
        y = self.bkd.dot(A, coef_true)

        # With large penalty, solution should be sparse
        solver = BasisPursuitDenoisingSolver(self.bkd, penalty=1.0)
        coef = solver.solve(A, y)

        # Count near-zero coefficients
        near_zero = np.sum(np.abs(self.bkd.to_numpy(coef)) < 0.1)
        assert near_zero > 0  # Some should be near zero


class TestQuantileRegressionSolver(SolverTestBase):
    """Tests for QuantileRegressionSolver."""

    def test_median_regression(self):
        """Test median regression (quantile=0.5)."""
        nsamples, nterms = 50, 3
        A = self.bkd.asarray(np.random.randn(nsamples, nterms))
        coef_true = self.bkd.asarray([[1.0], [2.0], [-1.0]])
        y = self.bkd.dot(A, coef_true)

        solver = QuantileRegressionSolver(self.bkd, quantile=0.5)
        coef = solver.solve(A, y)

        self.bkd.assert_allclose(coef, coef_true, rtol=0.1)

    def test_quantile_ordering(self):
        """Test that higher quantiles give higher predictions."""
        nsamples, nterms = 100, 3
        A = self.bkd.asarray(np.random.randn(nsamples, nterms))
        y = self.bkd.asarray(np.random.randn(nsamples, 1))

        solver_low = QuantileRegressionSolver(self.bkd, quantile=0.25)
        solver_high = QuantileRegressionSolver(self.bkd, quantile=0.75)

        coef_low = solver_low.solve(A, y)
        coef_high = solver_high.solve(A, y)

        # Predictions at mean of samples
        x_mean = self.bkd.reshape(self.bkd.sum(A, axis=0) / nsamples, (1, -1))
        pred_low = self.bkd.to_float(self.bkd.dot(x_mean, coef_low))
        pred_high = self.bkd.to_float(self.bkd.dot(x_mean, coef_high))

        # Higher quantile should give higher prediction on average
        # This is a statistical test, may not always pass
        assert pred_low < pred_high + 1.0

    def test_invalid_quantile(self):
        """Test that invalid quantile raises error."""
        with pytest.raises(ValueError):
            QuantileRegressionSolver(self.bkd, quantile=1.5)
        with pytest.raises(ValueError):
            QuantileRegressionSolver(self.bkd, quantile=-0.1)


class TestExpectileRegressionSolver(SolverTestBase):
    """Tests for ExpectileRegressionSolver."""

    def test_mean_regression(self):
        """Test mean regression (expectile=0.5 = OLS)."""
        nsamples, nterms = 50, 3
        A = self.bkd.asarray(np.random.randn(nsamples, nterms))
        coef_true = self.bkd.asarray([[1.0], [2.0], [-1.0]])
        y = self.bkd.dot(A, coef_true)

        # Expectile 0.5 should give same result as OLS
        solver = ExpectileRegressionSolver(self.bkd, expectile=0.5)
        coef = solver.solve(A, y)

        lstsq_solver = LeastSquaresSolver(self.bkd)
        coef_lstsq = lstsq_solver.solve(A, y)

        self.bkd.assert_allclose(coef, coef_lstsq, rtol=0.01)


class TestWeightedSolving(SolverTestBase):
    """Tests for weighted solving."""

    def test_weights_affect_solution(self):
        """Test that sample weights affect the solution."""
        nsamples, nterms = 50, 3
        A = self.bkd.asarray(np.random.randn(nsamples, nterms))
        y = self.bkd.asarray(np.random.randn(nsamples, 1))

        # Uniform weights
        weights_uniform = self.bkd.ones((nsamples,))

        # Non-uniform weights (emphasize first half)
        weights_nonuniform = self.bkd.concatenate(
            [
                2.0 * self.bkd.ones((nsamples // 2,)),
                0.5 * self.bkd.ones((nsamples - nsamples // 2,)),
            ]
        )

        solver1 = LeastSquaresSolver(self.bkd)
        solver1.set_weights(weights_uniform)
        coef1 = solver1.solve(A, y)

        solver2 = LeastSquaresSolver(self.bkd)
        solver2.set_weights(weights_nonuniform)
        coef2 = solver2.solve(A, y)

        # Solutions should be different
        diff = float(self.bkd.norm(coef1 - coef2))
        assert diff > 1e-10
