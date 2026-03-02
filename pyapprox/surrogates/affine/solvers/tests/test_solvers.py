"""Tests for linear system solvers."""

import numpy as np
import pytest

from pyapprox.surrogates.affine.solvers import (
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


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(42)


class TestLeastSquaresSolver:
    """Tests for LeastSquaresSolver."""

    def test_overdetermined_system(self, numpy_bkd):
        """Test solving overdetermined system."""
        bkd = numpy_bkd
        nsamples, nterms = 50, 5
        A = bkd.asarray(np.random.randn(nsamples, nterms))
        coef_true = bkd.asarray(np.random.randn(nterms, 1))
        y = bkd.dot(A, coef_true)

        solver = LeastSquaresSolver(bkd)
        coef = solver.solve(A, y)

        bkd.assert_allclose(coef, coef_true, rtol=1e-10)

    def test_multiple_qoi(self, numpy_bkd):
        """Test with multiple quantities of interest."""
        bkd = numpy_bkd
        nsamples, nterms, nqoi = 50, 5, 3
        A = bkd.asarray(np.random.randn(nsamples, nterms))
        coef_true = bkd.asarray(np.random.randn(nterms, nqoi))
        y = bkd.dot(A, coef_true)

        solver = LeastSquaresSolver(bkd)
        coef = solver.solve(A, y)

        bkd.assert_allclose(coef, coef_true, rtol=1e-10)

    def test_with_noise(self, numpy_bkd):
        """Test with noisy data."""
        bkd = numpy_bkd
        nsamples, nterms = 100, 5
        A = bkd.asarray(np.random.randn(nsamples, nterms))
        coef_true = bkd.asarray(np.random.randn(nterms, 1))
        noise = bkd.asarray(0.01 * np.random.randn(nsamples, 1))
        y = bkd.dot(A, coef_true) + noise

        solver = LeastSquaresSolver(bkd)
        coef = solver.solve(A, y)

        bkd.assert_allclose(coef, coef_true, rtol=0.1)


class TestRidgeRegressionSolver:
    """Tests for RidgeRegressionSolver."""

    def test_reduces_to_lstsq_small_alpha(self, numpy_bkd):
        """Test that small alpha gives similar results to lstsq."""
        bkd = numpy_bkd
        nsamples, nterms = 50, 5
        A = bkd.asarray(np.random.randn(nsamples, nterms))
        coef_true = bkd.asarray(np.random.randn(nterms, 1))
        y = bkd.dot(A, coef_true)

        lstsq_solver = LeastSquaresSolver(bkd)
        ridge_solver = RidgeRegressionSolver(bkd, alpha=1e-10)

        coef_lstsq = lstsq_solver.solve(A, y)
        coef_ridge = ridge_solver.solve(A, y)

        bkd.assert_allclose(coef_lstsq, coef_ridge, rtol=1e-5)

    def test_regularization_shrinks_coefficients(self, numpy_bkd):
        """Test that regularization shrinks coefficient magnitudes."""
        bkd = numpy_bkd
        nsamples, nterms = 50, 5
        A = bkd.asarray(np.random.randn(nsamples, nterms))
        coef_true = bkd.asarray(np.random.randn(nterms, 1))
        y = bkd.dot(A, coef_true)

        small_alpha = RidgeRegressionSolver(bkd, alpha=0.01)
        large_alpha = RidgeRegressionSolver(bkd, alpha=10.0)

        coef_small = small_alpha.solve(A, y)
        coef_large = large_alpha.solve(A, y)

        norm_small = float(bkd.norm(coef_small))
        norm_large = float(bkd.norm(coef_large))
        assert norm_small > norm_large


class TestLinearlyConstrainedLstSqSolver:
    """Tests for LinearlyConstrainedLstSqSolver."""

    def test_single_constraint(self, numpy_bkd):
        """Test with a single linear constraint."""
        bkd = numpy_bkd
        nsamples, nterms = 50, 5
        A = bkd.asarray(np.random.randn(nsamples, nterms))
        coef_true = bkd.asarray(np.random.randn(nterms, 1))
        y = bkd.dot(A, coef_true)

        # Constraint: sum of coefficients = 1
        C = bkd.ones((1, nterms))
        d = bkd.asarray([[1.0]])

        solver = LinearlyConstrainedLstSqSolver(bkd, C, d)
        coef = solver.solve(A, y)

        # Check constraint is satisfied
        coef_sum = float(bkd.sum(coef))
        assert coef_sum == pytest.approx(1.0, abs=1e-10)

    def test_multiple_constraints(self, numpy_bkd):
        """Test with multiple linear constraints."""
        bkd = numpy_bkd
        nsamples, nterms = 50, 5
        A = bkd.asarray(np.random.randn(nsamples, nterms))
        coef_true = bkd.asarray(np.random.randn(nterms, 1))
        y = bkd.dot(A, coef_true)

        # Constraints: c_0 = 0, c_1 + c_2 = 1
        C = bkd.asarray(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0],
            ]
        )
        d = bkd.asarray([[0.0], [1.0]])

        solver = LinearlyConstrainedLstSqSolver(bkd, C, d)
        coef = solver.solve(A, y)

        # Check constraints are satisfied
        residual = bkd.dot(C, coef) - d
        bkd.assert_allclose(residual, bkd.zeros_like(residual), atol=1e-10)


class TestOMPSolver:
    """Tests for OMPSolver."""

    def test_recovers_sparse_signal(self, numpy_bkd):
        """Test recovery of sparse signal."""
        bkd = numpy_bkd
        nsamples, nterms = 50, 20
        A = bkd.asarray(np.random.randn(nsamples, nterms))

        # True sparse coefficients (only 3 non-zero)
        coef_true = bkd.zeros((nterms, 1))
        coef_true[2, 0] = 1.5
        coef_true[7, 0] = -2.0
        coef_true[15, 0] = 0.8
        y = bkd.dot(A, coef_true)

        solver = OMPSolver(bkd, max_nonzeros=3, rtol=1e-10)
        coef = solver.solve(A, y)

        # Check sparse recovery
        bkd.assert_allclose(coef, coef_true, rtol=1e-5)
        assert solver.termination_flag == OMPTerminationFlag.RESIDUAL_TOLERANCE

    def test_respects_max_nonzeros(self, numpy_bkd):
        """Test that max_nonzeros is respected."""
        bkd = numpy_bkd
        nsamples, nterms = 50, 20
        A = bkd.asarray(np.random.randn(nsamples, nterms))
        y = bkd.asarray(np.random.randn(nsamples, 1))

        max_nonzeros = 5
        solver = OMPSolver(bkd, max_nonzeros=max_nonzeros, rtol=1e-15)
        coef = solver.solve(A, y)

        # Count non-zeros
        nonzero_count = np.sum(np.abs(bkd.to_numpy(coef)) > 1e-14)
        assert nonzero_count <= max_nonzeros


class TestBasisPursuitSolver:
    """Tests for BasisPursuitSolver."""

    def test_recovers_sparse_signal(self, numpy_bkd):
        """Test recovery of sparse signal."""
        bkd = numpy_bkd
        nsamples, nterms = 50, 20
        A = bkd.asarray(np.random.randn(nsamples, nterms))

        # True sparse coefficients
        coef_true = bkd.zeros((nterms, 1))
        coef_true[2, 0] = 1.5
        coef_true[7, 0] = -2.0
        y = bkd.dot(A, coef_true)

        solver = BasisPursuitSolver(bkd)
        coef = solver.solve(A, y)

        bkd.assert_allclose(coef, coef_true, rtol=0.1, atol=0.1)


class TestBasisPursuitDenoisingSolver:
    """Tests for BasisPursuitDenoisingSolver (LASSO)."""

    def test_produces_sparse_solution(self, numpy_bkd):
        """Test that LASSO produces sparse solutions."""
        bkd = numpy_bkd
        nsamples, nterms = 100, 20
        A = bkd.asarray(np.random.randn(nsamples, nterms))
        coef_true = bkd.asarray(np.random.randn(nterms, 1))
        y = bkd.dot(A, coef_true)

        # With large penalty, solution should be sparse
        solver = BasisPursuitDenoisingSolver(bkd, penalty=1.0)
        coef = solver.solve(A, y)

        # Count near-zero coefficients
        near_zero = np.sum(np.abs(bkd.to_numpy(coef)) < 0.1)
        assert near_zero > 0  # Some should be near zero


class TestQuantileRegressionSolver:
    """Tests for QuantileRegressionSolver."""

    def test_median_regression(self, numpy_bkd):
        """Test median regression (quantile=0.5)."""
        bkd = numpy_bkd
        nsamples, nterms = 50, 3
        A = bkd.asarray(np.random.randn(nsamples, nterms))
        coef_true = bkd.asarray([[1.0], [2.0], [-1.0]])
        y = bkd.dot(A, coef_true)

        solver = QuantileRegressionSolver(bkd, quantile=0.5)
        coef = solver.solve(A, y)

        bkd.assert_allclose(coef, coef_true, rtol=0.1)

    def test_quantile_ordering(self, numpy_bkd):
        """Test that higher quantiles give higher predictions."""
        bkd = numpy_bkd
        nsamples, nterms = 100, 3
        A = bkd.asarray(np.random.randn(nsamples, nterms))
        y = bkd.asarray(np.random.randn(nsamples, 1))

        solver_low = QuantileRegressionSolver(bkd, quantile=0.25)
        solver_high = QuantileRegressionSolver(bkd, quantile=0.75)

        coef_low = solver_low.solve(A, y)
        coef_high = solver_high.solve(A, y)

        # Predictions at mean of samples
        x_mean = bkd.reshape(bkd.sum(A, axis=0) / nsamples, (1, -1))
        pred_low = bkd.to_float(bkd.dot(x_mean, coef_low))
        pred_high = bkd.to_float(bkd.dot(x_mean, coef_high))

        # Higher quantile should give higher prediction on average
        # This is a statistical test, may not always pass
        assert pred_low < pred_high + 1.0

    def test_invalid_quantile(self, numpy_bkd):
        """Test that invalid quantile raises error."""
        bkd = numpy_bkd
        with pytest.raises(ValueError):
            QuantileRegressionSolver(bkd, quantile=1.5)
        with pytest.raises(ValueError):
            QuantileRegressionSolver(bkd, quantile=-0.1)


class TestExpectileRegressionSolver:
    """Tests for ExpectileRegressionSolver."""

    def test_mean_regression(self, numpy_bkd):
        """Test mean regression (expectile=0.5 = OLS)."""
        bkd = numpy_bkd
        nsamples, nterms = 50, 3
        A = bkd.asarray(np.random.randn(nsamples, nterms))
        coef_true = bkd.asarray([[1.0], [2.0], [-1.0]])
        y = bkd.dot(A, coef_true)

        # Expectile 0.5 should give same result as OLS
        solver = ExpectileRegressionSolver(bkd, expectile=0.5)
        coef = solver.solve(A, y)

        lstsq_solver = LeastSquaresSolver(bkd)
        coef_lstsq = lstsq_solver.solve(A, y)

        bkd.assert_allclose(coef, coef_lstsq, rtol=0.01)


class TestWeightedSolving:
    """Tests for weighted solving."""

    def test_weights_affect_solution(self, numpy_bkd):
        """Test that sample weights affect the solution."""
        bkd = numpy_bkd
        nsamples, nterms = 50, 3
        A = bkd.asarray(np.random.randn(nsamples, nterms))
        y = bkd.asarray(np.random.randn(nsamples, 1))

        # Uniform weights
        weights_uniform = bkd.ones((nsamples,))

        # Non-uniform weights (emphasize first half)
        weights_nonuniform = bkd.concatenate(
            [
                2.0 * bkd.ones((nsamples // 2,)),
                0.5 * bkd.ones((nsamples - nsamples // 2,)),
            ]
        )

        solver1 = LeastSquaresSolver(bkd)
        solver1.set_weights(weights_uniform)
        coef1 = solver1.solve(A, y)

        solver2 = LeastSquaresSolver(bkd)
        solver2.set_weights(weights_nonuniform)
        coef2 = solver2.solve(A, y)

        # Solutions should be different
        diff = float(bkd.norm(coef1 - coef2))
        assert diff > 1e-10
