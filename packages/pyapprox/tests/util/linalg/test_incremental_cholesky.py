"""Tests for incremental Cholesky factorization."""

import warnings

import numpy as np
import pytest

from pyapprox.util.linalg.incremental_cholesky import (
    IncrementalCholeskyFactorization,
)


class TestIncrementalCholesky:
    """Base tests for IncrementalCholeskyFactorization."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(1)

    def test_incremental_matches_full(self, bkd) -> None:
        """Incremental factorization matches full Cholesky."""
        n = 5
        B = bkd.asarray(np.random.normal(0, 1, (n, n)))
        K = B.T @ B
        fact = IncrementalCholeskyFactorization(K, bkd)
        for i in range(n):
            fact.add_pivot(i)

        L_inc = fact.L()
        L_full = bkd.cholesky(K)
        bkd.assert_allclose(L_inc, L_full, rtol=1e-10)

    def test_L_inv_is_inverse(self, bkd) -> None:
        """L_inv is the correct inverse of L."""
        n = 4
        B = bkd.asarray(np.random.normal(0, 1, (n, n)))
        K = B.T @ B
        fact = IncrementalCholeskyFactorization(K, bkd)
        for i in range(n):
            fact.add_pivot(i)

        L = fact.L()
        L_inv = fact.L_inv()
        bkd.assert_allclose(L_inv @ L, bkd.eye(n), atol=1e-12)

    def test_partial_pivots(self, bkd) -> None:
        """Incremental with subset of pivots matches submatrix Cholesky."""
        n = 5
        B = bkd.asarray(np.random.normal(0, 1, (n, n)))
        K = B.T @ B
        pivots = [2, 0, 4]
        fact = IncrementalCholeskyFactorization(K, bkd)
        for p in pivots:
            fact.add_pivot(p)

        assert fact.npivots() == 3
        K_sub = K[pivots, :][:, pivots]
        L_expected = bkd.cholesky(K_sub)
        bkd.assert_allclose(fact.L(), L_expected, rtol=1e-10)

    def test_pivots_array(self, bkd) -> None:
        """pivots() returns correct array."""
        K = bkd.eye(3)
        fact = IncrementalCholeskyFactorization(K, bkd)
        fact.add_pivot(2)
        fact.add_pivot(0)
        bkd.assert_allclose(fact.pivots(), bkd.asarray([2, 0]))

    def test_reset(self, bkd) -> None:
        """reset() clears all state."""
        K = bkd.eye(3)
        fact = IncrementalCholeskyFactorization(K, bkd)
        fact.add_pivot(0)
        fact.reset()
        assert fact.npivots() == 0
        with pytest.raises(RuntimeError):
            fact.L()

    def test_update_K(self, bkd) -> None:
        """update_K replaces the kernel matrix reference."""
        K1 = bkd.eye(3)
        K2 = 2.0 * bkd.eye(3)
        fact = IncrementalCholeskyFactorization(K1, bkd)
        fact.add_pivot(0)
        fact.update_K(K2)
        fact.reset()
        fact.add_pivot(0)
        bkd.assert_allclose(
            fact.L(),
            bkd.asarray([[bkd.sqrt(bkd.asarray([2.0]))[0]]]),
            rtol=1e-12,
        )

    def test_no_pivots_raises(self, bkd) -> None:
        """Accessing L or L_inv before adding pivots raises."""
        K = bkd.eye(3)
        fact = IncrementalCholeskyFactorization(K, bkd)
        with pytest.raises(RuntimeError):
            fact.L()
        with pytest.raises(RuntimeError):
            fact.L_inv()

    def test_degenerate_pivot_warns_and_skips(self, bkd) -> None:
        """Degenerate pivot triggers warning and is skipped."""
        # Rank-1 matrix: only first pivot is safe, second is degenerate
        K = bkd.asarray([[1.0, 1.0], [1.0, 1.0]])
        fact = IncrementalCholeskyFactorization(K, bkd)
        fact.add_pivot(0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fact.add_pivot(1)
            assert any("degenerate" in str(x.message) for x in w)
        # Pivot was skipped, so npivots remains 1
        assert fact.npivots() == 1
