"""Tests for incremental Cholesky factorization."""

import unittest
import warnings
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.linalg.incremental_cholesky import (
    IncrementalCholeskyFactorization,
)
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


class TestIncrementalCholesky(Generic[Array], unittest.TestCase):
    """Base tests for IncrementalCholeskyFactorization."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(1)

    def test_incremental_matches_full(self) -> None:
        """Incremental factorization matches full Cholesky."""
        bkd = self._bkd
        n = 5
        B = bkd.asarray(np.random.normal(0, 1, (n, n)))
        K = B.T @ B
        fact = IncrementalCholeskyFactorization(K, bkd)
        for i in range(n):
            fact.add_pivot(i)

        L_inc = fact.L()
        L_full = bkd.cholesky(K)
        bkd.assert_allclose(L_inc, L_full, rtol=1e-10)

    def test_L_inv_is_inverse(self) -> None:
        """L_inv is the correct inverse of L."""
        bkd = self._bkd
        n = 4
        B = bkd.asarray(np.random.normal(0, 1, (n, n)))
        K = B.T @ B
        fact = IncrementalCholeskyFactorization(K, bkd)
        for i in range(n):
            fact.add_pivot(i)

        L = fact.L()
        L_inv = fact.L_inv()
        bkd.assert_allclose(L_inv @ L, bkd.eye(n), atol=1e-12)

    def test_partial_pivots(self) -> None:
        """Incremental with subset of pivots matches submatrix Cholesky."""
        bkd = self._bkd
        n = 5
        B = bkd.asarray(np.random.normal(0, 1, (n, n)))
        K = B.T @ B
        pivots = [2, 0, 4]
        fact = IncrementalCholeskyFactorization(K, bkd)
        for p in pivots:
            fact.add_pivot(p)

        self.assertEqual(fact.npivots(), 3)
        K_sub = K[pivots, :][:, pivots]
        L_expected = bkd.cholesky(K_sub)
        bkd.assert_allclose(fact.L(), L_expected, rtol=1e-10)

    def test_pivots_array(self) -> None:
        """pivots() returns correct array."""
        bkd = self._bkd
        K = bkd.eye(3)
        fact = IncrementalCholeskyFactorization(K, bkd)
        fact.add_pivot(2)
        fact.add_pivot(0)
        bkd.assert_allclose(
            fact.pivots(), bkd.asarray([2, 0])
        )

    def test_reset(self) -> None:
        """reset() clears all state."""
        bkd = self._bkd
        K = bkd.eye(3)
        fact = IncrementalCholeskyFactorization(K, bkd)
        fact.add_pivot(0)
        fact.reset()
        self.assertEqual(fact.npivots(), 0)
        with self.assertRaises(RuntimeError):
            fact.L()

    def test_update_K(self) -> None:
        """update_K replaces the kernel matrix reference."""
        bkd = self._bkd
        K1 = bkd.eye(3)
        K2 = 2.0 * bkd.eye(3)
        fact = IncrementalCholeskyFactorization(K1, bkd)
        fact.add_pivot(0)
        fact.update_K(K2)
        fact.reset()
        fact.add_pivot(0)
        bkd.assert_allclose(
            fact.L(), bkd.asarray([[bkd.sqrt(bkd.asarray([2.0]))[0]]]),
            rtol=1e-12,
        )

    def test_no_pivots_raises(self) -> None:
        """Accessing L or L_inv before adding pivots raises."""
        bkd = self._bkd
        K = bkd.eye(3)
        fact = IncrementalCholeskyFactorization(K, bkd)
        with self.assertRaises(RuntimeError):
            fact.L()
        with self.assertRaises(RuntimeError):
            fact.L_inv()

    def test_degenerate_pivot_warns_and_skips(self) -> None:
        """Degenerate pivot triggers warning and is skipped."""
        bkd = self._bkd
        # Rank-1 matrix: only first pivot is safe, second is degenerate
        K = bkd.asarray([[1.0, 1.0], [1.0, 1.0]])
        fact = IncrementalCholeskyFactorization(K, bkd)
        fact.add_pivot(0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fact.add_pivot(1)
            self.assertTrue(
                any("degenerate" in str(x.message) for x in w)
            )
        # Pivot was skipped, so npivots remains 1
        self.assertEqual(fact.npivots(), 1)


class TestIncrementalCholeskyNumpy(TestIncrementalCholesky[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIncrementalCholeskyTorch(TestIncrementalCholesky[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
