"""Tests for pivoted Cholesky factorization.

Tests replicate legacy tests from pyapprox/util/tests/test_linalg.py
using the new typing module implementation.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.linalg.pivoted_cholesky import (
    PivotedCholeskyFactorizer,
)
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestPivotedCholesky(Generic[Array], unittest.TestCase):
    """Base tests for PivotedCholeskyFactorizer.

    Replicates legacy test_pivoted_cholesky_decomposition and
    test_update_pivoted_cholesky from pyapprox/util/tests/test_linalg.py.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(1)

    def test_full_factorization(self) -> None:
        """Full pivoted Cholesky recovers A = L @ L.T."""
        bkd = self._bkd
        nrows = 4
        A = bkd.asarray(np.random.normal(0.0, 1.0, (nrows, nrows)))
        A = A.T @ A
        fact = PivotedCholeskyFactorizer(A, bkd)
        fact.factorize(nrows)
        self.assertTrue(fact.success())
        L = fact.factor()
        bkd.assert_allclose(L @ L.T, A, rtol=1e-10)

    def test_low_rank_factorization(self) -> None:
        """Partial factorization of rank-deficient matrix."""
        bkd = self._bkd
        nrows, npivots = 4, 2
        A = bkd.asarray(np.random.normal(0.0, 1.0, (npivots, nrows)))
        A = A.T @ A
        fact = PivotedCholeskyFactorizer(A, bkd)
        fact.factorize(npivots)
        L = fact.factor()
        self.assertEqual(L.shape, (nrows, npivots))
        self.assertEqual(fact.pivots().shape[0], npivots)
        self.assertEqual(fact.npivots(), npivots)
        bkd.assert_allclose(L @ L.T, A, rtol=1e-10)

    def test_init_pivots_enforced(self) -> None:
        """init_pivots forces specific pivot ordering."""
        bkd = self._bkd
        nrows, npivots = 4, 3
        A = bkd.asarray(np.random.normal(0.0, 1.0, (npivots, nrows)))
        A = A.T @ A

        # Get natural pivot order
        fact1 = PivotedCholeskyFactorizer(A, bkd)
        fact1.factorize(npivots)
        pivots1 = fact1.pivots()

        # Force second natural pivot to go first
        fact2 = PivotedCholeskyFactorizer(A, bkd)
        fact2.factorize(npivots, init_pivots=pivots1[1:2])
        pivots2 = fact2.pivots()
        pivots1_np = bkd.to_numpy(pivots1)
        bkd.assert_allclose(
            pivots2,
            bkd.asarray(pivots1_np[[1, 0, 2]]),
        )

    def test_known_matrix(self) -> None:
        """Factorize a known 3x3 matrix and compare with numpy cholesky."""
        bkd = self._bkd
        A = bkd.asarray([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]])
        fact = PivotedCholeskyFactorizer(A, bkd)
        fact.factorize(A.shape[0])
        L = fact.factor()

        # Reorder A so cholesky needs no pivoting
        true_pivots = np.array([2, 1, 0])
        A_no_pivots = A[true_pivots, :][:, true_pivots]
        L_np = bkd.cholesky(A_no_pivots)
        bkd.assert_allclose(L[fact.pivots(), :], L_np, rtol=1e-10)

    def test_known_matrix_permuted(self) -> None:
        """Factorize permuted known matrix."""
        bkd = self._bkd
        A_orig = bkd.asarray(
            [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]]
        )
        true_pivots = np.array([1, 0, 2])
        A = A_orig[true_pivots, :][:, true_pivots]

        fact = PivotedCholeskyFactorizer(A, bkd)
        fact.factorize(A.shape[0])
        L = fact.factor()

        orig_pivots = np.array([2, 1, 0])
        A_no_pivots = A_orig[orig_pivots, :][:, orig_pivots]
        L_np = bkd.cholesky(A_no_pivots)
        bkd.assert_allclose(L[fact.pivots(), :], L_np, rtol=1e-10)

    def test_econ_false(self) -> None:
        """Full Schur complement pivot selection mode."""
        bkd = self._bkd
        nrows = 4
        A = bkd.asarray(np.random.normal(0.0, 1.0, (nrows, nrows)))
        A = A.T @ A
        fact = PivotedCholeskyFactorizer(A, bkd, econ=False)
        fact.factorize(nrows)
        self.assertTrue(fact.success())
        L = fact.factor()
        bkd.assert_allclose(L @ L.T, A, rtol=1e-10)

    def test_econ_false_rank_deficient(self) -> None:
        """econ=False on rank-deficient matrix raises ValueError."""
        bkd = self._bkd
        nrows, rank = 4, 2
        A = bkd.asarray(np.random.normal(0.0, 1.0, (nrows, rank)))
        A = A.T @ A
        fact = PivotedCholeskyFactorizer(A, bkd, econ=False)
        with self.assertRaises(ValueError):
            fact.factorize(nrows)

    def test_econ_true_rank_deficient(self) -> None:
        """econ=True on rank-deficient matrix recovers A."""
        bkd = self._bkd
        nrows, rank = 4, 2
        A = bkd.asarray(np.random.normal(0.0, 1.0, (nrows, rank)))
        A = A @ A.T
        fact = PivotedCholeskyFactorizer(A, bkd, econ=True)
        fact.factorize(nrows)
        L = fact.factor()
        bkd.assert_allclose(L @ L.T, A, rtol=1e-10)

    def test_econ_false_rank_deficient_wide(self) -> None:
        """econ=False on rank-deficient wide matrix recovers A."""
        bkd = self._bkd
        nrows, rank = 4, 2
        A = bkd.asarray(np.random.normal(0.0, 1.0, (nrows, rank)))
        A = A @ A.T
        fact = PivotedCholeskyFactorizer(A, bkd, econ=False)
        fact.factorize(nrows)
        L = fact.factor()
        bkd.assert_allclose(L @ L.T, A, rtol=1e-10)

    def test_update_continues_factorization(self) -> None:
        """update() continues from partial factorization and matches full.

        Replicates legacy test_update_pivoted_cholesky.
        """
        bkd = self._bkd
        nrows = 10
        A = bkd.asarray(np.random.normal(0, 1, (nrows, nrows)))
        A = A.T @ A
        pivot_weights = bkd.asarray(np.random.uniform(1, 2, nrows))

        # Full factorization
        fact1 = PivotedCholeskyFactorizer(A, bkd)
        fact1.factorize(nrows, pivot_weights=pivot_weights)
        L1 = fact1.factor()

        # Partial then update
        npivots_partial = nrows - 2
        fact2 = PivotedCholeskyFactorizer(A, bkd)
        fact2.factorize(npivots_partial, pivot_weights=pivot_weights)
        self.assertEqual(fact2.npivots(), npivots_partial)
        fact2.update(nrows)
        L2 = fact2.factor()

        bkd.assert_allclose(L2, L1, rtol=1e-10)
        bkd.assert_allclose(fact2.pivots(), fact1.pivots())

    def test_pivot_weights(self) -> None:
        """Pivot weights influence selection order."""
        bkd = self._bkd
        nrows = 5
        A = bkd.asarray(np.random.normal(0, 1, (nrows, nrows)))
        A = A.T @ A

        # Without weights
        fact1 = PivotedCholeskyFactorizer(A, bkd)
        fact1.factorize(nrows)
        fact1.pivots()

        # With weights heavily favoring index 0
        weights = bkd.asarray([100.0, 1.0, 1.0, 1.0, 1.0])
        fact2 = PivotedCholeskyFactorizer(A, bkd)
        fact2.factorize(nrows, pivot_weights=weights)
        pivots2 = fact2.pivots()

        # First pivot should be 0 with heavy weight
        self.assertEqual(int(bkd.to_numpy(pivots2[0:1])[0]), 0)

    def test_too_many_pivots_raises(self) -> None:
        """Requesting more pivots than rows raises ValueError."""
        bkd = self._bkd
        A = bkd.eye(3)
        fact = PivotedCholeskyFactorizer(A, bkd)
        with self.assertRaises(ValueError):
            fact.factorize(4)

    def test_factorize_before_update_raises(self) -> None:
        """Calling update() before factorize() raises."""
        bkd = self._bkd
        A = bkd.eye(3)
        fact = PivotedCholeskyFactorizer(A, bkd)
        with self.assertRaises(RuntimeError):
            fact.update(3)


class TestPivotedCholeskyNumpy(TestPivotedCholesky[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPivotedCholeskyTorch(TestPivotedCholesky[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
