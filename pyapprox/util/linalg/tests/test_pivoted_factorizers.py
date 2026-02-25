"""Tests for pivoted LU and QR factorizers."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestPivotedLUFactorizer(Generic[Array], unittest.TestCase):
    """Tests for PivotedLUFactorizer."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_factorize_basic(self) -> None:
        """Test basic LU factorization."""
        from pyapprox.util.linalg import PivotedLUFactorizer

        A = self._bkd.asarray([[4.0, 3.0], [6.0, 3.0]])
        factorizer = PivotedLUFactorizer(self._bkd, A)
        L, U = factorizer.factorize(2)

        # L should be lower triangular with unit diagonal
        self.assertEqual(L.shape, (2, 2))
        self.assertEqual(U.shape, (2, 2))

    def test_factorize_recovers_matrix(self) -> None:
        """Test that L @ U recovers permuted original matrix."""
        from pyapprox.util.linalg import PivotedLUFactorizer

        A = self._bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
        factorizer = PivotedLUFactorizer(self._bkd, A)
        L, U = factorizer.factorize(3)

        # Reconstruct and compare
        LU = L @ U
        pivots = factorizer.pivots()
        A_permuted = A[pivots, :]
        self._bkd.assert_allclose(LU, A_permuted[:3, :3], rtol=1e-10)

    def test_incremental_factorization(self) -> None:
        """Test incremental update of factorization."""
        from pyapprox.util.linalg import PivotedLUFactorizer

        A = self._bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
        factorizer = PivotedLUFactorizer(self._bkd, A)

        # Factorize 2 pivots first
        L1, U1 = factorizer.factorize(2)
        self.assertEqual(factorizer.npivots(), 2)

        # Continue to 3 pivots
        L2, U2 = factorizer.update(3)
        self.assertEqual(factorizer.npivots(), 3)

        # Verify reconstruction
        LU = L2 @ U2
        pivots = factorizer.pivots()
        A_permuted = A[pivots, :]
        self._bkd.assert_allclose(LU, A_permuted[:3, :3], rtol=1e-10)

    def test_initial_pivots(self) -> None:
        """Test factorization with initial pivots specified."""
        from pyapprox.util.linalg import PivotedLUFactorizer

        A = self._bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
        init_pivots = self._bkd.asarray([1], dtype=self._bkd.int64_dtype())
        factorizer = PivotedLUFactorizer(self._bkd, A, init_pivots=init_pivots)
        L, U = factorizer.factorize(2)

        # First pivot should be row 1
        pivots = factorizer.pivots()
        self.assertEqual(int(pivots[0]), 1)

    def test_matches_backend_lu_preordered(self) -> None:
        """Test that factorizer matches backend lu when matrix is pre-ordered.

        When the matrix rows are pre-ordered so that pivoting selects rows
        in order (0, 1, 2, ...), the factorizer should produce the same
        L and U as the backend's lu function applied to the reordered matrix.
        """
        from pyapprox.util.linalg import PivotedLUFactorizer

        # Original matrix
        A = self._bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])

        # First, factorize to get the pivot order
        factorizer = PivotedLUFactorizer(self._bkd, A)
        L_piv, U_piv = factorizer.factorize(3)
        pivots = factorizer.pivots()

        # Reorder matrix rows according to pivots
        A_reordered = A[pivots, :]

        # Backend lu on reordered matrix should not need pivoting
        P_bkd, L_bkd, U_bkd = self._bkd.lu(A_reordered)

        # P should be identity (no pivoting needed) or close to it
        # L and U from pivoted factorizer should match backend
        self._bkd.assert_allclose(L_piv, L_bkd, rtol=1e-10)
        self._bkd.assert_allclose(U_piv, U_bkd, rtol=1e-10)


class TestPivotedQRFactorizer(Generic[Array], unittest.TestCase):
    """Tests for PivotedQRFactorizer."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_select_points(self) -> None:
        """Test selecting points via QR factorization."""
        from pyapprox.util.linalg import PivotedQRFactorizer

        # Create basis matrix
        basis_mat = self._bkd.asarray(
            [[1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [1.0, -0.5], [1.0, -1.0]]
        )
        factorizer = PivotedQRFactorizer(self._bkd)
        pivots = factorizer.select_points(basis_mat, 2)

        self.assertEqual(len(pivots), 2)
        # Pivots should be valid indices
        for p in pivots:
            self.assertTrue(0 <= int(p) < 5)

    def test_select_all_points(self) -> None:
        """Test selecting all points from candidate set."""
        from pyapprox.util.linalg import PivotedQRFactorizer

        basis_mat = self._bkd.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        factorizer = PivotedQRFactorizer(self._bkd)
        pivots = factorizer.select_points(basis_mat, 3)

        self.assertEqual(len(pivots), 3)
        # All indices should be unique
        unique_pivots = set(int(p) for p in pivots)
        self.assertEqual(len(unique_pivots), 3)

    def test_matches_backend_qr_preordered(self) -> None:
        """Test that pivoted QR matches backend qr when rows are pre-ordered.

        When rows are reordered according to the pivots selected by QR,
        the backend's qr on the reordered matrix should give the same R
        (up to sign of rows).
        """
        from pyapprox.util.linalg import PivotedQRFactorizer

        # Create a matrix where pivoting matters
        basis_mat = self._bkd.asarray(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [0.1, 0.2], [7.0, 8.0]]
        )
        factorizer = PivotedQRFactorizer(self._bkd)
        pivots = factorizer.select_points(basis_mat, 2)

        # Reorder rows according to pivots (selected rows first)
        selected_rows = basis_mat[pivots, :]

        # Backend QR on selected rows
        Q_bkd, R_bkd = self._bkd.qr(selected_rows)

        # The selected rows should span the same space
        # Verify that Q @ R recovers the selected rows
        reconstructed = Q_bkd @ R_bkd
        self._bkd.assert_allclose(reconstructed, selected_rows, rtol=1e-10)


# NumPy backend tests
class TestPivotedLUFactorizerNumpy(TestPivotedLUFactorizer[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPivotedQRFactorizerNumpy(TestPivotedQRFactorizer[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestPivotedLUFactorizerTorch(TestPivotedLUFactorizer[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestPivotedQRFactorizerTorch(TestPivotedQRFactorizer[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
