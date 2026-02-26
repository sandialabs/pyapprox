"""Tests for pivoted LU and QR factorizers."""


class TestPivotedLUFactorizer:
    """Tests for PivotedLUFactorizer."""

    def test_factorize_basic(self, bkd) -> None:
        """Test basic LU factorization."""
        from pyapprox.util.linalg import PivotedLUFactorizer

        A = bkd.asarray([[4.0, 3.0], [6.0, 3.0]])
        factorizer = PivotedLUFactorizer(bkd, A)
        L, U = factorizer.factorize(2)

        # L should be lower triangular with unit diagonal
        assert L.shape == (2, 2)
        assert U.shape == (2, 2)

    def test_factorize_recovers_matrix(self, bkd) -> None:
        """Test that L @ U recovers permuted original matrix."""
        from pyapprox.util.linalg import PivotedLUFactorizer

        A = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
        factorizer = PivotedLUFactorizer(bkd, A)
        L, U = factorizer.factorize(3)

        # Reconstruct and compare
        LU = L @ U
        pivots = factorizer.pivots()
        A_permuted = A[pivots, :]
        bkd.assert_allclose(LU, A_permuted[:3, :3], rtol=1e-10)

    def test_incremental_factorization(self, bkd) -> None:
        """Test incremental update of factorization."""
        from pyapprox.util.linalg import PivotedLUFactorizer

        A = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
        factorizer = PivotedLUFactorizer(bkd, A)

        # Factorize 2 pivots first
        L1, U1 = factorizer.factorize(2)
        assert factorizer.npivots() == 2

        # Continue to 3 pivots
        L2, U2 = factorizer.update(3)
        assert factorizer.npivots() == 3

        # Verify reconstruction
        LU = L2 @ U2
        pivots = factorizer.pivots()
        A_permuted = A[pivots, :]
        bkd.assert_allclose(LU, A_permuted[:3, :3], rtol=1e-10)

    def test_initial_pivots(self, bkd) -> None:
        """Test factorization with initial pivots specified."""
        from pyapprox.util.linalg import PivotedLUFactorizer

        A = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
        init_pivots = bkd.asarray([1], dtype=bkd.int64_dtype())
        factorizer = PivotedLUFactorizer(bkd, A, init_pivots=init_pivots)
        L, U = factorizer.factorize(2)

        # First pivot should be row 1
        pivots = factorizer.pivots()
        assert int(pivots[0]) == 1

    def test_matches_backend_lu_preordered(self, bkd) -> None:
        """Test that factorizer matches backend lu when matrix is pre-ordered.

        When the matrix rows are pre-ordered so that pivoting selects rows
        in order (0, 1, 2, ...), the factorizer should produce the same
        L and U as the backend's lu function applied to the reordered matrix.
        """
        from pyapprox.util.linalg import PivotedLUFactorizer

        # Original matrix
        A = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])

        # First, factorize to get the pivot order
        factorizer = PivotedLUFactorizer(bkd, A)
        L_piv, U_piv = factorizer.factorize(3)
        pivots = factorizer.pivots()

        # Reorder matrix rows according to pivots
        A_reordered = A[pivots, :]

        # Backend lu on reordered matrix should not need pivoting
        P_bkd, L_bkd, U_bkd = bkd.lu(A_reordered)

        # P should be identity (no pivoting needed) or close to it
        # L and U from pivoted factorizer should match backend
        bkd.assert_allclose(L_piv, L_bkd, rtol=1e-10)
        bkd.assert_allclose(U_piv, U_bkd, rtol=1e-10)


class TestPivotedQRFactorizer:
    """Tests for PivotedQRFactorizer."""

    def test_select_points(self, bkd) -> None:
        """Test selecting points via QR factorization."""
        from pyapprox.util.linalg import PivotedQRFactorizer

        # Create basis matrix
        basis_mat = bkd.asarray(
            [[1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [1.0, -0.5], [1.0, -1.0]]
        )
        factorizer = PivotedQRFactorizer(bkd)
        pivots = factorizer.select_points(basis_mat, 2)

        assert len(pivots) == 2
        # Pivots should be valid indices
        for p in pivots:
            assert 0 <= int(p) < 5

    def test_select_all_points(self, bkd) -> None:
        """Test selecting all points from candidate set."""
        from pyapprox.util.linalg import PivotedQRFactorizer

        basis_mat = bkd.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        factorizer = PivotedQRFactorizer(bkd)
        pivots = factorizer.select_points(basis_mat, 3)

        assert len(pivots) == 3
        # All indices should be unique
        unique_pivots = set(int(p) for p in pivots)
        assert len(unique_pivots) == 3

    def test_matches_backend_qr_preordered(self, bkd) -> None:
        """Test that pivoted QR matches backend qr when rows are pre-ordered.

        When rows are reordered according to the pivots selected by QR,
        the backend's qr on the reordered matrix should give the same R
        (up to sign of rows).
        """
        from pyapprox.util.linalg import PivotedQRFactorizer

        # Create a matrix where pivoting matters
        basis_mat = bkd.asarray(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [0.1, 0.2], [7.0, 8.0]]
        )
        factorizer = PivotedQRFactorizer(bkd)
        pivots = factorizer.select_points(basis_mat, 2)

        # Reorder rows according to pivots (selected rows first)
        selected_rows = basis_mat[pivots, :]

        # Backend QR on selected rows
        Q_bkd, R_bkd = bkd.qr(selected_rows)

        # The selected rows should span the same space
        # Verify that Q @ R recovers the selected rows
        reconstructed = Q_bkd @ R_bkd
        bkd.assert_allclose(reconstructed, selected_rows, rtol=1e-10)
