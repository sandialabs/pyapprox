"""Tests for the concrete LinearOperatorProtocol implementations."""

import numpy as np
import pytest
from pyapprox.ode.linear_operator import (
    BlockDiagonalLinearOperator,
    LinearOperatorProtocol,
    MatrixOperator,
    SparseMatrixOperator,
)
from scipy.sparse import diags


class TestBlockDiagonalLinearOperator:
    def _make_invertible_blocks(self, bkd, k=3, m=4, seed=42):
        rng = np.random.RandomState(seed)
        blocks_np = np.empty((k, m, m))
        for i in range(k):
            A = rng.randn(m, m)
            blocks_np[i] = A @ A.T + 5.0 * np.eye(m)
        return bkd.array(blocks_np)

    def test_protocol_conformance(self, bkd):
        blocks = self._make_invertible_blocks(bkd, k=2, m=3)
        op = BlockDiagonalLinearOperator(blocks, bkd)
        assert isinstance(op, LinearOperatorProtocol)

    def test_apply(self, bkd):
        k, m = 3, 4
        blocks = self._make_invertible_blocks(bkd, k=k, m=m)
        op = BlockDiagonalLinearOperator(blocks, bkd)
        v = bkd.array(np.random.RandomState(7).randn(k * m))
        full = op.as_matrix()
        bkd.assert_allclose(op.apply(v), bkd.dot(full, v), rtol=1e-12)

    def test_solve(self, bkd):
        k, m = 3, 4
        blocks = self._make_invertible_blocks(bkd, k=k, m=m)
        op = BlockDiagonalLinearOperator(blocks, bkd)
        v = bkd.array(np.random.RandomState(8).randn(k * m))
        x = op.solve(v)
        bkd.assert_allclose(op.apply(x), v, rtol=1e-10)

    def test_apply_transpose(self, bkd):
        k, m = 3, 4
        blocks = self._make_invertible_blocks(bkd, k=k, m=m)
        op = BlockDiagonalLinearOperator(blocks, bkd)
        v = bkd.array(np.random.RandomState(9).randn(k * m))
        full = op.as_matrix()
        bkd.assert_allclose(
            op.apply_transpose(v), bkd.dot(full.T, v), rtol=1e-12
        )

    def test_solve_transpose(self, bkd):
        k, m = 3, 4
        blocks = self._make_invertible_blocks(bkd, k=k, m=m)
        op = BlockDiagonalLinearOperator(blocks, bkd)
        v = bkd.array(np.random.RandomState(10).randn(k * m))
        x = op.solve_transpose(v)
        bkd.assert_allclose(op.apply_transpose(x), v, rtol=1e-10)

    def test_as_matrix_shape_and_structure(self, bkd):
        k, m = 3, 4
        blocks = self._make_invertible_blocks(bkd, k=k, m=m)
        op = BlockDiagonalLinearOperator(blocks, bkd)
        full = op.as_matrix()
        assert full.shape == (k * m, k * m)
        for i in range(k):
            s = i * m
            bkd.assert_allclose(full[s : s + m, s : s + m], blocks[i], rtol=1e-14)
        for i in range(k):
            for j in range(k):
                if i != j:
                    bkd.assert_allclose(
                        full[i * m : (i + 1) * m, j * m : (j + 1) * m],
                        bkd.zeros((m, m)),
                        atol=1e-15,
                    )

    def test_single_block_matches_matrix_operator(self, bkd):
        rng = np.random.RandomState(55)
        m = 5
        A = rng.randn(m, m)
        A = A @ A.T + 3.0 * np.eye(m)
        blocks = bkd.array(A.reshape(1, m, m))
        mat = bkd.array(A)

        op_block = BlockDiagonalLinearOperator(blocks, bkd)
        op_mat = MatrixOperator(mat, bkd)

        v = bkd.array(rng.randn(m))
        bkd.assert_allclose(op_block.apply(v), op_mat.apply(v), rtol=1e-12)
        bkd.assert_allclose(op_block.solve(v), op_mat.solve(v), rtol=1e-10)
        bkd.assert_allclose(
            op_block.apply_transpose(v), op_mat.apply_transpose(v), rtol=1e-12
        )
        bkd.assert_allclose(
            op_block.solve_transpose(v), op_mat.solve_transpose(v), rtol=1e-10
        )

    def test_nonsymmetric_blocks(self, bkd):
        rng = np.random.RandomState(77)
        k, m = 2, 3
        blocks_np = np.empty((k, m, m))
        for i in range(k):
            blocks_np[i] = rng.randn(m, m) + 5.0 * np.eye(m)
        blocks = bkd.array(blocks_np)
        op = BlockDiagonalLinearOperator(blocks, bkd)

        v = bkd.array(rng.randn(k * m))
        x = op.solve(v)
        bkd.assert_allclose(op.apply(x), v, rtol=1e-10)
        xt = op.solve_transpose(v)
        bkd.assert_allclose(op.apply_transpose(xt), v, rtol=1e-10)

    def test_invalid_shape_not_3d(self, bkd):
        with pytest.raises(ValueError, match="3D"):
            BlockDiagonalLinearOperator(bkd.array(np.zeros((3, 3))), bkd)

    def test_invalid_shape_not_square(self, bkd):
        with pytest.raises(ValueError, match="square"):
            BlockDiagonalLinearOperator(bkd.array(np.zeros((2, 3, 4))), bkd)


class TestSparseMatrixOperator:
    def _make_sparse_system(self, n=12, seed=3):
        """SPD tridiagonal sparse matrix and a dense RHS."""
        rng = np.random.RandomState(seed)
        lower = rng.rand(n - 1)
        matrix = diags(
            [lower, 2.0 + np.arange(n), 0.5 * lower], [-1, 0, 1]
        ).tocsr()
        rhs = rng.randn(n)
        return matrix, rhs

    def test_protocol_conformance(self, numpy_bkd):
        matrix, _ = self._make_sparse_system()
        op = SparseMatrixOperator(matrix, numpy_bkd)
        assert isinstance(op, LinearOperatorProtocol)

    def test_solve_matches_dense(self, numpy_bkd):
        matrix, rhs = self._make_sparse_system()
        op = SparseMatrixOperator(matrix, numpy_bkd)
        dense = matrix.toarray()
        numpy_bkd.assert_allclose(
            op.solve(numpy_bkd.asarray(rhs)),
            numpy_bkd.asarray(np.linalg.solve(dense, rhs)),
            rtol=1e-12,
        )

    def test_solve_transpose_matches_dense(self, numpy_bkd):
        matrix, rhs = self._make_sparse_system()
        op = SparseMatrixOperator(matrix, numpy_bkd)
        dense = matrix.toarray()
        numpy_bkd.assert_allclose(
            op.solve_transpose(numpy_bkd.asarray(rhs)),
            numpy_bkd.asarray(np.linalg.solve(dense.T, rhs)),
            rtol=1e-12,
        )

    def test_apply_and_transpose_match_dense(self, numpy_bkd):
        matrix, rhs = self._make_sparse_system()
        op = SparseMatrixOperator(matrix, numpy_bkd)
        dense = matrix.toarray()
        v = numpy_bkd.asarray(rhs)
        numpy_bkd.assert_allclose(op.apply(v), dense @ rhs, rtol=1e-12)
        numpy_bkd.assert_allclose(
            op.apply_transpose(v), dense.T @ rhs, rtol=1e-12
        )
        numpy_bkd.assert_allclose(op.as_matrix(), dense, rtol=1e-15)

    def test_factorization_cached_and_shared(self, numpy_bkd, monkeypatch):
        """One splu call serves solve and solve_transpose alike."""
        import pyapprox.ode.linear_operator as lin_op_module

        matrix, rhs = self._make_sparse_system()
        op = SparseMatrixOperator(matrix, numpy_bkd)

        calls = []
        real_splu = lin_op_module.splu

        def counting_splu(*args, **kwargs):
            calls.append(1)
            return real_splu(*args, **kwargs)

        monkeypatch.setattr(lin_op_module, "splu", counting_splu)
        v = numpy_bkd.asarray(rhs)
        op.solve(v)
        op.solve(v)
        op.solve_transpose(v)
        assert len(calls) == 1

    def test_rejects_dense_matrix(self, numpy_bkd):
        with pytest.raises(TypeError, match="sparse"):
            SparseMatrixOperator(np.eye(3), numpy_bkd)

    def test_rejects_nonsquare(self, numpy_bkd):
        matrix, _ = self._make_sparse_system()
        with pytest.raises(ValueError, match="square"):
            SparseMatrixOperator(matrix[:, :-1], numpy_bkd)

    def test_rejects_torch_backend(self, torch_bkd):
        """Sparse factorization would break autograd; constructor raises."""
        matrix, _ = self._make_sparse_system()
        with pytest.raises(TypeError, match="NumpyBkd"):
            SparseMatrixOperator(matrix, torch_bkd)
