"""Tests for MassMatrix value objects."""

import numpy as np
import pytest
from pyapprox.ode.mass_matrix import (
    ConstantDenseMassMatrix,
    ConstantSparseMassMatrix,
    IdentityMassMatrix,
    MassMatrixProtocol,
    create_mass_matrix,
)
from scipy.sparse import issparse


class TestIdentityMassMatrix:
    def test_protocol_conformance(self, bkd):
        m = IdentityMassMatrix(3, bkd)
        assert isinstance(m, MassMatrixProtocol)

    def test_is_identity(self, bkd):
        m = IdentityMassMatrix(3, bkd)
        assert m.is_identity()

    def test_apply(self, bkd):
        m = IdentityMassMatrix(3, bkd)
        v = bkd.array([1.0, 2.0, 3.0])
        bkd.assert_allclose(m.apply(v), v)

    def test_solve(self, bkd):
        m = IdentityMassMatrix(3, bkd)
        v = bkd.array([1.0, 2.0, 3.0])
        bkd.assert_allclose(m.solve(v), v)

    def test_apply_transpose(self, bkd):
        m = IdentityMassMatrix(3, bkd)
        v = bkd.array([1.0, 2.0, 3.0])
        bkd.assert_allclose(m.apply_transpose(v), v)

    def test_solve_transpose(self, bkd):
        m = IdentityMassMatrix(3, bkd)
        v = bkd.array([1.0, 2.0, 3.0])
        bkd.assert_allclose(m.solve_transpose(v), v)

    def test_as_matrix(self, bkd):
        m = IdentityMassMatrix(3, bkd)
        bkd.assert_allclose(m.as_matrix(), bkd.eye(3))

    def test_as_matrix_cached(self, bkd):
        m = IdentityMassMatrix(3, bkd)
        mat1 = m.as_matrix()
        mat2 = m.as_matrix()
        assert mat1 is mat2


class TestConstantDenseMassMatrix:
    def _make_spd_matrix(self, bkd, n=3, seed=42):
        rng = np.random.RandomState(seed)
        A = rng.randn(n, n)
        return bkd.array(A @ A.T + np.eye(n))

    def test_protocol_conformance(self, bkd):
        M = self._make_spd_matrix(bkd)
        m = ConstantDenseMassMatrix(M, bkd)
        assert isinstance(m, MassMatrixProtocol)

    def test_is_not_identity(self, bkd):
        M = self._make_spd_matrix(bkd)
        m = ConstantDenseMassMatrix(M, bkd)
        assert not m.is_identity()

    def test_apply(self, bkd):
        M = self._make_spd_matrix(bkd)
        m = ConstantDenseMassMatrix(M, bkd)
        v = bkd.array([1.0, 2.0, 3.0])
        bkd.assert_allclose(m.apply(v), bkd.dot(M, v), rtol=1e-12)

    def test_solve(self, bkd):
        M = self._make_spd_matrix(bkd)
        m = ConstantDenseMassMatrix(M, bkd)
        v = bkd.array([1.0, 2.0, 3.0])
        x = m.solve(v)
        bkd.assert_allclose(bkd.dot(M, x), v, rtol=1e-10)

    def test_apply_transpose(self, bkd):
        M = self._make_spd_matrix(bkd)
        m = ConstantDenseMassMatrix(M, bkd)
        v = bkd.array([1.0, 2.0, 3.0])
        bkd.assert_allclose(m.apply_transpose(v), bkd.dot(M.T, v), rtol=1e-12)

    def test_solve_transpose(self, bkd):
        M = self._make_spd_matrix(bkd)
        m = ConstantDenseMassMatrix(M, bkd)
        v = bkd.array([1.0, 2.0, 3.0])
        x = m.solve_transpose(v)
        bkd.assert_allclose(bkd.dot(M.T, x), v, rtol=1e-10)

    def test_as_matrix(self, bkd):
        M = self._make_spd_matrix(bkd)
        m = ConstantDenseMassMatrix(M, bkd)
        bkd.assert_allclose(m.as_matrix(), M, rtol=1e-14)

    def test_solve_nonsymmetric(self, bkd):
        rng = np.random.RandomState(7)
        A = rng.randn(4, 4)
        A += 5.0 * np.eye(4)
        M = bkd.array(A)
        m = ConstantDenseMassMatrix(M, bkd)
        v = bkd.array(rng.randn(4))
        x = m.solve(v)
        bkd.assert_allclose(bkd.dot(M, x), v, rtol=1e-10)

    def test_solve_transpose_nonsymmetric(self, bkd):
        rng = np.random.RandomState(7)
        A = rng.randn(4, 4)
        A += 5.0 * np.eye(4)
        M = bkd.array(A)
        m = ConstantDenseMassMatrix(M, bkd)
        v = bkd.array(rng.randn(4))
        x = m.solve_transpose(v)
        bkd.assert_allclose(bkd.dot(M.T, x), v, rtol=1e-10)


class TestConstantSparseMassMatrix:
    def _make_sparse_matrix(self, n=5):
        from scipy.sparse import diags

        return diags([1.0, -0.5, -0.5], [0, -1, 1], shape=(n, n), format="csc")

    def test_protocol_conformance(self, numpy_bkd):
        S = self._make_sparse_matrix()
        m = ConstantSparseMassMatrix(S, numpy_bkd)
        assert isinstance(m, MassMatrixProtocol)

    def test_is_not_identity(self, numpy_bkd):
        S = self._make_sparse_matrix()
        m = ConstantSparseMassMatrix(S, numpy_bkd)
        assert not m.is_identity()

    def test_apply(self, numpy_bkd):
        bkd = numpy_bkd
        S = self._make_sparse_matrix()
        m = ConstantSparseMassMatrix(S, bkd)
        v = bkd.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = bkd.array(S @ v)
        bkd.assert_allclose(m.apply(v), expected, rtol=1e-12)

    def test_solve(self, numpy_bkd):
        bkd = numpy_bkd
        S = self._make_sparse_matrix()
        m = ConstantSparseMassMatrix(S, bkd)
        v = bkd.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = m.solve(v)
        bkd.assert_allclose(bkd.array(S @ x), v, rtol=1e-10)

    def test_apply_transpose(self, numpy_bkd):
        bkd = numpy_bkd
        S = self._make_sparse_matrix()
        m = ConstantSparseMassMatrix(S, bkd)
        v = bkd.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = bkd.array(S.T @ v)
        bkd.assert_allclose(m.apply_transpose(v), expected, rtol=1e-12)

    def test_solve_transpose(self, numpy_bkd):
        bkd = numpy_bkd
        S = self._make_sparse_matrix()
        m = ConstantSparseMassMatrix(S, bkd)
        v = bkd.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = m.solve_transpose(v)
        bkd.assert_allclose(bkd.array(S.T @ x), v, rtol=1e-10)

    def test_as_matrix(self, numpy_bkd):
        bkd = numpy_bkd
        S = self._make_sparse_matrix()
        m = ConstantSparseMassMatrix(S, bkd)
        result = m.as_matrix()
        # as_matrix() must not densify the sparse matrix
        assert issparse(result)
        bkd.assert_allclose(
            bkd.array(result.toarray()), bkd.array(S.toarray()), rtol=1e-14
        )


class TestCreateMassMatrix:
    def test_identity_detection(self, bkd):
        eye = bkd.eye(4)
        m = create_mass_matrix(eye, bkd)
        assert isinstance(m, IdentityMassMatrix)
        assert m.is_identity()

    def test_dense_matrix(self, bkd):
        rng = np.random.RandomState(0)
        A = rng.randn(3, 3)
        M = bkd.array(A @ A.T + 2.0 * np.eye(3))
        m = create_mass_matrix(M, bkd)
        assert isinstance(m, ConstantDenseMassMatrix)
        assert not m.is_identity()

    def test_sparse_matrix(self, numpy_bkd):
        from scipy.sparse import eye as speye

        S = speye(4, format="csc")
        m = create_mass_matrix(S, numpy_bkd)
        assert isinstance(m, ConstantSparseMassMatrix)


class TestSingularityDetection:
    def _make_stokes_block_mass(self, nvel=6, npres=3):
        """Stokes-style DAE mass [[M_vel, 0], [0, 0]] as sparse csc."""
        from scipy.sparse import bmat, csr_matrix, diags

        M_vel = diags(
            [2.0 + np.arange(nvel), -0.5 * np.ones(nvel - 1)], [0, -1]
        )
        zero = csr_matrix((npres, npres))
        return bmat(
            [[M_vel, None], [None, zero]], format="csc"
        ), list(range(nvel, nvel + npres))

    def test_identity_not_singular(self, bkd):
        m = IdentityMassMatrix(4, bkd)
        assert not m.is_singular()
        assert m.zero_rows() == []

    def test_dense_nonsingular(self, bkd):
        rng = np.random.RandomState(1)
        A = rng.randn(4, 4)
        m = ConstantDenseMassMatrix(bkd.array(A @ A.T + 2.0 * np.eye(4)), bkd)
        assert not m.is_singular()
        assert m.zero_rows() == []

    def test_dense_zero_rows_detected(self, bkd):
        A = np.diag([1.0, 0.0, 2.0, 0.0])
        m = ConstantDenseMassMatrix(bkd.array(A), bkd)
        assert m.is_singular()
        assert m.zero_rows() == [1, 3]

    def test_dense_singular_constructible_and_applies(self, bkd):
        """Deferred LU: construction and apply() work for a DAE mass."""
        A = np.diag([1.0, 2.0, 0.0])
        m = ConstantDenseMassMatrix(bkd.array(A), bkd)
        v = bkd.array([1.0, 2.0, 3.0])
        bkd.assert_allclose(m.apply(v), bkd.array([1.0, 4.0, 0.0]), rtol=1e-14)

    def test_sparse_nonsingular(self, numpy_bkd):
        from scipy.sparse import diags

        S = diags([1.0, -0.5, -0.5], [0, -1, 1], shape=(5, 5), format="csc")
        m = ConstantSparseMassMatrix(S, numpy_bkd)
        assert not m.is_singular()
        assert m.zero_rows() == []

    def test_stokes_block_mass(self, numpy_bkd):
        """Sparse DAE mass: singular, pressure rows identified, apply works,
        mass-only solve fails."""
        bkd = numpy_bkd
        S, pressure_rows = self._make_stokes_block_mass()
        m = ConstantSparseMassMatrix(S, bkd)

        assert m.is_singular()
        assert m.zero_rows() == pressure_rows

        n = S.shape[0]
        v = bkd.array(np.arange(1.0, n + 1.0))
        result = m.apply(v)
        bkd.assert_allclose(
            result[pressure_rows], bkd.zeros((len(pressure_rows),)), atol=1e-15
        )

        with pytest.raises(RuntimeError):
            m.solve(v)

    def test_zero_rows_returns_copy(self, numpy_bkd):
        """Mutating the returned list must not corrupt the operator."""
        S, pressure_rows = self._make_stokes_block_mass()
        m = ConstantSparseMassMatrix(S, numpy_bkd)
        rows = m.zero_rows()
        rows.append(999)
        assert m.zero_rows() == pressure_rows
