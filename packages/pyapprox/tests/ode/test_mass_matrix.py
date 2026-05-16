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
        bkd.assert_allclose(m.as_matrix(), bkd.array(S.toarray()), rtol=1e-14)


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
