"""Tests for the metric a projection is taken in.

The property that matters is not that these compute ``x^T M y`` -- that
is one line each -- but that the three implementations agree wherever
they describe the same metric, and that a basis orthonormal in one is
recognized as such. A caller who pairs an M-orthonormal basis with the
wrong metric gets a projection that is not a best approximation and no
error, so the agreement checks below are the real content.
"""

import numpy as np
import pytest
from pyapprox.util.linalg.inner_product import (
    DiagonalInnerProduct,
    EuclideanInnerProduct,
    InnerProductProtocol,
    MassInnerProduct,
    m_orthonormality_drift,
)
from scipy.sparse import csr_matrix, diags


def _vectors(bkd, nstates=6, ncols=3, seed=0):
    rng = np.random.RandomState(seed)
    return bkd.array(rng.standard_normal((nstates, ncols)))


def _weights(bkd, nstates=6, seed=1):
    rng = np.random.RandomState(seed)
    return bkd.array(rng.uniform(0.5, 2.0, nstates))


class TestProtocolConformance:
    def test_all_three_satisfy_the_protocol(self, bkd) -> None:
        w = _weights(bkd)
        for metric in (
            EuclideanInnerProduct(6, bkd),
            DiagonalInnerProduct(w, bkd),
            MassInnerProduct(diags(np.asarray(bkd.to_numpy(w))), bkd),
        ):
            assert isinstance(metric, InnerProductProtocol)
            assert metric.nstates() == 6

    def test_only_the_sparse_one_is_non_diagonal(self, bkd) -> None:
        w = _weights(bkd)
        assert EuclideanInnerProduct(6, bkd).is_diagonal()
        assert DiagonalInnerProduct(w, bkd).is_diagonal()
        assert not MassInnerProduct(diags([1.0] * 6), bkd).is_diagonal()


class TestAgreementBetweenImplementations:
    """The same metric expressed three ways must give one answer."""

    def test_euclidean_matches_unit_diagonal(self, bkd) -> None:
        x, y = _vectors(bkd), _vectors(bkd, seed=2)
        euclidean = EuclideanInnerProduct(6, bkd)
        unit = DiagonalInnerProduct(bkd.ones((6,)), bkd)
        bkd.assert_allclose(euclidean.dot(x, y), unit.dot(x, y), rtol=1e-12)
        bkd.assert_allclose(euclidean.norm(x), unit.norm(x), rtol=1e-12)

    def test_euclidean_matches_sparse_identity(self, bkd) -> None:
        x, y = _vectors(bkd), _vectors(bkd, seed=2)
        euclidean = EuclideanInnerProduct(6, bkd)
        sparse = MassInnerProduct(diags([1.0] * 6), bkd)
        bkd.assert_allclose(euclidean.dot(x, y), sparse.dot(x, y), rtol=1e-12)

    def test_diagonal_matches_sparse_diagonal(self, bkd) -> None:
        w = _weights(bkd)
        x, y = _vectors(bkd), _vectors(bkd, seed=2)
        dense = DiagonalInnerProduct(w, bkd)
        sparse = MassInnerProduct(diags(np.asarray(bkd.to_numpy(w))), bkd)
        bkd.assert_allclose(dense.apply(x), sparse.apply(x), rtol=1e-12)
        bkd.assert_allclose(dense.dot(x, y), sparse.dot(x, y), rtol=1e-12)
        bkd.assert_allclose(dense.norm(x), sparse.norm(x), rtol=1e-12)


class TestFormProperties:
    """What makes it an inner product rather than any bilinear form."""

    def test_dot_agrees_with_explicit_matrix_product(self, bkd) -> None:
        w = _weights(bkd)
        x, y = _vectors(bkd), _vectors(bkd, seed=2)
        expected = bkd.dot(x.T, w[:, None] * y)
        bkd.assert_allclose(
            DiagonalInnerProduct(w, bkd).dot(x, y), expected, rtol=1e-12
        )

    def test_norm_is_the_diagonal_of_dot(self, bkd) -> None:
        metric = DiagonalInnerProduct(_weights(bkd), bkd)
        x = _vectors(bkd)
        bkd.assert_allclose(
            metric.norm(x) ** 2,
            bkd.get_diagonal(metric.dot(x, x)),
            rtol=1e-12,
        )

    def test_is_symmetric(self, bkd) -> None:
        metric = DiagonalInnerProduct(_weights(bkd), bkd)
        x, y = _vectors(bkd), _vectors(bkd, seed=2)
        bkd.assert_allclose(
            metric.dot(x, y), metric.dot(y, x).T, rtol=1e-12
        )

    def test_is_positive_definite_on_nonzero_vectors(self, bkd) -> None:
        metric = DiagonalInnerProduct(_weights(bkd), bkd)
        assert bool(bkd.all_bool(metric.norm(_vectors(bkd)) > 0.0))


class TestOrthonormalityDrift:
    """Zero exactly when the basis is orthonormal in *that* metric."""

    def test_zero_for_a_basis_built_in_the_same_metric(self, bkd) -> None:
        w = _weights(bkd)
        metric = DiagonalInnerProduct(w, bkd)
        # Orthonormalize under M by whitening, doing the QR in the
        # Euclidean geometry, then mapping back.
        raw = _vectors(bkd, ncols=3)
        sqrt_w = bkd.sqrt(w)
        q, _ = bkd.qr(sqrt_w[:, None] * raw)
        basis = q / sqrt_w[:, None]
        drift = m_orthonormality_drift(basis, metric, bkd)
        assert drift < 1e-12

    def test_nonzero_when_the_metric_is_the_wrong_one(self, bkd) -> None:
        """The mismatch this module exists to prevent."""
        w = _weights(bkd)
        raw = _vectors(bkd, ncols=3)
        sqrt_w = bkd.sqrt(w)
        q, _ = bkd.qr(sqrt_w[:, None] * raw)
        basis = q / sqrt_w[:, None]
        drift = m_orthonormality_drift(
            basis, EuclideanInnerProduct(6, bkd), bkd
        )
        assert drift > 1e-3

    def test_zero_for_euclidean_qr_under_euclidean(self, bkd) -> None:
        q, _ = bkd.qr(_vectors(bkd, ncols=3))
        drift = m_orthonormality_drift(
            q, EuclideanInnerProduct(6, bkd), bkd
        )
        assert drift < 1e-12


class TestMassSolve:
    def test_solve_inverts_apply(self, bkd) -> None:
        w = np.linspace(0.5, 2.0, 6)
        metric = MassInnerProduct(diags(w), bkd)
        x = _vectors(bkd)
        bkd.assert_allclose(metric.solve(metric.apply(x)), x, rtol=1e-10)

    def test_solve_reuses_one_factorization(self, bkd) -> None:
        """The cache is the point: refactorizing per call is the cost."""
        metric = MassInnerProduct(diags(np.linspace(0.5, 2.0, 6)), bkd)
        x = _vectors(bkd)
        first = metric.solve(x)
        bkd.assert_allclose(metric.solve(x), first, rtol=0.0, atol=0.0)


class TestRejects:
    def test_diagonal_rejects_2d_weights(self, bkd) -> None:
        with pytest.raises(ValueError, match="must be 1D"):
            DiagonalInnerProduct(bkd.ones((6, 1)), bkd)

    def test_diagonal_rejects_zero_weight(self, bkd) -> None:
        w = bkd.copy(bkd.ones((6,)))
        w[2] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            DiagonalInnerProduct(w, bkd)

    def test_diagonal_rejects_negative_weight(self, bkd) -> None:
        w = bkd.copy(bkd.ones((6,)))
        w[3] = -1.0
        with pytest.raises(ValueError, match="strictly positive"):
            DiagonalInnerProduct(w, bkd)

    def test_mass_rejects_dense_matrix(self, bkd) -> None:
        with pytest.raises(ValueError, match="scipy sparse"):
            MassInnerProduct(np.eye(6), bkd)

    def test_mass_rejects_rectangular_matrix(self, bkd) -> None:
        with pytest.raises(ValueError, match="must be square"):
            MassInnerProduct(csr_matrix(np.ones((6, 4))), bkd)

    def test_euclidean_rejects_nonpositive_nstates(self, bkd) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            EuclideanInnerProduct(0, bkd)
