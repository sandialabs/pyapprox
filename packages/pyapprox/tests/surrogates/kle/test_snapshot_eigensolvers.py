"""Tests for extracting a basis from snapshot data.

The two solvers reach the same subspace by different routes -- one
symmetrizes and takes an SVD, the other eigendecomposes a Gram -- so the
assertions that matter are that they agree, and that each produces a
basis orthonormal in the metric it was given rather than in whichever
metric happened to be convenient. A basis that is orthonormal in the
wrong metric still has the right shape and still decodes plausibly; only
an explicit check catches it.
"""

import numpy as np
import pytest
from pyapprox.surrogates.kle.snapshot_eigensolvers import (
    MethodOfSnapshotsSolver,
    SnapshotEigenSolverProtocol,
    SVDSnapshotSolver,
    default_snapshot_eigensolver,
)
from pyapprox.util.linalg.inner_product import (
    DiagonalInnerProduct,
    EuclideanInnerProduct,
    MassInnerProduct,
    m_orthonormality_drift,
)
from scipy.sparse import diags


def _snapshots(bkd, nstates=12, nsamples=8, seed=0):
    rng = np.random.RandomState(seed)
    return bkd.array(rng.standard_normal((nstates, nsamples)))


def _weights(bkd, nstates=12, seed=1):
    rng = np.random.RandomState(seed)
    return bkd.array(rng.uniform(0.5, 2.0, nstates))


def _sparse_of(bkd, weights):
    return diags(np.asarray(bkd.to_numpy(weights)))


class TestProtocolConformance:
    def test_both_satisfy_the_protocol(self, bkd) -> None:
        for solver in (SVDSnapshotSolver(bkd), MethodOfSnapshotsSolver(bkd)):
            assert isinstance(solver, SnapshotEigenSolverProtocol)

    def test_default_follows_the_metric_not_the_caller(self, bkd) -> None:
        w = _weights(bkd)
        assert isinstance(
            default_snapshot_eigensolver(bkd, None), SVDSnapshotSolver
        )
        assert isinstance(
            default_snapshot_eigensolver(bkd, DiagonalInnerProduct(w, bkd)),
            SVDSnapshotSolver,
        )
        assert isinstance(
            default_snapshot_eigensolver(
                bkd, MassInnerProduct(_sparse_of(bkd, w), bkd)
            ),
            MethodOfSnapshotsSolver,
        )


class TestAgreement:
    """Two routes to one subspace."""

    def test_solvers_agree_without_a_metric(self, bkd) -> None:
        snaps = _snapshots(bkd)
        va, basis_a = SVDSnapshotSolver(bkd).solve(snaps, 4)
        vb, basis_b = MethodOfSnapshotsSolver(bkd).solve(snaps, 4)
        bkd.assert_allclose(va, vb, rtol=1e-10)
        bkd.assert_allclose(basis_a, basis_b, rtol=1e-8, atol=1e-10)

    def test_solvers_agree_under_a_diagonal_metric(self, bkd) -> None:
        """The same metric, reached through both code paths."""
        snaps, w = _snapshots(bkd), _weights(bkd)
        va, basis_a = SVDSnapshotSolver(bkd).solve(
            snaps, 4, DiagonalInnerProduct(w, bkd)
        )
        vb, basis_b = MethodOfSnapshotsSolver(bkd).solve(
            snaps, 4, MassInnerProduct(_sparse_of(bkd, w), bkd)
        )
        bkd.assert_allclose(va, vb, rtol=1e-10)
        bkd.assert_allclose(basis_a, basis_b, rtol=1e-8, atol=1e-10)

    def test_euclidean_metric_matches_no_metric(self, bkd) -> None:
        snaps = _snapshots(bkd)
        for solver in (SVDSnapshotSolver(bkd), MethodOfSnapshotsSolver(bkd)):
            bare = solver.solve(snaps, 4)
            explicit = solver.solve(snaps, 4, EuclideanInnerProduct(12, bkd))
            bkd.assert_allclose(bare[0], explicit[0], rtol=1e-12)
            bkd.assert_allclose(bare[1], explicit[1], rtol=1e-12)


class TestConvention:
    """What every solver must establish before returning."""

    @pytest.mark.parametrize(
        "solver_cls", [SVDSnapshotSolver, MethodOfSnapshotsSolver]
    )
    def test_basis_is_orthonormal_in_the_given_metric(
        self, bkd, solver_cls
    ) -> None:
        snaps, w = _snapshots(bkd), _weights(bkd)
        metric = (
            DiagonalInnerProduct(w, bkd)
            if solver_cls is SVDSnapshotSolver
            else MassInnerProduct(_sparse_of(bkd, w), bkd)
        )
        _, basis = solver_cls(bkd).solve(snaps, 4, metric)
        assert m_orthonormality_drift(basis, metric, bkd) < 1e-10

    @pytest.mark.parametrize(
        "solver_cls", [SVDSnapshotSolver, MethodOfSnapshotsSolver]
    )
    def test_weighted_basis_is_not_euclidean_orthonormal(
        self, bkd, solver_cls
    ) -> None:
        """The mismatch a shape check cannot see."""
        snaps, w = _snapshots(bkd), _weights(bkd)
        metric = (
            DiagonalInnerProduct(w, bkd)
            if solver_cls is SVDSnapshotSolver
            else MassInnerProduct(_sparse_of(bkd, w), bkd)
        )
        _, basis = solver_cls(bkd).solve(snaps, 4, metric)
        drift = m_orthonormality_drift(
            basis, EuclideanInnerProduct(12, bkd), bkd
        )
        assert drift > 1e-3

    @pytest.mark.parametrize(
        "solver_cls", [SVDSnapshotSolver, MethodOfSnapshotsSolver]
    )
    def test_eigenvalues_descend_and_are_nonnegative(
        self, bkd, solver_cls
    ) -> None:
        vals, _ = solver_cls(bkd).solve(_snapshots(bkd), 5)
        assert vals.shape == (5,)
        assert bool(bkd.all_bool(vals >= 0.0))
        assert bool(bkd.all_bool(vals[:-1] >= vals[1:]))

    @pytest.mark.parametrize(
        "solver_cls", [SVDSnapshotSolver, MethodOfSnapshotsSolver]
    )
    def test_shapes(self, bkd, solver_cls) -> None:
        vals, basis = solver_cls(bkd).solve(_snapshots(bkd), 3)
        assert vals.shape == (3,)
        assert basis.shape == (12, 3)

    @pytest.mark.parametrize(
        "solver_cls", [SVDSnapshotSolver, MethodOfSnapshotsSolver]
    )
    def test_eigenvalues_reproduce_the_covariance(
        self, bkd, solver_cls
    ) -> None:
        """The spectrum is that of S S^T, undivided by any sample count."""
        snaps = _snapshots(bkd)
        vals, _ = solver_cls(bkd).solve(snaps, 4)
        reference = bkd.eigh(bkd.dot(snaps, snaps.T))[0]
        expected = bkd.flip(reference, axis=(0,))[:4]
        bkd.assert_allclose(vals, expected, rtol=1e-8)


class TestRejects:
    @pytest.mark.parametrize(
        "solver_cls", [SVDSnapshotSolver, MethodOfSnapshotsSolver]
    )
    def test_rejects_1d_snapshots(self, bkd, solver_cls) -> None:
        with pytest.raises(ValueError, match="must be 2D"):
            solver_cls(bkd).solve(bkd.ones((12,)), 2)

    @pytest.mark.parametrize(
        "solver_cls", [SVDSnapshotSolver, MethodOfSnapshotsSolver]
    )
    def test_rejects_more_terms_than_rank(self, bkd, solver_cls) -> None:
        with pytest.raises(ValueError, match="rank of the snapshot"):
            solver_cls(bkd).solve(_snapshots(bkd), 9)

    @pytest.mark.parametrize(
        "solver_cls", [SVDSnapshotSolver, MethodOfSnapshotsSolver]
    )
    def test_rejects_nonpositive_terms(self, bkd, solver_cls) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            solver_cls(bkd).solve(_snapshots(bkd), 0)

    @pytest.mark.parametrize(
        "solver_cls", [SVDSnapshotSolver, MethodOfSnapshotsSolver]
    )
    def test_rejects_metric_of_the_wrong_size(self, bkd, solver_cls) -> None:
        with pytest.raises(ValueError, match="defined on"):
            solver_cls(bkd).solve(
                _snapshots(bkd), 2, EuclideanInnerProduct(5, bkd)
            )

    def test_svd_refuses_a_non_diagonal_metric(self, bkd) -> None:
        """Refusing beats densifying, and the message names the fix."""
        metric = MassInnerProduct(_sparse_of(bkd, _weights(bkd)), bkd)
        with pytest.raises(ValueError, match="MethodOfSnapshotsSolver"):
            SVDSnapshotSolver(bkd).solve(_snapshots(bkd), 4, metric)
