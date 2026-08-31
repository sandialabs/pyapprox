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
        decomposition = SVDSnapshotSolver(bkd).solve(snaps, 4)
        va = decomposition.eigenvalues
        basis_a = decomposition.eigenvectors
        decomposition = MethodOfSnapshotsSolver(bkd).solve(snaps, 4)
        vb = decomposition.eigenvalues
        basis_b = decomposition.eigenvectors
        bkd.assert_allclose(va, vb, rtol=1e-10)
        bkd.assert_allclose(basis_a, basis_b, rtol=1e-8, atol=1e-10)

    def test_solvers_agree_under_a_diagonal_metric(self, bkd) -> None:
        """The same metric, reached through both code paths."""
        snaps, w = _snapshots(bkd), _weights(bkd)
        decomposition = SVDSnapshotSolver(bkd).solve(
            snaps, 4, DiagonalInnerProduct(w, bkd)
        )
        va = decomposition.eigenvalues
        basis_a = decomposition.eigenvectors
        decomposition = MethodOfSnapshotsSolver(bkd).solve(
            snaps, 4, MassInnerProduct(_sparse_of(bkd, w), bkd)
        )
        vb = decomposition.eigenvalues
        basis_b = decomposition.eigenvectors
        bkd.assert_allclose(va, vb, rtol=1e-10)
        bkd.assert_allclose(basis_a, basis_b, rtol=1e-8, atol=1e-10)

    def test_euclidean_metric_matches_no_metric(self, bkd) -> None:
        snaps = _snapshots(bkd)
        for solver in (SVDSnapshotSolver(bkd), MethodOfSnapshotsSolver(bkd)):
            bare = solver.solve(snaps, 4)
            explicit = solver.solve(snaps, 4, EuclideanInnerProduct(12, bkd))
            bkd.assert_allclose(
                bare.eigenvalues, explicit.eigenvalues, rtol=1e-12
            )
            bkd.assert_allclose(
                bare.eigenvectors, explicit.eigenvectors, rtol=1e-12
            )
            bkd.assert_allclose(
                bare.coordinates, explicit.coordinates, rtol=1e-12
            )


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
        decomposition = solver_cls(bkd).solve(snaps, 4, metric)
        basis = decomposition.eigenvectors
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
        decomposition = solver_cls(bkd).solve(snaps, 4, metric)
        basis = decomposition.eigenvectors
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
        decomposition = solver_cls(bkd).solve(_snapshots(bkd), 5)
        vals = decomposition.eigenvalues
        assert vals.shape == (5,)
        assert bool(bkd.all_bool(vals >= 0.0))
        assert bool(bkd.all_bool(vals[:-1] >= vals[1:]))

    @pytest.mark.parametrize(
        "solver_cls", [SVDSnapshotSolver, MethodOfSnapshotsSolver]
    )
    def test_shapes(self, bkd, solver_cls) -> None:
        decomposition = solver_cls(bkd).solve(_snapshots(bkd), 3)
        vals = decomposition.eigenvalues
        basis = decomposition.eigenvectors
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
        decomposition = solver_cls(bkd).solve(snaps, 4)
        vals = decomposition.eigenvalues
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


class TestCoordinates:
    """The second factor, and its pairing with the first.

    The coordinates are what makes the decomposition reconstruct the
    data. An eigenvector's sign is free only in isolation: once a second
    factor is paired with it, a flip applied to one and not the other
    leaves two arrays that each look right and together are wrong. These
    tests pin that pairing, because nothing about the shapes or the
    orthonormality would reveal a break.
    """

    @pytest.mark.parametrize(
        "solver_cls", [SVDSnapshotSolver, MethodOfSnapshotsSolver]
    )
    def test_reconstructs_the_snapshots(self, bkd, solver_cls) -> None:
        snaps = _snapshots(bkd)
        decomposition = solver_cls(bkd).solve(snaps)
        bkd.assert_allclose(
            decomposition.eigenvectors @ decomposition.coordinates,
            snaps,
            atol=1e-12,
        )

    @pytest.mark.parametrize(
        "solver_cls", [SVDSnapshotSolver, MethodOfSnapshotsSolver]
    )
    def test_reconstructs_under_a_diagonal_metric(
        self, bkd, solver_cls
    ) -> None:
        # The metric changes the basis but not the identity: the two
        # factors still reproduce the snapshots they came from.
        snaps = _snapshots(bkd)
        metric = DiagonalInnerProduct(_weights(bkd), bkd)
        decomposition = solver_cls(bkd).solve(snaps, None, metric)
        bkd.assert_allclose(
            decomposition.eigenvectors @ decomposition.coordinates,
            snaps,
            atol=1e-12,
        )

    @pytest.mark.parametrize(
        "solver_cls", [SVDSnapshotSolver, MethodOfSnapshotsSolver]
    )
    def test_shape_and_truncation(self, bkd, solver_cls) -> None:
        snaps = _snapshots(bkd)
        decomposition = solver_cls(bkd).solve(snaps, 3)
        assert decomposition.nterms() == 3
        assert decomposition.coordinates.shape == (3, snaps.shape[1])

    @pytest.mark.parametrize(
        "solver_cls", [SVDSnapshotSolver, MethodOfSnapshotsSolver]
    )
    def test_truncated_reconstruction_is_the_best_rank_k(
        self, bkd, solver_cls
    ) -> None:
        # Truncating both factors together gives the leading-rank
        # approximation, whose error is the discarded energy.
        snaps = _snapshots(bkd)
        nterms = 3
        decomposition = solver_cls(bkd).solve(snaps, nterms)
        approximation = (
            decomposition.eigenvectors @ decomposition.coordinates
        )
        residual = bkd.to_numpy(approximation - snaps)
        full = solver_cls(bkd).solve(snaps)
        discarded = bkd.to_numpy(full.eigenvalues)[nterms:].sum()
        assert abs(float((residual**2).sum()) - discarded) < 1e-8

    @pytest.mark.parametrize(
        "solver_cls", [SVDSnapshotSolver, MethodOfSnapshotsSolver]
    )
    def test_coordinates_are_the_projection_onto_the_basis(
        self, bkd, solver_cls
    ) -> None:
        # Equivalent to eigenvectors.T @ snapshots, which is what a
        # caller would otherwise recompute.
        snaps = _snapshots(bkd)
        decomposition = solver_cls(bkd).solve(snaps)
        bkd.assert_allclose(
            decomposition.coordinates,
            bkd.dot(decomposition.eigenvectors.T, snaps),
            atol=1e-12,
        )

    def test_solvers_agree_on_coordinates(self, bkd) -> None:
        snaps = _snapshots(bkd)
        svd = SVDSnapshotSolver(bkd).solve(snaps, 4)
        gram = MethodOfSnapshotsSolver(bkd).solve(snaps, 4)
        bkd.assert_allclose(
            svd.coordinates, gram.coordinates, rtol=1e-8, atol=1e-10
        )

    def test_sign_convention_reaches_both_factors(self, bkd) -> None:
        """The pairing survives the sign fix applied to the basis.

        Taking the raw right factor of an SVD alongside the canonically
        signed left factor is the mistake this guards: on ordinary data
        most columns are flipped, and the resulting reconstruction is
        wrong by an O(1) amount rather than subtly.
        """
        snaps = _snapshots(bkd, nstates=20, nsamples=10)
        decomposition = SVDSnapshotSolver(bkd).solve(snaps)

        raw_left, svals, raw_right = bkd.svd(snaps, full_matrices=False)
        flipped = int(
            np.sum(
                np.sign(
                    np.sum(
                        bkd.to_numpy(decomposition.eigenvectors)
                        * bkd.to_numpy(raw_left),
                        axis=0,
                    )
                )
                < 0
            )
        )
        assert flipped > 0, "no column was flipped; test proves nothing"

        # The signed basis against the unsigned right factor is wrong.
        naive = bkd.to_numpy(
            decomposition.eigenvectors @ (svals[:, None] * raw_right)
        )
        assert np.abs(naive - bkd.to_numpy(snaps)).max() > 1e-3
        # The bundle's own pair is not.
        bkd.assert_allclose(
            decomposition.eigenvectors @ decomposition.coordinates,
            snaps,
            atol=1e-12,
        )
