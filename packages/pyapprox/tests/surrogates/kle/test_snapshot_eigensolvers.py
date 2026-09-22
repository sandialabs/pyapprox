"""Tests for extracting a basis from snapshot data.

The two solvers reach the same subspace by different routes -- one
symmetrizes and takes an SVD, the other eigendecomposes a Gram -- so the
assertions that matter are that they agree, and that each produces a
basis orthonormal in the metric it was given rather than in whichever
metric happened to be convenient. A basis that is orthonormal in the
wrong metric still has the right shape and still decodes plausibly; only
an explicit check catches it.
"""

import os

import numpy as np
import pytest
from pyapprox.surrogates.kle.basis_sinks import (
    ArrayBasisSink,
    MemmapBasisSink,
)
from pyapprox.surrogates.kle.snapshot_eigensolvers import (
    MethodOfSnapshotsSolver,
    RandomizedSnapshotSolver,
    SnapshotEigenSolverProtocol,
    SVDSnapshotSolver,
    default_snapshot_eigensolver,
)
from pyapprox.surrogates.kle.snapshot_sources import ArraySnapshotSource
from pyapprox.util.linalg.inner_product import (
    DiagonalInnerProduct,
    EuclideanInnerProduct,
    MassInnerProduct,
    m_orthonormality_drift,
)
from pyapprox.util.linalg.orthonormalize import HouseholderQR
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


def _decaying_snapshots(bkd, nstates=400, nsamples=60, rank=30, seed=0):
    """Snapshots with a spectrum spanning three decades.

    A sketch is accurate exactly when the discarded tail is small, so a
    flat spectrum would test the method at its worst and a rank-deficient
    one at its best. Three decades is the middle case, where the
    approximation is good but not free.
    """
    rng = np.random.RandomState(seed)
    left = rng.standard_normal((nstates, rank))
    right = rng.standard_normal((rank, nsamples))
    scale = np.logspace(0.0, -3.0, rank)
    return bkd.array(left @ np.diag(scale) @ right)


class TestRandomizedSnapshotSolver:
    """The approximate solver, whose basis is sized by the request.

    Tested against the dense SVD rather than against a threshold: the
    question is never whether a sketch hits some absolute accuracy, but
    whether it reaches what an exact solve would have on the same data.
    """

    def test_satisfies_the_protocol(self, bkd) -> None:
        assert isinstance(
            RandomizedSnapshotSolver(bkd), SnapshotEigenSolverProtocol
        )

    def test_basis_is_orthonormal(self, bkd) -> None:
        """The convention every solver here shares."""
        snaps = _decaying_snapshots(bkd)
        basis = RandomizedSnapshotSolver(bkd, seed=0).solve(
            snaps, 8
        ).eigenvectors
        bkd.assert_allclose(
            basis.T @ basis, bkd.eye(8), atol=1e-12
        )

    def test_coordinates_reconstruct_the_projection(self, bkd) -> None:
        """The two factors are consistent, as the bundle promises.

        Weaker than reconstructing the snapshots, which an approximate
        basis cannot do, and the right claim: the coordinates must be
        the components *in the basis returned*, whatever that basis is.
        """
        snaps = _decaying_snapshots(bkd)
        result = RandomizedSnapshotSolver(bkd, seed=0).solve(snaps, 8)
        basis = result.eigenvectors
        bkd.assert_allclose(
            basis @ result.coordinates,
            basis @ (basis.T @ snaps),
            atol=1e-10,
        )

    def test_reaches_the_dense_subspace_with_power_iterations(
        self, bkd
    ) -> None:
        """What the approximation costs, measured against the exact answer.

        Asserted as a ratio to the dense reconstruction error rather
        than as an absolute tolerance, since the achievable error is a
        property of the spectrum and would have to be retuned for any
        other data.
        """
        snaps = _decaying_snapshots(bkd)
        nterms = 8

        def relative_error(basis):
            residual = snaps - basis @ (basis.T @ snaps)
            return float(bkd.norm(residual) / bkd.norm(snaps))

        dense = relative_error(
            SVDSnapshotSolver(bkd).solve(snaps, nterms).eigenvectors
        )
        sketched = relative_error(
            RandomizedSnapshotSolver(
                bkd, noversampling=10, npower_iters=1, seed=0
            )
            .solve(snaps, nterms)
            .eigenvectors
        )
        assert sketched < 1.05 * dense

    def test_power_iterations_improve_a_poor_sketch(self, bkd) -> None:
        """Why the knob exists, asserted rather than documented.

        With no oversampling and no power iterations the sketch is
        deliberately starved, so this compares a bad configuration
        against a better one on identical data and seed. It would pass
        vacuously if both were already converged, hence the starvation.
        """
        snaps = _decaying_snapshots(bkd)
        nterms = 8

        def relative_error(npower):
            basis = (
                RandomizedSnapshotSolver(
                    bkd, noversampling=0, npower_iters=npower, seed=0
                )
                .solve(snaps, nterms)
                .eigenvectors
            )
            residual = snaps - basis @ (basis.T @ snaps)
            return float(bkd.norm(residual) / bkd.norm(snaps))

        assert relative_error(2) < relative_error(0)

    def test_the_seed_makes_the_basis_reproducible(self, bkd) -> None:
        """A randomized method is otherwise different every run."""
        snaps = _decaying_snapshots(bkd)
        first = RandomizedSnapshotSolver(bkd, seed=3).solve(snaps, 6)
        second = RandomizedSnapshotSolver(bkd, seed=3).solve(snaps, 6)
        bkd.assert_allclose(
            first.eigenvectors, second.eigenvectors, atol=0.0
        )

    def test_requires_an_explicit_nterms(self, bkd) -> None:
        """It never forms the spectrum the rank would be read from."""
        with pytest.raises(ValueError, match="explicit nterms"):
            RandomizedSnapshotSolver(bkd).solve(_decaying_snapshots(bkd))

    def test_refuses_a_metric(self, bkd) -> None:
        """Rather than silently returning a Euclidean basis."""
        nstates = 12
        snaps = _decaying_snapshots(
            bkd, nstates=nstates, nsamples=8, rank=6
        )
        metric = DiagonalInnerProduct(
            _weights(bkd, nstates=nstates), bkd
        )
        with pytest.raises(ValueError, match="does not accept a metric"):
            RandomizedSnapshotSolver(bkd).solve(snaps, 4, metric)

    def test_rejects_negative_settings(self, bkd) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            RandomizedSnapshotSolver(bkd, noversampling=-1)
        with pytest.raises(ValueError, match="non-negative"):
            RandomizedSnapshotSolver(bkd, npower_iters=-1)

    def test_a_source_gives_the_same_answer_as_an_array(
        self, bkd
    ) -> None:
        """Reading in pieces is the same computation, not an approximation.

        Asserted exactly rather than to a tolerance: the block loop
        re-associates nothing that the dense path associated
        differently, so any discrepancy would be a dropped or repeated
        term rather than rounding.
        """
        snaps = _decaying_snapshots(bkd)
        from_array = RandomizedSnapshotSolver(bkd, seed=0).solve(snaps, 8)
        from_source = RandomizedSnapshotSolver(bkd, seed=0).solve(
            ArraySnapshotSource(snaps, bkd), 8
        )
        bkd.assert_allclose(
            from_source.eigenvalues, from_array.eigenvalues, atol=0.0
        )
        bkd.assert_allclose(
            from_source.eigenvectors, from_array.eigenvectors, atol=0.0
        )
        bkd.assert_allclose(
            from_source.coordinates, from_array.coordinates, atol=0.0
        )

    def test_a_source_is_validated_like_an_array(self, bkd) -> None:
        """The checks are on the dimensions, so a source reaches them too."""
        source = ArraySnapshotSource(_decaying_snapshots(bkd), bkd)
        with pytest.raises(ValueError, match="exceeds the rank"):
            RandomizedSnapshotSolver(bkd).solve(source, 10_000)

    def test_oversampling_is_clamped_to_the_sample_count(
        self, bkd
    ) -> None:
        """Asking for more directions than exist is a setting, not an error.

        The default oversampling exceeds the sample count for a small
        problem, which a caller should not have to notice.
        """
        snaps = _decaying_snapshots(bkd, nstates=40, nsamples=12, rank=8)
        result = RandomizedSnapshotSolver(
            bkd, noversampling=100, seed=0
        ).solve(snaps, 4)
        assert result.eigenvectors.shape == (40, 4)


class TestTheExactSolversRefuseASource:
    """Neither exact solver can read snapshots in pieces.

    Not a limitation of the implementations but of the algorithms.
    Symmetrizing needs the whole matrix before the SVD, and the Gram
    accumulates over row blocks only when the metric leaves rows
    independent -- which the metric that motivates the method of
    snapshots, an assembled mass matrix, does not.

    The refusal is tested because the alternative is worse than an
    error: a block-wise Gram over a coupled metric returns a basis that
    is orthonormal, plausible, and wrong.
    """

    @pytest.mark.parametrize(
        "solver_cls", [SVDSnapshotSolver, MethodOfSnapshotsSolver]
    )
    def test_refuses_with_a_reason(self, bkd, solver_cls) -> None:
        source = ArraySnapshotSource(_snapshots(bkd), bkd)
        with pytest.raises(TypeError, match="needs the snapshots in memory"):
            solver_cls(bkd).solve(source)

    @pytest.mark.parametrize(
        "solver_cls", [SVDSnapshotSolver, MethodOfSnapshotsSolver]
    )
    def test_names_the_solver_that_can(self, bkd, solver_cls) -> None:
        """A dead end without an alternative is a poor error."""
        source = ArraySnapshotSource(_snapshots(bkd), bkd)
        with pytest.raises(TypeError, match="RandomizedSnapshotSolver"):
            solver_cls(bkd).solve(source)


class TestWritingTheBasisToASink:
    """The randomized solver with its output streamed rather than returned.

    The basis is the second-largest array the decomposition holds -- the
    sketch it comes from is wider by the oversampling -- so sending it
    to a sink lowers the peak without removing the ambient dimension
    from it. What must not change is the answer: the same data through
    ``solve`` and through ``solve_to_sink`` has to give the same basis,
    not merely an equally valid one.

    That is a sharper requirement than it sounds. Both paths sign their
    columns, by *different* rules -- the decomposition's own first-entry
    convention and the package's largest-magnitude one -- and a basis
    differing by a per-column flip reconstructs its own data perfectly
    while disagreeing with anything stored beside it.
    """

    def _solver(self, bkd, **kwargs):
        """Seeded, so the sketch is the same across both paths."""
        return RandomizedSnapshotSolver(
            bkd, noversampling=5, npower_iters=1, seed=7, **kwargs
        )

    def _data(self, bkd, nstates=40, nsamples=16):
        """Low-rank with a decaying spectrum, as snapshots are."""
        rng = np.random.RandomState(0)
        left = rng.standard_normal((nstates, 10))
        right = rng.standard_normal((10, nsamples))
        return bkd.array(
            left @ np.diag(np.logspace(0, -3, 10)) @ right
        )

    def test_the_basis_matches_solve_exactly(self, bkd) -> None:
        """Bit-identical, since both paths do the same arithmetic."""
        snapshots = self._data(bkd)
        expected = self._solver(bkd).solve(snapshots, nterms=4)
        got = self._solver(bkd).solve_to_sink(
            snapshots, 4, ArrayBasisSink(40, 4, bkd)
        )
        bkd.assert_allclose(
            got.basis.to_array(), expected.eigenvectors, atol=0.0
        )

    def test_the_eigenvalues_and_coordinates_match_solve(
        self, bkd
    ) -> None:
        """The coordinates must follow the basis through the signing.

        A flip applied to one and not the other leaves a pair that no
        longer reconstructs the snapshots, which is why this is checked
        beside the basis rather than trusted to follow.
        """
        snapshots = self._data(bkd)
        expected = self._solver(bkd).solve(snapshots, nterms=4)
        got = self._solver(bkd).solve_to_sink(
            snapshots, 4, ArrayBasisSink(40, 4, bkd)
        )
        bkd.assert_allclose(
            got.eigenvalues, expected.eigenvalues, atol=0.0
        )
        bkd.assert_allclose(
            got.coordinates, expected.coordinates, atol=0.0
        )

    def test_the_pair_reconstructs_the_snapshots(self, bkd) -> None:
        """The property the signing could break without changing shapes.

        At the data's full rank, so the residual is roundoff rather than
        the discarded spectrum -- truncating to fewer terms would leave
        an error of the next singular value's size and make the
        tolerance, not the signing, the thing under test.
        """
        snapshots = self._data(bkd)
        got = self._solver(bkd).solve_to_sink(
            snapshots, 10, ArrayBasisSink(40, 10, bkd)
        )
        bkd.assert_allclose(
            got.basis.apply(got.coordinates), snapshots, atol=1e-12
        )

    def test_a_memmap_sink_gives_the_same_basis(self, bkd, tmp_path) -> None:
        """The sink is a destination, not a participant in the answer."""
        snapshots = self._data(bkd)
        resident = self._solver(bkd).solve_to_sink(
            snapshots, 4, ArrayBasisSink(40, 4, bkd)
        )
        streamed = self._solver(bkd).solve_to_sink(
            snapshots,
            4,
            MemmapBasisSink(
                os.path.join(str(tmp_path), "basis.dat"), 40, 4, bkd
            ),
        )
        bkd.assert_allclose(
            streamed.basis.to_array(),
            resident.basis.to_array(),
            atol=0.0,
        )

    def test_a_source_gives_the_same_basis_as_an_array(self, bkd) -> None:
        """Both ends streamed, and the answer still does not move."""
        snapshots = self._data(bkd)
        from_array = self._solver(bkd).solve_to_sink(
            snapshots, 4, ArrayBasisSink(40, 4, bkd)
        )
        from_source = self._solver(bkd).solve_to_sink(
            ArraySnapshotSource(snapshots, bkd),
            4,
            ArrayBasisSink(40, 4, bkd),
        )
        bkd.assert_allclose(
            from_source.basis.to_array(),
            from_array.basis.to_array(),
            atol=0.0,
        )

    @pytest.mark.parametrize("max_bytes", [8, 400, 1 << 20])
    def test_the_block_size_does_not_change_the_basis(
        self, bkd, max_bytes
    ) -> None:
        snapshots = self._data(bkd)
        expected = self._solver(bkd).solve(snapshots, nterms=4)
        got = self._solver(bkd).solve_to_sink(
            snapshots,
            4,
            ArrayBasisSink(40, 4, bkd),
            max_bytes=max_bytes,
        )
        bkd.assert_allclose(
            got.basis.to_array(), expected.eigenvectors, atol=1e-14
        )

    def test_an_injected_orthonormalizer_reaches_the_sketch(
        self, bkd
    ) -> None:
        """The solver forwards it rather than holding it unused.

        Without this the seam would look present on the solver and be
        absent where it matters, since only the decomposition
        orthonormalizes anything.
        """
        calls = []

        class Counting:
            def __init__(self, inner):
                self._inner = inner

            def __call__(self, array):
                calls.append(1)
                return self._inner(array)

        snapshots = self._data(bkd)
        expected = self._solver(bkd).solve(snapshots, nterms=4)
        got = self._solver(
            bkd, orthonormalizer=Counting(HouseholderQR(bkd))
        ).solve(snapshots, nterms=4)
        assert len(calls) == 2
        bkd.assert_allclose(
            got.eigenvectors, expected.eigenvectors, atol=0.0
        )
