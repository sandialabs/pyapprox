"""Tests for fitting the manifold correction from a source.

The oracle is :class:`ManifoldScorer`: the streaming fit is correct when
it produces the weights that one produces from the same data. Everything
else here exists because agreeing on the answer is not enough --
the point of the module is that it agrees *while holding less*, and a
version that quietly reassembled the dataset would pass every accuracy
test in this file.

So the memory behaviour is tested directly: peak allocation must track
the block budget rather than the dataset, and the gamma grid must not
cost one ambient array per gamma.
"""

import os
import tracemalloc
from typing import Any, List

import numpy as np
import pytest
from pyapprox.surrogates.kle.basis_operator import ArrayBasis
from pyapprox.surrogates.kle.basis_sinks import (
    ArrayBasisSink,
    MemmapBasisSink,
)
from pyapprox.surrogates.kle.snapshot_sources import ArraySnapshotSource
from pyapprox.surrogates.reduction.feature_maps import MonomialFeatureMap
from pyapprox.surrogates.reduction.manifold_scoring import ManifoldScorer
from pyapprox.surrogates.reduction.manifold_streaming import (
    encode_from_source,
    fit_weights_from_source,
    select_gamma_from_source,
)
from pyapprox.surrogates.reduction.monomial_manifold import (
    MonomialManifoldEncoder,
)
from pyapprox.util.backends.protocols import Backend

NSTATES, NSAMPLES, NVALIDATION, NREDUCED = 600, 40, 24, 3
GAMMA = 1e-5


def _data(bkd: Backend, nsamples: int, seed: int) -> Any:
    """Low-rank centered snapshots, as the fit expects."""
    rng = np.random.RandomState(seed)
    left = rng.standard_normal((NSTATES, 8))
    right = rng.standard_normal((8, nsamples))
    raw = left @ np.diag(np.logspace(0, -3, 8)) @ right
    return bkd.array(raw - raw.mean(axis=1, keepdims=True))


def _basis(bkd: Backend) -> Any:
    raw = np.random.RandomState(9).standard_normal(
        (NSTATES, NREDUCED)
    )
    return bkd.array(np.linalg.qr(raw)[0])


def _setup(bkd: Backend) -> Any:
    """Snapshots, validation snapshots, basis, feature map, scorer."""
    centered = _data(bkd, NSAMPLES, 0)
    validation = _data(bkd, NVALIDATION, 5)
    basis = _basis(bkd)
    feature_map = MonomialFeatureMap(NREDUCED, bkd)
    scorer = ManifoldScorer(bkd.dot(basis.T, centered), GAMMA, bkd)
    return centered, validation, basis, feature_map, scorer


class TestEncodingFromASource:
    def test_matches_the_resident_product(self, bkd: Backend) -> None:
        """``V^T S`` accumulated over blocks rather than formed at once."""
        centered, _, basis, _, _ = _setup(bkd)
        bkd.assert_allclose(
            encode_from_source(
                ArraySnapshotSource(centered, bkd), basis, bkd
            ),
            bkd.dot(basis.T, centered),
            atol=1e-12,
        )

    @pytest.mark.parametrize("max_bytes", [8, 2000, 1 << 20])
    def test_is_independent_of_the_block_size(
        self, bkd: Backend, max_bytes: int
    ) -> None:
        centered, _, basis, _, _ = _setup(bkd)
        bkd.assert_allclose(
            encode_from_source(
                ArraySnapshotSource(centered, bkd),
                basis,
                bkd,
                max_bytes=max_bytes,
            ),
            bkd.dot(basis.T, centered),
            atol=1e-12,
        )

    def test_rejects_a_basis_of_the_wrong_height(
        self, bkd: Backend
    ) -> None:
        """A basis from a different mesh than the source."""
        centered, _, _, _, _ = _setup(bkd)
        wrong = bkd.array(
            np.random.RandomState(1).standard_normal(
                (NSTATES + 4, NREDUCED)
            )
        )
        with pytest.raises(ValueError, match="states"):
            encode_from_source(
                ArraySnapshotSource(centered, bkd), wrong, bkd
            )


class TestFittingWeightsFromASource:
    """Against ``ManifoldScorer.fit_weights``, the resident oracle."""

    def test_matches_the_resident_fit(self, bkd: Backend) -> None:
        centered, _, basis, feature_map, scorer = _setup(bkd)
        expected = scorer.fit_weights(
            centered, basis, feature_map, GAMMA
        )
        got = fit_weights_from_source(
            ArraySnapshotSource(centered, bkd),
            basis,
            feature_map,
            GAMMA,
            ArrayBasisSink(NSTATES, feature_map.nterms(), bkd),
            bkd,
        )
        bkd.assert_allclose(got.to_array(), expected, atol=1e-11)

    @pytest.mark.parametrize("max_bytes", [8, 2000, 1 << 20])
    def test_the_block_size_does_not_change_the_weights(
        self, bkd: Backend, max_bytes: int
    ) -> None:
        centered, _, basis, feature_map, scorer = _setup(bkd)
        expected = scorer.fit_weights(
            centered, basis, feature_map, GAMMA
        )
        got = fit_weights_from_source(
            ArraySnapshotSource(centered, bkd),
            basis,
            feature_map,
            GAMMA,
            ArrayBasisSink(NSTATES, feature_map.nterms(), bkd),
            bkd,
            max_bytes=max_bytes,
        )
        bkd.assert_allclose(got.to_array(), expected, atol=1e-11)

    def test_a_precomputed_encoding_gives_the_same_answer(
        self, bkd: Backend
    ) -> None:
        """Passing ``encoded`` saves a read and must change nothing.

        Gamma selection passes it, so a discrepancy here would make the
        selected gamma disagree with a direct fit at that gamma.
        """
        centered, _, basis, feature_map, _ = _setup(bkd)
        source = ArraySnapshotSource(centered, bkd)
        without = fit_weights_from_source(
            source,
            basis,
            feature_map,
            GAMMA,
            ArrayBasisSink(NSTATES, feature_map.nterms(), bkd),
            bkd,
        )
        with_encoded = fit_weights_from_source(
            source,
            basis,
            feature_map,
            GAMMA,
            ArrayBasisSink(NSTATES, feature_map.nterms(), bkd),
            bkd,
            encoded=encode_from_source(source, basis, bkd),
        )
        bkd.assert_allclose(
            with_encoded.to_array(), without.to_array(), atol=0.0
        )

    def test_a_mis_sized_sink_is_rejected(self, bkd: Backend) -> None:
        """Named before any I/O, rather than at the first write."""
        centered, _, basis, feature_map, _ = _setup(bkd)
        with pytest.raises(ValueError, match="terms"):
            fit_weights_from_source(
                ArraySnapshotSource(centered, bkd),
                basis,
                feature_map,
                GAMMA,
                ArrayBasisSink(NSTATES, feature_map.nterms() + 3, bkd),
                bkd,
            )


class TestSelectingGammaFromASource:
    GRID = [1e-8, 1e-6, 1e-3, 1.0]

    def test_picks_the_same_gamma_as_the_resident_selection(
        self, bkd: Backend
    ) -> None:
        centered, validation, basis, feature_map, scorer = _setup(bkd)
        expected, _ = scorer.select_gamma(
            centered,
            basis,
            feature_map,
            self.GRID,
            validation,
            return_diagnostics=True,
        )
        got, _ = select_gamma_from_source(
            ArraySnapshotSource(centered, bkd),
            ArraySnapshotSource(validation, bkd),
            basis,
            feature_map,
            self.GRID,
            bkd,
        )
        assert got == expected

    def test_the_per_gamma_errors_match(self, bkd: Backend) -> None:
        """The scores, not just the winner.

        Agreeing on the argmin can happen by luck when one gamma is far
        better than the rest; agreeing on every score cannot.
        """
        centered, validation, basis, feature_map, scorer = _setup(bkd)
        _, expected = scorer.select_gamma(
            centered,
            basis,
            feature_map,
            self.GRID,
            validation,
            return_diagnostics=True,
        )
        _, got = select_gamma_from_source(
            ArraySnapshotSource(centered, bkd),
            ArraySnapshotSource(validation, bkd),
            basis,
            feature_map,
            self.GRID,
            bkd,
        )
        for mine, theirs in zip(got["val_err"], expected["val_err"]):
            assert abs(mine - theirs) <= 1e-8 * abs(theirs)

    @pytest.mark.parametrize("max_bytes", [8, 2000, 1 << 20])
    def test_the_block_size_does_not_change_the_choice(
        self, bkd: Backend, max_bytes: int
    ) -> None:
        """Training and validation block differently at every budget.

        They hold different numbers of snapshots, so one byte budget
        buys each a different number of rows -- the case the paired walk
        exists for, exercised by varying the budget.
        """
        centered, validation, basis, feature_map, scorer = _setup(bkd)
        expected, _ = scorer.select_gamma(
            centered,
            basis,
            feature_map,
            self.GRID,
            validation,
            return_diagnostics=True,
        )
        got, _ = select_gamma_from_source(
            ArraySnapshotSource(centered, bkd),
            ArraySnapshotSource(validation, bkd),
            basis,
            feature_map,
            self.GRID,
            bkd,
            max_bytes=max_bytes,
        )
        assert got == expected

    def test_an_empty_grid_is_rejected(self, bkd: Backend) -> None:
        centered, validation, basis, feature_map, _ = _setup(bkd)
        with pytest.raises(ValueError, match="must not be empty"):
            select_gamma_from_source(
                ArraySnapshotSource(centered, bkd),
                ArraySnapshotSource(validation, bkd),
                basis,
                feature_map,
                [],
                bkd,
            )

    def test_mismatched_meshes_are_rejected(self, bkd: Backend) -> None:
        """Held-out error compares the two row by row."""
        centered, _, basis, feature_map, _ = _setup(bkd)
        rng = np.random.RandomState(3)
        other = bkd.array(rng.standard_normal((NSTATES + 7, 10)))
        with pytest.raises(ValueError, match="states"):
            select_gamma_from_source(
                ArraySnapshotSource(centered, bkd),
                ArraySnapshotSource(other, bkd),
                basis,
                feature_map,
                self.GRID,
                bkd,
            )


class TestNothingAmbientSizedIsFormed:
    """The property the module exists for, and the one that can rot.

    Every accuracy test above passes against an implementation that
    reads the blocks and concatenates them straight back together --
    which an earlier draft of the paired walk did. Peak allocation is
    the only thing that distinguishes the two, so it is asserted rather
    than assumed.
    """

    GRID = [1e-8, 1e-4, 1.0]

    def _peak_bytes(self, call: Any) -> int:
        tracemalloc.start()
        try:
            call()
            return int(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()

    def test_peak_memory_stays_under_the_block_budget(
        self, numpy_bkd: Backend
    ) -> None:
        """The budget must bound what is held, not merely influence it.

        Asserted against the budget rather than against a larger run,
        because "smaller budget allocates less" is satisfied by an
        implementation that reads blocks and joins them back together --
        it allocates the joined array either way, and the budget still
        changes the intermediate. Only an absolute bound rules that out.

        Numpy only: the torch allocator caches, so tracemalloc does not
        see tensor storage and the comparison would be vacuous.
        """
        bkd = numpy_bkd
        centered, validation, basis, feature_map, _ = _setup(bkd)
        budget = 1 << 12
        peak = self._peak_bytes(
            lambda: select_gamma_from_source(
                ArraySnapshotSource(centered, bkd),
                ArraySnapshotSource(validation, bkd),
                basis,
                feature_map,
                self.GRID,
                bkd,
                max_bytes=budget,
            )
        )
        # A few budgets' worth of working space is expected -- a block
        # of each dataset, the residual formed from one, the weights
        # solved from it. What is ruled out is anything proportional to
        # the dataset, which is orders of magnitude larger.
        assert peak < 16 * budget

    def test_the_gamma_grid_costs_no_ambient_array(
        self, numpy_bkd: Backend
    ) -> None:
        """Peak must not grow in proportion to the grid length.

        The resident selection holds one ``(nstates, p)`` weight matrix
        per gamma; here the gamma loop is inside the block loop, so a
        longer grid buys more small solves rather than more arrays.
        """
        bkd = numpy_bkd
        centered, validation, basis, feature_map, _ = _setup(bkd)
        source = ArraySnapshotSource(centered, bkd)
        validation_source = ArraySnapshotSource(validation, bkd)
        weights_bytes = NSTATES * feature_map.nterms() * 8

        def run(grid: List[float]) -> Any:
            return lambda: select_gamma_from_source(
                source,
                validation_source,
                basis,
                feature_map,
                grid,
                bkd,
                max_bytes=1 << 12,
            )

        one = self._peak_bytes(run([1e-4]))
        eight = self._peak_bytes(run([1e-8 * 10**i for i in range(8)]))
        # Seven more gammas, so seven more ambient weight matrices if
        # they were being held. Allow one, generously, for noise.
        assert eight - one < weights_bytes

    def test_the_fit_never_holds_the_dataset(
        self, numpy_bkd: Backend
    ) -> None:
        """Peak well under one ``(nstates, nsamples)`` array."""
        bkd = numpy_bkd
        centered, _, basis, feature_map, _ = _setup(bkd)
        source = ArraySnapshotSource(centered, bkd)
        dataset_bytes = NSTATES * NSAMPLES * 8
        budget = 1 << 12
        peak = self._peak_bytes(
            lambda: fit_weights_from_source(
                source,
                basis,
                feature_map,
                GAMMA,
                ArrayBasisSink(NSTATES, feature_map.nterms(), bkd),
                bkd,
                max_bytes=budget,
            )
        )
        # The sink is (nstates, p) and is the intended output, so it is
        # excluded; what remains must be bounded by the budget rather
        # than by the dataset, which is what a version reassembling the
        # blocks would need.
        sink_bytes = NSTATES * feature_map.nterms() * 8
        assert peak - sink_bytes < 16 * budget
        assert 16 * budget < dataset_bytes


class TestTheEncoderHoldsOperators:
    """Option 3: array accessors kept, operator accessors added.

    The operator is the primitive -- ``decode`` and ``encode`` contract
    against it, so they stay correct when the matrix is backed by a
    file. The array accessor materializes, which every existing caller
    and every small problem can afford.
    """

    def test_both_accessors_agree(self, bkd: Backend) -> None:
        centered, _, basis, feature_map, scorer = _setup(bkd)
        weights = scorer.fit_weights(
            centered, basis, feature_map, GAMMA
        )
        encoder = MonomialManifoldEncoder(
            basis, weights, feature_map, bkd.zeros((NSTATES, 1)), bkd
        )
        bkd.assert_allclose(
            encoder.weights_operator().to_array(),
            encoder.weights(),
            atol=0.0,
        )
        bkd.assert_allclose(
            encoder.basis_operator().to_array(),
            encoder.basis(),
            atol=0.0,
        )

    def test_an_array_is_accepted_unchanged(self, bkd: Backend) -> None:
        """Existing callers pass arrays and must keep working."""
        centered, _, basis, feature_map, scorer = _setup(bkd)
        weights = scorer.fit_weights(
            centered, basis, feature_map, GAMMA
        )
        encoder = MonomialManifoldEncoder(
            basis, weights, feature_map, bkd.zeros((NSTATES, 1)), bkd
        )
        bkd.assert_allclose(encoder.weights(), weights, atol=0.0)
        bkd.assert_allclose(encoder.basis(), basis, atol=0.0)

    def test_an_operator_backing_decodes_identically(
        self, bkd: Backend
    ) -> None:
        """The point of the change: decode does not care which it holds."""
        centered, _, basis, feature_map, scorer = _setup(bkd)
        weights = scorer.fit_weights(
            centered, basis, feature_map, GAMMA
        )
        mean = bkd.zeros((NSTATES, 1))
        from_arrays = MonomialManifoldEncoder(
            basis, weights, feature_map, mean, bkd
        )
        from_operators = MonomialManifoldEncoder(
            ArrayBasis(basis, bkd),
            ArrayBasis(weights, bkd),
            feature_map,
            mean,
            bkd,
        )
        latents = from_arrays.encode(centered)
        bkd.assert_allclose(
            from_operators.decode(latents),
            from_arrays.decode(latents),
            atol=0.0,
        )
        bkd.assert_allclose(
            from_operators.decode_jacobian(latents[:, :1]),
            from_arrays.decode_jacobian(latents[:, :1]),
            atol=0.0,
        )


class TestFittingAnEncoderFromASource:
    """``fit_from_source``, the streamed counterpart of the builder."""

    def test_matches_a_resident_fit(self, bkd: Backend) -> None:
        centered, _, basis, feature_map, scorer = _setup(bkd)
        mean = bkd.zeros((NSTATES, 1))
        expected = scorer.fit_weights(
            centered, basis, feature_map, GAMMA
        )
        encoder = MonomialManifoldEncoder.fit_from_source(
            ArraySnapshotSource(centered, bkd),
            basis,
            feature_map,
            mean,
            ArrayBasisSink(NSTATES, feature_map.nterms(), bkd),
            bkd,
            gamma=GAMMA,
        )
        bkd.assert_allclose(encoder.weights(), expected, atol=1e-11)

    def test_the_mean_is_removed_per_block(self, bkd: Backend) -> None:
        """A source of raw snapshots plus a mean, never a centered copy.

        Checked against the resident fit on an explicitly centered
        array: the two must agree, which they only do if every block had
        the mean subtracted.
        """
        centered, _, basis, feature_map, _ = _setup(bkd)
        rng = np.random.RandomState(11)
        mean = bkd.array(rng.standard_normal((NSTATES, 1)))
        raw = centered + mean
        scorer = ManifoldScorer(
            bkd.dot(basis.T, centered), GAMMA, bkd
        )
        expected = scorer.fit_weights(
            centered, basis, feature_map, GAMMA
        )
        encoder = MonomialManifoldEncoder.fit_from_source(
            ArraySnapshotSource(raw, bkd),
            basis,
            feature_map,
            mean,
            ArrayBasisSink(NSTATES, feature_map.nterms(), bkd),
            bkd,
            gamma=GAMMA,
        )
        bkd.assert_allclose(encoder.weights(), expected, atol=1e-11)

    def test_a_gamma_grid_selects_the_same_gamma(
        self, bkd: Backend
    ) -> None:
        centered, validation, basis, feature_map, scorer = _setup(bkd)
        grid = [1e-8, 1e-6, 1e-3, 1.0]
        expected = scorer.select_gamma(
            centered, basis, feature_map, grid, validation
        )
        encoder = MonomialManifoldEncoder.fit_from_source(
            ArraySnapshotSource(centered, bkd),
            basis,
            feature_map,
            bkd.zeros((NSTATES, 1)),
            ArrayBasisSink(NSTATES, feature_map.nterms(), bkd),
            bkd,
            gamma_grid=grid,
            validation=ArraySnapshotSource(validation, bkd),
        )
        assert encoder.fit_gamma() == expected

    def test_a_gamma_grid_without_validation_is_rejected(
        self, bkd: Backend
    ) -> None:
        centered, _, basis, feature_map, _ = _setup(bkd)
        with pytest.raises(ValueError, match="requires validation"):
            MonomialManifoldEncoder.fit_from_source(
                ArraySnapshotSource(centered, bkd),
                basis,
                feature_map,
                bkd.zeros((NSTATES, 1)),
                ArrayBasisSink(NSTATES, feature_map.nterms(), bkd),
                bkd,
                gamma_grid=[1e-6, 1e-3],
            )

    def test_a_memmap_backed_encoder_decodes_identically(
        self, bkd: Backend, tmp_path: Any
    ) -> None:
        """The weights live in a file and decode does not notice."""
        centered, _, basis, feature_map, _ = _setup(bkd)
        mean = bkd.zeros((NSTATES, 1))
        resident = MonomialManifoldEncoder.fit_from_source(
            ArraySnapshotSource(centered, bkd),
            basis,
            feature_map,
            mean,
            ArrayBasisSink(NSTATES, feature_map.nterms(), bkd),
            bkd,
            gamma=GAMMA,
        )
        streamed = MonomialManifoldEncoder.fit_from_source(
            ArraySnapshotSource(centered, bkd),
            basis,
            feature_map,
            mean,
            MemmapBasisSink(
                os.path.join(str(tmp_path), "w.dat"),
                NSTATES,
                feature_map.nterms(),
                bkd,
            ),
            bkd,
            gamma=GAMMA,
        )
        latents = resident.encode(centered)
        bkd.assert_allclose(
            streamed.decode(latents),
            resident.decode(latents),
            atol=1e-13,
        )
