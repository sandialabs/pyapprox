"""Tests for ManifoldScorer.

The manifold construction scores thousands of candidate subspaces, so the
objective is computed by a closed form that avoids materializing the
(c, N) corrected-residual matrix and reads ||R||^2 from precomputed row
energies. These tests pin that fast path to the brute-force definition
that explicitly forms the corrected residual, and pin the batched
candidate sweep to scoring each candidate independently.
"""

import numpy as np
import pytest
from pyapprox.surrogates.reduction.feature_maps import (
    MonomialFeatureMap,
    SparseMonomialFeatureMap,
)
from pyapprox.surrogates.reduction.manifold_scoring import (
    ManifoldScorer,
    center_and_decompose,
)


def _coords(rank, nsnapshots, seed, bkd):
    rng = np.random.RandomState(seed)
    return bkd.array(rng.normal(size=(rank, nsnapshots)))


def _indices(nreduced, bkd, degrees=(2,)):
    """The multi-index set of a monomial map, as greedy_scores wants it."""
    return MonomialFeatureMap(nreduced, bkd, degrees=degrees).indices()


class TestCenterAndDecompose:
    """The SVD factors and the centering convention."""

    def test_factors_reproduce_centered_data(self, bkd):
        data = bkd.array(np.random.RandomState(0).normal(size=(20, 50)))
        centered, mean, phi, svals, psi_t = center_and_decompose(data, bkd)
        bkd.assert_allclose(centered, data - mean)
        reconstructed = bkd.dot(phi * svals, psi_t)
        bkd.assert_allclose(reconstructed, centered, atol=1e-10)

    def test_uncentered_keeps_a_zero_mean(self, bkd):
        data = bkd.array(np.random.RandomState(1).normal(size=(10, 30)))
        centered, mean, _, _, _ = center_and_decompose(
            data, bkd, center=False
        )
        bkd.assert_allclose(mean, bkd.zeros((10, 1)))
        bkd.assert_allclose(centered, data)

    def test_supplied_mean_is_used_as_is(self, bkd):
        data = bkd.array(np.random.RandomState(2).normal(size=(10, 30)))
        supplied = bkd.array(np.arange(10, dtype=float))
        centered, mean, _, _, _ = center_and_decompose(
            data, bkd, mean=supplied
        )
        assert mean.shape == (10, 1)
        bkd.assert_allclose(centered, data - bkd.reshape(supplied, (10, 1)))

    def test_precomputed_svd_is_reused(self, bkd):
        data = bkd.array(np.random.RandomState(3).normal(size=(12, 40)))
        _, mean, phi, svals, psi_t = center_and_decompose(data, bkd)
        again = center_and_decompose(
            data, bkd, precomputed_svd=(phi, svals, psi_t, mean)
        )
        bkd.assert_allclose(again[2], phi)
        bkd.assert_allclose(again[3], svals)


class TestObjectiveMatchesBruteForce:
    """Fast objective == brute-force objective, across configurations."""

    @pytest.mark.parametrize(
        "active_rows", [[0], [0, 1], [2, 5, 7], [0, 3, 4, 9]]
    )
    @pytest.mark.parametrize("degrees", [(2,), (2, 3)])
    @pytest.mark.parametrize("gamma", [0.0, 1e-6, 1e-2])
    def test_match(self, bkd, active_rows, degrees, gamma) -> None:
        coords = _coords(rank=12, nsnapshots=300, seed=0, bkd=bkd)
        scorer = ManifoldScorer(coords, gamma, bkd)
        fmap = MonomialFeatureMap(len(active_rows), bkd, degrees=degrees)

        fast = scorer.objective(active_rows, fmap)
        brute = scorer.objective_bruteforce(active_rows, fmap)
        assert abs(fast - brute) <= 1e-8 * max(abs(brute), 1.0)

    def test_empty_feature_map_is_residual_energy(self, bkd) -> None:
        # With no correction terms the objective is just ||R||^2.
        coords = _coords(rank=10, nsnapshots=200, seed=1, bkd=bkd)
        scorer = ManifoldScorer(coords, 1e-6, bkd)
        empty = SparseMonomialFeatureMap(
            bkd.asarray(np.zeros((2, 0), dtype=int)), bkd
        )
        fast = scorer.objective([0, 1], empty)
        brute = scorer.objective_bruteforce([0, 1], empty)
        assert abs(fast - brute) <= 1e-9 * max(abs(brute), 1.0)

    def test_sparse_anisotropic_feature_map(self, bkd) -> None:
        # The closed form must hold for an arbitrary index set too.
        coords = _coords(rank=12, nsnapshots=300, seed=2, bkd=bkd)
        scorer = ManifoldScorer(coords, 1e-4, bkd)
        indices = bkd.asarray(np.array([[2, 3, 1], [0, 0, 1], [0, 0, 0]]))
        fmap = SparseMonomialFeatureMap(indices, bkd)

        fast = scorer.objective([0, 1, 2], fmap)
        brute = scorer.objective_bruteforce([0, 1, 2], fmap)
        assert abs(fast - brute) <= 1e-8 * max(abs(brute), 1.0)


class TestGreedyScoresMatchesPerCandidate:
    """Batched greedy_scores == scoring each candidate independently.

    ``greedy_scores`` computes, for a fixed selected set, the objective of
    ``selected + [j]`` for every candidate j, reusing the selected-set
    monomial work across candidates. It must return the same value per
    candidate as scoring each trial independently with ``objective``.
    """

    @pytest.mark.parametrize(
        "selected", [[], [3], [1, 4], [0, 2, 7]]
    )
    @pytest.mark.parametrize("degrees", [(2,), (2, 3)])
    @pytest.mark.parametrize("gamma", [0.0, 1e-4])
    def test_batched_matches_loop(
        self, bkd, selected, degrees, gamma
    ) -> None:
        coords = _coords(rank=12, nsnapshots=300, seed=4, bkd=bkd)
        scorer = ManifoldScorer(coords, gamma, bkd)
        candidates = [c for c in range(12) if c not in selected]
        nreduced = len(selected) + 1

        batched = scorer.greedy_scores(
            selected, candidates, _indices(nreduced, bkd, degrees)
        )

        for candidate, score in zip(candidates, batched):
            trial = list(selected) + [candidate]
            fmap = MonomialFeatureMap(len(trial), bkd, degrees=degrees)
            ref = scorer.objective(trial, fmap)
            assert abs(score - ref) <= 1e-8 * max(abs(ref), 1.0), (
                f"candidate {candidate}: batched {score} vs {ref}"
            )

    def test_picks_same_argmin(self, bkd) -> None:
        # The greedy decision, the argmin over candidates, must agree.
        coords = _coords(rank=10, nsnapshots=250, seed=5, bkd=bkd)
        scorer = ManifoldScorer(coords, 1e-6, bkd)
        selected = [2]
        candidates = [c for c in range(10) if c not in selected]
        batched = scorer.greedy_scores(
            selected, candidates, _indices(2, bkd)
        )
        loop = [
            scorer.objective(
                selected + [c], MonomialFeatureMap(2, bkd, degrees=(2,))
            )
            for c in candidates
        ]
        assert int(np.argmin(batched)) == int(np.argmin(loop))

    @pytest.mark.parametrize("degrees", [(2,), (2, 3)])
    def test_larger_stress(self, bkd, degrees) -> None:
        # Bigger rank and candidate pool, deeper selected set.
        coords = _coords(rank=25, nsnapshots=500, seed=6, bkd=bkd)
        scorer = ManifoldScorer(coords, 1e-5, bkd)
        selected = [1, 4, 8, 12]
        candidates = [c for c in range(25) if c not in selected]
        batched = scorer.greedy_scores(
            selected, candidates, _indices(len(selected) + 1, bkd, degrees)
        )
        for candidate, score in zip(candidates, batched):
            ref = scorer.objective(
                selected + [candidate],
                MonomialFeatureMap(len(selected) + 1, bkd, degrees=degrees),
            )
            assert abs(score - ref) <= 1e-7 * max(abs(ref), 1.0)

    @pytest.mark.parametrize("selected", [[], [3], [0, 2, 5]])
    @pytest.mark.parametrize("degrees", [(2,), (2, 3)])
    def test_chunking_matches_single_chunk(
        self, bkd, selected, degrees
    ) -> None:
        # greedy_scores chunks candidates to bound peak memory of the
        # (ncands, p_c, N) feature tensor. Forcing a tiny budget splits
        # even this small problem, exercising the seam without a huge
        # one. Chunked scores must equal the single-chunk path exactly:
        # chunking only reorders an independent per-candidate computation.
        coords = _coords(rank=12, nsnapshots=300, seed=8, bkd=bkd)
        candidates = [c for c in range(12) if c not in selected]
        indices = _indices(len(selected) + 1, bkd, degrees)

        big = ManifoldScorer(coords, 1e-5, bkd)
        big._CHUNK_MEM_BYTES = 10**15
        ref = big.greedy_scores(selected, candidates, indices)

        small = ManifoldScorer(coords, 1e-5, bkd)
        p_c = small._candidate_indices(indices, len(selected)).shape[1]
        if p_c > 0:
            small._CHUNK_MEM_BYTES = 2 * p_c * coords.shape[1] * 8
            assert small._candidate_chunk_size(
                indices, len(selected), len(candidates)
            ) < len(candidates)
        chunked = small.greedy_scores(selected, candidates, indices)

        for candidate, (one, many) in zip(
            candidates, zip(ref, chunked)
        ):
            assert abs(one - many) <= 1e-10 * max(abs(one), 1.0), (
                f"candidate {candidate}: single {one} vs chunked {many}"
            )

    def test_chunk_size_at_least_one(self, bkd) -> None:
        # Even when one candidate exceeds the budget the chunk size must
        # not collapse to zero, which would drop candidates or loop.
        coords = _coords(rank=8, nsnapshots=400, seed=9, bkd=bkd)
        scorer = ManifoldScorer(coords, 1e-6, bkd)
        scorer._CHUNK_MEM_BYTES = 1
        indices = _indices(4, bkd, (2, 3))
        assert scorer._candidate_chunk_size(indices, 3, 5) == 1

    def test_residual_energy_from_row_energies(self, bkd) -> None:
        # ||R||^2 via precomputed row energies must equal the explicit norm.
        coords = _coords(rank=15, nsnapshots=400, seed=3, bkd=bkd)
        scorer = ManifoldScorer(coords, 1e-6, bkd)
        active = [1, 4, 9]
        keep = [a for a in range(15) if a not in active]
        explicit = float(
            bkd.to_numpy(bkd.sum(coords[keep, :] * coords[keep, :]))
        )
        via_rows = float(bkd.to_numpy(scorer._residual_energy(active)))
        assert abs(via_rows - explicit) <= 1e-9 * max(abs(explicit), 1.0)


class TestGreedyScoresWithAnisotropicIndices:
    """Selection is driven by the index set the final fit will use.

    Taking a multi-index set rather than the parameters that generated one
    means an anisotropic set can drive selection, and that the terms used
    to score a trial subspace are the ones that subspace supports.
    """

    def test_anisotropic_index_set_matches_per_candidate(self, bkd) -> None:
        coords = _coords(rank=10, nsnapshots=200, seed=21, bkd=bkd)
        scorer = ManifoldScorer(coords, 1e-5, bkd)
        # Degree->=2 terms over 3 reduced dims: z0^2, z0 z1, z2^2. Not a
        # degree band, so no degrees tuple could express it.
        indices = bkd.asarray(
            np.array([[2, 1, 0], [0, 1, 0], [0, 0, 2]])
        )
        selected = [1, 4]
        candidates = [c for c in range(10) if c not in selected]

        batched = scorer.greedy_scores(selected, candidates, indices)

        for candidate, score in zip(candidates, batched):
            fmap = SparseMonomialFeatureMap(indices, bkd)
            ref = scorer.objective(selected + [candidate], fmap)
            assert abs(score - ref) <= 1e-8 * max(abs(ref), 1.0)

    def test_trial_indices_restrict_to_the_trial_dimension(
        self, bkd
    ) -> None:
        """A trial of dimension a scores with the terms a dims support."""
        coords = _coords(rank=8, nsnapshots=150, seed=22, bkd=bkd)
        scorer = ManifoldScorer(coords, 1e-6, bkd)
        full = _indices(4, bkd, (2,))
        trial = scorer._trial_indices(full, 2)
        # The degree-2 band on 4 variables restricted to 2 is the band on
        # 2 variables: z0^2, z0 z1, z1^2.
        assert trial.shape == (2, 3)

    def test_first_step_has_no_static_block(self, bkd) -> None:
        """With nothing selected each candidate is scored on its own."""
        coords = _coords(rank=6, nsnapshots=120, seed=23, bkd=bkd)
        scorer = ManifoldScorer(coords, 1e-6, bkd)
        candidates = list(range(6))
        batched = scorer.greedy_scores(
            [], candidates, _indices(1, bkd, (2,))
        )
        for candidate, score in zip(candidates, batched):
            ref = scorer.objective(
                [candidate], MonomialFeatureMap(1, bkd, degrees=(2,))
            )
            assert abs(score - ref) <= 1e-8 * max(abs(ref), 1.0)


class TestWeightFitting:
    """The ambient correction fit and its regularization."""

    def test_fit_weights_multi_matches_per_gamma(self, bkd) -> None:
        rng = np.random.RandomState(7)
        centered = bkd.array(rng.normal(size=(40, 200)))
        basis = bkd.array(np.linalg.qr(rng.normal(size=(40, 5)))[0])
        scorer = ManifoldScorer(bkd.dot(basis.T, centered), 1e-6, bkd)
        fmap = MonomialFeatureMap(5, bkd, degrees=(2,))
        gammas = [1e-8, 1e-4, 1e-1, 1.0]

        multi = scorer.fit_weights_multi(centered, basis, fmap, gammas)
        for gamma, weights in zip(gammas, multi):
            ref = scorer.fit_weights(centered, basis, fmap, gamma)
            bkd.assert_allclose(weights, ref, rtol=1e-9, atol=1e-10)

    def test_fit_weights_solves_the_ridge_problem(self, bkd) -> None:
        """W satisfies the normal equations it is defined by."""
        rng = np.random.RandomState(31)
        centered = bkd.array(rng.normal(size=(25, 120)))
        basis = bkd.array(np.linalg.qr(rng.normal(size=(25, 3)))[0])
        scorer = ManifoldScorer(bkd.dot(basis.T, centered), 1e-3, bkd)
        fmap = MonomialFeatureMap(3, bkd, degrees=(2,))

        weights = scorer.fit_weights(centered, basis, fmap)
        z = bkd.dot(basis.T, centered)
        feats = fmap(z)
        residual = centered - bkd.dot(basis, z)
        # (F F^T + gamma I) W^T = F R^T
        lhs = bkd.dot(
            bkd.dot(feats, feats.T)
            + 1e-3 * bkd.eye(feats.shape[0]),
            weights.T,
        )
        bkd.assert_allclose(lhs, bkd.dot(feats, residual.T), atol=1e-8)

    def test_empty_feature_map_gives_empty_weights(self, bkd) -> None:
        rng = np.random.RandomState(32)
        centered = bkd.array(rng.normal(size=(12, 40)))
        basis = bkd.array(np.eye(12)[:, :2])
        scorer = ManifoldScorer(bkd.dot(basis.T, centered), 1e-6, bkd)
        empty = SparseMonomialFeatureMap(
            bkd.asarray(np.zeros((2, 0), dtype=int)), bkd
        )
        assert scorer.fit_weights(centered, basis, empty).shape == (12, 0)


class TestGramCondition:
    """The conditioning diagnostic of the correction fit."""

    def test_matches_definition(self, bkd) -> None:
        rng = np.random.RandomState(12)
        centered = bkd.array(rng.normal(size=(40, 200)))
        basis = bkd.array(np.linalg.qr(rng.normal(size=(40, 4)))[0])
        scorer = ManifoldScorer(bkd.dot(basis.T, centered), 1e-6, bkd)
        fmap = MonomialFeatureMap(4, bkd, degrees=(2, 3))
        for gamma in (1e-8, 1e-3, 1.0):
            got = scorer.gram_condition(centered, basis, fmap, gamma)
            feats = fmap(bkd.dot(basis.T, centered))
            gram = bkd.dot(feats, feats.T) + gamma * bkd.eye(
                feats.shape[0]
            )
            ref = float(bkd.to_numpy(bkd.cond(gram)))
            assert abs(got - ref) <= 1e-6 * max(ref, 1.0)

    def test_decreases_with_gamma(self, bkd) -> None:
        # More regularization cannot make the Gram worse-conditioned.
        rng = np.random.RandomState(13)
        centered = bkd.array(rng.normal(size=(50, 300)))
        basis = bkd.array(np.linalg.qr(rng.normal(size=(50, 5)))[0])
        scorer = ManifoldScorer(bkd.dot(basis.T, centered), 1e-6, bkd)
        fmap = MonomialFeatureMap(5, bkd, degrees=(2, 3))
        conds = [
            scorer.gram_condition(centered, basis, fmap, gamma)
            for gamma in (1e-10, 1e-6, 1e-2, 1.0)
        ]
        for earlier, later in zip(conds, conds[1:]):
            assert later <= earlier + 1e-6 * earlier

    def test_empty_feature_map_condition_is_one(self, bkd) -> None:
        coords = _coords(rank=8, nsnapshots=200, seed=14, bkd=bkd)
        scorer = ManifoldScorer(coords, 1e-6, bkd)
        basis = bkd.array(np.eye(8)[:, :2])
        centered = bkd.array(
            np.random.RandomState(0).normal(size=(8, 200))
        )
        empty = SparseMonomialFeatureMap(
            bkd.asarray(np.zeros((2, 0), dtype=int)), bkd
        )
        assert scorer.gram_condition(centered, basis, empty, 1e-6) == 1.0


class TestSelectGamma:
    """Validation-set selection of the regularization."""

    def _problem(self, bkd, seed=41):
        rng = np.random.RandomState(seed)
        centered = bkd.array(rng.normal(size=(30, 150)))
        validation = bkd.array(rng.normal(size=(30, 40)))
        basis = bkd.array(np.linalg.qr(rng.normal(size=(30, 3)))[0])
        scorer = ManifoldScorer(bkd.dot(basis.T, centered), 1e-6, bkd)
        return scorer, centered, validation, basis

    def test_selects_the_grid_minimizer(self, bkd) -> None:
        scorer, centered, validation, basis = self._problem(bkd)
        fmap = MonomialFeatureMap(3, bkd, degrees=(2,))
        grid = [1e-8, 1e-4, 1e-1, 1.0]

        best, diagnostics = scorer.select_gamma(
            centered, basis, fmap, grid, validation,
            return_diagnostics=True,
        )
        assert best == grid[int(np.argmin(diagnostics["val_err"]))]
        assert diagnostics["gammas"] == grid
        assert len(diagnostics["gram_cond"]) == len(grid)

    def test_single_gamma_grid_returns_it(self, bkd) -> None:
        scorer, centered, validation, basis = self._problem(bkd)
        fmap = MonomialFeatureMap(3, bkd, degrees=(2,))
        best = scorer.select_gamma(
            centered, basis, fmap, [0.5], validation
        )
        assert best == 0.5

    def test_empty_grid_is_rejected(self, bkd) -> None:
        scorer, centered, validation, basis = self._problem(bkd)
        fmap = MonomialFeatureMap(3, bkd, degrees=(2,))
        with pytest.raises(ValueError):
            scorer.select_gamma(centered, basis, fmap, [], validation)
