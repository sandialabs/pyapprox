"""Tests for error indicators used in adaptive sparse grid refinement.

An indicator receives a candidate carrying its backward box and returns
an error. Cost is not applied here; the fitter turns errors into queue
order through a PriorityProtocol.

The helper builds a downward-closed selected set, registers a candidate
against it, and assembles the box from IncrementalSmolyakCoefficients,
mirroring what the fitter does. Where an indicator's value has a closed
form it is checked against that, and the box-based errors are also
checked against the surrogate difference they stand in for.

Tests run on both NumPy and PyTorch backends.
"""

from typing import List, Tuple

import pytest
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import LinearGrowthRule
from pyapprox.surrogates.sparsegrids.basis_factory import (
    GaussLagrangeFactory,
)
from pyapprox.surrogates.sparsegrids.candidate_info import Candidate
from pyapprox.surrogates.sparsegrids.combination_surrogate import (
    CombinationSurrogate,
)
from pyapprox.surrogates.sparsegrids.error_indicators import (
    ErrorIndicatorProtocol,
    L2GlobalSurplusIndicator,
    L2SurplusIndicator,
    SummedSubspaceVarianceIndicator,
)
from pyapprox.surrogates.sparsegrids.sample_tracker import SampleTracker
from pyapprox.surrogates.sparsegrids.smolyak import (
    IncrementalSmolyakCoefficients,
    compute_smolyak_coefficients,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)


def _level_set(nvars: int, level: int) -> List[Tuple[int, ...]]:
    """Total-degree index set {k : sum(k) <= level}, in level order."""
    keys: List[Tuple[int, ...]] = []
    for total in range(level + 1):
        for key in _compositions(nvars, total):
            keys.append(key)
    return keys


def _compositions(nvars: int, total: int) -> List[Tuple[int, ...]]:
    """All non-negative integer tuples of length nvars summing to total."""
    if nvars == 1:
        return [(total,)]
    out: List[Tuple[int, ...]] = []
    for first in range(total + 1):
        for rest in _compositions(nvars - 1, total - first):
            out.append((first,) + rest)
    return out


class _Grid:
    """Minimal stand-in for the fitter's read-only view.

    Supplies get_samples, which is all L2GlobalSurplusIndicator needs.
    """

    def __init__(self, samples) -> None:
        self._samples = samples

    def get_samples(self, subset: str = "all"):
        return self._samples


def _build_candidate(
    bkd,
    nvars: int,
    selected_level: int,
    candidate_index_tuple: Tuple[int, ...],
    target_fn,
    cost: float = 1.0,
):
    """Build a candidate, its grid view, and the two surrogates.

    Returns
    -------
    candidate : Candidate
        With its backward box assembled from the Smolyak coefficients.
    grid : _Grid
        Read-only view carrying every sample in the grid.
    selected : CombinationSurrogate
        Built from the selected set alone.
    sel_plus : CombinationSurrogate
        Built from the selected set plus the candidate.
    """
    marginal = UniformMarginal(-1.0, 1.0, bkd)
    factories = [GaussLagrangeFactory(marginal, bkd)] * nvars
    tp_factory = TensorProductSubspaceFactory(
        bkd, factories, LinearGrowthRule(scale=1, shift=1)
    )

    selected_keys = _level_set(nvars, selected_level)
    candidate_key = tuple(candidate_index_tuple)

    smolyak = IncrementalSmolyakCoefficients(nvars)
    for key in selected_keys:
        smolyak.add(key)

    # One subspace per key, registered so the tracker can dedupe and
    # then hand every subspace its values.
    tracker = SampleTracker(bkd, tp_factory)
    subspaces = {}
    positions = {}
    for key in selected_keys + [candidate_key]:
        idx = bkd.asarray(list(key), dtype=bkd.int64_dtype())
        subspace = tp_factory(idx)
        subspaces[key] = subspace
        positions[key] = tracker.register(idx, subspace)

    all_samples = tracker.collect_unique_samples()
    tracker.append_new_values(target_fn(all_samples))
    tracker.distribute_values_to_subspaces()

    box = [
        (sign, subspaces[key])
        for key, sign in smolyak.delta(candidate_key)
    ]
    candidate = Candidate(
        index=bkd.asarray(list(candidate_key), dtype=bkd.int64_dtype()),
        subspace=subspaces[candidate_key],
        box=box,
        new_sample_local_indices=tracker.get_unique_local_indices(
            positions[candidate_key]
        ),
        config_idx=None,
        cost=cost,
    )

    def surrogate(keys):
        indices = bkd.asarray(
            [[k[d] for k in keys] for d in range(nvars)],
            dtype=bkd.int64_dtype(),
        )
        coefs = compute_smolyak_coefficients(indices, bkd)
        return CombinationSurrogate(
            bkd,
            nvars,
            [subspaces[k] for k in keys],
            coefs,
            1,
            indices=indices,
        )

    return (
        candidate,
        _Grid(all_samples),
        surrogate(selected_keys),
        surrogate(selected_keys + [candidate_key]),
    )


class TestIndicatorsSatisfyProtocol:
    """Each indicator is usable where the protocol is required."""

    @pytest.mark.parametrize(
        "indicator_cls",
        [L2SurplusIndicator, L2GlobalSurplusIndicator, SummedSubspaceVarianceIndicator],
    )
    def test_isinstance(self, bkd, indicator_cls) -> None:
        assert isinstance(indicator_cls(bkd), ErrorIndicatorProtocol)


class TestL2GlobalSurplus:
    """RMS surplus over every sample in the grid."""

    def test_zero_for_exactly_represented_function(self, bkd) -> None:
        """A linear target is exact at level 1, so adding nothing moves."""

        def target_fn(samples):
            return bkd.reshape(samples[0, :] + samples[1, :], (1, -1))

        candidate, grid, _, _ = _build_candidate(
            bkd, 2, 1, (2, 0), target_fn
        )
        assert L2GlobalSurplusIndicator(bkd)(candidate, grid) < 1e-10

    @pytest.mark.slow_on("TorchBkd")
    def test_nonzero_for_underresolved_function(self, bkd) -> None:
        def target_fn(samples):
            x, y = samples[0, :], samples[1, :]
            return bkd.reshape(x**4 + y**4, (1, -1))

        candidate, grid, _, _ = _build_candidate(
            bkd, 2, 1, (2, 0), target_fn
        )
        assert L2GlobalSurplusIndicator(bkd)(candidate, grid) > 0

    def test_matches_surrogate_difference(self, bkd) -> None:
        """The box sum stands in for I_{K+k} - I_K; check it does."""

        def target_fn(samples):
            x, y = samples[0, :], samples[1, :]
            return bkd.reshape(x**4 + y**4, (1, -1))

        candidate, grid, selected, sel_plus = _build_candidate(
            bkd, 2, 1, (2, 0), target_fn
        )
        samples = grid.get_samples("all")
        diff = sel_plus(samples) - selected(samples)
        expected = bkd.to_float(
            bkd.sqrt(bkd.sum(diff * diff) / samples.shape[1])
        )
        got = L2GlobalSurplusIndicator(bkd)(candidate, grid)
        bkd.assert_allclose(
            bkd.asarray([got]), bkd.asarray([expected]), rtol=1e-10
        )


class TestL2Surplus:
    """RMS surplus on the candidate's new samples."""

    def test_zero_for_exactly_represented_function(self, bkd) -> None:
        def target_fn(samples):
            return bkd.reshape(samples[0, :] + samples[1, :], (1, -1))

        candidate, grid, _, _ = _build_candidate(
            bkd, 2, 1, (2, 0), target_fn
        )
        assert L2SurplusIndicator(bkd)(candidate, grid) < 1e-10

    def test_nonzero_for_underresolved_function(self, bkd) -> None:
        def target_fn(samples):
            x, y = samples[0, :], samples[1, :]
            return bkd.reshape(x**4 + y**4, (1, -1))

        candidate, grid, _, _ = _build_candidate(
            bkd, 2, 1, (2, 0), target_fn
        )
        assert L2SurplusIndicator(bkd)(candidate, grid) > 0

    def test_matches_surrogate_difference(self, bkd) -> None:
        """Checked on the new samples only, which is what it scores."""

        def target_fn(samples):
            x, y = samples[0, :], samples[1, :]
            return bkd.reshape(x**4 + y**4, (1, -1))

        candidate, grid, selected, sel_plus = _build_candidate(
            bkd, 2, 1, (2, 0), target_fn
        )
        local = bkd.asarray(
            list(candidate.new_sample_local_indices),
            dtype=bkd.int64_dtype(),
        )
        new_samples = candidate.subspace.get_samples()[:, local]
        diff = sel_plus(new_samples) - selected(new_samples)
        expected = bkd.to_float(
            bkd.sqrt(bkd.sum(diff * diff) / new_samples.shape[1])
        )
        got = L2SurplusIndicator(bkd)(candidate, grid)
        bkd.assert_allclose(
            bkd.asarray([got]), bkd.asarray([expected]), rtol=1e-10
        )


class TestVarianceChange:
    """Change in mean and in the summed per-subspace variance."""

    def test_zero_for_constant(self, bkd) -> None:
        """A constant has no variance to expose at any level."""

        def target_fn(samples):
            return bkd.full((1, samples.shape[1]), 5.0)

        candidate, grid, _, _ = _build_candidate(
            bkd, 2, 1, (2, 0), target_fn
        )
        assert SummedSubspaceVarianceIndicator(bkd)(candidate, grid) < 1e-6

    def test_zero_for_already_resolved_variance(self, bkd) -> None:
        def target_fn(samples):
            return bkd.reshape(samples[0, :], (1, -1))

        candidate, grid, _, _ = _build_candidate(
            bkd, 2, 1, (2, 0), target_fn
        )
        assert SummedSubspaceVarianceIndicator(bkd)(candidate, grid) < 1e-6

    def test_nonzero_for_underresolved_variance(self, bkd) -> None:
        def target_fn(samples):
            x, y = samples[0, :], samples[1, :]
            return bkd.reshape(x**4 + y**4, (1, -1))

        candidate, grid, _, _ = _build_candidate(
            bkd, 2, 1, (2, 0), target_fn
        )
        assert SummedSubspaceVarianceIndicator(bkd)(candidate, grid) > 0

    def test_uses_one_qoi_for_both_terms(self, bkd) -> None:
        """q* is chosen by |Delta V|, and the mean term uses that same q*.

        QoI 0 carries a large mean change and no variance change; QoI 1
        the reverse. Taking each term's max independently would sum the
        two, so the error would exceed what one QoI can produce.
        """

        def target_fn(samples):
            x, y = samples[0, :], samples[1, :]
            # QoI 0: a quartic in x, shifted far from zero. QoI 1: a
            # quartic in y with no offset.
            return bkd.stack([x**4 + 100.0, y**4], axis=0)

        candidate, grid, _, _ = _build_candidate(
            bkd, 2, 1, (2, 0), target_fn
        )
        indicator = SummedSubspaceVarianceIndicator(bkd)
        error = indicator(candidate, grid)

        # Recover the per-QoI changes the indicator saw.
        from pyapprox.surrogates.sparsegrids.statistics.cache import box_sum

        delta = box_sum(indicator._cache, candidate.box)
        delta_mean, delta_var = delta[0], delta[1]
        qstar = int(bkd.to_int(bkd.argmax(bkd.abs(delta_var))))
        expected = bkd.to_float(
            bkd.abs(delta_mean[qstar])
            + bkd.sqrt(bkd.abs(delta_var[qstar]))
        )
        bkd.assert_allclose(
            bkd.asarray([error]), bkd.asarray([expected]), rtol=1e-12
        )

        # And it is strictly less than mixing QoIs would give, unless the
        # same QoI happens to dominate both.
        mixed = bkd.to_float(
            bkd.max(bkd.abs(delta_mean))
            + bkd.max(bkd.sqrt(bkd.abs(delta_var)))
        )
        assert error <= mixed + 1e-12


class TestCacheReuse:
    """Per-subspace statistics are computed once."""

    def test_statistic_computed_once_per_subspace(self, bkd) -> None:
        """Scoring the same candidate repeatedly does not recompute."""

        def target_fn(samples):
            x, y = samples[0, :], samples[1, :]
            return bkd.reshape(x**4 + y**4, (1, -1))

        candidate, grid, _, _ = _build_candidate(
            bkd, 2, 1, (2, 0), target_fn
        )
        indicator = SummedSubspaceVarianceIndicator(bkd)
        for _ in range(4):
            indicator(candidate, grid)
        # The box holds 2^nnz = 2 subspaces for candidate (2, 0).
        assert indicator._cache.nentries() == len(candidate.box)
