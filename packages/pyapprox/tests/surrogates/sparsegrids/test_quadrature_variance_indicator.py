"""Tests for QuadratureVarianceIndicator.

The indicator reports |Delta m| + |Delta sigma| for the quantity
QuadratureMoments defines, so the checks are against that class applied
to the selected set and to the selected set plus the candidate. Building
those two surrogates is exactly the work the indicator avoids, which
makes it a genuine cross-check rather than a restatement.
"""

from typing import List, Tuple

import pytest
from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import (
    LinearGrowthRule,
    MaxLevelCriteria,
)
from pyapprox.surrogates.sparsegrids import create_basis_factories
from pyapprox.surrogates.sparsegrids.adaptive_fitter import (
    SingleFidelityAdaptiveSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.candidate_info import (
    Candidate,
    SmolyakSelection,
)
from pyapprox.surrogates.sparsegrids.combination_surrogate import (
    CombinationSurrogate,
)
from pyapprox.surrogates.sparsegrids.error_indicators import (
    ErrorIndicatorProtocol,
)
from pyapprox.surrogates.sparsegrids.quadrature_variance_indicator import (
    QuadratureVarianceIndicator,
)
from pyapprox.surrogates.sparsegrids.sample_tracker import SampleTracker
from pyapprox.surrogates.sparsegrids.smolyak import (
    IncrementalSmolyakCoefficients,
    compute_smolyak_coefficients,
)
from pyapprox.surrogates.sparsegrids.statistics.moments import (
    QuadratureMoments,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)


def _level_set(nvars: int, level: int) -> List[Tuple[int, ...]]:
    keys: List[Tuple[int, ...]] = []
    for total in range(level + 1):
        keys.extend(_compositions(nvars, total))
    return keys


def _compositions(nvars: int, total: int) -> List[Tuple[int, ...]]:
    if nvars == 1:
        return [(total,)]
    out: List[Tuple[int, ...]] = []
    for first in range(total + 1):
        for rest in _compositions(nvars - 1, total - first):
            out.append((first,) + rest)
    return out


class _Grid:
    """Supplies the selection snapshot, which is all this needs."""

    def __init__(self, selection) -> None:
        self._selection = selection

    def selection(self):
        return self._selection


def _build(bkd, nvars, selected_level, candidate_key, target_fn, cost=1.0):
    """Build a candidate, a grid view, and the two surrogates."""
    marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(nvars)]
    factories = create_basis_factories(marginals, bkd, "gauss")
    tp_factory = TensorProductSubspaceFactory(
        bkd, factories, LinearGrowthRule(scale=1, shift=1)
    )
    selected_keys = _level_set(nvars, selected_level)

    smolyak = IncrementalSmolyakCoefficients(nvars)
    for key in selected_keys:
        smolyak.add(key)

    tracker = SampleTracker(bkd, tp_factory)
    subspaces = {}
    positions = {}
    for key in selected_keys + [candidate_key]:
        idx = bkd.asarray(list(key), dtype=bkd.int64_dtype())
        subspace = tp_factory(idx)
        subspaces[key] = subspace
        positions[key] = tracker.register(idx, subspace)
    tracker.append_new_values(target_fn(tracker.collect_unique_samples()))
    tracker.distribute_values_to_subspaces()

    selection = SmolyakSelection(
        terms=tuple(
            (coef, subspaces[key])
            for key, coef in smolyak.nonzero_items()
        )
    )
    candidate = Candidate(
        index=bkd.asarray(list(candidate_key), dtype=bkd.int64_dtype()),
        subspace=subspaces[candidate_key],
        box=[
            (sign, subspaces[key])
            for key, sign in smolyak.delta(candidate_key)
        ],
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
        return CombinationSurrogate(
            bkd,
            nvars,
            [subspaces[k] for k in keys],
            compute_smolyak_coefficients(indices, bkd),
            1,
            indices=indices,
        )

    return (
        candidate,
        _Grid(selection),
        surrogate(selected_keys),
        surrogate(selected_keys + [candidate_key]),
    )


def _quartic(bkd):
    def fun(samples):
        return bkd.reshape(samples[0] ** 4 + samples[1] ** 4, (1, -1))

    return fun


def _expected_error(bkd, selected, sel_plus):
    """|Delta m| + |Delta sigma| from the two surrogates directly."""
    before = QuadratureMoments(selected)
    after = QuadratureMoments(sel_plus)
    delta_mean = after.mean() - before.mean()
    sigma_before = bkd.sqrt(bkd.abs(before.variance()))
    sigma_after = bkd.sqrt(bkd.abs(after.variance()))
    delta_variance = after.variance() - before.variance()
    qstar = bkd.to_int(bkd.argmax(bkd.abs(delta_variance)))
    return bkd.to_float(
        bkd.abs(delta_mean[qstar])
        + bkd.abs(sigma_after[qstar] - sigma_before[qstar])
    )


class TestMatchesQuadratureMoments:
    """The reported change is the one QuadratureMoments would give."""

    @pytest.mark.parametrize(
        "selected_level,candidate_key",
        [
            (1, (2, 0)),
            (1, (0, 2)),
            (1, (1, 1)),
            # (2, 1) needs (1, 1), so it is only admissible against a
            # level-2 set.
            (2, (2, 1)),
            (2, (3, 0)),
        ],
    )
    def test_error_equals_surrogate_difference(
        self, bkd, selected_level: int, candidate_key: Tuple[int, int]
    ) -> None:
        candidate, grid, selected, sel_plus = _build(
            bkd, 2, selected_level, candidate_key, _quartic(bkd)
        )
        got = QuadratureVarianceIndicator(bkd)(candidate, grid)
        expected = _expected_error(bkd, selected, sel_plus)
        bkd.assert_allclose(
            bkd.asarray([got]), bkd.asarray([expected]), rtol=1e-10
        )

    def test_selected_moments_match_the_surrogate(self, bkd) -> None:
        """The snapshot combination is QuadratureMoments of the selected."""
        candidate, grid, selected, _ = _build(
            bkd, 2, 2, (3, 0), _quartic(bkd)
        )
        indicator = QuadratureVarianceIndicator(bkd)
        combined = indicator._selected_moments(grid.selection())
        moments = QuadratureMoments(selected)
        bkd.assert_allclose(combined[0], moments.mean(), rtol=1e-10)
        bkd.assert_allclose(
            combined[1], moments.second_moment(), rtol=1e-10
        )


class TestErrorForm:
    """|Delta m| + |Delta sigma|, one QoI for both terms."""

    def test_positive_for_underresolved_target(self, bkd) -> None:
        candidate, grid, _, _ = _build(
            bkd, 2, 1, (2, 0), _quartic(bkd)
        )
        assert QuadratureVarianceIndicator(bkd)(candidate, grid) > 0

    def test_near_zero_for_resolved_target(self, bkd) -> None:
        """A linear target is already exact at level 1."""

        def linear(samples):
            return bkd.reshape(samples[0] + samples[1], (1, -1))

        candidate, grid, _, _ = _build(bkd, 2, 1, (2, 0), linear)
        assert QuadratureVarianceIndicator(bkd)(candidate, grid) < 1e-10

    def test_constant_target_is_near_zero(self, bkd) -> None:
        """A constant has no variance and no mean change to find.

        The reported error is not exactly zero, and cannot be: the
        variance rounds to order 1e-15 rather than 0, and a square root
        turns that into order 1e-8. Both terms of this error form take
        a square root, so this floor applies to any of them --- it is
        the scale below which a variance-driven indicator cannot
        distinguish signal from rounding.
        """
        candidate, grid, _, _ = _build(
            bkd, 2, 1, (2, 0), lambda s: bkd.full((1, s.shape[1]), 3.0)
        )
        error = QuadratureVarianceIndicator(bkd)(candidate, grid)
        assert error < 1e-6

    def test_mean_term_survives_when_the_variance_does_not(
        self, bkd
    ) -> None:
        """The mean term is what stops refinement stalling.

        For a target whose variance a candidate leaves untouched, an
        indicator built on the variance alone would report no error.
        Here the candidate changes the mean in a second QoI while the
        first carries all the variance change.
        """

        def two_qoi(samples):
            # QoI 0 varies in x only; QoI 1 is a large constant offset
            # plus a y term, so adding a y-refinement moves its mean.
            return bkd.stack(
                [samples[0] ** 4, 100.0 + samples[1] ** 4], axis=0
            )

        candidate, grid, selected, sel_plus = _build(
            bkd, 2, 1, (0, 2), two_qoi
        )
        error = QuadratureVarianceIndicator(bkd)(candidate, grid)
        expected = _expected_error(bkd, selected, sel_plus)
        bkd.assert_allclose(
            bkd.asarray([error]), bkd.asarray([expected]), rtol=1e-10
        )
        assert error > 0


class TestSnapshotReuse:
    """The selected set is combined once per round."""

    def test_combined_once_per_snapshot(self, bkd) -> None:
        candidate, grid, _, _ = _build(
            bkd, 2, 2, (3, 0), _quartic(bkd)
        )
        indicator = QuadratureVarianceIndicator(bkd)
        first = indicator._selected_moments(grid.selection())
        for _ in range(4):
            indicator(candidate, grid)
        # Same object, so the combination was not redone.
        assert indicator._selected_moments(grid.selection()) is first

    def test_rejects_empty_selection(self, bkd) -> None:
        indicator = QuadratureVarianceIndicator(bkd)
        with pytest.raises(ValueError, match="empty selected set"):
            indicator._selected_moments(SmolyakSelection(terms=()))


class TestProtocolAndIntegration:
    """Usable where the protocol is required, and drives refinement."""

    def test_satisfies_protocol(self, bkd) -> None:
        assert isinstance(
            QuadratureVarianceIndicator(bkd), ErrorIndicatorProtocol
        )

    def test_end_to_end_convergence(self, bkd) -> None:
        """Refinement driven by this indicator converges."""
        marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(2)]
        factories = create_basis_factories(marginals, bkd, "gauss")
        tp_factory = TensorProductSubspaceFactory(
            bkd, factories, LinearGrowthRule(scale=1, shift=1)
        )
        admis = MaxLevelCriteria(max_level=5, pnorm=1.0, bkd=bkd)
        fitter = SingleFidelityAdaptiveSparseGridFitter(
            bkd,
            tp_factory,
            admis,
            error_indicator=QuadratureVarianceIndicator(bkd),
        )

        def fun(samples):
            return bkd.reshape(
                samples[0] ** 2 + samples[0] * samples[1], (1, -1)
            )

        result = fitter.refine_to_tolerance(
            FunctionFromCallable(1, 2, fun, bkd), tol=1e-12, max_steps=40
        )
        samples = bkd.asarray(
            [[0.1, 0.35, 0.7], [0.2, 0.55, 0.9]]
        )
        bkd.assert_allclose(
            result.surrogate(samples), fun(samples), rtol=1e-8
        )
