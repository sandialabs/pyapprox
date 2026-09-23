"""Tests for SampleTracker value distribution and write-once subspaces.

Distribution used to rewrite every registered subspace on every call,
which costs O(total samples) per step and conflicts with subspace values
being write-once. These tests pin the replacement: each subspace is
written exactly once, when the values covering its samples arrive.
"""

from typing import List

import pytest
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import LinearGrowthRule
from pyapprox.surrogates.sparsegrids import create_basis_factories
from pyapprox.surrogates.sparsegrids.sample_tracker import SampleTracker
from pyapprox.surrogates.sparsegrids.subspace import TensorProductSubspace
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)


class _CountingSubspace(TensorProductSubspace):
    """Subspace that records how often its values are written."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nset_values = 0

    def set_values(self, values):
        self.nset_values += 1
        return super().set_values(values)


class _CountingFactory:
    """Subspace factory producing counting subspaces."""

    def __init__(self, inner: TensorProductSubspaceFactory) -> None:
        self._inner = inner

    def __call__(self, approx_index):
        return _CountingSubspace(
            self._inner._bkd,
            approx_index[: self._inner.nvars_physical()],
            self._inner._basis_factories,
            self._inner._growth_rules,
        )

    def nvars_physical(self) -> int:
        return self._inner.nvars_physical()

    def growth_rules(self):
        return self._inner.growth_rules()

    def is_nested(self) -> bool:
        return self._inner.is_nested()


def _make_factory(bkd, nvars: int, basis_type: str):
    marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(nvars)]
    factories = create_basis_factories(marginals, bkd, basis_type)
    return TensorProductSubspaceFactory(
        bkd, factories, LinearGrowthRule(scale=1, shift=1)
    )


def _register(tracker, factory, bkd, keys) -> List[int]:
    positions = []
    for key in keys:
        idx = bkd.asarray(list(key), dtype=bkd.int64_dtype())
        positions.append(tracker.register(idx, factory(idx)))
    return positions


def _supply_all(tracker, bkd) -> None:
    """Append values for every outstanding sample, then distribute."""
    samples = tracker.collect_unique_samples()
    held = 0 if tracker._values is None else tracker._values.shape[1]
    outstanding = samples[:, held:]
    tracker.append_new_values(
        bkd.reshape(outstanding[0] ** 2, (1, -1))
    )
    tracker.distribute_values_to_subspaces()


class TestWritesOnlyNewSubspaces:
    """Earlier subspaces are never rewritten as the grid grows."""

    def test_each_subspace_written_once(self, bkd) -> None:
        """Growing the grid in stages writes each subspace exactly once."""
        inner = _make_factory(bkd, 2, "leja")
        factory = _CountingFactory(inner)
        tracker = SampleTracker(bkd, factory)

        _register(tracker, factory, bkd, [(0, 0), (1, 0)])
        _supply_all(tracker, bkd)
        first_round = list(tracker._registered)
        assert [s.nset_values for s in first_round] == [1, 1]

        _register(tracker, factory, bkd, [(0, 1), (1, 1)])
        _supply_all(tracker, bkd)
        # The first two must not have been touched again.
        assert [s.nset_values for s in first_round] == [1, 1]
        assert all(s.nset_values == 1 for s in tracker._registered)

    def test_nested_reuse_still_writes_once(self, bkd) -> None:
        """Leja nesting means new subspaces inherit most values."""
        inner = _make_factory(bkd, 2, "leja")
        factory = _CountingFactory(inner)
        tracker = SampleTracker(bkd, factory)
        _register(tracker, factory, bkd, [(0, 0), (1, 0), (2, 0), (1, 1)])
        _supply_all(tracker, bkd)
        assert all(s.nset_values == 1 for s in tracker._registered)
        # Nesting: far fewer unique samples than the sum of subspace sizes.
        total = sum(s.nsamples() for s in tracker._registered)
        assert tracker.n_unique_samples() < total

    def test_gauss_rules_sharing_centre_point(self, bkd) -> None:
        """Non-nested Gauss rules dedupe by coordinate, not basis index."""
        inner = _make_factory(bkd, 1, "gauss")
        factory = _CountingFactory(inner)
        tracker = SampleTracker(bkd, factory)
        # Odd-point Gauss rules all contain the midpoint of [0, 1].
        _register(tracker, factory, bkd, [(0,), (2,)])
        _supply_all(tracker, bkd)
        assert all(s.nset_values == 1 for s in tracker._registered)
        total = sum(s.nsamples() for s in tracker._registered)
        assert tracker.n_unique_samples() < total


class TestPending:
    """npending tracks subspaces still awaiting values."""

    def test_everything_pending_before_any_values(self, bkd) -> None:
        """Distributing with no values leaves every subspace pending."""
        factory = _make_factory(bkd, 2, "leja")
        tracker = SampleTracker(bkd, factory)
        _register(tracker, factory, bkd, [(0, 0), (1, 0)])
        tracker.distribute_values_to_subspaces()
        assert tracker.npending() == 2
        for subspace in tracker._registered:
            assert subspace.get_values() is None

    def test_pending_clears_once_values_arrive(self, bkd) -> None:
        factory = _make_factory(bkd, 2, "leja")
        tracker = SampleTracker(bkd, factory)
        _register(tracker, factory, bkd, [(0, 0), (1, 0)])
        _supply_all(tracker, bkd)
        assert tracker.npending() == 0
        for subspace in tracker._registered:
            assert subspace.get_values() is not None


class TestAppendValidation:
    """append_new_values requires all outstanding samples, in order."""

    def test_short_batch_raises(self, bkd) -> None:
        """A batch covering only some outstanding samples is rejected."""
        factory = _make_factory(bkd, 2, "leja")
        tracker = SampleTracker(bkd, factory)
        _register(tracker, factory, bkd, [(0, 0), (1, 0), (0, 1)])
        with pytest.raises(ValueError, match="outstanding samples"):
            tracker.append_new_values(bkd.zeros((1, 1)))

    def test_long_batch_raises(self, bkd) -> None:
        """A batch claiming more samples than exist is rejected."""
        factory = _make_factory(bkd, 2, "leja")
        tracker = SampleTracker(bkd, factory)
        _register(tracker, factory, bkd, [(0, 0), (1, 0)])
        with pytest.raises(ValueError, match="outstanding samples"):
            tracker.append_new_values(bkd.zeros((1, 99)))

    def test_exact_batch_accepted(self, bkd) -> None:
        factory = _make_factory(bkd, 2, "leja")
        tracker = SampleTracker(bkd, factory)
        _register(tracker, factory, bkd, [(0, 0), (1, 0)])
        nout = tracker.n_unique_samples()
        tracker.append_new_values(bkd.zeros((1, nout)))
        assert tracker._values.shape[1] == nout

    def test_second_append_after_new_registrations(self, bkd) -> None:
        """A later append supplies only the newly registered samples."""
        factory = _make_factory(bkd, 2, "leja")
        tracker = SampleTracker(bkd, factory)
        _register(tracker, factory, bkd, [(0, 0), (1, 0)])
        first = tracker.n_unique_samples()
        tracker.append_new_values(bkd.zeros((1, first)))

        _register(tracker, factory, bkd, [(0, 1)])
        outstanding = tracker.n_unique_samples() - first
        assert outstanding > 0
        with pytest.raises(ValueError, match="outstanding samples"):
            tracker.append_new_values(bkd.zeros((1, tracker.n_unique_samples())))
        tracker.append_new_values(bkd.zeros((1, outstanding)))
        assert tracker._values.shape[1] == tracker.n_unique_samples()


class TestWriteOnceSubspaceValues:
    """TensorProductSubspace values cannot be replaced."""

    def test_second_set_values_raises(self, bkd) -> None:
        factory = _make_factory(bkd, 2, "leja")
        idx = bkd.asarray([1, 0], dtype=bkd.int64_dtype())
        subspace = factory(idx)
        values = bkd.zeros((1, subspace.nsamples()))
        subspace.set_values(values)
        with pytest.raises(ValueError, match="already set"):
            subspace.set_values(values)

    def test_first_set_values_succeeds(self, bkd) -> None:
        factory = _make_factory(bkd, 2, "leja")
        idx = bkd.asarray([1, 0], dtype=bkd.int64_dtype())
        subspace = factory(idx)
        values = bkd.zeros((1, subspace.nsamples()))
        subspace.set_values(values)
        assert subspace.get_values() is not None


class TestQuadratureWeights:
    """The weights are probability weights, not Lebesgue weights."""

    @pytest.mark.parametrize("basis_type", ["gauss", "leja", "clenshaw_curtis"])
    @pytest.mark.parametrize("nvars", [1, 2, 3])
    def test_weights_sum_to_one(
        self, bkd, basis_type: str, nvars: int
    ) -> None:
        """Sum is 1 in every dimension, not 2**nvars."""
        marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(nvars)]
        factories = create_basis_factories(marginals, bkd, basis_type)
        from pyapprox.surrogates.affine.indices import (
            ClenshawCurtisGrowthRule,
        )

        growth = (
            ClenshawCurtisGrowthRule()
            if basis_type == "clenshaw_curtis"
            else LinearGrowthRule(scale=1, shift=1)
        )
        factory = TensorProductSubspaceFactory(bkd, factories, growth)
        idx = bkd.asarray([2] * nvars, dtype=bkd.int64_dtype())
        weights = factory(idx).get_quadrature_weights()
        bkd.assert_allclose(
            bkd.asarray([bkd.sum(weights)]), bkd.asarray([1.0]), rtol=1e-12
        )
