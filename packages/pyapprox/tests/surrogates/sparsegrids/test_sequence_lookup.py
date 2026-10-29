"""Resuming a sparse grid from stored values, keyed by sequence position.

A ``SequencePositionLookup`` behind ``CachedObjective`` lets a second
run reuse every value a first run stored. Each run builds its own basis
factories and lookup, as a new process would, and shares only the store.

Tests run on both NumPy and PyTorch backends.
"""

from pathlib import Path
from typing import Generic, Set

import pytest

from pyapprox.interface.evaluation.caching import CachedObjective
from pyapprox.interface.evaluation.protocols import ResultStore, SampleLookup
from pyapprox.interface.evaluation.records import ComputeProvenance, Cost, Decoded
from pyapprox.interface.evaluation.stores import InMemoryResultStore, NpzResultStore
from pyapprox.interface.functions.fromcallable.function import FunctionFromCallable
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import LinearGrowthRule, MaxLevelCriteria
from pyapprox.surrogates.sparsegrids.adaptive_fitter import (
    SingleFidelityAdaptiveSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.basis_factory import (
    GaussLagrangeFactory,
    LejaLagrangeFactory,
)
from pyapprox.surrogates.sparsegrids.error_indicators import (
    ErrorIndicatorProtocol,
    L2GlobalSurplusIndicator,
    L2SurplusIndicator,
)
from pyapprox.surrogates.sparsegrids.sequence_lookup import SequencePositionLookup
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)
from pyapprox.util.backends.protocols import Array, Backend

_NVARS = 2
_MAX_LEVEL = 6
_BUDGET = 25


class _CountingTarget(Generic[Array]):
    """A smooth target that counts the samples it evaluates."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._nevaluated = 0

    def nevaluated(self) -> int:
        return self._nevaluated

    def __call__(self, samples: Array) -> Array:
        self._nevaluated += samples.shape[1]
        x0, x1 = samples[0], samples[1]
        values = self._bkd.exp(0.7 * x0 + 0.3 * x1) + 0.2 * x0 * x1
        return self._bkd.reshape(values, (1, -1))


def _leja_factory(
    bkd: Backend[Array], upper: float = 1.0, step: int = 2
) -> TensorProductSubspaceFactory[Array]:
    """A fresh factory, as a new process would build it."""
    leja = LejaLagrangeFactory(UniformMarginal(0.0, upper, bkd), bkd)
    growth = LinearGrowthRule(scale=step, shift=1)
    return TensorProductSubspaceFactory(bkd, [leja] * _NVARS, growth)


def _run(
    bkd: Backend[Array],
    factory: TensorProductSubspaceFactory[Array],
    indicator: ErrorIndicatorProtocol[Array],
    model: CachedObjective[Array] | _CountingTarget[Array],
) -> SingleFidelityAdaptiveSparseGridFitter[Array]:
    """Refine until the budget is reached; return the fitter."""
    fitter = SingleFidelityAdaptiveSparseGridFitter(
        bkd,
        factory,
        MaxLevelCriteria(_MAX_LEVEL, 1.0, bkd),
        error_indicator=indicator,
    )
    while True:
        samples = fitter.step_samples()
        if samples is None:
            break
        fitter.step_values(model(samples))
        if fitter.get_samples("all").shape[1] >= _BUDGET:
            break
    return fitter


def _cached(
    bkd: Backend[Array],
    target: _CountingTarget[Array],
    store: ResultStore[Array],
    factory: TensorProductSubspaceFactory[Array],
) -> CachedObjective[Array]:
    lookup = SequencePositionLookup(bkd, factory, store, _MAX_LEVEL)
    model = FunctionFromCallable(1, _NVARS, target, bkd)
    return CachedObjective(model, store, lookup)


def _keys(lookup: SequencePositionLookup[Array], samples: Array) -> Set[str]:
    return {lookup.remember(samples[:, i : i + 1]) for i in range(samples.shape[1])}


def _assert_same_grid(
    bkd: Backend[Array],
    actual: SingleFidelityAdaptiveSparseGridFitter[Array],
    desired: SingleFidelityAdaptiveSparseGridFitter[Array],
) -> None:
    bkd.assert_allclose(
        actual.get_samples("all"), desired.get_samples("all"), rtol=1e-12
    )
    actual_values = actual.get_values()
    desired_values = desired.get_values()
    assert actual_values is not None and desired_values is not None
    bkd.assert_allclose(actual_values, desired_values, rtol=1e-12)
    actual_selected = actual.get_selected_indices()
    desired_selected = desired.get_selected_indices()
    assert actual_selected.shape == desired_selected.shape
    assert bkd.all_bool(bkd.equal(actual_selected, desired_selected))


@pytest.fixture(params=["memory", "npz"])
def store(
    request: pytest.FixtureRequest, bkd: Backend[Array], tmp_path: Path
) -> ResultStore[Array]:
    if request.param == "memory":
        return InMemoryResultStore()
    return NpzResultStore(str(tmp_path / "store"), bkd)


class TestResume:
    def test_lookup_satisfies_protocol(
        self, bkd: Backend[Array], store: ResultStore[Array]
    ) -> None:
        lookup = SequencePositionLookup(bkd, _leja_factory(bkd), store, _MAX_LEVEL)
        assert isinstance(lookup, SampleLookup)

    def test_second_run_evaluates_nothing(
        self, bkd: Backend[Array], store: ResultStore[Array]
    ) -> None:
        """Same setup, same store: identical grid, zero model calls."""
        first_target = _CountingTarget(bkd)
        first = _run(
            bkd,
            _leja_factory(bkd),
            L2SurplusIndicator(bkd),
            _cached(bkd, first_target, store, _leja_factory(bkd)),
        )
        assert first_target.nevaluated() == first.get_samples("all").shape[1]

        second_target = _CountingTarget(bkd)
        second = _run(
            bkd,
            _leja_factory(bkd),
            L2SurplusIndicator(bkd),
            _cached(bkd, second_target, store, _leja_factory(bkd)),
        )
        assert second_target.nevaluated() == 0
        _assert_same_grid(bkd, second, first)

    def test_other_indicator_evaluates_only_new_samples(
        self, bkd: Backend[Array], store: ResultStore[Array]
    ) -> None:
        """Values are found by sample, so a new order reuses what it can."""
        first = _run(
            bkd,
            _leja_factory(bkd),
            L2SurplusIndicator(bkd),
            _cached(bkd, _CountingTarget(bkd), store, _leja_factory(bkd)),
        )
        resumed_target = _CountingTarget(bkd)
        resumed = _run(
            bkd,
            _leja_factory(bkd),
            L2GlobalSurplusIndicator(bkd),
            _cached(bkd, resumed_target, store, _leja_factory(bkd)),
        )
        fresh = _run(
            bkd,
            _leja_factory(bkd),
            L2GlobalSurplusIndicator(bkd),
            _CountingTarget(bkd),
        )
        _assert_same_grid(bkd, resumed, fresh)

        keys = SequencePositionLookup(
            bkd, _leja_factory(bkd), InMemoryResultStore(), _MAX_LEVEL
        )
        new_keys = _keys(keys, resumed.get_samples("all")) - _keys(
            keys, first.get_samples("all")
        )
        assert resumed_target.nevaluated() == len(new_keys)

    def test_growth_step_change_keeps_the_store(
        self, bkd: Backend[Array], store: ResultStore[Array]
    ) -> None:
        """One point per level to two keeps the Leja points, so the keys."""
        first = _run(
            bkd,
            _leja_factory(bkd, step=1),
            L2SurplusIndicator(bkd),
            _cached(bkd, _CountingTarget(bkd), store, _leja_factory(bkd, step=1)),
        )
        resumed_target = _CountingTarget(bkd)
        resumed = _run(
            bkd,
            _leja_factory(bkd, step=2),
            L2SurplusIndicator(bkd),
            _cached(bkd, resumed_target, store, _leja_factory(bkd, step=2)),
        )
        keys = SequencePositionLookup(
            bkd, _leja_factory(bkd), InMemoryResultStore(), _MAX_LEVEL
        )
        reused = _keys(keys, resumed.get_samples("all")) & _keys(
            keys, first.get_samples("all")
        )
        assert len(reused) > 0
        assert resumed_target.nevaluated() == (
            resumed.get_samples("all").shape[1] - len(reused)
        )


class TestRefusals:
    def test_different_marginal(
        self, bkd: Backend[Array], store: ResultStore[Array]
    ) -> None:
        SequencePositionLookup(bkd, _leja_factory(bkd), store, _MAX_LEVEL)
        with pytest.raises(ValueError, match="differ"):
            SequencePositionLookup(
                bkd, _leja_factory(bkd, upper=2.0), store, _MAX_LEVEL
            )

    def test_gauss_is_not_nested(
        self, bkd: Backend[Array], store: ResultStore[Array]
    ) -> None:
        gauss = GaussLagrangeFactory(UniformMarginal(0.0, 1.0, bkd), bkd)
        factory = TensorProductSubspaceFactory(
            bkd, [gauss] * _NVARS, LinearGrowthRule(scale=2, shift=1)
        )
        with pytest.raises(ValueError, match="nested"):
            SequencePositionLookup(bkd, factory, store, _MAX_LEVEL)

    def test_records_without_a_header(
        self, bkd: Backend[Array], store: ResultStore[Array]
    ) -> None:
        """A store from before headers: its keys' meaning is unknown."""
        store.save(
            "values:0-0",
            Decoded(values=bkd.array([[1.0]]), indices=[0]),
            Cost(
                wall_clock=0.0,
                compute=0.0,
                provenance=ComputeProvenance.NOT_APPLICABLE,
            ),
        )
        with pytest.raises(ValueError, match="no header"):
            SequencePositionLookup(bkd, _leja_factory(bkd), store, _MAX_LEVEL)

    def test_different_number_of_variables(
        self, bkd: Backend[Array], store: ResultStore[Array]
    ) -> None:
        store.save_header({"sequence_0": bkd.array([0.5])})
        with pytest.raises(ValueError, match="number of variables"):
            SequencePositionLookup(bkd, _leja_factory(bkd), store, _MAX_LEVEL)

    def test_lower_level_store_point_differs(
        self, bkd: Backend[Array], store: ResultStore[Array]
    ) -> None:
        """A level-0 grid still checks the one stored point it can build."""
        _extend_to_level_two(bkd, store)
        header = store.load_header()
        assert header is not None
        moved = bkd.copy(header["sequence_0"])
        moved[0] = moved[0] + 1e-6
        store.save_header({"sequence_0": moved, "sequence_1": header["sequence_1"]})
        with pytest.raises(ValueError, match="differ"):
            SequencePositionLookup(bkd, _leja_factory(bkd), store, 0)


def _extend_to_level_two(bkd: Backend[Array], store: ResultStore[Array]) -> int:
    """Key one level-2 point in dimension 0; return the stored length."""
    factory = _leja_factory(bkd)
    lookup = SequencePositionLookup(bkd, factory, store, _MAX_LEVEL)
    index = bkd.zeros((_NVARS,), dtype=bkd.int64_dtype())
    index[0] = 2
    level_two = factory(index).get_samples_1d(0)[0]
    sample = bkd.stack([level_two[-1:], lookup.sequence(1)[:1]], axis=0)
    assert lookup.find(sample) is not None
    return int(lookup.sequence(0).shape[0])


class TestLowerLevelGrid:
    """A grid capped below the store's level reuses what it can build."""

    def test_opens_a_higher_level_store(
        self, bkd: Backend[Array], store: ResultStore[Array]
    ) -> None:
        _extend_to_level_two(bkd, store)
        lower = SequencePositionLookup(bkd, _leja_factory(bkd), store, 0)
        midpoint = bkd.stack([lower.sequence(0)[:1], lower.sequence(1)[:1]], axis=0)
        assert lower.remember(midpoint) == "0-0"

    def test_keeps_the_longer_stored_sequence(
        self, bkd: Backend[Array], store: ResultStore[Array]
    ) -> None:
        """Saving from a lower-level grid must not cut the header."""
        nstored = _extend_to_level_two(bkd, store)
        lower = SequencePositionLookup(bkd, _leja_factory(bkd), store, 1)
        # Extending dimension 1 to level 1 makes the lower grid save.
        index = bkd.zeros((_NVARS,), dtype=bkd.int64_dtype())
        index[1] = 1
        level_one = _leja_factory(bkd)(index).get_samples_1d(1)[0]
        sample = bkd.stack([lower.sequence(0)[:1], level_one[-1:]], axis=0)
        assert lower.find(sample) is not None
        header = store.load_header()
        assert header is not None
        assert header["sequence_0"].shape[0] == nstored
        # And the higher-level grid still opens it.
        SequencePositionLookup(bkd, _leja_factory(bkd), store, _MAX_LEVEL)


class TestKeys:
    def test_positions_name_the_points(
        self, bkd: Backend[Array], store: ResultStore[Array]
    ) -> None:
        """Every key's positions index back to the sample's coordinates."""
        fitter = _run(
            bkd, _leja_factory(bkd), L2SurplusIndicator(bkd), _CountingTarget(bkd)
        )
        lookup = SequencePositionLookup(bkd, _leja_factory(bkd), store, _MAX_LEVEL)
        samples = fitter.get_samples("all")
        keys = [lookup.remember(samples[:, i : i + 1]) for i in range(samples.shape[1])]
        assert len(set(keys)) == samples.shape[1]
        for i, key in enumerate(keys):
            for dim, position in enumerate(key.split("-")):
                bkd.assert_allclose(
                    lookup.sequence(dim)[int(position)],
                    samples[dim, i],
                    atol=1e-12,
                )

    def test_off_grid_sample(
        self, bkd: Backend[Array], store: ResultStore[Array]
    ) -> None:
        lookup = SequencePositionLookup(bkd, _leja_factory(bkd), store, 2)
        off_grid = bkd.array([[0.123456], [0.5]])
        assert lookup.find(off_grid) is None
        with pytest.raises(ValueError, match="not a point"):
            lookup.remember(off_grid)

    def test_rejects_more_than_one_array(
        self, bkd: Backend[Array], store: ResultStore[Array]
    ) -> None:
        lookup = SequencePositionLookup(bkd, _leja_factory(bkd), store, _MAX_LEVEL)
        sample = bkd.array([[0.5], [0.5]])
        with pytest.raises(ValueError, match="alone"):
            lookup.find(sample, sample)

    def test_rejects_wrong_shape(
        self, bkd: Backend[Array], store: ResultStore[Array]
    ) -> None:
        lookup = SequencePositionLookup(bkd, _leja_factory(bkd), store, _MAX_LEVEL)
        with pytest.raises(ValueError, match="shape"):
            lookup.find(bkd.array([0.5, 0.5]))
