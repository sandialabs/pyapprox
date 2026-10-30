"""Refining to a budget, interrupted by evaluations outside pyapprox.

``refine_with_model`` drives a fitter whose model is an
``ExternalBatchModel`` behind ``CachedObjective`` and a
``SequencePositionLookup``. Each interruption exports a batch; the test
evaluates it as outside code would, imports the results and calls again.
Every run is compared with one that was never interrupted.

Tests run on both NumPy and PyTorch backends, except the two-process
round trip, which runs the NumPy backend in fresh interpreters.
"""

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Generic, List, Set

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.interface.evaluation.caching import CachedObjective
from pyapprox.interface.evaluation.external_batch import (
    ExternalBatchModel,
    PendingEvaluations,
)
from pyapprox.interface.evaluation.stores import NpzResultStore
from pyapprox.interface.functions.fromcallable.function import FunctionFromCallable
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import LinearGrowthRule, MaxLevelCriteria
from pyapprox.surrogates.sparsegrids.adaptive_fitter import (
    SingleFidelityAdaptiveSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.basis_factory import LejaLagrangeFactory
from pyapprox.surrogates.sparsegrids.error_indicators import L2SurplusIndicator
from pyapprox.surrogates.sparsegrids.refine import refine_with_model
from pyapprox.surrogates.sparsegrids.sequence_lookup import SequencePositionLookup
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.markers import slower_test

_NVARS = 2
_MAX_LEVEL = 6
_BUDGET = 20


def _target_numpy(
    samples: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """The model, as outside code computes it: (nvars, n) to (1, n)."""
    values: NDArray[np.floating[Any]] = (
        np.exp(0.7 * samples[0] + 0.3 * samples[1]) + 0.2 * samples[0] * samples[1]
    )
    return values[None, :]


class _Target(Generic[Array]):
    """The same model in pyapprox, counting the samples it evaluates."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._nevaluated = 0

    def nevaluated(self) -> int:
        return self._nevaluated

    def __call__(self, samples: Array) -> Array:
        self._nevaluated += samples.shape[1]
        return self._bkd.asarray(_target_numpy(self._bkd.to_numpy(samples)))


def _factory(bkd: Backend[Array]) -> TensorProductSubspaceFactory[Array]:
    leja = LejaLagrangeFactory(UniformMarginal(0.0, 1.0, bkd), bkd)
    return TensorProductSubspaceFactory(
        bkd, [leja] * _NVARS, LinearGrowthRule(scale=2, shift=1)
    )


def _fitter(bkd: Backend[Array]) -> SingleFidelityAdaptiveSparseGridFitter[Array]:
    return SingleFidelityAdaptiveSparseGridFitter(
        bkd,
        _factory(bkd),
        MaxLevelCriteria(_MAX_LEVEL, 1.0, bkd),
        error_indicator=L2SurplusIndicator(bkd),
    )


def _uninterrupted(
    bkd: Backend[Array],
) -> SingleFidelityAdaptiveSparseGridFitter[Array]:
    fitter = _fitter(bkd)
    refine_with_model(
        fitter, FunctionFromCallable(1, _NVARS, _Target(bkd), bkd), _BUDGET
    )
    return fitter


def _evaluate_batch(batch_path: str, results_path: str) -> List[str]:
    """Evaluate a batch file as outside code would; return its keys."""
    with np.load(batch_path, allow_pickle=False) as data:
        keys = [str(key) for key in data["keys"]]
        samples = data["samples"]
    np.savez(
        results_path, keys=np.asarray(keys, dtype=str), values=_target_numpy(samples)
    )
    return keys


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


class TestRefineWithModel:
    def test_reaches_the_budget(self, bkd: Backend[Array]) -> None:
        fitter = _fitter(bkd)
        target = _Target(bkd)
        nsamples = refine_with_model(
            fitter, FunctionFromCallable(1, _NVARS, target, bkd), _BUDGET
        )
        assert nsamples == fitter.get_samples("all").shape[1]
        assert nsamples >= _BUDGET
        assert target.nevaluated() == nsamples

    def test_matches_a_hand_written_loop(self, bkd: Backend[Array]) -> None:
        by_hand = _fitter(bkd)
        target = _Target(bkd)
        while True:
            samples = by_hand.step_samples()
            if samples is None:
                break
            by_hand.step_values(target(samples))
            if by_hand.get_samples("all").shape[1] >= _BUDGET:
                break
        _assert_same_grid(bkd, _uninterrupted(bkd), by_hand)

    def test_finishes_a_pending_batch_first(self, bkd: Backend[Array]) -> None:
        """A batch requested by hand is completed, not replaced."""
        fitter = _fitter(bkd)
        assert fitter.step_samples() is not None
        refine_with_model(
            fitter, FunctionFromCallable(1, _NVARS, _Target(bkd), bkd), _BUDGET
        )
        _assert_same_grid(bkd, fitter, _uninterrupted(bkd))

    def test_called_again_at_budget_takes_no_step(self, bkd: Backend[Array]) -> None:
        fitter = _uninterrupted(bkd)
        target = _Target(bkd)
        refine_with_model(fitter, FunctionFromCallable(1, _NVARS, target, bkd), _BUDGET)
        assert target.nevaluated() == 0

    def test_rejects_a_non_function(self, bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="FunctionProtocol"):
            refine_with_model(_fitter(bkd), object(), _BUDGET)  # type: ignore[arg-type]

    def test_rejects_a_non_positive_budget(self, bkd: Backend[Array]) -> None:
        model = FunctionFromCallable(1, _NVARS, _Target(bkd), bkd)
        with pytest.raises(ValueError, match="positive"):
            refine_with_model(_fitter(bkd), model, 0)


class TestInterruptedInOneProcess:
    def test_matches_an_uninterrupted_run(
        self, bkd: Backend[Array], tmp_path: Path
    ) -> None:
        """Same grid; every sample exported, evaluated and imported once."""
        store = NpzResultStore(str(tmp_path / "store"), bkd)
        lookup = SequencePositionLookup(bkd, _factory(bkd), store, _MAX_LEVEL)
        external = ExternalBatchModel(bkd, _NVARS, 1, lookup, str(tmp_path / "batches"))
        model = CachedObjective(external, store, lookup)
        fitter = _fitter(bkd)

        evaluated: List[str] = []
        ninterruptions = 0
        while True:
            try:
                refine_with_model(fitter, model, _BUDGET)
                break
            except PendingEvaluations as pending:
                ninterruptions += 1
                results = str(tmp_path / f"results-{ninterruptions}.npz")
                evaluated += _evaluate_batch(pending.batch_path(), results)
                external.import_results(results, model)

        _assert_same_grid(bkd, fitter, _uninterrupted(bkd))
        assert ninterruptions > 1
        assert len(evaluated) == len(set(evaluated))
        assert len(evaluated) == fitter.get_samples("all").shape[1]


_SCRIPT = textwrap.dedent(
    """
    import sys

    import numpy as np

    from pyapprox.interface.evaluation.caching import CachedObjective
    from pyapprox.interface.evaluation.external_batch import (
        ExternalBatchModel,
        PendingEvaluations,
    )
    from pyapprox.interface.evaluation.stores import NpzResultStore
    from pyapprox.probability import UniformMarginal
    from pyapprox.surrogates.affine.indices import (
        LinearGrowthRule,
        MaxLevelCriteria,
    )
    from pyapprox.surrogates.sparsegrids.adaptive_fitter import (
        SingleFidelityAdaptiveSparseGridFitter,
    )
    from pyapprox.surrogates.sparsegrids.basis_factory import LejaLagrangeFactory
    from pyapprox.surrogates.sparsegrids.error_indicators import (
        L2SurplusIndicator,
    )
    from pyapprox.surrogates.sparsegrids.refine import refine_with_model
    from pyapprox.surrogates.sparsegrids.sequence_lookup import (
        SequencePositionLookup,
    )
    from pyapprox.surrogates.sparsegrids.subspace_factory import (
        TensorProductSubspaceFactory,
    )
    from pyapprox.util.backends.numpy import NumpyBkd

    directory, budget, max_level = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    results = sys.argv[4:]
    bkd = NumpyBkd()
    leja = LejaLagrangeFactory(UniformMarginal(0.0, 1.0, bkd), bkd)
    factory = TensorProductSubspaceFactory(
        bkd, [leja, leja], LinearGrowthRule(scale=2, shift=1)
    )
    store = NpzResultStore(directory + "/store", bkd)
    lookup = SequencePositionLookup(bkd, factory, store, max_level)
    external = ExternalBatchModel(bkd, 2, 1, lookup, directory + "/batches")
    model = CachedObjective(external, store, lookup)
    for path in results:
        external.import_results(path, model)
    fitter = SingleFidelityAdaptiveSparseGridFitter(
        bkd,
        factory,
        MaxLevelCriteria(max_level, 1.0, bkd),
        error_indicator=L2SurplusIndicator(bkd),
    )
    try:
        refine_with_model(fitter, model, budget)
    except PendingEvaluations as pending:
        print("PENDING " + pending.batch_path())
        sys.exit(3)
    np.savez(
        directory + "/final.npz",
        samples=fitter.get_samples("all"),
        values=fitter.get_values(),
        selected=fitter.get_selected_indices(),
    )
    print("DONE")
    """
)


class TestInterruptedAcrossProcesses:
    # About 30 s: each round trip starts a fresh interpreter.
    @slower_test
    def test_matches_an_uninterrupted_run(
        self, numpy_bkd: Backend[Array], tmp_path: Path
    ) -> None:
        """Each run is a fresh interpreter that replays from the store."""
        script = tmp_path / "refine_script.py"
        script.write_text(_SCRIPT)
        evaluated: Set[str] = set()
        nevaluated = 0
        imports: List[str] = []
        for _ in range(50):
            run = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    str(tmp_path),
                    str(_BUDGET),
                    str(_MAX_LEVEL),
                    *imports,
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            imports = []
            if run.returncode == 0:
                break
            assert run.returncode == 3, run.stderr[-2000:]
            batch_path = run.stdout.strip().split("PENDING ", 1)[1]
            results = str(tmp_path / f"results-{len(evaluated)}.npz")
            keys = _evaluate_batch(batch_path, results)
            nevaluated += len(keys)
            evaluated.update(keys)
            imports = [results]
        else:
            raise AssertionError("refinement did not finish in 50 runs")

        with np.load(tmp_path / "final.npz", allow_pickle=False) as data:
            samples = numpy_bkd.asarray(data["samples"])
            values = numpy_bkd.asarray(data["values"])
            selected = numpy_bkd.asarray(data["selected"])
        desired = _uninterrupted(numpy_bkd)
        numpy_bkd.assert_allclose(samples, desired.get_samples("all"), rtol=1e-12)
        desired_values = desired.get_values()
        assert desired_values is not None
        numpy_bkd.assert_allclose(values, desired_values, rtol=1e-12)
        assert numpy_bkd.all_bool(
            numpy_bkd.equal(selected, desired.get_selected_indices())
        )
        # Every sample evaluated exactly once across all the processes.
        assert nevaluated == len(evaluated) == samples.shape[1]
