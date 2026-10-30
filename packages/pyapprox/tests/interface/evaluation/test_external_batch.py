"""Exporting samples for evaluation outside pyapprox and importing results.

``ExternalBatchModel`` sits behind ``CachedObjective``; results come back
through ``import_results``. The batch directory is the only state, so a
second model object on the same directory stands in for a new process.

Tests run on both NumPy and PyTorch backends.
"""

import os
from pathlib import Path
from typing import List

import numpy as np
import pytest

from pyapprox.interface.evaluation.caching import CachedObjective, RoundedHashLookup
from pyapprox.interface.evaluation.external_batch import (
    ExternalBatchModel,
    PendingEvaluations,
)
from pyapprox.interface.evaluation.stores import InMemoryResultStore
from pyapprox.interface.functions.protocols.objective import ObjectiveProtocol
from pyapprox.util.backends.protocols import Array, Backend

_NVARS = 2
_NQOI = 2


def _true_values(samples: np.ndarray) -> np.ndarray:
    """What the external code computes, shape (nqoi, n)."""
    return np.vstack([samples.sum(axis=0), (samples**2).sum(axis=0)])


def _write_results(path: str, keys: List[str], samples: np.ndarray) -> str:
    np.savez(path, keys=np.asarray(keys, dtype=str), values=_true_values(samples))
    return path


def _evaluate_batch(batch_path: str, results_path: str) -> str:
    """Evaluate every sample in a batch file, as the user's code would."""
    with np.load(batch_path, allow_pickle=False) as data:
        keys = [str(key) for key in data["keys"]]
        samples = data["samples"]
    return _write_results(results_path, keys, samples)


class _Setup:
    def __init__(self, bkd: Backend[Array], directory: Path) -> None:
        self.bkd = bkd
        self.lookup = RoundedHashLookup(bkd)
        self.store: InMemoryResultStore[Array] = InMemoryResultStore()
        self.external = ExternalBatchModel(
            bkd, _NVARS, _NQOI, self.lookup, str(directory / "batches")
        )
        self.model = CachedObjective(self.external, self.store, self.lookup)
        self.directory = directory

    def reopen(self) -> "_Setup":
        """A new process: new objects, the same directory and store."""
        other = _Setup(self.bkd, self.directory)
        other.store = self.store
        other.model = CachedObjective(other.external, self.store, other.lookup)
        return other

    def results_path(self, name: str) -> str:
        return str(self.directory / name)


@pytest.fixture
def setup(bkd: Backend[Array], tmp_path: Path) -> _Setup:
    return _Setup(bkd, tmp_path)


def _samples(bkd: Backend[Array]) -> Array:
    return bkd.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


class TestExport:
    def test_is_an_objective_without_derivatives(self, setup: _Setup) -> None:
        assert isinstance(setup.external, ObjectiveProtocol)
        assert setup.external.derivatives().jacobian is None

    def test_writes_a_batch_and_raises(self, setup: _Setup) -> None:
        with pytest.raises(PendingEvaluations) as pending:
            setup.model(_samples(setup.bkd))
        assert pending.value.nsamples() == 3
        assert os.path.basename(pending.value.batch_path()) == "batch-0000.npz"
        with np.load(pending.value.batch_path(), allow_pickle=False) as data:
            setup.bkd.assert_allclose(
                setup.bkd.asarray(data["samples"]), _samples(setup.bkd)
            )
            assert len(data["keys"]) == 3
        assert list(setup.store.keys()) == []

    def test_asking_again_returns_the_same_batch(self, setup: _Setup) -> None:
        """A rerun before importing must not write a duplicate batch."""
        with pytest.raises(PendingEvaluations) as first:
            setup.model(_samples(setup.bkd))
        with pytest.raises(PendingEvaluations) as again:
            setup.reopen().model(_samples(setup.bkd))
        assert again.value.batch_path() == first.value.batch_path()
        assert sorted(os.listdir(setup.external.directory())) == ["batch-0000.npz"]

    def test_only_unstored_samples_are_exported(self, setup: _Setup) -> None:
        samples = _samples(setup.bkd)
        with pytest.raises(PendingEvaluations) as pending:
            setup.model(samples[:, :2])
        setup.external.import_results(
            _evaluate_batch(pending.value.batch_path(), setup.results_path("r.npz")),
            setup.model,
        )
        with pytest.raises(PendingEvaluations) as second:
            setup.model(samples)
        assert second.value.nsamples() == 1


class TestImport:
    def test_round_trip_serves_the_imported_values(self, setup: _Setup) -> None:
        samples = _samples(setup.bkd)
        with pytest.raises(PendingEvaluations) as pending:
            setup.model(samples)
        results = _evaluate_batch(
            pending.value.batch_path(), setup.results_path("r.npz")
        )
        assert setup.external.import_results(results, setup.model) == 3
        expected = setup.bkd.asarray(_true_values(setup.bkd.to_numpy(samples)))
        setup.bkd.assert_allclose(setup.model(samples), expected)
        # And from a new process, served without exporting anything.
        setup.bkd.assert_allclose(setup.reopen().model(samples), expected)

    def test_results_in_any_order(self, setup: _Setup) -> None:
        samples = _samples(setup.bkd)
        with pytest.raises(PendingEvaluations) as pending:
            setup.model(samples)
        with np.load(pending.value.batch_path(), allow_pickle=False) as data:
            keys = [str(key) for key in data["keys"]][::-1]
            batch_samples = data["samples"][:, ::-1]
        setup.external.import_results(
            _write_results(setup.results_path("r.npz"), keys, batch_samples),
            setup.model,
        )
        expected = setup.bkd.asarray(_true_values(setup.bkd.to_numpy(samples)))
        setup.bkd.assert_allclose(setup.model(samples), expected)

    def test_partial_import_re_exports_only_the_missing(self, setup: _Setup) -> None:
        samples = _samples(setup.bkd)
        with pytest.raises(PendingEvaluations) as pending:
            setup.model(samples)
        with np.load(pending.value.batch_path(), allow_pickle=False) as data:
            keys = [str(key) for key in data["keys"]]
            batch_samples = data["samples"]
        setup.external.import_results(
            _write_results(
                setup.results_path("r0.npz"), keys[:2], batch_samples[:, :2]
            ),
            setup.model,
        )
        with pytest.raises(PendingEvaluations) as rest:
            setup.reopen().model(samples)
        assert rest.value.nsamples() == 1
        assert os.path.basename(rest.value.batch_path()) == "batch-0001.npz"
        setup.external.import_results(
            _evaluate_batch(rest.value.batch_path(), setup.results_path("r1.npz")),
            setup.model,
        )
        expected = setup.bkd.asarray(_true_values(setup.bkd.to_numpy(samples)))
        setup.bkd.assert_allclose(setup.model(samples), expected)

    def test_unrequested_key_is_rejected(self, setup: _Setup) -> None:
        with pytest.raises(PendingEvaluations):
            setup.model(_samples(setup.bkd))
        bogus = _write_results(
            setup.results_path("r.npz"), ["never-asked"], np.array([[0.9], [0.9]])
        )
        with pytest.raises(ValueError, match="never requested"):
            setup.external.import_results(bogus, setup.model)
        assert list(setup.store.keys()) == []

    def test_wrong_values_shape_is_rejected(self, setup: _Setup) -> None:
        with pytest.raises(PendingEvaluations) as pending:
            setup.model(_samples(setup.bkd))
        with np.load(pending.value.batch_path(), allow_pickle=False) as data:
            keys = [str(key) for key in data["keys"]]
        path = setup.results_path("r.npz")
        np.savez(path, keys=np.asarray(keys), values=np.zeros((1, len(keys))))
        with pytest.raises(ValueError, match="shape"):
            setup.external.import_results(path, setup.model)

    def test_into_must_wrap_this_model(self, setup: _Setup) -> None:
        other = CachedObjective(
            ExternalBatchModel(
                setup.bkd,
                _NVARS,
                _NQOI,
                setup.lookup,
                str(setup.directory / "other"),
            ),
            setup.store,
            setup.lookup,
        )
        with pytest.raises(ValueError, match="wrapping this model"):
            setup.external.import_results(setup.results_path("r.npz"), other)


class TestDivergence:
    def test_asking_for_other_samples_while_outstanding(self, setup: _Setup) -> None:
        """A replay that diverges must not export a second, unrelated batch."""
        with pytest.raises(PendingEvaluations):
            setup.model(_samples(setup.bkd))
        with pytest.raises(RuntimeError, match="outstanding"):
            setup.reopen().model(setup.bkd.array([[0.7], [0.8]]))
        assert sorted(os.listdir(setup.external.directory())) == ["batch-0000.npz"]

    def test_rejects_wrong_number_of_variables(self, setup: _Setup) -> None:
        with pytest.raises(ValueError, match="shape"):
            setup.external(setup.bkd.array([[0.1, 0.2]]))
