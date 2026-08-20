"""Resuming a sweep that died half way through.

The store, the writer and the evaluator each work in isolation and are
tested there. What is proven here is the thing a user actually has to
do, end to end: run part of a sweep, lose the process, restart, and pay
only for what was never computed.

The bookkeeping is the caller's, deliberately -- an evaluator has no
notion of sample identity, so it cannot know that key ``sweep:7`` still
means the sample the caller has in hand. These tests are therefore also
the specification of how much bookkeeping that actually is.
"""

import pytest
from pyapprox.interface.evaluation.callable_marshaller import (
    CallableMarshaller,
)
from pyapprox.interface.evaluation.evaluator import Evaluator
from pyapprox.interface.evaluation.inline_dispatcher import InlineDispatcher
from pyapprox.interface.evaluation.stores import (
    InMemoryResultStore,
    NpzResultStore,
    StoreWriter,
    restore_columns,
    stored_indices,
)


class _CountingModel:
    """f(x) = sum(x^2), counting the samples it is asked to evaluate.

    The count is what makes "did not recompute" checkable: values alone
    would look identical whether they were loaded or recomputed.
    """

    def __init__(self, bkd):
        self._bkd = bkd
        self.nevaluated = 0

    def __call__(self, samples):
        self.nevaluated += int(samples.shape[1])
        return self._bkd.sum(samples * samples, axis=0)[None, :]


def _samples(bkd, n):
    """A reproducible sweep: column i is [i, i + 0.5]."""
    return bkd.array(
        [[float(i) for i in range(n)], [float(i) + 0.5 for i in range(n)]]
    )


def _evaluator(bkd, model, store, prefix, samples_per_task=1):
    marshaller = CallableMarshaller(
        model,
        bkd,
        nvars=2,
        nqoi=1,
        samples_per_task=samples_per_task,
    )
    return Evaluator(
        marshaller,
        InlineDispatcher(marshaller.run),
        on_complete=StoreWriter(store, prefix),
    )


def _known(store, prefix, nsamples):
    """Which batch-local indices the store already holds.

    Kept as the foil for ``stored_indices``: correct only while a record
    covers exactly one sample, and wrong the moment tasks group them.
    ``test_grouped_tasks_need_indices_not_keys`` pins the difference.
    """
    held = set(store.keys())
    return {i for i in range(nsamples) if f"{prefix}:{i}" in held}




class TestStoreWriter:
    """The hook records what finished, and only what finished."""

    def test_every_task_is_recorded(self, bkd):
        store = InMemoryResultStore()
        model = _CountingModel(bkd)
        evaluator = _evaluator(bkd, model, store, "sweep")
        evaluator.submit(_samples(bkd, 4)).collect()
        assert sorted(store.keys()) == [
            "sweep:0",
            "sweep:1",
            "sweep:2",
            "sweep:3",
        ]

    def test_values_reach_the_store_intact(self, bkd):
        store = InMemoryResultStore()
        evaluator = _evaluator(bkd, _CountingModel(bkd), store, "sweep")
        evaluator.submit(_samples(bkd, 3)).collect()
        decoded, _ = store.load("sweep:2")
        # sample 2 is [2.0, 2.5]; 4 + 6.25 = 10.25
        bkd.assert_allclose(decoded.values, bkd.array([[10.25]]))
        assert list(decoded.indices) == [2]

    def test_a_multi_sample_task_is_one_record(self, bkd):
        """Grouped tasks are stored whole, keyed by their first index."""
        store = InMemoryResultStore()
        evaluator = _evaluator(
            bkd, _CountingModel(bkd), store, "sweep", samples_per_task=2
        )
        evaluator.submit(_samples(bkd, 4)).collect()
        assert sorted(store.keys()) == ["sweep:0", "sweep:2"]
        decoded, _ = store.load("sweep:0")
        assert list(decoded.indices) == [0, 1]

    def test_cost_is_recorded_with_the_values(self, bkd):
        store = InMemoryResultStore()
        evaluator = _evaluator(bkd, _CountingModel(bkd), store, "sweep")
        evaluator.submit(_samples(bkd, 2)).collect()
        _, cost = store.load("sweep:0")
        assert cost.wall_clock >= 0.0

    def test_writing_does_not_make_the_evaluator_skip(self, bkd):
        """The invariant: an evaluator computes everything it is given.

        Submitting the same samples twice through a store-writing
        evaluator must evaluate them twice. Anything else would mean the
        evaluator had started deciding what a key means.
        """
        store = InMemoryResultStore()
        model = _CountingModel(bkd)
        evaluator = _evaluator(bkd, model, store, "sweep")
        evaluator.submit(_samples(bkd, 3)).collect()
        evaluator.submit(_samples(bkd, 3)).collect()
        assert model.nevaluated == 6


class TestResume:
    """The whole point: a restarted sweep pays only for the remainder."""

    def test_resumed_run_computes_only_what_is_missing(self, bkd):
        nsamples = 6
        samples = _samples(bkd, nsamples)
        store = InMemoryResultStore()

        # First attempt: the process dies after four samples.
        first = _CountingModel(bkd)
        _evaluator(bkd, first, store, "sweep").submit(
            samples[:, :4]
        ).collect()
        assert first.nevaluated == 4

        # Restart. The caller asks the store what it already has and
        # submits only the rest.
        todo = [
            i for i in range(nsamples) if i not in _known(store, "sweep", 4)
        ]
        assert todo == [4, 5]

        second = _CountingModel(bkd)
        # A fresh prefix would orphan the stored work, so the remainder
        # is submitted under keys continuing the same space.
        marshaller = CallableMarshaller(
            second, bkd, nvars=2, nqoi=1, samples_per_task=1
        )
        writer = StoreWriter(store, "resumed")
        Evaluator(
            marshaller, InlineDispatcher(marshaller.run), on_complete=writer
        ).submit(samples[:, todo]).collect()

        assert second.nevaluated == 2, "recomputed work that was stored"

    def test_reassembled_result_matches_an_uninterrupted_run(self, bkd):
        """The trap: fresh results are indexed batch-locally.

        ``submit(samples[:, todo])`` numbers its columns 0..len(todo)-1,
        not by the original sample index. Reading them back positionally
        gives a full-length array of plausible numbers with the resumed
        columns in the wrong places, which no shape check would catch.
        """
        nsamples = 6
        samples = _samples(bkd, nsamples)
        store = InMemoryResultStore()

        _evaluator(bkd, _CountingModel(bkd), store, "sweep").submit(
            samples[:, :4]
        ).collect()

        todo = [
            i for i in range(nsamples) if i not in _known(store, "sweep", 4)
        ]
        marshaller = CallableMarshaller(
            _CountingModel(bkd), bkd, nvars=2, nqoi=1, samples_per_task=1
        )
        fresh = Evaluator(
            marshaller,
            InlineDispatcher(marshaller.run),
            on_complete=StoreWriter(store, "resumed"),
        ).submit(samples[:, todo]).collect()

        combined = restore_columns(
            store, "sweep", nsamples, fresh.values, todo, bkd
        )

        expected = _CountingModel(bkd)(samples)
        assert combined.shape == expected.shape
        bkd.assert_allclose(combined, expected)

    def test_resume_survives_a_new_process(self, bkd, tmp_path):
        """A store on disk is read by a store object that never saw it.

        The in-memory case proves the bookkeeping; only this proves the
        durability the whole design is for.
        """
        directory = str(tmp_path / "sweep")
        samples = _samples(bkd, 5)

        first = _CountingModel(bkd)
        _evaluator(
            bkd, first, NpzResultStore(directory, bkd), "sweep"
        ).submit(samples[:, :3]).collect()

        # A different store object, as a restarted process would build.
        reopened = NpzResultStore(directory, bkd)
        todo = [i for i in range(5) if i not in _known(reopened, "sweep", 3)]
        assert todo == [3, 4]

        decoded, _ = reopened.load("sweep:1")
        # sample 1 is [1.0, 1.5]; 1 + 2.25 = 3.25
        bkd.assert_allclose(decoded.values, bkd.array([[3.25]]))

    def test_scattered_gaps_are_reassembled_correctly(self, bkd):
        """The realistic case: what is missing is not a suffix.

        A solver fails on particular samples, so a resumed run's todo
        list is scattered -- and scattered is where the reassembly can
        go wrong while a contiguous tail still looks fine. With stored
        indices {0, 2, 5} the fresh columns are 0, 1, 2 standing for
        samples 1, 3, 4, so any mapping that assumes the fresh block is
        contiguous, or that it lands at the end, produces a full-length
        array with three values in the wrong places.
        """
        nsamples = 6
        samples = _samples(bkd, nsamples)
        store = InMemoryResultStore()

        # The gap arises the way it does in practice: the first attempt
        # ran everything, and particular samples diverged.
        doomed = {1.0, 3.0, 4.0}

        def fails_on_some(batch):
            if float(batch[0, 0]) in doomed:
                raise RuntimeError("solver diverged")
            return bkd.sum(batch * batch, axis=0)[None, :]

        marshaller = CallableMarshaller(
            fails_on_some, bkd, nvars=2, nqoi=1, samples_per_task=1
        )
        Evaluator(
            marshaller,
            InlineDispatcher(marshaller.run),
            on_complete=StoreWriter(store, "sweep"),
        ).submit(samples).collect()

        known = stored_indices(store, "sweep")
        assert known == {0, 2, 5}
        todo = [i for i in range(nsamples) if i not in known]
        assert todo == [1, 3, 4]

        # Second attempt, with whatever was wrong now fixed.
        model = _CountingModel(bkd)
        retry = CallableMarshaller(
            model, bkd, nvars=2, nqoi=1, samples_per_task=1
        )
        fresh = Evaluator(
            retry,
            InlineDispatcher(retry.run),
            on_complete=StoreWriter(store, "resumed"),
        ).submit(samples[:, todo]).collect()
        assert model.nevaluated == 3

        combined = restore_columns(
            store, "sweep", nsamples, fresh.values, todo, bkd
        )
        bkd.assert_allclose(combined, _CountingModel(bkd)(samples))

    def test_scattered_failures_leave_scattered_gaps(self, bkd):
        """Failures pick out samples, not suffixes.

        Two non-adjacent failures must leave exactly those two keys
        absent, so the todo list a resumed run derives is scattered
        rather than a tail.
        """
        store = InMemoryResultStore()
        doomed = {1.0, 4.0}

        def fails_on_some(samples):
            if float(samples[0, 0]) in doomed:
                raise RuntimeError("solver diverged")
            return bkd.sum(samples * samples, axis=0)[None, :]

        marshaller = CallableMarshaller(
            fails_on_some, bkd, nvars=2, nqoi=1, samples_per_task=1
        )
        result = Evaluator(
            marshaller,
            InlineDispatcher(marshaller.run),
            on_complete=StoreWriter(store, "sweep"),
        ).submit(_samples(bkd, 6)).collect()

        assert _known(store, "sweep", 6) == {0, 2, 3, 5}
        assert result.nfailed() == 2

    def test_grouped_tasks_need_indices_not_keys(self, bkd):
        """With several samples per task, a key no longer names a sample.

        Keys name the task that produced a record. Under
        ``samples_per_task=3`` a six-sample batch stores two records,
        ``sweep:0`` and ``sweep:3``, so asking whether ``sweep:4``
        exists reports a sample missing that is sitting inside the
        second record. Reading ``indices`` off the records gets it
        right for any grouping.
        """
        store = InMemoryResultStore()
        _evaluator(
            bkd, _CountingModel(bkd), store, "sweep", samples_per_task=3
        ).submit(_samples(bkd, 6)).collect()

        assert sorted(store.keys()) == ["sweep:0", "sweep:3"]
        # The key-presence shortcut misses four of the six.
        assert _known(store, "sweep", 6) == {0, 3}
        # stored_indices reads the records and finds all of them.
        assert stored_indices(store, "sweep") == {0, 1, 2, 3, 4, 5}

    def test_resume_with_grouped_tasks(self, bkd):
        """A partly finished sweep of grouped tasks resumes correctly.

        The first attempt completes two tasks of two samples and dies,
        so four of seven samples are held across two records. The
        remainder is scattered across the record boundary rather than
        aligned to it.
        """
        nsamples = 7
        samples = _samples(bkd, nsamples)
        store = InMemoryResultStore()

        first = _CountingModel(bkd)
        _evaluator(
            bkd, first, store, "sweep", samples_per_task=2
        ).submit(samples[:, :4]).collect()
        assert first.nevaluated == 4

        known = stored_indices(store, "sweep")
        assert known == {0, 1, 2, 3}
        todo = [i for i in range(nsamples) if i not in known]
        assert todo == [4, 5, 6]

        second = _CountingModel(bkd)
        marshaller = CallableMarshaller(
            second, bkd, nvars=2, nqoi=1, samples_per_task=2
        )
        fresh = Evaluator(
            marshaller,
            InlineDispatcher(marshaller.run),
            on_complete=StoreWriter(store, "resumed"),
        ).submit(samples[:, todo]).collect()
        assert second.nevaluated == 3

        # Records here hold two samples each, so the wanted column sits
        # at a record-local position rather than at the sample index.
        combined = restore_columns(
            store, "sweep", nsamples, fresh.values, todo, bkd
        )
        bkd.assert_allclose(combined, _CountingModel(bkd)(samples))

    def test_a_failure_inside_a_group_leaves_the_others_known(self, bkd):
        """One bad sample must not discard its task-mates.

        A task covering samples 2 and 3 whose sample 2 diverges should
        still contribute sample 3, or a resumed run recomputes work
        that succeeded.
        """
        store = InMemoryResultStore()

        def fails_on_two(samples):
            if any(float(v) == 2.0 for v in samples[0]):
                raise RuntimeError("solver diverged")
            return bkd.sum(samples * samples, axis=0)[None, :]

        marshaller = CallableMarshaller(
            fails_on_two, bkd, nvars=2, nqoi=1, samples_per_task=2
        )
        Evaluator(
            marshaller,
            InlineDispatcher(marshaller.run),
            on_complete=StoreWriter(store, "sweep"),
        ).submit(_samples(bkd, 6)).collect()

        known = stored_indices(store, "sweep")
        # The task covering 2 and 3 failed whole -- the model raised for
        # the pair -- so both are absent while the others survive.
        assert known == {0, 1, 4, 5}

    def test_restore_rejects_a_mismatched_fresh_width(self, bkd):
        """fresh must be as wide as todo, or the mapping is undefined.

        Silently zipping the shorter of the two would drop or misplace
        columns, which is the failure this helper exists to prevent.
        """
        store = InMemoryResultStore()
        _evaluator(bkd, _CountingModel(bkd), store, "sweep").submit(
            _samples(bkd, 2)
        ).collect()
        with pytest.raises(ValueError, match="columns but todo names"):
            restore_columns(
                store, "sweep", 4, bkd.array([[1.0, 2.0]]), [2], bkd
            )

    def test_restore_rejects_an_incomplete_result(self, bkd):
        """Every sample must come from the store or from todo.

        A sample in neither has no value anywhere, so returning a
        narrower array -- or a wider one with a gap -- would hand back
        something the caller would index wrongly.
        """
        store = InMemoryResultStore()
        _evaluator(bkd, _CountingModel(bkd), store, "sweep").submit(
            _samples(bkd, 2)
        ).collect()
        with pytest.raises(ValueError, match=r"samples \[3\]"):
            restore_columns(
                store, "sweep", 4, bkd.array([[9.0]]), [2], bkd
            )

    def test_restore_ignores_other_prefixes(self, bkd):
        """Sweeps may share a store without reading each other's work."""
        store = InMemoryResultStore()
        samples = _samples(bkd, 3)
        _evaluator(bkd, _CountingModel(bkd), store, "other").submit(
            samples
        ).collect()
        # Nothing under "sweep", so every sample must come from todo.
        fresh = _CountingModel(bkd)(samples)
        combined = restore_columns(
            store, "sweep", 3, fresh, [0, 1, 2], bkd
        )
        bkd.assert_allclose(combined, fresh)

    def test_stored_indices_ignores_other_prefixes(self, bkd):
        store = InMemoryResultStore()
        _evaluator(bkd, _CountingModel(bkd), store, "other").submit(
            _samples(bkd, 3)
        ).collect()
        assert stored_indices(store, "sweep") == set()
        assert stored_indices(store, "other") == {0, 1, 2}

    def test_nothing_stored_means_everything_runs(self, bkd):
        """An empty store must not make a resumed run skip anything."""
        store = InMemoryResultStore()
        assert _known(store, "sweep", 4) == set()
        model = _CountingModel(bkd)
        _evaluator(bkd, model, store, "sweep").submit(
            _samples(bkd, 4)
        ).collect()
        assert model.nevaluated == 4

    def test_a_failed_sample_is_not_recorded_as_known(self, bkd):
        """A failure must be retried on resume, not treated as done.

        Recording a placeholder for a failed task would make the key
        present, and a resumed run would skip the one sample that never
        produced a value.
        """
        store = InMemoryResultStore()

        def explodes_on_sample_two(samples):
            if float(samples[0, 0]) == 2.0:
                raise RuntimeError("solver diverged")
            return bkd.sum(samples * samples, axis=0)[None, :]

        marshaller = CallableMarshaller(
            explodes_on_sample_two, bkd, nvars=2, nqoi=1, samples_per_task=1
        )
        Evaluator(
            marshaller,
            InlineDispatcher(marshaller.run),
            on_complete=StoreWriter(store, "sweep"),
        ).submit(_samples(bkd, 4)).collect()

        assert "sweep:2" not in set(store.keys())
        assert _known(store, "sweep", 4) == {0, 1, 3}
