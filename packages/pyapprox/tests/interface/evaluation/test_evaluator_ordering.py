"""Results must follow their indices, not their completion order.

The inline dispatcher always finishes in submission order, so it cannot
produce the case that matters. A stub dispatcher can: it hands back
handles that complete in a scrambled order, which is what a real
scheduler does under priority queueing, retry-at-end, or simply
heterogeneous run times.

This is the property ``Outcome.indices`` exists for. Recovering the
mapping by sorting job ids, or by assuming completion order matches
dispatch order, silently scrambles results with no error -- so it is
worth a dedicated dispatcher to test.

Array-free where possible, but these assert on assembled values, so they
use the ``bkd`` fixture.
"""

from typing import Optional, Sequence

import pytest
from pyapprox.interface.evaluation.callable_marshaller import (
    CallableMarshaller,
)
from pyapprox.interface.evaluation.evaluator import Evaluator
from pyapprox.interface.evaluation.records import (
    ComputeProvenance,
    JobStatus,
    Outcome,
    Request,
)
from pyapprox.interface.functions.derivatives import Derivatives


class _ScrambledHandle:
    """A handle that is already finished, holding a prepared outcome."""

    def __init__(self, outcome):
        self._outcome = outcome

    def done(self) -> bool:
        return True

    def outcome(self, timeout: Optional[float] = None):
        return self._outcome

    def cancel(self) -> bool:
        return False


class _ScrambledDispatcher:
    """Runs tasks, then returns their handles in a chosen order.

    The reordering is of the *handles*, not the work: every task still
    runs, and each outcome still carries its own indices. What changes is
    the order in which the evaluator meets them, which is exactly what a
    scheduler varies and what nothing else here exercises.
    """

    def __init__(self, run, order=None):
        self._run = run
        self._order = order

    def concurrency(self) -> int:
        return 1

    def compute_provenance(self) -> ComputeProvenance:
        return ComputeProvenance.MEASURED

    def submit(self, tasks: Sequence[object]) -> Sequence[_ScrambledHandle]:
        handles = []
        for task in tasks:
            payload = self._run(task)
            handles.append(
                _ScrambledHandle(
                    Outcome(
                        task=task,
                        indices=task.indices,
                        status=JobStatus.SUCCEEDED,
                        payload=payload,
                        wall_time=0.0,
                    )
                )
            )
        if self._order is None:
            return list(reversed(handles))
        return [handles[i] for i in self._order]

    def close(self) -> None:
        return None


def _evaluator(bkd, order=None, samples_per_task=1):
    def model(samples):
        return bkd.sum(samples * samples, axis=0)[None, :]

    marshaller = CallableMarshaller(
        model, bkd, nvars=2, nqoi=1, samples_per_task=samples_per_task
    )
    return Evaluator(
        marshaller, _ScrambledDispatcher(marshaller.run, order)
    )


def _evaluator_with(bkd, derivatives, samples_per_task=1, order=None):
    """A scrambling evaluator over a model with derivative capability."""

    def model(samples):
        return bkd.sum(samples * samples, axis=0)[None, :]

    marshaller = CallableMarshaller(
        model,
        bkd,
        nvars=2,
        nqoi=1,
        samples_per_task=samples_per_task,
        derivatives=derivatives,
    )
    return Evaluator(
        marshaller, _ScrambledDispatcher(marshaller.run, order)
    )


def _columns(bkd, n):
    """(2, n) whose column j sums to 2*j^2 -- every value distinct."""
    return bkd.array([[float(j) for j in range(n)] for _ in range(2)])


class TestOutOfOrderCompletion:
    def test_values_follow_indices_not_arrival(self, bkd):
        """The headline: reversed arrival must not reverse the values."""
        X = _columns(bkd, 5)
        result = _evaluator(bkd).submit(X).collect()
        bkd.assert_allclose(result.succeeded, bkd.asarray([0, 1, 2, 3, 4]))
        bkd.assert_allclose(result.values, bkd.sum(X * X, axis=0)[None, :])

    @pytest.mark.parametrize(
        "order",
        [
            [3, 1, 0, 2],
            [2, 3, 1, 0],
            [1, 0, 3, 2],
            [0, 1, 2, 3],
        ],
    )
    def test_every_arrival_order_gives_the_same_result(self, bkd, order):
        """Permuting arrival must not change anything a caller sees."""
        X = _columns(bkd, 4)
        result = _evaluator(bkd, order=order).submit(X).collect()
        bkd.assert_allclose(result.succeeded, bkd.asarray([0, 1, 2, 3]))
        bkd.assert_allclose(result.values, bkd.sum(X * X, axis=0)[None, :])

    def test_streaming_out_of_order_still_covers_the_batch_once(self, bkd):
        """Partial collection under reordering keeps the union exact."""
        X = _columns(bkd, 6)
        batch = _evaluator(bkd).submit(X)
        first = batch.collect_ready()
        second = batch.collect()
        seen = sorted(
            [int(i) for i in first.succeeded]
            + [int(i) for i in second.succeeded]
        )
        assert seen == [0, 1, 2, 3, 4, 5]

    def test_multi_sample_tasks_keep_their_own_indices(self, bkd):
        """Reordered tasks each covering several columns."""
        X = _columns(bkd, 6)
        result = (
            _evaluator(bkd, samples_per_task=2).submit(X).collect()
        )
        bkd.assert_allclose(
            result.succeeded, bkd.asarray([0, 1, 2, 3, 4, 5])
        )
        bkd.assert_allclose(result.values, bkd.sum(X * X, axis=0)[None, :])

    def test_a_single_task_is_unaffected(self, bkd):
        """Sanity: nothing to reorder when the batch is one task."""
        X = _columns(bkd, 4)
        result = (
            _evaluator(bkd, samples_per_task=100).submit(X).collect()
        )
        bkd.assert_allclose(result.values, bkd.sum(X * X, axis=0)[None, :])


class TestDerivativeReassembly:
    """Derivatives reassemble on their own sample axis, in index order.

    The four fields split across two axes: ``jacobians`` and
    ``hessians`` are sample-**first**, ``(n, ...)``, while ``jvps`` and
    ``hvps`` are sample-**last**, ``(..., n)`` -- the same convention as
    ``values``. One test per *family* is the minimum: the two members of
    a family share a code path, but the families do not, and using the
    wrong axis transposes silently rather than raising.

    Only visible when a batch spans several tasks, since a single task
    is returned whole.
    """

    def test_jacobians_reassemble_sample_first(self, bkd):
        """Axis 0: (n, nqoi, nvars)."""

        def jac_batch(samples):
            n = samples.shape[1]
            return bkd.reshape(2.0 * samples.T, (n, 1, 2))

        ev = _evaluator_with(
            bkd, Derivatives(jacobian_batch=jac_batch), samples_per_task=1
        )
        X = _columns(bkd, 4)
        result = ev.submit(X, Request(jacobians=True)).collect()
        assert result.jacobians is not None
        assert result.jacobians.shape == (4, 1, 2)
        bkd.assert_allclose(
            result.jacobians, bkd.reshape(2.0 * X.T, (4, 1, 2))
        )

    def test_jvps_reassemble_sample_last(self, bkd):
        """Axis 1: (nqoi, n). The family jacobians alone cannot cover."""

        def jvp(samples, vecs):
            return bkd.sum(2.0 * samples * vecs, axis=0)[None, :]

        ev = _evaluator_with(
            bkd, Derivatives(jvp=jvp), samples_per_task=1
        )
        X = _columns(bkd, 4)
        vecs = bkd.ones((2, 4))
        result = ev.submit(X, Request(jvp_vecs=vecs)).collect()
        assert result.jvps is not None
        assert result.jvps.shape == (1, 4)
        bkd.assert_allclose(
            result.jvps, bkd.sum(2.0 * X * vecs, axis=0)[None, :]
        )

    def test_hvps_reassemble_sample_first(self, bkd):
        """Axis 0: (n, nvars) -- scalar-implicit, no nqoi axis.

        The trap the derivative bundle names explicitly: the batch form
        is not the single form widened. ``hvp`` is ``(nvars, 1)`` while
        ``hvp_batch`` is ``(n, nvars)``, so assuming the directional
        capabilities share an axis convention transposes the result.
        """

        def hvp_batch(samples, vecs):
            return (2.0 * vecs).T

        ev = _evaluator_with(
            bkd, Derivatives(hvp_batch=hvp_batch), samples_per_task=1
        )
        X = _columns(bkd, 3)
        vecs = bkd.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = ev.submit(X, Request(hvp_vecs=vecs)).collect()
        assert result.hvps is not None
        assert result.hvps.shape == (3, 2)
        bkd.assert_allclose(result.hvps, (2.0 * vecs).T)

    def test_weighted_hvp_records_its_weights(self, bkd):
        """A stored result must say what its hvps were weighted by."""

        def whvp_batch(samples, vecs, weights):
            return (2.0 * vecs).T

        ev = _evaluator_with(
            bkd, Derivatives(whvp_batch=whvp_batch), samples_per_task=1
        )
        weights = bkd.ones((1, 1))
        result = ev.submit(
            _columns(bkd, 3),
            Request(hvp_vecs=bkd.ones((2, 3)), hvp_weights=weights),
        ).collect()
        assert result.hvp_weights is not None
        bkd.assert_allclose(result.hvp_weights, weights)

    def test_plain_hvp_records_no_weights(self, bkd):
        """None distinguishes an unweighted product from a weighted one."""

        def hvp_batch(samples, vecs):
            return (2.0 * vecs).T

        ev = _evaluator_with(
            bkd, Derivatives(hvp_batch=hvp_batch), samples_per_task=1
        )
        result = ev.submit(
            _columns(bkd, 3), Request(hvp_vecs=bkd.ones((2, 3)))
        ).collect()
        assert result.hvps is not None
        assert result.hvp_weights is None

    def test_absent_derivatives_stay_none(self, bkd):
        """None means the capability was absent, not that it returned
        nothing."""
        result = _evaluator(bkd).submit(_columns(bkd, 3)).collect()
        assert result.jacobians is None
        assert result.jvps is None
        assert result.hvps is None
