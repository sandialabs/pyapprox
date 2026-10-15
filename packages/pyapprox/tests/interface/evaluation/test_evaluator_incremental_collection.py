"""Collection harvests each task as it finishes, not the batch at the end.

A completion hook is where a finished task becomes durable -- written
to a store a restarted run can read. If collection waited on every task
before harvesting any, a driver killed while waiting on the last task of
a long batch would have recorded none of the tasks that finished before
it. So each task is decoded, charged, passed to the hook and released as
soon as its outcome is in.

A stub dispatcher makes the order observable: its handles log each time
they are waited on, and one can be told to raise from ``outcome`` -- a
``BaseException`` for a killed process, which nothing catches, or an
ordinary exception a caller survives and collects past.
"""

from typing import Any, Callable, List, Optional, Sequence, Set, Type

import pytest

from pyapprox.interface.evaluation.callable_marshaller import (
    CallableMarshaller,
)
from pyapprox.interface.evaluation.evaluator import Evaluator
from pyapprox.interface.evaluation.records import (
    ComputeProvenance,
    JobStatus,
    Outcome,
)
from pyapprox.util.backends.protocols import Backend


class _Killed(BaseException):
    """A killed process, as the code it was running sees it."""


class _Interrupted(Exception):
    """An interruption the caller catches and carries on from."""


class _LoggingHandle:
    """A finished handle that logs being waited on, and may raise once."""

    def __init__(
        self,
        outcome: Outcome[Any, Any],
        events: List[str],
        raises: Optional[Type[BaseException]],
    ) -> None:
        self._outcome = outcome
        self._events = events
        self._raises = raises

    def done(self) -> bool:
        return True

    def outcome(self, timeout: Optional[float] = None) -> Outcome[Any, Any]:
        self._events.append(f"wait:{self._outcome.indices[0]}")
        if self._raises is not None:
            raises, self._raises = self._raises, None
            raise raises()
        return self._outcome

    def cancel(self) -> bool:
        return False


class _LoggingDispatcher:
    """Runs every task, and hands back handles that log being waited on;
    the one at ``raise_at`` raises ``raises`` the first time."""

    def __init__(
        self,
        run: Callable[[Any], Any],
        events: List[str],
        raise_at: Optional[int] = None,
        raises: Type[BaseException] = _Killed,
    ) -> None:
        self._run = run
        self._events = events
        self._raise_at = raise_at
        self._raises = raises

    def concurrency(self) -> int:
        return 1

    def compute_provenance(self) -> ComputeProvenance:
        return ComputeProvenance.MEASURED

    def submit(self, tasks: Sequence[Any]) -> List[_LoggingHandle]:
        handles = []
        for position, task in enumerate(tasks):
            payload = self._run(task)
            handles.append(
                _LoggingHandle(
                    Outcome(
                        task=task,
                        indices=task.indices,
                        status=JobStatus.SUCCEEDED,
                        payload=payload,
                        wall_time=0.0,
                    ),
                    self._events,
                    self._raises if position == self._raise_at else None,
                )
            )
        return handles

    def close(self) -> None:
        return None


def _evaluator(
    bkd: Backend[Any],
    events: List[str],
    recorded: Set[int],
    **dispatcher_kwargs: Any,
) -> Evaluator[Any, Any, Any]:
    def model(samples: Any) -> Any:
        return bkd.sum(samples * samples, axis=0)[None, :]

    def hook(outcome: Outcome[Any, Any], decoded: Any, cost: Any) -> None:
        events.append(f"hook:{outcome.indices[0]}")
        if decoded is not None:
            recorded.update(decoded.indices)

    marshaller = CallableMarshaller(
        model, bkd, nvars=2, nqoi=1, samples_per_task=1
    )
    return Evaluator(
        marshaller,
        _LoggingDispatcher(marshaller.run, events, **dispatcher_kwargs),
        on_complete=hook,
    )


def _samples(bkd: Backend[Any], n: int) -> Any:
    return bkd.array([[float(j) for j in range(n)] for _ in range(2)])


class TestEachTaskIsHarvestedAsItFinishes:
    def test_a_task_is_recorded_before_the_next_is_waited_on(
        self, numpy_bkd: Backend[Any]
    ) -> None:
        events: List[str] = []
        _evaluator(numpy_bkd, events, set()).submit(
            _samples(numpy_bkd, 4)
        ).collect()
        for k in range(3):
            assert events.index(f"hook:{k}") < events.index(f"wait:{k + 1}")

    def test_a_driver_killed_mid_collect_has_recorded_what_finished(
        self, numpy_bkd: Backend[Any]
    ) -> None:
        """Killed while waiting on task 3: tasks 0-2 are already kept."""
        recorded: Set[int] = set()
        batch = _evaluator(numpy_bkd, [], recorded, raise_at=3).submit(
            _samples(numpy_bkd, 6)
        )
        with pytest.raises(_Killed):
            batch.collect()
        assert recorded == {0, 1, 2}


class TestAnInterruptedCollectionLosesNothing:
    def test_every_index_is_returned_once_across_the_interruption(
        self, bkd: Backend[Any]
    ) -> None:
        """Tasks absorbed before the interruption are not marked returned
        and dropped: the next collection returns them, with the rest, in
        index order with their values."""
        X = _samples(bkd, 5)
        batch = _evaluator(
            bkd, [], set(), raise_at=2, raises=_Interrupted
        ).submit(X)
        with pytest.raises(_Interrupted):
            batch.collect()
        result = batch.collect()
        bkd.assert_allclose(result.succeeded, bkd.asarray([0, 1, 2, 3, 4]))
        bkd.assert_allclose(result.values, bkd.sum(X * X, axis=0)[None, :])
        assert sorted(batch.statuses()) == [0, 1, 2, 3, 4]
        assert batch.collect().nreturned() == 0
