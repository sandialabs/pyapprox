"""The protocols are satisfiable, and the pairing is checked statically.

Protocols typecheck trivially on their own; what confirms a design is
something checked *against* them. These stubs are signature-only
implementations of the two dispatchers that land later, plus two
marshallers producing *different* task families. They exist so that:

- mypy proves the protocol surface is implementable without ``cast``,
  ``type: ignore`` or ``Any``;
- ``isinstance`` proves the runtime backstop works on the bare
  protocols;
- the task-family seam is exercised here rather than only in the
  concrete code, where a mismatch would surface as a confusing inference
  error far from its cause.

The mispairing that must NOT typecheck -- a marshaller producing one task
family handed to a dispatcher accepting another -- lives in
``typecheck_negative/``, which is excluded from the repo-wide mypy run
and asserted to fail by ``test_negative_typing.py``.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pytest
from pyapprox.interface.evaluation.protocols import (
    BatchProtocol,
    DispatcherProtocol,
    EvaluatorProtocol,
    JobHandle,
    MarshalError,
    MarshallerProtocol,
)
from pyapprox.interface.evaluation.records import (
    ComputeProvenance,
    Cost,
    Decoded,
    EvalProgress,
    EvalResult,
    JobStatus,
    Outcome,
    Request,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.util.backends.protocols import Array, Backend


@dataclass(frozen=True)
class ShellTask:
    """A command to run in a directory. What a subprocess dispatcher takes."""

    argv: Sequence[str]
    workdir: str
    indices: Sequence[int]


@dataclass(frozen=True)
class CallableTask:
    """A call to make in-process. What an executor dispatcher takes."""

    indices: Sequence[int]


class _StubHandle:
    """Signature-only ``JobHandle``."""

    def __init__(self, task: CallableTask) -> None:
        self._task = task

    def done(self) -> bool:
        return True

    def outcome(
        self, timeout: Optional[float] = None
    ) -> Outcome[CallableTask, "np.ndarray"]:
        return Outcome(
            task=self._task,
            indices=self._task.indices,
            status=JobStatus.SUCCEEDED,
            payload=np.zeros((1, len(self._task.indices))),
            wall_time=0.0,
        )

    def cancel(self) -> bool:
        return False


class _StubShellHandle:
    """Signature-only ``JobHandle`` over a different task family."""

    def __init__(self, task: ShellTask) -> None:
        self._task = task

    def done(self) -> bool:
        return True

    def outcome(
        self, timeout: Optional[float] = None
    ) -> Outcome[ShellTask, str]:
        return Outcome(
            task=self._task,
            indices=self._task.indices,
            status=JobStatus.SUCCEEDED,
            payload=self._task.workdir,
            wall_time=0.0,
        )

    def cancel(self) -> bool:
        return True


class _StubExecutorDispatcher:
    """Signature-only stand-in for ``ExecutorDispatcher``."""

    def concurrency(self) -> int:
        return 4

    def compute_provenance(self) -> ComputeProvenance:
        return ComputeProvenance.MEASURED

    def submit(
        self, tasks: Sequence[CallableTask]
    ) -> Sequence[JobHandle[CallableTask, "np.ndarray"]]:
        return [_StubHandle(task) for task in tasks]

    def close(self) -> None:
        return None


class _StubSubprocessDispatcher:
    """Signature-only stand-in for ``SubprocessDispatcher``."""

    def concurrency(self) -> int:
        return 8

    def compute_provenance(self) -> ComputeProvenance:
        return ComputeProvenance.MEASURED

    def submit(
        self, tasks: Sequence[ShellTask]
    ) -> Sequence[JobHandle[ShellTask, str]]:
        return [_StubShellHandle(task) for task in tasks]

    def close(self) -> None:
        return None


class _StubCallableMarshaller:
    """Signature-only marshaller producing ``CallableTask``."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return 1

    def max_samples_per_task(self) -> int:
        return 1 << 30

    def tasks(
        self,
        samples: Array,
        indices: Sequence[int],
        request: Request[Array],
    ) -> Sequence[CallableTask]:
        """A fused code: one task satisfies every quantity asked for."""
        return [CallableTask(indices=indices)]

    def values(
        self, outcome: Outcome[CallableTask, "np.ndarray"]
    ) -> Decoded[Array]:
        if outcome.payload is None:
            raise MarshalError("no payload")
        return Decoded(
            values=self._bkd.asarray(outcome.payload),
            indices=outcome.indices,
        )

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.none()

    def release(self, outcome: Outcome[CallableTask, "np.ndarray"]) -> None:
        return None


class _StubTextFileMarshaller:
    """Signature-only marshaller producing a *different* task family.

    This is what makes the ``Task`` seam real rather than asserted: two
    marshallers, two task types, each usable only with its own
    dispatcher.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return 1

    def max_samples_per_task(self) -> int:
        return 1

    def tasks(
        self,
        samples: Array,
        indices: Sequence[int],
        request: Request[Array],
    ) -> Sequence[ShellTask]:
        """A split code: values and jacobians are separate executables.

        The counterpart to the fused marshaller above. Both satisfy the
        same protocol without either being penalized, which is what
        taking a request rather than exposing a method per quantity buys.
        """
        built = []
        if request.values:
            built.append(
                ShellTask(
                    argv=["./solver"], workdir="/tmp/x", indices=indices
                )
            )
        if request.jacobians:
            built.append(
                ShellTask(
                    argv=["./solver_adjoint"],
                    workdir="/tmp/x",
                    indices=indices,
                )
            )
        return built

    def values(self, outcome: Outcome[ShellTask, str]) -> Decoded[Array]:
        if outcome.payload is None:
            raise MarshalError("no payload")
        nsamples = len(outcome.indices)
        if outcome.task.argv[0].endswith("adjoint"):
            return Decoded(
                values=self._bkd.zeros((1, 0)),
                indices=outcome.indices,
                jacobians=self._bkd.zeros((nsamples, 1, 2)),
            )
        return Decoded(
            values=self._bkd.zeros((1, nsamples)), indices=outcome.indices
        )

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.first_order(jacobian=self._jacobian)

    def _jacobian(self, sample: Array) -> Array:
        return self._bkd.zeros((1, 2))

    def release(self, outcome: Outcome[ShellTask, str]) -> None:
        return None


class _StubBatch:
    """Signature-only ``BatchProtocol``."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def nsubmitted(self) -> int:
        return 0

    def progress(self) -> EvalProgress:
        return EvalProgress(
            nsucceeded=0,
            nfailed=0,
            noutstanding=0,
            cost=Cost.zero(),
            elapsed_seconds=0.0,
        )

    def _empty(self) -> EvalResult[Array]:
        return EvalResult(
            values=self._bkd.zeros((1, 0)),
            succeeded=self._bkd.zeros((0,), dtype=int),
            failed=self._bkd.zeros((0,), dtype=int),
            cancelled=self._bkd.zeros((0,), dtype=int),
            cost=Cost.zero(),
        )

    def collect_ready(self) -> EvalResult[Array]:
        return self._empty()

    def collect(self, timeout: Optional[float] = None) -> EvalResult[Array]:
        return self._empty()

    def cancel(self) -> EvalResult[Array]:
        return self._empty()


class _StubEvaluator:
    """Signature-only ``EvaluatorProtocol``, with Task/Payload erased."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return 1

    def submit(
        self, samples: Array, request: Optional[Request[Array]] = None
    ) -> BatchProtocol[Array]:
        return _StubBatch(self._bkd)

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.none()


class TestRuntimeConformance:
    """The ``isinstance`` backstop, on the bare protocols only."""

    def test_dispatchers_satisfy_protocol(self):
        assert isinstance(_StubExecutorDispatcher(), DispatcherProtocol)
        assert isinstance(_StubSubprocessDispatcher(), DispatcherProtocol)

    def test_handles_satisfy_protocol(self):
        assert isinstance(_StubHandle(CallableTask([0])), JobHandle)
        assert isinstance(
            _StubShellHandle(ShellTask(["x"], "/tmp", [0])), JobHandle
        )

    def test_marshallers_satisfy_protocol(self, numpy_bkd):
        assert isinstance(
            _StubCallableMarshaller(numpy_bkd), MarshallerProtocol
        )
        assert isinstance(
            _StubTextFileMarshaller(numpy_bkd), MarshallerProtocol
        )

    def test_batch_and_evaluator_satisfy_protocols(self, numpy_bkd):
        assert isinstance(_StubBatch(numpy_bkd), BatchProtocol)
        assert isinstance(_StubEvaluator(numpy_bkd), EvaluatorProtocol)

    def test_missing_method_fails_the_check(self):
        """The backstop catches an implementer who missed a method."""

        class Incomplete:
            def concurrency(self) -> int:
                return 1

        assert not isinstance(Incomplete(), DispatcherProtocol)

    def test_subscripted_protocol_cannot_be_used_with_isinstance(self):
        """Documented limitation, and why mypy is the primary guard.

        A subscripted protocol raises at runtime, so the task pairing
        cannot be checked this way at all.
        """
        with pytest.raises(TypeError):
            isinstance(
                _StubExecutorDispatcher(),
                DispatcherProtocol[CallableTask, "np.ndarray"],
            )


class TestTaskFamilySeam:
    """Each dispatcher runs only its own task family."""

    def test_matched_pairing_round_trips(self, numpy_bkd):
        marshaller = _StubCallableMarshaller(numpy_bkd)
        dispatcher = _StubExecutorDispatcher()
        tasks = marshaller.tasks(
            numpy_bkd.zeros((2, 1)), [0], Request.values_only()
        )
        handles = dispatcher.submit(tasks)
        decoded = marshaller.values(handles[0].outcome())
        assert list(decoded.indices) == [0]
        assert decoded.values.shape[1] == 1

    def test_shell_pairing_round_trips(self, numpy_bkd):
        marshaller = _StubTextFileMarshaller(numpy_bkd)
        dispatcher = _StubSubprocessDispatcher()
        tasks = marshaller.tasks(
            numpy_bkd.zeros((2, 1)), [3], Request.values_only()
        )
        handles = dispatcher.submit(tasks)
        decoded = marshaller.values(handles[0].outcome())
        assert list(decoded.indices) == [3]

    def test_fused_code_answers_one_request_with_one_task(self, numpy_bkd):
        """A fused marshaller solves once for values and jacobians."""
        marshaller = _StubCallableMarshaller(numpy_bkd)
        tasks = marshaller.tasks(
            numpy_bkd.zeros((2, 2)),
            [0, 1],
            Request(values=True, jacobians=True),
        )
        assert len(tasks) == 1

    def test_split_code_answers_one_request_with_two_tasks(self, numpy_bkd):
        """A split marshaller dispatches value and adjoint separately.

        Same protocol, same request, different number of invocations --
        which is the point of the marshaller owning that decision.
        """
        marshaller = _StubTextFileMarshaller(numpy_bkd)
        tasks = marshaller.tasks(
            numpy_bkd.zeros((2, 1)),
            [0],
            Request(values=True, jacobians=True),
        )
        assert len(tasks) == 2
        assert [t.argv[0] for t in tasks] == ["./solver", "./solver_adjoint"]

    def test_split_code_decodes_per_quantity(self, numpy_bkd):
        """Each task decodes only what it computed; merging is by index."""
        marshaller = _StubTextFileMarshaller(numpy_bkd)
        tasks = marshaller.tasks(
            numpy_bkd.zeros((2, 1)),
            [0],
            Request(values=True, jacobians=True),
        )
        dispatcher = _StubSubprocessDispatcher()
        handles = dispatcher.submit(tasks)
        value_decoded = marshaller.values(handles[0].outcome())
        jac_decoded = marshaller.values(handles[1].outcome())

        assert value_decoded.values.shape == (1, 1)
        assert value_decoded.jacobians is None
        assert jac_decoded.jacobians is not None
        assert jac_decoded.jacobians.shape == (1, 1, 2)
        assert list(value_decoded.indices) == list(jac_decoded.indices)

    def test_marshaller_advertises_capability_through_bundle(
        self, numpy_bkd
    ):
        """Discovery is bundle inspection, not a capability predicate."""
        assert (
            _StubCallableMarshaller(numpy_bkd).derivatives().jacobian is None
        )
        assert (
            _StubTextFileMarshaller(numpy_bkd).derivatives().jacobian
            is not None
        )

    def test_grouping_differs_by_marshaller(self, numpy_bkd):
        """One workdir per sample versus vectorized in-memory work."""
        assert _StubTextFileMarshaller(numpy_bkd).max_samples_per_task() == 1
        assert (
            _StubCallableMarshaller(numpy_bkd).max_samples_per_task() > 1000
        )


class TestOutcomeTimeoutContract:
    def test_outstanding_outcome_is_expressible(self):
        """A timeout returns a record rather than raising."""
        outcome: Outcome[CallableTask, "np.ndarray"] = Outcome(
            task=CallableTask([0]),
            indices=[0],
            status=JobStatus.OUTSTANDING,
        )
        assert not outcome.status.is_finished()
        assert outcome.payload is None
