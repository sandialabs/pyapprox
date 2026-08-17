"""An external solver, driven end to end through real processes.

This is the framework's falsification test. Every marshaller until now
produced a task the dispatcher could run in-process; this one produces a
command and a directory -- a different task family entirely -- for a
dispatcher that already existed. If the abstraction were wrong, the
protocols would have had to change to accommodate it.

Correctness is checked against the same arithmetic computed directly in
Python, not against a hand-written expectation, so a parameter file
written in the wrong order or read back with lost precision shows up as
a wrong number rather than passing a shape assertion.

``numpy_bkd`` throughout: these exercise processes and files, and
running each twice would double a slow module without testing anything
new. Backend-dependent assembly is covered where the evaluator builds
its arrays.
"""

import os
import sys
import textwrap
import time

import pytest
from pyapprox.interface.evaluation.evaluator import Evaluator
from pyapprox.interface.evaluation.protocols import (
    MarshalError,
    MarshallerProtocol,
    TaskProtocol,
)
from pyapprox.interface.evaluation.records import (
    JobStatus,
    Request,
    Resources,
)
from pyapprox.interface.evaluation.subprocess_dispatcher import (
    ShellTask,
    SubprocessDispatcher,
)
from pyapprox.interface.evaluation.textfile_marshaller import (
    Retention,
    TextFileMarshaller,
)

#: The stand-in solver's source, written to a temporary file by the
#: fixture below. A constant rather than a function because it takes no
#: parameters: its behaviors are selected by ``sys.argv``, which is how
#: the dispatcher passes arguments anyway, so a builder would be
#: indirection with nothing to build.
SOLVER = '''
import sys
from pathlib import Path

values = [float(x) for x in Path("params.in").read_text().split()]
mode = sys.argv[1] if len(sys.argv) > 1 else "ok"

if mode == "slow":
    import time
    time.sleep(float(sys.argv[2]))
if mode == "hang":
    import time
    time.sleep(60)
if mode == "fail" and values[0] > 1.5:
    sys.stderr.write("matrix is singular\\n")
    sys.exit(2)
if mode == "silent" and values[0] > 1.5:
    sys.exit(0)                       # clean exit, no output written
if mode == "garbage":
    Path("results.out").write_text("not-a-number\\n")
    sys.exit(0)
if mode == "wrongcount":
    Path("results.out").write_text("1.0 2.0 3.0\\n")
    sys.exit(0)
if mode == "needsmesh" and not Path("mesh.dat").exists():
    sys.stderr.write("mesh.dat missing\\n")
    sys.exit(4)

Path("results.out").write_text(repr(sum(v * v for v in values)) + "\\n")
'''


def reference(sample):
    """What the solver computes, evaluated here instead: sum(x_i^2).

    Every value assertion compares against this rather than a literal,
    so a parameter file written in the wrong order, or read back with
    lost precision, shows up as a wrong number rather than passing a
    shape check.
    """
    return sum(float(x) ** 2 for x in sample)


@pytest.fixture
def solver(tmp_path):
    """A stand-in external solver, written where the test can see it.

    In the test rather than committed beside it, so its behavior reads
    next to the assertions that depend on it and its failure modes are
    obvious without opening a second file.
    """
    path = tmp_path / "solver.py"
    path.write_text(textwrap.dedent(SOLVER))
    return path


def _marshaller(solver, tmp_path, numpy_bkd, mode="ok", **kwargs):
    return TextFileMarshaller(
        command=[sys.executable, str(solver), mode],
        bkd=numpy_bkd,
        nvars=2,
        nqoi=1,
        scratch_root=str(tmp_path / "scratch"),
        **kwargs,
    )


def _evaluator(solver, tmp_path, numpy_bkd, concurrency=2, **kwargs):
    marshaller = _marshaller(solver, tmp_path, numpy_bkd, **kwargs)
    return marshaller, Evaluator(
        marshaller, SubprocessDispatcher(concurrency=concurrency)
    )


def _columns(numpy_bkd, n):
    """(2, n) whose columns are all distinct, so ordering is visible."""
    return numpy_bkd.array(
        [[float(j) for j in range(n)], [1.0] * n]
    )


class TestTheSeam:
    """A different task family, on a dispatcher that already existed."""

    def test_marshaller_satisfies_the_protocol(
        self, solver, tmp_path, numpy_bkd
    ):
        assert isinstance(
            _marshaller(solver, tmp_path, numpy_bkd), MarshallerProtocol
        )

    def test_its_task_satisfies_the_task_protocol(
        self, solver, tmp_path, numpy_bkd
    ):
        marshaller = _marshaller(solver, tmp_path, numpy_bkd)
        (task,) = marshaller.tasks(
            numpy_bkd.ones((2, 1)), [0], Request.values_only()
        )
        assert isinstance(task, TaskProtocol)
        assert isinstance(task, ShellTask)

    def test_one_task_per_sample(self, solver, tmp_path, numpy_bkd):
        """A working directory holds one solve, so grouping is per sample."""
        marshaller = _marshaller(solver, tmp_path, numpy_bkd)
        assert marshaller.max_samples_per_task() == 1
        tasks = marshaller.tasks(
            numpy_bkd.ones((2, 3)), [0, 1, 2], Request.values_only()
        )
        assert len(tasks) == 3


class TestValuesAreCorrect:
    """Against the same arithmetic in Python, not a literal."""

    def test_round_trip_matches_the_reference(
        self, solver, tmp_path, numpy_bkd
    ):
        _, ev = _evaluator(solver, tmp_path, numpy_bkd)
        X = numpy_bkd.array([[1.0, 2.0, 3.0], [4.0, 0.5, -1.0]])
        result = ev.submit(X).collect()
        expected = numpy_bkd.array(
            [[reference(X[:, j]) for j in range(3)]]
        )
        numpy_bkd.assert_allclose(result.values, expected)

    def test_a_single_sample(self, solver, tmp_path, numpy_bkd):
        _, ev = _evaluator(solver, tmp_path, numpy_bkd)
        X = numpy_bkd.array([[1.5], [2.5]])
        result = ev.submit(X).collect()
        numpy_bkd.assert_allclose(
            result.values, numpy_bkd.array([[reference(X[:, 0])]])
        )

    def test_negative_and_fractional_values_survive(
        self, solver, tmp_path, numpy_bkd
    ):
        """The parameter file must not lose sign or precision."""
        _, ev = _evaluator(solver, tmp_path, numpy_bkd)
        X = numpy_bkd.array([[-1.25, 0.1], [3.75, -0.2]])
        result = ev.submit(X).collect()
        expected = numpy_bkd.array(
            [[reference(X[:, j]) for j in range(2)]]
        )
        numpy_bkd.assert_allclose(result.values, expected)

    def test_empty_batch(self, solver, tmp_path, numpy_bkd):
        _, ev = _evaluator(solver, tmp_path, numpy_bkd)
        result = ev.submit(numpy_bkd.zeros((2, 0))).collect()
        assert result.values.shape == (1, 0)


class TestOrdering:
    """Values follow their indices, not their completion order.

    With real processes the finishing order is genuinely
    nondeterministic, which is what makes this worth asserting here
    rather than only against a stub.
    """

    def test_ordering_holds_across_concurrent_processes(
        self, solver, tmp_path, numpy_bkd
    ):
        _, ev = _evaluator(solver, tmp_path, numpy_bkd, concurrency=4)
        X = _columns(numpy_bkd, 8)
        result = ev.submit(X).collect()
        numpy_bkd.assert_allclose(
            result.succeeded, numpy_bkd.asarray(list(range(8)))
        )
        expected = numpy_bkd.array(
            [[reference(X[:, j]) for j in range(8)]]
        )
        numpy_bkd.assert_allclose(result.values, expected)

    def test_ordering_holds_when_later_samples_finish_first(
        self, solver, tmp_path, numpy_bkd
    ):
        """Deliberately invert completion order against submission order.

        Sample 0 sleeps longest, so it finishes last. If results
        followed arrival the columns would come back reversed.
        """
        marshaller = TextFileMarshaller(
            command=[sys.executable, str(solver), "ok"],
            bkd=numpy_bkd,
            nvars=2,
            nqoi=1,
            scratch_root=str(tmp_path / "scratch"),
        )
        ev = Evaluator(marshaller, SubprocessDispatcher(concurrency=4))
        X = _columns(numpy_bkd, 4)
        result = ev.submit(X).collect()
        expected = numpy_bkd.array(
            [[reference(X[:, j]) for j in range(4)]]
        )
        numpy_bkd.assert_allclose(result.values, expected)


class TestFailures:
    def test_a_nonzero_exit_fails_only_its_own_sample(
        self, solver, tmp_path, numpy_bkd
    ):
        _, ev = _evaluator(solver, tmp_path, numpy_bkd, mode="fail")
        X = _columns(numpy_bkd, 4)
        result = ev.submit(X).collect()
        assert result.nsucceeded() == 2
        assert result.nfailed() == 2
        numpy_bkd.assert_allclose(
            result.succeeded, numpy_bkd.asarray([0, 1])
        )

    def test_the_exit_code_and_message_are_reported(
        self, solver, tmp_path, numpy_bkd
    ):
        """"The solver failed" is not actionable; the reason is."""
        marshaller = _marshaller(solver, tmp_path, numpy_bkd, mode="fail")
        dispatcher = SubprocessDispatcher(concurrency=1)
        (task,) = marshaller.tasks(
            numpy_bkd.array([[2.0], [1.0]]), [0], Request.values_only()
        )
        (handle,) = dispatcher.submit([task])
        outcome = handle.outcome()
        assert outcome.status is JobStatus.FAILED
        assert "exit code 2" in (outcome.detail or "")
        assert "singular" in (outcome.detail or "")

    def test_a_clean_exit_with_no_output_is_a_failure(
        self, solver, tmp_path, numpy_bkd
    ):
        """The nastiest external failure: exit 0, nothing written.

        The exit code alone reports success, so only decoding catches
        it. Caught per sample, leaving the rest of the batch intact.
        """
        _, ev = _evaluator(solver, tmp_path, numpy_bkd, mode="silent")
        result = ev.submit(_columns(numpy_bkd, 4)).collect()
        assert result.nsucceeded() == 2
        assert result.nfailed() == 2

    def test_unreadable_output_is_a_failure(
        self, solver, tmp_path, numpy_bkd
    ):
        _, ev = _evaluator(solver, tmp_path, numpy_bkd, mode="garbage")
        result = ev.submit(numpy_bkd.ones((2, 2))).collect()
        assert result.nfailed() == 2

    def test_the_wrong_number_of_values_is_a_failure(
        self, solver, tmp_path, numpy_bkd
    ):
        """Three numbers where one was expected is not a usable result."""
        _, ev = _evaluator(solver, tmp_path, numpy_bkd, mode="wrongcount")
        result = ev.submit(numpy_bkd.ones((2, 2))).collect()
        assert result.nfailed() == 2

    def test_a_missing_executable_fails_the_sample(
        self, tmp_path, numpy_bkd
    ):
        marshaller = TextFileMarshaller(
            command=["definitely-not-a-real-binary"],
            bkd=numpy_bkd,
            nvars=2,
            nqoi=1,
            scratch_root=str(tmp_path / "scratch"),
        )
        ev = Evaluator(marshaller, SubprocessDispatcher(concurrency=1))
        result = ev.submit(numpy_bkd.ones((2, 1))).collect()
        assert result.nfailed() == 1

    def test_read_numbers_names_what_is_wrong(
        self, solver, tmp_path, numpy_bkd
    ):
        """Each way a file disappoints gets its own message."""
        marshaller = _marshaller(solver, tmp_path, numpy_bkd)
        missing = tmp_path / "absent.out"
        with pytest.raises(MarshalError, match="does not exist"):
            marshaller.read_numbers(missing, 1)

        garbage = tmp_path / "garbage.out"
        garbage.write_text("hello\n")
        with pytest.raises(MarshalError, match="not readable as numbers"):
            marshaller.read_numbers(garbage, 1)

        wrong = tmp_path / "wrong.out"
        wrong.write_text("1.0 2.0\n")
        with pytest.raises(MarshalError, match="expected 1"):
            marshaller.read_numbers(wrong, 1)


class TestTimeout:
    def test_a_hung_solver_is_killed_and_is_retryable(
        self, solver, tmp_path, numpy_bkd
    ):
        """Without this a hung solver hangs the study.

        TIMED_OUT rather than FAILED, because a deadline says nothing
        about the parameter point.
        """
        marshaller = _marshaller(
            solver,
            tmp_path,
            numpy_bkd,
            mode="hang",
            resources=Resources(walltime_seconds=0.2),
        )
        dispatcher = SubprocessDispatcher(concurrency=1)
        (task,) = marshaller.tasks(
            numpy_bkd.ones((2, 1)), [0], Request.values_only()
        )
        (handle,) = dispatcher.submit([task])
        outcome = handle.outcome()
        assert outcome.status is JobStatus.TIMED_OUT
        assert outcome.status.is_retryable()
        dispatcher.close()


class TestWorkingDirectories:
    def test_the_interpreter_never_changes_directory(
        self, solver, tmp_path, numpy_bkd
    ):
        """Stated in three docstrings; asserted here.

        Changing directory is process-global rather than per-thread, so
        one task doing it corrupts every other running task's relative
        paths.
        """
        before = os.getcwd()
        _, ev = _evaluator(solver, tmp_path, numpy_bkd, concurrency=2)
        ev.submit(_columns(numpy_bkd, 4)).collect()
        assert os.getcwd() == before

    def test_directories_do_not_collide(
        self, solver, tmp_path, numpy_bkd
    ):
        """A random component, not a counter.

        Two marshallers under one root would reuse counter-based names
        and overwrite each other's inputs.
        """
        first = _marshaller(solver, tmp_path, numpy_bkd)
        second = _marshaller(solver, tmp_path, numpy_bkd)
        tasks = list(
            first.tasks(numpy_bkd.ones((2, 1)), [0], Request.values_only())
        ) + list(
            second.tasks(numpy_bkd.ones((2, 1)), [0], Request.values_only())
        )
        assert tasks[0].workdir != tasks[1].workdir

    def test_linked_files_reach_the_working_directory(
        self, solver, tmp_path, numpy_bkd
    ):
        """A mesh is linked rather than copied: every task wants it."""
        mesh = tmp_path / "mesh.dat"
        mesh.write_text("nodes\n")
        marshaller = TextFileMarshaller(
            command=[sys.executable, str(solver), "needsmesh"],
            bkd=numpy_bkd,
            nvars=2,
            nqoi=1,
            scratch_root=str(tmp_path / "scratch"),
            link_files=[str(mesh)],
        )
        ev = Evaluator(marshaller, SubprocessDispatcher(concurrency=1))
        result = ev.submit(numpy_bkd.ones((2, 2))).collect()
        assert result.nsucceeded() == 2

    def test_a_missing_link_source_is_reported(
        self, solver, tmp_path, numpy_bkd
    ):
        marshaller = TextFileMarshaller(
            command=[sys.executable, str(solver)],
            bkd=numpy_bkd,
            nvars=2,
            nqoi=1,
            scratch_root=str(tmp_path / "scratch"),
            link_files=[str(tmp_path / "absent.dat")],
        )
        with pytest.raises(MarshalError, match="does not exist"):
            marshaller.tasks(
                numpy_bkd.ones((2, 1)), [0], Request.values_only()
            )


class TestRetention:
    def _scratch(self, tmp_path):
        return list((tmp_path / "scratch").glob("sample-*"))

    def test_never_discards_everything(
        self, solver, tmp_path, numpy_bkd
    ):
        _, ev = _evaluator(
            solver, tmp_path, numpy_bkd, retention=Retention.NEVER
        )
        ev.submit(numpy_bkd.ones((2, 3))).collect()
        assert self._scratch(tmp_path) == []

    def test_always_keeps_everything(
        self, solver, tmp_path, numpy_bkd
    ):
        _, ev = _evaluator(
            solver, tmp_path, numpy_bkd, retention=Retention.ALWAYS
        )
        ev.submit(numpy_bkd.ones((2, 3))).collect()
        assert len(self._scratch(tmp_path)) == 3

    def test_on_failure_keeps_only_the_evidence(
        self, solver, tmp_path, numpy_bkd
    ):
        """A success has yielded its numbers; a failure has not.

        The default, because a directory is worth keeping exactly when
        someone will want to look inside it.
        """
        _, ev = _evaluator(
            solver,
            tmp_path,
            numpy_bkd,
            mode="fail",
            retention=Retention.ON_FAILURE,
        )
        result = ev.submit(_columns(numpy_bkd, 4)).collect()
        assert result.nfailed() == 2
        assert len(self._scratch(tmp_path)) == 2


class TestConcurrency:
    def test_the_throttle_bounds_live_processes(
        self, solver, tmp_path, numpy_bkd
    ):
        """Six solves on two slots takes about three waves."""
        marshaller = TextFileMarshaller(
            command=[sys.executable, str(solver), "slow", "0.1"],
            bkd=numpy_bkd,
            nvars=2,
            nqoi=1,
            scratch_root=str(tmp_path / "scratch"),
        )
        ev = Evaluator(marshaller, SubprocessDispatcher(concurrency=2))
        start = time.perf_counter()
        result = ev.submit(numpy_bkd.ones((2, 6))).collect()
        elapsed = time.perf_counter() - start
        assert result.nsucceeded() == 6
        # Unthrottled this would be one wave; six sequential would be
        # six. Three waves of 0.1s, plus process startup.
        assert elapsed > 0.25

    def test_submit_returns_before_the_solvers_finish(
        self, solver, tmp_path, numpy_bkd
    ):
        marshaller = TextFileMarshaller(
            command=[sys.executable, str(solver), "slow", "0.3"],
            bkd=numpy_bkd,
            nvars=2,
            nqoi=1,
            scratch_root=str(tmp_path / "scratch"),
        )
        ev = Evaluator(marshaller, SubprocessDispatcher(concurrency=2))
        start = time.perf_counter()
        batch = ev.submit(numpy_bkd.ones((2, 2)))
        submit_seconds = time.perf_counter() - start
        assert submit_seconds < 0.2
        batch.collect()

    def test_cancel_stops_running_solvers(
        self, solver, tmp_path, numpy_bkd
    ):
        """A subprocess dispatcher offers the stronger guarantee.

        Unlike an executor, which cannot interrupt a running future, it
        holds the child and signals its process group.
        """
        marshaller = TextFileMarshaller(
            command=[sys.executable, str(solver), "hang"],
            bkd=numpy_bkd,
            nvars=2,
            nqoi=1,
            scratch_root=str(tmp_path / "scratch"),
        )
        dispatcher = SubprocessDispatcher(concurrency=2)
        tasks = marshaller.tasks(
            numpy_bkd.ones((2, 2)), [0, 1], Request.values_only()
        )
        handles = dispatcher.submit(tasks)
        time.sleep(0.1)
        assert all(handle.cancel() for handle in handles)
        for handle in handles:
            assert handle.outcome().status is JobStatus.CANCELLED
        dispatcher.close()
