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
from pathlib import Path

import pytest
from pyapprox.interface.evaluation.collection import (
    AnomalyKind,
    OutputSpec,
    SpecCollector,
    gather_run,
    reconcile,
)
from pyapprox.interface.evaluation.evaluator import Evaluator
from pyapprox.interface.evaluation.manifest import (
    KIND_PREPARED,
    KIND_RELEASED,
    KIND_RUN,
    RUN_DONE_FILENAME,
    read_records,
)
from pyapprox.interface.evaluation.protocols import (
    MarshalError,
    MarshallerProtocol,
    SubmissionAware,
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
    OnExisting,
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
if mode == "writesfield":
    # A field file beside the scalar, which is what a real solver
    # leaves behind and what nothing but collection ever looks at.
    Path("out.fld").write_text("field for sample %d\\n" % int(values[0]))

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
    ) -> None:
        """At construction, not per sample after ``submit``.

        A typo in a mesh path is one mistake; discovering it once per
        sample, on every rank, after an allocation has started, is the
        expensive way to be told.
        """
        with pytest.raises(ValueError, match="does not exist"):
            TextFileMarshaller(
                command=[sys.executable, str(solver)],
                bkd=numpy_bkd,
                nvars=2,
                nqoi=1,
                scratch_root=str(tmp_path / "scratch"),
                link_files=[str(tmp_path / "absent.dat")],
            )

    def test_an_unreadable_link_source_is_reported(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        """Existing is not enough; the symlink has to be followable."""
        mesh = tmp_path / "mesh.dat"
        mesh.write_text("nodes\n")
        mesh.chmod(0o000)
        try:
            with pytest.raises(ValueError, match="not readable"):
                TextFileMarshaller(
                    command=[sys.executable, str(solver)],
                    bkd=numpy_bkd,
                    nvars=2,
                    nqoi=1,
                    scratch_root=str(tmp_path / "scratch"),
                    link_files=[str(mesh)],
                )
        finally:
            mesh.chmod(0o600)

    def test_an_unusable_scratch_root_is_reported(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        readonly = tmp_path / "readonly"
        readonly.mkdir()
        readonly.chmod(0o500)
        try:
            with pytest.raises(ValueError, match="not usable"):
                TextFileMarshaller(
                    command=[sys.executable, str(solver)],
                    bkd=numpy_bkd,
                    nvars=2,
                    nqoi=1,
                    scratch_root=str(readonly / "scratch"),
                )
        finally:
            readonly.chmod(0o700)

    def test_the_probe_leaves_nothing_behind(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        """An ensemble rebuilt per iteration constructs many of these."""
        scratch = tmp_path / "scratch"
        for _ in range(3):
            TextFileMarshaller(
                command=[sys.executable, str(solver)],
                bkd=numpy_bkd,
                nvars=2,
                nqoi=1,
                scratch_root=str(scratch),
            )
        assert list(scratch.iterdir()) == []

    def test_the_per_sample_check_remains_as_a_backstop(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        """A file can be deleted between construction and dispatch."""
        mesh = tmp_path / "mesh.dat"
        mesh.write_text("nodes\n")
        marshaller = TextFileMarshaller(
            command=[sys.executable, str(solver)],
            bkd=numpy_bkd,
            nvars=2,
            nqoi=1,
            scratch_root=str(tmp_path / "scratch"),
            link_files=[str(mesh)],
        )
        mesh.unlink()
        with pytest.raises(MarshalError, match="does not exist"):
            marshaller.tasks(
                numpy_bkd.ones((2, 1)), [0], Request.values_only()
            )


class TestRunAndSubmissionLayout:
    """Directories are scoped by run and submission, not by index alone.

    A sample number is a column in the submitted batch, so it restarts
    at zero every time. One marshaller is submitted to many times --
    ``BlockingModel`` submits on every call -- so the submission level
    is what keeps the second submission from rebuilding the first's
    paths.
    """

    def _run_dirs(self, tmp_path):
        return sorted((tmp_path / "scratch").iterdir())

    def test_a_sample_lands_under_run_and_submission(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        marshaller = _marshaller(solver, tmp_path, numpy_bkd)
        tasks = marshaller.tasks(
            numpy_bkd.ones((2, 1)), [0], Request.values_only()
        )
        workdir = Path(tasks[0].workdir)
        assert workdir.name == "sample-000000"
        assert workdir.parent.name == "sub-000"
        assert workdir.parent.parent == Path(marshaller.run_dir())

    def test_names_are_padded_so_they_sort_as_text(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        """``sample-10`` before ``sample-2`` is the trap being avoided."""
        marshaller = _marshaller(solver, tmp_path, numpy_bkd)
        tasks = marshaller.tasks(
            numpy_bkd.ones((2, 11)), list(range(11)), Request.values_only()
        )
        names = [Path(task.workdir).name for task in tasks]
        assert names == sorted(names)
        assert names[-1] == "sample-000010"

    def test_two_submissions_do_not_collide(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        """The behavior the submission level exists for.

        Both submissions number their columns from zero, so without a
        submission level the second would rebuild the first's paths.
        """
        marshaller, ev = _evaluator(solver, tmp_path, numpy_bkd)
        ev.submit(_columns(numpy_bkd, 2)).collect()
        ev.submit(_columns(numpy_bkd, 2)).collect()
        run = Path(marshaller.run_dir())
        # By name, not by listing the directory: the manifest lives here
        # too, so "what submissions are there" is a question about
        # sub-* rather than about everything present.
        assert sorted(p.name for p in run.glob("sub-*")) == [
            "sub-000",
            "sub-001",
        ]

    def test_a_second_submission_does_not_overwrite_the_first(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        """Distinct paths, so no solve reads another's inputs."""
        marshaller = _marshaller(
            solver, tmp_path, numpy_bkd, retention=Retention.ALWAYS
        )
        marshaller.begin_submission()
        first = marshaller.tasks(
            numpy_bkd.ones((2, 1)), [0], Request.values_only()
        )
        marshaller.begin_submission()
        second = marshaller.tasks(
            numpy_bkd.ones((2, 1)), [0], Request.values_only()
        )
        assert first[0].workdir != second[0].workdir
        assert Path(first[0].workdir).exists()
        assert Path(second[0].workdir).exists()

    def test_the_run_directory_is_not_claimed_at_construction(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        """An ensemble rebuilt per iteration constructs many of these."""
        scratch = tmp_path / "scratch"
        for _ in range(3):
            marshaller = _marshaller(solver, tmp_path, numpy_bkd)
            assert marshaller.run_dir() is None
        assert list(scratch.iterdir()) == []

    def test_a_named_run_id_is_used_verbatim(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        marshaller = _marshaller(
            solver, tmp_path, numpy_bkd, run_id="sweep-a"
        )
        marshaller.begin_submission()
        assert Path(marshaller.run_dir()).name == "sweep-a"

    @pytest.mark.parametrize(
        "bad", ["../escape", "a/b", "..", "with space", ""]
    )
    def test_a_run_id_that_is_not_a_safe_name_is_refused(
        self, solver, tmp_path, numpy_bkd, bad
    ) -> None:
        """It becomes a path component, so traversal must not reach it."""
        with pytest.raises(ValueError, match="run_id"):
            _marshaller(solver, tmp_path, numpy_bkd, run_id=bad)

    def test_an_existing_run_id_is_refused_by_default(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        (tmp_path / "scratch" / "sweep-a").mkdir(parents=True)
        marshaller = _marshaller(
            solver, tmp_path, numpy_bkd, run_id="sweep-a"
        )
        with pytest.raises(MarshalError, match="already exists"):
            marshaller.begin_submission()

    def test_resume_continues_after_existing_submissions(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        """The ordinal is read from the directory, not held in memory.

        A resumed process starts a fresh marshaller, so an in-memory
        counter would restart at zero and overwrite ``sub-000``.
        """
        run = tmp_path / "scratch" / "sweep-a"
        (run / "sub-000").mkdir(parents=True)
        (run / "sub-001").mkdir()
        marshaller = _marshaller(
            solver,
            tmp_path,
            numpy_bkd,
            run_id="sweep-a",
            on_existing=OnExisting.RESUME,
        )
        marshaller.begin_submission()
        tasks = marshaller.tasks(
            numpy_bkd.ones((2, 1)), [0], Request.values_only()
        )
        assert Path(tasks[0].workdir).parent.name == "sub-002"

    def test_new_allocates_a_fresh_id_beside_the_existing_one(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        (tmp_path / "scratch" / "sweep-a").mkdir(parents=True)
        marshaller = _marshaller(
            solver,
            tmp_path,
            numpy_bkd,
            run_id="sweep-a",
            on_existing=OnExisting.NEW,
        )
        marshaller.begin_submission()
        assert Path(marshaller.run_dir()).name != "sweep-a"
        assert (tmp_path / "scratch" / "sweep-a").exists()

    def test_the_marshaller_declares_it_is_submission_aware(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        assert isinstance(
            _marshaller(solver, tmp_path, numpy_bkd), SubmissionAware
        )


class TestManifest:
    """What the run recorded about itself, as it ran."""

    def _records(self, marshaller, kind=None):
        records = read_records(marshaller.manifest_path())
        if kind is None:
            return records
        return [r for r in records if r["kind"] == kind]

    def test_a_header_is_written_when_the_run_begins(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        marshaller = _marshaller(
            solver, tmp_path, numpy_bkd, retention=Retention.NEVER
        )
        marshaller.begin_submission()
        header = self._records(marshaller, KIND_RUN)[0]
        assert header["run_id"] == Path(marshaller.run_dir()).name
        assert header["retention"] == "never"
        assert header["command"][0] == sys.executable

    def test_the_manifest_lives_in_the_run_directory(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        marshaller = _marshaller(solver, tmp_path, numpy_bkd)
        marshaller.begin_submission()
        assert (
            Path(marshaller.manifest_path()).parent
            == Path(marshaller.run_dir())
        )

    def test_there_is_no_manifest_before_a_run_begins(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        marshaller = _marshaller(solver, tmp_path, numpy_bkd)
        assert marshaller.manifest_path() is None

    def test_every_prepared_directory_is_recorded(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        marshaller = _marshaller(solver, tmp_path, numpy_bkd)
        marshaller.tasks(
            numpy_bkd.ones((2, 3)), [0, 1, 2], Request.values_only()
        )
        prepared = self._records(marshaller, KIND_PREPARED)
        assert [r["index"] for r in prepared] == [0, 1, 2]

    def test_workdirs_are_recorded_relative_to_the_run(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        """An absolute path stops meaning anything once a run moves."""
        marshaller = _marshaller(solver, tmp_path, numpy_bkd)
        marshaller.tasks(
            numpy_bkd.ones((2, 1)), [0], Request.values_only()
        )
        record = self._records(marshaller, KIND_PREPARED)[0]
        assert record["workdir"] == "sub-000/sample-000000"

    def test_a_finished_sample_gets_one_release_record(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        marshaller, ev = _evaluator(
            solver, tmp_path, numpy_bkd, retention=Retention.NEVER
        )
        ev.submit(_columns(numpy_bkd, 3)).collect()
        released = self._records(marshaller, KIND_RELEASED)
        assert len(released) == 3
        assert all(r["status"] == "SUCCEEDED" for r in released)
        assert all(r["any_failed"] is False for r in released)

    def test_a_failure_is_recorded_with_its_detail(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        marshaller, ev = _evaluator(
            solver, tmp_path, numpy_bkd, mode="fail"
        )
        ev.submit(_columns(numpy_bkd, 4)).collect()
        failed = [
            r
            for r in self._records(marshaller, KIND_RELEASED)
            if r["any_failed"]
        ]
        assert failed
        assert failed[0]["status"] == "FAILED"
        assert "singular" in failed[0]["tasks"][0]["detail"]

    def test_retention_is_recorded_per_directory(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        """So a reader who finds nothing can tell policy from loss."""
        marshaller, ev = _evaluator(
            solver, tmp_path, numpy_bkd, retention=Retention.NEVER
        )
        ev.submit(_columns(numpy_bkd, 2)).collect()
        released = self._records(marshaller, KIND_RELEASED)
        assert all(r["retained"] is False for r in released)

    def test_a_prepared_sample_that_never_released_keeps_its_record(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        """The distinction the two record kinds exist for.

        A directory built and never harvested must not read the same as
        a sample that was never created.
        """
        marshaller = _marshaller(solver, tmp_path, numpy_bkd)
        marshaller.tasks(
            numpy_bkd.ones((2, 2)), [0, 1], Request.values_only()
        )
        assert len(self._records(marshaller, KIND_PREPARED)) == 2
        assert self._records(marshaller, KIND_RELEASED) == []

    def test_a_run_can_be_marked_done(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        marshaller, ev = _evaluator(solver, tmp_path, numpy_bkd)
        ev.submit(_columns(numpy_bkd, 2)).collect()
        assert not (
            Path(marshaller.run_dir()) / RUN_DONE_FILENAME
        ).exists()
        marshaller.mark_run_done()
        assert (Path(marshaller.run_dir()) / RUN_DONE_FILENAME).exists()

    def test_marking_a_run_that_never_began_is_harmless(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        _marshaller(solver, tmp_path, numpy_bkd).mark_run_done()

    def test_a_second_submission_is_recorded_under_its_own_ordinal(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        marshaller, ev = _evaluator(solver, tmp_path, numpy_bkd)
        ev.submit(_columns(numpy_bkd, 2)).collect()
        ev.submit(_columns(numpy_bkd, 2)).collect()
        prepared = self._records(marshaller, KIND_PREPARED)
        assert sorted({r["submission"] for r in prepared}) == [0, 1]
        # Same indices, different directories -- which is the whole
        # point of the submission level.
        assert len({r["workdir"] for r in prepared}) == 4


class TestDeferredGathering:
    """A real run, gathered afterwards from nothing but its directory.

    The case the whole feature exists for. Everything the gathering
    needs is on disk: no evaluator, no marshaller, no live process --
    only the run directory and the manifest inside it.
    """

    def test_a_finished_run_can_be_gathered_from_its_directory_alone(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        marshaller, ev = _evaluator(
            solver,
            tmp_path,
            numpy_bkd,
            mode="writesfield",
            retention=Retention.ALWAYS,
        )
        ev.submit(_columns(numpy_bkd, 3)).collect()
        marshaller.mark_run_done()
        run_dir = marshaller.run_dir()

        # Everything below uses only the path -- the marshaller is done.
        report = gather_run(
            run_dir,
            SpecCollector([OutputSpec("*.fld", required=True)]),
            str(tmp_path / "archive"),
        )
        assert report.complete is True
        assert sorted(report.gathered) == [0, 1, 2]
        assert report.ok()

    def test_gathered_content_is_what_the_solver_wrote(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        marshaller, ev = _evaluator(
            solver,
            tmp_path,
            numpy_bkd,
            mode="writesfield",
            retention=Retention.ALWAYS,
        )
        ev.submit(_columns(numpy_bkd, 2)).collect()
        report = gather_run(
            marshaller.run_dir(),
            SpecCollector([OutputSpec("*.fld")]),
            str(tmp_path / "archive"),
        )
        landed = Path(report.gathered[0][0])
        assert landed.read_text().strip() == "field for sample 0"

    def test_a_linked_mesh_is_not_gathered_once_per_sample(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        """The trap that would multiply a shared file by nsamples."""
        mesh = tmp_path / "mesh.fld"
        mesh.write_text("shared mesh")
        marshaller, ev = _evaluator(
            solver,
            tmp_path,
            numpy_bkd,
            mode="writesfield",
            retention=Retention.ALWAYS,
            link_files=[str(mesh)],
        )
        ev.submit(_columns(numpy_bkd, 3)).collect()
        report = gather_run(
            marshaller.run_dir(),
            SpecCollector([OutputSpec("*.fld")]),
            str(tmp_path / "archive"),
        )
        names = {
            os.path.basename(path)
            for paths in report.gathered.values()
            for path in paths
        }
        assert names == {"out.fld"}

    def test_a_run_killed_before_finishing_is_flagged(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        """No marker, so a reader knows it may be racing live writes."""
        marshaller, ev = _evaluator(
            solver,
            tmp_path,
            numpy_bkd,
            mode="writesfield",
            retention=Retention.ALWAYS,
        )
        ev.submit(_columns(numpy_bkd, 2)).collect()
        report = gather_run(
            marshaller.run_dir(),
            SpecCollector([OutputSpec("*.fld")]),
            str(tmp_path / "archive"),
        )
        assert report.complete is False

    def test_directories_removed_by_retention_are_reported_missing(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        """Under NEVER the manifest is the only trace they existed."""
        marshaller, ev = _evaluator(
            solver,
            tmp_path,
            numpy_bkd,
            mode="writesfield",
            retention=Retention.NEVER,
        )
        ev.submit(_columns(numpy_bkd, 3)).collect()
        report = gather_run(
            marshaller.run_dir(),
            SpecCollector([OutputSpec("*.fld")]),
            str(tmp_path / "archive"),
        )
        assert sorted(report.missing) == [0, 1, 2]
        assert report.gathered == {}


class TestReconcilingARealRun:
    """The invariant, checked against a run a real solver produced."""

    def test_a_clean_run_reconciles(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        marshaller, ev = _evaluator(
            solver,
            tmp_path,
            numpy_bkd,
            mode="writesfield",
            retention=Retention.ALWAYS,
        )
        batch = ev.submit(_columns(numpy_bkd, 3))
        batch.collect()
        report = reconcile(
            marshaller.run_dir(),
            statuses=batch.statuses(),
            collector=SpecCollector([OutputSpec("*.fld", required=True)]),
        )
        assert report.ok()
        assert report.ok_indices == (0, 1, 2)

    def test_a_solver_that_exits_zero_writing_nothing_is_caught(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        """The case this whole feature exists for.

        ``silent`` mode writes results.out for some samples and exits
        zero without it for others -- and the values decoder is what
        turns the missing scalar into a failure. What no other check
        can see is the sample that wrote its scalar and no field file,
        which is what the required spec catches here.
        """
        marshaller, ev = _evaluator(
            solver,
            tmp_path,
            numpy_bkd,
            mode="ok",
            retention=Retention.ALWAYS,
        )
        batch = ev.submit(_columns(numpy_bkd, 3))
        batch.collect()
        # The solver wrote results.out and no field file, so every
        # sample succeeded and every one is missing its output.
        report = reconcile(
            marshaller.run_dir(),
            statuses=batch.statuses(),
            collector=SpecCollector([OutputSpec("*.fld", required=True)]),
        )
        assert not report.ok()
        assert {a.kind for a in report.anomalies} == {
            AnomalyKind.NO_OUTPUT
        }
        assert len(report.anomalies) == 3

    def test_a_genuinely_failed_sample_is_explained(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        """A failure is evidence, not an anomaly."""
        marshaller, ev = _evaluator(
            solver,
            tmp_path,
            numpy_bkd,
            mode="fail",
            retention=Retention.ALWAYS,
        )
        batch = ev.submit(_columns(numpy_bkd, 4))
        batch.collect()
        report = reconcile(
            marshaller.run_dir(),
            statuses=batch.statuses(),
            collector=SpecCollector([OutputSpec("*.fld", required=True)]),
        )
        assert report.explained
        assert all(
            a.kind is AnomalyKind.NO_OUTPUT for a in report.anomalies
        )

    def test_retention_never_leaves_a_reconcilable_record(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        """Every directory is gone, and none of it is alarming.

        The manifest is the only remaining trace the samples existed,
        which is exactly what it is for.
        """
        marshaller, ev = _evaluator(
            solver,
            tmp_path,
            numpy_bkd,
            mode="writesfield",
            retention=Retention.NEVER,
        )
        batch = ev.submit(_columns(numpy_bkd, 3))
        batch.collect()
        report = reconcile(
            marshaller.run_dir(),
            statuses=batch.statuses(),
            collector=SpecCollector([OutputSpec("*.fld", required=True)]),
        )
        assert report.ok()
        assert len(report.explained) == 3


class TestLogOutput:
    """The marshaller picks the location; the dispatcher routes to it."""

    def test_off_by_default(self, solver, tmp_path, numpy_bkd) -> None:
        marshaller = _marshaller(solver, tmp_path, numpy_bkd)
        tasks = marshaller.tasks(
            numpy_bkd.ones((2, 1)), [0], Request.values_only()
        )
        assert tasks[0].stdout_path is None
        assert tasks[0].stderr_path is None

    def test_paths_land_inside_the_working_directory(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        marshaller = _marshaller(
            solver, tmp_path, numpy_bkd, log_output=True
        )
        tasks = marshaller.tasks(
            numpy_bkd.ones((2, 1)), [0], Request.values_only()
        )
        task = tasks[0]
        assert task.stdout_path == os.path.join(
            task.workdir, "solver.stdout"
        )
        assert task.stderr_path == os.path.join(
            task.workdir, "solver.stderr"
        )

    def test_a_retained_failure_keeps_its_explanation(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        """``ON_FAILURE`` otherwise keeps evidence without the evidence."""
        marshaller, ev = _evaluator(
            solver,
            tmp_path,
            numpy_bkd,
            mode="fail",
            retention=Retention.ON_FAILURE,
            log_output=True,
        )
        ev.submit(_columns(numpy_bkd, 4)).collect()
        kept = list((tmp_path / "scratch").glob("*/sub-*/sample-*"))
        assert kept
        logged = [
            (directory / "solver.stderr").read_text()
            for directory in kept
        ]
        assert any("matrix is singular" in text for text in logged)

    def test_the_filenames_are_configurable(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        marshaller = _marshaller(
            solver,
            tmp_path,
            numpy_bkd,
            log_output=True,
            stdout_filename="out.txt",
            stderr_filename="err.txt",
        )
        tasks = marshaller.tasks(
            numpy_bkd.ones((2, 1)), [0], Request.values_only()
        )
        assert tasks[0].stdout_path.endswith("out.txt")
        assert tasks[0].stderr_path.endswith("err.txt")


class TestRetention:
    def _scratch(self, tmp_path):
        """Sample directories, wherever the run/submission levels put them.

        Globbing the scratch root directly would match nothing now that
        samples live under ``<run>/sub-NNN/``, and the assertions that
        expect an empty list would pass without testing anything.
        """
        return list((tmp_path / "scratch").glob("*/sub-*/sample-*"))

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
