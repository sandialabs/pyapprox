"""Output routing, tested against the dispatcher directly.

The dispatcher's other behaviors -- throttling, timeouts, cancellation
-- are already exercised end to end through the marshaller's tests.
What is not reachable from there is where a child's stdout and stderr
go, because the marshaller only ever asked for the default. These tests
build ``ShellTask`` by hand and drive ``SubprocessDispatcher`` with no
marshaller at all, which is also the point: the task carries the log
paths, so nothing about routing needs the marshaller to be involved.

No backend fixture. Nothing here touches an array; these are processes,
descriptors and files, and running every case twice over NumPy and
Torch would double the module without testing anything new.
"""

import subprocess
import sys
import textwrap

import pytest
from pyapprox.interface.evaluation.records import JobStatus, Resources
from pyapprox.interface.evaluation.subprocess_dispatcher import (
    DETAIL_TAIL_BYTES,
    ShellTask,
    SubprocessDispatcher,
)

#: A child that writes to both streams and exits how it is told.
#:
#: ``argv[1]`` is the exit code; ``argv[2]``, when present, is a count
#: of filler lines sent to stderr, for the cases about size.
TALKER = """
import sys

nfiller = int(sys.argv[2]) if len(sys.argv) > 2 else 0
for i in range(nfiller):
    sys.stderr.write("filler line %d\\n" % i)
sys.stdout.write("solver said hello\\n")
sys.stderr.write("matrix is singular\\n")
sys.exit(int(sys.argv[1]))
"""


@pytest.fixture
def talker(tmp_path):
    path = tmp_path / "talker.py"
    path.write_text(textwrap.dedent(TALKER))
    return path


def _run(task):
    """Submit one task and wait for its outcome."""
    with SubprocessDispatcher(concurrency=1) as dispatcher:
        return dispatcher.submit([task])[0].outcome()


def _task(talker, tmp_path, code=0, nfiller=0, **kwargs):
    return ShellTask(
        indices=(0,),
        argv=[sys.executable, str(talker), str(code), str(nfiller)],
        workdir=str(tmp_path),
        **kwargs,
    )


class TestDefaultRouting:
    """Unchanged: stdout discarded, stderr piped into the detail."""

    def test_stdout_is_discarded_by_default(
        self, talker, tmp_path
    ) -> None:
        outcome = _run(_task(talker, tmp_path))
        assert outcome.status is JobStatus.SUCCEEDED
        assert list(tmp_path.glob("*.stdout")) == []

    def test_the_last_stderr_line_becomes_the_detail(
        self, talker, tmp_path
    ) -> None:
        outcome = _run(_task(talker, tmp_path, code=2))
        assert outcome.status is JobStatus.FAILED
        assert outcome.detail is not None
        assert "matrix is singular" in outcome.detail


class TestRedirectionToFiles:
    def test_stdout_reaches_the_named_file(self, talker, tmp_path) -> None:
        out = tmp_path / "solver.stdout"
        outcome = _run(_task(talker, tmp_path, stdout_path=str(out)))
        assert outcome.status is JobStatus.SUCCEEDED
        assert "solver said hello" in out.read_text()

    def test_stderr_reaches_the_named_file(self, talker, tmp_path) -> None:
        err = tmp_path / "solver.stderr"
        outcome = _run(_task(talker, tmp_path, stderr_path=str(err)))
        assert outcome.status is JobStatus.SUCCEEDED
        assert "matrix is singular" in err.read_text()

    def test_a_redirected_failure_still_reports_a_detail(
        self, talker, tmp_path
    ) -> None:
        """The detail must not be the price of keeping the log."""
        err = tmp_path / "solver.stderr"
        outcome = _run(
            _task(talker, tmp_path, code=2, stderr_path=str(err))
        )
        assert outcome.status is JobStatus.FAILED
        assert outcome.detail is not None
        assert "matrix is singular" in outcome.detail

    def test_the_streams_can_be_routed_independently(
        self, talker, tmp_path
    ) -> None:
        out = tmp_path / "only.stdout"
        outcome = _run(_task(talker, tmp_path, stdout_path=str(out)))
        assert out.exists()
        assert outcome.detail is None or "singular" not in str(
            outcome.detail
        )

    def test_logs_append_rather_than_truncate(
        self, talker, tmp_path
    ) -> None:
        """Two invocations may share one working directory."""
        out = tmp_path / "solver.stdout"
        for _ in range(2):
            _run(_task(talker, tmp_path, stdout_path=str(out)))
        assert out.read_text().count("solver said hello") == 2

    def test_an_unopenable_log_fails_only_that_sample(
        self, talker, tmp_path
    ) -> None:
        """The same rule as a missing executable."""
        outcome = _run(
            _task(
                talker,
                tmp_path,
                stdout_path=str(tmp_path / "absent" / "solver.stdout"),
            )
        )
        assert outcome.status is JobStatus.FAILED
        assert outcome.detail is not None
        assert "could not open log file" in outcome.detail


class TestTheDetailIsBounded:
    def test_a_large_log_does_not_become_the_detail(
        self, talker, tmp_path
    ) -> None:
        """A diverging solver's stderr is routinely enormous.

        The detail is one line either way; what must not happen is
        reading the whole file to find it.
        """
        err = tmp_path / "solver.stderr"
        outcome = _run(
            _task(
                talker, tmp_path, code=2, nfiller=4000, stderr_path=str(err)
            )
        )
        assert err.stat().st_size > DETAIL_TAIL_BYTES
        assert outcome.detail is not None
        assert len(outcome.detail) < 200
        assert "matrix is singular" in outcome.detail


class TestTimeoutGainsADetail:
    def test_a_timed_out_job_reports_where_it_got_to(
        self, tmp_path
    ) -> None:
        """A piped stderr is closed unread, so this is only possible
        once the child is writing somewhere seekable."""
        slow = tmp_path / "slow.py"
        slow.write_text(
            textwrap.dedent(
                """
                import sys, time
                sys.stderr.write("starting timestep 41\\n")
                sys.stderr.flush()
                time.sleep(60)
                """
            )
        )
        err = tmp_path / "solver.stderr"
        task = ShellTask(
            indices=(0,),
            argv=[sys.executable, str(slow)],
            workdir=str(tmp_path),
            resources=Resources(walltime_seconds=0.3),
            stderr_path=str(err),
        )
        outcome = _run(task)
        assert outcome.status is JobStatus.TIMED_OUT
        assert outcome.detail is not None
        assert "exceeded walltime" in outcome.detail
        assert "timestep 41" in outcome.detail


class TestDescriptorsDoNotLeak:
    def test_many_redirected_tasks_do_not_exhaust_the_table(
        self, talker, tmp_path
    ) -> None:
        """The parent's copies are dead weight once the child forks.

        Leaking two per sample would end a long sweep with ``EMFILE``
        rather than a result.
        """
        tasks = [
            _task(
                talker,
                tmp_path,
                stdout_path=str(tmp_path / f"out-{i}.log"),
                stderr_path=str(tmp_path / f"err-{i}.log"),
            )
            for i in range(40)
        ]
        with SubprocessDispatcher(concurrency=4) as dispatcher:
            handles = dispatcher.submit(tasks)
            outcomes = [handle.outcome() for handle in handles]
        assert all(
            outcome.status is JobStatus.SUCCEEDED for outcome in outcomes
        )


class TestStderrHeavyChildDoesNotDeadlock:
    """A pipe holds ~64 KB and nothing drains it until the child exits.

    With a file there is no buffer to fill. Without ``stderr_path`` this
    same child would block forever, which is why the timeout is the
    safety net rather than the assertion.
    """

    def test_a_chatty_child_completes_when_redirected(
        self, talker, tmp_path
    ) -> None:
        err = tmp_path / "solver.stderr"
        outcome = _run(
            _task(
                talker,
                tmp_path,
                nfiller=20000,
                stderr_path=str(err),
            )
        )
        assert outcome.status is JobStatus.SUCCEEDED
        assert err.stat().st_size > 64 * 1024


class TestShellTaskDefaults:
    def test_log_paths_default_to_absent(self) -> None:
        task = ShellTask(indices=(0,), argv=["true"], workdir=".")
        assert task.stdout_path is None
        assert task.stderr_path is None

    def test_the_sentinels_are_negative(self) -> None:
        """What tells a real descriptor from DEVNULL/PIPE when closing."""
        assert subprocess.DEVNULL < 0
        assert subprocess.PIPE < 0
