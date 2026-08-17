"""Dispatch to external executables.

For models that are not Python: a solver binary, a shell script, an
`mpirun` invocation. The dispatcher launches child processes, throttles
how many run at once, notices when they finish, and reports how they
exited. It never opens a file and never reads a format -- the task
carries a command and a directory, and what those mean is marshalling's
business.

**A task carries its own working directory**, which the dispatcher
passes as ``cwd`` and nothing more. Never ``os.chdir``: it is
process-global rather than per-thread, so it is unsafe the moment
anything runs in a pool, and an exception between changing directory
and changing back leaves the interpreter stranded somewhere it does not
expect, breaking every relative path afterwards. Absolute paths and
``cwd=`` remove the need for it.

**Throttling is a pending queue, advanced by pumping.** Two terms worth
defining, because a pool would hide both.

The *throttle* is the cap on how many children run at once.
``Popen`` starts a process the instant it is called, so without a cap,
submitting two thousand tasks forks two thousand solvers. Tasks beyond
the cap wait in a queue and get handles anyway, which is what lets
``submit`` return immediately while still bounding the machine.

*Pumping* is the step that advances that queue: reap any child that has
exited, freeing its slot, then launch queued tasks while slots remain.
It is explicit because nothing here runs in the background -- there are
no worker threads to notice an exit and pick up the next task, so the
queue moves only when someone calls in. ``submit`` pumps once, and every
poll pumps again.

That has one consequence worth stating: waiting on a *queued* task means
waiting for an earlier one to free a slot, so a handle poll advances the
whole dispatcher rather than just itself. Polling one handle in
isolation would wait forever on a process nothing had started.

**Polling never spins.** The interval is a constructor argument rather
than a busy loop, so waiting costs no CPU, and a test can assert that a
finished batch collects without sleeping.

**Two settings that look similar and are not.** Where a child's stdout
and stderr go -- discarded, piped, or written to a file -- is a property
of the process, so this dispatcher decides it. Whether the framework
itself announces "launched task 3" is a property of the library, and
belongs to logging configuration. Only the first is settled here; a
single knob covering both would mean something different for every
dispatcher, since a thread pool has no child output to route at all.
"""

import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    TypeVar,
)

from pyapprox.interface.evaluation.records import (
    ComputeProvenance,
    JobStatus,
    Outcome,
    Resources,
)

Payload = TypeVar("Payload")

#: How often to check whether children have finished, in seconds.
#:
#: Explicit rather than a busy loop, which would burn a core watching
#: processes that take minutes. Small enough that a short test is not
#: dominated by it.
DEFAULT_POLL_INTERVAL = 0.01


@dataclass(frozen=True)
class ShellTask:
    """A command to run, and where to run it.

    Attributes
    ----------
    indices : Sequence[int]
        Which columns of the submitted batch this task covers.
    argv : Sequence[str]
        The command and its arguments. A list rather than a string, so
        nothing is passed through a shell and no argument needs quoting.
    workdir : str
        Absolute path the child runs in. Created and owned by the
        marshaller, which also decides whether it survives.
    resources : Resources
        What the task needs. Only ``ncores`` is read here, for
        accounting; a scheduler dispatcher would read the rest.
    env : Mapping[str, str], optional
        Extra environment variables for the child, merged over the
        parent's. ``None`` inherits unchanged.
    """

    indices: Sequence[int]
    argv: Sequence[str]
    workdir: str
    resources: Resources = field(default_factory=Resources)
    env: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class ShellPayload:
    """What a finished child left behind.

    Deliberately not the output itself. The dispatcher reports where the
    work happened and how it ended; reading ``results.out`` is
    marshalling's job, and a dispatcher that parsed it would be tied to
    one file format.
    """

    workdir: str
    returncode: int


class SubprocessJobHandle(Generic[Payload]):
    """One child process, or one waiting to become one."""

    def __init__(
        self,
        task: ShellTask,
        poll_interval: float,
        pump: Optional[Callable[[], None]] = None,
    ) -> None:
        self._task = task
        self._poll_interval = poll_interval
        # Waiting on a queued job means waiting for a slot, and only the
        # dispatcher can hand one out. Without this the handle would
        # spin on a job that can never start by itself, and a batch
        # larger than the throttle would never finish.
        self._pump = pump
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._started: Optional[float] = None
        self._outcome: Optional[Outcome[ShellTask, ShellPayload]] = None
        self._cancelled = False

    def is_pending(self) -> bool:
        """Whether this job is waiting for a slot."""
        return (
            self._process is None
            and self._outcome is None
            and not self._cancelled
        )

    def is_running(self) -> bool:
        """Whether a child is alive for this job."""
        return self._process is not None and self._outcome is None

    def launch(self) -> None:
        """Start the child. Called by the dispatcher when a slot frees."""
        if not self.is_pending():
            return
        self._started = time.perf_counter()
        env = None
        if self._task.env is not None:
            env = {**os.environ, **self._task.env}
        try:
            self._process = subprocess.Popen(
                list(self._task.argv),
                cwd=self._task.workdir,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                # Its own process group, so cancelling can signal the
                # whole tree. A solver that spawns children of its own
                # would otherwise leave them running after the parent
                # was killed.
                start_new_session=True,
            )
        except OSError as exc:
            # A missing executable is a failed sample, not a failed
            # batch -- the same rule as a solver that diverges.
            self._outcome = Outcome(
                task=self._task,
                indices=self._task.indices,
                status=JobStatus.FAILED,
                resources=self._task.resources,
                detail=f"could not start: {exc}",
            )

    def poll(self) -> None:
        """Reap the child if it has finished. Never blocks."""
        if self._process is None or self._outcome is not None:
            return
        returncode = self._process.poll()
        if returncode is None:
            return
        self._finish(returncode)

    def _finish(self, returncode: int) -> None:
        """Record how a child ended, reading whatever it said on stderr."""
        process = self._process
        if process is None:
            raise RuntimeError(
                "a job cannot finish before it has started"
            )
        stderr = b""
        if process.stderr is not None:
            stderr = process.stderr.read()
            process.stderr.close()
        elapsed = (
            0.0
            if self._started is None
            else time.perf_counter() - self._started
        )
        if returncode == 0:
            self._outcome = Outcome(
                task=self._task,
                indices=self._task.indices,
                status=JobStatus.SUCCEEDED,
                payload=ShellPayload(
                    workdir=self._task.workdir, returncode=returncode
                ),
                wall_time=elapsed,
                resources=self._task.resources,
            )
            return
        # A non-zero exit is evidence about this parameter point, so it
        # is FAILED rather than retryable. The code and the last line of
        # stderr are kept because "the solver failed" is not actionable
        # and "exit 2: matrix is singular" is.
        detail = f"exit code {returncode}"
        message = stderr.decode("utf-8", errors="replace").strip()
        if message:
            detail = f"{detail}: {message.splitlines()[-1]}"
        self._outcome = Outcome(
            task=self._task,
            indices=self._task.indices,
            status=JobStatus.FAILED,
            wall_time=elapsed,
            resources=self._task.resources,
            detail=detail,
        )

    def check_timeout(self) -> None:
        """Kill a child that has outrun its walltime.

        Without this a hung solver hangs the whole study, since nothing
        else will ever notice. A killed job is ``TIMED_OUT`` rather than
        ``FAILED``, because a deadline says nothing about the parameter
        point and the sample may be worth resubmitting.
        """
        limit = self._task.resources.walltime_seconds
        if limit is None or not self.is_running() or self._started is None:
            return
        if time.perf_counter() - self._started < limit:
            return
        self._terminate()
        self._outcome = Outcome(
            task=self._task,
            indices=self._task.indices,
            status=JobStatus.TIMED_OUT,
            wall_time=time.perf_counter() - self._started,
            resources=self._task.resources,
            detail=f"exceeded walltime of {limit}s",
        )

    def _terminate(self) -> None:
        """Signal the child's whole process group, then reap it.

        The group rather than the process, so a solver that spawned
        workers of its own does not leave them orphaned.
        """
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            process.terminate()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
            process.wait()
        if process.stderr is not None:
            process.stderr.close()

    def done(self) -> bool:
        """Whether this job has finished, without blocking."""
        self.poll()
        return self._outcome is not None

    def outcome(
        self, timeout: Optional[float] = None
    ) -> Outcome[ShellTask, ShellPayload]:
        """What the job produced, waiting up to ``timeout`` seconds.

        Returns an ``OUTSTANDING`` outcome on expiry rather than
        raising, and sleeps between polls rather than spinning.
        """
        deadline = (
            None if timeout is None else time.perf_counter() + timeout
        )
        while self._outcome is None:
            # Advance the whole queue, not just this job: waiting on a
            # queued task means waiting for an earlier one to free a
            # slot, so polling this handle alone would wait forever.
            if self._pump is not None:
                self._pump()
            self.poll()
            self.check_timeout()
            if self._outcome is not None:
                break
            if deadline is not None and time.perf_counter() >= deadline:
                break
            time.sleep(self._poll_interval)
        if self._outcome is None:
            return Outcome(
                task=self._task,
                indices=self._task.indices,
                status=JobStatus.OUTSTANDING,
            )
        return self._outcome

    def cancel(self) -> bool:
        """Stop the job, whether queued or running.

        Unlike an executor, a subprocess dispatcher **can** interrupt
        work already started: it holds the child and signals its process
        group. So this offers the stronger of the two guarantees the
        protocol allows, and returns ``True`` for a running job as well
        as a pending one.
        """
        if self._outcome is not None:
            return False
        self._cancelled = True
        if self._process is not None:
            self._terminate()
            elapsed = (
                0.0
                if self._started is None
                else time.perf_counter() - self._started
            )
            self._outcome = Outcome(
                task=self._task,
                indices=self._task.indices,
                status=JobStatus.CANCELLED,
                wall_time=elapsed,
                resources=self._task.resources,
                detail="terminated while running",
            )
        else:
            self._outcome = Outcome(
                task=self._task,
                indices=self._task.indices,
                status=JobStatus.CANCELLED,
                resources=self._task.resources,
                detail="cancelled before it started",
            )
        return True


class SubprocessDispatcher:
    """Runs external commands, throttled, without blocking submit.

    Parameters
    ----------
    concurrency : int
        How many children may run at once.
    poll_interval : float
        Seconds between checks for finished children. Explicit rather
        than a busy loop, so waiting costs no CPU.
    """

    def __init__(
        self,
        concurrency: int,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> None:
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")
        if poll_interval <= 0.0:
            raise ValueError(
                f"poll_interval must be positive, got {poll_interval}"
            )
        self._concurrency = concurrency
        self._poll_interval = poll_interval
        self._handles: List[SubprocessJobHandle[ShellPayload]] = []
        self._closed = False

    def concurrency(self) -> int:
        """How many children may run at once."""
        return self._concurrency

    def compute_provenance(self) -> ComputeProvenance:
        """Measured: each child is timed from launch to exit."""
        return ComputeProvenance.MEASURED

    def submit(
        self, tasks: Sequence[ShellTask]
    ) -> Sequence[SubprocessJobHandle[ShellPayload]]:
        """Queue tasks and launch what the throttle allows.

        Returns immediately. Tasks beyond the throttle wait in the
        pending queue and start as slots free -- ``Popen`` launches a
        process the moment it is called, so the queue is what bounds how
        many solvers run at once.
        """
        if self._closed:
            raise RuntimeError("cannot submit to a closed dispatcher")
        handles: List[SubprocessJobHandle[ShellPayload]] = [
            SubprocessJobHandle(task, self._poll_interval, self._pump)
            for task in tasks
        ]
        self._handles.extend(handles)
        self._pump()
        return handles

    def _pump(self) -> None:
        """Reap finished children and launch queued ones.

        Called from ``submit`` and from every poll, so the queue drains
        without a background thread. Two passes over the handle list:
        finish first, then launch, so a slot freed by one child is
        available to the next in the same call.
        """
        for handle in self._handles:
            handle.poll()
            handle.check_timeout()
        running = sum(1 for h in self._handles if h.is_running())
        for handle in self._handles:
            if running >= self._concurrency:
                break
            if handle.is_pending():
                handle.launch()
                running += 1

    def pump(self) -> None:
        """Advance the queue: reap what finished, start what fits.

        A caller polling handles directly drives this implicitly, but an
        ensemble waiting on several models may need to advance one
        dispatcher without asking a particular handle.
        """
        self._pump()

    def close(self) -> None:
        """Stop everything still running, and refuse further work.

        Idempotent. Unlike a thread pool, leaking here leaves live
        processes behind, so this terminates rather than merely
        declining new submissions.
        """
        self._closed = True
        for handle in self._handles:
            if handle.is_running() or handle.is_pending():
                handle.cancel()

    def __enter__(self) -> "SubprocessDispatcher":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
