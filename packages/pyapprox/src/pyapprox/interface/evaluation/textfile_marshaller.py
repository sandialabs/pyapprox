"""Marshalling to and from files on disk.

For a solver that reads its inputs from a file and writes its outputs to
another: the marshaller creates a working directory per task, writes the
samples into it, builds the command, and afterwards reads the results
back. The dispatcher only runs the command in that directory.

**The marshaller owns working directories, end to end.** It creates
them, writes into them, puts the path in the task, and decides in
``release`` whether they survive. The alternative -- dispatch owning
them -- cannot work: the marshaller has to write inputs *before* the
command runs, so the directory must exist before dispatch sees the task.
Assigning them to dispatch would also be dishonest, since a thread pool
or an HTTP client has no working directory at all.

**Directories are scoped by run and submission**, as
``<run_id>/sub-NNN/sample-NNNNNN``. The sample number alone will not do:
it is a column in the submitted batch, so it restarts at zero every
time, and one marshaller is submitted to many times. Without the
submission level a second submission would rebuild the first's paths and
overwrite its inputs. The run level is what makes a sweep quotable --
"gather from run 20260902T143011Z-3f9b21c40a17" is an instruction
someone can act on a month later -- and gives its directories one name
to archive or delete. Names are zero-padded so they sort numerically as
text.

**Retention has two axes, not one.** Whether a scratch directory
survives, and whether a summary artifact is written, are separate
questions -- a caller may want the results file without the multi-
gigabyte scratch, or the scratch for debugging without an artifact.
Folding them into one setting makes at least one combination
unreachable, and makes "no" ambiguous.

Failure keeps its evidence by default: a directory whose solve failed is
retained even when successful ones are discarded, because that is when
someone will want to look inside.

**Never ``os.chdir``.** The command runs with ``cwd`` set by the
dispatcher and every path here is absolute. Changing the interpreter's
directory is process-global rather than per-thread, so it corrupts
anything running in parallel, and an exception before changing back
leaves every later relative path resolving somewhere unintended.
"""

import datetime
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from pyapprox.interface.evaluation.protocols import MarshalError
from pyapprox.interface.evaluation.records import (
    Decoded,
    JobStatus,
    Outcome,
    Request,
    Resources,
)
from pyapprox.interface.evaluation.subprocess_dispatcher import (
    ShellPayload,
    ShellTask,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.util.backends.protocols import Array, Backend

#: What the solver reads and writes, relative to its working directory.
DEFAULT_PARAMS_FILENAME = "params.in"
DEFAULT_RESULTS_FILENAME = "results.out"

#: Where a solver's own output goes when ``log_output`` is on. Named by
#: the framework rather than the solver, so a code that writes its own
#: ``stdout.log`` is not silently overwritten.
DEFAULT_STDOUT_FILENAME = "solver.stdout"
DEFAULT_STDERR_FILENAME = "solver.stderr"

#: What a caller-supplied ``run_id`` may contain. A run id becomes a
#: path component, so ``../..`` would escape the scratch root and
#: ``a/b`` would nest silently.
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+")

#: How many times to retry an auto-generated run id before giving up.
#:
#: A collision on an id the library invented is not the caller's
#: mistake, so it retries rather than raising. Bounded because a
#: scratch root that refuses every mkdir must not spin.
_RUN_ID_ATTEMPTS = 5

#: Digits in the zero-padded directory names.
#:
#: Padded so names sort numerically as text: ``sample-10`` before
#: ``sample-2`` is the trap ``protocols.py`` documents for output
#: listings, and it is cheaper to remove at the source than to require
#: every consumer to sort by a parsed integer.
_SUBMISSION_DIGITS = 3
_SAMPLE_DIGITS = 6


def _generate_run_id() -> str:
    """A run id that sorts by time and does not collide.

    Twelve hex digits rather than four: ranks starting in the same
    second are a birthday problem, and four digits collide about 7% of
    the time at 100 ranks and 26% at 200. Four extra characters in a
    string written down once cost nothing.
    """
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    return f"{stamp}-{uuid.uuid4().hex[:12]}"


def _highest_submission(run_dir: Path) -> int:
    """The largest ``sub-NNN`` ordinal present, or -1 if there is none.

    Read back from the directory so a resumed run continues after what
    is already there. Unparseable names are ignored rather than
    refused: the run directory is also where the manifest lives, and a
    user may well have left something beside it.
    """
    highest = -1
    for entry in run_dir.iterdir():
        matched = re.fullmatch(r"sub-(\d+)", entry.name)
        if matched is not None and entry.is_dir():
            highest = max(highest, int(matched.group(1)))
    return highest


class OnExisting(Enum):
    """What to do when a named run directory is already there.

    Only consulted for a ``run_id`` the caller supplied. An
    auto-generated id that collides is retried with fresh entropy
    instead, because that collision says nothing about the caller's
    intent.
    """

    #: Refuse, before any job is built.
    ERROR = "error"
    #: Reuse the directory, continuing after its existing submissions.
    RESUME = "resume"
    #: Leave it alone and allocate a fresh id.
    NEW = "new"

#: How many numbers one sample's worth of each quantity holds, and the
#: shape it takes in a :class:`Decoded`, keyed by derivative-bundle
#: field name.
#:
#: Shipped rather than left to each marshaller because the conventions
#: are not uniform and getting one wrong **transposes a result rather
#: than raising**. Jacobians, Hessians and Hessian-vector products are
#: sample-*first*; values and jacobian-vector products are sample-*last*.
#: A Hessian-vector product additionally carries no ``nqoi`` axis at
#: all -- its batch form is scalar-implicit, which the derivative bundle
#: itself flags as a recurring trap when reshaping.
#:
#: Bundle field names rather than a new vocabulary, so there is nothing
#: extra to learn and the mapping to result fields is the one the
#: records already publish.
QUANTITY_LAYOUT: Dict[str, "_QuantityLayout"] = {}


@dataclass(frozen=True)
class _QuantityLayout:
    """Where one quantity lands, and what shape it takes."""

    field: str
    count: Callable[[int, int], int]
    shape: Callable[[int, int], Tuple[int, ...]]


QUANTITY_LAYOUT.update(
    {
        "values": _QuantityLayout(
            "values",
            lambda nqoi, nvars: nqoi,
            lambda nqoi, nvars: (nqoi, 1),
        ),
        "jacobian_batch": _QuantityLayout(
            "jacobians",
            lambda nqoi, nvars: nqoi * nvars,
            lambda nqoi, nvars: (1, nqoi, nvars),
        ),
        "hessian_batch": _QuantityLayout(
            "hessians",
            lambda nqoi, nvars: nvars * nvars,
            lambda nqoi, nvars: (1, nvars, nvars),
        ),
        "jvp": _QuantityLayout(
            "jvps",
            lambda nqoi, nvars: nqoi,
            lambda nqoi, nvars: (nqoi, 1),
        ),
        "hvp_batch": _QuantityLayout(
            "hvps",
            lambda nqoi, nvars: nvars,
            lambda nqoi, nvars: (1, nvars),
        ),
        "whvp_batch": _QuantityLayout(
            "hvps",
            lambda nqoi, nvars: nvars,
            lambda nqoi, nvars: (1, nvars),
        ),
    }
)


class Retention(Enum):
    """When a task's scratch directory survives its task."""

    ALWAYS = "always"
    """Keep every directory. For debugging, or when the solver writes
    output worth more than the disk it occupies."""

    ON_FAILURE = "on_failure"
    """Keep only directories whose solve failed.

    The default: a successful run has already yielded its numbers, while
    a failed one holds the only evidence of why.
    """

    NEVER = "never"
    """Discard every directory once its results are read."""


@dataclass(frozen=True)
class _Layout:
    """Where one task's files live. Absolute paths throughout."""

    workdir: Path
    params: Path
    results: Path


class TextFileMarshaller(Generic[Array]):
    """Writes samples to a file, runs a command, reads values back.

    Parameters
    ----------
    command : Sequence[str]
        The solver and its fixed arguments. The working directory is
        supplied by the dispatcher rather than appended here, so the
        command needs no per-task substitution.
    bkd : Backend[Array]
        Backend used to build the decoded arrays.
    nvars : int
        Number of input variables.
    nqoi : int
        Number of quantities of interest the solver writes.
    scratch_root : str
        Parent directory for per-task working directories.
    link_files : Sequence[str], optional
        Files each working directory needs -- meshes, configuration,
        restart data. Symlinked rather than copied, since a mesh may be
        large and every task wants the same one.
    retention : Retention
        Whether working directories survive. Defaults to keeping only
        the failures.
    resources : Resources, optional
        What one solve needs from the machine.
    params_filename, results_filename : str
        What the solver reads and writes, relative to its directory.
    log_output : bool
        Send each child's stdout and stderr to files in its working
        directory instead of discarding stdout and piping stderr. Off by
        default, since it writes two files per sample. Worth turning on
        whenever a directory is retained for diagnosis: stderr otherwise
        survives only as its last line, stdout not at all, and a job
        killed on walltime reports nothing about itself.
    stdout_filename, stderr_filename : str
        Names for those files, relative to the working directory.
    run_id : str, optional
        Names this run's directory under ``scratch_root``. Generated
        from the time and some entropy when omitted. Worth supplying
        when a run has to be found again later, since a name chosen by
        the caller is one they can quote; must match
        ``[A-Za-z0-9._-]+``, because it becomes a path component.
    on_existing : OnExisting
        What to do when a supplied ``run_id`` is already there. Ignored
        for a generated id, which is simply retried on collision.
    """

    def __init__(
        self,
        command: Sequence[str],
        bkd: Backend[Array],
        nvars: int,
        nqoi: int,
        scratch_root: str,
        link_files: Optional[Sequence[str]] = None,
        retention: Retention = Retention.ON_FAILURE,
        resources: Optional[Resources] = None,
        params_filename: str = DEFAULT_PARAMS_FILENAME,
        results_filename: str = DEFAULT_RESULTS_FILENAME,
        log_output: bool = False,
        stdout_filename: str = DEFAULT_STDOUT_FILENAME,
        stderr_filename: str = DEFAULT_STDERR_FILENAME,
        run_id: Optional[str] = None,
        on_existing: OnExisting = OnExisting.ERROR,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        if nvars < 1:
            raise ValueError(f"nvars must be >= 1, got {nvars}")
        if nqoi < 1:
            raise ValueError(f"nqoi must be >= 1, got {nqoi}")
        self._command = list(command)
        self._bkd = bkd
        self._nvars = nvars
        self._nqoi = nqoi
        self._scratch_root = Path(scratch_root).resolve()
        self._link_files = [
            Path(path).resolve() for path in (link_files or ())
        ]
        self._retention = retention
        self._resources = (
            Resources() if resources is None else resources
        )
        self._params_filename = params_filename
        self._results_filename = results_filename
        self._log_output = log_output
        self._stdout_filename = stdout_filename
        self._stderr_filename = stderr_filename
        # ``.`` and ``..`` match the pattern -- the dot is legal inside a
        # name -- and both are traversal, so they are excluded by name
        # rather than by the character class.
        if run_id is not None and (
            not _RUN_ID_PATTERN.fullmatch(run_id) or run_id in (".", "..")
        ):
            raise ValueError(
                f"run_id {run_id!r} must match [A-Za-z0-9._-]+ and be "
                "neither '.' nor '..'; it becomes a directory name, so a "
                "separator would nest it and '..' would escape the "
                "scratch root"
            )
        self._requested_run_id = run_id
        self._on_existing = on_existing
        # Claimed at the first ``tasks``, not here. Creating it now would
        # give the constructor a filesystem side effect and leave a
        # directory behind for every marshaller that is built and never
        # used -- an ensemble rebuilt per iteration builds one per model
        # per iteration. Nothing is lost: ``submit`` builds every task
        # before dispatching any, so a failure at the first ``tasks``
        # still precedes every solver launch.
        self._run_dir: Optional[Path] = None
        # -1 rather than 0 because ``begin_submission`` increments before
        # use, and because "no submission has begun" has to be
        # distinguishable from "the first one has".
        self._submission = -1
        self._check_link_files()
        self._check_scratch_root()
        # How many tasks still expect each working directory to exist.
        #
        # A directory belongs to a *sample*, but ``release`` is called
        # per *task*, and one sample may be covered by several: a solver
        # computing its values and its derivatives by separate
        # invocations shares one directory between them. Deleting on the
        # first release would remove the inputs the second still has to
        # read -- and it would do so silently, since that task simply
        # finds no output where it expected one.
        self._outstanding: Dict[str, int] = {}
        # Directories whose sample failed in at least one invocation, so
        # a retained-on-failure directory is kept when any of the tasks
        # sharing it failed rather than only the last to be released.
        self._failed_dirs: Set[str] = set()

    def _check_link_files(self) -> None:
        """Refuse unusable link sources before any job is built.

        ``_prepare`` checks the same thing, but per sample and after
        ``submit`` -- so a typo in a mesh path is discovered once per
        sample rather than once, and on a scheduler every rank discovers
        it independently after the allocation has started. The per-sample
        check stays as a backstop, since a file can be deleted between
        construction and dispatch; this one is what makes the common case
        cheap to diagnose.

        ``ValueError`` rather than ``MarshalError``: nothing has been
        marshalled yet, and a bad argument to a constructor is what
        ``ValueError`` is for.
        """
        for source in self._link_files:
            if not source.exists():
                raise ValueError(
                    f"cannot link {source}: it does not exist"
                )
            if not os.access(source, os.R_OK):
                raise ValueError(f"cannot link {source}: it is not readable")

    def _check_scratch_root(self) -> None:
        """Refuse a scratch root that cannot hold working directories.

        Probed by creating and removing a directory rather than by
        testing permission bits, which answer a different question on a
        read-only mount, over NFS with root-squash, or under an ACL.

        The probe is removed again and no run directory is claimed, so
        constructing a marshaller still leaves nothing behind -- a
        constructed-and-unused marshaller is common enough (an ensemble
        rebuilt per iteration) that a residue per construction would be
        its own problem.
        """
        try:
            self._scratch_root.mkdir(parents=True, exist_ok=True)
            probe = Path(tempfile.mkdtemp(dir=self._scratch_root))
        except OSError as exc:
            raise ValueError(
                f"scratch_root {self._scratch_root} is not usable: {exc}"
            ) from exc
        probe.rmdir()

    def _claim_run_dir(self) -> Path:
        """Create the directory this run's samples live under.

        ``mkdir(exist_ok=False)`` is the claim: it creates or raises, in
        one operation. Checking for the name first and then creating it
        is check-then-act, and two array-job ranks starting in the same
        second would both see nothing and both proceed.

        An auto-generated id that collides is retried with fresh
        entropy, because the library invented that id and a collision
        says nothing about what the caller asked for. Retrying also
        covers a quirk of NFS: a retransmitted MKDIR can return EEXIST
        to the client whose original request actually succeeded, so a
        legitimate first creator can see a collision that never
        happened.
        """
        if self._requested_run_id is not None:
            return self._claim_named_run_dir(self._requested_run_id)
        for _ in range(_RUN_ID_ATTEMPTS):
            candidate = self._scratch_root / _generate_run_id()
            try:
                candidate.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            return candidate
        raise MarshalError(
            f"could not claim a run directory under {self._scratch_root} "
            f"after {_RUN_ID_ATTEMPTS} attempts"
        )

    def _claim_named_run_dir(self, run_id: str) -> Path:
        """Claim a directory for an id the caller chose."""
        candidate = self._scratch_root / run_id
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            if self._on_existing is OnExisting.RESUME:
                return candidate
            if self._on_existing is OnExisting.NEW:
                self._requested_run_id = None
                return self._claim_run_dir()
            raise MarshalError(
                f"run directory {candidate} already exists; pass "
                "on_existing=OnExisting.RESUME to continue it or "
                "OnExisting.NEW to allocate a fresh id"
            ) from None
        return candidate

    def run_dir(self) -> Optional[str]:
        """Where this run's directories live, once one has been claimed.

        ``None`` until the first submission, since nothing is created
        before then.
        """
        return None if self._run_dir is None else str(self._run_dir)

    def begin_submission(self) -> None:
        """Start a new submission, giving its samples their own level.

        Batch-local indices restart at zero on every submission, so
        without this a second submission would rebuild the same
        ``sample-NNNNNN`` paths -- overwriting the first submission's
        inputs, and corrupting the refcount that decides when a
        directory may be deleted, since that is keyed by path.

        The ordinal is recovered from the directory rather than kept
        only in memory, so a resumed run continues after the
        submissions already there instead of overwriting ``sub-000``.
        The directory is the record; a counter in a file would be a
        second one to keep in step.
        """
        if self._run_dir is None:
            self._run_dir = self._claim_run_dir()
            self._submission = _highest_submission(self._run_dir)
        self._submission += 1
        (self._run_dir / self._submission_name()).mkdir(exist_ok=True)

    def _submission_name(self) -> str:
        return f"sub-{self._submission:0{_SUBMISSION_DIGITS}d}"

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Number of input variables."""
        return self._nvars

    def nqoi(self) -> int:
        """Number of quantities of interest."""
        return self._nqoi

    def max_samples_per_task(self) -> int:
        """One.

        A working directory holds one solve's inputs and outputs, so a
        task is inherently per-sample. This is the case the evaluator's
        grouping rule reduces to one-task-per-sample for, and it is why
        that rule takes a marshaller's own limit rather than deciding
        alone.
        """
        return 1

    def derivatives(self) -> Derivatives[Array]:
        """None: this reads values and nothing else.

        A solver that also produces derivatives -- by whatever means, an
        adjoint solve, a tangent-linear one, algorithmic
        differentiation, or hand-coded formulae, none of which this
        layer can distinguish or needs to -- populates the bundle in a
        subclass and overrides three things:

        - :meth:`write_inputs`, for a quantity whose computation takes
          an input of its own. A jacobian-vector or Hessian-vector
          product is evaluated *with* its direction, so the direction is
          written into the input file rather than applied to a returned
          matrix. Use :meth:`direction_for` to slice it;
        - :meth:`commands_for`, where a second executable computes them.
          Use :meth:`quantities_in` to read the request;
        - :meth:`values`, using :meth:`read_numbers` for each extra file
          and :meth:`decode_quantity` to reshape what it holds.

        Those three helpers exist because the work they do is identical
        for every solver and silently wrong when hand-written: the
        direction must be indexed by the sample's batch column rather
        than its position in a task, and each quantity has its own axis
        convention, so a wrong one transposes a result instead of
        raising.

        Everything else -- directory creation, symlinking, retention,
        grouping -- is unchanged.
        """
        return Derivatives.none()

    def tasks(
        self,
        samples: Array,
        indices: Sequence[int],
        request: Request[Array],
    ) -> Sequence[ShellTask]:
        """Create a directory per sample, write inputs, build commands.

        The directory must exist before the task does, because the task
        carries its path and the dispatcher will run there -- which is
        why directory creation belongs to marshalling rather than to
        dispatch.
        """
        built: List[ShellTask] = []
        for position, index in enumerate(indices):
            layout = self._prepare(index)
            self.write_inputs(
                layout, samples[:, position : position + 1], index, request
            )
            commands = [
                self._with_logs(task)
                for task in self.commands_for(layout, index, request)
            ]
            self._outstanding[str(layout.workdir)] = len(commands)
            built.extend(commands)
        return built

    def _with_logs(self, task: ShellTask) -> ShellTask:
        """Point a task's output at files inside its working directory.

        Applied here rather than in :meth:`commands_for` so that every
        subclass gets it without knowing about it, and so a subclass
        that deliberately set its own paths keeps them.

        Several tasks may share one directory -- a solver invoked once
        for values and again for a jacobian -- so the files are opened
        for append and both invocations accumulate into one log rather
        than the second erasing the first.
        """
        if not self._log_output:
            return task
        if task.stdout_path is not None or task.stderr_path is not None:
            return task
        workdir = Path(task.workdir)
        return replace(
            task,
            stdout_path=str(workdir / self._stdout_filename),
            stderr_path=str(workdir / self._stderr_filename),
        )

    def write_inputs(
        self,
        layout: _Layout,
        sample: Array,
        index: int,
        request: Request[Array],
    ) -> None:
        """Write everything the solver needs to read for one sample.

        Here, just the sample. It takes the request because that is not
        the only thing a solver may need: a directional derivative --
        a jacobian-vector or Hessian-vector product -- is computed
        *with* its direction as an input, so a marshaller supporting one
        writes ``request.jvp_vecs`` or ``request.hvp_vecs`` alongside
        the sample, and the weighted form writes ``hvp_weights`` too.

        ``index`` is the sample's column in the **submitted batch**,
        which is what indexes those vectors: they span the submission,
        not this task. Slicing them by a position within the task
        instead would hand every task the first sample's direction --
        correct-looking output for the wrong input, and invisible
        wherever a task holds a single sample.

        A subclass overrides this and leaves the directory creation,
        symlinking and grouping above untouched.
        """
        self._write_params(layout, sample)

    def commands_for(
        self, layout: _Layout, index: int, request: Request[Array]
    ) -> Sequence[ShellTask]:
        """Which commands satisfy ``request`` for one prepared directory.

        One task running one command, for a solver whose single
        invocation produces everything asked of it.

        Separated from :meth:`tasks` because that is not the only shape.
        A code that computes its derivatives in a second executable
        answers a request for values and a jacobian with *two* tasks over
        the same directory, and a subclass expresses that by overriding
        this alone.

        A request for anything this marshaller cannot produce is refused
        here rather than in :meth:`tasks`, because which quantities are
        available is exactly what a subclass changes -- and the base
        class refusing on its behalf would make the seam unusable.

        **Tasks returned here are independent.** They share a working
        directory but the dispatcher may run them in any order, or at
        the same time, so none may depend on another having finished. A
        solver whose derivative step reuses the forward solution is
        therefore one invocation that writes both, not two -- express it
        by returning a single task and decoding several files from it.
        Returning two would read a forward solution that may not exist
        yet, and only sometimes.

        **Ask for everything in one request.** Requesting values now and
        a jacobian later means two submissions, and the directory from
        the first is gone by the second, so the sample is solved again
        from scratch. Directory setup is milliseconds and irrelevant;
        the repeated solve is the cost, and for a code that could have
        produced both in one invocation it is the entire cost. Whether
        the second request can be avoided at all is the caller's to
        know, since only the caller can say two submissions concern the
        same point.
        """
        if not request.values:
            raise MarshalError(
                "this marshaller produces values and nothing else, so a "
                "request for anything else has no command to run"
            )
        return [
            ShellTask(
                indices=(index,),
                argv=self._command,
                workdir=str(layout.workdir),
                resources=self._resources,
            )
        ]

    def quantities_in(self, request: Request[Array]) -> List[str]:
        """Which quantities a request asks for, by bundle field name.

        Restricted to what :meth:`derivatives` advertises, so a
        marshaller is never asked to produce something it does not
        offer. Shipped rather than left to each subclass because the
        chain is the same for everyone and the names have to match the
        bundle's exactly.
        """
        derivs = self.derivatives()
        wanted: List[str] = []
        if request.values:
            wanted.append("values")
        if request.jacobians and derivs.jacobian_batch is not None:
            wanted.append("jacobian_batch")
        if request.hessians and derivs.hessian_batch is not None:
            wanted.append("hessian_batch")
        if request.wants_jvp() and derivs.jvp is not None:
            wanted.append("jvp")
        if request.wants_hvp():
            if request.is_weighted_hvp():
                if derivs.whvp_batch is not None:
                    wanted.append("whvp_batch")
            elif derivs.hvp_batch is not None:
                wanted.append("hvp_batch")
        return wanted

    def direction_for(
        self, request: Request[Array], index: int
    ) -> Optional[Array]:
        """This sample's direction vector, or ``None`` if none applies.

        ``index`` is the sample's column in the **submitted batch**,
        which is what indexes these vectors: a request carries one per
        sample across the whole submission, not per task. Slicing by a
        position within the task instead gives every task the *first*
        sample's direction -- an answer of the right shape to the wrong
        question, and invisible wherever a task holds one sample.

        Shipped for exactly that reason: the slice is identical for
        every marshaller and silently wrong when it is not.
        """
        vectors = (
            request.hvp_vecs
            if request.hvp_vecs is not None
            else request.jvp_vecs
        )
        if vectors is None:
            return None
        return vectors[:, index : index + 1]

    def decode_quantity(
        self, quantity: str, numbers: Sequence[float]
    ) -> Tuple[str, Array]:
        """Reshape one quantity's numbers, and say which field it fills.

        Returns the ``Decoded`` field name and the array, so a subclass
        assembles a record without having to remember which axis a
        quantity uses -- the part that transposes silently when wrong.

        Numbers are expected flat and in the order the shape implies,
        row-major for a jacobian.
        """
        layout = QUANTITY_LAYOUT.get(quantity)
        if layout is None:
            raise MarshalError(
                f"unknown quantity {quantity!r}; expected one of "
                f"{sorted(QUANTITY_LAYOUT)}"
            )
        nqoi, nvars = self._nqoi, self._nvars
        expected = layout.count(nqoi, nvars)
        if len(numbers) != expected:
            raise MarshalError(
                f"{quantity} holds {len(numbers)} numbers, expected "
                f"{expected}"
            )
        return layout.field, self._bkd.reshape(
            self._bkd.asarray(list(numbers)), layout.shape(nqoi, nvars)
        )

    def _prepare(self, index: int) -> _Layout:
        """Make a working directory and link whatever it needs.

        The path is ``<run>/sub-NNN/sample-NNNNNN``, deterministic
        rather than randomized. What keeps it unique is the submission
        level, since the index alone restarts at zero on every
        submission; what makes it worth having is that a name a person
        can predict is a name they can be told to look in a month
        later.

        Called once per *sample*, not once per task. Several
        invocations may share this directory -- a solver run once for
        values and again for a jacobian reads the inputs written here
        exactly once -- so preparing per task would give each quantity
        its own directory and its own copy of the inputs.
        """
        if self._run_dir is None:
            # A marshaller driven directly, without an evaluator to
            # announce the submission. One submission is the honest
            # reading of it.
            self.begin_submission()
        if self._run_dir is None:
            raise MarshalError("no run directory could be claimed")
        workdir = (
            self._run_dir
            / self._submission_name()
            / f"sample-{index:0{_SAMPLE_DIGITS}d}"
        )
        workdir.mkdir(parents=True)
        for source in self._link_files:
            if not source.exists():
                raise MarshalError(
                    f"cannot link {source}: it does not exist"
                )
            (workdir / source.name).symlink_to(source)
        return _Layout(
            workdir=workdir,
            params=workdir / self._params_filename,
            results=workdir / self._results_filename,
        )

    def _write_params(self, layout: _Layout, sample: Array) -> None:
        """Write one sample, one value per line."""
        values = [
            repr(self._bkd.to_float(sample[row, 0]))
            for row in range(self._nvars)
        ]
        layout.params.write_text("\n".join(values) + "\n")

    def values(
        self, outcome: Outcome[ShellTask, ShellPayload]
    ) -> Decoded[Array]:
        """Read the solver's output back into an array.

        Raises :class:`MarshalError` for a missing, unparseable or
        wrong-shaped result. That is a statement about one task, so the
        evaluator records its samples as failed and the rest of the
        batch continues -- rather than the whole submission dying on one
        malformed file.
        """
        payload = outcome.payload
        if payload is None:
            raise MarshalError(
                f"no output for sample {list(outcome.indices)}: the "
                "solver did not finish"
            )
        workdir = Path(payload.workdir)
        numbers = self.read_numbers(
            workdir / self._results_filename, self._nqoi
        )
        return Decoded(
            values=self._bkd.reshape(
                self._bkd.asarray(numbers), (self._nqoi, 1)
            ),
            indices=outcome.indices,
        )

    def read_numbers(self, path: Path, expected: int) -> List[float]:
        """Read exactly ``expected`` whitespace-separated numbers.

        Each way a file can disappoint gets its own message, because
        "could not read the output" leaves a user with a directory of
        files and no idea which is wrong. The first case is the nastiest:
        a solver that exits cleanly without writing anything reports
        success by its exit code alone.

        Takes a path and a count rather than assuming the results file
        and ``nqoi``, since a marshaller reading a derivative reads a
        different file and a different number of values -- a jacobian is
        ``nqoi * nvars``, a Hessian ``nvars * nvars``, a
        Hessian-vector product ``nvars``.
        """
        if not path.exists():
            raise MarshalError(
                f"{path} does not exist: the solver exited cleanly "
                "without writing its output"
            )
        try:
            numbers = [float(token) for token in path.read_text().split()]
        except ValueError as exc:
            raise MarshalError(
                f"{path} is not readable as numbers: {exc}"
            ) from exc
        if len(numbers) != expected:
            raise MarshalError(
                f"{path} holds {len(numbers)} numbers, expected {expected}"
            )
        return numbers

    def release(
        self, outcome: Outcome[ShellTask, ShellPayload]
    ) -> None:
        """Apply the retention policy to one task's directory.

        ``shutil.rmtree`` rather than removing matched files: a solver
        may write subdirectories and dotfiles, and a glob-and-unlink
        misses the second and raises on the first.
        """
        key = outcome.task.workdir
        remaining = self._outstanding.get(key, 1) - 1
        if remaining > 0:
            # Another task still needs this directory. Any failure it
            # reported is remembered, so a sample that failed in one
            # invocation keeps its evidence even if a sibling succeeded.
            self._outstanding[key] = remaining
            if outcome.status is not JobStatus.SUCCEEDED:
                self._failed_dirs.add(key)
            return
        self._outstanding.pop(key, None)

        failed = (
            key in self._failed_dirs
            or outcome.status is not JobStatus.SUCCEEDED
        )
        self._failed_dirs.discard(key)

        if self._retention is Retention.ALWAYS:
            return
        if self._retention is Retention.ON_FAILURE and failed:
            return
        workdir = Path(key)
        if workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)
