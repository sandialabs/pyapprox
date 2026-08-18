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

**Directories are named with a random component**, not a counter. A
counter is shared mutable state: two marshallers pointed at one parent
directory, or a run resumed after a crash, would reuse names and one
solver would overwrite another's inputs.

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

import shutil
import uuid
from dataclasses import dataclass
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
            commands = list(self.commands_for(layout, index, request))
            self._outstanding[str(layout.workdir)] = len(commands)
            built.extend(commands)
        return built

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

        The name carries a random component rather than a counter: a
        counter is shared mutable state, so two marshallers writing
        under one root, or a run resumed after a crash, would collide
        and one solve would overwrite another's inputs.
        """
        self._scratch_root.mkdir(parents=True, exist_ok=True)
        workdir = self._scratch_root / f"sample-{index}-{uuid.uuid4().hex[:8]}"
        workdir.mkdir()
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
