"""Getting a solver's output files out of a working directory.

The scalar a marshaller parses out of ``results.out`` is a summary; the
mesh, field or restart file beside it is the simulation. Nothing else in
this package gathers those, so a caller who wants to re-plot a field,
run a different functional over the same solutions, or find out why
sample 47 diverged, has to locate the directories by hand -- and by then
retention may have removed them.

Three layers, each built on the one before. :func:`gather_into` takes a
single live directory; :func:`gather_run` takes a whole run from its
directory alone, days later; :func:`reconcile` copies nothing and asks
whether every sample either produced what it should have or was excused.

Nothing here is generic in ``Array`` -- these are files, and a report
about files. The one thing that would drag genericity in is taking an
``EvalResult``; reconciliation takes a plain mapping of statuses
instead.

**No format is privileged.** Selection is by glob and by an injected
collector; nothing here parses, validates or even opens a matched file,
and the only names it knows are the ones this framework itself writes
into a working directory. A solver emitting HDF5, VTK, CGNS, plain CSV
or a directory per timestep is served by the same code, because the
caller supplies the patterns.

**Never copy a symlink.** Shared input is linked into every working
directory, so a walk that follows links copies the mesh once per sample
and turns a deliberately shared 2 GB file into 2 GB times nsamples. Two
separate checks are needed, and the order of one of them matters:
``is_file`` follows a link and answers ``True`` for the linked mesh, so
``is_symlink`` has to be asked first.

**Write atomically.** A copy interrupted by a wall-clock kill or a full
disk leaves a truncated file at the destination path, which a later
reader cannot tell from a complete one -- it has a plausible size and an
mtime. Every transfer lands on a temporary name and is renamed into
place, the same rule the result stores follow.
"""

import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    runtime_checkable,
)

from pyapprox.interface.evaluation.manifest import (
    KIND_PREPARED,
    KIND_RELEASED,
    KIND_RUN,
    is_run_complete,
    stream_records,
)
from pyapprox.interface.evaluation.records import JobStatus

#: Files the framework itself puts in a working directory.
#:
#: Skipped by default so a pattern as ordinary as ``"*.out"`` does not
#: sweep up the inputs beside the outputs, and so a framework-written
#: log is never collected as though the solver produced it. Each stays
#: collectible by naming it in a spec.
DEFAULT_SKIP = (
    "params.in",
    "results.out",
    "direction.in",
    "solver.stdout",
    "solver.stderr",
)

#: Suffix for a transfer in progress.
_PARTIAL_SUFFIX = ".part"


class TransferMode(Enum):
    """How a matched file reaches its destination."""

    #: Copy the bytes. The default, because the usual reason to gather
    #: is to get data off a scratch filesystem before it is purged.
    COPY = "copy"
    #: Link it. Moves no bytes, so the file still lives on the
    #: filesystem that is about to be purged -- right for a stable
    #: named view on the same filesystem, wrong for an archive.
    HARDLINK = "hardlink"
    #: Move it, draining scratch as it goes. For a run that will not be
    #: inspected where it ran.
    MOVE = "move"


@dataclass(frozen=True)
class OutputSpec:
    """Which files to take out of a working directory.

    Attributes
    ----------
    pattern : str
        A glob relative to the working directory, in whatever the
        solver happens to write: ``"*.h5"``, ``"logs/*.log"``,
        ``"**/*.vtu"``, ``"restart/*"``.
    required : bool
        Whether a sample that succeeded must match at least one file.
        Per-spec rather than global because both extremes are wrong:
        everything mandatory flags every sample over a missing optional
        log, and nothing mandatory never notices a solver that exited
        zero and wrote no field data at all.
    rename : Callable[[Path], str], optional
        Renames the *basename* at the destination. The directory part
        of a match is kept regardless, since flattening would collide
        the moment two samples both write a file of the same name --
        which they usually do, being runs of one solver.
    """

    pattern: str
    required: bool = False
    rename: Optional[Callable[[Path], str]] = None


@runtime_checkable
class OutputCollectorProtocol(Protocol):
    """Decides which files in a working directory are worth keeping.

    A protocol rather than another argument on :class:`OutputSpec`,
    because a solver whose output names cannot be written as a glob --
    a file whose name encodes the final timestep it reached -- needs to
    supply the rule itself rather than a richer pattern language.
    """

    def matches(self, workdir: Path) -> Iterable[Path]:
        """Every file to collect, as absolute paths."""
        ...


class SpecCollector:
    """Selects by glob, skipping symlinks and the framework's own files.

    Parameters
    ----------
    specs : Sequence[OutputSpec]
        Patterns to match. A file matched by several specs is collected
        once, under the first spec's rename.
    skip : Sequence[str], optional
        Names never collected regardless of pattern. Defaults to
        :data:`DEFAULT_SKIP`; pass an empty sequence to collect them.
    """

    def __init__(
        self,
        specs: Sequence[OutputSpec],
        skip: Optional[Sequence[str]] = None,
    ) -> None:
        self._specs = list(specs)
        self._skip = set(DEFAULT_SKIP if skip is None else skip)

    def specs(self) -> Sequence[OutputSpec]:
        """The patterns this collector selects by."""
        return tuple(self._specs)

    def matches(self, workdir: Path) -> Iterable[Path]:
        """Files to collect, in spec order, without duplicates."""
        return [path for path, _ in self.matches_with_specs(workdir)]

    def matches_with_specs(
        self, workdir: Path
    ) -> List[Tuple[Path, OutputSpec]]:
        """Each match paired with the spec that first claimed it.

        The pairing is what lets a caller apply the right ``rename``
        and tell which ``required`` spec went unmatched.
        """
        seen: Set[Path] = set()
        found: List[Tuple[Path, OutputSpec]] = []
        for spec in self._specs:
            for path in sorted(workdir.glob(spec.pattern)):
                if path in seen or not _is_collectable(path, self._skip):
                    continue
                seen.add(path)
                found.append((path, spec))
        return found

    def unmatched_required(self, workdir: Path) -> List[OutputSpec]:
        """Required specs that matched nothing.

        The case this whole module exists for: a solver that exits zero
        having written no field file raises nothing anywhere else,
        because the values decoder only ever reads ``results.out``.
        """
        matched = {spec for _, spec in self.matches_with_specs(workdir)}
        return [
            spec
            for spec in self._specs
            if spec.required and spec not in matched
        ]


def _is_collectable(path: Path, skip: Set[str]) -> bool:
    """Whether one matched path is a file worth copying.

    ``is_symlink`` before ``is_file``, deliberately. ``is_file``
    resolves the link and answers ``True`` for the mesh symlinked into
    every working directory, so asking it first would admit precisely
    what has to be excluded -- and would do so silently, as a copy that
    works and costs one mesh per sample.
    """
    if path.is_symlink():
        return False
    if not path.is_file():
        return False
    return path.name not in skip


@dataclass(frozen=True)
class TransferResult:
    """What became of one directory's files."""

    #: Destination paths, in the order they were written.
    paths: Sequence[str] = field(default_factory=tuple)
    #: Total bytes transferred.
    nbytes: int = 0
    #: Per-file failures, as ``(source, reason)``.
    failures: Sequence[Tuple[str, str]] = field(default_factory=tuple)
    #: Sources that fell back to a copy because a hardlink could not
    #: cross a filesystem boundary. Recorded rather than silent: a
    #: caller who asked for links and got copies has different disk
    #: usage than they planned for.
    fellback: Sequence[str] = field(default_factory=tuple)

    def ok(self) -> bool:
        """Whether every file transferred."""
        return not self.failures


def gather_into(
    workdir: str,
    collector: OutputCollectorProtocol,
    dest: str,
    mode: TransferMode = TransferMode.COPY,
    overwrite: bool = False,
) -> TransferResult:
    """Copy one directory's matched files into ``dest``.

    The primitive the rest is built on: one live directory, before
    anything deletes it. Deferred gathering over a whole run is a loop
    over this.

    Relative paths under the working directory are preserved, so a
    solver writing ``fields/step_0400.dat`` keeps that shape. Failures
    are collected rather than raised -- a caller gathering hundreds of
    directories wants the ones that worked and a list of the ones that
    did not, and this runs where an exception would strand the rest.

    Parameters
    ----------
    workdir : str
        The directory to gather from.
    collector : OutputCollectorProtocol
        Decides what to take.
    dest : str
        Where files land. Created if absent.
    mode : TransferMode
        How each file gets there.
    overwrite : bool
        Whether an existing destination file may be replaced. Off by
        default, so a second gather with different specs, or a repeated
        move, cannot quietly destroy what the first one wrote.
    """
    if not isinstance(collector, OutputCollectorProtocol):
        raise TypeError(
            "collector must satisfy OutputCollectorProtocol, got "
            f"{type(collector).__name__}"
        )
    source_dir = Path(workdir)
    dest_dir = Path(dest)
    paths: List[str] = []
    failures: List[Tuple[str, str]] = []
    fellback: List[str] = []
    nbytes = 0

    for path, target in _planned(source_dir, dest_dir, collector, failures):
        try:
            transferred, did_fallback = _transfer(
                path, target, mode, overwrite
            )
        except OSError as exc:
            failures.append((str(path), str(exc)))
            continue
        paths.append(str(target))
        nbytes += transferred
        if did_fallback:
            fellback.append(str(path))

    return TransferResult(
        paths=tuple(paths),
        nbytes=nbytes,
        failures=tuple(failures),
        fellback=tuple(fellback),
    )


@dataclass(frozen=True)
class GatherReport:
    """What a whole run yielded.

    Attributes
    ----------
    run_id : str
        The run gathered from.
    gathered : Mapping[int, Sequence[str]]
        Destination paths per sample index. Destinations rather than
        sources, which is the only choice that still means anything
        under :attr:`TransferMode.MOVE`.
    missing : Sequence[int]
        Indices whose directory the manifest recorded but which is no
        longer on disk. Expected under a retention policy that removed
        them; :func:`reconcile` is what decides whether that is alarming.
    failures : Sequence[Tuple[str, str]]
        Per-file failures, as ``(source, reason)``.
    nbytes : int
        Total bytes transferred.
    complete : bool
        Whether the run had marked itself finished. ``False`` means
        gathering may have raced writes still in progress.
    """

    run_id: str
    gathered: Mapping[int, Sequence[str]] = field(default_factory=dict)
    missing: Sequence[int] = field(default_factory=tuple)
    failures: Sequence[Tuple[str, str]] = field(default_factory=tuple)
    nbytes: int = 0
    complete: bool = True

    def nfiles(self) -> int:
        """How many files were transferred."""
        return sum(len(paths) for paths in self.gathered.values())

    def ok(self) -> bool:
        """Whether every file transferred."""
        return not self.failures


def gather_run(
    run_dir: str,
    collector: OutputCollectorProtocol,
    dest: str,
    mode: TransferMode = TransferMode.COPY,
    indices: Optional[Sequence[int]] = None,
    overwrite: bool = False,
) -> GatherReport:
    """Gather every directory a run's manifests recorded.

    The deferred case, and the one that motivated all of this: it needs
    no evaluator, no marshaller and no live process, so a different
    session or a shell script a week later can run it. What it needs is
    the run directory, which is why directories are named rather than
    randomized.

    The manifests are **streamed**. A large sweep records one line per
    sample, and building an index-to-record map before doing any work
    would hold the whole sweep in memory to copy one file at a time.

    Sample directories are recreated under ``dest/<run_id>/`` with their
    submission and sample levels intact, so two submissions that both
    number their samples from zero stay apart at the destination exactly
    as they do at the source.

    Parameters
    ----------
    run_dir : str
        A run directory, as named by the marshaller that wrote it.
    collector : OutputCollectorProtocol
        Decides what to take from each directory.
    dest : str
        Where the gathered tree is written.
    mode : TransferMode
        How each file gets there.
    indices : Sequence[int], optional
        Restrict to these sample indices. ``None`` gathers everything.
    overwrite : bool
        Whether existing destination files may be replaced.
    """
    if not isinstance(collector, OutputCollectorProtocol):
        raise TypeError(
            "collector must satisfy OutputCollectorProtocol, got "
            f"{type(collector).__name__}"
        )
    run = Path(run_dir)
    wanted = None if indices is None else set(indices)
    gathered: Dict[int, List[str]] = {}
    missing: List[int] = []
    failures: List[Tuple[str, str]] = []
    nbytes = 0

    for record in stream_records(run_dir):
        if record.get("kind") != KIND_PREPARED:
            continue
        index = record.get("index")
        relative = record.get("workdir")
        if not isinstance(index, int) or not isinstance(relative, str):
            continue
        if wanted is not None and index not in wanted:
            continue
        workdir = run / relative
        if not workdir.is_dir():
            # Recorded but gone. Retention may explain it, and only a
            # reconciliation against recorded status can say.
            missing.append(index)
            continue
        result = gather_into(
            str(workdir),
            collector,
            str(Path(dest) / run.name / relative),
            mode=mode,
            overwrite=overwrite,
        )
        gathered.setdefault(index, []).extend(result.paths)
        failures.extend(result.failures)
        nbytes += result.nbytes

    return GatherReport(
        run_id=run.name,
        gathered={index: tuple(paths) for index, paths in gathered.items()},
        missing=tuple(missing),
        failures=tuple(failures),
        nbytes=nbytes,
        complete=is_run_complete(run_dir),
    )


class AnomalyKind(Enum):
    """Why one sample did not satisfy the invariant."""

    #: A required spec matched nothing though the sample succeeded. The
    #: case this whole feature exists for.
    NO_OUTPUT = "no_output"
    #: The manifest recorded the directory and it is gone, with no
    #: retention setting that explains it.
    VANISHED = "vanished"
    #: Prepared and never released: the run stopped between building
    #: the directory and finishing with it, so it is still there.
    NEVER_RELEASED = "never_released"
    #: Two release records for one directory.
    DUPLICATE = "duplicate"
    #: A status map says the index was submitted and no record mentions
    #: it, so nothing ever built it a directory.
    NEVER_PREPARED = "never_prepared"


@dataclass(frozen=True)
class Anomaly:
    """One sample that did not satisfy the invariant."""

    index: int
    kind: AnomalyKind
    detail: str

    def __str__(self) -> str:
        return f"sample {self.index}: {self.kind.value} ({self.detail})"


class CollectionError(RuntimeError):
    """A reconciliation found anomalies and the caller wanted a raise."""


@dataclass(frozen=True)
class CollectionReport:
    """What a run's directories say about themselves.

    Attributes
    ----------
    run_id : str
        The run reconciled.
    ok_indices : Sequence[int]
        Samples that satisfied the invariant.
    explained : Mapping[int, str]
        Samples that produced no output for a reason the record gives:
        the job failed, was cancelled, timed out, or its directory was
        removed by the retention policy in force.
    retryable : Sequence[int]
        Explained samples whose failure says nothing about the parameter
        point -- a timeout or a cancellation. Separate because that is
        the difference between "resubmit this" and "this point is bad".
    partial : Mapping[int, str]
        Samples reported as succeeded whose own record shows some
        invocation failed. Their outputs may legitimately be incomplete.
    anomalies : Sequence[Anomaly]
        Everything the record does not explain.
    """

    run_id: str
    ok_indices: Sequence[int] = field(default_factory=tuple)
    explained: Mapping[int, str] = field(default_factory=dict)
    retryable: Sequence[int] = field(default_factory=tuple)
    partial: Mapping[int, str] = field(default_factory=dict)
    anomalies: Sequence[Anomaly] = field(default_factory=tuple)

    def ok(self) -> bool:
        """Whether every sample was accounted for."""
        return not self.anomalies

    def raise_if_anomalous(self) -> None:
        """Raise :class:`CollectionError` if anything was unexplained.

        Available rather than automatic. Someone reconciling two
        thousand samples with six bad ones wants the list and the other
        1994, not a traceback; a script that must not proceed on partial
        data asks for this explicitly.
        """
        if not self.anomalies:
            return
        listed = "\n".join(f"  {anomaly}" for anomaly in self.anomalies)
        raise CollectionError(
            f"run {self.run_id} has {len(self.anomalies)} "
            f"unexplained sample(s):\n{listed}"
        )


def reconcile(
    run_dir: str,
    statuses: Optional[Mapping[int, JobStatus]] = None,
    collector: Optional[OutputCollectorProtocol] = None,
) -> CollectionReport:
    """Check that every sample either produced its output or was excused.

    The invariant, stated for the multi-quantity case:

        Either index ``i`` produced every required output, or ``i`` was
        reported as not having succeeded for the quantity that would
        have written it.

    Checked against what the run recorded, **never against a directory
    listing**. A listing cannot tell "the solver wrote nothing" from
    "the pattern did not match", and it cannot see a sample whose
    directory retention has already removed.

    Copies nothing.

    Parameters
    ----------
    run_dir : str
        The run to check.
    statuses : Mapping[int, JobStatus], optional
        What the caller knows about each index, from
        ``Batch.statuses()``. Supplying it adds one check the manifest
        alone cannot make -- that an index submitted was ever prepared
        -- because a marshaller never learns how many samples a batch
        held.
    collector : OutputCollectorProtocol, optional
        Supplies the required specs. Without one, nothing is required
        and the output check cannot fire.
    """
    run = Path(run_dir)
    # Keyed by (submission, index), never by index alone. Indices are
    # batch-local and restart at zero, so one marshaller submitted twice
    # produces two samples numbered 0 -- distinct work in distinct
    # directories, which keying by index would report as a duplicate.
    prepared: Dict[Tuple[int, int], Dict[str, object]] = {}
    released: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
    retention = ""

    for record in stream_records(run_dir):
        kind = record.get("kind")
        if kind == KIND_RUN:
            retention = str(record.get("retention", ""))
            continue
        if kind not in (KIND_PREPARED, KIND_RELEASED):
            continue
        key = _sample_key(record)
        if key is None:
            continue
        if kind == KIND_PREPARED:
            prepared[key] = record
        else:
            released.setdefault(key, []).append(record)

    report = _Reconciler(run, prepared, released, retention, collector)
    return report.run(statuses)


def _sample_key(record: Mapping[str, object]) -> Optional[Tuple[int, int]]:
    """A record's ``(submission, index)``, or ``None`` if malformed.

    Records written before submissions were recorded, or edited by
    hand, fall back to submission zero rather than being dropped.
    """
    index = record.get("index")
    if not isinstance(index, int):
        return None
    submission = record.get("submission")
    return (submission if isinstance(submission, int) else 0, index)


class _Reconciler:
    """Applies the decision procedure to one run's records.

    The rows are **ordered and first-match-wins**, which a flat table
    cannot express: they overlap. A succeeded sample under a retention
    policy of ``never`` whose required output was never collected
    matches both "missing required output" and "directory absent", and
    only one of those is the useful thing to say.
    """

    def __init__(
        self,
        run: Path,
        prepared: Dict[Tuple[int, int], Dict[str, object]],
        released: Dict[Tuple[int, int], List[Dict[str, object]]],
        retention: str,
        collector: Optional[OutputCollectorProtocol],
    ) -> None:
        self._run = run
        self._prepared = prepared
        self._released = released
        self._retention = retention
        self._collector = collector

    def run(
        self, statuses: Optional[Mapping[int, JobStatus]]
    ) -> CollectionReport:
        ok: List[int] = []
        explained: Dict[int, str] = {}
        retryable: List[int] = []
        partial: Dict[int, str] = {}
        anomalies: List[Anomaly] = []

        # Judged per (submission, index) but reported per index, which
        # is what a caller holds: they submitted columns, not
        # submissions. A sample asked about twice therefore contributes
        # two verdicts under one index, and the worse one has to win --
        # otherwise a later clean resubmission would paper over an
        # earlier anomaly that is still true of the run.
        for key in self._keys(statuses):
            index = key[1]
            verdict, detail = self._verdict(key, statuses)
            if isinstance(verdict, AnomalyKind):
                anomalies.append(Anomaly(index, verdict, detail))
            elif verdict == "partial":
                partial[index] = detail
            elif verdict == "retryable":
                explained[index] = detail
                retryable.append(index)
            elif verdict == "explained":
                explained[index] = detail
            elif index not in partial and index not in explained:
                ok.append(index)

        flagged = (
            {anomaly.index for anomaly in anomalies}
            | set(partial)
            | set(explained)
        )
        return CollectionReport(
            run_id=self._run.name,
            ok_indices=tuple(
                sorted({index for index in ok if index not in flagged})
            ),
            explained=dict(explained),
            retryable=tuple(sorted(set(retryable))),
            partial=dict(partial),
            anomalies=tuple(anomalies),
        )

    def _keys(
        self, statuses: Optional[Mapping[int, JobStatus]]
    ) -> List[Tuple[int, int]]:
        """Every sample worth judging, from records and caller alike.

        A status map is keyed by index alone -- it comes from one batch,
        which knows nothing of submissions -- so an index it names that
        no record mentions is attributed to submission zero purely to
        give it a key.
        """
        known = set(self._prepared) | set(self._released)
        if statuses is not None:
            seen = {index for _, index in known}
            known |= {
                (0, index) for index in statuses if index not in seen
            }
        return sorted(known)

    def _verdict(
        self,
        key: Tuple[int, int],
        statuses: Optional[Mapping[int, JobStatus]],
    ) -> Tuple[object, str]:
        """One sample's verdict, first matching rule winning."""
        records = self._released.get(key, [])

        # Two release records for one sample. Only reachable through a
        # bug or a hand-edited manifest, and worth saying plainly
        # rather than letting one silently win.
        if len(records) > 1:
            return AnomalyKind.DUPLICATE, f"{len(records)} release records"

        # Submitted, and no record mentions it: nothing ever built it a
        # directory. Needs the caller's status map, since a marshaller
        # never learns how many samples a batch held.
        if key not in self._prepared and not records:
            return (
                AnomalyKind.NEVER_PREPARED,
                "submitted, but no record mentions it",
            )

        # Prepared and never released. The run stopped in between, so
        # the directory is still there and worth looking inside.
        if not records:
            return (
                AnomalyKind.NEVER_RELEASED,
                "prepared, never released; the directory should remain",
            )

        record = records[0]
        status = str(record.get("status", ""))
        any_failed = bool(record.get("any_failed"))

        # A failure is the evidence case, not an anomaly: the run said
        # this sample did not succeed, so no output is expected.
        if status in _RETRYABLE_STATUSES:
            return "retryable", f"job {status.lower()}"
        if status not in ("", JobStatus.SUCCEEDED.name):
            return "explained", f"job {status.lower()}"

        # Succeeded overall, but some invocation did not. Checked
        # BEFORE the output rule: success is per quantity, so a sample
        # whose jacobian failed is legitimately missing that quantity's
        # file, and calling that "the solver wrote nothing" would flag
        # every partially-failed derivative sample in the run.
        if any_failed:
            return "partial", "succeeded, but an invocation failed"

        workdir = self._run / str(record.get("workdir", ""))
        if not workdir.is_dir():
            if bool(record.get("retained")) is False:
                return (
                    "explained",
                    f"removed by retention policy {self._retention!r}",
                )
            return (
                AnomalyKind.VANISHED,
                "recorded as retained, but the directory is gone",
            )

        unmatched = self._unmatched_required(workdir)
        if unmatched:
            patterns = ", ".join(spec.pattern for spec in unmatched)
            return (
                AnomalyKind.NO_OUTPUT,
                f"succeeded but matched no {patterns}",
            )
        return "ok", ""

    def _unmatched_required(self, workdir: Path) -> List[OutputSpec]:
        """Required specs this directory did not satisfy."""
        if not isinstance(self._collector, SpecCollector):
            return []
        return self._collector.unmatched_required(workdir)


#: Statuses whose failure says nothing about the parameter point.
_RETRYABLE_STATUSES = (
    JobStatus.TIMED_OUT.name,
    JobStatus.CANCELLED.name,
)


def _planned(
    source_dir: Path,
    dest_dir: Path,
    collector: OutputCollectorProtocol,
    failures: List[Tuple[str, str]],
) -> List[Tuple[Path, Path]]:
    """Pair each match with where it will land, refusing collisions.

    Two matches renaming onto one destination is data loss the caller
    cannot see -- the second silently replaces the first -- so it is
    reported as a failure of the second rather than performed.
    """
    planned: List[Tuple[Path, Path]] = []
    claimed: Set[Path] = set()
    for path, spec in _pairs(collector, source_dir):
        relative = path.relative_to(source_dir)
        name = spec.rename(path) if spec.rename is not None else path.name
        target = dest_dir / relative.parent / name
        if target in claimed:
            failures.append(
                (str(path), f"renames onto {target}, already claimed")
            )
            continue
        claimed.add(target)
        planned.append((path, target))
    return planned


def _pairs(
    collector: OutputCollectorProtocol, source_dir: Path
) -> List[Tuple[Path, OutputSpec]]:
    """Matches with their specs, for any collector.

    A collector supplying only ``matches`` gets a spec with no rename,
    which is the whole of what the pairing is used for.
    """
    if isinstance(collector, SpecCollector):
        return collector.matches_with_specs(source_dir)
    plain = OutputSpec(pattern="")
    return [(path, plain) for path in collector.matches(source_dir)]


def _transfer(
    source: Path, target: Path, mode: TransferMode, overwrite: bool
) -> Tuple[int, bool]:
    """Move one file into place, atomically. Returns bytes and fallback.

    Landing on a temporary name and renaming is what keeps a partial
    transfer from looking like a complete one: ``os.replace`` either
    happens or does not, so a reader never sees a half-written file at
    the real path.
    """
    if target.exists() and not overwrite:
        raise OSError(f"{target} exists; pass overwrite=True to replace it")
    target.parent.mkdir(parents=True, exist_ok=True)
    nbytes = source.stat().st_size
    partial = target.with_name(target.name + _PARTIAL_SUFFIX)
    fellback = False
    try:
        if mode is TransferMode.HARDLINK:
            try:
                os.link(source, partial)
            except OSError:
                # A hardlink cannot cross a filesystem boundary, and
                # copying is what the caller wanted anyway. Reported,
                # because links and copies differ in what a purge takes.
                shutil.copy2(source, partial)
                fellback = True
        else:
            shutil.copy2(source, partial)
        os.replace(partial, target)
    except OSError:
        # Never leave the scratch name behind: a later gather would see
        # it as a stray file, and under MOVE the source is still there.
        if partial.exists():
            partial.unlink()
        raise
    if mode is TransferMode.MOVE:
        source.unlink()
    return nbytes, fellback
