"""Getting a solver's output files out of a working directory.

The scalar a marshaller parses out of ``results.out`` is a summary; the
mesh, field or restart file beside it is the simulation. Nothing else in
this package gathers those, so a caller who wants to re-plot a field,
run a different functional over the same solutions, or find out why
sample 47 diverged, has to locate the directories by hand -- and by then
retention may have removed them.

This module is the primitive: one live directory, some patterns, a
destination. Nothing here knows about runs, manifests or batches, and
nothing here is generic in ``Array`` -- these are files.

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
    is_run_complete,
    stream_records,
)

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
