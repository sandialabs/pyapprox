"""What happened to each working directory, written as it happens.

A directory listing answers "what is here". It cannot answer "did
sample 47 exit zero", "was this directory deleted by policy or lost",
or "what command produced any of this" -- and once retention has
removed a directory, nothing answers them. That is what this records.

**JSON Lines, appended per record.** A run killed by a wall-clock limit
has already durably written every sample that finished, and a partial
file is readable up to its last complete line. One JSON document would
be readable only once complete, which is exactly when it is least
needed.

**One file per writing process**, ``manifest.<host>.<pid>.jsonl``.
``O_APPEND`` is not honored atomically by NFS clients -- the client
seeks to the size it believes and writes -- and NFS-mounted project
filesystems are where these runs live. Two ranks resuming into one run
directory would interleave and truncate each other's records. Per-writer
files remove the question on every filesystem, give each resume its own
header, and cost a glob at read time.

**A record is one ``write`` of at most 4096 bytes.** Beyond that a
record could split across syscalls and interleave with another writer's,
which is the failure the per-writer file was meant to rule out. Records
that would exceed the cap drop their optional parts rather than being
split or silently truncated mid-JSON.

Paths are stored **relative to the run directory**. Absolute paths break
the moment a run is archived, copied to a workstation, or read from a
node that mounts the filesystem elsewhere -- which is precisely when
someone reads a manifest.

This is not a results store. Values live in a ``ResultStore``, which
already handles resume; this records what happened and where, and
nothing about numbers.
"""

import json
import os
import socket
from typing import Any, Dict, List, Optional, Sequence

#: The most one record may occupy, in bytes.
#:
#: A single ``write`` of no more than this cannot be split by the
#: kernel, which is what keeps two writers from interleaving halves of a
#: line. Large enough for a command, a link list and a sample.
MAX_RECORD_BYTES = 4096

#: Record kinds. A closed set, written into every line so a reader can
#: dispatch without positional assumptions.
KIND_RUN = "run"
KIND_PREPARED = "prepared"
KIND_RELEASED = "released"

#: Written into the run directory when a run is known to be finished.
#:
#: Its absence is the interesting case: a run killed by a wall-clock
#: limit never writes it, so a reader gathering from a run without one
#: may be looking at partial writes.
RUN_DONE_FILENAME = "run.done"


def manifest_filename(host: Optional[str] = None,
                      pid: Optional[int] = None) -> str:
    """Name this process's manifest file.

    Host and pid rather than a counter, because the writers that must
    not collide are in different processes and often on different
    machines, and neither can see the other's counter.
    """
    resolved_host = socket.gethostname() if host is None else host
    resolved_pid = os.getpid() if pid is None else pid
    # A hostname may be a dotted FQDN; the leading label is enough to
    # tell ranks apart and keeps the filename short.
    label = resolved_host.split(".")[0]
    return f"manifest.{label}.{resolved_pid}.jsonl"


class ManifestWriter:
    """Appends records to one run's manifest.

    Parameters
    ----------
    path : str
        The manifest file. Created on first write.

    Notes
    -----
    The handle is opened per append rather than held. Holding one would
    make the owning marshaller unpicklable and demand a ``close`` that
    ``MarshallerProtocol`` does not have; at a few thousand samples the
    open costs nothing next to running a solver.

    Writing never raises. A manifest is a record *about* a run, so
    failing to write one must not end the run it describes -- the run
    is the thing with value. Failures are counted and reported through
    :meth:`nfailed` rather than propagating.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._nfailed = 0

    def path(self) -> str:
        """Where records are appended."""
        return self._path

    def nfailed(self) -> int:
        """How many records could not be written."""
        return self._nfailed

    def append(self, record: Dict[str, Any]) -> None:
        """Write one record, dropping optional parts if it is too long.

        Order matters: the cap is checked against the encoded bytes,
        because a multi-byte character costs more than one column.
        """
        line = _encode(record)
        if line is None:
            self._nfailed += 1
            return
        try:
            handle = os.open(
                self._path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o644,
            )
        except OSError:
            self._nfailed += 1
            return
        try:
            os.write(handle, line)
        except OSError:
            self._nfailed += 1
        finally:
            os.close(handle)


def _encode(record: Dict[str, Any]) -> Optional[bytes]:
    """Encode one record, shedding optional fields to fit the cap.

    Returns ``None`` when even the required fields do not fit, which
    means something is wrong with the record rather than with its
    extras -- a 4 KB command, say -- and a truncated line would be worse
    than no line.
    """
    line = _line(record)
    if len(line) <= MAX_RECORD_BYTES:
        return line
    # Shed in order of what a reader can most afford to lose. ``sample``
    # is a convenience for self-describing runs; ``detail`` and
    # ``link_files`` are diagnostics; the identity fields never go.
    trimmed = dict(record)
    for field in ("sample", "link_files", "tasks", "command"):
        if field not in trimmed:
            continue
        trimmed.pop(field)
        trimmed["truncated"] = True
        line = _line(trimmed)
        if len(line) <= MAX_RECORD_BYTES:
            return line
    return None


def _line(record: Dict[str, Any]) -> bytes:
    """One record as the bytes that will be written, newline included."""
    return (
        json.dumps(record, separators=(",", ":"), default=str) + "\n"
    ).encode("utf-8")


def run_record(
    run_id: str,
    retention: str,
    command: Sequence[str],
    link_files: Sequence[str],
    created: str,
    host: Optional[str] = None,
    pid: Optional[int] = None,
) -> Dict[str, Any]:
    """The header: what this process was configured to do.

    Written once per manifest file rather than once per run, so a
    resumed run's second file says which settings applied to *its*
    records. Retention is here because it is what tells "absent by
    configuration" from "absent by failure" when the directories are
    gone.

    ``nsubmitted`` is deliberately absent: a marshaller never learns it.
    It belongs to the batch, and ``tasks`` receives only its own chunk's
    indices, so a count here would be a guess.
    """
    return {
        "kind": KIND_RUN,
        "run_id": run_id,
        "host": socket.gethostname() if host is None else host,
        "pid": os.getpid() if pid is None else pid,
        "created": created,
        "retention": retention,
        "command": list(command),
        "link_files": list(link_files),
    }


def prepared_record(
    submission: int,
    index: int,
    workdir: str,
    key: Optional[str] = None,
    sample: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """A directory exists and inputs were written into it.

    Separate from the release record because the two failures they
    distinguish are different. ``tasks`` prepares directory k, registers
    it, then prepares k+1; an error at k+1 leaves 0..k created,
    registered and never released. With only a release record those
    samples get no line at all, and a reader cannot tell them from
    samples that were never created -- one verdict for two situations,
    useful for neither.

    ``key`` is the caller's own name for this sample, carried so that a
    resubmission can be joined to what it retried. Batch-local indices
    cannot do that: a sample resubmitted after a timeout becomes index
    zero of a new submission, and nothing else links the two.
    """
    record: Dict[str, Any] = {
        "kind": KIND_PREPARED,
        "submission": submission,
        "index": index,
        "workdir": workdir,
    }
    if key is not None:
        record["key"] = key
    if sample is not None:
        record["sample"] = [float(value) for value in sample]
    return record


def released_record(
    submission: int,
    index: int,
    workdir: str,
    status: str,
    any_failed: bool,
    tasks: Sequence[Dict[str, Any]],
    retained: bool,
) -> Dict[str, Any]:
    """A directory has been finished with, and what became of it.

    Written **once per directory**, when the last task sharing it is
    released -- not once per ``release`` call. A sample covered by a
    values invocation and a jacobian invocation would otherwise get two
    lines, which a reader can only see as a duplicate.

    ``status`` is a derivation, not whichever task happened to finish
    last. Because ``release`` fires per task, the final call sees only
    one outcome, so an identical run could record ``SUCCEEDED`` or
    ``FAILED`` for the same sample depending on completion order.
    ``any_failed`` carries the aggregate and ``tasks`` carries the
    detail, so a partially-failed sample is legible as exactly that
    rather than collapsing to one word.

    ``retained`` says whether the directory survived, so a reader who
    finds nothing on disk can tell policy from loss.
    """
    return {
        "kind": KIND_RELEASED,
        "submission": submission,
        "index": index,
        "workdir": workdir,
        "status": status,
        "any_failed": any_failed,
        "tasks": list(tasks),
        "retained": retained,
    }


def task_record(
    status: str, detail: Optional[str], wall_time: float
) -> Dict[str, Any]:
    """One invocation's own outcome, for the ``tasks`` list."""
    return {
        "status": status,
        "detail": detail,
        "wall_time": wall_time,
    }


def read_records(path: str) -> List[Dict[str, Any]]:
    """Every complete record in one manifest file.

    A trailing partial line -- a run killed mid-write -- is skipped
    rather than raising: reading up to the last complete record is the
    property the line-delimited format exists to provide.
    """
    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.endswith("\n"):
                    break
                try:
                    parsed = json.loads(line)
                except ValueError:
                    continue
                if isinstance(parsed, dict):
                    records.append(parsed)
    except OSError:
        return records
    return records
