"""Where finished results outlive the process that produced them.

A handle is an in-memory object, so a crashed workflow loses every
result still in flight. For a model taking hours per sample that is the
sharpest version of the failure this subpackage exists to prevent, and
it is why :class:`~pyapprox.interface.evaluation.protocols.ResultStore`
records results as they are collected rather than at the end -- a store
written only on completion is empty in exactly the case it exists for.

Two implementations ship here because one implementation never proves a
protocol is implementable. :class:`NpzResultStore` survives the machine;
:class:`InMemoryResultStore` does not, and exists so that code consuming
a store is testable without a filesystem.

**None of them decides anything.** A store answers what it holds and
records what it is given. It never tells an evaluator to skip work,
because sample identity belongs to the caller that built the samples --
see the protocol for why that trade is deliberate.

Relationship to a marshaller's scratch directories
--------------------------------------------------
:class:`~pyapprox.interface.evaluation.textfile_marshaller.Retention`
also keeps data past the end of a task, so the two are easy to confuse.
They answer different questions and are not substitutes:

- A **retained working directory** holds what the solver wrote: input
  decks, logs, native output, whatever else it produced. It is keyed by
  task, in the solver's own formats, and exists mainly so a failure can
  be diagnosed -- which is why ``ON_FAILURE`` is the default.
- A **store** holds decoded arrays keyed by the caller's key. It exists
  so a restarted workflow can skip completed work, which needs a
  uniform representation rather than whatever the solver emitted.

They compose, and the combination worth knowing about is
``Retention.NEVER`` with a store: the scratch directories go away while
the numbers survive, which is usually what a long sweep wants. The
opposite, ``Retention.ALWAYS`` with a store, keeps every result twice,
in two formats, and is worth choosing deliberately rather than by
accident.

A marshaller owning a durable native format may instead implement this
protocol itself, recording where the solver already wrote its output and
re-decoding on ``load``. That avoids the duplication entirely and keeps
whatever the native format carries that ``values`` does not.
"""

import os
import pickle
import tempfile
from typing import (
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

import numpy as np
from numpy.typing import ArrayLike

from pyapprox.interface.evaluation.protocols import (
    ResultStore,
    TaskProtocol,
)
from pyapprox.interface.evaluation.records import (
    ComputeProvenance,
    Cost,
    Decoded,
    Outcome,
)
from pyapprox.util.backends.protocols import Array, Backend

# Optional Decoded fields that are arrays. Named once so adding a
# capability upstream is a single edit here rather than four.
_OPTIONAL_ARRAYS = ("jacobians", "hessians", "jvps", "hvps", "hvp_weights")

# Carried through StoreWriter so it satisfies CompletionHook exactly,
# rather than erasing the pair to Any and losing the check at the point
# where a hook is paired with an evaluator.
Task = TypeVar("Task", bound=TaskProtocol)
Payload = TypeVar("Payload")


class InMemoryResultStore(Generic[Array]):
    """A store that keeps results in a dictionary.

    Loses everything when the process exits, which makes it useless for
    the resume case and ideal for testing code that consults a store.

    Records are held as given. Nothing is copied, because a ``Decoded``
    is frozen and its arrays are not mutated after construction.
    """

    def __init__(self) -> None:
        self._records: Dict[str, Tuple[Decoded[Array], Cost]] = {}

    def save(self, key: str, decoded: Decoded[Array], cost: Cost) -> None:
        """Record one task's decoded output under ``key``."""
        self._records[key] = (decoded, cost)

    def load(self, key: str) -> Optional[Tuple[Decoded[Array], Cost]]:
        """Return what was stored under ``key``, or None if absent."""
        return self._records.get(key)

    def keys(self) -> Sequence[str]:
        """Every key currently stored."""
        return list(self._records)


class NpzResultStore(Generic[Array]):
    """A store that writes one ``.npz`` file per key.

    Parameters
    ----------
    directory : str
        Where the files live. Created if absent.
    bkd : Backend[Array]
        Used to rebuild arrays on ``load``. A store written by one
        backend can be read by another, since what reaches disk is
        plain numpy.

    Notes
    -----
    **Arrays are converted to numpy to be written.** That is what
    ``.npz`` is, and serializing is not computing -- the restriction on
    ``to_numpy`` exists to keep backend arrays out of *calculations*,
    not out of files. A caller wanting a native format (``torch.save``,
    HDF5, a database) injects a different implementation of the same
    protocol; nothing here enumerates the possibilities.

    **Writes are atomic.** The file is written to a temporary name in
    the same directory and renamed into place, because ``rename`` is
    atomic within a filesystem. A save interrupted midway therefore
    leaves either the previous file or none -- never a truncated one
    that a resumed run would load and trust. This matters more than
    usual here: the whole point of the store is to be read after a
    crash, so the crash-during-write case is the expected one rather
    than the unlucky one.
    """

    def __init__(self, directory: str, bkd: Backend[Array]) -> None:
        self._directory = directory
        self._bkd = bkd
        os.makedirs(directory, exist_ok=True)

    def bkd(self) -> Backend[Array]:
        """Return the backend used to rebuild loaded arrays."""
        return self._bkd

    def directory(self) -> str:
        """Return the directory holding the stored files."""
        return self._directory

    def _path(self, key: str) -> str:
        return os.path.join(self._directory, f"{_encode_key(key)}.npz")

    def save(self, key: str, decoded: Decoded[Array], cost: Cost) -> None:
        """Record one task's decoded output under ``key``.

        Safe to call again with the same key: the rename replaces the
        previous file, so a resumed run that recomputes a sample whose
        save was interrupted simply overwrites it.
        """
        payload: Dict[str, ArrayLike] = {
            "values": self._bkd.to_numpy(decoded.values),
            "indices": np.asarray(list(decoded.indices), dtype=np.int64),
            "cost_wall_clock": np.asarray(cost.wall_clock),
            "cost_compute": np.asarray(cost.compute),
            "cost_provenance": np.asarray(cost.provenance.value),
        }
        for name in _OPTIONAL_ARRAYS:
            field = getattr(decoded, name)
            if field is not None:
                payload[name] = self._bkd.to_numpy(field)
        if decoded.wall_time is not None:
            payload["wall_time"] = np.asarray(decoded.wall_time)

        # Written beside the target so the rename stays within one
        # filesystem; across filesystems rename is not atomic.
        handle, temp_path = tempfile.mkstemp(
            dir=self._directory, suffix=".npz.tmp"
        )
        os.close(handle)
        try:
            # Written through an open handle rather than by name:
            # np.savez appends .npz to a name that lacks it, which would
            # leave the real file beside the temporary one and defeat the
            # atomic rename below.
            with open(temp_path, "wb") as stream:
                # allow_pickle is passed explicitly so that everything
                # reaching disk is plain array data: load() reads with
                # allow_pickle=False, and a file this store wrote must
                # be one it can read back.
                np.savez(stream, allow_pickle=False, **payload)
            os.replace(temp_path, self._path(key))
        except BaseException:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def load(self, key: str) -> Optional[Tuple[Decoded[Array], Cost]]:
        """Return what was stored under ``key``, or None if absent."""
        path = self._path(key)
        if not os.path.exists(path):
            return None
        with np.load(path, allow_pickle=False) as data:
            optional = {
                name: self._bkd.asarray(data[name])
                for name in _OPTIONAL_ARRAYS
                if name in data.files
            }
            wall_time = (
                float(data["wall_time"]) if "wall_time" in data.files else None
            )
            decoded: Decoded[Array] = Decoded(
                values=self._bkd.asarray(data["values"]),
                indices=[int(i) for i in data["indices"]],
                wall_time=wall_time,
                **optional,
            )
            cost = Cost(
                wall_clock=float(data["cost_wall_clock"]),
                compute=float(data["cost_compute"]),
                provenance=ComputeProvenance(str(data["cost_provenance"])),
            )
        return decoded, cost

    def keys(self) -> Sequence[str]:
        """Every key currently stored.

        Partially written temporaries are skipped by extension, so a
        crash during a save cannot make a resumed run believe it holds
        a result it does not.
        """
        found: List[str] = []
        for name in os.listdir(self._directory):
            if name.endswith(".npz"):
                found.append(_decode_key(name[: -len(".npz")]))
        return found


class PickleResultStore(Generic[Array]):
    """A store that pickles one record per key.

    Parameters
    ----------
    directory : str
        Where the files live. Created if absent.

    Notes
    -----
    Exists because ``.npz`` holds arrays and nothing else. A marshaller
    whose ``Decoded`` carries something that is not an array has nowhere
    to put it in an npz, and a torch user pays a numpy round trip on
    every save and load. Pickle stores the record whole, in whatever
    types it already has, and needs no backend to rebuild it.

    **Loading a pickle executes code from the file.** That is a property
    of the format, not of this class, and it means a store is only as
    trustworthy as whoever could write to its directory. Fine for a
    scratch directory owned by the person running the sweep; wrong for
    anything shared, downloaded, or on a filesystem others can write to.
    Prefer :class:`NpzResultStore` when the data must cross a trust
    boundary -- it reads with ``allow_pickle=False``, so a hostile file
    fails to load rather than running.

    Writes are atomic by the same temp-and-rename argument as
    :class:`NpzResultStore`.
    """

    def __init__(self, directory: str) -> None:
        self._directory = directory
        os.makedirs(directory, exist_ok=True)

    def directory(self) -> str:
        """Return the directory holding the stored files."""
        return self._directory

    def _path(self, key: str) -> str:
        return os.path.join(self._directory, f"{_encode_key(key)}.pkl")

    def save(self, key: str, decoded: Decoded[Array], cost: Cost) -> None:
        """Record one task's decoded output under ``key``."""
        handle, temp_path = tempfile.mkstemp(
            dir=self._directory, suffix=".pkl.tmp"
        )
        try:
            with os.fdopen(handle, "wb") as stream:
                pickle.dump((decoded, cost), stream)
            os.replace(temp_path, self._path(key))
        except BaseException:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def load(self, key: str) -> Optional[Tuple[Decoded[Array], Cost]]:
        """Return what was stored under ``key``, or None if absent."""
        path = self._path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as stream:
            record: Tuple[Decoded[Array], Cost] = pickle.load(stream)
        return record

    def keys(self) -> Sequence[str]:
        """Every key currently stored."""
        return [
            _decode_key(name[: -len(".pkl")])
            for name in os.listdir(self._directory)
            if name.endswith(".pkl")
        ]


class StoreWriter(Generic[Task, Payload, Array]):
    """Records each finished task in a store as it is collected.

    Built to be passed as an evaluator's ``on_complete`` hook, which
    fires once per finished task before its scratch is released and
    carries exactly what a store needs::

        writer = StoreWriter(store, prefix="sweep7")
        evaluator = Evaluator(marshaller, dispatcher, on_complete=writer)

    Saving here rather than after ``collect`` returns is the difference
    between a store that survives a crash and one that does not: a batch
    interrupted half way has already written the tasks that finished.

    Parameters
    ----------
    store : ResultStore[Array]
        Where records are written.
    prefix : str
        Namespaces this submission's keys. Keys are formed as
        ``f"{prefix}:{index}"`` from the batch-local sample indices, so
        two submissions sharing a prefix share an index space and the
        later one overwrites -- which is what a resumed run wants, and
        why the prefix must change when the samples do.

    Notes
    -----
    **Writing a record does not make an evaluator consult it.** Nothing
    here causes work to be skipped: an evaluator with this hook still
    computes everything it is given. Deciding that a stored key still
    refers to the sample a caller means is the caller's, because only
    the caller knows how its samples were built.

    A task covering several samples is stored under one key, formed from
    the first index it covers. Splitting it per sample would mean
    re-slicing every array in the record, and a resumed caller reads
    whole records anyway.
    """

    def __init__(self, store: ResultStore[Array], prefix: str) -> None:
        self._store = store
        self._prefix = prefix

    def store(self) -> ResultStore[Array]:
        """The store being written to."""
        return self._store

    def prefix(self) -> str:
        """The namespace applied to this submission's keys."""
        return self._prefix

    def key_for(self, index: int) -> str:
        """The key a sample at batch-local ``index`` is stored under."""
        return f"{self._prefix}:{index}"

    def __call__(
        self,
        outcome: Outcome[Task, Payload],
        decoded: Optional[Decoded[Array]],
        cost: Cost,
    ) -> None:
        """Record one finished task, if it produced anything."""
        if decoded is None or not decoded.indices:
            # A task that failed outright has nothing to record. Storing
            # a placeholder would make a resumed run treat the key as
            # known and never retry it.
            return
        self._store.save(self.key_for(min(decoded.indices)), decoded, cost)


def stored_indices(store: ResultStore[Array], prefix: str) -> Set[int]:
    """Which sample indices ``store`` already holds under ``prefix``.

    Parameters
    ----------
    store : ResultStore[Array]
        The store to interrogate.
    prefix : str
        The namespace a :class:`StoreWriter` was writing under. Keys
        outside it are ignored, so several sweeps may share a store.

    Returns
    -------
    Set[int]
        Every batch-local index covered by a stored record.

    Notes
    -----
    Reads ``indices`` off each record rather than parsing the keys,
    because **a key names the task that produced a record, not a
    sample**. Under ``samples_per_task=3`` a six-sample batch stores two
    records, ``prefix:0`` and ``prefix:3``, and asking whether
    ``prefix:4`` exists reports a sample missing that is sitting inside
    the second record. The failure is silent -- a resumed run simply
    recomputes work it already had, with nothing to indicate why -- and
    the shortcut is right often enough to survive casual testing, since
    it is correct whenever a task covers exactly one sample.

    Costs one ``load`` per record. A store whose loads are expensive can
    implement this more cheaply by knowing its own layout; this is the
    form that works for any store.
    """
    known: Set[int] = set()
    marker = f"{prefix}:"
    for key in store.keys():
        if not key.startswith(marker):
            continue
        record = store.load(key)
        if record is None:
            # Removed between keys() and load(): treat as absent rather
            # than failing a resume that would otherwise succeed.
            continue
        known.update(record[0].indices)
    return known


def restore_columns(
    store: ResultStore[Array],
    prefix: str,
    nsamples: int,
    fresh: Array,
    todo: Sequence[int],
    bkd: Backend[Array],
) -> Array:
    """Rebuild a full ``(nqoi, nsamples)`` array from stored and fresh parts.

    Parameters
    ----------
    store : ResultStore[Array]
        Holds the records written by an earlier attempt.
    prefix : str
        The namespace those records were written under.
    nsamples : int
        Width of the array to rebuild.
    fresh : Array
        Values just computed, shape ``(nqoi, len(todo))``. Column ``j``
        is sample ``todo[j]``.
    todo : Sequence[int]
        The sample indices ``fresh`` covers, in submission order.
    bkd : Backend[Array]
        Used to join the columns.

    Returns
    -------
    Array
        Shape ``(nqoi, nsamples)``, column ``i`` being sample ``i``.

    Raises
    ------
    ValueError
        If ``fresh`` is not as wide as ``todo``, or if the stored
        records and ``todo`` together do not cover every sample.

    Notes
    -----
    Exists because getting this wrong produces a **full-length array of
    plausible numbers with columns in the wrong places**, which no shape
    check catches and no exception announces. Two mappings have to be
    right at once: a fresh column is at its position in ``todo`` rather
    than at its sample index, and a stored column is at its position
    within its own record rather than at either.

    Deciding *which* samples are stale is still the caller's: this takes
    the ``todo`` list rather than deriving one, so nothing here judges
    whether a stored key still refers to the sample the caller means.
    """
    if int(fresh.shape[1]) != len(todo):
        raise ValueError(
            f"fresh has {int(fresh.shape[1])} columns but todo names "
            f"{len(todo)} samples; column j of fresh must be sample "
            "todo[j]"
        )

    position_in_todo = {index: j for j, index in enumerate(todo)}
    columns: Dict[int, Array] = {}
    marker = f"{prefix}:"
    for key in store.keys():
        if not key.startswith(marker):
            continue
        record = store.load(key)
        if record is None:
            continue
        decoded = record[0]
        for position, index in enumerate(decoded.indices):
            if index < nsamples and index not in position_in_todo:
                columns[index] = decoded.values[:, position : position + 1]

    missing = [
        i
        for i in range(nsamples)
        if i not in columns and i not in position_in_todo
    ]
    if missing:
        raise ValueError(
            f"samples {missing} are neither stored under {prefix!r} nor "
            "named in todo, so the result cannot be completed"
        )

    ordered: List[Array] = []
    for index in range(nsamples):
        if index in position_in_todo:
            j = position_in_todo[index]
            ordered.append(fresh[:, j : j + 1])
        else:
            ordered.append(columns[index])
    return bkd.hstack(ordered)


def _encode_key(key: str) -> str:
    """Make a key safe to use as a filename.

    Keys are the caller's and may contain separators -- ``sweep7/0`` is
    a natural thing to write -- which would otherwise create
    directories or escape the store. Percent-encoding every byte that
    is not plainly safe is reversible, so ``keys()`` can return exactly
    what was saved.
    """
    safe = []
    for char in key:
        if char.isalnum() or char in "-_.":
            safe.append(char)
        else:
            safe.append(f"%{ord(char):02x}")
    return "".join(safe)


def _decode_key(name: str) -> str:
    """Invert :func:`_encode_key`."""
    out: List[str] = []
    index = 0
    while index < len(name):
        if name[index] == "%":
            out.append(chr(int(name[index + 1 : index + 3], 16)))
            index += 3
        else:
            out.append(name[index])
            index += 1
    return "".join(out)
