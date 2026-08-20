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
from typing import Dict, Generic, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

from pyapprox.interface.evaluation.records import (
    ComputeProvenance,
    Cost,
    Decoded,
)
from pyapprox.util.backends.protocols import Array, Backend

# Optional Decoded fields that are arrays. Named once so adding a
# capability upstream is a single edit here rather than four.
_OPTIONAL_ARRAYS = ("jacobians", "hessians", "jvps", "hvps", "hvp_weights")


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
