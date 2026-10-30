"""Evaluate a model outside pyapprox, one batch of samples at a time.

Some models run where pyapprox cannot call them: a batch system, another
machine, a code only a colleague can run. :class:`ExternalBatchModel`
stands in for such a model. Called on samples, it never evaluates: it
writes them to a batch file and raises :class:`PendingEvaluations`. The
user evaluates the batch however they like, writes the results, and
imports them with :meth:`ExternalBatchModel.import_results`.

It is meant to sit behind
:class:`~pyapprox.interface.evaluation.caching.CachedObjective`, which
calls it only for samples its store lacks. Imported values go into that
store through the same objective, so they are saved under exactly the
keys and in exactly the form a live evaluation would be, and the next
call is served from the store::

    lookup = ...  # what makes two samples the same, e.g. by position
    external = ExternalBatchModel(bkd, nvars, nqoi, lookup, directory)
    model = CachedObjective(external, store, lookup)
    try:
        values = model(samples)
    except PendingEvaluations as pending:
        ...  # evaluate pending.batch_path(), write results_path
        external.import_results(results_path, model)
        values = model(samples)  # now served from the store

**The directory is the state.** Batch files record what was requested,
import files what came back, so a new process holding a new
``ExternalBatchModel`` on the same directory knows exactly what is
outstanding. A call while samples are outstanding must ask for exactly
those samples: asking again before importing gets the same batch file
back rather than a duplicate; asking after a partial import writes a
new batch of only the samples still missing. Asking for anything else
means the caller has diverged from the run that wrote the batch (its
code or setup changed), and raises instead of exporting, since values
for the old batch would then never be used.

File formats (``.npz``, read with ``allow_pickle=False``):

- ``batch-NNNN.npz``: ``samples`` (nvars, n) in the model's input
  domain, ``keys`` (n,) strings.
- results, written by the user: ``keys`` (m,) strings, each one from a
  batch, and ``values`` (nqoi, m), column ``j`` for ``keys[j]``. Any
  order, any subset.
"""

import os
from typing import Dict, Generic, List, Set, Tuple

import numpy as np

from pyapprox.interface.evaluation.caching import CachedObjective
from pyapprox.interface.evaluation.protocols import SampleLookup
from pyapprox.interface.evaluation.stores import save_npz_atomically
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.util.backends.protocols import Array, Backend

_BATCH_PREFIX = "batch-"
_IMPORT_PREFIX = "imported-"


class PendingEvaluations(Exception):
    """Samples await evaluation outside pyapprox.

    Parameters
    ----------
    batch_path : str
        The batch file holding the samples to evaluate.
    nsamples : int
        How many samples it holds.
    """

    def __init__(self, batch_path: str, nsamples: int) -> None:
        super().__init__(
            f"{nsamples} samples await evaluation outside pyapprox, in "
            f"{batch_path}. Evaluate its 'samples', write an .npz with "
            "the batch's 'keys' and their 'values' (nqoi, n), pass it to "
            "ExternalBatchModel.import_results, then call again."
        )
        self._batch_path = batch_path
        self._nsamples = nsamples

    def batch_path(self) -> str:
        """Return the batch file holding the samples to evaluate."""
        return self._batch_path

    def nsamples(self) -> int:
        """Return how many samples await evaluation."""
        return self._nsamples


class ExternalBatchModel(Generic[Array]):
    """A model whose evaluations happen outside pyapprox.

    Satisfies ``ObjectiveProtocol`` with no derivatives, so it can sit
    behind ``CachedObjective``.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    nvars : int
        Number of input variables.
    nqoi : int
        Number of quantities of interest the external model returns.
    lookup : SampleLookup[Array]
        Names each sample in the batch files. Must be the lookup of the
        ``CachedObjective`` this model sits behind.
    directory : str
        Where batch and import files live. Created if absent.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        nvars: int,
        nqoi: int,
        lookup: SampleLookup[Array],
        directory: str,
    ) -> None:
        if not isinstance(lookup, SampleLookup):
            raise TypeError(
                f"lookup must satisfy SampleLookup, got {type(lookup).__name__}"
            )
        self._bkd = bkd
        self._nvars = nvars
        self._nqoi = nqoi
        self._lookup = lookup
        self._directory = directory
        os.makedirs(directory, exist_ok=True)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of input variables."""
        return self._nvars

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        return self._nqoi

    def directory(self) -> str:
        """Return the directory holding batch and import files."""
        return self._directory

    def derivatives(self) -> Derivatives[Array]:
        """No derivatives: the external model returns values only."""
        return Derivatives.none()

    def __call__(self, samples: Array) -> Array:
        """Export the samples not yet requested and raise.

        Raises
        ------
        PendingEvaluations
            Always, naming the batch file to evaluate.
        RuntimeError
            If samples are outstanding and these are not exactly them.
        """
        if samples.ndim != 2 or samples.shape[0] != self._nvars:
            raise ValueError(
                f"samples must have shape ({self._nvars}, n), got "
                f"{tuple(samples.shape)}"
            )
        keys: List[str] = []
        columns: List[Array] = []
        for i in range(samples.shape[1]):
            column = samples[:, i : i + 1]
            key = self._lookup.remember(column)
            if key not in keys:
                keys.append(key)
                columns.append(column)

        batches = self._batches()
        outstanding = self._requested_keys(batches) - self._imported_keys()
        if outstanding:
            if set(keys) != outstanding:
                raise RuntimeError(
                    f"{len(outstanding)} samples requested earlier are still "
                    f"outstanding, but this call asks for {len(keys)} samples "
                    f"of which {len(set(keys) - outstanding)} are new. The "
                    "caller no longer requests what the run that wrote the "
                    "batch did, so its code or setup has changed; import "
                    "the outstanding results first or start a new directory."
                )
            latest_path, latest_keys, _ = batches[-1]
            if set(latest_keys) == outstanding:
                raise PendingEvaluations(latest_path, len(latest_keys))

        path = os.path.join(self._directory, f"{_BATCH_PREFIX}{len(batches):04d}.npz")
        save_npz_atomically(
            path,
            {
                "samples": self._bkd.to_numpy(self._bkd.hstack(columns)),
                "keys": np.asarray(keys, dtype=str),
            },
        )
        raise PendingEvaluations(path, len(keys))

    def import_results(self, results_path: str, into: CachedObjective[Array]) -> int:
        """Store externally computed values; return how many were read.

        Parameters
        ----------
        results_path : str
            An ``.npz`` with ``keys`` (m,) and ``values`` (nqoi, m).
        into : CachedObjective[Array]
            The objective this model sits behind. Values are stored
            through it, so they land exactly as a live evaluation would.

        Raises
        ------
        ValueError
            If ``into`` does not wrap this model, if the file's shapes
            are wrong, or if any key was never requested.
        """
        if into.model() is not self:
            raise ValueError(
                "into must be the CachedObjective wrapping this model, so "
                "values are stored under its lookup's keys"
            )
        with np.load(results_path, allow_pickle=False) as data:
            result_keys = [str(key) for key in data["keys"]]
            values = self._bkd.asarray(data["values"])
        if values.ndim != 2 or values.shape != (self._nqoi, len(result_keys)):
            raise ValueError(
                f"values must have shape ({self._nqoi}, {len(result_keys)}) "
                f"to match {len(result_keys)} keys, got {tuple(values.shape)}"
            )
        requested = self._requested_samples(self._batches())
        unknown = [key for key in result_keys if key not in requested]
        if unknown:
            raise ValueError(
                f"{len(unknown)} result keys were never requested, e.g. "
                f"{unknown[:3]}; refusing to store values for samples no "
                "batch asked for"
            )
        if not result_keys:
            return 0
        samples = self._bkd.hstack([requested[key] for key in result_keys])
        supply = _ImportedValues(self._bkd, self._lookup, result_keys, values)
        into.cached(samples, supply, "values")
        nimports = len(self._files(_IMPORT_PREFIX))
        save_npz_atomically(
            os.path.join(self._directory, f"{_IMPORT_PREFIX}{nimports:04d}.npz"),
            {"keys": np.asarray(result_keys, dtype=str)},
        )
        return len(result_keys)

    def _files(self, prefix: str) -> List[str]:
        return sorted(
            name
            for name in os.listdir(self._directory)
            if name.startswith(prefix) and name.endswith(".npz")
        )

    def _batches(self) -> List[Tuple[str, List[str], Array]]:
        """Every batch file, in order: (path, keys, samples)."""
        batches: List[Tuple[str, List[str], Array]] = []
        for name in self._files(_BATCH_PREFIX):
            path = os.path.join(self._directory, name)
            with np.load(path, allow_pickle=False) as data:
                keys = [str(key) for key in data["keys"]]
                samples = self._bkd.asarray(data["samples"])
            batches.append((path, keys, samples))
        return batches

    def _requested_keys(self, batches: List[Tuple[str, List[str], Array]]) -> Set[str]:
        return {key for _, keys, _ in batches for key in keys}

    def _requested_samples(
        self, batches: List[Tuple[str, List[str], Array]]
    ) -> Dict[str, Array]:
        requested: Dict[str, Array] = {}
        for _, keys, samples in batches:
            for i, key in enumerate(keys):
                requested[key] = samples[:, i : i + 1]
        return requested

    def _imported_keys(self) -> Set[str]:
        imported: Set[str] = set()
        for name in self._files(_IMPORT_PREFIX):
            path = os.path.join(self._directory, name)
            with np.load(path, allow_pickle=False) as data:
                imported.update(str(key) for key in data["keys"])
        return imported


class _ImportedValues(Generic[Array]):
    """Serve imported values to ``CachedObjective`` as if computed."""

    def __init__(
        self,
        bkd: Backend[Array],
        lookup: SampleLookup[Array],
        keys: List[str],
        values: Array,
    ) -> None:
        self._bkd = bkd
        self._lookup = lookup
        self._columns = {key: j for j, key in enumerate(keys)}
        self._values = values

    def __call__(self, samples: Array) -> Array:
        """Return the imported values for ``samples``, shape (nqoi, n)."""
        pieces: List[Array] = []
        for i in range(samples.shape[1]):
            key = self._lookup.remember(samples[:, i : i + 1])
            j = self._columns[key]
            pieces.append(self._values[:, j : j + 1])
        return self._bkd.hstack(pieces)
