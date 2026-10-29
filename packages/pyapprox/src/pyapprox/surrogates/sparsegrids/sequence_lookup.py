"""Identify a sparse-grid sample by its position in each point sequence.

A nested sparse grid draws every coordinate in dimension ``d`` from one
sequence of distinct points, the union of that dimension's univariate
rules in order of first appearance. A sample's key is the tuple of its
coordinates' positions in those sequences, such as ``"3-0-1-0"``.

That key survives what a resumed run may change: the error indicator,
the refinement order, and the number of points added per level (a
Leja sequence keeps its points when only the level boundaries move). It
does not survive a change of basis or marginal, which moves the points.
The sequences are therefore saved in the store's header, and a lookup
opened on a store whose sequences differ from its own refuses, so a
value is never attached to the wrong point.

Positions rather than coordinates, because the points come from an
optimizer or a formula and can move by round-off between code versions:
an exact float key would then miss, and a rounded one can match the
wrong point. Matching against a short list of known distinct points at
a tolerance is unambiguous.

Pair it with :class:`~pyapprox.interface.evaluation.caching.CachedObjective`
to evaluate only the samples a store lacks::

    lookup = SequencePositionLookup(bkd, subspace_factory, store, max_level)
    model = CachedObjective(target, store, lookup)

The model must be deterministic, as any interpolated model must be: a
stored value is reused as what the model would return now.
"""

from typing import Dict, Generic, List, Optional

from pyapprox.interface.evaluation.protocols import ResultStore
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    SubspaceFactoryProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend

_SEQUENCE_ENTRY = "sequence_{}"


class SequencePositionLookup(Generic[Array]):
    """Key a sample by its position in each dimension's point sequence.

    Satisfies :class:`~pyapprox.interface.evaluation.protocols.SampleLookup`.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    subspace_factory : SubspaceFactoryProtocol[Array]
        The factory the fitter uses. Each dimension's sequence is built
        from it one level at a time, only as far as the samples seen so
        far require.
    store : ResultStore[Array]
        Where the sequences are recorded, in the header. The same store
        the values go in, so the two cannot be paired wrongly.
    max_level : int
        The highest level any dimension may reach, as in the fitter's
        admissibility criterion. A coordinate not found by this level is
        not a point of the grid. It may be lower than the level the
        store was written to: only the points this grid can build are
        checked, and the store's longer sequences are kept.
    atol : float, optional
        Two points closer than this are the same point. Default 1e-12.

    Raises
    ------
    ValueError
        If the factory's rules are not nested; if the store holds
        records but no header, so what their keys mean is unknown; or if
        the store's sequences differ from the factory's.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        subspace_factory: SubspaceFactoryProtocol[Array],
        store: ResultStore[Array],
        max_level: int,
        atol: float = 1e-12,
    ) -> None:
        if not isinstance(subspace_factory, SubspaceFactoryProtocol):
            raise TypeError(
                "subspace_factory must satisfy SubspaceFactoryProtocol, got "
                f"{type(subspace_factory).__name__}"
            )
        if not isinstance(store, ResultStore):
            raise TypeError(
                f"store must satisfy ResultStore, got {type(store).__name__}"
            )
        if not subspace_factory.is_nested():
            raise ValueError(
                "Keying samples by sequence position needs a nested rule "
                "in every dimension (Leja, Clenshaw-Curtis), so that each "
                "level's points extend the previous level's. Gauss rules "
                "are not nested: each level's points are new."
            )
        if max_level < 0:
            raise ValueError(f"max_level must be non-negative, got {max_level}")
        self._bkd = bkd
        self._factory = subspace_factory
        self._store = store
        self._max_level = max_level
        self._atol = atol
        self._nvars = subspace_factory.nvars_physical()
        self._sequences: List[Array] = []
        self._levels: List[int] = []
        # What the store's header held when opened, if anything.
        self._stored: List[Optional[Array]] = [None] * self._nvars
        for dim in range(self._nvars):
            self._sequences.append(self._nodes(dim, 0))
            self._levels.append(0)
        self._check_or_write_header()

    def nvars(self) -> int:
        """Return the number of physical variables."""
        return self._nvars

    def sequence(self, dim: int) -> Array:
        """Return dimension ``dim``'s points built so far, shape (npoints,)."""
        return self._sequences[dim]

    def find(self, *columns: Array) -> Optional[str]:
        """Return the key of this sample, or None if it is not a grid point.

        Parameters
        ----------
        *columns : Array
            Exactly one array, the sample, shape ``(nvars, 1)``. Keys
            depend on the sample alone, so directional quantities, which
            also depend on a direction, are not supported.
        """
        sample = self._single_sample(columns)
        nextended = sum(self._levels)
        positions: List[str] = []
        for dim in range(self._nvars):
            position = self._position(dim, sample[dim, 0])
            if position is None:
                return None
            positions.append(str(position))
        if sum(self._levels) != nextended:
            self._save_header()
        return "-".join(positions)

    def remember(self, *columns: Array) -> str:
        """Return the key of this sample, which must be a grid point.

        Positions are fixed by the sequences, so there is nothing to
        record; this only refuses a sample no key can describe.
        """
        key = self.find(*columns)
        if key is None:
            raise ValueError(
                "sample is not a point of the sparse grid: some coordinate "
                f"is in no sequence up to level {self._max_level}. Only "
                "samples the fitter requested can be keyed by position."
            )
        return key

    def _single_sample(self, columns: tuple[Array, ...]) -> Array:
        if len(columns) != 1:
            raise ValueError(
                f"SequencePositionLookup keys a sample alone; got {len(columns)} arrays"
            )
        sample = columns[0]
        if sample.ndim != 2 or sample.shape != (self._nvars, 1):
            raise ValueError(
                f"sample must have shape ({self._nvars}, 1), got {tuple(sample.shape)}"
            )
        return sample

    def _nodes(self, dim: int, level: int) -> Array:
        """Dimension ``dim``'s univariate nodes at ``level``, shape (n,)."""
        index = self._bkd.zeros((self._nvars,), dtype=self._bkd.int64_dtype())
        index[dim] = level
        return self._factory(index).get_samples_1d(dim)[0]

    def _index_of(self, sequence: Array, coordinate: Array) -> Optional[int]:
        distances = self._bkd.abs(sequence - coordinate)
        nearest = self._bkd.to_int(self._bkd.argmin(distances))
        if self._bkd.to_float(distances[nearest]) <= self._atol:
            return nearest
        return None

    def _extend(self, dim: int) -> bool:
        """Append the next level's new points; False past ``max_level``."""
        level = self._levels[dim] + 1
        if level > self._max_level:
            return False
        sequence = self._sequences[dim]
        nodes = self._nodes(dim, level)
        new = [
            nodes[i : i + 1]
            for i in range(nodes.shape[0])
            if self._index_of(sequence, nodes[i]) is None
        ]
        if new:
            sequence = self._bkd.concatenate([sequence, *new])
        self._sequences[dim] = sequence
        self._levels[dim] = level
        return True

    def _position(self, dim: int, coordinate: Array) -> Optional[int]:
        while True:
            position = self._index_of(self._sequences[dim], coordinate)
            if position is not None:
                return position
            if not self._extend(dim):
                return None

    def _save_header(self) -> None:
        # Keep a stored sequence longer than this grid's: a run capped at
        # a lower level must not cut what a higher-level run recorded.
        # The two agree wherever both exist, as checked on opening.
        header: Dict[str, Array] = {}
        for dim in range(self._nvars):
            stored = self._stored[dim]
            built = self._sequences[dim]
            longer = (
                stored
                if stored is not None and stored.shape[0] > built.shape[0]
                else built
            )
            header[_SEQUENCE_ENTRY.format(dim)] = longer
        self._store.save_header(header)

    def _check_or_write_header(self) -> None:
        header = self._store.load_header()
        if header is None:
            if len(self._store.keys()) > 0:
                raise ValueError(
                    "The store holds records but no header, so nothing "
                    "says which grid points their keys name. Use an empty "
                    "store, or the one a SequencePositionLookup wrote."
                )
            self._save_header()
            return
        expected = {_SEQUENCE_ENTRY.format(dim) for dim in range(self._nvars)}
        if set(header) != expected:
            raise ValueError(
                f"The store's header has entries {sorted(header)}, not the "
                f"{self._nvars} sequences this grid needs: it was written "
                "for a different number of variables or by another caller."
            )
        for dim in range(self._nvars):
            stored = header[_SEQUENCE_ENTRY.format(dim)]
            self._check_sequence(dim, stored)
            self._stored[dim] = stored

    def _check_sequence(self, dim: int, stored: Array) -> None:
        """Compare the stored points this grid can build with its own.

        A grid capped below the level the store reached builds only a
        prefix of the stored sequence, and that prefix is all it can
        ever key, so the points past it need no check. Everything
        buildable within the stored length is built here, so a later
        extension only ever adds points beyond it.
        """
        nstored = stored.shape[0]
        while self._sequences[dim].shape[0] < nstored and self._extend(dim):
            pass
        noverlap = min(nstored, self._sequences[dim].shape[0])
        built = self._sequences[dim][:noverlap]
        mismatch = self._bkd.to_float(
            self._bkd.max(self._bkd.abs(built - stored[:noverlap]))
        )
        if mismatch > self._atol:
            raise ValueError(
                f"The store's points for dimension {dim} differ from this "
                f"grid's by up to {mismatch:.3e}, so its keys name other "
                "points. The marginal, the basis, or the code that places "
                "the points has changed since the store was written; use "
                "a new store."
            )
