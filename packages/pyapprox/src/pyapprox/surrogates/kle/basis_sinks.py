r"""Where an ambient-sized basis is written, and how it is read back.

:mod:`~pyapprox.surrogates.kle.snapshot_sources` handles the input side:
a snapshot matrix too large to hold, delivered as row blocks. This module
is the output side of the same problem. A basis is
``(nstates, nterms)``, so at :math:`10^7` states and 50 terms it is 4 GB
-- smaller than the data it came from, but past the point where it can be
assembled in memory and handed back as an array.

The lift that produces it -- :func:`~pyapprox.surrogates.kle.basis_lift.lift_basis`
-- computes one row block at a time and has nowhere to put the result.
A sink is that somewhere. It receives ``(rows, block)`` in the same
shape the source yields, so the lift is a loop that reads a block and
writes a block, holding neither end.

**A sink is write-once and then readable.** :meth:`BasisSinkProtocol.write`
accumulates, :meth:`BasisSinkProtocol.finalize` closes and returns a
:class:`~pyapprox.surrogates.kle.basis_operator.BasisOperatorProtocol`.
Splitting the two keeps a half-written basis from being read: the only
object that can be contracted against is the one ``finalize`` returned,
so a lift that raised partway through cannot leave a plausible-looking
basis behind.

**Why the sink returns the operator rather than the caller constructing
it.** Only the sink knows what it wrote to -- an array, a memmap, a file
in some format -- and so only it can produce the reader that matches.
A caller that constructed its own would have to know the storage, which
is the coupling the seam exists to remove.
"""

from typing import (
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from pyapprox.surrogates.kle.basis_operator import (
    ArrayBasis,
    BasisOperatorProtocol,
)
from pyapprox.surrogates.kle.snapshot_sources import rows_per_block
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class RowFetcherProtocol(Protocol, Generic[Array]):
    """Fetches scattered rows of a stored basis directly.

    Optional, because whether it can be done cheaply is a property of
    the storage: a memmap indexes into a page table, while a format
    that must be decompressed or read in order cannot, and would have
    to walk to the same place a caller could walk itself.

    A sink supplies one only when its storage indexes randomly. Where
    it does, the difference is not marginal: fetching a few hundred
    scattered rows of a gigabyte-scale basis touches a few hundred
    pages, against a full pass to keep the same rows.
    """

    def __call__(self, indices: Sequence[int]) -> Array:
        """Return rows ``indices``, shape ``(len(indices), nterms)``."""
        ...


@runtime_checkable
class RowBlockReaderProtocol(Protocol, Generic[Array]):
    """Reads a stored basis back as row blocks.

    A protocol rather than a bare ``Callable`` so the byte budget is a
    named, optional argument: a reader that took a required positional
    or named it something else would otherwise typecheck and then fail
    at the first pass.
    """

    def __call__(
        self, max_bytes: Optional[int] = None
    ) -> Iterator[Tuple[slice, Array]]:
        """Yield ``(rows, block)`` covering every row once, in order."""
        ...


@runtime_checkable
class BasisSinkProtocol(Protocol, Generic[Array]):
    r"""Somewhere to put an ambient-sized basis, one row block at a time.

    The write-side mirror of
    :class:`~pyapprox.surrogates.kle.snapshot_sources.SnapshotSourceProtocol`:
    that yields ``(rows, block)``, this accepts them.

    Implementations promise that after every row has been written once,
    :meth:`finalize` returns a basis equal to the concatenation of the
    blocks in row order.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend blocks arrive in."""
        ...

    def nstates(self) -> int:
        """Return the ambient dimension the sink was sized for."""
        ...

    def nterms(self) -> int:
        """Return the number of basis vectors the sink was sized for."""
        ...

    def write(self, rows: slice, block: Array) -> None:
        """Store ``block`` at ``rows``.

        ``block`` has shape ``(rows.stop - rows.start, nterms)``. Blocks
        may arrive in any order and must not overlap; the sink places
        each by its slice rather than by arrival, so a producer that
        reads its input out of order does not have to buffer.
        """
        ...

    def finalize(self) -> BasisOperatorProtocol[Array]:
        """Close the sink and return the basis as an operator.

        Raises if any row was never written, since the alternative is a
        basis silently containing whatever the storage was initialized
        to -- zeros make an orthonormality check fail somewhere far from
        the lift that skipped the block.
        """
        ...


class ArrayBasisSink(Generic[Array]):
    """A sink that assembles the basis in memory.

    The implementation for a basis that does fit, and the reference any
    other is checked against: a streaming sink is correct when it
    finalizes to the same basis as this one on a problem small enough to
    run both ways.

    Parameters
    ----------
    nstates : int
        Ambient dimension.
    nterms : int
        Number of basis vectors.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self, nstates: int, nterms: int, bkd: Backend[Array]
    ) -> None:
        if nstates < 1 or nterms < 1:
            raise ValueError(
                f"nstates and nterms must be positive, got "
                f"nstates={nstates}, nterms={nterms}"
            )
        self._nstates = nstates
        self._nterms = nterms
        self._bkd = bkd
        self._basis = bkd.zeros((nstates, nterms))
        self._written: List[slice] = []
        self._finalized = False

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nstates(self) -> int:
        """Return the ambient dimension."""
        return self._nstates

    def nterms(self) -> int:
        """Return the number of basis vectors."""
        return self._nterms

    def write(self, rows: slice, block: Array) -> None:
        """Store ``block`` at ``rows``."""
        if self._finalized:
            raise RuntimeError(
                "cannot write to a sink that has been finalized"
            )
        validate_block(
            rows, block, self._nstates, self._nterms, self._written
        )
        self._basis[rows, :] = block
        self._written.append(rows)

    def finalize(self) -> BasisOperatorProtocol[Array]:
        """Return the assembled basis, checking every row was written."""
        require_complete_coverage(self._written, self._nstates)
        self._finalized = True
        return ArrayBasis(self._basis, self._bkd)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nstates={self._nstates}, "
            f"nterms={self._nterms})"
        )


class MemmapBasisSink(Generic[Array]):
    """A sink that writes the basis to a file on disk.

    The implementation the ambient sizes call for: blocks go to a
    ``numpy.memmap``, so what stays resident is one block and the
    operating system's page cache rather than the whole basis.
    :meth:`finalize` returns a :class:`StreamingBasis` reading the same
    file back, which is why this is the sink a lift at scale writes to
    and ``ArrayBasisSink`` is the one its test writes to.

    NumPy-backed regardless of the computational backend, because a
    memmap is a file layout rather than a computation: the file holds
    bytes in a fixed dtype and a backend array has to become those bytes
    to be stored. Blocks are converted on the way in and back on the way
    out, per block, so the cost is a block's worth of copy and never the
    basis's.

    Parameters
    ----------
    path : str
        Where to write. Overwritten if it exists.
    nstates : int
        Ambient dimension.
    nterms : int
        Number of basis vectors.
    bkd : Backend[Array]
        Backend blocks arrive in and are returned in.
    max_bytes : int, optional
        Byte budget the returned :class:`StreamingBasis` reads with.
    """

    def __init__(
        self,
        path: str,
        nstates: int,
        nterms: int,
        bkd: Backend[Array],
        max_bytes: Optional[int] = None,
    ) -> None:
        import numpy as np

        if nstates < 1 or nterms < 1:
            raise ValueError(
                f"nstates and nterms must be positive, got "
                f"nstates={nstates}, nterms={nterms}"
            )
        self._path = path
        self._nstates = nstates
        self._nterms = nterms
        self._bkd = bkd
        self._max_bytes = max_bytes
        self._memmap = np.memmap(
            path, dtype=np.float64, mode="w+", shape=(nstates, nterms)
        )
        self._written: List[slice] = []
        self._finalized = False

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nstates(self) -> int:
        """Return the ambient dimension."""
        return self._nstates

    def nterms(self) -> int:
        """Return the number of basis vectors."""
        return self._nterms

    def path(self) -> str:
        """Return the file the basis is written to."""
        return self._path

    def write(self, rows: slice, block: Array) -> None:
        """Store ``block`` at ``rows``, converting to the file's dtype."""
        if self._finalized:
            raise RuntimeError(
                "cannot write to a sink that has been finalized"
            )
        validate_block(
            rows, block, self._nstates, self._nterms, self._written
        )
        self._memmap[rows, :] = self._bkd.to_numpy(block)
        self._written.append(rows)

    def finalize(self) -> BasisOperatorProtocol[Array]:
        """Flush the file and return a basis that reads it back."""
        import numpy as np

        require_complete_coverage(self._written, self._nstates)
        self._memmap.flush()
        self._finalized = True
        path, nstates, nterms = self._path, self._nstates, self._nterms
        bkd = self._bkd

        def read_blocks(
            max_bytes: Optional[int] = None,
        ) -> Iterator[Tuple[slice, Array]]:
            """Re-open per pass, so a reader holds no file handle."""
            handle = np.memmap(
                path, dtype=np.float64, mode="r", shape=(nstates, nterms)
            )
            nrows = basis_rows_per_block(
                nterms, handle.dtype, max_bytes
            )
            for start in range(0, nstates, nrows):
                stop = min(start + nrows, nstates)
                yield slice(start, stop), bkd.array(
                    np.asarray(handle[start:stop, :])
                )

        def fetch_rows(indices: Sequence[int]) -> Array:
            """Read scattered rows directly, touching only their pages.

            A memmap indexes through the page table, so this is what
            makes decoding at a subsample cheap: a few hundred rows of
            a gigabyte-scale basis cost a few hundred pages rather than
            a pass over the file.
            """
            handle = np.memmap(
                path, dtype=np.float64, mode="r", shape=(nstates, nterms)
            )
            return bkd.array(
                np.asarray(handle[np.asarray(indices, dtype=int), :])
            )

        return StreamingBasis(
            read_blocks,
            nstates,
            nterms,
            bkd,
            max_bytes=self._max_bytes,
            fetch_rows=fetch_rows,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(path={self._path!r}, "
            f"nstates={self._nstates}, nterms={self._nterms})"
        )


class StreamingBasis(Generic[Array]):
    r"""A basis read from a source in row blocks.

    Satisfies
    :class:`~pyapprox.surrogates.kle.basis_operator.BasisOperatorProtocol`
    without ever holding ``(nstates, nterms)`` values at once. What it
    holds instead is the *recipe*: the row-block reader, plus the column
    selection, scaling and squaring that have been asked for. Those are
    applied to each block as it arrives, which is why
    :meth:`select` and :meth:`scale` can compose without materializing --
    they extend the recipe and read nothing.

    :meth:`apply_transpose` is the operation this exists for. It
    contracts over rows, so it accumulates a ``(nterms, ncols)`` result
    over blocks and the ambient dimension never appears in the output.

    :meth:`apply` cannot do that: its result *is* ambient-sized. It is
    offered anyway, because a caller reconstructing a handful of fields
    for a plot has already accepted an array of that size, and refusing
    would push them to :meth:`to_array`, which is far larger. A caller
    who cannot hold the result should write it to a sink instead.

    Parameters
    ----------
    read_blocks : RowBlockReaderProtocol[Array]
        Called with an optional byte budget, returns an iterator of
        ``(rows, block)`` covering every row once in increasing order.
        ``block`` is ``(nrows_in_block, nterms_stored)`` -- the *stored*
        width, before any selection this basis has recorded.
    nstates : int
        Ambient dimension.
    nterms : int
        Number of stored basis vectors, before selection.
    bkd : Backend[Array]
        Computational backend.
    columns : Sequence[int], optional
        Columns of the stored basis this one presents, in order. None
        means all of them, unpermuted.
    factors : Array, optional
        One scaling per presented column, applied after selection.
    squared : bool
        Whether each block is squared elementwise as it is read.
    max_bytes : int, optional
        Byte budget passed to ``read_blocks`` on every pass.
    fetch_rows : RowFetcherProtocol[Array], optional
        Fetches scattered rows directly. Supplied by a sink whose
        storage indexes randomly, None otherwise -- absence is the
        honest report that the storage cannot do better than a pass,
        not a missing feature. :meth:`rows` falls back to walking the
        blocks when it is None, which is correct either way and
        differs only in what it reads.
    """

    def __init__(
        self,
        read_blocks: RowBlockReaderProtocol[Array],
        nstates: int,
        nterms: int,
        bkd: Backend[Array],
        columns: Optional[Sequence[int]] = None,
        factors: Optional[Array] = None,
        squared: bool = False,
        max_bytes: Optional[int] = None,
        fetch_rows: Optional[RowFetcherProtocol[Array]] = None,
    ) -> None:
        if not isinstance(read_blocks, RowBlockReaderProtocol):
            raise TypeError(
                "read_blocks must be callable, returning row blocks; got "
                f"{type(read_blocks).__name__}"
            )
        self._read_blocks = read_blocks
        self._nstates = nstates
        self._stored_nterms = nterms
        self._bkd = bkd
        self._columns = (
            list(range(nterms)) if columns is None else list(columns)
        )
        self._factors = factors
        self._squared = squared
        self._max_bytes = max_bytes
        self._fetch_rows = fetch_rows

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nstates(self) -> int:
        """Return the ambient dimension."""
        return self._nstates

    def nterms(self) -> int:
        """Return the number of basis vectors presented."""
        return len(self._columns)

    def nrows(self) -> int:
        """Alias for :meth:`nstates`, for operator consumers."""
        return self.nstates()

    def ncols(self) -> int:
        """Alias for :meth:`nterms`, for operator consumers."""
        return self.nterms()

    def _derive(
        self,
        columns: Optional[Sequence[int]] = None,
        factors: Optional[Array] = None,
        squared: Optional[bool] = None,
    ) -> "StreamingBasis[Array]":
        """Return a copy of this basis with part of the recipe replaced."""
        return StreamingBasis(
            self._read_blocks,
            self._nstates,
            self._stored_nterms,
            self._bkd,
            self._columns if columns is None else columns,
            self._factors if factors is None else factors,
            self._squared if squared is None else squared,
            self._max_bytes,
            self._fetch_rows,
        )

    def select(self, columns: Sequence[int]) -> "StreamingBasis[Array]":
        """Return the basis restricted to ``columns``, in that order.

        Composes against the columns currently presented rather than the
        stored ones, so selecting twice narrows -- matching
        :class:`~pyapprox.surrogates.kle.basis_operator.ArrayBasis`.
        Any scaling already recorded is restricted alongside, since a
        factor belongs to the column it was given for.
        """
        chosen = list(columns)
        for column in chosen:
            if not 0 <= column < self.nterms():
                raise ValueError(
                    f"column {column} out of range for a basis with "
                    f"{self.nterms()} terms"
                )
        factors = self._factors
        if factors is not None:
            factors = factors[self._bkd.asarray(chosen, dtype=int)]
        return self._derive(
            columns=[self._columns[column] for column in chosen],
            factors=factors,
        )

    def scale(self, factors: Array) -> "StreamingBasis[Array]":
        """Return the basis with column ``j`` scaled by ``factors[j]``."""
        if factors.ndim != 1 or int(factors.shape[0]) != self.nterms():
            raise ValueError(
                f"factors must be 1D with one entry per term "
                f"({self.nterms()}), got shape {tuple(factors.shape)}"
            )
        combined = (
            factors if self._factors is None else self._factors * factors
        )
        return self._derive(factors=combined)

    def square(self) -> "StreamingBasis[Array]":
        """Return the elementwise square of the basis.

        The recorded scaling is squared with it, so that squaring after
        scaling gives :math:`(V d)^2` rather than :math:`V^2 d` -- the
        two differ, and the block only sees the combined recipe.
        """
        factors = None if self._factors is None else self._factors**2
        return self._derive(factors=factors, squared=True)

    def blocks(
        self, max_bytes: Optional[int] = None
    ) -> Iterator[Tuple[slice, Array]]:
        """Yield ``(rows, block)`` of the basis this object presents.

        The recipe applied, so each block is
        ``(nrows_in_block, nterms)`` and already selected, squared and
        scaled. What a consumer iterates when it wants the basis itself
        rather than a product with it -- writing to a sink, say.
        """
        budget = self._max_bytes if max_bytes is None else max_bytes
        for rows, block in self._read_blocks(budget):
            yield rows, self._apply_recipe(block)

    def _apply_recipe(self, block: Array) -> Array:
        """Select, square and scale a block, in the recorded order.

        Shared by :meth:`blocks` and :meth:`rows` so a row obtained
        either way carries the same transformations -- the fetched path
        would otherwise return stored columns where the walked path
        returns presented ones.
        """
        chosen = block[:, self._bkd.asarray(self._columns, dtype=int)]
        if self._squared:
            chosen = chosen**2
        if self._factors is not None:
            chosen = chosen * self._factors
        return chosen

    def rows(self, indices: Sequence[int]) -> Array:
        r"""Return the named rows, shape ``(len(indices), nterms)``.

        Uses the sink's row fetcher when it supplied one, which reads
        only the pages those rows fall on. Without one the blocks are
        walked and the wanted rows kept -- correct, and a full pass to
        retain a handful of rows, which is why a storage that can index
        randomly says so rather than leaving this to guess.
        """
        wanted = _validate_row_indices(indices, self._nstates)
        if self._fetch_rows is not None:
            return self._apply_recipe(self._fetch_rows(wanted))
        out = self._bkd.zeros((len(wanted), self.nterms()))
        for block_rows, block in self.blocks():
            start = 0 if block_rows.start is None else int(block_rows.start)
            stop = (
                self._nstates
                if block_rows.stop is None
                else int(block_rows.stop)
            )
            for position, index in enumerate(wanted):
                if start <= index < stop:
                    out[position, :] = block[index - start, :]
        return out

    def apply(self, coefs: Array) -> Array:
        r"""Return :math:`V c`, shape ``(nstates, ncols)``.

        Ambient-sized. Assembled block by block, so the peak cost is the
        result plus one block rather than the result plus the basis --
        but the result is still ``(nstates, ncols)`` and a caller who
        cannot hold that should write to a sink instead.
        """
        out = self._bkd.zeros((self._nstates, int(coefs.shape[1])))
        for rows, block in self.blocks():
            out[rows, :] = self._bkd.dot(block, coefs)
        return out

    def apply_transpose(self, fields: Array) -> Array:
        r"""Return :math:`V^T f`, shape ``(nterms, ncols)``.

        A sum over row blocks, so nothing ambient-sized is formed and
        the result is small whatever ``nstates`` is. The reason this
        class exists.
        """
        out = self._bkd.zeros((self.nterms(), int(fields.shape[1])))
        for rows, block in self.blocks():
            out = out + self._bkd.dot(block.T, fields[rows, :])
        return out

    def to_array(self) -> Array:
        """Return the basis as a dense array, reading every block.

        Allowed rather than refused because the sizes that make this a
        mistake are not knowable here: a streaming basis over a small
        file is ordinary, and a caller serializing or plotting one needs
        the array. The cost is explicit in the call.
        """
        out = self._bkd.zeros((self._nstates, self.nterms()))
        for rows, block in self.blocks():
            out[rows, :] = block
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nstates={self._nstates}, "
            f"nterms={self.nterms()})"
        )


def _validate_row_indices(
    indices: Sequence[int], nstates: int
) -> List[int]:
    """Check every requested row exists, and return them as a list.

    Checked up front rather than per block: an out-of-range index would
    otherwise be silently skipped by the walk, leaving that row of the
    result as whatever it was initialized to.
    """
    wanted = [int(index) for index in indices]
    for index in wanted:
        if not 0 <= index < nstates:
            raise ValueError(
                f"row {index} out of range for a basis with "
                f"{nstates} states"
            )
    return wanted


def validate_block(
    rows: slice,
    block: Array,
    nstates: int,
    nterms: int,
    written: Sequence[slice],
) -> None:
    """Check ``(rows, block)`` is a placeable, non-overlapping piece.

    Shared by every sink so the diagnosis does not depend on which one
    received the block. Each condition here would otherwise surface as a
    wrong basis rather than an error: a mis-shaped block broadcasts, an
    out-of-range slice silently clips, and an overlapping one overwrites
    rows already written.
    """
    if rows.step not in (None, 1):
        raise ValueError(
            f"rows must be contiguous, got step={rows.step}"
        )
    start = 0 if rows.start is None else int(rows.start)
    stop = nstates if rows.stop is None else int(rows.stop)
    if start < 0 or stop > nstates or start >= stop:
        raise ValueError(
            f"rows {start}:{stop} is not a non-empty range within "
            f"nstates={nstates}"
        )
    if block.ndim != 2:
        raise ValueError(
            f"block must be 2D (nrows, nterms), got ndim={block.ndim}"
        )
    if int(block.shape[0]) != stop - start:
        raise ValueError(
            f"block has {block.shape[0]} rows but rows {start}:{stop} "
            f"expects {stop - start}"
        )
    if int(block.shape[1]) != nterms:
        raise ValueError(
            f"block has {block.shape[1]} columns but the sink was sized "
            f"for {nterms} terms"
        )
    for other in written:
        other_start = 0 if other.start is None else int(other.start)
        other_stop = nstates if other.stop is None else int(other.stop)
        if start < other_stop and other_start < stop:
            raise ValueError(
                f"rows {start}:{stop} overlap {other_start}:{other_stop}, "
                "which has already been written"
            )


def require_complete_coverage(
    written: Sequence[slice], nstates: int
) -> None:
    """Raise unless ``written`` covers ``0:nstates`` exactly once.

    Called at finalize. An unwritten row keeps whatever the storage was
    initialized to, which for zeros means a basis that fails an
    orthonormality check somewhere with no indication of which block was
    skipped.
    """
    bounds = sorted(
        (
            0 if rows.start is None else int(rows.start),
            nstates if rows.stop is None else int(rows.stop),
        )
        for rows in written
    )
    covered = 0
    for start, stop in bounds:
        if start != covered:
            raise ValueError(
                f"rows {covered}:{start} were never written; a basis "
                "with unwritten rows is not the basis it claims to be"
            )
        covered = stop
    if covered != nstates:
        raise ValueError(
            f"rows {covered}:{nstates} were never written; a basis with "
            "unwritten rows is not the basis it claims to be"
        )


def basis_rows_per_block(
    nterms: int, dtype: object, max_bytes: Optional[int] = None
) -> int:
    """How many basis rows fit in ``max_bytes``, at least one.

    The basis-shaped counterpart of
    :func:`~pyapprox.surrogates.kle.snapshot_sources.rows_per_block`,
    which divides by ``nsamples``. A basis block is ``(k, nterms)``, so
    the same budget holds more rows of it than of the snapshots it came
    from whenever the expansion is a reduction.
    """
    return rows_per_block(nterms, dtype, max_bytes)
