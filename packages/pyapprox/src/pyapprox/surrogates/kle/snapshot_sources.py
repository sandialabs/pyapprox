r"""Snapshots read in pieces, for data too large to hold at once.

A snapshot matrix :math:`S` is ``(nstates, nsamples)``, and the solvers
here take it as one resident array. That is the right interface until
``nstates`` is large: at :math:`10^7` states and 1000 snapshots the
matrix is 80 GB, and the decomposition allocates several more arrays of
that size, so an ordinary fit needs several times the data in memory.

This module supplies the seam a file-backed or generated source plugs
into. The algorithms ask for what the mathematics needs, and the source
is responsible for making that cheap on whatever format it wraps.

**Blocks are rows, and this is not a preference.** A row block is
``(k, nsamples)`` -- every snapshot, but only ``k`` of the mesh points.
The Gram :math:`S^T M S` a snapshot decomposition is built on is a *sum
over rows*,

.. math:: S^T M S = \sum_i m_i\, (\mathrm{row}_i \otimes \mathrm{row}_i)

so a row block contributes a term to the whole matrix and the sum
accumulates in one pass. A *column* block of shape ``(nstates, k)``
instead yields one ``(k, k)`` diagonal block, and assembling the full
Gram from column blocks needs every *pair* of them --
:math:`O(n_{\mathrm{blocks}}^2)` reads rather than
:math:`O(n_{\mathrm{blocks}})`. For 100 GB in 1 GB blocks that is the
difference between reading 100 GB and reading 10 TB.

The awkward part, and the reason this is a protocol rather than a
utility: most solver output is written one record per timestep, which is
column-major in these terms. An adapter for such a format has to
transpose, cache, or read strided, and that cost is real. It belongs to
the adapter, not to the algorithm -- an algorithm that tried to
accommodate a format's layout would be wrong for the next format.
"""

from typing import (
    Generic,
    Iterator,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.randomized import MatVecOperator

#: Default working size for one row block. Large enough that a read is
#: sequential on any storage worth using, small enough to sit alongside
#: the arrays a decomposition builds from it.
DEFAULT_BLOCK_BYTES = 1 << 30


@runtime_checkable
class SnapshotSourceProtocol(Protocol, Generic[Array]):
    r"""Snapshots delivered as row blocks rather than one array.

    Implementations may hold the data, read it from a file, or generate
    it. What they promise is that iterating :meth:`row_blocks` visits
    every row exactly once, in increasing order, and that the blocks
    concatenate to the full ``(nstates, nsamples)`` matrix.

    The ordering matters to more than tidiness: a consumer writing an
    ambient-sized result back out block by block relies on it to know
    where each piece belongs without tracking position itself.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend blocks are produced in."""
        ...

    def nstates(self) -> int:
        """Return the ambient dimension, the number of rows."""
        ...

    def nsamples(self) -> int:
        """Return the number of snapshots, the number of columns."""
        ...

    def row_blocks(
        self, max_bytes: Optional[int] = None
    ) -> Iterator[Tuple[slice, Array]]:
        """Yield ``(rows, block)`` covering every row once, in order.

        ``block`` has shape ``(rows.stop - rows.start, nsamples)`` and
        ``rows`` indexes the ambient dimension, so a caller can place a
        result computed from the block without counting.

        Parameters
        ----------
        max_bytes : int, optional
            Roughly how much memory one block may occupy. A budget
            rather than a row count, because the caller knows what it
            can afford while only the source knows its element size and
            chunk layout -- a row count would force every caller to do
            arithmetic it cannot do correctly for an arbitrary format.
            None lets the source choose.

            Advisory. A source yields at least one row per block however
            small the budget, and may exceed it when its storage has a
            natural unit larger than the request.
        """
        ...


class ArraySnapshotSource(Generic[Array]):
    """A source backed by an array already in memory.

    The adapter that lets a consumer be written against the protocol
    without penalising the case that does not need it: the blocks are
    views, so iterating costs nothing beyond the arithmetic.

    Its other role is as the reference an out-of-core source is checked
    against. A streaming implementation is correct when it produces the
    same decomposition as this one on data small enough to hold both
    ways, which is a sharper test than any property asserted about the
    streaming path alone.

    Parameters
    ----------
    snapshots : Array
        Shape ``(nstates, nsamples)``, columns are snapshots.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, snapshots: Array, bkd: Backend[Array]) -> None:
        if snapshots.ndim != 2:
            raise ValueError(
                "snapshots must be 2D (nstates, nsamples), got "
                f"ndim={snapshots.ndim}"
            )
        self._snapshots = snapshots
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nstates(self) -> int:
        """Return the ambient dimension."""
        return int(self._snapshots.shape[0])

    def nsamples(self) -> int:
        """Return the number of snapshots."""
        return int(self._snapshots.shape[1])

    def row_blocks(
        self, max_bytes: Optional[int] = None
    ) -> Iterator[Tuple[slice, Array]]:
        """Yield views of ``max_bytes``-sized row blocks, in order."""
        nrows = rows_per_block(
            self.nsamples(), self._snapshots.dtype, max_bytes
        )
        for start in range(0, self.nstates(), nrows):
            rows = slice(start, min(start + nrows, self.nstates()))
            yield rows, self._snapshots[rows, :]

    def to_array(self) -> Array:
        """Return the underlying array."""
        return self._snapshots

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nstates={self.nstates()}, "
            f"nsamples={self.nsamples()})"
        )


def rows_per_block(
    nsamples: int, dtype: object, max_bytes: Optional[int] = None
) -> int:
    """How many rows fit in ``max_bytes``, at least one.

    Shared so every source divides the budget the same way, and so the
    floor of one row is not re-derived per implementation. That floor is
    what keeps a budget smaller than a single row a slow read rather
    than an empty iterator.
    """
    budget = DEFAULT_BLOCK_BYTES if max_bytes is None else max_bytes
    if budget < 1:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    itemsize = getattr(dtype, "itemsize", 8)
    return max(1, int(budget // (max(nsamples, 1) * itemsize)))


class SnapshotSourceOperator(MatVecOperator[Array]):
    r"""A snapshot source as a matrix-free operator.

    The adapter that lets a randomized decomposition read from storage
    without knowing it has: every product it needs is a row-block
    accumulation, so the source is asked for blocks and nothing of size
    ``(nstates, nsamples)`` is ever formed.

    Each of the four products decomposes differently, and the difference
    is the whole content of this class:

    - :meth:`apply`, :math:`S x`, assigns to the rows of the result, so
      it streams to an output of the ambient size the caller already
      accepted.
    - :meth:`apply_transpose`, :math:`S^T y`, and :meth:`right_apply`,
      :math:`y S`, are *sums* over rows, so their results are small
      whatever the ambient dimension.

    Parameters
    ----------
    source : SnapshotSourceProtocol[Array]
        Supplies the row blocks.
    bkd : Backend[Array]
        Computational backend.
    max_bytes : int, optional
        Passed to :meth:`SnapshotSourceProtocol.row_blocks` on every
        pass. None lets the source choose.
    """

    def __init__(
        self,
        source: SnapshotSourceProtocol[Array],
        bkd: Backend[Array],
        max_bytes: Optional[int] = None,
    ) -> None:
        super().__init__(bkd)
        self._source = source
        self._max_bytes = max_bytes

    def source(self) -> SnapshotSourceProtocol[Array]:
        """Return the source the blocks come from."""
        return self._source

    def nrows(self) -> int:
        """Return the ambient dimension."""
        return self._source.nstates()

    def ncols(self) -> int:
        """Return the number of snapshots."""
        return self._source.nsamples()

    def apply(self, vecs: Array) -> Array:
        r"""Return :math:`S x`, shape ``(nstates, ncols)``.

        Ambient-sized, so it is the one product here whose result the
        caller must be able to hold. A randomized decomposition asks for
        it with ``ncols = nterms + noversampling``, which is the output
        size rather than the data's.
        """
        bkd = self._bkd
        out = bkd.zeros((self.nrows(), int(vecs.shape[1])))
        for rows, block in self._source.row_blocks(self._max_bytes):
            out[rows, :] = bkd.dot(block, vecs)
        return out

    def apply_transpose(self, vecs: Array) -> Array:
        r"""Return :math:`S^T y`, shape ``(nsamples, ncols)``."""
        bkd = self._bkd
        out = bkd.zeros((self.ncols(), int(vecs.shape[1])))
        for rows, block in self._source.row_blocks(self._max_bytes):
            out = out + bkd.dot(block.T, vecs[rows, :])
        return out

    def right_apply(self, vecs: Array) -> Array:
        r"""Return :math:`y S`, shape ``(nrows_of_vecs, nsamples)``."""
        bkd = self._bkd
        out = bkd.zeros((int(vecs.shape[0]), self.ncols()))
        for rows, block in self._source.row_blocks(self._max_bytes):
            out = out + bkd.dot(vecs[:, rows], block)
        return out

    def right_apply_implemented(self) -> bool:
        """Return True; the product is a row-block sum like the others."""
        return True

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nrows={self.nrows()}, "
            f"ncols={self.ncols()})"
        )


def as_snapshot_source(
    snapshots: "Array | SnapshotSourceProtocol[Array]",
    bkd: Backend[Array],
) -> SnapshotSourceProtocol[Array]:
    """Return ``snapshots`` as a source, wrapping a bare array.

    Lets a function widen from ``Array`` to either without its callers
    changing, and without the function branching on which it received.
    """
    if isinstance(snapshots, SnapshotSourceProtocol):
        return snapshots
    return ArraySnapshotSource(snapshots, bkd)
