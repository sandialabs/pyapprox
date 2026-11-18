r"""How many rows of a tall array to process at a time.

Several dense computations walk a tall ``nrows x ncols`` array whose rows
are independent: evaluating a kernel between many points and a few
landmarks, a tall-skinny QR, or the column update of a pivoted Cholesky
factorization. Processed in one piece, each allocates temporaries of the
full size, and a few of them can exceed memory even when the result is
small. Processed in blocks of rows, the temporaries are bounded by the
block size instead.

The consumer owns the loop; a :class:`RowBlockingProtocol` only decides
the block size, so how large a block may be is chosen by the caller and
injected. :class:`NoBlocking` (one block, the full array) is the default
everywhere it is accepted, so small problems run exactly the unblocked
arithmetic. :class:`ByteBudgetBlocking` bounds the bytes of one block and
therefore leaves arrays that fit the budget in one block too.
:class:`FixedRowBlocking` uses a constant row count.

Consumers iterate over :func:`row_block_ranges`, which also checks what
the blocking returns.
"""

from typing import Protocol, runtime_checkable

#: Bytes per entry assumed when converting a byte budget to rows (float64,
#: the library's default dtype).
_ITEMSIZE = 8


@runtime_checkable
class RowBlockingProtocol(Protocol):
    """Chooses the rows per block for a tall ``nrows x ncols`` array."""

    def rows_per_block(self, nrows: int, ncols: int) -> int:
        """Return the rows of one block, in ``[1, nrows]``.

        Called only with ``nrows >= 1``. ``ncols`` is the width of the
        largest temporary one block allocates and may be 0.
        """
        ...


class NoBlocking:
    """One block holding every row: the unblocked computation."""

    def rows_per_block(self, nrows: int, ncols: int) -> int:
        return nrows

    def __repr__(self) -> str:
        return "NoBlocking()"


class ByteBudgetBlocking:
    """Blocks of at most ``nbytes`` bytes of float64 entries.

    A block has ``max(1, nbytes // (8 * ncols))`` rows, capped at
    ``nrows``, so an array that fits the budget stays in one block. A
    consumer that allocates ``k`` temporaries of the block's size peaks
    at about ``k * nbytes``, independent of ``nrows``.

    Parameters
    ----------
    nbytes : int
        Bytes allowed for one ``rows x ncols`` block.

    Raises
    ------
    ValueError
        If ``nbytes < 1``.
    """

    def __init__(self, nbytes: int) -> None:
        if nbytes < 1:
            raise ValueError(f"nbytes must be >= 1, got {nbytes}")
        self._nbytes = nbytes

    def nbytes(self) -> int:
        """Return the byte budget of one block."""
        return self._nbytes

    def rows_per_block(self, nrows: int, ncols: int) -> int:
        row_bytes = _ITEMSIZE * max(ncols, 1)
        return min(nrows, max(1, self._nbytes // row_bytes))

    def __repr__(self) -> str:
        return f"ByteBudgetBlocking(nbytes={self._nbytes})"


class FixedRowBlocking:
    """Blocks of a constant number of rows (the last may be shorter).

    Parameters
    ----------
    nrows_per_block : int
        Rows per block.

    Raises
    ------
    ValueError
        If ``nrows_per_block < 1``.
    """

    def __init__(self, nrows_per_block: int) -> None:
        if nrows_per_block < 1:
            raise ValueError(
                f"nrows_per_block must be >= 1, got {nrows_per_block}"
            )
        self._nrows_per_block = nrows_per_block

    def rows_per_block(self, nrows: int, ncols: int) -> int:
        return min(nrows, self._nrows_per_block)

    def __repr__(self) -> str:
        return f"FixedRowBlocking(nrows_per_block={self._nrows_per_block})"


def row_block_ranges(
    blocking: RowBlockingProtocol, nrows: int, ncols: int,
) -> list[tuple[int, int]]:
    """Return the ``(start, stop)`` row ranges of the blocks, in order.

    The ranges are contiguous and cover ``range(nrows)``; every block but
    the last has the size ``blocking`` chose. ``nrows == 0`` gives no
    blocks.

    Raises
    ------
    ValueError
        If ``nrows`` or ``ncols`` is negative, or ``blocking`` returns a
        size outside ``[1, nrows]``.
    """
    if nrows < 0 or ncols < 0:
        raise ValueError(
            f"nrows and ncols must be >= 0, got nrows={nrows}, ncols={ncols}"
        )
    if nrows == 0:
        return []
    size = blocking.rows_per_block(nrows, ncols)
    if not 1 <= size <= nrows:
        raise ValueError(
            f"{blocking!r} returned {size} rows per block for nrows={nrows}; "
            "expected a value in [1, nrows]"
        )
    return [(start, min(start + size, nrows)) for start in range(0, nrows, size)]
