"""Tests for the row-blocking strategies and the block ranges they give."""

import pytest

from pyapprox.util.linalg import (
    ByteBudgetBlocking,
    FixedRowBlocking,
    NoBlocking,
    RowBlockingProtocol,
    row_block_ranges,
)


def _covers(ranges: list[tuple[int, int]], nrows: int) -> bool:
    """True if the ranges are contiguous, non-empty and cover range(nrows)."""
    stop = 0
    for start, end in ranges:
        if start != stop or end <= start:
            return False
        stop = end
    return stop == nrows


class _ReturnsZero:
    def rows_per_block(self, nrows: int, ncols: int) -> int:
        return 0


class _ReturnsTooMany:
    def rows_per_block(self, nrows: int, ncols: int) -> int:
        return nrows + 1


class TestStrategies:
    def test_satisfy_protocol(self) -> None:
        for blocking in (
            NoBlocking(), ByteBudgetBlocking(1024), FixedRowBlocking(3),
        ):
            assert isinstance(blocking, RowBlockingProtocol)

    def test_no_blocking_is_one_block(self) -> None:
        assert NoBlocking().rows_per_block(318124, 1000) == 318124
        assert row_block_ranges(NoBlocking(), 17, 5) == [(0, 17)]

    def test_byte_budget_rows(self) -> None:
        # 64 MiB of float64 at 1000 columns is 8388 rows.
        blocking = ByteBudgetBlocking(64 * 1024**2)
        assert blocking.rows_per_block(318124, 1000) == 8388
        assert blocking.nbytes() == 64 * 1024**2

    def test_byte_budget_fits_in_one_block(self) -> None:
        # An array smaller than the budget is not split.
        assert ByteBudgetBlocking(10**9).rows_per_block(500, 100) == 500

    def test_byte_budget_at_least_one_row(self) -> None:
        # A budget smaller than one row still makes progress.
        assert ByteBudgetBlocking(1).rows_per_block(50, 1000) == 1

    def test_byte_budget_zero_columns(self) -> None:
        # The first pivoted-Cholesky update has no previous columns.
        assert ByteBudgetBlocking(80).rows_per_block(50, 0) == 10

    def test_fixed_rows_capped_at_nrows(self) -> None:
        assert FixedRowBlocking(7).rows_per_block(5, 3) == 5
        assert FixedRowBlocking(7).rows_per_block(50, 3) == 7

    def test_invalid_sizes_raise(self) -> None:
        with pytest.raises(ValueError, match="nbytes"):
            ByteBudgetBlocking(0)
        with pytest.raises(ValueError, match="nrows_per_block"):
            FixedRowBlocking(0)


class TestRowBlockRanges:
    @pytest.mark.parametrize("nrows", [1, 6, 7, 20, 21])
    def test_ragged_and_exact_cover(self, nrows: int) -> None:
        ranges = row_block_ranges(FixedRowBlocking(7), nrows, 4)
        assert _covers(ranges, nrows)
        assert all(end - start == 7 for start, end in ranges[:-1])

    def test_single_row_blocks(self) -> None:
        ranges = row_block_ranges(FixedRowBlocking(1), 4, 2)
        assert ranges == [(0, 1), (1, 2), (2, 3), (3, 4)]

    def test_no_rows_gives_no_blocks(self) -> None:
        assert row_block_ranges(NoBlocking(), 0, 3) == []

    def test_negative_sizes_raise(self) -> None:
        with pytest.raises(ValueError, match="nrows and ncols"):
            row_block_ranges(NoBlocking(), -1, 3)
        with pytest.raises(ValueError, match="nrows and ncols"):
            row_block_ranges(NoBlocking(), 3, -1)

    def test_bad_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="rows per block"):
            row_block_ranges(_ReturnsZero(), 5, 2)
        with pytest.raises(ValueError, match="rows per block"):
            row_block_ranges(_ReturnsTooMany(), 5, 2)
