"""Tests for delivering snapshots as row blocks.

The contract is easy to satisfy loosely and hard to satisfy exactly: a
source that dropped the last short block, or overlapped two, or yielded
them out of order would still produce blocks of a plausible shape. Each
of those failures is silent at the point it happens and wrong only much
later, in a Gram that is missing terms or has counted some twice, so the
tests here check the tiling itself rather than only the shapes.
"""

from typing import Any

import numpy as np
import pytest
from pyapprox.surrogates.kle.snapshot_sources import (
    ArraySnapshotSource,
    SnapshotSourceOperator,
    SnapshotSourceProtocol,
    as_snapshot_source,
    rows_per_block,
)
from pyapprox.util.backends.protocols import Backend
from pyapprox.util.linalg.randomized import DenseMatVecOperator


def _snapshots(bkd: Backend, nstates: int = 97, nsamples: int = 12) -> Any:
    """Deliberately not a multiple of any block size used below."""
    rng = np.random.RandomState(0)
    return bkd.array(rng.standard_normal((nstates, nsamples)))


class TestArraySnapshotSource:
    def test_satisfies_the_protocol(self, bkd: Backend) -> None:
        assert isinstance(
            ArraySnapshotSource(_snapshots(bkd), bkd),
            SnapshotSourceProtocol,
        )

    def test_reports_its_shape(self, bkd: Backend) -> None:
        source = ArraySnapshotSource(_snapshots(bkd, 97, 12), bkd)
        assert source.nstates() == 97
        assert source.nsamples() == 12

    def test_rejects_a_1d_array(self, bkd: Backend) -> None:
        """One snapshot someone forgot to give a second axis."""
        with pytest.raises(ValueError, match="2D"):
            ArraySnapshotSource(bkd.array([1.0, 2.0]), bkd)


class TestTheBlocksTileTheRows:
    """Every row once, in order, with nothing dropped or repeated.

    Checked by reassembly rather than by inspecting the slices: a
    reconstructed matrix that equals the original can only come from a
    correct tiling, and it catches an ordering error that per-block
    shape assertions would miss.
    """

    @pytest.mark.parametrize("max_bytes", [64, 1024, 1 << 20])
    def test_blocks_reassemble_the_matrix(
        self, bkd: Backend, max_bytes: int
    ) -> None:
        snapshots = _snapshots(bkd, 97, 12)
        source = ArraySnapshotSource(snapshots, bkd)
        pieces = [
            bkd.to_numpy(block)
            for _, block in source.row_blocks(max_bytes=max_bytes)
        ]
        bkd.assert_allclose(
            bkd.array(np.concatenate(pieces, axis=0)),
            snapshots,
            atol=0.0,
        )

    @pytest.mark.parametrize("max_bytes", [64, 1024, 1 << 20])
    def test_slices_are_contiguous_and_cover_everything(
        self, bkd: Backend, max_bytes: int
    ) -> None:
        """The slice must say where the block belongs, not merely how big.

        A consumer writing an ambient-sized result back out places each
        piece by this slice, so an off-by-one here corrupts the output
        rather than raising.
        """
        source = ArraySnapshotSource(_snapshots(bkd, 97, 12), bkd)
        bounds = [
            (rows.start, rows.stop)
            for rows, _ in source.row_blocks(max_bytes=max_bytes)
        ]
        assert bounds[0][0] == 0
        assert bounds[-1][1] == 97
        assert all(
            bounds[i][1] == bounds[i + 1][0]
            for i in range(len(bounds) - 1)
        )

    def test_the_slice_matches_the_block_it_accompanies(
        self, bkd: Backend
    ) -> None:
        """Otherwise a caller indexes with a slice describing another block."""
        snapshots = _snapshots(bkd, 97, 12)
        source = ArraySnapshotSource(snapshots, bkd)
        for rows, block in source.row_blocks(max_bytes=1024):
            assert block.shape[0] == rows.stop - rows.start
            bkd.assert_allclose(block, snapshots[rows, :], atol=0.0)

    def test_every_block_spans_all_samples(self, bkd: Backend) -> None:
        """The property that makes these row blocks rather than column ones.

        A block short of a column would make the Gram accumulation wrong
        in a way that still produces a symmetric matrix of the right
        shape.
        """
        source = ArraySnapshotSource(_snapshots(bkd, 97, 12), bkd)
        for _, block in source.row_blocks(max_bytes=64):
            assert block.shape[1] == 12

    def test_a_budget_below_one_row_still_yields_rows(
        self, bkd: Backend
    ) -> None:
        """A tiny budget is a slow read, not an empty iterator.

        Rounding down without a floor gives zero rows per block and an
        iterator that terminates immediately, which reads as "no data"
        rather than as a configuration error.
        """
        source = ArraySnapshotSource(_snapshots(bkd, 20, 12), bkd)
        blocks = list(source.row_blocks(max_bytes=1))
        assert len(blocks) == 20
        assert all(block.shape[0] == 1 for _, block in blocks)

    def test_the_default_budget_is_used_when_none_is_given(
        self, bkd: Backend
    ) -> None:
        """Small data is one block, so the common case costs one read."""
        source = ArraySnapshotSource(_snapshots(bkd, 97, 12), bkd)
        assert len(list(source.row_blocks())) == 1


class TestRowsPerBlock:
    def test_divides_the_budget_by_the_row_size(self) -> None:
        # 12 columns of 8 bytes is 96 per row; 1024 // 96 == 10.
        assert rows_per_block(12, np.dtype("float64"), 1024) == 10

    def test_never_returns_zero(self) -> None:
        assert rows_per_block(1000, np.dtype("float64"), 1) == 1

    def test_rejects_a_nonpositive_budget(self) -> None:
        """Silently substituting a default would hide a caller's bug."""
        with pytest.raises(ValueError, match="positive"):
            rows_per_block(12, np.dtype("float64"), 0)


class TestSnapshotSourceOperator:
    """A source used as a matrix-free operator.

    Checked against ``DenseMatVecOperator`` on the same data rather than
    against hand-computed products: the two must be interchangeable,
    which is a stronger statement than either being individually
    correct, and it is what a randomized decomposition relies on when it
    cannot tell which it holds.

    The block size is deliberately small enough to force many blocks, so
    an accumulation that forgot to sum -- or a placement that overwrote
    instead -- cannot pass.
    """

    def _pair(self, bkd: Backend, nstates: int = 97, nsamples: int = 12):
        snapshots = _snapshots(bkd, nstates, nsamples)
        return (
            SnapshotSourceOperator(
                ArraySnapshotSource(snapshots, bkd), bkd, max_bytes=800
            ),
            DenseMatVecOperator(snapshots, bkd),
        )

    def test_reports_the_matrix_shape(self, bkd: Backend) -> None:
        streaming, dense = self._pair(bkd)
        assert streaming.nrows() == dense.nrows()
        assert streaming.ncols() == dense.ncols()

    def test_apply_matches_the_dense_operator(self, bkd: Backend) -> None:
        """``S x``: the one product whose result is ambient-sized."""
        streaming, dense = self._pair(bkd)
        vecs = bkd.array(np.random.RandomState(1).standard_normal((12, 5)))
        bkd.assert_allclose(
            streaming.apply(vecs), dense.apply(vecs), atol=1e-12
        )

    def test_apply_transpose_matches_the_dense_operator(
        self, bkd: Backend
    ) -> None:
        """``S^T y``: a sum over rows, so the result stays small."""
        streaming, dense = self._pair(bkd)
        vecs = bkd.array(np.random.RandomState(2).standard_normal((97, 5)))
        bkd.assert_allclose(
            streaming.apply_transpose(vecs),
            dense.apply_transpose(vecs),
            atol=1e-12,
        )

    def test_right_apply_matches_the_dense_operator(
        self, bkd: Backend
    ) -> None:
        """``y S``: also a sum over rows."""
        streaming, dense = self._pair(bkd)
        vecs = bkd.array(np.random.RandomState(3).standard_normal((5, 97)))
        bkd.assert_allclose(
            streaming.right_apply(vecs),
            dense.right_apply(vecs),
            atol=1e-12,
        )

    def test_right_apply_is_advertised(self, bkd: Backend) -> None:
        """Consumers ask before using it, and it is available here."""
        streaming, _ = self._pair(bkd)
        assert streaming.right_apply_implemented()

    def test_the_block_size_does_not_change_the_answer(
        self, bkd: Backend
    ) -> None:
        """Different tilings of the same sum, so they must agree.

        A block size that changed the result would mean an accumulation
        was dropping or repeating a term, which no single-block-size
        test detects.
        """
        snapshots = _snapshots(bkd, 97, 12)
        source = ArraySnapshotSource(snapshots, bkd)
        vecs = bkd.array(np.random.RandomState(4).standard_normal((97, 3)))
        results = [
            SnapshotSourceOperator(
                source, bkd, max_bytes=size
            ).apply_transpose(vecs)
            for size in (96, 800, 1 << 20)
        ]
        for other in results[1:]:
            bkd.assert_allclose(results[0], other, atol=1e-12)


class TestTheAdapter:
    def test_wraps_a_bare_array(self, bkd: Backend) -> None:
        assert isinstance(
            as_snapshot_source(_snapshots(bkd), bkd),
            SnapshotSourceProtocol,
        )

    def test_passes_a_source_through_unchanged(self, bkd: Backend) -> None:
        """Wrapping one twice would bury a file-backed source in an array."""
        source = ArraySnapshotSource(_snapshots(bkd), bkd)
        assert as_snapshot_source(source, bkd) is source
