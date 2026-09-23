"""Tests for writing an ambient-sized basis and reading it back.

Two things are being checked and they pull in opposite directions.

A :class:`StreamingBasis` must be *indistinguishable* from an
:class:`ArrayBasis` holding the same values -- so most tests here compare
the two, and the array implementation is the oracle rather than a
hand-written expectation.

A sink must be *distinguishable* from a careless one -- so the rest check
that a partial write, an overlap or a mis-shaped block raises, since each
of those otherwise produces a basis that is merely wrong.
"""

import os
from typing import Any, Iterator, Optional, Tuple

import numpy as np
import pytest
from pyapprox.surrogates.kle.basis_operator import (
    ArrayBasis,
    BasisOperatorProtocol,
)
from pyapprox.surrogates.kle.basis_sinks import (
    ArrayBasisSink,
    BasisSinkProtocol,
    MemmapBasisSink,
    StreamingBasis,
)
from pyapprox.util.backends.protocols import Backend


def _orthonormal(bkd: Backend, nstates: int = 120, nterms: int = 5) -> Any:
    """Orthonormal, so a scaling is visible against a known norm."""
    raw = np.random.RandomState(0).standard_normal((nstates, nterms))
    return bkd.array(np.linalg.qr(raw)[0])


def _fill(sink: Any, basis: Any, block_rows: int = 37) -> Any:
    """Write ``basis`` to ``sink`` in blocks and finalize.

    A block size that does not divide ``nstates``, so the short final
    block is exercised every time rather than only when a test thinks to.
    """
    nstates = int(basis.shape[0])
    for start in range(0, nstates, block_rows):
        stop = min(start + block_rows, nstates)
        sink.write(slice(start, stop), basis[start:stop, :])
    return sink.finalize()


def _reader(basis: Any, block_rows: int = 25) -> Any:
    """A row-block reader over a resident array, and a pass counter.

    Lets a test build a :class:`StreamingBasis` directly, without going
    through a sink, when what it is checking is the reading rather than
    the writing.
    """
    passes: list = []

    def read_blocks(
        max_bytes: Optional[int] = None,
    ) -> Iterator[Tuple[slice, Any]]:
        passes.append(1)
        nstates = int(basis.shape[0])
        rows = block_rows if max_bytes is None else max(1, max_bytes)
        for start in range(0, nstates, rows):
            stop = min(start + rows, nstates)
            yield slice(start, stop), basis[start:stop, :]

    return read_blocks, passes


def _sinks(bkd: Backend, tmp_path: Any, nstates: int, nterms: int) -> Any:
    """Both sinks, so every behavioural test runs against each."""
    return [
        ("array", ArrayBasisSink(nstates, nterms, bkd)),
        (
            "memmap",
            MemmapBasisSink(
                os.path.join(str(tmp_path), "basis.dat"),
                nstates,
                nterms,
                bkd,
            ),
        ),
    ]


class TestASinkRoundTripsTheBasis:
    """What a sink is for: values in, the same values back out."""

    @pytest.mark.parametrize("kind", ["array", "memmap"])
    def test_finalize_returns_a_basis_operator(
        self, bkd: Backend, tmp_path: Any, kind: str
    ) -> None:
        """So a consumer written against the protocol accepts it."""
        basis = _orthonormal(bkd)
        sink = dict(_sinks(bkd, tmp_path, 120, 5))[kind]
        assert isinstance(sink, BasisSinkProtocol)
        assert isinstance(_fill(sink, basis), BasisOperatorProtocol)

    @pytest.mark.parametrize("kind", ["array", "memmap"])
    def test_blocks_in_reassemble_to_the_basis(
        self, bkd: Backend, tmp_path: Any, kind: str
    ) -> None:
        basis = _orthonormal(bkd)
        sink = dict(_sinks(bkd, tmp_path, 120, 5))[kind]
        bkd.assert_allclose(
            _fill(sink, basis).to_array(), basis, atol=1e-15
        )

    @pytest.mark.parametrize("kind", ["array", "memmap"])
    def test_blocks_may_arrive_out_of_order(
        self, bkd: Backend, tmp_path: Any, kind: str
    ) -> None:
        """Placed by slice, not by arrival.

        A producer reading its input in whatever order the storage makes
        cheap should not have to buffer to write in order.
        """
        basis = _orthonormal(bkd, 100, 4)
        sink = dict(_sinks(bkd, tmp_path, 100, 4))[kind]
        for start in (60, 0, 80, 20, 40):
            stop = start + 20
            sink.write(slice(start, stop), basis[start:stop, :])
        bkd.assert_allclose(sink.finalize().to_array(), basis, atol=1e-15)


class TestASinkRefusesAnIncompleteBasis:
    """Each of these otherwise yields a basis that is quietly wrong."""

    @pytest.mark.parametrize("kind", ["array", "memmap"])
    def test_an_unwritten_tail_raises_at_finalize(
        self, bkd: Backend, tmp_path: Any, kind: str
    ) -> None:
        """Zeros in the last rows fail an orthonormality check far away."""
        basis = _orthonormal(bkd, 100, 4)
        sink = dict(_sinks(bkd, tmp_path, 100, 4))[kind]
        sink.write(slice(0, 60), basis[:60, :])
        with pytest.raises(ValueError, match="60:100 were never written"):
            sink.finalize()

    @pytest.mark.parametrize("kind", ["array", "memmap"])
    def test_an_unwritten_gap_raises_at_finalize(
        self, bkd: Backend, tmp_path: Any, kind: str
    ) -> None:
        """Harder to spot than a tail, and named the same way."""
        basis = _orthonormal(bkd, 100, 4)
        sink = dict(_sinks(bkd, tmp_path, 100, 4))[kind]
        sink.write(slice(0, 20), basis[:20, :])
        sink.write(slice(40, 100), basis[40:, :])
        with pytest.raises(ValueError, match="20:40 were never written"):
            sink.finalize()

    @pytest.mark.parametrize("kind", ["array", "memmap"])
    def test_overlapping_writes_raise(
        self, bkd: Backend, tmp_path: Any, kind: str
    ) -> None:
        """Two producers writing the same rows: one silently wins."""
        basis = _orthonormal(bkd, 100, 4)
        sink = dict(_sinks(bkd, tmp_path, 100, 4))[kind]
        sink.write(slice(0, 60), basis[:60, :])
        with pytest.raises(ValueError, match="overlap"):
            sink.write(slice(50, 100), basis[50:, :])

    @pytest.mark.parametrize("kind", ["array", "memmap"])
    def test_a_block_of_the_wrong_height_raises(
        self, bkd: Backend, tmp_path: Any, kind: str
    ) -> None:
        basis = _orthonormal(bkd, 100, 4)
        sink = dict(_sinks(bkd, tmp_path, 100, 4))[kind]
        with pytest.raises(ValueError, match="expects 20"):
            sink.write(slice(0, 20), basis[:19, :])

    @pytest.mark.parametrize("kind", ["array", "memmap"])
    def test_a_block_of_the_wrong_width_raises(
        self, bkd: Backend, tmp_path: Any, kind: str
    ) -> None:
        """A basis truncated to fewer terms than the sink was sized for."""
        basis = _orthonormal(bkd, 100, 4)
        sink = dict(_sinks(bkd, tmp_path, 100, 4))[kind]
        with pytest.raises(ValueError, match="sized for 4 terms"):
            sink.write(slice(0, 20), basis[:20, :2])

    @pytest.mark.parametrize("kind", ["array", "memmap"])
    def test_writing_after_finalize_raises(
        self, bkd: Backend, tmp_path: Any, kind: str
    ) -> None:
        """The basis handed out would change under its holder."""
        basis = _orthonormal(bkd, 100, 4)
        sink = dict(_sinks(bkd, tmp_path, 100, 4))[kind]
        _fill(sink, basis, block_rows=50)
        with pytest.raises(RuntimeError, match="finalized"):
            sink.write(slice(0, 20), basis[:20, :])


class TestAStreamingBasisMatchesAnArrayBasis:
    """The array implementation as oracle, across the whole protocol.

    A streaming basis that agreed on ``to_array`` but not on a product
    would be worse than one that failed outright, since the products are
    what a consumer at scale actually calls.
    """

    @pytest.fixture
    def pair(self, bkd: Backend, tmp_path: Any) -> Any:
        """The same values, held both ways."""
        basis = _orthonormal(bkd, 120, 5)
        sink = MemmapBasisSink(
            os.path.join(str(tmp_path), "b.dat"), 120, 5, bkd
        )
        return ArrayBasis(basis, bkd), _fill(sink, basis)

    def test_dimensions_agree(self, bkd: Backend, pair: Any) -> None:
        array, streamed = pair
        assert (streamed.nstates(), streamed.nterms()) == (
            array.nstates(),
            array.nterms(),
        )
        assert (streamed.nrows(), streamed.ncols()) == (120, 5)

    def test_apply_transpose_agrees(
        self, bkd: Backend, pair: Any
    ) -> None:
        """The operation the class exists for."""
        array, streamed = pair
        fields = bkd.array(
            np.random.RandomState(1).standard_normal((120, 3))
        )
        bkd.assert_allclose(
            streamed.apply_transpose(fields),
            array.apply_transpose(fields),
            atol=1e-14,
        )

    def test_apply_agrees(self, bkd: Backend, pair: Any) -> None:
        array, streamed = pair
        coefs = bkd.array(
            np.random.RandomState(2).standard_normal((5, 3))
        )
        bkd.assert_allclose(
            streamed.apply(coefs), array.apply(coefs), atol=1e-14
        )

    def test_select_then_scale_then_contract_agrees(
        self, bkd: Backend, pair: Any
    ) -> None:
        """The composition a streaming implementation gets wrong.

        Deferring two transformations to block time is the failure mode;
        the resident implementation gets it right by accident, which is
        what makes it a usable oracle here.
        """
        array, streamed = pair
        chosen = [3, 1, 4]
        factors = bkd.array([2.0, 0.5, 4.0])
        fields = bkd.array(
            np.random.RandomState(3).standard_normal((120, 2))
        )
        bkd.assert_allclose(
            streamed.select(chosen)
            .scale(factors)
            .apply_transpose(fields),
            array.select(chosen).scale(factors).apply_transpose(fields),
            atol=1e-14,
        )

    def test_scale_then_square_agrees(
        self, bkd: Backend, pair: Any
    ) -> None:
        """``(V d)^2``, not ``V^2 d``.

        The block sees one combined recipe, so the order the caller
        asked for has to be encoded in it rather than in the sequence of
        operations applied to the block.
        """
        array, streamed = pair
        factors = bkd.array(np.arange(1.0, 6.0))
        bkd.assert_allclose(
            streamed.scale(factors).square().to_array(),
            array.scale(factors).square().to_array(),
            atol=1e-13,
        )

    def test_selecting_twice_narrows_against_current_columns(
        self, bkd: Backend, pair: Any
    ) -> None:
        """Matching ``ArrayBasis``, so neither leaks its history."""
        array, streamed = pair
        bkd.assert_allclose(
            streamed.select([4, 2, 0]).select([1]).to_array(),
            array.select([4, 2, 0]).select([1]).to_array(),
            atol=1e-15,
        )

    def test_select_restricts_a_recorded_scaling_with_it(
        self, bkd: Backend, pair: Any
    ) -> None:
        """A factor belongs to its column and follows the permutation.

        Scaling before selecting is the order that exposes this: the
        recipe holds five factors and must end up holding the three that
        go with the columns kept, in their new order.
        """
        array, streamed = pair
        factors = bkd.array(np.arange(1.0, 6.0))
        bkd.assert_allclose(
            streamed.scale(factors).select([4, 0, 2]).to_array(),
            array.scale(factors).select([4, 0, 2]).to_array(),
            atol=1e-14,
        )


class TestTheBlockSizeDoesNotChangeTheAnswer:
    """Blocking is an implementation detail and must stay one.

    A result that depended on the budget would mean an accumulation was
    order-sensitive in a way the caller cannot see or control.
    """

    @pytest.mark.parametrize("rows_per_block", [1, 7, 120])
    def test_apply_transpose_is_independent_of_the_block_size(
        self, bkd: Backend, rows_per_block: int
    ) -> None:
        basis = _orthonormal(bkd, 120, 5)
        reader, _ = _reader(basis, rows_per_block)
        fields = bkd.array(
            np.random.RandomState(4).standard_normal((120, 2))
        )
        bkd.assert_allclose(
            StreamingBasis(reader, 120, 5, bkd).apply_transpose(fields),
            ArrayBasis(basis, bkd).apply_transpose(fields),
            atol=1e-13,
        )

    def test_a_memmap_budget_below_one_row_still_progresses(
        self, bkd: Backend, tmp_path: Any
    ) -> None:
        """One row per block rather than an empty iterator.

        A budget smaller than a single row should be slow, not a silent
        no-op that reads nothing and returns zeros.
        """
        basis = _orthonormal(bkd, 30, 4)
        sink = MemmapBasisSink(
            os.path.join(str(tmp_path), "b.dat"), 30, 4, bkd, max_bytes=1
        )
        streamed = _fill(sink, basis)
        assert len(list(streamed.blocks())) == 30
        bkd.assert_allclose(streamed.to_array(), basis, atol=1e-15)


class TestTheStreamingBasisReadsLazily:
    """What distinguishes it from an array behind an interface.

    If ``select`` or ``scale`` read blocks, a consumer composing a few
    restrictions before contracting would pay a full pass for each, and
    at 25 GB that is the cost the seam exists to avoid.
    """

    def test_select_and_scale_read_nothing(self, bkd: Backend) -> None:
        basis = _orthonormal(bkd, 100, 5)
        reader, passes = _reader(basis)
        streamed = StreamingBasis(reader, 100, 5, bkd)
        streamed.select([3, 1]).scale(bkd.array([2.0, 0.5])).square()
        assert passes == []

    def test_a_composed_contraction_takes_one_pass(
        self, bkd: Backend
    ) -> None:
        """Not one per transformation."""
        basis = _orthonormal(bkd, 100, 5)
        reader, passes = _reader(basis)
        streamed = StreamingBasis(reader, 100, 5, bkd)
        fields = bkd.array(
            np.random.RandomState(5).standard_normal((100, 2))
        )
        streamed.select([3, 1]).scale(
            bkd.array([2.0, 0.5])
        ).apply_transpose(fields)
        assert len(passes) == 1


class TestTheStreamingBasisValidatesItsArguments:
    def test_rejects_a_non_callable_reader(self, bkd: Backend) -> None:
        """An array passed where the recipe was expected."""
        with pytest.raises(TypeError, match="callable"):
            StreamingBasis(_orthonormal(bkd), 120, 5, bkd)

    def test_rejects_a_column_out_of_range(self, bkd: Backend) -> None:
        """Caught at select rather than at the block that indexes with it."""
        reader, _ = _reader(_orthonormal(bkd, 100, 5))
        with pytest.raises(ValueError, match="out of range"):
            StreamingBasis(reader, 100, 5, bkd).select([7])

    def test_rejects_a_scaling_of_the_wrong_length(
        self, bkd: Backend
    ) -> None:
        """A length-one vector broadcasts and would scale every column."""
        reader, _ = _reader(_orthonormal(bkd, 100, 5))
        with pytest.raises(ValueError, match="one entry per term"):
            StreamingBasis(reader, 100, 5, bkd).scale(bkd.array([2.0]))


class TestTheSinksValidateTheirSize:
    def test_a_sink_rejects_a_non_positive_size(
        self, bkd: Backend
    ) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            ArrayBasisSink(0, 5, bkd)


class TestSelectingRows:
    """Row selection, which decoding at a subsample needs.

    Distinct from ``select``, which takes columns and returns a basis: a
    row subset is a sample of the ambient space rather than a smaller
    basis for it, so it returns an array sized by the request.
    """

    def test_matches_row_indexing(self, bkd: Backend) -> None:
        array = _orthonormal(bkd, 120, 5)
        chosen = [0, 77, 3, 1]
        bkd.assert_allclose(
            ArrayBasis(array, bkd).rows(chosen),
            array[bkd.asarray(chosen, dtype=int), :],
            atol=0.0,
        )

    def test_a_fetched_and_a_walked_basis_agree(
        self, bkd: Backend, tmp_path: Any
    ) -> None:
        """The two paths through ``StreamingBasis.rows``.

        A memmap sink supplies a fetcher that indexes directly; without
        one the blocks are walked. The reads differ enormously and the
        answer must not.
        """
        basis = _orthonormal(bkd, 120, 5)
        sink = MemmapBasisSink(
            os.path.join(str(tmp_path), "b.dat"), 120, 5, bkd
        )
        fetched = _fill(sink, basis)
        walked = StreamingBasis(
            fetched._read_blocks, 120, 5, bkd
        )
        chosen = [0, 77, 3, 1]
        expected = ArrayBasis(basis, bkd).rows(chosen)
        bkd.assert_allclose(fetched.rows(chosen), expected, atol=0.0)
        bkd.assert_allclose(walked.rows(chosen), expected, atol=0.0)

    def test_the_recipe_applies_on_both_paths(
        self, bkd: Backend, tmp_path: Any
    ) -> None:
        """Rows come back selected and scaled, however they were read.

        The failure this rules out is a fetched row carrying *stored*
        columns where a walked row carries presented ones -- same shape
        when the selection happens to be the identity, wrong otherwise.
        """
        basis = _orthonormal(bkd, 120, 5)
        sink = MemmapBasisSink(
            os.path.join(str(tmp_path), "b.dat"), 120, 5, bkd
        )
        fetched = _fill(sink, basis)
        walked = StreamingBasis(fetched._read_blocks, 120, 5, bkd)
        chosen_columns = [4, 1, 0]
        factors = bkd.array([2.0, 0.5, 3.0])
        chosen_rows = [0, 77, 3]
        expected = (
            ArrayBasis(basis, bkd)
            .select(chosen_columns)
            .scale(factors)
            .rows(chosen_rows)
        )
        bkd.assert_allclose(
            fetched.select(chosen_columns)
            .scale(factors)
            .rows(chosen_rows),
            expected,
            atol=1e-14,
        )
        bkd.assert_allclose(
            walked.select(chosen_columns)
            .scale(factors)
            .rows(chosen_rows),
            expected,
            atol=1e-14,
        )

    def test_the_order_given_is_the_order_returned(
        self, bkd: Backend
    ) -> None:
        """A caller passes the ordering its plot wants."""
        array = _orthonormal(bkd, 120, 5)
        rows = ArrayBasis(array, bkd).rows([77, 0, 3])
        bkd.assert_allclose(rows[0], array[77], atol=0.0)
        bkd.assert_allclose(rows[1], array[0], atol=0.0)

    def test_a_repeated_row_is_returned_twice(
        self, bkd: Backend, tmp_path: Any
    ) -> None:
        """Nothing here deduplicates, and a caller may have reason to."""
        basis = _orthonormal(bkd, 120, 5)
        sink = MemmapBasisSink(
            os.path.join(str(tmp_path), "b.dat"), 120, 5, bkd
        )
        streamed = _fill(sink, basis)
        rows = streamed.rows([9, 9, 9])
        assert rows.shape == (3, 5)
        bkd.assert_allclose(rows[0], rows[2], atol=0.0)

    @pytest.mark.parametrize("bad", [-1, 120])
    def test_an_out_of_range_row_raises(
        self, bkd: Backend, bad: int
    ) -> None:
        """Silently skipped, it would leave that row as initialized."""
        array = _orthonormal(bkd, 120, 5)
        with pytest.raises(ValueError, match="out of range"):
            ArrayBasis(array, bkd).rows([0, bad])


class TestFetchingRowsReadsLessThanWalking:
    """Why the row fetcher is worth an optional capability at all.

    Both paths return the same rows, so every correctness test above
    passes without a fetcher. What differs is what gets read: indexing
    a memmap touches the pages those rows fall on, while walking reads
    the basis. Asserted rather than assumed, because a fetcher silently
    dropped from a sink would leave the tests green and the reads
    proportional to the file.

    Numpy only: the torch allocator caches, so tracemalloc does not see
    tensor storage and the comparison would be vacuous.
    """

    def test_the_fetched_path_allocates_less(
        self, numpy_bkd: Backend, tmp_path: Any
    ) -> None:
        import tracemalloc

        bkd = numpy_bkd
        nstates, nterms = 40000, 10
        sink = MemmapBasisSink(
            os.path.join(str(tmp_path), "b.dat"),
            nstates,
            nterms,
            bkd,
        )
        rng = np.random.RandomState(0)
        for start in range(0, nstates, 5000):
            stop = min(start + 5000, nstates)
            sink.write(
                slice(start, stop),
                bkd.array(rng.standard_normal((stop - start, nterms))),
            )
        fetched = sink.finalize()
        walked = StreamingBasis(
            fetched._read_blocks,
            nstates,
            nterms,
            bkd,
            max_bytes=1 << 16,
        )
        chosen = sorted(
            rng.choice(nstates, 50, replace=False).tolist()
        )

        def peak(call: Any) -> int:
            tracemalloc.start()
            try:
                call()
                return int(tracemalloc.get_traced_memory()[1])
            finally:
                tracemalloc.stop()

        bkd.assert_allclose(
            fetched.rows(chosen), walked.rows(chosen), atol=0.0
        )
        assert peak(lambda: fetched.rows(chosen)) < peak(
            lambda: walked.rows(chosen)
        )
