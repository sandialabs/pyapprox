"""Block-diagonal operators, their format and the batched LU, on both
backends.

References are the dense block-diagonal matrices of the same blocks
(``scipy.linalg.block_diag``), so every check compares the blockwise
computation with an independent dense one.
"""

import numpy as np
import pytest
import scipy.linalg
from numpy.typing import NDArray

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_block_diagonal import (
    BackendBlockDiagonalOperator,
    BlockDiagonalFormat,
    BlockDiagonalLU,
)
from pyapprox.util.linalg.backend_dense import (
    BackendDenseOperator,
    BackendRowSet,
    BackendVectorSpace,
)
from pyapprox.util.linalg.operators import (
    DiagonalCapableProtocol,
    OperatorProperties,
    RowReplacedOperator,
    RowsZeroedOperator,
    ScaledSum,
    SolverHandleProtocol,
    SpaceDescriptor,
    Transposed,
)
from pyapprox.util.linalg.scipy_stack import NumpyRowSet

_K, _M = 3, 2
_U = SpaceDescriptor("u", _K * _M)
_ROWS = [1, 4]  # one row in block 0, one in block 2


def _blocks(seed: int) -> NDArray[np.float64]:
    """Well-conditioned nonsymmetric blocks."""
    rng = np.random.default_rng(seed)
    blocks: NDArray[np.float64] = rng.normal(size=(_K, _M, _M)) + 3.0 * np.eye(_M)
    return blocks


def _dense(blocks: NDArray[np.float64]) -> NDArray[np.float64]:
    dense: NDArray[np.float64] = scipy.linalg.block_diag(*blocks)
    return dense


def _operator(
    blocks: NDArray[np.float64], bkd: Backend[Array]
) -> BackendBlockDiagonalOperator[Array]:
    return BackendBlockDiagonalOperator(bkd.asarray(blocks), _U, bkd)


def _vec(seed: int, bkd: Backend[Array]) -> Array:
    return bkd.asarray(np.random.default_rng(seed).normal(size=(_K * _M, 2)))


class TestOperator:
    def test_actions_and_diagonal(self, bkd: Backend[Array]) -> None:
        blocks = _blocks(0)
        op = _operator(blocks, bkd)
        assert isinstance(op, DiagonalCapableProtocol)
        assert (op.nblocks(), op.block_size()) == (_K, _M)
        dense, x = bkd.asarray(_dense(blocks)), _vec(1, bkd)
        bkd.assert_allclose(op.apply(x), bkd.dot(dense, x), rtol=1e-14)
        bkd.assert_allclose(op.apply_transpose(x), bkd.dot(dense.T, x), rtol=1e-14)
        bkd.assert_allclose(
            op.diagonal(), bkd.reshape(bkd.diag(dense), (_K * _M, 1)), rtol=1e-14
        )

    def test_verifies_declared_properties(self, bkd: Backend[Array]) -> None:
        symmetric = _blocks(2) + np.transpose(_blocks(2), (0, 2, 1))
        BackendBlockDiagonalOperator(
            bkd.asarray(symmetric), _U, bkd, OperatorProperties(symmetric=True)
        )
        with pytest.raises(ValueError, match="symmetric"):
            BackendBlockDiagonalOperator(
                bkd.asarray(_blocks(2)), _U, bkd, OperatorProperties(symmetric=True)
            )
        identity = np.repeat(np.eye(_M)[None], _K, axis=0)
        BackendBlockDiagonalOperator(
            bkd.asarray(identity),
            _U,
            bkd,
            OperatorProperties(symmetric=True, spd=True, diagonal=True, identity=True),
        )
        with pytest.raises(ValueError, match="diagonal"):
            BackendBlockDiagonalOperator(
                bkd.asarray(_blocks(3)), _U, bkd, OperatorProperties(diagonal=True)
            )

    def test_refuses_wrong_shapes(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match=r"\(k, m, m\)"):
            BackendBlockDiagonalOperator(bkd.zeros((_K, _M, _M + 1)), _U, bkd)
        with pytest.raises(ValueError, match="cover"):
            BackendBlockDiagonalOperator(bkd.zeros((_K + 1, _M, _M)), _U, bkd)
        op = _operator(_blocks(4), bkd)
        with pytest.raises(ValueError, match="shape"):
            op.apply(bkd.zeros((_K * _M,)))


class TestFormat:
    def test_combinators_stay_block_diagonal(self, bkd: Backend[Array]) -> None:
        a, b = _blocks(5), _blocks(6)
        space = BackendVectorSpace(_U, bkd)
        total = ScaledSum([(2.0, _operator(a, bkd)), (-0.5, _operator(b, bkd))], space)
        fmt = total.format()
        assert isinstance(fmt, BlockDiagonalFormat)
        bkd.assert_allclose(fmt.blocks(), bkd.asarray(2.0 * a - 0.5 * b), rtol=1e-14)
        transposed = Transposed(_operator(a, bkd)).format()
        bkd.assert_allclose(
            transposed.blocks(), bkd.asarray(np.transpose(a, (0, 2, 1))), rtol=1e-14
        )

    def test_row_edits_match_the_dense_ones(self, bkd: Backend[Array]) -> None:
        """Replaced rows become rows of the identity; zeroed rows zero."""
        blocks = _blocks(7)
        rows = BackendRowSet(_U, _ROWS, bkd)
        dense_op = BackendDenseOperator(bkd.asarray(_dense(blocks)), _U, bkd)
        block_op = _operator(blocks, bkd)
        for edit in (RowReplacedOperator, RowsZeroedOperator):
            block_fmt = edit(block_op, rows).format()
            assert isinstance(block_fmt, BlockDiagonalFormat)
            expected = edit(dense_op, rows).format().matrix()
            assembled = BackendBlockDiagonalOperator(block_fmt.blocks(), _U, bkd)
            identity = bkd.eye(_K * _M, dtype=bkd.default_dtype())
            bkd.assert_allclose(assembled.apply(identity), expected, rtol=1e-14)

    def test_refuses_another_stacks_row_set_and_mismatched_sums(
        self, bkd: Backend[Array]
    ) -> None:
        fmt = _operator(_blocks(8), bkd).format()
        with pytest.raises(TypeError, match="BackendRowSet"):
            fmt.rows_zeroed(NumpyRowSet(_U, [0]))  # type: ignore[misc]
        other = BlockDiagonalFormat(bkd.zeros((_K * _M, 1, 1)), bkd)
        with pytest.raises(ValueError, match="same blocks"):
            fmt.scaled_sum(1.0, [(1.0, other)])


class TestBlockDiagonalLU:
    def test_solve_and_transpose(self, bkd: Backend[Array]) -> None:
        blocks = _blocks(9)
        handle = BlockDiagonalLU(bkd).setup(_operator(blocks, bkd))
        assert isinstance(handle, SolverHandleProtocol)
        dense, b = bkd.asarray(_dense(blocks)), _vec(10, bkd)
        x, info = handle.solve(b)
        assert info.converged
        bkd.assert_allclose(bkd.dot(dense, x), b, rtol=1e-12)
        transposed = handle.transpose()
        assert transposed is not None
        xt, info = transposed.solve(b)
        assert info.converged
        bkd.assert_allclose(bkd.dot(dense.T, xt), b, rtol=1e-12)
        bkd.assert_allclose(handle.apply(b), x, rtol=1e-14)

    def test_singular_block_reports_without_raising(self, bkd: Backend[Array]) -> None:
        blocks = _blocks(11)
        blocks[1] = 0.0
        handle = BlockDiagonalLU(bkd).setup(_operator(blocks, bkd))
        _, info = handle.solve(_vec(12, bkd))
        assert not info.converged

    def test_refuses_dense_storage(self, bkd: Backend[Array]) -> None:
        dense = BackendDenseOperator(bkd.asarray(_dense(_blocks(13))), _U, bkd)
        with pytest.raises(TypeError, match="BlockDiagonalFormat"):
            BlockDiagonalLU(bkd).setup(dense)  # type: ignore[arg-type]
