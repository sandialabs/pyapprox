r"""Block vectors, block operators and block lower-triangular solves, on
every stack.

Components of sizes 3 and 2 (and 3, 2, 4 for the three-block solves).
Every result is compared with the same computation on the dense matrix
assembled from the blocks: the product space's operations component by
component, the block operator's action and transpose (a zero block
included), forward substitution against a dense solve with the assembled
lower-triangular matrix, and backward substitution against a dense solve
with its transpose. On the mock stack the components cannot be looked
inside, so the block layer is shown to use only the component spaces and
operators.
"""

from typing import Optional, Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    BlockInnerProductSpace,
    BlockLowerTriangularHandle,
    BlockOperator,
    BlockVector,
    BlockVectorSpace,
    InnerProductSpaceProtocol,
    LinearMapProtocol,
    MatrixFree,
    SolveInfo,
    SolverHandleBase,
    SolverHandleProtocol,
    SpaceDescriptor,
    SpaceMismatchError,
    VectorSpaceProtocol,
)
from tests._helpers.linalg.stacks import (
    A,
    F,
    TypedStackCase,
    V,
    random_matrix,
    spd_matrix,
)

_U1 = SpaceDescriptor("u1", 3)
_U2 = SpaceDescriptor("u2", 2)
_U3 = SpaceDescriptor("u3", 4)
_Np = NDArray[np.float64]


def _product(
    case: TypedStackCase[V, F, A], descriptors: Sequence[SpaceDescriptor]
) -> BlockVectorSpace[V]:
    return BlockVectorSpace([case.space(d) for d in descriptors], "product")


def _block_vec(
    case: TypedStackCase[V, F, A],
    descriptors: Sequence[SpaceDescriptor],
    ncols: int,
    seed: int,
) -> BlockVector[V]:
    return BlockVector(
        tuple(case.vec(d, ncols, seed + k) for k, d in enumerate(descriptors))
    )


def _stacked(case: TypedStackCase[V, F, A], x: BlockVector[V]) -> A:
    """The components of ``x`` stacked into one dense array."""
    return case.bkd.vstack([case.array(b) for b in x.blocks])


def _assembled(grid: Sequence[Sequence[Optional[_Np]]], sizes: Sequence[int]) -> _Np:
    """The dense matrix of a grid of dense blocks (``None`` is zero)."""
    rows = [
        np.hstack(
            [
                np.zeros((sizes[i], sizes[j])) if block is None else block
                for j, block in enumerate(row)
            ]
        )
        for i, row in enumerate(grid)
    ]
    return np.vstack(rows)


def _operator(
    case: TypedStackCase[V, F, A],
    grid: Sequence[Sequence[Optional[_Np]]],
    descriptors: Sequence[SpaceDescriptor],
) -> BlockOperator[V]:
    space = _product(case, descriptors)
    blocks: list[list[Optional[LinearMapProtocol[V]]]] = [
        [
            None if block is None else case.build(block, descriptors[j], descriptors[i])
            for j, block in enumerate(row)
        ]
        for i, row in enumerate(grid)
    ]
    return BlockOperator(blocks, space, space)


def _lower_grid(descriptors: Sequence[SpaceDescriptor]) -> list[list[Optional[_Np]]]:
    """A block lower-triangular grid with invertible diagonal blocks and
    one zero block below the diagonal when there are three blocks."""
    sizes = [d.size for d in descriptors]
    grid: list[list[Optional[_Np]]] = []
    for i, ni in enumerate(sizes):
        row: list[Optional[_Np]] = []
        for j, nj in enumerate(sizes):
            if j > i or (i == 2 and j == 0):
                row.append(None)
            elif i == j:
                row.append(spd_matrix(ni, 10 + i) + 0.3 * random_matrix(ni, ni, 20 + i))
            else:
                row.append(random_matrix(ni, nj, 30 + 3 * i + j))
        grid.append(row)
    return grid


def _handles(
    case: TypedStackCase[V, F, A],
    grid: Sequence[Sequence[Optional[_Np]]],
    descriptors: Sequence[SpaceDescriptor],
) -> list[SolverHandleProtocol[V]]:
    handles: list[SolverHandleProtocol[V]] = []
    for i, d in enumerate(descriptors):
        block = grid[i][i]
        assert block is not None
        handles.append(case.direct_solver().setup(case.build(block, d)))
    return handles


class TestBlockVectorSpace:
    def test_is_a_space_of_the_summed_size(self, case: TypedStackCase[V, F, A]) -> None:
        space = _product(case, [_U1, _U2])
        assert isinstance(space, VectorSpaceProtocol)
        assert space.descriptor() == SpaceDescriptor("product", 5)
        assert space.nblocks() == 2
        assert space.component(1).descriptor() == _U2

    def test_operations_act_component_by_component(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        space = _product(case, [_U1, _U2])
        x = _block_vec(case, [_U1, _U2], 2, 0)
        y = _block_vec(case, [_U1, _U2], 2, 5)
        bkd = case.bkd
        xa, ya = _stacked(case, x), _stacked(case, y)
        assert space.ncols(x) == 2
        case.bkd.assert_allclose(
            _stacked(case, space.axpby(2.0, x, -0.5, y)), 2.0 * xa - 0.5 * ya
        )
        bkd.assert_allclose(_stacked(case, space.scale(3.0, x)), 3.0 * xa)
        bkd.assert_allclose(_stacked(case, space.copy(x)), xa, rtol=0.0)
        bkd.assert_allclose(_stacked(case, space.column(x, 1)), xa[:, 1:2], rtol=0.0)
        bkd.assert_allclose(
            _stacked(case, space.concatenate([x, y])), bkd.hstack([xa, ya]), rtol=0.0
        )
        bkd.assert_allclose(_stacked(case, space.zeros_like(x)), 0.0 * xa)
        d = space.axpby(1.0, space.column(y, 0), 0.0, space.column(y, 0))
        bkd.assert_allclose(
            _stacked(case, space.divide_entries(x, d)), xa / _stacked(case, d)
        )
        assert space.all_finite(x)

    def test_inner_product_sums_the_components(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        """The block dot and norm equal those of the stacked arrays."""
        space = BlockInnerProductSpace([case.space(_U1), case.space(_U2)], "product")
        assert isinstance(space, InnerProductSpaceProtocol)
        x = _block_vec(case, [_U1, _U2], 2, 0)
        y = _block_vec(case, [_U1, _U2], 3, 5)
        bkd = case.bkd
        xa, ya = _stacked(case, x), _stacked(case, y)
        bkd.assert_allclose(space.dot(x, y), bkd.dot(xa.T, ya), rtol=1e-12)
        bkd.assert_allclose(
            space.norm(x), bkd.sqrt(bkd.sum(xa * xa, axis=0, keepdims=True)), rtol=1e-12
        )
        assert space.component(1).descriptor() == _U2

    def test_refusals(self, case: TypedStackCase[V, F, A]) -> None:
        space = _product(case, [_U1, _U2])
        with pytest.raises(ValueError, match="at least one component"):
            BlockVectorSpace([], "empty")
        with pytest.raises(ValueError, match="1 components, the space 2"):
            space.ncols(BlockVector((case.vec(_U1, 1, 0),)))
        mismatched = BlockVector((case.vec(_U1, 1, 0), case.vec(_U2, 2, 1)))
        with pytest.raises(ValueError, match="different numbers of columns"):
            space.ncols(mismatched)


class TestBlockOperator:
    def test_acts_as_the_assembled_matrix(self, case: TypedStackCase[V, F, A]) -> None:
        """A 2 x 2 grid with a zero block above the diagonal: the action and
        the transpose equal the assembled matrix's."""
        descriptors = [_U1, _U2]
        grid: list[list[Optional[_Np]]] = [
            [random_matrix(3, 3, 1), None],
            [random_matrix(2, 3, 2), random_matrix(2, 2, 3)],
        ]
        op = _operator(case, grid, descriptors)
        assert isinstance(op, ActionOperatorProtocol)
        assert isinstance(op.format(), MatrixFree)
        assert op.is_lower_triangular()
        dense = case.dense(_assembled(grid, [3, 2]))
        x = _block_vec(case, descriptors, 2, 7)
        case.bkd.assert_allclose(
            _stacked(case, op.apply(x)), case.bkd.dot(dense, _stacked(case, x))
        )
        case.bkd.assert_allclose(
            _stacked(case, op.apply_transpose(x)),
            case.bkd.dot(dense.T, _stacked(case, x)),
        )

    def test_a_zero_row_gives_zeros(self, case: TypedStackCase[V, F, A]) -> None:
        grid: list[list[Optional[_Np]]] = [[random_matrix(3, 3, 1), None], [None, None]]
        op = _operator(case, grid, [_U1, _U2])
        x = _block_vec(case, [_U1, _U2], 1, 0)
        case.assert_vector(op.apply(x).blocks[1], case.dense(np.zeros((2, 1))))
        assert not BlockOperator(
            [[None, case.build(random_matrix(3, 2, 4), _U2, _U1)], [None, None]],
            _product(case, [_U1, _U2]),
            _product(case, [_U1, _U2]),
        ).is_lower_triangular()

    def test_refuses_a_grid_that_does_not_match(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        space = _product(case, [_U1, _U2])
        block = case.build(random_matrix(3, 3, 1), _U1)
        with pytest.raises(ValueError, match="1 rows, the range space 2"):
            BlockOperator([[block, None]], space, space)
        with pytest.raises(ValueError, match="row 0 of blocks has 1 entries"):
            BlockOperator([[block], [None, None]], space, space)
        with pytest.raises(SpaceMismatchError, match=r"block \(1, 1\)"):
            BlockOperator([[block, None], [None, block]], space, space)


class _FailingHandle(SolverHandleBase[V]):
    """A handle whose solve never converges and that has no transpose."""

    def solve(
        self, rhs: V, tol: Optional[float] = None, x0: Optional[V] = None
    ) -> tuple[V, SolveInfo]:
        return rhs, SolveInfo(False, message="planted failure")

    def transpose(self) -> None:
        return None


class TestBlockLowerTriangularHandle:
    @pytest.mark.parametrize("nblocks", [2, 3])
    def test_forward_and_backward_substitution(
        self, case: TypedStackCase[V, F, A], nblocks: int
    ) -> None:
        descriptors = [_U1, _U2, _U3][:nblocks]
        sizes = [d.size for d in descriptors]
        grid = _lower_grid(descriptors)
        op = _operator(case, grid, descriptors)
        handle = BlockLowerTriangularHandle(op, _handles(case, grid, descriptors))
        assert isinstance(handle, SolverHandleProtocol)
        dense = _assembled(grid, sizes)
        rhs = _block_vec(case, descriptors, 2, 40)
        rhs_dense = np.asarray(case.bkd.to_numpy(_stacked(case, rhs)))
        solution, info = handle.solve(rhs)
        assert info.converged
        case.bkd.assert_allclose(
            _stacked(case, solution),
            case.dense(np.linalg.solve(dense, rhs_dense)),
            rtol=1e-10,
        )
        transposed = handle.transpose()
        assert transposed is not None
        case.bkd.assert_allclose(
            _stacked(case, transposed.apply(rhs)),
            case.dense(np.linalg.solve(dense.T, rhs_dense)),
            rtol=1e-10,
        )
        case.bkd.assert_allclose(
            _stacked(case, handle.apply_transpose(rhs)),
            _stacked(case, transposed.apply(rhs)),
            rtol=0.0,
        )
        assert (transposed.domain(), transposed.range()) == (op.domain(), op.range())
        assert (handle.domain(), handle.range()) == (op.range(), op.domain())

    def test_a_failed_block_is_named(self, case: TypedStackCase[V, F, A]) -> None:
        descriptors = [_U1, _U2]
        grid = _lower_grid(descriptors)
        op = _operator(case, grid, descriptors)
        handles = _handles(case, grid, descriptors)
        block = op.block(1, 1)
        assert block is not None
        handle = BlockLowerTriangularHandle(op, [handles[0], _FailingHandle(block)])
        _, info = handle.solve(_block_vec(case, descriptors, 1, 0))
        assert not info.converged
        assert info.message == "block 1: planted failure"
        assert handle.transpose() is None

    def test_refusals(self, case: TypedStackCase[V, F, A]) -> None:
        descriptors = [_U1, _U2]
        grid = _lower_grid(descriptors)
        handles = _handles(case, grid, descriptors)
        upper = [row[:] for row in grid]
        upper[0][1] = random_matrix(3, 2, 9)
        with pytest.raises(ValueError, match="above the diagonal"):
            BlockLowerTriangularHandle(_operator(case, upper, descriptors), handles)
        missing = [row[:] for row in grid]
        missing[1][1] = None
        with pytest.raises(ValueError, match="diagonal block 1 is missing"):
            BlockLowerTriangularHandle(_operator(case, missing, descriptors), handles)
        with pytest.raises(ValueError, match="1 diagonal handles for 2 blocks"):
            BlockLowerTriangularHandle(_operator(case, grid, descriptors), handles[:1])
