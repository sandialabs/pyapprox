r"""Block vectors, block operators and block lower-triangular solves.

A *block vector* is a tuple of vectors, one per component space, each a
block of the same ``ncols`` columns; the *product space* of the
components combines them component by component, so it satisfies
:class:`~.protocols.VectorSpaceProtocol` over block vectors and generic
code uses it as any other space. Nothing here looks inside a component:
every operation goes through the component spaces and operators, so the
components may belong to any stack.

A *block operator* :math:`A = [A_{ij}]` maps a product space to a product
space. It is known by its blocks' actions (a missing block is zero), so
it is matrix-free; whoever builds it declares its properties.

A block lower-triangular operator, :math:`A_{ij} = 0` for :math:`j > i`,
is solved by forward substitution,

.. math::

    x_i = A_{ii}^{-1}\Bigl(b_i - \sum_{j<i} A_{ij} x_j\Bigr),

with one solver handle per diagonal block, and its transpose by backward
substitution,

.. math::

    x_i = A_{ii}^{-\top}\Bigl(b_i - \sum_{j>i} A_{ji}^\top x_j\Bigr).
"""

from dataclasses import dataclass
from typing import Generic, Optional, Sequence, TypeVar

from pyapprox.util.linalg.operators.descriptor import SpaceDescriptor
from pyapprox.util.linalg.operators.formats import MatrixFree
from pyapprox.util.linalg.operators.protocols import (
    LinearMapProtocol,
    OperatorProperties,
    VectorSpaceProtocol,
)
from pyapprox.util.linalg.operators.solvers import (
    SolveInfo,
    SolverHandleBase,
    SolverHandleProtocol,
)

V = TypeVar("V")


@dataclass(frozen=True)
class BlockVector(Generic[V]):
    """A tuple of vectors, one per component space of a product space.

    Parameters
    ----------
    blocks : tuple of V
        The components, in the product space's order; each is a block of
        the same number of columns.
    """

    blocks: tuple[V, ...]


def _optional_component(out: Optional[BlockVector[V]], i: int) -> Optional[V]:
    return None if out is None else out.blocks[i]


class BlockVectorSpace(Generic[V]):
    r"""The product of component spaces, over :class:`BlockVector`.

    Every operation acts component by component through the component
    spaces, passing ``out`` hints through; the aliasing rules are the
    components'. The descriptor's size is the sum of the components'.

    Parameters
    ----------
    spaces : sequence of VectorSpaceProtocol
        Nonempty; the component spaces, all of one dtype.
    name : str
        The product space's name.

    Raises
    ------
    ValueError
        If ``spaces`` is empty or the components' dtypes differ. From any
        operation, if a block vector has the wrong number of components
        or components with different numbers of columns.
    """

    def __init__(self, spaces: Sequence[VectorSpaceProtocol[V]], name: str) -> None:
        if not spaces:
            raise ValueError("a product space needs at least one component")
        self._spaces = tuple(spaces)
        dtypes = {space.descriptor().dtype for space in self._spaces}
        if len(dtypes) != 1:
            raise ValueError(
                f"the component spaces must share one dtype, got {sorted(dtypes)}"
            )
        self._descriptor = SpaceDescriptor(
            name,
            sum(space.descriptor().size for space in self._spaces),
            dtypes.pop(),
        )

    def descriptor(self) -> SpaceDescriptor:
        return self._descriptor

    def nblocks(self) -> int:
        """The number of component spaces."""
        return len(self._spaces)

    def component(self, i: int) -> VectorSpaceProtocol[V]:
        """The ``i``-th component space."""
        return self._spaces[i]

    def _check(self, x: BlockVector[V], name: str) -> int:
        """Check ``x``'s components and return their common ``ncols``."""
        if len(x.blocks) != len(self._spaces):
            raise ValueError(
                f"{name} has {len(x.blocks)} components, the space {len(self._spaces)}"
            )
        counts = {space.ncols(b) for space, b in zip(self._spaces, x.blocks)}
        if len(counts) != 1:
            raise ValueError(
                f"the components of {name} have different numbers of columns: "
                f"{sorted(counts)}"
            )
        return counts.pop()

    def zeros(self, ncols: int = 1) -> BlockVector[V]:
        return BlockVector(tuple(space.zeros(ncols) for space in self._spaces))

    def zeros_like(self, x: BlockVector[V]) -> BlockVector[V]:
        return self.zeros(self._check(x, "x"))

    def ncols(self, x: BlockVector[V]) -> int:
        return self._check(x, "x")

    def column(self, x: BlockVector[V], k: int) -> BlockVector[V]:
        self._check(x, "x")
        return BlockVector(
            tuple(space.column(b, k) for space, b in zip(self._spaces, x.blocks))
        )

    def concatenate(self, blocks: Sequence[BlockVector[V]]) -> BlockVector[V]:
        for block in blocks:
            self._check(block, "block")
        return BlockVector(
            tuple(
                space.concatenate([block.blocks[i] for block in blocks])
                for i, space in enumerate(self._spaces)
            )
        )

    def copy(
        self, x: BlockVector[V], out: Optional[BlockVector[V]] = None
    ) -> BlockVector[V]:
        self._check(x, "x")
        return BlockVector(
            tuple(
                space.copy(b, out=_optional_component(out, i))
                for i, (space, b) in enumerate(zip(self._spaces, x.blocks))
            )
        )

    def scale(
        self, a: float, x: BlockVector[V], out: Optional[BlockVector[V]] = None
    ) -> BlockVector[V]:
        self._check(x, "x")
        return BlockVector(
            tuple(
                space.scale(a, b, out=_optional_component(out, i))
                for i, (space, b) in enumerate(zip(self._spaces, x.blocks))
            )
        )

    def axpby(
        self,
        a: float,
        x: BlockVector[V],
        b: float,
        y: BlockVector[V],
        out: Optional[BlockVector[V]] = None,
    ) -> BlockVector[V]:
        if self._check(x, "x") != self._check(y, "y"):
            raise ValueError("x and y must have the same number of columns")
        return BlockVector(
            tuple(
                space.axpby(a, xb, b, yb, out=_optional_component(out, i))
                for i, (space, xb, yb) in enumerate(
                    zip(self._spaces, x.blocks, y.blocks)
                )
            )
        )

    def divide_entries(
        self,
        x: BlockVector[V],
        d: BlockVector[V],
        out: Optional[BlockVector[V]] = None,
    ) -> BlockVector[V]:
        self._check(x, "x")
        self._check(d, "d")
        return BlockVector(
            tuple(
                space.divide_entries(xb, db, out=_optional_component(out, i))
                for i, (space, xb, db) in enumerate(
                    zip(self._spaces, x.blocks, d.blocks)
                )
            )
        )

    def all_finite(self, x: BlockVector[V]) -> bool:
        self._check(x, "x")
        return all(space.all_finite(b) for space, b in zip(self._spaces, x.blocks))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._descriptor!r})"


class BlockOperator(Generic[V]):
    r"""The operator :math:`[A_{ij}]` between two product spaces.

    :math:`(A x)_i = \sum_j A_{ij} x_j` and
    :math:`(A^\top y)_j = \sum_i A_{ij}^\top y_i`; a ``None`` block is zero.
    Matrix-free.

    Parameters
    ----------
    blocks : sequence of sequences of operators or None
        One row per component of ``range_space``, each with one entry per
        component of ``domain_space``; block :math:`(i, j)` maps the
        ``j``-th domain component to the ``i``-th range component.
    domain_space, range_space : BlockVectorSpace
        The product spaces.
    properties : OperatorProperties, optional
        Declared by the caller; nothing is inferred from the blocks.

    Raises
    ------
    ValueError
        If the grid's shape does not match the spaces, or a block's
        spaces do not match its position.
    """

    def __init__(
        self,
        blocks: Sequence[Sequence[Optional[LinearMapProtocol[V]]]],
        domain_space: BlockVectorSpace[V],
        range_space: BlockVectorSpace[V],
        properties: Optional[OperatorProperties] = None,
    ) -> None:
        if len(blocks) != range_space.nblocks():
            raise ValueError(
                f"blocks has {len(blocks)} rows, the range space "
                f"{range_space.nblocks()} components"
            )
        for i, row in enumerate(blocks):
            if len(row) != domain_space.nblocks():
                raise ValueError(
                    f"row {i} of blocks has {len(row)} entries, the domain "
                    f"space {domain_space.nblocks()} components"
                )
            for j, block in enumerate(row):
                if block is None:
                    continue
                domain_space.component(j).descriptor().require_same(
                    block.domain(), f"domain of block ({i}, {j})"
                )
                range_space.component(i).descriptor().require_same(
                    block.range(), f"range of block ({i}, {j})"
                )
        self._blocks = tuple(tuple(row) for row in blocks)
        self._domain_space = domain_space
        self._range_space = range_space
        self._properties = OperatorProperties() if properties is None else properties

    def domain(self) -> SpaceDescriptor:
        return self._domain_space.descriptor()

    def range(self) -> SpaceDescriptor:
        return self._range_space.descriptor()

    def domain_space(self) -> BlockVectorSpace[V]:
        """The product space of the input of :meth:`apply`."""
        return self._domain_space

    def range_space(self) -> BlockVectorSpace[V]:
        """The product space of the output of :meth:`apply`."""
        return self._range_space

    def block(self, i: int, j: int) -> Optional[LinearMapProtocol[V]]:
        """Block :math:`(i, j)`, or ``None`` for a zero block."""
        return self._blocks[i][j]

    def is_lower_triangular(self) -> bool:
        """Whether every block above the diagonal is ``None``."""
        return all(
            block is None
            for i, row in enumerate(self._blocks)
            for j, block in enumerate(row)
            if j > i
        )

    def properties(self) -> OperatorProperties:
        return self._properties

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(
        self, x: BlockVector[V], out: Optional[BlockVector[V]] = None
    ) -> BlockVector[V]:
        ncols = self._domain_space.ncols(x)
        return BlockVector(
            tuple(
                _accumulate(
                    self._range_space.component(i),
                    [
                        (block, x.blocks[j], False)
                        for j, block in enumerate(row)
                        if block is not None
                    ],
                    ncols,
                    _optional_component(out, i),
                )
                for i, row in enumerate(self._blocks)
            )
        )

    def apply_transpose(
        self, y: BlockVector[V], out: Optional[BlockVector[V]] = None
    ) -> BlockVector[V]:
        ncols = self._range_space.ncols(y)
        return BlockVector(
            tuple(
                _accumulate(
                    self._domain_space.component(j),
                    [
                        (row[j], y.blocks[i], True)
                        for i, row in enumerate(self._blocks)
                        if row[j] is not None
                    ],
                    ncols,
                    _optional_component(out, j),
                )
                for j in range(self._domain_space.nblocks())
            )
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self._range_space.nblocks()} x "
            f"{self._domain_space.nblocks()} blocks)"
        )


def _accumulate(
    space: VectorSpaceProtocol[V],
    terms: Sequence[tuple[Optional[LinearMapProtocol[V]], V, bool]],
    ncols: int,
    out: Optional[V],
) -> V:
    """The sum of the terms' actions (or transposed actions) in ``space``;
    zeros when there are none."""
    total: Optional[V] = None
    for op, x, transposed in terms:
        if op is None:
            continue
        term = op.apply_transpose(x) if transposed else op.apply(x)
        total = term if total is None else space.axpby(1.0, term, 1.0, total, out=total)
    if total is None:
        return space.zeros(ncols)
    return space.copy(total, out=out) if out is not None else total


class BlockLowerTriangularHandle(SolverHandleBase[BlockVector[V]]):
    r"""A solver handle for a block lower-triangular operator.

    Solves by forward substitution with one handle per diagonal block;
    its :meth:`transpose` solves with :math:`A^\top` by backward
    substitution with the diagonal handles' transposes. ``tol`` is passed
    to every diagonal solve and ``x0``'s components serve as their initial
    guesses. A failed diagonal solve stops the substitution: the report
    names the block, and the remaining components are zero.

    Parameters
    ----------
    op : BlockOperator
        Block lower triangular, square block by block.
    diagonal_handles : sequence of SolverHandleProtocol
        One per diagonal block, set up for it (their transposes when
        ``transposed``).
    transposed : bool, default False
        Solve with :math:`A^\top` instead; :meth:`transpose` builds this.

    Raises
    ------
    ValueError
        If ``op`` is not block lower triangular, a diagonal block is
        missing or not square, or the handles do not match the diagonal
        blocks.
    """

    def __init__(
        self,
        op: BlockOperator[V],
        diagonal_handles: Sequence[SolverHandleProtocol[V]],
        transposed: bool = False,
    ) -> None:
        super().__init__(op)
        nblocks = op.range_space().nblocks()
        if op.domain_space().nblocks() != nblocks:
            raise ValueError("a block triangular operator needs square block structure")
        if not transposed and not op.is_lower_triangular():
            raise ValueError("the operator has a nonzero block above the diagonal")
        if len(diagonal_handles) != nblocks:
            raise ValueError(
                f"got {len(diagonal_handles)} diagonal handles for {nblocks} blocks"
            )
        for i, handle in enumerate(diagonal_handles):
            block = op.block(i, i)
            if block is None:
                raise ValueError(f"diagonal block {i} is missing")
            block.domain().require_same(block.range(), f"range of diagonal block {i}")
            block.range().require_same(
                handle.domain(), f"domain of the handle for diagonal block {i}"
            )
        self._op = op
        self._handles = tuple(diagonal_handles)
        self._transposed = transposed

    def domain(self) -> SpaceDescriptor:
        """The space of right-hand sides: the operator's range, or its
        domain for the transposed solve."""
        return self._op.domain() if self._transposed else self._op.range()

    def range(self) -> SpaceDescriptor:
        """The space of solutions."""
        return self._op.range() if self._transposed else self._op.domain()

    def _order(self) -> list[int]:
        """Forward substitution order, or backward for the transpose."""
        order = list(range(len(self._handles)))
        return order[::-1] if self._transposed else order

    def _coupling(
        self, i: int, x: list[Optional[V]]
    ) -> list[tuple[Optional[LinearMapProtocol[V]], V, bool]]:
        """The solved components' contributions to row ``i``."""
        if self._transposed:
            pairs = [(self._op.block(j, i), x[j]) for j in range(i + 1, len(x))]
        else:
            pairs = [(self._op.block(i, j), x[j]) for j in range(i)]
        return [
            (block, xj, self._transposed)
            for block, xj in pairs
            if block is not None and xj is not None
        ]

    def solve(
        self,
        rhs: BlockVector[V],
        tol: Optional[float] = None,
        x0: Optional[BlockVector[V]] = None,
    ) -> tuple[BlockVector[V], SolveInfo]:
        space = (
            self._op.range_space() if not self._transposed else self._op.domain_space()
        )
        ncols = space.ncols(rhs)
        x: list[Optional[V]] = [None] * len(self._handles)
        iterations = 0
        norms: list[float] = []
        for i in self._order():
            component = space.component(i)
            coupling = _accumulate(component, self._coupling(i, x), ncols, None)
            residual = component.axpby(1.0, rhs.blocks[i], -1.0, coupling)
            x[i], info = self._handles[i].solve(
                residual, tol=tol, x0=_optional_component(x0, i)
            )
            iterations += info.iterations
            if info.residual_norm is not None:
                norms.append(info.residual_norm)
            if not info.converged:
                solution = BlockVector(
                    tuple(
                        space.component(k).zeros(ncols) if xk is None else xk
                        for k, xk in enumerate(x)
                    )
                )
                return solution, SolveInfo(
                    False, iterations, None, f"block {i}: {info.message}"
                )
        solved = tuple(xk for xk in x if xk is not None)
        return BlockVector(solved), SolveInfo(
            True, iterations, max(norms) if norms else None
        )

    def transpose(self) -> Optional["BlockLowerTriangularHandle[V]"]:
        handles: list[SolverHandleProtocol[V]] = []
        for handle in self._handles:
            transposed = handle.transpose()
            if transposed is None:
                return None
            handles.append(transposed)
        return BlockLowerTriangularHandle(self._op, handles, not self._transposed)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nblocks={len(self._handles)})"
