r"""The two state blocks with solves condensed to size :math:`n`, on any
stack.

:class:`CondensedStateBlocks` satisfies
:class:`~pyapprox.ode.second_order.reduction.StateBlocksProtocol`. The
state is one of the stack's vectors of :math:`2n` rows, as in
:class:`~pyapprox.ode.second_order.assembled_blocks.AssembledStateBlocks`,
but a block operator keeps its blocks (:class:`CondensedFormat`). The
reduction's operators have the first block row

.. math::

    A = \begin{bmatrix} c_1 I & c_2 I \\ L & R \end{bmatrix}

(the block mass :math:`c_1 = 1, c_2 = 0`, the Jacobian :math:`c_1 = 0,
c_2 = 1`, and their combinations), so the solver eliminates the
displacement and factors only the :math:`n \times n` Schur complement

.. math::

    S = R - \frac{c_2}{c_1} L Z,

with the stack's own solver, then recovers
:math:`u = (r_u - c_2 v)/c_1`. :math:`Z` zeros the columns of the
displacement rows pinned by a row replacement (there :math:`u = r_u`).
For the stage operator :math:`a\,\mathrm{blockdiag}(I, M) - sJ`,
:math:`S = aM - s g_v - (s^2/a)(f_u + g_u)`, symmetric and SPD whenever
:math:`M`, :math:`-f_u` and :math:`-g_v` are.

Requirements beyond the assembled storage, each refused when violated:

- each first-row block is zero or declared the identity;
- :math:`c_1 \ne 0` when solving (true for the mass and every implicit
  stage operator);
- a displacement row is edited only by a row replacement (an essential
  pin); zeroing one leaves :math:`A` singular;
- the format is neither summed after a row edit nor edited or summed
  once transposed.

A stack supplies its spaces, a row partition that also splits and joins
row sets, the identity on one block, and a solver at size :math:`n`.
"""

from typing import Generic, List, Optional, Protocol, Sequence, Tuple, TypeVar

from pyapprox.ode.residual_derivatives import backend_state_space
from pyapprox.ode.second_order.reduction import Block, Blocks
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendDenseOperator,
    BackendRowPartition,
    DenseFormat,
)
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    BlockAssembled,
    FormatProtocol,
    InnerProductSpaceProtocol,
    LinearSolverProtocol,
    OperatorProperties,
    RowReplacedOperator,
    RowSetProtocol,
    RowsZeroedOperator,
    ScaledSum,
    SolveInfo,
    SolverHandleBase,
    SolverHandleProtocol,
    SpaceDescriptor,
    Transposed,
    VectorSpaceProtocol,
)
from pyapprox.util.linalg.operators.solvers import require_format, require_square

V = TypeVar("V")
A = TypeVar("A")
F = TypeVar("F", bound=FormatProtocol)
W = TypeVar("W")


class StatePartitionProtocol(Protocol[V]):
    """What the condensed storage needs of a stack's row partition: the
    partition's split and join, and row sets split and joined the same
    way."""

    def space(self) -> VectorSpaceProtocol[V]: ...

    def parts(self) -> Sequence[VectorSpaceProtocol[V]]: ...

    def split(self, x: V) -> Sequence[V]: ...

    def join(self, parts: Sequence[V]) -> V: ...

    def split_rows(self, rows: RowSetProtocol[W]) -> Sequence[RowSetProtocol[V]]:
        """The rows of ``rows`` in each part, indexed within the part."""
        ...

    def join_rows(self, parts: Sequence[RowSetProtocol[W]]) -> RowSetProtocol[V]:
        """The row set of the whole space holding each part's rows."""
        ...


class _Context(Generic[V, F]):
    """What every condensed format of one storage shares."""

    def __init__(
        self,
        half: VectorSpaceProtocol[V],
        partition: StatePartitionProtocol[V],
        identity: ActionOperatorProtocol[V, F],
    ) -> None:
        self.half, self.partition, self.identity = half, partition, identity


def _sum(
    terms: Sequence[Tuple[float, Optional[ActionOperatorProtocol[V, F]]]],
    space: VectorSpaceProtocol[V],
) -> Optional[ActionOperatorProtocol[V, F]]:
    present = [(c, op) for c, op in terms if op is not None]
    if not present:
        return None
    return ScaledSum(present, space)


class CondensedFormat(Generic[V, F]):
    r"""The format of :math:`A = [c_1 I, c_2 I; L, R]` with its blocks
    kept, displacement rows possibly pinned; with ``transposed``, of
    :math:`A^\top`.

    Built by :class:`CondensedStateBlocks`; read by
    :class:`CondensedSolver`. Its combinations act on the blocks, so a
    stage operator's format is again condensed.
    """

    def __init__(
        self,
        coefficients: Tuple[float, float],
        lower: Tuple[
            Optional[ActionOperatorProtocol[V, F]],
            Optional[ActionOperatorProtocol[V, F]],
        ],
        context: _Context[V, F],
        pins: Optional[RowSetProtocol[V]] = None,
        transposed: bool = False,
    ) -> None:
        self._coefficients, self._lower = coefficients, lower
        self._context, self._pins, self._transposed = context, pins, transposed

    def coefficients(self) -> Tuple[float, float]:
        r""":math:`(c_1, c_2)`, off the pinned rows."""
        return self._coefficients

    def lower_left(self) -> Optional[ActionOperatorProtocol[V, F]]:
        """:math:`L`, or None for a zero block."""
        return self._lower[0]

    def lower_right(self) -> Optional[ActionOperatorProtocol[V, F]]:
        """:math:`R`, or None for a zero block."""
        return self._lower[1]

    def pins(self) -> Optional[RowSetProtocol[V]]:
        """The pinned displacement rows, or None."""
        return self._pins

    def is_transposed(self) -> bool:
        return self._transposed

    def _require_editable(self, what: str) -> None:
        if self._transposed:
            raise TypeError(f"a transposed condensed format takes no {what}")

    def scaled_sum(
        self,
        coefficient: float,
        others: Sequence[Tuple[float, "CondensedFormat[V, F]"]],
    ) -> "CondensedFormat[V, F]":
        """Combine block by block.

        Raises
        ------
        TypeError
            If a term is transposed or has pinned rows.
        """
        terms = [(coefficient, self), *others]
        for _, fmt in terms:
            fmt._require_editable("sum")
            if fmt._pins is not None:
                raise TypeError("a condensed format with pinned rows takes no sum")
        half = self._context.half
        c1 = sum(c * fmt._coefficients[0] for c, fmt in terms)
        c2 = sum(c * fmt._coefficients[1] for c, fmt in terms)
        return CondensedFormat(
            (c1, c2),
            (
                _sum([(c, fmt._lower[0]) for c, fmt in terms], half),
                _sum([(c, fmt._lower[1]) for c, fmt in terms], half),
            ),
            self._context,
        )

    def transposed(self) -> "CondensedFormat[V, F]":
        return CondensedFormat(
            self._coefficients,
            self._lower,
            self._context,
            self._pins,
            not self._transposed,
        )

    def _split(
        self, rows: RowSetProtocol[W]
    ) -> Tuple[RowSetProtocol[V], RowSetProtocol[V]]:
        top, bottom = self._context.partition.split_rows(rows)
        return top, bottom

    def rows_replaced(self, rows: RowSetProtocol[W]) -> "CondensedFormat[V, F]":
        """Pin displacement rows (:math:`u = r_u` there) and replace
        velocity rows of :math:`[L, R]` by identity rows.

        Raises
        ------
        TypeError
            If transposed, or displacement rows are already pinned.
        """
        self._require_editable("row edit")
        top, bottom = self._split(rows)
        pins = self._pins
        if top.count() > 0:
            if pins is not None:
                raise TypeError("the condensed format's displacement rows are pinned")
            pins = top
        lower_left, lower_right = self._lower
        if bottom.count() > 0:
            ctx = self._context
            if lower_left is not None:
                lower_left = RowsZeroedOperator(lower_left, bottom)
            base = (
                ScaledSum([(0.0, ctx.identity)], ctx.half)
                if lower_right is None
                else lower_right
            )
            lower_right = RowReplacedOperator(base, bottom)
        return CondensedFormat(
            self._coefficients, (lower_left, lower_right), self._context, pins
        )

    def rows_zeroed(self, rows: RowSetProtocol[W]) -> "CondensedFormat[V, F]":
        """Zero velocity rows of :math:`[L, R]`.

        Raises
        ------
        ValueError
            If a displacement row is among ``rows``: the operator would be
            singular, with no displacement to eliminate.
        """
        self._require_editable("row edit")
        top, bottom = self._split(rows)
        if top.count() > 0:
            raise ValueError(
                "zeroing a displacement row leaves the operator singular; the "
                "condensed storage cannot eliminate it"
            )
        lower_left, lower_right = self._lower
        if bottom.count() > 0:
            if lower_left is not None:
                lower_left = RowsZeroedOperator(lower_left, bottom)
            if lower_right is not None:
                lower_right = RowsZeroedOperator(lower_right, bottom)
        return CondensedFormat(
            self._coefficients, (lower_left, lower_right), self._context, self._pins
        )

    def block_matrix(
        self,
        blocks: Sequence[Sequence[Optional["CondensedFormat[V, F]"]]],
        sizes: Sequence[int],
    ) -> "CondensedFormat[V, F]":
        raise TypeError("condensed formats are not nested in a block operator")

    def __repr__(self) -> str:
        c1, c2 = self._coefficients
        return (
            f"{type(self).__name__}(c1={c1}, c2={c2}, "
            f"pinned={self._pins is not None}, transposed={self._transposed})"
        )


class CondensedOperator(Generic[V, F]):
    """A block operator of the reduction stored in a
    :class:`CondensedFormat`; its action and properties are its blocks'."""

    def __init__(self, action: BlockAssembled[V, F], fmt: CondensedFormat[V, F]):
        self._action, self._format = action, fmt

    def domain(self) -> SpaceDescriptor:
        return self._action.domain()

    def range(self) -> SpaceDescriptor:
        return self._action.range()

    def properties(self) -> OperatorProperties:
        return self._action.properties()

    def format(self) -> CondensedFormat[V, F]:
        return self._format

    def apply(self, x: V, out: Optional[V] = None) -> V:
        return self._action.apply(x, out=out)

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        return self._action.apply_transpose(y, out=out)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._format!r})"


class _CondensedHandle(SolverHandleBase[V], Generic[V, F]):
    """Solves with :math:`A` or :math:`A^\\top` through the Schur
    complement's handle (see the module docstring)."""

    def __init__(
        self,
        op: ActionOperatorProtocol[V, CondensedFormat[V, F]],
        fmt: CondensedFormat[V, F],
        schur: SolverHandleProtocol[V],
        context: _Context[V, F],
    ) -> None:
        super().__init__(op)
        self._op, self._format, self._schur, self._context = op, fmt, schur, context

    def _recover(self, coarse: V, exact: V) -> V:
        """``coarse`` scaled by 1/c1 off the pins, ``exact`` on them."""
        c1 = self._format.coefficients()[0]
        scaled = self._context.half.scale(1.0 / c1, coarse)
        pins = self._format.pins()
        return scaled if pins is None else pins.replace_rows(scaled, exact)

    def solve(
        self, rhs: V, tol: Optional[float] = None, x0: Optional[V] = None
    ) -> Tuple[V, SolveInfo]:
        half, fmt = self._context.half, self._format
        c1, c2 = fmt.coefficients()
        first, second = self._context.partition.split(rhs)
        lower_left, pins = fmt.lower_left(), fmt.pins()
        if not fmt.is_transposed():
            reduced = second
            if lower_left is not None:
                eliminated = lower_left.apply(self._recover(first, first))
                reduced = half.axpby(-1.0, eliminated, 1.0, second)
            velocity, info = self._schur.solve(reduced, tol=tol)
            displacement = self._recover(half.axpby(-c2, velocity, 1.0, first), first)
            return self._context.partition.join([displacement, velocity]), info
        # A^T = [D1, L^T; D2, R^T]: D2 D1^{-1} = (c2/c1) Z.
        unpinned = first if pins is None else pins.zero_rows(first)
        reduced = half.axpby(-c2 / c1, unpinned, 1.0, second)
        velocity, info = self._schur.solve(reduced, tol=tol)
        remainder = first
        if lower_left is not None:
            remainder = half.axpby(
                -1.0, lower_left.apply_transpose(velocity), 1.0, first
            )
        displacement = self._recover(remainder, remainder)
        return self._context.partition.join([displacement, velocity]), info

    def transpose(self) -> Optional["_CondensedHandle[V, F]"]:
        schur = self._schur.transpose()
        if schur is None:
            return None
        return _CondensedHandle(
            Transposed(self._op), self._format.transposed(), schur, self._context
        )


class CondensedSolver(Generic[V, F]):
    r"""Solves with a :class:`CondensedFormat` operator by factoring its
    :math:`n \times n` Schur complement with ``half_solver``.

    Parameters
    ----------
    half_solver : LinearSolverProtocol
        The stack's solver for operators of format ``F`` on one block.
    context : _Context
        The storage's spaces, partition and identity.
    """

    def __init__(
        self, half_solver: LinearSolverProtocol[V, F], context: _Context[V, F]
    ) -> None:
        self._half_solver, self._context = half_solver, context

    def setup(
        self, op: ActionOperatorProtocol[V, CondensedFormat[V, F]]
    ) -> SolverHandleProtocol[V]:
        """Factor the Schur complement.

        Raises
        ------
        TypeError
            If ``op`` is not in a condensed format.
        ValueError
            If :math:`c_1 = 0`, or the Schur complement is zero.
        """
        fmt = require_format(op, CondensedFormat, "CondensedSolver")
        require_square(op, "CondensedSolver")
        c1, c2 = fmt.coefficients()
        if c1 == 0.0:
            raise ValueError(
                "the first block row has no displacement coefficient (c1 = 0); "
                "the displacement cannot be eliminated"
            )
        lower_left, lower_right, pins = fmt.lower_left(), fmt.lower_right(), fmt.pins()
        if lower_left is not None and pins is not None and pins.count() > 0:
            # L Z: L with the pinned columns zeroed.
            lower_left = Transposed(RowsZeroedOperator(Transposed(lower_left), pins))
        terms: List[Tuple[float, Optional[ActionOperatorProtocol[V, F]]]] = [
            (1.0, lower_right),
            (-c2 / c1, lower_left if c2 != 0.0 else None),
        ]
        schur = _sum(terms, self._context.half)
        if schur is None:
            raise ValueError("the Schur complement is zero: the operator is singular")
        handle = self._half_solver.setup(schur)
        if fmt.is_transposed():
            transposed = handle.transpose()
            if transposed is None:
                raise ValueError(
                    "the Schur complement's solver has no transpose, needed for "
                    "a transposed operator"
                )
            handle = transposed
        return _CondensedHandle(op, fmt, handle, self._context)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._half_solver!r})"


def _coefficient(block: Block[V, F], position: str) -> float:
    if block is None:
        return 0.0
    if not block.properties().identity:
        raise ValueError(
            f"the condensed storage needs the {position} block zero or declared "
            "the identity"
        )
    return 1.0


class CondensedStateBlocks(Generic[V, A, F]):
    r""":math:`w = [u; v]` as one of the stack's vectors, block operators
    kept in their blocks and solved at size :math:`n` (see the module
    docstring).

    Parameters
    ----------
    half_space, space : InnerProductSpaceProtocol
        The space of one block (size :math:`n`) and of :math:`w`
        (:math:`2n`).
    partition : StatePartitionProtocol
        Splits :math:`w`, and row sets of its space, into the two blocks.
    identity : ActionOperatorProtocol
        The identity on one block, in format ``F``.
    half_solver : LinearSolverProtocol
        Solves with operators of format ``F`` on one block.

    Raises
    ------
    ValueError
        If the partition does not have two parts of ``half_space`` over
        ``space``, or the identity is not on ``half_space``.
    """

    def __init__(
        self,
        half_space: InnerProductSpaceProtocol[V, A],
        space: InnerProductSpaceProtocol[V, A],
        partition: StatePartitionProtocol[V],
        identity: ActionOperatorProtocol[V, F],
        half_solver: LinearSolverProtocol[V, F],
    ) -> None:
        half = half_space.descriptor()
        parts = partition.parts()
        if len(parts) != 2:
            raise ValueError(f"the partition must have two parts, got {len(parts)}")
        for part in parts:
            half.require_same(part.descriptor(), "part of the state partition")
        space.descriptor().require_same(
            partition.space().descriptor(), "space of the state partition"
        )
        half.require_same(identity.domain(), "domain of the block identity")
        half.require_same(identity.range(), "range of the block identity")
        self._half, self._space, self._partition = half_space, space, partition
        self._context = _Context(half_space, partition, identity)
        self._solver = CondensedSolver(half_solver, self._context)

    def half_space(self) -> InnerProductSpaceProtocol[V, A]:
        return self._half

    def space(self) -> InnerProductSpaceProtocol[V, A]:
        return self._space

    def displacement(self, state: V) -> V:
        return self._partition.split(state)[0]

    def velocity(self, state: V) -> V:
        return self._partition.split(state)[1]

    def join(self, displacement: V, velocity: V) -> V:
        return self._partition.join([displacement, velocity])

    def identity(self) -> ActionOperatorProtocol[V, F]:
        return self._context.identity

    def lift_rows(
        self, displacement: RowSetProtocol[W], velocity: RowSetProtocol[W]
    ) -> RowSetProtocol[V]:
        """The rows of :math:`w` that are ``displacement``'s rows of the
        first block and ``velocity``'s of the second."""
        return self._partition.join_rows([displacement, velocity])

    def assemble(self, blocks: Blocks[V, F]) -> CondensedOperator[V, F]:
        """The condensed operator of ``blocks``, a None block zero.

        Raises
        ------
        ValueError
            If a first-row block is neither zero nor declared the identity.
        """
        (upper_left, upper_right), (lower_left, lower_right) = blocks
        fmt = CondensedFormat(
            (
                _coefficient(upper_left, "upper-left"),
                _coefficient(upper_right, "upper-right"),
            ),
            (lower_left, lower_right),
            self._context,
        )
        return CondensedOperator(BlockAssembled(blocks, self._partition), fmt)

    def solver(self) -> CondensedSolver[V, F]:
        return self._solver

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nstates={self._half.descriptor().size})"


def backend_dense_condensed_state_blocks(
    bkd: Backend[Array],
    nstates: int,
    solver: Optional[LinearSolverProtocol[Array, DenseFormat[Array]]] = None,
) -> CondensedStateBlocks[Array, Array, DenseFormat[Array]]:
    """The condensed storage on the dense backend stack.

    Parameters
    ----------
    bkd : Backend
        The backend of the states.
    nstates : int
        :math:`n`, the size of one block.
    solver : LinearSolverProtocol, optional
        Factors the :math:`n \\times n` Schur complement; a dense LU by
        default.
    """
    half = backend_state_space(bkd, nstates)
    space = backend_state_space(bkd, 2 * nstates)
    return CondensedStateBlocks(
        half,
        space,
        BackendRowPartition(space, [half, half], bkd),
        BackendDenseOperator.identity(half.descriptor(), bkd),
        BackendDenseLU(bkd) if solver is None else solver,
    )
