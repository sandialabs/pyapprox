r"""Rows of a step's blocks replaced by an affine relation, on any stack.

Some rows of a semi-discrete system are not evolution laws but relations
:math:`a(y, t) = A(t)\, y - b(t) = 0` on rows :math:`S` (a natural boundary
condition imposed in strong form, an algebraic constraint).
:class:`AffineRowPinning` imposes them in the blocks a placement selects,
alongside essential rows pinned by the shared
:class:`~pyapprox.ode.pinned_rows.RowPin`. In a selected block:

- ``residual``: rows :math:`S` read :math:`a(Y, t)`;
- ``own``: rows :math:`S` are the rows of :math:`A(t)`;
- ``right_hand_side``: rows :math:`S` take :math:`b(t) = -a(0, t)`, so a
  linear solve with ``own`` satisfies :math:`a(Y, t) = 0`;
- ``coupled`` and ``adjoint``: rows :math:`S` are zero, as :math:`a` reads
  only the block's own unknown;

and the essential rows as ``RowPin`` treats them. A block not selected
keeps its raw rows.

Every edit goes through the format's own algebra (row zeroing, sums, row
replacement), so the pin serves every format. They are ordered: rows
:math:`S` are zeroed and :math:`A` added first, the essential rows pinned
last, since a format may refuse a sum after a row replacement (the
condensed second-order storage does).

The relation must be affine in the state and independent of the
parameters: the pin gives the data terms no derivative of :math:`a`. Its
rows and the essential rows must be disjoint.
"""

from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.ode.pinned_rows import (
    PinnedRowsProtocol,
    PinPlacementProtocol,
    RowPin,
    StepBlock,
)
from pyapprox.ode.residual_derivatives import DerivativeOperator
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    RowSetProtocol,
    RowsZeroedOperator,
    ScaledSum,
    VectorSpaceProtocol,
)

V = TypeVar("V")
F = TypeVar("F", bound=FormatProtocol)
G = TypeVar("G", bound=FormatProtocol)
F_co = TypeVar("F_co", bound=FormatProtocol, covariant=True)


@runtime_checkable
class AffineRowsProtocol(Protocol[V, F_co]):
    r"""An affine relation :math:`a(y, t) = A(t)\, y - b(t)` on rows
    :math:`S` of the stepped state, as :class:`AffineRowPinning` reads it;
    independent of the parameters."""

    def rows(self) -> RowSetProtocol[V]:
        """The rows :math:`S`, a row set of the stepped space."""
        ...

    def residual(self, state: V, time: float) -> V:
        r""":math:`a(y, t)`, a single column of the stepped space; rows
        outside :math:`S` are ignored."""
        ...

    def jacobian(self, time: float) -> ActionOperatorProtocol[V, F_co]:
        r""":math:`A(t)` on the stepped space, in its format; rows outside
        :math:`S` are ignored."""
        ...

    def is_time_invariant(self) -> bool:
        r"""Whether :math:`A` does not change with time."""
        ...


class _AffineBlockPin(Generic[V, F]):
    """A selected block: rows S replaced by the relation, the essential
    rows pinned by ``RowPin``."""

    def __init__(self, pinning: "AffineRowPinning[V, F]") -> None:
        self._p = pinning

    def residual(
        self, block: V, state: V, time: float, space: VectorSpaceProtocol[V]
    ) -> V:
        p = self._p
        related = p._rows.replace_rows(block, p._affine.residual(state, time))
        return p._essential.residual(related, state, time, space)

    def right_hand_side(self, rhs: V, time: float) -> V:
        p = self._p
        data = p._space.scale(-1.0, p._affine.residual(p._zero, time))
        return p._essential.right_hand_side(p._rows.replace_rows(rhs, data), time)

    def own(
        self, op: ActionOperatorProtocol[V, F], time: float
    ) -> ActionOperatorProtocol[V, F]:
        p = self._p
        jacobian = p._affine.jacobian(time)
        # (I - P) A: the relation's rows of A, zero elsewhere
        rows_of_a: ScaledSum[V, F] = ScaledSum(
            [(1.0, jacobian), (-1.0, RowsZeroedOperator(jacobian, p._rows))],
            p._space,
        )
        related: ScaledSum[V, F] = ScaledSum(
            [(1.0, RowsZeroedOperator(op, p._rows)), (1.0, rows_of_a)], p._space
        )
        return p._essential.own(related, time)

    def own_is_time_invariant(self) -> bool:
        return self._p._affine.is_time_invariant()

    def coupled(self, op: ActionOperatorProtocol[V, G]) -> ActionOperatorProtocol[V, G]:
        p = self._p
        return p._essential.coupled(RowsZeroedOperator(op, p._rows))

    def adjoint(self, block: V, space: VectorSpaceProtocol[V]) -> V:
        p = self._p
        return p._essential.adjoint(p._rows.zero_rows(space.copy(block)), space)

    def data_param_jacobian(
        self,
        time: float,
        space: VectorSpaceProtocol[V],
        params: VectorSpaceProtocol[V],
    ) -> Optional[DerivativeOperator[V]]:
        # The relation does not depend on the parameters
        return self._p._essential.data_param_jacobian(time, space, params)

    def data_param_curvature(
        self,
        weights: V,
        time: float,
        space: VectorSpaceProtocol[V],
        params: VectorSpaceProtocol[V],
    ) -> Optional[DerivativeOperator[V]]:
        return self._p._essential.data_param_curvature(weights, time, space, params)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._p!r})"


class AffineRowPinning(Generic[V, F]):
    r"""Rows replaced by an affine relation, and essential rows pinned, in
    the blocks a placement selects.

    Implements :class:`~pyapprox.ode.pinned_rows.RowPinningProtocol` for
    every format.

    Parameters
    ----------
    affine : AffineRowsProtocol
        The relation and its rows.
    essential : PinnedRowsProtocol
        The essential rows and their data, disjoint from the relation's.
    space : VectorSpaceProtocol
        The stepped space.
    placement : PinPlacementProtocol
        Which blocks.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    """

    def __init__(
        self,
        affine: AffineRowsProtocol[V, F],
        essential: PinnedRowsProtocol[V],
        space: VectorSpaceProtocol[V],
        placement: PinPlacementProtocol,
    ) -> None:
        if not isinstance(affine, AffineRowsProtocol):
            raise TypeError(
                f"affine must satisfy AffineRowsProtocol, got {type(affine).__name__}"
            )
        if not isinstance(essential, PinnedRowsProtocol):
            raise TypeError(
                "essential must satisfy PinnedRowsProtocol, got "
                f"{type(essential).__name__}"
            )
        if not isinstance(placement, PinPlacementProtocol):
            raise TypeError(
                "placement must satisfy PinPlacementProtocol, got "
                f"{type(placement).__name__}"
            )
        self._affine, self._space, self._placement = affine, space, placement
        self._rows = affine.rows()
        self._essential: RowPin[V] = RowPin(essential)
        self._zero = space.zeros()
        self._pin: _AffineBlockPin[V, F] = _AffineBlockPin(self)

    def at(self, block: StepBlock) -> Optional[_AffineBlockPin[V, F]]:
        """The block's pin, or None when the placement leaves it."""
        return self._pin if self._placement.pins(block) else None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nrows={self._rows.count()}, "
            f"placement={self._placement!r})"
        )
