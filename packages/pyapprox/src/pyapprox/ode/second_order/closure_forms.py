r"""Relations on the displacement imposed in a second-order system.

A relation on the displacement, :math:`c(u, t) = C(t)\, u - h(t) = 0` on
rows :math:`R` (a traction or Robin condition imposed in strong form),
replaces a block row of the reduced system :math:`w = [u; v]`. The
displacement rows stay :math:`\dot u = v`: editing them would break the
first block row the condensed storage eliminates. So a closure form
places the relation on the velocity rows :math:`R` of the second block,
as an :class:`~pyapprox.ode.affine_rows.AffineRowsProtocol` on :math:`w`
that :class:`~pyapprox.ode.affine_rows.AffineRowPinning` imposes.

:class:`DirectClosure` imposes :math:`c(u, t) = 0` itself: the stored
displacement satisfies the relation exactly; the velocity on :math:`R`
is then fixed only through :math:`\dot u = v`. A form imposing the
relation's rate instead, :math:`C v - \dot h = 0`, is another
implementation of the same protocol.
"""

from typing import Generic, Protocol, TypeVar, runtime_checkable

from pyapprox.ode.second_order.reduction import Blocks
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    RowSetProtocol,
    VectorSpaceProtocol,
)

V = TypeVar("V")
F = TypeVar("F", bound=FormatProtocol)
G = TypeVar("G", bound=FormatProtocol)
F_co = TypeVar("F_co", bound=FormatProtocol, covariant=True)
F_contra = TypeVar("F_contra", bound=FormatProtocol, contravariant=True)
G_co = TypeVar("G_co", bound=FormatProtocol, covariant=True)


@runtime_checkable
class DisplacementConstraintProtocol(Protocol[V, F_co]):
    r""":math:`c(u, t) = C(t)\, u - h(t)` on rows :math:`R` of one block,
    affine in the displacement and independent of the parameters."""

    def rows(self) -> RowSetProtocol[V]:
        """The rows :math:`R`, a row set of one block."""
        ...

    def residual(self, displacement: V, time: float) -> V:
        r""":math:`c(u, t)`, a single column of one block; rows outside
        :math:`R` are ignored."""
        ...

    def jacobian(self, time: float) -> ActionOperatorProtocol[V, F_co]:
        r""":math:`C(t)` on one block, in format ``F``; rows outside
        :math:`R` are ignored."""
        ...

    def is_time_invariant(self) -> bool:
        r"""Whether :math:`C` does not change with time."""
        ...


@runtime_checkable
class ClosureBlocksProtocol(Protocol[V, F_contra, G_co]):
    """What a closure form needs of a storage of :math:`w = [u; v]`; both
    storages of :mod:`pyapprox.ode.second_order` satisfy it."""

    def half_space(self) -> VectorSpaceProtocol[V]:
        """The space of one block."""
        ...

    def displacement(self, state: V) -> V:
        """The first block of ``state``."""
        ...

    def join(self, displacement: V, velocity: V) -> V:
        """The vector with blocks ``displacement`` and ``velocity``."""
        ...

    def lift_rows(
        self, displacement: RowSetProtocol[V], velocity: RowSetProtocol[V]
    ) -> RowSetProtocol[V]:
        """The rows of :math:`w` that are ``displacement``'s rows of the
        first block and ``velocity``'s of the second."""
        ...

    def assemble(self, blocks: Blocks[V, F_contra]) -> ActionOperatorProtocol[V, G_co]:
        """The operator with these blocks, a None block zero."""
        ...


class DirectClosure(Generic[V, F, G]):
    r""":math:`c(u, t) = 0` on the velocity rows :math:`R`.

    Satisfies :class:`~pyapprox.ode.affine_rows.AffineRowsProtocol` on
    :math:`w`: rows :math:`R` of the second block, residual :math:`[\,\cdot\,;
    c(u, t)]`, Jacobian :math:`[0, 0; C, 0]` assembled by the storage.

    Parameters
    ----------
    constraint : DisplacementConstraintProtocol
        The relation on the displacement.
    blocks : ClosureBlocksProtocol
        The storage of :math:`w`, whose block space is the constraint's.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    """

    def __init__(
        self,
        constraint: DisplacementConstraintProtocol[V, F],
        blocks: ClosureBlocksProtocol[V, F, G],
    ) -> None:
        if not isinstance(constraint, DisplacementConstraintProtocol):
            raise TypeError(
                "constraint must satisfy DisplacementConstraintProtocol, got "
                f"{type(constraint).__name__}"
            )
        if not isinstance(blocks, ClosureBlocksProtocol):
            raise TypeError(
                "blocks must satisfy ClosureBlocksProtocol, got "
                f"{type(blocks).__name__}"
            )
        self._constraint, self._blocks = constraint, blocks
        base = constraint.rows()
        self._rows = blocks.lift_rows(base.difference(base), base)
        self._zero = blocks.half_space().zeros()

    def rows(self) -> RowSetProtocol[V]:
        r"""Rows :math:`R` of the second block, a row set of :math:`w`."""
        return self._rows

    def residual(self, state: V, time: float) -> V:
        r""":math:`c(u, t)` on the velocity rows :math:`R`, for
        :math:`u` the displacement of ``state``."""
        blocks = self._blocks
        relation = self._constraint.residual(blocks.displacement(state), time)
        return blocks.join(self._zero, relation)

    def jacobian(self, time: float) -> ActionOperatorProtocol[V, G]:
        r""":math:`[0, 0; C(t), 0]`, in the storage's format."""
        return self._blocks.assemble(
            ((None, None), (self._constraint.jacobian(time), None))
        )

    def is_time_invariant(self) -> bool:
        return self._constraint.is_time_invariant()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._constraint!r})"
