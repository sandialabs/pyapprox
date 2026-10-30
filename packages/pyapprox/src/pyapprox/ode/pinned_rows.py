r"""Rows of a state pinned to boundary data, inside a step or after it.

A discretization declares the rows :math:`R` to pin and their data
:math:`g(t)` once (:class:`PinnedRowsProtocol`). Two consumers apply the
declaration at different points of a step:

- **inside the solve**: a stepper given a :class:`RowPinningProtocol`
  replaces rows of its own equations, block by block. In a pinned block
  the rows :math:`R` read :math:`Y_R - g_R(t)` at the block's time, its
  own diagonal block is :math:`P A + (I - P)` (identity rows,
  :class:`~pyapprox.util.linalg.operators.RowReplacedOperator`), its
  couplings to other unknowns, to the stored states and to the
  parameters lose those rows (:math:`P A`,
  :class:`~pyapprox.util.linalg.operators.RowsZeroedOperator`), and the
  step adjoint's rows :math:`R` there carry no curvature. The other rows
  are solved with the data, so for implicit methods this is a different
  scheme from pinning after the solve. :class:`RowPinning` pins the
  blocks a placement selects: :class:`AtStepEnd` the one producing
  :math:`y_{n+1}`. A rule that does not split into rows and a placement
  implements :class:`RowPinningProtocol` itself; the steppers need no
  change for it.
- **after the solve**: :class:`PinnedRowsConstraint`, a state constraint
  :math:`\Pi_t(y) = P y + (I - P)\, g(t)` the integrator applies to stored
  states.

The data must not depend on the parameters. The transposed solves of the
adjoint need no edit: a row-replaced block's transpose already is the
adjoint of the replaced equation.
"""

from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.ode.residual_derivatives import STATED_ZERO
from pyapprox.ode.state_constraint import StateConstraintDerivatives
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    IdentityOperator,
    MatrixFree,
    RowReplacedOperator,
    RowSetProtocol,
    RowsZeroedOperator,
    VectorSpaceProtocol,
)

V = TypeVar("V")
F = TypeVar("F", bound=FormatProtocol)


@runtime_checkable
class PinnedRowsProtocol(Protocol[V]):
    """The rows of a state pinned to boundary data, and the data."""

    def rows(self) -> RowSetProtocol[V]:
        """The rows :math:`R`, a row set of the state space."""
        ...

    def values(self, time: float) -> V:
        r""":math:`g(t)`: a single column of the state space whose rows
        :math:`R` hold the data at ``time``; its other rows are ignored."""
        ...


@runtime_checkable
class BlockPinProtocol(Protocol[V]):
    """How one residual block of a step is pinned (see the module
    docstring); ``time`` is the block's time."""

    def residual(
        self, block: V, state: V, time: float, space: VectorSpaceProtocol[V]
    ) -> V:
        r"""A new residual block, its pinned rows reading :math:`Y_R -
        g_R(t)` for ``state`` the block's unknown :math:`Y`."""
        ...

    def right_hand_side(self, rhs: V, time: float) -> V:
        """A new right-hand side for a linear solve with a row-replaced
        operator: ``rhs`` with its pinned rows the data."""
        ...

    def own(self, op: ActionOperatorProtocol[V, F]) -> ActionOperatorProtocol[V, F]:
        """The block's own diagonal block, its pinned rows identity rows."""
        ...

    def coupled(self, op: ActionOperatorProtocol[V, F]) -> ActionOperatorProtocol[V, F]:
        """A coupling block of the block's row, its pinned rows zeroed."""
        ...

    def adjoint(self, block: V, space: VectorSpaceProtocol[V]) -> V:
        """A new copy of the step adjoint's block with its pinned rows
        zeroed, for the second-derivative contractions."""
        ...


@runtime_checkable
class RowPinningProtocol(Protocol[V]):
    """Which residual blocks of a step a stepper pins, and how."""

    def at(self, block: int, nblocks: int) -> Optional[BlockPinProtocol[V]]:
        """How to pin residual block ``block`` of ``nblocks`` (the last
        produces :math:`y_{n+1}`), or None to leave it."""
        ...


@runtime_checkable
class PinPlacementProtocol(Protocol):
    """Which residual blocks of a step to pin."""

    def pins(self, block: int, nblocks: int) -> bool:
        """Whether to pin block ``block`` of ``nblocks``; the last produces
        :math:`y_{n+1}`."""
        ...


class AtStepEnd:
    """Pin only the block producing :math:`y_{n+1}`: the final block, or
    the last stage of a stiffly accurate method."""

    def pins(self, block: int, nblocks: int) -> bool:
        return block == nblocks - 1

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RowPin(Generic[V]):
    """Pins the declared rows of a residual block.

    Parameters
    ----------
    rows : PinnedRowsProtocol
        The rows and their data.
    """

    def __init__(self, rows: PinnedRowsProtocol[V]) -> None:
        self._declared, self._rows = rows, rows.rows()

    def residual(
        self, block: V, state: V, time: float, space: VectorSpaceProtocol[V]
    ) -> V:
        pinned = space.axpby(1.0, state, -1.0, self._declared.values(time))
        return self._rows.replace_rows(block, pinned)

    def right_hand_side(self, rhs: V, time: float) -> V:
        return self._rows.replace_rows(rhs, self._declared.values(time))

    def own(self, op: ActionOperatorProtocol[V, F]) -> RowReplacedOperator[V, F]:
        return RowReplacedOperator(op, self._rows)

    def coupled(self, op: ActionOperatorProtocol[V, F]) -> RowsZeroedOperator[V, F]:
        return RowsZeroedOperator(op, self._rows)

    def adjoint(self, block: V, space: VectorSpaceProtocol[V]) -> V:
        return self._rows.zero_rows(space.copy(block))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nrows={self._rows.count()})"


class RowPinning(Generic[V]):
    """Pins the declared rows in the blocks a placement selects.

    Parameters
    ----------
    rows : PinnedRowsProtocol
        The rows and their data, the same in every pinned block.
    placement : PinPlacementProtocol, optional
        Which blocks; :class:`AtStepEnd` by default.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    """

    def __init__(
        self,
        rows: PinnedRowsProtocol[V],
        placement: Optional[PinPlacementProtocol] = None,
    ) -> None:
        if not isinstance(rows, PinnedRowsProtocol):
            raise TypeError(
                f"rows must satisfy PinnedRowsProtocol, got {type(rows).__name__}"
            )
        chosen: PinPlacementProtocol = AtStepEnd() if placement is None else placement
        if not isinstance(chosen, PinPlacementProtocol):
            raise TypeError(
                "placement must satisfy PinPlacementProtocol, got "
                f"{type(chosen).__name__}"
            )
        self._pin, self._placement = RowPin(rows), chosen

    def at(self, block: int, nblocks: int) -> Optional[RowPin[V]]:
        return self._pin if self._placement.pins(block, nblocks) else None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._pin!r}, {self._placement!r})"


class _PinTangent(Generic[V]):
    """The pin's tangent, the identity with the rows zeroed, the same at
    every state and time."""

    def __init__(self, op: RowsZeroedOperator[V, MatrixFree]) -> None:
        self._op = op

    def __call__(self, state: V, time: float) -> RowsZeroedOperator[V, MatrixFree]:
        return self._op


class PinnedRowsConstraint(Generic[V]):
    r""":math:`\Pi_t(y) = P y + (I - P)\, g(t)`: the declared rows of each
    stored state set to the data after the step, the others unchanged.

    Affine in the state: the tangent is the identity with the rows zeroed
    (so :math:`\Pi'^\top` zeros them in an adjoint), and the parameter and
    second derivatives are stated zeros. Its derivatives take stack
    vectors, so it is given to the integrator directly.

    Parameters
    ----------
    rows : PinnedRowsProtocol
        The rows and their data.
    space : VectorSpaceProtocol
        The state space.
    param_space : VectorSpaceProtocol, optional
        The parameter space, when parameter derivatives are needed.

    Raises
    ------
    TypeError
        If ``rows`` does not satisfy its protocol.
    ValueError
        If the row set belongs to another space.
    """

    def __init__(
        self,
        rows: PinnedRowsProtocol[V],
        space: VectorSpaceProtocol[V],
        param_space: Optional[VectorSpaceProtocol[V]] = None,
    ) -> None:
        if not isinstance(rows, PinnedRowsProtocol):
            raise TypeError(
                f"rows must satisfy PinnedRowsProtocol, got {type(rows).__name__}"
            )
        row_set = rows.rows()
        space.descriptor().require_same(row_set.descriptor(), "the pinned rows")
        self._declared, self._rows = rows, row_set
        tangent = _PinTangent(RowsZeroedOperator(IdentityOperator(space), row_set))
        self._derivs: StateConstraintDerivatives[V] = (
            StateConstraintDerivatives.state_only(tangent)
            if param_space is None
            else StateConstraintDerivatives.second_order(
                space,
                param_space,
                tangent,
                param=STATED_ZERO,
                state_state=STATED_ZERO,
                state_param=STATED_ZERO,
                param_param=STATED_ZERO,
            )
        )

    def constrain(self, state: V, time: float) -> V:
        """``state`` with its rows :math:`R` set to :math:`g_R(t)`."""
        return self._rows.replace_rows(state, self._declared.values(time))

    def derivatives(self) -> StateConstraintDerivatives[V]:
        return self._derivs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nrows={self._rows.count()})"
