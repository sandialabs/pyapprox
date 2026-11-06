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
  :math:`y_{n+1}`, :class:`AtEveryBlock` every one. A stepper describes
  each block to the rule as a :class:`StepBlock`. A rule that does not
  split into rows and a placement implements :class:`RowPinningProtocol`
  itself; the steppers need no change for it.
- **after the solve**: :class:`PinnedRowsConstraint`, a state constraint
  :math:`\Pi_t(y) = P y + (I - P)\, g(t)` the integrator applies to stored
  states.

The data may depend on the parameters, :math:`g(t; p)`; the declaration
says how through its :class:`PinnedRowsParamDerivatives` bundle, and
every consumer adds the data's derivatives on the rows :math:`R` where it
would otherwise state them zero. The transposed solves of the adjoint
need no edit: a row-replaced block's transpose already is the adjoint of
the replaced equation.
"""

from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

from pyapprox.ode.residual_derivatives import (
    STATED_ZERO,
    DerivativeOperator,
    StatedZero,
)
from pyapprox.ode.state_constraint import StateConstraintDerivatives
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    IdentityOperator,
    MatrixFree,
    OperatorProperties,
    RowReplacedOperator,
    RowSetProtocol,
    RowsZeroedOperator,
    SpaceDescriptor,
    VectorSpaceProtocol,
)

V = TypeVar("V")
W = TypeVar("W")
F = TypeVar("F", bound=FormatProtocol)

DataParamFn = Callable[[float], DerivativeOperator[V]]
r""":math:`t \mapsto \partial d/\partial p`, parameters to states, for
data :math:`d` (the values :math:`g` or their rate :math:`\dot g`); its
rows outside :math:`R` are ignored."""
DataCurvatureFn = Callable[[float, V], DerivativeOperator[V]]
r""":math:`(t, w) \mapsto \sum_i w_i\, \partial^2 d_i/\partial p^2`,
parameters to parameters, for a weight column :math:`w` that is zero
outside :math:`R`."""


@dataclass(frozen=True)
class PinnedRowsParamDerivatives(Generic[V]):
    r"""How a declaration's data depend on the parameters.

    Every field is a function, :data:`~pyapprox.ode.residual_derivatives.
    STATED_ZERO` (the data do not depend on the parameters that way), or
    None (not available: a consumer needing it refuses). Build with the
    named constructors.

    Attributes
    ----------
    values : DataParamFn, STATED_ZERO or None
        :math:`\partial g/\partial p`; also the initial state's rows
        :math:`R` at :math:`t_0`.
    rates : DataParamFn, STATED_ZERO or None
        :math:`\partial \dot g/\partial p`.
    values_curvature, rates_curvature : DataCurvatureFn, STATED_ZERO or None
        The adjoint-weighted second derivatives of :math:`g` and
        :math:`\dot g`.
    """

    values: Union[DataParamFn[V], StatedZero, None] = None
    rates: Union[DataParamFn[V], StatedZero, None] = None
    values_curvature: Union[DataCurvatureFn[V], StatedZero, None] = None
    rates_curvature: Union[DataCurvatureFn[V], StatedZero, None] = None

    @staticmethod
    def independent() -> "PinnedRowsParamDerivatives[W]":
        """Data that do not depend on the parameters: every derivative the
        stated zero."""
        return PinnedRowsParamDerivatives(
            STATED_ZERO, STATED_ZERO, STATED_ZERO, STATED_ZERO
        )

    @staticmethod
    def unavailable() -> "PinnedRowsParamDerivatives[W]":
        """Data whose parameter derivatives are not known."""
        return PinnedRowsParamDerivatives()

    @staticmethod
    def first_order(
        values: Union[DataParamFn[W], StatedZero],
        rates: Union[DataParamFn[W], StatedZero],
    ) -> "PinnedRowsParamDerivatives[W]":
        """First derivatives only: gradients, no Hessian-vector products."""
        return PinnedRowsParamDerivatives(values, rates)

    @staticmethod
    def second_order(
        values: Union[DataParamFn[W], StatedZero],
        rates: Union[DataParamFn[W], StatedZero],
        values_curvature: Union[DataCurvatureFn[W], StatedZero],
        rates_curvature: Union[DataCurvatureFn[W], StatedZero],
    ) -> "PinnedRowsParamDerivatives[W]":
        """First and second derivatives."""
        return PinnedRowsParamDerivatives(
            values, rates, values_curvature, rates_curvature
        )


class _RowsKept(Generic[V]):
    r""":math:`(I - P) A`: an operator with the rows outside :math:`R`
    zeroed; ``MatrixFree``. ``out`` is ignored."""

    def __init__(
        self,
        op: DerivativeOperator[V],
        rows: RowSetProtocol[V],
        states: VectorSpaceProtocol[V],
    ) -> None:
        self._op, self._rows, self._states = op, rows, states

    def domain(self) -> SpaceDescriptor:
        return self._op.domain()

    def range(self) -> SpaceDescriptor:
        return self._op.range()

    def properties(self) -> OperatorProperties:
        return OperatorProperties()

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(self, x: V, out: Optional[V] = None) -> V:
        y = self._op.apply(x)
        return self._states.axpby(1.0, y, -1.0, self._rows.zero_rows(y))

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        kept = self._states.axpby(1.0, y, -1.0, self._rows.zero_rows(y))
        return self._op.apply_transpose(kept)


def _present(
    field: Union[Callable[..., DerivativeOperator[V]], StatedZero, None],
    what: str,
    owner: object,
) -> Optional[Callable[..., DerivativeOperator[V]]]:
    """The field's function, None for the stated zero; refused when not
    available."""
    if isinstance(field, StatedZero):
        return None
    if field is None:
        raise ValueError(
            f"{type(owner).__name__} does not declare {what} of its data, "
            "which these derivatives need: declare it, or that the data do "
            "not depend on the parameters (PinnedRowsParamDerivatives."
            "independent())"
        )
    return field


class PinnedData(Generic[V]):
    r"""A declaration's data derivatives read on its rows :math:`R`: what
    every consumer adds there, None where the declaration states zero.

    Parameters
    ----------
    declaration : PinnedRowsProtocol
        The rows and their data.
    states : VectorSpaceProtocol
        The state space.

    Raises
    ------
    ValueError
        From a method, if the declaration does not make available the
        derivative it reads.
    """

    def __init__(
        self, declaration: "PinnedRowsProtocol[V]", states: VectorSpaceProtocol[V]
    ) -> None:
        self._declaration, self._states = declaration, states
        self._rows = declaration.rows()
        self._bundle = declaration.parameter_derivatives()

    def _kept(self, op: DerivativeOperator[V]) -> DerivativeOperator[V]:
        return _RowsKept(op, self._rows, self._states)

    def _weights(self, weights: V) -> V:
        return self._states.axpby(1.0, weights, -1.0, self._rows.zero_rows(weights))

    def values_param(self, time: float) -> Optional[DerivativeOperator[V]]:
        r""":math:`(I - P)\, \partial g/\partial p` at ``time``."""
        fn = _present(self._bundle.values, "the parameter Jacobian", self._declaration)
        return None if fn is None else self._kept(fn(time))

    def rates_param(self, time: float) -> Optional[DerivativeOperator[V]]:
        r""":math:`(I - P)\, \partial \dot g/\partial p` at ``time``."""
        fn = _present(
            self._bundle.rates, "the rate's parameter Jacobian", self._declaration
        )
        return None if fn is None else self._kept(fn(time))

    def values_curvature(
        self, time: float, weights: V
    ) -> Optional[DerivativeOperator[V]]:
        r""":math:`\sum_{i \in R} w_i\, \partial^2 g_i/\partial p^2`."""
        fn = _present(
            self._bundle.values_curvature,
            "the second parameter derivatives",
            self._declaration,
        )
        return None if fn is None else fn(time, self._weights(weights))

    def rates_curvature(
        self, time: float, weights: V
    ) -> Optional[DerivativeOperator[V]]:
        r""":math:`\sum_{i \in R} w_i\, \partial^2 \dot g_i/\partial p^2`."""
        fn = _present(
            self._bundle.rates_curvature,
            "the rate's second parameter derivatives",
            self._declaration,
        )
        return None if fn is None else fn(time, self._weights(weights))


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

    def rates(self) -> Optional[Callable[[float], V]]:
        r""":math:`t \mapsto \dot g(t)`, a single column laid out as
        :meth:`values`; None when the data's rate is not supplied. Fixed
        for the declaration's life."""
        ...

    def parameter_derivatives(self) -> PinnedRowsParamDerivatives[V]:
        """How the data depend on the parameters;
        :meth:`PinnedRowsParamDerivatives.independent` when they do not.
        Which fields are present is fixed for the declaration's life."""
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

    def data_param(
        self, time: float, space: VectorSpaceProtocol[V]
    ) -> Optional[DerivativeOperator[V]]:
        r""":math:`(I - P)\, \partial g/\partial p` at the block's time: the
        pinned rows' parameter Jacobian is its negative. None when the
        data do not depend on the parameters."""
        ...

    def data_curvature(
        self, weights: V, time: float, space: VectorSpaceProtocol[V]
    ) -> Optional[DerivativeOperator[V]]:
        r""":math:`\sum_{i \in R} w_i\, \partial^2 g_i/\partial p^2` for
        ``weights`` :math:`w`, the step adjoint's block unzeroed; the
        pinned rows' parameter curvature is its negative. None when the
        data do not depend on the parameters."""
        ...


@dataclass(frozen=True)
class StepBlock:
    """One residual block of a step, as a stepper describes it to a pin
    rule.

    A stepper builds one per block; a rule reads only the fields it needs,
    so a stepper with more to say about its blocks (a split method's
    partition, say) describes them with more fields and every existing
    rule still works.

    Attributes
    ----------
    index : int
        The block's position in the step's unknown.
    count : int
        The number of blocks; the last produces :math:`y_{n+1}`.

    Raises
    ------
    ValueError
        If ``index`` is not in ``[0, count)``.
    """

    index: int
    count: int

    def __post_init__(self) -> None:
        if not 0 <= self.index < self.count:
            raise ValueError(
                f"block {self.index} is not one of the step's {self.count} blocks"
            )

    def is_last(self) -> bool:
        """Whether this block produces :math:`y_{n+1}`."""
        return self.index == self.count - 1


@runtime_checkable
class RowPinningProtocol(Protocol[V]):
    """Which residual blocks of a step a stepper pins, and how."""

    def at(self, block: StepBlock) -> Optional[BlockPinProtocol[V]]:
        """How to pin ``block``, or None to leave it."""
        ...


@runtime_checkable
class PinPlacementProtocol(Protocol):
    """Which residual blocks of a step to pin."""

    def pins(self, block: StepBlock) -> bool:
        """Whether to pin ``block``."""
        ...


class AtStepEnd:
    """Pin only the block producing :math:`y_{n+1}`: the final block, or
    the last stage of a stiffly accurate method."""

    def pins(self, block: StepBlock) -> bool:
        return block.is_last()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AtEveryBlock:
    """Pin every block: each stage's and the step end's. Every block that
    is solved then holds the data at its own time, so no stage integrates
    the pinned rows' equations and no boundary rate is needed."""

    def pins(self, block: StepBlock) -> bool:
        return True

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

    def data_param(
        self, time: float, space: VectorSpaceProtocol[V]
    ) -> Optional[DerivativeOperator[V]]:
        return PinnedData(self._declared, space).values_param(time)

    def data_curvature(
        self, weights: V, time: float, space: VectorSpaceProtocol[V]
    ) -> Optional[DerivativeOperator[V]]:
        return PinnedData(self._declared, space).values_curvature(time, weights)

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

    def at(self, block: StepBlock) -> Optional[RowPin[V]]:
        return self._pin if self._placement.pins(block) else None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._pin!r}, {self._placement!r})"


class _PinTangent(Generic[V]):
    """The pin's tangent, the identity with the rows zeroed, the same at
    every state and time."""

    def __init__(self, op: RowsZeroedOperator[V, MatrixFree]) -> None:
        self._op = op

    def __call__(self, state: V, time: float) -> RowsZeroedOperator[V, MatrixFree]:
        return self._op


class _ConstraintParam(Generic[V]):
    r""":math:`\partial \Pi/\partial p = (I - P)\, \partial g/\partial p`."""

    def __init__(self, data: PinnedData[V]) -> None:
        self._data = data

    def __call__(self, state: V, time: float) -> DerivativeOperator[V]:
        op = self._data.values_param(time)
        if op is None:
            raise AssertionError("the data's parameter Jacobian is not zero")
        return op


class _ConstraintCurvature(Generic[V]):
    r""":math:`\lambda^\top \partial^2 \Pi/\partial p^2 = \sum_{i \in R}
    \lambda_i\, \partial^2 g_i/\partial p^2`."""

    def __init__(self, data: PinnedData[V]) -> None:
        self._data = data

    def __call__(self, state: V, time: float, adj: V) -> DerivativeOperator[V]:
        op = self._data.values_curvature(time, adj)
        if op is None:
            raise AssertionError("the data's curvature is not zero")
        return op


class PinnedRowsConstraint(Generic[V]):
    r""":math:`\Pi_t(y) = P y + (I - P)\, g(t; p)`: the declared rows of
    each stored state set to the data after the step, the others
    unchanged.

    Affine in the state: the tangent is the identity with the rows zeroed
    (so :math:`\Pi'^\top` zeros them in an adjoint), the state curvatures
    are stated zeros, and the parameter derivatives are the data's on the
    rows (stated zeros for data independent of the parameters; without
    the data's second derivatives, first order only). Its derivatives
    take stack vectors, so it is given to the integrator directly.

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
        If the row set belongs to another space, or ``param_space`` is
        given and the declaration makes its data's parameter Jacobian
        unavailable.
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
        self._derivs: StateConstraintDerivatives[V]
        if param_space is None:
            self._derivs = StateConstraintDerivatives.state_only(tangent)
            return
        # The data's parameter derivatives on the rows; the state enters
        # linearly, with no curvature.
        bundle, data = rows.parameter_derivatives(), PinnedData(rows, space)
        param: Union[_ConstraintParam[V], StatedZero] = (
            STATED_ZERO
            if _present(bundle.values, "the parameter Jacobian", rows) is None
            else _ConstraintParam(data)
        )
        if bundle.values_curvature is None:
            self._derivs = StateConstraintDerivatives.first_order(
                space, param_space, tangent, param
            )
            return
        self._derivs = StateConstraintDerivatives.second_order(
            space,
            param_space,
            tangent,
            param=param,
            state_state=STATED_ZERO,
            state_param=STATED_ZERO,
            param_param=(
                STATED_ZERO
                if isinstance(bundle.values_curvature, StatedZero)
                else _ConstraintCurvature(data)
            ),
        )

    def constrain(self, state: V, time: float) -> V:
        """``state`` with its rows :math:`R` set to :math:`g_R(t)`."""
        return self._rows.replace_rows(state, self._declared.values(time))

    def derivatives(self) -> StateConstraintDerivatives[V]:
        return self._derivs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nrows={self._rows.count()})"
