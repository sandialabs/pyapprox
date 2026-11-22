r"""Collocation's closure rows imposed at the end of each step.

On a natural or periodic row the collocation equation is a relation
:math:`c(y, t) = 0` (Robin: :math:`\alpha y_i + \beta (\partial_n y)_i -
g_i(t)`), not an evolution law. :class:`ClosurePinning` imposes it on the
blocks a placement selects, by default the block producing
:math:`y_{n+1}`; the essential rows of those blocks are pinned to their data
by the shared :class:`~pyapprox.ode.pinned_rows.RowPin`. In a selected
block:

- ``residual``: the closure rows read :math:`c(Y, t)`;
- ``own``: the closure rows are the rows of :math:`c_u` at the block's
  time;
- ``right_hand_side``: the closure rows take :math:`-c(0, t)`, so a
  linear solve with ``own`` satisfies :math:`c(Y, t) = 0`;
- ``coupled`` and ``adjoint``: the closure rows are zero, as :math:`c`
  sees only the block's own unknown and is affine.

A block not selected keeps the raw collocation row: the PDE collocated at
the boundary node, a consistent evolution law for a smooth solution. A
stage therefore advances the boundary values by the PDE and the step end
enforces the relation, so the stored states satisfy it exactly.

The block pin is given no state, and its data terms no closure
derivative, so the closure rows must be affine in the state and
independent of the parameters; others are refused at construction. The
edit inserts rows of a dense matrix, so the pin is for dense systems.
"""

from typing import Generic, Optional, TypeVar

from pyapprox.implicitfunction.pinned_rows import RowReplacement
from pyapprox.ode.pinned_rows import (
    AtStepEnd,
    PinPlacementProtocol,
    RowPin,
    StepBlock,
)
from pyapprox.ode.residual_derivatives import (
    DerivativeOperator,
    StatedZero,
    backend_state_space,
)
from pyapprox.pde.collocation.boundary.closure_rows import ClosureRowsProtocol
from pyapprox.pde.models.collocation.closed_system import (
    CollocationSystem,
    NoEssentialRows,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.backend_dense import (
    BackendDenseOperator,
    BackendRowSet,
    BackendVectorSpace,
    DenseFormat,
)
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    RowsZeroedOperator,
    ScaledSum,
    VectorSpaceProtocol,
)

G = TypeVar("G", bound=FormatProtocol)


def _refuse_unpinnable(
    k: int, closure: ClosureRowsProtocol[Array], parameterized: bool
) -> None:
    """Refuse closure rows a block pin cannot represent: nonlinear in the
    state, or depending on the parameters."""
    derivatives = closure.derivatives()
    name = f"closure {k} ({type(closure).__name__})"
    if not isinstance(derivatives.state_curvature, StatedZero):
        raise ValueError(
            f"{name} is not declared affine in the state; a step-end closure "
            "replaces the block's rows with c_u, given no state"
        )
    param = derivatives.param_jacobian
    if isinstance(param, StatedZero):
        return
    if param is not None:
        raise ValueError(
            f"{name} holds parameters of its own; a step-end closure supplies "
            "no parameter derivative for its rows"
        )
    if parameterized:
        raise ValueError(
            f"{name} reads a physics coefficient, so with parameters its rows "
            "depend on them; a step-end closure supplies no parameter "
            "derivative for its rows"
        )


class _ClosureBlockPin(Generic[Array]):
    """A selected block: the essential rows pinned by ``RowPin``, the
    closure rows replaced by :math:`c`."""

    def __init__(self, pinning: "ClosurePinning[Array]") -> None:
        self._p = pinning

    def residual(
        self, block: Array, state: Array, time: float, space: VectorSpaceProtocol[Array]
    ) -> Array:
        p = self._p
        pinned = p._essential.residual(block, state, time, space)
        return p._with_closure_rows(pinned, p._closure_residual(state[:, 0], time))

    def right_hand_side(self, rhs: Array, time: float) -> Array:
        p = self._p
        pinned = p._essential.right_hand_side(rhs, time)
        return p._with_closure_rows(pinned, -p._closure_residual(p._zero, time))

    def own(
        self, op: ActionOperatorProtocol[Array, DenseFormat[Array]], time: float
    ) -> ActionOperatorProtocol[Array, DenseFormat[Array]]:
        p = self._p
        pinned = RowsZeroedOperator(p._essential.own(op, time), p._closure_rows)
        rows = BackendDenseOperator(
            p._closure_jacobian(time), p._space.descriptor(), p._bkd
        )
        return ScaledSum([(1.0, pinned), (1.0, rows)], p._space)

    def own_is_time_invariant(self) -> bool:
        return self._p._own_time_invariant

    def coupled(
        self, op: ActionOperatorProtocol[Array, G]
    ) -> RowsZeroedOperator[Array, G]:
        return self._p._replacement.coupled(op)

    def adjoint(self, block: Array, space: VectorSpaceProtocol[Array]) -> Array:
        return self._p._replacement.adjoint(block, space)

    def data_param_jacobian(
        self,
        time: float,
        space: VectorSpaceProtocol[Array],
        params: VectorSpaceProtocol[Array],
    ) -> Optional[DerivativeOperator[Array]]:
        # The closure rows do not depend on the parameters
        return self._p._essential.data_param_jacobian(time, space, params)

    def data_param_curvature(
        self,
        weights: Array,
        time: float,
        space: VectorSpaceProtocol[Array],
        params: VectorSpaceProtocol[Array],
    ) -> Optional[DerivativeOperator[Array]]:
        return self._p._essential.data_param_curvature(weights, time, space, params)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._p!r})"


class ClosurePinning(Generic[Array]):
    r"""The essential rows pinned and the closure rows imposed in the
    blocks a placement selects.

    Implements :class:`~pyapprox.ode.pinned_rows.RowPinningProtocol` for
    dense systems.

    Parameters
    ----------
    system : CollocationSystem
        Supplies the essential rows and the closure declarations, with
        shared rows already resolved (essential rows beat closure rows).
    placement : PinPlacementProtocol, optional
        Which blocks; :class:`~pyapprox.ode.pinned_rows.AtStepEnd` by
        default.
    parameters : ParameterState, optional
        The system's parameters, if any; with them, closure rows reading a
        physics coefficient are refused.

    Raises
    ------
    TypeError
        If ``system`` is not a ``CollocationSystem`` or ``placement`` does
        not satisfy ``PinPlacementProtocol``.
    ValueError
        If a closure is not declared affine in the state, holds
        parameters of its own, or reads a physics coefficient while the
        system has parameters.
    """

    def __init__(
        self,
        system: CollocationSystem[Array],
        placement: Optional[PinPlacementProtocol] = None,
        parameters: Optional[ParameterState[Array]] = None,
    ) -> None:
        if not isinstance(system, CollocationSystem):
            raise TypeError(
                f"system must be a CollocationSystem, got {type(system).__name__}"
            )
        chosen: PinPlacementProtocol = AtStepEnd() if placement is None else placement
        if not isinstance(chosen, PinPlacementProtocol):
            raise TypeError(
                "placement must satisfy PinPlacementProtocol, got "
                f"{type(chosen).__name__}"
            )
        closures = system.closures()
        for k, closure in enumerate(closures):
            _refuse_unpinnable(k, closure, parameters is not None)
        bkd, nstates = system.bkd(), system.nstates()
        self._bkd, self._placement = bkd, chosen
        self._closures = closures
        self._masks = system.closure_masks()
        self._space: BackendVectorSpace[Array] = backend_state_space(bkd, nstates)
        descriptor = self._space.descriptor()
        declared = system.essential()
        essential_rows = NoEssentialRows(bkd, nstates) if declared is None else declared
        self._essential: RowPin[Array] = RowPin(essential_rows)
        mask = sum(self._masks[1:], self._masks[0]) if self._masks else None
        self._mask = bkd.zeros((nstates,)) if mask is None else mask
        # Read once at construction to build the row sets
        closure_rows = [
            row for row in range(nstates) if bkd.to_float(self._mask[row]) > 0.5
        ]
        self._closure_rows = BackendRowSet(descriptor, closure_rows, bkd)
        # Every row the selected blocks replace: the essential rows, read
        # back as a 0/1 column, and the closure rows
        essential_mask = essential_rows.rows().replace_rows(
            bkd.zeros((nstates, 1)), bkd.ones((nstates, 1))
        )[:, 0]
        replaced = [
            row
            for row in range(nstates)
            if bkd.to_float(essential_mask[row] + self._mask[row]) > 0.5
        ]
        self._replacement: RowReplacement[Array] = RowReplacement(
            BackendRowSet(descriptor, replaced, bkd)
        )
        self._zero = bkd.zeros((nstates,))
        self._own_time_invariant = all(
            closure.is_time_invariant() for closure in closures
        )
        self._pin = _ClosureBlockPin(self)

    def at(self, block: StepBlock) -> Optional[_ClosureBlockPin[Array]]:
        """The block's pin, or None when the placement leaves it."""
        return self._pin if self._placement.pins(block) else None

    def _closure_residual(self, state: Array, time: float) -> Array:
        r""":math:`c(y, t)` on the closure rows, zero elsewhere, each row
        from the declaration that supplies it. Shape: (nstates,)"""
        total = self._zero
        for mask, closure in zip(self._masks, self._closures):
            total = total + mask * closure.residual(state, time)
        return total

    def _closure_jacobian(self, time: float) -> Array:
        r""":math:`c_u` at ``time`` on the closure rows, zero elsewhere;
        affine, so taken at the zero state. Shape: (nstates, nstates)"""
        nstates = self._zero.shape[0]
        total = self._bkd.zeros((nstates, nstates))
        for mask, closure in zip(self._masks, self._closures):
            total = total + mask[:, None] * closure.jacobian(self._zero, time)
        return total

    def _with_closure_rows(self, column: Array, values: Array) -> Array:
        """``column`` with its closure rows ``values``. Shape: (nstates, 1)"""
        mask = self._mask[:, None]
        return (1.0 - mask) * column + mask * values[:, None]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nclosure_rows={self._closure_rows.count()}, "
            f"placement={self._placement!r})"
        )
