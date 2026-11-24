r"""Collocation's closure rows imposed at the end of each step.

On a natural or periodic row the collocation equation is a relation
:math:`c(y, t) = 0` (Robin: :math:`\alpha y_i + \beta (\partial_n y)_i -
g_i(t)`), not an evolution law. :class:`ClosurePinning` imposes it on the
blocks a placement selects, by default the block producing
:math:`y_{n+1}`, through the shared
:class:`~pyapprox.ode.affine_rows.AffineRowPinning`: in a selected block
the closure rows read :math:`c(Y, t)` and the essential rows are pinned to
their data by :class:`~pyapprox.ode.pinned_rows.RowPin`.

A block not selected keeps the raw collocation row: the PDE collocated at
the boundary node, a consistent evolution law for a smooth solution. A
stage therefore advances the boundary values by the PDE and the step end
enforces the relation, so the stored states satisfy it exactly.

:class:`ClosureRelation` is the system's closure rows as that one affine
relation. The block pin is given no state, and its data terms no closure
derivative, so the closure rows must be affine in the state and
independent of the parameters; others are refused at construction.
"""

from typing import Generic, Optional

from pyapprox.ode.affine_rows import AffineRowPinning
from pyapprox.ode.pinned_rows import (
    AtStepEnd,
    BlockPinProtocol,
    PinPlacementProtocol,
    StepBlock,
)
from pyapprox.ode.residual_derivatives import StatedZero, backend_state_space
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


class ClosureRelation(Generic[Array]):
    r"""A collocation system's closure rows as one affine relation
    :math:`c(y, t) = C(t)\, y - b(t)` on their rows, each row from the
    declaration that supplies it.

    Satisfies :class:`~pyapprox.ode.affine_rows.AffineRowsProtocol` on the
    system's state, for :class:`~pyapprox.ode.affine_rows.AffineRowPinning`,
    and :class:`~pyapprox.ode.second_order.closure_forms.
    DisplacementConstraintProtocol` on the displacement of a second-order
    system built on it, for a closure form: the two read the same methods.

    Parameters
    ----------
    system : CollocationSystem
        Supplies the closure declarations, with shared rows already
        resolved.
    parameters : ParameterState, optional
        The system's parameters, if any; with them, closure rows reading a
        physics coefficient are refused.

    Raises
    ------
    TypeError
        If ``system`` is not a ``CollocationSystem``.
    ValueError
        If a closure is not declared affine in the state, holds
        parameters of its own, or reads a physics coefficient while the
        system has parameters.
    """

    def __init__(
        self,
        system: CollocationSystem[Array],
        parameters: Optional[ParameterState[Array]] = None,
    ) -> None:
        if not isinstance(system, CollocationSystem):
            raise TypeError(
                f"system must be a CollocationSystem, got {type(system).__name__}"
            )
        closures = system.closures()
        for k, closure in enumerate(closures):
            _refuse_unpinnable(k, closure, parameters is not None)
        bkd, nstates = system.bkd(), system.nstates()
        self._bkd, self._closures = bkd, closures
        self._masks = system.closure_masks()
        mask = sum(self._masks[1:], self._masks[0]) if self._masks else None
        total = bkd.zeros((nstates,)) if mask is None else mask
        self._space: BackendVectorSpace[Array] = backend_state_space(bkd, nstates)
        # Read once at construction to build the row set
        rows = [row for row in range(nstates) if bkd.to_float(total[row]) > 0.5]
        self._rows = BackendRowSet(self._space.descriptor(), rows, bkd)
        self._zero = bkd.zeros((nstates,))
        self._time_invariant = all(closure.is_time_invariant() for closure in closures)

    def space(self) -> BackendVectorSpace[Array]:
        """The state space."""
        return self._space

    def rows(self) -> BackendRowSet[Array]:
        """The closure rows."""
        return self._rows

    def residual(self, state: Array, time: float) -> Array:
        r""":math:`c(y, t)` on the closure rows, zero elsewhere, for a
        single column ``state``. Shape: (nstates, 1)"""
        flat = state[:, 0]
        total = self._zero
        for mask, closure in zip(self._masks, self._closures):
            total = total + mask * closure.residual(flat, time)
        return self._bkd.reshape(total, (-1, 1))

    def jacobian(self, time: float) -> BackendDenseOperator[Array]:
        r""":math:`C(t)` on the closure rows, zero elsewhere; affine, so
        taken at the zero state."""
        nstates = self._zero.shape[0]
        total = self._bkd.zeros((nstates, nstates))
        for mask, closure in zip(self._masks, self._closures):
            total = total + mask[:, None] * closure.jacobian(self._zero, time)
        return BackendDenseOperator(total, self._space.descriptor(), self._bkd)

    def is_time_invariant(self) -> bool:
        """Whether every closure's Jacobian and data are time-invariant."""
        return self._time_invariant

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nrows={self._rows.count()})"


class ClosurePinning(Generic[Array]):
    r"""The essential rows pinned and the closure rows imposed in the
    blocks a placement selects.

    Implements :class:`~pyapprox.ode.pinned_rows.RowPinningProtocol`: the
    shared :class:`~pyapprox.ode.affine_rows.AffineRowPinning` of the
    system's :class:`ClosureRelation` and essential rows.

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
        relation = ClosureRelation(system, parameters)
        declared = system.essential()
        essential = (
            NoEssentialRows(system.bkd(), system.nstates())
            if declared is None
            else declared
        )
        self._pinning: AffineRowPinning[Array, DenseFormat[Array]] = AffineRowPinning(
            relation, essential, relation.space(), chosen
        )

    def at(
        self, block: StepBlock
    ) -> Optional[BlockPinProtocol[Array, DenseFormat[Array]]]:
        """The block's pin, or None when the placement leaves it."""
        return self._pinning.at(block)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._pinning!r})"
