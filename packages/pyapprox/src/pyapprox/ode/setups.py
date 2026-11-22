r"""Setup requests of a mass-form stepper.

Every operator a stepper of :math:`M \dot y = f(y, t; p)` solves with is

.. math::

    P\,(a \hat M - s \hat J(y, t)) + I - P,

with :math:`\hat M`, :math:`\hat J` the system's mass and Jacobian, and
:math:`P` keeping the rows a block's pin leaves free (:math:`P = I`
unpinned). An implicit stage has :math:`a = 1`, :math:`s = \Delta t\,
a_{ii}`; an explicit stage, a final block or an explicit multistep step
:math:`s = 0`. A :class:`MassShiftRequest` describes one such operator
and :class:`MassShiftBuilder` forms it; the strategies of
:mod:`pyapprox.implicitfunction.setups` decide which setups to keep.

A converged state is named by its :class:`StatePosition` (the origin a
trajectory store stamps on the step's window, and the block of that
step), never by the state object, so a store that recomputes states from
checkpoints still finds the setups kept for them, and a kept setup holds
no store. An iterate of a stage solve, or a state of a window built by
hand, has no position.
"""

from typing import Generic, Hashable, List, Optional, Tuple, TypeVar

from pyapprox.implicitfunction.setups import (
    OperatorBuilderProtocol,
    SetupKey,
    SetupStrategyProtocol,
    state_key,
)
from pyapprox.ode.mass_form_system import MassFormSystem, parameters_version_reader
from pyapprox.ode.pinned_rows import BlockPinProtocol
from pyapprox.ode.stepper_derivatives import StepOrigin
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    LinearizationPoint,
    ScaledSum,
)

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
F = TypeVar("F", bound=FormatProtocol)


class StatePosition:
    """Where a converged state sits: a step of a trajectory (the origin a
    store stamps on the step's window) and the block of that step.

    Parameters
    ----------
    origin : StepOrigin
        The trajectory and step.
    block : int
        The residual block of the step (a stage, say), counted from 0.

    Raises
    ------
    TypeError
        If ``origin`` is not a :class:`~pyapprox.ode.stepper_derivatives.StepOrigin`.
    ValueError
        If ``block`` is negative.
    """

    __slots__ = ("_origin", "_block")

    def __init__(self, origin: StepOrigin, block: int) -> None:
        if not isinstance(origin, StepOrigin):
            raise TypeError(f"origin must be a StepOrigin, got {type(origin).__name__}")
        if block < 0:
            raise ValueError(f"block must be nonnegative, got {block}")
        self._origin, self._block = origin, block

    def origin(self) -> StepOrigin:
        return self._origin

    def block(self) -> int:
        return self._block

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StatePosition):
            return NotImplemented
        return other._origin == self._origin and other._block == self._block

    def __hash__(self) -> int:
        return hash((self._origin, self._block))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._origin!r}, block={self._block})"


class MassShiftRequest(Generic[V, A, F]):
    r""":math:`P\,(a \hat M - s \hat J(y, t)) + I - P` of a mass-form
    system.

    The key holds the system and the pin (by identity) and :math:`a`,
    :math:`s`; when :math:`s \ne 0`, also the time if :math:`J` sees it
    and the state's position if :math:`J` sees the state. Which it sees is
    read from the system's bundle. The time is also held when the pin's
    own block changes with it.

    Parameters
    ----------
    source : MassFormSystem
        The system; compared by identity.
    a, s : float
        The coefficients.
    pin : BlockPinProtocol or None
        How the block's rows are pinned; compared by identity. None for
        none.
    time : float
        Where :math:`J` is evaluated.
    state : V
        Where :math:`J` is evaluated, a single column of the state space;
        read by the builder only, and only when :math:`s \ne 0`.
    position : StatePosition or None
        Names ``state`` when it is converged; None for an iterate.

    Raises
    ------
    ValueError
        If ``a`` and ``s`` are both zero.
    TypeError
        If :math:`s \ne 0` and the system's bundle has no Jacobian.
    """

    __slots__ = (
        "_source",
        "_a",
        "_s",
        "_pin",
        "_time",
        "_state",
        "_position",
        "_sees_state",
        "_key",
    )

    def __init__(
        self,
        source: MassFormSystem[V, A, F],
        a: float,
        s: float,
        pin: Optional[BlockPinProtocol[V, F]],
        time: float,
        state: V,
        position: Optional[StatePosition],
    ) -> None:
        self._a, self._s, self._time = float(a), float(s), float(time)
        if self._a == 0.0 and self._s == 0.0:
            raise ValueError("a and s are both zero; there is no operator")
        sees_time = False
        self._sees_state = False
        if self._s != 0.0:
            jacobian = source.rhs.derivatives().jacobian
            if jacobian is None:
                raise TypeError(
                    f"{type(source.rhs).__name__}'s derivatives bundle has no "
                    "jacobian; only an operator with s = 0 can be requested"
                )
            self._sees_state = not jacobian.affine()
            sees_time = not jacobian.time_invariant()
        # A pin whose own block changes with time makes the operator do so
        # whatever s is
        sees_time = sees_time or (pin is not None and not pin.own_is_time_invariant())
        self._source, self._pin = source, pin
        self._state, self._position = state, position
        when = self._time if sees_time else None
        where = state_key(position) if self._sees_state else None
        self._key = SetupKey((source, pin), (self._a, self._s, when, where))

    def source(self) -> MassFormSystem[V, A, F]:
        """The system; compared by identity."""
        return self._source

    def a(self) -> float:
        return self._a

    def s(self) -> float:
        return self._s

    def pin(self) -> Optional[BlockPinProtocol[V, F]]:
        return self._pin

    def time(self) -> float:
        return self._time

    def state(self) -> V:
        return self._state

    def position(self) -> Optional[StatePosition]:
        return self._position

    def at_solution(self) -> bool:
        return self._position is not None

    def sees_state(self) -> bool:
        return self._sees_state

    def key(self) -> SetupKey:
        return self._key

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(a={self._a}, s={self._s}, "
            f"time={self._time}, pinned={self._pin is not None}, "
            f"position={self._position!r})"
        )


class MassShiftBuilder(Generic[V, A, F]):
    r"""Forms a request's operator from the system's own mass and
    Jacobian: the scaled sum (the mass alone when :math:`s = 0`), then
    the pin's own block.

    Raises
    ------
    ValueError
        From :meth:`build`, for a request of another system.
    """

    def __init__(self, system: MassFormSystem[V, A, F]) -> None:
        self._system = system
        self._version = parameters_version_reader(system.rhs)

    def build(self, request: MassShiftRequest[V, A, F]) -> ActionOperatorProtocol[V, F]:
        system = self._system
        if request.source() is not system:
            raise ValueError(
                f"a request of {request.source()!r} given to the builder of {system!r}"
            )
        terms: List[Tuple[float, ActionOperatorProtocol[V, F]]] = [
            (request.a(), system.mass)
        ]
        # With s = 0 the system may have no Jacobian (an explicit method).
        if request.s() != 0.0:
            jacobian = system.jacobian(
                system.form.to_residual(request.state()),
                request.time(),
                self._label(request),
            )
            terms.append((-request.s(), jacobian))
        operator = ScaledSum(terms, system.space)
        pin = request.pin()
        return operator if pin is None else pin.own(operator, request.time())

    def _label(self, request: MassShiftRequest[V, A, F]) -> LinearizationPoint:
        # Nothing reads it; kept truthful so a label-keyed cache could not
        # confuse two operators.
        version = self._version()
        if request.sees_state():
            return LinearizationPoint(
                version, object(), provisional=not request.at_solution()
            )
        point: Hashable = None
        if not self._system.jacobian_time_invariant():
            point = ("time", request.time())
        return LinearizationPoint(version, point)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._system!r})"


MassShiftStrategy = SetupStrategyProtocol[MassShiftRequest[V, A, F], V, F]
"""A strategy serving a mass-form stepper's requests."""

MassShiftBuilderProtocol = OperatorBuilderProtocol[MassShiftRequest[V, A, F], V, F]
"""A builder of a mass-form stepper's operators."""


def stepper_setups(
    system: MassFormSystem[V, A, F],
    setups: MassShiftStrategy[V, A, F],
    builder: Optional[MassShiftBuilderProtocol[V, A, F]],
) -> Tuple[MassShiftStrategy[V, A, F], MassShiftBuilderProtocol[V, A, F]]:
    """What a stepper of ``system`` sets up through: ``setups``, checked,
    and ``builder``, by default :class:`MassShiftBuilder` of ``system``.

    Raises
    ------
    TypeError
        If ``setups`` or ``builder`` does not satisfy its protocol.
    """
    if not isinstance(setups, SetupStrategyProtocol):
        raise TypeError(
            f"setups must satisfy SetupStrategyProtocol, got {type(setups).__name__}"
        )
    if builder is None:
        return setups, MassShiftBuilder(system)
    if not isinstance(builder, OperatorBuilderProtocol):
        raise TypeError(
            "builder must satisfy OperatorBuilderProtocol, got "
            f"{type(builder).__name__}"
        )
    return setups, builder
