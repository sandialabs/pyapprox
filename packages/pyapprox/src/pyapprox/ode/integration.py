r"""Marching a stepper forward and recording the trajectory.

:class:`Integrator` advances a stepper through given times and writes,
into an injected sink, every stored state and every step's unknown;
when the sink is a store, the sweeps rebuild each step's window from it
later. The state constraint
:math:`\Pi_t` is applied at the stored states only, the initial state
included,

.. math::

    y_0 = \Pi_{t_0}(\hat y_0), \qquad u_n = \mathrm{step}(y_n),
    \qquad y_{n+1} = \Pi_{t_{n+1}}(E\, u_n),

where :math:`E` is the stepper's readout; the unknown :math:`u_n` is
recorded unconstrained, since it is where step :math:`n` is linearized,
and so is the value :math:`\Pi` was applied to (:math:`\hat y_0` and
:math:`E u_n`), where the sweeps linearize :math:`\Pi`.

This is one of two places a constraint enters. The other is inside the
step: a stepper may replace rows of its own equations (boundary rows set
to their data at the end of a step, or at every stage), and then the
step's derivative operators carry that treatment, so the sweeps see it
without a branch of their own.

States are stack vectors throughout. A constraint written for the
residual's own state form (:class:`~pyapprox.ode.protocols.StateConstraintProtocol`)
is called through :class:`ResidualFormConstraint`, as the stepper calls
the residual.
"""

from collections import deque
from dataclasses import dataclass
from typing import (
    Deque,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from pyapprox.ode.protocols import ForwardStepperProtocol, StateConstraintProtocol
from pyapprox.ode.residual_derivatives import DerivativeOperator, SecondDerivativeFn
from pyapprox.ode.residual_form import ResidualFormProtocol
from pyapprox.ode.state_constraint import StateConstraintDerivatives, TangentFn
from pyapprox.ode.step_plan import StepPlanProtocol
from pyapprox.ode.trajectory_store import TrajectorySinkProtocol
from pyapprox.util.backends.protocols import Array

V = TypeVar("V")


@runtime_checkable
class StackConstraintProtocol(Protocol[V]):
    r""":math:`\Pi_t` on single-column stack vectors."""

    def constrain(self, state: V, time: float) -> V:
        """The state satisfying the constraints at ``time``."""
        ...


@runtime_checkable
class DifferentiableStackConstraintProtocol(StackConstraintProtocol[V], Protocol[V]):
    r""":math:`\Pi_t` on stack vectors, with its derivatives.

    The bundle's callables take stack vectors (states and adjoints as
    single columns) and return operators of the stack; see
    :mod:`pyapprox.ode.state_constraint`.
    """

    def derivatives(self) -> StateConstraintDerivatives[V]:
        """The constraint's optional derivatives."""
        ...


@dataclass(frozen=True)
class _AtStackState(Generic[V]):
    """A ``(state, time)`` derivative of a residual-form constraint,
    called with a stack vector."""

    field: TangentFn[V]
    form: ResidualFormProtocol[V, V]

    def __call__(self, state: V, time: float) -> DerivativeOperator[V]:
        return self.field(self.form.to_residual(state), time)


@dataclass(frozen=True)
class _AtStackStateAndAdjoint(Generic[V]):
    """A ``(state, time, adj)`` second derivative of a residual-form
    constraint, called with stack vectors."""

    field: SecondDerivativeFn[V]
    form: ResidualFormProtocol[V, V]

    def __call__(self, state: V, time: float, adj: V) -> DerivativeOperator[V]:
        to_residual = self.form.to_residual
        return self.field(to_residual(state), time, to_residual(adj))


def _at_state(
    field: Optional[TangentFn[V]], form: ResidualFormProtocol[V, V]
) -> Optional[TangentFn[V]]:
    return None if field is None else _AtStackState(field, form)


def _at_state_and_adjoint(
    field: Optional[SecondDerivativeFn[V]], form: ResidualFormProtocol[V, V]
) -> Optional[SecondDerivativeFn[V]]:
    return None if field is None else _AtStackStateAndAdjoint(field, form)


class ResidualFormConstraint(Generic[Array]):
    """A residual-form state constraint called on stack vectors.

    Parameters
    ----------
    constraint : StateConstraintProtocol
        Constrains states in the residual's form.
    form : ResidualFormProtocol
        Converts stack vectors to that form and back.
    """

    def __init__(
        self,
        constraint: StateConstraintProtocol[Array],
        form: ResidualFormProtocol[Array, Array],
    ) -> None:
        self._constraint, self._form = constraint, form

    def constrain(self, state: Array, time: float) -> Array:
        form = self._form
        return form.from_residual(
            self._constraint.constrain(form.to_residual(state), time)
        )

    def derivatives(self) -> StateConstraintDerivatives[Array]:
        """The wrapped constraint's derivatives, field for field, taking
        stack vectors; absent fields stay absent."""
        inner, form = self._constraint.derivatives(), self._form
        return StateConstraintDerivatives(
            tangent=_at_state(inner.tangent, form),
            param=_at_state(inner.param, form),
            state_state=_at_state_and_adjoint(inner.state_state, form),
            state_param=_at_state_and_adjoint(inner.state_param, form),
            param_param=_at_state_and_adjoint(inner.param_param, form),
            param_space=inner.param_space,
        )

    def constraint(self) -> StateConstraintProtocol[Array]:
        """The wrapped residual-form constraint."""
        return self._constraint

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._constraint!r})"


class Integrator(Generic[V]):
    r"""March a stepper through given times, writing into a sink.

    Parameters
    ----------
    stepper : ForwardStepperProtocol
        Solves each step and reads out the new state.
    constraint : StackConstraintProtocol, optional
        :math:`\Pi_t`, applied at every stored state; none by default.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    """

    def __init__(
        self,
        stepper: ForwardStepperProtocol[V],
        constraint: Optional[StackConstraintProtocol[V]] = None,
    ) -> None:
        if not isinstance(stepper, ForwardStepperProtocol):
            raise TypeError(
                "stepper must satisfy ForwardStepperProtocol, got "
                f"{type(stepper).__name__}"
            )
        if constraint is not None and not isinstance(
            constraint, StackConstraintProtocol
        ):
            raise TypeError(
                "constraint must satisfy StackConstraintProtocol, got "
                f"{type(constraint).__name__}"
            )
        self._stepper = stepper
        self._constraint = constraint

    def stepper(self) -> ForwardStepperProtocol[V]:
        """The stepper."""
        return self._stepper

    def constraint(self) -> Optional[StackConstraintProtocol[V]]:
        """The state constraint, or ``None``."""
        return self._constraint

    def _constrained(self, state: V, time: float) -> V:
        if self._constraint is None:
            return state
        return self._constraint.constrain(state, time)

    def solve(
        self,
        init_state: V,
        steps: StepPlanProtocol,
        sink: TrajectorySinkProtocol[V],
    ) -> None:
        """March from ``init_state`` through the steps of ``steps``,
        writing into ``sink``.

        Parameters
        ----------
        init_state : V
            :math:`\\hat y_0`, a single column of the state space; it is
            constrained, not modified.
        steps : StepPlanProtocol
            Which steps to take, from its origin :math:`t_0`; a fresh
            sequence of them is started for this solve, and it takes each
            step through this solve's session.
        sink : TrajectorySinkProtocol
            Empty; receives :math:`y_0, \\dots, y_N` and each step's span
            and unknown. A store, for the sweeps to read back; any sink
            when no sweep follows.

        Raises
        ------
        TypeError
            If ``steps`` does not satisfy its protocol.
        ValueError
            If ``sink`` is not empty.
        SolveFailedError
            If a step's solve fails and the plan does not retry it.
        """
        if not isinstance(steps, StepPlanProtocol):
            raise TypeError(
                f"steps must satisfy StepPlanProtocol, got {type(steps).__name__}"
            )
        if sink.ntimes() != 0:
            raise ValueError("the sink must be empty")
        stepper = self._stepper
        # One session for this solve: its steps may share setups (a factored
        # stage operator), which are dropped with it, so nothing reused
        # outlives the parameters the solve ran at.
        session = stepper.session()
        sequence = steps.start()
        # With a constraint, each stored state is recorded with the value
        # the constraint was applied to, where the sweeps linearize it.
        constrained = self._constraint is not None
        origin = steps.origin()
        state = self._constrained(init_state, origin)
        sink.append(origin, state, init_state if constrained else None)
        # The states the next step reads, most recent first: the
        # stepper's depth of them once there are that many.
        recent: Deque[V] = deque([state], maxlen=stepper.depth())
        while (taken := sequence.next_step(session, tuple(recent))) is not None:
            readout = stepper.readout().apply(taken.unknown).blocks[0]
            state = self._constrained(readout, taken.span.end())
            recent.appendleft(state)
            sink.append_step(
                taken.span,
                state,
                taken.unknown,
                stepper.parameters_version(),
                readout if constrained else None,
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._stepper!r})"
