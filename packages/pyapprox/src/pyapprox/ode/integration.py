r"""Marching a stepper forward and recording the trajectory.

:class:`Integrator` advances a stepper through given times and records,
in an injected store, every stored state and every step's unknown, so
the sweeps can rebuild each step's window later. The state constraint
:math:`\Pi_t` is applied at the stored states only, the initial state
included,

.. math::

    y_0 = \Pi_{t_0}(\hat y_0), \qquad u_n = \mathrm{step}(y_n),
    \qquad y_{n+1} = \Pi_{t_{n+1}}(E\, u_n),

where :math:`E` is the stepper's readout; the unknown :math:`u_n` is
recorded unconstrained, since it is where step :math:`n` is linearized.

States are stack vectors throughout. A constraint written for the
residual's own state form (:class:`~pyapprox.ode.protocols.StateConstraintProtocol`)
is called through :class:`ResidualFormConstraint`, as the stepper calls
the residual.
"""

from typing import Generic, Optional, Protocol, Sequence, TypeVar, runtime_checkable

from pyapprox.ode.protocols import ForwardStepperProtocol, StateConstraintProtocol
from pyapprox.ode.residual_form import ResidualFormProtocol
from pyapprox.ode.trajectory_store import TrajectoryStoreProtocol
from pyapprox.util.backends.protocols import Array

V = TypeVar("V")


@runtime_checkable
class StackConstraintProtocol(Protocol[V]):
    r""":math:`\Pi_t` on single-column stack vectors."""

    def constrain(self, state: V, time: float) -> V:
        """The state satisfying the constraints at ``time``."""
        ...


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

    def constraint(self) -> StateConstraintProtocol[Array]:
        """The wrapped residual-form constraint."""
        return self._constraint

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._constraint!r})"


class Integrator(Generic[V]):
    r"""March a stepper through given times, recording into a store.

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
        times: Sequence[float],
        store: TrajectoryStoreProtocol[V],
    ) -> TrajectoryStoreProtocol[V]:
        """March from ``init_state`` through ``times``, recording into
        ``store``.

        Parameters
        ----------
        init_state : V
            :math:`\\hat y_0`, a single column of the state space; it is
            constrained, not modified.
        times : sequence of float
            The step boundaries :math:`t_0 < t_1 < \\dots < t_N`, at least
            two.
        store : TrajectoryStoreProtocol
            Empty; receives :math:`y_0, \\dots, y_N` and each step's
            unknown.

        Returns
        -------
        TrajectoryStoreProtocol
            ``store``, filled.

        Raises
        ------
        ValueError
            If ``times`` has fewer than two entries or does not strictly
            increase, or ``store`` is not empty.
        SolveFailedError
            If a step's solve fails.
        """
        if len(times) < 2:
            raise ValueError(f"times needs at least two entries, got {len(times)}")
        if any(not later > earlier for earlier, later in zip(times, times[1:])):
            raise ValueError("times must strictly increase")
        if store.ntimes() != 0:
            raise ValueError("the store must be empty")
        stepper = self._stepper
        state = self._constrained(init_state, times[0])
        store.append(times[0], state)
        for time, next_time in zip(times, times[1:]):
            unknown = stepper.step(state, time, next_time - time)
            readout = stepper.readout().apply(unknown).blocks[0]
            state = self._constrained(readout, next_time)
            store.append_step(next_time, state, unknown, stepper.parameters_version())
        return store

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._stepper!r})"
