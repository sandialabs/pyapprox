r"""Tangent-linear and adjoint sweeps over a recorded trajectory.

A trajectory recorded by :class:`~pyapprox.ode.integration.Integrator`
satisfies, for each step :math:`n`, the step equation
:math:`R_n(u_n;\, y_n, \dots, y_{n+1-k};\, p) = 0` with
:math:`y_{n+1} = \Pi(E u_n)` and :math:`y_0 = \Pi(\hat y_0(p))`, where
:math:`E` is the stepper's readout and :math:`\Pi` the state constraint
(the identity without one). For a scalar functional
:math:`Q(y_0, \dots, y_N;\, p)`:

- the **tangent sweep** propagates a parameter direction :math:`v`
  forward, :math:`w_0 = \Pi'(\partial \hat y_0/\partial p)\, v
  + \Pi_p v` and, per step,

  .. math::

      \frac{\partial R_n}{\partial u_n}\, \delta u_n
      = -\Big(\sum_{l \ge 1} \frac{\partial R_n}{\partial y_{n+1-l}}
      w_{n+1-l} + \frac{\partial R_n}{\partial p} v\Big), \qquad
      w_{n+1} = \Pi'\, E\, \delta u_n + \Pi_p v,

  accumulating :math:`dQ = \sum_m \langle \partial Q/\partial y_m,
  w_m\rangle + \langle \partial Q/\partial p, v\rangle`;

- the **adjoint sweep** runs backward with the cotangents
  :math:`\bar\nu_m = \partial Q/\partial y_m + \sum_l
  (\partial R_{m-1+l}/\partial y_m)^\top \mu_{m-1+l}` of the stored
  states, one transposed block solve per step,

  .. math::

      \Big(\frac{\partial R_n}{\partial u_n}\Big)^\top \mu_n
      = -E^\top \Pi'^\top \bar\nu_{n+1},

  and the gradient :math:`\partial Q/\partial p + \sum_n
  (\partial R_n/\partial p)^\top \mu_n + \sum_m \Pi_p^\top \bar\nu_m
  + (\partial \hat y_0/\partial p)^\top \Pi'^\top \bar\nu_0`.

:math:`\Pi'` and :math:`\Pi_p` are evaluated at the value the constraint
was applied to, which the store records. :math:`y_0` is data, so its
cotangent :math:`\bar\nu_0` is a sum of applies: no solve at
:math:`t_0`. A solve there belongs to whatever defines :math:`y_0` by an
equation, inside :math:`\Pi'` (a mass-consistent projection) or as a
recorded step of its own.

A constraint applied inside the step (rows of the step equation replaced
by boundary data) is part of the stepper's derivative operators, so
these sweeps handle it unchanged.

Every parameter term is an operator applied to one vector: no
state-by-parameter matrix is ever formed, so a field-valued parameter as
large as the state costs one apply per term.
"""

from collections import deque
from typing import Deque, Dict, Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.ode.integration import DifferentiableStackConstraintProtocol
from pyapprox.ode.mass_form_system import RightHandSideProtocol
from pyapprox.ode.protocols import DifferentiableStepperProtocol
from pyapprox.ode.residual_derivatives import DerivativeOperator
from pyapprox.ode.stepper_derivatives import (
    OffDiagFn,
    StepParamFn,
    StepWindow,
)
from pyapprox.ode.trajectory_store import TrajectoryStoreProtocol
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import (
    BlockVector,
    InnerProductSpaceProtocol,
    SolveFailedError,
    SolverHandleProtocol,
)

V = TypeVar("V")
V_co = TypeVar("V_co", covariant=True)
A = TypeVar("A", bound=ArrayProtocol)


@runtime_checkable
class TrajectoryFunctionalProtocol(Protocol[V_co]):
    r"""What the sweeps read of a scalar functional of one recorded
    trajectory: its first derivatives, as stack vectors."""

    def state_cotangent(self, index: int) -> V_co:
        r""":math:`\partial Q/\partial y_{\mathrm{index}}`, a single column
        of the state space."""
        ...

    def param_cotangent(self) -> V_co:
        r"""The direct :math:`\partial Q/\partial p`, a single column of
        the parameter space."""
        ...


@runtime_checkable
class InitialStateDerivativeProtocol(Protocol[V]):
    r""":math:`\partial \hat y_0/\partial p`, for an initial state that
    depends on the parameters."""

    def param_derivative(self) -> DerivativeOperator[V]:
        """Parameters to state, valid for the parameters currently set;
        a zero operator declares ``zero`` and is skipped."""
        ...


class ResidualInitialState(Generic[V]):
    r""":math:`\partial \hat y_0/\partial p` from the residual's
    derivatives bundle (its ``initial_state`` field).

    Parameters
    ----------
    rhs : RightHandSideProtocol
        The residual; its bundle is fetched on every call, so it is valid
        for the parameters currently set.
    """

    def __init__(self, rhs: RightHandSideProtocol[V]) -> None:
        self._rhs = rhs

    def param_derivative(self) -> DerivativeOperator[V]:
        """The bundle's ``initial_state()``.

        Raises
        ------
        TypeError
            If the bundle has no ``initial_state``.
        """
        field = self._rhs.derivatives().initial_state
        if field is None:
            raise TypeError(
                f"{type(self._rhs).__name__}'s derivatives() has no initial_state"
            )
        return field()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._rhs!r})"


class _Cotangents(Generic[V]):
    r"""The cotangents :math:`\bar\nu_m` of the stored states, built up
    during the backward sweep.

    A stored state's cotangent is the functional's direct
    :math:`\partial Q/\partial y_m` plus a contribution
    :math:`(\partial R_n/\partial y_m)^\top \mu_n` from every step
    :math:`n` that reads it. The sweep runs backward, so when it reaches
    the step producing :math:`y_m`, every step reading :math:`y_m` has
    already contributed and the cotangent is complete. Only states still
    to be read are held: at most the stepper's depth.

    Parameters
    ----------
    functional : TrajectoryFunctionalProtocol
        Supplies the direct :math:`\partial Q/\partial y_m`.
    space : InnerProductSpaceProtocol
        The state space.
    """

    def __init__(
        self,
        functional: TrajectoryFunctionalProtocol[V],
        space: InnerProductSpaceProtocol[V, ArrayProtocol],
    ) -> None:
        self._functional, self._space = functional, space
        self._pending: Dict[int, V] = {}

    def add(self, index: int, contribution: V) -> None:
        """Add a reading step's contribution to state ``index``'s
        cotangent. ``contribution`` is kept and may be modified."""
        earlier = self._pending.get(index)
        self._pending[index] = (
            contribution
            if earlier is None
            else self._space.axpby(1.0, contribution, 1.0, earlier, out=earlier)
        )

    def complete(self, index: int) -> V:
        """State ``index``'s cotangent, once every step reading it has
        contributed; its contributions are released."""
        direct = self._functional.state_cotangent(index)
        later = self._pending.pop(index, None)
        return direct if later is None else self._space.axpby(1.0, later, 1.0, direct)

    def npending(self) -> int:
        """The number of states with contributions held."""
        return len(self._pending)


class TrajectoryDerivatives(Generic[V, A]):
    r"""The tangent and adjoint sweeps of a stepper's recorded
    trajectories (see the module docstring).

    Parameters
    ----------
    stepper : DifferentiableStepperProtocol
        The stepper the trajectories were recorded with; its bundle must
        have ``off_diag`` and ``param``.
    constraint : DifferentiableStackConstraintProtocol, optional
        The constraint the integrator applied at stored states; its
        bundle must have ``tangent`` and ``param``. None for no
        constraint.
    initial_state : InitialStateDerivativeProtocol, optional
        :math:`\partial \hat y_0/\partial p`; None for an initial state
        independent of the parameters.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    """

    def __init__(
        self,
        stepper: DifferentiableStepperProtocol[V, A],
        constraint: Optional[DifferentiableStackConstraintProtocol[V]] = None,
        initial_state: Optional[InitialStateDerivativeProtocol[V]] = None,
    ) -> None:
        if not isinstance(stepper, DifferentiableStepperProtocol):
            raise TypeError(
                "stepper must satisfy DifferentiableStepperProtocol, got "
                f"{type(stepper).__name__}"
            )
        if constraint is not None and not isinstance(
            constraint, DifferentiableStackConstraintProtocol
        ):
            raise TypeError(
                "constraint must satisfy DifferentiableStackConstraintProtocol, "
                f"got {type(constraint).__name__}"
            )
        if initial_state is not None and not isinstance(
            initial_state, InitialStateDerivativeProtocol
        ):
            raise TypeError(
                "initial_state must satisfy InitialStateDerivativeProtocol, got "
                f"{type(initial_state).__name__}"
            )
        self._stepper = stepper
        self._constraint = constraint
        self._initial_state = initial_state

    # ------------------------------------------------------------------
    # What both sweeps read
    # ------------------------------------------------------------------

    def _state_space(self) -> InnerProductSpaceProtocol[V, A]:
        return self._stepper.state_space().component(0)

    def _param_space(self) -> InnerProductSpaceProtocol[V, A]:
        """The parameter space, as a plain (one-block) space.

        Raises
        ------
        TypeError
            If the stepper has no parameters.
        """
        space = self._stepper.param_space()
        if space is None:
            raise TypeError(
                f"{type(self._stepper).__name__} has no parameter space, so it "
                "has no parameter derivatives"
            )
        return space.component(0)

    def _first_order(self) -> tuple[OffDiagFn[V], StepParamFn[V]]:
        """The stepper bundle's ``off_diag`` and ``param``, fetched for
        the parameters currently set.

        Raises
        ------
        TypeError
            If the bundle lacks either.
        """
        derivs = self._stepper.derivatives()
        if derivs.off_diag is None or derivs.param is None:
            raise TypeError(
                f"{type(self._stepper).__name__}'s derivatives() lacks off_diag "
                "or param, so it has no parameter gradients"
            )
        return derivs.off_diag, derivs.param

    def _window(self, store: TrajectoryStoreProtocol[V], step: int) -> StepWindow[V]:
        """Step ``step``'s window, refused when it was recorded at other
        parameters than those set now (the bundle would not match it).

        Raises
        ------
        ValueError
            If the window's parameter version is not the stepper's.
        """
        window = store.window(step, self._stepper.depth())
        current = self._stepper.parameters_version()
        if window.parameters_version != current:
            raise ValueError(
                f"step {step} was recorded at parameter version "
                f"{window.parameters_version}, not the current {current}: "
                "solve again before differentiating"
            )
        return window

    def _transposed_handle(
        self, window: StepWindow[V]
    ) -> SolverHandleProtocol[BlockVector[V]]:
        """A solver handle for the transposed diagonal block.

        Raises
        ------
        TypeError
            If the stepper's handle cannot solve transposed.
        """
        transposed = self._stepper.diag_handle(window).transpose()
        if transposed is None:
            raise TypeError(
                f"{type(self._stepper).__name__}'s diag_handle cannot solve "
                "transposed, so it has no adjoint"
            )
        return transposed

    @staticmethod
    def _solve(
        handle: SolverHandleProtocol[BlockVector[V]],
        rhs: BlockVector[V],
        what: str,
    ) -> BlockVector[V]:
        solution, info = handle.solve(rhs)
        if not info.converged:
            raise SolveFailedError(f"{what}: {info.message}")
        return solution

    def _constraint_terms(
        self, store: TrajectoryStoreProtocol[V], index: int
    ) -> Optional[tuple[DerivativeOperator[V], DerivativeOperator[V]]]:
        r""":math:`(\Pi', \Pi_p)` at stored state ``index``, linearized at
        the value the constraint was applied to; None without a
        constraint.

        Raises
        ------
        TypeError
            If the constraint's bundle lacks ``tangent`` or ``param``.
        """
        if self._constraint is None:
            return None
        derivs = self._constraint.derivatives()
        if derivs.tangent is None or derivs.param is None:
            raise TypeError(
                f"{type(self._constraint).__name__}'s derivatives() lacks "
                "tangent or param, so it has no parameter gradients"
            )
        point, time = store.unconstrained(index), store.time(index)
        return derivs.tangent(point, time), derivs.param(point, time)

    # ------------------------------------------------------------------
    # The tangent sweep
    # ------------------------------------------------------------------

    def _constrained_tangent(
        self, store: TrajectoryStoreProtocol[V], index: int, dz: V, direction: V
    ) -> V:
        r""":math:`w = \Pi' dz + \Pi_p v` at stored state ``index``."""
        terms = self._constraint_terms(store, index)
        if terms is None:
            return dz
        tangent, param = terms
        w = tangent.apply(dz)
        if not param.properties().zero:
            w = self._state_space().axpby(1.0, param.apply(direction), 1.0, w, out=w)
        return w

    def tangent(
        self,
        store: TrajectoryStoreProtocol[V],
        functional: TrajectoryFunctionalProtocol[V],
        direction: V,
    ) -> A:
        r""":math:`dQ/dp \cdot v`, by one forward sweep.

        Parameters
        ----------
        store : TrajectoryStoreProtocol
            A trajectory recorded at the parameters currently set.
        functional : TrajectoryFunctionalProtocol
            :math:`Q`'s derivatives on that trajectory.
        direction : V
            :math:`v`, a single column of the parameter space.

        Returns
        -------
        A
            The directional derivative, shape ``(1, 1)``.

        Raises
        ------
        TypeError
            If a derivative the sweep needs is missing.
        ValueError
            If the trajectory was recorded at other parameters.
        SolveFailedError
            If a step's solve fails.
        """
        stepper = self._stepper
        off_diag, param = self._first_order()
        space, params = self._state_space(), self._param_space()
        residuals = stepper.residual_space()
        readout = stepper.readout()
        p_dir = BlockVector((direction,))
        dz0 = (
            space.zeros()
            if self._initial_state is None
            else self._initial_state.param_derivative().apply(direction)
        )
        w0 = self._constrained_tangent(store, 0, dz0, direction)
        total = space.dot(functional.state_cotangent(0), w0) + params.dot(
            functional.param_cotangent(), direction
        )
        # The tangents of the states the next step reads, most recent
        # first: recent[l - 1] is w_{n+1-l}.
        recent: Deque[V] = deque([w0], maxlen=stepper.depth())
        for n in range(store.nsteps()):
            window = self._window(store, n)
            # rhs = -(sum_l dR/dy_{n+1-l} w_{n+1-l} + dR/dp v)
            rhs = param(window).apply(p_dir)
            for lag, w in enumerate(recent, start=1):
                coupled = off_diag(window, lag).apply(BlockVector((w,)))
                rhs = residuals.axpby(1.0, coupled, 1.0, rhs, out=rhs)
            rhs = residuals.scale(-1.0, rhs, out=rhs)
            delta = self._solve(
                stepper.diag_handle(window), rhs, f"tangent sweep, step {n}"
            )
            dz = readout.apply(delta).blocks[0]
            w = self._constrained_tangent(store, n + 1, dz, direction)
            total = total + space.dot(functional.state_cotangent(n + 1), w)
            recent.appendleft(w)
        return total

    # ------------------------------------------------------------------
    # The adjoint sweep
    # ------------------------------------------------------------------

    def _pull_constraint(
        self,
        store: TrajectoryStoreProtocol[V],
        index: int,
        cotangent: V,
        gradient: V,
    ) -> tuple[V, V]:
        r"""Pull :math:`\bar\nu` back through :math:`\Pi` at stored state
        ``index``.

        Returns
        -------
        tuple
            :math:`\Pi'^\top \bar\nu`, and ``gradient`` plus
            :math:`\Pi_p^\top \bar\nu` (``gradient`` may be reused for
            it: use the returned one).
        """
        terms = self._constraint_terms(store, index)
        if terms is None:
            return cotangent, gradient
        tangent, param = terms
        if not param.properties().zero:
            gradient = self._param_space().axpby(
                1.0, param.apply_transpose(cotangent), 1.0, gradient, out=gradient
            )
        return tangent.apply_transpose(cotangent), gradient

    def gradient(
        self,
        store: TrajectoryStoreProtocol[V],
        functional: TrajectoryFunctionalProtocol[V],
    ) -> V:
        r""":math:`dQ/dp`, by one backward sweep.

        Parameters
        ----------
        store : TrajectoryStoreProtocol
            A trajectory recorded at the parameters currently set.
        functional : TrajectoryFunctionalProtocol
            :math:`Q`'s derivatives on that trajectory.

        Returns
        -------
        V
            The gradient, a single column of the parameter space.

        Raises
        ------
        TypeError
            If a derivative the sweep needs is missing.
        ValueError
            If the trajectory was recorded at other parameters.
        SolveFailedError
            If a step's transposed solve fails.
        """
        stepper = self._stepper
        off_diag, param = self._first_order()
        space, params = self._state_space(), self._param_space()
        unknowns = stepper.unknown_space()
        readout = stepper.readout()
        cotangents = _Cotangents(functional, space)
        gradient = params.copy(functional.param_cotangent())
        for n in reversed(range(store.nsteps())):
            window = self._window(store, n)
            pulled, gradient = self._pull_constraint(
                store, n + 1, cotangents.complete(n + 1), gradient
            )
            # (dR_n/du_n)^T mu_n = -E^T Pi'^T nu_{n+1}
            rhs = readout.apply_transpose(BlockVector((pulled,)))
            rhs = unknowns.scale(-1.0, rhs, out=rhs)
            mu = self._solve(
                self._transposed_handle(window), rhs, f"adjoint sweep, step {n}"
            )
            contribution = param(window).apply_transpose(mu).blocks[0]
            gradient = params.axpby(1.0, contribution, 1.0, gradient, out=gradient)
            for lag in range(1, stepper.depth() + 1):
                coupled = off_diag(window, lag).apply_transpose(mu).blocks[0]
                cotangents.add(n + 1 - lag, coupled)
        pulled, gradient = self._pull_constraint(
            store, 0, cotangents.complete(0), gradient
        )
        if self._initial_state is not None:
            initial = self._initial_state.param_derivative()
            if not initial.properties().zero:
                gradient = params.axpby(
                    1.0, initial.apply_transpose(pulled), 1.0, gradient, out=gradient
                )
        return gradient

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._stepper!r})"
