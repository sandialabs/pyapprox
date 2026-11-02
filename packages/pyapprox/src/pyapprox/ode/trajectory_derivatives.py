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
  + (\partial \hat y_0/\partial p)^\top \Pi'^\top \bar\nu_0`;

- the **Hessian-vector product** differentiates that gradient along
  :math:`v` (forward over reverse): the tangent sweep gives the
  tangents :math:`w_m`, :math:`\delta u_n` and the unconstrained
  :math:`dx_m`, and one backward sweep solves, per step and with the same
  transposed handle, for :math:`\mu_n` and then for its derivative
  :math:`\dot\mu_n`,

  .. math::

      \Big(\frac{\partial R_n}{\partial u_n}\Big)^\top \dot\mu_n
      = -\Big(E^\top \dot{\bar z}_{n+1} + \sum_b H_n(0, b)\, dX_b
      + H_n(0, p)\, v\Big),

  where :math:`H_n(a, b)` is the lag-:math:`b` to lag-:math:`a` block of
  :math:`\mu_n^\top \partial^2 R_n` (the stepper's ``state_state``),
  :math:`H_n(a, p)` its ``state_param``, :math:`dX_0 = \delta u_n` and
  :math:`dX_l = w_{n+1-l}`, and :math:`\dot{\bar z}_m = \Pi'^\top
  \dot{\bar\nu}_m + (\bar\nu_m^\top \Pi)_{xx}\, dx_m + (\bar\nu_m^\top
  \Pi)_{xp}\, v`. The cotangent derivatives :math:`\dot{\bar\nu}_m`
  gather :math:`Q_{yy} w_m + Q_{yp} v` and, from each step reading
  :math:`y_m` at lag :math:`l`, :math:`(\partial R_n/\partial
  y_m)^\top \dot\mu_n + \sum_b H_n(l, b)\, dX_b + H_n(l, p)\, v`; the
  product collects every derivative of the gradient's terms. The
  first- and second-order adjoints are solved and pulled back by the
  same code; only their sources differ.

A functional (:class:`TrajectoryFunctional`) is a sum of terms read from
the stored states, :math:`Q_y(y_0, \dots, y_N;\, p)`
(:class:`StateTermProtocol`), and terms read from each step beyond its
stored states, :math:`\sum_n Q_n(u_n;\, y_n, \dots, y_{n+1-k};\, p)`
(:class:`StepTermProtocol`; a functional integrating over time reads the
stages); the sweeps add the terms' derivatives. In the adjoint sweep
:math:`\partial Q_n/\partial u_n` joins the step solve's right-hand side
(:math:`-(E^\top \Pi'^\top \bar\nu_{n+1} + \partial Q_n/\partial u_n)`),
:math:`\partial Q_n/\partial y_{n+1-l}` the cotangent of the state read
at lag :math:`l`, and :math:`\partial Q_n/\partial p` the gradient; the
tangent sweep adds their inner products with :math:`\delta u_n`,
:math:`w_{n+1-l}` and :math:`v`; and the second-order sweep adds, in the
same three places, their derivatives along the tangent
(:class:`StepTermDerivatives`). A step term's :math:`Q_n` and a state
term's :math:`Q_y` read the same stored state through different
channels, so a functional reports each dependence in exactly one.

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
from dataclasses import dataclass
from typing import (
    Callable,
    Deque,
    Dict,
    FrozenSet,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    runtime_checkable,
)

from pyapprox.ode.integration import DifferentiableStackConstraintProtocol
from pyapprox.ode.mass_form_system import RightHandSideProtocol
from pyapprox.ode.protocols import (
    DifferentiableSessionProtocol,
    DifferentiableStepperProtocol,
)
from pyapprox.ode.residual_derivatives import DerivativeOperator
from pyapprox.ode.residual_form import ResidualFormProtocol
from pyapprox.ode.stepper_derivatives import (
    OffDiagFn,
    ParamParamFn,
    StateParamFn,
    StateStateFn,
    StepParamFn,
    StepWindow,
)
from pyapprox.ode.trajectory_store import (
    InMemoryTrajectoryStore,
    TrajectoryStoreProtocol,
)
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import (
    BlockInnerProductSpace,
    BlockVector,
    FormatProtocol,
    InnerProductSpaceProtocol,
    SolveFailedError,
    SolverHandleProtocol,
    VectorSpaceProtocol,
)

V = TypeVar("V")
W = TypeVar("W")
A = TypeVar("A", bound=ArrayProtocol)


StateCurvatureFn = Callable[[int, V], Optional[V]]
"""``(index, x) ->`` a state term's second derivative at stored state
``index`` applied to the column ``x``; None for a zero block."""

ParamCurvatureFn = Callable[[V], Optional[V]]
"""``(v) ->`` a state term's second parameter derivative applied to the
parameter column ``v``; None for a zero block."""

StepUnknownCurvatureFn = Callable[
    [int, StepWindow[V], StepWindow[V], V], Optional[BlockVector[V]]
]
"""``(step, window, tangents, direction) ->`` the derivative of a step
term's :math:`\\partial Q/\\partial u_n` along the tangent."""

StepLagCurvatureFn = Callable[[int, StepWindow[V], StepWindow[V], int, V], Optional[V]]
"""``(step, window, tangents, lag, direction) ->`` the derivative of a
step term's :math:`\\partial Q/\\partial y_{n+1-\\mathrm{lag}}` along the
tangent."""

StepParamCurvatureFn = Callable[[int, StepWindow[V], StepWindow[V], V], Optional[V]]
"""``(step, window, tangents, direction) ->`` the derivative of a step
term's :math:`\\partial Q/\\partial p` along the tangent."""


@dataclass(frozen=True)
class StateTermDerivatives(Generic[V]):
    r"""A state term's optional second derivatives applied to vectors,
    for Hessian-vector products.

    The second derivative in the states is block diagonal over the stored
    states (:math:`\partial^2 Q/\partial y_m \partial y_k = 0` for
    :math:`m \ne k`), as for terms that sum or integrate over the states.
    Build with :meth:`first_order` or :meth:`second_order`.

    Parameters
    ----------
    state_state : callable, optional
        :math:`(\partial^2 Q/\partial y_m^2)\, w`, state to state.
    state_param : callable, optional
        :math:`(\partial^2 Q/\partial y_m \partial p)\, v`, parameters to
        state.
    param_state : callable, optional
        :math:`(\partial^2 Q/\partial p\, \partial y_m)\, w`, state to
        parameters.
    param_param : callable, optional
        :math:`(\partial^2 Q/\partial p^2)\, v`, parameters to parameters.

    Raises
    ------
    ValueError
        If some but not all four are given: the second-order sweep needs
        every one.
    """

    state_state: Optional[StateCurvatureFn[V]] = None
    state_param: Optional[StateCurvatureFn[V]] = None
    param_state: Optional[StateCurvatureFn[V]] = None
    param_param: Optional[ParamCurvatureFn[V]] = None

    def __post_init__(self) -> None:
        given = [
            field is not None
            for field in (
                self.state_state,
                self.state_param,
                self.param_state,
                self.param_param,
            )
        ]
        if any(given) and not all(given):
            raise ValueError("a state term's second derivatives come all four or none")

    @staticmethod
    def first_order() -> "StateTermDerivatives[V]":
        """No second derivatives: gradients only."""
        return StateTermDerivatives()

    @staticmethod
    def second_order(
        state_state: StateCurvatureFn[V],
        state_param: StateCurvatureFn[V],
        param_state: StateCurvatureFn[V],
        param_param: ParamCurvatureFn[V],
    ) -> "StateTermDerivatives[V]":
        """All four second derivatives."""
        return StateTermDerivatives(state_state, state_param, param_state, param_param)


@dataclass(frozen=True)
class StepTermDerivatives(Generic[V]):
    r"""A step term's optional derivatives along a tangent, for
    Hessian-vector products: each is the derivative of the matching
    cotangent of :class:`StepTermProtocol` along :math:`(dX, v)`, with
    :math:`dX` the tangents of the step's arguments read from
    ``tangents`` (the tangent trajectory's window of the step:
    :math:`\delta u_n` as its unknown, :math:`w_{n+1-l}` as its past
    states); a call returns None for a zero term. Build with
    :meth:`first_order` or :meth:`second_order`.

    Parameters
    ----------
    curvature_wrt_unknown : callable, optional
        Along the tangent, :math:`\partial Q/\partial u_n`'s derivative,
        an unknown-space column.
    curvature_wrt_lag : callable, optional
        :math:`\partial Q/\partial y_{n+1-\mathrm{lag}}`'s, a state column.
    curvature_wrt_param : callable, optional
        :math:`\partial Q/\partial p`'s through the step, a parameter
        column.

    Raises
    ------
    ValueError
        If some but not all three are given.
    """

    curvature_wrt_unknown: Optional[StepUnknownCurvatureFn[V]] = None
    curvature_wrt_lag: Optional[StepLagCurvatureFn[V]] = None
    curvature_wrt_param: Optional[StepParamCurvatureFn[V]] = None

    def __post_init__(self) -> None:
        given = [
            field is not None
            for field in (
                self.curvature_wrt_unknown,
                self.curvature_wrt_lag,
                self.curvature_wrt_param,
            )
        ]
        if any(given) and not all(given):
            raise ValueError("a step term's curvatures come all three or none")

    @staticmethod
    def first_order() -> "StepTermDerivatives[V]":
        """No derivatives along a tangent: gradients only."""
        return StepTermDerivatives()

    @staticmethod
    def second_order(
        curvature_wrt_unknown: StepUnknownCurvatureFn[V],
        curvature_wrt_lag: StepLagCurvatureFn[V],
        curvature_wrt_param: StepParamCurvatureFn[V],
    ) -> "StepTermDerivatives[V]":
        """All three derivatives along a tangent."""
        return StepTermDerivatives(
            curvature_wrt_unknown, curvature_wrt_lag, curvature_wrt_param
        )


@runtime_checkable
class StateTermProtocol(Protocol[V]):
    r"""A term of a scalar functional read from the stored states and the
    parameters directly, :math:`Q_y(y_0, \dots, y_N;\, p)`, on one
    recorded trajectory: its first derivatives, and its optional second
    derivatives in a bundle.

    A returned vector may be the term's own storage, valid until its next
    call: the sweeps copy whatever they keep, combine or write into, and
    never modify a returned vector. None for a derivative that is zero
    (a state the term does not read).
    """

    def state_cotangent(self, index: int) -> Optional[V]:
        r""":math:`\partial Q/\partial y_{\mathrm{index}}`, a single column
        of the state space."""
        ...

    def param_cotangent(self) -> Optional[V]:
        r"""The direct :math:`\partial Q/\partial p`, a single column of
        the parameter space."""
        ...

    def derivatives(self) -> StateTermDerivatives[V]:
        """The optional second derivatives."""
        ...


@runtime_checkable
class StepTermProtocol(Protocol[V]):
    r"""A term of a scalar functional read from each step,
    :math:`\sum_n Q_n(u_n;\, y_n, \dots, y_{n+1-k};\, p)` (a functional
    integrating over time reads the step's stages), on one recorded
    trajectory: its first derivatives at the step's window, and its
    optional derivatives along a tangent in a bundle.

    Each is :math:`\partial Q/\partial(\cdot)` through step :math:`n`'s
    part only; None for a derivative that is zero. A returned vector is
    new or the term's own storage, valid until its next call, as for
    :class:`StateTermProtocol`.
    """

    def cotangent_wrt_unknown(
        self, step: int, window: StepWindow[V]
    ) -> Optional[BlockVector[V]]:
        r""":math:`\partial Q/\partial u_n`, a single column of the
        stepper's unknown space."""
        ...

    def cotangent_wrt_lag(
        self, step: int, window: StepWindow[V], lag: int
    ) -> Optional[V]:
        r""":math:`\partial Q/\partial y_{n+1-\mathrm{lag}}` through the
        step, a single column of the state space."""
        ...

    def cotangent_wrt_param(self, step: int, window: StepWindow[V]) -> Optional[V]:
        r""":math:`\partial Q/\partial p` through the step, a single column
        of the parameter space."""
        ...

    def derivatives(self) -> StepTermDerivatives[V]:
        """The optional derivatives along a tangent."""
        ...


@dataclass(frozen=True)
class TrajectoryFunctional(Generic[V]):
    r"""A scalar functional of one recorded trajectory as the sweeps read
    it: a sum of terms read from the stored states and terms read from
    the steps, :math:`Q = \sum Q_y + \sum \sum_n Q_n`.

    A functional made of several (a misfit and a regularization, a time
    integral and an endpoint value) is the sum of theirs, ``a + b``; the
    sweeps add the terms' derivatives, so no term knows of another.

    Parameters
    ----------
    state_terms : tuple of StateTermProtocol
        The terms read from the stored states and the parameters.
    step_terms : tuple of StepTermProtocol
        The terms read from the steps.

    Raises
    ------
    TypeError
        If a term does not satisfy its protocol.
    """

    state_terms: Tuple[StateTermProtocol[V], ...] = ()
    step_terms: Tuple[StepTermProtocol[V], ...] = ()

    def __post_init__(self) -> None:
        for state_term in self.state_terms:
            if not isinstance(state_term, StateTermProtocol):
                raise TypeError(
                    "each state term must satisfy StateTermProtocol, got "
                    f"{type(state_term).__name__}"
                )
        for step_term in self.step_terms:
            if not isinstance(step_term, StepTermProtocol):
                raise TypeError(
                    "each step term must satisfy StepTermProtocol, got "
                    f"{type(step_term).__name__}"
                )

    def __add__(self, other: "TrajectoryFunctional[V]") -> "TrajectoryFunctional[V]":
        """The sum of the two functionals: their terms together."""
        return TrajectoryFunctional(
            self.state_terms + other.state_terms, self.step_terms + other.step_terms
        )


def _accumulate(
    total: Optional[W], term: Optional[W], space: VectorSpaceProtocol[W]
) -> Optional[W]:
    """``total + term``, into ``total`` when there is one (a new vector the
    caller owns); a term starting the total is copied, since it may be a
    term's own storage, overwritten by that term's next call. So each
    term is added as soon as it is returned, never kept for later."""
    if term is None:
        return total
    if total is None:
        return space.copy(term)
    return space.axpby(1.0, term, 1.0, total, out=total)


@runtime_checkable
class InitialStateDerivativeProtocol(Protocol[V]):
    r""":math:`\partial \hat y_0/\partial p`, for an initial state that
    depends on the parameters."""

    def param_derivative(self) -> DerivativeOperator[V]:
        """Parameters to state, valid for the parameters currently set;
        a zero operator declares ``zero`` and is skipped."""
        ...

    def curvature(self, weight: V) -> Optional[DerivativeOperator[V]]:
        r""":math:`\sum_i w_i\, \partial^2 \hat y_{0,i}/\partial p^2`,
        parameters to parameters, for a state column ``weight``; None when
        it is not available (no Hessian-vector products), and a zero
        operator declares ``zero`` and is skipped."""
        ...


class ResidualInitialState(Generic[V]):
    r""":math:`\partial \hat y_0/\partial p` and its curvature from the
    residual's derivatives bundle (its ``initial_state`` and
    ``initial_state_curvature`` fields).

    Parameters
    ----------
    rhs : RightHandSideProtocol
        The residual; its bundle is fetched on every call, so it is valid
        for the parameters currently set.
    form : ResidualFormProtocol
        Converts a stack vector to the residual's form for the
        curvature's weight.
    """

    def __init__(
        self,
        rhs: RightHandSideProtocol[V, FormatProtocol],
        form: ResidualFormProtocol[V, V],
    ) -> None:
        self._rhs, self._form = rhs, form

    def curvature(self, weight: V) -> Optional[DerivativeOperator[V]]:
        """The bundle's ``initial_state_curvature`` at ``weight``; None
        when the bundle has none."""
        field = self._rhs.derivatives().initial_state_curvature
        return None if field is None else field(self._form.to_residual(weight))

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

    The second-order sweep keeps the derivatives :math:`\dot{\bar\nu}_m`
    the same way, with its own direct term.

    Parameters
    ----------
    direct : callable
        ``index ->`` the direct term of state ``index``'s cotangent, a new
        state column the cotangent takes over (:math:`\partial
        Q/\partial y_m` for the first order), or None when it is zero.
    space : InnerProductSpaceProtocol
        The state space.
    """

    def __init__(
        self,
        direct: Callable[[int], Optional[V]],
        space: InnerProductSpaceProtocol[V, ArrayProtocol],
    ) -> None:
        self._direct, self._space = direct, space
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
        contributed; its contributions are released. Always a vector the
        caller owns and may write into."""
        direct = self._direct(index)
        later = self._pending.pop(index, None)
        if direct is None:
            return self._space.zeros() if later is None else later
        if later is None:
            return direct
        return self._space.axpby(1.0, direct, 1.0, later, out=later)

    def npending(self) -> int:
        """The number of states with contributions held."""
        return len(self._pending)


@dataclass(frozen=True)
class TangentState(Generic[V]):
    r"""The tangent of one stored state :math:`y_m` along :math:`v`.

    Attributes
    ----------
    unconstrained : V
        :math:`dx_m`, the tangent of the value the constraint was applied
        to (:math:`\hat y_0` or :math:`E u_{m-1}`).
    state : V
        :math:`w_m`, the tangent of the stored state.
    unknown : BlockVector, optional
        :math:`\delta u_{m-1}`, the tangent of the producing step's
        unknown; None for :math:`y_0`.
    """

    unconstrained: V
    state: V
    unknown: Optional[BlockVector[V]]


@dataclass(frozen=True)
class _StepSecondOrder(Generic[V]):
    """The stepper bundle's second-order fields and its declared blocks."""

    state_state: StateStateFn[V]
    state_param: StateParamFn[V]
    param_param: ParamParamFn[V]
    state_pairs: FrozenSet[Tuple[int, int]]
    param_lags: FrozenSet[int]


@dataclass(frozen=True)
class _ConstraintSecondOrder(Generic[V]):
    r"""The constraint's three :math:`\bar\nu`-weighted second
    derivatives at one stored state."""

    state_state: DerivativeOperator[V]
    state_param: DerivativeOperator[V]
    param_param: DerivativeOperator[V]


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

        The window reads the stepper's depth of past states, or all the
        states before the step when fewer precede it: a multistep method's
        startup steps read the history there is, and every lag loop runs
        over ``len(window.past_states)``.

        Raises
        ------
        ValueError
            If the window's parameter version is not the stepper's.
        """
        depth = min(self._stepper.depth(), step + 1)
        window = store.window(step, depth)
        current = self._stepper.parameters_version()
        if window.parameters_version != current:
            raise ValueError(
                f"step {step} was recorded at parameter version "
                f"{window.parameters_version}, not the current {current}: "
                "solve again before differentiating"
            )
        return window

    def _session(
        self, session: Optional[DifferentiableSessionProtocol[V]]
    ) -> DifferentiableSessionProtocol[V]:
        """``session``, or a new one of the stepper for this call."""
        return self._stepper.session() if session is None else session

    def _transposed_handle(
        self, window: StepWindow[V], session: DifferentiableSessionProtocol[V]
    ) -> SolverHandleProtocol[BlockVector[V]]:
        """A solver handle for the transposed diagonal block, set up
        through ``session``.

        Raises
        ------
        TypeError
            If the stepper's handle cannot solve transposed.
        """
        transposed = session.diag_handle(window).transpose()
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
        functional: TrajectoryFunctional[V],
        direction: V,
        session: Optional[DifferentiableSessionProtocol[V]] = None,
    ) -> A:
        r""":math:`dQ/dp \cdot v`, by one forward sweep.

        Parameters
        ----------
        store : TrajectoryStoreProtocol
            A trajectory recorded at the parameters currently set.
        functional : TrajectoryFunctional
            :math:`Q`'s terms on that trajectory.
        direction : V
            :math:`v`, a single column of the parameter space.
        session : DifferentiableSessionProtocol, optional
            Sets up the step solves; a new one of the stepper by default,
            so steps whose diagonal block repeats share a setup.

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
        space, params = self._state_space(), self._param_space()
        products: List[A] = []
        for term in functional.state_terms:
            by_param = term.param_cotangent()
            if by_param is not None:
                products.append(params.dot(by_param, direction))
        # The tangents of the states the next step reads, most recent first.
        recent: Deque[V] = deque(maxlen=self._stepper.depth())
        states = self._tangent_states(store, direction, self._session(session))
        for index, tangent in enumerate(states):
            for term in functional.state_terms:
                by_state = term.state_cotangent(index)
                if by_state is not None:
                    products.append(space.dot(by_state, tangent.state))
            if functional.step_terms and tangent.unknown is not None:
                products += self._step_terms_tangent(
                    functional.step_terms,
                    store,
                    index - 1,
                    tangent.unknown,
                    recent,
                    direction,
                )
            recent.appendleft(tangent.state)
        total = params.dot(params.zeros(), direction)
        for product in products:
            total = total + product
        return total

    def _step_terms_tangent(
        self,
        step_terms: Sequence[StepTermProtocol[V]],
        store: TrajectoryStoreProtocol[V],
        step: int,
        unknown: BlockVector[V],
        recent: Sequence[V],
        direction: V,
    ) -> List[A]:
        r"""Step ``step``'s terms along the tangent, those present of
        :math:`\langle \partial Q/\partial u_n, \delta u_n\rangle`,
        :math:`\langle \partial Q/\partial y_{n+1-l}, w_{n+1-l}\rangle` and
        :math:`\langle \partial Q/\partial p, v\rangle` through the step
        for each term, ``recent[l - 1]`` being :math:`w_{n+1-l}`."""
        window = self._window(store, step)
        space, params = self._state_space(), self._param_space()
        unknowns = self._stepper.unknown_space()
        products: List[A] = []
        for term in step_terms:
            by_unknown = term.cotangent_wrt_unknown(step, window)
            if by_unknown is not None:
                products.append(unknowns.dot(by_unknown, unknown))
            for lag in range(1, len(window.past_states) + 1):
                by_lag = term.cotangent_wrt_lag(step, window, lag)
                if by_lag is not None:
                    products.append(space.dot(by_lag, recent[lag - 1]))
            by_param = term.cotangent_wrt_param(step, window)
            if by_param is not None:
                products.append(params.dot(by_param, direction))
        return products

    def _tangent_states(
        self,
        store: TrajectoryStoreProtocol[V],
        direction: V,
        session: DifferentiableSessionProtocol[V],
    ) -> Iterator[TangentState[V]]:
        r"""The tangent sweep, one stored state at a time from :math:`y_0`:
        :meth:`tangent_initial`, then :meth:`tangent_step` per step.

        Parameters
        ----------
        store : TrajectoryStoreProtocol
            A trajectory recorded at the parameters currently set.
        direction : V
            :math:`v`, a single column of the parameter space.
        session : DifferentiableSessionProtocol
            Sets up the step solves.
        """
        initial = self.tangent_initial(store, direction)
        yield initial
        # The tangents of the states the next step reads, most recent
        # first: recent[l - 1] is w_{n+1-l}.
        recent: Deque[V] = deque([initial.state], maxlen=self._stepper.depth())
        for n in range(store.nsteps()):
            tangent = self.tangent_step(
                store, n, self._window(store, n), tuple(recent), direction, session
            )
            yield tangent
            recent.appendleft(tangent.state)

    def tangent_initial(
        self, store: TrajectoryStoreProtocol[V], direction: V
    ) -> TangentState[V]:
        r"""The tangent of :math:`y_0`: :math:`dx_0 = (\partial \hat
        y_0/\partial p)\, v` and :math:`w_0 = \Pi' dx_0 + \Pi_p v`.

        Parameters
        ----------
        store : TrajectoryStoreProtocol
            The trajectory (for the constraint's linearization point).
        direction : V
            :math:`v`, a single column of the parameter space.
        """
        dx0 = (
            self._state_space().zeros()
            if self._initial_state is None
            else self._initial_state.param_derivative().apply(direction)
        )
        return TangentState(
            dx0, self._constrained_tangent(store, 0, dx0, direction), None
        )

    def tangent_step(
        self,
        store: TrajectoryStoreProtocol[V],
        step: int,
        window: StepWindow[V],
        recent: Sequence[V],
        direction: V,
        session: Optional[DifferentiableSessionProtocol[V]] = None,
    ) -> TangentState[V]:
        r"""One step of the tangent sweep: from the tangents of the states
        step ``step`` reads to the tangent of the state it produces,

        .. math::

            \frac{\partial R_n}{\partial u_n}\, \delta u_n
            = -\Big(\sum_l \frac{\partial R_n}{\partial y_{n+1-l}}\,
            w_{n+1-l} + \frac{\partial R_n}{\partial p}\, v\Big), \quad
            dx_{n+1} = E\, \delta u_n, \quad
            w_{n+1} = \Pi' dx_{n+1} + \Pi_p v.

        The tangent sweep is a loop over this; a store that recomputes
        tangents from checkpoints calls it to replay a segment, with the
        history it keeps.

        Parameters
        ----------
        store : TrajectoryStoreProtocol
            The trajectory (for the constraint's linearization point).
        step : int
            :math:`n`.
        window : StepWindow
            Step ``step``'s window.
        recent : sequence of V
            The tangents of the states the step reads, most recent first:
            ``recent[l - 1]`` is :math:`w_{n+1-l}`, one per past state of
            the window.
        direction : V
            :math:`v`, a single column of the parameter space.
        session : DifferentiableSessionProtocol, optional
            Sets up the step's solve; the stepper's own, of this one step,
            by default.

        Raises
        ------
        ValueError
            If ``recent`` does not match the window's past states.
        SolveFailedError
            If the step's solve fails.
        """
        if len(recent) != len(window.past_states):
            raise ValueError(
                f"step {step} reads {len(window.past_states)} states, but "
                f"{len(recent)} tangents were given"
            )
        stepper = self._stepper
        off_diag, param = self._first_order()
        residuals = stepper.residual_space()
        # rhs = -(sum_l dR/dy_{n+1-l} w_{n+1-l} + dR/dp v)
        rhs = param(window).apply(BlockVector((direction,)))
        for lag, w in enumerate(recent, start=1):
            coupled = off_diag(window, lag).apply(BlockVector((w,)))
            rhs = residuals.axpby(1.0, coupled, 1.0, rhs, out=rhs)
        rhs = residuals.scale(-1.0, rhs, out=rhs)
        handle = self._session(session).diag_handle(window)
        delta = self._solve(handle, rhs, f"tangent sweep, step {step}")
        dx = stepper.readout().apply(delta).blocks[0]
        w = self._constrained_tangent(store, step + 1, dx, direction)
        return TangentState(dx, w, delta)

    # ------------------------------------------------------------------
    # The adjoint sweeps: one recurrence (_AdjointSweep), with the first-
    # or second-order sources
    # ------------------------------------------------------------------

    def stepper(self) -> DifferentiableStepperProtocol[V, A]:
        """The stepper."""
        return self._stepper

    def initial_state(self) -> Optional[InitialStateDerivativeProtocol[V]]:
        r""":math:`\partial \hat y_0/\partial p`, or None."""
        return self._initial_state

    def _step_second_order(self) -> _StepSecondOrder[V]:
        """The stepper bundle's second-order fields.

        Raises
        ------
        TypeError
            If the bundle has none.
        """
        derivs = self._stepper.derivatives()
        if (
            derivs.state_state is None
            or derivs.state_param is None
            or derivs.param_param is None
        ):
            raise TypeError(
                f"{type(self._stepper).__name__}'s derivatives() has no second "
                "derivatives, so it has no Hessian-vector products"
            )
        return _StepSecondOrder(
            derivs.state_state,
            derivs.state_param,
            derivs.param_param,
            derivs.state_pairs,
            derivs.param_lags,
        )

    def _constraint_second_order(
        self, store: TrajectoryStoreProtocol[V], index: int, cotangent: V
    ) -> Optional[_ConstraintSecondOrder[V]]:
        r"""The constraint's second derivatives at stored state ``index``,
        weighted by ``cotangent`` (:math:`\bar\nu_m`); None without a
        constraint.

        Raises
        ------
        TypeError
            If the constraint's bundle has no second derivatives.
        """
        if self._constraint is None:
            return None
        derivs = self._constraint.derivatives()
        if (
            derivs.state_state is None
            or derivs.state_param is None
            or derivs.param_param is None
        ):
            raise TypeError(
                f"{type(self._constraint).__name__}'s derivatives() has no "
                "second derivatives, so it has no Hessian-vector products"
            )
        point, time = store.unconstrained(index), store.time(index)
        return _ConstraintSecondOrder(
            derivs.state_state(point, time, cotangent),
            derivs.state_param(point, time, cotangent),
            derivs.param_param(point, time, cotangent),
        )

    def gradient(
        self,
        store: TrajectoryStoreProtocol[V],
        functional: TrajectoryFunctional[V],
        session: Optional[DifferentiableSessionProtocol[V]] = None,
    ) -> V:
        r""":math:`dQ/dp`, by one backward sweep.

        Parameters
        ----------
        store : TrajectoryStoreProtocol
            A trajectory recorded at the parameters currently set.
        functional : TrajectoryFunctional
            :math:`Q`'s terms on that trajectory.
        session : DifferentiableSessionProtocol, optional
            Sets up the step solves; a new one of the stepper by default,
            so steps whose diagonal block repeats share a setup.

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
        sources = _FirstOrderSources(functional, self)
        handles = _SharedHandles(self, self._session(session))
        sweep = _AdjointSweep(self, store, sources, handles)
        for n in reversed(range(store.nsteps())):
            sweep.step(n, self._window(store, n))
        return sweep.finish()

    def hvp(
        self,
        store: TrajectoryStoreProtocol[V],
        functional: TrajectoryFunctional[V],
        direction: V,
        tangent_store: Optional[TrajectoryStoreProtocol[V]] = None,
        session: Optional[DifferentiableSessionProtocol[V]] = None,
    ) -> V:
        r""":math:`(d^2 Q/dp^2)\, v`; see :meth:`gradient_and_hvp`, which
        also returns the gradient at no extra cost."""
        return self.gradient_and_hvp(
            store, functional, direction, tangent_store, session
        )[1]

    def gradient_and_hvp(
        self,
        store: TrajectoryStoreProtocol[V],
        functional: TrajectoryFunctional[V],
        direction: V,
        tangent_store: Optional[TrajectoryStoreProtocol[V]] = None,
        session: Optional[DifferentiableSessionProtocol[V]] = None,
    ) -> tuple[V, V]:
        r""":math:`dQ/dp` and :math:`(d^2 Q/dp^2)\, v`: the forward tangent
        sweep, then the backward first- and second-order adjoint sweeps in
        lockstep; the first-order sweep's total is the gradient.

        The tangent sweep records the tangent trajectory into
        ``tangent_store`` as a trajectory of its own: :math:`w_m` as the
        states, :math:`dx_m` as the unconstrained values and
        :math:`\delta u_n` as the step unknowns, so a tangent window holds
        exactly the step's :math:`dX_b`. At each step, backward, the
        first-order sweep solves for :math:`\mu_n`, then the second-order
        sweep, whose sources read :math:`\mu_n` and the tangent store,
        solves for :math:`\dot\mu_n` with the same transposed handle.

        Parameters
        ----------
        store : TrajectoryStoreProtocol
            A trajectory recorded at the parameters currently set.
        functional : TrajectoryFunctional
            :math:`Q`'s terms on that trajectory; each term's bundle must
            carry its second derivatives.
        direction : V
            :math:`v`, a single column of the parameter space.
        tangent_store : TrajectoryStoreProtocol, optional
            Empty; receives the tangent trajectory, which the backward
            sweeps read in reverse step order. In memory by default.
        session : DifferentiableSessionProtocol, optional
            Sets up the step solves of all three sweeps; a new one of the
            stepper by default, so steps whose diagonal block repeats
            share a setup, across the sweeps too.

        Returns
        -------
        tuple
            The gradient and the product, single columns of the parameter
            space.

        Raises
        ------
        TypeError
            If a term's bundle has no second derivatives, or a derivative
            the sweeps need is missing.
        ValueError
            If the trajectory was recorded at other parameters, or
            ``tangent_store`` is not empty.
        SolveFailedError
            If a step's solve fails.
        """
        # Read every term's bundle once, refusing before any sweep runs.
        curvatures = _FunctionalCurvatures.of(functional)
        tangents: TrajectoryStoreProtocol[V] = (
            InMemoryTrajectoryStore() if tangent_store is None else tangent_store
        )
        shared = self._session(session)
        self._record_tangents(store, direction, tangents, shared)
        handles = _SharedHandles(self, shared)
        adjoint = _AdjointSweep(
            self, store, _FirstOrderSources(functional, self), handles
        )
        second_sources = _SecondOrderSources(
            self, store, curvatures, tangents, direction, adjoint
        )
        second_adjoint = _AdjointSweep(self, store, second_sources, handles)
        for n in reversed(range(store.nsteps())):
            # One window for both sweeps; the second-order sources read
            # this step's mu_n, so the first-order sweep goes first.
            window = self._window(store, n)
            adjoint.step(n, window)
            second_adjoint.step(n, window)
        gradient = adjoint.finish()
        return gradient, second_adjoint.finish()

    def _record_tangents(
        self,
        store: TrajectoryStoreProtocol[V],
        direction: V,
        tangents: TrajectoryStoreProtocol[V],
        session: DifferentiableSessionProtocol[V],
    ) -> None:
        """Run the tangent sweep through ``session``, recording it into
        ``tangents`` at the primal trajectory's times.

        Raises
        ------
        ValueError
            If ``tangents`` is not empty.
        """
        if tangents.ntimes() != 0:
            raise ValueError("the tangent store must be empty")
        version = self._stepper.parameters_version()
        states = self._tangent_states(store, direction, session)
        for index, tangent in enumerate(states):
            # Only the initial tangent has no unknown: no step produced it.
            if tangent.unknown is None:
                tangents.append(store.time(index), tangent.state, tangent.unconstrained)
            else:
                # The primal step's own span, so the tangent store's times
                # are the primal's, derived from the same sizes.
                tangents.append_step(
                    store.span(index - 1),
                    tangent.state,
                    tangent.unknown,
                    version,
                    tangent.unconstrained,
                )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._stepper!r})"


class _AdjointSourcesProtocol(Protocol[V]):
    r"""What an adjoint sweep adds to its recurrence (see
    :class:`_AdjointSweep`); None for a term that is absent."""

    def state(self, index: int) -> Optional[V]:
        """The direct term of state ``index``'s cotangent, a new vector the
        sweep takes over."""
        ...

    def param(self) -> V:
        """The direct parameter term; a new vector the sweep accumulates
        into."""
        ...

    def at_state(self, index: int) -> tuple[Optional[V], Optional[V]]:
        """Extra terms once the cotangent of state ``index`` is complete,
        added as it is pulled through the constraint: to the pulled
        cotangent, and to the parameters. Called for states in reverse
        order, so sources reading per-state data read it backward."""
        ...

    def begin_step(self, step: int, window: StepWindow[V]) -> None:
        """Step ``step`` is being swept, at ``window``: called once before
        the step's other hooks, so per-step data (a tangent window) is
        read once."""
        ...

    def step_unknown(
        self, step: int, window: StepWindow[V]
    ) -> Optional[BlockVector[V]]:
        """An extra term of the step solve's right-hand side, in the
        unknown space."""
        ...

    def step_param(self, step: int, window: StepWindow[V]) -> Optional[V]:
        """An extra parameter term of the step."""
        ...

    def step_lag(self, step: int, window: StepWindow[V], lag: int) -> Optional[V]:
        """An extra term of the cotangent of the state read at ``lag``."""
        ...

    def initial(self) -> Optional[V]:
        r"""An extra parameter term at :math:`t_0`."""
        ...


class _FirstOrderAdjointProtocol(Protocol[V]):
    r"""The first-order adjoint as second-order sources read it, by index.

    The live sweep (:class:`_AdjointSweep`) answers for the step it has
    just swept only, so the second-order sweep runs in lockstep with it; a
    record of a finished first-order sweep could answer for any index
    instead, without changing the sources.
    """

    def adjoint(self, step: int) -> BlockVector[V]:
        r""":math:`\mu_n` for :math:`n` = ``step``."""
        ...

    def cotangent(self, index: int) -> V:
        r""":math:`\bar\nu_m` for :math:`m` = ``index``."""
        ...

    def pulled(self, index: int) -> V:
        r""":math:`\bar z_m = \Pi'^\top \bar\nu_m` for :math:`m` =
        ``index``."""
        ...


class _SharedHandles(Generic[V, A]):
    """Each step's transposed handle, set up once for the sweeps running
    in lockstep over that step, and released at the next step; its setup
    goes through ``session``, which may share it across steps too when
    the step's diagonal block repeats.

    Parameters
    ----------
    owner : TrajectoryDerivatives
        Sets up the handles.
    session : DifferentiableSessionProtocol
        The sweeps' session.
    """

    def __init__(
        self,
        owner: TrajectoryDerivatives[V, A],
        session: DifferentiableSessionProtocol[V],
    ) -> None:
        self._owner, self._session = owner, session
        self._step: Optional[int] = None
        self._handle: Optional[SolverHandleProtocol[BlockVector[V]]] = None

    def get(
        self, step: int, window: StepWindow[V]
    ) -> SolverHandleProtocol[BlockVector[V]]:
        """Step ``step``'s transposed handle."""
        if self._handle is None or self._step != step:
            self._handle = self._owner._transposed_handle(window, self._session)
            self._step = step
        return self._handle


class _AdjointSweep(Generic[V, A]):
    r"""The backward recurrence shared by the first- and second-order
    adjoints, run one step at a time.

    With :math:`\bar\nu_m` the cotangent of stored state :math:`m` and
    :math:`\bar z_m` its pull through the constraint, each step :math:`n`
    (from the last) does

    .. math::

        \bar z_{n+1} = \Pi'^\top \bar\nu_{n+1} + c_z, \qquad
        g \mathrel{+}= \Pi_p^\top \bar\nu_{n+1} + c_p,

        \Big(\frac{\partial R_n}{\partial u_n}\Big)^\top \mu_n
        = -\big(E^\top \bar z_{n+1} + s_u\big),

        g \mathrel{+}= \Big(\frac{\partial R_n}{\partial p}\Big)^\top \mu_n
        + s_p, \qquad
        \bar\nu_{n+1-l} \mathrel{+}= \Big(\frac{\partial R_n}{\partial
        y_{n+1-l}}\Big)^\top \mu_n + s_l,

    and :meth:`finish` pulls :math:`\bar\nu_0` through the constraint and
    :math:`\partial \hat y_0/\partial p`. The sources (the direct terms
    and :math:`c, s`) come from a :class:`_AdjointSourcesProtocol`: the
    functional's first derivatives for the gradient, the curvature terms
    for the second-order adjoint. Every operation is a linear operator
    or its transpose at the recorded trajectory, or a transposed solve.

    Parameters
    ----------
    owner : TrajectoryDerivatives
        The stepper, constraint and initial state.
    store : TrajectoryStoreProtocol
        The trajectory.
    sources : _AdjointSourcesProtocol
        What this adjoint adds.
    handles : _SharedHandles
        The steps' transposed handles, shared with a sweep in lockstep.
    """

    def __init__(
        self,
        owner: TrajectoryDerivatives[V, A],
        store: TrajectoryStoreProtocol[V],
        sources: _AdjointSourcesProtocol[V],
        handles: _SharedHandles[V, A],
    ) -> None:
        self._owner, self._store = owner, store
        self._sources, self._handles = sources, handles
        self._off_diag, self._param = owner._first_order()
        self._space, self._params = owner._state_space(), owner._param_space()
        self._cotangents = _Cotangents(sources.state, self._space)
        self._total = sources.param()
        # The latest step adjoint, completed cotangent and its pull, with
        # their indices, for a second-order sweep in lockstep to read.
        self._adjoint: Optional[tuple[int, BlockVector[V]]] = None
        self._cotangent: Optional[tuple[int, V]] = None
        self._pulled: Optional[tuple[int, V]] = None

    @staticmethod
    def _latest(held: Optional[tuple[int, W]], index: int, what: str) -> W:
        """``held``'s value when it is for ``index``: a live sweep holds
        only its latest step.

        Raises
        ------
        LookupError
            If ``held`` is for another index, or nothing is held.
        """
        if held is None or held[0] != index:
            latest = None if held is None else held[0]
            raise LookupError(
                f"{what} {index} is not held: this sweep holds only its latest "
                f"({latest}); read it in lockstep"
            )
        return held[1]

    def adjoint(self, step: int) -> BlockVector[V]:
        r""":math:`\mu_n` for :math:`n` = ``step``, the latest step."""
        return self._latest(self._adjoint, step, "the adjoint of step")

    def cotangent(self, index: int) -> V:
        r""":math:`\bar\nu_m` for :math:`m` = ``index``, the latest
        completed."""
        return self._latest(self._cotangent, index, "the cotangent of state")

    def pulled(self, index: int) -> V:
        r""":math:`\bar z_m` for :math:`m` = ``index``, the latest
        pulled."""
        return self._latest(self._pulled, index, "the pulled cotangent of state")

    def _add(self, term: Optional[V]) -> None:
        if term is not None:
            self._total = self._params.axpby(
                1.0, term, 1.0, self._total, out=self._total
            )

    def _pull_constraint(self, index: int) -> V:
        r"""Complete :math:`\bar\nu_m` and pull it through the constraint
        at stored state ``index``, adding the parameter terms."""
        cotangent = self._cotangents.complete(index)
        self._cotangent = (index, cotangent)
        pulled = cotangent
        terms = self._owner._constraint_terms(self._store, index)
        if terms is not None:
            tangent, param = terms
            pulled = tangent.apply_transpose(cotangent)
            if not param.properties().zero:
                self._add(param.apply_transpose(cotangent))
        extra_state, extra_param = self._sources.at_state(index)
        if extra_state is not None:
            pulled = self._space.axpby(1.0, extra_state, 1.0, pulled, out=pulled)
        self._add(extra_param)
        self._pulled = (index, pulled)
        return pulled

    def step(self, step: int, window: StepWindow[V]) -> None:
        """Sweep step ``step`` at its ``window``; steps go from the last
        to the first. The caller fetches the window once for every sweep
        running in lockstep."""
        owner, sources = self._owner, self._sources
        stepper = owner.stepper()
        sources.begin_step(step, window)
        pulled = self._pull_constraint(step + 1)
        # (dR_n/du_n)^T mu_n = -(E^T z_{n+1} + s_u)
        unknowns = stepper.unknown_space()
        rhs = stepper.readout().apply_transpose(BlockVector((pulled,)))
        source = sources.step_unknown(step, window)
        if source is not None:
            rhs = unknowns.axpby(1.0, source, 1.0, rhs, out=rhs)
        rhs = unknowns.scale(-1.0, rhs, out=rhs)
        mu = owner._solve(
            self._handles.get(step, window), rhs, f"adjoint sweep, step {step}"
        )
        self._adjoint = (step, mu)
        self._add(self._param(window).apply_transpose(mu).blocks[0])
        self._add(sources.step_param(step, window))
        # The states this step read: a startup step reads fewer than the
        # stepper's depth.
        for lag in range(1, len(window.past_states) + 1):
            coupled = self._off_diag(window, lag).apply_transpose(mu).blocks[0]
            extra = sources.step_lag(step, window, lag)
            if extra is not None:
                coupled = self._space.axpby(1.0, extra, 1.0, coupled, out=coupled)
            self._cotangents.add(step + 1 - lag, coupled)

    def finish(self) -> V:
        r"""Pull :math:`\bar\nu_0` back to the parameters; returns the
        accumulated total (the gradient, or the Hessian-vector product)."""
        pulled = self._pull_constraint(0)
        initial = self._owner.initial_state()
        if initial is not None:
            derivative = initial.param_derivative()
            if not derivative.properties().zero:
                self._add(derivative.apply_transpose(pulled))
        self._add(self._sources.initial())
        return self._total


@dataclass(frozen=True)
class _StateCurvature(Generic[V]):
    """A state term's second derivatives, read from its bundle and known
    present."""

    state_state: StateCurvatureFn[V]
    state_param: StateCurvatureFn[V]
    param_state: StateCurvatureFn[V]
    param_param: ParamCurvatureFn[V]


@dataclass(frozen=True)
class _StepCurvature(Generic[V]):
    """A step term's derivatives along a tangent, read from its bundle
    and known present."""

    wrt_unknown: StepUnknownCurvatureFn[V]
    wrt_lag: StepLagCurvatureFn[V]
    wrt_param: StepParamCurvatureFn[V]


@dataclass(frozen=True)
class _FunctionalCurvatures(Generic[V]):
    """Every term's second derivatives, read once from the bundles."""

    states: Tuple[_StateCurvature[V], ...]
    steps: Tuple[_StepCurvature[V], ...]

    @staticmethod
    def of(functional: TrajectoryFunctional[V]) -> "_FunctionalCurvatures[V]":
        """Read the terms' bundles.

        Raises
        ------
        TypeError
            If a term's bundle has no second derivatives.
        """
        states = []
        for state_term in functional.state_terms:
            bundle = state_term.derivatives()
            if (
                bundle.state_state is None
                or bundle.state_param is None
                or bundle.param_state is None
                or bundle.param_param is None
            ):
                raise TypeError(
                    f"{type(state_term).__name__}'s derivatives() has no second "
                    "derivatives, so the functional has no Hessian-vector products"
                )
            states.append(
                _StateCurvature(
                    bundle.state_state,
                    bundle.state_param,
                    bundle.param_state,
                    bundle.param_param,
                )
            )
        steps = []
        for step_term in functional.step_terms:
            step_bundle = step_term.derivatives()
            if (
                step_bundle.curvature_wrt_unknown is None
                or step_bundle.curvature_wrt_lag is None
                or step_bundle.curvature_wrt_param is None
            ):
                raise TypeError(
                    f"{type(step_term).__name__}'s derivatives() has no curvature, "
                    "so the functional has no Hessian-vector products"
                )
            steps.append(
                _StepCurvature(
                    step_bundle.curvature_wrt_unknown,
                    step_bundle.curvature_wrt_lag,
                    step_bundle.curvature_wrt_param,
                )
            )
        return _FunctionalCurvatures(tuple(states), tuple(steps))


class _FirstOrderSources(Generic[V, A]):
    """The gradient's sources: the terms' first derivatives, summed; those
    read from each step added to the step's terms.

    Parameters
    ----------
    functional : TrajectoryFunctional
        :math:`Q`'s terms.
    owner : TrajectoryDerivatives
        The spaces.
    """

    def __init__(
        self,
        functional: TrajectoryFunctional[V],
        owner: "TrajectoryDerivatives[V, A]",
    ) -> None:
        self._functional = functional
        self._space, self._params = owner._state_space(), owner._param_space()
        self._unknowns = owner.stepper().unknown_space()

    def state(self, index: int) -> Optional[V]:
        total: Optional[V] = None
        for term in self._functional.state_terms:
            total = _accumulate(total, term.state_cotangent(index), self._space)
        return total

    def param(self) -> V:
        total: Optional[V] = None
        for term in self._functional.state_terms:
            total = _accumulate(total, term.param_cotangent(), self._params)
        return self._params.zeros() if total is None else total

    def at_state(self, index: int) -> tuple[Optional[V], Optional[V]]:
        return None, None

    def begin_step(self, step: int, window: StepWindow[V]) -> None:
        return None

    def step_unknown(
        self, step: int, window: StepWindow[V]
    ) -> Optional[BlockVector[V]]:
        total: Optional[BlockVector[V]] = None
        for term in self._functional.step_terms:
            total = _accumulate(
                total, term.cotangent_wrt_unknown(step, window), self._unknowns
            )
        return total

    def step_param(self, step: int, window: StepWindow[V]) -> Optional[V]:
        total: Optional[V] = None
        for term in self._functional.step_terms:
            total = _accumulate(
                total, term.cotangent_wrt_param(step, window), self._params
            )
        return total

    def step_lag(self, step: int, window: StepWindow[V], lag: int) -> Optional[V]:
        total: Optional[V] = None
        for term in self._functional.step_terms:
            total = _accumulate(
                total, term.cotangent_wrt_lag(step, window, lag), self._space
            )
        return total

    def initial(self) -> Optional[V]:
        return None


class _SecondOrderSources(Generic[V, A]):
    r"""The second-order adjoint's sources: the derivative along
    :math:`v` of every first-order term, given the tangents and the
    first-order sweep running in lockstep.

    Parameters
    ----------
    owner : TrajectoryDerivatives
        The stepper, constraint and initial state.
    store : TrajectoryStoreProtocol
        The trajectory.
    curvatures : _FunctionalCurvatures
        The terms' second derivatives.
    tangents : TrajectoryStoreProtocol
        The tangent trajectory: :math:`w_m` as states, :math:`dx_m` as
        unconstrained values, :math:`\delta u_n` as step unknowns. Read in
        reverse step order only, like the primal store.
    direction : V
        :math:`v`.
    adjoint : _FirstOrderAdjointProtocol
        The first-order adjoint, read by index at the step being swept:
        its :math:`\mu_n`, :math:`\bar\nu_m` and :math:`\bar z_m` weight
        the second derivatives.
    """

    def __init__(
        self,
        owner: TrajectoryDerivatives[V, A],
        store: TrajectoryStoreProtocol[V],
        curvatures: _FunctionalCurvatures[V],
        tangents: TrajectoryStoreProtocol[V],
        direction: V,
        adjoint: _FirstOrderAdjointProtocol[V],
    ) -> None:
        self._owner, self._store, self._curvatures = owner, store, curvatures
        self._tangents, self._direction, self._adjoint = tangents, direction, adjoint
        self._p_dir = BlockVector((direction,))
        self._second = owner._step_second_order()
        self._space, self._params = owner._state_space(), owner._param_space()
        self._unknowns = owner.stepper().unknown_space()
        # The step being swept, its tangent window and its dX_b by lag, set
        # by begin_step.
        self._step: Optional[int] = None
        self._tangent_window: Optional[StepWindow[V]] = None
        self._lag_tangents: Dict[int, BlockVector[V]] = {}

    def state(self, index: int) -> Optional[V]:
        # Q_yy w_m + Q_yp v over the state terms, each added as returned.
        w, v = self._tangents.state(index), self._direction
        total: Optional[V] = None
        for term in self._curvatures.states:
            total = _accumulate(total, term.state_state(index, w), self._space)
            total = _accumulate(total, term.state_param(index, v), self._space)
        return total

    def param(self) -> V:
        # Q_pp v, new: the sweep accumulates into it. The terms' Q_py_m w_m
        # come per state, from at_state, so the tangents are read backward
        # only.
        total: Optional[V] = None
        for term in self._curvatures.states:
            total = _accumulate(total, term.param_param(self._direction), self._params)
        return self._params.zeros() if total is None else total

    def at_state(self, index: int) -> tuple[Optional[V], Optional[V]]:
        # To the parameters: Q_py_m w_m, and with a constraint
        # (nu^T Pi)_px dx + (nu^T Pi)_pp v; to the pulled cotangent, with a
        # constraint, (nu^T Pi)_xx dx + (nu^T Pi)_xp v; nu the first
        # order's cotangent of this state.
        w, v = self._tangents.state(index), self._direction
        to_param: Optional[V] = None
        for term in self._curvatures.states:
            to_param = _accumulate(to_param, term.param_state(index, w), self._params)
        ops = self._owner._constraint_second_order(
            self._store, index, self._adjoint.cotangent(index)
        )
        if ops is None:
            return None, to_param
        dx = self._tangents.unconstrained(index)
        to_state: Optional[V] = None
        for op, x in ((ops.state_state, dx), (ops.state_param, v)):
            if not op.properties().zero:
                to_state = _accumulate(to_state, op.apply(x), self._space)
        if not ops.state_param.properties().zero:
            to_param = _accumulate(
                to_param, ops.state_param.apply_transpose(dx), self._params
            )
        if not ops.param_param.properties().zero:
            to_param = _accumulate(to_param, ops.param_param.apply(v), self._params)
        return to_state, to_param

    def begin_step(self, step: int, window: StepWindow[V]) -> None:
        """Read the tangent store's window of step ``step`` once, at the
        primal window's depth: :math:`dX_b` by lag, the step's unknown at
        0 and the states it read at :math:`l \\ge 1`."""
        tangent_window = self._tangents.window(step, len(window.past_states))
        self._lag_tangents = {0: tangent_window.unknown}
        for lag in range(1, len(window.past_states) + 1):
            self._lag_tangents[lag] = BlockVector((tangent_window.past_state(lag),))
        self._tangent_window = tangent_window
        self._step = step

    def _tangents_of(self, step: int) -> Dict[int, BlockVector[V]]:
        """The :math:`dX_b` of step ``step``, read by :meth:`begin_step`.

        Raises
        ------
        RuntimeError
            If ``step`` is not the step begun.
        """
        if self._step != step:
            raise RuntimeError(f"step {step} was not begun (begun: {self._step})")
        return self._lag_tangents

    def _tangent_window_of(self, step: int) -> StepWindow[V]:
        """Step ``step``'s tangent window, read by :meth:`begin_step`.

        Raises
        ------
        RuntimeError
            If ``step`` is not the step begun.
        """
        if self._step != step or self._tangent_window is None:
            raise RuntimeError(f"step {step} was not begun (begun: {self._step})")
        return self._tangent_window

    def _curvature_at_lag(
        self, step: int, window: StepWindow[V], lag: int
    ) -> Optional[BlockVector[V]]:
        r""":math:`\sum_b H_n(a, b)\, dX_b + H_n(a, p)\, v` for :math:`a` =
        ``lag``, over the declared blocks the step has (a startup step
        reads fewer lags), with :math:`H_n` weighted by the first order's
        :math:`\mu_n`; None when none is declared."""
        second, mu = self._second, self._adjoint.adjoint(step)
        stepper = self._owner.stepper()
        space: BlockInnerProductSpace[V, A] = (
            stepper.unknown_space() if lag == 0 else stepper.state_space()
        )
        lag_tangents = self._tangents_of(step)
        terms: List[BlockVector[V]] = [
            second.state_state(window, mu, lag, other).apply(lag_tangents[other])
            for a, other in sorted(second.state_pairs)
            if a == lag and other in lag_tangents
        ]
        if lag in second.param_lags:
            terms.append(second.state_param(window, mu, lag).apply(self._p_dir))
        total: Optional[BlockVector[V]] = None
        for term in terms:
            total = (
                term if total is None else space.axpby(1.0, term, 1.0, total, out=total)
            )
        return total

    def step_unknown(
        self, step: int, window: StepWindow[V]
    ) -> Optional[BlockVector[V]]:
        # The stepper's curvature (a new vector), then each step term's own
        # added as returned.
        tangents, v = self._tangent_window_of(step), self._direction
        total = self._curvature_at_lag(step, window, 0)
        for term in self._curvatures.steps:
            total = _accumulate(
                total, term.wrt_unknown(step, window, tangents, v), self._unknowns
            )
        return total

    def step_param(self, step: int, window: StepWindow[V]) -> Optional[V]:
        # sum_a H(a, p)^T dX_a + H(p, p) v, and the step terms' own
        second, mu = self._second, self._adjoint.adjoint(step)
        lag_tangents = self._tangents_of(step)
        params = self._params
        total: Optional[V] = second.param_param(window, mu).apply(self._p_dir).blocks[0]
        for lag in sorted(second.param_lags):
            if lag in lag_tangents:
                mixed = second.state_param(window, mu, lag)
                total = _accumulate(
                    total, mixed.apply_transpose(lag_tangents[lag]).blocks[0], params
                )
        tangents, v = self._tangent_window_of(step), self._direction
        for term in self._curvatures.steps:
            total = _accumulate(
                total, term.wrt_param(step, window, tangents, v), params
            )
        return total

    def step_lag(self, step: int, window: StepWindow[V], lag: int) -> Optional[V]:
        read = self._curvature_at_lag(step, window, lag)
        tangents, v = self._tangent_window_of(step), self._direction
        total = None if read is None else read.blocks[0]
        for term in self._curvatures.steps:
            total = _accumulate(
                total, term.wrt_lag(step, window, tangents, lag, v), self._space
            )
        return total

    def initial(self) -> Optional[V]:
        # (sum_i z_0,i d^2 y0_i/dp^2) v, with z_0 the first order's pull.
        initial = self._owner.initial_state()
        if initial is None or initial.param_derivative().properties().zero:
            return None
        curvature = initial.curvature(self._adjoint.pulled(0))
        if curvature is None:
            raise TypeError(
                f"{type(initial).__name__} has no curvature, so its initial "
                "state has no Hessian-vector products"
            )
        if curvature.properties().zero:
            return None
        return curvature.apply(self._direction)
