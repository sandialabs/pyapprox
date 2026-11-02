r"""Functionals integrating over time, with the stepper's own method.

:math:`Q = \int_{t_0}^{t_N} q(y, t)\, dt` is computed as the stepper's
method applied to :math:`\dot z = q(y, t)`, :math:`z(t_0) = 0` (see
:mod:`pyapprox.ode.step_quadrature`): the integral is consistent with the
steps by construction, and no rule is chosen apart from the method.

A :class:`TimeIntegral` is reused across solves; each solve takes a fresh
sink from :meth:`TimeIntegral.sink`, which advances :math:`z` as the steps
arrive, reading each step's nodes from its unknown and the stored states
it read, so the solve keeps no trajectory.

Its derivatives (:meth:`TimeIntegral.on`, for a scalar :math:`q`) are the
recursion's, read per step by the sweeps of
:mod:`pyapprox.ode.trajectory_derivatives`. With step :math:`n`'s
recursion :math:`\alpha^{(n)}_0 z_{n+1} + \sum_{l \ge 1} \alpha^{(n)}_l
z_{n+1-l} = \sum w\, q(\text{node})`, the integral's sensitivity to step
:math:`n`'s right-hand side is :math:`\theta_n = \zeta_{n+1} /
\alpha^{(n)}_0`, where :math:`\zeta_m = \partial Q/\partial z_m` solves
the transposed recursion backward from :math:`\zeta_N = 1`,
:math:`\zeta_m = [m = N] - \sum \theta_n \alpha^{(n)}_l` over the steps
reading :math:`z_m` at lag :math:`l`. The weights depend on the step
sizes only, so :math:`\theta` is computed once per trajectory; it is 1
throughout for a Runge-Kutta or Adams method (:math:`\alpha = (1, -1,
0, \dots)`). Step :math:`n` then contributes :math:`\theta_n w\, \nabla
q(\text{node})` to the derivative in each node's state.
"""

from collections import deque
from dataclasses import dataclass
from typing import (
    Callable,
    Deque,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

from pyapprox.ode.protocols import StepperProtocol
from pyapprox.ode.step_quadrature import StepQuadratureProtocol, StepRule
from pyapprox.ode.step_span import StepSpan
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.trajectory_store import TrajectoryStoreProtocol
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import (
    BlockInnerProductSpace,
    BlockVector,
    InnerProductSpaceProtocol,
)

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
R = TypeVar("R", bound=ArrayProtocol)
R_co = TypeVar("R_co", bound=ArrayProtocol, covariant=True)

GradientFn = Callable[[V, float], V]
"""``(state, time) -> `` a gradient of :math:`q`, a single column."""


@dataclass(frozen=True)
class IntegrandDerivatives(Generic[V]):
    r"""The optional derivatives of a scalar integrand :math:`q(y, t;
    p)`, as single columns of stack vectors.

    Build with :meth:`none`, :meth:`param_free` or :meth:`first_order`.

    Parameters
    ----------
    state : callable, optional
        :math:`\nabla_y q`, a state column; None when :math:`q` has no
        derivatives.
    param : callable, optional
        :math:`\nabla_p q`, a parameter column; None when :math:`q` does
        not depend on the parameters.

    Raises
    ------
    ValueError
        If ``param`` is given without ``state``.
    """

    state: Optional[GradientFn[V]] = None
    param: Optional[GradientFn[V]] = None

    def __post_init__(self) -> None:
        if self.param is not None and self.state is None:
            raise ValueError("a parameter gradient needs the state gradient")

    @staticmethod
    def none() -> "IntegrandDerivatives[V]":
        """No derivatives: the integral is only evaluated."""
        return IntegrandDerivatives()

    @staticmethod
    def param_free(state: GradientFn[V]) -> "IntegrandDerivatives[V]":
        """A :math:`q` independent of the parameters, with its state
        gradient."""
        return IntegrandDerivatives(state)

    @staticmethod
    def first_order(
        state: GradientFn[V], param: GradientFn[V]
    ) -> "IntegrandDerivatives[V]":
        """The state and parameter gradients."""
        return IntegrandDerivatives(state, param)


@runtime_checkable
class IntegrandProtocol(Protocol[V, R_co]):
    r"""The integrand :math:`q(y, t)` of a functional integrating over
    time."""

    def value(self, state: V, time: float) -> R_co:
        """:math:`q` at a single column ``state`` of the state space and
        ``time``; the state must not be modified."""
        ...

    def derivatives(self) -> IntegrandDerivatives[V]:
        """The optional derivatives, valid for the parameters currently
        set."""
        ...


class TimeIntegral(Generic[V, A, R]):
    r""":math:`\int_{t_0}^{t_N} q(y, t)\, dt` by the stepper's method.

    Parameters
    ----------
    integrand : IntegrandProtocol
        :math:`q`.
    stepper : StepperProtocol
        The stepper the trajectory is computed with; its method integrates
        :math:`q`, and its spaces hold the derivatives.

    Raises
    ------
    TypeError
        If ``integrand`` or ``stepper`` does not satisfy its protocol.
    """

    def __init__(
        self, integrand: IntegrandProtocol[V, R], stepper: StepperProtocol[V, A]
    ) -> None:
        if not isinstance(integrand, IntegrandProtocol):
            raise TypeError(
                "integrand must satisfy IntegrandProtocol, got "
                f"{type(integrand).__name__}"
            )
        if not isinstance(stepper, StepperProtocol):
            raise TypeError(
                f"stepper must satisfy StepperProtocol, got {type(stepper).__name__}"
            )
        self._integrand = integrand
        self._stepper = stepper
        self._quadrature = stepper.time_quadrature()
        self._depth = stepper.depth()

    def integrand(self) -> IntegrandProtocol[V, R]:
        """:math:`q`."""
        return self._integrand

    def quadrature(self) -> StepQuadratureProtocol:
        """The stepper's method applied to :math:`\\dot z = q`."""
        return self._quadrature

    def sink(self) -> "TimeIntegralSink[V, R]":
        """A fresh sink for one solve, whose value is the integral of what
        it receives."""
        return TimeIntegralSink(self._integrand, self._quadrature, self._depth)

    def on(self, store: TrajectoryStoreProtocol[V]) -> "BoundTimeIntegral[V, A]":
        """The integral's derivatives on the trajectory ``store`` recorded,
        as the sweeps read them; :math:`q` must be scalar.

        Raises
        ------
        TypeError
            If the integrand has no state gradient, or the stepper has no
            parameters.
        """
        return BoundTimeIntegral(
            self._integrand.derivatives(),
            self._quadrature,
            self._stepper,
            store,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._integrand!r}, {self._quadrature!r})"


def recursion_sensitivities(rules: Sequence[StepRule]) -> List[float]:
    r""":math:`\theta_n = \partial Q/\partial(\text{step } n\text{'s
    right-hand side})` for :math:`Q = z_N`, by the transposed recursion run
    backward (see the module docstring).

    Parameters
    ----------
    rules : sequence of StepRule
        Every step's recursion, in step order.
    """
    nsteps = len(rules)
    # zeta[m] = dQ/dz_m, built up from the steps reading z_m.
    zeta = [0.0] * (nsteps + 1)
    zeta[nsteps] = 1.0
    theta = [0.0] * nsteps
    for n in reversed(range(nsteps)):
        alpha = rules[n].alpha
        theta[n] = zeta[n + 1] / alpha[0]
        for lag, coefficient in enumerate(alpha[1:], start=1):
            zeta[n + 1 - lag] -= theta[n] * coefficient
    return theta


class BoundTimeIntegral(Generic[V, A]):
    r"""A scalar time integral's derivatives on one recorded trajectory:
    the integral reads the steps, so its stored-state and direct parameter
    cotangents are zero and its derivatives are its step terms
    (:class:`TimeIntegralStepTerms`). Satisfies
    :class:`~pyapprox.ode.trajectory_derivatives.TrajectoryFunctionalProtocol`.

    Parameters
    ----------
    derivatives : IntegrandDerivatives
        :math:`q`'s gradients.
    quadrature : StepQuadratureProtocol
        The stepper's method applied to :math:`\dot z = q`.
    stepper : StepperProtocol
        The stepper the trajectory was recorded with.
    store : TrajectoryStoreProtocol
        The trajectory.

    Raises
    ------
    TypeError
        If the integrand has no state gradient, or the stepper has no
        parameters.
    """

    def __init__(
        self,
        derivatives: IntegrandDerivatives[V],
        quadrature: StepQuadratureProtocol,
        stepper: StepperProtocol[V, A],
        store: TrajectoryStoreProtocol[V],
    ) -> None:
        if derivatives.state is None:
            raise TypeError("the integrand has no state gradient")
        params = stepper.param_space()
        if params is None:
            raise TypeError(
                f"{type(stepper).__name__} has no parameter space, so the "
                "integral has no parameter derivatives"
            )
        space = stepper.state_space().component(0)
        depth = stepper.depth()
        # The recursion of every step, at the window the sweeps read.
        rules = [
            quadrature.rule(store.window(n, min(depth, n + 1)).span)
            for n in range(store.nsteps())
        ]
        self._steps = TimeIntegralStepTerms(
            derivatives.state,
            derivatives.param,
            stepper.unknown_space(),
            space,
            params.component(0),
            rules,
        )
        self._zero_state = space.zeros()
        self._zero_param = params.component(0).zeros()

    def state_cotangent(self, index: int) -> V:
        """Zero: the integral reads the stored states through the steps."""
        return self._zero_state

    def param_cotangent(self) -> V:
        """Zero: the integral reads the parameters through the steps."""
        return self._zero_param

    def step_terms(self) -> "TimeIntegralStepTerms[V, A]":
        """The integral's derivatives, read per step."""
        return self._steps

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._steps!r})"


class TimeIntegralStepTerms(Generic[V, A]):
    r"""A scalar time integral's derivatives through each step,
    :math:`\theta_n \sum w\, \nabla q(\text{node})` (see the module
    docstring); satisfies
    :class:`~pyapprox.ode.trajectory_derivatives.StepTermsProtocol`.

    Parameters
    ----------
    state_gradient : callable
        :math:`\nabla_y q`.
    param_gradient : callable, optional
        :math:`\nabla_p q`; None when :math:`q` does not depend on the
        parameters.
    unknowns : BlockInnerProductSpace
        The stepper's unknown space.
    space : InnerProductSpaceProtocol
        The state space.
    params : InnerProductSpaceProtocol
        The parameter space.
    rules : sequence of StepRule
        Every step's recursion, in step order.
    """

    def __init__(
        self,
        state_gradient: GradientFn[V],
        param_gradient: Optional[GradientFn[V]],
        unknowns: BlockInnerProductSpace[V, A],
        space: InnerProductSpaceProtocol[V, A],
        params: InnerProductSpaceProtocol[V, A],
        rules: Sequence[StepRule],
    ) -> None:
        self._state_gradient, self._param_gradient = state_gradient, param_gradient
        self._unknowns, self._space, self._params = unknowns, space, params
        self._rules = tuple(rules)
        self._theta = recursion_sensitivities(self._rules)

    def _weighted_sum(
        self,
        step: int,
        window: StepWindow[V],
        gradient: GradientFn[V],
        space: InnerProductSpaceProtocol[V, A],
        lag: int,
        block: Optional[int],
    ) -> Optional[V]:
        r""":math:`\theta_n \sum w\, \nabla q` over step ``step``'s nodes at
        ``lag`` (and ``block``, unless None), each node's state read from
        the window; None when there is no such node."""
        theta, total = self._theta[step], None
        for node in self._rules[step].nodes:
            if node.lag != lag or (block is not None and node.block != block):
                continue
            state = (
                window.unknown.blocks[node.block]
                if node.lag == 0
                else window.past_state(node.lag)
            )
            term = gradient(state, node.time)
            total = (
                space.scale(theta * node.weight, term)
                if total is None
                else space.axpby(theta * node.weight, term, 1.0, total, out=total)
            )
        return total

    def unknown_cotangent(
        self, step: int, window: StepWindow[V]
    ) -> Optional[BlockVector[V]]:
        r""":math:`\theta_n \sum w\, \nabla_y q` at the nodes in the step's
        unknown, block by block; None when no node is there."""
        blocks = [
            self._weighted_sum(step, window, self._state_gradient, self._space, 0, b)
            for b in range(self._unknowns.nblocks())
        ]
        if all(block is None for block in blocks):
            return None
        return BlockVector(
            tuple(
                self._space.zeros() if block is None else block for block in blocks
            )
        )

    def lag_cotangent(self, step: int, window: StepWindow[V], lag: int) -> Optional[V]:
        r""":math:`\theta_n \sum w\, \nabla_y q` at the nodes on the stored
        state read at ``lag``; None when no node is there."""
        return self._weighted_sum(
            step, window, self._state_gradient, self._space, lag, None
        )

    def param_cotangent(self, step: int, window: StepWindow[V]) -> Optional[V]:
        r""":math:`\theta_n \sum w\, \nabla_p q` over every node of the step;
        None when :math:`q` does not depend on the parameters."""
        gradient = self._param_gradient
        if gradient is None:
            return None
        total: Optional[V] = None
        for lag in range(len(self._rules[step].alpha)):
            term = self._weighted_sum(step, window, gradient, self._params, lag, None)
            if term is not None:
                total = (
                    term
                    if total is None
                    else self._params.axpby(1.0, term, 1.0, total, out=total)
                )
        return total

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nsteps={len(self._rules)})"


def step_increment(
    integrand: IntegrandProtocol[V, R],
    rule: StepRule,
    unknown: BlockVector[V],
    past_states: List[V],
) -> R:
    r""":math:`\sum w\, q(\text{node})` for one step: lag 0 reads the
    step's unknown, lag :math:`l \ge 1` the stored state ``past_states[l -
    1]`` (most recent first).

    Raises
    ------
    ValueError
        If the rule has no node.
    """
    total: Optional[R] = None
    for node in rule.nodes:
        state = (
            unknown.blocks[node.block] if node.lag == 0 else past_states[node.lag - 1]
        )
        term = node.weight * integrand.value(state, node.time)
        total = term if total is None else total + term
    if total is None:
        raise ValueError("a step's rule has no node")
    return total


class TimeIntegralSink(Generic[V, R]):
    r"""One solve's :math:`z`, advanced as the steps arrive.

    Keeps the stored states and the values of :math:`z` the method's
    deepest step reads, nothing more.

    Parameters
    ----------
    integrand : IntegrandProtocol
        :math:`q`.
    quadrature : StepQuadratureProtocol
        The stepper's method applied to :math:`\dot z = q`.
    depth : int
        The most stored states a step reads.

    Raises
    ------
    ValueError
        From :meth:`append`, if a state was already received; from
        :meth:`append_step`, if none was, or the step does not start where
        the last one ended; from :meth:`value`, before any step.
    """

    def __init__(
        self,
        integrand: IntegrandProtocol[V, R],
        quadrature: StepQuadratureProtocol,
        depth: int,
    ) -> None:
        self._integrand = integrand
        self._quadrature = quadrature
        # Most recent first; z_0 = 0 is None, since no zero of q's type is
        # at hand before the first step.
        self._states: Deque[V] = deque(maxlen=depth)
        self._integrals: Deque[Optional[R]] = deque(maxlen=depth)
        self._ntimes = 0
        self._last_time: Optional[float] = None

    def append(self, time: float, state: V, unconstrained: Optional[V] = None) -> None:
        """Take the initial state, where :math:`z = 0`."""
        if self._last_time is not None:
            raise ValueError(
                "only the initial state is appended at a time; every later state "
                "comes with its step (append_step)"
            )
        self._states.appendleft(state)
        self._integrals.appendleft(None)
        self._ntimes, self._last_time = 1, time

    def append_step(
        self,
        span: StepSpan,
        state: V,
        unknown: BlockVector[V],
        parameters_version: int,
        unconstrained: Optional[V] = None,
    ) -> None:
        r"""Advance :math:`z` over the step, reading its nodes from
        ``unknown`` and the stored states the step read."""
        if self._last_time is None:
            raise ValueError("a step starts from a received state; append one first")
        if span.time() != self._last_time:
            raise ValueError(
                f"the step starts at {span.time()}, not where the last one ended, "
                f"{self._last_time}"
            )
        rule = self._quadrature.rule(span)
        increment = step_increment(
            self._integrand, rule, unknown, list(self._states)
        )
        past = [
            increment * 0.0 if value is None else value
            for value in list(self._integrals)[: len(rule.alpha) - 1]
        ]
        self._integrals.appendleft(rule.advance(past, increment))
        self._states.appendleft(state)
        self._ntimes += 1
        self._last_time = span.end()

    def ntimes(self) -> int:
        """The number of states received."""
        return self._ntimes

    def value(self) -> R:
        r""":math:`z` at the last state received: the integral so far."""
        latest = self._integrals[0] if self._integrals else None
        if latest is None:
            raise ValueError("no step has been received, so there is no integral")
        return latest

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._integrand!r}, ntimes={self._ntimes})"
