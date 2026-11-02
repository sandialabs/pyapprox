r"""Functionals integrating over time, with the stepper's own method.

:math:`Q = \int_{t_0}^{t_N} q(y, t)\, dt` is computed as the stepper's
method applied to :math:`\dot z = q(y, t)`, :math:`z(t_0) = 0` (see
:mod:`pyapprox.ode.step_quadrature`): the integral is consistent with the
steps by construction, and no rule is chosen apart from the method.

A :class:`TimeIntegral` is reused across solves; each solve takes a fresh
sink from :meth:`TimeIntegral.sink`, which advances :math:`z` as the steps
arrive, reading each step's nodes from its unknown and the stored states
it read, so the solve keeps no trajectory.

Its derivatives (:meth:`TimeIntegral.trajectory_functional`, a step term
for a scalar :math:`q`) are the
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
from functools import partial
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
from pyapprox.ode.step_quadrature import (
    QuadratureNode,
    StepQuadratureProtocol,
    StepRule,
)
from pyapprox.ode.step_span import StepSpan
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.trajectory_derivatives import (
    StepTermDerivatives,
    TrajectoryFunctional,
)
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

CurvatureFn = Callable[[V, float, V], V]
"""``(state, time, x) -> `` a second derivative of :math:`q` applied to
the single column ``x``, a single column."""


@dataclass(frozen=True)
class IntegrandDerivatives(Generic[V]):
    r"""The optional derivatives of a scalar integrand :math:`q(y, t;
    p)`, as single columns of stack vectors.

    Build with :meth:`none`, :meth:`param_free`, :meth:`first_order`,
    :meth:`param_free_second_order` or :meth:`second_order`.

    Parameters
    ----------
    state : callable, optional
        :math:`\nabla_y q`, a state column; None when :math:`q` has no
        derivatives.
    param : callable, optional
        :math:`\nabla_p q`, a parameter column; None when :math:`q` does
        not depend on the parameters.
    state_state : callable, optional
        :math:`q_{yy}\, w`, state to state; None without second
        derivatives.
    state_param : callable, optional
        :math:`q_{yp}\, v`, parameters to state.
    param_state : callable, optional
        :math:`q_{py}\, w`, state to parameters.
    param_param : callable, optional
        :math:`q_{pp}\, v`, parameters to parameters.

    Raises
    ------
    ValueError
        If ``param`` is given without ``state``; a second derivative
        without ``state_state`` and the gradients it builds on; or, with
        second derivatives, the three parameter ones not exactly when
        ``param`` is given (the second-order sweep needs every one).
    """

    state: Optional[GradientFn[V]] = None
    param: Optional[GradientFn[V]] = None
    state_state: Optional[CurvatureFn[V]] = None
    state_param: Optional[CurvatureFn[V]] = None
    param_state: Optional[CurvatureFn[V]] = None
    param_param: Optional[CurvatureFn[V]] = None

    def __post_init__(self) -> None:
        if self.param is not None and self.state is None:
            raise ValueError("a parameter gradient needs the state gradient")
        mixed = (self.state_param, self.param_state, self.param_param)
        if self.state_state is None:
            if any(field is not None for field in mixed):
                raise ValueError("second derivatives need state_state")
            return
        if self.state is None:
            raise ValueError("second derivatives need the state gradient")
        given = [field is not None for field in mixed]
        if self.param is None and any(given):
            raise ValueError(
                "a q independent of the parameters has no parameter second "
                "derivatives"
            )
        if self.param is not None and not all(given):
            raise ValueError(
                "a q depending on the parameters needs state_param, param_state "
                "and param_param with state_state"
            )

    def is_second_order(self) -> bool:
        """Whether the second derivatives are given."""
        return self.state_state is not None

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

    @staticmethod
    def param_free_second_order(
        state: GradientFn[V], state_state: CurvatureFn[V]
    ) -> "IntegrandDerivatives[V]":
        """A :math:`q` independent of the parameters, with its state
        gradient and Hessian."""
        return IntegrandDerivatives(state, None, state_state)

    @staticmethod
    def second_order(
        state: GradientFn[V],
        param: GradientFn[V],
        state_state: CurvatureFn[V],
        state_param: CurvatureFn[V],
        param_state: CurvatureFn[V],
        param_param: CurvatureFn[V],
    ) -> "IntegrandDerivatives[V]":
        """Both gradients and every second derivative."""
        return IntegrandDerivatives(
            state, param, state_state, state_param, param_state, param_param
        )


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

    def _params(self) -> InnerProductSpaceProtocol[V, A]:
        """The stepper's parameter space, as a plain space.

        Raises
        ------
        TypeError
            If the stepper has no parameters.
        """
        params = self._stepper.param_space()
        if params is None:
            raise TypeError(
                f"{type(self._stepper).__name__} has no parameter space, so the "
                "integral has no parameter derivatives"
            )
        return params.component(0)

    def _rules(self, store: TrajectoryStoreProtocol[V]) -> List[StepRule]:
        """Every step's recursion, over the span the sweeps' window has,
        from the step sizes alone: no state is read."""
        depth = self._stepper.depth()
        return [
            self._quadrature.rule(store.span(n, min(depth, n + 1)))
            for n in range(store.nsteps())
        ]

    def trajectory_functional(
        self, store: TrajectoryStoreProtocol[V]
    ) -> TrajectoryFunctional[V]:
        """The integral on the trajectory ``store`` recorded, as the sweeps
        differentiate it: one step term, second order when the integrand
        is; :math:`q` must be scalar.

        Raises
        ------
        TypeError
            If the integrand has no state gradient, or the stepper has no
            parameters.
        """
        term = TimeIntegralStepTerm(
            self._integrand.derivatives(),
            self._stepper.unknown_space(),
            self._stepper.state_space().component(0),
            self._params(),
            self._rules(store),
        )
        return TrajectoryFunctional(step_terms=(term,))

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


class TimeIntegralStepTerm(Generic[V, A]):
    r"""A scalar time integral on one recorded trajectory, as a step term:
    its derivatives through each step are :math:`\theta_n \sum w\, \nabla
    q(\text{node})` (see the module docstring), and its bundle carries
    their derivatives along a tangent when the integrand's has second
    derivatives. Satisfies
    :class:`~pyapprox.ode.trajectory_derivatives.StepTermProtocol`.

    The weights depend on the step sizes only, so differentiating
    :math:`\theta_n w\, \nabla q(\text{node})` along the tangent leaves
    :math:`\theta_n w\, (q_{yy}\, dX + q_{yp}\, v)` for a node's state and
    :math:`\theta_n w\, (q_{py}\, dX + q_{pp}\, v)` for the parameters;
    the second derivative is block diagonal over the nodes.

    Parameters
    ----------
    derivatives : IntegrandDerivatives
        :math:`q`'s gradients, and second derivatives when it has them.
    unknowns : BlockInnerProductSpace
        The stepper's unknown space.
    space : InnerProductSpaceProtocol
        The state space.
    params : InnerProductSpaceProtocol
        The parameter space.
    rules : sequence of StepRule
        Every step's recursion, in step order.

    Raises
    ------
    TypeError
        If the integrand has no state gradient.
    """

    def __init__(
        self,
        derivatives: IntegrandDerivatives[V],
        unknowns: BlockInnerProductSpace[V, A],
        space: InnerProductSpaceProtocol[V, A],
        params: InnerProductSpaceProtocol[V, A],
        rules: Sequence[StepRule],
    ) -> None:
        if derivatives.state is None:
            raise TypeError("the integrand has no state gradient")
        self._state_gradient: GradientFn[V] = derivatives.state
        self._param_gradient: Optional[GradientFn[V]] = derivatives.param
        self._unknowns, self._space, self._params = unknowns, space, params
        self._rules = tuple(rules)
        self._theta = recursion_sensitivities(self._rules)
        self._fields: Optional[_SecondOrderFields[V]] = (
            None
            if derivatives.state_state is None
            else _SecondOrderFields(
                derivatives.state_state,
                derivatives.state_param,
                derivatives.param_state,
                derivatives.param_param,
            )
        )
        self._derivatives: StepTermDerivatives[V] = (
            StepTermDerivatives.first_order()
            if self._fields is None
            else StepTermDerivatives.second_order(
                self._curvature_wrt_unknown,
                self._curvature_wrt_lag,
                self._curvature_wrt_param,
            )
        )

    def _weighted_sum(
        self,
        step: int,
        window: StepWindow[V],
        term: Callable[[QuadratureNode, V], V],
        space: InnerProductSpaceProtocol[V, A],
        lag: Optional[int],
        block: Optional[int],
    ) -> Optional[V]:
        r""":math:`\theta_n \sum w\, \mathrm{term}(\text{node}, \text{its
        state})` over step ``step``'s nodes at ``lag`` and ``block`` (each
        unless None), each node's state read from the window; None when
        there is no such node."""
        theta, total = self._theta[step], None
        for node in self._rules[step].nodes:
            if (lag is not None and node.lag != lag) or (
                block is not None and node.block != block
            ):
                continue
            value = term(node, node_state(window, node))
            total = (
                space.scale(theta * node.weight, value)
                if total is None
                else space.axpby(theta * node.weight, value, 1.0, total, out=total)
            )
        return total

    def _blocks(self, sums: List[Optional[V]]) -> Optional[BlockVector[V]]:
        """The unknown-space vector of per-block sums, zero where a block
        has none; None when no block has one."""
        if all(block is None for block in sums):
            return None
        return BlockVector(
            tuple(self._space.zeros() if block is None else block for block in sums)
        )

    def _state_term(self, node: QuadratureNode, state: V) -> V:
        return self._state_gradient(state, node.time)

    def _param_term(self, node: QuadratureNode, state: V) -> V:
        gradient = self._param_gradient
        if gradient is None:
            raise RuntimeError("q does not depend on the parameters")
        return gradient(state, node.time)

    def cotangent_wrt_unknown(
        self, step: int, window: StepWindow[V]
    ) -> Optional[BlockVector[V]]:
        r""":math:`\theta_n \sum w\, \nabla_y q` at the nodes in the step's
        unknown, block by block; None when no node is there."""
        return self._blocks(
            [
                self._weighted_sum(step, window, self._state_term, self._space, 0, b)
                for b in range(self._unknowns.nblocks())
            ]
        )

    def cotangent_wrt_lag(
        self, step: int, window: StepWindow[V], lag: int
    ) -> Optional[V]:
        r""":math:`\theta_n \sum w\, \nabla_y q` at the nodes on the stored
        state read at ``lag``; None when no node is there."""
        return self._weighted_sum(
            step, window, self._state_term, self._space, lag, None
        )

    def cotangent_wrt_param(self, step: int, window: StepWindow[V]) -> Optional[V]:
        r""":math:`\theta_n \sum w\, \nabla_p q` over every node of the step;
        None when :math:`q` does not depend on the parameters."""
        if self._param_gradient is None:
            return None
        return self._weighted_sum(
            step, window, self._param_term, self._params, None, None
        )

    def derivatives(self) -> StepTermDerivatives[V]:
        """The derivatives along a tangent, when the integrand has second
        derivatives."""
        return self._derivatives

    def _curvature(
        self, tangents: StepWindow[V], direction: V
    ) -> "_NodeCurvature[V, A]":
        if self._fields is None:
            raise RuntimeError(
                "the integrand has no second derivatives; check derivatives() "
                "before calling"
            )
        return _NodeCurvature(self._fields, tangents, direction, self._space)

    def _curvature_wrt_unknown(
        self,
        step: int,
        window: StepWindow[V],
        tangents: StepWindow[V],
        direction: V,
    ) -> Optional[BlockVector[V]]:
        r""":math:`\theta_n \sum w\, (q_{yy}\, dX + q_{yp}\, v)` at the
        nodes in the step's unknown, block by block."""
        curvature = self._curvature(tangents, direction)
        return self._blocks(
            [
                self._weighted_sum(
                    step, window, curvature.to_state, self._space, 0, b
                )
                for b in range(self._unknowns.nblocks())
            ]
        )

    def _curvature_wrt_lag(
        self,
        step: int,
        window: StepWindow[V],
        tangents: StepWindow[V],
        lag: int,
        direction: V,
    ) -> Optional[V]:
        r""":math:`\theta_n \sum w\, (q_{yy}\, dX + q_{yp}\, v)` at the
        nodes on the stored state read at ``lag``."""
        curvature = self._curvature(tangents, direction)
        return self._weighted_sum(
            step, window, curvature.to_state, self._space, lag, None
        )

    def _curvature_wrt_param(
        self,
        step: int,
        window: StepWindow[V],
        tangents: StepWindow[V],
        direction: V,
    ) -> Optional[V]:
        r""":math:`\theta_n \sum w\, (q_{py}\, dX + q_{pp}\, v)` over every
        node of the step; None when :math:`q` does not depend on the
        parameters."""
        curvature = self._curvature(tangents, direction)
        if self._param_gradient is None:
            return None
        params = self._params
        return self._weighted_sum(
            step,
            window,
            partial(curvature.to_param, params=params),
            params,
            None,
            None,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nsteps={len(self._rules)})"


def node_state(window: StepWindow[V], node: QuadratureNode) -> V:
    """The state at ``node`` in ``window``: a block of the step's unknown
    at lag 0, the stored state read at the node's lag otherwise."""
    if node.lag == 0:
        return window.unknown.blocks[node.block]
    return window.past_state(node.lag)


class _NodeCurvature(Generic[V, A]):
    r"""One node's second-order term along a tangent: :math:`q_{yy}\,
    dX + q_{yp}\, v` (``to_state``) or :math:`q_{py}\, dX + q_{pp}\, v`
    (to the parameters), :math:`dX` the node's state's tangent read from
    the tangent window. Parameter terms are skipped for a :math:`q`
    independent of the parameters."""

    def __init__(
        self,
        derivatives: "_SecondOrderFields[V]",
        tangents: StepWindow[V],
        direction: V,
        space: InnerProductSpaceProtocol[V, A],
    ) -> None:
        self._fields, self._tangents = derivatives, tangents
        self._direction, self._space = direction, space

    def to_state(self, node: QuadratureNode, state: V) -> V:
        fields = self._fields
        value = fields.state_state(state, node.time, node_state(self._tangents, node))
        if fields.state_param is None:
            return value
        mixed = fields.state_param(state, node.time, self._direction)
        return self._space.axpby(1.0, mixed, 1.0, value, out=value)

    def to_param(
        self,
        node: QuadratureNode,
        state: V,
        params: InnerProductSpaceProtocol[V, A],
    ) -> V:
        fields = self._fields
        if fields.param_state is None or fields.param_param is None:
            raise RuntimeError("q does not depend on the parameters")
        value = fields.param_state(state, node.time, node_state(self._tangents, node))
        curvature = fields.param_param(state, node.time, self._direction)
        return params.axpby(1.0, curvature, 1.0, value, out=value)


@dataclass(frozen=True)
class _SecondOrderFields(Generic[V]):
    """An integrand bundle's second derivatives, ``state_state`` known
    present; the parameter ones None for a :math:`q` independent of the
    parameters."""

    state_state: CurvatureFn[V]
    state_param: Optional[CurvatureFn[V]]
    param_state: Optional[CurvatureFn[V]]
    param_param: Optional[CurvatureFn[V]]


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
