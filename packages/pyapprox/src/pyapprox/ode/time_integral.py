r"""Functionals integrating over time, with the stepper's own method.

:math:`Q = \int_{t_0}^{t_N} q(y, t)\, dt` is computed as the stepper's
method applied to :math:`\dot z = q(y, t)`, :math:`z(t_0) = 0` (see
:mod:`pyapprox.ode.step_quadrature`): the integral is consistent with the
steps by construction, and no rule is chosen apart from the method.

Step :math:`n` contributes :math:`c_n = \sum w\, q(\text{node})` over
its rule's nodes, read from its window (a block of its unknown, or a
stored state it read): a :class:`QuadratureStepTerm`, which holds no
trajectory. The contributions combine through the method's recursion,
:math:`\alpha^{(n)}_0 z_{n+1} + \sum_{l \ge 1} \alpha^{(n)}_l z_{n+1-l} =
c_n`: forward, step by step, for the value (:meth:`TimeIntegral.sink`,
keeping no trajectory), and through its transpose :math:`\theta`
(:func:`~pyapprox.ode.step_quadrature.recursion_sensitivities`, from the
step sizes alone) for the derivatives
(:meth:`TimeIntegral.trajectory_functional`, the term weighed by
:class:`~pyapprox.ode.trajectory_terms.RecursionWeighted`). Both read the
nodes through the same term, so the value and the derivatives cannot
read different states.

The integrand's derivatives are linearizations, so a vector-valued
:math:`q` has derivatives in both modes; its second derivatives are
weighted by the adjoint weights :math:`\omega` of its entries.
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
    Tuple,
    TypeVar,
    runtime_checkable,
)

from pyapprox.ode.protocols import StepperProtocol
from pyapprox.ode.step_quadrature import (
    QuadratureNode,
    StepQuadratureProtocol,
    StepRule,
    recursion_sensitivities,
)
from pyapprox.ode.step_span import StepSpan
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.trajectory_store import TrajectoryStoreProtocol
from pyapprox.ode.trajectory_terms import (
    LinearizationProtocol,
    RecursionWeighted,
    StepTermDerivatives,
    TrajectoryFunctional,
)
from pyapprox.util.backends.protocols import ArrayProtocol, Backend
from pyapprox.util.linalg.operators import (
    BlockInnerProductSpace,
    BlockVector,
    InnerProductSpaceProtocol,
)

V = TypeVar("V")
X = TypeVar("X")
A = TypeVar("A", bound=ArrayProtocol)

LinearizationFn = Callable[[V, float], LinearizationProtocol[V, A]]
"""``(state, time) ->`` a first derivative of :math:`q` there."""

CurvatureFn = Callable[[V, float, A, V], V]
"""``(state, time, weights, x) ->`` a second derivative of
:math:`\\omega^\\top q` applied to the column ``x``."""


@dataclass(frozen=True)
class IntegrandDerivatives(Generic[V, A]):
    r"""The optional derivatives of an integrand :math:`q(y, t; p)`: its
    first derivatives as linearizations (forward to value columns,
    transposed from value weights), its second derivatives weighted by
    the value weights :math:`\omega`.

    Build with :meth:`none`, :meth:`param_free`, :meth:`first_order`,
    :meth:`param_free_second_order` or :meth:`second_order`.

    Parameters
    ----------
    state : callable, optional
        :math:`\partial q/\partial y`; None when :math:`q` has no
        derivatives.
    param : callable, optional
        :math:`\partial q/\partial p`; None when :math:`q` does not depend
        on the parameters.
    state_state : callable, optional
        :math:`(\omega^\top q)_{yy}\, w`, state to state; None without
        second derivatives.
    state_param : callable, optional
        :math:`(\omega^\top q)_{yp}\, v`, parameters to state.
    param_state : callable, optional
        :math:`(\omega^\top q)_{py}\, w`, state to parameters.
    param_param : callable, optional
        :math:`(\omega^\top q)_{pp}\, v`, parameters to parameters.

    Raises
    ------
    ValueError
        If ``param`` is given without ``state``; a second derivative
        without ``state_state`` and the gradients it builds on; or, with
        second derivatives, the three parameter ones not exactly when
        ``param`` is given (the second-order sweep needs every one).
    """

    state: Optional[LinearizationFn[V, A]] = None
    param: Optional[LinearizationFn[V, A]] = None
    state_state: Optional[CurvatureFn[V, A]] = None
    state_param: Optional[CurvatureFn[V, A]] = None
    param_state: Optional[CurvatureFn[V, A]] = None
    param_param: Optional[CurvatureFn[V, A]] = None

    def __post_init__(self) -> None:
        if self.param is not None and self.state is None:
            raise ValueError("a parameter derivative needs the state derivative")
        mixed = (self.state_param, self.param_state, self.param_param)
        if self.state_state is None:
            if any(field is not None for field in mixed):
                raise ValueError("second derivatives need state_state")
            return
        if self.state is None:
            raise ValueError("second derivatives need the state derivative")
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
    def none() -> "IntegrandDerivatives[V, A]":
        """No derivatives: the integral is only evaluated."""
        return IntegrandDerivatives()

    @staticmethod
    def param_free(state: LinearizationFn[V, A]) -> "IntegrandDerivatives[V, A]":
        """A :math:`q` independent of the parameters, with its state
        derivative."""
        return IntegrandDerivatives(state)

    @staticmethod
    def first_order(
        state: LinearizationFn[V, A], param: LinearizationFn[V, A]
    ) -> "IntegrandDerivatives[V, A]":
        """The state and parameter derivatives."""
        return IntegrandDerivatives(state, param)

    @staticmethod
    def param_free_second_order(
        state: LinearizationFn[V, A], state_state: CurvatureFn[V, A]
    ) -> "IntegrandDerivatives[V, A]":
        """A :math:`q` independent of the parameters, with its state
        derivative and weighted Hessian."""
        return IntegrandDerivatives(state, None, state_state)

    @staticmethod
    def second_order(
        state: LinearizationFn[V, A],
        param: LinearizationFn[V, A],
        state_state: CurvatureFn[V, A],
        state_param: CurvatureFn[V, A],
        param_state: CurvatureFn[V, A],
        param_param: CurvatureFn[V, A],
    ) -> "IntegrandDerivatives[V, A]":
        """Both first derivatives and every second derivative."""
        return IntegrandDerivatives(
            state, param, state_state, state_param, param_state, param_param
        )


@runtime_checkable
class IntegrandProtocol(Protocol[V, A]):
    r"""The integrand :math:`q(y, t)` of a functional integrating over
    time, with values of ``nqoi`` entries."""

    def nqoi(self) -> int:
        """The number of entries of :math:`q`."""
        ...

    def value(self, state: V, time: float) -> A:
        """:math:`q` at a single column ``state`` of the state space and
        ``time``, a column of ``nqoi`` entries; the state must not be
        modified."""
        ...

    def derivatives(self) -> IntegrandDerivatives[V, A]:
        """The optional derivatives, valid for the parameters currently
        set."""
        ...


class TimeIntegral(Generic[V, A]):
    r""":math:`\int_{t_0}^{t_N} q(y, t)\, dt` by the stepper's method.

    Parameters
    ----------
    integrand : IntegrandProtocol
        :math:`q`.
    stepper : StepperProtocol
        The stepper the trajectory is computed with; its method integrates
        :math:`q`, and its spaces hold the derivatives.
    bkd : Backend
        The backend of the integral's values.

    Raises
    ------
    TypeError
        If ``integrand`` or ``stepper`` does not satisfy its protocol.
    """

    def __init__(
        self,
        integrand: IntegrandProtocol[V, A],
        stepper: StepperProtocol[V, A],
        bkd: Backend[A],
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
        self._integrand, self._stepper, self._bkd = integrand, stepper, bkd
        self._quadrature = stepper.time_quadrature()
        self._depth = stepper.depth()
        params = stepper.param_space()
        self._term = QuadratureStepTerm(
            integrand,
            self._quadrature,
            stepper.unknown_space(),
            stepper.state_space().component(0),
            None if params is None else params.component(0),
        )

    def integrand(self) -> IntegrandProtocol[V, A]:
        """:math:`q`."""
        return self._integrand

    def quadrature(self) -> StepQuadratureProtocol:
        """The stepper's method applied to :math:`\\dot z = q`."""
        return self._quadrature

    def step_term(self) -> "QuadratureStepTerm[V, A]":
        """Each step's contribution :math:`c_n`, unweighted."""
        return self._term

    def sink(self) -> "TimeIntegralSink[V, A]":
        """A fresh sink for one solve, whose value is the integral of what
        it receives."""
        return TimeIntegralSink(self._term, self._quadrature, self._depth)

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
    ) -> TrajectoryFunctional[V, A]:
        """The integral on the trajectory ``store`` recorded, as the sweeps
        differentiate it: the step term weighed by the recursion's
        sensitivities on that trajectory, from its step sizes alone (no
        state is read).

        Raises
        ------
        TypeError
            If the integrand has no state derivative.
        """
        if self._integrand.derivatives().state is None:
            raise TypeError("the integrand has no state derivative")
        weighted = RecursionWeighted(
            self._term, recursion_sensitivities(self._rules(store))
        )
        return TrajectoryFunctional.from_terms(
            self._bkd, self._integrand.nqoi(), step_terms=(weighted,)
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._integrand!r}, {self._quadrature!r})"


def _add_scaled(
    total: Optional[V],
    weight: float,
    term: V,
    space: InnerProductSpaceProtocol[V, A],
) -> V:
    """``total + weight * term``, into ``total`` when there is one; a term
    starting the total is scaled into a new vector, since it may be the
    integrand's own storage."""
    if total is None:
        return space.scale(weight, term)
    return space.axpby(weight, term, 1.0, total, out=total)


class _NodeSum(Generic[X, A]):
    r""":math:`\sum w\, D` over nodes, each node's :math:`D` a
    linearization of :math:`q` at its state: forward a sum of value
    columns, transposed a sum of vectors of ``space``."""

    def __init__(
        self,
        pieces: Sequence[Tuple[float, LinearizationProtocol[X, A]]],
        space: InnerProductSpaceProtocol[X, A],
    ) -> None:
        self._pieces, self._space = tuple(pieces), space

    def apply(self, x: X) -> A:
        total: Optional[A] = None
        for weight, linearization in self._pieces:
            term = weight * linearization.apply(x)
            total = term if total is None else total + term
        if total is None:
            raise ValueError("a node sum has no node")
        return total

    def apply_transpose(self, weights: A) -> X:
        total: Optional[X] = None
        for weight, linearization in self._pieces:
            total = _add_scaled(
                total, weight, linearization.apply_transpose(weights), self._space
            )
        if total is None:
            raise ValueError("a node sum has no node")
        return total


class _UnknownNodes(Generic[V, A]):
    r""":math:`\sum w\, D` over the nodes in a step's unknown, each acting
    on its block: forward from an unknown-space vector, transposed into
    one (zero blocks where no node lies)."""

    def __init__(
        self,
        pieces: Sequence[Tuple[int, float, LinearizationProtocol[V, A]]],
        unknowns: BlockInnerProductSpace[V, A],
        space: InnerProductSpaceProtocol[V, A],
    ) -> None:
        self._pieces, self._unknowns, self._space = tuple(pieces), unknowns, space

    def apply(self, x: BlockVector[V]) -> A:
        total: Optional[A] = None
        for block, weight, linearization in self._pieces:
            term = weight * linearization.apply(x.blocks[block])
            total = term if total is None else total + term
        if total is None:
            raise ValueError("no node lies in the unknown")
        return total

    def apply_transpose(self, weights: A) -> BlockVector[V]:
        blocks: List[Optional[V]] = [None] * self._unknowns.nblocks()
        for block, weight, linearization in self._pieces:
            blocks[block] = _add_scaled(
                blocks[block],
                weight,
                linearization.apply_transpose(weights),
                self._space,
            )
        return BlockVector(
            tuple(self._space.zeros() if b is None else b for b in blocks)
        )


@dataclass(frozen=True)
class _SecondOrderFields(Generic[V, A]):
    """An integrand bundle's second derivatives, ``state_state`` known
    present; the parameter ones None for a :math:`q` independent of the
    parameters."""

    state_state: CurvatureFn[V, A]
    state_param: Optional[CurvatureFn[V, A]]
    param_state: Optional[CurvatureFn[V, A]]
    param_param: Optional[CurvatureFn[V, A]]


class QuadratureStepTerm(Generic[V, A]):
    r"""Each step's contribution to a time integral, :math:`c_n = \sum w\,
    q(y_{\text{node}}, t_{\text{node}})` over the stepper's rule's nodes on
    the step, read from the step's window: a step term, unweighted (the
    recursion's weights are applied by
    :class:`~pyapprox.ode.trajectory_terms.RecursionWeighted`). It holds no
    trajectory; the rule is taken from each window's span.

    Its derivatives are the integrand's at each node, summed with the
    nodes' weights; the second derivative is block diagonal over the
    nodes, and the weights depend on the step sizes only, so the
    derivative along a tangent of :math:`w\, D q` is :math:`w\,
    ((\omega^\top q)_{yy}\, dX + (\omega^\top q)_{yp}\, v)` at a node's
    state and :math:`w\, ((\omega^\top q)_{py}\, dX + (\omega^\top
    q)_{pp}\, v)` for the parameters. Satisfies
    :class:`~pyapprox.ode.trajectory_terms.StepTermProtocol`.

    Parameters
    ----------
    integrand : IntegrandProtocol
        :math:`q`; its derivatives are read once, here.
    quadrature : StepQuadratureProtocol
        The stepper's method applied to :math:`\dot z = q`.
    unknowns : BlockInnerProductSpace
        The stepper's unknown space.
    space : InnerProductSpaceProtocol
        The state space.
    params : InnerProductSpaceProtocol, optional
        The parameter space; None when the stepper has none.
    """

    def __init__(
        self,
        integrand: IntegrandProtocol[V, A],
        quadrature: StepQuadratureProtocol,
        unknowns: BlockInnerProductSpace[V, A],
        space: InnerProductSpaceProtocol[V, A],
        params: Optional[InnerProductSpaceProtocol[V, A]],
    ) -> None:
        self._integrand, self._quadrature = integrand, quadrature
        self._unknowns, self._space, self._params = unknowns, space, params
        derivatives = integrand.derivatives()
        self._state: Optional[LinearizationFn[V, A]] = derivatives.state
        self._param: Optional[LinearizationFn[V, A]] = derivatives.param
        self._fields: Optional[_SecondOrderFields[V, A]] = (
            None
            if derivatives.state_state is None
            else _SecondOrderFields(
                derivatives.state_state,
                derivatives.state_param,
                derivatives.param_state,
                derivatives.param_param,
            )
        )
        self._derivatives: StepTermDerivatives[V, A] = (
            StepTermDerivatives.first_order()
            if self._fields is None
            else StepTermDerivatives.second_order(
                self._curvature_wrt_unknown,
                self._curvature_wrt_lag,
                self._curvature_wrt_param,
            )
        )

    def nqoi(self) -> int:
        return self._integrand.nqoi()

    def _nodes(self, window: StepWindow[V]) -> Tuple[QuadratureNode, ...]:
        return self._quadrature.rule(window.span).nodes

    def value(self, step: int, window: StepWindow[V]) -> A:
        r""":math:`c_n = \sum w\, q(\text{node})`.

        Raises
        ------
        ValueError
            If the step's rule has no node.
        """
        total: Optional[A] = None
        for node in self._nodes(window):
            term = node.weight * self._integrand.value(
                node_state(window, node), node.time
            )
            total = term if total is None else total + term
        if total is None:
            raise ValueError("a step's rule has no node")
        return total

    def _state_derivative(self) -> LinearizationFn[V, A]:
        if self._state is None:
            raise TypeError("the integrand has no state derivative")
        return self._state

    def wrt_unknown(
        self, step: int, window: StepWindow[V]
    ) -> Optional[LinearizationProtocol[BlockVector[V], A]]:
        r""":math:`\sum w\, \partial q/\partial y` at the nodes in the
        step's unknown; None when no node lies there."""
        derivative = self._state_derivative()
        pieces = [
            (node.block, node.weight, derivative(node_state(window, node), node.time))
            for node in self._nodes(window)
            if node.lag == 0
        ]
        if not pieces:
            return None
        return _UnknownNodes(pieces, self._unknowns, self._space)

    def wrt_lag(
        self, step: int, window: StepWindow[V], lag: int
    ) -> Optional[LinearizationProtocol[V, A]]:
        r""":math:`\sum w\, \partial q/\partial y` at the nodes on the stored
        state read at ``lag``; None when no node lies there."""
        derivative = self._state_derivative()
        pieces = [
            (node.weight, derivative(node_state(window, node), node.time))
            for node in self._nodes(window)
            if node.lag == lag
        ]
        return _NodeSum(pieces, self._space) if pieces else None

    def wrt_param(
        self, step: int, window: StepWindow[V]
    ) -> Optional[LinearizationProtocol[V, A]]:
        r""":math:`\sum w\, \partial q/\partial p` over every node; None
        when :math:`q` does not depend on the parameters."""
        derivative, params = self._param, self._params
        if derivative is None or params is None:
            return None
        pieces = [
            (node.weight, derivative(node_state(window, node), node.time))
            for node in self._nodes(window)
        ]
        return _NodeSum(pieces, params)

    def derivatives(self) -> StepTermDerivatives[V, A]:
        """The derivatives along a tangent, when the integrand has second
        derivatives."""
        return self._derivatives

    def _second_order(self) -> _SecondOrderFields[V, A]:
        if self._fields is None:
            raise RuntimeError(
                "the integrand has no second derivatives; check derivatives() "
                "before calling"
            )
        return self._fields

    def _to_state(
        self,
        node: QuadratureNode,
        window: StepWindow[V],
        tangents: StepWindow[V],
        direction: V,
        weights: A,
    ) -> V:
        r""":math:`(\omega^\top q)_{yy}\, dX + (\omega^\top q)_{yp}\, v` at
        ``node``."""
        fields, state = self._second_order(), node_state(window, node)
        value = fields.state_state(
            state, node.time, weights, node_state(tangents, node)
        )
        if fields.state_param is None:
            return value
        mixed = fields.state_param(state, node.time, weights, direction)
        return self._space.axpby(1.0, mixed, 1.0, value)

    def _curvature_wrt_unknown(
        self,
        step: int,
        window: StepWindow[V],
        tangents: StepWindow[V],
        direction: V,
        weights: A,
    ) -> Optional[BlockVector[V]]:
        blocks: List[Optional[V]] = [None] * self._unknowns.nblocks()
        for node in self._nodes(window):
            if node.lag == 0:
                blocks[node.block] = _add_scaled(
                    blocks[node.block],
                    node.weight,
                    self._to_state(node, window, tangents, direction, weights),
                    self._space,
                )
        if all(block is None for block in blocks):
            return None
        return BlockVector(
            tuple(self._space.zeros() if b is None else b for b in blocks)
        )

    def _curvature_wrt_lag(
        self,
        step: int,
        window: StepWindow[V],
        tangents: StepWindow[V],
        lag: int,
        direction: V,
        weights: A,
    ) -> Optional[V]:
        total: Optional[V] = None
        for node in self._nodes(window):
            if node.lag == lag:
                total = _add_scaled(
                    total,
                    node.weight,
                    self._to_state(node, window, tangents, direction, weights),
                    self._space,
                )
        return total

    def _curvature_wrt_param(
        self,
        step: int,
        window: StepWindow[V],
        tangents: StepWindow[V],
        direction: V,
        weights: A,
    ) -> Optional[V]:
        fields, params = self._second_order(), self._params
        if fields.param_state is None or fields.param_param is None or params is None:
            return None
        total: Optional[V] = None
        for node in self._nodes(window):
            state = node_state(window, node)
            mixed = fields.param_state(
                state, node.time, weights, node_state(tangents, node)
            )
            curvature = fields.param_param(state, node.time, weights, direction)
            total = _add_scaled(
                total, node.weight, params.axpby(1.0, curvature, 1.0, mixed), params
            )
        return total

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._integrand!r}, {self._quadrature!r})"


def node_state(window: StepWindow[V], node: QuadratureNode) -> V:
    """The state at ``node`` in ``window``: a block of the step's unknown
    at lag 0, the stored state read at the node's lag otherwise."""
    if node.lag == 0:
        return window.unknown.blocks[node.block]
    return window.past_state(node.lag)


class TimeIntegralSink(Generic[V, A]):
    r"""One solve's :math:`z`, advanced as the steps arrive: each step's
    contribution read through the step term, from the step's window, and
    combined by the step's recursion.

    Keeps the stored states and the values of :math:`z` the method's
    deepest step reads, nothing more.

    Parameters
    ----------
    term : QuadratureStepTerm
        Each step's contribution.
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
        term: QuadratureStepTerm[V, A],
        quadrature: StepQuadratureProtocol,
        depth: int,
    ) -> None:
        self._term = term
        self._quadrature = quadrature
        # Most recent first; z_0 = 0 is None, since no zero of q's type is
        # at hand before the first step.
        self._states: Deque[V] = deque(maxlen=depth)
        self._integrals: Deque[Optional[A]] = deque(maxlen=depth)
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
        # The step's window, as a store would give it for the sweeps.
        read = tuple(list(self._states)[: len(span.sizes)])
        window = StepWindow(span, read, unknown, parameters_version)
        increment = self._term.value(self._ntimes - 1, window)
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

    def value(self) -> A:
        r""":math:`z` at the last state received: the integral so far."""
        latest = self._integrals[0] if self._integrals else None
        if latest is None:
            raise ValueError("no step has been received, so there is no integral")
        return latest

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._term!r}, ntimes={self._ntimes})"
