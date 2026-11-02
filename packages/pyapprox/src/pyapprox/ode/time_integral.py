r"""Functionals integrating over time, with the stepper's own method.

:math:`Q = \int_{t_0}^{t_N} q(y, t)\, dt` is computed as the stepper's
method applied to :math:`\dot z = q(y, t)`, :math:`z(t_0) = 0` (see
:mod:`pyapprox.ode.step_quadrature`): the integral is consistent with the
steps by construction, and no rule is chosen apart from the method.

A :class:`TimeIntegral` is reused across solves; each solve takes a fresh
sink from :meth:`TimeIntegral.sink`, which advances :math:`z` as the steps
arrive, reading each step's nodes from its unknown and the stored states
it read, so the solve keeps no trajectory.
"""

from collections import deque
from typing import Deque, Generic, List, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.ode.protocols import ForwardStepperProtocol
from pyapprox.ode.step_quadrature import StepQuadratureProtocol, StepRule
from pyapprox.ode.step_span import StepSpan
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import BlockVector

V = TypeVar("V")
V_contra = TypeVar("V_contra", contravariant=True)
R = TypeVar("R", bound=ArrayProtocol)
R_co = TypeVar("R_co", bound=ArrayProtocol, covariant=True)


@runtime_checkable
class IntegrandProtocol(Protocol[V_contra, R_co]):
    r"""The integrand :math:`q(y, t)` of a functional integrating over
    time."""

    def value(self, state: V_contra, time: float) -> R_co:
        """:math:`q` at a single column ``state`` of the state space and
        ``time``; the state must not be modified."""
        ...


class TimeIntegral(Generic[V, R]):
    r""":math:`\int_{t_0}^{t_N} q(y, t)\, dt` by the stepper's method.

    Parameters
    ----------
    integrand : IntegrandProtocol
        :math:`q`.
    stepper : ForwardStepperProtocol
        The stepper the trajectory is computed with; its method integrates
        :math:`q`.

    Raises
    ------
    TypeError
        If ``integrand`` or ``stepper`` does not satisfy its protocol.
    """

    def __init__(
        self, integrand: IntegrandProtocol[V, R], stepper: ForwardStepperProtocol[V]
    ) -> None:
        if not isinstance(integrand, IntegrandProtocol):
            raise TypeError(
                "integrand must satisfy IntegrandProtocol, got "
                f"{type(integrand).__name__}"
            )
        if not isinstance(stepper, ForwardStepperProtocol):
            raise TypeError(
                "stepper must satisfy ForwardStepperProtocol, got "
                f"{type(stepper).__name__}"
            )
        self._integrand = integrand
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

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._integrand!r}, {self._quadrature!r})"


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
