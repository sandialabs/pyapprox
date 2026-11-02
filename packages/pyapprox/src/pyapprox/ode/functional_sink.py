r"""The value of a functional, computed as its trajectory is computed.

A :class:`FunctionalSink` is the one value engine for the terms of
:mod:`pyapprox.ode.trajectory_terms`: it receives the trajectory from the
integrator, one stored state and one step window at a time, hands each
term the state or window it reads, and adds the values, keeping no
trajectory. The terms are the same objects the sweeps differentiate, read
at the same places, so the value and the derivatives cannot read
different states.

A state term reads a state by index and whether it is the last; the sink
holds the newest state one step back, so it knows a state was not the
last once the next arrives, and reads the one it holds as the last when
asked for the value. A trajectory's length need not be known in advance.

How a step term's contributions :math:`c_n` combine is injected
(:class:`StepCombinationProtocol`): summed (:class:`PlainSum`), or
through the stepper's recursion, so that their combination is the
method's own integral over time, either run forward step by step
(:class:`RunningRecursion`, memory of the method's depth, a running
value) or kept and weighed at the end by the recursion's transpose
(:class:`StoredContributions`, every contribution kept). A new strategy
is another implementation, passed in.
"""

from collections import deque
from typing import (
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

from pyapprox.ode.step_quadrature import (
    StepQuadratureProtocol,
    StepRule,
    recursion_sensitivities,
)
from pyapprox.ode.step_span import StepSpan
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.trajectory_terms import (
    ParameterTermProtocol,
    StateTermProtocol,
    StepTermProtocol,
)
from pyapprox.util.backends.protocols import ArrayProtocol

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)


@runtime_checkable
class StepCombinationProtocol(Protocol[A]):
    """How one solve's step contributions combine into a value. Opened
    for one solve and dropped with it."""

    def add(self, span: StepSpan, contribution: A) -> None:
        """Take the next step's contribution, the step over ``span``."""
        ...

    def value(self) -> Optional[A]:
        """The combination of the contributions taken so far; None before
        any."""
        ...


class PlainSum(Generic[A]):
    r""":math:`\sum_n c_n`: each step's contribution counted once."""

    def __init__(self) -> None:
        self._total: Optional[A] = None

    def add(self, span: StepSpan, contribution: A) -> None:
        self._total = (
            contribution if self._total is None else self._total + contribution
        )

    def value(self) -> Optional[A]:
        return self._total


class RunningRecursion(Generic[A]):
    r"""The stepper's recursion run forward: :math:`\alpha^{(n)}_0
    z_{n+1} + \sum_{l \ge 1} \alpha^{(n)}_l z_{n+1-l} = c_n`, :math:`z_0 =
    0`, the value :math:`z` at the last step. Keeps the method's depth of
    values of :math:`z`; its value so far is the integral so far.

    Parameters
    ----------
    quadrature : StepQuadratureProtocol
        The stepper's rule, giving each step's recursion.
    depth : int
        The most stored states a step reads.
    """

    def __init__(self, quadrature: StepQuadratureProtocol, depth: int) -> None:
        self._quadrature = quadrature
        # Most recent first; z_0 = 0 is None, since no zero of the values'
        # type is at hand before the first contribution.
        self._values: Deque[Optional[A]] = deque([None], maxlen=depth)

    def add(self, span: StepSpan, contribution: A) -> None:
        rule = self._quadrature.rule(span)
        past = [
            contribution * 0.0 if value is None else value
            for value in list(self._values)[: len(rule.alpha) - 1]
        ]
        self._values.appendleft(rule.advance(past, contribution))

    def value(self) -> Optional[A]:
        return self._values[0]


class StoredContributions(Generic[A]):
    r"""The stepper's recursion through its transpose: every contribution
    kept, the value :math:`\sum_n \theta_n c_n` with :math:`\theta` the
    recursion's sensitivities (:func:`~pyapprox.ode.step_quadrature.
    recursion_sensitivities`), the weights the derivatives use. Equal to
    :class:`RunningRecursion` by duality; keeps one contribution per
    step.

    Parameters
    ----------
    quadrature : StepQuadratureProtocol
        The stepper's rule, giving each step's recursion.
    depth : int
        Not needed here; taken so both recursions are built alike.
    """

    def __init__(self, quadrature: StepQuadratureProtocol, depth: int) -> None:
        self._quadrature = quadrature
        self._rules: List[StepRule] = []
        self._contributions: List[A] = []

    def add(self, span: StepSpan, contribution: A) -> None:
        self._rules.append(self._quadrature.rule(span))
        self._contributions.append(contribution)

    def value(self) -> Optional[A]:
        total: Optional[A] = None
        theta = recursion_sensitivities(self._rules)
        for weight, contribution in zip(theta, self._contributions):
            term = weight * contribution
            total = term if total is None else total + term
        return total


class FunctionalSink(Generic[V, A]):
    r"""A functional's value as the trajectory arrives: the parameter
    terms' values, the state terms' at the states they read, and each step
    term's contributions combined as given.

    Parameters
    ----------
    zero : A
        A zero value, shape ``(nqoi, 1)``.
    parameter_terms, state_terms : sequence
        The parameter and state terms.
    step_terms : sequence of (StepTermProtocol, StepCombinationProtocol)
        Each step term with how its contributions combine, fresh for this
        solve.

    Raises
    ------
    ValueError
        From :meth:`append`, if a state was already received; from
        :meth:`append_step`, if none was, or the step does not start where
        the last one ended.
    """

    def __init__(
        self,
        zero: A,
        parameter_terms: Sequence[ParameterTermProtocol[V, A]] = (),
        state_terms: Sequence[StateTermProtocol[V, A]] = (),
        step_terms: Sequence[
            Tuple[StepTermProtocol[V, A], StepCombinationProtocol[A]]
        ] = (),
    ) -> None:
        self._zero = zero
        self._parameter_terms = tuple(parameter_terms)
        self._state_terms = tuple(state_terms)
        self._step_terms = tuple(step_terms)
        # The state terms' values at the states known not to be last.
        self._states_total: A = zero
        # The newest state, held until the next shows it was not the last.
        self._held: Optional[Tuple[int, float, V]] = None
        self._nsteps = 0

    def _state_values(self, index: int, time: float, state: V, last: bool) -> A:
        total = self._zero
        for term in self._state_terms:
            if term.reads(index, last):
                total = total + term.value(index, time, state)
        return total

    def append(self, time: float, state: V, unconstrained: Optional[V] = None) -> None:
        """Take the initial state, held until the next arrives."""
        if self._held is not None:
            raise ValueError(
                "only the initial state is appended at a time; every later state "
                "comes with its step (append_step)"
            )
        self._held = (0, time, state)

    def append_step(
        self, window: StepWindow[V], state: V, unconstrained: Optional[V] = None
    ) -> None:
        """Read the step's window for the step terms, the state it held
        as one not the last for the state terms, and hold the new state."""
        if self._held is None:
            raise ValueError("a step starts from a received state; append one first")
        index, time, held = self._held
        span = window.span
        if span.time() != time:
            raise ValueError(
                f"the step starts at {span.time()}, not where the last one ended, "
                f"{time}"
            )
        self._states_total = self._states_total + self._state_values(
            index, time, held, False
        )
        for term, combination in self._step_terms:
            combination.add(span, term.value(self._nsteps, window))
        self._nsteps += 1
        self._held = (index + 1, span.end(), state)

    def ntimes(self) -> int:
        """The number of states received."""
        return 0 if self._held is None else self._held[0] + 1

    def value(self) -> A:
        """The functional of the trajectory received so far, its newest
        state read as the last.

        Raises
        ------
        ValueError
            If no state was received.
        """
        if self._held is None:
            raise ValueError("no state has been received, so there is no value")
        index, time, state = self._held
        total = self._states_total + self._state_values(index, time, state, True)
        for parameter_term in self._parameter_terms:
            total = total + parameter_term.value()
        for _, combination in self._step_terms:
            combined = combination.value()
            if combined is not None:
                total = total + combined
        return total

    def __repr__(self) -> str:
        return f"{type(self).__name__}(ntimes={self.ntimes()})"
