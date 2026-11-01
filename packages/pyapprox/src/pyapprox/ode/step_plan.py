r"""Which steps a solve takes: injected, and taking them itself.

An integrator records accepted steps and applies the state constraint; it
does not choose steps. A :class:`StepPlanProtocol` does: each solve
starts a fresh :class:`StepSequenceProtocol` from it, which, asked for the
next step, chooses its size, takes it through the solve's step session
and returns it, or returns None when the plan is done. A sequence that
takes the step itself always knows the time exactly (it made the span),
so it lands on the times it must hit without being told where it is, and
the integrator sees accepted steps only.

A fixed grid takes the steps it was given. An adaptive plan is another
implementation of the same protocols: its sequence proposes a size,
takes the trial step, estimates the error, and accepts it or retries
with a smaller one, retries reusing the session's factorizations; the
estimator, the norm and the step-size policy are its own parts, so the
integrator and the steppers need not change for it.

Sizes are the stored truth (see :mod:`pyapprox.ode.step_span`): a plan
holds its start time and its sizes, and its times are derived from them.
"""

import math
from dataclasses import dataclass
from typing import (
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    runtime_checkable,
)

from pyapprox.ode.protocols import StepSessionProtocol
from pyapprox.ode.step_span import StepSpan
from pyapprox.util.linalg.operators import BlockVector

V = TypeVar("V")
W = TypeVar("W")


@dataclass(frozen=True)
class TakenStep(Generic[V]):
    """An accepted step: where it lies and its converged unknown."""

    span: StepSpan
    unknown: BlockVector[V]


@runtime_checkable
class StepSequenceProtocol(Protocol):
    """One solve's steps, chosen and taken one at a time. Opened for one
    solve and dropped with it: it is never shared or reused."""

    def next_step(
        self, session: StepSessionProtocol[W], past_states: Sequence[W]
    ) -> Optional[TakenStep[W]]:
        """Take the next step from ``past_states`` (most recent first, the
        stored states it reads) through ``session``, or return None when
        the plan is done."""
        ...


@runtime_checkable
class StepPlanProtocol(Protocol):
    """Which steps a solve takes, from its start time."""

    def origin(self) -> float:
        """:math:`t_0`, the time of the initial state."""
        ...

    def start(self) -> StepSequenceProtocol:
        """A fresh sequence of this plan's steps, for one solve."""
        ...


class _GivenSequence:
    """The steps of a given list of sizes, taken in order."""

    def __init__(self, origin: float, sizes: Tuple[float, ...]) -> None:
        self._origin, self._sizes = origin, sizes
        self._taken = 0
        self._last: Optional[StepSpan] = None

    def next_step(
        self, session: StepSessionProtocol[W], past_states: Sequence[W]
    ) -> Optional[TakenStep[W]]:
        if self._taken == len(self._sizes):
            return None
        size = self._sizes[self._taken]
        if self._last is None:
            if len(past_states) != 1:
                raise ValueError(
                    f"the first step reads the initial state alone, got "
                    f"{len(past_states)} states"
                )
            span = StepSpan.first(self._origin, size)
        else:
            span = self._last.advanced(size, len(past_states))
        unknown = session.step(past_states, span)
        self._last = span
        self._taken += 1
        return TakenStep(span, unknown)


def _derived_times(origin: float, sizes: Sequence[float]) -> Tuple[float, ...]:
    times = [origin]
    for size in sizes:
        times.append(times[-1] + size)
    return tuple(times)


class GivenSizes:
    """Steps of given sizes from a given start: a fixed grid, a replay of
    an accepted grid, or a frozen grid for derivative checks.

    Parameters
    ----------
    origin : float
        :math:`t_0`.
    sizes : sequence of float
        The step sizes, in order.

    Raises
    ------
    ValueError
        If there is no size, or a size does not advance the time it is
        added to.
    """

    def __init__(self, origin: float, sizes: Sequence[float]) -> None:
        if not sizes:
            raise ValueError("a step plan needs at least one step")
        self._origin = float(origin)
        self._sizes = tuple(float(size) for size in sizes)
        # Every size must advance the time it is added to.
        StepSpan(self._origin, self._sizes)

    @staticmethod
    def from_times(times: Sequence[float]) -> "GivenSizes":
        """The steps between increasing ``times``, sizes their
        differences.

        Raises
        ------
        ValueError
            If adding the differences back does not reproduce every time
            exactly (see :meth:`StepSpan.from_times`).
        """
        span = StepSpan.from_times(times)
        return GivenSizes(span.origin, span.sizes)

    def origin(self) -> float:
        return self._origin

    def sizes(self) -> Tuple[float, ...]:
        """The step sizes, in order."""
        return self._sizes

    def times(self) -> Tuple[float, ...]:
        """Every step boundary, derived from the sizes."""
        return _derived_times(self._origin, self._sizes)

    def start(self) -> _GivenSequence:
        return _GivenSequence(self._origin, self._sizes)

    def __repr__(self) -> str:
        nsteps = len(self._sizes)
        return f"{type(self).__name__}(origin={self._origin}, nsteps={nsteps})"


def _segment_sizes(start: float, stop: float, deltat: float) -> List[float]:
    """Steps of ``deltat`` from ``start`` landing exactly on ``stop``: the
    count decided first (a whole number of steps to a relative 1e-9 is
    that many, otherwise one more), the last size computed by
    subtraction from the derived time before it."""
    ratio = (stop - start) / deltat
    whole = round(ratio)
    if abs(ratio - whole) <= 1e-9 * max(1.0, ratio):
        nfull = whole - 1
    else:
        nfull = math.floor(ratio)
    if nfull < 0:
        raise ValueError(
            f"the interval from {start} to {stop} is shorter than a step "
            f"of {deltat} by more than the 1e-9 whole-step tolerance"
        )
    time = start
    for _ in range(nfull):
        time = time + deltat
    last = stop - time
    if time + last != stop:
        raise ValueError(
            f"the last step from {time} cannot land exactly on {stop}; "
            "choose a stop or start that makes the subtraction exact"
        )
    return [deltat] * nfull + [last]


class FixedStep:
    r"""Steps of one size over an interval, each stop hit exactly.

    Between consecutive stops (and the interval's ends) the steps have
    size ``deltat``, the last of each stretch shortened to land exactly on
    its stop: the number of steps is decided first, so rounding never adds
    a vanishing step, and only the last size is computed, by subtraction
    from the derived time before it.

    Parameters
    ----------
    deltat : float
        The step size.
    interval : tuple of float
        :math:`(t_0, T)`.
    stops : sequence of float, optional
        Times strictly inside the interval, increasing, that the
        trajectory must contain exactly (output or observation times).

    Raises
    ------
    ValueError
        If ``deltat`` is not positive, the interval is empty, the stops
        are not increasing and strictly inside it, or a stretch cannot end
        exactly on its stop.
    """

    def __init__(
        self,
        deltat: float,
        interval: Tuple[float, float],
        stops: Sequence[float] = (),
    ) -> None:
        start, final = float(interval[0]), float(interval[1])
        if not deltat > 0.0:
            raise ValueError(f"deltat must be positive, got {deltat}")
        if not final > start:
            raise ValueError(f"the interval {interval} is empty")
        points = [start, *(float(stop) for stop in stops), final]
        if any(not later > earlier for earlier, later in zip(points, points[1:])):
            raise ValueError(
                f"stops must increase strictly inside {interval}, got {tuple(stops)}"
            )
        sizes: List[float] = []
        for earlier, later in zip(points, points[1:]):
            sizes += _segment_sizes(earlier, later, float(deltat))
        self._deltat = float(deltat)
        self._given = GivenSizes(start, sizes)

    def deltat(self) -> float:
        """The step size."""
        return self._deltat

    def origin(self) -> float:
        return self._given.origin()

    def sizes(self) -> Tuple[float, ...]:
        """The step sizes, in order."""
        return self._given.sizes()

    def times(self) -> Tuple[float, ...]:
        """Every step boundary, derived from the sizes; the stops and the
        interval's end among them exactly."""
        return self._given.times()

    def start(self) -> _GivenSequence:
        return self._given.start()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(deltat={self._deltat}, "
            f"nsteps={len(self.sizes())})"
        )
