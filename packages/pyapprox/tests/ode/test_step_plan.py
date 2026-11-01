"""Step plans: which steps a solve takes, each taken by its sequence.

- ``FixedStep`` decides the number of steps first and computes only the
  last size, so it lands exactly on the interval's end and on every stop:
  0.1 over [0, 1] is 10 steps ending on 1.0 (not 11 with a vanishing
  last one); 10^6 steps of 0.001 end exactly on 1000, the last absorbing
  the drift of the summed times; 0.3 over [0, 1] is three full steps and
  a short one; a stop at 0.0105 is a stored time.
- ``GivenSizes`` keeps its sizes as given; ``from_times`` keeps a grid's
  times exactly.
- A sequence asks its session for each step over the span it made,
  reading as many states as it is given, advancing the span by the next
  size, and returns None when the plan is done; each start is a fresh
  sequence.
- Refusals: a non-positive step, an empty interval, stops not strictly
  inside and increasing, a stretch shorter than a step, no sizes, and a
  first step given more than the initial state.
"""

from typing import List, Sequence, Tuple

import pytest

from pyapprox.ode.step_plan import (
    FixedStep,
    GivenSizes,
    StepPlanProtocol,
    StepSequenceProtocol,
)
from pyapprox.ode.step_span import StepSpan
from pyapprox.util.linalg.operators import BlockVector


class _RecordingSession:
    """Records the spans and state counts it is asked to step; the
    unknown is the first past state."""

    def __init__(self) -> None:
        self.asked: List[Tuple[StepSpan, int]] = []

    def step(self, past_states: Sequence[float], span: StepSpan) -> BlockVector[float]:
        self.asked.append((span, len(past_states)))
        return BlockVector((past_states[0],))


class TestFixedStep:
    def test_a_whole_number_of_steps_lands_on_the_end(self) -> None:
        plan = FixedStep(0.1, (0.0, 1.0))
        assert isinstance(plan, StepPlanProtocol)
        assert len(plan.sizes()) == 10
        assert plan.times()[-1] == 1.0
        assert plan.sizes()[:-1] == (0.1,) * 9

    def test_a_million_steps_land_exactly(self) -> None:
        plan = FixedStep(0.001, (0.0, 1000.0))
        sizes = plan.sizes()
        assert len(sizes) == 10**6
        assert plan.times()[-1] == 1000.0
        assert set(sizes[:-1]) == {0.001}
        # The last step absorbs the drift of the summed times.
        assert abs(sizes[-1] - 0.001) < 1e-7

    def test_a_shortened_last_step(self) -> None:
        plan = FixedStep(0.3, (0.0, 1.0))
        assert plan.sizes()[:3] == (0.3, 0.3, 0.3)
        assert len(plan.sizes()) == 4
        assert plan.times()[-1] == 1.0
        assert abs(plan.sizes()[-1] - 0.1) < 1e-15

    def test_stops_are_stored_times(self) -> None:
        plan = FixedStep(0.001, (0.0, 0.02), stops=(0.0105,))
        times = plan.times()
        assert 0.0105 in times
        assert times[-1] == 0.02
        assert len(times) == 22

    @pytest.mark.parametrize(
        "deltat, interval, stops, match",
        [
            (0.0, (0.0, 1.0), (), "positive"),
            (0.1, (1.0, 1.0), (), "empty"),
            (0.1, (0.0, 1.0), (1.0,), "strictly inside"),
            (0.1, (0.0, 1.0), (0.6, 0.4), "strictly inside"),
            (1.0, (0.0, 1e-12), (), "shorter than a step"),
        ],
    )
    def test_refusals(
        self,
        deltat: float,
        interval: Tuple[float, float],
        stops: Tuple[float, ...],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            FixedStep(deltat, interval, stops)


class TestGivenSizes:
    def test_sizes_kept_and_times_derived(self) -> None:
        plan = GivenSizes(0.5, (0.25, 0.5))
        assert plan.origin() == 0.5
        assert plan.sizes() == (0.25, 0.5)
        assert plan.times() == (0.5, 0.75, 1.25)
        grid = [0.0, 0.05, 0.08, 0.15]
        assert GivenSizes.from_times(grid).times() == tuple(grid)

    def test_refusals(self) -> None:
        with pytest.raises(ValueError, match="at least one step"):
            GivenSizes(0.0, ())
        with pytest.raises(ValueError, match="does not advance"):
            GivenSizes(0.0, (0.1, 0.0))


class TestSequences:
    def test_a_sequence_takes_each_step_over_its_span(self) -> None:
        plan = GivenSizes(0.0, (0.25, 0.5, 0.25))
        sequence = plan.start()
        assert isinstance(sequence, StepSequenceProtocol)
        session = _RecordingSession()
        # The states read grow to two, as a two-step method's would.
        nread = [1, 2, 2]
        for n, count in enumerate(nread):
            taken = sequence.next_step(session, [float(k) for k in range(count)])
            assert taken is not None
            assert taken.span is session.asked[-1][0]
            assert taken.unknown.blocks == (0.0,)
            assert len(session.asked) == n + 1
        assert sequence.next_step(session, [0.0, 1.0]) is None
        spans = [span for span, _ in session.asked]
        assert [span.times() for span in spans] == [
            (0.0, 0.25),
            (0.0, 0.25, 0.75),
            (0.25, 0.75, 1.0),
        ]
        assert [span.deltat() for span in spans] == [0.25, 0.5, 0.25]
        # Each start is a fresh sequence.
        fresh = plan.start().next_step(_RecordingSession(), [0.0])
        assert fresh is not None and fresh.span.times() == (0.0, 0.25)

    def test_the_first_step_reads_the_initial_state_alone(self) -> None:
        sequence = FixedStep(0.5, (0.0, 1.0)).start()
        with pytest.raises(ValueError, match="initial state alone"):
            sequence.next_step(_RecordingSession(), [0.0, 1.0])
