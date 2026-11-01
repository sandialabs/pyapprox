"""Where a step lies: sizes given, times derived by adding forward.

- A span's times are the running sums of its sizes from its origin, its
  size is the one given, and lags count back from the step's end.
- Advancing a span continues the same chain of additions: the times of a
  long run of spans equal the running sums bit for bit, never a product
  such as ``t0 + n h``.
- ``from_times`` recovers sizes by subtraction only when adding them back
  reproduces every time exactly, and refuses otherwise.
- Refusals: no size, a size that does not advance the time (not positive,
  or below the resolution of the time it is added to), a lag out of
  range, and advancing to more states than the step has lags.
"""

import pytest

from pyapprox.ode.step_span import StepSpan


class TestTimes:
    def test_times_are_running_sums_and_the_size_is_given(self) -> None:
        span = StepSpan(0.1, (0.1, 0.1))
        assert span.times() == (0.1, 0.1 + 0.1, 0.1 + 0.1 + 0.1)
        assert span.end() == 0.1 + 0.1 + 0.1
        assert span.end() != 0.3
        assert span.time() == 0.1 + 0.1
        assert span.deltat() == 0.1
        assert [span.lag_time(lag) for lag in range(3)] == list(span.times())[::-1]
        assert span.nlags() == 3

    def test_advancing_continues_the_chain_of_additions(self) -> None:
        """A thousand steps of 0.001, reading up to three states: every
        span's times are the running sums, so a long run drifts from
        0.001 n while its size stays exactly 0.001."""
        sums = [0.0]
        for _ in range(1000):
            sums.append(sums[-1] + 0.001)
        span = StepSpan.first(0.0, 0.001)
        for n in range(1, 1000):
            span = span.advanced(0.001, min(3, n + 1))
            oldest = n + 1 - len(span.sizes)
            assert span.times() == tuple(sums[oldest : n + 2])
            assert span.deltat() == 0.001
        assert span.end() == sums[-1]
        assert span.end() != 1.0


class TestFromTimes:
    def test_round_trips_exactly(self) -> None:
        span = StepSpan.from_times((0.2, 0.25, 0.4))
        assert span.times() == (0.2, 0.25, 0.4)
        assert span.sizes == (0.25 - 0.2, 0.4 - 0.25)

    def test_refuses_times_that_do_not_determine_their_sizes(self) -> None:
        # 1e-17 - (-1) rounds to 1, and -1 + 1 is 0, not 1e-17.
        with pytest.raises(ValueError, match="do not determine their sizes"):
            StepSpan.from_times((-1.0, 1e-17))
        with pytest.raises(ValueError, match="at least two times"):
            StepSpan.from_times((0.0,))


class TestRefusals:
    @pytest.mark.parametrize(
        "origin, sizes",
        [(0.0, (0.0,)), (0.0, (0.1, -0.1)), (1e16, (0.5,))],
        ids=["zero", "negative", "below_resolution"],
    )
    def test_a_size_that_does_not_advance_the_time(
        self, origin: float, sizes: tuple[float, ...]
    ) -> None:
        with pytest.raises(ValueError, match="does not advance"):
            StepSpan(origin, sizes)

    def test_no_size_a_lag_out_of_range_and_too_many_states(self) -> None:
        with pytest.raises(ValueError, match="at least one size"):
            StepSpan(0.0, ())
        span = StepSpan(0.0, (0.5, 0.5))
        with pytest.raises(ValueError, match="between 0 and 2"):
            span.lag_time(3)
        with pytest.raises(ValueError, match="between 1 and 3"):
            span.advanced(0.5, 4)
