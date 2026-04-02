"""Tests for HP refit schedules."""

import pytest

from pyapprox.optimization.bayesian.hp_schedule import (
    AlwaysOptimizeSchedule,
    EveryKSchedule,
    GeometricSchedule,
    HPRefitScheduleProtocol,
)


class TestAlwaysOptimizeSchedule:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(AlwaysOptimizeSchedule(), HPRefitScheduleProtocol)

    def test_always_true(self) -> None:
        schedule = AlwaysOptimizeSchedule()
        for step in range(20):
            assert schedule.should_optimize(step) is True


class TestEveryKSchedule:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(EveryKSchedule(3), HPRefitScheduleProtocol)

    def test_k_equals_1(self) -> None:
        schedule = EveryKSchedule(1)
        for step in range(10):
            assert schedule.should_optimize(step) is True

    def test_k_equals_3(self) -> None:
        schedule = EveryKSchedule(3)
        expected = {0, 3, 6, 9, 12, 15}
        for step in range(18):
            assert schedule.should_optimize(step) == (step in expected)

    def test_step_zero_always_optimizes(self) -> None:
        for k in [1, 2, 5, 10]:
            assert EveryKSchedule(k).should_optimize(0) is True

    def test_invalid_k(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            EveryKSchedule(0)
        with pytest.raises(ValueError, match="k must be >= 1"):
            EveryKSchedule(-1)


class TestGeometricSchedule:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(GeometricSchedule(), HPRefitScheduleProtocol)

    def test_step_zero_always_optimizes(self) -> None:
        schedule = GeometricSchedule(base=2.0)
        assert schedule.should_optimize(0) is True

    def test_early_steps_dense(self) -> None:
        schedule = GeometricSchedule(base=1.5)
        # Steps 0, 1, 2 should all optimize (intervals are small early)
        for step in range(3):
            assert schedule.should_optimize(step) is True

    def test_later_steps_sparse(self) -> None:
        schedule = GeometricSchedule(base=1.5)
        # Collect which steps optimize in 0..50
        optimize_steps = [s for s in range(51) if schedule.should_optimize(s)]
        # Should have fewer optimize steps than total steps
        assert len(optimize_steps) < 30
        # Should still have some
        assert len(optimize_steps) > 5

    def test_intervals_increase(self) -> None:
        schedule = GeometricSchedule(base=2.0)
        optimize_steps = sorted(
            s for s in range(200) if schedule.should_optimize(s)
        )
        # Gaps between consecutive optimize steps should generally increase
        gaps = [
            optimize_steps[i + 1] - optimize_steps[i]
            for i in range(len(optimize_steps) - 1)
        ]
        # Allow for rounding but overall trend should be non-decreasing
        for i in range(len(gaps) - 1):
            assert gaps[i + 1] >= gaps[i] - 1  # tolerance for rounding

    def test_lazy_extension(self) -> None:
        schedule = GeometricSchedule(base=1.5)
        # Query a step well beyond initial precomputation
        result = schedule.should_optimize(500)
        assert isinstance(result, bool)

    def test_invalid_base(self) -> None:
        with pytest.raises(ValueError, match="base must be > 1.0"):
            GeometricSchedule(base=1.0)
        with pytest.raises(ValueError, match="base must be > 1.0"):
            GeometricSchedule(base=0.5)
