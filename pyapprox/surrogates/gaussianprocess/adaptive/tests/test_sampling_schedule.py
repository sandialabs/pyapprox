"""Tests for sampling schedule implementations."""

import pytest

from pyapprox.surrogates.gaussianprocess.adaptive.protocols import (
    SamplingScheduleProtocol,
)
from pyapprox.surrogates.gaussianprocess.adaptive.sampling_schedule import (
    ConstantSamplingSchedule,
    ListSamplingSchedule,
)


class TestConstantSamplingSchedule:
    def test_protocol_compliance(self) -> None:
        schedule = ConstantSamplingSchedule(5, 20)
        assert isinstance(schedule, SamplingScheduleProtocol)

    def test_increments(self) -> None:
        schedule = ConstantSamplingSchedule(5, 20)
        counts = []
        while not schedule.is_exhausted():
            counts.append(schedule.nnew_samples())
        assert counts == [5, 5, 5, 5]

    def test_partial_last_increment(self) -> None:
        schedule = ConstantSamplingSchedule(7, 20)
        counts = []
        while not schedule.is_exhausted():
            counts.append(schedule.nnew_samples())
        assert counts == [7, 7, 6]

    def test_exhausted_raises(self) -> None:
        schedule = ConstantSamplingSchedule(10, 10)
        schedule.nnew_samples()
        assert schedule.is_exhausted()
        with pytest.raises(StopIteration):
            schedule.nnew_samples()

    def test_invalid_increment(self) -> None:
        with pytest.raises(ValueError):
            ConstantSamplingSchedule(0, 10)

    def test_invalid_max(self) -> None:
        with pytest.raises(ValueError):
            ConstantSamplingSchedule(5, 0)


class TestListSamplingSchedule:
    def test_protocol_compliance(self) -> None:
        schedule = ListSamplingSchedule([10, 20, 30])
        assert isinstance(schedule, SamplingScheduleProtocol)

    def test_sequence(self) -> None:
        schedule = ListSamplingSchedule([10, 20, 30])
        counts = []
        while not schedule.is_exhausted():
            counts.append(schedule.nnew_samples())
        assert counts == [10, 20, 30]

    def test_exhausted_raises(self) -> None:
        schedule = ListSamplingSchedule([5])
        schedule.nnew_samples()
        assert schedule.is_exhausted()
        with pytest.raises(StopIteration):
            schedule.nnew_samples()

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            ListSamplingSchedule([])

    def test_negative_increment_raises(self) -> None:
        with pytest.raises(ValueError):
            ListSamplingSchedule([5, -1, 10])
