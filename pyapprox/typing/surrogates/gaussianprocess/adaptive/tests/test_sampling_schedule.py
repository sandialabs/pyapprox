"""Tests for sampling schedule implementations."""

import unittest

from pyapprox.typing.surrogates.gaussianprocess.adaptive.protocols import (
    SamplingScheduleProtocol,
)
from pyapprox.typing.surrogates.gaussianprocess.adaptive.sampling_schedule import (
    ConstantSamplingSchedule,
    ListSamplingSchedule,
)


class TestConstantSamplingSchedule(unittest.TestCase):
    def test_protocol_compliance(self) -> None:
        schedule = ConstantSamplingSchedule(5, 20)
        self.assertIsInstance(schedule, SamplingScheduleProtocol)

    def test_increments(self) -> None:
        schedule = ConstantSamplingSchedule(5, 20)
        counts = []
        while not schedule.is_exhausted():
            counts.append(schedule.nnew_samples())
        self.assertEqual(counts, [5, 5, 5, 5])

    def test_partial_last_increment(self) -> None:
        schedule = ConstantSamplingSchedule(7, 20)
        counts = []
        while not schedule.is_exhausted():
            counts.append(schedule.nnew_samples())
        self.assertEqual(counts, [7, 7, 6])

    def test_exhausted_raises(self) -> None:
        schedule = ConstantSamplingSchedule(10, 10)
        schedule.nnew_samples()
        self.assertTrue(schedule.is_exhausted())
        with self.assertRaises(StopIteration):
            schedule.nnew_samples()

    def test_invalid_increment(self) -> None:
        with self.assertRaises(ValueError):
            ConstantSamplingSchedule(0, 10)

    def test_invalid_max(self) -> None:
        with self.assertRaises(ValueError):
            ConstantSamplingSchedule(5, 0)


class TestListSamplingSchedule(unittest.TestCase):
    def test_protocol_compliance(self) -> None:
        schedule = ListSamplingSchedule([10, 20, 30])
        self.assertIsInstance(schedule, SamplingScheduleProtocol)

    def test_sequence(self) -> None:
        schedule = ListSamplingSchedule([10, 20, 30])
        counts = []
        while not schedule.is_exhausted():
            counts.append(schedule.nnew_samples())
        self.assertEqual(counts, [10, 20, 30])

    def test_exhausted_raises(self) -> None:
        schedule = ListSamplingSchedule([5])
        schedule.nnew_samples()
        self.assertTrue(schedule.is_exhausted())
        with self.assertRaises(StopIteration):
            schedule.nnew_samples()

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            ListSamplingSchedule([])

    def test_negative_increment_raises(self) -> None:
        with self.assertRaises(ValueError):
            ListSamplingSchedule([5, -1, 10])


if __name__ == "__main__":
    unittest.main()
