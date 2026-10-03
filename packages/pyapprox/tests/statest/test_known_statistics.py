"""Known statistics, and each statistic's say over which it accepts.

A known value is a fact about a model, stated without a statistic; the
statistic it is given to decides whether it is consistent. These tests
pin each statistic's rules, and that a new statistic can bring its own
known kind without anything in the library changing.
"""

from dataclasses import dataclass
from typing import Generic, List, final

import pytest

from pyapprox.statest import (
    KnownMean,
    KnownStatistic,
    KnownVariance,
    MultiOutputMean,
    MultiOutputMeanAndVariance,
    MultiOutputVariance,
)
from pyapprox.statest.protocols import GroupBlockStatistic
from pyapprox.util.backends.protocols import Array, Backend

NQOI = 2


class TestAKnownValueChecksWhatItCan:
    """Its length is the statistic's to check; its form is its own."""

    def test_values_must_be_a_vector(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            KnownMean(1, bkd.zeros((NQOI, 1)))

    @pytest.mark.parametrize("model", [-1, True])
    def test_the_model_must_be_a_non_negative_int(
        self, bkd: Backend[Array], model: int
    ) -> None:
        with pytest.raises(ValueError, match="non-negative int"):
            KnownVariance(model, bkd.zeros((NQOI,)))

    def test_it_satisfies_the_protocol(self, bkd: Backend[Array]) -> None:
        assert isinstance(KnownMean(1, bkd.zeros((NQOI,))), KnownStatistic)


class TestEachStatisticAcceptsItsOwnKinds:
    def test_a_mean_takes_a_known_mean(self, bkd: Backend[Array]) -> None:
        stat = MultiOutputMean(NQOI, bkd)
        assert stat.known_slots(KnownMean(1, bkd.zeros((NQOI,)))) == [0, 1]

    def test_a_mean_refuses_a_known_variance(
        self, bkd: Backend[Array]
    ) -> None:
        stat = MultiOutputMean(NQOI, bkd)
        with pytest.raises(ValueError, match="accepts no known KnownVariance"):
            stat.check_known([KnownVariance(1, bkd.zeros((NQOI,)))])

    def test_a_variance_takes_a_known_variance(
        self, bkd: Backend[Array]
    ) -> None:
        stat = MultiOutputVariance(NQOI, bkd)
        slots = stat.known_slots(KnownVariance(1, bkd.zeros((stat.nstats(),))))
        assert slots == list(range(stat.nstats()))

    def test_a_variance_refuses_a_known_mean(
        self, bkd: Backend[Array]
    ) -> None:
        stat = MultiOutputVariance(NQOI, bkd)
        with pytest.raises(ValueError, match="accepts no known KnownMean"):
            stat.check_known([KnownMean(1, bkd.zeros((NQOI,)))])

    def test_mean_and_variance_place_each_in_its_slots(
        self, bkd: Backend[Array]
    ) -> None:
        stat = MultiOutputMeanAndVariance(NQOI, bkd)
        mean = KnownMean(1, bkd.zeros((NQOI,)))
        variance = KnownVariance(1, bkd.zeros((stat.nstats() - NQOI,)))
        assert stat.known_slots(mean) == [0, 1]
        assert stat.known_slots(variance) == list(range(NQOI, stat.nstats()))


class TestMeanAndVarianceComeTogether:
    def _known(
        self, bkd: Backend[Array], stat: MultiOutputMeanAndVariance[Array]
    ) -> List[KnownStatistic[Array]]:
        return [
            KnownMean(1, bkd.zeros((NQOI,))),
            KnownVariance(1, bkd.zeros((stat.nstats() - NQOI,))),
        ]

    def test_both_are_accepted(self, bkd: Backend[Array]) -> None:
        stat = MultiOutputMeanAndVariance(NQOI, bkd)
        stat.check_known(self._known(bkd, stat))

    @pytest.mark.parametrize("keep", [0, 1])
    def test_one_without_the_other_is_refused(
        self, bkd: Backend[Array], keep: int
    ) -> None:
        stat = MultiOutputMeanAndVariance(NQOI, bkd)
        with pytest.raises(ValueError, match="together: model 1"):
            stat.check_known([self._known(bkd, stat)[keep]])

    def test_each_model_is_judged_alone(self, bkd: Backend[Array]) -> None:
        stat = MultiOutputMeanAndVariance(NQOI, bkd)
        both = self._known(bkd, stat)
        with pytest.raises(ValueError, match="model 2"):
            stat.check_known(both + [KnownMean(2, bkd.zeros((NQOI,)))])


class TestEveryStatisticChecks:
    def test_a_kind_given_twice_for_one_model_is_refused(
        self, bkd: Backend[Array]
    ) -> None:
        stat = MultiOutputMean(NQOI, bkd)
        twice = [KnownMean(1, bkd.zeros((NQOI,)))] * 2
        with pytest.raises(ValueError, match="more than one known KnownMean"):
            stat.check_known(twice)

    def test_the_length_must_match_the_slots(
        self, bkd: Backend[Array]
    ) -> None:
        stat = MultiOutputMean(NQOI, bkd)
        with pytest.raises(ValueError, match=r"expects \(2,\)"):
            stat.check_known([KnownMean(1, bkd.zeros((NQOI + 1,)))])

    @pytest.mark.parametrize(
        "make", [MultiOutputMean, MultiOutputVariance, MultiOutputMeanAndVariance]
    )
    def test_every_statistic_still_satisfies_the_group_contract(
        self, bkd: Backend[Array], make: type
    ) -> None:
        assert isinstance(make(NQOI, bkd), GroupBlockStatistic)


@final
@dataclass(frozen=True)
class _KnownMedian(Generic[Array]):
    """A known kind no library code has heard of."""

    model: int
    values: Array


class _MeanAndMedian(MultiOutputMean[Array]):
    """A new statistic that also accepts a known median."""

    def known_slots(self, known: KnownStatistic[Array]) -> List[int]:
        if isinstance(known, _KnownMedian):
            return list(range(self.nqoi()))
        return super().known_slots(known)


class TestANewStatisticBringsItsOwnKind:
    """The point of the design: nothing in the library is edited."""

    def test_its_kind_is_accepted(self, bkd: Backend[Array]) -> None:
        stat = _MeanAndMedian(NQOI, bkd)
        stat.check_known(
            [_KnownMedian(1, bkd.zeros((NQOI,))), KnownMean(1, bkd.zeros((NQOI,)))]
        )

    def test_another_statistic_refuses_it(self, bkd: Backend[Array]) -> None:
        stat = MultiOutputMean(NQOI, bkd)
        with pytest.raises(ValueError, match="accepts no known _KnownMedian"):
            stat.check_known([_KnownMedian(1, bkd.zeros((NQOI,)))])
