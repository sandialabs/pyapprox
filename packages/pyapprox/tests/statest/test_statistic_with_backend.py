"""Tests for moving a statistic, with its pilot quantities, between backends."""

from typing import Callable, List

import numpy as np
import pytest

from pyapprox.statest.acv.variants import GMFEstimator
from pyapprox.statest.statistics import (
    MultiOutputMean,
    MultiOutputMeanAndVariance,
    MultiOutputStatistic,
    MultiOutputVariance,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd

NQOI = 2
NMODELS = 3

StatFactory = Callable[[Backend[Array]], MultiOutputStatistic[Array]]

FACTORIES: List[StatFactory] = [
    lambda bkd: MultiOutputMean(NQOI, bkd),
    lambda bkd: MultiOutputVariance(NQOI, bkd),
    lambda bkd: MultiOutputMeanAndVariance(NQOI, bkd),
]
IDS = ["mean", "variance", "mean-var"]

TARGETS = [NumpyBkd(), TorchBkd()]
TARGET_IDS = ["to-numpy", "to-torch"]


def _with_pilot(
    factory: StatFactory, bkd: Backend[Array]
) -> MultiOutputStatistic[Array]:
    stat = factory(bkd)
    np.random.seed(1)
    base = np.random.normal(0, 1, (NQOI, 200))
    pilot = [
        bkd.asarray(base * 0.9**m + 0.3 * np.random.normal(0, 1, base.shape))
        for m in range(NMODELS)
    ]
    stat.set_pilot_quantities(*stat.compute_pilot_quantities(pilot))
    return stat


class TestWithBackend:
    @pytest.mark.parametrize("factory", FACTORIES, ids=IDS)
    @pytest.mark.parametrize("target", TARGETS, ids=TARGET_IDS)
    def test_moved_statistic_lives_on_target(
        self, bkd: Backend[Array], factory: StatFactory, target: Backend[Array]
    ) -> None:
        stat = _with_pilot(factory, bkd)
        moved = stat.with_backend(target)
        assert type(moved) is type(stat)
        assert moved.bkd() is target
        assert isinstance(moved.pilot_covariance(), type(target.zeros((1,))))
        assert moved.nqoi() == stat.nqoi()
        assert moved.nmodels() == stat.nmodels()
        assert moved.nstats() == stat.nstats()

    @pytest.mark.parametrize("factory", FACTORIES, ids=IDS)
    @pytest.mark.parametrize("target", TARGETS, ids=TARGET_IDS)
    def test_high_fidelity_covariance_unchanged(
        self, bkd: Backend[Array], factory: StatFactory, target: Backend[Array]
    ) -> None:
        stat = _with_pilot(factory, bkd)
        moved = stat.with_backend(target)
        expected = stat.high_fidelity_estimator_covariance(bkd.asarray(10.0))
        actual = moved.high_fidelity_estimator_covariance(target.asarray(10.0))
        target.assert_allclose(
            actual, target.asarray(expected), rtol=1e-12
        )

    @pytest.mark.parametrize("factory", FACTORIES, ids=IDS)
    @pytest.mark.parametrize("target", TARGETS, ids=TARGET_IDS)
    def test_estimator_covariance_unchanged(
        self, bkd: Backend[Array], factory: StatFactory, target: Backend[Array]
    ) -> None:
        # The estimator covariance reads every pilot quantity a statistic
        # holds (W and B as well as the covariance), so agreement here
        # means nothing was dropped or reshaped in the move.
        stat = _with_pilot(factory, bkd)
        moved = stat.with_backend(target)
        costs = [1.0, 0.1, 0.01]
        nps = [10.0, 20.0, 40.0]
        expected = GMFEstimator(stat, costs).covariance_at_npartition_samples(
            bkd.asarray(nps)
        )
        actual = GMFEstimator(moved, costs).covariance_at_npartition_samples(
            target.asarray(nps)
        )
        target.assert_allclose(
            actual, target.asarray(expected), rtol=1e-12
        )

    @pytest.mark.parametrize("factory", FACTORIES, ids=IDS)
    @pytest.mark.parametrize("target", TARGETS, ids=TARGET_IDS)
    def test_without_pilot_quantities(
        self, bkd: Backend[Array], factory: StatFactory, target: Backend[Array]
    ) -> None:
        moved = factory(bkd).with_backend(target)
        assert moved.bkd() is target
        assert not moved.has_pilot_covariance()

    @pytest.mark.parametrize("factory", FACTORIES, ids=IDS)
    def test_source_is_left_alone(
        self, bkd: Backend[Array], factory: StatFactory
    ) -> None:
        stat = _with_pilot(factory, bkd)
        before = bkd.copy(stat.pilot_covariance())
        stat.with_backend(TorchBkd())
        assert stat.bkd() is bkd
        bkd.assert_allclose(stat.pilot_covariance(), before)

    def test_tensor_requiring_grad_is_refused_not_detached(
        self, torch_bkd: TorchBkd
    ) -> None:
        stat = _with_pilot(FACTORIES[0], torch_bkd)
        stat.set_pilot_quantities(
            stat.pilot_covariance().clone().requires_grad_(True)
        )
        with pytest.raises(RuntimeError, match="requires grad"):
            stat.with_backend(NumpyBkd())
