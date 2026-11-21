"""Tests for moving an ACV estimator, statistic included, between backends."""

from typing import Callable, List

import numpy as np
import pytest

from pyapprox.statest.acv.base import ACVEstimator
from pyapprox.statest.acv.variants import (
    GISEstimator,
    GMFEstimator,
    GRDEstimator,
    MFMCEstimator,
    MLMCEstimator,
)
from pyapprox.statest.statistics import MultiOutputMean
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend

EstimatorFactory = Callable[
    [MultiOutputMean[Array], Array, Backend[Array]], ACVEstimator[Array]
]

FACTORIES: List[EstimatorFactory] = [
    lambda s, c, b: GMFEstimator(s, c, recursion_index=b.array([0, 0])),
    lambda s, c, b: GISEstimator(s, c, recursion_index=b.array([0, 1])),
    lambda s, c, b: GRDEstimator(s, c, recursion_index=b.array([0, 1])),
    lambda s, c, b: MFMCEstimator(s, c),
    lambda s, c, b: MLMCEstimator(s, c),
]
IDS = ["gmf", "gis", "grd", "mfmc", "mlmc"]


@pytest.fixture(params=["numpy_bkd", "torch_bkd"], ids=["to-numpy", "to-torch"])
def target(request: pytest.FixtureRequest) -> Backend[Array]:
    """The backend an estimator is moved to; the torch case skips without
    torch, through the ``torch_bkd`` fixture."""
    backend: Backend[Array] = request.getfixturevalue(request.param)
    return backend


def _estimator(factory: EstimatorFactory, bkd: Backend[Array]) -> ACVEstimator[Array]:
    np.random.seed(1)
    A = np.random.normal(0, 1, (3, 3))
    stat = MultiOutputMean(1, bkd)
    stat.set_pilot_quantities(bkd.asarray(A @ A.T + 3 * np.eye(3)))
    return factory(stat, bkd.asarray([1.0, 0.1, 0.01]), bkd)


class TestEstimatorWithBackend:
    @pytest.mark.parametrize("factory", FACTORIES, ids=IDS)
    def test_same_estimator_on_target(
        self,
        bkd: Backend[Array],
        factory: EstimatorFactory,
        target: Backend[Array],
    ) -> None:
        est = _estimator(factory, bkd)
        moved = est.with_backend(target)
        assert type(moved) is type(est)
        assert moved.bkd() is target
        target.assert_allclose(
            moved.allocation_matrix(),
            target.asarray(est.allocation_matrix()),
        )

    @pytest.mark.parametrize("factory", FACTORIES, ids=IDS)
    def test_covariance_unchanged(
        self,
        bkd: Backend[Array],
        factory: EstimatorFactory,
        target: Backend[Array],
    ) -> None:
        est = _estimator(factory, bkd)
        moved = est.with_backend(target)
        nps = [10.0, 20.0, 40.0]
        target.assert_allclose(
            moved.covariance_at_npartition_samples(target.asarray(nps)),
            target.asarray(est.covariance_at_npartition_samples(bkd.asarray(nps))),
            rtol=1e-12,
        )

    def test_subclass_must_override(self, numpy_bkd: NumpyBkd) -> None:
        """Inheriting with_backend would silently rebuild the parent class."""

        class _Custom(GMFEstimator[Array]):
            pass

        est = _estimator(
            lambda s, c, b: _Custom(s, c, recursion_index=b.array([0, 0])),
            numpy_bkd,
        )
        with pytest.raises(NotImplementedError, match="_Custom must override"):
            est.with_backend(NumpyBkd())
