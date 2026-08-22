"""Tests that MLBLUEEstimator rejects variance and mean+variance statistics."""

import pytest

from pyapprox.statest.groupacv.mlblue import MLBLUEEstimator
from pyapprox.statest.statistics import (
    MultiOutputMeanAndVariance,
    MultiOutputVariance,
)


class TestMLBLUERejectsNonMeanStats:
    def test_rejects_variance_stat(self, bkd):
        nmodels, nqoi = 3, 1
        cov = bkd.eye(nmodels)
        W = bkd.eye(nmodels)
        costs = bkd.ones((nmodels,))
        stat = MultiOutputVariance(nqoi, bkd)
        stat.set_pilot_quantities(cov, W)
        with pytest.raises(NotImplementedError, match="mean estimation"):
            MLBLUEEstimator(stat, costs)

    def test_rejects_mean_and_variance_stat(self, bkd):
        nmodels, nqoi = 3, 1
        n = nmodels * nqoi
        nsq = nmodels * nqoi**2
        cov = bkd.eye(n)
        V = bkd.eye(nsq)
        W = bkd.eye(nsq)
        B = bkd.zeros((n, nsq))
        costs = bkd.ones((nmodels,))
        stat = MultiOutputMeanAndVariance(nqoi, bkd)
        stat.set_pilot_quantities(cov, V, W, B)
        with pytest.raises(NotImplementedError, match="mean estimation"):
            MLBLUEEstimator(stat, costs)
