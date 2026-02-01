"""Legacy comparison tests for GroupACV and MLBLUE estimators.

These tests compare the typing implementation against the legacy
implementation to ensure numerical equivalence. These tests should
be removed once the legacy module is deleted.

NOTE: These tests only run with NumPy backend since the legacy code
uses a different backend interface.
"""

import unittest

import numpy as np

from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

# Typing imports
from pyapprox.typing.statest.groupacv import (
    GroupACVEstimator as TypingGroupACV,
    get_model_subsets as typing_get_subsets,
)
from pyapprox.typing.statest.statistics import MultiOutputMean
from pyapprox.typing.util.backends.numpy import NumpyBkd

# Legacy imports
from pyapprox.multifidelity.groupacv import (
    GroupACVEstimator as LegacyGroupACV,
    get_model_subsets as legacy_get_subsets,
)
from pyapprox.multifidelity.factory import multioutput_stats
from pyapprox.util.backends.numpy import NumpyMixin as LegacyBkd


class TestGroupACVLegacyComparison(unittest.TestCase):
    """Compare typing GroupACV implementation against legacy."""

    def setUp(self):
        self.legacy_bkd = LegacyBkd
        self.typing_bkd = NumpyBkd()
        np.random.seed(1)

    def test_get_model_subsets(self):
        """Test that get_model_subsets produces identical results."""
        nmodels = 3
        legacy_subsets = legacy_get_subsets(nmodels, self.legacy_bkd)
        typing_subsets = typing_get_subsets(nmodels, self.typing_bkd)

        self.assertEqual(len(legacy_subsets), len(typing_subsets))
        for ls, ts in zip(legacy_subsets, typing_subsets):
            np.testing.assert_allclose(ls, ts)

    def test_groupacv_basic_properties(self):
        """Test that GroupACVEstimator has identical basic properties."""
        nmodels = 3
        cov = np.random.normal(0, 1, (nmodels, nmodels))
        cov = cov.T @ cov
        costs = np.arange(nmodels, 0, -1, dtype=float)

        legacy_stat = multioutput_stats["mean"](1, backend=self.legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyGroupACV(legacy_stat, costs)

        typing_stat = MultiOutputMean(1, self.typing_bkd)
        typing_stat.set_pilot_quantities(self.typing_bkd.array(cov))
        typing_est = TypingGroupACV(
            typing_stat, self.typing_bkd.array(costs)
        )

        self.assertEqual(legacy_est.nmodels(), typing_est.nmodels())
        self.assertEqual(legacy_est.nsubsets(), typing_est.nsubsets())
        self.assertEqual(legacy_est.npartitions(), typing_est.npartitions())

    def test_compute_nsamples_per_model(self):
        """Test that _compute_nsamples_per_model produces identical results."""
        nmodels = 3
        cov = np.random.normal(0, 1, (nmodels, nmodels))
        cov = cov.T @ cov
        costs = np.arange(nmodels, 0, -1, dtype=float)

        legacy_stat = multioutput_stats["mean"](1, backend=self.legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyGroupACV(legacy_stat, costs)

        typing_stat = MultiOutputMean(1, self.typing_bkd)
        typing_stat.set_pilot_quantities(self.typing_bkd.array(cov))
        typing_est = TypingGroupACV(
            typing_stat, self.typing_bkd.array(costs)
        )

        npartition_samples_np = np.arange(
            2.0, 2 + legacy_est.nsubsets(), dtype=float
        )
        npartition_samples = self.typing_bkd.array(npartition_samples_np)

        legacy_nsamples = legacy_est._compute_nsamples_per_model(
            npartition_samples_np
        )
        typing_nsamples = typing_est._compute_nsamples_per_model(
            npartition_samples
        )

        np.testing.assert_allclose(legacy_nsamples, typing_nsamples)

    def test_estimator_cost(self):
        """Test that _estimator_cost produces identical results."""
        nmodels = 3
        cov = np.random.normal(0, 1, (nmodels, nmodels))
        cov = cov.T @ cov
        costs = np.arange(nmodels, 0, -1, dtype=float)

        legacy_stat = multioutput_stats["mean"](1, backend=self.legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyGroupACV(legacy_stat, costs)

        typing_stat = MultiOutputMean(1, self.typing_bkd)
        typing_stat.set_pilot_quantities(self.typing_bkd.array(cov))
        typing_est = TypingGroupACV(
            typing_stat, self.typing_bkd.array(costs)
        )

        npartition_samples_np = np.arange(
            2.0, 2 + legacy_est.nsubsets(), dtype=float
        )
        npartition_samples = self.typing_bkd.array(npartition_samples_np)

        legacy_cost = legacy_est._estimator_cost(npartition_samples_np)
        typing_cost = typing_est._estimator_cost(npartition_samples)

        np.testing.assert_allclose(legacy_cost, typing_cost)

    def test_nintersect_samples(self):
        """Test that _nintersect_samples produces identical results."""
        nmodels = 3
        cov = np.random.normal(0, 1, (nmodels, nmodels))
        cov = cov.T @ cov
        costs = np.arange(nmodels, 0, -1, dtype=float)

        legacy_stat = multioutput_stats["mean"](1, backend=self.legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyGroupACV(legacy_stat, costs)

        typing_stat = MultiOutputMean(1, self.typing_bkd)
        typing_stat.set_pilot_quantities(self.typing_bkd.array(cov))
        typing_est = TypingGroupACV(
            typing_stat, self.typing_bkd.array(costs)
        )

        npartition_samples_np = np.arange(
            2.0, 2 + legacy_est.nsubsets(), dtype=float
        )
        npartition_samples = self.typing_bkd.array(npartition_samples_np)

        legacy_intersect = legacy_est._nintersect_samples(npartition_samples_np)
        typing_intersect = typing_est._nintersect_samples(npartition_samples)

        np.testing.assert_allclose(legacy_intersect, typing_intersect)

    def test_sigma_matrix(self):
        """Test that _sigma produces identical results."""
        nmodels = 3
        cov = np.random.normal(0, 1, (nmodels, nmodels))
        cov = cov.T @ cov
        costs = np.arange(nmodels, 0, -1, dtype=float)

        legacy_stat = multioutput_stats["mean"](1, backend=self.legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyGroupACV(legacy_stat, costs)

        typing_stat = MultiOutputMean(1, self.typing_bkd)
        typing_stat.set_pilot_quantities(self.typing_bkd.array(cov))
        typing_est = TypingGroupACV(
            typing_stat, self.typing_bkd.array(costs)
        )

        npartition_samples_np = np.arange(
            2.0, 2 + legacy_est.nsubsets(), dtype=float
        )
        npartition_samples = self.typing_bkd.array(npartition_samples_np)

        legacy_sigma = legacy_est._sigma(npartition_samples_np)
        typing_sigma = typing_est._sigma(npartition_samples)

        np.testing.assert_allclose(legacy_sigma, typing_sigma)

    def test_covariance_from_npartition_samples(self):
        """Test that _covariance_from_npartition_samples produces identical results."""
        nmodels = 3
        cov = np.random.normal(0, 1, (nmodels, nmodels))
        cov = cov.T @ cov
        costs = np.arange(nmodels, 0, -1, dtype=float)

        legacy_stat = multioutput_stats["mean"](1, backend=self.legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyGroupACV(legacy_stat, costs)

        typing_stat = MultiOutputMean(1, self.typing_bkd)
        typing_stat.set_pilot_quantities(self.typing_bkd.array(cov))
        typing_est = TypingGroupACV(
            typing_stat, self.typing_bkd.array(costs)
        )

        npartition_samples_np = np.arange(
            2.0, 2 + legacy_est.nsubsets(), dtype=float
        )
        npartition_samples = self.typing_bkd.array(npartition_samples_np)

        legacy_cov = legacy_est._covariance_from_npartition_samples(
            npartition_samples_np
        )
        typing_cov = typing_est._covariance_from_npartition_samples(
            npartition_samples
        )

        np.testing.assert_allclose(legacy_cov, typing_cov)

    def test_nested_estimation(self):
        """Test nested estimation type produces identical results."""
        nmodels = 3
        np.random.seed(1)
        cov = np.random.normal(0, 1, (nmodels, nmodels))
        cov = cov.T @ cov
        costs = np.arange(nmodels, 0, -1, dtype=float)

        legacy_stat = multioutput_stats["mean"](1, backend=self.legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyGroupACV(legacy_stat, costs, est_type="nested")

        typing_stat = MultiOutputMean(1, self.typing_bkd)
        typing_stat.set_pilot_quantities(self.typing_bkd.array(cov))
        typing_est = TypingGroupACV(
            typing_stat, self.typing_bkd.array(costs), est_type="nested"
        )

        self.assertEqual(legacy_est.nsubsets(), typing_est.nsubsets())

        npartition_samples_np = np.arange(
            2.0, 2 + legacy_est.nsubsets(), dtype=float
        )
        npartition_samples = self.typing_bkd.array(npartition_samples_np)

        legacy_nsamples = legacy_est._compute_nsamples_per_model(
            npartition_samples_np
        )
        typing_nsamples = typing_est._compute_nsamples_per_model(
            npartition_samples
        )
        np.testing.assert_allclose(legacy_nsamples, typing_nsamples)


if __name__ == "__main__":
    unittest.main()
