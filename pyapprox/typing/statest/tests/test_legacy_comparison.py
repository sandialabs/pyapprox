"""Regression tests comparing typing vs legacy implementations.

These tests ensure the new typing module produces identical output to the
legacy multifidelity module.

# TODO: Delete after refactor complete
"""

import unittest

import numpy as np
import torch

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd

# Legacy imports - use legacy backend class (not instance) for legacy code
from pyapprox.util.backends.numpy import NumpyMixin as LegacyNumpyBackend
from pyapprox.util.backends.torch import TorchMixin as LegacyTorchBackend
from pyapprox.multifidelity.stats import (
    MultiOutputMean as LegacyMultiOutputMean,
    MultiOutputVariance as LegacyMultiOutputVariance,
    MultiOutputMeanAndVariance as LegacyMultiOutputMeanAndVariance,
)
from pyapprox.multifidelity.acv import (
    MCEstimator as LegacyMCEstimator,
    CVEstimator as LegacyCVEstimator,
    GMFEstimator as LegacyGMFEstimator,
    GISEstimator as LegacyGISEstimator,
    GRDEstimator as LegacyGRDEstimator,
    MFMCEstimator as LegacyMFMCEstimator,
    MLMCEstimator as LegacyMLMCEstimator,
    _get_allocation_matrix_gmf as legacy_get_allocation_matrix_gmf,
    _get_allocation_matrix_acvis as legacy_get_allocation_matrix_acvis,
    _get_allocation_matrix_acvrd as legacy_get_allocation_matrix_acvrd,
)

# Typing imports
from pyapprox.typing.statest.statistics import (
    MultiOutputMean as TypingMultiOutputMean,
    MultiOutputVariance as TypingMultiOutputVariance,
    MultiOutputMeanAndVariance as TypingMultiOutputMeanAndVariance,
)
from pyapprox.typing.statest.mc_estimator import (
    MCEstimator as TypingMCEstimator,
)
from pyapprox.typing.statest.cv_estimator import (
    CVEstimator as TypingCVEstimator,
)
from pyapprox.typing.statest.acv.optimization import (
    _get_allocation_matrix_gmf as typing_get_allocation_matrix_gmf,
    _get_allocation_matrix_acvis as typing_get_allocation_matrix_acvis,
    _get_allocation_matrix_acvrd as typing_get_allocation_matrix_acvrd,
)
from pyapprox.typing.statest.acv.variants import (
    GMFEstimator as TypingGMFEstimator,
    GISEstimator as TypingGISEstimator,
    GRDEstimator as TypingGRDEstimator,
    MFMCEstimator as TypingMFMCEstimator,
    MLMCEstimator as TypingMLMCEstimator,
)
from pyapprox.typing.util.test_utils import (
    slow_test,
    allocate_with_allocator,
)


class TestLegacyComparisonMultiOutputMean(unittest.TestCase):
    """Compare MultiOutputMean between legacy and typing."""

    # TODO: Delete after refactor complete

    def setUp(self) -> None:
        np.random.seed(42)
        # Legacy backend is the class itself (static methods)
        self._legacy_bkd = LegacyNumpyBackend
        self._typing_bkd = NumpyBkd()

    def test_nstats(self) -> None:
        """Compare nstats."""
        nqoi = 3
        legacy_stat = LegacyMultiOutputMean(nqoi, self._legacy_bkd)
        typing_stat = TypingMultiOutputMean(nqoi, self._typing_bkd)
        self.assertEqual(typing_stat.nstats(), legacy_stat.nstats())

    def test_sample_estimate(self) -> None:
        """Compare sample_estimate."""
        nqoi = 2
        nsamples = 100
        # Use typing convention (nqoi, nsamples)
        values = np.random.randn(nqoi, nsamples)

        legacy_stat = LegacyMultiOutputMean(nqoi, self._legacy_bkd)
        typing_stat = TypingMultiOutputMean(nqoi, self._typing_bkd)

        # Legacy uses (nsamples, nqoi) convention
        legacy_result = legacy_stat.sample_estimate(values.T)
        typing_result = typing_stat.sample_estimate(values)
        np.testing.assert_allclose(typing_result, legacy_result, rtol=1e-12)

    def test_high_fidelity_estimator_covariance(self) -> None:
        """Compare high_fidelity_estimator_covariance."""
        nqoi = 2
        nmodels = 2
        cov = np.eye(nmodels * nqoi) * 4.0

        legacy_stat = LegacyMultiOutputMean(nqoi, self._legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)

        typing_stat = TypingMultiOutputMean(nqoi, self._typing_bkd)
        typing_stat.set_pilot_quantities(cov)

        nhf_samples = 50
        legacy_result = legacy_stat.high_fidelity_estimator_covariance(nhf_samples)
        typing_result = typing_stat.high_fidelity_estimator_covariance(nhf_samples)
        np.testing.assert_allclose(typing_result, legacy_result, rtol=1e-12)

    def test_compute_pilot_quantities(self) -> None:
        """Compare compute_pilot_quantities."""
        nqoi = 2
        nsamples = 500
        # Use typing convention (nqoi, nsamples)
        values1 = np.random.randn(nqoi, nsamples)
        values2 = values1 + np.random.randn(nqoi, nsamples) * 0.1
        pilot_values = [values1, values2]

        legacy_stat = LegacyMultiOutputMean(nqoi, self._legacy_bkd)
        typing_stat = TypingMultiOutputMean(nqoi, self._typing_bkd)

        # Legacy uses (nsamples, nqoi) convention
        pilot_values_legacy = [v.T for v in pilot_values]
        (legacy_cov,) = legacy_stat.compute_pilot_quantities(pilot_values_legacy)
        (typing_cov,) = typing_stat.compute_pilot_quantities(pilot_values)
        np.testing.assert_allclose(typing_cov, legacy_cov, rtol=1e-12)

    def test_get_cv_discrepancy_covariances(self) -> None:
        """Compare _get_cv_discrepancy_covariances."""
        nqoi = 2
        nmodels = 3
        cov = np.eye(nmodels * nqoi) * 2.0
        npartition_samples = np.array([100.0])

        legacy_stat = LegacyMultiOutputMean(nqoi, self._legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)

        typing_stat = TypingMultiOutputMean(nqoi, self._typing_bkd)
        typing_stat.set_pilot_quantities(cov)

        legacy_CF, legacy_cf = legacy_stat._get_cv_discrepancy_covariances(npartition_samples)
        typing_CF, typing_cf = typing_stat._get_cv_discrepancy_covariances(npartition_samples)
        np.testing.assert_allclose(typing_CF, legacy_CF, rtol=1e-12)
        np.testing.assert_allclose(typing_cf, legacy_cf, rtol=1e-12)


class TestLegacyComparisonMultiOutputVariance(unittest.TestCase):
    """Compare MultiOutputVariance between legacy and typing."""

    # TODO: Delete after refactor complete

    def setUp(self) -> None:
        np.random.seed(42)
        self._legacy_bkd = LegacyNumpyBackend
        self._typing_bkd = NumpyBkd()

    def test_nstats_tril(self) -> None:
        """Compare nstats with tril=True."""
        nqoi = 3
        nmodels = 2
        cov = np.eye(nmodels * nqoi)
        W = np.eye(nmodels * nqoi**2)

        legacy_stat = LegacyMultiOutputVariance(nqoi, self._legacy_bkd, tril=True)
        legacy_stat.set_pilot_quantities(cov, W)

        typing_stat = TypingMultiOutputVariance(nqoi, self._typing_bkd, tril=True)
        typing_stat.set_pilot_quantities(cov, W)

        self.assertEqual(typing_stat.nstats(), legacy_stat.nstats())

    def test_sample_estimate(self) -> None:
        """Compare sample_estimate."""
        nqoi = 2
        nmodels = 2
        nsamples = 100
        # Use typing convention (nqoi, nsamples)
        values = np.random.randn(nqoi, nsamples)
        cov = np.eye(nmodels * nqoi)
        W = np.eye(nmodels * nqoi**2)

        legacy_stat = LegacyMultiOutputVariance(nqoi, self._legacy_bkd, tril=True)
        legacy_stat.set_pilot_quantities(cov, W)

        typing_stat = TypingMultiOutputVariance(nqoi, self._typing_bkd, tril=True)
        typing_stat.set_pilot_quantities(cov, W)

        # Legacy uses (nsamples, nqoi) convention
        legacy_result = legacy_stat.sample_estimate(values.T)
        typing_result = typing_stat.sample_estimate(values)
        np.testing.assert_allclose(typing_result, legacy_result, rtol=1e-12)


class TestLegacyComparisonMultiOutputMeanAndVariance(unittest.TestCase):
    """Compare MultiOutputMeanAndVariance between legacy and typing."""

    # TODO: Delete after refactor complete

    def setUp(self) -> None:
        np.random.seed(42)
        self._legacy_bkd = LegacyNumpyBackend
        self._typing_bkd = NumpyBkd()

    def test_nstats(self) -> None:
        """Compare nstats."""
        nqoi = 2
        nmodels = 2
        cov = np.eye(nmodels * nqoi)
        V_shape = nmodels * nqoi**2
        W = np.eye(V_shape)
        B = np.zeros((nmodels * nqoi, V_shape))

        legacy_stat = LegacyMultiOutputMeanAndVariance(nqoi, self._legacy_bkd)
        legacy_stat.set_pilot_quantities(cov, W, B)

        typing_stat = TypingMultiOutputMeanAndVariance(nqoi, self._typing_bkd)
        typing_stat.set_pilot_quantities(cov, W, B)

        self.assertEqual(typing_stat.nstats(), legacy_stat.nstats())

    def test_sample_estimate(self) -> None:
        """Compare sample_estimate."""
        nqoi = 2
        nmodels = 2
        nsamples = 100
        # Use typing convention (nqoi, nsamples)
        values = np.random.randn(nqoi, nsamples)
        cov = np.eye(nmodels * nqoi)
        V_shape = nmodels * nqoi**2
        W = np.eye(V_shape)
        B = np.zeros((nmodels * nqoi, V_shape))

        legacy_stat = LegacyMultiOutputMeanAndVariance(nqoi, self._legacy_bkd)
        legacy_stat.set_pilot_quantities(cov, W, B)

        typing_stat = TypingMultiOutputMeanAndVariance(nqoi, self._typing_bkd)
        typing_stat.set_pilot_quantities(cov, W, B)

        # Legacy uses (nsamples, nqoi) convention
        legacy_result = legacy_stat.sample_estimate(values.T)
        typing_result = typing_stat.sample_estimate(values)
        np.testing.assert_allclose(typing_result, legacy_result, rtol=1e-12)


class TestLegacyComparisonMCEstimator(unittest.TestCase):
    """Compare MCEstimator between legacy and typing."""

    # TODO: Delete after refactor complete

    def setUp(self) -> None:
        np.random.seed(42)
        self._legacy_bkd = LegacyNumpyBackend
        self._typing_bkd = NumpyBkd()

    def test_allocate_samples(self) -> None:
        """Compare allocate_samples."""
        nqoi = 2
        nmodels = 1
        cov = np.eye(nmodels * nqoi) * 4.0
        costs = np.array([2.0])
        target_cost = 100.0

        legacy_stat = LegacyMultiOutputMean(nqoi, self._legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyMCEstimator(legacy_stat, costs)
        legacy_est.allocate_samples(target_cost)

        typing_stat = TypingMultiOutputMean(nqoi, self._typing_bkd)
        typing_stat.set_pilot_quantities(cov)
        typing_est = TypingMCEstimator(typing_stat, costs)
        allocate_with_allocator(typing_est, target_cost)

        np.testing.assert_allclose(
            typing_est._rounded_nsamples_per_model,
            legacy_est._rounded_nsamples_per_model,
            rtol=1e-12
        )
        np.testing.assert_allclose(
            typing_est._rounded_target_cost,
            legacy_est._rounded_target_cost,
            rtol=1e-12
        )

    def test_optimized_covariance(self) -> None:
        """Compare optimized_covariance."""
        nqoi = 2
        nmodels = 1
        cov = np.eye(nmodels * nqoi) * 4.0
        costs = np.array([1.0])
        target_cost = 100.0

        legacy_stat = LegacyMultiOutputMean(nqoi, self._legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyMCEstimator(legacy_stat, costs)
        legacy_est.allocate_samples(target_cost)

        typing_stat = TypingMultiOutputMean(nqoi, self._typing_bkd)
        typing_stat.set_pilot_quantities(cov)
        typing_est = TypingMCEstimator(typing_stat, costs)
        allocate_with_allocator(typing_est, target_cost)

        np.testing.assert_allclose(
            typing_est.optimized_covariance(),
            legacy_est.optimized_covariance(),
            rtol=1e-12
        )

    def test_call(self) -> None:
        """Compare __call__."""
        nqoi = 2
        nmodels = 1
        cov = np.eye(nmodels * nqoi) * 4.0
        costs = np.array([1.0])
        target_cost = 10.0

        legacy_stat = LegacyMultiOutputMean(nqoi, self._legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyMCEstimator(legacy_stat, costs)
        legacy_est.allocate_samples(target_cost)

        typing_stat = TypingMultiOutputMean(nqoi, self._typing_bkd)
        typing_stat.set_pilot_quantities(cov)
        typing_est = TypingMCEstimator(typing_stat, costs)
        allocate_with_allocator(typing_est, target_cost)

        # Use typing convention (nqoi, nsamples)
        values = np.random.randn(nqoi, 10)
        # Legacy uses (nsamples, nqoi) convention
        legacy_result = legacy_est(values.T)
        typing_result = typing_est(values)
        np.testing.assert_allclose(typing_result, legacy_result, rtol=1e-12)

    def test_generate_samples_per_model(self) -> None:
        """Compare generate_samples_per_model."""
        nqoi = 2
        nmodels = 1
        nvars = 3
        cov = np.eye(nmodels * nqoi) * 4.0
        costs = np.array([1.0])
        target_cost = 50.0

        legacy_stat = LegacyMultiOutputMean(nqoi, self._legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyMCEstimator(legacy_stat, costs)
        legacy_est.allocate_samples(target_cost)

        typing_stat = TypingMultiOutputMean(nqoi, self._typing_bkd)
        typing_stat.set_pilot_quantities(cov)
        typing_est = TypingMCEstimator(typing_stat, costs)
        allocate_with_allocator(typing_est, target_cost)

        # Use same seed for rvs
        def rvs(n):
            return np.random.randn(nvars, n)

        np.random.seed(123)
        legacy_samples = legacy_est.generate_samples_per_model(rvs)

        np.random.seed(123)
        typing_samples = typing_est.generate_samples_per_model(rvs)

        self.assertEqual(len(typing_samples), len(legacy_samples))
        for i in range(len(legacy_samples)):
            np.testing.assert_allclose(
                typing_samples[i], legacy_samples[i], rtol=1e-12
            )


class TestLegacyComparisonCVEstimator(unittest.TestCase):
    """Compare CVEstimator between legacy and typing."""

    # TODO: Delete after refactor complete

    def setUp(self) -> None:
        np.random.seed(42)
        self._legacy_bkd = LegacyNumpyBackend
        self._typing_bkd = NumpyBkd()

    def test_allocate_samples(self) -> None:
        """Compare allocate_samples."""
        nqoi = 2
        nmodels = 2
        cov = np.eye(nmodels * nqoi) * 4.0
        costs = np.array([2.0, 1.0])
        target_cost = 90.0
        lowfi_stats = np.zeros((nmodels - 1, nqoi))

        legacy_stat = LegacyMultiOutputMean(nqoi, self._legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyCVEstimator(legacy_stat, costs, lowfi_stats)
        legacy_est.allocate_samples(target_cost)

        typing_stat = TypingMultiOutputMean(nqoi, self._typing_bkd)
        typing_stat.set_pilot_quantities(cov)
        typing_est = TypingCVEstimator(typing_stat, costs, lowfi_stats)
        allocate_with_allocator(typing_est, target_cost)

        np.testing.assert_allclose(
            typing_est._rounded_npartition_samples,
            legacy_est._rounded_npartition_samples,
            rtol=1e-12
        )
        np.testing.assert_allclose(
            typing_est._rounded_nsamples_per_model,
            legacy_est._rounded_nsamples_per_model,
            rtol=1e-12
        )

    def test_optimized_covariance(self) -> None:
        """Compare optimized_covariance."""
        nqoi = 1
        nmodels = 2
        # Covariance with correlation
        cov = np.array([[1.0, 0.8], [0.8, 1.0]])
        costs = np.array([1.0, 0.5])
        target_cost = 100.0
        lowfi_stats = np.zeros((nmodels - 1, nqoi))

        legacy_stat = LegacyMultiOutputMean(nqoi, self._legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyCVEstimator(legacy_stat, costs, lowfi_stats)
        legacy_est.allocate_samples(target_cost)

        typing_stat = TypingMultiOutputMean(nqoi, self._typing_bkd)
        typing_stat.set_pilot_quantities(cov)
        typing_est = TypingCVEstimator(typing_stat, costs, lowfi_stats)
        allocate_with_allocator(typing_est, target_cost)

        np.testing.assert_allclose(
            typing_est.optimized_covariance(),
            legacy_est.optimized_covariance(),
            rtol=1e-12
        )

    def test_optimized_weights(self) -> None:
        """Compare optimized_weights."""
        nqoi = 1
        nmodels = 2
        cov = np.array([[1.0, 0.9], [0.9, 1.0]])
        costs = np.array([1.0, 0.1])
        target_cost = 100.0
        lowfi_stats = np.zeros((nmodels - 1, nqoi))

        legacy_stat = LegacyMultiOutputMean(nqoi, self._legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyCVEstimator(legacy_stat, costs, lowfi_stats)
        legacy_est.allocate_samples(target_cost)

        typing_stat = TypingMultiOutputMean(nqoi, self._typing_bkd)
        typing_stat.set_pilot_quantities(cov)
        typing_est = TypingCVEstimator(typing_stat, costs, lowfi_stats)
        allocate_with_allocator(typing_est, target_cost)

        np.testing.assert_allclose(
            typing_est._optimized_weights,
            legacy_est._optimized_weights,
            rtol=1e-12
        )

    def test_call(self) -> None:
        """Compare __call__."""
        nqoi = 1
        nmodels = 2
        nsamples = 20
        cov = np.array([[1.0, 0.8], [0.8, 1.0]])
        costs = np.array([1.0, 0.5])
        target_cost = float(nsamples) * costs.sum()
        lowfi_stats = np.zeros((nmodels - 1, nqoi))

        legacy_stat = LegacyMultiOutputMean(nqoi, self._legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyCVEstimator(legacy_stat, costs, lowfi_stats)
        legacy_est.allocate_samples(target_cost)

        typing_stat = TypingMultiOutputMean(nqoi, self._typing_bkd)
        typing_stat.set_pilot_quantities(cov)
        typing_est = TypingCVEstimator(typing_stat, costs, lowfi_stats)
        allocate_with_allocator(typing_est, target_cost)

        # Use typing convention (nqoi, nsamples)
        hf_values = np.random.randn(nqoi, nsamples)
        lf_values = np.random.randn(nqoi, nsamples)
        values_per_model = [hf_values, lf_values]

        # Legacy uses (nsamples, nqoi) convention
        values_per_model_legacy = [v.T for v in values_per_model]
        legacy_result = legacy_est(values_per_model_legacy)
        typing_result = typing_est(values_per_model)
        np.testing.assert_allclose(typing_result, legacy_result, rtol=1e-12)

    def test_generate_samples_per_model(self) -> None:
        """Compare generate_samples_per_model."""
        nqoi = 2
        nmodels = 3
        nvars = 4
        cov = np.eye(nmodels * nqoi) * 4.0
        costs = np.array([1.0, 0.5, 0.25])
        target_cost = 100.0
        lowfi_stats = np.zeros((nmodels - 1, nqoi))

        legacy_stat = LegacyMultiOutputMean(nqoi, self._legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyCVEstimator(legacy_stat, costs, lowfi_stats)
        legacy_est.allocate_samples(target_cost)

        typing_stat = TypingMultiOutputMean(nqoi, self._typing_bkd)
        typing_stat.set_pilot_quantities(cov)
        typing_est = TypingCVEstimator(typing_stat, costs, lowfi_stats)
        allocate_with_allocator(typing_est, target_cost)

        def rvs(n):
            return np.random.randn(nvars, n)

        np.random.seed(123)
        legacy_samples = legacy_est.generate_samples_per_model(rvs)

        np.random.seed(123)
        typing_samples = typing_est.generate_samples_per_model(rvs)

        self.assertEqual(len(typing_samples), len(legacy_samples))
        for i in range(len(legacy_samples)):
            np.testing.assert_allclose(
                typing_samples[i], legacy_samples[i], rtol=1e-12
            )

    def test_discrepancy_covariances(self) -> None:
        """Compare _get_discrepancy_covariances."""
        nqoi = 2
        nmodels = 3
        cov = np.eye(nmodels * nqoi) * 2.0
        costs = np.array([1.0, 0.5, 0.25])
        target_cost = 100.0
        lowfi_stats = np.zeros((nmodels - 1, nqoi))

        legacy_stat = LegacyMultiOutputMean(nqoi, self._legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyCVEstimator(legacy_stat, costs, lowfi_stats)
        legacy_est.allocate_samples(target_cost)

        typing_stat = TypingMultiOutputMean(nqoi, self._typing_bkd)
        typing_stat.set_pilot_quantities(cov)
        typing_est = TypingCVEstimator(typing_stat, costs, lowfi_stats)
        allocate_with_allocator(typing_est, target_cost)

        np.testing.assert_allclose(
            typing_est._optimized_CF,
            legacy_est._optimized_CF,
            rtol=1e-12
        )
        np.testing.assert_allclose(
            typing_est._optimized_cf,
            legacy_est._optimized_cf,
            rtol=1e-12
        )


class TestLegacyComparisonAllocationMatrices(unittest.TestCase):
    """Compare allocation matrix functions between legacy and typing."""

    # TODO: Delete after refactor complete

    def setUp(self) -> None:
        self._legacy_bkd = LegacyNumpyBackend
        self._typing_bkd = NumpyBkd()

    def test_get_allocation_matrix_gmf_2models(self) -> None:
        """Compare GMF allocation matrix for 2 models."""
        recursion_index = np.array([0])
        legacy_mat = legacy_get_allocation_matrix_gmf(
            recursion_index, self._legacy_bkd
        )
        typing_mat = typing_get_allocation_matrix_gmf(
            recursion_index, self._typing_bkd
        )
        np.testing.assert_allclose(typing_mat, legacy_mat, rtol=1e-12)

    def test_get_allocation_matrix_gmf_3models(self) -> None:
        """Compare GMF allocation matrix for 3 models."""
        recursion_index = np.array([0, 0])
        legacy_mat = legacy_get_allocation_matrix_gmf(
            recursion_index, self._legacy_bkd
        )
        typing_mat = typing_get_allocation_matrix_gmf(
            recursion_index, self._typing_bkd
        )
        np.testing.assert_allclose(typing_mat, legacy_mat, rtol=1e-12)

    def test_get_allocation_matrix_gmf_chain(self) -> None:
        """Compare GMF allocation matrix with chain recursion."""
        recursion_index = np.array([0, 1, 2])
        legacy_mat = legacy_get_allocation_matrix_gmf(
            recursion_index, self._legacy_bkd
        )
        typing_mat = typing_get_allocation_matrix_gmf(
            recursion_index, self._typing_bkd
        )
        np.testing.assert_allclose(typing_mat, legacy_mat, rtol=1e-12)

    def test_get_allocation_matrix_acvis_2models(self) -> None:
        """Compare ACVIS allocation matrix for 2 models."""
        recursion_index = np.array([0])
        legacy_mat = legacy_get_allocation_matrix_acvis(
            recursion_index, self._legacy_bkd
        )
        typing_mat = typing_get_allocation_matrix_acvis(
            recursion_index, self._typing_bkd
        )
        np.testing.assert_allclose(typing_mat, legacy_mat, rtol=1e-12)

    def test_get_allocation_matrix_acvis_3models(self) -> None:
        """Compare ACVIS allocation matrix for 3 models."""
        recursion_index = np.array([0, 1])
        legacy_mat = legacy_get_allocation_matrix_acvis(
            recursion_index, self._legacy_bkd
        )
        typing_mat = typing_get_allocation_matrix_acvis(
            recursion_index, self._typing_bkd
        )
        np.testing.assert_allclose(typing_mat, legacy_mat, rtol=1e-12)

    def test_get_allocation_matrix_acvrd_2models(self) -> None:
        """Compare ACVRD allocation matrix for 2 models."""
        recursion_index = np.array([0])
        legacy_mat = legacy_get_allocation_matrix_acvrd(
            recursion_index, self._legacy_bkd
        )
        typing_mat = typing_get_allocation_matrix_acvrd(
            recursion_index, self._typing_bkd
        )
        np.testing.assert_allclose(typing_mat, legacy_mat, rtol=1e-12)

    def test_get_allocation_matrix_acvrd_3models(self) -> None:
        """Compare ACVRD allocation matrix for 3 models."""
        recursion_index = np.array([0, 0])
        legacy_mat = legacy_get_allocation_matrix_acvrd(
            recursion_index, self._legacy_bkd
        )
        typing_mat = typing_get_allocation_matrix_acvrd(
            recursion_index, self._typing_bkd
        )
        np.testing.assert_allclose(typing_mat, legacy_mat, rtol=1e-12)


class TestLegacyComparisonGMFEstimator(unittest.TestCase):
    """Compare GMFEstimator between legacy and typing.

    Uses torch backend since GMF optimization requires jacobians.
    """

    # TODO: Delete after refactor complete

    def setUp(self) -> None:
        np.random.seed(42)
        torch.set_default_dtype(torch.float64)
        self._legacy_bkd = LegacyTorchBackend
        self._typing_bkd = TorchBkd()

    @slow_test
    def test_allocate_samples(self) -> None:
        """Compare allocate_samples."""
        nqoi = 1
        # Create covariance with correlations
        cov = torch.tensor([
            [1.0, 0.9, 0.7],
            [0.9, 1.0, 0.8],
            [0.7, 0.8, 1.0],
        ])
        costs = torch.tensor([1.0, 0.1, 0.01])
        target_cost = 100.0
        recursion_index = torch.tensor([0, 0])

        legacy_stat = LegacyMultiOutputMean(nqoi, self._legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyGMFEstimator(
            legacy_stat, costs, recursion_index=recursion_index
        )
        legacy_est.allocate_samples(target_cost)

        typing_stat = TypingMultiOutputMean(nqoi, self._typing_bkd)
        typing_stat.set_pilot_quantities(cov)
        typing_est = TypingGMFEstimator(
            typing_stat, costs, recursion_index=recursion_index
        )
        allocate_with_allocator(typing_est, target_cost)

        np.testing.assert_allclose(
            typing_est._rounded_npartition_samples.numpy(),
            legacy_est._rounded_npartition_samples.numpy(),
            rtol=1e-10
        )
        np.testing.assert_allclose(
            typing_est.optimized_covariance().numpy(),
            legacy_est.optimized_covariance().numpy(),
            rtol=1e-10
        )


class TestLegacyComparisonGISEstimator(unittest.TestCase):
    """Compare GISEstimator between legacy and typing.

    Uses torch backend since GIS optimization requires jacobians.
    """

    # TODO: Delete after refactor complete

    def setUp(self) -> None:
        np.random.seed(42)
        torch.set_default_dtype(torch.float64)
        self._legacy_bkd = LegacyTorchBackend
        self._typing_bkd = TorchBkd()

    @slow_test
    def test_allocate_samples(self) -> None:
        """Compare allocate_samples."""
        nqoi = 1
        cov = torch.tensor([
            [1.0, 0.9, 0.7],
            [0.9, 1.0, 0.8],
            [0.7, 0.8, 1.0],
        ])
        costs = torch.tensor([1.0, 0.1, 0.01])
        target_cost = 100.0
        recursion_index = torch.tensor([0, 1])

        legacy_stat = LegacyMultiOutputMean(nqoi, self._legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyGISEstimator(
            legacy_stat, costs, recursion_index=recursion_index
        )
        legacy_est.allocate_samples(target_cost)

        typing_stat = TypingMultiOutputMean(nqoi, self._typing_bkd)
        typing_stat.set_pilot_quantities(cov)
        typing_est = TypingGISEstimator(
            typing_stat, costs, recursion_index=recursion_index
        )
        allocate_with_allocator(typing_est, target_cost)

        np.testing.assert_allclose(
            typing_est._rounded_npartition_samples.numpy(),
            legacy_est._rounded_npartition_samples.numpy(),
            rtol=1e-10
        )
        np.testing.assert_allclose(
            typing_est.optimized_covariance().numpy(),
            legacy_est.optimized_covariance().numpy(),
            rtol=1e-10
        )


class TestLegacyComparisonGRDEstimator(unittest.TestCase):
    """Compare GRDEstimator between legacy and typing.

    Uses torch backend since GRD optimization requires jacobians.
    """

    # TODO: Delete after refactor complete

    def setUp(self) -> None:
        np.random.seed(42)
        torch.set_default_dtype(torch.float64)
        self._legacy_bkd = LegacyTorchBackend
        self._typing_bkd = TorchBkd()

    @slow_test
    def test_allocate_samples(self) -> None:
        """Compare allocate_samples."""
        nqoi = 1
        cov = torch.tensor([
            [1.0, 0.9, 0.7],
            [0.9, 1.0, 0.8],
            [0.7, 0.8, 1.0],
        ])
        costs = torch.tensor([1.0, 0.1, 0.01])
        target_cost = 100.0
        recursion_index = torch.tensor([0, 0])

        legacy_stat = LegacyMultiOutputMean(nqoi, self._legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyGRDEstimator(
            legacy_stat, costs, recursion_index=recursion_index
        )
        legacy_est.allocate_samples(target_cost)

        typing_stat = TypingMultiOutputMean(nqoi, self._typing_bkd)
        typing_stat.set_pilot_quantities(cov)
        typing_est = TypingGRDEstimator(
            typing_stat, costs, recursion_index=recursion_index
        )
        allocate_with_allocator(typing_est, target_cost)

        np.testing.assert_allclose(
            typing_est._rounded_npartition_samples.numpy(),
            legacy_est._rounded_npartition_samples.numpy(),
            rtol=1e-10
        )
        np.testing.assert_allclose(
            typing_est.optimized_covariance().numpy(),
            legacy_est.optimized_covariance().numpy(),
            rtol=1e-10
        )


class TestLegacyComparisonMFMCEstimator(unittest.TestCase):
    """Compare MFMCEstimator between legacy and typing."""

    # TODO: Delete after refactor complete

    def setUp(self) -> None:
        np.random.seed(42)
        self._legacy_bkd = LegacyNumpyBackend
        self._typing_bkd = NumpyBkd()

    def test_allocate_samples(self) -> None:
        """Compare allocate_samples."""
        nqoi = 1
        # Correlations must decrease monotonically for MFMC
        cov = np.array([
            [1.0, 0.9, 0.7],
            [0.9, 1.0, 0.8],
            [0.7, 0.8, 1.0],
        ])
        # Costs must decrease monotonically for MFMC hierarchy
        costs = np.array([1.0, 0.1, 0.01])
        target_cost = 100.0

        legacy_stat = LegacyMultiOutputMean(nqoi, self._legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyMFMCEstimator(legacy_stat, costs)
        legacy_est.allocate_samples(target_cost)

        typing_stat = TypingMultiOutputMean(nqoi, self._typing_bkd)
        typing_stat.set_pilot_quantities(cov)
        typing_est = TypingMFMCEstimator(typing_stat, costs)
        allocate_with_allocator(typing_est, target_cost)

        np.testing.assert_allclose(
            typing_est._rounded_npartition_samples,
            legacy_est._rounded_npartition_samples,
            rtol=1e-10
        )
        np.testing.assert_allclose(
            typing_est.optimized_covariance(),
            legacy_est.optimized_covariance(),
            rtol=1e-10
        )


class TestLegacyComparisonMLMCEstimator(unittest.TestCase):
    """Compare MLMCEstimator between legacy and typing."""

    # TODO: Delete after refactor complete

    def setUp(self) -> None:
        np.random.seed(42)
        self._legacy_bkd = LegacyNumpyBackend
        self._typing_bkd = NumpyBkd()

    def test_allocate_samples(self) -> None:
        """Compare allocate_samples."""
        nqoi = 1
        # Costs must decrease monotonically for MLMC
        cov = np.array([
            [1.0, 0.9, 0.7],
            [0.9, 1.0, 0.8],
            [0.7, 0.8, 1.0],
        ])
        costs = np.array([1.0, 0.1, 0.01])
        target_cost = 100.0

        legacy_stat = LegacyMultiOutputMean(nqoi, self._legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyMLMCEstimator(legacy_stat, costs)
        legacy_est.allocate_samples(target_cost)

        typing_stat = TypingMultiOutputMean(nqoi, self._typing_bkd)
        typing_stat.set_pilot_quantities(cov)
        typing_est = TypingMLMCEstimator(typing_stat, costs)
        allocate_with_allocator(typing_est, target_cost)

        np.testing.assert_allclose(
            typing_est._rounded_npartition_samples,
            legacy_est._rounded_npartition_samples,
            rtol=1e-10
        )
        np.testing.assert_allclose(
            typing_est.optimized_covariance(),
            legacy_est.optimized_covariance(),
            rtol=1e-10
        )


if __name__ == "__main__":
    unittest.main()
