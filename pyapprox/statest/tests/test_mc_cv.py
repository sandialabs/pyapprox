"""Unit tests for MC and CV estimators.

Tests core functionality with hardcoded matrices.
Integration tests with benchmarks will be added in Phase 8.
"""

import numpy as np
import pytest

from pyapprox.statest.cv_estimator import CVEstimator
from pyapprox.statest.mc_estimator import MCEstimator
from pyapprox.statest.statistics import MultiOutputMean


class TestMCEstimator:
    """Test MCEstimator class."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_init(self, bkd) -> None:
        """Test MCEstimator initialization."""
        nqoi = 2
        nmodels = 1
        stat = MultiOutputMean(nqoi, bkd)
        cov = bkd.eye(nmodels * nqoi) * 4.0
        stat.set_pilot_quantities(cov)
        costs = bkd.array([1.0])
        est = MCEstimator(stat, costs)
        bkd.assert_allclose(
            bkd.asarray([est._nmodels]), bkd.asarray([nmodels])
        )

    def test_allocate_samples(self, bkd) -> None:
        """Test allocate_samples computes correct sample count."""
        nqoi = 2
        nmodels = 1
        stat = MultiOutputMean(nqoi, bkd)
        cov = bkd.eye(nmodels * nqoi) * 4.0
        stat.set_pilot_quantities(cov)
        costs = bkd.array([2.0])
        est = MCEstimator(stat, costs)
        target_cost = 100.0
        est.allocate_samples(target_cost)
        # With cost=2.0 and target_cost=100, nsamples = floor(100/2) = 50
        bkd.assert_allclose(
            bkd.asarray([est._rounded_nsamples_per_model[0]]),
            bkd.asarray([50]),
        )

    def test_optimized_covariance(self, bkd) -> None:
        """Test optimized_covariance returns correct shape."""
        nqoi = 2
        nmodels = 1
        stat = MultiOutputMean(nqoi, bkd)
        cov = bkd.eye(nmodels * nqoi) * 4.0
        stat.set_pilot_quantities(cov)
        costs = bkd.array([1.0])
        est = MCEstimator(stat, costs)
        est.allocate_samples(100.0)
        opt_cov = est.optimized_covariance()
        # Shape should be (nqoi, nqoi)
        bkd.assert_allclose(
            bkd.asarray([opt_cov.shape[0], opt_cov.shape[1]]),
            bkd.asarray([nqoi, nqoi]),
        )

    def test_generate_samples_per_model(self, bkd) -> None:
        """Test generate_samples_per_model returns correct shape."""
        nqoi = 2
        nmodels = 1
        nvars = 3
        stat = MultiOutputMean(nqoi, bkd)
        cov = bkd.eye(nmodels * nqoi) * 4.0
        stat.set_pilot_quantities(cov)
        costs = bkd.array([1.0])
        est = MCEstimator(stat, costs)
        est.allocate_samples(100.0)

        def rvs(nsamples):
            return bkd.asarray(np.random.randn(nvars, nsamples))

        samples = est.generate_samples_per_model(rvs)
        # Should return list with one entry
        bkd.assert_allclose(
            bkd.asarray([len(samples)]), bkd.asarray([1])
        )
        # Shape should be (nvars, nsamples)
        bkd.assert_allclose(
            bkd.asarray([samples[0].shape[0]]), bkd.asarray([nvars])
        )
        bkd.assert_allclose(
            bkd.asarray([samples[0].shape[1]]), bkd.asarray([100])
        )

    def test_call(self, bkd) -> None:
        """Test __call__ computes statistic correctly."""
        nqoi = 2
        nmodels = 1
        stat = MultiOutputMean(nqoi, bkd)
        cov = bkd.eye(nmodels * nqoi) * 4.0
        stat.set_pilot_quantities(cov)
        costs = bkd.array([1.0])
        est = MCEstimator(stat, costs)
        est.allocate_samples(10.0)  # 10 samples
        # Create values with known mean, using typing convention (nqoi, nsamples)
        values = bkd.ones((nqoi, 10)) * 3.0
        result = est(values)
        expected = bkd.ones((nqoi,)) * 3.0
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_call_wrong_shape_raises(self, bkd) -> None:
        """Test __call__ raises ValueError for wrong shape."""
        nqoi = 2
        nmodels = 1
        stat = MultiOutputMean(nqoi, bkd)
        cov = bkd.eye(nmodels * nqoi) * 4.0
        stat.set_pilot_quantities(cov)
        costs = bkd.array([1.0])
        est = MCEstimator(stat, costs)
        est.allocate_samples(10.0)  # 10 samples
        # Wrong number of samples, using typing convention (nqoi, nsamples)
        values = bkd.ones((nqoi, 5))
        with pytest.raises(ValueError):
            est(values)

    def test_covariance_formula(self, bkd) -> None:
        """Test covariance is cov/nsamples for mean statistic."""
        nqoi = 2
        nmodels = 1
        variance = 4.0
        stat = MultiOutputMean(nqoi, bkd)
        cov = bkd.eye(nmodels * nqoi) * variance
        stat.set_pilot_quantities(cov)
        costs = bkd.array([1.0])
        est = MCEstimator(stat, costs)
        nsamples = 100
        est.allocate_samples(float(nsamples))
        opt_cov = est.optimized_covariance()
        # For mean, estimator covariance = cov / nsamples
        expected = bkd.eye(nqoi) * (variance / nsamples)
        bkd.assert_allclose(opt_cov, expected, rtol=1e-12)


class TestCVEstimator:
    """Test CVEstimator class."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_init(self, bkd) -> None:
        """Test CVEstimator initialization."""
        nqoi = 2
        nmodels = 2
        stat = MultiOutputMean(nqoi, bkd)
        cov = bkd.eye(nmodels * nqoi) * 4.0
        stat.set_pilot_quantities(cov)
        costs = bkd.array([1.0, 0.5])
        lowfi_stats = bkd.zeros((nmodels - 1, nqoi))
        est = CVEstimator(stat, costs, lowfi_stats)
        bkd.assert_allclose(
            bkd.asarray([est._nmodels]), bkd.asarray([nmodels])
        )

    def test_init_lowfi_stats_wrong_shape_raises(self, bkd) -> None:
        """Test CVEstimator raises ValueError for wrong lowfi_stats shape."""
        nqoi = 2
        nmodels = 2
        stat = MultiOutputMean(nqoi, bkd)
        cov = bkd.eye(nmodels * nqoi) * 4.0
        stat.set_pilot_quantities(cov)
        costs = bkd.array([1.0, 0.5])
        # Wrong shape: should be (nmodels-1, nstats) = (1, 2)
        lowfi_stats = bkd.zeros((2, nqoi))
        with pytest.raises(ValueError):
            CVEstimator(stat, costs, lowfi_stats)

    def test_allocate_samples(self, bkd) -> None:
        """Test allocate_samples computes correct sample count."""
        nqoi = 2
        nmodels = 2
        stat = MultiOutputMean(nqoi, bkd)
        cov = bkd.eye(nmodels * nqoi) * 4.0
        stat.set_pilot_quantities(cov)
        costs = bkd.array([2.0, 1.0])  # total cost = 3.0
        lowfi_stats = bkd.zeros((nmodels - 1, nqoi))
        est = CVEstimator(stat, costs, lowfi_stats)
        target_cost = 90.0
        est.allocate_samples(target_cost)
        # nsamples = floor(90 / 3) = 30
        bkd.assert_allclose(
            bkd.asarray([est._rounded_npartition_samples[0]]),
            bkd.asarray([30]),
        )

    def test_generate_samples_per_model(self, bkd) -> None:
        """Test generate_samples_per_model returns samples for all models."""
        nqoi = 2
        nmodels = 3
        nvars = 4
        stat = MultiOutputMean(nqoi, bkd)
        cov = bkd.eye(nmodels * nqoi) * 4.0
        stat.set_pilot_quantities(cov)
        costs = bkd.array([1.0, 0.5, 0.25])
        lowfi_stats = bkd.zeros((nmodels - 1, nqoi))
        est = CVEstimator(stat, costs, lowfi_stats)
        est.allocate_samples(100.0)

        def rvs(nsamples):
            return bkd.asarray(np.random.randn(nvars, nsamples))

        samples = est.generate_samples_per_model(rvs)
        # Should return list with nmodels entries
        bkd.assert_allclose(
            bkd.asarray([len(samples)]), bkd.asarray([nmodels])
        )
        # All should have same shape since CV uses same samples for all models
        for i in range(nmodels):
            bkd.assert_allclose(
                bkd.asarray([samples[i].shape[0]]), bkd.asarray([nvars])
            )

    def test_weights(self, bkd) -> None:
        """Test _weights computes optimal control variate weights."""
        nqoi = 1
        nmodels = 2
        stat = MultiOutputMean(nqoi, bkd)
        # Simple diagonal covariance
        cov = bkd.array([[4.0, 2.0], [2.0, 4.0]])
        stat.set_pilot_quantities(cov)
        costs = bkd.array([1.0, 0.5])
        lowfi_stats = bkd.zeros((nmodels - 1, nqoi))
        est = CVEstimator(stat, costs, lowfi_stats)
        est.allocate_samples(100.0)
        # Weights should be set after allocation
        assert est._optimized_weights is not None

    def test_call(self, bkd) -> None:
        """Test __call__ computes CV estimate correctly."""
        nqoi = 1
        nsamples = 10
        stat = MultiOutputMean(nqoi, bkd)
        # Covariance with correlation between models
        cov = bkd.array([[1.0, 0.8], [0.8, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = bkd.array([1.0, 0.5])
        # Known low-fidelity mean
        lowfi_stats = bkd.array([[0.0]])
        est = CVEstimator(stat, costs, lowfi_stats)
        est.allocate_samples(float(nsamples) * costs.sum())

        # Create values for both models using typing convention (nqoi, nsamples)
        hf_values = bkd.ones((nqoi, nsamples)) * 2.0
        lf_values = bkd.ones((nqoi, nsamples)) * 1.0
        values_per_model = [hf_values, lf_values]

        result = est(values_per_model)
        # Result should be scalar-ish (nqoi,)
        bkd.assert_allclose(
            bkd.asarray([result.shape[0]]), bkd.asarray([nqoi])
        )

    def test_call_wrong_type_raises(self, bkd) -> None:
        """Test __call__ raises ValueError for wrong input type."""
        nqoi = 1
        nmodels = 2
        stat = MultiOutputMean(nqoi, bkd)
        cov = bkd.eye(nmodels * nqoi)
        stat.set_pilot_quantities(cov)
        costs = bkd.array([1.0, 0.5])
        lowfi_stats = bkd.zeros((nmodels - 1, nqoi))
        est = CVEstimator(stat, costs, lowfi_stats)
        est.allocate_samples(10.0)

        # Pass list instead of proper array type
        values_per_model = [[[1.0]], [[0.5]]]
        with pytest.raises(ValueError):
            est(values_per_model)

    def test_optimized_covariance_reduced(self, bkd) -> None:
        """Test CV estimator has lower variance than MC."""
        nqoi = 1
        nmodels = 2
        stat = MultiOutputMean(nqoi, bkd)
        # High correlation between models
        cov = bkd.array([[1.0, 0.9], [0.9, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = bkd.array([1.0, 0.1])
        lowfi_stats = bkd.zeros((nmodels - 1, nqoi))
        cv_est = CVEstimator(stat, costs, lowfi_stats)
        cv_est.allocate_samples(100.0)

        # Compare to MC with same budget
        mc_stat = MultiOutputMean(nqoi, bkd)
        mc_cov = bkd.array([[1.0]])
        mc_stat.set_pilot_quantities(mc_cov)
        mc_costs = bkd.array([1.0])
        mc_est = MCEstimator(mc_stat, mc_costs)
        mc_est.allocate_samples(100.0)

        # CV variance should be less than or equal to MC variance
        cv_var = cv_est.optimized_covariance()[0, 0]
        mc_var = mc_est.optimized_covariance()[0, 0]
        # With high correlation, CV should reduce variance
        assert float(cv_var) <= float(mc_var)
