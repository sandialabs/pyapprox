"""Unit tests for statistics module.

Tests core mathematical functionality with hardcoded matrices.
Integration tests with benchmarks will be added in Phase 8.
"""

import numpy as np
import pytest

from pyapprox.statest.statistics import (
    MultiOutputMean,
    MultiOutputMeanAndVariance,
    MultiOutputVariance,
    _covariance_of_variance_estimator,
    _get_nsamples_intersect,
    _get_nsamples_subset,
    _get_V_from_covariance,
    block_2x2,
)


class TestHelperFunctions:
    """Test helper functions for statistics."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_block_2x2(self, bkd) -> None:
        """Test 2x2 block matrix construction."""
        a = bkd.array([[1.0, 2.0], [3.0, 4.0]])
        b = bkd.array([[5.0, 6.0], [7.0, 8.0]])
        c = bkd.array([[9.0, 10.0], [11.0, 12.0]])
        d = bkd.array([[13.0, 14.0], [15.0, 16.0]])
        result = block_2x2([[a, b], [c, d]], bkd)
        expected = bkd.array(
            [
                [1.0, 2.0, 5.0, 6.0],
                [3.0, 4.0, 7.0, 8.0],
                [9.0, 10.0, 13.0, 14.0],
                [11.0, 12.0, 15.0, 16.0],
            ]
        )
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_get_nsamples_subset_simple(self, bkd) -> None:
        """Test _get_nsamples_subset with simple allocation matrix."""
        # 2 models, allocation matrix for CV estimator
        allocation_mat = bkd.array(
            [
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        npartition_samples = bkd.array([10.0, 5.0])
        result = _get_nsamples_subset(allocation_mat, npartition_samples, bkd)
        # Column 0: no 1s -> 0
        # Column 1: row 0 has 1 -> 10
        # Column 2: row 0 has 1 -> 10
        # Column 3: row 1 has 1 -> 5
        expected = bkd.array([0.0, 10.0, 10.0, 5.0])
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_get_nsamples_intersect_shape(self, bkd) -> None:
        """Test _get_nsamples_intersect returns correct shape."""
        nmodels = 3
        allocation_mat = bkd.zeros((nmodels, 2 * nmodels))
        for ii in range(nmodels):
            allocation_mat[ii, 2 * ii + 1] = 1.0
        npartition_samples = bkd.array([10.0, 5.0, 3.0])
        result = _get_nsamples_intersect(allocation_mat, npartition_samples, bkd)
        bkd.assert_allclose(
            bkd.asarray([result.shape[0], result.shape[1]]),
            bkd.asarray([2 * nmodels, 2 * nmodels]),
        )

    def test_get_V_from_covariance_symmetric(self, bkd) -> None:
        """Test _get_V_from_covariance produces symmetric result."""
        nmodels = 2
        nqoi = 2
        cov = bkd.eye(nmodels * nqoi)
        V = _get_V_from_covariance(cov, nmodels, bkd)
        # V should be symmetric
        bkd.assert_allclose(V, V.T, rtol=1e-12)

    def test_covariance_of_variance_estimator(self, bkd) -> None:
        """Test _covariance_of_variance_estimator formula."""
        nqsq = 4
        W = bkd.eye(nqsq) * 2.0
        V = bkd.eye(nqsq) * 3.0
        nsamples = 10
        result = _covariance_of_variance_estimator(W, V, nsamples)
        # result = W/n + V/(n*(n-1)) = 2/10 + 3/90 = 0.2 + 0.0333... = 0.2333...
        expected_diag = 2.0 / 10 + 3.0 / (10 * 9)
        expected = bkd.eye(nqsq) * expected_diag
        bkd.assert_allclose(result, expected, rtol=1e-10)


class TestMultiOutputMean:
    """Test MultiOutputMean class."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_nstats(self, bkd) -> None:
        """Test nstats returns nqoi."""
        nqoi = 3
        stat = MultiOutputMean(nqoi, bkd)
        bkd.assert_allclose(
            bkd.asarray([stat.nstats()]), bkd.asarray([nqoi])
        )

    def test_sample_estimate(self, bkd) -> None:
        """Test sample_estimate computes mean correctly."""
        nqoi = 2
        nsamples = 100
        stat = MultiOutputMean(nqoi, bkd)
        # Use typing convention (nqoi, nsamples)
        values = bkd.asarray(np.random.randn(nqoi, nsamples))
        estimate = stat.sample_estimate(values)
        expected = bkd.mean(values, axis=1)
        bkd.assert_allclose(estimate, expected, rtol=1e-12)

    def test_set_pilot_quantities(self, bkd) -> None:
        """Test set_pilot_quantities initializes correctly."""
        nqoi = 2
        nmodels = 3
        stat = MultiOutputMean(nqoi, bkd)
        cov = bkd.eye(nmodels * nqoi)
        stat.set_pilot_quantities(cov)
        bkd.assert_allclose(
            bkd.asarray([stat._nmodels]), bkd.asarray([nmodels])
        )

    def test_high_fidelity_estimator_covariance(self, bkd) -> None:
        """Test high_fidelity_estimator_covariance computation."""
        nqoi = 2
        nmodels = 2
        stat = MultiOutputMean(nqoi, bkd)
        # Set diagonal covariance for simplicity
        cov = bkd.eye(nmodels * nqoi) * 4.0
        stat.set_pilot_quantities(cov)
        nhf_samples = 10
        result = stat.high_fidelity_estimator_covariance(nhf_samples)
        # Result should be cov[:nqoi, :nqoi] / nhf_samples
        expected = bkd.eye(nqoi) * (4.0 / nhf_samples)
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_min_nsamples(self, bkd) -> None:
        """Test min_nsamples returns 1."""
        stat = MultiOutputMean(2, bkd)
        bkd.assert_allclose(
            bkd.asarray([stat.min_nsamples()]), bkd.asarray([1])
        )

    def test_subset_creates_valid_statistic(self, bkd) -> None:
        """Test subset creates a valid statistic for model subset."""
        nqoi = 2
        nmodels = 4
        stat = MultiOutputMean(nqoi, bkd)
        # Create a covariance matrix
        cov = bkd.eye(nmodels * nqoi) * 2.0
        stat.set_pilot_quantities(cov)

        # Create subset with models 0, 1, 3
        subset_stat = stat.subset([0, 1, 3])

        # Check nmodels is updated
        bkd.assert_allclose(
            bkd.asarray([subset_stat._nmodels]), bkd.asarray([3])
        )
        # Check nqoi is preserved
        bkd.assert_allclose(
            bkd.asarray([subset_stat._nqoi]), bkd.asarray([nqoi])
        )

    def test_subset_requires_model_zero(self, bkd) -> None:
        """Test subset raises ValueError if 0 not in model_indices."""
        nqoi = 2
        nmodels = 3
        stat = MultiOutputMean(nqoi, bkd)
        cov = bkd.eye(nmodels * nqoi)
        stat.set_pilot_quantities(cov)

        with pytest.raises(ValueError, match="must include 0"):
            stat.subset([1, 2])

    def test_subset_covariance_shape(self, bkd) -> None:
        """Test subset statistic has correct covariance shape."""
        nqoi = 2
        nmodels = 4
        stat = MultiOutputMean(nqoi, bkd)
        # Create a covariance matrix with distinct values
        cov = bkd.asarray(np.random.randn(nmodels * nqoi, nmodels * nqoi))
        cov = cov @ cov.T  # Make positive definite
        stat.set_pilot_quantities(cov)

        # Create subset with models 0, 2
        subset_stat = stat.subset([0, 2])

        # Expected shape: (2*nqoi, 2*nqoi) = (4, 4)
        bkd.assert_allclose(
            bkd.asarray([subset_stat._cov.shape[0], subset_stat._cov.shape[1]]),
            bkd.asarray([2 * nqoi, 2 * nqoi]),
        )

    def test_compute_pilot_quantities(self, bkd) -> None:
        """Test compute_pilot_quantities computes covariance."""
        nqoi = 2
        nsamples = 1000
        stat = MultiOutputMean(nqoi, bkd)
        # Create correlated data with typing convention (nqoi, nsamples)
        values1 = bkd.asarray(np.random.randn(nqoi, nsamples))
        values2 = values1 + bkd.asarray(np.random.randn(nqoi, nsamples) * 0.1)
        pilot_values = [values1, values2]
        (cov,) = stat.compute_pilot_quantities(pilot_values)
        # Check shape is (nmodels * nqoi, nmodels * nqoi)
        nmodels = 2
        bkd.assert_allclose(
            bkd.asarray([cov.shape[0], cov.shape[1]]),
            bkd.asarray([nmodels * nqoi, nmodels * nqoi]),
        )
        # Check symmetry
        bkd.assert_allclose(cov, cov.T, rtol=1e-10)


class TestMultiOutputVariance:
    """Test MultiOutputVariance class."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_nstats_tril(self, bkd) -> None:
        """Test nstats returns nqoi*(nqoi+1)/2 for tril=True."""
        nqoi = 3
        nmodels = 2
        stat = MultiOutputVariance(nqoi, bkd, tril=True)
        cov = bkd.eye(nmodels * nqoi)
        W = bkd.eye(nmodels * nqoi**2)
        stat.set_pilot_quantities(cov, W)
        expected_nstats = nqoi * (nqoi + 1) // 2  # 3*4/2 = 6
        bkd.assert_allclose(
            bkd.asarray([stat.nstats()]), bkd.asarray([expected_nstats])
        )

    def test_min_nsamples(self, bkd) -> None:
        """Test min_nsamples returns 1."""
        stat = MultiOutputVariance(2, bkd)
        bkd.assert_allclose(
            bkd.asarray([stat.min_nsamples()]), bkd.asarray([1])
        )

    def test_subset_creates_valid_statistic(self, bkd) -> None:
        """Test subset creates a valid statistic for model subset."""
        nqoi = 2
        nmodels = 4
        stat = MultiOutputVariance(nqoi, bkd)
        # Create pilot quantities
        cov = bkd.eye(nmodels * nqoi) * 2.0
        W = bkd.eye(nmodels * nqoi**2)
        stat.set_pilot_quantities(cov, W)

        # Create subset with models 0, 1, 3
        subset_stat = stat.subset([0, 1, 3])

        # Check nmodels is updated
        bkd.assert_allclose(
            bkd.asarray([subset_stat._nmodels]), bkd.asarray([3])
        )
        # Check nqoi is preserved
        bkd.assert_allclose(
            bkd.asarray([subset_stat._nqoi]), bkd.asarray([nqoi])
        )
        # Check tril is preserved
        assert subset_stat._tril == stat._tril

    def test_subset_requires_model_zero(self, bkd) -> None:
        """Test subset raises ValueError if 0 not in model_indices."""
        nqoi = 2
        nmodels = 3
        stat = MultiOutputVariance(nqoi, bkd)
        cov = bkd.eye(nmodels * nqoi)
        W = bkd.eye(nmodels * nqoi**2)
        stat.set_pilot_quantities(cov, W)

        with pytest.raises(ValueError, match="must include 0"):
            stat.subset([1, 2])

    def test_subset_pilot_quantities_shapes(self, bkd) -> None:
        """Test subset statistic has correct pilot quantities shapes."""
        nqoi = 2
        nmodels = 4
        stat = MultiOutputVariance(nqoi, bkd)
        cov = bkd.eye(nmodels * nqoi) * 2.0
        W = bkd.eye(nmodels * nqoi**2)
        stat.set_pilot_quantities(cov, W)

        # Create subset with models 0, 2
        subset_stat = stat.subset([0, 2])
        nsub = 2

        # Expected cov shape: (nsub*nqoi, nsub*nqoi)
        bkd.assert_allclose(
            bkd.asarray([subset_stat._cov.shape[0], subset_stat._cov.shape[1]]),
            bkd.asarray([nsub * nqoi, nsub * nqoi]),
        )
        # Expected W shape: (nsub*nqoi^2, nsub*nqoi^2)
        bkd.assert_allclose(
            bkd.asarray([subset_stat._W.shape[0], subset_stat._W.shape[1]]),
            bkd.asarray([nsub * nqoi**2, nsub * nqoi**2]),
        )


class TestMultiOutputMeanAndVariance:
    """Test MultiOutputMeanAndVariance class."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_nstats(self, bkd) -> None:
        """Test nstats returns nqoi + nqoi*(nqoi+1)/2."""
        nqoi = 2
        nmodels = 2
        stat = MultiOutputMeanAndVariance(nqoi, bkd)
        cov = bkd.eye(nmodels * nqoi)
        V_shape = nmodels * nqoi**2
        W = bkd.eye(V_shape)
        B = bkd.zeros((nmodels * nqoi, V_shape))
        stat.set_pilot_quantities(cov, W, B)
        # nstats = nqoi + nqoi*(nqoi+1)/2 = 2 + 3 = 5
        expected_nstats = nqoi + nqoi * (nqoi + 1) // 2
        bkd.assert_allclose(
            bkd.asarray([stat.nstats()]), bkd.asarray([expected_nstats])
        )

    def test_min_nsamples(self, bkd) -> None:
        """Test min_nsamples returns 1."""
        stat = MultiOutputMeanAndVariance(2, bkd)
        bkd.assert_allclose(
            bkd.asarray([stat.min_nsamples()]), bkd.asarray([1])
        )

    def test_set_pilot_quantities_shape_check(self, bkd) -> None:
        """Test set_pilot_quantities validates W shape."""
        nqoi = 2
        nmodels = 2
        stat = MultiOutputMeanAndVariance(nqoi, bkd)
        cov = bkd.eye(nmodels * nqoi)
        # Wrong W shape
        W_wrong = bkd.eye(3)
        B = bkd.zeros((nmodels * nqoi, nmodels * nqoi**2))
        with pytest.raises(ValueError):
            stat.set_pilot_quantities(cov, W_wrong, B)

    def test_subset_creates_valid_statistic(self, bkd) -> None:
        """Test subset creates a valid statistic for model subset."""
        nqoi = 2
        nmodels = 4
        stat = MultiOutputMeanAndVariance(nqoi, bkd)
        # Create pilot quantities
        cov = bkd.eye(nmodels * nqoi) * 2.0
        W = bkd.eye(nmodels * nqoi**2)
        B = bkd.zeros((nmodels * nqoi, nmodels * nqoi**2))
        stat.set_pilot_quantities(cov, W, B)

        # Create subset with models 0, 1, 3
        subset_stat = stat.subset([0, 1, 3])

        # Check nmodels is updated
        bkd.assert_allclose(
            bkd.asarray([subset_stat._nmodels]), bkd.asarray([3])
        )
        # Check nqoi is preserved
        bkd.assert_allclose(
            bkd.asarray([subset_stat._nqoi]), bkd.asarray([nqoi])
        )
        # Check tril is preserved
        assert subset_stat._tril == stat._tril

    def test_subset_requires_model_zero(self, bkd) -> None:
        """Test subset raises ValueError if 0 not in model_indices."""
        nqoi = 2
        nmodels = 3
        stat = MultiOutputMeanAndVariance(nqoi, bkd)
        cov = bkd.eye(nmodels * nqoi)
        W = bkd.eye(nmodels * nqoi**2)
        B = bkd.zeros((nmodels * nqoi, nmodels * nqoi**2))
        stat.set_pilot_quantities(cov, W, B)

        with pytest.raises(ValueError, match="must include 0"):
            stat.subset([1, 2])

    def test_subset_pilot_quantities_shapes(self, bkd) -> None:
        """Test subset statistic has correct pilot quantities shapes."""
        nqoi = 2
        nmodels = 4
        stat = MultiOutputMeanAndVariance(nqoi, bkd)
        cov = bkd.eye(nmodels * nqoi) * 2.0
        W = bkd.eye(nmodels * nqoi**2)
        B = bkd.zeros((nmodels * nqoi, nmodels * nqoi**2))
        stat.set_pilot_quantities(cov, W, B)

        # Create subset with models 0, 2
        subset_stat = stat.subset([0, 2])
        nsub = 2

        # Expected cov shape: (nsub*nqoi, nsub*nqoi)
        bkd.assert_allclose(
            bkd.asarray([subset_stat._cov.shape[0], subset_stat._cov.shape[1]]),
            bkd.asarray([nsub * nqoi, nsub * nqoi]),
        )
        # Expected W shape: (nsub*nqoi^2, nsub*nqoi^2)
        bkd.assert_allclose(
            bkd.asarray([subset_stat._W.shape[0], subset_stat._W.shape[1]]),
            bkd.asarray([nsub * nqoi**2, nsub * nqoi**2]),
        )
        # Expected B shape: (nsub*nqoi, nsub*nqoi^2)
        bkd.assert_allclose(
            bkd.asarray([subset_stat._B.shape[0], subset_stat._B.shape[1]]),
            bkd.asarray([nsub * nqoi, nsub * nqoi**2]),
        )
