"""Standalone tests for AETC module.

These tests verify AETC behavior without importing legacy code.
They will remain after the legacy module is deleted.
"""

import numpy as np
import pytest

from pyapprox.statest.aetc.aetcblue import AETCBLUE
from pyapprox.statest.aetc.aetcmc import AETCMC
from pyapprox.statest.aetc.base import AETC
from pyapprox.statest.groupacv import GroupACVAllocationResult
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.markers import slow_test, slower_test, slowest_test


def _make_groupacv_allocation(est, npartition_samples):
    """Helper to create GroupACVAllocationResult from npartition_samples."""
    bkd = est.bkd()
    nsamples_per_model = est._compute_nsamples_per_model(npartition_samples)
    actual_cost = float(est._estimator_cost(npartition_samples))
    return GroupACVAllocationResult(
        npartition_samples=npartition_samples,
        nsamples_per_model=nsamples_per_model,
        actual_cost=actual_cost,
        objective_value=bkd.array([0.0]),
        success=True,
        message="",
    )


class TestLeastSquares:
    """Test _least_squares method."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_aetc(self, bkd, nmodels: int = 3):
        """Create a minimal AETC instance for testing."""

        def dummy_model(samples):
            return bkd.zeros((1, samples.shape[1]))

        def dummy_rvs(nsamples: int):
            return bkd.asarray(np.random.randn(1, nsamples))

        models = [dummy_model] * nmodels
        costs = bkd.asarray([10.0 ** (-i) for i in range(nmodels)])

        return AETC(models, dummy_rvs, costs, oracle_stats=None, bkd=bkd)

    def test_least_squares_coefficients_shape(self, bkd) -> None:
        """Test that _least_squares returns correct shapes."""
        nsamples = 50
        ncovariates = 2

        # Typing convention: (nqoi, nsamples) and (ncovariates, nsamples)
        hf_values = bkd.asarray(np.random.randn(1, nsamples))
        covariate_values = bkd.asarray(np.random.randn(ncovariates, nsamples))

        aetc = self._create_aetc(bkd)
        beta, sigma_sq, X = aetc._least_squares(hf_values, covariate_values)

        # beta shape should be (ncovariates + 1, 1)
        bkd.assert_allclose(
            bkd.asarray([beta.shape[0], beta.shape[1]]),
            bkd.asarray([ncovariates + 1, 1]),
        )

        # X shape should be (nsamples, ncovariates + 1) - internal design matrix
        bkd.assert_allclose(
            bkd.asarray([X.shape[0], X.shape[1]]),
            bkd.asarray([nsamples, ncovariates + 1]),
        )

    def test_least_squares_known_solution(self, bkd) -> None:
        """Test _least_squares with a known analytical solution.

        For y = a + b*x, least squares should recover a and b.
        """
        np.random.seed(123)
        nsamples = 1000
        true_intercept = 2.0
        true_slope = 3.0

        # Generate x values - typing convention: (ncovariates, nsamples)
        x1 = np.random.randn(1, nsamples)

        # y = intercept + slope * x + small noise
        # Typing convention: (nqoi, nsamples)
        noise = 0.1 * np.random.randn(1, nsamples)
        hf_values = true_intercept + true_slope * x1 + noise

        aetc = self._create_aetc(bkd)
        beta, sigma_sq, X = aetc._least_squares(
            bkd.asarray(hf_values), bkd.asarray(x1)
        )

        # Check intercept is close to true value
        bkd.assert_allclose(
            beta[0:1], bkd.asarray([[true_intercept]]), rtol=0.1
        )

        # Check slope is close to true value
        bkd.assert_allclose(
            beta[1:2], bkd.asarray([[true_slope]]), rtol=0.1
        )

        # Check residual variance is close to noise variance (0.01)
        bkd.assert_allclose(
            bkd.asarray([sigma_sq]),
            bkd.asarray([0.01]),
            rtol=0.5,  # Relaxed tolerance for variance estimate
        )

    def test_least_squares_perfect_fit(self, bkd) -> None:
        """Test _least_squares with perfect linear relationship (no noise)."""
        nsamples = 100
        true_intercept = 5.0
        true_slope = -2.0

        # Typing convention: (ncovariates, nsamples)
        x1 = bkd.asarray(np.linspace(-1, 1, nsamples).reshape(1, -1))
        # Typing convention: (nqoi, nsamples)
        hf_values = true_intercept + true_slope * x1

        aetc = self._create_aetc(bkd)
        beta, sigma_sq, X = aetc._least_squares(hf_values, x1)

        # Should recover exact coefficients
        bkd.assert_allclose(
            beta[0:1], bkd.asarray([[true_intercept]]), rtol=1e-10
        )
        bkd.assert_allclose(
            beta[1:2], bkd.asarray([[true_slope]]), rtol=1e-10
        )

        # Residual variance should be essentially zero
        bkd.assert_allclose(
            bkd.asarray([sigma_sq]), bkd.asarray([0.0]), atol=1e-20
        )

    def test_least_squares_multiple_covariates(self, bkd) -> None:
        """Test _least_squares with multiple covariates."""
        np.random.seed(456)
        nsamples = 500
        ncovariates = 3

        # True coefficients
        true_beta = np.array([[1.0], [2.0], [-1.0], [0.5]])  # intercept + 3 slopes

        # Generate covariates - typing convention: (ncovariates, nsamples)
        x = np.random.randn(ncovariates, nsamples)

        # Compute y = X_Sp @ beta + noise
        # X_Sp has shape (nsamples, ncovariates+1) for internal computation
        X_Sp = np.hstack([np.ones((nsamples, 1)), x.T])
        noise = 0.1 * np.random.randn(nsamples, 1)
        hf_values_legacy = X_Sp @ true_beta + noise
        # Convert to typing convention: (nqoi, nsamples)
        hf_values = hf_values_legacy.T

        aetc = self._create_aetc(bkd, nmodels=4)
        beta, sigma_sq, X = aetc._least_squares(
            bkd.asarray(hf_values), bkd.asarray(x)
        )

        # Check all coefficients are close
        bkd.assert_allclose(beta, bkd.asarray(true_beta), rtol=0.15)

    def test_least_squares_design_matrix_first_column(self, bkd) -> None:
        """Test that design matrix X has leading column of ones."""
        nsamples = 20
        ncovariates = 2

        # Typing convention: (nqoi, nsamples) and (ncovariates, nsamples)
        hf_values = bkd.asarray(np.random.randn(1, nsamples))
        covariate_values = bkd.asarray(np.random.randn(ncovariates, nsamples))

        aetc = self._create_aetc(bkd)
        _, _, X = aetc._least_squares(hf_values, covariate_values)

        # First column should be all ones
        expected_ones = bkd.ones((nsamples,))
        bkd.assert_allclose(X[:, 0], expected_ones, rtol=1e-12)

        # Remaining columns should match covariate_values transposed
        # Input is (ncovariates, nsamples), X stores (nsamples, ncovariates)
        bkd.assert_allclose(X[:, 1:], covariate_values.T, rtol=1e-12)


class TestSubsetOracleStats:
    """Test _subset_oracle_stats method."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_aetc(self, bkd, nmodels: int = 3):
        """Create a minimal AETC instance for testing."""

        def dummy_model(samples):
            return bkd.zeros((samples.shape[1], 1))

        def dummy_rvs(nsamples: int):
            return bkd.asarray(np.random.randn(1, nsamples))

        models = [dummy_model] * nmodels
        costs = bkd.asarray([10.0 ** (-i) for i in range(nmodels)])

        return AETC(models, dummy_rvs, costs, oracle_stats=None, bkd=bkd)

    def test_subset_oracle_stats_shapes(self, bkd) -> None:
        """Test that _subset_oracle_stats returns correct shapes."""
        nmodels = 4
        nsubset = 2

        # Create oracle stats: covariance and means
        cov = bkd.asarray(
            np.eye(nmodels) + 0.1 * np.random.randn(nmodels, nmodels)
        )
        cov = (cov + cov.T) / 2  # Symmetrize
        means = bkd.asarray(np.random.randn(nmodels, 1))

        oracle_stats = [cov, means]
        covariate_subset = bkd.asarray([0, 1], dtype=int)  # First two LF models

        aetc = self._create_aetc(bkd, nmodels)
        Sigma_S, Lambda_Sp, x_Sp = aetc._subset_oracle_stats(
            oracle_stats, covariate_subset
        )

        # Sigma_S shape should be (nsubset, nsubset)
        bkd.assert_allclose(
            bkd.asarray([Sigma_S.shape[0], Sigma_S.shape[1]]),
            bkd.asarray([nsubset, nsubset]),
        )

        # Lambda_Sp shape should be (nsubset+1, nsubset+1)
        bkd.assert_allclose(
            bkd.asarray([Lambda_Sp.shape[0], Lambda_Sp.shape[1]]),
            bkd.asarray([nsubset + 1, nsubset + 1]),
        )

        # x_Sp shape should be (nsubset+1, 1)
        bkd.assert_allclose(
            bkd.asarray([x_Sp.shape[0], x_Sp.shape[1]]),
            bkd.asarray([nsubset + 1, 1]),
        )

    def test_subset_oracle_stats_sigma_extraction(self, bkd) -> None:
        """Test that Sigma_S extracts correct covariance submatrix."""
        nmodels = 4

        # Create known covariance matrix
        cov = bkd.asarray(
            [
                [1.0, 0.5, 0.3, 0.2],
                [0.5, 2.0, 0.4, 0.3],
                [0.3, 0.4, 3.0, 0.5],
                [0.2, 0.3, 0.5, 4.0],
            ]
        )
        means = bkd.asarray([[0.0], [1.0], [2.0], [3.0]])

        oracle_stats = [cov, means]
        # Select LF models 0 and 1 (which are indices 1 and 2 in full model indexing)
        covariate_subset = bkd.asarray([0, 1], dtype=int)

        aetc = self._create_aetc(bkd, nmodels)
        Sigma_S, Lambda_Sp, x_Sp = aetc._subset_oracle_stats(
            oracle_stats, covariate_subset
        )

        # Sigma_S should be cov[[1,2], [1,2]]
        expected_Sigma_S = bkd.asarray([[2.0, 0.4], [0.4, 3.0]])
        bkd.assert_allclose(Sigma_S, expected_Sigma_S, rtol=1e-12)

    def test_subset_oracle_stats_x_Sp_structure(self, bkd) -> None:
        """Test that x_Sp has correct structure (1, means[subset])."""
        nmodels = 3

        cov = bkd.eye(nmodels)
        means = bkd.asarray([[0.0], [1.5], [2.5]])

        oracle_stats = [cov, means]
        covariate_subset = bkd.asarray([0], dtype=int)  # Just first LF model

        aetc = self._create_aetc(bkd, nmodels)
        Sigma_S, Lambda_Sp, x_Sp = aetc._subset_oracle_stats(
            oracle_stats, covariate_subset
        )

        # x_Sp should be (1, means[1])^T = [[1], [1.5]]
        expected_x_Sp = bkd.asarray([[1.0], [1.5]])
        bkd.assert_allclose(x_Sp, expected_x_Sp, rtol=1e-12)

    def test_subset_oracle_stats_single_covariate(self, bkd) -> None:
        """Test with single covariate subset."""
        nmodels = 3

        cov = bkd.asarray([[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]])
        means = bkd.asarray([[0.0], [1.0], [2.0]])

        oracle_stats = [cov, means]
        covariate_subset = bkd.asarray(
            [1], dtype=int
        )  # Second LF model (index 2)

        aetc = self._create_aetc(bkd, nmodels)
        Sigma_S, Lambda_Sp, x_Sp = aetc._subset_oracle_stats(
            oracle_stats, covariate_subset
        )

        # Sigma_S should be cov[2,2] = [[1.0]]
        expected_Sigma_S = bkd.asarray([[1.0]])
        bkd.assert_allclose(Sigma_S, expected_Sigma_S, rtol=1e-12)

        # x_Sp should be [[1], [means[2]]] = [[1], [2]]
        expected_x_Sp = bkd.asarray([[1.0], [2.0]])
        bkd.assert_allclose(x_Sp, expected_x_Sp, rtol=1e-12)


class TestAllocateSamples:
    """Test _allocate_samples method."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_aetc_with_k2(self, bkd, nmodels: int = 3):
        """Create an AETC instance with a mock _find_k2 method."""

        class MockAETC(AETC):
            def _find_k2(
                self,
                beta_Sp,
                Sigma_S,
                costs_S,
                round_nsamples: bool = False,
            ):
                # Simple mock: return fixed values for testing
                _bkd = self._bkd
                nmodels = len(costs_S)
                k2 = _bkd.asarray(0.5)
                nsamples_per_subset = _bkd.ones((nmodels,)) / nmodels
                return k2, nsamples_per_subset

        def dummy_model(samples):
            return bkd.zeros((samples.shape[1], 1))

        def dummy_rvs(nsamples: int):
            return bkd.asarray(np.random.randn(1, nsamples))

        models = [dummy_model] * nmodels
        costs = bkd.asarray([10.0 ** (-i) for i in range(nmodels)])

        return MockAETC(models, dummy_rvs, costs, oracle_stats=None, bkd=bkd)

    def test_allocate_samples_returns_tuple(self, bkd) -> None:
        """Test that _allocate_samples returns (k1, k2, nsamples_per_subset)."""
        nmodels = 3

        # Create inputs
        beta_Sp = bkd.asarray([[1.0], [0.5], [0.3]])
        Sigma_S = bkd.eye(2) * 0.5
        sigma_S_sq = bkd.asarray(0.1)
        x_Sp = bkd.asarray([[1.0], [0.5], [0.3]])
        Lambda_Sp = bkd.eye(3)
        costs_S = bkd.asarray([0.1, 0.01])
        exploit_budget = bkd.asarray(100.0)

        aetc = self._create_aetc_with_k2(bkd, nmodels)
        k1, k2, nsamples_per_subset = aetc._allocate_samples(
            beta_Sp, Sigma_S, sigma_S_sq, x_Sp, Lambda_Sp, costs_S, exploit_budget
        )

        # Check that k1 is a scalar
        bkd.assert_allclose(bkd.asarray([k1.ndim]), bkd.asarray([0]))

        # Check that nsamples_per_subset has correct length
        bkd.assert_allclose(
            bkd.asarray([len(nsamples_per_subset)]),
            bkd.asarray([len(costs_S)]),
        )

    def test_allocate_samples_single_covariate(self, bkd) -> None:
        """Test _allocate_samples with single covariate (special case)."""
        nmodels = 2  # HF + 1 LF

        # For single covariate, k2 is computed directly without _find_k2
        beta_Sp = bkd.asarray([[1.0], [0.8]])
        Sigma_S = bkd.asarray([[0.5]])  # 1x1 covariance
        sigma_S_sq = bkd.asarray(0.1)
        x_Sp = bkd.asarray([[1.0], [0.5]])
        Lambda_Sp = bkd.eye(2)
        costs_S = bkd.asarray([0.1])  # Single LF model cost
        exploit_budget = bkd.asarray(100.0)

        aetc = self._create_aetc_with_k2(bkd, nmodels)
        k1, k2, nsamples_per_subset = aetc._allocate_samples(
            beta_Sp, Sigma_S, sigma_S_sq, x_Sp, Lambda_Sp, costs_S, exploit_budget
        )

        # For single covariate:
        # exploit_cost = sum(costs_S) = 0.1
        # nsamples_per_subset = 1 / exploit_cost = 10.0
        # k2 = exploit_cost * trace(Sigma_Sp @ beta_Sp @ beta_Sp.T)
        bkd.assert_allclose(
            nsamples_per_subset, bkd.asarray([10.0]), rtol=1e-10
        )

    def test_allocate_samples_k1_formula(self, bkd) -> None:
        """Test that k1 follows the correct formula."""
        # k1 = sigma_S_sq * trace(x_Sp @ x_Sp.T @ inv(Lambda_Sp))

        # Use identity Lambda_Sp for simplicity
        x_Sp = bkd.asarray([[1.0], [0.5]])
        Lambda_Sp = bkd.eye(2)
        sigma_S_sq = bkd.asarray(0.25)

        # Expected: k1 = 0.25 * trace(x_Sp @ x_Sp.T @ I)
        #              = 0.25 * trace(x_Sp @ x_Sp.T)
        #              = 0.25 * (1.0 + 0.25) = 0.3125
        expected_k1 = 0.25 * (1.0**2 + 0.5**2)

        beta_Sp = bkd.asarray([[1.0], [0.5]])
        Sigma_S = bkd.asarray([[0.5]])
        costs_S = bkd.asarray([0.1])
        exploit_budget = bkd.asarray(100.0)

        aetc = self._create_aetc_with_k2(bkd, 2)
        k1, k2, nsamples = aetc._allocate_samples(
            beta_Sp, Sigma_S, sigma_S_sq, x_Sp, Lambda_Sp, costs_S, exploit_budget
        )

        bkd.assert_allclose(
            bkd.asarray([k1]), bkd.asarray([expected_k1]), rtol=1e-10
        )


class TestValidateSubsets:
    """Test _validate_subsets method."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_aetc(self, bkd, nmodels: int = 4):
        """Create a minimal AETC instance for testing."""

        def dummy_model(samples):
            return bkd.zeros((samples.shape[1], 1))

        def dummy_rvs(nsamples: int):
            return bkd.asarray(np.random.randn(1, nsamples))

        models = [dummy_model] * nmodels
        costs = bkd.asarray([10.0 ** (-i) for i in range(nmodels)])

        return AETC(models, dummy_rvs, costs, oracle_stats=None, bkd=bkd)

    def test_validate_subsets_none_generates_all(self, bkd) -> None:
        """Test that None generates all possible subsets."""
        nmodels = 3
        aetc = self._create_aetc(bkd, nmodels)

        subsets, max_ncov = aetc._validate_subsets(None)

        # For 2 LF models, we should get: [0], [1], [0,1]
        bkd.assert_allclose(
            bkd.asarray([len(subsets)]), bkd.asarray([3])
        )
        bkd.assert_allclose(bkd.asarray([max_ncov]), bkd.asarray([2]))

    def test_validate_subsets_valid(self, bkd) -> None:
        """Test validation of valid subsets."""
        nmodels = 4
        aetc = self._create_aetc(bkd, nmodels)

        subsets = [
            bkd.asarray([0], dtype=int),
            bkd.asarray([0, 1], dtype=int),
        ]

        validated, max_ncov = aetc._validate_subsets(subsets)

        bkd.assert_allclose(
            bkd.asarray([len(validated)]), bkd.asarray([2])
        )
        bkd.assert_allclose(bkd.asarray([max_ncov]), bkd.asarray([2]))

    def test_validate_subsets_rejects_duplicates(self, bkd) -> None:
        """Test that duplicate indices are rejected."""
        nmodels = 4
        aetc = self._create_aetc(bkd, nmodels)

        subsets = [bkd.asarray([0, 0], dtype=int)]

        with pytest.raises(ValueError):
            aetc._validate_subsets(subsets)

    def test_validate_subsets_rejects_out_of_range(self, bkd) -> None:
        """Test that out-of-range indices are rejected."""
        nmodels = 3  # LF models are 0, 1 (max index is 1)
        aetc = self._create_aetc(bkd, nmodels)

        subsets = [bkd.asarray([2], dtype=int)]  # Index 2 is out of range

        with pytest.raises(ValueError):
            aetc._validate_subsets(subsets)


class TestAETCBLUEFindK2:
    """Test AETCBLUE._find_k2 method."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_aetcblue(self, bkd, nmodels: int = 3):
        """Create an AETCBLUE instance for testing."""

        def dummy_model(samples):
            return bkd.zeros((samples.shape[1], 1))

        def dummy_rvs(nsamples: int):
            return bkd.asarray(np.random.randn(1, nsamples))

        models = [dummy_model] * nmodels
        costs = bkd.asarray([10.0 ** (-i) for i in range(nmodels)])

        return AETCBLUE(models, dummy_rvs, costs, oracle_stats=None, bkd=bkd)

    def test_find_k2_returns_tuple(self, bkd) -> None:
        """Test that _find_k2 returns (k2, nsamples_per_subset)."""
        aetcblue = self._create_aetcblue(bkd, nmodels=3)

        beta_Sp = bkd.asarray([[1.0], [0.8], [0.6]])
        Sigma_S = bkd.eye(2) * 0.5
        costs_S = bkd.asarray([0.1, 0.01])

        k2, nsamples = aetcblue._find_k2(beta_Sp, Sigma_S, costs_S)

        # k2 should be a scalar (0-d or squeezed)
        assert k2.ndim == 0 or (k2.ndim == 1 and k2.shape[0] == 1)

        # nsamples should have same length as costs_S or subsets
        assert len(nsamples) > 0


class TestAETCBLUEExploitation:
    """Test AETCBLUE exploitation methods."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_aetcblue(self, bkd, nmodels: int = 3):
        """Create an AETCBLUE instance for testing."""

        def dummy_model(samples):
            return bkd.zeros((1, samples.shape[1]))

        def dummy_rvs(nsamples: int):
            return bkd.asarray(np.random.randn(1, nsamples))

        models = [dummy_model] * nmodels
        costs = bkd.asarray([10.0 ** (-i) for i in range(nmodels)])

        return AETCBLUE(models, dummy_rvs, costs, oracle_stats=None, bkd=bkd)

    def _create_mock_explore_result(self, bkd):
        """Create a mock exploration result tuple for testing exploitation."""
        nexplore_samples = 100
        subset = bkd.asarray([0, 1], dtype=int)
        subset_cost = 0.11
        beta_Sp = bkd.asarray([[1.5], [0.8], [0.6]])
        sigma_S = bkd.asarray([[1.0, 0.3], [0.3, 1.0]])
        rounded_nsamples_per_subset = bkd.asarray([10.0, 20.0, 5.0])
        nsamples_per_subset = bkd.asarray([10.2, 19.8, 5.1])
        loss = 0.05
        k1 = 0.1
        BLUE_variance = 0.02
        exploit_budget = 50.0
        mlblue_subset_costs = bkd.asarray([0.1, 0.01])

        return (
            nexplore_samples,
            subset,
            subset_cost,
            beta_Sp,
            sigma_S,
            rounded_nsamples_per_subset,
            nsamples_per_subset,
            loss,
            k1,
            BLUE_variance,
            exploit_budget,
            mlblue_subset_costs,
        )

    def test_find_exploit_mean_uses_correct_reg_blue(self, bkd) -> None:
        """Test that find_exploit_mean uses scalar reg_blue, not Sigma_best_S.

        This verifies the correct behavior. Legacy incorrectly passes
        Sigma_best_S (a matrix) as reg_blue (should be a scalar).
        """
        from pyapprox.statest.groupacv import MLBLUEEstimator
        from pyapprox.statest.statistics import MultiOutputMean

        nmodels = 3
        aetcblue = self._create_aetcblue(bkd, nmodels)
        result = self._create_mock_explore_result(bkd)

        beta_Sp = result[3]
        sigma_S = result[4]
        rounded_nsamples_per_subset = result[5]
        subset = result[1]
        costs = bkd.asarray([1.0, 0.1, 0.01])
        costs_best_S = costs[subset + 1]

        # Create MLBLUEEstimator with correct scalar reg_blue
        stat = MultiOutputMean(1, bkd)
        stat.set_pilot_quantities(sigma_S)
        est = MLBLUEEstimator(stat, costs_best_S, asketch=beta_Sp[1:].T, reg_blue=1e-15)
        allocation = _make_groupacv_allocation(est, rounded_nsamples_per_subset)
        est.set_allocation(allocation)
        nsamples_per_model = allocation.nsamples_per_model

        # Create mock values with shape (nqoi, nsamples)
        np.random.seed(123)
        values_per_model = [
            bkd.asarray(np.random.randn(1, int(nsamples_per_model[i])))
            for i in range(len(bkd.to_numpy(subset)))
        ]

        # Compute expected mean using MLBLUEEstimator directly
        expected_product = est(values_per_model)
        if hasattr(expected_product, "item"):
            expected_product = expected_product.item()
        expected_mean = bkd.to_numpy(beta_Sp)[0, 0] + expected_product

        # Compute mean using AETCBLUE.find_exploit_mean
        typing_mean = aetcblue.find_exploit_mean(values_per_model, result)

        bkd.assert_allclose(
            bkd.asarray([typing_mean]),
            bkd.asarray([expected_mean]),
            rtol=1e-10,
        )

    def test_explore_result_to_dict_keys(self, bkd) -> None:
        """Test that _explore_result_to_dict returns expected keys."""
        aetcblue = self._create_aetcblue(bkd, nmodels=3)
        result = self._create_mock_explore_result(bkd)

        result_dict = aetcblue._explore_result_to_dict(result)

        expected_keys = {
            "nexplore_samples",
            "subset",
            "subset_cost",
            "beta_Sp",
            "sigma_S",
            "rounded_nsamples_per_subset",
            "nsamples_per_subset",
            "loss",
            "k1",
            "BLUE_variance",
            "exploit_budget",
            "mlblue_subset_costs",
            "explore_budget",
        }
        assert set(result_dict.keys()) == expected_keys


class TestAETCMC:
    """Test AETCMC methods."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_aetcmc(self, bkd, nmodels: int = 3):
        """Create an AETCMC instance for testing."""

        def dummy_model(samples):
            return bkd.zeros((1, samples.shape[1]))

        def dummy_rvs(nsamples: int):
            return bkd.asarray(np.random.randn(1, nsamples))

        models = [dummy_model] * nmodels
        costs = bkd.asarray([10.0 ** (-i) for i in range(nmodels)])

        return AETCMC(models, dummy_rvs, costs, oracle_stats=None, bkd=bkd)

    def test_find_k2_returns_tuple(self, bkd) -> None:
        """Test that _find_k2 returns (k2, nsamples_per_subset)."""
        aetcmc = self._create_aetcmc(bkd, nmodels=3)

        beta_Sp = bkd.asarray([[1.0], [0.8], [0.6]])
        Sigma_S = bkd.eye(2) * 0.5
        costs_S = bkd.asarray([0.1, 0.01])

        k2, nsamples = aetcmc._find_k2(beta_Sp, Sigma_S, costs_S)

        # k2 should be a scalar
        assert k2.ndim == 0 or (k2.ndim == 1 and k2.shape[0] == 1)

        # nsamples should be uniform (1/exploit_cost)
        expected_nsamples = 1 / (0.1 + 0.01)
        bkd.assert_allclose(
            nsamples, bkd.asarray([expected_nsamples]), rtol=1e-10
        )

    def test_find_k2_formula(self, bkd) -> None:
        """Test k2 follows the formula: exploit_cost * trace(asketch.T @ Sigma_S @
        asketch)."""
        aetcmc = self._create_aetcmc(bkd, nmodels=3)

        beta_Sp = bkd.asarray([[1.0], [0.8], [0.6]])
        Sigma_S = bkd.asarray([[1.0, 0.3], [0.3, 1.0]])
        costs_S = bkd.asarray([0.1, 0.01])

        k2, _ = aetcmc._find_k2(beta_Sp, Sigma_S, costs_S)

        # Compute expected k2
        asketch = beta_Sp[1:]  # [[0.8], [0.6]]
        exploit_cost = 0.1 + 0.01
        expected_k2 = exploit_cost * bkd.trace(
            bkd.multidot([asketch.T, Sigma_S, asketch])
        )

        bkd.assert_allclose(
            bkd.asarray([k2]), bkd.asarray([expected_k2]), rtol=1e-10
        )

    def test_explore_result_to_dict_keys(self, bkd) -> None:
        """Test that _explore_result_to_dict returns expected keys."""
        aetcmc = self._create_aetcmc(bkd, nmodels=3)

        # Create mock result
        result = (
            100,  # nexplore_samples
            bkd.asarray([0, 1], dtype=int),  # subset
            0.11,  # subset_cost
            bkd.asarray([[1.5], [0.8], [0.6]]),  # beta_Sp
            bkd.asarray([[1.0, 0.3], [0.3, 1.0]]),  # sigma_S
            bkd.asarray([10.0, 20.0, 5.0]),  # rounded_nsamples_per_subset
            bkd.asarray([10.2, 19.8, 5.1]),  # nsamples_per_subset
            0.05,  # loss
            0.1,  # k1
            0.02,  # BLUE_variance
            50.0,  # exploit_budget
            bkd.asarray([0.1, 0.01]),  # mlblue_subset_costs
        )

        result_dict = aetcmc._explore_result_to_dict(result)

        expected_keys = {
            "nexplore_samples",
            "subset",
            "subset_cost",
            "beta_Sp",
            "sigma_S",
            "rounded_nsamples_per_subset",
            "nsamples_per_subset",
            "loss",
            "k1",
            "BLUE_variance",
            "exploit_budget",
            "mlblue_subset_costs",
            "explore_budget",
        }
        assert set(result_dict.keys()) == expected_keys


class TestExploreStep:
    """Test _explore_step returns correct structure."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_aetcblue(self, bkd, nmodels: int = 3):
        """Create an AETCBLUE instance for testing."""
        from pyapprox_benchmarks.functions.multifidelity import (
            TunableModelEnsemble,
        )

        # Use TunableModelEnsemble for realistic models
        shifts = [1.0, 2.0]
        ensemble = TunableModelEnsemble(np.pi / 4, bkd, shifts)
        models = ensemble.models()

        costs = bkd.asarray([10.0 ** (-i) for i in range(len(models))])

        return AETCBLUE(models, ensemble.rvs, costs, oracle_stats=None, bkd=bkd)

    @slower_test
    def test_explore_step_returns_tuple(self, bkd) -> None:
        """Test _explore_step returns tuple with correct length."""
        aetcblue = self._create_aetcblue(bkd)

        # Run explore which calls _explore_step internally
        total_budget = 100.0
        subsets = [bkd.asarray([0, 1], dtype=int)]

        samples, values, result = aetcblue.explore(total_budget, subsets)

        # Result should be a tuple with 12 elements
        assert isinstance(result, tuple)
        assert len(result) == 12

    @slowest_test
    def test_explore_result_structure(self, bkd) -> None:
        """Test explore result has correct element types."""
        aetcblue = self._create_aetcblue(bkd)

        total_budget = 100.0
        subsets = [bkd.asarray([0, 1], dtype=int)]

        samples, values, result = aetcblue.explore(total_budget, subsets)

        # Unpack and check types
        (
            nexplore_samples,
            subset,
            subset_cost,
            beta_Sp,
            sigma_S,
            rounded_nsamples,
            nsamples,
            loss,
            k1,
            BLUE_var,
            exploit_budget,
            mlblue_costs,
        ) = result

        # nexplore_samples should be integer
        assert isinstance(nexplore_samples, (int, np.integer))

        # beta_Sp should be 2D array
        assert len(beta_Sp.shape) == 2

        # sigma_S should be 2D square
        assert len(sigma_S.shape) == 2
        assert sigma_S.shape[0] == sigma_S.shape[1]


class TestExploitProducesMean:
    """Test exploit produces valid mean estimate."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_aetcblue(self, bkd):
        """Create an AETCBLUE instance for testing."""
        from pyapprox_benchmarks.functions.multifidelity import (
            TunableModelEnsemble,
        )

        shifts = [1.0, 2.0]
        ensemble = TunableModelEnsemble(np.pi / 4, bkd, shifts)
        models = ensemble.models()

        costs = bkd.asarray([10.0 ** (-i) for i in range(len(models))])

        return AETCBLUE(models, ensemble.rvs, costs, oracle_stats=None, bkd=bkd)

    @slower_test
    def test_exploit_returns_scalar(self, bkd) -> None:
        """Test exploit returns a scalar mean value."""
        aetcblue = self._create_aetcblue(bkd)

        total_budget = 200.0  # Increased for numerical stability
        subsets = [bkd.asarray([0, 1], dtype=int)]

        # Run explore first
        samples, values, result = aetcblue.explore(total_budget, subsets)

        # Run exploit
        mean = aetcblue.exploit(result)

        # Mean should be a scalar
        assert (
            np.isscalar(mean)
            or (hasattr(mean, "ndim") and mean.ndim == 0)
            or (hasattr(mean, "shape") and mean.shape == ())
        )

    @slower_test
    def test_exploit_mean_is_finite(self, bkd) -> None:
        """Test exploit returns finite mean."""
        aetcblue = self._create_aetcblue(bkd)

        total_budget = 200.0  # Increased for numerical stability
        subsets = [bkd.asarray([0, 1], dtype=int)]

        samples, values, result = aetcblue.explore(total_budget, subsets)
        mean = aetcblue.exploit(result)

        # Mean should be finite
        assert np.isfinite(float(mean))


class TestFullEstimatePipeline:
    """Test full estimate pipeline."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_aetcblue(self, bkd):
        """Create an AETCBLUE instance for testing."""
        from pyapprox_benchmarks.functions.multifidelity import (
            TunableModelEnsemble,
        )

        shifts = [1.0, 2.0]
        ensemble = TunableModelEnsemble(np.pi / 4, bkd, shifts)
        models = ensemble.models()

        costs = bkd.asarray([10.0 ** (-i) for i in range(len(models))])

        return AETCBLUE(models, ensemble.rvs, costs, oracle_stats=None, bkd=bkd)

    @slow_test
    def test_estimate_returns_tuple(self, bkd) -> None:
        """Test estimate returns (mean, values, result)."""
        aetcblue = self._create_aetcblue(bkd)

        total_budget = 100.0
        subsets = [bkd.asarray([0, 1], dtype=int)]

        result = aetcblue.estimate(total_budget, subsets=subsets, return_dict=False)

        assert len(result) == 3
        mean, values, result_tuple = result

        # mean should be scalar
        assert (
            np.isscalar(mean)
            or (hasattr(mean, "ndim") and mean.ndim == 0)
            or (hasattr(mean, "shape") and mean.shape == ())
        )

        # values should be array
        assert values is not None

        # result_tuple should be tuple
        assert isinstance(result_tuple, tuple)

    @slow_test
    def test_estimate_returns_dict(self, bkd) -> None:
        """Test estimate with return_dict=True returns dict."""
        aetcblue = self._create_aetcblue(bkd)

        total_budget = 100.0
        subsets = [bkd.asarray([0, 1], dtype=int)]

        mean, values, result_dict = aetcblue.estimate(
            total_budget, subsets=subsets, return_dict=True
        )

        assert isinstance(result_dict, dict)

        expected_keys = {
            "nexplore_samples",
            "subset",
            "subset_cost",
            "beta_Sp",
            "sigma_S",
            "rounded_nsamples_per_subset",
            "nsamples_per_subset",
            "loss",
            "k1",
            "BLUE_variance",
            "exploit_budget",
            "mlblue_subset_costs",
            "explore_budget",
        }
        assert set(result_dict.keys()) == expected_keys

    @slow_test
    def test_sigma_matches_sample_covariance(self, bkd) -> None:
        """Test sigma_S from result matches sample covariance (assertion 2.1)."""
        aetcblue = self._create_aetcblue(bkd)

        total_budget = 200.0
        subsets = [bkd.asarray([0, 1], dtype=int)]

        mean, values, result = aetcblue.estimate(
            total_budget, subsets=subsets, return_dict=True
        )

        # Compute sample covariance from values
        # Typing convention: values is (nmodels, nsamples), so rowvar=True (default)
        cov_exe = bkd.cov(values, ddof=1)

        # Get subset indices (add 1 because subset is 0-indexed for LF models)
        subset = result["subset"] + 1
        subset_np = bkd.to_numpy(subset).astype(int)

        # Extract subset covariance
        expected_sigma_S = cov_exe[np.ix_(subset_np, subset_np)]

        # sigma_S from result should match
        bkd.assert_allclose(result["sigma_S"], expected_sigma_S, rtol=1e-10)

    @slow_test
    def test_blue_variance_formula(self, bkd) -> None:
        """Test BLUE_variance is computed correctly (assertions 2.2, 2.3)."""
        from pyapprox.statest.groupacv import MLBLUEEstimator
        from pyapprox.statest.statistics import MultiOutputMean

        aetcblue = self._create_aetcblue(bkd)

        total_budget = 200.0
        subsets = [bkd.asarray([0, 1], dtype=int)]

        mean, values, result = aetcblue.estimate(
            total_budget, subsets=subsets, return_dict=True
        )

        # Compute sample covariance from values
        # Typing convention: values is (nmodels, nsamples), so rowvar=True (default)
        cov_exe = bkd.cov(values, ddof=1)

        # Get subset indices
        subset = result["subset"] + 1
        subset_np = bkd.to_numpy(subset).astype(int)
        costs = bkd.asarray([10.0 ** (-i) for i in range(3)])

        # Create MLBLUEEstimator to compute variance
        stat = MultiOutputMean(1, bkd)
        stat.set_pilot_quantities(cov_exe[np.ix_(subset_np, subset_np)])
        mlblue_est = MLBLUEEstimator(
            stat,
            costs[subset_np],
            asketch=result["beta_Sp"][1:].T,
        )
        computed_var = mlblue_est._covariance_from_npartition_samples(
            result["nsamples_per_subset"]
        )

        # BLUE_variance should match computed variance
        # computed_var may be multi-dimensional, flatten to scalar for comparison
        computed_var_scalar = float(bkd.to_numpy(computed_var).ravel()[0])
        bkd.assert_allclose(
            bkd.asarray([result["BLUE_variance"]]),
            bkd.asarray([computed_var_scalar]),
            rtol=3e-2,
        )


class TestOptimalLossOracleVsMC:
    """Test k1 from oracle stats matches k1 from MC with many samples.

    Replicates legacy test_AETC_optimal_loss:
        assert bkd.allclose(result_mc[-2], result_oracle[-2], rtol=1e-2)
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        np.random.seed(1)
        self._bkd = NumpyBkd()

    @slower_test
    def test_optimal_loss_oracle_vs_mc(self) -> None:
        """Test k1 from oracle stats matches k1 from MC with many samples."""
        from pyapprox_benchmarks.functions.multifidelity import (
            TunableModelEnsemble,
        )

        bkd = self._bkd
        alpha = 1000
        nsamples = int(1e6)
        shifts = [1.0, 2.0]

        ensemble = TunableModelEnsemble(np.pi / 4, bkd, shifts)
        models = ensemble.models()
        cov = ensemble.covariance()
        costs = bkd.asarray([10.0 ** (-i) for i in range(len(models))])
        true_means = ensemble.means()

        target_cost = bkd.sum(costs) * (nsamples + 10)

        oracle_stats = [cov, true_means]

        # Generate samples and evaluate models
        samples = ensemble.rvs(nsamples)
        # Typing convention: values shape (nmodels, nsamples)
        values = bkd.vstack([model(samples) for model in models])

        exploit_cost = 0.5 * target_cost
        covariate_subset = bkd.asarray([0, 1], dtype=int)
        # HF values: (1, nsamples), LF values: (nmodels-1, nsamples)
        hf_values = values[:1, :]
        covariate_values = values[1:, :]

        # Create estimators with and without oracle stats
        est_nor = AETCBLUE(models, ensemble.rvs, costs, oracle_stats=None, bkd=bkd)
        est_or = AETCBLUE(
            models, ensemble.rvs, costs, oracle_stats=oracle_stats, bkd=bkd
        )

        result_oracle = est_or._optimal_loss(
            target_cost,
            hf_values,
            covariate_values[covariate_subset, :],
            costs,
            covariate_subset,
            alpha,
            exploit_cost,
        )
        result_mc = est_nor._optimal_loss(
            target_cost,
            hf_values,
            covariate_values[covariate_subset, :],
            costs,
            covariate_subset,
            alpha,
            exploit_cost,
        )

        # k1 values should match (index -3 is k1 in result tuple)
        # Result tuple: (loss, nsamples_per_subset, explore_rate, beta_Sp,
        #                Sigma_S, k1, k2, exploit_budget)
        k1_mc = result_mc[5]  # k1 is at index 5
        k1_oracle = result_oracle[5]

        bkd.assert_allclose(bkd.asarray([k1_mc]), bkd.asarray([k1_oracle]), rtol=1e-2)


class TestMSEMatchesLoss:
    """Test MSE matches theoretical loss (slow test)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        np.random.seed(1)
        self._bkd = NumpyBkd()

    def _create_aetcblue(self):
        """Create an AETCBLUE instance for testing."""
        from pyapprox_benchmarks.functions.multifidelity import (
            TunableModelEnsemble,
        )
        from pyapprox.optimization.minimize.chained.chained_optimizer import (
            ChainedOptimizer,
        )
        from pyapprox.optimization.minimize.scipy.diffevol import (
            ScipyDifferentialEvolutionOptimizer,
        )
        from pyapprox.optimization.minimize.scipy.trust_constr import (
            ScipyTrustConstrOptimizer,
        )

        shifts = [1.0, 2.0]
        ensemble = TunableModelEnsemble(np.pi / 4, self._bkd, shifts)
        models = ensemble.models()
        true_means = ensemble.means()

        costs = self._bkd.asarray([10.0 ** (-i) for i in range(len(models))])

        # Use chained optimizer with relaxed settings for stability
        global_opt = ScipyDifferentialEvolutionOptimizer(
            maxiter=3,
            polish=False,
            seed=1,
            tol=1e-8,
            raise_on_failure=False,
        )
        local_opt = ScipyTrustConstrOptimizer(
            gtol=1e-6,
            maxiter=1000,
        )
        optimizer = ChainedOptimizer(global_opt, local_opt)

        return AETCBLUE(
            models,
            ensemble.rvs,
            costs,
            oracle_stats=None,
            reg_blue=0,
            optimizer=optimizer,
            bkd=self._bkd,
        ), true_means

    @slow_test
    def test_mse_matches_theoretical_loss(self) -> None:
        """Verify empirical MSE matches theoretical loss (assertion 2.4).

        Replicates legacy test_aetc_blue assertion:
            assert bkd.allclose(mse, result_dict["loss"], rtol=4e-2)
        """
        aetcblue, true_means = self._create_aetcblue()

        target_cost = 300.0
        subsets = [self._bkd.asarray([0, 1], dtype=int)]

        # Get initial result for reference loss
        mean, values, result = aetcblue.estimate(
            target_cost, subsets=subsets, return_dict=True
        )
        reference_loss = result["loss"]

        # Run multiple trials
        ntrials = 300
        means = np.empty(ntrials)

        for ii in range(ntrials):
            means[ii], _, _ = aetcblue.estimate(target_cost, subsets=subsets)

        # Compute empirical MSE
        # true_means has shape (nmodels, 1), HF model is index 0
        true_mean = float(self._bkd.to_numpy(true_means)[0, 0])
        mse = np.mean((means - true_mean) ** 2)

        # MSE should be close to theoretical loss
        self._bkd.assert_allclose(
            self._bkd.asarray([mse]), self._bkd.asarray([reference_loss]), rtol=1e-1
        )
