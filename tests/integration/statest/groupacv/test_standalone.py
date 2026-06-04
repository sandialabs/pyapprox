"""Standalone tests for GroupACV and MLBLUE estimators.

These tests do not depend on legacy code and will remain after
the legacy module is removed. All expected values are derived
from mathematical definitions.
"""

import numpy as np
import pytest
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.optimization.minimize.chained.chained_optimizer import (
    ChainedOptimizer,
)
from pyapprox.optimization.minimize.scipy.diffevol import (
    ScipyDifferentialEvolutionOptimizer,
)
from pyapprox.optimization.minimize.scipy.slsqp import ScipySLSQPOptimizer
from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.statest.acv import GMFEstimator, GRDEstimator, MFMCEstimator
from pyapprox.statest.acv.variants import _allocate_samples_mfmc
from pyapprox.statest.groupacv import (
    FittedGroupACVEstimator,
    GroupACVCostConstraint,
    GroupACVEstimatorIS,
    GroupACVEstimatorNested,
    GroupACVLogDetObjective,
    GroupACVTraceObjective,
    MLBLUEEstimator,
    MLBLUEObjective,
    MLBLUESPDAllocationOptimizer,
    _get_allocation_matrix_is,
    _get_allocation_matrix_nested,
    _nest_subsets,
    get_model_subsets,
)
from pyapprox.statest.groupacv.allocation import (
    GroupACVAllocationOptimizer,
    GroupACVAllocationResult,
)
from pyapprox.statest.groupacv.utils import _grouped_acv_sigma_block
from pyapprox.statest.groupacv.variable_space import (
    AllocationProblemConfig,
    _NormalizedConstraint,
    _RescaledConstraint,
    _RescaledObjective,
)
from pyapprox.statest.statistics import (
    MultiOutputMean,
    MultiOutputMeanAndVariance,
    MultiOutputVariance,
)
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.optional_deps import package_available
from pyapprox_benchmarks.statest import (
    MultiOutputEnsembleBenchmark,
    PolynomialEnsembleBenchmark,
)

from tests._helpers.acv_utils import allocate_with_allocator
from tests._helpers.markers import slow_test, slower_test


def _make_groupacv_allocation(est, npartition_samples):
    """Helper to create GroupACVAllocationResult from npartition_samples.

    Rounds the samples and computes derived quantities.
    """
    bkd = est.bkd()
    rounded = bkd.asarray(
        bkd.floor(npartition_samples + 1e-4), dtype=bkd.int64_dtype()
    )
    nsamples_per_model = bkd.asarray(
        est._compute_nsamples_per_model(
            bkd.asarray(rounded, dtype=bkd.double_dtype())
        ),
        dtype=bkd.int64_dtype(),
    )
    actual_cost = float(est._estimator_cost(
        bkd.asarray(rounded, dtype=bkd.double_dtype())
    ))
    return GroupACVAllocationResult(
        npartition_samples=rounded,
        nsamples_per_model=nsamples_per_model,
        actual_cost=actual_cost,
        objective_value=bkd.array([0.0]),  # Placeholder
        success=True,
        message="",
    )


class TestGroupACVUtils:
    """Tests for GroupACV utility functions."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(1)

    def test_get_model_subsets(self, bkd):
        """Test model subset generation."""
        nmodels = 3
        subsets = get_model_subsets(nmodels, bkd)

        # Should have 2^nmodels - 1 = 7 subsets
        bkd.assert_allclose(
            bkd.asarray([len(subsets)]),
            bkd.asarray([7]),
        )

        # Check first few subsets
        bkd.assert_allclose(subsets[0], bkd.asarray([0], dtype=int))
        bkd.assert_allclose(subsets[1], bkd.asarray([1], dtype=int))
        bkd.assert_allclose(subsets[2], bkd.asarray([2], dtype=int))

    def test_allocation_matrix_is(self, bkd):
        """Test independent sampling allocation matrix."""
        nmodels = 3
        subsets = get_model_subsets(nmodels, bkd)
        allocation_mat = _get_allocation_matrix_is(subsets, bkd)

        # IS allocation should be identity
        expected = bkd.eye(len(subsets))
        bkd.assert_allclose(allocation_mat, expected)

    def test_allocation_matrix_nested(self, bkd):
        """Test nested sampling allocation matrix."""
        nmodels = 3
        # Remove subset 0 for nested
        subsets = get_model_subsets(nmodels, bkd)[1:]
        subsets = _nest_subsets(subsets, nmodels, bkd)[0]

        # Re-sort as in legacy code
        idx = sorted(
            list(range(len(subsets))),
            key=lambda ii: (len(subsets[ii]), tuple(nmodels - subsets[ii])),
            reverse=True,
        )
        subsets = [subsets[ii] for ii in idx]

        nsubsets = len(subsets)
        allocation_mat = _get_allocation_matrix_nested(subsets, bkd)

        # Nested allocation should be lower triangular of ones
        expected = bkd.asarray(np.tril(np.ones((nsubsets, nsubsets))))
        bkd.assert_allclose(allocation_mat, expected)


class TestGroupACVEstimator:
    """Tests for GroupACVEstimator."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(1)

    def _create_estimator(self, bkd, nmodels, estimator_cls=GroupACVEstimatorIS):
        """Helper to create a test estimator."""
        np.random.seed(1)
        cov = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        costs = bkd.arange(nmodels, 0, -1, dtype=bkd.double_dtype())

        stat = MultiOutputMean(1, bkd)
        stat.set_pilot_quantities(cov)
        est = estimator_cls(stat, costs)
        return est

    def test_nsamples_per_model_is(self, bkd):
        """Test sample count computation for IS estimation."""
        nmodels = 3
        est = self._create_estimator(bkd, nmodels, estimator_cls=GroupACVEstimatorIS)

        npartition_samples = bkd.arange(
            2.0, 2 + est.nsubsets(), dtype=bkd.double_dtype()
        )

        # Expected values derived mathematically from subset structure
        expected_nsamples = bkd.asarray([21.0, 23.0, 25.0])
        bkd.assert_allclose(
            est._compute_nsamples_per_model(npartition_samples),
            expected_nsamples,
        )

        # Check total cost: sum(nsamples_i * cost_i) where costs = [3, 2, 1]
        expected_cost = bkd.asarray([21 * 3 + 23 * 2 + 25 * 1.0])
        bkd.assert_allclose(
            bkd.asarray([est._estimator_cost(npartition_samples)]),
            expected_cost,
        )

        # For IS, intersection samples matrix is diagonal
        bkd.assert_allclose(
            est._nintersect_samples(npartition_samples),
            bkd.diag(npartition_samples),
        )

    def test_nsamples_per_model_nested(self, bkd):
        """Test sample count computation for nested estimation."""
        nmodels = 3
        np.random.seed(1)
        est = self._create_estimator(
            bkd, nmodels, estimator_cls=GroupACVEstimatorNested
        )

        npartition_samples = bkd.arange(
            2.0, 2.0 + est.nsubsets(), dtype=bkd.double_dtype()
        )

        # Expected values from nested structure
        expected_nsamples = bkd.asarray([9, 20, 27.0])
        bkd.assert_allclose(
            est._compute_nsamples_per_model(npartition_samples),
            expected_nsamples,
        )

        # Check total cost
        expected_cost = bkd.asarray([9 * 3 + 20 * 2 + 27 * 1.0])
        bkd.assert_allclose(
            bkd.asarray([est._estimator_cost(npartition_samples)]),
            expected_cost,
        )

        # Check intersection samples matrix (nested structure)
        expected_intersect = bkd.asarray(
            [
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                [2.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                [2.0, 5.0, 9.0, 9.0, 9.0, 9.0],
                [2.0, 5.0, 9.0, 14.0, 14.0, 14.0],
                [2.0, 5.0, 9.0, 14.0, 20.0, 20.0],
                [2.0, 5.0, 9.0, 14.0, 20.0, 27.0],
            ]
        )
        bkd.assert_allclose(
            est._nintersect_samples(npartition_samples),
            expected_intersect,
        )

    def test_basic_accessors(self, bkd):
        """Test basic accessor methods."""
        nmodels = 3
        est = self._create_estimator(bkd, nmodels)

        bkd.assert_allclose(
            bkd.asarray([est.nmodels()]),
            bkd.asarray([nmodels]),
        )
        # For IS with nmodels=3, we get 2^3 - 1 = 7 subsets
        bkd.assert_allclose(
            bkd.asarray([est.nsubsets()]),
            bkd.asarray([7]),
        )
        bkd.assert_allclose(
            bkd.asarray([est.npartitions()]),
            bkd.asarray([7]),
        )

    def _check_separate_samples(self, bkd, est):
        """Test sample separation logic."""
        NN = 2
        npartition_samples = bkd.full(
            (est.nsubsets(),), float(NN), dtype=bkd.double_dtype()
        )
        allocation = _make_groupacv_allocation(est, npartition_samples)
        fitted = FittedGroupACVEstimator(est, allocation)

        samples_per_model = fitted.generate_samples_per_model(
            lambda n: bkd.arange(int(n), dtype=bkd.double_dtype())[None, :]
        )
        for ii in range(est.nmodels()):
            bkd.assert_allclose(
                bkd.asarray([samples_per_model[ii].shape[1]]),
                bkd.asarray([int(allocation.nsamples_per_model[ii])]),
            )

        # values shape is (nqoi, nsamples) - same as samples but with nqoi rows
        values_per_model = [(ii + 1) * s for ii, s in enumerate(samples_per_model)]
        values_per_subset = fitted._separate_values_per_model(values_per_model)

        test_samples = bkd.arange(
            int(fitted.npartition_samples().sum()),
            dtype=bkd.double_dtype(),
        )[None, :]
        # test_values shape is (nqoi, nsamples)
        [(ii + 1) * test_samples for ii in range(est.nmodels())]

        for ii in range(est.nsubsets()):
            active_partitions = bkd.where(est._allocation_mat[ii] == 1)[0]
            (
                bkd.arange(test_samples.shape[1], dtype=bkd.int64_dtype())
                .reshape(est.npartitions(), NN)[active_partitions]
                .flatten()
            )
            # expected_shape is (nqoi*nmodels_in_subset, nsamples_in_subset)
            expected_shape = (
                len(est._subsets[ii]),
                int(est._nintersect_samples(npartition_samples)[ii][ii]),
            )
            bkd.assert_allclose(
                bkd.asarray(list(values_per_subset[ii].shape)),
                bkd.asarray(list(expected_shape)),
            )

    def test_separate_samples_is(self, bkd):
        """Test sample separation for IS estimation."""
        est = self._create_estimator(bkd, 3, estimator_cls=GroupACVEstimatorIS)
        self._check_separate_samples(bkd, est)

    def test_separate_samples_nested(self, bkd):
        """Test sample separation for nested estimation."""
        est = self._create_estimator(bkd, 3, estimator_cls=GroupACVEstimatorNested)
        self._check_separate_samples(bkd, est)


class TestGroupACVObjectiveDerivativesTorchOnly:
    """Test derivative correctness for GroupACV objectives using DerivativeChecker.

    These tests only run with Torch backend because the base objectives use
    bkd.jacobian() which requires autograd (only available in Torch).
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        import torch

        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def _create_estimator(self, nmodels, nqoi=1):
        """Helper to create a test estimator."""
        np.random.seed(1)
        cov_size = nmodels * nqoi
        cov = self._bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
        cov = cov.T @ cov
        costs = self._bkd.arange(nmodels, 0, -1, dtype=self._bkd.double_dtype())

        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(cov)
        est = GroupACVEstimatorIS(stat, costs)
        return est

    @pytest.mark.parametrize(
        "nmodels,nqoi",
        [(2, 1), (3, 1), (2, 2)],
    )
    def test_trace_objective_jacobian(self, nmodels: int, nqoi: int):
        """Test trace objective Jacobian with DerivativeChecker."""
        np.random.seed(1)
        est = self._create_estimator(nmodels, nqoi)
        target_cost = 100

        obj = GroupACVTraceObjective(self._bkd)
        obj.set_estimator(est)

        iterate = est._init_guess(target_cost)
        checker = DerivativeChecker(obj)
        errors = checker.check_derivatives(iterate, verbosity=0)

        # Check Jacobian accuracy
        assert checker.error_ratio(errors[0]) <= 2e-6

    @pytest.mark.parametrize(
        "nmodels,nqoi",
        [(2, 1), (3, 1), (2, 2)],
    )
    def test_logdet_objective_jacobian(self, nmodels: int, nqoi: int):
        """Test log-determinant objective Jacobian with DerivativeChecker."""
        np.random.seed(1)
        est = self._create_estimator(nmodels, nqoi)
        target_cost = 100

        obj = GroupACVLogDetObjective(self._bkd)
        obj.set_estimator(est)

        iterate = est._init_guess(target_cost)
        checker = DerivativeChecker(obj)
        errors = checker.check_derivatives(iterate, verbosity=0)

        # Check Jacobian accuracy
        assert checker.error_ratio(errors[0]) <= 2e-6


class TestGroupACVConstraintDerivatives:
    """Test derivative correctness for GroupACV constraints."""

    def _create_estimator(self, bkd, nmodels):
        """Helper to create a test estimator."""
        np.random.seed(1)
        cov = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        costs = bkd.arange(nmodels, 0, -1, dtype=bkd.double_dtype())

        stat = MultiOutputMean(1, bkd)
        stat.set_pilot_quantities(cov)
        est = GroupACVEstimatorIS(stat, costs)
        return est

    @pytest.mark.parametrize(
        "nmodels",
        [2, 3],
    )
    def test_constraint_jacobian(self, bkd, nmodels: int):
        """Test constraint Jacobian with DerivativeChecker."""
        np.random.seed(1)
        est = self._create_estimator(bkd, nmodels)
        target_cost = 100
        min_nhf_samples = 1

        constraint = GroupACVCostConstraint(bkd)
        constraint.set_estimator(est)
        constraint.set_budget(target_cost, min_nhf_samples)

        iterate = est._init_guess(target_cost)
        checker = DerivativeChecker(constraint)
        # Pass weights for multi-QoI constraint (nqoi=2) with whvp
        weights = bkd.asarray([[0.6, 0.4]])  # Shape (1, nqoi)
        errors = checker.check_derivatives(iterate, weights=weights, verbosity=0)

        # Constraints are linear, so Jacobian should be exact
        assert float(errors[0][0]) < 1e-12

    @pytest.mark.parametrize(
        "nmodels",
        [2, 3],
    )
    def test_constraint_hessian_zero(self, bkd, nmodels: int):
        """Test that constraint Hessian is zero (linear constraints)."""
        np.random.seed(1)
        est = self._create_estimator(bkd, nmodels)

        constraint = GroupACVCostConstraint(bkd)
        constraint.set_estimator(est)
        constraint.set_budget(target_cost=100.0, min_nhf_samples=1)

        npartition_samples = bkd.full(
            (est.npartitions(), 1), 1.0, dtype=bkd.double_dtype()
        )
        hess = constraint.hessian(npartition_samples)

        # Hessian should be all zeros since constraints are linear
        bkd.assert_allclose(
            hess,
            bkd.zeros((2, est.npartitions(), est.npartitions())),
        )


class TestMLBLUEEstimator:
    """Tests for MLBLUEEstimator."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(1)

    def test_mlblue_uses_is(self, bkd):
        """Test that MLBLUE uses independent sampling."""
        nmodels = 3
        cov = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        costs = bkd.arange(nmodels, 0, -1, dtype=bkd.double_dtype())

        stat = MultiOutputMean(1, bkd)
        stat.set_pilot_quantities(cov)
        est = MLBLUEEstimator(stat, costs)

        # MLBLUE should use IS, so allocation should be identity-like
        allocation_mat = est._allocation_mat
        bkd.assert_allclose(
            allocation_mat,
            bkd.eye(est.nsubsets()),
        )

    def test_mlblue_psi_blocks(self, bkd):
        """Test that MLBLUE precomputes psi blocks."""
        nmodels = 2
        cov = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        costs = bkd.arange(nmodels, 0, -1, dtype=bkd.double_dtype())

        stat = MultiOutputMean(1, bkd)
        stat.set_pilot_quantities(cov)
        est = MLBLUEEstimator(stat, costs)

        # Should have precomputed psi blocks
        bkd.assert_allclose(
            bkd.asarray([len(est._psi_blocks)]),
            bkd.asarray([est.nsubsets()]),
        )

    def test_mlblue_inherits_groupacv(self, bkd):
        """Test that MLBLUEEstimator inherits from GroupACVEstimatorIS."""
        nmodels = 2
        cov = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        costs = bkd.arange(nmodels, 0, -1, dtype=bkd.double_dtype())

        stat = MultiOutputMean(1, bkd)
        stat.set_pilot_quantities(cov)
        est = MLBLUEEstimator(stat, costs)

        # Should be instance of GroupACVEstimatorIS
        assert isinstance(est, GroupACVEstimatorIS)


class TestMLBLUEObjectiveDerivatives:
    """Test MLBLUE-specific objective derivatives."""

    def _create_mlblue_estimator(self, bkd, nmodels, nqoi=1):
        """Helper to create a test MLBLUE estimator."""
        np.random.seed(1)
        cov_size = nmodels * nqoi
        cov = bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
        cov = cov.T @ cov
        costs = bkd.arange(nmodels, 0, -1, dtype=bkd.double_dtype())

        stat = MultiOutputMean(nqoi, bkd)
        stat.set_pilot_quantities(cov)
        est = MLBLUEEstimator(stat, costs)
        return est

    @pytest.mark.parametrize(
        "nmodels,nqoi",
        [(2, 1), (3, 1), (2, 2)],
    )
    def test_mlblue_objective_jacobian(self, bkd, nmodels: int, nqoi: int):
        """Test MLBLUE objective analytical Jacobian."""
        np.random.seed(1)
        est = self._create_mlblue_estimator(bkd, nmodels, nqoi)
        target_cost = 100

        obj = MLBLUEObjective(bkd)
        obj.set_estimator(est)

        iterate = est._init_guess(target_cost)
        checker = DerivativeChecker(obj)
        errors = checker.check_derivatives(iterate, verbosity=0)

        # Check Jacobian accuracy
        assert checker.error_ratio(errors[0]) <= 1e-6

    def test_mlblue_objective_inherits_trace(self, bkd):
        """Test that MLBLUEObjective inherits from GroupACVTraceObjective."""
        obj = MLBLUEObjective(bkd)
        assert isinstance(obj, GroupACVTraceObjective)


class TestRestrictionMatrices:
    """Tests for restriction matrix functionality."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(1)

    def test_restriction_matrices_nqoi_1(self, bkd):
        """Test restriction matrices with nqoi=1."""
        qoi_idx = [0]
        costs = bkd.array([1, 0.5, 0.25], dtype=bkd.double_dtype())
        stat = MultiOutputMean(len(qoi_idx), bkd)
        subsets = [[0, 1, 2], [1], [2]]
        subsets = [bkd.array(s, dtype=bkd.int64_dtype()) for s in subsets]
        est = GroupACVEstimatorNested(stat, costs, model_subsets=subsets)

        # Vector containing model ids
        Lvec = bkd.arange(3, dtype=bkd.double_dtype())[:, None]

        # Make sure each restriction matrix recovers correct subset model ids
        cnt = 0
        for ii in range(len(subsets)):
            recovered = (est._R[:, cnt : cnt + len(subsets[ii])].T @ Lvec)[:, 0]
            expected = bkd.asarray(subsets[ii], dtype=bkd.double_dtype())
            bkd.assert_allclose(recovered, expected)
            cnt += len(subsets[ii])

    def test_restriction_matrices_nqoi_2(self, bkd):
        """Test restriction matrices with nqoi=2."""
        qoi_idx = [0, 1]
        costs = bkd.array([1, 0.5, 0.25], dtype=bkd.double_dtype())
        stat = MultiOutputMean(len(qoi_idx), bkd)
        subsets = [[0, 1, 2], [1], [2]]
        subsets = [bkd.array(s, dtype=bkd.int64_dtype()) for s in subsets]
        est = GroupACVEstimatorNested(stat, costs, model_subsets=subsets)

        # Vector containing flattened model qoi ids
        Lvec = bkd.arange(3 * len(qoi_idx), dtype=bkd.double_dtype())[:, None]

        # Check restriction matrix recovers all correct qoi of all subset model ids
        bkd.assert_allclose(
            (est._R[:, :6].T @ Lvec)[:, 0],
            bkd.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        )
        bkd.assert_allclose((est._R[:, 6:8].T @ Lvec)[:, 0], bkd.array([2.0, 3.0]))
        bkd.assert_allclose((est._R[:, 8:10].T @ Lvec)[:, 0], bkd.array([4.0, 5.0]))


# CVXPY-dependent tests
HAS_CVXPY = package_available("cvxpy")


@pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")
class TestMLBLUESPDAllocationOptimizer:
    """Tests for MLBLUESPDAllocationOptimizer (requires cvxpy)."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(1)

    def _create_mlblue_estimator(self, bkd, nmodels, nqoi=1):
        """Helper to create a test MLBLUE estimator."""
        np.random.seed(1)
        cov_size = nmodels * nqoi
        cov = bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
        cov = cov.T @ cov
        # Normalize to correlation matrix for better conditioning
        d = bkd.sqrt(bkd.diag(cov))
        cov = cov / bkd.outer(d, d)
        costs = bkd.array(
            [1.0 / (10**i) for i in range(nmodels)],
            dtype=bkd.double_dtype(),
        )

        stat = MultiOutputMean(nqoi, bkd)
        stat.set_pilot_quantities(cov)
        est = MLBLUEEstimator(stat, costs, reg_blue=0)
        return est

    @pytest.mark.parametrize(
        "nmodels,min_nhf_samples",
        [(2, 1), (3, 1), (4, 1), (3, 10)],
    )
    def test_spd_allocator_solves(self, bkd, nmodels: int, min_nhf_samples: int):
        """Test that SPD allocator finds a solution."""
        np.random.seed(1)
        est = self._create_mlblue_estimator(bkd, nmodels, nqoi=1)
        target_cost = 100

        allocator = MLBLUESPDAllocationOptimizer(est)
        result = allocator.optimize(target_cost, min_nhf_samples)

        # Check that optimization succeeded
        assert result.success

        # Check that result has correct shape
        bkd.assert_allclose(
            bkd.asarray([result.npartition_samples.shape[0]]),
            bkd.asarray([est.nsubsets()]),
        )

        # Check that all sample counts are non-negative
        assert float(bkd.min(result.npartition_samples)) >= -1e-10

    @pytest.mark.parametrize(
        "nmodels",
        [2, 3],
    )
    def test_spd_respects_budget(self, bkd, nmodels: int):
        """Test that SPD allocator respects budget constraint."""
        np.random.seed(1)
        est = self._create_mlblue_estimator(bkd, nmodels, nqoi=1)
        target_cost = 100
        min_nhf_samples = 1

        allocator = MLBLUESPDAllocationOptimizer(est)
        result = allocator.optimize(target_cost, min_nhf_samples)

        # Compute actual cost
        nps_float = bkd.asarray(
            result.npartition_samples, dtype=bkd.double_dtype()
        )
        actual_cost = est._estimator_cost(nps_float)

        # Cost should be <= target (with small tolerance)
        assert float(bkd.to_numpy(actual_cost)) <= target_cost + 1e-6

    def test_spd_raises_for_multioutput(self, bkd):
        """Test that SPD allocator raises error for multi-output."""
        est = self._create_mlblue_estimator(bkd, 2, nqoi=2)
        allocator = MLBLUESPDAllocationOptimizer(est)
        with pytest.raises(RuntimeError):
            allocator.optimize(100, 1)


class TestPilotSampleInsertionTorchOnly:
    """Tests for pilot sample insertion and removal.

    Torch-only test since optimization requires bkd.jacobian.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        import torch

        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def _covariance_to_correlation(self, cov):
        """Convert covariance matrix to correlation matrix."""
        d = self._bkd.sqrt(self._bkd.diag(cov))
        return cov / self._bkd.outer(d, d)

    def _generate_correlated_values(self, chol_factor, means, samples):
        """Generate correlated values from samples."""
        return (chol_factor @ samples + means[:, None]).T

    def _create_mlblue_estimator(self, nmodels: int):
        """Create an MLBLUE estimator for testing."""
        cov = self._bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        cov = self._covariance_to_correlation(cov)
        costs = self._bkd.flip(self._bkd.logspace(-nmodels + 1, 0, nmodels))
        stat = MultiOutputMean(1, self._bkd)
        stat.set_pilot_quantities(cov)
        return MLBLUEEstimator(stat, costs, reg_blue=1e-10)

    def _create_optimizer(self):
        """Create optimizer with sufficient iterations for test."""


        global_opt = ScipyDifferentialEvolutionOptimizer(
            maxiter=3,
            polish=False,
            seed=1,
            tol=1e-8,
            raise_on_failure=False,
        )
        local_opt = ScipyTrustConstrOptimizer(
            gtol=1e-6,
            maxiter=2000,
        )
        return ChainedOptimizer(global_opt, local_opt)

    @pytest.mark.parametrize(
        "nmodels,min_nhf_samples",
        [(2, 11), (3, 11), (4, 11)],
    )
    @slow_test
    def test_insert_pilot_samples(self, nmodels: int, min_nhf_samples: int):
        """Test that pilot values can be correctly inserted."""


        ntrials = 5  # Match legacy test
        npilot_samples = 8

        for seed in range(ntrials):
            np.random.seed(seed)
            est = self._create_mlblue_estimator(nmodels)

            # Allocate samples using new API
            target_cost = 100.0
            iterate = est._init_guess(target_cost)
            allocator = GroupACVAllocationOptimizer(
                est, optimizer=self._create_optimizer()
            )
            result = allocator.optimize(
                target_cost, min_nhf_samples, init_guess=iterate
            )
            fitted = FittedGroupACVEstimator(est, result)

            # Get covariance and create value generator
            cov = est._stat._cov
            chol_factor = self._bkd.cholesky(cov)
            exact_means = self._bkd.arange(nmodels, dtype=self._bkd.double_dtype())

            # Sample generator with nmodels variables (required for correlation)
            def rvs(n: int):
                return self._bkd.array(np.random.randn(nmodels, int(n)))

            # Generate full samples
            np.random.seed(seed)
            samples_per_model = fitted.generate_samples_per_model(rvs)

            # Remove pilot samples
            _, pilot_samples = fitted._remove_pilot_samples(
                npilot_samples, samples_per_model
            )

            # Generate pilot values - shape (nqoi, nsamples) = (1, npilot)
            pilot_values = [
                self._generate_correlated_values(
                    chol_factor, exact_means, pilot_samples
                )[:, ii : ii + 1].T
                for ii in range(nmodels)
            ]

            # Generate samples without pilot
            np.random.seed(seed)
            samples_per_model_wo_pilot = fitted.generate_samples_per_model(
                rvs, npilot_samples
            )

            # Generate values without pilot - shape (nqoi, nsamples) = (1,
            # nsamples_for_model)
            values_per_model_wo_pilot = [
                self._generate_correlated_values(
                    chol_factor,
                    exact_means,
                    samples_per_model_wo_pilot[ii],
                )[:, ii : ii + 1].T
                for ii in range(est.nmodels())
            ]

            # Insert pilot values
            values_per_model_recovered = fitted.insert_pilot_values(
                pilot_values, values_per_model_wo_pilot
            )

            # Generate reference values with full samples - shape (nqoi, nsamples)
            np.random.seed(seed)
            samples_per_model_full = fitted.generate_samples_per_model(rvs)
            values_per_model = [
                self._generate_correlated_values(
                    chol_factor, exact_means, samples_per_model_full[ii]
                )[:, ii : ii + 1].T
                for ii in range(est.nmodels())
            ]

            # Check that recovered values match expected
            for v1, v2 in zip(values_per_model, values_per_model_recovered):
                self._bkd.assert_allclose(v1, v2)


class TestGroupACVRecoversMFMC:
    """Test that GroupACV with MFMC subsets recovers MFMC variance."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(1)

    def _covariance_to_correlation(self, bkd, cov):
        """Convert covariance matrix to correlation matrix."""
        d = bkd.sqrt(bkd.diag(cov))
        return cov / bkd.outer(d, d)

    def test_groupacv_recovers_mfmc(self, bkd):
        """Test that GroupACV with MFMC subsets matches MFMC variance.

        When GroupACV uses the same nested subsets as MFMC, it should
        produce the same variance estimate as MFMC for the same sample
        allocation.
        """
        nmodels = 3
        cov = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        cov = self._covariance_to_correlation(bkd, cov)

        target_cost = 100.0
        costs = bkd.copy(bkd.flip(bkd.logspace(-nmodels + 1, 0, nmodels)))

        # MFMC subsets: [[0,1], [1,2], [2]]
        subsets = [[0, 1], [1, 2], [2]]
        subsets = [bkd.asarray(s, dtype=int) for s in subsets]

        stat = MultiOutputMean(1, bkd)
        stat.set_pilot_quantities(cov)

        # Create GroupACV estimator with MFMC subsets
        groupacv_est = GroupACVEstimatorNested(
            stat,
            costs,
            reg_blue=1e-8,
            model_subsets=subsets,
        )

        # Get MFMC sample allocation
        mfmc_model_ratios, mfmc_log_variance = _allocate_samples_mfmc(
            cov, costs, target_cost, bkd
        )

        # Create MFMC estimator to get partition samples
        mfmc_est = MFMCEstimator(stat, costs)
        partition_ratios = mfmc_est._native_ratios_to_npartition_ratios(
            mfmc_model_ratios
        )
        npartition_samples = mfmc_est._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios
        )

        # Check that nsamples per model match
        bkd.assert_allclose(
            groupacv_est._compute_nsamples_per_model(npartition_samples),
            mfmc_est._compute_nsamples_per_model(npartition_samples),
        )

        # Check that variance matches
        # mfmc_log_variance is scalar, covariance is (1,1) matrix
        groupacv_variance = groupacv_est._covariance_from_npartition_samples(
            npartition_samples
        )
        bkd.assert_allclose(
            bkd.exp(mfmc_log_variance),
            groupacv_variance[0, 0],
        )


class TestMFMCNestedEstimationTorchOnly:
    """Test that GroupACV matches GMF estimator for nested MFMC estimation.

    Torch-only test since GMF optimization requires bkd.jacobian.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        import torch

        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(1)

    def _get_stat(self, stat_type: str, nqoi: int):
        """Create statistic object by type."""
        if stat_type == "mean":
            return MultiOutputMean(nqoi, self._bkd)
        elif stat_type == "variance":
            return MultiOutputVariance(nqoi, self._bkd)
        elif stat_type == "mean_variance":
            return MultiOutputMeanAndVariance(nqoi, self._bkd)
        else:
            raise ValueError(f"Unknown stat_type: {stat_type}")

    def _setup_problem(self, nmodels: int, qoi_idx: list, stat_type: str):
        """Set up the multi-output problem using PSD ensemble for conditioning."""
        bm = MultiOutputEnsembleBenchmark(self._bkd, psd=True)
        model_idx = list(range(nmodels))

        # Get subproblem data
        cov = bm.covariance_subproblem(model_idx, qoi_idx)
        costs = bm.costs_subproblem(model_idx)
        models = bm.models_subproblem(model_idx, qoi_idx)

        nqoi = len(qoi_idx)
        stat = self._get_stat(stat_type, nqoi)

        # Set pilot quantities based on stat type
        if stat_type == "mean":
            stat.set_pilot_quantities(cov)
        elif stat_type == "variance":
            W = bm.covariance_of_centered_values_kronecker_product_subproblem(
                model_idx, qoi_idx
            )
            stat.set_pilot_quantities(cov, W)
        elif stat_type == "mean_variance":
            W = bm.covariance_of_centered_values_kronecker_product_subproblem(
                model_idx, qoi_idx
            )
            B = bm.covariance_of_mean_and_variance_estimators_subproblem(
                model_idx, qoi_idx
            )
            stat.set_pilot_quantities(cov, W, B)

        return bm, cov, costs, models, stat

    def _setup_polynomial_problem(self, nmodels: int, stat_type: str):
        """Set up problem using PolynomialEnsembleBenchmark (nqoi=1).

        Uses exact quadrature-based pilot statistics from the benchmark
        mixin rather than Monte Carlo pilot samples.
        """
        bm = PolynomialEnsembleBenchmark(self._bkd, nmodels=nmodels)
        costs = bm.problem().costs()
        models = bm.problem().models()
        cov = bm.covariance_matrix()

        stat = self._get_stat(stat_type, 1)
        if stat_type == "mean":
            stat.set_pilot_quantities(cov)
        elif stat_type == "variance":
            W = bm.covariance_of_centered_values_kronecker_product()
            stat.set_pilot_quantities(cov, W)
        elif stat_type == "mean_variance":
            W = bm.covariance_of_centered_values_kronecker_product()
            B = bm.covariance_of_mean_and_variance_estimators()
            stat.set_pilot_quantities(cov, W, B)

        return cov, costs, models, stat

    @pytest.mark.parametrize(
        "nmodels,qoi_idx,stat_type",
        [
            (2, [0], "mean"),
            (3, [0], "mean"),
            (2, [0, 1], "mean"),
            (2, [0, 1, 2], "mean"),
            (3, [0, 1, 2], "mean"),
            (2, [0], "variance"),
            (2, [0, 1, 2], "variance"),
            (2, [0], "mean_variance"),
            (2, [0, 1], "mean_variance"),
        ],
    )
    def test_mfmc_nested_estimation(self, nmodels: int, qoi_idx: list, stat_type: str):
        """Test GroupACV matches GMF for nested MFMC estimation.

        When GroupACV uses MFMC subsets and the same sample allocation as
        a GMF estimator, both should produce the same covariance and
        estimate values. Uses MultiOutputEnsembleBenchmark (up to 3 models,
        multi-QoI).
        """
        np.random.seed(1)
        target_cost = 10.0

        ensemble, cov, costs, models, stat = self._setup_problem(
            nmodels, qoi_idx, stat_type
        )

        # MFMC subsets: consecutive pairs + tail singleton
        # {0,1}, {1,2}, ..., {L-1,L}, {L}
        subsets = (
            [[k, k + 1] for k in range(nmodels - 1)]
            + [[nmodels - 1]]
        )
        recursion_index = self._bkd.asarray(
            list(range(nmodels - 1)), dtype=int
        )

        subsets = [self._bkd.asarray(s, dtype=int) for s in subsets]

        # Create GroupACV estimator with MFMC subsets
        groupacv_est = GroupACVEstimatorNested(
            stat,
            costs,
            model_subsets=subsets,
            reg_blue=0,
            use_pseudo_inv=False,
        )

        # Create GMF estimator (MFMC uses GMF internally)
        gmf_stat = self._get_stat(stat_type, len(qoi_idx))
        if stat_type == "mean":
            gmf_stat.set_pilot_quantities(cov)
        elif stat_type == "variance":
            W = ensemble.covariance_of_centered_values_kronecker_product_subproblem(
                list(range(nmodels)), qoi_idx
            )
            gmf_stat.set_pilot_quantities(cov, W)
        elif stat_type == "mean_variance":
            W = ensemble.covariance_of_centered_values_kronecker_product_subproblem(
                list(range(nmodels)), qoi_idx
            )
            B = ensemble.covariance_of_mean_and_variance_estimators_subproblem(
                list(range(nmodels)), qoi_idx
            )
            gmf_stat.set_pilot_quantities(cov, W, B)

        self._check_acv_groupacv_match(
            groupacv_est, gmf_stat, costs, models, recursion_index, target_cost
        )

    @pytest.mark.parametrize(
        "nmodels,stat_type,acv_type",
        [
            (4, "mean", "gmf_nested"),
            (5, "mean", "gmf_nested"),
            (4, "variance", "gmf_nested"),
            (5, "variance", "gmf_nested"),
            (4, "mean_variance", "gmf_nested"),
            (5, "mean_variance", "gmf_nested"),
            (3, "mean", "grd_is"),
            (4, "mean", "grd_is"),
            (5, "mean", "grd_is"),
            (3, "variance", "grd_is"),
            (4, "variance", "grd_is"),
            (5, "variance", "grd_is"),
            (3, "mean_variance", "grd_is"),
            (4, "mean_variance", "grd_is"),
            (5, "mean_variance", "grd_is"),
        ],
    )
    def test_mfmc_estimation_many_models(
        self, nmodels: int, stat_type: str, acv_type: str
    ):
        """Test GroupACV matches ACV for consecutive-pair MFMC subsets.

        GMF/Nested: nested samples, matches GroupACVEstimatorNested.
        GRD/IS: independent samples, matches GroupACVEstimatorIS.

        Uses PolynomialEnsembleBenchmark (nqoi=1, arbitrary nmodels).
        Both estimators share the same stat object so pilot quantities
        are identical.
        """
        np.random.seed(1)
        target_cost = 20.0

        cov, costs, models, stat = self._setup_polynomial_problem(
            nmodels, stat_type
        )

        # MFMC subsets: consecutive pairs + tail singleton
        # {0,1}, {1,2}, ..., {L-1,L}, {L}
        subsets = (
            [[k, k + 1] for k in range(nmodels - 1)]
            + [[nmodels - 1]]
        )
        recursion_index = self._bkd.asarray(
            list(range(nmodels - 1)), dtype=int
        )
        subsets = [self._bkd.asarray(s, dtype=int) for s in subsets]

        if acv_type == "gmf_nested":
            groupacv_cls = GroupACVEstimatorNested
            acv_cls = GMFEstimator
        else:
            groupacv_cls = GroupACVEstimatorIS
            acv_cls = GRDEstimator

        groupacv_est = groupacv_cls(
            stat,
            costs,
            model_subsets=subsets,
            reg_blue=0,
            use_pseudo_inv=False,
        )

        self._check_acv_groupacv_match(
            groupacv_est, stat, costs, models, recursion_index, target_cost,
            acv_cls=acv_cls,
        )

    def _check_acv_groupacv_match(
        self, groupacv_est, acv_stat, costs, models, recursion_index,
        target_cost, acv_cls=GMFEstimator,
    ):
        """Verify GroupACV and ACV estimator produce identical covariance/estimates."""
        acv_template = acv_cls(acv_stat, costs, recursion_index=recursion_index)
        acv_fitted = allocate_with_allocator(acv_template, target_cost)

        def rvs(n: int):
            return self._bkd.asarray(np.random.rand(1, int(n)))

        samples_per_model = acv_fitted.generate_samples_per_model(rvs)
        values_per_model = [
            model(samples) for model, samples in zip(models, samples_per_model)
        ]

        acv_est_val = acv_fitted(values_per_model)

        acv_npartition = acv_fitted.npartition_samples()
        groupacv_allocation = _make_groupacv_allocation(groupacv_est, acv_npartition)
        groupacv_fitted = FittedGroupACVEstimator(groupacv_est, groupacv_allocation)

        groupacv_est_val = groupacv_fitted(values_per_model)

        self._bkd.assert_allclose(
            groupacv_fitted.covariance(),
            acv_fitted.covariance(),
            rtol=1e-11,
            atol=1e-11,
        )

        self._bkd.assert_allclose(
            self._bkd.asarray(groupacv_est_val),
            self._bkd.asarray(acv_est_val),
            rtol=1e-11,
            atol=1e-11,
        )


class TestSigmaMatrixMonteCarlo:
    """Monte Carlo validation of sigma matrix computation.

    Validates that:
    1. Analytical sigma matrix matches MC covariance of subset estimates
    2. Analytical estimator variance matches MC variance of ACV estimates

    This replicates the legacy test_sigma_matrix with Monte Carlo validation.
    """

    def _rvs(self, bkd, n: int):
        """Generate uniform [0, 1] random samples."""
        return bkd.asarray(np.random.rand(1, int(n)))

    def _setup_problem(self, bkd, nmodels: int, qoi_idx: list):
        """Set up the multi-output problem."""
        bm = MultiOutputEnsembleBenchmark(bkd)
        model_idx = list(range(nmodels))
        cov = bm.covariance_subproblem(model_idx, qoi_idx)
        costs = bm.costs_subproblem(model_idx)
        models = bm.models_subproblem(model_idx, qoi_idx)
        return bm, cov, costs, models

    def _check_sigma_matrix_of_estimator(self, bkd, fitted, ntrials: int, models: list):
        """Check sigma matrix and estimator variance via Monte Carlo.

        Parameters
        ----------
        bkd : Backend
            The backend to use.
        fitted : FittedGroupACVEstimator
            The fitted estimator to check.
        ntrials : int
            Number of Monte Carlo trials.
        models : list
            List of model functions.
        """
        # Get analytical values
        t = fitted._template
        nps_float = bkd.asarray(
            fitted.npartition_samples(), dtype=bkd.double_dtype()
        )
        est_var = t._covariance_from_npartition_samples(nps_float)
        Sigma = t._sigma(nps_float)

        subset_vars_list = []
        acv_ests_list = []

        for nn in range(ntrials):
            samples_per_model = fitted.generate_samples_per_model(
                lambda n: self._rvs(bkd, n)
            )
            # Values shape: (nqoi, nsamples) for typing convention
            values_per_model = [
                models[ii](samples_per_model[ii]) for ii in range(t.nmodels())
            ]
            subset_values = fitted._separate_values_per_model(values_per_model)

            subset_vars_nn = []
            for kk in range(t.nsubsets()):
                if subset_values[kk].shape[1] > 0:
                    subset_var = fitted._stat.sample_estimate(subset_values[kk])
                else:
                    subset_var = bkd.zeros(len(t._subsets[kk]))
                subset_vars_nn.append(subset_var)

            acv_est = fitted(values_per_model)
            subset_vars_list.append(bkd.hstack(subset_vars_nn))
            acv_ests_list.append(acv_est)

        # Compute Monte Carlo covariances
        acv_ests = bkd.stack(acv_ests_list)
        subset_vars = bkd.stack(subset_vars_list)
        mc_group_cov = bkd.cov(subset_vars, ddof=1, rowvar=False)
        est_var_mc = bkd.cov(acv_ests, ddof=1, rowvar=False)

        # Check sigma matrix
        atol, rtol = 4e-3, 3e-2
        bkd.assert_allclose(
            bkd.diag(mc_group_cov),
            bkd.diag(Sigma),
            rtol=rtol,
        )
        bkd.assert_allclose(
            mc_group_cov,
            Sigma,
            rtol=rtol,
            atol=atol,
        )

        # Check estimator variance
        bkd.assert_allclose(
            est_var_mc,
            est_var,
            rtol=rtol,
            atol=atol,
        )

    def _check_sigma_matrix(
        self,
        bkd,
        nmodels: int,
        ntrials: int,
        group_type: str,
        stat_name: str,
        qoi_idx: list,
        asketch=None,
    ):
        """Set up and check sigma matrix for a test case."""
        ntrials = int(ntrials)
        ensemble, cov, costs, models = self._setup_problem(bkd, nmodels, qoi_idx)

        # Create statistic
        nqoi = len(qoi_idx)
        if stat_name == "mean":
            stat = MultiOutputMean(nqoi, bkd)
            stat.set_pilot_quantities(cov)
        else:
            raise ValueError(f"Unsupported stat_name: {stat_name}")

        # Create model subsets for 3 models
        if nmodels == 3:
            model_subsets = [[0, 1], [1, 2], [2]]
            model_subsets = [bkd.asarray(s, dtype=int) for s in model_subsets]
        else:
            model_subsets = None

        # Select estimator class based on group_type
        estimator_classes = {
            "is": GroupACVEstimatorIS,
            "nested": GroupACVEstimatorNested,
        }
        estimator_cls = estimator_classes[group_type]

        # Create estimator
        est = estimator_cls(
            stat,
            costs,
            asketch=asketch,
            reg_blue=0,
            model_subsets=model_subsets,
        )

        # Set sample allocation
        npartition_samples = bkd.arange(
            2.0, 2 + est.nsubsets(), dtype=bkd.double_dtype()
        )
        allocation = _make_groupacv_allocation(est, npartition_samples)
        fitted = FittedGroupACVEstimator(est, allocation)

        # Run Monte Carlo check
        self._check_sigma_matrix_of_estimator(bkd, fitted, ntrials, models)

    @pytest.mark.parametrize(
        "nmodels,ntrials,group_type,stat_name,qoi_idx",
        [
            (2, 50000, "is", "mean", [0]),
            (2, 50000, "is", "mean", [0, 1]),
            (2, 50000, "nested", "mean", [0]),
            (3, 50000, "is", "mean", [0]),
            (3, 50000, "is", "mean", [0, 1, 2]),
            (3, 20000, "nested", "mean", [0]),
        ],
    )
    @slower_test
    def test_sigma_matrix(
        self,
        bkd,
        nmodels: int,
        ntrials: int,
        group_type: str,
        stat_name: str,
        qoi_idx: list,
    ):
        """Test sigma matrix via Monte Carlo validation."""
        np.random.seed(1)
        self._check_sigma_matrix(bkd, nmodels, ntrials, group_type, stat_name, qoi_idx)


class TestGradientOptimizationTorchOnly:
    """Test full optimization loop with GroupACVAllocationOptimizer.

    Validates that:
    1. GroupACV and MLBLUE produce equivalent results for single QoI
    2. Optimization completes successfully for various configurations
    3. Multi-output estimation works correctly

    Torch-only test since optimization requires bkd.jacobian.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        import torch

        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def _covariance_to_correlation(self, cov):
        """Convert covariance matrix to correlation matrix."""
        d = self._bkd.sqrt(self._bkd.diag(cov))
        return cov / self._bkd.outer(d, d)

    def _create_optimizer(self):
        """Create the chained optimizer for sample allocation."""


        global_opt = ScipyDifferentialEvolutionOptimizer(
            maxiter=3,
            polish=False,
            seed=1,
            tol=1e-8,
            raise_on_failure=False,
        )
        local_opt = ScipyTrustConstrOptimizer(
            gtol=1e-8,
            maxiter=2000,
        )
        return ChainedOptimizer(global_opt, local_opt)

    def _check_gradient_optimization(
        self, nmodels: int, min_nhf_samples: int, nqoi: int
    ):
        """Check that optimization completes and produces valid results.

        Parameters
        ----------
        nmodels : int
            Number of models.
        min_nhf_samples : int
            Minimum high-fidelity samples.
        nqoi : int
            Number of quantities of interest.
        """
        # Create random covariance matrix
        cov = self._bkd.array(np.random.normal(0, 1, (nmodels * nqoi, nmodels * nqoi)))
        cov = cov.T @ cov
        cov = self._covariance_to_correlation(cov)

        target_cost = 100.0
        costs = self._bkd.flip(self._bkd.logspace(-nmodels + 1, 0, nmodels))



        # Create GroupACV estimator
        stat_g = MultiOutputMean(nqoi, self._bkd)
        stat_g.set_pilot_quantities(cov)
        gest = GroupACVEstimatorIS(stat_g, costs, reg_blue=0)

        # Create MLBLUE estimator
        stat_m = MultiOutputMean(nqoi, self._bkd)
        stat_m.set_pilot_quantities(cov)
        mlest = MLBLUEEstimator(stat_m, costs, reg_blue=0)

        # Get initial guess
        iterate = gest._init_guess(target_cost)

        # Allocate samples using new API
        gest_allocator = GroupACVAllocationOptimizer(
            gest, optimizer=self._create_optimizer()
        )
        gest_result = gest_allocator.optimize(
            target_cost, min_nhf_samples, init_guess=iterate
        )
        gest_fitted = FittedGroupACVEstimator(gest, gest_result)

        mlest_allocator = GroupACVAllocationOptimizer(
            mlest, optimizer=self._create_optimizer()
        )
        mlest_result = mlest_allocator.optimize(
            target_cost, min_nhf_samples, init_guess=iterate
        )
        mlest_fitted = FittedGroupACVEstimator(mlest, mlest_result)

        # Check that both estimators produce identical covariance matrices
        # when using the same partition samples (GroupACV allocation)
        # This is the key test from legacy: given same samples, both should agree
        nps_float = self._bkd.asarray(
            gest_fitted.npartition_samples(), dtype=self._bkd.double_dtype()
        )
        groupacv_cov = gest._covariance_from_npartition_samples(nps_float)
        mlblue_cov_at_groupacv = mlest._covariance_from_npartition_samples(
            nps_float
        )
        self._bkd.assert_allclose(
            groupacv_cov,
            mlblue_cov_at_groupacv,
        )  # default tolerance like legacy

        # Check that sample counts are valid (non-negative)
        assert float(self._bkd.min(gest_fitted.npartition_samples())) >= 0
        assert float(self._bkd.min(mlest_fitted.npartition_samples())) >= 0

        # Check cost constraint is satisfied
        gest_nps = gest_fitted.npartition_samples()
        gest_cost = gest._estimator_cost(
            self._bkd.asarray(gest_nps, dtype=self._bkd.double_dtype())
        )
        mlest_nps = mlest_fitted.npartition_samples()
        mlest_cost = mlest._estimator_cost(
            self._bkd.asarray(mlest_nps, dtype=self._bkd.double_dtype())
        )
        assert float(gest_cost) <= target_cost * 1.01  # Allow 1% tolerance
        assert float(mlest_cost) <= target_cost * 1.01

    @pytest.mark.parametrize(
        "nmodels,min_nhf_samples,nqoi",
        [
            (2, 1, 1),
            (3, 1, 1),
            (2, 1, 2),
            (3, 1, 2),
        ],
    )
    @slow_test
    def test_gradient_optimization(self, nmodels: int, min_nhf_samples: int, nqoi: int):
        """Test full optimization loop with GroupACVAllocationOptimizer."""
        np.random.seed(1)
        self._check_gradient_optimization(nmodels, min_nhf_samples, nqoi)


class TestGroupACVAllocationOptimizerTorchOnly:
    """Tests for the new allocation optimizer.

    Torch-only test since optimization requires bkd.jacobian.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        import torch

        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(1)

    def _create_estimator(self, nmodels, nqoi=1):
        """Helper to create a test estimator."""
        np.random.seed(1)
        cov_size = nmodels * nqoi
        cov = self._bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
        cov = cov.T @ cov
        costs = self._bkd.arange(nmodels, 0, -1, dtype=self._bkd.double_dtype())

        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(cov)
        return GroupACVEstimatorIS(stat, costs)

    @slow_test
    def test_allocator_produces_valid_result(self):
        """Test that allocator returns valid GroupACVAllocationResult."""


        est = self._create_estimator(3)
        allocator = GroupACVAllocationOptimizer(est)
        result = allocator.optimize(target_cost=100, min_nhf_samples=1)

        assert isinstance(result, GroupACVAllocationResult)
        assert result.success
        assert float(self._bkd.min(result.npartition_samples)) >= 0

    def test_allocator_with_custom_optimizer(self):
        """Test that custom optimizer is used."""


        est = self._create_estimator(3)
        opt = ScipyTrustConstrOptimizer(gtol=1e-6, maxiter=500)
        allocator = GroupACVAllocationOptimizer(est, optimizer=opt)
        result = allocator.optimize(target_cost=100, min_nhf_samples=1)

        assert result.success

    @slow_test
    def test_allocator_respects_budget(self):
        """Test that allocator respects budget constraint."""


        est = self._create_estimator(3)
        target_cost = 100.0
        allocator = GroupACVAllocationOptimizer(est)
        result = allocator.optimize(target_cost=target_cost, min_nhf_samples=1)

        nps_float = self._bkd.asarray(
            result.npartition_samples, dtype=self._bkd.double_dtype()
        )
        actual_cost = est._estimator_cost(nps_float)
        assert float(self._bkd.to_numpy(actual_cost)) <= target_cost + 1e-6

    @slow_test
    def test_allocator_with_mlblue_objective(self):
        """Test allocator with MLBLUEObjective for analytical derivatives."""


        np.random.seed(1)
        cov = self._bkd.array(np.random.normal(0, 1, (3, 3)))
        cov = cov.T @ cov
        costs = self._bkd.arange(3, 0, -1, dtype=self._bkd.double_dtype())

        stat = MultiOutputMean(1, self._bkd)
        stat.set_pilot_quantities(cov)
        est = MLBLUEEstimator(stat, costs)

        obj = MLBLUEObjective(self._bkd)
        allocator = GroupACVAllocationOptimizer(est, objective=obj)
        result = allocator.optimize(target_cost=100, min_nhf_samples=1)

        assert result.success

    @slow_test
    def test_allocator_result_into_fitted(self):
        """Test that allocator result can construct a fitted estimator."""


        est = self._create_estimator(3)
        allocator = GroupACVAllocationOptimizer(est)
        result = allocator.optimize(target_cost=100, min_nhf_samples=1)

        # Construct fitted estimator
        fitted = FittedGroupACVEstimator(est, result)

        # Verify allocation is accessible
        self._bkd.assert_allclose(
            fitted.npartition_samples(), result.npartition_samples
        )

        # Verify covariance can be computed
        cov = fitted.covariance()
        assert cov.shape[0] == est._stat.nstats()
        assert cov.shape[1] == est._stat.nstats()

    @pytest.mark.parametrize(
        "nmodels,nqoi",
        [(2, 1), (3, 1), (2, 2)],
    )
    @slow_test
    def test_allocator_various_configurations(self, nmodels: int, nqoi: int):
        """Test allocator with various model/qoi configurations."""


        np.random.seed(1)
        est = self._create_estimator(nmodels, nqoi)
        allocator = GroupACVAllocationOptimizer(est)
        result = allocator.optimize(target_cost=100, min_nhf_samples=1)

        assert result.success
        assert result.npartition_samples.shape[0] == est.npartitions()


class TestGroupACVRecoversMLBLUETorchOnly:
    """Test that GroupACV with IS partitions recovers MLBLUE results.

    Torch-only test since autograd optimization requires torch.

    This test validates that the new separation of allocation optimization
    allows us to verify:
    1. GroupACV with IS partitions and torch autograd optimization matches
       MLBLUE with analytical jacobian/hessian
    2. Both gradient-based approaches match MLBLUE SPD optimization
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        import torch

        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(1)

    def _covariance_to_correlation(self, cov):
        """Convert covariance matrix to correlation matrix."""
        d = self._bkd.sqrt(self._bkd.diag(cov))
        return cov / self._bkd.outer(d, d)

    def _create_optimizer(self):
        """Create the chained optimizer for sample allocation."""


        global_opt = ScipyDifferentialEvolutionOptimizer(
            maxiter=10,
            polish=False,
            seed=1,
            tol=1e-8,
            raise_on_failure=False,
        )
        local_opt = ScipyTrustConstrOptimizer(
            gtol=1e-8,
            maxiter=2000,
        )
        return ChainedOptimizer(global_opt, local_opt)

    @pytest.mark.parametrize(
        "nmodels",
        [2, 3, 4],
    )
    @slow_test
    def test_groupacv_autograd_recovers_mlblue_analytical(self, nmodels: int):
        """Test GroupACV with autograd jacobian matches MLBLUE analytical jacobian.

        This verifies that:
        1. GroupACV with IS partitions using torch autograd (default jacobian)
           produces the same allocation as MLBLUE with analytical jacobian
        2. The estimator covariances are identical for the same allocation
        """


        np.random.seed(42)
        cov_size = nmodels
        cov = self._bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
        cov = cov.T @ cov
        cov = self._covariance_to_correlation(cov)

        target_cost = 100.0
        costs = self._bkd.flip(self._bkd.logspace(-nmodels + 1, 0, nmodels))

        # Create GroupACV estimator (uses IS partitions like MLBLUE)
        stat_g = MultiOutputMean(1, self._bkd)
        stat_g.set_pilot_quantities(cov)
        groupacv_est = GroupACVEstimatorIS(stat_g, costs, reg_blue=0)

        # Create MLBLUE estimator
        stat_m = MultiOutputMean(1, self._bkd)
        stat_m.set_pilot_quantities(cov)
        mlblue_est = MLBLUEEstimator(stat_m, costs, reg_blue=0)

        # Get initial guess (same for both)
        iterate = groupacv_est._init_guess(target_cost)

        # GroupACV with autograd jacobian (via GroupACVTraceObjective)
        groupacv_obj = GroupACVTraceObjective(self._bkd)
        groupacv_allocator = GroupACVAllocationOptimizer(
            groupacv_est,
            optimizer=self._create_optimizer(),
            objective=groupacv_obj,
        )
        groupacv_result = groupacv_allocator.optimize(
            target_cost, min_nhf_samples=1, init_guess=iterate
        )

        # MLBLUE with analytical jacobian (via MLBLUEObjective)
        mlblue_obj = MLBLUEObjective(self._bkd)
        mlblue_allocator = GroupACVAllocationOptimizer(
            mlblue_est,
            optimizer=self._create_optimizer(),
            objective=mlblue_obj,
        )
        mlblue_result = mlblue_allocator.optimize(
            target_cost, min_nhf_samples=1, init_guess=iterate
        )

        # Both should have same number of partitions (IS uses 2^nmodels - 1)
        assert groupacv_est.npartitions() == mlblue_est.npartitions()

        # Covariances at same allocation should match
        nps_float = self._bkd.asarray(
            groupacv_result.npartition_samples,
            dtype=self._bkd.double_dtype(),
        )
        groupacv_cov = groupacv_est._covariance_from_npartition_samples(
            nps_float
        )
        mlblue_cov = mlblue_est._covariance_from_npartition_samples(
            nps_float
        )
        self._bkd.assert_allclose(groupacv_cov, mlblue_cov, rtol=1e-10)

        # Objective values should be close (both minimize trace)
        self._bkd.assert_allclose(
            groupacv_result.objective_value,
            mlblue_result.objective_value,
            rtol=1e-2,  # Allow some tolerance due to optimization
        )

    @pytest.mark.parametrize(
        "nmodels",
        [2, 3],
    )
    @pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")
    def test_groupacv_autograd_recovers_mlblue_spd(self, nmodels: int):
        """Test GroupACV with autograd matches MLBLUE SPD optimization.

        This verifies that:
        1. GroupACV with IS partitions using gradient-based optimization
           produces similar results to MLBLUE SPD (convex) optimization
        2. SPD should find global optimum, gradient-based should be close
        """


        np.random.seed(42)
        cov_size = nmodels
        cov = self._bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
        cov = cov.T @ cov
        cov = self._covariance_to_correlation(cov)

        target_cost = 100.0
        costs = self._bkd.flip(self._bkd.logspace(-nmodels + 1, 0, nmodels))

        # Create GroupACV estimator
        stat_g = MultiOutputMean(1, self._bkd)
        stat_g.set_pilot_quantities(cov)
        groupacv_est = GroupACVEstimatorIS(stat_g, costs, reg_blue=0)

        # Create MLBLUE estimator for SPD
        stat_m = MultiOutputMean(1, self._bkd)
        stat_m.set_pilot_quantities(cov)
        mlblue_est = MLBLUEEstimator(stat_m, costs, reg_blue=0)

        # GroupACV with gradient-based optimization
        groupacv_obj = GroupACVTraceObjective(self._bkd)
        groupacv_allocator = GroupACVAllocationOptimizer(
            groupacv_est,
            optimizer=self._create_optimizer(),
            objective=groupacv_obj,
        )
        groupacv_result = groupacv_allocator.optimize(
            target_cost, min_nhf_samples=1, round_nsamples=False
        )

        # MLBLUE with SPD optimization (convex, finds global optimum)
        spd_allocator = MLBLUESPDAllocationOptimizer(mlblue_est)
        spd_result = spd_allocator.optimize(
            target_cost, min_nhf_samples=1, round_nsamples=False
        )

        # SPD finds optimal variance (trace of covariance)
        spd_allocation = spd_result.npartition_samples
        spd_cov = mlblue_est._covariance_from_npartition_samples(spd_allocation)
        spd_trace = self._bkd.trace(spd_cov)

        # Gradient-based should find similar variance
        groupacv_cov = groupacv_est._covariance_from_npartition_samples(
            groupacv_result.npartition_samples
        )
        groupacv_trace = self._bkd.trace(groupacv_cov)

        # Gradient-based should be close to SPD optimum (within 5%)
        # SPD is convex so it finds global optimum
        self._bkd.assert_allclose(
            groupacv_trace,
            spd_trace,
            rtol=0.05,
        )


class TestSigmaBlockDerivatives:
    """Finite-difference tests for _group_acv_sigma_block_derivs."""

    def _check_derivs(self, stat, subset, n, bkd):
        d1, d2 = stat._group_acv_sigma_block_derivs(subset, n)
        eps = 1e-7
        # d1 via central FD of sigma_block
        sp = stat._group_acv_sigma_block(subset, subset, n + eps, n + eps, n + eps)
        sm = stat._group_acv_sigma_block(subset, subset, n - eps, n - eps, n - eps)
        fd1 = (sp - sm) / (2 * eps)
        bkd.assert_allclose(d1, fd1, rtol=1e-5)
        # d2 via central FD of d1
        d1p = stat._group_acv_sigma_block_derivs(subset, n + eps)[0]
        d1m = stat._group_acv_sigma_block_derivs(subset, n - eps)[0]
        fd2 = (d1p - d1m) / (2 * eps)
        bkd.assert_allclose(d2, fd2, rtol=1e-4)

    @pytest.mark.parametrize("nqoi", [1, 2])
    def test_mean_sigma_block_derivs(self, numpy_bkd, nqoi):
        np.random.seed(0)
        nmodels = 3
        cov_size = nmodels * nqoi
        cov = np.random.randn(cov_size, cov_size)
        cov = cov @ cov.T / 10
        stat = MultiOutputMean(nqoi, numpy_bkd)
        stat.set_pilot_quantities(cov)
        subset = numpy_bkd.arange(nqoi, dtype=numpy_bkd.int64_dtype())
        self._check_derivs(stat, subset, 50.0, numpy_bkd)

    @pytest.mark.parametrize("nqoi", [1, 2])
    def test_variance_sigma_block_derivs(self, numpy_bkd, nqoi):
        np.random.seed(0)
        nmodels = 3
        nsamples = 500
        pilot_values = [
            np.random.randn(nqoi, nsamples) * (0.5 + 0.5 * i)
            for i in range(nmodels)
        ]
        stat = MultiOutputVariance(nqoi, numpy_bkd)
        cov, W = stat.compute_pilot_quantities(pilot_values)
        stat.set_pilot_quantities(cov, W)
        ncov_stats = stat._tril_idx_flat.shape[0]
        subset = numpy_bkd.arange(ncov_stats, dtype=numpy_bkd.int64_dtype())
        self._check_derivs(stat, subset, 10.0, numpy_bkd)

    @pytest.mark.parametrize("nqoi", [1, 2])
    def test_joint_sigma_block_derivs(self, numpy_bkd, nqoi):
        np.random.seed(0)
        nmodels = 3
        nsamples = 500
        pilot_values = [
            np.random.randn(nqoi, nsamples) * (0.5 + 0.5 * i)
            for i in range(nmodels)
        ]
        stat = MultiOutputMeanAndVariance(nqoi, numpy_bkd)
        cov, W, B = stat.compute_pilot_quantities(pilot_values)
        stat.set_pilot_quantities(cov, W, B)
        nstats = stat.nstats()
        subset = numpy_bkd.arange(nstats, dtype=numpy_bkd.int64_dtype())
        self._check_derivs(stat, subset, 10.0, numpy_bkd)


class TestAnalyticalGroupACVDerivatives:
    """Test analytical jacobian/hessian/hvp for GroupACV objectives.

    Uses DerivativeChecker for jacobian and hvp, and parametrizes
    over all stat types and both objectives.
    """

    @staticmethod
    def _create_estimator(bkd, nmodels, nqoi, stat_type):
        np.random.seed(1)
        nsamples = 500
        pilot_values = [
            np.random.randn(nqoi, nsamples) * (0.5 + 0.5 * i)
            for i in range(nmodels)
        ]
        costs = bkd.arange(nmodels, 0, -1, dtype=bkd.double_dtype())
        if stat_type == "mean":
            stat = MultiOutputMean(nqoi, bkd)
            cov = stat.compute_pilot_quantities(pilot_values)[0]
            stat.set_pilot_quantities(cov)
        elif stat_type == "variance":
            stat = MultiOutputVariance(nqoi, bkd)
            cov, W = stat.compute_pilot_quantities(pilot_values)
            stat.set_pilot_quantities(cov, W)
        elif stat_type == "joint":
            stat = MultiOutputMeanAndVariance(nqoi, bkd)
            cov, W, B = stat.compute_pilot_quantities(pilot_values)
            stat.set_pilot_quantities(cov, W, B)
        else:
            raise ValueError(f"Unknown stat_type: {stat_type}")
        return GroupACVEstimatorIS(stat, costs)

    @pytest.mark.parametrize(
        "nmodels,nqoi,stat_type",
        [
            (2, 1, "mean"),
            (3, 1, "mean"),
            (2, 2, "mean"),
            (2, 1, "variance"),
            (3, 1, "variance"),
            (2, 2, "variance"),
            (2, 1, "joint"),
            (3, 1, "joint"),
            (2, 2, "joint"),
        ],
    )
    @pytest.mark.parametrize(
        "objective_cls",
        [GroupACVTraceObjective, GroupACVLogDetObjective],
    )
    def test_analytical_jacobian_and_hvp(
        self, numpy_bkd, nmodels, nqoi, stat_type, objective_cls,
    ):
        est = self._create_estimator(numpy_bkd, nmodels, nqoi, stat_type)
        obj = objective_cls(numpy_bkd)
        obj.set_estimator(est)
        assert obj._use_analytical

        iterate = est._init_guess(100)
        checker = DerivativeChecker(obj)
        errors = checker.check_derivatives(iterate, verbosity=0)

        assert checker.error_ratio(errors[0]) <= 5e-6
        if len(errors) > 1:
            assert checker.error_ratio(errors[1]) <= 5e-6

    @pytest.mark.parametrize(
        "nmodels,nqoi",
        [(2, 1), (3, 1), (2, 2)],
    )
    def test_trace_matches_mlblue_for_mean(self, numpy_bkd, nmodels, nqoi):
        """Verify GroupACVTraceObjective analytical jac matches MLBLUEObjective."""
        est = self._create_estimator(numpy_bkd, nmodels, nqoi, "mean")
        trace_obj = GroupACVTraceObjective(numpy_bkd)
        trace_obj.set_estimator(est)
        mlblue_obj = MLBLUEObjective(numpy_bkd)
        mlblue_obj.set_estimator(est)

        iterate = est._init_guess(100)
        jac_trace = trace_obj.jacobian(iterate)
        jac_mlblue = mlblue_obj.jacobian(iterate)
        numpy_bkd.assert_allclose(jac_trace, jac_mlblue, rtol=1e-10)


class TestGroupACVPsiConsistency:
    """Test that GroupACV and MLBLUE Psi matrices agree for IS estimators.

    For IS (independent sampling), GroupACV computes
        Psi = R @ inv(Sigma) @ R^T
    where Sigma = blkdiag(C_0/n_0, ..., C_S/n_S).

    MLBLUE precomputes Psi_block_m = R_m @ inv(C_mm) @ R_m^T once, then
        Psi = sum(n_m * Psi_block_m)

    These are mathematically identical, but GroupACV's full-Sigma inversion
    becomes numerically unstable when some n_m are near zero (Sigma becomes
    ill-conditioned and pinv truncates directions). This class tests that
    the two formulations agree across the feasible region.
    """

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_benchmark_estimators(self, bkd):
        """Create paired GroupACV-IS and MLBLUE estimators.

        Uses 5 models with costs [1, 0.1, 0.01, 0.001, 0.0001] and
        the PolynomialEnsembleBenchmark covariance. This setup produces
        31 partitions (2^5 - 1) with a wide cost range, making it a
        good stress test for numerical stability.
        """
        nmodels = 5
        bench = PolynomialEnsembleBenchmark(bkd, nmodels=nmodels)
        costs = bench.problem().costs()
        cov = bench.ensemble_covariance()
        subsets = get_model_subsets(nmodels, bkd)

        stat_g = MultiOutputMean(1, bkd)
        stat_g.set_pilot_quantities(cov)
        gacv = GroupACVEstimatorIS(stat_g, costs, model_subsets=subsets)

        stat_m = MultiOutputMean(1, bkd)
        stat_m.set_pilot_quantities(cov)
        mlblue = MLBLUEEstimator(stat_m, costs, reg_blue=0)

        return gacv, mlblue, costs, subsets

    @pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")
    def test_trace_agrees_at_slsqp_solution(self, numpy_bkd):
        """GroupACV and MLBLUE trace must agree at optimizer solutions.

        SLSQP can push partitions to the lower bound (1e-8), making
        Sigma rank-deficient. pinv then truncates directions, causing
        GroupACV's Psi to differ from MLBLUE's linear formulation.
        """
        bkd = numpy_bkd
        gacv, mlblue, costs, subsets = self._create_benchmark_estimators(bkd)

        # SDP gives a good starting point with many near-zero partitions
        spd = MLBLUESPDAllocationOptimizer(mlblue)
        spd_result = spd.optimize(target_cost=100.0, round_nsamples=False)
        spd_nps = spd_result.npartition_samples

        # SLSQP from SDP init — it pushes near-zero partitions to the
        # lower bound, creating the rank-deficient Sigma
        trace_obj = GroupACVTraceObjective(bkd)
        slsqp = ScipySLSQPOptimizer(maxiter=1000, ftol=1e-15)
        alloc = GroupACVAllocationOptimizer(
            gacv, optimizer=slsqp, objective=trace_obj,
        )
        slsqp_result = alloc.optimize(
            target_cost=100.0,
            round_nsamples=False,
            init_guess=bkd.asarray(
                np.array(spd_nps)[:, None], dtype=bkd.double_dtype()
            ),
        )
        slsqp_nps = slsqp_result.npartition_samples

        # Evaluate both formulations at the SLSQP solution
        nps_bkd = bkd.asarray(slsqp_nps, dtype=bkd.double_dtype())
        mlblue_trace = bkd.trace(
            mlblue._covariance_from_npartition_samples(nps_bkd)
        )
        gacv_trace = trace_obj(
            bkd.asarray(np.array(slsqp_nps)[:, None], dtype=bkd.double_dtype())
        ).flatten()

        bkd.assert_allclose(
            gacv_trace,
            bkd.asarray([mlblue_trace]),
            rtol=1e-6,
        )

    @pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")
    def test_all_optimizers_agree_for_mean_estimation(self, numpy_bkd):
        """GroupACV and MLBLUE formulations agree at every optimizer solution.

        Tests SDP, SLSQP, Trust-Constr, and (if available) ROL.
        SLSQP may find better allocations than SDP by pushing unused
        partitions to the lower bound — that is fine as long as both
        formulations agree at the solution point.
        """
        bkd = numpy_bkd
        gacv, mlblue, costs, subsets = self._create_benchmark_estimators(bkd)

        # SDP provides init for gradient-based optimizers
        spd = MLBLUESPDAllocationOptimizer(mlblue)
        spd_result = spd.optimize(target_cost=100.0, round_nsamples=False)
        spd_nps = spd_result.npartition_samples

        sdp_init = bkd.asarray(
            np.array(spd_nps)[:, None], dtype=bkd.double_dtype()
        )

        # Collect (optimizer_name, result_nps) pairs
        optimizer_results = []

        # SLSQP
        trace_obj_slsqp = GroupACVTraceObjective(bkd)
        slsqp = ScipySLSQPOptimizer(maxiter=1000, ftol=1e-15)
        alloc_slsqp = GroupACVAllocationOptimizer(
            gacv, optimizer=slsqp, objective=trace_obj_slsqp,
        )
        res_slsqp = alloc_slsqp.optimize(
            target_cost=100.0, round_nsamples=False, init_guess=sdp_init,
        )
        optimizer_results.append(("SLSQP", res_slsqp.npartition_samples))

        # Trust-Constr
        stat_tc = MultiOutputMean(1, bkd)
        stat_tc.set_pilot_quantities(
            mlblue._stat.pilot_covariance()
        )
        gacv_tc = GroupACVEstimatorIS(
            stat_tc, costs, model_subsets=subsets,
        )
        trace_obj_tc = GroupACVTraceObjective(bkd)
        tc = ScipyTrustConstrOptimizer(gtol=1e-12, maxiter=1000)
        alloc_tc = GroupACVAllocationOptimizer(
            gacv_tc, optimizer=tc, objective=trace_obj_tc,
        )
        res_tc = alloc_tc.optimize(
            target_cost=100.0, round_nsamples=False, init_guess=sdp_init,
        )
        optimizer_results.append(("Trust-Constr", res_tc.npartition_samples))

        # ROL (optional dependency)
        if package_available("pyrol"):
            from pyapprox.optimization.minimize.rol.rol_optimizer import (
                ROLOptimizer,
            )
            stat_rol = MultiOutputMean(1, bkd)
            stat_rol.set_pilot_quantities(
                mlblue._stat.pilot_covariance()
            )
            gacv_rol = GroupACVEstimatorIS(
                stat_rol, costs, model_subsets=subsets,
            )
            trace_obj_rol = GroupACVTraceObjective(bkd)
            rol = ROLOptimizer()
            alloc_rol = GroupACVAllocationOptimizer(
                gacv_rol, optimizer=rol, objective=trace_obj_rol,
            )
            res_rol = alloc_rol.optimize(
                target_cost=100.0, round_nsamples=False, init_guess=sdp_init,
            )
            optimizer_results.append(("ROL", res_rol.npartition_samples))

        # Every optimizer's solution must have MLBLUE trace within 5%
        # of SDP optimum, and GroupACV trace must agree with MLBLUE trace
        for name, nps in optimizer_results:
            nps_bkd = bkd.asarray(nps, dtype=bkd.double_dtype())

            # MLBLUE trace (ground truth at this point)
            mlblue_trace = float(bkd.trace(
                mlblue._covariance_from_npartition_samples(nps_bkd)
            ))

            # GroupACV trace (what the optimizer actually minimized)
            trace_obj_eval = GroupACVTraceObjective(bkd)
            trace_obj_eval.set_estimator(gacv)
            gacv_trace = float(trace_obj_eval(
                bkd.asarray(np.array(nps)[:, None], dtype=bkd.double_dtype())
            ).flatten()[0])

            # The two formulations must agree at any feasible point
            bkd.assert_allclose(
                bkd.asarray([gacv_trace]),
                bkd.asarray([mlblue_trace]),
                rtol=1e-6,
            )


class TestDeadThresholdGuard:
    """Tests for the continuous_dead_threshold guard in sigma blocks."""

    def _make_variance_stat(self, bkd):
        np.random.seed(42)
        nsamples = 100
        vals0 = bkd.array(np.random.randn(1, nsamples))
        vals1 = bkd.array(
            0.8 * np.random.randn(1, nsamples)
            + 0.2 * np.random.randn(1, nsamples)
        )
        stat = MultiOutputVariance(1, bkd)
        cov, W = stat.compute_pilot_quantities([vals0, vals1])
        stat.set_pilot_quantities(cov, W)
        return stat

    def test_variance_block_below_threshold_is_zero(self, numpy_bkd):
        bkd = numpy_bkd
        stat = self._make_variance_stat(bkd)
        subset = bkd.array([0], dtype=int)
        block = _grouped_acv_sigma_block(subset, subset, 1.5, 1.5, 1.5, stat)
        bkd.assert_allclose(block, bkd.full((1, 1), 0.0))

    def test_variance_block_at_threshold_is_nonzero(self, numpy_bkd):
        bkd = numpy_bkd
        stat = self._make_variance_stat(bkd)
        subset = bkd.array([0], dtype=int)
        block = _grouped_acv_sigma_block(subset, subset, 2.0, 2.0, 2.0, stat)
        assert not bkd.allclose(block, bkd.full((1, 1), 0.0))
        assert np.all(np.isfinite(bkd.to_numpy(block)))

    def test_variance_block_above_threshold_is_nonzero(self, numpy_bkd):
        bkd = numpy_bkd
        stat = self._make_variance_stat(bkd)
        subset = bkd.array([0], dtype=int)
        block = _grouped_acv_sigma_block(subset, subset, 2.5, 2.5, 2.5, stat)
        assert not bkd.allclose(block, bkd.full((1, 1), 0.0))

    def test_mean_block_at_small_n_is_nonzero(self, numpy_bkd):
        bkd = numpy_bkd
        cov = bkd.array([[1.0, 0.5], [0.5, 1.0]])
        stat = MultiOutputMean(1, bkd)
        stat.set_pilot_quantities(cov)
        subset = bkd.array([0], dtype=int)
        block = _grouped_acv_sigma_block(
            subset, subset, 0.001, 0.001, 0.001, stat
        )
        assert not bkd.allclose(block, bkd.full((1, 1), 0.0))

    def test_continuous_dead_threshold_values(self, numpy_bkd):
        bkd = numpy_bkd
        assert MultiOutputMean(1, bkd).continuous_dead_threshold() == 0.0
        assert MultiOutputVariance(1, bkd).continuous_dead_threshold() == 2.0
        assert (
            MultiOutputMeanAndVariance(1, bkd).continuous_dead_threshold()
            == 2.0
        )


class TestFullSigmaPathMatchesMLBLUE:
    """Validate the full-Sigma codepath used by GroupACVEstimatorNested.

    Since nested has no independent reference, we cross-check the shared
    base-class full-Sigma path (sigma -> psi_matrix_from_sigma) against
    MLBLUE using GroupACVEstimatorIS at benign allocations.
    """

    @pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")
    def test_full_sigma_psi_matches_mlblue(self, numpy_bkd):
        bkd = numpy_bkd
        nmodels = 5
        bench = PolynomialEnsembleBenchmark(bkd, nmodels=nmodels)
        costs = bench.problem().costs()
        cov = bench.ensemble_covariance()
        subsets = get_model_subsets(nmodels, bkd)

        stat_m = MultiOutputMean(1, bkd)
        stat_m.set_pilot_quantities(cov)
        mlblue = MLBLUEEstimator(stat_m, costs, reg_blue=0)

        stat_g = MultiOutputMean(1, bkd)
        stat_g.set_pilot_quantities(cov)
        gacv = GroupACVEstimatorIS(stat_g, costs, model_subsets=subsets)

        spd = MLBLUESPDAllocationOptimizer(mlblue)
        spd_result = spd.optimize(target_cost=100.0, round_nsamples=False)
        nps = bkd.asarray(
            spd_result.npartition_samples, dtype=bkd.double_dtype()
        )

        # Manually call the base-class full-Sigma path (not the IS override)
        from pyapprox.statest.groupacv.base import BaseGroupACVEstimator

        sigma = gacv._sigma(nps)
        psi_full = BaseGroupACVEstimator._psi_matrix_from_sigma(gacv, sigma)

        # MLBLUE Psi for comparison
        psi_mlblue = mlblue._psi_matrix(nps)

        bkd.assert_allclose(psi_full, psi_mlblue, rtol=1e-8)


class TestPsiWellConditionedSparseAllocations:
    """Verify block-by-block Psi matches MLBLUE with sparse allocations."""

    @pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")
    def test_sparse_mean_allocations_match_mlblue(self, numpy_bkd):
        bkd = numpy_bkd
        nmodels = 5
        bench = PolynomialEnsembleBenchmark(bkd, nmodels=nmodels)
        costs = bench.problem().costs()
        cov = bench.ensemble_covariance()
        subsets = get_model_subsets(nmodels, bkd)

        stat_m = MultiOutputMean(1, bkd)
        stat_m.set_pilot_quantities(cov)
        mlblue = MLBLUEEstimator(stat_m, costs, reg_blue=0)

        stat_g = MultiOutputMean(1, bkd)
        stat_g.set_pilot_quantities(cov)
        gacv = GroupACVEstimatorIS(stat_g, costs, model_subsets=subsets)

        spd = MLBLUESPDAllocationOptimizer(mlblue)
        spd_result = spd.optimize(target_cost=100.0, round_nsamples=False)
        nps = np.array(spd_result.npartition_samples)

        # Force many partitions to 1e-8
        nps_sparse = nps.copy()
        nps_sparse[nps_sparse < 1.0] = 1e-8
        pc = np.array(bkd.einsum(
            "m,mp->p", costs, gacv._partitions_per_model
        ))
        cost = np.sum(nps_sparse * pc)
        if cost > 100.0:
            nps_sparse *= 100.0 / cost

        nps_bkd = bkd.asarray(nps_sparse, dtype=bkd.double_dtype())
        mlblue_tr = float(bkd.trace(
            mlblue._covariance_from_npartition_samples(nps_bkd)
        ))
        gacv_tr = float(bkd.trace(
            gacv._covariance_from_npartition_samples(nps_bkd)
        ))

        bkd.assert_allclose(
            bkd.asarray([gacv_tr]),
            bkd.asarray([mlblue_tr]),
            rtol=1e-8,
        )


class TestISBetaEquivalence:
    """Verify IS block-by-block beta matches full-Sigma reference."""

    @staticmethod
    def _create_estimator(bkd, nmodels, nqoi, stat_type):
        np.random.seed(1)
        nsamples = 500
        pilot_values = [
            np.random.randn(nqoi, nsamples) * (0.5 + 0.5 * i)
            for i in range(nmodels)
        ]
        costs = bkd.arange(nmodels, 0, -1, dtype=bkd.double_dtype())
        subsets = get_model_subsets(nmodels, bkd)
        if stat_type == "mean":
            stat = MultiOutputMean(nqoi, bkd)
            cov = stat.compute_pilot_quantities(pilot_values)[0]
            stat.set_pilot_quantities(cov)
        elif stat_type == "variance":
            stat = MultiOutputVariance(nqoi, bkd)
            cov, W = stat.compute_pilot_quantities(pilot_values)
            stat.set_pilot_quantities(cov, W)
        elif stat_type == "joint":
            stat = MultiOutputMeanAndVariance(nqoi, bkd)
            cov, W, B = stat.compute_pilot_quantities(pilot_values)
            stat.set_pilot_quantities(cov, W, B)
        else:
            raise ValueError(f"Unknown stat_type: {stat_type}")
        return GroupACVEstimatorIS(stat, costs, model_subsets=subsets)

    @pytest.mark.parametrize(
        "nmodels,nqoi,stat_type",
        [
            (3, 1, "mean"),
            (3, 1, "variance"),
            (3, 1, "joint"),
            (3, 2, "mean"),
        ],
    )
    def test_is_beta_matches_full_sigma_reference(
        self, numpy_bkd, nmodels, nqoi, stat_type,
    ):
        bkd = numpy_bkd
        est = self._create_estimator(bkd, nmodels, nqoi, stat_type)
        nps = bkd.full((est.nsubsets(),), 50.0, dtype=bkd.double_dtype())

        # IS override: block-by-block beta
        beta_is = est._grouped_acv_beta(nps)

        # Full-Sigma reference (base class path)
        from pyapprox.statest.groupacv.base import BaseGroupACVEstimator

        sigma = est._sigma(nps)
        psi_ref = BaseGroupACVEstimator._psi_matrix_from_sigma(est, sigma)
        beta_ref = bkd.stack(
            [
                bkd.multidot(
                    [est._inv(sigma), est._R.T, bkd.solve(psi_ref, a)]
                )
                for a in est._asketch
            ],
            axis=0,
        )

        bkd.assert_allclose(beta_is, beta_ref, rtol=1e-10)

    def test_nested_beta_unchanged(self, numpy_bkd):
        bkd = numpy_bkd
        nmodels = 3
        np.random.seed(1)
        nsamples = 500
        pilot_values = [
            np.random.randn(1, nsamples) * (0.5 + 0.5 * i)
            for i in range(nmodels)
        ]
        costs = bkd.arange(nmodels, 0, -1, dtype=bkd.double_dtype())
        subsets = get_model_subsets(nmodels, bkd)

        stat = MultiOutputMean(1, bkd)
        cov = stat.compute_pilot_quantities(pilot_values)[0]
        stat.set_pilot_quantities(cov)
        est = GroupACVEstimatorNested(
            stat, costs, model_subsets=subsets,
        )
        nps = bkd.full((est.nsubsets(),), 50.0, dtype=bkd.double_dtype())

        # Nested uses base-class path (full-Sigma)
        beta = est._grouped_acv_beta(nps)

        # Reference: manually call same path
        sigma = est._sigma(nps)
        psi = est._psi_matrix_from_sigma(sigma)
        beta_ref = bkd.stack(
            [
                bkd.multidot(
                    [est._inv(sigma), est._R.T, bkd.solve(psi, a)]
                )
                for a in est._asketch
            ],
            axis=0,
        )

        bkd.assert_allclose(beta, beta_ref, rtol=1e-12)

    def test_fitted_estimator_end_to_end(self, numpy_bkd):
        bkd = numpy_bkd
        nmodels = 3
        est = self._create_estimator(bkd, nmodels, 1, "mean")

        trace_obj = GroupACVTraceObjective(bkd)
        slsqp = ScipySLSQPOptimizer(maxiter=1000, ftol=1e-15)
        alloc_opt = GroupACVAllocationOptimizer(
            est, optimizer=slsqp, objective=trace_obj,
        )
        alloc = alloc_opt.optimize(
            target_cost=100.0, round_nsamples=True,
        )
        FittedGroupACVEstimator(est, alloc)

        nps_float = bkd.asarray(
            alloc.npartition_samples, dtype=bkd.double_dtype()
        )
        beta = est._grouped_acv_beta(nps_float)

        # Beta shape: (nstats_output, ntotal_stats)
        assert beta.shape[0] == len(est._asketch)

        # Beta rows sum to 1 for mean estimation
        row_sums = bkd.sum(beta, axis=1)
        bkd.assert_allclose(
            row_sums, bkd.ones((beta.shape[0],)), rtol=1e-10,
        )

        # No NaN in covariance
        cov = est._covariance_from_npartition_samples(nps_float)
        assert np.all(np.isfinite(bkd.to_numpy(cov)))


# ---------------------------------------------------------------------------
# Variable rescaling: derivative checks for wrapper classes
# ---------------------------------------------------------------------------


class TestRescaledObjectiveDerivativesTorchOnly:
    """DerivativeChecker tests for _RescaledObjective wrapper."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        import torch

        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def _create_estimator(self, nmodels, nqoi=1):
        np.random.seed(1)
        cov_size = nmodels * nqoi
        cov = self._bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
        cov = cov.T @ cov
        costs = self._bkd.arange(
            nmodels, 0, -1, dtype=self._bkd.double_dtype()
        )
        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(cov)
        return GroupACVEstimatorIS(stat, costs)

    @pytest.mark.parametrize("nmodels,nqoi", [(2, 1), (3, 1), (2, 2)])
    def test_rescaled_trace_jacobian(self, nmodels: int, nqoi: int):
        np.random.seed(1)
        est = self._create_estimator(nmodels, nqoi)
        bkd = self._bkd
        obj = GroupACVTraceObjective(bkd)
        obj.set_estimator(est)
        partition_costs = bkd.einsum(
            "m,mp->p", est._costs, est._partitions_per_model
        )
        scale = partition_costs / bkd.min(partition_costs)
        wrapped = _RescaledObjective(obj, scale)
        iterate = est._init_guess(100) * scale[:, None]
        checker = DerivativeChecker(wrapped)
        errors = checker.check_derivatives(iterate, verbosity=0)
        assert checker.error_ratio(errors[0]) <= 3e-6

    @pytest.mark.parametrize("nmodels,nqoi", [(2, 1), (3, 1), (2, 2)])
    def test_rescaled_logdet_jacobian(self, nmodels: int, nqoi: int):
        np.random.seed(1)
        est = self._create_estimator(nmodels, nqoi)
        bkd = self._bkd
        obj = GroupACVLogDetObjective(bkd)
        obj.set_estimator(est)
        partition_costs = bkd.einsum(
            "m,mp->p", est._costs, est._partitions_per_model
        )
        scale = partition_costs / bkd.min(partition_costs)
        wrapped = _RescaledObjective(obj, scale)
        iterate = est._init_guess(100) * scale[:, None]
        checker = DerivativeChecker(wrapped)
        errors = checker.check_derivatives(iterate, verbosity=0)
        assert checker.error_ratio(errors[0]) <= 3e-6


class TestRescaledConstraintDerivatives:
    """DerivativeChecker tests for _RescaledConstraint wrapper."""

    def _create_estimator(self, bkd, nmodels):
        np.random.seed(1)
        cov = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        costs = bkd.arange(nmodels, 0, -1, dtype=bkd.double_dtype())
        stat = MultiOutputMean(1, bkd)
        stat.set_pilot_quantities(cov)
        return GroupACVEstimatorIS(stat, costs)

    @pytest.mark.parametrize("nmodels", [2, 3])
    def test_rescaled_constraint_jacobian(self, bkd, nmodels: int):
        np.random.seed(1)
        est = self._create_estimator(bkd, nmodels)
        con = GroupACVCostConstraint(bkd)
        con.set_estimator(est)
        con.set_budget(100.0, 1)
        partition_costs = bkd.einsum(
            "m,mp->p", est._costs, est._partitions_per_model
        )
        scale = partition_costs / bkd.min(partition_costs)
        wrapped = _RescaledConstraint(con, scale)
        iterate = est._init_guess(100) * scale[:, None]
        checker = DerivativeChecker(wrapped)
        weights = bkd.asarray([[0.6, 0.4]])
        errors = checker.check_derivatives(
            iterate, weights=weights, verbosity=0
        )
        assert float(errors[0][0]) < 1e-12


class TestNormalizedConstraintDerivatives:
    """DerivativeChecker tests for _NormalizedConstraint wrapper."""

    def _create_estimator(self, bkd, nmodels):
        np.random.seed(1)
        cov = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        costs = bkd.arange(nmodels, 0, -1, dtype=bkd.double_dtype())
        stat = MultiOutputMean(1, bkd)
        stat.set_pilot_quantities(cov)
        return GroupACVEstimatorIS(stat, costs)

    @pytest.mark.parametrize("nmodels", [2, 3])
    def test_normalized_constraint_jacobian(self, bkd, nmodels: int):
        """_NormalizedConstraint wrapping raw constraint."""
        np.random.seed(1)
        est = self._create_estimator(bkd, nmodels)
        con = GroupACVCostConstraint(bkd)
        con.set_estimator(est)
        con.set_budget(100.0, 1)
        norm = bkd.array([100.0, 1.0])
        wrapped = _NormalizedConstraint(con, norm)
        iterate = est._init_guess(100)
        checker = DerivativeChecker(wrapped)
        weights = bkd.asarray([[0.6, 0.4]])
        errors = checker.check_derivatives(
            iterate, weights=weights, verbosity=0
        )
        assert float(errors[0][0]) < 1e-12

    @pytest.mark.parametrize("nmodels", [2, 3])
    def test_composed_rescaled_normalized_jacobian(self, bkd, nmodels: int):
        """_NormalizedConstraint wrapping _RescaledConstraint (FullCostSpace)."""
        np.random.seed(1)
        est = self._create_estimator(bkd, nmodels)
        con = GroupACVCostConstraint(bkd)
        con.set_estimator(est)
        con.set_budget(100.0, 1)
        partition_costs = bkd.einsum(
            "m,mp->p", est._costs, est._partitions_per_model
        )
        scale = partition_costs / bkd.min(partition_costs)
        norm = bkd.array([100.0, 1.0])
        rescaled = _RescaledConstraint(con, scale)
        composed = _NormalizedConstraint(rescaled, norm)
        iterate = est._init_guess(100) * scale[:, None]
        checker = DerivativeChecker(composed)
        weights = bkd.asarray([[0.6, 0.4]])
        errors = checker.check_derivatives(
            iterate, weights=weights, verbosity=0
        )
        assert float(errors[0][0]) < 1e-12


# ---------------------------------------------------------------------------
# Variable rescaling: integration tests for allocation optimizer
# ---------------------------------------------------------------------------


class TestVariableRescalingIntegrationTorchOnly:
    """Integration tests for variable rescaling in GroupACVAllocationOptimizer.

    Torch-only since optimization requires bkd.jacobian (autograd).
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        import torch

        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def _create_estimator(self, nmodels=3, nqoi=1, cost_ratio=None):
        np.random.seed(1)
        bkd = self._bkd
        cov_size = nmodels * nqoi
        cov = bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
        cov = cov.T @ cov
        if cost_ratio is not None:
            log_costs = np.linspace(0, np.log10(cost_ratio), nmodels)
            costs = bkd.array(
                10.0 ** log_costs[::-1], dtype=bkd.double_dtype()
            )
        else:
            costs = bkd.arange(nmodels, 0, -1, dtype=bkd.double_dtype())
        stat = MultiOutputMean(nqoi, bkd)
        stat.set_pilot_quantities(cov)
        return GroupACVEstimatorIS(stat, costs)

    @slow_test
    def test_identity_config_matches_default(self):
        """Default AllocationProblemConfig produces same result as no config."""
        est = self._create_estimator(3)
        allocator_default = GroupACVAllocationOptimizer(est)
        result_default = allocator_default.optimize(target_cost=100)

        est2 = self._create_estimator(3)
        config = AllocationProblemConfig()
        allocator_config = GroupACVAllocationOptimizer(
            est2, problem_config=config
        )
        result_config = allocator_config.optimize(target_cost=100)

        self._bkd.assert_allclose(
            result_config.npartition_samples,
            result_default.npartition_samples,
        )

    @slow_test
    def test_well_conditioned_full_matches_none(self):
        """Full rescaling matches identity on well-conditioned problem."""
        est_none = self._create_estimator(3, cost_ratio=10)
        config_none = AllocationProblemConfig(
            variable_scaling="none", bounds_lb=1e-8
        )
        allocator_none = GroupACVAllocationOptimizer(
            est_none, problem_config=config_none
        )
        result_none = allocator_none.optimize(target_cost=100)

        est_full = self._create_estimator(3, cost_ratio=10)
        config_full = AllocationProblemConfig(
            variable_scaling="full", bounds_lb=1e-8
        )
        allocator_full = GroupACVAllocationOptimizer(
            est_full, problem_config=config_full
        )
        result_full = allocator_full.optimize(target_cost=100)

        assert result_none.success
        assert result_full.success
        self._bkd.assert_allclose(
            result_full.npartition_samples,
            result_none.npartition_samples,
            rtol=1e-4,
        )

    @slow_test
    def test_constraint_only_matches_none(self):
        """Constraint-only normalization matches identity in allocation."""
        est_none = self._create_estimator(3)
        config_none = AllocationProblemConfig(
            variable_scaling="none", bounds_lb=1e-8
        )
        allocator_none = GroupACVAllocationOptimizer(
            est_none, problem_config=config_none
        )
        result_none = allocator_none.optimize(target_cost=100)

        est_cs = self._create_estimator(3)
        config_cs = AllocationProblemConfig(
            variable_scaling="constraint_only", bounds_lb=1e-8
        )
        allocator_cs = GroupACVAllocationOptimizer(
            est_cs, problem_config=config_cs
        )
        result_cs = allocator_cs.optimize(target_cost=100)

        assert result_none.success
        assert result_cs.success
        self._bkd.assert_allclose(
            result_cs.npartition_samples,
            result_none.npartition_samples,
            rtol=1e-4,
        )

    @slow_test
    def test_equality_budget_uses_full_budget(self):
        """Equality budget produces continuous allocation using full budget."""
        est = self._create_estimator(3)
        target_cost = 100.0
        config = AllocationProblemConfig(
            budget_constraint_form="equality", bounds_lb=1e-8
        )
        allocator = GroupACVAllocationOptimizer(est, problem_config=config)
        result = allocator.optimize(
            target_cost=target_cost, round_nsamples=False
        )

        assert result.success
        assert result.actual_cost == pytest.approx(target_cost, rel=1e-4)

    @slow_test
    def test_repeated_optimize_no_state_leakage(self):
        """Calling optimize() twice with different configs gives correct results."""
        est = self._create_estimator(3)
        target_cost = 100.0

        config_ineq = AllocationProblemConfig(
            budget_constraint_form="inequality", bounds_lb=1e-8
        )
        allocator = GroupACVAllocationOptimizer(est, problem_config=config_ineq)
        result_ineq = allocator.optimize(target_cost=target_cost)

        est2 = self._create_estimator(3)
        config_eq = AllocationProblemConfig(
            budget_constraint_form="equality", bounds_lb=1e-8
        )
        allocator2 = GroupACVAllocationOptimizer(est2, problem_config=config_eq)
        result_eq = allocator2.optimize(target_cost=target_cost)

        assert result_ineq.success
        assert result_eq.success
        assert result_eq.actual_cost >= result_ineq.actual_cost - 1e-6

    @pytest.mark.parametrize("cost_ratio", [1e2, 1e4])
    @slow_test
    def test_ill_conditioned_full_succeeds(self, cost_ratio: float):
        """Full rescaling succeeds on ill-conditioned cost ratios."""
        est = self._create_estimator(3, cost_ratio=cost_ratio)
        target_cost = 10.0 * cost_ratio
        config = AllocationProblemConfig(
            variable_scaling="full", bounds_lb=1e-8,
        )
        allocator = GroupACVAllocationOptimizer(est, problem_config=config)
        result = allocator.optimize(target_cost=target_cost)
        assert result.success
        assert result.actual_cost <= target_cost + 1e-6
