"""Standalone tests for GroupACV and MLBLUE estimators.

These tests do not depend on legacy code and will remain after
the legacy module is removed. All expected values are derived
from mathematical definitions.
"""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.statest.groupacv import (
    get_model_subsets,
    _get_allocation_matrix_is,
    _get_allocation_matrix_nested,
    _nest_subsets,
    GroupACVEstimatorIS,
    GroupACVEstimatorNested,
    MLBLUEEstimator,
    GroupACVTraceObjective,
    GroupACVLogDetObjective,
    MLBLUEObjective,
    GroupACVCostConstraint,
    MLBLUESPDAllocationOptimizer,
)
from pyapprox.typing.statest.groupacv.allocation import GroupACVAllocationResult
from pyapprox.typing.statest.acv import MFMCEstimator, GMFEstimator
from pyapprox.typing.statest.acv.variants import _allocate_samples_mfmc
from pyapprox.typing.benchmarks.functions.multifidelity.multioutput_ensemble import (
    MultiOutputModelEnsemble,
    PSDMultiOutputModelEnsemble,
)
from pyapprox.typing.statest.statistics import (
    MultiOutputVariance,
    MultiOutputMeanAndVariance,
)
from pyapprox.typing.util.optional_deps import package_available
from pyapprox.typing.statest.statistics import MultiOutputMean
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.typing.util.test_utils import slower_test, allocate_with_allocator


def _make_groupacv_allocation(est, npartition_samples):
    """Helper to create GroupACVAllocationResult from npartition_samples.

    Rounds the samples and computes derived quantities.
    """
    bkd = est.bkd()
    rounded = bkd.floor(npartition_samples + 1e-4)
    nsamples_per_model = est._compute_nsamples_per_model(rounded)
    actual_cost = float(est._estimator_cost(rounded))
    return GroupACVAllocationResult(
        npartition_samples=rounded,
        nsamples_per_model=nsamples_per_model,
        actual_cost=actual_cost,
        objective_value=bkd.array([0.0]),  # Placeholder
        success=True,
        message="",
    )


class TestGroupACVUtils(Generic[Array], unittest.TestCase):
    """Tests for GroupACV utility functions."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(1)

    def test_get_model_subsets(self):
        """Test model subset generation."""
        nmodels = 3
        subsets = get_model_subsets(nmodels, self._bkd)

        # Should have 2^nmodels - 1 = 7 subsets
        self._bkd.assert_allclose(
            self._bkd.asarray([len(subsets)]),
            self._bkd.asarray([7]),
        )

        # Check first few subsets
        self._bkd.assert_allclose(subsets[0], self._bkd.asarray([0]))
        self._bkd.assert_allclose(subsets[1], self._bkd.asarray([1]))
        self._bkd.assert_allclose(subsets[2], self._bkd.asarray([2]))

    def test_allocation_matrix_is(self):
        """Test independent sampling allocation matrix."""
        nmodels = 3
        subsets = get_model_subsets(nmodels, self._bkd)
        allocation_mat = _get_allocation_matrix_is(subsets, self._bkd)

        # IS allocation should be identity
        expected = self._bkd.eye(len(subsets))
        self._bkd.assert_allclose(allocation_mat, expected)

    def test_allocation_matrix_nested(self):
        """Test nested sampling allocation matrix."""
        nmodels = 3
        # Remove subset 0 for nested
        subsets = get_model_subsets(nmodels, self._bkd)[1:]
        subsets = _nest_subsets(subsets, nmodels, self._bkd)[0]

        # Re-sort as in legacy code
        idx = sorted(
            list(range(len(subsets))),
            key=lambda ii: (len(subsets[ii]), tuple(nmodels - subsets[ii])),
            reverse=True,
        )
        subsets = [subsets[ii] for ii in idx]

        nsubsets = len(subsets)
        allocation_mat = _get_allocation_matrix_nested(subsets, self._bkd)

        # Nested allocation should be lower triangular of ones
        expected = self._bkd.asarray(np.tril(np.ones((nsubsets, nsubsets))))
        self._bkd.assert_allclose(allocation_mat, expected)


class TestGroupACVUtilsNumpy(TestGroupACVUtils[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGroupACVUtilsTorch(TestGroupACVUtils[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGroupACVEstimator(Generic[Array], unittest.TestCase):
    """Tests for GroupACVEstimator."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(1)

    def _create_estimator(self, nmodels, estimator_cls=GroupACVEstimatorIS):
        """Helper to create a test estimator."""
        np.random.seed(1)
        cov = self._bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        costs = self._bkd.arange(nmodels, 0, -1, dtype=self._bkd.double_dtype())

        stat = MultiOutputMean(1, self._bkd)
        stat.set_pilot_quantities(cov)
        est = estimator_cls(stat, costs)
        return est

    def test_nsamples_per_model_is(self):
        """Test sample count computation for IS estimation."""
        nmodels = 3
        est = self._create_estimator(nmodels, estimator_cls=GroupACVEstimatorIS)

        npartition_samples = self._bkd.arange(
            2.0, 2 + est.nsubsets(), dtype=self._bkd.double_dtype()
        )

        # Expected values derived mathematically from subset structure
        expected_nsamples = self._bkd.asarray([21.0, 23.0, 25.0])
        self._bkd.assert_allclose(
            est._compute_nsamples_per_model(npartition_samples),
            expected_nsamples,
        )

        # Check total cost: sum(nsamples_i * cost_i) where costs = [3, 2, 1]
        expected_cost = self._bkd.asarray([21 * 3 + 23 * 2 + 25 * 1.0])
        self._bkd.assert_allclose(
            self._bkd.asarray([est._estimator_cost(npartition_samples)]),
            expected_cost,
        )

        # For IS, intersection samples matrix is diagonal
        self._bkd.assert_allclose(
            est._nintersect_samples(npartition_samples),
            self._bkd.diag(npartition_samples),
        )

    def test_nsamples_per_model_nested(self):
        """Test sample count computation for nested estimation."""
        nmodels = 3
        np.random.seed(1)
        est = self._create_estimator(nmodels, estimator_cls=GroupACVEstimatorNested)

        npartition_samples = self._bkd.arange(
            2.0, 2.0 + est.nsubsets(), dtype=self._bkd.double_dtype()
        )

        # Expected values from nested structure
        expected_nsamples = self._bkd.asarray([9, 20, 27.0])
        self._bkd.assert_allclose(
            est._compute_nsamples_per_model(npartition_samples),
            expected_nsamples,
        )

        # Check total cost
        expected_cost = self._bkd.asarray([9 * 3 + 20 * 2 + 27 * 1.0])
        self._bkd.assert_allclose(
            self._bkd.asarray([est._estimator_cost(npartition_samples)]),
            expected_cost,
        )

        # Check intersection samples matrix (nested structure)
        expected_intersect = self._bkd.asarray(
            [
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                [2.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                [2.0, 5.0, 9.0, 9.0, 9.0, 9.0],
                [2.0, 5.0, 9.0, 14.0, 14.0, 14.0],
                [2.0, 5.0, 9.0, 14.0, 20.0, 20.0],
                [2.0, 5.0, 9.0, 14.0, 20.0, 27.0],
            ]
        )
        self._bkd.assert_allclose(
            est._nintersect_samples(npartition_samples),
            expected_intersect,
        )

    def test_basic_accessors(self):
        """Test basic accessor methods."""
        nmodels = 3
        est = self._create_estimator(nmodels)

        self._bkd.assert_allclose(
            self._bkd.asarray([est.nmodels()]),
            self._bkd.asarray([nmodels]),
        )
        # For IS with nmodels=3, we get 2^3 - 1 = 7 subsets
        self._bkd.assert_allclose(
            self._bkd.asarray([est.nsubsets()]),
            self._bkd.asarray([7]),
        )
        self._bkd.assert_allclose(
            self._bkd.asarray([est.npartitions()]),
            self._bkd.asarray([7]),
        )

    def _check_separate_samples(self, est):
        """Test sample separation logic."""
        NN = 2
        npartition_samples = self._bkd.full(
            (est.nsubsets(),), float(NN), dtype=self._bkd.double_dtype()
        )
        allocation = _make_groupacv_allocation(est, npartition_samples)
        est.set_allocation(allocation)

        samples_per_model = est.generate_samples_per_model(
            lambda n: self._bkd.arange(int(n), dtype=self._bkd.double_dtype())[
                None, :
            ]
        )
        for ii in range(est.nmodels()):
            self._bkd.assert_allclose(
                self._bkd.asarray([samples_per_model[ii].shape[1]]),
                self._bkd.asarray([int(allocation.nsamples_per_model[ii])]),
            )

        # values shape is (nqoi, nsamples) - same as samples but with nqoi rows
        values_per_model = [
            (ii + 1) * s for ii, s in enumerate(samples_per_model)
        ]
        values_per_subset = est._separate_values_per_model(values_per_model)

        test_samples = self._bkd.arange(
            int(est.npartition_samples().sum()),
            dtype=self._bkd.double_dtype(),
        )[None, :]
        # test_values shape is (nqoi, nsamples)
        test_values = [
            (ii + 1) * test_samples for ii in range(est.nmodels())
        ]

        for ii in range(est.nsubsets()):
            active_partitions = self._bkd.where(est._allocation_mat[ii] == 1)[0]
            indices = (
                self._bkd.arange(
                    test_samples.shape[1], dtype=self._bkd.int64_dtype()
                )
                .reshape(est.npartitions(), NN)[active_partitions]
                .flatten()
            )
            # expected_shape is (nqoi*nmodels_in_subset, nsamples_in_subset)
            expected_shape = (
                len(est._subsets[ii]),
                int(est._nintersect_samples(npartition_samples)[ii][ii]),
            )
            self._bkd.assert_allclose(
                self._bkd.asarray(list(values_per_subset[ii].shape)),
                self._bkd.asarray(list(expected_shape)),
            )

    def test_separate_samples_is(self):
        """Test sample separation for IS estimation."""
        est = self._create_estimator(3, estimator_cls=GroupACVEstimatorIS)
        self._check_separate_samples(est)

    def test_separate_samples_nested(self):
        """Test sample separation for nested estimation."""
        est = self._create_estimator(3, estimator_cls=GroupACVEstimatorNested)
        self._check_separate_samples(est)


class TestGroupACVEstimatorNumpy(TestGroupACVEstimator[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGroupACVEstimatorTorch(TestGroupACVEstimator[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGroupACVObjectiveDerivativesTorchOnly(ParametrizedTestCase, unittest.TestCase):
    """Test derivative correctness for GroupACV objectives using DerivativeChecker.

    These tests only run with Torch backend because the base objectives use
    bkd.jacobian() which requires autograd (only available in Torch).
    """

    def setUp(self):
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

    @parametrize(
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
        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-6)

    @parametrize(
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
        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-6)


class TestGroupACVConstraintDerivatives(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Test derivative correctness for GroupACV constraints."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _create_estimator(self, nmodels):
        """Helper to create a test estimator."""
        np.random.seed(1)
        cov = self._bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        costs = self._bkd.arange(nmodels, 0, -1, dtype=self._bkd.double_dtype())

        stat = MultiOutputMean(1, self._bkd)
        stat.set_pilot_quantities(cov)
        est = GroupACVEstimatorIS(stat, costs)
        return est

    @parametrize(
        "nmodels",
        [(2,), (3,)],
    )
    def test_constraint_jacobian(self, nmodels: int):
        """Test constraint Jacobian with DerivativeChecker."""
        np.random.seed(1)
        est = self._create_estimator(nmodels)
        target_cost = 100
        min_nhf_samples = 1

        constraint = GroupACVCostConstraint(self._bkd)
        constraint.set_estimator(est)
        constraint.set_budget(target_cost, min_nhf_samples)

        iterate = est._init_guess(target_cost)
        checker = DerivativeChecker(constraint)
        # Pass weights for multi-QoI constraint (nqoi=2) with whvp
        weights = self._bkd.asarray([[0.6, 0.4]])  # Shape (1, nqoi)
        errors = checker.check_derivatives(iterate, weights=weights, verbosity=0)

        # Constraints are linear, so Jacobian should be exact
        self.assertLess(float(errors[0][0]), 1e-12)

    @parametrize(
        "nmodels",
        [(2,), (3,)],
    )
    def test_constraint_hessian_zero(self, nmodels: int):
        """Test that constraint Hessian is zero (linear constraints)."""
        np.random.seed(1)
        est = self._create_estimator(nmodels)

        constraint = GroupACVCostConstraint(self._bkd)
        constraint.set_estimator(est)
        constraint.set_budget(target_cost=100.0, min_nhf_samples=1)

        npartition_samples = self._bkd.full(
            (est.npartitions(), 1), 1.0, dtype=self._bkd.double_dtype()
        )
        hess = constraint.hessian(npartition_samples)

        # Hessian should be all zeros since constraints are linear
        self._bkd.assert_allclose(
            hess,
            self._bkd.zeros((2, est.npartitions(), est.npartitions())),
        )


class TestGroupACVConstraintDerivativesNumpy(
    TestGroupACVConstraintDerivatives[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGroupACVConstraintDerivativesTorch(
    TestGroupACVConstraintDerivatives[torch.Tensor]
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestMLBLUEEstimator(Generic[Array], unittest.TestCase):
    """Tests for MLBLUEEstimator."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(1)

    def test_mlblue_uses_is(self):
        """Test that MLBLUE uses independent sampling."""
        nmodels = 3
        cov = self._bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        costs = self._bkd.arange(nmodels, 0, -1, dtype=self._bkd.double_dtype())

        stat = MultiOutputMean(1, self._bkd)
        stat.set_pilot_quantities(cov)
        est = MLBLUEEstimator(stat, costs)

        # MLBLUE should use IS, so allocation should be identity-like
        allocation_mat = est._allocation_mat
        self._bkd.assert_allclose(
            allocation_mat,
            self._bkd.eye(est.nsubsets()),
        )

    def test_mlblue_psi_blocks(self):
        """Test that MLBLUE precomputes psi blocks."""
        nmodels = 2
        cov = self._bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        costs = self._bkd.arange(nmodels, 0, -1, dtype=self._bkd.double_dtype())

        stat = MultiOutputMean(1, self._bkd)
        stat.set_pilot_quantities(cov)
        est = MLBLUEEstimator(stat, costs)

        # Should have precomputed psi blocks
        self._bkd.assert_allclose(
            self._bkd.asarray([len(est._psi_blocks)]),
            self._bkd.asarray([est.nsubsets()]),
        )

    def test_mlblue_inherits_groupacv(self):
        """Test that MLBLUEEstimator inherits from GroupACVEstimatorIS."""
        nmodels = 2
        cov = self._bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        costs = self._bkd.arange(nmodels, 0, -1, dtype=self._bkd.double_dtype())

        stat = MultiOutputMean(1, self._bkd)
        stat.set_pilot_quantities(cov)
        est = MLBLUEEstimator(stat, costs)

        # Should be instance of GroupACVEstimatorIS
        assert isinstance(est, GroupACVEstimatorIS)


class TestMLBLUEEstimatorNumpy(TestMLBLUEEstimator[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMLBLUEEstimatorTorch(TestMLBLUEEstimator[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestMLBLUEObjectiveDerivatives(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Test MLBLUE-specific objective derivatives."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _create_mlblue_estimator(self, nmodels, nqoi=1):
        """Helper to create a test MLBLUE estimator."""
        np.random.seed(1)
        cov_size = nmodels * nqoi
        cov = self._bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
        cov = cov.T @ cov
        costs = self._bkd.arange(nmodels, 0, -1, dtype=self._bkd.double_dtype())

        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(cov)
        est = MLBLUEEstimator(stat, costs)
        return est

    @parametrize(
        "nmodels,nqoi",
        [(2, 1), (3, 1), (2, 2)],
    )
    def test_mlblue_objective_jacobian(self, nmodels: int, nqoi: int):
        """Test MLBLUE objective analytical Jacobian."""
        np.random.seed(1)
        est = self._create_mlblue_estimator(nmodels, nqoi)
        target_cost = 100

        obj = MLBLUEObjective(self._bkd)
        obj.set_estimator(est)

        iterate = est._init_guess(target_cost)
        checker = DerivativeChecker(obj)
        errors = checker.check_derivatives(iterate, verbosity=0)

        # Check Jacobian accuracy
        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-6)

    def test_mlblue_objective_inherits_trace(self):
        """Test that MLBLUEObjective inherits from GroupACVTraceObjective."""
        obj = MLBLUEObjective(self._bkd)
        assert isinstance(obj, GroupACVTraceObjective)


class TestMLBLUEObjectiveDerivativesNumpy(
    TestMLBLUEObjectiveDerivatives[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMLBLUEObjectiveDerivativesTorch(
    TestMLBLUEObjectiveDerivatives[torch.Tensor]
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestRestrictionMatrices(Generic[Array], unittest.TestCase):
    """Tests for restriction matrix functionality."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(1)

    def test_restriction_matrices_nqoi_1(self):
        """Test restriction matrices with nqoi=1."""
        qoi_idx = [0]
        costs = self._bkd.array(
            [1, 0.5, 0.25], dtype=self._bkd.double_dtype()
        )
        stat = MultiOutputMean(len(qoi_idx), self._bkd)
        subsets = [[0, 1, 2], [1], [2]]
        subsets = [
            self._bkd.array(s, dtype=self._bkd.int64_dtype()) for s in subsets
        ]
        est = GroupACVEstimatorNested(stat, costs, model_subsets=subsets)

        # Vector containing model ids
        Lvec = self._bkd.arange(3, dtype=self._bkd.double_dtype())[:, None]

        # Make sure each restriction matrix recovers correct subset model ids
        cnt = 0
        for ii in range(len(subsets)):
            recovered = (est._R[:, cnt : cnt + len(subsets[ii])].T @ Lvec)[:, 0]
            expected = self._bkd.asarray(
                subsets[ii], dtype=self._bkd.double_dtype()
            )
            self._bkd.assert_allclose(recovered, expected)
            cnt += len(subsets[ii])

    def test_restriction_matrices_nqoi_2(self):
        """Test restriction matrices with nqoi=2."""
        qoi_idx = [0, 1]
        costs = self._bkd.array(
            [1, 0.5, 0.25], dtype=self._bkd.double_dtype()
        )
        stat = MultiOutputMean(len(qoi_idx), self._bkd)
        subsets = [[0, 1, 2], [1], [2]]
        subsets = [
            self._bkd.array(s, dtype=self._bkd.int64_dtype()) for s in subsets
        ]
        est = GroupACVEstimatorNested(stat, costs, model_subsets=subsets)

        # Vector containing flattened model qoi ids
        Lvec = self._bkd.arange(
            3 * len(qoi_idx), dtype=self._bkd.double_dtype()
        )[:, None]

        # Check restriction matrix recovers all correct qoi of all subset model ids
        self._bkd.assert_allclose(
            (est._R[:, :6].T @ Lvec)[:, 0],
            self._bkd.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        )
        self._bkd.assert_allclose(
            (est._R[:, 6:8].T @ Lvec)[:, 0], self._bkd.array([2.0, 3.0])
        )
        self._bkd.assert_allclose(
            (est._R[:, 8:10].T @ Lvec)[:, 0], self._bkd.array([4.0, 5.0])
        )


class TestRestrictionMatricesNumpy(TestRestrictionMatrices[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestRestrictionMatricesTorch(TestRestrictionMatrices[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# CVXPY-dependent tests
HAS_CVXPY = package_available("cvxpy")


@unittest.skipIf(not HAS_CVXPY, "cvxpy not installed")
class TestMLBLUESPDAllocationOptimizer(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Tests for MLBLUESPDAllocationOptimizer (requires cvxpy)."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(1)

    def _create_mlblue_estimator(self, nmodels, nqoi=1):
        """Helper to create a test MLBLUE estimator."""
        np.random.seed(1)
        cov_size = nmodels * nqoi
        cov = self._bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
        cov = cov.T @ cov
        # Normalize to correlation matrix for better conditioning
        d = self._bkd.sqrt(self._bkd.diag(cov))
        cov = cov / self._bkd.outer(d, d)
        costs = self._bkd.array(
            [1.0 / (10 ** i) for i in range(nmodels)],
            dtype=self._bkd.double_dtype(),
        )

        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(cov)
        est = MLBLUEEstimator(stat, costs, reg_blue=0)
        return est

    @parametrize(
        "nmodels,min_nhf_samples",
        [(2, 1), (3, 1), (4, 1), (3, 10)],
    )
    def test_spd_allocator_solves(self, nmodels: int, min_nhf_samples: int):
        """Test that SPD allocator finds a solution."""
        np.random.seed(1)
        est = self._create_mlblue_estimator(nmodels, nqoi=1)
        target_cost = 100

        allocator = MLBLUESPDAllocationOptimizer(est)
        result = allocator.optimize(target_cost, min_nhf_samples)

        # Check that optimization succeeded
        self.assertTrue(result.success)

        # Check that result has correct shape
        self._bkd.assert_allclose(
            self._bkd.asarray([result.npartition_samples.shape[0]]),
            self._bkd.asarray([est.nsubsets()]),
        )

        # Check that all sample counts are non-negative
        self.assertTrue(
            float(self._bkd.min(result.npartition_samples)) >= -1e-10
        )

    @parametrize(
        "nmodels",
        [(2,), (3,)],
    )
    def test_spd_respects_budget(self, nmodels: int):
        """Test that SPD allocator respects budget constraint."""
        np.random.seed(1)
        est = self._create_mlblue_estimator(nmodels, nqoi=1)
        target_cost = 100
        min_nhf_samples = 1

        allocator = MLBLUESPDAllocationOptimizer(est)
        result = allocator.optimize(target_cost, min_nhf_samples)

        # Compute actual cost
        actual_cost = est._estimator_cost(result.npartition_samples)

        # Cost should be <= target (with small tolerance)
        self.assertLessEqual(
            float(self._bkd.to_numpy(actual_cost)),
            target_cost + 1e-6,
        )

    def test_spd_raises_for_multioutput(self):
        """Test that SPD allocator raises error for multi-output."""
        est = self._create_mlblue_estimator(2, nqoi=2)
        allocator = MLBLUESPDAllocationOptimizer(est)
        with self.assertRaises(RuntimeError):
            allocator.optimize(100, 1)


@unittest.skipIf(not HAS_CVXPY, "cvxpy not installed")
class TestMLBLUESPDAllocationOptimizerNumpy(
    TestMLBLUESPDAllocationOptimizer[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


@unittest.skipIf(not HAS_CVXPY, "cvxpy not installed")
class TestMLBLUESPDAllocationOptimizerTorch(
    TestMLBLUESPDAllocationOptimizer[torch.Tensor]
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestPilotSampleInsertion(Generic[Array], ParametrizedTestCase):
    """Tests for pilot sample insertion and removal."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _covariance_to_correlation(self, cov: Array) -> Array:
        """Convert covariance matrix to correlation matrix."""
        d = self._bkd.sqrt(self._bkd.diag(cov))
        return cov / self._bkd.outer(d, d)

    def _generate_correlated_values(
        self, chol_factor: Array, means: Array, samples: Array
    ) -> Array:
        """Generate correlated values from samples."""
        return (chol_factor @ samples + means[:, None]).T

    def _create_mlblue_estimator(self, nmodels: int) -> MLBLUEEstimator:
        """Create an MLBLUE estimator for testing."""
        cov = self._bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        cov = self._covariance_to_correlation(cov)
        costs = self._bkd.flip(
            self._bkd.logspace(-nmodels + 1, 0, nmodels)
        )
        stat = MultiOutputMean(1, self._bkd)
        stat.set_pilot_quantities(cov)
        return MLBLUEEstimator(stat, costs, reg_blue=1e-10)

    def _create_optimizer(self):
        """Create optimizer with sufficient iterations for test."""
        from pyapprox.typing.optimization.minimize.chained.chained_optimizer import (
            ChainedOptimizer,
        )
        from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
            ScipyTrustConstrOptimizer,
        )
        from pyapprox.typing.optimization.minimize.scipy.diffevol import (
            ScipyDifferentialEvolutionOptimizer,
        )

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

    @parametrize(
        "nmodels,min_nhf_samples",
        [(2, 11), (3, 11), (4, 11)],
    )
    def test_insert_pilot_samples(self, nmodels: int, min_nhf_samples: int):
        """Test that pilot values can be correctly inserted."""
        from pyapprox.typing.statest.groupacv.allocation import (
            GroupACVAllocationOptimizer,
        )

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
            est.set_allocation(result)

            # Get covariance and create value generator
            cov = est._stat._cov
            chol_factor = self._bkd.cholesky(cov)
            exact_means = self._bkd.arange(
                nmodels, dtype=self._bkd.double_dtype()
            )

            # Sample generator with nmodels variables (required for correlation)
            def rvs(n: int) -> Array:
                return self._bkd.array(np.random.randn(nmodels, int(n)))

            # Generate full samples
            np.random.seed(seed)
            samples_per_model = est.generate_samples_per_model(rvs)

            # Remove pilot samples
            _, pilot_samples = est._remove_pilot_samples(
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
            samples_per_model_wo_pilot = est.generate_samples_per_model(
                rvs, npilot_samples
            )

            # Generate values without pilot - shape (nqoi, nsamples) = (1, nsamples_for_model)
            values_per_model_wo_pilot = [
                self._generate_correlated_values(
                    chol_factor,
                    exact_means,
                    samples_per_model_wo_pilot[ii],
                )[:, ii : ii + 1].T
                for ii in range(est.nmodels())
            ]

            # Insert pilot values
            values_per_model_recovered = est.insert_pilot_values(
                pilot_values, values_per_model_wo_pilot
            )

            # Generate reference values with full samples - shape (nqoi, nsamples)
            np.random.seed(seed)
            samples_per_model_full = est.generate_samples_per_model(rvs)
            values_per_model = [
                self._generate_correlated_values(
                    chol_factor, exact_means, samples_per_model_full[ii]
                )[:, ii : ii + 1].T
                for ii in range(est.nmodels())
            ]

            # Check that recovered values match expected
            for v1, v2 in zip(values_per_model, values_per_model_recovered):
                self._bkd.assert_allclose(v1, v2)


class TestPilotSampleInsertionTorchOnly(TestPilotSampleInsertion[torch.Tensor]):
    """Torch-only test since optimization requires bkd.jacobian."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGroupACVRecoversMFMC(Generic[Array], unittest.TestCase):
    """Test that GroupACV with MFMC subsets recovers MFMC variance."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(1)

    def _covariance_to_correlation(self, cov: Array) -> Array:
        """Convert covariance matrix to correlation matrix."""
        d = self._bkd.sqrt(self._bkd.diag(cov))
        return cov / self._bkd.outer(d, d)

    def test_groupacv_recovers_mfmc(self):
        """Test that GroupACV with MFMC subsets matches MFMC variance.

        When GroupACV uses the same nested subsets as MFMC, it should
        produce the same variance estimate as MFMC for the same sample
        allocation.
        """
        nmodels = 3
        cov = self._bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        cov = self._covariance_to_correlation(cov)

        target_cost = 100.0
        costs = self._bkd.copy(
            self._bkd.flip(self._bkd.logspace(-nmodels + 1, 0, nmodels))
        )

        # MFMC subsets: [[0,1], [1,2], [2]]
        subsets = [[0, 1], [1, 2], [2]]
        subsets = [self._bkd.asarray(s, dtype=int) for s in subsets]

        stat = MultiOutputMean(1, self._bkd)
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
            cov, costs, target_cost, self._bkd
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
        self._bkd.assert_allclose(
            groupacv_est._compute_nsamples_per_model(npartition_samples),
            mfmc_est._compute_nsamples_per_model(npartition_samples),
        )

        # Check that variance matches
        # mfmc_log_variance is scalar, covariance is (1,1) matrix
        groupacv_variance = groupacv_est._covariance_from_npartition_samples(
            npartition_samples
        )
        self._bkd.assert_allclose(
            self._bkd.exp(mfmc_log_variance),
            groupacv_variance[0, 0],
        )


class TestGroupACVRecoversMFMCNumpy(TestGroupACVRecoversMFMC[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGroupACVRecoversMFMCTorch(TestGroupACVRecoversMFMC[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestMFMCNestedEstimation(Generic[Array], ParametrizedTestCase):
    """Test that GroupACV matches GMF estimator for nested MFMC estimation."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
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
        ensemble = PSDMultiOutputModelEnsemble(self._bkd)
        model_idx = list(range(nmodels))

        # Get subproblem data
        cov = ensemble.covariance_subproblem(model_idx, qoi_idx)
        costs = ensemble.costs_subproblem(model_idx)
        models = ensemble.models_subproblem(model_idx, qoi_idx)

        nqoi = len(qoi_idx)
        stat = self._get_stat(stat_type, nqoi)

        # Set pilot quantities based on stat type
        if stat_type == "mean":
            stat.set_pilot_quantities(cov)
        elif stat_type == "variance":
            W = ensemble.covariance_of_centered_values_kronecker_product_subproblem(
                model_idx, qoi_idx
            )
            stat.set_pilot_quantities(cov, W)
        elif stat_type == "mean_variance":
            W = ensemble.covariance_of_centered_values_kronecker_product_subproblem(
                model_idx, qoi_idx
            )
            B = ensemble.covariance_of_mean_and_variance_estimators_subproblem(
                model_idx, qoi_idx
            )
            stat.set_pilot_quantities(cov, W, B)

        return ensemble, cov, costs, models, stat

    @parametrize(
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
    def test_mfmc_nested_estimation(
        self, nmodels: int, qoi_idx: list, stat_type: str
    ):
        """Test GroupACV matches GMF for nested MFMC estimation.

        When GroupACV uses MFMC subsets and the same sample allocation as
        a GMF estimator, both should produce the same covariance and
        estimate values.
        """
        np.random.seed(1)
        target_cost = 10.0

        ensemble, cov, costs, models, stat = self._setup_problem(
            nmodels, qoi_idx, stat_type
        )

        # MFMC subsets for nested estimation
        if nmodels == 3:
            subsets = [[0, 1], [1, 2], [2]]
            recursion_index = self._bkd.asarray([0, 1], dtype=int)
        elif nmodels == 2:
            subsets = [[0, 1], [1]]
            recursion_index = self._bkd.asarray([0], dtype=int)
        else:
            raise ValueError(f"Unsupported nmodels: {nmodels}")

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

        gmf_est = GMFEstimator(gmf_stat, costs, recursion_index=recursion_index)
        allocate_with_allocator(gmf_est, target_cost)

        # Generate samples and values using GMF allocation
        def rvs(n: int) -> Array:
            return self._bkd.asarray(np.random.rand(1, int(n)))

        samples_per_model = gmf_est.generate_samples_per_model(rvs)
        # Values shape: (nqoi, nsamples) - samples in columns
        values_per_model = [
            model(samples)
            for model, samples in zip(models, samples_per_model)
        ]

        # Compute GMF estimate (expects samples in columns)
        gmf_est_val = gmf_est(values_per_model)

        # Apply GMF sample allocation to GroupACV
        # Create GroupACV allocation from GMF allocation
        gmf_npartition = gmf_est.npartition_samples()
        groupacv_allocation = _make_groupacv_allocation(
            groupacv_est, gmf_npartition
        )
        groupacv_est.set_allocation(groupacv_allocation)

        # Compute GroupACV estimate (now expects samples in columns like other typing code)
        groupacv_est_val = groupacv_est(values_per_model)

        # Check covariances match (use default bkd.allclose tolerances)
        self._bkd.assert_allclose(
            groupacv_est.optimized_covariance(),
            gmf_est.optimized_covariance(),
            rtol=1e-05,
            atol=1e-08,
        )

        # Check estimate values match (use default bkd.allclose tolerances)
        self._bkd.assert_allclose(
            self._bkd.asarray(groupacv_est_val),
            self._bkd.asarray(gmf_est_val),
            rtol=1e-05,
            atol=1e-08,
        )


class TestMFMCNestedEstimationTorchOnly(TestMFMCNestedEstimation[torch.Tensor]):
    """Torch-only test since GMF optimization requires bkd.jacobian."""
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestSigmaMatrixMonteCarlo(Generic[Array], ParametrizedTestCase):
    """Monte Carlo validation of sigma matrix computation.

    Validates that:
    1. Analytical sigma matrix matches MC covariance of subset estimates
    2. Analytical estimator variance matches MC variance of ACV estimates

    This replicates the legacy test_sigma_matrix with Monte Carlo validation.
    """

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _rvs(self, n: int) -> Array:
        """Generate uniform [0, 1] random samples."""
        return self._bkd.asarray(np.random.rand(1, int(n)))

    def _setup_problem(self, nmodels: int, qoi_idx: list):
        """Set up the multi-output problem."""
        ensemble = MultiOutputModelEnsemble(self._bkd)
        model_idx = list(range(nmodels))
        cov = ensemble.covariance_subproblem(model_idx, qoi_idx)
        costs = ensemble.costs_subproblem(model_idx)
        models = ensemble.models_subproblem(model_idx, qoi_idx)
        return ensemble, cov, costs, models

    def _check_sigma_matrix_of_estimator(
        self, est, ntrials: int, models: list
    ):
        """Check sigma matrix and estimator variance via Monte Carlo.

        Parameters
        ----------
        est : GroupACVEstimator
            The estimator to check.
        ntrials : int
            Number of Monte Carlo trials.
        models : list
            List of model functions.
        """
        # Get analytical values
        est_var = est._covariance_from_npartition_samples(
            est.npartition_samples()
        )
        Sigma = est._sigma(est.npartition_samples())

        subset_vars_list = []
        acv_ests_list = []

        for nn in range(ntrials):
            samples_per_model = est.generate_samples_per_model(self._rvs)
            # Values shape: (nqoi, nsamples) for typing convention
            values_per_model = [
                models[ii](samples_per_model[ii])
                for ii in range(est.nmodels())
            ]
            subset_values = est._separate_values_per_model(values_per_model)

            subset_vars_nn = []
            for kk in range(est.nsubsets()):
                if subset_values[kk].shape[1] > 0:
                    subset_var = est._stat.sample_estimate(subset_values[kk])
                else:
                    subset_var = self._bkd.zeros(len(est._subsets[kk]))
                subset_vars_nn.append(subset_var)

            acv_est = est(values_per_model)
            subset_vars_list.append(self._bkd.hstack(subset_vars_nn))
            acv_ests_list.append(acv_est)

        # Compute Monte Carlo covariances
        acv_ests = self._bkd.stack(acv_ests_list)
        subset_vars = self._bkd.stack(subset_vars_list)
        mc_group_cov = self._bkd.cov(subset_vars, ddof=1, rowvar=False)
        est_var_mc = self._bkd.cov(acv_ests, ddof=1, rowvar=False)

        # Check sigma matrix
        atol, rtol = 4e-3, 3e-2
        self._bkd.assert_allclose(
            self._bkd.diag(mc_group_cov),
            self._bkd.diag(Sigma),
            rtol=rtol,
        )
        self._bkd.assert_allclose(
            mc_group_cov,
            Sigma,
            rtol=rtol,
            atol=atol,
        )

        # Check estimator variance
        self._bkd.assert_allclose(
            est_var_mc,
            est_var,
            rtol=rtol,
            atol=atol,
        )

    def _check_sigma_matrix(
        self,
        nmodels: int,
        ntrials: int,
        group_type: str,
        stat_name: str,
        qoi_idx: list,
        asketch=None,
    ):
        """Set up and check sigma matrix for a test case."""
        ntrials = int(ntrials)
        ensemble, cov, costs, models = self._setup_problem(nmodels, qoi_idx)

        # Create statistic
        nqoi = len(qoi_idx)
        if stat_name == "mean":
            stat = MultiOutputMean(nqoi, self._bkd)
            stat.set_pilot_quantities(cov)
        else:
            raise ValueError(f"Unsupported stat_name: {stat_name}")

        # Create model subsets for 3 models
        if nmodels == 3:
            model_subsets = [[0, 1], [1, 2], [2]]
            model_subsets = [
                self._bkd.asarray(s, dtype=int) for s in model_subsets
            ]
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
        npartition_samples = self._bkd.arange(
            2.0, 2 + est.nsubsets(), dtype=self._bkd.double_dtype()
        )
        allocation = _make_groupacv_allocation(est, npartition_samples)
        est.set_allocation(allocation)

        # Run Monte Carlo check
        self._check_sigma_matrix_of_estimator(est, ntrials, models)

    @parametrize(
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
        nmodels: int,
        ntrials: int,
        group_type: str,
        stat_name: str,
        qoi_idx: list,
    ):
        """Test sigma matrix via Monte Carlo validation."""
        np.random.seed(1)
        self._check_sigma_matrix(
            nmodels, ntrials, group_type, stat_name, qoi_idx
        )


class TestSigmaMatrixMonteCarloNumpy(TestSigmaMatrixMonteCarlo[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSigmaMatrixMonteCarloTorch(TestSigmaMatrixMonteCarlo[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGradientOptimization(Generic[Array], ParametrizedTestCase):
    """Test full optimization loop with GroupACVAllocationOptimizer.

    Validates that:
    1. GroupACV and MLBLUE produce equivalent results for single QoI
    2. Optimization completes successfully for various configurations
    3. Multi-output estimation works correctly

    This replicates the full optimization portion of legacy test_gradient_optimization.
    """

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _covariance_to_correlation(self, cov: Array) -> Array:
        """Convert covariance matrix to correlation matrix."""
        d = self._bkd.sqrt(self._bkd.diag(cov))
        return cov / self._bkd.outer(d, d)

    def _create_optimizer(self):
        """Create the chained optimizer for sample allocation."""
        from pyapprox.typing.optimization.minimize.chained.chained_optimizer import (
            ChainedOptimizer,
        )
        from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
            ScipyTrustConstrOptimizer,
        )
        from pyapprox.typing.optimization.minimize.scipy.diffevol import (
            ScipyDifferentialEvolutionOptimizer,
        )

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
        cov = self._bkd.array(
            np.random.normal(0, 1, (nmodels * nqoi, nmodels * nqoi))
        )
        cov = cov.T @ cov
        cov = self._covariance_to_correlation(cov)

        target_cost = 100.0
        costs = self._bkd.flip(
            self._bkd.logspace(-nmodels + 1, 0, nmodels)
        )

        from pyapprox.typing.statest.groupacv.allocation import (
            GroupACVAllocationOptimizer,
        )

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
        gest.set_allocation(gest_result)

        mlest_allocator = GroupACVAllocationOptimizer(
            mlest, optimizer=self._create_optimizer()
        )
        mlest_result = mlest_allocator.optimize(
            target_cost, min_nhf_samples, init_guess=iterate
        )
        mlest.set_allocation(mlest_result)

        # Check that both estimators produce identical covariance matrices
        # when using the same partition samples (GroupACV allocation)
        # This is the key test from legacy: given same samples, both should agree
        groupacv_cov = gest._covariance_from_npartition_samples(
            gest.npartition_samples()
        )
        mlblue_cov_at_groupacv = mlest._covariance_from_npartition_samples(
            gest.npartition_samples()
        )
        self._bkd.assert_allclose(
            groupacv_cov,
            mlblue_cov_at_groupacv,
        )  # default tolerance like legacy

        # Check that sample counts are valid (non-negative)
        self.assertTrue(
            float(self._bkd.min(gest.npartition_samples())) >= 0
        )
        self.assertTrue(
            float(self._bkd.min(mlest.npartition_samples())) >= 0
        )

        # Check cost constraint is satisfied
        gest_cost = gest._estimator_cost(gest.npartition_samples())
        mlest_cost = mlest._estimator_cost(mlest.npartition_samples())
        self.assertTrue(float(gest_cost) <= target_cost * 1.01)  # Allow 1% tolerance
        self.assertTrue(float(mlest_cost) <= target_cost * 1.01)

    @parametrize(
        "nmodels,min_nhf_samples,nqoi",
        [
            (2, 1, 1),
            (3, 1, 1),
            (2, 1, 2),
            (3, 1, 2),
        ],
    )
    def test_gradient_optimization(
        self, nmodels: int, min_nhf_samples: int, nqoi: int
    ):
        """Test full optimization loop with GroupACVAllocationOptimizer."""
        np.random.seed(1)
        self._check_gradient_optimization(nmodels, min_nhf_samples, nqoi)


class TestGradientOptimizationTorchOnly(
    TestGradientOptimization[torch.Tensor]
):
    """Torch-only test since optimization requires bkd.jacobian."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGroupACVAllocationOptimizer(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Tests for the new allocation optimizer."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
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

    def test_allocator_produces_valid_result(self):
        """Test that allocator returns valid GroupACVAllocationResult."""
        from pyapprox.typing.statest.groupacv.allocation import (
            GroupACVAllocationOptimizer,
        )

        est = self._create_estimator(3)
        allocator = GroupACVAllocationOptimizer(est)
        result = allocator.optimize(target_cost=100, min_nhf_samples=1)

        self.assertIsInstance(result, GroupACVAllocationResult)
        self.assertTrue(result.success)
        self.assertTrue(
            float(self._bkd.min(result.npartition_samples)) >= 0
        )

    def test_allocator_with_custom_optimizer(self):
        """Test that custom optimizer is used."""
        from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
            ScipyTrustConstrOptimizer,
        )
        from pyapprox.typing.statest.groupacv.allocation import (
            GroupACVAllocationOptimizer,
        )

        est = self._create_estimator(3)
        opt = ScipyTrustConstrOptimizer(gtol=1e-6, maxiter=500)
        allocator = GroupACVAllocationOptimizer(est, optimizer=opt)
        result = allocator.optimize(target_cost=100, min_nhf_samples=1)

        self.assertTrue(result.success)

    def test_allocator_respects_budget(self):
        """Test that allocator respects budget constraint."""
        from pyapprox.typing.statest.groupacv.allocation import (
            GroupACVAllocationOptimizer,
        )

        est = self._create_estimator(3)
        target_cost = 100.0
        allocator = GroupACVAllocationOptimizer(est)
        result = allocator.optimize(target_cost=target_cost, min_nhf_samples=1)

        actual_cost = est._estimator_cost(result.npartition_samples)
        self.assertLessEqual(
            float(self._bkd.to_numpy(actual_cost)), target_cost + 1e-6
        )

    def test_allocator_with_mlblue_objective(self):
        """Test allocator with MLBLUEObjective for analytical derivatives."""
        from pyapprox.typing.statest.groupacv.allocation import (
            GroupACVAllocationOptimizer,
        )

        np.random.seed(1)
        cov = self._bkd.array(np.random.normal(0, 1, (3, 3)))
        cov = cov.T @ cov
        costs = self._bkd.arange(3, 0, -1, dtype=self._bkd.double_dtype())

        stat = MultiOutputMean(1, self._bkd)
        stat.set_pilot_quantities(cov)
        est = MLBLUEEstimator(stat, costs)

        # Use MLBLUE's default objective (has analytical derivatives)
        obj = est.default_objective()
        allocator = GroupACVAllocationOptimizer(est, objective=obj)
        result = allocator.optimize(target_cost=100, min_nhf_samples=1)

        self.assertTrue(result.success)

    def test_allocator_set_allocation_integration(self):
        """Test that allocator result can be used with estimator setter."""
        from pyapprox.typing.statest.groupacv.allocation import (
            GroupACVAllocationOptimizer,
        )

        est = self._create_estimator(3)
        allocator = GroupACVAllocationOptimizer(est)
        result = allocator.optimize(target_cost=100, min_nhf_samples=1)

        # Use new setter API
        est.set_allocation(result)

        # Verify allocation is stored
        self._bkd.assert_allclose(
            est.npartition_samples(), result.npartition_samples
        )

        # Verify covariance can be computed
        cov = est.covariance()
        self.assertEqual(cov.shape[0], est._stat.nstats())
        self.assertEqual(cov.shape[1], est._stat.nstats())

    @parametrize(
        "nmodels,nqoi",
        [(2, 1), (3, 1), (2, 2)],
    )
    def test_allocator_various_configurations(self, nmodels: int, nqoi: int):
        """Test allocator with various model/qoi configurations."""
        from pyapprox.typing.statest.groupacv.allocation import (
            GroupACVAllocationOptimizer,
        )

        np.random.seed(1)
        est = self._create_estimator(nmodels, nqoi)
        allocator = GroupACVAllocationOptimizer(est)
        result = allocator.optimize(target_cost=100, min_nhf_samples=1)

        self.assertTrue(result.success)
        self.assertEqual(
            result.npartition_samples.shape[0], est.npartitions()
        )


class TestGroupACVAllocationOptimizerTorchOnly(
    TestGroupACVAllocationOptimizer[torch.Tensor]
):
    """Torch-only test since optimization requires bkd.jacobian."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGroupACVRecoversMLBLUE(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Test that GroupACV with IS partitions recovers MLBLUE results.

    This test validates that the new separation of allocation optimization
    allows us to verify:
    1. GroupACV with IS partitions and torch autograd optimization matches
       MLBLUE with analytical jacobian/hessian
    2. Both gradient-based approaches match MLBLUE SPD optimization

    This is an important validation because MLBLUE is a special case of
    GroupACV with independent sampling (IS) partitions.
    """

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(1)

    def _covariance_to_correlation(self, cov: Array) -> Array:
        """Convert covariance matrix to correlation matrix."""
        d = self._bkd.sqrt(self._bkd.diag(cov))
        return cov / self._bkd.outer(d, d)

    def _create_optimizer(self):
        """Create the chained optimizer for sample allocation."""
        from pyapprox.typing.optimization.minimize.chained.chained_optimizer import (
            ChainedOptimizer,
        )
        from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
            ScipyTrustConstrOptimizer,
        )
        from pyapprox.typing.optimization.minimize.scipy.diffevol import (
            ScipyDifferentialEvolutionOptimizer,
        )

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

    @parametrize(
        "nmodels",
        [(2,), (3,), (4,)],
    )
    def test_groupacv_autograd_recovers_mlblue_analytical(self, nmodels: int):
        """Test GroupACV with autograd jacobian matches MLBLUE analytical jacobian.

        This verifies that:
        1. GroupACV with IS partitions using torch autograd (default jacobian)
           produces the same allocation as MLBLUE with analytical jacobian
        2. The estimator covariances are identical for the same allocation
        """
        from pyapprox.typing.statest.groupacv.allocation import (
            GroupACVAllocationOptimizer,
        )

        np.random.seed(42)
        cov_size = nmodels
        cov = self._bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
        cov = cov.T @ cov
        cov = self._covariance_to_correlation(cov)

        target_cost = 100.0
        costs = self._bkd.flip(
            self._bkd.logspace(-nmodels + 1, 0, nmodels)
        )

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
        groupacv_est.set_allocation(groupacv_result)

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
        mlblue_est.set_allocation(mlblue_result)

        # Both should have same number of partitions (IS uses 2^nmodels - 1)
        self.assertEqual(
            groupacv_est.npartitions(), mlblue_est.npartitions()
        )

        # Covariances at same allocation should match
        groupacv_cov = groupacv_est._covariance_from_npartition_samples(
            groupacv_result.npartition_samples
        )
        mlblue_cov = mlblue_est._covariance_from_npartition_samples(
            groupacv_result.npartition_samples
        )
        self._bkd.assert_allclose(groupacv_cov, mlblue_cov, rtol=1e-10)

        # Objective values should be close (both minimize trace)
        self._bkd.assert_allclose(
            groupacv_result.objective_value,
            mlblue_result.objective_value,
            rtol=1e-2,  # Allow some tolerance due to optimization
        )

    @unittest.skipIf(not HAS_CVXPY, "cvxpy not installed")
    @parametrize(
        "nmodels",
        [(2,), (3,)],
    )
    def test_groupacv_autograd_recovers_mlblue_spd(self, nmodels: int):
        """Test GroupACV with autograd matches MLBLUE SPD optimization.

        This verifies that:
        1. GroupACV with IS partitions using gradient-based optimization
           produces similar results to MLBLUE SPD (convex) optimization
        2. SPD should find global optimum, gradient-based should be close
        """
        from pyapprox.typing.statest.groupacv.allocation import (
            GroupACVAllocationOptimizer,
        )

        np.random.seed(42)
        cov_size = nmodels
        cov = self._bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
        cov = cov.T @ cov
        cov = self._covariance_to_correlation(cov)

        target_cost = 100.0
        costs = self._bkd.flip(
            self._bkd.logspace(-nmodels + 1, 0, nmodels)
        )

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


class TestGroupACVRecoversMLBLUETorchOnly(
    TestGroupACVRecoversMLBLUE[torch.Tensor]
):
    """Torch-only test since autograd optimization requires torch."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
