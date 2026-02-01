"""Tests for GroupACV and MLBLUE estimators.

This module provides dual-backend tests for:
- GroupACVEstimator
- MLBLUEEstimator
- GroupACV optimization objectives and constraints
"""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.statest.groupacv import (
    get_model_subsets,
    _get_allocation_matrix_is,
    _get_allocation_matrix_nested,
    _nest_subsets,
    GroupACVEstimator,
    MLBLUEEstimator,
    GroupACVTraceObjective,
    GroupACVLogDetObjective,
    MLBLUEObjective,
    GroupACVCostConstraint,
)
from pyapprox.typing.statest.statistics import MultiOutputMean


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

    def _create_estimator(self, nmodels, est_type="is"):
        """Helper to create a test estimator."""
        cov = self._bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        costs = self._bkd.arange(nmodels, 0, -1, dtype=self._bkd.double_dtype())

        stat = MultiOutputMean(1, self._bkd)
        stat.set_pilot_quantities(cov)
        est = GroupACVEstimator(stat, costs, est_type=est_type)
        return est

    def test_nsamples_per_model_is(self):
        """Test sample count computation for IS estimation."""
        nmodels = 3
        est = self._create_estimator(nmodels, est_type="is")

        npartition_samples = self._bkd.arange(
            2.0, 2 + est.nsubsets(), dtype=self._bkd.double_dtype()
        )

        # Check nsamples per model
        expected_nsamples = self._bkd.asarray([21.0, 23.0, 25.0])
        self._bkd.assert_allclose(
            est._compute_nsamples_per_model(npartition_samples),
            expected_nsamples,
        )

        # Check total cost
        expected_cost = self._bkd.asarray([21 * 3 + 23 * 2 + 25 * 1.0])
        self._bkd.assert_allclose(
            self._bkd.asarray([est._estimator_cost(npartition_samples)]),
            expected_cost,
        )

        # Check intersection samples (diagonal for IS)
        self._bkd.assert_allclose(
            est._nintersect_samples(npartition_samples),
            self._bkd.diag(npartition_samples),
        )

    def test_nsamples_per_model_nested(self):
        """Test sample count computation for nested estimation."""
        nmodels = 3
        np.random.seed(1)
        est = self._create_estimator(nmodels, est_type="nested")

        npartition_samples = self._bkd.arange(
            2.0, 2.0 + est.nsubsets(), dtype=self._bkd.double_dtype()
        )

        # Check nsamples per model (from legacy test)
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

        # Check intersection samples matrix (from legacy test)
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


class TestGroupACVEstimatorNumpy(TestGroupACVEstimator[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGroupACVEstimatorTorch(TestGroupACVEstimator[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGroupACVObjective(Generic[Array], unittest.TestCase):
    """Tests for GroupACV optimization objectives."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(1)

    def _create_estimator(self, nmodels):
        """Helper to create a test estimator."""
        cov = self._bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        costs = self._bkd.arange(nmodels, 0, -1, dtype=self._bkd.double_dtype())

        stat = MultiOutputMean(1, self._bkd)
        stat.set_pilot_quantities(cov)
        est = GroupACVEstimator(stat, costs)
        return est

    def test_trace_objective_shape(self):
        """Test trace objective returns correct shapes."""
        nmodels = 3
        est = self._create_estimator(nmodels)

        obj = GroupACVTraceObjective(self._bkd)
        obj.set_estimator(est)

        # Check nvars and nqoi
        self._bkd.assert_allclose(
            self._bkd.asarray([obj.nvars()]),
            self._bkd.asarray([est.npartitions()]),
        )
        self._bkd.assert_allclose(
            self._bkd.asarray([obj.nqoi()]),
            self._bkd.asarray([1]),
        )

        # Test evaluation
        npartition_samples = self._bkd.full(
            (est.npartitions(), 1), 10.0, dtype=self._bkd.double_dtype()
        )
        value = obj(npartition_samples)

        # Shape should be (1, 1)
        self._bkd.assert_allclose(
            self._bkd.asarray(list(value.shape)),
            self._bkd.asarray([1, 1]),
        )

    def test_logdet_objective_shape(self):
        """Test log-determinant objective returns correct shapes."""
        nmodels = 3
        est = self._create_estimator(nmodels)

        obj = GroupACVLogDetObjective(self._bkd)
        obj.set_estimator(est)

        # Test evaluation
        npartition_samples = self._bkd.full(
            (est.npartitions(), 1), 10.0, dtype=self._bkd.double_dtype()
        )
        value = obj(npartition_samples)

        # Shape should be (1, 1)
        self._bkd.assert_allclose(
            self._bkd.asarray(list(value.shape)),
            self._bkd.asarray([1, 1]),
        )


class TestGroupACVObjectiveNumpy(TestGroupACVObjective[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGroupACVObjectiveTorch(TestGroupACVObjective[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGroupACVConstraint(Generic[Array], unittest.TestCase):
    """Tests for GroupACV cost constraint."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(1)

    def _create_estimator(self, nmodels):
        """Helper to create a test estimator."""
        cov = self._bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        costs = self._bkd.arange(nmodels, 0, -1, dtype=self._bkd.double_dtype())

        stat = MultiOutputMean(1, self._bkd)
        stat.set_pilot_quantities(cov)
        est = GroupACVEstimator(stat, costs)
        return est

    def test_constraint_shape(self):
        """Test constraint returns correct shapes."""
        nmodels = 3
        est = self._create_estimator(nmodels)

        constraint = GroupACVCostConstraint(self._bkd)
        constraint.set_estimator(est)
        constraint.set_budget(target_cost=100.0, min_nhf_samples=1)

        # Check nqoi (2 constraints: cost and min HF)
        self._bkd.assert_allclose(
            self._bkd.asarray([constraint.nqoi()]),
            self._bkd.asarray([2]),
        )

        # Test evaluation
        npartition_samples = self._bkd.full(
            (est.npartitions(), 1), 1.0, dtype=self._bkd.double_dtype()
        )
        value = constraint(npartition_samples)

        # Shape should be (2, 1)
        self._bkd.assert_allclose(
            self._bkd.asarray(list(value.shape)),
            self._bkd.asarray([2, 1]),
        )

    def test_constraint_jacobian_shape(self):
        """Test constraint Jacobian returns correct shapes."""
        nmodels = 3
        est = self._create_estimator(nmodels)

        constraint = GroupACVCostConstraint(self._bkd)
        constraint.set_estimator(est)
        constraint.set_budget(target_cost=100.0, min_nhf_samples=1)

        npartition_samples = self._bkd.full(
            (est.npartitions(), 1), 1.0, dtype=self._bkd.double_dtype()
        )
        jac = constraint.jacobian(npartition_samples)

        # Shape should be (2, npartitions)
        self._bkd.assert_allclose(
            self._bkd.asarray(list(jac.shape)),
            self._bkd.asarray([2, est.npartitions()]),
        )


class TestGroupACVConstraintNumpy(TestGroupACVConstraint[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGroupACVConstraintTorch(TestGroupACVConstraint[torch.Tensor]):
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


class TestMLBLUEEstimatorNumpy(TestMLBLUEEstimator[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMLBLUEEstimatorTorch(TestMLBLUEEstimator[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestMLBLUEObjective(Generic[Array], unittest.TestCase):
    """Tests for MLBLUE-specific objective."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(1)

    def test_mlblue_objective_shape(self):
        """Test MLBLUE objective returns correct shapes."""
        nmodels = 2
        cov = self._bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        costs = self._bkd.arange(nmodels, 0, -1, dtype=self._bkd.double_dtype())

        stat = MultiOutputMean(1, self._bkd)
        stat.set_pilot_quantities(cov)
        est = MLBLUEEstimator(stat, costs)

        obj = MLBLUEObjective(self._bkd)
        obj.set_estimator(est)

        # Test evaluation
        npartition_samples = self._bkd.full(
            (est.npartitions(), 1), 10.0, dtype=self._bkd.double_dtype()
        )
        value = obj(npartition_samples)

        # Shape should be (1, 1)
        self._bkd.assert_allclose(
            self._bkd.asarray(list(value.shape)),
            self._bkd.asarray([1, 1]),
        )


class TestMLBLUEObjectiveNumpy(TestMLBLUEObjective[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMLBLUEObjectiveTorch(TestMLBLUEObjective[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
