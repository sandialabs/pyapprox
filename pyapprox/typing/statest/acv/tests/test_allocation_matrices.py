"""Standalone tests for ACV allocation matrices.

These tests verify allocation matrix construction for GRD, GMF, GIS
estimators with various recursion indices. Uses hardcoded expected
values from mathematical definitions.

Tests use typing array convention: (nqoi, nsamples) for outputs.
"""

import unittest
from typing import Any, Generic, List

import numpy as np
from numpy.typing import NDArray
import torch
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.statest.statistics import (
    MultiOutputMean,
    _get_nsamples_intersect,
    _get_nsamples_subset,
)
from pyapprox.typing.statest.acv.optimization import (
    _get_allocation_matrix_gmf,
    _get_allocation_matrix_acvis,
    _get_allocation_matrix_acvrd,
)
from pyapprox.typing.statest.acv.variants import (
    GMFEstimator,
    GISEstimator,
    GRDEstimator,
)


class TestGRDAllocationMatrices(Generic[Array], ParametrizedTestCase):
    """Test GRD allocation matrix construction."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError()

    def setUp(self) -> None:
        np.random.seed(1)
        self._bkd = self.bkd()

    @parametrize(
        "recursion_index,expected",
        [
            # recursion_index=[2, 0]
            (
                [2, 0],
                [
                    [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                ],
            ),
            # recursion_index=[0, 1]
            (
                [0, 1],
                [
                    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
            ),
            # recursion_index=[0, 0]
            (
                [0, 0],
                [
                    [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
            ),
        ],
        ids=["rec_2_0", "rec_0_1", "rec_0_0"],
    )
    def test_allocation_matrix(
        self, recursion_index: List[int], expected: List[List[float]]
    ) -> None:
        """Test GRD allocation matrix for given recursion index."""
        rec_idx = self._bkd.array(recursion_index, dtype=int)
        allocation_mat = _get_allocation_matrix_acvrd(rec_idx, self._bkd)
        expected_mat = self._bkd.array(expected)
        self._bkd.assert_allclose(allocation_mat, expected_mat, rtol=1e-12)

    def test_allocation_matrix_via_estimator(self) -> None:
        """Test allocation matrix via GRDEstimator matches direct computation."""
        nqoi = 1
        nmodels = 3
        recursion_index = self._bkd.array([0, 1], dtype=int)

        # Create covariance
        cov = self._bkd.eye(nmodels * nqoi)
        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(cov)
        costs = self._bkd.array([1.0, 0.5, 0.25])

        est = GRDEstimator(stat, costs, recursion_index=recursion_index)

        # Compare with direct computation
        expected_mat = _get_allocation_matrix_acvrd(recursion_index, self._bkd)
        self._bkd.assert_allclose(
            est._allocation_mat, expected_mat, rtol=1e-12
        )


class TestGMFAllocationMatrices(Generic[Array], ParametrizedTestCase):
    """Test GMF allocation matrix construction."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError()

    def setUp(self) -> None:
        np.random.seed(1)
        self._bkd = self.bkd()

    @parametrize(
        "recursion_index,expected",
        [
            # recursion_index=[2, 0]
            (
                [2, 0],
                [
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                ],
            ),
            # recursion_index=[0, 1]
            (
                [0, 1],
                [
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
            ),
            # recursion_index=[0, 0]
            (
                [0, 0],
                [
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
            ),
        ],
        ids=["rec_2_0", "rec_0_1", "rec_0_0"],
    )
    def test_allocation_matrix(
        self, recursion_index: List[int], expected: List[List[float]]
    ) -> None:
        """Test GMF allocation matrix for given recursion index."""
        rec_idx = self._bkd.array(recursion_index, dtype=int)
        allocation_mat = _get_allocation_matrix_gmf(rec_idx, self._bkd)
        expected_mat = self._bkd.array(expected)
        self._bkd.assert_allclose(allocation_mat, expected_mat, rtol=1e-12)

    def test_allocation_matrix_via_estimator(self) -> None:
        """Test allocation matrix via GMFEstimator matches direct computation."""
        nqoi = 1
        nmodels = 3
        recursion_index = self._bkd.array([0, 0], dtype=int)

        cov = self._bkd.eye(nmodels * nqoi)
        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(cov)
        costs = self._bkd.array([1.0, 0.5, 0.25])

        est = GMFEstimator(stat, costs, recursion_index=recursion_index)

        expected_mat = _get_allocation_matrix_gmf(recursion_index, self._bkd)
        self._bkd.assert_allclose(
            est._allocation_mat, expected_mat, rtol=1e-12
        )


class TestGISAllocationMatrices(Generic[Array], ParametrizedTestCase):
    """Test GIS allocation matrix construction."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError()

    def setUp(self) -> None:
        np.random.seed(1)
        self._bkd = self.bkd()

    @parametrize(
        "recursion_index,expected",
        [
            # recursion_index=[2, 0]
            (
                [2, 0],
                [
                    [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                ],
            ),
            # recursion_index=[0, 1]
            (
                [0, 1],
                [
                    [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
            ),
            # recursion_index=[0, 0]
            (
                [0, 0],
                [
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
            ),
        ],
        ids=["rec_2_0", "rec_0_1", "rec_0_0"],
    )
    def test_allocation_matrix(
        self, recursion_index: List[int], expected: List[List[float]]
    ) -> None:
        """Test GIS allocation matrix for given recursion index."""
        rec_idx = self._bkd.array(recursion_index, dtype=int)
        allocation_mat = _get_allocation_matrix_acvis(rec_idx, self._bkd)
        expected_mat = self._bkd.array(expected)
        self._bkd.assert_allclose(allocation_mat, expected_mat, rtol=1e-12)

    def test_allocation_matrix_via_estimator(self) -> None:
        """Test allocation matrix via GISEstimator matches direct computation."""
        nqoi = 1
        nmodels = 3
        recursion_index = self._bkd.array([0, 1], dtype=int)

        cov = self._bkd.eye(nmodels * nqoi)
        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(cov)
        costs = self._bkd.array([1.0, 0.5, 0.25])

        est = GISEstimator(stat, costs, recursion_index=recursion_index)

        expected_mat = _get_allocation_matrix_acvis(recursion_index, self._bkd)
        self._bkd.assert_allclose(
            est._allocation_mat, expected_mat, rtol=1e-12
        )


class TestNsamplesIntersectAndSubset(Generic[Array], unittest.TestCase):
    """Test _get_nsamples_intersect and _get_nsamples_subset with GRD matrices."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError()

    def setUp(self) -> None:
        np.random.seed(1)
        self._bkd = self.bkd()

    def test_nsamples_intersect_grd_4models(self) -> None:
        """Test _get_nsamples_intersect with 4-model GRD allocation."""
        nmodels = 4
        recursion_index = self._bkd.array([0, 1, 2], dtype=int)
        allocation_mat = _get_allocation_matrix_acvrd(recursion_index, self._bkd)

        npartition_samples = self._bkd.array([2.0, 2.0, 4.0, 4.0])
        nsamples_intersect = _get_nsamples_intersect(
            allocation_mat, npartition_samples, self._bkd
        )

        expected = self._bkd.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
            ]
        )
        self._bkd.assert_allclose(nsamples_intersect, expected, rtol=1e-12)

    def test_nsamples_subset_grd_4models(self) -> None:
        """Test _get_nsamples_subset with 4-model GRD allocation."""
        nmodels = 4
        recursion_index = self._bkd.array([0, 1, 2], dtype=int)
        allocation_mat = _get_allocation_matrix_acvrd(recursion_index, self._bkd)

        npartition_samples = self._bkd.array([2.0, 2.0, 4.0, 4.0])
        nsamples_subset = _get_nsamples_subset(
            allocation_mat, npartition_samples, self._bkd
        )

        expected = self._bkd.array([0.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0])
        self._bkd.assert_allclose(nsamples_subset, expected, rtol=1e-12)


# NumPy backend tests


class TestGRDAllocationMatricesNumpy(TestGRDAllocationMatrices[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGMFAllocationMatricesNumpy(TestGMFAllocationMatrices[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGISAllocationMatricesNumpy(TestGISAllocationMatrices[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestNsamplesIntersectAndSubsetNumpy(
    TestNsamplesIntersectAndSubset[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests


class TestGRDAllocationMatricesTorch(TestGRDAllocationMatrices[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGMFAllocationMatricesTorch(TestGMFAllocationMatrices[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGISAllocationMatricesTorch(TestGISAllocationMatrices[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestNsamplesIntersectAndSubsetTorch(
    TestNsamplesIntersectAndSubset[torch.Tensor]
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
