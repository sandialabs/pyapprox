"""Standalone tests for ACV allocation matrices.

These tests verify allocation matrix construction for GRD, GMF, GIS
estimators with various recursion indices. Uses hardcoded expected
values from mathematical definitions.

Tests use typing array convention: (nqoi, nsamples) for outputs.
"""

from typing import List

import numpy as np
import pytest

from pyapprox.statest.acv.optimization import (
    _get_allocation_matrix_acvis,
    _get_allocation_matrix_acvrd,
    _get_allocation_matrix_gmf,
)
from pyapprox.statest.acv.variants import (
    GISEstimator,
    GMFEstimator,
    GRDEstimator,
)
from pyapprox.statest.statistics import (
    MultiOutputMean,
    _get_nsamples_intersect,
    _get_nsamples_subset,
)


class TestGRDAllocationMatrices:
    """Test GRD allocation matrix construction."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(1)

    @pytest.mark.parametrize(
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
        self, bkd, recursion_index: List[int], expected: List[List[float]]
    ) -> None:
        """Test GRD allocation matrix for given recursion index."""
        rec_idx = bkd.array(recursion_index, dtype=int)
        allocation_mat = _get_allocation_matrix_acvrd(rec_idx, bkd)
        expected_mat = bkd.array(expected)
        bkd.assert_allclose(allocation_mat, expected_mat, rtol=1e-12)

    def test_allocation_matrix_via_estimator(self, bkd) -> None:
        """Test allocation matrix via GRDEstimator matches direct computation."""
        nqoi = 1
        nmodels = 3
        recursion_index = bkd.array([0, 1], dtype=int)

        # Create covariance
        cov = bkd.eye(nmodels * nqoi)
        stat = MultiOutputMean(nqoi, bkd)
        stat.set_pilot_quantities(cov)
        costs = bkd.array([1.0, 0.5, 0.25])

        est = GRDEstimator(stat, costs, recursion_index=recursion_index)

        # Compare with direct computation
        expected_mat = _get_allocation_matrix_acvrd(recursion_index, bkd)
        bkd.assert_allclose(est._allocation_mat, expected_mat, rtol=1e-12)


class TestGMFAllocationMatrices:
    """Test GMF allocation matrix construction."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(1)

    @pytest.mark.parametrize(
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
        self, bkd, recursion_index: List[int], expected: List[List[float]]
    ) -> None:
        """Test GMF allocation matrix for given recursion index."""
        rec_idx = bkd.array(recursion_index, dtype=int)
        allocation_mat = _get_allocation_matrix_gmf(rec_idx, bkd)
        expected_mat = bkd.array(expected)
        bkd.assert_allclose(allocation_mat, expected_mat, rtol=1e-12)

    def test_allocation_matrix_via_estimator(self, bkd) -> None:
        """Test allocation matrix via GMFEstimator matches direct computation."""
        nqoi = 1
        nmodels = 3
        recursion_index = bkd.array([0, 0], dtype=int)

        cov = bkd.eye(nmodels * nqoi)
        stat = MultiOutputMean(nqoi, bkd)
        stat.set_pilot_quantities(cov)
        costs = bkd.array([1.0, 0.5, 0.25])

        est = GMFEstimator(stat, costs, recursion_index=recursion_index)

        expected_mat = _get_allocation_matrix_gmf(recursion_index, bkd)
        bkd.assert_allclose(est._allocation_mat, expected_mat, rtol=1e-12)


class TestGISAllocationMatrices:
    """Test GIS allocation matrix construction."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(1)

    @pytest.mark.parametrize(
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
        self, bkd, recursion_index: List[int], expected: List[List[float]]
    ) -> None:
        """Test GIS allocation matrix for given recursion index."""
        rec_idx = bkd.array(recursion_index, dtype=int)
        allocation_mat = _get_allocation_matrix_acvis(rec_idx, bkd)
        expected_mat = bkd.array(expected)
        bkd.assert_allclose(allocation_mat, expected_mat, rtol=1e-12)

    def test_allocation_matrix_via_estimator(self, bkd) -> None:
        """Test allocation matrix via GISEstimator matches direct computation."""
        nqoi = 1
        nmodels = 3
        recursion_index = bkd.array([0, 1], dtype=int)

        cov = bkd.eye(nmodels * nqoi)
        stat = MultiOutputMean(nqoi, bkd)
        stat.set_pilot_quantities(cov)
        costs = bkd.array([1.0, 0.5, 0.25])

        est = GISEstimator(stat, costs, recursion_index=recursion_index)

        expected_mat = _get_allocation_matrix_acvis(recursion_index, bkd)
        bkd.assert_allclose(est._allocation_mat, expected_mat, rtol=1e-12)


class TestNsamplesIntersectAndSubset:
    """Test _get_nsamples_intersect and _get_nsamples_subset with GRD matrices."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(1)

    def test_nsamples_intersect_grd_4models(self, bkd) -> None:
        """Test _get_nsamples_intersect with 4-model GRD allocation."""
        recursion_index = bkd.array([0, 1, 2], dtype=int)
        allocation_mat = _get_allocation_matrix_acvrd(recursion_index, bkd)

        npartition_samples = bkd.array([2.0, 2.0, 4.0, 4.0])
        nsamples_intersect = _get_nsamples_intersect(
            allocation_mat, npartition_samples, bkd
        )

        expected = bkd.array(
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
        bkd.assert_allclose(nsamples_intersect, expected, rtol=1e-12)

    def test_nsamples_subset_grd_4models(self, bkd) -> None:
        """Test _get_nsamples_subset with 4-model GRD allocation."""
        recursion_index = bkd.array([0, 1, 2], dtype=int)
        allocation_mat = _get_allocation_matrix_acvrd(recursion_index, bkd)

        npartition_samples = bkd.array([2.0, 2.0, 4.0, 4.0])
        nsamples_subset = _get_nsamples_subset(
            allocation_mat, npartition_samples, bkd
        )

        expected = bkd.array([0.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0])
        bkd.assert_allclose(nsamples_subset, expected, rtol=1e-12)
