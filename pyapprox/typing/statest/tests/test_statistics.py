"""Unit tests for statistics module.

Tests core mathematical functionality with hardcoded matrices.
Integration tests with benchmarks will be added in Phase 8.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.statest.statistics import (
    MultiOutputMean,
    MultiOutputVariance,
    MultiOutputMeanAndVariance,
    _get_nsamples_intersect,
    _get_nsamples_subset,
    _get_V_from_covariance,
    _covariance_of_variance_estimator,
    block_2x2,
)


class TestHelperFunctions(Generic[Array], unittest.TestCase):
    """Test helper functions for statistics."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError()

    def setUp(self) -> None:
        np.random.seed(42)
        self._bkd = self.bkd()

    def test_block_2x2(self) -> None:
        """Test 2x2 block matrix construction."""
        a = self._bkd.array([[1.0, 2.0], [3.0, 4.0]])
        b = self._bkd.array([[5.0, 6.0], [7.0, 8.0]])
        c = self._bkd.array([[9.0, 10.0], [11.0, 12.0]])
        d = self._bkd.array([[13.0, 14.0], [15.0, 16.0]])
        result = block_2x2([[a, b], [c, d]], self._bkd)
        expected = self._bkd.array([
            [1.0, 2.0, 5.0, 6.0],
            [3.0, 4.0, 7.0, 8.0],
            [9.0, 10.0, 13.0, 14.0],
            [11.0, 12.0, 15.0, 16.0],
        ])
        self._bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_get_nsamples_subset_simple(self) -> None:
        """Test _get_nsamples_subset with simple allocation matrix."""
        # 2 models, allocation matrix for CV estimator
        allocation_mat = self._bkd.array([
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        npartition_samples = self._bkd.array([10.0, 5.0])
        result = _get_nsamples_subset(
            allocation_mat, npartition_samples, self._bkd
        )
        # Column 0: no 1s -> 0
        # Column 1: row 0 has 1 -> 10
        # Column 2: row 0 has 1 -> 10
        # Column 3: row 1 has 1 -> 5
        expected = self._bkd.array([0.0, 10.0, 10.0, 5.0])
        self._bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_get_nsamples_intersect_shape(self) -> None:
        """Test _get_nsamples_intersect returns correct shape."""
        nmodels = 3
        allocation_mat = self._bkd.zeros((nmodels, 2 * nmodels))
        for ii in range(nmodels):
            allocation_mat[ii, 2 * ii + 1] = 1.0
        npartition_samples = self._bkd.array([10.0, 5.0, 3.0])
        result = _get_nsamples_intersect(
            allocation_mat, npartition_samples, self._bkd
        )
        self._bkd.assert_allclose(
            self._bkd.asarray([result.shape[0], result.shape[1]]),
            self._bkd.asarray([2 * nmodels, 2 * nmodels])
        )

    def test_get_V_from_covariance_symmetric(self) -> None:
        """Test _get_V_from_covariance produces symmetric result."""
        nmodels = 2
        nqoi = 2
        cov = self._bkd.eye(nmodels * nqoi)
        V = _get_V_from_covariance(cov, nmodels, self._bkd)
        # V should be symmetric
        self._bkd.assert_allclose(V, V.T, rtol=1e-12)

    def test_covariance_of_variance_estimator(self) -> None:
        """Test _covariance_of_variance_estimator formula."""
        nqsq = 4
        W = self._bkd.eye(nqsq) * 2.0
        V = self._bkd.eye(nqsq) * 3.0
        nsamples = 10
        result = _covariance_of_variance_estimator(W, V, nsamples)
        # result = W/n + V/(n*(n-1)) = 2/10 + 3/90 = 0.2 + 0.0333... = 0.2333...
        expected_diag = 2.0 / 10 + 3.0 / (10 * 9)
        expected = self._bkd.eye(nqsq) * expected_diag
        self._bkd.assert_allclose(result, expected, rtol=1e-10)


class TestMultiOutputMean(Generic[Array], unittest.TestCase):
    """Test MultiOutputMean class."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError()

    def setUp(self) -> None:
        np.random.seed(42)
        self._bkd = self.bkd()

    def test_nstats(self) -> None:
        """Test nstats returns nqoi."""
        nqoi = 3
        stat = MultiOutputMean(nqoi, self._bkd)
        self._bkd.assert_allclose(
            self._bkd.asarray([stat.nstats()]),
            self._bkd.asarray([nqoi])
        )

    def test_sample_estimate(self) -> None:
        """Test sample_estimate computes mean correctly."""
        nqoi = 2
        nsamples = 100
        stat = MultiOutputMean(nqoi, self._bkd)
        # Use typing convention (nqoi, nsamples)
        values = self._bkd.asarray(np.random.randn(nqoi, nsamples))
        estimate = stat.sample_estimate(values)
        expected = self._bkd.mean(values, axis=1)
        self._bkd.assert_allclose(estimate, expected, rtol=1e-12)

    def test_set_pilot_quantities(self) -> None:
        """Test set_pilot_quantities initializes correctly."""
        nqoi = 2
        nmodels = 3
        stat = MultiOutputMean(nqoi, self._bkd)
        cov = self._bkd.eye(nmodels * nqoi)
        stat.set_pilot_quantities(cov)
        self._bkd.assert_allclose(
            self._bkd.asarray([stat._nmodels]),
            self._bkd.asarray([nmodels])
        )

    def test_high_fidelity_estimator_covariance(self) -> None:
        """Test high_fidelity_estimator_covariance computation."""
        nqoi = 2
        nmodels = 2
        stat = MultiOutputMean(nqoi, self._bkd)
        # Set diagonal covariance for simplicity
        cov = self._bkd.eye(nmodels * nqoi) * 4.0
        stat.set_pilot_quantities(cov)
        nhf_samples = 10
        result = stat.high_fidelity_estimator_covariance(nhf_samples)
        # Result should be cov[:nqoi, :nqoi] / nhf_samples
        expected = self._bkd.eye(nqoi) * (4.0 / nhf_samples)
        self._bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_min_nsamples(self) -> None:
        """Test min_nsamples returns 1."""
        stat = MultiOutputMean(2, self._bkd)
        self._bkd.assert_allclose(
            self._bkd.asarray([stat.min_nsamples()]),
            self._bkd.asarray([1])
        )

    def test_compute_pilot_quantities(self) -> None:
        """Test compute_pilot_quantities computes covariance."""
        nqoi = 2
        nsamples = 1000
        stat = MultiOutputMean(nqoi, self._bkd)
        # Create correlated data with typing convention (nqoi, nsamples)
        values1 = self._bkd.asarray(np.random.randn(nqoi, nsamples))
        values2 = values1 + self._bkd.asarray(
            np.random.randn(nqoi, nsamples) * 0.1
        )
        pilot_values = [values1, values2]
        (cov,) = stat.compute_pilot_quantities(pilot_values)
        # Check shape is (nmodels * nqoi, nmodels * nqoi)
        nmodels = 2
        self._bkd.assert_allclose(
            self._bkd.asarray([cov.shape[0], cov.shape[1]]),
            self._bkd.asarray([nmodels * nqoi, nmodels * nqoi])
        )
        # Check symmetry
        self._bkd.assert_allclose(cov, cov.T, rtol=1e-10)


class TestMultiOutputVariance(Generic[Array], unittest.TestCase):
    """Test MultiOutputVariance class."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError()

    def setUp(self) -> None:
        np.random.seed(42)
        self._bkd = self.bkd()

    def test_nstats_tril(self) -> None:
        """Test nstats returns nqoi*(nqoi+1)/2 for tril=True."""
        nqoi = 3
        nmodels = 2
        stat = MultiOutputVariance(nqoi, self._bkd, tril=True)
        cov = self._bkd.eye(nmodels * nqoi)
        W = self._bkd.eye(nmodels * nqoi**2)
        stat.set_pilot_quantities(cov, W)
        expected_nstats = nqoi * (nqoi + 1) // 2  # 3*4/2 = 6
        self._bkd.assert_allclose(
            self._bkd.asarray([stat.nstats()]),
            self._bkd.asarray([expected_nstats])
        )

    def test_min_nsamples(self) -> None:
        """Test min_nsamples returns 1."""
        stat = MultiOutputVariance(2, self._bkd)
        self._bkd.assert_allclose(
            self._bkd.asarray([stat.min_nsamples()]),
            self._bkd.asarray([1])
        )


class TestMultiOutputMeanAndVariance(Generic[Array], unittest.TestCase):
    """Test MultiOutputMeanAndVariance class."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError()

    def setUp(self) -> None:
        np.random.seed(42)
        self._bkd = self.bkd()

    def test_nstats(self) -> None:
        """Test nstats returns nqoi + nqoi*(nqoi+1)/2."""
        nqoi = 2
        nmodels = 2
        stat = MultiOutputMeanAndVariance(nqoi, self._bkd)
        cov = self._bkd.eye(nmodels * nqoi)
        V_shape = nmodels * nqoi**2
        W = self._bkd.eye(V_shape)
        B = self._bkd.zeros((nmodels * nqoi, V_shape))
        stat.set_pilot_quantities(cov, W, B)
        # nstats = nqoi + nqoi*(nqoi+1)/2 = 2 + 3 = 5
        expected_nstats = nqoi + nqoi * (nqoi + 1) // 2
        self._bkd.assert_allclose(
            self._bkd.asarray([stat.nstats()]),
            self._bkd.asarray([expected_nstats])
        )

    def test_min_nsamples(self) -> None:
        """Test min_nsamples returns 1."""
        stat = MultiOutputMeanAndVariance(2, self._bkd)
        self._bkd.assert_allclose(
            self._bkd.asarray([stat.min_nsamples()]),
            self._bkd.asarray([1])
        )

    def test_set_pilot_quantities_shape_check(self) -> None:
        """Test set_pilot_quantities validates W shape."""
        nqoi = 2
        nmodels = 2
        stat = MultiOutputMeanAndVariance(nqoi, self._bkd)
        cov = self._bkd.eye(nmodels * nqoi)
        # Wrong W shape
        W_wrong = self._bkd.eye(3)
        B = self._bkd.zeros((nmodels * nqoi, nmodels * nqoi**2))
        with self.assertRaises(ValueError):
            stat.set_pilot_quantities(cov, W_wrong, B)


# NumPy backend tests


class TestHelperFunctionsNumpy(TestHelperFunctions[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMultiOutputMeanNumpy(TestMultiOutputMean[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMultiOutputVarianceNumpy(TestMultiOutputVariance[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMultiOutputMeanAndVarianceNumpy(
    TestMultiOutputMeanAndVariance[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests


class TestHelperFunctionsTorch(TestHelperFunctions[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestMultiOutputMeanTorch(TestMultiOutputMean[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestMultiOutputVarianceTorch(TestMultiOutputVariance[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestMultiOutputMeanAndVarianceTorch(
    TestMultiOutputMeanAndVariance[torch.Tensor]
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
