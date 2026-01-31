"""Tests for allocation matrix utilities."""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.stats.allocation.matrices import (
    get_allocation_matrix_from_recursion,
    get_allocation_matrix_gis,
    get_allocation_matrix_grd,
    get_allocation_matrix_gmf,
    get_npartitions_from_nmodels,
    get_nsamples_per_model,
    validate_allocation_matrix,
    get_recursion_index_mfmc,
    get_recursion_index_mlmc,
    allocation_matrix_to_string,
)


class TestAllocationMatrixUtilities(Generic[Array], unittest.TestCase):
    """Tests for allocation matrix utilities."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self):
        self._bkd = self.bkd()

    def test_get_npartitions_from_nmodels(self):
        """Test partition count formula."""
        # 2 models: 2*(2-1) + 1 = 3 partitions
        self.assertEqual(get_npartitions_from_nmodels(2), 3)

        # 3 models: 2*(3-1) + 1 = 5 partitions
        self.assertEqual(get_npartitions_from_nmodels(3), 5)

        # 4 models: 2*(4-1) + 1 = 7 partitions
        self.assertEqual(get_npartitions_from_nmodels(4), 7)

    def test_allocation_matrix_mfmc_2_models(self):
        """Test MFMC allocation matrix with 2 models."""
        ridx = get_recursion_index_mfmc(2)
        A = get_allocation_matrix_from_recursion(2, ridx, self._bkd)

        # Expected: Model 0 in P0 and P1, Model 1 in P1 and P2
        expected = self._bkd.asarray([
            [1.0, 1.0, 0.0],  # HF: P0 (HF only), P1 (shared)
            [0.0, 1.0, 1.0],  # LF: P1 (shared), P2 (LF only)
        ])
        self._bkd.assert_allclose(A, expected)

    def test_allocation_matrix_mfmc_3_models(self):
        """Test MFMC allocation matrix with 3 models."""
        ridx = get_recursion_index_mfmc(3)
        A = get_allocation_matrix_from_recursion(3, ridx, self._bkd)

        # MFMC: All LF models coupled with HF
        # P0: HF only
        # P1: HF + LF1 shared
        # P2: LF1 only
        # P3: HF + LF2 shared
        # P4: LF2 only
        expected = self._bkd.asarray([
            [1.0, 1.0, 0.0, 1.0, 0.0],  # HF: P0, P1, P3
            [0.0, 1.0, 1.0, 0.0, 0.0],  # LF1: P1, P2
            [0.0, 0.0, 0.0, 1.0, 1.0],  # LF2: P3, P4
        ])
        self._bkd.assert_allclose(A, expected)

    def test_allocation_matrix_mlmc_3_models(self):
        """Test MLMC allocation matrix with 3 models."""
        ridx = get_recursion_index_mlmc(3)
        A = get_allocation_matrix_from_recursion(3, ridx, self._bkd)

        # MLMC: Successive coupling
        # ridx = [0, 1] means LF1 coupled with HF, LF2 coupled with LF1
        expected = self._bkd.asarray([
            [1.0, 1.0, 0.0, 0.0, 0.0],  # HF: P0, P1
            [0.0, 1.0, 1.0, 1.0, 0.0],  # LF1: P1, P2, P3
            [0.0, 0.0, 0.0, 1.0, 1.0],  # LF2: P3, P4
        ])
        self._bkd.assert_allclose(A, expected)

    def test_get_nsamples_per_model(self):
        """Test samples per model computation."""
        A = np.array([
            [1, 1, 0],
            [0, 1, 1],
        ])
        n_part = np.array([10, 20, 30])

        n_model = get_nsamples_per_model(A, n_part)

        # Model 0: P0 + P1 = 10 + 20 = 30
        # Model 1: P1 + P2 = 20 + 30 = 50
        expected = np.array([30, 50])
        self._bkd.assert_allclose(
            self._bkd.asarray(n_model), self._bkd.asarray(expected)
        )

    def test_validate_allocation_matrix_valid(self):
        """Test validation passes for valid matrix."""
        A = np.array([[1, 1, 0], [0, 1, 1]])
        # Should not raise
        validate_allocation_matrix(A)

    def test_validate_allocation_matrix_not_binary(self):
        """Test validation fails for non-binary matrix."""
        A = np.array([[1, 0.5, 0], [0, 1, 1]])
        with self.assertRaises(ValueError):
            validate_allocation_matrix(A)

    def test_validate_allocation_matrix_no_hf(self):
        """Test validation fails when HF has no partitions."""
        A = np.array([[0, 0, 0], [0, 1, 1]])
        with self.assertRaises(ValueError):
            validate_allocation_matrix(A)

    def test_recursion_index_mfmc(self):
        """Test MFMC recursion index."""
        ridx = get_recursion_index_mfmc(4)
        expected = np.array([0, 0, 0])
        self._bkd.assert_allclose(
            self._bkd.asarray(ridx), self._bkd.asarray(expected)
        )

    def test_recursion_index_mlmc(self):
        """Test MLMC recursion index."""
        ridx = get_recursion_index_mlmc(4)
        expected = np.array([0, 1, 2])
        self._bkd.assert_allclose(
            self._bkd.asarray(ridx), self._bkd.asarray(expected)
        )

    def test_allocation_matrix_from_recursion_validation(self):
        """Test recursion index validation."""
        # Wrong length
        with self.assertRaises(ValueError):
            get_allocation_matrix_from_recursion(
                3, np.array([0]), self._bkd
            )

        # Invalid value (too large)
        with self.assertRaises(ValueError):
            get_allocation_matrix_from_recursion(
                3, np.array([0, 2]), self._bkd  # ridx[1] = 2 > 1
            )

        # Invalid value (negative)
        with self.assertRaises(ValueError):
            get_allocation_matrix_from_recursion(
                3, np.array([-1, 0]), self._bkd
            )

    def test_allocation_matrix_to_string(self):
        """Test string representation of allocation matrix."""
        A = np.array([[1, 1, 0], [0, 1, 1]])
        s = allocation_matrix_to_string(A)

        self.assertIn("2 models", s)
        self.assertIn("3 partitions", s)
        self.assertIn("X", s)
        self.assertIn(".", s)


class TestAllocationMatrixUtilitiesNumpy(TestAllocationMatrixUtilities[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestAllocationMatrixUtilitiesTorch(TestAllocationMatrixUtilities[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestAllocationMatrixProperties(Generic[Array], unittest.TestCase):
    """Test mathematical properties of allocation matrices."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self):
        self._bkd = self.bkd()

    def test_hf_model_in_first_partition(self):
        """Verify HF model is always in partition 0."""
        for nmodels in range(2, 6):
            ridx = get_recursion_index_mfmc(nmodels)
            A = get_allocation_matrix_from_recursion(nmodels, ridx, self._bkd)
            A_np = self._bkd.to_numpy(A)

            # HF (row 0) should have 1 in partition 0
            self.assertEqual(A_np[0, 0], 1)

    def test_each_lf_has_shared_and_only_partitions(self):
        """Verify each LF model has both shared and only partitions."""
        for nmodels in range(2, 6):
            ridx = get_recursion_index_mfmc(nmodels)
            A = get_allocation_matrix_from_recursion(nmodels, ridx, self._bkd)
            A_np = self._bkd.to_numpy(A)

            for m in range(1, nmodels):
                # Each LF model should have exactly 2 partitions
                n_parts = np.sum(A_np[m, :])
                self.assertEqual(
                    n_parts, 2,
                    f"Model {m} has {n_parts} partitions, expected 2"
                )

    def test_sample_allocation_consistency(self):
        """Verify sample allocation is consistent."""
        nmodels = 3
        ridx = get_recursion_index_mfmc(nmodels)
        A = get_allocation_matrix_from_recursion(nmodels, ridx, self._bkd)
        A_np = self._bkd.to_numpy(A)

        # Set partition samples
        n_part = np.array([100, 50, 200, 30, 150])

        # Compute samples per model
        n_model = get_nsamples_per_model(A_np, n_part)

        # Total samples should match
        total_samples = np.sum(n_part)
        self.assertEqual(total_samples, np.sum(n_part))


class TestAllocationMatrixPropertiesNumpy(TestAllocationMatrixProperties[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestAllocationMatrixPropertiesTorch(TestAllocationMatrixProperties[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestGRDAllocationMatrices(Generic[Array], unittest.TestCase):
    """Tests for GRD allocation matrices matching legacy test cases.

    Replicates test_generalized_recursive_difference_allocation_matrices
    from pyapprox/multifidelity/tests/test_acv.py
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self):
        self._bkd = self.bkd()

    def test_grd_recursion_2_0(self):
        """Test GRD with recursion_index=[2, 0]."""
        ridx = np.array([2, 0])
        A = get_allocation_matrix_grd(3, ridx, self._bkd)

        expected = self._bkd.asarray([
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ])
        self._bkd.assert_allclose(A, expected)

    def test_grd_recursion_0_1(self):
        """Test GRD with recursion_index=[0, 1]."""
        ridx = np.array([0, 1])
        A = get_allocation_matrix_grd(3, ridx, self._bkd)

        expected = self._bkd.asarray([
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        self._bkd.assert_allclose(A, expected)

    def test_grd_recursion_0_0(self):
        """Test GRD with recursion_index=[0, 0]."""
        ridx = np.array([0, 0])
        A = get_allocation_matrix_grd(3, ridx, self._bkd)

        expected = self._bkd.asarray([
            [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        self._bkd.assert_allclose(A, expected)


class TestGRDAllocationMatricesNumpy(TestGRDAllocationMatrices[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestGRDAllocationMatricesTorch(TestGRDAllocationMatrices[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestGMFAllocationMatrices(Generic[Array], unittest.TestCase):
    """Tests for GMF allocation matrices matching legacy test cases.

    Replicates test_generalized_multifidelity_allocation_matrices
    from pyapprox/multifidelity/tests/test_acv.py
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self):
        self._bkd = self.bkd()

    def test_gmf_recursion_2_0(self):
        """Test GMF with recursion_index=[2, 0]."""
        ridx = np.array([2, 0])
        A = get_allocation_matrix_gmf(3, ridx, self._bkd)

        expected = self._bkd.asarray([
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ])
        self._bkd.assert_allclose(A, expected)

    def test_gmf_recursion_0_1(self):
        """Test GMF with recursion_index=[0, 1]."""
        ridx = np.array([0, 1])
        A = get_allocation_matrix_gmf(3, ridx, self._bkd)

        expected = self._bkd.asarray([
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        self._bkd.assert_allclose(A, expected)

    def test_gmf_recursion_0_0(self):
        """Test GMF with recursion_index=[0, 0]."""
        ridx = np.array([0, 0])
        A = get_allocation_matrix_gmf(3, ridx, self._bkd)

        expected = self._bkd.asarray([
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        self._bkd.assert_allclose(A, expected)


class TestGMFAllocationMatricesNumpy(TestGMFAllocationMatrices[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestGMFAllocationMatricesTorch(TestGMFAllocationMatrices[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestGISAllocationMatrices(Generic[Array], unittest.TestCase):
    """Tests for GIS allocation matrices matching legacy test cases.

    Replicates test_generalized_independent_samples_allocation_matrices
    from pyapprox/multifidelity/tests/test_acv.py
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self):
        self._bkd = self.bkd()

    def test_gis_recursion_2_0(self):
        """Test GIS with recursion_index=[2, 0]."""
        ridx = np.array([2, 0])
        A = get_allocation_matrix_gis(3, ridx, self._bkd)

        expected = self._bkd.asarray([
            [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        ])
        self._bkd.assert_allclose(A, expected)

    def test_gis_recursion_0_1(self):
        """Test GIS with recursion_index=[0, 1]."""
        ridx = np.array([0, 1])
        A = get_allocation_matrix_gis(3, ridx, self._bkd)

        expected = self._bkd.asarray([
            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        self._bkd.assert_allclose(A, expected)

    def test_gis_recursion_0_0(self):
        """Test GIS with recursion_index=[0, 0]."""
        ridx = np.array([0, 0])
        A = get_allocation_matrix_gis(3, ridx, self._bkd)

        expected = self._bkd.asarray([
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        self._bkd.assert_allclose(A, expected)


class TestGISAllocationMatricesNumpy(TestGISAllocationMatrices[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestGISAllocationMatricesTorch(TestGISAllocationMatrices[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


from pyapprox.typing.util.test_utils import load_tests  # noqa: F401, E402


if __name__ == "__main__":
    unittest.main()
