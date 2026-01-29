"""Tests for PCEFunctionTrainCore."""

import unittest
from typing import Any, Generic, List

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.surrogates.affine.univariate import create_bases_1d
from pyapprox.typing.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.typing.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.typing.surrogates.affine.expansions import BasisExpansion
from pyapprox.typing.probability import UniformMarginal

from pyapprox.typing.surrogates.functiontrain.core import FunctionTrainCore
from pyapprox.typing.surrogates.functiontrain.pce_core import (
    PCEFunctionTrainCore,
)


class TestPCEFunctionTrainCore(Generic[Array], unittest.TestCase):
    """Base class for PCEFunctionTrainCore tests."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_univariate_pce(
        self, max_level: int, nqoi: int = 1
    ) -> BasisExpansion[Array]:
        """Create a univariate PCE expansion with Legendre basis."""
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def _create_core_with_coefficients(
        self,
        r_left: int,
        r_right: int,
        max_level: int,
        coefficients: List[List[Array]],
    ) -> FunctionTrainCore[Array]:
        """Create a FunctionTrain core with specified coefficients.

        Parameters
        ----------
        r_left : int
            Left rank.
        r_right : int
            Right rank.
        max_level : int
            Polynomial degree.
        coefficients : List[List[Array]]
            Coefficient arrays for each (i, j) position.
            Each should have shape (nterms, 1).
        """
        basisexps: List[List[BasisExpansion[Array]]] = []
        for ii in range(r_left):
            row: List[BasisExpansion[Array]] = []
            for jj in range(r_right):
                pce = self._create_univariate_pce(max_level)
                pce = pce.with_params(coefficients[ii][jj])
                row.append(pce)
            basisexps.append(row)
        return FunctionTrainCore(basisexps, self._bkd)

    def test_construction_validates_coefficients(self) -> None:
        """Test PCEFunctionTrainCore construction succeeds for valid core."""
        bkd = self._bkd
        r_left, r_right = 2, 3
        max_level = 2
        nterms = max_level + 1

        # Create coefficients
        coefficients = [
            [bkd.asarray(np.random.randn(nterms, 1)) for _ in range(r_right)]
            for _ in range(r_left)
        ]
        core = self._create_core_with_coefficients(
            r_left, r_right, max_level, coefficients
        )

        pce_core = PCEFunctionTrainCore(core)

        self.assertEqual(pce_core.nterms(), nterms)
        self.assertEqual(pce_core.ranks(), (r_left, r_right))

    def test_construction_rejects_multi_qoi(self) -> None:
        """Test construction fails for nqoi > 1."""
        bkd = self._bkd
        # Create core with nqoi=2
        basisexps = [[self._create_univariate_pce(2, nqoi=2)]]
        core = FunctionTrainCore(basisexps, bkd)

        with self.assertRaises(ValueError) as ctx:
            PCEFunctionTrainCore(core)
        self.assertIn("nqoi=2", str(ctx.exception))
        self.assertIn("only nqoi=1", str(ctx.exception))

    def test_construction_rejects_inconsistent_nterms(self) -> None:
        """Test construction fails for inconsistent nterms."""
        bkd = self._bkd
        # Create core with different nterms in each position
        pce_deg2 = self._create_univariate_pce(2)  # nterms=3
        pce_deg3 = self._create_univariate_pce(3)  # nterms=4

        basisexps = [[pce_deg2, pce_deg3]]
        core = FunctionTrainCore(basisexps, bkd)

        with self.assertRaises(ValueError) as ctx:
            PCEFunctionTrainCore(core)
        self.assertIn("inconsistent nterms", str(ctx.exception))

    def test_coefficient_matrix_extraction(self) -> None:
        """Test coefficient_matrix extracts correct values."""
        bkd = self._bkd
        r_left, r_right = 2, 2
        max_level = 2
        nterms = max_level + 1

        # Create known coefficients
        coefficients = [
            [
                bkd.asarray([[1.0], [2.0], [3.0]]),  # (0, 0)
                bkd.asarray([[4.0], [5.0], [6.0]]),  # (0, 1)
            ],
            [
                bkd.asarray([[7.0], [8.0], [9.0]]),  # (1, 0)
                bkd.asarray([[10.0], [11.0], [12.0]]),  # (1, 1)
            ],
        ]
        core = self._create_core_with_coefficients(
            r_left, r_right, max_level, coefficients
        )
        pce_core = PCEFunctionTrainCore(core)

        # Check Θ^{(0)} (constant terms)
        theta0 = pce_core.coefficient_matrix(0)
        expected0 = bkd.asarray([[1.0, 4.0], [7.0, 10.0]])
        bkd.assert_allclose(theta0, expected0, rtol=1e-12)

        # Check Θ^{(1)} (linear terms)
        theta1 = pce_core.coefficient_matrix(1)
        expected1 = bkd.asarray([[2.0, 5.0], [8.0, 11.0]])
        bkd.assert_allclose(theta1, expected1, rtol=1e-12)

        # Check Θ^{(2)} (quadratic terms)
        theta2 = pce_core.coefficient_matrix(2)
        expected2 = bkd.asarray([[3.0, 6.0], [9.0, 12.0]])
        bkd.assert_allclose(theta2, expected2, rtol=1e-12)

    def test_coefficient_matrix_bounds_checking(self) -> None:
        """Test coefficient_matrix raises IndexError for invalid basis_idx."""
        bkd = self._bkd
        max_level = 2
        nterms = max_level + 1
        coefficients = [[bkd.asarray(np.random.randn(nterms, 1))]]
        core = self._create_core_with_coefficients(1, 1, max_level, coefficients)
        pce_core = PCEFunctionTrainCore(core)

        with self.assertRaises(IndexError):
            pce_core.coefficient_matrix(-1)

        with self.assertRaises(IndexError):
            pce_core.coefficient_matrix(nterms)

    def test_coefficient_matrix_caching(self) -> None:
        """Test coefficient matrices are cached."""
        bkd = self._bkd
        max_level = 2
        nterms = max_level + 1
        coefficients = [[bkd.asarray(np.random.randn(nterms, 1))]]
        core = self._create_core_with_coefficients(1, 1, max_level, coefficients)
        pce_core = PCEFunctionTrainCore(core)

        theta0_first = pce_core.coefficient_matrix(0)
        theta0_second = pce_core.coefficient_matrix(0)

        # Same object (cached)
        self.assertIs(theta0_first, theta0_second)

    def test_expected_core(self) -> None:
        """Test expected_core returns Θ^{(0)}."""
        bkd = self._bkd
        max_level = 2
        nterms = max_level + 1
        coefficients = [[bkd.asarray([[1.5], [0.0], [0.0]])]]
        core = self._create_core_with_coefficients(1, 1, max_level, coefficients)
        pce_core = PCEFunctionTrainCore(core)

        expected = pce_core.expected_core()
        bkd.assert_allclose(expected, bkd.asarray([[1.5]]), rtol=1e-12)

    def test_kronecker_product_dimensions(self) -> None:
        """Test Kronecker product matrices have correct dimensions."""
        bkd = self._bkd
        r_left, r_right = 2, 3
        max_level = 2
        nterms = max_level + 1
        coefficients = [
            [bkd.asarray(np.random.randn(nterms, 1)) for _ in range(r_right)]
            for _ in range(r_left)
        ]
        core = self._create_core_with_coefficients(
            r_left, r_right, max_level, coefficients
        )
        pce_core = PCEFunctionTrainCore(core)

        # Expected dimensions: (r_left², r_right²)
        expected_shape = (r_left * r_left, r_right * r_right)

        M = pce_core.expected_kron_core()
        self.assertEqual(M.shape, expected_shape)

        delta_M = pce_core.delta_kron_core()
        self.assertEqual(delta_M.shape, expected_shape)

        M0 = pce_core.mean_kron_core()
        self.assertEqual(M0.shape, expected_shape)

    def test_kron_decomposition_identity(self) -> None:
        """Test M_k = M_k^{(0)} + ΔM_k."""
        bkd = self._bkd
        r_left, r_right = 2, 2
        max_level = 3
        nterms = max_level + 1
        coefficients = [
            [bkd.asarray(np.random.randn(nterms, 1)) for _ in range(r_right)]
            for _ in range(r_left)
        ]
        core = self._create_core_with_coefficients(
            r_left, r_right, max_level, coefficients
        )
        pce_core = PCEFunctionTrainCore(core)

        M = pce_core.expected_kron_core()
        M0 = pce_core.mean_kron_core()
        delta_M = pce_core.delta_kron_core()

        bkd.assert_allclose(M, M0 + delta_M, rtol=1e-12)

    def test_kronecker_caching(self) -> None:
        """Test Kronecker products are cached."""
        bkd = self._bkd
        max_level = 2
        nterms = max_level + 1
        coefficients = [[bkd.asarray(np.random.randn(nterms, 1))]]
        core = self._create_core_with_coefficients(1, 1, max_level, coefficients)
        pce_core = PCEFunctionTrainCore(core)

        # First calls
        M_first = pce_core.expected_kron_core()
        delta_first = pce_core.delta_kron_core()
        M0_first = pce_core.mean_kron_core()

        # Second calls
        M_second = pce_core.expected_kron_core()
        delta_second = pce_core.delta_kron_core()
        M0_second = pce_core.mean_kron_core()

        # Same objects (cached)
        self.assertIs(M_first, M_second)
        self.assertIs(delta_first, delta_second)
        self.assertIs(M0_first, M0_second)

    def test_rank_1_core(self) -> None:
        """Test with rank-1 core (simplest case)."""
        bkd = self._bkd
        max_level = 2
        nterms = max_level + 1
        # Rank 1x1 core: single scalar function
        coefficients = [[bkd.asarray([[1.0], [2.0], [3.0]])]]
        core = self._create_core_with_coefficients(1, 1, max_level, coefficients)
        pce_core = PCEFunctionTrainCore(core)

        # Θ^{(ℓ)} are 1x1 matrices
        for ell in range(nterms):
            theta = pce_core.coefficient_matrix(ell)
            self.assertEqual(theta.shape, (1, 1))

        # M_k is 1x1: M_k = sum_ell theta_ell^2
        M = pce_core.expected_kron_core()
        self.assertEqual(M.shape, (1, 1))
        expected_M = bkd.asarray([[1.0**2 + 2.0**2 + 3.0**2]])
        bkd.assert_allclose(M, expected_M, rtol=1e-12)

    def test_core_accessor(self) -> None:
        """Test core() returns underlying FunctionTrainCore."""
        bkd = self._bkd
        max_level = 2
        nterms = max_level + 1
        coefficients = [[bkd.asarray(np.random.randn(nterms, 1))]]
        core = self._create_core_with_coefficients(1, 1, max_level, coefficients)
        pce_core = PCEFunctionTrainCore(core)

        self.assertIs(pce_core.core(), core)


class TestPCEFunctionTrainCoreNumpy(TestPCEFunctionTrainCore[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPCEFunctionTrainCoreTorch(TestPCEFunctionTrainCore[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
