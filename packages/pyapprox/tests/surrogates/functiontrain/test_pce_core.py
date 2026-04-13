"""Tests for PCEFunctionTrainCore."""

import numpy as np
import pytest

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.functiontrain.core import FunctionTrainCore
from pyapprox.surrogates.functiontrain.pce_core import (
    PCEFunctionTrainCore,
)


class TestPCEFunctionTrainCore:
    """Base class for PCEFunctionTrainCore tests."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_univariate_pce(self, bkd, max_level, nqoi=1):
        """Create a univariate PCE expansion with Legendre basis."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def _create_core_with_coefficients(
        self, bkd, r_left, r_right, max_level, coefficients
    ):
        """Create a FunctionTrain core with specified coefficients."""
        basisexps = []
        for ii in range(r_left):
            row = []
            for jj in range(r_right):
                pce = self._create_univariate_pce(bkd, max_level)
                pce = pce.with_params(coefficients[ii][jj])
                row.append(pce)
            basisexps.append(row)
        return FunctionTrainCore(basisexps, bkd)

    def test_construction_validates_coefficients(self, bkd) -> None:
        """Test PCEFunctionTrainCore construction succeeds for valid core."""
        r_left, r_right = 2, 3
        max_level = 2
        nterms = max_level + 1

        # Create coefficients
        coefficients = [
            [bkd.asarray(np.random.randn(nterms, 1)) for _ in range(r_right)]
            for _ in range(r_left)
        ]
        core = self._create_core_with_coefficients(
            bkd, r_left, r_right, max_level, coefficients
        )

        pce_core = PCEFunctionTrainCore(core)

        assert pce_core.nterms() == nterms
        assert pce_core.ranks() == (r_left, r_right)

    def test_construction_rejects_multi_qoi(self, bkd) -> None:
        """Test construction fails for nqoi > 1."""
        # Create core with nqoi=2
        basisexps = [[self._create_univariate_pce(bkd, 2, nqoi=2)]]
        core = FunctionTrainCore(basisexps, bkd)

        with pytest.raises(ValueError) as ctx:
            PCEFunctionTrainCore(core)
        assert "nqoi=2" in str(ctx.value)
        assert "only nqoi=1" in str(ctx.value)

    def test_construction_rejects_inconsistent_nterms(self, bkd) -> None:
        """Test construction fails for inconsistent nterms."""
        # Create core with different nterms in each position
        pce_deg2 = self._create_univariate_pce(bkd, 2)  # nterms=3
        pce_deg3 = self._create_univariate_pce(bkd, 3)  # nterms=4

        basisexps = [[pce_deg2, pce_deg3]]
        core = FunctionTrainCore(basisexps, bkd)

        with pytest.raises(ValueError) as ctx:
            PCEFunctionTrainCore(core)
        assert "inconsistent nterms" in str(ctx.value)

    def test_coefficient_matrix_extraction(self, bkd) -> None:
        """Test coefficient_matrix extracts correct values."""
        r_left, r_right = 2, 2
        max_level = 2

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
            bkd, r_left, r_right, max_level, coefficients
        )
        pce_core = PCEFunctionTrainCore(core)

        # Check Theta^{(0)} (constant terms)
        theta0 = pce_core.coefficient_matrix(0)
        expected0 = bkd.asarray([[1.0, 4.0], [7.0, 10.0]])
        bkd.assert_allclose(theta0, expected0, rtol=1e-12)

        # Check Theta^{(1)} (linear terms)
        theta1 = pce_core.coefficient_matrix(1)
        expected1 = bkd.asarray([[2.0, 5.0], [8.0, 11.0]])
        bkd.assert_allclose(theta1, expected1, rtol=1e-12)

        # Check Theta^{(2)} (quadratic terms)
        theta2 = pce_core.coefficient_matrix(2)
        expected2 = bkd.asarray([[3.0, 6.0], [9.0, 12.0]])
        bkd.assert_allclose(theta2, expected2, rtol=1e-12)

    def test_coefficient_matrix_bounds_checking(self, bkd) -> None:
        """Test coefficient_matrix raises IndexError for invalid basis_idx."""
        max_level = 2
        nterms = max_level + 1
        coefficients = [[bkd.asarray(np.random.randn(nterms, 1))]]
        core = self._create_core_with_coefficients(bkd, 1, 1, max_level, coefficients)
        pce_core = PCEFunctionTrainCore(core)

        with pytest.raises(IndexError):
            pce_core.coefficient_matrix(-1)

        with pytest.raises(IndexError):
            pce_core.coefficient_matrix(nterms)

    def test_coefficient_matrix_caching(self, bkd) -> None:
        """Test coefficient matrices are cached."""
        max_level = 2
        nterms = max_level + 1
        coefficients = [[bkd.asarray(np.random.randn(nterms, 1))]]
        core = self._create_core_with_coefficients(bkd, 1, 1, max_level, coefficients)
        pce_core = PCEFunctionTrainCore(core)

        theta0_first = pce_core.coefficient_matrix(0)
        theta0_second = pce_core.coefficient_matrix(0)

        # Same object (cached)
        assert theta0_first is theta0_second

    def test_expected_core(self, bkd) -> None:
        """Test expected_core returns Theta^{(0)}."""
        max_level = 2
        coefficients = [[bkd.asarray([[1.5], [0.0], [0.0]])]]
        core = self._create_core_with_coefficients(bkd, 1, 1, max_level, coefficients)
        pce_core = PCEFunctionTrainCore(core)

        expected = pce_core.expected_core()
        bkd.assert_allclose(expected, bkd.asarray([[1.5]]), rtol=1e-12)

    def test_kronecker_product_dimensions(self, bkd) -> None:
        """Test Kronecker product matrices have correct dimensions."""
        r_left, r_right = 2, 3
        max_level = 2
        nterms = max_level + 1
        coefficients = [
            [bkd.asarray(np.random.randn(nterms, 1)) for _ in range(r_right)]
            for _ in range(r_left)
        ]
        core = self._create_core_with_coefficients(
            bkd, r_left, r_right, max_level, coefficients
        )
        pce_core = PCEFunctionTrainCore(core)

        # Expected dimensions: (r_left^2, r_right^2)
        expected_shape = (r_left * r_left, r_right * r_right)

        M = pce_core.expected_kron_core()
        assert M.shape == expected_shape

        delta_M = pce_core.delta_kron_core()
        assert delta_M.shape == expected_shape

        M0 = pce_core.mean_kron_core()
        assert M0.shape == expected_shape

    def test_kron_decomposition_identity(self, bkd) -> None:
        """Test M_k = M_k^{(0)} + DeltaM_k."""
        r_left, r_right = 2, 2
        max_level = 3
        nterms = max_level + 1
        coefficients = [
            [bkd.asarray(np.random.randn(nterms, 1)) for _ in range(r_right)]
            for _ in range(r_left)
        ]
        core = self._create_core_with_coefficients(
            bkd, r_left, r_right, max_level, coefficients
        )
        pce_core = PCEFunctionTrainCore(core)

        M = pce_core.expected_kron_core()
        M0 = pce_core.mean_kron_core()
        delta_M = pce_core.delta_kron_core()

        bkd.assert_allclose(M, M0 + delta_M, rtol=1e-12)

    def test_kronecker_caching(self, bkd) -> None:
        """Test Kronecker products are cached."""
        max_level = 2
        nterms = max_level + 1
        coefficients = [[bkd.asarray(np.random.randn(nterms, 1))]]
        core = self._create_core_with_coefficients(bkd, 1, 1, max_level, coefficients)
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
        assert M_first is M_second
        assert delta_first is delta_second
        assert M0_first is M0_second

    def test_rank_1_core(self, bkd) -> None:
        """Test with rank-1 core (simplest case)."""
        max_level = 2
        nterms = max_level + 1
        # Rank 1x1 core: single scalar function
        coefficients = [[bkd.asarray([[1.0], [2.0], [3.0]])]]
        core = self._create_core_with_coefficients(bkd, 1, 1, max_level, coefficients)
        pce_core = PCEFunctionTrainCore(core)

        # Theta^{(l)} are 1x1 matrices
        for ell in range(nterms):
            theta = pce_core.coefficient_matrix(ell)
            assert theta.shape == (1, 1)

        # M_k is 1x1: M_k = sum_ell theta_ell^2
        M = pce_core.expected_kron_core()
        assert M.shape == (1, 1)
        expected_M = bkd.asarray([[1.0**2 + 2.0**2 + 3.0**2]])
        bkd.assert_allclose(M, expected_M, rtol=1e-12)

    def test_core_accessor(self, bkd) -> None:
        """Test core() returns underlying FunctionTrainCore."""
        max_level = 2
        nterms = max_level + 1
        coefficients = [[bkd.asarray(np.random.randn(nterms, 1))]]
        core = self._create_core_with_coefficients(bkd, 1, 1, max_level, coefficients)
        pce_core = PCEFunctionTrainCore(core)

        assert pce_core.core() is core
