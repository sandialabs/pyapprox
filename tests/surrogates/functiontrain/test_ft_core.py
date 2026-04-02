"""Tests for FunctionTrainCore."""

import numpy as np
import pytest

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.functiontrain.core import FunctionTrainCore


class TestFunctionTrainCore:
    """Base class for FunctionTrainCore tests."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_univariate_expansion(self, bkd, max_level, nqoi=1):
        """Create a univariate polynomial expansion."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_core_creation(self, bkd) -> None:
        """Test basic core creation."""
        # Create 2x3 grid of expansions
        r_left, r_right = 2, 3
        basisexps = []
        for _ in range(r_left):
            row = [self._create_univariate_expansion(bkd, 2) for _ in range(r_right)]
            basisexps.append(row)

        core = FunctionTrainCore(basisexps, bkd)

        assert core.ranks() == (r_left, r_right)
        assert core.nqoi() == 1

    def test_core_ranks(self, bkd) -> None:
        """Test various rank configurations."""
        # (1, 2) - first core shape
        basisexps_1_2 = [[self._create_univariate_expansion(bkd, 2) for _ in range(2)]]
        core_1_2 = FunctionTrainCore(basisexps_1_2, bkd)
        assert core_1_2.ranks() == (1, 2)

        # (2, 1) - last core shape
        basisexps_2_1 = [
            [self._create_univariate_expansion(bkd, 2)],
            [self._create_univariate_expansion(bkd, 2)],
        ]
        core_2_1 = FunctionTrainCore(basisexps_2_1, bkd)
        assert core_2_1.ranks() == (2, 1)

        # (2, 2) - middle core shape
        basisexps_2_2 = [
            [self._create_univariate_expansion(bkd, 2) for _ in range(2)],
            [self._create_univariate_expansion(bkd, 2) for _ in range(2)],
        ]
        core_2_2 = FunctionTrainCore(basisexps_2_2, bkd)
        assert core_2_2.ranks() == (2, 2)

    def test_core_call(self, bkd) -> None:
        """Test core evaluation."""
        # Create (2, 3) core
        basisexps = [
            [self._create_univariate_expansion(bkd, 2) for _ in range(3)],
            [self._create_univariate_expansion(bkd, 2) for _ in range(3)],
        ]
        core = FunctionTrainCore(basisexps, bkd)

        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 5)))
        result = core(samples)

        # Shape should be (r_left, r_right, nsamples, nqoi)
        assert result.shape == (2, 3, 5, 1)

    def test_nparams(self, bkd) -> None:
        """Test parameter counting."""
        max_level = 2
        nterms = max_level + 1  # For univariate: 0, 1, 2

        # (2, 3) core with nqoi=1
        basisexps = [
            [self._create_univariate_expansion(bkd, max_level) for _ in range(3)],
            [self._create_univariate_expansion(bkd, max_level) for _ in range(3)],
        ]
        core = FunctionTrainCore(basisexps, bkd)

        expected = 2 * 3 * nterms * 1  # r_left * r_right * nterms * nqoi
        assert core.nparams() == expected

    def test_with_params_roundtrip(self, bkd) -> None:
        """Test flatten/unflatten roundtrip."""
        basisexps = [
            [self._create_univariate_expansion(bkd, 2) for _ in range(2)],
            [self._create_univariate_expansion(bkd, 2) for _ in range(2)],
        ]
        core = FunctionTrainCore(basisexps, bkd)

        # Set random parameters
        nparams = core.nparams()
        random_params = bkd.asarray(np.random.randn(nparams))

        # Apply and flatten
        new_core = core.with_params(random_params)
        recovered_params = new_core._flatten_params()

        bkd.assert_allclose(recovered_params, random_params, rtol=1e-12)

    def test_with_params_immutability(self, bkd) -> None:
        """Test that with_params doesn't modify original."""
        basisexps = [
            [self._create_univariate_expansion(bkd, 2) for _ in range(2)],
        ]
        core = FunctionTrainCore(basisexps, bkd)

        # Get original params
        original_params = bkd.copy(core._flatten_params())

        # Create new core with different params
        new_params = bkd.asarray(np.random.randn(core.nparams()))
        _ = core.with_params(new_params)

        # Original should be unchanged
        bkd.assert_allclose(core._flatten_params(), original_params, rtol=1e-12)

    def test_basis_matrix(self, bkd) -> None:
        """Test basis matrix extraction."""
        basisexps = [
            [self._create_univariate_expansion(bkd, 2) for _ in range(2)],
            [self._create_univariate_expansion(bkd, 2) for _ in range(2)],
        ]
        core = FunctionTrainCore(basisexps, bkd)

        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 5)))
        basis_mat = core.basis_matrix(samples, 0, 1)

        # Shape: (nsamples, nterms)
        assert basis_mat.shape[0] == 5
        assert basis_mat.shape[1] == 3  # max_level + 1

    def test_get_basisexp(self, bkd) -> None:
        """Test get_basisexp returns correct expansion."""
        # Create distinct expansions for each position
        basisexps = [
            [self._create_univariate_expansion(bkd, 2) for _ in range(3)],
            [self._create_univariate_expansion(bkd, 2) for _ in range(3)],
        ]
        core = FunctionTrainCore(basisexps, bkd)

        # Verify each position returns the correct expansion
        for ii in range(2):
            for jj in range(3):
                retrieved = core.get_basisexp(ii, jj)
                # Check it's the same object reference
                assert retrieved is basisexps[ii][jj]

    def test_get_basisexp_bounds_checking(self, bkd) -> None:
        """Test get_basisexp raises IndexError for out-of-bounds indices."""
        basisexps = [
            [self._create_univariate_expansion(bkd, 2) for _ in range(3)],
            [self._create_univariate_expansion(bkd, 2) for _ in range(3)],
        ]
        core = FunctionTrainCore(basisexps, bkd)

        # Test left rank out of bounds
        with pytest.raises(IndexError) as ctx:
            core.get_basisexp(2, 0)
        assert "Left rank index" in str(ctx.value)

        with pytest.raises(IndexError) as ctx:
            core.get_basisexp(-1, 0)
        assert "Left rank index" in str(ctx.value)

        # Test right rank out of bounds
        with pytest.raises(IndexError) as ctx:
            core.get_basisexp(0, 3)
        assert "Right rank index" in str(ctx.value)

        with pytest.raises(IndexError) as ctx:
            core.get_basisexp(0, -1)
        assert "Right rank index" in str(ctx.value)

    def test_get_basisexp_interface(self, bkd) -> None:
        """Test get_basisexp returns expansion with expected interface."""
        basisexps = [[self._create_univariate_expansion(bkd, 3, nqoi=2)]]
        core = FunctionTrainCore(basisexps, bkd)

        bexp = core.get_basisexp(0, 0)

        # Verify expected interface methods exist and work
        assert hasattr(bexp, "get_coefficients")
        assert hasattr(bexp, "nterms")
        assert hasattr(bexp, "nqoi")

        # Check values
        assert bexp.nterms() == 4  # max_level + 1 = 3 + 1
        assert bexp.nqoi() == 2

        # get_coefficients should return shape (nterms, nqoi)
        coef = bexp.get_coefficients()
        assert coef.shape == (4, 2)
