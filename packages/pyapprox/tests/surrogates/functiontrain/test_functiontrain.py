"""Tests for FunctionTrain."""

import numpy as np
import pytest

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.functiontrain.additive import (
    ConstantExpansion,
    create_additive_functiontrain,
)
from pyapprox.surrogates.functiontrain.core import FunctionTrainCore
from pyapprox.surrogates.functiontrain.functiontrain import FunctionTrain


class TestFunctionTrain:
    """Base class for FunctionTrain tests."""

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

    def _create_simple_functiontrain(self, bkd, nvars=3, max_level=2, nqoi=1):
        """Create a simple additive FunctionTrain for testing."""
        univariate_bases = [
            self._create_univariate_expansion(bkd, max_level, nqoi)
            for _ in range(nvars)
        ]
        return create_additive_functiontrain(univariate_bases, bkd, nqoi)

    def test_functiontrain_creation(self, bkd) -> None:
        """Test basic FunctionTrain creation."""
        ft = self._create_simple_functiontrain(bkd, nvars=3)

        assert ft.nvars() == 3
        assert ft.nqoi() == 1
        assert len(ft.cores()) == 3

    def test_functiontrain_ranks(self, bkd) -> None:
        """Test that additive FunctionTrain has correct rank structure."""
        ft = self._create_simple_functiontrain(bkd, nvars=4)

        # Additive structure: (1,2), (2,2), (2,2), (2,1)
        cores = ft.cores()
        assert cores[0].ranks() == (1, 2)
        assert cores[1].ranks() == (2, 2)
        assert cores[2].ranks() == (2, 2)
        assert cores[3].ranks() == (2, 1)

    def test_functiontrain_call(self, bkd) -> None:
        """Test FunctionTrain evaluation."""
        ft = self._create_simple_functiontrain(bkd, nvars=3)

        samples = bkd.asarray(np.random.uniform(-1, 1, (3, 10)))
        result = ft(samples)

        # Shape: (nqoi, nsamples)
        assert result.shape == (1, 10)

    def test_nparams_correct(self, bkd) -> None:
        """Test that nparams matches sum of core params."""
        ft = self._create_simple_functiontrain(bkd, nvars=3, max_level=2)

        total_from_cores = sum(core.nparams() for core in ft.cores())
        assert ft.nparams() == total_from_cores

    def test_with_params_roundtrip(self, bkd) -> None:
        """Test flatten/unflatten is exact inverse."""
        ft = self._create_simple_functiontrain(bkd, nvars=3, max_level=2)

        # Set random parameters
        nparams = ft.nparams()
        random_params = bkd.asarray(np.random.randn(nparams))

        # Apply and flatten
        new_ft = ft.with_params(random_params)
        recovered_params = new_ft._flatten_params()

        bkd.assert_allclose(recovered_params, random_params, rtol=1e-12)

    def test_with_params_immutability(self, bkd) -> None:
        """Test that with_params doesn't modify original."""
        ft = self._create_simple_functiontrain(bkd, nvars=3)

        # Get original params
        original_params = bkd.copy(ft._flatten_params())

        # Create new FT with different params
        new_params = bkd.asarray(np.random.randn(ft.nparams()))
        _ = ft.with_params(new_params)

        # Original should be unchanged
        bkd.assert_allclose(ft._flatten_params(), original_params, rtol=1e-12)

    def test_with_cores(self, bkd) -> None:
        """Test with_cores creates new FunctionTrain."""
        ft = self._create_simple_functiontrain(bkd, nvars=3)

        original_cores = ft.cores()
        new_ft = ft.with_cores(original_cores)

        # Should be same structure
        assert new_ft.nvars() == ft.nvars()
        assert new_ft.nqoi() == ft.nqoi()

    def test_additive_structure_sums_functions(self, bkd) -> None:
        """Test that additive FunctionTrain computes sum of functions."""
        # Create univariate expansions with known coefficients
        max_level = 1  # Linear polynomials
        univariate_bases = []
        for _ in range(3):
            exp = self._create_univariate_expansion(bkd, max_level)
            # Set coefficients to [1, 1] so f_i(x_i) = 1 + sqrt(3)*x_i
            # (using Legendre normalization)
            exp.set_coefficients(bkd.asarray([[1.0], [1.0]]))
            univariate_bases.append(exp)

        ft = create_additive_functiontrain(univariate_bases, bkd, nqoi=1)

        # Evaluate at a point
        sample = bkd.asarray([[0.5], [0.3], [-0.2]])  # (3, 1)

        ft_result = ft(sample)

        # Manually compute sum of univariate functions
        expected = bkd.asarray([[0.0]])
        for ii in range(3):
            exp_val = univariate_bases[ii](sample[ii : ii + 1])
            expected = expected + exp_val

        bkd.assert_allclose(ft_result, expected, rtol=1e-10)

    def test_invalid_rank_structure_raises(self, bkd) -> None:
        """Test that invalid rank structure raises error."""
        # Create cores with mismatched ranks
        exp = self._create_univariate_expansion(bkd, 2)

        # Core 0: (1, 3)
        core0 = FunctionTrainCore([[exp, exp, exp]], bkd)
        # Core 1: (2, 1) - wrong! Should be (3, 1)
        core1 = FunctionTrainCore([[exp], [exp]], bkd)

        with pytest.raises(ValueError):
            FunctionTrain([core0, core1], bkd)

    def test_first_core_must_have_left_rank_1(self, bkd) -> None:
        """Test that first core must have left rank 1."""
        exp = self._create_univariate_expansion(bkd, 2)

        # Core with left rank 2 (invalid for first core)
        core = FunctionTrainCore([[exp], [exp]], bkd)

        with pytest.raises(ValueError):
            FunctionTrain([core], bkd)

    def test_last_core_must_have_right_rank_1(self, bkd) -> None:
        """Test that last core must have right rank 1."""
        exp = self._create_univariate_expansion(bkd, 2)

        # First core (1, 2)
        core0 = FunctionTrainCore([[exp, exp]], bkd)
        # Last core (2, 2) - invalid! Should be (2, 1)
        core1 = FunctionTrainCore([[exp, exp], [exp, exp]], bkd)

        with pytest.raises(ValueError):
            FunctionTrain([core0, core1], bkd)


class TestAdditiveFunctionTrain:
    """Tests for create_additive_functiontrain factory."""

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

    def test_requires_at_least_2_variables(self, bkd) -> None:
        """Test that additive FT requires at least 2 variables."""
        exp = self._create_univariate_expansion(bkd, 2)

        with pytest.raises(ValueError):
            create_additive_functiontrain([exp], bkd)

    def test_validates_univariate(self, bkd) -> None:
        """Test that non-univariate bases are rejected."""
        # Create a 2D expansion
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(2, 2, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        exp_2d = BasisExpansion(basis, bkd, nqoi=1)

        exp_1d = self._create_univariate_expansion(bkd, 2)

        with pytest.raises(ValueError):
            create_additive_functiontrain([exp_1d, exp_2d], bkd)


class TestConstantExpansion:
    """Tests for ConstantExpansion helper class."""

    def test_constant_evaluation(self, bkd) -> None:
        """Test constant returns same value for all inputs."""
        const = ConstantExpansion(3.14, bkd, nqoi=1)

        samples = bkd.asarray(np.random.randn(1, 5))
        result = const(samples)

        expected = bkd.full((1, 5), 3.14)
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_constant_nparams_zero(self, bkd) -> None:
        """Test that constants have no trainable parameters."""
        const = ConstantExpansion(1.0, bkd, nqoi=1)

        assert const.nparams() == 0

    def test_constant_basis_matrix(self, bkd) -> None:
        """Test constant basis matrix is all ones."""
        const = ConstantExpansion(1.0, bkd, nqoi=1)

        samples = bkd.asarray(np.random.randn(1, 5))
        basis_mat = const.basis_matrix(samples)

        expected = bkd.ones((5, 1))
        bkd.assert_allclose(basis_mat, expected, rtol=1e-12)
