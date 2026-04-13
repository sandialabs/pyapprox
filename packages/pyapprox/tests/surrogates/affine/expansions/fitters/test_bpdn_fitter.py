"""Tests for BPDNFitter (Basis Pursuit Denoising).

Tests focus on:
- Returning SparseResult
- Producing sparse solutions
- Support/n_nonzero correctness
- Regularization behavior
"""

import numpy as np
import pytest

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.expansions.fitters.bpdn import (
    BPDNFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.results import (
    SparseResult,
)
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d


class TestBPDNFitter:
    """Base test class for BPDNFitter."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_expansion(self, bkd, nvars: int, max_level: int, nqoi: int = 1):
        """Create test expansion."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_fit_returns_sparse_result(self, bkd) -> None:
        """Fit returns SparseResult."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        fitter = BPDNFitter(bkd, penalty=0.1)
        result = fitter.fit(expansion, samples, values)

        assert isinstance(result, SparseResult)

    def test_result_params_shape(self, bkd) -> None:
        """Result params have correct shape."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        fitter = BPDNFitter(bkd, penalty=0.1)
        result = fitter.fit(expansion, samples, values)

        assert result.params().shape == (expansion.nterms(), 1)

    def test_handles_1d_values(self, bkd) -> None:
        """Fitter handles 1D values array."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values_1d = bkd.asarray(np.random.randn(50))

        fitter = BPDNFitter(bkd, penalty=0.1)
        result = fitter.fit(expansion, samples, values_1d)

        # Should work without error
        assert result.params().shape[1] == 1

    def test_produces_sparse_solution(self, bkd) -> None:
        """Higher penalty leads to sparser solution."""
        expansion_low = self._create_expansion(bkd, nvars=2, max_level=4)
        expansion_high = self._create_expansion(bkd, nvars=2, max_level=4)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values = bkd.asarray(np.random.randn(1, 50))

        # Low regularization (less sparse)
        fitter_low = BPDNFitter(bkd, penalty=0.01)
        result_low = fitter_low.fit(expansion_low, samples, values)

        # High regularization (more sparse)
        fitter_high = BPDNFitter(bkd, penalty=0.5)
        result_high = fitter_high.fit(expansion_high, samples, values)

        # Higher penalty should give fewer nonzeros
        assert result_high.n_nonzero() <= result_low.n_nonzero()

    def test_support_matches_nonzero(self, bkd) -> None:
        """Support contains exactly the nonzero coefficient indices."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 40)))
        values = bkd.asarray(np.random.randn(1, 40))

        fitter = BPDNFitter(bkd, penalty=0.1)
        result = fitter.fit(expansion, samples, values)

        # Count nonzeros manually
        params = result.params()[:, 0]
        tol = 1e-10
        abs_params = bkd.abs(params)
        nonzero_mask = abs_params > tol
        expected_support = bkd.where(nonzero_mask)[0]

        # support should match
        bkd.assert_allclose(result.support(), expected_support)

    @pytest.mark.slow_on("TorchBkd")
    def test_n_nonzero_correct(self, bkd) -> None:
        """n_nonzero matches actual count."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 40)))
        values = bkd.asarray(np.random.randn(1, 40))

        fitter = BPDNFitter(bkd, penalty=0.1)
        result = fitter.fit(expansion, samples, values)

        # n_nonzero should match support length
        assert result.n_nonzero() == len(result.support())

    def test_invalid_penalty_raises(self, bkd) -> None:
        """Non-positive penalty raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            BPDNFitter(bkd, penalty=0.0)

        with pytest.raises(ValueError, match="positive"):
            BPDNFitter(bkd, penalty=-1.0)

    def test_multi_qoi_raises(self, bkd) -> None:
        """nqoi > 1 raises ValueError."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3, nqoi=2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(2, 30))

        fitter = BPDNFitter(bkd, penalty=0.1)
        with pytest.raises(ValueError, match="nqoi=1"):
            fitter.fit(expansion, samples, values)

    def test_regularization_strength_stored(self, bkd) -> None:
        """Result stores regularization_strength correctly."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        penalty = 0.25
        fitter = BPDNFitter(bkd, penalty=penalty)
        result = fitter.fit(expansion, samples, values)

        bkd.assert_allclose(
            bkd.asarray([result.regularization_strength()]),
            bkd.asarray([penalty]),
        )

    def test_penalty_accessor(self, bkd) -> None:
        """Penalty accessor returns correct value."""
        fitter = BPDNFitter(bkd, penalty=0.75)
        bkd.assert_allclose(
            bkd.asarray([fitter.penalty()]),
            bkd.asarray([0.75]),
        )

    def test_fitted_surrogate_evaluates(self, bkd) -> None:
        """Fitted surrogate can evaluate at new points."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        fitter = BPDNFitter(bkd, penalty=0.1)
        result = fitter.fit(expansion, samples, values)

        # Evaluate at new points
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        predictions = result(test_samples)

        assert predictions.shape == (1, 10)

    def test_recovers_target_expansion_with_decaying_coefficients(self, bkd) -> None:
        """BPDN can recover a sparse target expansion with decaying coefficients.

        Creates a 1D target expansion with 20 coefficients that decay as 1/k^2.
        Verifies that fitting with BPDN (small penalty) recovers the target with
        high accuracy.
        """
        nvars = 1
        max_level = 19  # 20 terms for 1D

        # Create target expansion with decaying coefficients
        target_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()

        # Create decaying coefficients: c_k = 1 / (k+1)^2
        # This gives coefficients like [1, 0.25, 0.11, 0.0625, ...]
        true_coef = bkd.asarray([[1.0 / ((k + 1) ** 2)] for k in range(nterms)])
        target_expansion = target_expansion.with_params(true_coef)

        # Generate training samples (more samples than coefficients)
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        # Generate training values from target expansion
        values = target_expansion(samples)

        # Create expansion to fit
        fit_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)

        # Fit with small penalty (weak regularization) to allow recovery
        fitter = BPDNFitter(bkd, penalty=1e-6, max_iter=5000, tol=1e-8)
        result = fitter.fit(fit_expansion, samples, values)

        # Evaluate at test points
        ntest = 50
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntest)))
        target_values = target_expansion(test_samples)
        fitted_values = result(test_samples)

        # Fitted expansion should predict target with high accuracy
        # Use relative error since values can vary in magnitude
        rel_error = bkd.norm(fitted_values - target_values) / bkd.norm(target_values)
        assert float(rel_error) < 1e-4, "Fitted expansion should match target"

        # Also verify coefficients are similar
        fitted_coef = result.params()
        coef_rel_error = bkd.norm(fitted_coef - true_coef) / bkd.norm(true_coef)
        assert float(coef_rel_error) < 1e-3, "Coefficients should be recovered"
