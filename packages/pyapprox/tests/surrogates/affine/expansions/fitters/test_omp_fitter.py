"""Tests for OMPFitter (Orthogonal Matching Pursuit).

Tests focus on:
- Returning OMPResult
- Respecting max_nonzeros limit
- Respecting rtol stopping criterion
- Selection order and residual history tracking
- Sparse signal recovery
"""

import numpy as np
import pytest

from pyapprox.optimization.linear.sparse import OMPTerminationFlag
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.expansions.fitters.omp import (
    OMPFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.results import (
    OMPResult,
)
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d


class TestOMPFitter:
    """Base test class."""

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

    @pytest.mark.slow_on("TorchBkd")
    def test_fit_returns_omp_result(self, bkd) -> None:
        """Fit returns OMPResult."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        fitter = OMPFitter(bkd, max_nonzeros=5)
        result = fitter.fit(expansion, samples, values)

        assert isinstance(result, OMPResult)

    def test_result_params_shape(self, bkd) -> None:
        """Result params have correct shape."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        fitter = OMPFitter(bkd, max_nonzeros=5)
        result = fitter.fit(expansion, samples, values)

        assert result.params().shape == (expansion.nterms(), 1)

    def test_handles_1d_values(self, bkd) -> None:
        """Fitter handles 1D values array."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values_1d = bkd.asarray(np.random.randn(50))

        fitter = OMPFitter(bkd, max_nonzeros=5)
        result = fitter.fit(expansion, samples, values_1d)

        # Should work without error
        assert result.params().shape[1] == 1

    def test_respects_max_nonzeros(self, bkd) -> None:
        """Selects at most max_nonzeros terms."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=5)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values = bkd.asarray(np.random.randn(1, 50))

        max_nonzeros = 3
        fitter = OMPFitter(bkd, max_nonzeros=max_nonzeros, rtol=1e-15)
        result = fitter.fit(expansion, samples, values)

        assert result.n_nonzero() <= max_nonzeros
        assert len(result.selection_order()) <= max_nonzeros

    def test_respects_rtol(self, bkd) -> None:
        """Stops when residual norm below rtol."""
        nvars = 1
        max_level = 10

        # Create a target with only 2 nonzero coefficients
        target_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        true_coef[0, 0] = 1.0
        true_coef[1, 0] = 0.5
        target_expansion = target_expansion.with_params(true_coef)

        # Generate noiseless data
        nsamples = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        # Fit with high max_nonzeros but tight rtol
        fit_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        fitter = OMPFitter(bkd, max_nonzeros=nterms, rtol=1e-10)
        result = fitter.fit(fit_expansion, samples, values)

        # Should stop early due to rtol, not max_nonzeros
        assert result.termination_flag() == OMPTerminationFlag.RESIDUAL_TOLERANCE
        # Should select ~2 terms (the true sparse support)
        assert result.n_nonzero() <= 4

    def test_selection_order_recorded(self, bkd) -> None:
        """selection_order has correct length and values."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 40)))
        values = bkd.asarray(np.random.randn(1, 40))

        fitter = OMPFitter(bkd, max_nonzeros=5, rtol=1e-15)
        result = fitter.fit(expansion, samples, values)

        # selection_order should have n_nonzero entries
        assert len(result.selection_order()) == result.n_nonzero()

        # selection_order should match support (for OMP they're the same)
        bkd.assert_allclose(result.selection_order(), result.support())

    def test_residual_history_recorded(self, bkd) -> None:
        """residual_history has correct length."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 40)))
        values = bkd.asarray(np.random.randn(1, 40))

        fitter = OMPFitter(bkd, max_nonzeros=5, rtol=1e-15)
        result = fitter.fit(expansion, samples, values)

        # residual_history should have n_nonzero entries (one per iteration)
        assert len(result.residual_history()) == result.n_nonzero()

    def test_residual_history_decreasing(self, bkd) -> None:
        """Residual norm decreases (or stays same) each iteration."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=4)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values = bkd.asarray(np.random.randn(1, 50))

        fitter = OMPFitter(bkd, max_nonzeros=8, rtol=1e-15)
        result = fitter.fit(expansion, samples, values)

        residuals = result.residual_history()
        for ii in range(len(residuals) - 1):
            # Each residual should be <= previous (OMP is greedy optimal)
            assert float(residuals[ii + 1]) <= float(residuals[ii]) + 1e-10

    def test_n_nonzero_correct(self, bkd) -> None:
        """n_nonzero matches support length."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 40)))
        values = bkd.asarray(np.random.randn(1, 40))

        fitter = OMPFitter(bkd, max_nonzeros=5)
        result = fitter.fit(expansion, samples, values)

        assert result.n_nonzero() == len(result.support())

    def test_invalid_max_nonzeros_raises(self, bkd) -> None:
        """max_nonzeros < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_nonzeros"):
            OMPFitter(bkd, max_nonzeros=0)

    def test_multi_qoi_raises(self, bkd) -> None:
        """nqoi > 1 raises ValueError."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3, nqoi=2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(2, 30))

        fitter = OMPFitter(bkd, max_nonzeros=5)
        with pytest.raises(ValueError, match="nqoi=1"):
            fitter.fit(expansion, samples, values)

    def test_accessors(self, bkd) -> None:
        """Accessors return correct values."""
        fitter = OMPFitter(bkd, max_nonzeros=7, rtol=0.01)
        assert fitter.max_nonzeros() == 7
        bkd.assert_allclose(
            bkd.asarray([fitter.rtol()]),
            bkd.asarray([0.01]),
        )

    def test_fitted_surrogate_evaluates(self, bkd) -> None:
        """Fitted surrogate can evaluate at new points."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        fitter = OMPFitter(bkd, max_nonzeros=5)
        result = fitter.fit(expansion, samples, values)

        # Evaluate at new points
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        predictions = result(test_samples)

        assert predictions.shape == (1, 10)

    def test_recovers_sparse_signal(self, bkd) -> None:
        """OMP recovers a known sparse signal accurately."""
        nvars = 1
        max_level = 19  # 20 terms

        # Create target with exactly 3 nonzero coefficients
        target_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        # Set 3 nonzero coefficients at indices 0, 2, 5
        true_coef[0, 0] = 1.0
        true_coef[2, 0] = -0.5
        true_coef[5, 0] = 0.3
        target_expansion = target_expansion.with_params(true_coef)

        # Generate noiseless data
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        # Fit
        fit_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        fitter = OMPFitter(bkd, max_nonzeros=5, rtol=1e-10)
        result = fitter.fit(fit_expansion, samples, values)

        # Should recover the 3 terms
        assert result.n_nonzero() <= 5

        # Evaluate at test points
        ntest = 50
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntest)))
        target_values = target_expansion(test_samples)
        fitted_values = result(test_samples)

        # Should predict accurately
        rel_error = bkd.norm(fitted_values - target_values) / bkd.norm(target_values)
        assert float(rel_error) < 1e-8

    def test_recovers_target_expansion_with_decaying_coefficients(self, bkd) -> None:
        """OMP can recover a target expansion with decaying coefficients.

        Creates a 1D target expansion with 20 coefficients that decay as 1/k^2.
        Verifies that fitting with OMP recovers the target with high accuracy.
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

        # Fit with OMP - allow enough terms and tight tolerance
        fitter = OMPFitter(bkd, max_nonzeros=nterms, rtol=1e-10)
        result = fitter.fit(fit_expansion, samples, values)

        # Evaluate at test points
        ntest = 50
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntest)))
        target_values = target_expansion(test_samples)
        fitted_values = result(test_samples)

        # Fitted expansion should predict target with high accuracy
        rel_error = bkd.norm(fitted_values - target_values) / bkd.norm(target_values)
        assert float(rel_error) < 1e-4, "Fitted expansion should match target"

        # Also verify coefficients are similar
        fitted_coef = result.params()
        coef_rel_error = bkd.norm(fitted_coef - true_coef) / bkd.norm(true_coef)
        assert float(coef_rel_error) < 1e-3, "Coefficients should be recovered"
