"""Tests for PCEDegreeSelectionFitter.

Tests verify that cross-validation-based level selection correctly
identifies the optimal index set level for various test functions.
"""

import numpy as np
import pytest

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.expansions.fitters.pce_cv import (
    PCEDegreeSelectionFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.results import (
    CVSelectionResult,
)
from pyapprox.surrogates.affine.indices import (
    HyperbolicIndexSequence,
    SparseGridIndexSequence,
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.indices.growth_rules import (
    LinearGrowthRule,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d


class TestPCEDegreeSelectionFitter:
    """Base test class."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_expansion(self, bkd, nvars: int, max_level: int, nqoi: int = 1):
        """Create test expansion with initial indices."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_selects_correct_degree_for_quadratic(self, bkd) -> None:
        """For x^2, selects level >= 2 and CV score at level 2 is near zero."""
        nvars = 1
        expansion = self._create_expansion(bkd, nvars, max_level=5)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        fitter = PCEDegreeSelectionFitter(
            bkd, levels=[1, 2, 3, 4, 5], index_sequence=index_seq
        )
        result = fitter.fit(expansion, samples, values)

        # level 1 should have large error, levels >= 2 should be near zero
        cv_scores = result.cv_scores()
        bkd.assert_allclose(cv_scores[1:2], bkd.zeros(1), atol=1e-12)
        # level 1 CV score should be much larger
        assert float(cv_scores[0]) > 1e-2
        # selected level should be >= 2 (exact choice among >=2 is noise)
        assert result.best_label() >= 2

    def test_selects_correct_degree_for_cubic(self, bkd) -> None:
        """For x^3, selects level >= 3 and CV score at level 3 is near zero."""
        nvars = 1
        expansion = self._create_expansion(bkd, nvars, max_level=5)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        values = bkd.reshape(samples[0, :] ** 3, (1, -1))

        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        fitter = PCEDegreeSelectionFitter(
            bkd, levels=[1, 2, 3, 4, 5], index_sequence=index_seq
        )
        result = fitter.fit(expansion, samples, values)

        # levels 1-2 should have large error, levels >= 3 near zero
        cv_scores = result.cv_scores()
        bkd.assert_allclose(cv_scores[2:3], bkd.zeros(1), atol=1e-12)
        assert float(cv_scores[0]) > 1e-2
        assert result.best_label() >= 3

    def test_multivariate_degree_selection(self, bkd) -> None:
        """Selects correct level for 2D quadratic."""
        nvars = 2
        expansion = self._create_expansion(bkd, nvars, max_level=4)
        ntrain = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        # f(x,y) = x^2 + y^2 -> degree 2 polynomial
        values = bkd.reshape(samples[0, :] ** 2 + samples[1, :] ** 2, (1, -1))

        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        fitter = PCEDegreeSelectionFitter(
            bkd, levels=[1, 2, 3, 4], index_sequence=index_seq
        )
        result = fitter.fit(expansion, samples, values)

        # Level 2 exactly represents x^2 + y^2; higher levels also fit it
        # exactly so their CV scores are comparable—backend numerics may
        # pick level 2 or 3.
        assert result.best_label() >= 2

    def test_returns_cv_selection_result(self, bkd) -> None:
        """Fit returns CVSelectionResult."""
        nvars = 1
        expansion = self._create_expansion(bkd, nvars=1, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 20)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        fitter = PCEDegreeSelectionFitter(
            bkd, levels=[1, 2, 3], index_sequence=index_seq
        )
        result = fitter.fit(expansion, samples, values)

        assert isinstance(result, CVSelectionResult)

    def test_cv_scores_shape(self, bkd) -> None:
        """CV scores array matches number of candidates."""
        nvars = 1
        levels = [1, 2, 3, 4]
        expansion = self._create_expansion(bkd, nvars=1, max_level=4)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 30)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        fitter = PCEDegreeSelectionFitter(bkd, levels=levels, index_sequence=index_seq)
        result = fitter.fit(expansion, samples, values)

        assert result.cv_scores().shape[0] == len(levels)

    def test_underdetermined_skipped(self, bkd) -> None:
        """Levels with too many terms for nsamples get inf score."""
        nvars = 2
        expansion = self._create_expansion(bkd, nvars, max_level=10)
        ntrain = 8  # Very few samples
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        values = bkd.reshape(samples[0, :] + samples[1, :], (1, -1))

        # Level 10 has (10+1)*(10+2)/2 = 66 terms >> 8 samples
        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        fitter = PCEDegreeSelectionFitter(bkd, levels=[1, 10], index_sequence=index_seq)
        result = fitter.fit(expansion, samples, values)

        # Level 10 should be inf, so level 1 is selected
        assert result.best_label() == 1
        cv_scores = result.cv_scores()
        assert float(cv_scores[1]) == float("inf")

    def test_accessors(self, bkd) -> None:
        """Accessors return correct values."""
        nvars = 1
        levels = [1, 2, 3]
        index_seq = HyperbolicIndexSequence(nvars, 0.5, bkd)
        fitter = PCEDegreeSelectionFitter(
            bkd, levels=levels, index_sequence=index_seq, alpha=1e-3
        )
        assert fitter.levels() == levels
        assert fitter.index_sequence() is index_seq
        assert fitter.alpha() == pytest.approx(1e-3)

    def test_with_ridge_regularization(self, bkd) -> None:
        """Level selection works with ridge regularization."""
        nvars = 1
        expansion = self._create_expansion(bkd, nvars=1, max_level=5)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        fitter = PCEDegreeSelectionFitter(
            bkd, levels=[1, 2, 3, 4, 5], index_sequence=index_seq, alpha=1e-6
        )
        result = fitter.fit(expansion, samples, values)

        # With small alpha, should still select level 2
        assert result.best_label() == 2

    def test_with_lmo_nfolds(self, bkd) -> None:
        """Level selection works with LMO CV."""
        nvars = 1
        expansion = self._create_expansion(bkd, nvars=1, max_level=5)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        fitter = PCEDegreeSelectionFitter(
            bkd, levels=[1, 2, 3, 4, 5], index_sequence=index_seq, nfolds=5
        )
        result = fitter.fit(expansion, samples, values)

        # Level >= 2 should be selected (exact noiseless fit)
        assert result.best_label() >= 2
        # Level 1 should have significantly larger error
        assert float(result.cv_scores()[0]) > float(result.cv_scores()[1]) + 1e-3

    def test_fitted_surrogate_evaluates(self, bkd) -> None:
        """Fitted surrogate from result evaluates correctly."""
        nvars = 1
        expansion = self._create_expansion(bkd, nvars=1, max_level=3)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        fitter = PCEDegreeSelectionFitter(
            bkd, levels=[1, 2, 3], index_sequence=index_seq
        )
        result = fitter.fit(expansion, samples, values)

        # Evaluate on test samples
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (1, 10)))
        predicted = result(test_samples)
        expected = bkd.reshape(test_samples[0, :] ** 2, (1, -1))

        bkd.assert_allclose(predicted, expected, atol=1e-10)

    def test_empty_levels_raises(self, bkd) -> None:
        """Empty levels list raises ValueError."""
        nvars = 1
        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        with pytest.raises(ValueError):
            PCEDegreeSelectionFitter(bkd, levels=[], index_sequence=index_seq)

    def test_all_params_stored(self, bkd) -> None:
        """All candidate params are stored in result."""
        nvars = 1
        levels = [1, 2, 3]
        expansion = self._create_expansion(bkd, nvars=1, max_level=3)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        fitter = PCEDegreeSelectionFitter(bkd, levels=levels, index_sequence=index_seq)
        result = fitter.fit(expansion, samples, values)

        assert len(result.all_params()) == len(levels)

    def test_with_sparse_grid_index_sequence(self, bkd) -> None:
        """Level selection works with SparseGridIndexSequence."""
        nvars = 2
        expansion = self._create_expansion(bkd, nvars, max_level=4)
        ntrain = 80
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        # f(x,y) = x^2 + y^2 -> degree 2 polynomial
        values = bkd.reshape(samples[0, :] ** 2 + samples[1, :] ** 2, (1, -1))

        # Use LinearGrowthRule(1,1) so sparse grid level = total degree
        index_seq = SparseGridIndexSequence(
            nvars, bkd, growth_rules=LinearGrowthRule(scale=1, shift=1)
        )
        fitter = PCEDegreeSelectionFitter(
            bkd, levels=[1, 2, 3, 4], index_sequence=index_seq
        )
        result = fitter.fit(expansion, samples, values)

        assert result.best_label() == 2
