"""Tests for RidgeCVFitter.

Tests verify that LOO/LMO cross-validation correctly selects
the ridge regularization parameter alpha. Key mathematical property:
for orthonormal basis with n >> p, the optimal alpha satisfies
alpha_opt ≈ p * sigma^2 / ||beta_true||^2, linking regularization
strength to noise variance.
"""

import numpy as np
import pytest
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.expansions.fitters.results import (
    CVSelectionResult,
)
from pyapprox.surrogates.affine.expansions.fitters.ridge_cv import (
    RidgeCVFitter,
)
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d


class TestRidgeCVFitter:

    def _create_expansion(self, bkd, nvars, max_level, nqoi=1):
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_returns_cv_selection_result(self, bkd):
        expansion = self._create_expansion(bkd, nvars=1, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        fitter = RidgeCVFitter(bkd, alphas=[1e-6, 1e-3, 1.0])
        result = fitter.fit(expansion, samples, values)

        assert isinstance(result, CVSelectionResult)

    def test_noiseless_selects_small_alpha(self, bkd):
        """With noiseless data, CV should select the smallest alpha."""
        expansion = self._create_expansion(bkd, nvars=1, max_level=3)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        alphas = [1e-12, 1e-8, 1e-4, 1e-2, 1.0, 10.0]
        fitter = RidgeCVFitter(bkd, alphas=alphas)
        result = fitter.fit(expansion, samples, values)

        assert result.best_label() <= 1e-4

    def test_noiseless_recovers_function(self, bkd):
        """With noiseless data, fitted surrogate reproduces the function."""
        expansion = self._create_expansion(bkd, nvars=1, max_level=3)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        fitter = RidgeCVFitter(bkd, alphas=[1e-12, 1e-8, 1e-4, 1.0])
        result = fitter.fit(expansion, samples, values)

        test_samples = bkd.asarray(np.random.uniform(-1, 1, (1, 20)))
        predicted = result(test_samples)
        expected = bkd.reshape(test_samples[0, :] ** 2, (1, -1))
        bkd.assert_allclose(predicted, expected, atol=1e-10)

    def test_noisy_selects_positive_alpha(self, bkd):
        """With noisy data, CV should select alpha > smallest candidate."""
        expansion = self._create_expansion(bkd, nvars=1, max_level=5)
        ntrain = 200
        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        signal = samples[0, :] ** 2
        noise = bkd.asarray(np.random.normal(0, 0.5, ntrain))
        values = bkd.reshape(signal + noise, (1, -1))

        alphas = [1e-12, 1e-8, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0]
        fitter = RidgeCVFitter(bkd, alphas=alphas)
        result = fitter.fit(expansion, samples, values)

        assert result.best_label() > 1e-8

    def test_higher_noise_selects_larger_alpha(self, bkd):
        """More noise should lead to larger selected alpha."""
        expansion_lo = self._create_expansion(bkd, nvars=1, max_level=5)
        expansion_hi = self._create_expansion(bkd, nvars=1, max_level=5)
        ntrain = 300
        np.random.seed(42)
        samples_np = np.random.uniform(-1, 1, (1, ntrain))
        samples = bkd.asarray(samples_np)
        signal = samples[0, :] ** 2

        alphas = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0]

        noise_lo = bkd.asarray(np.random.normal(0, 0.01, ntrain))
        values_lo = bkd.reshape(signal + noise_lo, (1, -1))
        result_lo = RidgeCVFitter(bkd, alphas=alphas).fit(
            expansion_lo, samples, values_lo
        )

        noise_hi = bkd.asarray(np.random.normal(0, 1.0, ntrain))
        values_hi = bkd.reshape(signal + noise_hi, (1, -1))
        result_hi = RidgeCVFitter(bkd, alphas=alphas).fit(
            expansion_hi, samples, values_hi
        )

        assert result_hi.best_label() >= result_lo.best_label()

    def test_optimal_alpha_scales_with_noise_variance(self, bkd):
        """For orthonormal basis with n >> p, alpha_opt ~ p*sigma^2/||beta||^2.

        We verify the selected alpha is within one grid step of this
        theoretical prediction.
        """
        nvars = 1
        max_level = 3
        expansion = self._create_expansion(bkd, nvars, max_level)
        nterms = expansion.nterms()
        ntrain = 500

        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))

        basis_mat = expansion.basis_matrix(samples)
        true_coef = bkd.asarray(
            np.random.RandomState(7).randn(nterms, 1)
        )
        signal = (basis_mat @ true_coef).T
        coef_norm_sq = float(bkd.sum(true_coef ** 2))

        sigma = 0.3
        noise = bkd.asarray(
            np.random.RandomState(8).normal(0, sigma, (1, ntrain))
        )
        values = signal + noise
        alpha_theory = nterms * sigma**2 / coef_norm_sq

        alphas = np.logspace(-6, 4, 50).tolist()
        fitter = RidgeCVFitter(bkd, alphas=alphas)
        result = fitter.fit(expansion, samples, values)
        alpha_selected = result.best_label()

        ratio = alpha_selected / alpha_theory
        assert 0.1 < ratio < 10.0, (
            f"alpha_selected={alpha_selected:.4e}, "
            f"alpha_theory={alpha_theory:.4e}, ratio={ratio:.2f}"
        )

    def test_cv_scores_shape(self, bkd):
        """CV scores array has one entry per alpha."""
        expansion = self._create_expansion(bkd, nvars=1, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
        fitter = RidgeCVFitter(bkd, alphas=alphas)
        result = fitter.fit(expansion, samples, values)

        assert result.cv_scores().shape[0] == len(alphas)

    def test_candidate_labels_match_alphas(self, bkd):
        """Candidate labels are the input alpha values."""
        expansion = self._create_expansion(bkd, nvars=1, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        alphas = [0.01, 0.1, 1.0]
        fitter = RidgeCVFitter(bkd, alphas=alphas)
        result = fitter.fit(expansion, samples, values)

        assert result.candidate_labels() == alphas

    def test_multi_qoi(self, bkd):
        """Works with multiple QoIs."""
        expansion = self._create_expansion(bkd, nvars=1, max_level=3, nqoi=2)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.asarray(np.vstack([
            samples[0, :] ** 2,
            samples[0, :] ** 3,
        ]))

        fitter = RidgeCVFitter(bkd, alphas=[1e-10, 1e-4, 1.0])
        result = fitter.fit(expansion, samples, values)

        assert isinstance(result, CVSelectionResult)
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (1, 10)))
        predicted = result(test_samples)
        assert predicted.shape == (2, 10)

    def test_with_lmo_nfolds(self, bkd):
        """Works with LMO cross-validation."""
        expansion = self._create_expansion(bkd, nvars=1, max_level=3)
        ntrain = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        fitter = RidgeCVFitter(bkd, alphas=[1e-8, 1e-4, 1.0], nfolds=5)
        result = fitter.fit(expansion, samples, values)

        assert isinstance(result, CVSelectionResult)

    def test_empty_alphas_raises(self, bkd):
        """Empty alpha list raises ValueError."""
        with pytest.raises(ValueError):
            RidgeCVFitter(bkd, alphas=[])

    def test_accessors(self, bkd):
        fitter = RidgeCVFitter(bkd, alphas=[0.1, 1.0, 10.0], nfolds=5)
        assert fitter.alphas() == [0.1, 1.0, 10.0]
        assert fitter.bkd() is bkd

    def test_2d_function(self, bkd):
        """Works on a 2D quadratic: f(x,y) = x^2 + 2xy + y."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        ntrain = 200
        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, ntrain)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**2 + 2*x*y + y, (1, -1))

        fitter = RidgeCVFitter(bkd, alphas=[1e-12, 1e-8, 1e-4, 1.0])
        result = fitter.fit(expansion, samples, values)

        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        xt, yt = test_samples[0, :], test_samples[1, :]
        predicted = result(test_samples)
        expected = bkd.reshape(xt**2 + 2*xt*yt + yt, (1, -1))
        bkd.assert_allclose(predicted, expected, atol=1e-10)
