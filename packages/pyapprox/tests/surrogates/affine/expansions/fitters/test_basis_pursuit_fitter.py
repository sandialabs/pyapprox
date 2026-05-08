"""Tests for BasisPursuitFitter (Basis Pursuit L1 minimization).

Tests focus on:
- Returning SparseResult
- Sparse coefficient recovery
- Support identification
- Multi-QoI rejection
"""

import numpy as np
import pytest

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.expansions.fitters.basis_pursuit import (
    BasisPursuitFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.results import (
    SparseResult,
)
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d


class TestBasisPursuitFitter:
    """Base test class for BasisPursuitFitter."""

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
        nvars, max_level = 1, 5

        # Create a sparse target
        target_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        true_coef[0, 0] = 1.0
        true_coef[2, 0] = 0.5
        target_expansion = target_expansion.with_params(true_coef)

        # Generate noiseless data (BP requires exact data)
        nsamples = nterms + 5  # slightly overdetermined
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        fitter = BasisPursuitFitter(bkd)
        fit_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        result = fitter.fit(fit_expansion, samples, values)

        assert isinstance(result, SparseResult)

    def test_result_params_shape(self, bkd) -> None:
        """Result params have correct shape."""
        nvars, max_level = 1, 5

        target_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        true_coef[0, 0] = 1.0
        target_expansion = target_expansion.with_params(true_coef)

        nsamples = nterms + 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        fitter = BasisPursuitFitter(bkd)
        fit_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        result = fitter.fit(fit_expansion, samples, values)

        assert result.params().shape == (nterms, 1)

    def test_handles_1d_values(self, bkd) -> None:
        """Fitter handles 1D values array."""
        nvars, max_level = 1, 3

        target_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        true_coef[0, 0] = 1.0
        target_expansion = target_expansion.with_params(true_coef)

        nsamples = nterms + 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values_2d = target_expansion(samples)
        values_1d = values_2d[0, :]  # flatten to 1D

        fitter = BasisPursuitFitter(bkd)
        fit_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        result = fitter.fit(fit_expansion, samples, values_1d)

        assert result.params().shape[1] == 1

    def test_fitted_surrogate_evaluates(self, bkd) -> None:
        """Fitted surrogate can evaluate at new points."""
        nvars, max_level = 1, 5

        target_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        true_coef[0, 0] = 1.0
        true_coef[2, 0] = 0.5
        target_expansion = target_expansion.with_params(true_coef)

        nsamples = nterms + 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        fitter = BasisPursuitFitter(bkd)
        fit_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        result = fitter.fit(fit_expansion, samples, values)

        # Evaluate at new points
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, 10)))
        predictions = result(test_samples)

        assert predictions.shape == (1, 10)

    def test_multi_qoi_raises(self, bkd) -> None:
        """nqoi > 1 raises ValueError."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3, nqoi=2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(2, 30))

        fitter = BasisPursuitFitter(bkd)
        with pytest.raises(ValueError, match="nqoi=1"):
            fitter.fit(expansion, samples, values)

    def test_recovers_sparse_signal(self, bkd) -> None:
        """BP recovers a known sparse signal accurately.

        Basis Pursuit recovers the sparsest solution when data is noiseless
        and the system is sufficiently overdetermined.
        """
        nvars = 1
        max_level = 9  # 10 terms

        # Create target with exactly 3 nonzero coefficients
        target_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        true_coef[0, 0] = 1.0
        true_coef[2, 0] = -0.5
        true_coef[5, 0] = 0.3
        target_expansion = target_expansion.with_params(true_coef)

        # Generate noiseless data (more samples than terms)
        nsamples = nterms + 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        # Fit
        fit_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        fitter = BasisPursuitFitter(bkd)
        result = fitter.fit(fit_expansion, samples, values)

        # Evaluate at test points
        ntest = 50
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntest)))
        target_values = target_expansion(test_samples)
        fitted_values = result(test_samples)

        # Should predict accurately
        rel_error = bkd.norm(fitted_values - target_values) / bkd.norm(target_values)
        assert float(rel_error) < 1e-8

    def test_support_identification(self, bkd) -> None:
        """BP correctly identifies support of sparse signal."""
        nvars = 1
        max_level = 9  # 10 terms

        # Create target with known support
        target_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        true_support = [0, 3, 7]  # indices of nonzero coefficients
        for idx in true_support:
            true_coef[idx, 0] = 1.0 / (idx + 1)
        target_expansion = target_expansion.with_params(true_coef)

        # Generate noiseless data
        nsamples = nterms + 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        # Fit
        fit_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        fitter = BasisPursuitFitter(bkd)
        result = fitter.fit(fit_expansion, samples, values)

        # Check support
        assert result.n_nonzero() == len(true_support)

        # Support should contain the true indices
        recovered_support = sorted([int(s) for s in result.support()])
        assert recovered_support == true_support

    def test_n_nonzero_correct(self, bkd) -> None:
        """n_nonzero matches support length."""
        nvars = 1
        max_level = 5

        target_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        true_coef[0, 0] = 1.0
        true_coef[1, 0] = 0.5
        target_expansion = target_expansion.with_params(true_coef)

        nsamples = nterms + 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        fitter = BasisPursuitFitter(bkd)
        fit_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        result = fitter.fit(fit_expansion, samples, values)

        assert result.n_nonzero() == len(result.support())

    def test_regularization_strength_zero(self, bkd) -> None:
        """regularization_strength is 0 for BP (no regularization)."""
        nvars = 1
        max_level = 3

        target_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        true_coef[0, 0] = 1.0
        target_expansion = target_expansion.with_params(true_coef)

        nsamples = nterms + 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        fitter = BasisPursuitFitter(bkd)
        fit_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        result = fitter.fit(fit_expansion, samples, values)

        bkd.assert_allclose(
            bkd.asarray([result.regularization_strength()]),
            bkd.asarray([0.0]),
        )

    def test_coefficient_recovery(self, bkd) -> None:
        """BP recovers exact coefficients for sparse signal."""
        nvars = 1
        max_level = 9  # 10 terms

        # Create target with 2 nonzero coefficients
        target_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        true_coef[0, 0] = 2.0
        true_coef[4, 0] = -1.5
        target_expansion = target_expansion.with_params(true_coef)

        # Generate noiseless data
        nsamples = nterms + 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        # Fit
        fit_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        fitter = BasisPursuitFitter(bkd)
        result = fitter.fit(fit_expansion, samples, values)

        # Check coefficient recovery
        bkd.assert_allclose(result.params(), true_coef, atol=1e-8)

    def test_recovers_1d_expansion_random_sparse(self, bkd) -> None:
        """BP recovers a 1D expansion with random sparse coefficients.

        Replicates legacy test_basis_pursuit: creates random sparse
        coefficients and verifies exact recovery.
        """
        nvars = 1
        max_level = 7  # 8 terms (degree 7 polynomial)
        sparsity = 2

        # Create expansion
        target_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()

        # Create random sparse coefficients (like legacy test)
        true_coef = bkd.zeros((nterms, 1))
        np.random.seed(42)  # ensure reproducibility
        sparse_indices = np.random.permutation(nterms)[:sparsity]
        for idx in sparse_indices:
            true_coef[idx, 0] = 1.0
        target_expansion = target_expansion.with_params(true_coef)

        # Generate noiseless data (overdetermined)
        nsamples = nterms + 2
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        # Fit
        fit_expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        fitter = BasisPursuitFitter(bkd)
        result = fitter.fit(fit_expansion, samples, values)

        # Should recover exactly
        bkd.assert_allclose(result.params(), true_coef, atol=1e-8)
