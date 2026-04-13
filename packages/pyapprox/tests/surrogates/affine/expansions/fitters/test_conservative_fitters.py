"""Tests for Conservative Fitters.

Tests focus on:
- Conservativeness: risk_measure(surrogate) >= risk_measure(data) for 500 trials
- Result types and shapes
- Single QoI validation
- Accessor methods
"""

import numpy as np
import pytest

from pyapprox.probability import UniformMarginal
from pyapprox.risk import ExactAVaR, SampleAverageMeanPlusStdev
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.expansions.fitters.conservative import (
    ConservativeLstSqFitter,
    ConservativeQuantileFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.results import (
    DirectSolverResult,
)
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d
from tests._helpers.markers import slow_test


class TestConservativeFitters:
    """Base test class for conservative fitters."""

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

    # --- ConservativeLstSqFitter Tests ---

    def test_lstsq_fit_returns_direct_solver_result(self, bkd) -> None:
        """ConservativeLstSqFitter returns DirectSolverResult."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        fitter = ConservativeLstSqFitter(bkd, strength=1.0)
        result = fitter.fit(expansion, samples, values)

        assert isinstance(result, DirectSolverResult)

    def test_lstsq_result_params_shape(self, bkd) -> None:
        """ConservativeLstSqFitter result params have correct shape."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        fitter = ConservativeLstSqFitter(bkd, strength=1.0)
        result = fitter.fit(expansion, samples, values)

        assert result.params().shape == (expansion.nterms(), 1)

    def test_lstsq_handles_1d_values(self, bkd) -> None:
        """ConservativeLstSqFitter handles 1D values array."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values_1d = bkd.asarray(np.random.randn(50))

        fitter = ConservativeLstSqFitter(bkd, strength=1.0)
        result = fitter.fit(expansion, samples, values_1d)

        assert result.params().shape[1] == 1

    def test_lstsq_fitted_surrogate_evaluates(self, bkd) -> None:
        """ConservativeLstSqFitter fitted surrogate can evaluate."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        fitter = ConservativeLstSqFitter(bkd, strength=1.0)
        result = fitter.fit(expansion, samples, values)

        # Evaluate at new points
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        predictions = result(test_samples)

        assert predictions.shape == (1, 10)

    def test_lstsq_multi_qoi_raises(self, bkd) -> None:
        """ConservativeLstSqFitter: nqoi > 1 raises ValueError."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3, nqoi=2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(2, 30))

        fitter = ConservativeLstSqFitter(bkd, strength=1.0)
        with pytest.raises(ValueError, match="nqoi=1"):
            fitter.fit(expansion, samples, values)

    def test_lstsq_accessors(self, bkd) -> None:
        """ConservativeLstSqFitter accessors return correct values."""
        fitter = ConservativeLstSqFitter(bkd, strength=2.5)
        bkd.assert_allclose(
            bkd.asarray([fitter.strength()]),
            bkd.asarray([2.5]),
        )

    @slow_test
    def test_lstsq_conservative_500_trials(self, bkd) -> None:
        """ConservativeLstSqFitter: risk(surrogate) >= risk(data) for 500 trials.

        This is the main conservativeness test, replicating the legacy test.
        """
        nsamples, nvars = 100, 2
        max_level = 2

        # Create expansion
        expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)

        # Generate true coefficients for underlying function
        nterms = expansion.nterms()
        true_coefs = bkd.ones((nterms, 1))
        true_expansion = expansion.with_params(true_coefs)

        # Test parameters
        noise_std = 0.1
        ntrials = 500
        strength = 1.0

        # Create risk measure for comparison
        risk_stat = SampleAverageMeanPlusStdev(strength, bkd)

        fitter = ConservativeLstSqFitter(bkd, strength=strength)

        for _ in range(ntrials):
            # Generate samples and noisy values
            samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
            noiseless_values = true_expansion(samples)  # (1, nsamples)
            noise = bkd.asarray(np.random.normal(0, noise_std, (1, nsamples)))
            train_values = noiseless_values + noise

            # Fit conservative surrogate
            result = fitter.fit(expansion, samples, train_values)

            # Compute risk of surrogate predictions
            predictions = result(samples)  # (1, nsamples)
            weights = bkd.full((1, nsamples), 1.0 / nsamples)
            surrogate_risk = float(risk_stat(predictions, weights)[0, 0])

            # Compute risk of training data
            data_risk = float(risk_stat(train_values, weights)[0, 0])

            # Conservative condition must hold
            assert surrogate_risk >= data_risk - 1e-10, (
                f"Conservative condition violated: "
                f"surrogate_risk={surrogate_risk} < data_risk={data_risk}"
            )

    # --- ConservativeQuantileFitter Tests ---

    def test_quantile_fit_returns_direct_solver_result(self, bkd) -> None:
        """ConservativeQuantileFitter returns DirectSolverResult."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        fitter = ConservativeQuantileFitter(bkd, quantile=0.5)
        result = fitter.fit(expansion, samples, values)

        assert isinstance(result, DirectSolverResult)

    def test_quantile_result_params_shape(self, bkd) -> None:
        """ConservativeQuantileFitter result params have correct shape."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        fitter = ConservativeQuantileFitter(bkd, quantile=0.5)
        result = fitter.fit(expansion, samples, values)

        assert result.params().shape == (expansion.nterms(), 1)

    def test_quantile_handles_1d_values(self, bkd) -> None:
        """ConservativeQuantileFitter handles 1D values array."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values_1d = bkd.asarray(np.random.randn(50))

        fitter = ConservativeQuantileFitter(bkd, quantile=0.5)
        result = fitter.fit(expansion, samples, values_1d)

        assert result.params().shape[1] == 1

    def test_quantile_fitted_surrogate_evaluates(self, bkd) -> None:
        """ConservativeQuantileFitter fitted surrogate can evaluate."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        fitter = ConservativeQuantileFitter(bkd, quantile=0.5)
        result = fitter.fit(expansion, samples, values)

        # Evaluate at new points
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        predictions = result(test_samples)

        assert predictions.shape == (1, 10)

    def test_quantile_multi_qoi_raises(self, bkd) -> None:
        """ConservativeQuantileFitter: nqoi > 1 raises ValueError."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3, nqoi=2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(2, 30))

        fitter = ConservativeQuantileFitter(bkd, quantile=0.5)
        with pytest.raises(ValueError, match="nqoi=1"):
            fitter.fit(expansion, samples, values)

    def test_quantile_invalid_quantile_raises(self, bkd) -> None:
        """ConservativeQuantileFitter: quantile outside [0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="(?i)quantile"):
            ConservativeQuantileFitter(bkd, quantile=1.0)

        with pytest.raises(ValueError, match="(?i)quantile"):
            ConservativeQuantileFitter(bkd, quantile=-0.1)

    def test_quantile_accessors(self, bkd) -> None:
        """ConservativeQuantileFitter accessors return correct values."""
        fitter = ConservativeQuantileFitter(bkd, quantile=0.75)
        bkd.assert_allclose(
            bkd.asarray([fitter.quantile()]),
            bkd.asarray([0.75]),
        )

    @slow_test
    def test_quantile_conservative_500_trials(self, bkd) -> None:
        """ConservativeQuantileFitter: risk(surrogate) >= risk(data) for 500 trials.

        This is the main conservativeness test, replicating the legacy test.
        Uses AVaR (CVaR) at the specified quantile as the risk measure.
        """
        nsamples, nvars = 100, 2
        max_level = 2

        # Create expansion
        expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)

        # Generate true coefficients for underlying function
        nterms = expansion.nterms()
        true_coefs = bkd.ones((nterms, 1))
        true_expansion = expansion.with_params(true_coefs)

        # Test parameters
        noise_std = 0.1
        ntrials = 500
        quantile = 0.5

        # Create risk measure for comparison
        risk_stat = ExactAVaR(quantile, bkd)

        fitter = ConservativeQuantileFitter(bkd, quantile=quantile)

        for _ in range(ntrials):
            # Generate samples and noisy values
            samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
            noiseless_values = true_expansion(samples)  # (1, nsamples)
            noise = bkd.asarray(np.random.normal(0, noise_std, (1, nsamples)))
            train_values = noiseless_values + noise

            # Fit conservative surrogate
            result = fitter.fit(expansion, samples, train_values)

            # Compute risk of surrogate predictions
            predictions = result(samples)  # (1, nsamples)
            weights = bkd.full((1, nsamples), 1.0 / nsamples)
            surrogate_risk = float(risk_stat(predictions, weights)[0, 0])

            # Compute risk of training data
            data_risk = float(risk_stat(train_values, weights)[0, 0])

            # Conservative condition must hold
            assert surrogate_risk >= data_risk - 1e-10, (
                f"Conservative condition violated: "
                f"surrogate_risk={surrogate_risk} < data_risk={data_risk}"
            )

    def test_different_strengths_affect_conservativeness(self, bkd) -> None:
        """Higher strength parameter produces more conservative surrogates."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)

        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 100)))
        values = bkd.asarray(np.random.randn(1, 100))

        # Fit with different strengths
        fitter_low = ConservativeLstSqFitter(bkd, strength=0.5)
        fitter_high = ConservativeLstSqFitter(bkd, strength=2.0)

        result_low = fitter_low.fit(expansion, samples, values)
        result_high = fitter_high.fit(expansion, samples, values)

        # Higher strength should give higher constant coefficient (more conservative)
        # The constant term is at index 0
        coef_low = float(result_low.params()[0, 0])
        coef_high = float(result_high.params()[0, 0])

        assert coef_high > coef_low

    @pytest.mark.slow_on("TorchBkd")
    def test_different_quantiles_affect_conservativeness(self, bkd) -> None:
        """Higher quantile produces more conservative surrogates."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)

        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 100)))
        values = bkd.asarray(np.random.randn(1, 100))

        # Fit with different quantiles
        fitter_low = ConservativeQuantileFitter(bkd, quantile=0.3)
        fitter_high = ConservativeQuantileFitter(bkd, quantile=0.8)

        result_low = fitter_low.fit(expansion, samples, values)
        result_high = fitter_high.fit(expansion, samples, values)

        # Higher quantile should give higher constant coefficient (more conservative)
        coef_low = float(result_low.params()[0, 0])
        coef_high = float(result_high.params()[0, 0])

        assert coef_high > coef_low
