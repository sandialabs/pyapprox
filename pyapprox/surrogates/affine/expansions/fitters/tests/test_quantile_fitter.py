"""Tests for QuantileFitter (Quantile Regression).

Tests focus on:
- Returning DirectSolverResult
- Median finding (quantile=0.5)
- Quantile statistic equal to zero at solution
- Multi-QoI rejection
"""

import numpy as np
import pytest

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.expansions.fitters.quantile import (
    QuantileFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.results import (
    DirectSolverResult,
)
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d


class TestQuantileFitter:
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

    def test_fit_returns_direct_solver_result(self, bkd) -> None:
        """Fit returns DirectSolverResult."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        fitter = QuantileFitter(bkd, quantile=0.5)
        result = fitter.fit(expansion, samples, values)

        assert isinstance(result, DirectSolverResult)

    def test_result_params_shape(self, bkd) -> None:
        """Result params have correct shape."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        fitter = QuantileFitter(bkd, quantile=0.5)
        result = fitter.fit(expansion, samples, values)

        assert result.params().shape == (expansion.nterms(), 1)

    def test_handles_1d_values(self, bkd) -> None:
        """Fitter handles 1D values array."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values_1d = bkd.asarray(np.random.randn(50))

        fitter = QuantileFitter(bkd, quantile=0.5)
        result = fitter.fit(expansion, samples, values_1d)

        assert result.params().shape[1] == 1

    def test_fitted_surrogate_evaluates(self, bkd) -> None:
        """Fitted surrogate can evaluate at new points."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        fitter = QuantileFitter(bkd, quantile=0.5)
        result = fitter.fit(expansion, samples, values)

        # Evaluate at new points
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        predictions = result(test_samples)

        assert predictions.shape == (1, 10)

    def test_multi_qoi_raises(self, bkd) -> None:
        """nqoi > 1 raises ValueError."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3, nqoi=2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(2, 30))

        fitter = QuantileFitter(bkd, quantile=0.5)
        with pytest.raises(ValueError, match="nqoi=1"):
            fitter.fit(expansion, samples, values)

    def test_invalid_quantile_raises(self, bkd) -> None:
        """Quantile outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="(?i)quantile"):
            QuantileFitter(bkd, quantile=1.5)

        with pytest.raises(ValueError, match="(?i)quantile"):
            QuantileFitter(bkd, quantile=-0.1)

    def test_accessors(self, bkd) -> None:
        """Accessors return correct values."""
        fitter = QuantileFitter(bkd, quantile=0.75)
        bkd.assert_allclose(
            bkd.asarray([fitter.quantile()]),
            bkd.asarray([0.75]),
        )

    def test_median_finding_degree_zero(self, bkd) -> None:
        """Median of constant polynomial equals median of data.

        For degree 0 polynomial (constant), quantile regression should
        find the quantile of the data.
        """
        nvars = 1
        max_level = 0  # Degree 0: only constant term

        expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)
        # Verify it's just the constant term
        assert expansion.nterms() == 1

        # Create known data with clear median
        nsamples = 101  # Odd number for clear median
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        # Values: 0, 1, 2, ..., 100 (median = 50)
        values = bkd.reshape(
            bkd.arange(nsamples, dtype=bkd.asarray([0.0]).dtype),
            (1, nsamples),
        )

        # Fit with median (quantile=0.5)
        fitter = QuantileFitter(bkd, quantile=0.5)
        result = fitter.fit(expansion, samples, values)

        # The constant coefficient should equal the median
        expected_median = bkd.asarray([50.0])
        fitted_coef = result.params()[0:1, 0]
        bkd.assert_allclose(fitted_coef, expected_median, rtol=1e-10)

    def test_quantile_statistic_zero_at_solution(self, bkd) -> None:
        """The quantile risk measure statistic is approximately zero at solution.

        For quantile regression with check function rho_tau(u) = u(tau - I(u < 0)),
        at the optimal solution, the fraction of residuals below zero
        approximately equals tau (asymptotically).
        """
        nvars = 1
        max_level = 2

        expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)

        # Generate data
        nsamples = 200
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        # Add noise to make it interesting
        true_values = samples[0, :] ** 2  # quadratic
        noise = bkd.asarray(np.random.randn(nsamples) * 0.5)
        values = bkd.reshape(true_values + noise, (1, nsamples))

        # Test multiple quantiles
        for tau in [0.25, 0.5, 0.75]:
            fitter = QuantileFitter(bkd, quantile=tau)
            result = fitter.fit(expansion, samples, values)

            # Compute residuals
            predictions = result(samples)
            residuals = values - predictions  # (1, nsamples)

            # Fraction of residuals below zero should approximately equal tau
            below_zero = bkd.asarray(
                residuals[0, :] < 0, dtype=bkd.asarray([0.0]).dtype
            )
            fraction_below = bkd.sum(below_zero) / nsamples

            # Allow some tolerance since we have finite samples
            # Convert to array for assert_allclose
            bkd.assert_allclose(
                bkd.reshape(fraction_below, (1,)),
                bkd.asarray([tau]),
                atol=0.015,
            )

    def test_different_quantiles_give_different_results(self, bkd) -> None:
        """Different quantile levels produce different fitted surrogates."""
        nvars = 1
        max_level = 0  # Degree 0 for simplicity

        expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)

        # Generate data with spread
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        # Fit at different quantiles
        fitter_25 = QuantileFitter(bkd, quantile=0.25)
        fitter_50 = QuantileFitter(bkd, quantile=0.50)
        fitter_75 = QuantileFitter(bkd, quantile=0.75)

        result_25 = fitter_25.fit(expansion, samples, values)
        result_50 = fitter_50.fit(expansion, samples, values)
        result_75 = fitter_75.fit(expansion, samples, values)

        # The constant coefficients should be ordered: 25th < 50th < 75th
        coef_25 = result_25.params()[0, 0]
        coef_50 = result_50.params()[0, 0]
        coef_75 = result_75.params()[0, 0]

        assert coef_25 < coef_50
        assert coef_50 < coef_75
