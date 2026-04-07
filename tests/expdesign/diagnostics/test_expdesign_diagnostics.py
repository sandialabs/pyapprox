"""Tests for KLOEDDiagnostics and PredictionOEDDiagnostics.

Tests verify the new raw-sample API: diagnostics accept pre-generated
arrays, not benchmarks with generate_data methods.
"""

from typing import Tuple

import numpy as np
import pytest

from pyapprox_benchmarks.functions.algebraic.linear_gaussian_oed import (
    _build_vandermonde,
)
from pyapprox_benchmarks.instances.oed.linear_gaussian import (
    build_linear_gaussian_kl_benchmark,
)
from pyapprox.expdesign.diagnostics import (
    KLOEDDiagnostics,
    compute_convergence_rate,
    compute_estimator_mse,
)
from pyapprox.expdesign.diagnostics.prediction_diagnostics import (
    create_prediction_oed_diagnostics,
    get_registered_utility_types,
)
from pyapprox.util.backends.protocols import Array, Backend


def _generate_samples(
    nobs: int,
    nparams: int,
    nsamples: int,
    noise_std: float,
    prior_std: float,
    bkd: Backend[Array],
    seed: int,
) -> Tuple[Array, Array]:
    """Generate outer_shapes, latent_samples, and inner_shapes for testing."""
    np.random.seed(seed)
    # Prior samples: theta ~ N(0, prior_std^2 I)
    theta = bkd.asarray(prior_std * np.random.randn(nparams, nsamples))
    # obs_map: y = A @ theta (Vandermonde)
    obs_locations = bkd.linspace(-1.0, 1.0, nobs)
    degree = nparams - 1
    A = _build_vandermonde(obs_locations, 0, degree, bkd)
    shapes = bkd.dot(A, theta)
    # Latent samples: z ~ N(0, I)
    latent = bkd.asarray(np.random.randn(nobs, nsamples))
    return shapes, latent


class TestKLOEDDiagnostics:
    """Tests for KLOEDDiagnostics with raw sample API."""

    def _setup(self, bkd: Backend[Array]) -> None:
        self._nobs = 5
        self._degree = 2
        self._nparams = self._degree + 1
        self._noise_std = 0.5
        self._prior_std = 0.5

    def _noise_variances(self, bkd: Backend[Array]) -> Array:
        return bkd.full((self._nobs,), self._noise_std**2)

    def test_compute_numerical_eig_finite(
        self, bkd: Backend[Array],
    ) -> None:
        """Numerical EIG is finite."""
        self._setup(bkd)
        diag = KLOEDDiagnostics(self._noise_variances(bkd), bkd)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        outer_shapes, latent = _generate_samples(
            self._nobs, self._nparams, 50,
            self._noise_std, self._prior_std, bkd, seed=42,
        )
        inner_shapes, _ = _generate_samples(
            self._nobs, self._nparams, 30,
            self._noise_std, self._prior_std, bkd, seed=123,
        )

        eig = diag.compute_numerical_eig(
            weights, outer_shapes, latent, inner_shapes,
        )
        assert np.isfinite(eig)

    def test_compute_numerical_eig_reproducible(
        self, bkd: Backend[Array],
    ) -> None:
        """Same inputs produce same EIG."""
        self._setup(bkd)
        diag = KLOEDDiagnostics(self._noise_variances(bkd), bkd)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        outer_shapes, latent = _generate_samples(
            self._nobs, self._nparams, 50,
            self._noise_std, self._prior_std, bkd, seed=42,
        )
        inner_shapes, _ = _generate_samples(
            self._nobs, self._nparams, 30,
            self._noise_std, self._prior_std, bkd, seed=123,
        )

        eig1 = diag.compute_numerical_eig(
            weights, outer_shapes, latent, inner_shapes,
        )
        eig2 = diag.compute_numerical_eig(
            weights, outer_shapes, latent, inner_shapes,
        )
        bkd.assert_allclose(
            bkd.asarray([eig1]), bkd.asarray([eig2]), rtol=1e-12,
        )

    def test_compute_numerical_eig_varies_with_samples(
        self, bkd: Backend[Array],
    ) -> None:
        """Different samples produce different EIG."""
        self._setup(bkd)
        diag = KLOEDDiagnostics(self._noise_variances(bkd), bkd)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        outer1, latent1 = _generate_samples(
            self._nobs, self._nparams, 50,
            self._noise_std, self._prior_std, bkd, seed=42,
        )
        inner1, _ = _generate_samples(
            self._nobs, self._nparams, 30,
            self._noise_std, self._prior_std, bkd, seed=123,
        )
        outer2, latent2 = _generate_samples(
            self._nobs, self._nparams, 50,
            self._noise_std, self._prior_std, bkd, seed=999,
        )
        inner2, _ = _generate_samples(
            self._nobs, self._nparams, 30,
            self._noise_std, self._prior_std, bkd, seed=888,
        )

        eig1 = diag.compute_numerical_eig(weights, outer1, latent1, inner1)
        eig2 = diag.compute_numerical_eig(weights, outer2, latent2, inner2)
        assert abs(eig1 - eig2) > 1e-5

    def test_bkd_accessor(self, bkd: Backend[Array]) -> None:
        """bkd() returns the backend."""
        self._setup(bkd)
        diag = KLOEDDiagnostics(self._noise_variances(bkd), bkd)
        assert diag.bkd() is bkd

    def test_numerical_eig_close_to_exact(
        self, bkd: Backend[Array],
    ) -> None:
        """Numerical EIG with many samples is close to exact."""
        self._setup(bkd)
        bench = build_linear_gaussian_kl_benchmark(
            self._nobs, self._degree, self._noise_std, self._prior_std, bkd,
        )
        weights = bkd.ones((self._nobs, 1)) / self._nobs
        exact_eig = bench.exact_eig(weights)

        diag = KLOEDDiagnostics(self._noise_variances(bkd), bkd)

        outer_shapes, latent = _generate_samples(
            self._nobs, self._nparams, 500,
            self._noise_std, self._prior_std, bkd, seed=42,
        )
        inner_shapes, _ = _generate_samples(
            self._nobs, self._nparams, 500,
            self._noise_std, self._prior_std, bkd, seed=123,
        )

        numerical_eig = diag.compute_numerical_eig(
            weights, outer_shapes, latent, inner_shapes,
        )

        # Should be within 30% of exact (MC has variance)
        relative_error = abs(numerical_eig - exact_eig) / exact_eig
        assert relative_error < 0.3


class TestDiagnosticUtils:
    """Tests for shared diagnostic utilities."""

    def test_compute_estimator_mse_decomposition(
        self, bkd: Backend[Array],
    ) -> None:
        """MSE = bias^2 + variance."""
        estimates = [1.1, 0.9, 1.05, 0.95, 1.0]
        exact = 1.0
        bias, variance, mse = compute_estimator_mse(exact, estimates)

        expected_mse = bias**2 + variance
        bkd.assert_allclose(
            bkd.asarray([mse]), bkd.asarray([expected_mse]), rtol=1e-10,
        )
        assert variance >= 0.0

    def test_compute_estimator_mse_unbiased(
        self, bkd: Backend[Array],
    ) -> None:
        """Unbiased estimator has bias near zero."""
        estimates = [1.0, 1.0, 1.0]
        exact = 1.0
        bias, variance, mse = compute_estimator_mse(exact, estimates)
        bkd.assert_allclose(
            bkd.asarray([bias]), bkd.asarray([0.0]), atol=1e-15,
        )

    def test_convergence_rate_o1n(self, bkd: Backend[Array]) -> None:
        """O(1/n) data gives rate ~1."""
        sample_counts = [10, 20, 40, 80, 160]
        values = [1.0 / n for n in sample_counts]
        rate = compute_convergence_rate(sample_counts, values)
        bkd.assert_allclose(
            bkd.asarray([rate]), bkd.asarray([1.0]), rtol=1e-10,
        )

    def test_convergence_rate_o1n2(self, bkd: Backend[Array]) -> None:
        """O(1/n^2) data gives rate ~2."""
        sample_counts = [10, 20, 40, 80, 160]
        values = [1.0 / (n**2) for n in sample_counts]
        rate = compute_convergence_rate(sample_counts, values)
        bkd.assert_allclose(
            bkd.asarray([rate]), bkd.asarray([2.0]), rtol=1e-10,
        )

    def test_convergence_rate_o1sqrtn(
        self, bkd: Backend[Array],
    ) -> None:
        """O(1/sqrt(n)) data gives rate ~0.5."""
        sample_counts = [10, 20, 40, 80, 160]
        values = [1.0 / np.sqrt(n) for n in sample_counts]
        rate = compute_convergence_rate(sample_counts, values)
        bkd.assert_allclose(
            bkd.asarray([rate]), bkd.asarray([0.5]), rtol=1e-10,
        )


class TestPredictionOEDDiagnosticsFactory:
    """Tests for prediction OED diagnostics factory."""

    def test_unknown_utility_type_raises(
        self, bkd: Backend[Array],
    ) -> None:
        """Unknown utility type raises ValueError."""
        noise_variances = bkd.full((5,), 0.25)
        with pytest.raises(ValueError):
            create_prediction_oed_diagnostics(
                noise_variances, 1, "nonexistent_type", bkd,
            )

    def test_registered_utility_types(
        self, bkd: Backend[Array],
    ) -> None:
        """All expected utility types are registered."""
        types = get_registered_utility_types()
        assert "nonlinear_mean_stdev" in types
        assert "nonlinear_avar_stdev" in types
        assert "linear_stdev" in types
        assert "linear_avar" in types

    def test_create_diagnostics_returns_correct_type(
        self, bkd: Backend[Array],
    ) -> None:
        """Factory returns PredictionOEDDiagnostics."""
        from pyapprox.expdesign.diagnostics import PredictionOEDDiagnostics

        noise_variances = bkd.full((5,), 0.25)
        diag = create_prediction_oed_diagnostics(
            noise_variances, 1, "nonlinear_mean_stdev", bkd,
        )
        assert isinstance(diag, PredictionOEDDiagnostics)
