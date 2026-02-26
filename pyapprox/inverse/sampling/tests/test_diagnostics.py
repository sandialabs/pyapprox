"""
Tests for MCMC diagnostic functions.
"""

import pytest

import numpy as np

from pyapprox.inverse.sampling.diagnostics import (
    MCMCDiagnostics,
    autocorrelation,
    compute_diagnostics,
    effective_sample_size,
    rhat,
)


class TestAutocorrelationBase:
    """Base test class for autocorrelation."""

    def test_lag_zero_is_one(self, bkd) -> None:
        """Test autocorrelation at lag 0 is always 1."""
        samples = bkd.asarray(np.random.randn(2, 100))
        acf = autocorrelation(samples, max_lag=10, bkd=bkd)
        acf_np = bkd.to_numpy(acf)
        np.testing.assert_allclose(acf_np[:, 0], 1.0, rtol=1e-10)

    def test_independent_samples_low_autocorrelation(self, bkd) -> None:
        """Test independent samples have low autocorrelation at lag > 0."""
        np.random.seed(42)
        samples = bkd.asarray(np.random.randn(1, 1000))
        acf = autocorrelation(samples, max_lag=50, bkd=bkd)
        acf_np = bkd.to_numpy(acf)
        # For independent samples, autocorrelation should be near 0
        np.testing.assert_array_less(np.abs(acf_np[0, 10:]), 0.15)

    def test_ar1_process(self, bkd) -> None:
        """Test autocorrelation for AR(1) process matches theory."""
        np.random.seed(123)
        n = 5000
        phi = 0.7  # AR(1) coefficient

        # Generate AR(1): x_t = phi * x_{t-1} + epsilon_t
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + np.random.randn()

        samples = bkd.asarray(x.reshape(1, -1))
        acf = autocorrelation(samples, max_lag=10, bkd=bkd)
        acf_np = bkd.to_numpy(acf)[0, :]

        # Theoretical autocorrelation for AR(1): rho_k = phi^k
        for k in range(1, 6):
            expected = phi**k
            np.testing.assert_allclose(acf_np[k], expected, atol=0.1)

    def test_1d_input(self, bkd) -> None:
        """Test autocorrelation works with 1D input."""
        samples = bkd.asarray(np.random.randn(100))
        acf = autocorrelation(samples, max_lag=10, bkd=bkd)
        assert acf.shape == (11,)


class TestESSBase:
    """Base test class for effective sample size."""

    def test_independent_samples_ess_near_n(self, bkd) -> None:
        """Test ESS is close to n for independent samples."""
        np.random.seed(42)
        n = 1000
        samples = bkd.asarray(np.random.randn(2, n))
        ess = effective_sample_size(samples, bkd=bkd)
        ess_np = bkd.to_numpy(ess)
        # ESS should be close to n for independent samples
        assert np.all(ess_np > 0.5 * n)

    def test_correlated_samples_lower_ess(self, bkd) -> None:
        """Test correlated samples have lower ESS."""
        np.random.seed(123)
        n = 2000
        phi = 0.9  # High autocorrelation

        # Generate AR(1)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + np.random.randn()

        samples = bkd.asarray(x.reshape(1, -1))
        ess = effective_sample_size(samples, bkd=bkd)
        ess_val = float(bkd.to_numpy(ess)[0])
        # ESS should be much less than n for highly correlated samples
        assert ess_val < 0.3 * n

    def test_ess_positive(self, bkd) -> None:
        """Test ESS is always positive."""
        samples = bkd.asarray(np.random.randn(3, 500))
        ess = effective_sample_size(samples, bkd=bkd)
        ess_np = bkd.to_numpy(ess)
        assert np.all(ess_np > 0)


class TestRhatBase:
    """Base test class for R-hat."""

    def test_identical_chains_rhat_one(self, bkd) -> None:
        """Test R-hat is approximately 1 for converged chains."""
        np.random.seed(42)
        # Two chains from same distribution
        chain1 = bkd.asarray(np.random.randn(2, 500))
        chain2 = bkd.asarray(np.random.randn(2, 500))
        rhat_val = rhat([chain1, chain2], bkd=bkd)
        rhat_np = bkd.to_numpy(rhat_val)
        # R-hat should be close to 1
        np.testing.assert_array_less(rhat_np, 1.1)

    def test_divergent_chains_high_rhat(self, bkd) -> None:
        """Test R-hat is high for divergent chains."""
        # Two chains with different means
        chain1 = bkd.asarray(np.random.randn(1, 500))
        chain2 = bkd.asarray(np.random.randn(1, 500) + 5.0)
        rhat_val = rhat([chain1, chain2], bkd=bkd)
        rhat_np = bkd.to_numpy(rhat_val)
        # R-hat should be much greater than 1
        assert rhat_np[0] > 1.5

    def test_requires_two_chains(self, bkd) -> None:
        """Test R-hat raises error with fewer than 2 chains."""
        chain = bkd.asarray(np.random.randn(2, 100))
        with pytest.raises(ValueError):
            rhat([chain], bkd=bkd)

    def test_multiple_chains(self, bkd) -> None:
        """Test R-hat works with more than 2 chains."""
        np.random.seed(42)
        chains = [bkd.asarray(np.random.randn(2, 300)) for _ in range(4)]
        rhat_val = rhat(chains, bkd=bkd)
        rhat_np = bkd.to_numpy(rhat_val)
        np.testing.assert_array_less(rhat_np, 1.1)


class TestComputeDiagnosticsBase:
    """Base test class for compute_diagnostics."""

    def test_returns_dataclass(self, bkd) -> None:
        """Test compute_diagnostics returns MCMCDiagnostics."""
        samples = bkd.asarray(np.random.randn(2, 500))
        diag = compute_diagnostics(samples, bkd=bkd)
        assert isinstance(diag, MCMCDiagnostics)

    def test_no_rhat_without_other_chains(self, bkd) -> None:
        """Test rhat is None when no other chains provided."""
        samples = bkd.asarray(np.random.randn(2, 500))
        diag = compute_diagnostics(samples, bkd=bkd)
        assert diag.rhat is None

    def test_rhat_with_other_chains(self, bkd) -> None:
        """Test rhat is computed when other chains provided."""
        np.random.seed(42)
        samples = bkd.asarray(np.random.randn(2, 500))
        other = [bkd.asarray(np.random.randn(2, 500))]
        diag = compute_diagnostics(samples, bkd=bkd, other_chains=other)
        assert diag.rhat is not None
