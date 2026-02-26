"""
Tests for Gaussian conjugate posterior.
"""

import pytest

import numpy as np

from pyapprox.inverse.conjugate.gaussian import DenseGaussianConjugatePosterior


class TestDenseGaussianConjugateBase:
    """Base test class for DenseGaussianConjugatePosterior."""

    def _make_solver(self, bkd):
        """Create solver and observations for tests."""
        nvars = 2
        nobs = 3

        # Linear model: y = A @ x + offset
        A = bkd.asarray([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]])
        offset = bkd.zeros((nobs, 1))

        # Prior
        prior_mean = bkd.zeros((nvars, 1))
        prior_cov = bkd.eye(nvars)

        # Noise
        noise_var = 0.1
        noise_cov = noise_var * bkd.eye(nobs)

        # Create solver
        solver = DenseGaussianConjugatePosterior(
            A,
            prior_mean,
            prior_cov,
            noise_cov,
            bkd,
            offset,
        )

        # Observations
        obs = bkd.asarray([[1.0], [1.5], [2.0]])

        return solver, obs, nvars, nobs, prior_mean

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct value."""
        solver, _, nvars, _, _ = self._make_solver(bkd)
        assert solver.nvars() == nvars

    def test_nobs(self, bkd) -> None:
        """Test nobs returns correct value."""
        solver, _, _, nobs, _ = self._make_solver(bkd)
        assert solver.nobs() == nobs

    def test_posterior_mean_shape(self, bkd) -> None:
        """Test posterior mean has correct shape."""
        solver, obs, nvars, _, _ = self._make_solver(bkd)
        solver.compute(obs)
        mean = solver.posterior_mean()
        assert mean.shape == (nvars, 1)

    def test_posterior_covariance_shape(self, bkd) -> None:
        """Test posterior covariance has correct shape."""
        solver, obs, nvars, _, _ = self._make_solver(bkd)
        solver.compute(obs)
        cov = solver.posterior_covariance()
        assert cov.shape == (nvars, nvars)

    def test_posterior_covariance_symmetric(self, bkd) -> None:
        """Test posterior covariance is symmetric."""
        solver, obs, _, _, _ = self._make_solver(bkd)
        solver.compute(obs)
        cov = solver.posterior_covariance()
        cov_np = bkd.to_numpy(cov)
        np.testing.assert_array_almost_equal(cov_np, cov_np.T)

    def test_posterior_covariance_positive_definite(self, bkd) -> None:
        """Test posterior covariance is positive definite."""
        solver, obs, _, _, _ = self._make_solver(bkd)
        solver.compute(obs)
        cov = solver.posterior_covariance()
        cov_np = bkd.to_numpy(cov)
        eigenvalues = np.linalg.eigvalsh(cov_np)
        assert all(eigenvalues > 0)

    def test_evidence_positive(self, bkd) -> None:
        """Test evidence is positive."""
        solver, obs, _, _, _ = self._make_solver(bkd)
        solver.compute(obs)
        evidence = solver.evidence()
        assert evidence > 0

    def test_expected_kl_divergence_nonnegative(self, bkd) -> None:
        """Test expected KL divergence is non-negative."""
        solver, obs, _, _, _ = self._make_solver(bkd)
        solver.compute(obs)
        kl = solver.expected_kl_divergence()
        assert kl >= 0

    def test_posterior_variable_returns_gaussian(self, bkd) -> None:
        """Test posterior_variable returns a Gaussian distribution."""
        solver, obs, _, _, _ = self._make_solver(bkd)
        solver.compute(obs)
        post = solver.posterior_variable()
        # Check it has the expected methods
        assert hasattr(post, "logpdf")
        assert hasattr(post, "rvs")

    def test_compute_not_called_raises(self, bkd) -> None:
        """Test accessing results before compute raises error."""
        solver, _, _, _, _ = self._make_solver(bkd)
        with pytest.raises(RuntimeError):
            solver.posterior_mean()

    def test_wrong_obs_shape_raises(self, bkd) -> None:
        """Test wrong observation shape raises error."""
        solver, _, _, _, _ = self._make_solver(bkd)
        bad_obs = bkd.zeros((5, 1))  # Wrong nobs
        with pytest.raises(ValueError):
            solver.compute(bad_obs)

    def test_posterior_differs_from_prior(self, bkd) -> None:
        """Test posterior mean differs from prior mean after data."""
        solver, obs, _, _, prior_mean = self._make_solver(bkd)
        solver.compute(obs)
        post_mean = bkd.to_numpy(solver.posterior_mean())
        prior_mean_np = bkd.to_numpy(prior_mean)

        # Posterior should move toward data direction
        diff = np.linalg.norm(post_mean - prior_mean_np)
        assert diff > 0.01  # Not identical to prior


class TestDenseGaussianConjugateAnalytical:
    """Test against analytical formulas."""

    def _make_solver(self, bkd):
        """Create 1D solver for analytical verification."""
        A = bkd.asarray([[1.0]])
        prior_mean = bkd.asarray([[0.0]])
        prior_var = 1.0
        prior_cov = prior_var * bkd.eye(1)
        noise_var = 0.5
        noise_cov = noise_var * bkd.eye(1)

        solver = DenseGaussianConjugatePosterior(
            A, prior_mean, prior_cov, noise_cov, bkd
        )
        return solver, prior_var, noise_var

    def test_posterior_mean_analytical(self, bkd) -> None:
        """Test posterior mean matches analytical formula."""
        solver, prior_var, noise_var = self._make_solver(bkd)
        obs = bkd.asarray([[2.0]])
        solver.compute(obs)

        # Analytical: post_mean = prior_var / (noise_var + prior_var) * obs
        expected = prior_var / (noise_var + prior_var) * 2.0

        post_mean = bkd.to_numpy(solver.posterior_mean())
        assert post_mean[0, 0] == pytest.approx(expected, abs=1e-5)

    def test_posterior_variance_analytical(self, bkd) -> None:
        """Test posterior variance matches analytical formula."""
        solver, prior_var, noise_var = self._make_solver(bkd)
        obs = bkd.asarray([[2.0]])
        solver.compute(obs)

        # Analytical: post_var = (1/prior_var + 1/noise_var)^{-1}
        expected = 1.0 / (1.0 / prior_var + 1.0 / noise_var)

        post_cov = bkd.to_numpy(solver.posterior_covariance())
        assert post_cov[0, 0] == pytest.approx(expected, abs=1e-5)
