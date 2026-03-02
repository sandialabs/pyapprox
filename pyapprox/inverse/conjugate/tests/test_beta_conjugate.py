"""
Tests for Beta conjugate posterior.
"""

import pytest
from scipy import stats

from pyapprox.inverse.conjugate.beta import BetaConjugatePosterior


class TestBetaConjugateBase:
    """Base test class for BetaConjugatePosterior."""

    def _make_solver(self, bkd):
        """Create solver and observations for tests."""
        # Uniform prior Beta(1, 1)
        alpha_prior = 1.0
        beta_prior = 1.0
        solver = BetaConjugatePosterior(alpha_prior, beta_prior, bkd)

        # Observations: 3 successes, 2 failures
        obs = bkd.asarray([[1, 1, 0, 1, 0]])
        return solver, obs

    def test_nvars(self, bkd) -> None:
        """Test nvars returns 1."""
        solver, _ = self._make_solver(bkd)
        assert solver.nvars() == 1

    def test_posterior_alpha(self, bkd) -> None:
        """Test posterior alpha = prior_alpha + successes."""
        solver, obs = self._make_solver(bkd)
        solver.compute(obs)
        # 3 successes
        assert solver.posterior_alpha() == pytest.approx(1.0 + 3.0, abs=1e-5)

    def test_posterior_beta(self, bkd) -> None:
        """Test posterior beta = prior_beta + failures."""
        solver, obs = self._make_solver(bkd)
        solver.compute(obs)
        # 2 failures
        assert solver.posterior_beta() == pytest.approx(1.0 + 2.0, abs=1e-5)

    def test_posterior_mean(self, bkd) -> None:
        """Test posterior mean = alpha / (alpha + beta)."""
        solver, obs = self._make_solver(bkd)
        solver.compute(obs)
        expected = 4.0 / (4.0 + 3.0)  # Beta(4, 3)
        assert solver.posterior_mean() == pytest.approx(expected, abs=1e-5)

    def test_posterior_variance(self, bkd) -> None:
        """Test posterior variance formula."""
        solver, obs = self._make_solver(bkd)
        solver.compute(obs)
        a, b = 4.0, 3.0  # Posterior Beta(4, 3)
        expected = (a * b) / ((a + b) ** 2 * (a + b + 1))
        assert solver.posterior_variance() == pytest.approx(expected, abs=1e-5)

    def test_evidence_positive(self, bkd) -> None:
        """Test evidence is positive."""
        solver, obs = self._make_solver(bkd)
        solver.compute(obs)
        assert solver.evidence() > 0

    def test_posterior_variable(self, bkd) -> None:
        """Test posterior_variable returns scipy Beta distribution."""
        solver, obs = self._make_solver(bkd)
        solver.compute(obs)
        post = solver.posterior_variable()
        # Check it's callable (has pdf method)
        assert callable(post.pdf)

    def test_compute_not_called_raises(self, bkd) -> None:
        """Test accessing results before compute raises error."""
        solver, _ = self._make_solver(bkd)
        with pytest.raises(RuntimeError):
            solver.posterior_alpha()


class TestBetaConjugateVsScipy:
    """Test against scipy.stats.beta."""

    def test_posterior_matches_scipy(self, bkd) -> None:
        """Test posterior matches scipy Beta."""
        alpha, beta = 2.0, 3.0
        solver = BetaConjugatePosterior(alpha, beta, bkd)

        obs = bkd.asarray([[1, 0, 1, 1, 0, 0]])  # 3 successes, 3 failures
        solver.compute(obs)

        # Expected: Beta(2+3, 3+3) = Beta(5, 6)
        expected_dist = stats.beta(5, 6)

        assert solver.posterior_mean() == pytest.approx(expected_dist.mean(), abs=1e-5)
        assert solver.posterior_variance() == pytest.approx(
            expected_dist.var(), abs=1e-5
        )
