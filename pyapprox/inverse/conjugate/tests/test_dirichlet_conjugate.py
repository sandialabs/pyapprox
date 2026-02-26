"""
Tests for Dirichlet conjugate posterior.
"""

import pytest

import numpy as np
from scipy import stats

from pyapprox.inverse.conjugate.dirichlet import DirichletConjugatePosterior


class TestDirichletConjugateBase:
    """Base test class for DirichletConjugatePosterior."""

    def _make_solver(self, bkd):
        """Create solver and observations for tests."""
        # Uniform prior over 3 categories
        alphas_prior = bkd.asarray([1.0, 1.0, 1.0])
        solver = DirichletConjugatePosterior(alphas_prior, bkd)

        # Observations: category indices
        # 3 in cat 0, 1 in cat 1, 1 in cat 2
        obs = bkd.asarray([[0, 0, 1, 2, 0]])
        return solver, obs

    def test_nvars(self, bkd) -> None:
        """Test nvars returns number of categories."""
        solver, _ = self._make_solver(bkd)
        assert solver.nvars() == 3

    def test_posterior_alphas(self, bkd) -> None:
        """Test posterior alphas = prior_alphas + counts."""
        solver, obs = self._make_solver(bkd)
        solver.compute(obs)
        post_alphas = bkd.to_numpy(solver.posterior_alphas())
        # Expected: [1+3, 1+1, 1+1] = [4, 2, 2]
        np.testing.assert_array_almost_equal(post_alphas, [4.0, 2.0, 2.0])

    def test_posterior_mean(self, bkd) -> None:
        """Test posterior mean = alpha_k / sum(alpha)."""
        solver, obs = self._make_solver(bkd)
        solver.compute(obs)
        post_mean = bkd.to_numpy(solver.posterior_mean())
        # Expected: [4/8, 2/8, 2/8] = [0.5, 0.25, 0.25]
        np.testing.assert_array_almost_equal(post_mean, [0.5, 0.25, 0.25])

    def test_posterior_mean_sums_to_one(self, bkd) -> None:
        """Test posterior mean is a proper probability distribution."""
        solver, obs = self._make_solver(bkd)
        solver.compute(obs)
        post_mean = bkd.to_numpy(solver.posterior_mean())
        assert sum(post_mean) == pytest.approx(1.0, abs=1e-5)

    def test_evidence_positive(self, bkd) -> None:
        """Test evidence is positive."""
        solver, obs = self._make_solver(bkd)
        solver.compute(obs)
        assert solver.evidence() > 0

    def test_posterior_variable(self, bkd) -> None:
        """Test posterior_variable returns scipy Dirichlet distribution."""
        solver, obs = self._make_solver(bkd)
        solver.compute(obs)
        post = solver.posterior_variable()
        assert callable(post.pdf)

    def test_compute_not_called_raises(self, bkd) -> None:
        """Test accessing results before compute raises error."""
        solver, _ = self._make_solver(bkd)
        with pytest.raises(RuntimeError):
            solver.posterior_alphas()

    def test_invalid_category_raises(self, bkd) -> None:
        """Test observation with invalid category raises error."""
        solver, _ = self._make_solver(bkd)
        bad_obs = bkd.asarray([[0, 1, 5]])  # 5 is out of range
        with pytest.raises(ValueError):
            solver.compute(bad_obs)


class TestDirichletConjugateVsScipy:
    """Test against scipy.stats.dirichlet."""

    def test_posterior_matches_scipy(self, bkd) -> None:
        """Test posterior matches scipy Dirichlet."""
        alphas = bkd.asarray([2.0, 3.0, 1.0])
        solver = DirichletConjugatePosterior(alphas, bkd)

        # 2 in cat 0, 1 in cat 1, 3 in cat 2
        obs = bkd.asarray([[0, 0, 1, 2, 2, 2]])
        solver.compute(obs)

        # Expected: Dirichlet(2+2, 3+1, 1+3) = Dirichlet(4, 4, 4)
        expected_dist = stats.dirichlet([4, 4, 4])

        post_mean = bkd.to_numpy(solver.posterior_mean())
        np.testing.assert_array_almost_equal(post_mean, expected_dist.mean(), decimal=5)
