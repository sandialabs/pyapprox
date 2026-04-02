"""Tests for BayesianConjugateFitter."""

import numpy as np
import pytest

from pyapprox.optimization.linear import RidgeRegressionSolver
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.expansions.fitters.bayesian import (
    BayesianConjugateFitter,
)
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d


class TestBayesianConjugateFitter:

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_expansion(self, bkd, nvars: int, max_level: int, nqoi: int = 1):
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_posterior_mean_equals_ridge(self, bkd) -> None:
        """Posterior mean equals Ridge solution when alpha = sigma^2/tau^2."""
        expansion_bayes = self._create_expansion(bkd, nvars=2, max_level=3)
        expansion_ridge = self._create_expansion(bkd, nvars=2, max_level=3)

        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values = bkd.asarray(np.random.randn(1, 50))

        # Prior variance and noise variance
        tau_sq = 10.0  # Prior variance
        sigma_sq = 0.1  # Noise variance
        alpha = sigma_sq / tau_sq  # Ridge regularization

        # Bayesian fitter
        bayes_fitter = BayesianConjugateFitter(
            bkd,
            prior_var=tau_sq,
            noise_var=sigma_sq,
        )
        bayes_result = bayes_fitter.fit(expansion_bayes, samples, values)

        # Ridge solver (direct)
        Phi = expansion_ridge.basis_matrix(samples)
        ridge_solver = RidgeRegressionSolver(bkd, alpha=alpha)
        ridge_coef = ridge_solver.solve(Phi, values.T)

        # Compare
        bkd.assert_allclose(
            bayes_result.posterior_mean(),
            ridge_coef,
            rtol=1e-8,
        )

    def test_posterior_covariance_shape(self, bkd) -> None:
        """Posterior covariance has correct shape."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        nterms = expansion.nterms()

        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        fitter = BayesianConjugateFitter(bkd, prior_var=1.0, noise_var=0.1)
        result = fitter.fit(expansion, samples, values)

        cov = result.posterior_covariance()
        assert cov.shape == (nterms, nterms)

    def test_sample_posterior(self, bkd) -> None:
        """Can sample from posterior distribution."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)

        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        fitter = BayesianConjugateFitter(bkd, prior_var=1.0, noise_var=0.1)
        result = fitter.fit(expansion, samples, values)

        # Sample from posterior
        posterior_dist = result.posterior_variable()
        n_draws = 100
        draws = posterior_dist.rvs(n_draws)

        assert draws.shape[1] == n_draws

    def test_fitted_surrogate_evaluates(self, bkd) -> None:
        """Fitted surrogate can evaluate at new points."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)

        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        fitter = BayesianConjugateFitter(bkd, prior_var=1.0, noise_var=0.1)
        result = fitter.fit(expansion, samples, values)

        # Evaluate fitted surrogate
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        predictions = result.surrogate()(test_samples)

        assert predictions.shape == (1, 10)

    def test_evidence_is_positive(self, bkd) -> None:
        """Model evidence is positive."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=2)

        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        values = bkd.asarray(np.random.randn(1, 20))

        fitter = BayesianConjugateFitter(bkd, prior_var=1.0, noise_var=0.1)
        result = fitter.fit(expansion, samples, values)

        evidence = result.evidence()
        assert evidence > 0.0

    def test_handles_1d_values(self, bkd) -> None:
        """Fitter handles 1D values array."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values_1d = bkd.asarray(np.random.randn(30))  # 1D

        fitter = BayesianConjugateFitter(bkd, prior_var=1.0, noise_var=0.1)
        result = fitter.fit(expansion, samples, values_1d)

        # Should work
        assert result.posterior_mean().shape == (expansion.nterms(), 1)
