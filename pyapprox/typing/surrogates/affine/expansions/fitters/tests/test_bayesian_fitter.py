"""Tests for BayesianConjugateFitter."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.surrogates.affine.expansions.fitters.bayesian import (
    BayesianConjugateFitter,
)
from pyapprox.typing.optimization.linear import RidgeRegressionSolver

from pyapprox.typing.surrogates.affine.univariate import create_bases_1d
from pyapprox.typing.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.typing.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.typing.surrogates.affine.expansions import BasisExpansion
from pyapprox.typing.probability import UniformMarginal


class TestBayesianConjugateFitter(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_expansion(self, nvars: int, max_level: int, nqoi: int = 1):
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_posterior_mean_equals_ridge(self) -> None:
        """Posterior mean equals Ridge solution when alpha = sigma^2/tau^2."""
        expansion_bayes = self._create_expansion(nvars=2, max_level=3)
        expansion_ridge = self._create_expansion(nvars=2, max_level=3)

        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values = self._bkd.asarray(np.random.randn(1, 50))

        # Prior variance and noise variance
        tau_sq = 10.0  # Prior variance
        sigma_sq = 0.1  # Noise variance
        alpha = sigma_sq / tau_sq  # Ridge regularization

        # Bayesian fitter
        bayes_fitter = BayesianConjugateFitter(
            self._bkd,
            prior_var=tau_sq,
            noise_var=sigma_sq,
        )
        bayes_result = bayes_fitter.fit(expansion_bayes, samples, values)

        # Ridge solver (direct)
        Phi = expansion_ridge.basis_matrix(samples)
        ridge_solver = RidgeRegressionSolver(self._bkd, alpha=alpha)
        ridge_coef = ridge_solver.solve(Phi, values.T)

        # Compare
        self._bkd.assert_allclose(
            bayes_result.posterior_mean(),
            ridge_coef,
            rtol=1e-8,
        )

    def test_posterior_covariance_shape(self) -> None:
        """Posterior covariance has correct shape."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        nterms = expansion.nterms()

        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(1, 30))

        fitter = BayesianConjugateFitter(
            self._bkd, prior_var=1.0, noise_var=0.1
        )
        result = fitter.fit(expansion, samples, values)

        cov = result.posterior_covariance()
        self.assertEqual(cov.shape, (nterms, nterms))

    def test_sample_posterior(self) -> None:
        """Can sample from posterior distribution."""
        expansion = self._create_expansion(nvars=2, max_level=3)

        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(1, 30))

        fitter = BayesianConjugateFitter(
            self._bkd, prior_var=1.0, noise_var=0.1
        )
        result = fitter.fit(expansion, samples, values)

        # Sample from posterior
        posterior_dist = result.posterior_variable()
        n_draws = 100
        draws = posterior_dist.rvs(n_draws)

        self.assertEqual(draws.shape[1], n_draws)

    def test_fitted_surrogate_evaluates(self) -> None:
        """Fitted surrogate can evaluate at new points."""
        expansion = self._create_expansion(nvars=2, max_level=3)

        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(1, 30))

        fitter = BayesianConjugateFitter(
            self._bkd, prior_var=1.0, noise_var=0.1
        )
        result = fitter.fit(expansion, samples, values)

        # Evaluate fitted surrogate
        test_samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        predictions = result.surrogate()(test_samples)

        self.assertEqual(predictions.shape, (1, 10))

    def test_evidence_is_positive(self) -> None:
        """Model evidence is positive."""
        expansion = self._create_expansion(nvars=2, max_level=2)

        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        values = self._bkd.asarray(np.random.randn(1, 20))

        fitter = BayesianConjugateFitter(
            self._bkd, prior_var=1.0, noise_var=0.1
        )
        result = fitter.fit(expansion, samples, values)

        evidence = result.evidence()
        self.assertGreater(evidence, 0.0)

    def test_handles_1d_values(self) -> None:
        """Fitter handles 1D values array."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values_1d = self._bkd.asarray(np.random.randn(30))  # 1D

        fitter = BayesianConjugateFitter(
            self._bkd, prior_var=1.0, noise_var=0.1
        )
        result = fitter.fit(expansion, samples, values_1d)

        # Should work
        self.assertEqual(result.posterior_mean().shape, (expansion.nterms(), 1))


class TestBayesianConjugateFitterNumpy(
    TestBayesianConjugateFitter[NDArray[Any]]
):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestBayesianConjugateFitterTorch(
    TestBayesianConjugateFitter[torch.Tensor]
):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
