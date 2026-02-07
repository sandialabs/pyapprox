"""
Tests for recovering conjugate posteriors via variational inference.

These tests verify that optimizing the ELBO with a Gaussian variational
family recovers the exact Gaussian conjugate posterior for linear models.
"""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import slow_test
from pyapprox.typing.probability.gaussian.diagonal import (
    DiagonalMultivariateGaussian,
)
from pyapprox.typing.probability.likelihood.gaussian import (
    DiagonalGaussianLogLikelihood,
    MultiExperimentLogLikelihood,
)
from pyapprox.typing.inverse.conjugate.gaussian import (
    DenseGaussianConjugatePosterior,
)
from pyapprox.typing.inverse.variational.gaussian_family import (
    GaussianVariationalFamily,
)
from pyapprox.typing.inverse.variational.elbo import (
    make_single_problem_elbo,
)
from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)


class TestGaussianConjugateRecoveryBase(
    Generic[Array], unittest.TestCase
):
    """Base test class for Gaussian conjugate recovery."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _run_vi_recovery(
        self,
        nvars: int,
        obs_matrix: Array,
        prior_mean_vals: list,
        prior_var_vals: list,
        noise_var: float,
        observations: Array,
        nsamples: int = 500,
        maxiter: int = 200,
    ) -> tuple:
        """Run VI and return (vi_mean, vi_var, exact_mean, exact_var)."""
        bkd = self._bkd
        nobs = obs_matrix.shape[0]

        # Exact conjugate posterior
        prior_mean_arr = bkd.reshape(
            bkd.asarray(prior_mean_vals), (nvars, 1)
        )
        prior_cov_arr = bkd.diag(bkd.asarray(prior_var_vals))
        noise_cov_arr = noise_var * bkd.eye(nobs)
        conjugate = DenseGaussianConjugatePosterior(
            obs_matrix, prior_mean_arr, prior_cov_arr, noise_cov_arr, bkd,
        )
        conjugate.compute(observations)
        exact_mean = conjugate.posterior_mean()
        exact_cov = conjugate.posterior_covariance()

        # VI setup
        family = GaussianVariationalFamily(nvars, bkd)
        prior = DiagonalMultivariateGaussian(
            prior_mean_arr, bkd.asarray(prior_var_vals), bkd,
        )

        # Build log-likelihood using MultiExperimentLogLikelihood
        noise_variances = bkd.full((nobs,), noise_var)
        base_lik = DiagonalGaussianLogLikelihood(noise_variances, bkd)
        multi_lik = MultiExperimentLogLikelihood(
            base_lik, observations, bkd,
        )

        def log_likelihood_fn(z: Array) -> Array:
            return multi_lik.logpdf(obs_matrix @ z)

        np.random.seed(42)
        base_samples = bkd.asarray(
            np.random.normal(0, 1, (nvars, nsamples))
        )
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        elbo = make_single_problem_elbo(
            family, log_likelihood_fn, prior, base_samples, weights, bkd,
        )

        # Use wide bounds for unconstrained optimization
        bounds = bkd.asarray(
            [[-1e6, 1e6]] * elbo.nvars()
        )
        optimizer = ScipyTrustConstrOptimizer(
            objective=elbo, bounds=bounds, maxiter=maxiter, gtol=1e-8,
        )
        init_guess = bkd.zeros((elbo.nvars(), 1))
        result = optimizer.minimize(init_guess)

        # Push fitted params back to family so it's the source of truth
        elbo(result.optima())
        elbo.push_params_to_family()
        vi_mean, vi_stdev = family._get_mean_stdev()
        vi_var = vi_stdev ** 2

        return vi_mean, vi_var, exact_mean, exact_cov

    @slow_test
    def test_gaussian_1d_conjugate(self) -> None:
        """1D linear model: A=[[1]], prior N(0,1), noise_var=0.5, obs=[2.0]."""
        bkd = self._bkd
        obs_matrix = bkd.asarray([[1.0]])
        observations = bkd.asarray([[2.0]])

        vi_mean, vi_var, exact_mean, exact_cov = self._run_vi_recovery(
            nvars=1,
            obs_matrix=obs_matrix,
            prior_mean_vals=[0.0],
            prior_var_vals=[1.0],
            noise_var=0.5,
            observations=observations,
            nsamples=1000,
            maxiter=300,
        )

        exact_mean_flat = bkd.flatten(exact_mean)
        exact_var_diag = bkd.diag(exact_cov)

        bkd.assert_allclose(vi_mean, exact_mean_flat, atol=0.15)
        bkd.assert_allclose(vi_var, exact_var_diag, rtol=0.3)

    @slow_test
    def test_gaussian_2d_conjugate(self) -> None:
        """2D linear model: A=[[1,0],[0,1]], prior N(0,I), noise_var=0.5."""
        bkd = self._bkd
        obs_matrix = bkd.eye(2)
        observations = bkd.asarray([[1.5], [2.5]])

        vi_mean, vi_var, exact_mean, exact_cov = self._run_vi_recovery(
            nvars=2,
            obs_matrix=obs_matrix,
            prior_mean_vals=[0.0, 0.0],
            prior_var_vals=[1.0, 1.0],
            noise_var=0.5,
            observations=observations,
            nsamples=1000,
            maxiter=300,
        )

        exact_mean_flat = bkd.flatten(exact_mean)
        exact_var_diag = bkd.diag(exact_cov)

        bkd.assert_allclose(vi_mean, exact_mean_flat, atol=0.15)
        bkd.assert_allclose(vi_var, exact_var_diag, rtol=0.3)

    @slow_test
    def test_gaussian_more_data_closer(self) -> None:
        """More observations should make VI closer to exact posterior."""
        bkd = self._bkd
        obs_matrix = bkd.asarray([[1.0]])

        # Few observations
        obs_few = bkd.asarray([[2.0]])
        vi_mean_few, vi_var_few, exact_mean_few, exact_cov_few = (
            self._run_vi_recovery(
                nvars=1,
                obs_matrix=obs_matrix,
                prior_mean_vals=[0.0],
                prior_var_vals=[1.0],
                noise_var=0.5,
                observations=obs_few,
                nsamples=1000,
                maxiter=300,
            )
        )

        # Many observations (3 experiments)
        obs_many = bkd.asarray([[2.0, 1.8, 2.2]])
        vi_mean_many, vi_var_many, exact_mean_many, exact_cov_many = (
            self._run_vi_recovery(
                nvars=1,
                obs_matrix=obs_matrix,
                prior_mean_vals=[0.0],
                prior_var_vals=[1.0],
                noise_var=0.5,
                observations=obs_many,
                nsamples=1000,
                maxiter=300,
            )
        )

        # More data → smaller posterior variance
        exact_var_few = exact_cov_few[0, 0]
        exact_var_many = exact_cov_many[0, 0]
        self.assertLess(float(bkd.flatten(exact_var_many)[0]),
                        float(bkd.flatten(exact_var_few)[0]))

        self.assertLess(float(bkd.flatten(vi_var_many)[0]),
                        float(bkd.flatten(vi_var_few)[0]))


class TestGaussianConjugateRecoveryNumpy(
    TestGaussianConjugateRecoveryBase[NDArray[Any]], unittest.TestCase
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGaussianConjugateRecoveryTorch(
    TestGaussianConjugateRecoveryBase[torch.Tensor], unittest.TestCase
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
