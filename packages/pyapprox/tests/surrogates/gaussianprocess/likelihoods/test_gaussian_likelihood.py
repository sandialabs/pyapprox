"""Tests for GaussianLikelihood."""

import math

import numpy as np

from pyapprox.surrogates.gaussianprocess.likelihoods.gaussian import (
    GaussianLikelihood,
)
from pyapprox.surrogates.gaussianprocess.likelihoods.protocol import (
    LikelihoodProtocol,
)


class TestGaussianLikelihood:
    def test_satisfies_protocol(self, bkd):
        lik = GaussianLikelihood(0.1, (1e-6, 1.0), bkd)
        assert isinstance(lik, LikelihoodProtocol)

    def test_noise_std(self, bkd):
        lik = GaussianLikelihood(0.5, (1e-6, 1.0), bkd)
        bkd.assert_allclose(lik.noise_std(), bkd.array([0.5]), rtol=1e-12)

    def test_log_prob_matches_manual(self, bkd):
        sigma = 0.3
        lik = GaussianLikelihood(sigma, (1e-6, 1.0), bkd)

        y = bkd.array([[1.0, 2.0, 3.0]])
        f = bkd.array([[1.1, 1.8, 3.2]])

        result = lik.log_prob(y, f)
        assert result.shape == (1, 3)

        y_np = np.array([1.0, 2.0, 3.0])
        f_np = np.array([1.1, 1.8, 3.2])
        expected = -0.5 * (
            math.log(2 * math.pi) + 2 * math.log(sigma)
            + (y_np - f_np) ** 2 / sigma**2
        )
        bkd.assert_allclose(result, bkd.array(expected.reshape(1, 3)), rtol=1e-12)

    def test_expected_log_prob_matches_gauss_hermite(self, bkd):
        """Tier 0: closed-form expected_log_prob matches high-order GH quadrature."""
        sigma = 0.2
        lik = GaussianLikelihood(sigma, (1e-6, 1.0), bkd)

        y = bkd.array([[1.5, -0.3]])
        f_mean = bkd.array([[1.4, -0.5]])
        f_var = bkd.array([[0.05, 0.1]])

        closed_form = lik.expected_log_prob(y, f_mean, f_var)
        assert closed_form.shape == (1, 2)

        # GH quadrature reference (high order)
        nodes_np, weights_np = np.polynomial.hermite.hermgauss(128)
        y_np = np.array([1.5, -0.3])
        mu_np = np.array([1.4, -0.5])
        var_np = np.array([0.05, 0.1])

        gh_result = np.zeros(2)
        for j in range(2):
            f_samples = mu_np[j] + np.sqrt(2.0 * var_np[j]) * nodes_np
            log_p = -0.5 * (
                math.log(2 * math.pi) + 2 * math.log(sigma)
                + (y_np[j] - f_samples) ** 2 / sigma**2
            )
            gh_result[j] = np.sum(weights_np * log_p) / np.sqrt(np.pi)

        bkd.assert_allclose(
            closed_form, bkd.array(gh_result.reshape(1, 2)), rtol=1e-10
        )

    def test_expected_log_prob_zero_variance_equals_log_prob(self, bkd):
        sigma = 0.1
        lik = GaussianLikelihood(sigma, (1e-6, 1.0), bkd)
        y = bkd.array([[1.0, 2.0]])
        f_mean = bkd.array([[0.9, 2.1]])
        f_var = bkd.array([[0.0, 0.0]])

        elp = lik.expected_log_prob(y, f_mean, f_var)
        lp = lik.log_prob(y, f_mean)
        bkd.assert_allclose(elp, lp, rtol=1e-12)

    def test_hyp_list_nparams(self, bkd):
        lik = GaussianLikelihood(0.1, (1e-6, 1.0), bkd)
        assert lik.hyp_list().nparams() == 1

    def test_fixed_noise(self, bkd):
        lik = GaussianLikelihood(0.1, (1e-6, 1.0), bkd, fixed=True)
        assert lik.hyp_list().nactive_params() == 0
