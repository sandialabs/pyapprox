import unittest
import numpy as np
from scipy import stats
from functools import partial
import copy
from scipy.special import erfinv

from pyapprox.bayesian_oed import (
    gaussian_loglike_fun, gaussian_kl_divergence,
    precompute_compute_expected_kl_utility_data,
    compute_expected_kl_utility_monte_carlo,
    BayesianBatchKLOED, BayesianSequentialKLOED, d_optimal_utility,
    BayesianBatchDeviationOED, oed_variance_deviation,
    oed_conditional_value_at_risk_deviation, get_oed_inner_quadrature_rule,
    get_posterior_2d_interpolant_from_oed_data, oed_entropic_deviation
)
from pyapprox.variables import IndependentMultivariateRandomVariable
from pyapprox.probability_measure_sampling import (
    generate_independent_random_samples
)
from pyapprox.univariate_polynomials.quadrature import (
    gauss_hermite_pts_wts_1D
)
from pyapprox.bayesian_inference.laplace import (
    laplace_posterior_approximation_for_linear_models
)
from pyapprox.risk_measures import conditional_value_at_risk
from pyapprox.tests.test_risk_measures import (
    get_lognormal_example_exact_quantities
)
from pyapprox.bayesian_inference.laplace import laplace_evidence
from pyapprox.variable_transformations import (
    AffineRandomVariableTransformation
)
from pyapprox.utilities import cartesian_product, outer_product
from pyapprox.indexing import compute_hyperbolic_indices
from pyapprox.monomial import monomial_basis_matrix

import warnings
warnings.filterwarnings('error')


def linear_obs_fun(Amat, samples):
    """
    Linear model. Fix some unknowns
    """
    assert samples.ndim == 2
    nvars = Amat.shape[1]
    assert samples.shape[0] <= nvars
    coef = np.ones((nvars, samples.shape[1]))
    coef[nvars-samples.shape[0]:, :] = samples
    return Amat.dot(coef).T


def exponential_qoi_fun(samples):
    return np.exp(samples.sum(axis=0))[:, None]


class TestBayesianOED(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def check_loglike_fun(self, noise_std, active_indices):
        nvars = 1

        def fun(design, samples):
            assert design.ndim == 2
            assert samples.ndim == 2
            Amat = design.T
            return Amat.dot(samples).T

        prior_mean = np.zeros((nvars, 1))
        prior_cov = np.eye(nvars)

        design = np.linspace(-1, 1, 4)[None, :]
        true_sample = np.ones((nvars, 1))*0.4
        obs = fun(design, true_sample)

        if type(noise_std) == np.ndarray:
            assert noise_std.shape[0] == obs.shape[1] and noise_std.ndim == 2
            obs += np.random.normal(0, 1, obs.shape)*noise_std.T
        else:
            obs += np.random.normal(0, 1, obs.shape)*noise_std

        noise_cov_inv = np.eye(obs.shape[1])/(noise_std**2)
        obs_matrix = design.T

        if active_indices is None:
            exact_post_mean, exact_post_cov = \
                laplace_posterior_approximation_for_linear_models(
                    obs_matrix, prior_mean, np.linalg.inv(prior_cov),
                    noise_cov_inv, obs.T)
        else:
            exact_post_mean, exact_post_cov = \
                laplace_posterior_approximation_for_linear_models(
                    obs_matrix[active_indices, :], prior_mean,
                    np.linalg.inv(prior_cov),
                    noise_cov_inv[np.ix_(active_indices, active_indices)],
                    obs[:, active_indices].T)

        n_xx = 100
        lb, ub = stats.norm(0, 1).interval(0.99)
        xx = np.linspace(lb, ub, n_xx)
        true_pdf_vals = stats.norm(
            exact_post_mean[0], np.sqrt(exact_post_cov[0])).pdf(xx)[:, None]

        prior_pdf = stats.norm(prior_mean[0], np.sqrt(prior_cov[0])).pdf
        pred_obs = fun(design, xx[None, :])
        lvals = (np.exp(
            gaussian_loglike_fun(obs, pred_obs, noise_std, active_indices)) *
                 prior_pdf(xx)[:, None])
        assert lvals.shape == (n_xx, 1)

        xx_gauss, ww_gauss = gauss_hermite_pts_wts_1D(300)
        pred_obs = fun(design, xx_gauss[None, :])
        evidence = np.exp(
            gaussian_loglike_fun(
                obs, pred_obs, noise_std, active_indices)[:, 0]).dot(ww_gauss)
        post_pdf_vals = lvals/evidence

        gauss_evidence = laplace_evidence(
            lambda x: np.exp(gaussian_loglike_fun(
                obs, fun(design, x), noise_std, active_indices)[:, 0]),
            prior_pdf, exact_post_cov, exact_post_mean)
        print(evidence, gauss_evidence)
        assert np.allclose(evidence, gauss_evidence)

        # accuracy depends on quadrature rule and size of noise
        # print(post_pdf_vals - true_pdf_vals)
        assert np.allclose(post_pdf_vals, true_pdf_vals)

        # plt.plot(xx, true_pdf_vals)
        # plt.plot(xx, prior_pdf(xx))
        # plt.plot(xx, post_pdf_vals, '--')
        # plt.show()

    def test_gaussian_loglike_fun(self):
        np.random.seed(1)
        self.check_loglike_fun(0.3, None)
        np.random.seed(1)
        self.check_loglike_fun(np.array([[0.25, 0.3, 0.35, 0.4]]).T, None)
        active_indices = np.array([0, 1, 2, 3])
        np.random.seed(1)
        self.check_loglike_fun(0.3, active_indices)
        np.random.seed(1)
        self.check_loglike_fun(
            np.array([[0.25, 0.3, 0.35, 0.4]]).T, active_indices)
        active_indices = np.array([0, 2, 3])
        np.random.seed(1)
        self.check_loglike_fun(0.3, active_indices)
        np.random.seed(1)
        self.check_loglike_fun(
            np.array([[0.25, 0.3, 0.35, 0.4]]).T, active_indices)

    def check_gaussian_loglike_fun_3d(self, noise_std, active_indices):
        nvars = 1

        def fun(design, samples):
            assert design.ndim == 2
            assert samples.ndim == 2
            Amat = design.T
            return Amat.dot(samples).T

        design = np.linspace(-1, 1, 4)[None, :]
        true_sample = np.ones((nvars, 1))*0.4
        obs = fun(design, true_sample)

        if type(noise_std) == np.ndarray:
            assert noise_std.shape[0] == obs.shape[1] and noise_std.ndim == 2
            obs += np.random.normal(0, 1, obs.shape)*noise_std.T
        else:
            obs += np.random.normal(0, 1, obs.shape)*noise_std

        n_xx = 11
        lb, ub = stats.norm(0, 1).interval(0.99)
        xx = np.linspace(lb, ub, n_xx)
        pred_obs = fun(design, xx[None, :])

        loglike = gaussian_loglike_fun(
            obs, pred_obs, noise_std, active_indices)[:, 0]
        loglike_3d = gaussian_loglike_fun(
            obs[:, None, :], pred_obs[:, None, :], noise_std,
            active_indices)[:, 0]
        assert np.allclose(loglike, loglike_3d)

        loglike_3d_econ = gaussian_loglike_fun(
            np.vstack([obs[:, None, :]]*pred_obs.shape[0]),
            pred_obs[:, None, :], noise_std,
            active_indices)[:, 0]
        assert np.allclose(loglike_3d, loglike_3d_econ)

        loglike_3d_econ = gaussian_loglike_fun(
            np.vstack([obs[:, None, :]]*pred_obs.shape[0]),
            np.hstack([pred_obs[:, None, :]]*3), noise_std,
            active_indices)[:, 0]
        assert np.allclose(loglike_3d, loglike_3d_econ)

    def test_gaussian_loglike_fun_3d(self):
        self.check_gaussian_loglike_fun_3d(0.3, None)
        self.check_loglike_fun(
            np.array([[0.25, 0.3, 0.35, 0.4]]).T, None)
        self.check_gaussian_loglike_fun_3d(0.3, np.array([0, 2, 3]))
        self.check_gaussian_loglike_fun_3d(
            np.array([[0.25, 0.3, 0.35, 0.4]]).T, np.array([0, 2, 3]))

    def test_gaussian_kl_divergence(self):
        nvars = 1
        mean1, sigma1 = np.zeros((nvars, 1)), np.eye(nvars)*2
        mean2, sigma2 = np.ones((nvars, 1)), np.eye(nvars)*3

        kl_div = gaussian_kl_divergence(mean1, sigma1, mean2, sigma2)

        rv1 = stats.multivariate_normal(mean1, sigma1)
        rv2 = stats.multivariate_normal(mean2, sigma2)

        xx, ww = gauss_hermite_pts_wts_1D(300)
        xx = xx*np.sqrt(sigma1[0, 0]) + mean1[0]
        kl_div_quad = np.log(rv1.pdf(xx)/rv2.pdf(xx)).dot(ww)
        assert np.allclose(kl_div, kl_div_quad)

        xx = np.random.normal(mean1[0], np.sqrt(sigma1[0, 0]), (int(1e6)))
        ww = np.ones(xx.shape[0])/xx.shape[0]
        kl_div_quad = np.log(rv1.pdf(xx)/rv2.pdf(xx)).dot(ww)
        assert np.allclose(kl_div, kl_div_quad, rtol=1e-2)

    def test_compute_expected_kl_utility_monte_carlo(self):
        nrandom_vars = 1
        noise_std = .3
        design = np.linspace(-1, 1, 2)[None, :]
        Amat = design.T

        def obs_fun(x):
            return (Amat.dot(x)).T

        def noise_fun(values):
            return np.random.normal(0, noise_std, (values.shape))

        # specify the first design point
        collected_design_indices = np.array([0])

        prior_variable = IndependentMultivariateRandomVariable(
            [stats.norm(0, 1)]*nrandom_vars)

        prior_mean = prior_variable.get_statistics('mean')
        prior_cov = np.diag(prior_variable.get_statistics('var')[:, 0])
        prior_cov_inv = np.linalg.inv(prior_cov)
        noise_cov_inv = np.eye(Amat.shape[0])/noise_std**2

        def generate_random_prior_samples(n):
            return (generate_independent_random_samples(prior_variable, n),
                    np.ones(n)/n)

        def generate_inner_prior_samples_mc(n):
            return generate_random_prior_samples(n), np.ones(n)/n

        ninner_loop_samples = 300
        x, w = gauss_hermite_pts_wts_1D(ninner_loop_samples)

        def generate_inner_prior_samples_gauss(n):
            # use precomputed samples so to avoid cost of regenerating
            assert n == x.shape[0]
            return x[None, :], w

        generate_inner_prior_samples = generate_inner_prior_samples_gauss

        nouter_loop_samples = 10000
        outer_loop_obs, outer_loop_pred_obs, inner_loop_pred_obs, \
            inner_loop_weights, __, __ = \
            precompute_compute_expected_kl_utility_data(
                generate_random_prior_samples, nouter_loop_samples, obs_fun,
                noise_fun, ninner_loop_samples,
                generate_inner_prior_samples=generate_inner_prior_samples)

        new_design_indices = np.array([1])

        outer_loop_weights = np.ones(
            (nouter_loop_samples, 1))/nouter_loop_samples

        def log_likelihood_fun(obs, pred_obs, active_indices=None):
            return gaussian_loglike_fun(
                obs, pred_obs, noise_std, active_indices)
        utility = compute_expected_kl_utility_monte_carlo(
            log_likelihood_fun, outer_loop_obs, outer_loop_pred_obs,
            inner_loop_pred_obs, inner_loop_weights, outer_loop_weights,
            collected_design_indices, new_design_indices, False)['utility_val']

        kl_divs = []
        # overwrite subset of obs with previously collected data
        # make copy so that outerloop obs can be used again
        outer_loop_obs_copy = outer_loop_obs.copy()
        for ii in range(nouter_loop_samples):
            idx = np.hstack(
                (collected_design_indices, new_design_indices))
            obs_ii = outer_loop_obs_copy[ii:ii+1, idx]

            idx = np.hstack((collected_design_indices, new_design_indices))
            exact_post_mean, exact_post_cov = \
                laplace_posterior_approximation_for_linear_models(
                    Amat[idx, :], prior_mean, prior_cov_inv,
                    noise_cov_inv[np.ix_(idx, idx)], obs_ii.T)

            kl_div = gaussian_kl_divergence(
                exact_post_mean, exact_post_cov, prior_mean, prior_cov)
            kl_divs.append(kl_div)

        # print(utility-np.mean(kl_divs), utility, np.mean(kl_divs))
        assert np.allclose(utility, np.mean(kl_divs), rtol=2e-2)

    def test_batch_kl_oed(self):
        """
        No observations collected to inform subsequent designs
        """
        np.random.seed(1)
        nrandom_vars = 1
        noise_std = 1
        ndesign = 4
        nouter_loop_samples = 10000
        ninner_loop_samples = 31

        ncandidates = 11
        design_candidates = np.linspace(-1, 1, ncandidates)[None, :]

        def obs_fun(samples):
            assert design_candidates.ndim == 2
            assert samples.ndim == 2
            Amat = design_candidates.T
            return Amat.dot(samples).T

        prior_variable = IndependentMultivariateRandomVariable(
            [stats.norm(0, 1)]*nrandom_vars)

        x_quad, w_quad = gauss_hermite_pts_wts_1D(ninner_loop_samples)

        def generate_inner_prior_samples_gauss(n):
            # use precomputed samples so to avoid cost of regenerating
            assert n == x_quad.shape[0]
            return x_quad[None, :], w_quad

        generate_inner_prior_samples = generate_inner_prior_samples_gauss

        # Define initial design
        init_design_indices = np.array([ncandidates//2])
        oed = BayesianBatchKLOED(
            design_candidates, obs_fun, noise_std, prior_variable,
            nouter_loop_samples, ninner_loop_samples,
            generate_inner_prior_samples)
        oed.populate()
        oed.set_collected_design_indices(init_design_indices)

        for ii in range(len(init_design_indices), ndesign):
            # loop must be before oed.updated design because
            # which updates oed.collected_design_indices and thus
            # changes problem
            d_utility_vals = np.zeros(ncandidates)
            for kk in range(ncandidates):
                if kk not in oed.collected_design_indices:
                    new_design = np.hstack(
                        (design_candidates[:, oed.collected_design_indices],
                         design_candidates[:, kk:kk+1]))
                    Amat = new_design.T
                    d_utility_vals[kk] = d_optimal_utility(Amat, noise_std)

            utility_vals, selected_indices = oed.update_design()
            # ignore entries of previously collected data
            II = np.where(d_utility_vals > 0)[0]
            print(d_utility_vals[II])
            print(utility_vals[II])
            print((np.absolute(
                d_utility_vals[II]-utility_vals[II])/d_utility_vals[II]).max())
            assert np.allclose(d_utility_vals[II], utility_vals[II], rtol=4e-2)

    def test_batch_prediction_oed(self):
        """
        No observations collected to inform subsequent designs
        """
        np.random.seed(1)
        noise_std = 1
        ndesign = 5
        nouter_loop_samples = 2
        # outerloop samples does not effect this problem
        # because variance is independent of noise only on design
        ninner_loop_samples_1d = 41
        degree = 2
        nrandom_vars = degree+1
        quad_method = "quadratic"
        risk_fun = None
        from pyapprox import conditional_value_at_risk
        quantile = 0.8
        risk_fun = partial(conditional_value_at_risk, alpha=quantile)

        ncandidates = 21
        design_candidates = np.linspace(-1, 1, ncandidates)[None, :]
        # design_candidates = np.hstack(
        #    (design_candidates, np.array([[-1/np.sqrt(5), 1/np.sqrt(5)]])))
        nprediction_samples = 201
        prediction_candidates = np.linspace(
            -1, 1, nprediction_samples)[None, :]

        def basis_matrix(degree, samples):
            return samples.T**np.arange(degree+1)[None, :]

        def obs_fun(samples):
            assert design_candidates.ndim == 2
            assert samples.ndim == 2
            Amat = basis_matrix(degree, design_candidates)
            return Amat.dot(samples).T

        def qoi_fun(samples):
            Amat = basis_matrix(degree, prediction_candidates)
            return Amat.dot(samples).T

        prior_variable = IndependentMultivariateRandomVariable(
            [stats.norm(0, 1)]*nrandom_vars)

        x_quad, w_quad = get_oed_inner_quadrature_rule(
            ninner_loop_samples_1d, prior_variable, quad_method)
        ninner_loop_samples = x_quad.shape[1]

        def generate_inner_prior_samples_gauss(n):
            # use precomputed samples so to avoid cost of regenerating
            assert n == x_quad.shape[1]
            return x_quad, w_quad
        print(x_quad.shape)

        generate_inner_prior_samples = generate_inner_prior_samples_gauss

        # Define initial design
        # init_design_indices = np.array([0])
        init_design_indices = np.empty((0), dtype=int)
        oed = BayesianBatchDeviationOED(
            design_candidates, obs_fun, noise_std, prior_variable,
            qoi_fun, nouter_loop_samples, ninner_loop_samples,
            generate_inner_prior_samples, deviation_fun=oed_variance_deviation,
            risk_fun=risk_fun)
        oed.populate()
        oed.set_collected_design_indices(init_design_indices)

        for ii in range(len(init_design_indices), ndesign):
            utility_vals, selected_indices = oed.update_design(False)

        prior_mean = prior_variable.get_statistics("mean")
        prior_cov = np.diag(prior_variable.get_statistics("var")[:, 0])
        prior_cov_inv = np.linalg.inv(prior_cov)
        obs_matrix = basis_matrix(degree, design_candidates)
        noise_cov_inv = np.eye(design_candidates.shape[1])/(noise_std**2)
        pred_matrix = basis_matrix(degree, prediction_candidates)

        # print(oed.collected_design_indices)
        # print(design_candidates[0, oed.collected_design_indices])

        # Check expected variance when choosing the final design
        # point. Compare with exact value computed using Laplace formula
        # Note variance is independent of data so no need to generate
        # realizations of data
        ii = 0
        data = []
        for jj in range(design_candidates.shape[1]):
            if jj not in oed.collected_design_indices[:-1]:
                idx = np.hstack((
                    oed.collected_design_indices[:-1], jj))
                # realization of data does not matter
                obs_ii = oed.outer_loop_obs[ii:ii+1, idx]
                exact_post_mean, exact_post_cov = \
                    laplace_posterior_approximation_for_linear_models(
                        obs_matrix[idx, :], prior_mean, prior_cov_inv,
                        noise_cov_inv[np.ix_(idx, idx)], obs_ii.T)
                pointwise_post_variance = np.diag(
                    pred_matrix.dot(exact_post_cov.dot(pred_matrix.T))
                )[:, None]
                exact_variance_risk = oed.risk_fun(pointwise_post_variance)
                data.append([pointwise_post_variance, -exact_variance_risk])
                # print(f"Candidate {jj}", exact_variance_risk)
            else:
                data.append([None, -np.inf])
        jdx = np.argmax([d[1] for d in data])
        # print(np.mean(pointwise_post_variance))
        # print(np.mean(
        #     conditional_value_at_risk(pointwise_post_variance, quantile)))
        # print(jdx, oed.collected_design_indices)
        assert jdx == oed.collected_design_indices[-1]
        # print(utility_vals[oed.collected_design_indices[-1]]-data[jdx][1])
        assert np.allclose(
            utility_vals[oed.collected_design_indices[-1]],
            data[jdx][1], rtol=1e-4)
        # from matplotlib import pyplot as plt
        # pointwise_prior_variance = np.diag(
        #     pred_matrix.dot(prior_cov.dot(pred_matrix.T)))
        # plt.plot(
        #     prediction_candidates[0, :], pointwise_prior_variance, '-')
        # plt.plot(
        #     prediction_candidates[0, :], data[jdx][0], '--')
        # collected_design_indices = np.hstack(
        #     (oed.collected_design_indices[:-1], jdx))
        # print(collected_design_indices)
        # print(design_candidates[0, oed.collected_design_indices])
        # for idx in collected_design_indices:
        #     plt.axvline(x=design_candidates[0, idx], ls='--', c='k')
        # plt.show()

    def test_sequential_kl_oed(self):
        """
        Observations collected ARE used to inform subsequent designs
        """
        nrandom_vars = 1
        noise_std = 1
        ndesign = 5
        nouter_loop_samples = int(1e4)
        ninner_loop_samples = 31

        ncandidates = 6
        design_candidates = np.linspace(-1, 1, ncandidates)[None, :]

        def obs_fun(samples):
            assert design_candidates.ndim == 2
            assert samples.ndim == 2
            Amat = design_candidates.T
            return Amat.dot(samples).T

        prior_variable = IndependentMultivariateRandomVariable(
            [stats.norm(0, 1)]*nrandom_vars)

        true_sample = np.array([.4]*nrandom_vars)[:, None]

        def obs_process(new_design_indices):
            obs = obs_fun(true_sample)[:, new_design_indices]
            obs += oed.noise_fun(obs)
            return obs

        x_quad, w_quad = gauss_hermite_pts_wts_1D(ninner_loop_samples)

        def generate_inner_prior_samples_gauss(n):
            # use precomputed samples so to avoid cost of regenerating
            assert n == x_quad.shape[0]
            return x_quad[None, :], w_quad

        generate_inner_prior_samples = generate_inner_prior_samples_gauss

        # Define initial design
        init_design_indices = np.array([ncandidates//2])
        oed = BayesianSequentialKLOED(
            design_candidates, obs_fun, noise_std, prior_variable,
            obs_process, nouter_loop_samples, ninner_loop_samples,
            generate_inner_prior_samples)
        oed.populate()
        oed.set_collected_design_indices(init_design_indices)

        prior_mean = oed.prior_variable.get_statistics('mean')
        prior_cov = np.diag(prior_variable.get_statistics('var')[:, 0])
        prior_cov_inv = np.linalg.inv(prior_cov)

        exact_post_mean_prev = prior_mean
        exact_post_cov_prev = prior_cov
        post_var_prev = stats.multivariate_normal(
            mean=exact_post_mean_prev[:, 0], cov=exact_post_cov_prev)
        selected_indices = init_design_indices

        # Because of Monte Carlo error set step tols individually
        # It is too expensive to up the number of outer_loop samples to
        # reduce errors
        step_tols = [7.3e-3, 6.5e-2, 3.3e-2, 1.6e-1]

        for step in range(len(init_design_indices), ndesign):
            current_design = design_candidates[:, oed.collected_design_indices]
            noise_cov_inv = np.eye(current_design.shape[1])/noise_std**2

            # Compute posterior moving from previous posterior and using
            # only the most recently collected data
            noise_cov_inv_incr = np.eye(
                selected_indices.shape[0])/noise_std**2
            exact_post_mean, exact_post_cov = \
                laplace_posterior_approximation_for_linear_models(
                    design_candidates[:, selected_indices].T,
                    exact_post_mean_prev, np.linalg.inv(exact_post_cov_prev),
                    noise_cov_inv_incr, oed.collected_obs[:, -1:].T)

            # check using current posteior as prior and only using new
            # data (above) produces the same posterior as using original prior
            # and all collected data (from_prior approach). The posteriors
            # should be the same but the evidences will be difference.
            # This is tested below
            exact_post_mean_from_prior, exact_post_cov_from_prior = \
                laplace_posterior_approximation_for_linear_models(
                    current_design.T, prior_mean, prior_cov_inv, noise_cov_inv,
                    oed.collected_obs.T)

            assert np.allclose(exact_post_mean, exact_post_mean_from_prior)
            assert np.allclose(exact_post_cov, exact_post_cov_from_prior)

            # Compute PDF of current posterior that uses all collected data
            post_var = stats.multivariate_normal(
                mean=exact_post_mean[:, 0].copy(), cov=exact_post_cov.copy())

            # Compute evidence moving from previous posterior to
            # new posterior (not initial prior to posterior).
            # Values can be computed exactly for Gaussian prior and noise
            gauss_evidence = laplace_evidence(
                lambda x: np.exp(gaussian_loglike_fun(
                    oed.collected_obs[:, -1:],
                    obs_fun(x)[:, oed.collected_design_indices[-1:]],
                    noise_std))[:, 0],
                lambda y: np.atleast_2d(post_var_prev.pdf(y.T)).T,
                exact_post_cov, exact_post_mean)

            # Compute evidence using Gaussian quadrature rule. This
            # is possible for this low-dimensional example.
            quad_loglike_vals = np.exp(gaussian_loglike_fun(
                oed.collected_obs[:, -1:],
                obs_fun(
                    x_quad[None, :])[:, oed.collected_design_indices[-1:]],
                noise_std))[:, 0]
            # we must divide integarnd by initial prior_pdf since it is
            # already implicilty included via the quadrature weights
            integrand_vals = quad_loglike_vals*post_var_prev.pdf(
                x_quad[:, None])/prior_variable.pdf(x_quad[None, :])[:, 0]
            quad_evidence = integrand_vals.dot(w_quad)
            # print(quad_evidence, gauss_evidence)
            assert np.allclose(gauss_evidence, quad_evidence), step

            # print('G', gauss_evidence, oed.evidence)
            assert np.allclose(gauss_evidence, oed.evidence), step

            # compute the evidence of moving from the initial prior
            # to the current posterior. This will be used for testing later
            gauss_evidence_from_prior = laplace_evidence(
                lambda x: np.exp(gaussian_loglike_fun(
                    oed.collected_obs,
                    obs_fun(x)[:, oed.collected_design_indices],
                    noise_std))[:, 0], prior_variable.pdf,
                exact_post_cov, exact_post_mean)

            # Copy current state of OED before new data is determined
            # This copy will be used to compute Laplace based utility and
            # evidence values for testing
            oed_copy = copy.deepcopy(oed)

            # Update the design
            utility_vals, selected_indices = oed.update_design()
            new_obs = oed.obs_process(selected_indices)
            oed.update_observations(new_obs)
            utility = utility_vals[selected_indices]

            # Re-compute the evidences that were used to update the design
            # above. This will be used for testing later
            # print('D', oed_copy.evidence)
            results = oed_copy.compute_expected_utility(
                oed_copy.collected_design_indices, selected_indices, True)
            evidences = results["evidences"]

            # print('Collected plus selected indices',
            #       oed.collected_design_indices,
            #       oed_copy.collected_design_indices, selected_indices)

            # For all outer loop samples compute the posterior exactly
            # and compute intermediate values for testing. While OED
            # considers all possible candidate design indices
            # Here we just test the one that was chosen last when
            # design was updated
            exact_evidences = np.empty(nouter_loop_samples)
            exact_kl_divs = np.empty_like(exact_evidences)
            for jj in range(nouter_loop_samples):
                # Fill obs with those predicted by outer loop sample
                idx = oed.collected_design_indices
                obs_jj = oed_copy.outer_loop_obs[jj:jj+1, idx]
                # Overwrite the previouly simulated obs with collected obs.
                # Do not ovewrite the last value which is the potential
                # data used to compute expected utility
                obs_jj[:, :oed_copy.collected_obs.shape[1]] = \
                    oed_copy.collected_obs

                # Compute the posterior obtained by using predicted value
                # of outer loop sample
                noise_cov_inv_jj = np.eye(
                    selected_indices.shape[0])/noise_std**2
                exact_post_mean_jj, exact_post_cov_jj = \
                    laplace_posterior_approximation_for_linear_models(
                        design_candidates[:, selected_indices].T,
                        exact_post_mean, np.linalg.inv(exact_post_cov),
                        noise_cov_inv_jj, obs_jj[:, -1].T)

                # Use post_pdf so measure change from current posterior (prior)
                # to new posterior
                gauss_evidence_jj = laplace_evidence(
                    lambda x: np.exp(gaussian_loglike_fun(
                        obs_jj[:, -1:], obs_fun(x)[:, selected_indices],
                        noise_std))[:, 0],
                    lambda y: np.atleast_2d(post_var.pdf(y.T)).T,
                    exact_post_cov_jj, exact_post_mean_jj)
                exact_evidences[jj] = gauss_evidence_jj

                # Check quadrature gets the same answer
                quad_loglike_vals = np.exp(gaussian_loglike_fun(
                    obs_jj[:, -1:],
                    obs_fun(
                        x_quad[None, :])[:, selected_indices],
                    noise_std))[:, 0]
                integrand_vals = quad_loglike_vals*post_var.pdf(
                    x_quad[:, None])/prior_variable.pdf(x_quad[None, :])[:, 0]
                quad_evidence = integrand_vals.dot(w_quad)
                # print(quad_evidence, gauss_evidence_jj)
                assert np.allclose(gauss_evidence_jj, quad_evidence), step

                # Check that evidence of moving from current posterior
                # to new posterior with (potential data from outer-loop sample)
                # is equal to the evidence of moving from
                # intitial prior to new posterior divide by the evidence
                # from moving from the initial prior to the current posterior
                gauss_evidence_jj_from_prior = laplace_evidence(
                    lambda x: np.exp(gaussian_loglike_fun(
                        obs_jj, obs_fun(x)[:, idx],
                        noise_std))[:, 0], prior_variable.pdf,
                    exact_post_cov_jj, exact_post_mean_jj)
                # print(gauss_evidence_jj_from_prior/gauss_evidence_from_prior,
                #       gauss_evidence_jj)
                # print('gauss_evidence_from_prior', gauss_evidence_from_prior)
                assert np.allclose(
                    gauss_evidence_jj_from_prior/gauss_evidence_from_prior,
                    gauss_evidence_jj)

                gauss_kl_div = gaussian_kl_divergence(
                    exact_post_mean_jj, exact_post_cov_jj,
                    exact_post_mean, exact_post_cov)
                # gauss_kl_div = gaussian_kl_divergence(
                #     exact_post_mean, exact_post_cov,
                #     exact_post_mean_jj, exact_post_cov_jj)
                exact_kl_divs[jj] = gauss_kl_div

            # print(evidences[:, 0], exact_evidences)
            assert np.allclose(evidences[:, 0], exact_evidences)

            # Outer loop samples are from prior. Use importance reweighting
            # to sample from previous posterior. This step is only relevant
            # for open loop design (used here)
            # where observed data informs current estimate
            # of parameters. Closed loop design (not used here)
            # never collects data and so it always samples from the prior.
            post_weights = post_var.pdf(
                oed.outer_loop_prior_samples.T)/post_var_prev.pdf(
                    oed.outer_loop_prior_samples.T)/oed.nouter_loop_samples
            laplace_utility = np.sum(exact_kl_divs*post_weights)
            # print('u', (utility-laplace_utility)/laplace_utility, step)
            assert np.allclose(
                utility, laplace_utility, rtol=step_tols[step-1])

            exact_post_mean_prev = exact_post_mean
            exact_post_cov_prev = exact_post_cov
            post_var_prev = post_var

    def help_compare_sequential_kl_oed_econ(self, use_gauss_quadrature):
        """
        Use the same inner loop samples for all outer loop samples
        """
        nrandom_vars = 1
        noise_std = 1
        ndesign = 5
        nouter_loop_samples = int(1e1)
        ninner_loop_samples = 31

        ncandidates = 6
        design_candidates = np.linspace(-1, 1, ncandidates)[None, :]

        def obs_fun(samples):
            assert design_candidates.ndim == 2
            assert samples.ndim == 2
            Amat = design_candidates.T
            return Amat.dot(samples).T

        prior_variable = IndependentMultivariateRandomVariable(
            [stats.norm(0, 1)]*nrandom_vars)

        true_sample = np.array([.4]*nrandom_vars)[:, None]

        def obs_process(new_design_indices):
            obs = obs_fun(true_sample)[:, new_design_indices]
            obs += oed.noise_fun(obs)
            return obs

        generate_random_prior_samples = partial(
            generate_independent_random_samples, prior_variable)

        def generate_inner_prior_samples_mc(n):
            # fix seed that when econ is False we are still creating
            # the samples each time. This is just for testing purposes
            # to make sure that econ is True does this in effect
            np.random.seed(1)
            return generate_random_prior_samples(n), np.ones(n)/n

        x_quad, w_quad = gauss_hermite_pts_wts_1D(ninner_loop_samples)

        def generate_inner_prior_samples_gauss(n):
            # use precomputed samples so to avoid cost of regenerating
            assert n == x_quad.shape[0]
            return x_quad[None, :], w_quad

        if use_gauss_quadrature:
            generate_inner_prior_samples = generate_inner_prior_samples_gauss
        else:
            generate_inner_prior_samples = generate_inner_prior_samples_mc

        # Define initial design
        init_design_indices = np.array([ncandidates//2])
        np.random.seed(1)
        oed = BayesianSequentialKLOED(
            design_candidates, obs_fun, noise_std, prior_variable,
            obs_process, nouter_loop_samples, ninner_loop_samples,
            generate_inner_prior_samples)
        oed.populate()
        oed.set_collected_design_indices(init_design_indices)

        # randomness only enters during populate and when collecting
        # real observations, i.e. evaluating obs model so setting seed
        # here is sufficient when below we just evaluate obs model once
        # and use same value for both oed instances
        np.random.seed(1)
        oed_econ = BayesianSequentialKLOED(
            design_candidates, obs_fun, noise_std, prior_variable,
            obs_process, nouter_loop_samples, ninner_loop_samples,
            generate_inner_prior_samples, econ=True)
        oed_econ.populate()
        oed_econ.set_collected_design_indices(init_design_indices)

        for step in range(len(init_design_indices), ndesign):
            utility_vals, selected_indices = oed.update_design()
            new_obs = oed.obs_process(selected_indices)
            oed.update_observations(new_obs)

            econ_utility_vals, econ_selected_indices = oed_econ.update_design()
            assert np.allclose(econ_utility_vals, utility_vals)
            assert np.allclose(econ_selected_indices, selected_indices)
            # use same data as non econ version do not call model
            # as different noise will be added
            oed_econ.update_observations(new_obs)

    def test_sequential_kl_oed_econ(self):
        self.help_compare_sequential_kl_oed_econ(False)
        self.help_compare_sequential_kl_oed_econ(True)

    def test_bayesian_importance_sampling_avar(self):
        np.random.seed(1)
        nrandom_vars = 2
        Amat = np.array([[-0.5, 1]])
        noise_std = 0.1
        prior_variable = IndependentMultivariateRandomVariable(
            [stats.norm(0, 1)]*nrandom_vars)
        prior_mean = prior_variable.get_statistics('mean')
        prior_cov = np.diag(prior_variable.get_statistics('var')[:, 0])
        prior_cov_inv = np.linalg.inv(prior_cov)
        noise_cov_inv = np.eye(Amat.shape[0])/noise_std**2
        true_sample = np.array([.4]*nrandom_vars)[:, None]
        collected_obs = Amat.dot(true_sample)
        collected_obs += np.random.normal(0, noise_std, (collected_obs.shape))
        exact_post_mean, exact_post_cov = \
            laplace_posterior_approximation_for_linear_models(
                Amat, prior_mean, prior_cov_inv, noise_cov_inv,
                collected_obs)

        chol_factor = np.linalg.cholesky(exact_post_cov)
        chol_factor_inv = np.linalg.inv(chol_factor)

        def g_model(samples):
            return np.exp(np.sum(chol_factor_inv.dot(samples-exact_post_mean),
                                 axis=0))[:, None]

        nsamples = int(1e6)
        prior_samples = generate_independent_random_samples(
            prior_variable, nsamples)
        posterior_samples = chol_factor.dot(
            np.random.normal(0, 1, (nrandom_vars, nsamples)))+exact_post_mean

        g_mu, g_sigma = 0, np.sqrt(nrandom_vars)
        f, f_cdf, f_pdf, VaR, CVaR, ssd, ssd_disutil = \
            get_lognormal_example_exact_quantities(g_mu, g_sigma)

        beta = .1
        cvar_exact = CVaR(beta)

        cvar_mc = conditional_value_at_risk(g_model(posterior_samples), beta)

        prior_pdf = prior_variable.pdf
        post_pdf = stats.multivariate_normal(
            mean=exact_post_mean[:, 0], cov=exact_post_cov).pdf
        weights = post_pdf(prior_samples.T)/prior_pdf(prior_samples)[:, 0]
        weights /= weights.sum()
        cvar_im = conditional_value_at_risk(
            g_model(prior_samples), beta, weights)
        # print(cvar_exact, cvar_mc, cvar_im)
        assert np.allclose(cvar_exact, cvar_mc, rtol=1e-3)
        assert np.allclose(cvar_exact, cvar_im, rtol=2e-3)

    def test_oed_variance_deviation(self):
        MM, NN = 2, 3
        samples = np.random.normal(0, 1, (MM, NN, 1))
        weights = np.ones((MM, NN))/NN
        variances = oed_variance_deviation(samples, weights)
        print(variances, samples.var(axis=1))
        assert np.allclose(variances, samples.var(axis=1))

    def help_compare_prediction_based_oed(
            self, deviation_fun, gauss_deviation_fun, use_gauss_quadrature,
            ninner_loop_samples, ndesign_vars, tol):
        ncandidates_1d = 5
        design_candidates = cartesian_product(
            [np.linspace(-1, 1, ncandidates_1d)]*ndesign_vars)
        ncandidates = design_candidates.shape[1]

        # Define model used to predict likely observable data
        indices = compute_hyperbolic_indices(ndesign_vars, 1)[:, 1:]
        Amat = monomial_basis_matrix(indices, design_candidates)
        obs_fun = partial(linear_obs_fun, Amat)

        # Define model used to predict unobservable QoI
        qoi_fun = exponential_qoi_fun

        # Define the prior PDF of the unknown variables
        nrandom_vars = indices.shape[1]
        prior_variable = IndependentMultivariateRandomVariable(
            [stats.norm(0, 0.5)]*nrandom_vars)

        # Define the independent observational noise
        noise_std = 1

        # Define initial design
        init_design_indices = np.array([ncandidates//2])

        # Define OED options
        nouter_loop_samples = 100
        if use_gauss_quadrature:
            # 301 needed for cvar deviation
            # only 31 needed for variance deviation
            ninner_loop_samples_1d = ninner_loop_samples
            var_trans = AffineRandomVariableTransformation(prior_variable)
            x_quad, w_quad = gauss_hermite_pts_wts_1D(
                ninner_loop_samples_1d)
            x_quad = cartesian_product([x_quad]*nrandom_vars)
            w_quad = outer_product([w_quad]*nrandom_vars)
            x_quad = var_trans.map_from_canonical_space(x_quad)
            ninner_loop_samples = x_quad.shape[1]

            def generate_inner_prior_samples(nsamples):
                assert nsamples == x_quad.shape[1], (nsamples, x_quad.shape)
                return x_quad, w_quad
        else:
            # use default Monte Carlo sampling
            generate_inner_prior_samples = None

        # Define initial design
        init_design_indices = np.array([ncandidates//2])

        # Setup OED problem
        oed = BayesianBatchDeviationOED(
            design_candidates, obs_fun, noise_std, prior_variable,
            qoi_fun, nouter_loop_samples, ninner_loop_samples,
            generate_inner_prior_samples, deviation_fun=deviation_fun)
        oed.populate()
        oed.set_collected_design_indices(init_design_indices)

        prior_mean = oed.prior_variable.get_statistics('mean')
        prior_cov = np.diag(prior_variable.get_statistics('var')[:, 0])
        prior_cov_inv = np.linalg.inv(prior_cov)
        selected_indices = init_design_indices

        # Generate experimental design
        nexperiments = 3
        for step in range(len(init_design_indices), nexperiments):
            # Copy current state of OED before new data is determined
            # This copy will be used to compute Laplace based utility and
            # evidence values for testing
            oed_copy = copy.deepcopy(oed)

            # Update the design
            utility_vals, selected_indices = oed.update_design()

            results = oed_copy.compute_expected_utility(
                oed_copy.collected_design_indices, selected_indices, True)
            # utility = results["utility_val"]
            deviations = results["deviations"]

            exact_deviations = np.empty(nouter_loop_samples)
            for jj in range(nouter_loop_samples):
                # only test intermediate quantities associated with design
                # chosen by the OED step
                idx = oed.collected_design_indices
                obs_jj = oed_copy.outer_loop_obs[jj:jj+1, idx]

                noise_cov_inv_jj = np.eye(idx.shape[0])/noise_std**2
                exact_post_mean_jj, exact_post_cov_jj = \
                    laplace_posterior_approximation_for_linear_models(
                        Amat[idx, :],
                        prior_mean, prior_cov_inv, noise_cov_inv_jj, obs_jj.T)

                exact_deviations[jj] = gauss_deviation_fun(
                    exact_post_mean_jj, exact_post_cov_jj)
            # print('d', np.absolute(exact_deviations-deviations[:, 0]).max(),
            #       tol)
            # print('eee', exact_deviations, deviations[:, 0])
            assert np.allclose(exact_deviations, deviations[:, 0], atol=tol)
            assert np.allclose(
                utility_vals[selected_indices], -np.mean(exact_deviations),
                atol=tol)

    def test_prediction_based_oed(self):
        def gauss_oed_variance_deviation(mean, cov):
            # compute variance of exp(X) where X has mean and variance
            # mean, cov. I.e. compute vairance of lognormal
            mu_g, sigma_g = mean.sum(), np.sqrt(cov.sum())
            pred_variable = stats.lognorm(s=sigma_g, scale=np.exp(mu_g))
            return pred_variable.var()

        ndesign_vars = 1
        self.help_compare_prediction_based_oed(
            oed_variance_deviation, gauss_oed_variance_deviation, True,
            21, ndesign_vars, 1e-8)

        ndesign_vars = 2
        self.help_compare_prediction_based_oed(
            oed_variance_deviation, gauss_oed_variance_deviation, True,
            21, ndesign_vars, 1e-8)

        def gauss_cvar_fun(p, mean, cov):
            # compute conditionalv value at risk of exp(X) where X has
            # mean and variance mean, cov. I.e. compute CVaR of lognormal
            mu_g, sigma_g = mean.sum(), np.sqrt(cov.sum())
            mean = np.exp(mu_g+sigma_g**2/2)
            value_at_risk = np.exp(mu_g+sigma_g*np.sqrt(2)*erfinv(2*p-1))
            cvar = mean*stats.norm.cdf(
                (mu_g+sigma_g**2-np.log(value_at_risk))/sigma_g)/(1-p)
            return cvar-mean

        beta = 0.5
        ndesign_vars = 1
        self.help_compare_prediction_based_oed(
            partial(oed_conditional_value_at_risk_deviation, beta,
                    samples_sorted=True), partial(gauss_cvar_fun, beta),
            True, 301, ndesign_vars, 3e-3)

        # Warning can only use samples_sorted=True for a single QoI
        # it will return an incorrect answer when more QoI are present
        # but this is difficult to catch if we continue to allow the user
        # to specify custom deviation functions
        beta = 0.5
        ndesign_vars = 1
        self.help_compare_prediction_based_oed(
            partial(oed_conditional_value_at_risk_deviation, beta,
                    samples_sorted=True), partial(gauss_cvar_fun, beta),
            False, 10000, ndesign_vars, 3e-2)

        beta = 0.5
        ndesign_vars = 2
        self.help_compare_prediction_based_oed(
            partial(oed_conditional_value_at_risk_deviation, beta,
                    samples_sorted=True), partial(gauss_cvar_fun, beta),
            False, 10000, ndesign_vars, 7e-2)

    def check_get_posterior_2d_interpolant_from_oed_data(
            self, method, rtol, ninner_loop_samples_1d):
        np.random.seed(1)
        nrandom_vars = 2
        noise_std = 1
        ndesign = 4
        nouter_loop_samples = 10

        ncandidates = 11
        design_candidates = np.linspace(-1, 1, ncandidates)[None, :]
        Amat = np.hstack((design_candidates.T, design_candidates.T**2))

        def obs_fun(samples):
            assert design_candidates.ndim == 2
            assert samples.ndim == 2
            return Amat.dot(samples).T

        prior_variable = IndependentMultivariateRandomVariable(
            [stats.norm(0, 1)]*nrandom_vars)

        x_quad, w_quad = get_oed_inner_quadrature_rule(
            ninner_loop_samples_1d, prior_variable, method)
        ninner_loop_samples = x_quad.shape[1]

        def generate_inner_prior_samples_gauss(n):
            # use precomputed samples so to avoid cost of regenerating
            assert n == x_quad.shape[1]
            return x_quad, w_quad

        generate_inner_prior_samples = generate_inner_prior_samples_gauss

        init_design_indices = np.array([ncandidates//2])
        oed = BayesianBatchKLOED(
            design_candidates, obs_fun, noise_std, prior_variable,
            nouter_loop_samples, ninner_loop_samples,
            generate_inner_prior_samples)
        oed.populate()
        oed.set_collected_design_indices(init_design_indices)

        for ii in range(1, ndesign):
            utility_vals, selected_indices = oed.update_design()

        nn, outer_loop_idx = 2, 0
        fun = get_posterior_2d_interpolant_from_oed_data(
            oed, prior_variable, nn, outer_loop_idx, method)
        samples = generate_independent_random_samples(prior_variable, 100)
        post_vals = fun(samples)

        prior_mean = oed.prior_variable.get_statistics('mean')
        prior_cov = np.diag(prior_variable.get_statistics('var')[:, 0])
        prior_cov_inv = np.linalg.inv(prior_cov)
        noise_cov_inv = np.eye(nn)/noise_std**2
        obs = oed.outer_loop_obs[
            outer_loop_idx, oed.collected_design_indices[:nn]][None, :]
        exact_post_mean, exact_post_cov = \
            laplace_posterior_approximation_for_linear_models(
                Amat[oed.collected_design_indices[:nn], :], prior_mean,
                prior_cov_inv, noise_cov_inv, obs.T)
        true_pdf = stats.multivariate_normal(
                mean=exact_post_mean[:, 0], cov=exact_post_cov).pdf
        true_post_vals = true_pdf(samples.T)
        # II = np.where(
        #     np.absolute(post_vals[:, 0]-true_post_vals) >
        #     rtol*np.absolute(true_post_vals))[0]
        # print(np.absolute(post_vals[:, 0]-true_post_vals)[II],
        #       rtol*np.absolute(true_post_vals)[II])
        assert np.allclose(post_vals[:, 0], true_post_vals, rtol=rtol)

        # plot_2d_posterior_from_oed_data(oed, prior_variable, 2, 0, method)

    def test_get_posterior_2d_interpolant_from_oed_data(self):
        ninner_loop_samples_1d = 301
        method, rtol = "linear", 2e-2
        self.check_get_posterior_2d_interpolant_from_oed_data(
            method, rtol, ninner_loop_samples_1d)
        ninner_loop_samples_1d = 301
        method, rtol = "quadratic", 2e-2
        # note rtol is the same for quadratic and linear because
        # linear interpolation is used for both even though different order
        # quadrature rules are used to compute evidence
        self.check_get_posterior_2d_interpolant_from_oed_data(
            method, rtol, ninner_loop_samples_1d)
        ninner_loop_samples_1d = 101
        method, rtol = "gauss", 1e-6
        self.check_get_posterior_2d_interpolant_from_oed_data(
            method, rtol, ninner_loop_samples_1d)

    def test_oed_entropic_risk_deviation(self):
        ninner_samples = int(1e4)
        vals = np.vstack((
            np.random.normal(0, 1, (1, ninner_samples)),
            np.random.normal(-1, 2, (1, ninner_samples)))
                         )[:, :, None]
        weights = np.ones((2, ninner_samples))/ninner_samples
        true_risk_measures = [
            np.log(np.exp(1/2))-0, np.log(np.exp(-1+4/2))--1]
        assert np.allclose(
            oed_entropic_deviation(vals, weights)[:, 0], true_risk_measures,
            atol=1e-2)


if __name__ == "__main__":
    bayesian_oed_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestBayesianOED)
    unittest.TextTestRunner(verbosity=2).run(bayesian_oed_test_suite)
    
