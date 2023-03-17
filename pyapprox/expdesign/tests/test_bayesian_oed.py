import unittest
import copy
import numpy as np
from scipy import stats
from functools import partial
from scipy.special import erfinv
import itertools

from pyapprox.expdesign.bayesian_oed import (
    gaussian_loglike_fun, d_optimal_utility, oed_variance_deviation,
    get_posterior_2d_interpolant_from_oed_data, oed_entropic_deviation,
    oed_prediction_average, get_data_risk_fun,
    get_deviation_fun, extract_independent_noise_cov,
    sequential_oed_synthetic_observation_process,
    gaussian_noise_fun, get_bayesian_oed_optimizer, OEDQOIDeviation)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.orthopoly.quadrature import gauss_hermite_pts_wts_1D
from pyapprox.bayes.laplace import (
    laplace_posterior_approximation_for_linear_models
)
from pyapprox.variables.risk import (
    conditional_value_at_risk, lognormal_variance,
    lognormal_cvar_deviation, gaussian_cvar, lognormal_kl_divergence,
    gaussian_kl_divergence, lognormal_cvar, lognormal_mean,
    conditional_value_at_risk_vectorized
)
from pyapprox.variables.tests.test_risk_measures import (
    get_lognormal_example_exact_quantities
)
from pyapprox.bayes.laplace import laplace_evidence
from pyapprox.variables.transforms import (
    AffineTransform
)
from pyapprox.util.utilities import (
    cartesian_product, outer_product
)
from pyapprox.surrogates.interp.tensorprod import (
    piecewise_univariate_linear_quad_rule)
from pyapprox.variables.algebra import (
    weighted_sum_dependent_gaussian_variables
)
from pyapprox.surrogates.interp.indexing import compute_hyperbolic_indices
from pyapprox.surrogates.interp.monomial import monomial_basis_matrix


def setup_linear_gaussian_model_inference(prior_variable, noise_std, obs_mat):
    # prior_variable = IndependentMarginalsVariable(
    #     [stats.norm(0, 1)]*nrandom_vars)
    nobs = obs_mat.shape[0]
    prior_mean = prior_variable.get_statistics("mean")
    prior_cov = np.diag(prior_variable.get_statistics("var")[:, 0])
    if type(noise_std) == np.ndarray:
        noise_cov = np.diag(noise_std**2)
    else:
        noise_cov = np.eye(nobs)*noise_std**2
    prior_cov_inv = np.linalg.inv(prior_cov)
    noise_cov_inv = np.linalg.inv(noise_cov)
    return prior_mean, prior_cov, noise_cov, prior_cov_inv, noise_cov_inv


def posterior_mean_data_stats(
        prior_mean, prior_cov, prior_cov_inv, post_cov, omat, nmat, nmat_inv):
    """
    Compute the mean and variance of the posterior mean with respect to
    uncertainty in the observation data. The posterior mean is a
    Gaussian variable

    Parameters
    ----------
    prior_mean : np.ndarray (nvars, 1)
        The mean of the prior

    prior_cov : np.ndarray (nvars, nvars)
        The covariance of the prior

    prior_cov_inv : np.ndarray (nvars, nvars)
        The inverse of the covariance of the prior

    post_cov : np.ndarray (nvars, nvars)
        The covariance of the posterior

    omat : np.ndaray (nobs, nvars)
        The linear observational model

    nmat : np.ndarray (nvars, nvars)
        The covariance of the observational noise

    nmat_inv : np.ndarray (nvars, nvars)
        The inverse of the covariance of the observational noise

    Returns
    -------
    nu_vec : np.ndarray (nvars, 1)
        The mean of the posterior mean

    Cmat : np.ndarray (nvars, nvars)
        The covariance of the posterior mean
    """
    Rmat = np.linalg.multi_dot(
        (post_cov, omat.T, nmat_inv))
    ROmat = Rmat.dot(omat)
    nu_vec = (
        np.linalg.multi_dot((ROmat, prior_mean)) +
        np.linalg.multi_dot((post_cov, prior_cov_inv,
                             prior_mean)))
    Cmat = (np.linalg.multi_dot((ROmat, prior_cov, ROmat.T)) +
            np.linalg.multi_dot((Rmat, nmat, Rmat.T)))
    return nu_vec, Cmat


def expected_kl_divergence_gaussian_inference(
        prior_mean, prior_cov, prior_cov_inv, post_cov, Cmat, nu_vec):
    """
    Compute the expected KL divergence between a Gaussian posterior and prior
    where average is taken with respect to the data

    prior_mean : np.ndarray (nvars, 1)
        The mean of the prior

    prior_cov : np.ndarray (nvars, nvars)
        The covariance of the prior

    prior_cov_inv : np.ndarray (nvars, nvars)
        The inverse of the covariance of the prior

    post_cov : np.ndarray (nvars, nvars)
        The covariance of the posterior

    nu_vec : np.ndarray (nvars, 1)
        The mean of the posterior mean

    Cmat : np.ndarray (nvars, nvars)
        The covariance of the posterior mean

    Returns
    -------
    kl_div : float
        The KL divergence
    """
    nvars = prior_mean.shape[0]
    kl_div = np.trace(prior_cov_inv.dot(post_cov))-nvars
    kl_div += np.log(
        np.linalg.det(prior_cov)/np.linalg.det(post_cov))
    kl_div += np.trace(prior_cov_inv.dot(Cmat))
    xi = prior_mean-nu_vec
    kl_div += np.linalg.multi_dot((xi.T, prior_cov_inv, xi))
    kl_div *= 0.5
    return kl_div[0, 0]


def compute_linear_gaussian_prior_prediction_stats(
        pred_mat, mean, cov, deviation_quantile, nonlinear):
    push_forward_cov = pred_mat.dot(cov.dot(pred_mat.T))
    pointwise_post_variance = np.diag(push_forward_cov)[:, None]
    push_forward_mean = pred_mat.dot(mean)
    if not nonlinear:
        if deviation_quantile is None:
            return np.sqrt(pointwise_post_variance)
        return gaussian_cvar(
            push_forward_mean, np.sqrt(pointwise_post_variance),
            deviation_quantile)

    if deviation_quantile is None:
        return np.sqrt(lognormal_variance(
            push_forward_mean, pointwise_post_variance))

    pointwise_post_stat = np.empty_like(pointwise_post_variance)
    for ii in range(pointwise_post_variance.shape[0]):
        pointwise_post_stat[ii] = lognormal_cvar(
            deviation_quantile, push_forward_mean[ii],
            pointwise_post_variance[ii])
    return pointwise_post_stat


def compute_linear_gaussian_oed_prediction_stats(
        nonlinear, deviation_quantile, post_cov, pred_mat, nu_vec, Cmat,
        data_quantile):
    """
    Compute the deviation of the posterior push-forward obtained from
    a linear Gaussian model
    """
    push_forward_cov = pred_mat.dot(post_cov.dot(pred_mat.T))
    pointwise_post_variance = np.diag(push_forward_cov)[:, None]
    if not nonlinear:
        # data quantile does not effect linear oed. because deviation
        # will be the same for all realizations of the observational data
        if deviation_quantile is None:
            pointwise_post_stat = np.sqrt(pointwise_post_variance)
        else:
            pointwise_post_stat = gaussian_cvar(
                0, np.sqrt(pointwise_post_variance), deviation_quantile)
    else:
        tau_hat = pred_mat.dot(nu_vec)
        sigma_hat_sq = np.diag(
            np.linalg.multi_dot((pred_mat, Cmat, pred_mat.T)))[:, None]
        tmp = np.exp(pointwise_post_variance)
        if deviation_quantile is None:
            factor = np.sqrt((tmp-1)*tmp)
        else:
            factor = np.sqrt(tmp)*(1/(1-deviation_quantile)*stats.norm.cdf(
                np.sqrt(pointwise_post_variance) -
                np.sqrt(2)*erfinv(2*deviation_quantile-1))-1)
        if data_quantile is None:
            pointwise_post_stat = factor*lognormal_mean(tau_hat, sigma_hat_sq)
        else:
            pointwise_post_stat = np.empty_like(tau_hat)
            for ii in range(tau_hat.shape[0]):
                pointwise_post_stat[ii] = factor[ii]*lognormal_cvar(
                    data_quantile, tau_hat[ii, 0], sigma_hat_sq[ii, 0])
    return pointwise_post_stat


def compute_linear_gaussian_oed_utility(
        oed_type, prior_mean, prior_cov, prior_cov_inv, post_cov, pred_mat,
        nu_vec, Cmat, pred_risk_fun, pointwise_post_stat):
    """
    Compute the expected utility for a linear Gaussian model
    """
    if oed_type == "KL-param":
        kl_div = expected_kl_divergence_gaussian_inference(
            prior_mean, prior_cov, prior_cov_inv, post_cov, Cmat,
            nu_vec)
        return kl_div

    if oed_type == "KL-pred":
        push_pr_mean = pred_mat.dot(prior_mean)
        push_pr_cov = pred_mat.dot(prior_cov.dot(pred_mat.T))
        push_post_cov = pred_mat.dot(post_cov.dot(pred_mat.T))
        nu_push = pred_mat.dot(nu_vec)
        C_push = pred_mat.dot(Cmat.dot(pred_mat.T))
        push_pr_cov_inv = np.linalg.inv(push_pr_cov)
        kl_div = expected_kl_divergence_gaussian_inference(
            push_pr_mean, push_pr_cov, push_pr_cov_inv,
            push_post_cov, C_push, nu_push)
        return kl_div
        # NOTE: The KL of two lognormals is the same as the KL of the
        # two associated Normals so we don't need two cases
        # if nolinear:
        #     assert pred_mat.shape[0] == 1
        #     kl_div = (
        #         1/(2*push_pr_cov)*(
        #             C_push+(push_pr_mean-nu_push)**2 +
        #             push_post_cov-push_pr_cov) +
        #         np.log(np.sqrt(push_pr_cov/push_post_cov)))
        #     return kl_div

    if oed_type != "dev-pred":
        raise ValueError(f"Incorrect oed_type: {oed_type}")

    if pred_risk_fun is None:
        assert pointwise_post_stat.shape == (1, 1)
        return -pointwise_post_stat[0, 0]
    utility_val = -pred_risk_fun(pointwise_post_stat)
    return utility_val


def linear_gaussian_posterior_from_observation_subset(
        obs, obs_mat, noise_cov_inv, prior_cov_inv, prior_mean, idx):
    obs_ii = obs[idx]
    omat = obs_mat[idx, :]
    nmat_inv = extract_independent_noise_cov(noise_cov_inv, idx)
    post_mean, post_cov = \
        laplace_posterior_approximation_for_linear_models(
            omat, prior_mean, prior_cov_inv, nmat_inv, obs_ii)
    return post_mean, post_cov, obs_ii, omat, nmat_inv


def linear_gaussian_prediction_deviation_based_oed(
        design_candidates, ndesign, prior_mean, prior_cov,
        noise_cov, obs_mat, pred_mat, pred_risk_fun,
        pre_collected_design_indices=[], oed_type="linear-pred",
        deviation_quantile=None,
        round_decimals=16, nonlinear=True, data_quantile=None,
        nnew=1):
    """
    Parameters
    ----------
    design_candidates : np.ndarray (nvars, nsamples)
        The location of all design sample candidates

    ndesign : integer
        The number of experiments to select

    prior_mean : np.ndarray (nvars, 1)
        The mean of the prior

    prior_cov : np.ndarray (nvars, nvars)
        The covariance of the prior

    noise_cov : np.ndarray (nvars, nvars)
        The covariance of the observational noise

    obs_mat : np.ndaray (nobs, nvars)
        The linear observational model

    pred_mat : np.ndaray (nobs, nvars)
        The linear prediction model

    pred_risk_fun : callable
        Function to compute the risk measure of the predictions stats
        over the prediction space

    pre_collected_design_indices : array_like
        The indices of the experiments that must be in the final design

    oed_type : string
        "linear-pred" - Divergence of prediction model pred_mat.dot(sample)
        "nonlinear-pred" - Divergence of prediction model np.exp(pred_mat.dot(sample))
        "linear-kl" - KL divergence of parameters posterior from prior

    deviation_quantile : float
        The deviation quantile of the conditional value at risk used to
        compute the
        statistic summarizing the variability at each prediction point
        If None then the statistic is standard deviation

    round_decimals : integer
        The number of decimal places to round utility_vals to when choosing
        the optimal design. This can be useful when comparing with
        numerical solutions where 2 designs are equivalent analytically
        but numerically there are slight differences that causes design to be
        different

    nonlinear : boolean
        True - prediction is exp(pred_mat.dot(samples))
        False - prediction is pred_mat.dot(samples)

    data_quantile : float
        The quantile of the conditional value at risk used to
        compute the statistic summarizing the deviation at each
        prediction point for all outerloop observation data

    nnew : integer
        The number of design points to choose at once.

    Returns
    -------
    collected_design_indices : np.ndarray (nobs)
        The indices into the qoi vector associated with the
        collected observations

    data_history : list
        Each entry of the list contains the following for each design candidate

    pointwise_post_stat : np.ndarray (nqoi, 1)
        The pointwise statistic at each prediction location

    risk_val : float
        The scalar -pred_risk_fun(pointwise_post_stat)
    """
    prior_sample = prior_mean + np.linalg.cholesky(prior_cov).dot(
        np.random.normal(0, 1, (prior_mean.shape[0], 1)))
    obs = obs_mat.dot(prior_sample) + np.linalg.cholesky(noise_cov).dot(
        np.random.normal(0, 1, (obs_mat.shape[0], 1)))

    prior_cov_inv = np.linalg.inv(prior_cov)
    noise_cov_inv = np.linalg.inv(noise_cov)

    data_history = []
    assert len(pre_collected_design_indices) < ndesign
    # make deep copy
    # collected_indices = [ii for ii in pre_collected_design_indices]
    # for jj in range(len(pre_collected_design_indices), ndesign):
    # for testing purposes compute utlity even for pre collected design indices
    # so we can plot this information
    collected_indices = np.asarray([], dtype=int)
    assert ndesign % nnew == 0
    assert len(pre_collected_design_indices) % nnew == 0
    new_indices = np.asarray(list(itertools.combinations_with_replacement(
            np.arange(design_candidates.shape[1]), nnew)))
    for jj in range(ndesign//nnew):
        data = []
        for ii in range(new_indices.shape[0]):
            # realization of data does not matter so if location appears
            # twice we can just use the same data
            idx = np.hstack((collected_indices, new_indices[ii])).astype(int)
            post_cov, obs_ii, omat, nmat_inv = \
                linear_gaussian_posterior_from_observation_subset(
                    obs, obs_mat, noise_cov_inv, prior_cov_inv, prior_mean,
                    idx)[1:]
            nmat = extract_independent_noise_cov(noise_cov, idx)
            nu_vec, Cmat = posterior_mean_data_stats(
                prior_mean, prior_cov, prior_cov_inv, post_cov, omat, nmat,
                nmat_inv)
            pointwise_post_stat = compute_linear_gaussian_oed_prediction_stats(
                nonlinear, deviation_quantile, post_cov, pred_mat, nu_vec,
                Cmat, data_quantile)
            utility_val = compute_linear_gaussian_oed_utility(
                oed_type, prior_mean, prior_cov, prior_cov_inv, post_cov,
                pred_mat, nu_vec, Cmat, pred_risk_fun,
                pointwise_post_stat)
            data.append({"expected_deviations": pointwise_post_stat,
                         "utility_val": utility_val})
        data_history.append(data)
        if jj < len(pre_collected_design_indices):
            selected_indices = (
                pre_collected_design_indices[jj*nnew:(jj+1)*nnew])
        else:
            # round digits to remove numerical noise
            # print(([d["utility_val"] for d in data_history[-1]]))
            selected_idx = np.argmax(
                np.round([d["utility_val"] for d in data_history[-1]],
                         round_decimals))
            selected_indices = new_indices[selected_idx]
        collected_indices = np.hstack((collected_indices, selected_indices))
        # print(collected_indices.shape, len(data_history))
    return collected_indices, data_history


def conditional_posterior_moments_gaussian_linear_model(
        obs_mat, prior_mean, prior_cov_inv, noise_cov_inv, idx, xx):
    assert xx.shape[1] == 1
    nvars = obs_mat.shape[1]
    obs = obs_mat[idx, :].dot(xx[:nvars, :])
    # need to add noise here instead of a priori so that if
    # idx has unique elements the observational noises are still independent
    nmat_inv = extract_independent_noise_cov(noise_cov_inv, idx)
    nmat = np.linalg.inv(nmat_inv)
    obs += np.linalg.cholesky(nmat).dot(
        np.random.normal(0, 1, (nmat.shape[0], 1)))
    mu_g, sigma_g_sq = \
        laplace_posterior_approximation_for_linear_models(
            obs_mat[idx, :], prior_mean, prior_cov_inv,
            nmat_inv, obs)
    return mu_g, sigma_g_sq


def conditional_posterior_prediction_moments_gaussian_linear_model(
        obs_mat, prior_mean, prior_cov_inv, noise_cov_inv, pred_mat, idx, xx):
    mu_g, sigma_g_sq = conditional_posterior_moments_gaussian_linear_model(
        obs_mat, prior_mean, prior_cov_inv, noise_cov_inv, idx, xx)
    # compute variance of polynomial
    mu, sigma_sq = weighted_sum_dependent_gaussian_variables(
        mu_g, sigma_g_sq, pred_mat.T)
    return mu, sigma_sq


def conditional_lognormal_stat(
        p, obs_mat, prior_mean, prior_cov_inv, noise_cov_inv, Bmat,
        idx, xx):
    mu, sigma_sq = \
        conditional_posterior_prediction_moments_gaussian_linear_model(
            obs_mat, prior_mean, prior_cov_inv, noise_cov_inv, Bmat, idx, xx)
    # compute statistic of exp(polynomial)
    if p is None:
        # return lognormal_variance(mu, sigma_sq)
        return np.sqrt(lognormal_variance(mu, sigma_sq))
    return lognormal_cvar_deviation(p, mu, sigma_sq)


def conditional_gaussian_stat(
        p, obs_mat, prior_mean, prior_cov_inv, noise_cov_inv, Bmat,
        idx, xx):
    mu, sigma_sq = \
        conditional_posterior_prediction_moments_gaussian_linear_model(
            obs_mat, prior_mean, prior_cov_inv, noise_cov_inv, Bmat, idx, xx)
    # compute statistic of polynomial
    if p is None:
        # return lognormal_variance(mu, sigma_sq)
        return np.sqrt(sigma_sq)[:, None]
    return (gaussian_cvar(mu, np.sqrt(sigma_sq), p)-mu)[:, None]


def conditional_gaussian_kl_param_stat(
        obs_mat, prior_mean, prior_cov_inv, noise_cov_inv, idx, xx):
    mu_g, sigma_g_sq = conditional_posterior_moments_gaussian_linear_model(
        obs_mat, prior_mean, prior_cov_inv, noise_cov_inv, idx, xx)
    kl_div = gaussian_kl_divergence(
        mu_g, sigma_g_sq, prior_mean, np.linalg.inv(prior_cov_inv))
    return np.array([[kl_div]])


def conditional_gaussian_kl_pred_stat(
        obs_mat, prior_mean, prior_cov_inv, noise_cov_inv, pred_mat, prior_cov,
        idx, xx):
    post_mean, post_cov = conditional_posterior_moments_gaussian_linear_model(
        obs_mat, prior_mean, prior_cov_inv, noise_cov_inv, idx, xx)
    push_post_mean = pred_mat.dot(post_mean)
    push_post_cov = pred_mat.dot(post_cov.dot(pred_mat.T))
    push_pr_mean = pred_mat.dot(prior_mean)
    push_pr_cov = pred_mat.dot(prior_cov.dot(pred_mat.T))
    kl_div = gaussian_kl_divergence(
        push_post_mean, push_post_cov, push_pr_mean, push_pr_cov)
    return np.array([[kl_div]])


def conditional_lognormal_kl_pred_stat(
        obs_mat, prior_mean, prior_cov_inv, noise_cov_inv, pred_mat, prior_cov,
        idx, xx):
    post_mean, post_cov = conditional_posterior_moments_gaussian_linear_model(
        obs_mat, prior_mean, prior_cov_inv, noise_cov_inv, idx, xx)
    push_post_mean = pred_mat.dot(post_mean)
    push_post_std = np.sqrt(pred_mat.dot(post_cov.dot(pred_mat.T)))
    push_pr_mean = pred_mat.dot(prior_mean)
    push_pr_std = np.sqrt(pred_mat.dot(prior_cov.dot(pred_mat.T)))
    kl_div = lognormal_kl_divergence(
        push_post_mean, push_post_std, push_pr_mean, push_pr_std)
    return kl_div


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

    def _check_loglike_fun(self, noise_std, active_indices, design=None,
                           degree=1):

        nvars = degree+1
        if design is None:
            design = np.linspace(-1, 1, 4)[None, :]

        obs_matrix = design.T**(np.arange(degree+1)[None, :])

        def fun(design, samples):
            assert design.ndim == 2
            assert samples.ndim == 2
            return obs_matrix.dot(samples).T

        prior_mean = np.zeros((nvars, 1))
        prior_cov = np.eye(nvars)

        true_sample = np.ones((nvars, 1))*0.4
        obs = fun(design, true_sample)

        if type(noise_std) == np.ndarray:
            assert noise_std.shape[0] == obs.shape[1] and noise_std.ndim == 2
            obs += np.random.normal(0, 1, obs.shape)*noise_std.T
        else:
            obs += np.random.normal(0, 1, obs.shape)*noise_std

        noise_cov_inv = np.eye(obs.shape[1])/(noise_std**2)

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
                    extract_independent_noise_cov(
                        noise_cov_inv, active_indices),
                    obs[:, active_indices].T)

        n_xx = 100
        lb, ub = stats.norm(0, 1).interval(0.99)
        xx = cartesian_product([np.linspace(lb, ub, n_xx)]*nvars)
        true_pdf_vals = stats.multivariate_normal(
            exact_post_mean[:, 0], cov=exact_post_cov).pdf(xx.T)[:, None]

        def prior_pdf(x):
            return np.atleast_1d(stats.multivariate_normal(
                prior_mean[:, 0], cov=prior_cov).pdf(x.T))[:, None]
        pred_obs = fun(design, xx)
        lvals = (np.exp(
            gaussian_loglike_fun(obs, pred_obs, noise_std, active_indices)) *
                 prior_pdf(xx))
        assert lvals.shape == (xx.shape[1], 1)

        xx_gauss, ww_gauss = gauss_hermite_pts_wts_1D(400)
        xx_gauss = cartesian_product([xx_gauss]*nvars)
        ww_gauss = outer_product([ww_gauss]*nvars)
        pred_obs = fun(design, xx_gauss)
        loglike_vals = gaussian_loglike_fun(
            obs, pred_obs, noise_std, active_indices)[:, 0]
        evidence = np.exp(loglike_vals).dot(ww_gauss)
        post_pdf_vals = lvals/evidence

        post_cov = np.cov(
            xx_gauss, aweights=ww_gauss*np.exp(loglike_vals), ddof=0)
        # print((post_cov, exact_post_cov))
        assert np.allclose(post_cov, exact_post_cov)

        gauss_evidence = laplace_evidence(
            lambda x: np.exp(gaussian_loglike_fun(
                obs, fun(design, x), noise_std, active_indices)[:, 0]),
            prior_pdf, exact_post_cov, exact_post_mean)
        # print(evidence, gauss_evidence)
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
        self._check_loglike_fun(0.3, None)
        np.random.seed(1)
        self._check_loglike_fun(np.array([[0.25, 0.3, 0.35, 0.4]]).T, None)
        active_indices = np.array([0, 1, 2, 3])
        np.random.seed(1)
        self._check_loglike_fun(0.3, active_indices)
        np.random.seed(1)
        self._check_loglike_fun(
            np.array([[0.25, 0.3, 0.35, 0.4]]).T, active_indices)
        active_indices = np.array([0, 2, 3])
        np.random.seed(1)
        self._check_loglike_fun(0.3, active_indices)
        np.random.seed(1)
        self._check_loglike_fun(
            np.array([[0.25, 0.3, 0.35, 0.4]]).T, active_indices)
        np.random.seed(1)
        # repeated observations at the same design location
        design = np.array([[-1, -1]])
        self._check_loglike_fun(
            np.array([[0.25, 0.25]]).T, [0, 0], design)

    def _check_gaussian_loglike_fun_3d(self, noise_std, active_indices):
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
            obs[:, :], pred_obs[:, None, :], noise_std,
            active_indices)[:, 0]
        assert np.allclose(loglike, loglike_3d)

        loglike_3d_econ = gaussian_loglike_fun(
            np.vstack([obs[:, :]]*pred_obs.shape[0]),
            pred_obs[:, None, :], noise_std,
            active_indices)[:, 0]
        assert np.allclose(loglike_3d, loglike_3d_econ)

        loglike_3d_econ = gaussian_loglike_fun(
            np.vstack([obs[:, None, :]]*pred_obs.shape[0]),
            np.hstack([pred_obs[:, None, :]]*3), noise_std,
            active_indices)[:, 0]
        assert np.allclose(loglike_3d, loglike_3d_econ)

    def test_gaussian_loglike_fun_3d(self):
        self._check_gaussian_loglike_fun_3d(0.3, None)
        self._check_loglike_fun(
            np.array([[0.25, 0.3, 0.35, 0.4]]).T, None)
        self._check_gaussian_loglike_fun_3d(0.3, np.array([0, 2, 3]))
        self._check_gaussian_loglike_fun_3d(
            np.array([[0.25, 0.3, 0.35, 0.4]]).T, np.array([0, 2, 3]))

    def _setup_linear_gaussian_problem(self, prior_std):
        nrandom_vars = 1
        design = np.linspace(-1, 1, 2)[None, :]
        Amat = design.T

        def obs_fun(x):
            return (Amat.dot(x)).T

        prior_variable = IndependentMarginalsVariable(
            [stats.norm(0, prior_std)]*nrandom_vars)

        return obs_fun, prior_variable, Amat, design

    def _check_compute_expected_kl_utility(
            self, out_quad_opts, in_quad_opts, prior_std, tol):

        # import warnings
        # warnings.filterwarnings('error')
        # warnings.filterwarnings(
        #     "error", category=np.VisibleDeprecationWarning)

        noise_std = .3
        (obs_fun, prior_variable, Amat, design_candidates) = \
            self._setup_linear_gaussian_problem(prior_std)

        ndesign_candidates = design_candidates.shape[1]
        init_design_indices = np.array([0])
        oed = get_bayesian_oed_optimizer(
            "kl_params", ndesign_candidates, obs_fun, noise_std,
            prior_variable, out_quad_opts, in_quad_opts,
            pre_collected_design_indices=init_design_indices)

        utility_vals, selected_indices, results = oed.select_design(
            oed.collected_design_indices, 1, True)
        new_design_indices = np.array([1])
        # new_design_indices = selected_indices
        utility = utility_vals[new_design_indices[0]]
        # print(utility_vals[new_design_indices[0]])

        prior_mean = prior_variable.get_statistics('mean')
        prior_cov = np.diag(prior_variable.get_statistics('var')[:, 0])
        prior_cov_inv = np.linalg.inv(prior_cov)
        noise_cov_inv = np.eye(Amat.shape[0])/noise_std**2

        kl_divs = []
        active_indices = np.hstack(
            (oed.collected_design_indices, new_design_indices))
        noise = oed.noise_samples[:, :active_indices.shape[0]]
        noisy_out_pred_obs = oed.out_pred_obs+noise
        for ii in range(oed.nout_samples):
            idx = np.hstack(
                (oed.collected_design_indices, new_design_indices))
            # obs_ii = out_obs_copy[ii:ii+1, idx]
            obs_ii = noisy_out_pred_obs[ii:ii+1, idx]

            idx = np.hstack((oed.collected_design_indices, new_design_indices))
            exact_post_mean, exact_post_cov = \
                laplace_posterior_approximation_for_linear_models(
                    Amat[idx, :], prior_mean, prior_cov_inv,
                    extract_independent_noise_cov(noise_cov_inv, idx),
                    obs_ii.T)

            kl_div = gaussian_kl_divergence(
                exact_post_mean, exact_post_cov, prior_mean, prior_cov)
            kl_divs.append(kl_div)

        ave_kl_divs = np.sum(kl_divs*oed.out_quad_data[1][:, 0])
        print(utility-ave_kl_divs, utility, ave_kl_divs)
        assert np.allclose(utility, ave_kl_divs, rtol=tol)

    def test_compute_expected_kl_utility(self):
        test_scenarios = [
            [{"method": "quasimontecarlo", "kwargs": {"nsamples": 1000}},
             {"method": "quasimontecarlo", "kwargs": {"nsamples": 1000}},
             1, 2e-3],
            [{"method": "quasimontecarlo", "kwargs": {"nsamples": 1000}},
             {"method": "quasimontecarlo", "kwargs": {"nsamples": 1000}},
             2, 2e-3],
            [{"method": "quasimontecarlo", "kwargs": {"nsamples": 1000}},
             {"method": "tensorproduct",
              "kwargs": {"levels": 200, "rule": "gauss"}},
             2, 2e-3],
            [{"method": "tensorproduct",
              "kwargs": {"levels": 30, "rule": "quadratic"}},
             {"method": "tensorproduct",
              "kwargs": {"levels": 200, "rule": "quadratic"}},
             2, 2e-3]
        ]
        for test_scenario in test_scenarios:
            self._check_compute_expected_kl_utility(*test_scenario)

    def _obs_fun(self, Amat, obs_locations, samples):
        assert obs_locations.ndim == 2
        assert samples.ndim == 2
        return Amat.dot(samples).T

    def _check_batch_kl_oed(self, ndata_per_candidate, nnew):
        """
        No observations collected to inform subsequent designs
        """
        np.random.seed(1)
        nrandom_vars = 2
        noise_std = 1
        ndesign = 5
        nprocs = 1

        ndesign_candidates = 11
        obs_locations = np.linspace(
            -1, 1, ndesign_candidates*ndata_per_candidate)[None, :]

        Amat = np.hstack(
            (np.ones((ndesign_candidates*ndata_per_candidate, 1)),
             obs_locations.T))

        # def obs_fun(samples):
        #     assert obs_locations.ndim == 2
        #     assert samples.ndim == 2
        #     return Amat.dot(samples).T
        obs_fun = partial(self._obs_fun, Amat, obs_locations)

        prior_variable = IndependentMarginalsVariable(
            [stats.norm(0, 1)]*nrandom_vars)

        out_quad_opts = {
            "method": "quasimontecarlo", "kwargs": {"nsamples": 10000}}
        in_quad_opts = {
            "method": "quasimontecarlo", "kwargs": {"nsamples": 1000}}

        # Define initial design
        init_design_indices = np.array([ndesign_candidates//2])
        oed = get_bayesian_oed_optimizer(
            "kl_params", ndesign_candidates, obs_fun, noise_std,
            prior_variable, out_quad_opts, in_quad_opts,
            pre_collected_design_indices=init_design_indices,
            max_ncollected_obs=ndesign*ndata_per_candidate, nprocs=nprocs,
            ndata_per_candidate=ndata_per_candidate)

        indices = np.asarray(list(itertools.combinations_with_replacement(
            np.arange(ndesign_candidates), nnew)))

        for ii in range(len(init_design_indices), ndesign, nnew):
            # loop must be before oed.updated design because
            # which updates oed.collected_design_indices and thus
            # changes problem
            d_utility_vals = np.zeros(indices.shape[0])
            for kk, idx in enumerate(indices):
                design_indices = np.hstack(
                    (oed.collected_design_indices, idx))
                active_indices = np.hstack(
                    [idx*ndata_per_candidate + np.arange(
                        ndata_per_candidate) for idx in design_indices])
                Amat_idx = Amat[active_indices]
                d_utility_vals[kk] = d_optimal_utility(Amat_idx, noise_std)

            utility_vals, selected_indices = oed.update_design(nnew=nnew)[:2]
            # ignore entries of previously collected data
            II = np.where(d_utility_vals > 0)[0]
            # print(d_utility_vals[II])
            # print(utility_vals[II])
            # print((np.absolute(
            #     d_utility_vals[II]-utility_vals[II])/d_utility_vals[II]).max())
            assert np.allclose(d_utility_vals[II], utility_vals[II], rtol=1e-2)

    def test_batch_kl_oed(self):
        test_cases = [[1, 1], [2, 1], [1, 2]]
        for test_case in test_cases:
            self._check_batch_kl_oed(*test_case)

    def test_batch_kl_data_risk(self):
        np.random.seed(1)
        nrandom_vars = 1 # 2
        noise_std = 1
        ndesign = 1
        nprocs = 1
        ndata_per_candidate = 1
        nnew = 1

        ndesign_candidates = 11
        obs_locations = np.linspace(
            -1, 1, ndesign_candidates*ndata_per_candidate)[None, :]

        Amat = np.hstack(
            (np.ones((ndesign_candidates*ndata_per_candidate, 1)),
             obs_locations.T))[:, :nrandom_vars]

        obs_fun = partial(self._obs_fun, Amat, obs_locations)

        prior_variable = IndependentMarginalsVariable(
            [stats.norm(0, 1)]*nrandom_vars)

        out_quad_opts = {
            "method": "quasimontecarlo", "kwargs": {"nsamples": 10}}
        # out_quad_opts = {
        #     "method": "tensorproduct",
        #     "kwargs": {"levels": 30, "rule": "gauss"}}
        in_quad_opts = {
            "method": "tensorproduct",
            "kwargs": {"levels": 100, "rule": "gauss"}}

        data_quantile = 0.5
        data_risk_fun = get_data_risk_fun(
            "cvar", {"quantile": data_quantile})
        # data_risk_fun = get_data_risk_fun("mean")

        # Define initial design
        init_design_indices = np.array([ndesign_candidates//2])
        init_design_indices = np.empty(0, dtype=int)
        oed = get_bayesian_oed_optimizer(
            "kl_params", ndesign_candidates, obs_fun, noise_std,
            prior_variable, out_quad_opts, in_quad_opts,
            pre_collected_design_indices=init_design_indices,
            max_ncollected_obs=ndesign*ndata_per_candidate, nprocs=nprocs,
            ndata_per_candidate=ndata_per_candidate,
            data_risk_fun=data_risk_fun)

        prior_mean, prior_cov, noise_cov, prior_cov_inv, _ = (
            setup_linear_gaussian_model_inference(
                prior_variable, noise_std, Amat))

        for ii in range(len(init_design_indices), ndesign):
            utilities, _, results = oed.update_design(True)

        kl_divs = np.empty(oed.nout_samples)
        idx = oed.collected_design_indices
        noise_cov_inv_idx = np.eye(idx.shape[0])/noise_std**2
        for jj in range(oed.nout_samples):
            noisy_obs = oed.out_pred_obs[jj:jj+1, idx] +\
                    oed.noise_samples[jj, :idx.shape[0]]
            post_mean, post_cov = \
                laplace_posterior_approximation_for_linear_models(
                    Amat[idx], prior_mean, prior_cov_inv,
                    noise_cov_inv_idx, noisy_obs.T)
            gauss_evidence = laplace_evidence(
                lambda x: np.exp(gaussian_loglike_fun(
                    noisy_obs,
                    obs_fun(x)[:, idx],
                    noise_std))[:, 0],
                lambda y: np.atleast_2d(prior_variable.pdf(y.T)).T,
                post_cov, post_mean)
            loglike_val = gaussian_loglike_fun(
                noisy_obs, oed.out_pred_obs[jj:jj+1, idx],
                noise_std)[:, 0]
            # kl_divs is not exactly the KL divergence
            kl_divs[jj] = loglike_val-np.log(gauss_evidence)
        ref_utilities = data_risk_fun(kl_divs[:, None], oed.out_weights)
        # print(ref_utilities, utilities[idx[-1]])
        assert np.allclose(ref_utilities, utilities)

    def _check_batch_prediction_oed(self, ndata_per_candidate, nnew):
        """
        No observations collected to inform subsequent designs
        """
        np.random.seed(1)
        noise_std = 1
        ndesign = 4
        degree = 2
        nrandom_vars = degree+1
        nprocs = 1

        nprediction_samples = 51 # 201
        quantile = 0.8
        pred_risk_fun = partial(conditional_value_at_risk, alpha=quantile)

        ndesign_candidates = 11
        obs_locations = np.linspace(
            -1, 1, ndesign_candidates*ndata_per_candidate)[None, :]

        prediction_candidates = np.linspace(
            -1, 1, nprediction_samples)[None, :]

        def basis_matrix(degree, samples):
            return samples.T**np.arange(degree+1)[None, :]

        def obs_fun(samples):
            assert obs_locations.ndim == 2
            assert samples.ndim == 2
            Amat = basis_matrix(degree, obs_locations)
            return Amat.dot(samples).T

        def qoi_fun(samples):
            Amat = basis_matrix(degree, prediction_candidates)
            qoi = Amat.dot(samples).T
            return qoi

        prior_variable = IndependentMarginalsVariable(
            [stats.norm(0, 1)]*nrandom_vars)

        # outerloop samples does not effect this problem
        # because variance is independent of noise only on design
        out_quad_opts = {
            "method": "montecarlo", "kwargs": {"nsamples": 2}}
        # in_quad_opts = {
        #     "method": "quasimontecarlo", "kwargs": {"nsamples": 10000}}
        in_quad_opts = {
            "method": "tensorproduct",
            "kwargs": {"levels": 41, "rule": "quadratic"}}

        # Define initial design
        # init_design_indices = np.array([ndesign_candidates//2])
        init_design_indices = np.empty((0), dtype=int)
        oed = get_bayesian_oed_optimizer(
            "dev_pred", ndesign_candidates, obs_fun, noise_std,
            prior_variable, out_quad_opts, in_quad_opts,
            qoi_fun=qoi_fun,
            pre_collected_design_indices=init_design_indices,
            deviation_fun=OEDQOIDeviation("variance"),
            pred_risk_fun=pred_risk_fun,
            max_ncollected_obs=ndesign*ndata_per_candidate, nprocs=nprocs,
            ndata_per_candidate=ndata_per_candidate)

        for ii in range(len(init_design_indices), ndesign, nnew):
            utility_vals, selected_indices = oed.update_design(False, nnew)[:2]

        prior_mean = prior_variable.get_statistics("mean")
        prior_cov = np.diag(prior_variable.get_statistics("var")[:, 0])
        prior_cov_inv = np.linalg.inv(prior_cov)
        obs_matrix = basis_matrix(degree, obs_locations)
        noise_cov_inv = np.eye(obs_locations.shape[1])/(noise_std**2)
        pred_matrix = basis_matrix(degree, prediction_candidates)

        # print(oed.collected_design_indices)
        # print(design_candidates[0, oed.collected_design_indices])

        indices = np.asarray(list(itertools.combinations_with_replacement(
            np.arange(ndesign_candidates), nnew)))

        # Check expected variance when choosing the final design
        # point. Compare with exact value computed using Laplace formula
        # Note variance is independent of data so no need to generate
        # realizations of data
        ii = 0
        data = []
        for idx in indices:
            design_indices = np.hstack((
                oed.collected_design_indices[:-nnew], idx))
            active_idx = np.hstack([idx*ndata_per_candidate + np.arange(
                ndata_per_candidate) for idx in design_indices])
            # realization of data does not matter so just take noisless obs
            obs_ii = oed.out_pred_obs[ii:ii+1, active_idx]
            exact_post_mean, exact_post_cov = \
                laplace_posterior_approximation_for_linear_models(
                    obs_matrix[active_idx, :], prior_mean, prior_cov_inv,
                    extract_independent_noise_cov(noise_cov_inv, active_idx),
                    obs_ii.T)
            pointwise_post_variance = np.diag(
                pred_matrix.dot(exact_post_cov.dot(pred_matrix.T))
            )[:, None]
            exact_variance_risk = oed.pred_risk_fun(pointwise_post_variance)
            data.append([pointwise_post_variance, -exact_variance_risk])
            # print(f"Candidate {idx}", exact_variance_risk)
        jdx = np.argmax(np.round(np.array([d[1] for d in data]), 4))
        selected_index = indices[jdx]
        # because of rounding error do not use following assert but the one after
        # that rounds
        # assert np.allclose(selected_index, oed.collected_design_indices[-nnew:])
        assert np.allclose(
            selected_index, indices[np.argmax(np.round(utility_vals, 4))])
        # print(utility_vals[oed.collected_design_indices[-1]]-data[jdx][1])
        assert np.allclose(utility_vals[jdx], data[jdx][1], rtol=1e-4)
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

    def test_batch_prediction_oed(self):
        test_cases = [[1, 1], [2, 1], [1, 2]]
        for test_case in test_cases:
            self._check_batch_prediction_oed(*test_case)

    def _check_sequential_kl_oed(self, oed_type, step_tols):
        """
        Observations collected ARE used to inform subsequent designs
        """
        degree = 0
        nrandom_vars = degree+1
        noise_std = 1
        ndesign = 5

        out_quad_opts = {
            "method": "quasimontecarlo", "kwargs": {"nsamples": 1e4}}
        in_quad_opts = {
            "method": "tensorproduct",
            "kwargs": {"levels": 51, "rule": "gauss"}}

        ndesign_candidates = 6
        design_candidates = np.linspace(-1, 1, ndesign_candidates)[None, :]
        Amat = design_candidates.T**np.arange(degree+1)[None, :]
        pred_mat = Amat[:1, :]

        def obs_fun(samples):
            assert design_candidates.ndim == 2
            assert samples.ndim == 2
            return Amat.dot(samples).T

        prior_variable = IndependentMarginalsVariable(
            [stats.norm(0, 1)]*nrandom_vars)

        true_sample = np.array([.4]*nrandom_vars)[:, None]
        obs_process = partial(
            sequential_oed_synthetic_observation_process,
            obs_fun, true_sample, partial(gaussian_noise_fun, noise_std))

        if "kl_params" not in oed_type:
            pred_risk_fun = oed_prediction_average
            def qoi_fun(samples): return pred_mat.dot(samples).T
            kwargs = {
                "qoi_fun": qoi_fun, "pred_risk_fun": pred_risk_fun,
                "deviation_fun": OEDQOIDeviation("variance")}
        else:
            kwargs = {}
        kwargs["obs_process"] = obs_process

        # Define initial design
        init_design_indices = np.array([ndesign_candidates//2])
        oed = get_bayesian_oed_optimizer(
            oed_type, ndesign_candidates, obs_fun, noise_std,
            prior_variable, out_quad_opts, in_quad_opts,
            pre_collected_design_indices=init_design_indices,
            max_ncollected_obs=ndesign,
            **kwargs)
        print(oed)
        # following assumes oed.econ = True
        x_quad = oed.in_samples
        w_quad = oed.in_weights

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
        for step in range(len(init_design_indices), ndesign):
            current_design = design_candidates[:, oed.collected_design_indices]
            noise_cov_inv = np.eye(current_design.shape[1])/noise_std**2

            # Compute posterior moving from previous posterior and using
            # only the most recently collected data
            noise_cov_inv_incr = np.eye(
                selected_indices.shape[0])/noise_std**2
            exact_post_mean, exact_post_cov = \
                laplace_posterior_approximation_for_linear_models(
                    Amat[selected_indices, :],
                    exact_post_mean_prev, np.linalg.inv(exact_post_cov_prev),
                    noise_cov_inv_incr, oed.collected_obs[:, -1:].T)

            # check using current posterior as prior and only using new
            # data (above) produces the same posterior as using original prior
            # and all collected data (from_prior approach). The posteriors
            # should be the same but the evidences will be difference.
            # This is tested below
            exact_post_mean_from_prior, exact_post_cov_from_prior = \
                laplace_posterior_approximation_for_linear_models(
                    Amat[oed.collected_design_indices, :],
                    prior_mean, prior_cov_inv, noise_cov_inv,
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
                obs_fun(x_quad)[:, oed.collected_design_indices[-1:]],
                noise_std))[:, 0]
            # we must divide integrand by initial prior_pdf since it is
            # already implicilty included via the quadrature weights
            integrand_vals = quad_loglike_vals*post_var_prev.pdf(
                x_quad.T)/prior_variable.pdf(x_quad)[:, 0]
            quad_evidence = integrand_vals.dot(w_quad)
            # print(quad_evidence, gauss_evidence)
            assert np.allclose(gauss_evidence, quad_evidence), step

            print('G', gauss_evidence, oed.evidence)
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
            utility_vals, selected_indices = oed.update_design()[:2]
            new_obs = oed.obs_process(selected_indices)
            oed.update_observations(new_obs)
            utility = utility_vals[selected_indices]
            # ensure noise realizations are the same for both approaches
            oed_copy.noise_samples = oed.noise_samples

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
            exact_evidences = np.empty(oed.nout_samples)
            exact_stats = np.empty_like(exact_evidences)
            for jj in range(oed.nout_samples):
                # Fill obs with those predicted by outer loop sample
                idx = oed.collected_design_indices
                # Merge collected observations and new predicted observation.
                noisy_obs = oed_copy.out_pred_obs[jj:jj+1, selected_indices] +\
                    oed_copy.noise_samples[jj, :selected_indices.shape[0]]
                # noisy_obs = oed_copy.get_out_obs(selected_indices)[jj:jj+1, :]
                obs_jj = np.hstack((oed_copy.collected_obs, noisy_obs))

                # Compute the posterior obtained by using predicted value
                # of outer loop sample
                noise_cov_inv_jj = np.eye(
                    selected_indices.shape[0])/noise_std**2
                exact_post_mean_jj, exact_post_cov_jj = \
                    laplace_posterior_approximation_for_linear_models(
                        Amat[selected_indices, :],
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
                    obs_jj[:, -1:], obs_fun(x_quad)[:, selected_indices],
                    noise_std))[:, 0]
                integrand_vals = quad_loglike_vals*post_var.pdf(
                    x_quad.T)/prior_variable.pdf(x_quad)[:, 0]
                quad_evidence = integrand_vals.dot(w_quad)
                # print(quad_evidence, gauss_evidence_jj)
                assert np.allclose(gauss_evidence_jj, quad_evidence), step

                # Check that evidence of moving from current posterior
                # to new posterior with (potential data from outer-loop sample)
                # is equal to the evidence of moving from
                # intitial prior to new posterior divided by the evidence
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

                if oed_type == "kl_params":
                    gauss_kl_div = gaussian_kl_divergence(
                        exact_post_mean_jj, exact_post_cov_jj,
                        exact_post_mean, exact_post_cov)
                    exact_stats[jj] = gauss_kl_div
                else:
                    exact_stats[jj] = -pred_mat.dot(
                        exact_post_cov_jj.dot(pred_mat.T))[0, 0]
                    # print(results["deviations"][jj], -exact_stats[jj], "D")
                    assert np.allclose(
                        results["deviations"][jj], -exact_stats[jj])

            print(step, evidences[:, 0]-exact_evidences)
            assert np.allclose(evidences[:, 0], exact_evidences)

            # Outer loop samples are from prior. Use importance reweighting
            # to sample from previous posterior. This step is only relevant
            # for open loop design (used here)
            # where observed data informs current estimate
            # of parameters. Closed loop design (not used here)
            # never collects data and so it always samples from the prior.
            # post_weights = post_var.pdf(
            #     oed.out_prior_samples.T)/prior_variable.pdf(
            #         oed.out_prior_samples)[:, 0]/oed.nout_samples
            post_weights = post_var.pdf(
                oed.out_prior_samples.T)/prior_variable.pdf(
                    oed.out_prior_samples)[:, 0]*oed.out_weights[:, 0]
            # accuracy of expected KL depends on nout_samples because
            # these samples are used to compute the kl divergences
            laplace_utility = np.sum(exact_stats*post_weights)
            print(utility, laplace_utility, "U")
            print(exact_stats)
            print('u', (utility-laplace_utility)/laplace_utility, step,
                  step_tols[step-1])
            assert np.allclose(
                utility, laplace_utility, rtol=step_tols[step-1])

            exact_post_mean_prev = exact_post_mean
            exact_post_cov_prev = exact_post_cov
            post_var_prev = post_var

    def test_sequential_kl_oed(self):
        self._check_sequential_kl_oed("kl_params", [2e-3, 3.e-3, 3e-3, 3e-3])
        self._check_sequential_kl_oed("dev_pred", [2e-15, 3e-12, 3e-6, 9e-9])

    def test_bayesian_importance_sampling_avar(self):
        np.random.seed(1)
        nrandom_vars = 2
        Amat = np.array([[-0.5, 1]])
        noise_std = 0.1
        prior_variable = IndependentMarginalsVariable(
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
        prior_samples = prior_variable.rvs(nsamples)
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
            self, deviation_fun, gauss_deviation_fun, inner_quad_type,
            nin_samples, ndesign_vars, tol):
        ndesign_candidates_1d = 5
        design_candidates = cartesian_product(
            [np.linspace(-1, 1, ndesign_candidates_1d)]*ndesign_vars)
        ndesign_candidates = design_candidates.shape[1]

        # Define model used to predict likely observable data
        indices = compute_hyperbolic_indices(ndesign_vars, 1)[:, 1:]
        Amat = monomial_basis_matrix(indices, design_candidates)
        obs_fun = partial(linear_obs_fun, Amat)

        # Define model used to predict unobservable QoI
        qoi_fun = exponential_qoi_fun

        # Define the prior PDF of the unknown variables
        nrandom_vars = indices.shape[1]
        prior_variable = IndependentMarginalsVariable(
            [stats.norm(0, 0.5)]*nrandom_vars)

        # Define the independent observational noise
        noise_std = 1

        # Define initial design
        init_design_indices = np.array([ndesign_candidates//2])

        # Define OED options
        nout_samples = 9  # 100
        out_quad_opts = {
            "method": "montecarlo", "kwargs": {"nsamples": nout_samples}}
        if (inner_quad_type == "quasimontecarlo" or
                inner_quad_type == "montecarlo"):
            in_quad_opts = {
                "method": inner_quad_type, "kwargs": {"nsamples": nin_samples}}
        elif inner_quad_type == "gauss":
            in_quad_opts = {
                "method": "tensorproduct",
                "kwargs": {"levels": nin_samples, "rule": "gauss"}}
        elif inner_quad_type == "quadratic" or inner_quad_type == "linear":
            in_quad_opts = {
                "method": "tensorproduct",
                "kwargs": {"levels": nin_samples, "rule": inner_quad_type}}

        # Define initial design
        init_design_indices = np.array([ndesign_candidates//2])

        # Setup OED problem
        nexperiments = 3

        # np.random.seed(1)
        oed = get_bayesian_oed_optimizer(
            "dev_pred", ndesign_candidates, obs_fun, noise_std,
            prior_variable, out_quad_opts, in_quad_opts, qoi_fun=qoi_fun,
            pre_collected_design_indices=init_design_indices,
            deviation_fun=deviation_fun, max_ncollected_obs=nexperiments)

        prior_mean = oed.prior_variable.get_statistics('mean')
        prior_cov = np.diag(prior_variable.get_statistics('var')[:, 0])
        prior_cov_inv = np.linalg.inv(prior_cov)
        selected_indices = init_design_indices

        # Generate experimental design
        for step in range(len(init_design_indices), nexperiments):
            # Copy current state of OED before new data is determined
            # This copy will be used to compute Laplace based utility and
            # evidence values for testing
            oed_copy = copy.deepcopy(oed)

            # Update the design
            utility_vals, selected_indices = oed.update_design()[:2]
            # make sure oed_copy has the same copy of noise_realizations as
            # oed which may have updated its copy when update_design was called
            oed_copy.noise_samples = oed.noise_samples

            results = oed_copy.compute_expected_utility(
                oed_copy.collected_design_indices, selected_indices, True)
            # utility = results["utility_val"]
            deviations = results["deviations"]

            exact_deviations = np.empty(nout_samples)
            for jj in range(nout_samples):
                # only test intermediate quantities associated with design
                # chosen by the OED step
                idx = oed.collected_design_indices
                # obs_jj = oed.get_out_obs(idx)[jj:jj+1, :]
                obs_jj = (oed.out_pred_obs[jj:jj+1, idx] +
                          oed.noise_samples[jj, :idx.shape[0]])

                noise_cov_inv_jj = np.eye(idx.shape[0])/noise_std**2
                exact_post_mean_jj, exact_post_cov_jj = \
                    laplace_posterior_approximation_for_linear_models(
                        Amat[idx, :],
                        prior_mean, prior_cov_inv, noise_cov_inv_jj, obs_jj.T)

                exact_deviations[jj] = gauss_deviation_fun(
                    exact_post_mean_jj, exact_post_cov_jj)

                # nsamples = 100000
                # L = np.linalg.cholesky(exact_post_cov_jj)
                # samples = exact_post_mean_jj+L.dot(
                #     np.random.normal(0, 1, (L.shape[0], nsamples)))
                # vals = qoi_fun(samples)
                # # print(vals.var(), 'variance', jj, exact_deviations[jj])
                # # cvar_dev = conditional_value_at_risk(vals, 0.5)-vals.mean()
                # from pyapprox.variables.risk import (
                #     conditional_value_at_risk_vectorized)
                # cvar = conditional_value_at_risk_vectorized(
                #     vals.T, 0.5)
                # cvar_dev = cvar-vals.mean()
                # print(idx)
                # print(cvar, vals.mean())
                # print(cvar_dev, 'cvar', jj, exact_deviations[jj])
                # # assert False

            print('d', np.absolute(exact_deviations-deviations[:, 0]).max(),
                  tol)
            # print('eee', exact_deviations, deviations[:, 0])
            assert np.allclose(exact_deviations, deviations[:, 0], atol=tol)
            assert np.allclose(
                utility_vals[selected_indices],
                # -np.mean(exact_deviations),
                -np.sum(oed.out_weights[:, 0]*exact_deviations),
                atol=tol)

    def test_prediction_based_oed(self):
        def lognorm_oed_variance_deviation(mean, cov):
            # compute variance of exp(X) where X has mean and variance
            # mean, cov. I.e. compute variance of lognormal
            mu_g, sigma_g = mean.sum(), np.sqrt(cov.sum())
            pred_variable = stats.lognorm(s=sigma_g, scale=np.exp(mu_g))
            return pred_variable.var()

        def lognorm_oed_cvar_deviation(p, mean, cov):
            # compute conditionalv value at risk of exp(X) where X has
            # mean and variance mean, cov. I.e. compute CVaR of lognormal
            mu_g, sigma_g = mean.sum(), np.sqrt(cov.sum())
            # mean = np.exp(mu_g+sigma_g**2/2)
            # value_at_risk = np.exp(mu_g+sigma_g*np.sqrt(2)*erfinv(2*p-1))
            # cvar = mean*stats.norm.cdf(
            #     (mu_g+sigma_g**2-np.log(value_at_risk))/sigma_g)/(1-p)
            # dev = cvar-mean
            dev = lognormal_cvar_deviation(p, mu_g, sigma_g**2)
            return dev

        beta = 0.5
        oed_variance_deviation = OEDQOIDeviation("variance")
        oed_cvar_deviation = OEDQOIDeviation("cvar", beta)
        test_cases = [
            [oed_variance_deviation, lognorm_oed_variance_deviation, "gauss",
             21, 1, 1e-8],
            [oed_variance_deviation, lognorm_oed_variance_deviation,
             "quasimontecarlo", 100000, 1, 2e-3],
            [oed_variance_deviation, lognorm_oed_variance_deviation, "gauss",
             21, 2, 1e-8],
            [oed_cvar_deviation, partial(lognorm_oed_cvar_deviation, beta),
             "gauss", 301, 1, 3e-3],
            [oed_cvar_deviation, partial(lognorm_oed_cvar_deviation, beta),
             "linear", 301, 1, 3e-3],
            [oed_cvar_deviation, partial(lognorm_oed_cvar_deviation, beta),
             "quadratic", 301, 1, 3e-3],
            [oed_cvar_deviation, partial(lognorm_oed_cvar_deviation, beta),
             "quasimontecarlo", 10000, 1, 4e-3],
            [oed_cvar_deviation, partial(lognorm_oed_cvar_deviation, beta),
             "quasimontecarlo", 10000, 2, 3e-3]
        ]
        for test_case in test_cases:
            print("#")
            self.help_compare_prediction_based_oed(*test_case)

    def check_get_posterior_2d_interpolant_from_oed_data(
            self, rule, rtol, nin_samples_1d):
        np.random.seed(1)
        nrandom_vars = 2
        noise_std = 1
        ndesign = 4
        nout_samples = 10

        ndesign_candidates = 11
        design_candidates = np.linspace(-1, 1, ndesign_candidates)[None, :]
        Amat = np.hstack((design_candidates.T, design_candidates.T**2))

        def obs_fun(samples):
            assert design_candidates.ndim == 2
            assert samples.ndim == 2
            return Amat.dot(samples).T

        prior_variable = IndependentMarginalsVariable(
            [stats.norm(0, 1)]*nrandom_vars)

        out_quad_opts = {
            "method": "quasimontecarlo", "kwargs": {"nsamples": nout_samples}}
        in_quad_opts = {
            "method": "tensorproduct",
            "kwargs": {"levels": nin_samples_1d, "rule": rule}}

        init_design_indices = np.array([ndesign_candidates//2])
        oed = get_bayesian_oed_optimizer(
            "kl_params", ndesign_candidates, obs_fun, noise_std,
            prior_variable, out_quad_opts, in_quad_opts,
            pre_collected_design_indices=init_design_indices,
            max_ncollected_obs=ndesign)

        for ii in range(1, ndesign):
            utility_vals, selected_indices = oed.update_design()[:2]

        nn, out_idx = 2, 0
        fun = get_posterior_2d_interpolant_from_oed_data(
            oed, prior_variable, nn, out_idx, rule)
        samples = prior_variable.rvs(100)
        post_vals = fun(samples)

        prior_mean = oed.prior_variable.get_statistics('mean')
        prior_cov = np.diag(prior_variable.get_statistics('var')[:, 0])
        prior_cov_inv = np.linalg.inv(prior_cov)
        noise_cov_inv = np.eye(nn)/noise_std**2
        # obs = oed.get_out_obs(oed.collected_design_indices[:nn])[
        #     out_idx][None, :]
        obs = (
            oed.out_pred_obs[out_idx, oed.collected_design_indices[:nn]] +
            oed.noise_samples[out_idx, :nn])[None, :]
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

        # from pyapprox.expdesign.bayesian_oed import (
        #     plot_2d_posterior_from_oed_data)
        # import matplotlib.pyplot as plt
        # plot_2d_posterior_from_oed_data(oed, prior_variable, 2, 0, rule)
        # plt.show()

    def test_get_posterior_2d_interpolant_from_oed_data(self):
        nin_samples_1d = 301
        method, rtol = "linear", 2e-2
        self.check_get_posterior_2d_interpolant_from_oed_data(
            method, rtol, nin_samples_1d)
        nin_samples_1d = 301
        method, rtol = "quadratic", 2e-2
        # note rtol is the same for quadratic and linear because
        # linear interpolation is used for both even though different order
        # quadrature rules are used to compute evidence
        self.check_get_posterior_2d_interpolant_from_oed_data(
            method, rtol, nin_samples_1d)
        nin_samples_1d = 101
        method, rtol = "gauss", 1e-6
        self.check_get_posterior_2d_interpolant_from_oed_data(
            method, rtol, nin_samples_1d)

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

    def check_analytical_gaussian_prediction_deviation_based_oed(
            self, nonlinear, oed_type, quantile, rtol, noise_std=.5,
            nnew=1, ndesign=2, nsamples=None):
        np.random.seed(1)
        degree = 3
        ndesign_candidates = 11
        pre_collected_design_indices = []
        lb, ub = -0.5, 0.5
        if nsamples is None:
            nsamples = int(1e4)

        def basis_matrix(degree, samples):
            return samples.T**np.arange(degree+1)[None, :]

        nprediction_samples = ndesign_candidates
        xx, ww = piecewise_univariate_linear_quad_rule(
            [lb, ub], nprediction_samples)
        # uncomment if care about prediction everywhere
        prediction_candidates = xx[None, :]
        if quantile is not None or nonlinear or oed_type == "KL-pred":
        # set this so only care about prediction at left point
            prediction_candidates = xx[None, :1]
            ww = ww[:1]*0+1

        def pred_risk_fun(x):
            return oed_prediction_average(x, weights=ww[:, None])[0]

        nrandom_vars = degree+1
        prior_variable = IndependentMarginalsVariable(
            # [stats.norm(-0.25, .5)]*nrandom_vars)
            [stats.norm(0.0, .5)]*nrandom_vars)

        design_candidates = np.linspace(lb, ub, ndesign_candidates)[None, :]
        obs_mat = basis_matrix(degree, design_candidates)
        pred_mat = basis_matrix(degree, prediction_candidates)

        (prior_mean, prior_cov, noise_cov, prior_cov_inv,
         noise_cov_inv) = setup_linear_gaussian_model_inference(
             prior_variable, noise_std, obs_mat)

        collected_indices, data = \
            linear_gaussian_prediction_deviation_based_oed(
                design_candidates, ndesign, prior_mean, prior_cov,
                noise_cov, obs_mat, pred_mat, pred_risk_fun,
                pre_collected_design_indices, oed_type, quantile,
                nonlinear=nonlinear, nnew=nnew)
        print(collected_indices)

        random_samples = prior_variable.rvs(nsamples)

        if oed_type == "dev-pred" and not nonlinear:
            stat_fun = partial(
                conditional_gaussian_stat, quantile, obs_mat, prior_mean,
                prior_cov_inv, noise_cov_inv, pred_mat)
        elif oed_type == "dev-pred" and nonlinear:
            stat_fun = partial(
                conditional_lognormal_stat, quantile, obs_mat, prior_mean,
                prior_cov_inv, noise_cov_inv, pred_mat)
        elif oed_type == "KL-param":
            stat_fun = partial(
                conditional_gaussian_kl_param_stat,
                obs_mat, prior_mean, prior_cov_inv, noise_cov_inv)
        elif oed_type == "KL-pred" and not nonlinear:
            stat_fun = partial(
                conditional_gaussian_kl_pred_stat,
                obs_mat, prior_mean, prior_cov_inv, noise_cov_inv, pred_mat,
                prior_cov)
        elif oed_type == "KL-pred" and nonlinear:
            # NOTE: The KL of two lognormals is the same as the KL of the
            # two associated Normals so we don't need two cases, this
            # test is making sure this is true
            stat_fun = partial(
                conditional_lognormal_kl_pred_stat,
                obs_mat, prior_mean, prior_cov_inv, noise_cov_inv, pred_mat,
                prior_cov)
        else:
            raise ValueError(f"Incorrect oed_type {oed_type}")

        # print(data[-1][collected_indices[-1]]['utility_val'])

        xx = random_samples
        metrics = []
        for ii in range(nsamples):
            if quantile is None and not nonlinear:
                metrics.append(pred_risk_fun(
                    stat_fun(collected_indices, xx[:, ii:ii+1])))
            else:
                metrics.append(
                    stat_fun(collected_indices, xx[:, ii:ii+1])[0, 0])

        metrics = np.array(metrics)
        # print(metrics)
        if oed_type == "KL-param" or oed_type == "KL-pred":
            sign = 1
        else:
            sign = -1

        new_indices = np.asarray(
            list(itertools.combinations_with_replacement(
                np.arange(design_candidates.shape[1]), nnew)))
        last_selected_indices = collected_indices[-nnew:]
        for ii in range(new_indices.shape[0]):
            if np.allclose(last_selected_indices, new_indices[ii]):
                idx = ii
                break
        print(metrics.mean(axis=0) -
              sign*data[-1][idx]['utility_val'], 'err')
        assert np.allclose(
            metrics.mean(axis=0),
            sign*data[-1][idx]['utility_val'], rtol=rtol)

    def test_analytical_gaussian_prediction_deviation_based_oed(self):
        self.check_analytical_gaussian_prediction_deviation_based_oed(
            False, "dev-pred", None, 1e-2)
        self.check_analytical_gaussian_prediction_deviation_based_oed(
            False, "dev-pred", 0.8, 1e-2)
        self.check_analytical_gaussian_prediction_deviation_based_oed(
            True, "dev-pred", None, 2e-2)
        self.check_analytical_gaussian_prediction_deviation_based_oed(
            True, "dev-pred", 0.8, 2e-2)
        self.check_analytical_gaussian_prediction_deviation_based_oed(
            False, "KL-param", None, 1e-2)
        self.check_analytical_gaussian_prediction_deviation_based_oed(
            False, "KL-pred", None, 1e-2)
        self.check_analytical_gaussian_prediction_deviation_based_oed(
            True, "KL-pred", None, 1e-2)
        # test heteroscedastic noise
        noise_std = np.linspace(0.5, 1, 11)
        self.check_analytical_gaussian_prediction_deviation_based_oed(
            True, "KL-pred", None, 1e-2, noise_std)
        # test multiple indices selected nnew > 1
        self.check_analytical_gaussian_prediction_deviation_based_oed(
            False, "dev-pred", None, 1e-2, 1., 8, 8)

    def check_numerical_gaussian_prediction_deviation_based_oed(
            self, nonlinear, oed_type, deviation_quantile, pred_risk_quantile,
            nout_samples, nin_samples_1d, in_rule, tols,
            data_quantile=None):
        ndesign = 2
        degree = 1
        ndesign_candidates = 3
        noise_std = 1
        pre_collected_design_indices = [1]

        if deviation_quantile is None:
            deviation_fun = OEDQOIDeviation("std_dev")
        else:
            deviation_fun = OEDQOIDeviation("cvar", deviation_quantile)

        if data_quantile is None:
            data_risk_fun = get_data_risk_fun("mean")
        else:
            data_risk_fun = get_data_risk_fun(
                "cvar", {"quantile": data_quantile})

        def basis_matrix(degree, samples):
            return samples.T**np.arange(degree+1)[None, :]

        nprediction_samples = ndesign_candidates
        xx, ww = piecewise_univariate_linear_quad_rule(
            [-1, 1], nprediction_samples)
        ww /= 2.0  # assume uniform distribution over prediction space
        prediction_candidates = xx[None, :]

        if pred_risk_quantile is None:
            pred_risk_fun = oed_prediction_average
        else:
            pred_risk_fun = partial(
                conditional_value_at_risk, alpha=pred_risk_quantile,
                weights=ww, prob=False)

        nrandom_vars = degree+1
        prior_variable = IndependentMarginalsVariable(
            [stats.norm(0, .1)]*nrandom_vars)

        design_candidates = np.linspace(-1, 1, ndesign_candidates)[None, :]
        obs_mat = basis_matrix(degree, design_candidates)
        pred_mat = basis_matrix(degree, prediction_candidates)
        # pred_mat = pred_mat[:1, :]  # this line is hack for debugging

        (prior_mean, prior_cov, noise_cov, prior_cov_inv,
         noise_cov_inv) = setup_linear_gaussian_model_inference(
             prior_variable, noise_std, obs_mat)

        round_decimals = 4
        collected_indices, analytical_results = \
            linear_gaussian_prediction_deviation_based_oed(
                design_candidates, ndesign, prior_mean, prior_cov,
                noise_cov, obs_mat, pred_mat, pred_risk_fun,
                pre_collected_design_indices, oed_type, deviation_quantile,
                round_decimals=round_decimals, nonlinear=nonlinear,
                data_quantile=data_quantile)

        def obs_fun(x): return obs_mat.dot(x).T

        def qoi_fun(x):
            vals = pred_mat.dot(x).T
            if not nonlinear:
                return vals
            return np.exp(vals)

        out_quad_opts = {
            "method": "montecarlo", "kwargs": {"nsamples": nout_samples}}
        if "montecarlo" not in in_rule:
            in_quad_opts = {
                "method": "tensorproduct",
                "kwargs": {"levels": nin_samples_1d, "rule": in_rule}}
        else:
            in_quad_opts = {
                "method": in_rule, "kwargs": {"nsamples": nin_samples_1d}}

        oed = get_bayesian_oed_optimizer(
            "dev_pred", ndesign_candidates, obs_fun, noise_std,
            prior_variable, out_quad_opts, in_quad_opts, qoi_fun=qoi_fun,
            pre_collected_design_indices=pre_collected_design_indices,
            deviation_fun=deviation_fun,
            pred_risk_fun=pred_risk_fun, data_risk_fun=data_risk_fun)
        oed_results = []
        for step in range(len(pre_collected_design_indices), ndesign):
            # print("#", step)
            results_step = oed.update_design(
                return_all=True)[2]  # , rounding_decimals=round_decimals)[2]
            oed_results.append(results_step)

        # print(collected_indices, oed.collected_design_indices)
        # for jj in range(ndesign_candidates):
        #     print(analytical_results[0][jj]["expected_deviations"][:, 0],
        #           oed_results[0][jj]["expected_deviations"][:, 0], 'exp dev')
        # print(oed_results[0][jj]["deviations"][:, 0], 'dev')

        for ii in range(len(oed_results)):
            kk = ii+len(pre_collected_design_indices)
            print(collected_indices[kk], oed.collected_design_indices[kk])
            # analytical_results stores data for pre_collected_design_indices
            # and the subsequently collected indices, but the numerical code
            # does not
            anlyt_utility_vals = np.array(
                [d["utility_val"] for d in analytical_results[kk]])
            oed_utility_vals = np.array(
                [d["utility_val"] for d in oed_results[ii]])
            print((anlyt_utility_vals-oed_utility_vals)/anlyt_utility_vals,
                  'rel err')
            print(anlyt_utility_vals, '\n', oed_utility_vals, 'vals')
            assert np.allclose(
               oed_utility_vals, anlyt_utility_vals, rtol=tols[0])
            # this depends on rounding
            # assert (collected_indices[kk] == oed.collected_design_indices[kk])

        idx = collected_indices[-1]
        # print((analytical_results[-1][idx]["expected_deviations"] -
        #        oed_results[-1][idx]["expected_deviations"]) /
        #       analytical_results[-1][idx]["expected_deviations"])
        assert np.allclose(
            oed_results[-1][idx]["expected_deviations"],
            analytical_results[-1][idx]["expected_deviations"],
            rtol=tols[1])

    def test_numerical_gaussian_prediction_deviation_based_oed(self):
        test_cases = [
            [False, "dev-pred", None, None, 2, 30, "gauss", [1e-15, 1e-15]],
            [False, "dev-pred", None, None, 2, 10000, "quasimontecarlo",
             [2e-3, 2e-3]],
            [True, "dev-pred", None, None, int(2e4), 40, "gauss",
             [4e-4, 2e-4]],
            [True, "dev-pred", None, None, int(1e3), int(1e3),
             "quasimontecarlo", [7.1e-3, 1.2e-2]],
            [False, "dev-pred", 0.8, None, 2, 80, "linear", [1e-3, 1e-3]],
            [False, "dev-pred", 0.8, None, 2, int(3e3),
             "quasimontecarlo", [8e-3, 1.2e-2]],
            [True, "dev-pred", 0.8, None, int(1e3), 80, "linear",
             [2e-3, 2e-3]],
            [True, "dev-pred", 0.8, 0.9, int(1e3), 80, "linear", [2e-3, 2e-3]],
            [True, "dev-pred", 0.8, None, int(2e3), 80, "linear", [2e-3, 2e-3],
             0.90]
        ]
        for test_case in test_cases:
            print('#')
            np.random.seed(1)
            self.check_numerical_gaussian_prediction_deviation_based_oed(
                *test_case)

    def test_linear_gaussian_posterior_mean_moments(self):
        degree = 2
        noise_std = 0.5
        nrandom_vars = degree+1
        ndesign_candidates = 11
        prior_variable = IndependentMarginalsVariable(
            [stats.norm(1, 0.5)]*nrandom_vars)
        design_candidates = np.linspace(-1, 1, ndesign_candidates)[None, :]

        def basis_matrix(degree, samples):
            return samples.T**np.arange(degree+1)[None, :]
        obs_mat = basis_matrix(degree, design_candidates)
        (prior_mean, prior_cov, noise_cov, prior_cov_inv,
         noise_cov_inv) = setup_linear_gaussian_model_inference(
             prior_variable, noise_std, obs_mat)

        idx = np.array([0, 0])
        nmat_inv = extract_independent_noise_cov(noise_cov_inv, idx)
        nmat = extract_independent_noise_cov(noise_cov, idx)
        omat = obs_mat[idx, :]

        means = []
        nsamples = int(1e5)
        for ii in range(nsamples):
            sample = prior_variable.rvs(1)
            obs_ii = obs_mat.dot(sample)[idx] + np.linalg.cholesky(nmat).dot(
                np.random.normal(0, 1, (nmat.shape[0], 1)))
            post_mean, post_cov = \
                laplace_posterior_approximation_for_linear_models(
                    omat, prior_mean, prior_cov_inv, nmat_inv, obs_ii)
            means.append(post_mean)

        Rmat = np.linalg.multi_dot(
                (post_cov, omat.T, nmat_inv))
        ROmat = Rmat.dot(omat)
        nu_vec = (
            np.linalg.multi_dot((ROmat, prior_mean)) +
            np.linalg.multi_dot((post_cov, prior_cov_inv,
                                 prior_mean)))
        Cmat = (np.linalg.multi_dot((ROmat, prior_cov, ROmat.T)) +
                np.linalg.multi_dot((Rmat, nmat, Rmat.T)))
        means = np.array(means)[:, :, 0]
        print(np.mean(means, axis=0)-nu_vec)
        assert np.allclose(np.mean(means, axis=0), nu_vec, rtol=1e-2)

        print(np.cov(means, rowvar=False)-Cmat)
        assert np.allclose(np.cov(means, rowvar=False), Cmat, rtol=1e-2)

    def test_compute_linear_gaussian_prior_prediction_stats(self):
        noise_std = 0.5
        degree = 3
        nrandom_vars = degree+1
        ndesign_candidates = 11
        design_candidates = np.linspace(-1, 1, ndesign_candidates)[None, :]
        npred_candidates = 12
        pred_candidates = np.linspace(-1, 1, npred_candidates)[None, :]
        prior_variable = IndependentMarginalsVariable(
            [stats.norm(1, 0.5)]*nrandom_vars)

        def basis_matrix(degree, samples):
            return samples.T**np.arange(degree+1)[None, :]
        obs_mat = basis_matrix(degree, design_candidates)
        pred_mat = basis_matrix(degree, pred_candidates)

        (prior_mean, prior_cov, noise_cov, prior_cov_inv,
         noise_cov_inv) = setup_linear_gaussian_model_inference(
             prior_variable, noise_std, obs_mat)

        deviation_quantile, nonlinear = None, False
        pred_stats = compute_linear_gaussian_prior_prediction_stats(
            pred_mat, prior_mean, prior_cov, deviation_quantile, nonlinear)
        samples = prior_variable.rvs(int(1e6))
        pred_vals = pred_mat.dot(samples)
        # print(pred_vals.std(axis=1))
        # print(pred_stats[:, 0])
        assert np.allclose(pred_vals.std(axis=1), pred_stats[:, 0], atol=1e-2)

        deviation_quantile, nonlinear = 0.9, False
        pred_stats = compute_linear_gaussian_prior_prediction_stats(
            pred_mat, prior_mean, prior_cov, deviation_quantile, nonlinear)
        pred_stats_mc = conditional_value_at_risk_vectorized(
            pred_vals, deviation_quantile)
        # print(pred_vals.shape)
        # print(pred_stats[:, 0])
        # print(pred_stats_mc)
        assert np.allclose(pred_stats_mc, pred_stats[:, 0], atol=1e-2)

        deviation_quantile, nonlinear = None, True
        pred_stats = compute_linear_gaussian_prior_prediction_stats(
            pred_mat, prior_mean, prior_cov, deviation_quantile, nonlinear)
        # print(pred_stats)
        # print(np.exp(pred_vals).std(axis=1))
        # print(pred_stats[:, 0])
        assert np.allclose(
            np.exp(pred_vals).std(axis=1), pred_stats[:, 0], rtol=1e-2)

        deviation_quantile, nonlinear = 0.9, True
        pred_stats = compute_linear_gaussian_prior_prediction_stats(
            pred_mat, prior_mean, prior_cov, deviation_quantile, nonlinear)
        pred_stats_mc = conditional_value_at_risk_vectorized(
            np.exp(pred_vals), deviation_quantile)
        # print(pred_vals.shape)
        # print(pred_stats[:, 0])
        # print(pred_stats_mc)
        assert np.allclose(pred_stats_mc, pred_stats[:, 0], rtol=1e-2)


if __name__ == "__main__":
    bayesian_oed_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestBayesianOED)
    unittest.TextTestRunner(verbosity=2).run(bayesian_oed_test_suite)

# notes about fast oed by tempone
# We assume that the experiment can completely determine the model, i.e.,
# is a full rank matrix even without help from the prior. After equation (9)

# The key step is propagating the uncertainty from the parameters to the
# quantity of interest using a small noise approximation. Therefore, the pdf of the quantity of interest can be approximated by a Gaussian one.
# Last para section 2.


#TODO shrink oed.in_weights to just in weights for a single outerloop
# we now assume that all inner weights are the same
#Also shrink oed.in_pred_obs
