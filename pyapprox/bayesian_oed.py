import numpy as np
from scipy.spatial.distance import cdist
from numba import njit
from functools import partial

from pyapprox.risk_measures import conditional_value_at_risk
from pyapprox.probability_measure_sampling import (
    generate_independent_random_samples
)
from pyapprox.variable_transformations import (
    AffineRandomVariableTransformation
)
from pyapprox.utilities import (
    get_tensor_product_quadrature_rule,
    get_tensor_product_piecewise_polynomial_quadrature_rule
)
from pyapprox import get_univariate_quadrature_rules_from_variable
from pyapprox.barycentric_interpolation import (
    compute_barycentric_weights_1d,
    multivariate_barycentric_lagrange_interpolation
)


def gaussian_loglike_fun_broadcast(
        obs, pred_obs, noise_stdev, active_indices=None):
    """
    Conmpute the log-likelihood values from a set of real and predicted
    observations

    Parameters
    ----------
    obs : np.ndarray (nsamples, nobs)
        The real observations repeated for each sample

    pred_obs : np.ndarray (nsamples, nobs)
        The observations predicited by the model for a set of samples

    noise_stdev : float or np.ndarray (nobs, 1)
        The standard deviation of Gaussian noise added to each observation

    active_indices : np.ndarray (nobs, 1)
        The subset of indices of the observations used to compute the
        likelihood

    Returns
    -------
    llike : np.ndaray (nsamples, 1)

    Notes
    -----
    This can handle 1d, 2d, 3d arrays but is slower
    due to broadcasting when computing obs-pred_obs
    """
    if (type(noise_stdev) == np.ndarray and
            noise_stdev.shape[0] != obs.shape[-1]):
        raise ValueError("noise must be provided for each observation")

    if type(noise_stdev) != np.ndarray:
        noise_stdev = np.ones((obs.shape[-1], 1), dtype=float)*noise_stdev

    if active_indices is None:
        # avoid copy if possible
        # using special indexing with array e.g array[:, I] where I is an array
        # makes a copy which is slow
        tmp = 1/(2*noise_stdev[:, 0]**2)
        llike = 0.5*np.sum(np.log(tmp/np.pi))
        llike += np.sum(-(obs-pred_obs)**2*tmp, axis=-1)
    else:
        tmp = 1/(2*noise_stdev[active_indices, 0]**2)
        llike = 0.5*np.sum(np.log(tmp/np.pi))
        llike += np.sum(
            -(obs[..., active_indices]-pred_obs[..., active_indices])**2*tmp,
            axis=-1)
    if llike.ndim == 1:
        llike = llike[:, None]
    return llike


@njit(cache=True)
def sq_dists_numba_3d(XX, YY, a=1, b=0, active_indices=None):
    """
    Compute the scaled l2-norm distance between two sets of samples
    E.g. for one point


    a[ii]*(XX[ii]-YY[ii])+b


    Parameters
    ----------
    XX : np.ndarray (LL, 1, NN)
        The first set of samples

    YY : np.ndarray (LL, MM, NN)
        The second set of samples. The 3D arrays are useful when computing
        squared distances for multiple sets of samples

    a : float or np.ndarray (NN, 1)
        scalar multiplying l2 distance

    a : float or np.ndarray (NN, 1)
        scalar added to l2 distance

    Returns
    -------
    ss : np.ndarray (LL, MM)
        The scaled distances
    """
    Yshape = YY.shape
    assert XX.shape[0] == YY.shape[0]
    assert XX.shape[1] == 1
    assert a.shape[0] == XX.shape[2]
    assert a.shape[0] == YY.shape[2]
    ss = np.empty(Yshape[:2])
    if active_indices is None:
        active_indices = np.arange(Yshape[2])
    nactive_indices = active_indices.shape[0]
    for ii in range(Yshape[0]):
        for jj in range(Yshape[1]):
            ss[ii, jj] = 0.0
            # for kk in range(Yshape[2]):
            #     ss[ii, jj] += (XX[ii, 0, kk] - YY[ii, jj, kk])**2
            for kk in range(nactive_indices):
                ss[ii, jj] += a[active_indices[kk]]*(
                    XX[ii, 0, active_indices[kk]] -
                    YY[ii, jj, active_indices[kk]])**2
            ss[ii, jj] = ss[ii, jj]+b
    return ss


def gaussian_loglike_fun_economial_3D(
        obs, pred_obs, noise_stdev, active_indices=None):
    if pred_obs.ndim != 3:
        raise ValueError("pred_obs must be 3D")
    # cdist has a lot of overhead and cannot be used with active_indices
    # sq_dists = sq_dists_cdist_3d(obs, pred_obs)

    if type(noise_stdev) != np.ndarray:
        noise_stdev = np.ones((pred_obs.shape[-1], 1), dtype=float)*noise_stdev

    tmp1 = -1/(2*noise_stdev[:, 0]**2)
    if active_indices is None:
        tmp2 = 0.5*np.sum(np.log(-tmp1/np.pi))
    else:
        tmp2 = 0.5*np.sum(np.log(-tmp1[active_indices]/np.pi))
    llike = sq_dists_numba_3d(obs, pred_obs, tmp1, tmp2, active_indices)
    if llike.ndim == 1:
        llike = llike[:, None]
    return llike


def compute_weighted_sqeuclidian_distance(obs, pred_obs, noise_stdev,
                                          active_indices):
    if obs.ndim != 2 or pred_obs.ndim != 2:
        raise ValueError("obs and pred_obs must be 2D arrays")
    if type(noise_stdev) != np.ndarray or noise_stdev.ndim != 2:
        msg = "noise_stdev must be a 2d np.ndarray with one column"
        raise ValueError(msg)

    if active_indices is None:
        weights = 1/(np.sqrt(2)*noise_stdev[:, 0])
        # avoid copy is possible
        # using special indexing with array makes a copy which is slow
        weighted_obs = obs*weights
        weighted_pred_obs = pred_obs*weights
        sq_dists = cdist(weighted_obs, weighted_pred_obs, "sqeuclidean")
        return sq_dists

    weights = 1/(np.sqrt(2)*noise_stdev[active_indices, 0])
    weighted_obs = obs[:, active_indices]*weights
    weighted_pred_obs = pred_obs[:, active_indices]*weights
    sq_dists = cdist(weighted_obs, weighted_pred_obs, "sqeuclidean")
    return sq_dists


def gaussian_loglike_fun_economial_2D(
        obs, pred_obs, noise_stdev, active_indices=None):
    if type(noise_stdev) != np.ndarray:
        noise_stdev = np.ones((obs.shape[-1], 1), dtype=float)*noise_stdev
    sq_dists = compute_weighted_sqeuclidian_distance(
        obs, pred_obs, noise_stdev, active_indices)
    if active_indices is None:
        llike = (0.5*np.sum(np.log(1/(2*np.pi*noise_stdev[:, 0]**2))) -
                 sq_dists[0, :])
    else:
        llike = (0.5*np.sum(
            np.log(1/(2*np.pi*noise_stdev[active_indices, 0]**2))) -
                 sq_dists[0, :])

    if llike.ndim == 1:
        llike = llike[:, None]
    return llike


def gaussian_loglike_fun(obs, pred_obs, noise_stdev, active_indices=None):
    assert pred_obs.shape[-1] == obs.shape[-1]
    if obs.ndim == 3 and obs.shape[0] != 1:
        return gaussian_loglike_fun_economial_3D(
            obs, pred_obs, noise_stdev, active_indices)
    elif obs.ndim == 2 and obs.shape != pred_obs.shape:
        return gaussian_loglike_fun_economial_2D(
            obs, pred_obs, noise_stdev, active_indices)
    else:
        return gaussian_loglike_fun_broadcast(
            obs, pred_obs, noise_stdev, active_indices)


def __compute_expected_kl_utility_monte_carlo(
        outer_loop_obs, log_likelihood_fun, outer_loop_pred_obs,
        inner_loop_pred_obs, inner_loop_weights, outer_loop_weights,
        active_indices, return_all):

    nouter_loop_samples = outer_loop_pred_obs.shape[0]
    ninner_loop_samples = int(
        inner_loop_pred_obs.shape[0]//nouter_loop_samples)

    outer_log_likelihood_vals = log_likelihood_fun(
        outer_loop_obs, outer_loop_pred_obs, active_indices)
    nobs = outer_loop_obs.shape[1]
    tmp = inner_loop_pred_obs.reshape(
        nouter_loop_samples, ninner_loop_samples, nobs)
    inner_log_likelihood_vals = log_likelihood_fun(
        outer_loop_obs[:, None, :], tmp, active_indices)

    # above is a faster version of loop below
    # outer_log_likelihood_vals = np.empty((nouter_loop_samples, 1))
    # inner_log_likelihood_vals = np.empty(
    #     (nouter_loop_samples, ninner_loop_samples))
    # idx1 = 0
    # for ii in range(nouter_loop_samples):
    #     outer_log_likelihood_vals[ii] = log_likelihood_fun(
    #         outer_loop_obs[ii:ii+1, :], outer_loop_pred_obs[ii:ii+1, :])
    #     idx2 = idx1 + ninner_loop_samples
    #     inner_log_likelihood_vals[ii, :] = log_likelihood_fun(
    #         outer_loop_obs[ii:ii+1, :], inner_loop_pred_obs[idx1:idx2, :])
    #     idx1 = idx2

    evidences = np.einsum(
        "ij,ij->i", np.exp(inner_log_likelihood_vals),
        inner_loop_weights)[:, None]
    utility_val = np.sum((outer_log_likelihood_vals - np.log(evidences)) *
                         outer_loop_weights)
    if not return_all:
        return utility_val
    weights = np.exp(inner_log_likelihood_vals)*inner_loop_weights/evidences
    return utility_val, evidences, weights


def precompute_compute_expected_kl_utility_data(
        generate_outer_prior_samples, nouter_loop_samples, obs_fun,
        noise_fun, ninner_loop_samples, generate_inner_prior_samples=None,
        econ=False):
    r"""
    Parameters
    ----------
    generate_outer_prior_samples : callable
        Function with the signature

        `generate_outer_prior_samples(nsamples) -> np.ndarray(nvars, nsamples)`

        That returns a set of random samples randomly drawn from the prior
        These samples are used to draw condition samples of the obsevations
        which are used to take an expectation with respect to the data space
        in the outer loop

    nouter_loop_samples : integer
        The number of Monte Carlo samples used to compute the outer integral
        over all possible observations

    obs_fun : callable
        Function with the signature

        `obs_fun(samples) -> np.ndarray(nsamples, nqoi)`

        That returns evaluations of the forward model. Observations are assumed
        to be :math:`f(z)+\epsilon` where :math:`\epsilon` is additive noise
        nsamples : np.ndarray (nvars, nsamples)

    noise_fun : callable
        Function with the signature

        `noise_fun(samples) -> np.ndarray(nsamples)`

        That returns the noise of observations

    ninner_loop_samples : integer
        The number of quadrature samples used for the inner integral that
        computes the evidence for each realiaztion of the predicted
        observations

    generate_inner_prior_samples : callable
       Function with the signature

        `generate_inner_prior_samples(nsamples) -> np.ndarray(nvars, nsamples), np.ndarray(nsamples, 1)`

        Generate samples and associated weights used to evaluate
        the evidence computed by the inner loop
        If None then the function generate_outer_prior_samples is used and
        weights are assumed to be 1/nsamples. This function is useful if
        wanting to use multivariate quadrature to evaluate the evidence

    econ : boolean
        Make all inner loop samples the same for all outer loop samples.
        This reduces number of evaluations of prediction model. Currently
        this common data is copied and repeated for each outer loop sample
        so the rest of the code can remain the same. Eventually the data has
        to be tiled anyway when computing exepcted utility so this is not a big
        deal.

    Returns
    -------
    outer_loop_obs : np.ndarray (nsamples, ncandidates)
        Noisy observations predicted by the numerical model. That is
        random samples from the density P(Y|\theta,X). These are generated by
        randomly drawing a sample from the prior and then computing
        obs_fun(sample)+noise

    outer_loop_pred_obs : np.ndarray (nouter_loop_samples, ncandidates)
        The noiseless values of outer_loop_obs with noise removed

    inner_loop_pred_obs : np.ndarray (nouter_loop_samples*ninner_loop_samples, ncandidates)
        The noiseless values of obs_fun at all sets of innerloop samples
        used for each outerloop iteration. The values are stacked such
        that np.vstack((inner_loop1_vals, inner_loop2_vals, ...))

    inner_loop_weights  : np.ndarray (nouter_loop_samples, ninner_loop_samples)
        The quadrature weights associated with each inner_loop_pred_obs set
        used to compute the inner integral.
    """

    # generate samples and values for the outer loop
    outer_loop_prior_samples = generate_outer_prior_samples(
        nouter_loop_samples)[0]
    outer_loop_pred_obs = obs_fun(outer_loop_prior_samples)
    outer_loop_obs = outer_loop_pred_obs + noise_fun(outer_loop_pred_obs)

    # generate samples and values for all inner loops
    if generate_inner_prior_samples is None:
        generate_inner_prior_samples = generate_outer_prior_samples

    inner_loop_prior_samples = np.empty(
        (outer_loop_prior_samples.shape[0],
         ninner_loop_samples*nouter_loop_samples))
    inner_loop_weights = np.empty(
        (nouter_loop_samples, ninner_loop_samples))
    idx1 = 0
    for ii in range(nouter_loop_samples):
        idx2 = idx1 + ninner_loop_samples
        if not econ or ii == 0:
            # when econ is True use samples from ii == 0 for all ii
            in_samples, in_weights = generate_inner_prior_samples(
                ninner_loop_samples)
            if in_samples.ndim != 2:
                msg = "Generate_inner_prior_samples must return 2d np.ndarray"
                raise ValueError(msg)
        inner_loop_prior_samples[:, idx1:idx2] = in_samples
        inner_loop_weights[ii, :] = in_weights
        idx1 = idx2

    if not econ:
        inner_loop_pred_obs = obs_fun(inner_loop_prior_samples)
    else:
        shared_inner_loop_pred_obs = obs_fun(in_samples)
        inner_loop_pred_obs = np.tile(
            shared_inner_loop_pred_obs, (nouter_loop_samples, 1))

    return (outer_loop_obs, outer_loop_pred_obs, inner_loop_pred_obs,
            inner_loop_weights, outer_loop_prior_samples,
            inner_loop_prior_samples)


def precompute_compute_expected_deviation_data(
        generate_outer_prior_samples, nouter_loop_samples, obs_fun,
        noise_fun, qoi_fun, ninner_loop_samples,
        generate_inner_prior_samples=None, econ=False):
    (outer_loop_obs, outer_loop_pred_obs, inner_loop_pred_obs,
     inner_loop_weights, outer_loop_prior_samples,
     inner_loop_prior_samples) = precompute_compute_expected_kl_utility_data(
             generate_outer_prior_samples, nouter_loop_samples, obs_fun,
             noise_fun, ninner_loop_samples, generate_inner_prior_samples,
             econ)

    if not econ:
        inner_loop_pred_qois = qoi_fun(inner_loop_prior_samples)
    else:
        in_samples = inner_loop_prior_samples[:, :ninner_loop_samples]
        shared_inner_loop_pred_qois = qoi_fun(in_samples)
        inner_loop_pred_qois = np.tile(
            shared_inner_loop_pred_qois, (nouter_loop_samples, 1))
    return (outer_loop_obs, outer_loop_pred_obs, inner_loop_pred_obs,
            inner_loop_weights, outer_loop_prior_samples,
            inner_loop_prior_samples, inner_loop_pred_qois)


def compute_expected_kl_utility_monte_carlo(
        log_likelihood_fun, outer_loop_obs, outer_loop_pred_obs,
        inner_loop_pred_obs, inner_loop_weights, outer_loop_weights,
        collected_design_indices, new_design_indices,
        return_all):
    r"""
    Compute the expected Kullback–Leibler (KL) divergence.

    Parameters
    ----------
    log_likelihood_fun : callable
        Function with signature

        `log_likelihood_fun(obs, pred_obs) -> np.ndarray (nsamples, 1)`

        That returns the log likelihood for a set of observations and
        predictions.
        obs : np.ndarray(nsamples, nobs+nnew_obs)
        pred_obs : np.ndarray(nsamples, nobs+nnew_obs)

    outer_loop_obs : np.ndarray (nsamples, ncandidates)
        Noisy observations predicted by the numerical model. That is
        random samples from the density P(Y|\theta,X). These are generated by
        randomly drawing a sample from the prior and then computing
        obs_fun(sample)+noise

    outer_loop_pred_obs : np.ndarray (nouter_loop_samples, ncandidates)
        The noiseless values of outer_loop_obs with noise removed

    inner_loop_pred_obs : np.ndarray (nouter_loop_samples*ninner_loop_samples, ncandidates)
        The noiseless values of obs_fun at all sets of innerloop samples
        used for each outerloop iteration. The values are stacked such
        that np.vstack((inner_loop1_vals, inner_loop2_vals, ...))

    inner_loop_weights  : np.ndarray (nouter_loop_samples, ninner_loop_samples)
        The quadrature weights associated with each inner_loop_pred_obs set
        used to compute the inner integral.

    outer_loop_weights  : np.ndarray (nouter_loop_samples, 1)
        The quadrature weights associated with each outer_loop_pred_obs set
        used to compute the outer integral.

    collected_design_indices : np.ndarray (nobs)
        The indices into the qoi vector associated with the
        collected observations

    new_design_indices : np.ndarray (nnew_obs)
        The indices into the qoi vector associated with new design locations
        under consideration

    Returns
    -------
    utility : float
        The expected utility
    """
    if collected_design_indices is not None:
        active_indices = np.hstack(
            (collected_design_indices, new_design_indices))
    else:
        # assume the observations at the collected_design_indices are already
        # incorporated into the inner and outer loop weights
        active_indices = np.asarray(new_design_indices)
    return __compute_expected_kl_utility_monte_carlo(
        outer_loop_obs, log_likelihood_fun, outer_loop_pred_obs,
        inner_loop_pred_obs, inner_loop_weights, outer_loop_weights,
        active_indices, return_all)


def __compute_negative_expected_deviation_monte_carlo(
        outer_loop_obs, log_likelihood_fun, outer_loop_pred_obs,
        inner_loop_pred_obs, inner_loop_weights, outer_loop_weights,
        inner_loop_pred_qois, deviation_fun, active_indices, return_all):

    nouter_loop_samples = outer_loop_pred_obs.shape[0]
    ninner_loop_samples = int(
        inner_loop_pred_obs.shape[0]//nouter_loop_samples)

    # TODO remove outer_loop obs and outerloop pred obs from function
    # arguments
    # outer_log_likelihood_vals = log_likelihood_fun(
    #     outer_loop_obs, outer_loop_pred_obs, active_indices)
    nobs = outer_loop_obs.shape[1]
    tmp = inner_loop_pred_obs.reshape(
        nouter_loop_samples, ninner_loop_samples, nobs)
    inner_log_likelihood_vals = log_likelihood_fun(
        outer_loop_obs[:, None, :], tmp, active_indices)

    inner_likelihood_vals = np.exp(inner_log_likelihood_vals)
    evidences = np.einsum(
        "ij,ij->i", inner_likelihood_vals, inner_loop_weights)[:, None]

    weights = inner_likelihood_vals*inner_loop_weights/evidences

    # make deviation_fun operate on columns of samples
    # so that it returns a vector of deviations one for each column
    deviations = deviation_fun(
        inner_loop_pred_qois.reshape(nouter_loop_samples, ninner_loop_samples),
        weights)
    expected_deviation = np.sum(deviations*outer_loop_weights)
    if not return_all:
        return -expected_deviation
    else:
        return -expected_deviation, evidences, weights, deviations


def compute_negative_expected_deviation_monte_carlo(
        log_likelihood_fun, outer_loop_obs, outer_loop_pred_obs,
        inner_loop_pred_obs, inner_loop_weights, outer_loop_weights,
        inner_loop_pred_qois, deviation_fun, collected_design_indices,
        new_design_indices, return_all):
    if collected_design_indices is not None:
        active_indices = np.hstack(
            (collected_design_indices, new_design_indices))
    else:
        # assume the observations at the collected_design_indices are already
        # incorporated into the inner and outer loop weights
        active_indices = np.asarray(new_design_indices)
    return __compute_negative_expected_deviation_monte_carlo(
        outer_loop_obs, log_likelihood_fun, outer_loop_pred_obs,
        inner_loop_pred_obs, inner_loop_weights, outer_loop_weights,
        inner_loop_pred_qois, deviation_fun, active_indices, return_all)


def select_design(design_candidates, collected_design_indices,
                  compute_expected_utility):
    """
    Update an experimental design.

    Parameters
    ----------
    design_candidates : np.ndarray (nvars, nsamples)
        The location of all design sample candidates

    collected_design_indices : np.ndarray (nobs)
        The indices into the qoi vector associated with the
        collected observations

    Returns
    -------
    utility_vals : np.ndarray (ncandidates)
        The utility vals at the candidate design samples. If the candidate
        sample is already in collected design then the utility value will
        be set to -np.inf

    selected_index : integer
        The index of the best design, i.e. the largest utility
    """
    ncandidates = design_candidates.shape[1]
    utility_vals = -np.ones(ncandidates)*np.inf
    for ii in range(ncandidates):
        if ii not in collected_design_indices:
            # print(f'Candidate {ii}')
            utility_vals[ii] = compute_expected_utility(
                collected_design_indices, np.array([ii]))
    selected_index = np.argmax(utility_vals)

    return utility_vals, selected_index


def update_observations(design_candidates, collected_design_indices,
                        new_design_indices, obs_process,
                        collected_obs):
    """
    Updated the real collected data with obsevations at the new selected
    candidate locations.

    Parameters
    ---------
    design_candidates : np.ndarray (nvars, nsamples)
        The location of all design sample candidates

    collected_design_indices : np.ndarray (nobs)
        The indices into the qoi vector associated with the
        collected observations

    new_design_indices : np.ndarray (nnew_design_samples)
        The indices into the design_candidates array of the new selected
        design samples

    obs_process : callable
        The true data generation model with the signature

        `obs_process(design_samples) -> np.ndarray (1, ndesign_samples)`

        where design_samples is np.ndarary (nvars, ndesign_samples)

    collected_obs : np.ndarray (1, nobs)
        The observations at the previously selected design samples

    Returns
    -------
    updated_collected_obs : np.ndarray (1, nobs+nnew_design_samples)
        The updated collected observations with the new observations
        appended to the previous observations

    updated_collected_design_indices : np.ndarray (1, nobs+nnew_design_samples)
        The updated indices associated with all the collected observations
    """
    new_obs = obs_process(new_design_indices)

    if collected_obs is None:
        return new_obs, new_design_indices

    updated_collected_obs = np.hstack((collected_obs, new_obs))
    updated_collected_design_indices = np.hstack(
        (collected_design_indices, new_design_indices))
    return updated_collected_obs, updated_collected_design_indices


def d_optimal_utility(Amat, noise_std):
    """
    Compute the d-optimality criterion for a linear model f(x) = Amat.dot(x)

    Assume R = sigma^2 I

    Posterior covaiance
    sigma^2 inv(A^TA+R)

    References
    ----------
    Alen Alexanderian and Arvind K. Saibaba
    Efficient D-Optimal Design of Experiments for
    Infinite-Dimensional Bayesian Linear Inverse Problems
    SIAM Journal on Scientific Computing 2018 40:5, A2956-A2985
    https://doi.org/10.1137/17M115712X
    Theorem 1
    """
    nvars = Amat.shape[1]
    hess_misfit = Amat.T.dot(Amat)/noise_std**2
    ident = np.eye(nvars)
    return 0.5*np.linalg.slogdet(hess_misfit+ident)[1]


def gaussian_kl_divergence(mean1, sigma1, mean2, sigma2):
    r"""
    Compute KL( N(mean1, sigma1) || N(mean2, sigma2) )
    """
    if mean1.ndim != 2 or mean2.ndim != 2:
        raise ValueError("means must have shape (nvars, 1)")
    nvars = mean1.shape[0]
    sigma2_inv = np.linalg.inv(sigma2)
    val = np.log(np.linalg.det(sigma2)/np.linalg.det(sigma1))-float(nvars)
    val += np.trace(sigma2_inv.dot(sigma1))
    val += (mean2-mean1).T.dot(sigma2_inv.dot(mean2-mean1))
    return 0.5*val.item()


class BayesianBatchKLOED(object):
    def __init__(self, design_candidates, obs_fun, noise_std,
                 prior_variable, nouter_loop_samples=1000,
                 ninner_loop_samples=1000, generate_inner_prior_samples=None,
                 econ=False):
        self.design_candidates = design_candidates
        self.obs_fun = obs_fun
        self.noise_std = noise_std
        self.prior_variable = prior_variable
        self.nouter_loop_samples = nouter_loop_samples
        self.ninner_loop_samples = ninner_loop_samples
        self.generate_inner_prior_samples = generate_inner_prior_samples
        self.econ = econ

        self.collected_design_indices = None
        self.outer_loop_obs = None
        self.outer_loop_pred_obs = None
        self.inner_loop_pred_obs = None
        self.inner_loop_weights = None
        self.outer_loop_prior_samples = None

    def noise_fun(self, values):
        return np.random.normal(0, self.noise_std, (values.shape))

    def generate_prior_samples(self, nsamples):
        return generate_independent_random_samples(
            self.prior_variable, nsamples), np.ones(nsamples)/nsamples

    def loglike_fun(self, obs, pred_obs, active_indices=None):
        return gaussian_loglike_fun(
            obs, pred_obs, self.noise_std, active_indices)

    def populate(self):
        (self.outer_loop_obs, self.outer_loop_pred_obs,
         self.inner_loop_pred_obs, self.inner_loop_weights,
         self.outer_loop_prior_samples, self.inner_loop_prior_samples) = \
             precompute_compute_expected_kl_utility_data(
                 self.generate_prior_samples, self.nouter_loop_samples,
                 self.obs_fun, self.noise_fun, self.ninner_loop_samples,
                 generate_inner_prior_samples=self.generate_inner_prior_samples,
                 econ=self.econ)
        self.outer_loop_weights = np.ones(
            (self.inner_loop_weights.shape[0], 1)) / \
            self.inner_loop_weights.shape[0]

    def compute_expected_utility(self, collected_design_indices,
                                 new_design_indices, return_all=False):
        """
        return_all true used for debugging returns more than just utilities
        and also returns itermediate data useful for testing
        """
        # unlike open loop design (closed loop batch design)
        # we do not update inner and outer loop weights but rather
        # just compute likelihood for all collected and new design indices
        # If want to update weights then we must have a different set of
        # weights for each inner iteration of the inner loop that is
        # computed using
        # the associated outerloop data
        return compute_expected_kl_utility_monte_carlo(
            self.loglike_fun, self.outer_loop_obs, self.outer_loop_pred_obs,
            self.inner_loop_pred_obs, self.inner_loop_weights,
            self.outer_loop_weights, collected_design_indices,
            new_design_indices, return_all)

    def set_collected_design_indices(self, indices):
        self.collected_design_indices = indices.copy()

    def update_design(self):
        if not hasattr(self, "outer_loop_obs"):
            raise ValueError("Must call self.populate before creating designs")
        if self.collected_design_indices is None:
            self.collected_design_indices = np.zeros((0), dtype=int)
        utility_vals, selected_index = select_design(
            self.design_candidates, self.collected_design_indices,
            self.compute_expected_utility)

        new_design_indices = np.array([selected_index])
        self.collected_design_indices = np.hstack(
            (self.collected_design_indices, new_design_indices))
        return utility_vals, new_design_indices


def oed_variance_deviation(samples, weights):
    means = np.einsum(
        "ij,ij->i", samples, weights)[:, None]
    variances = np.einsum(
        "ij,ij->i", samples**2, weights)[:, None]-means**2
    return variances


def oed_conditional_value_at_risk_deviation(beta, samples, weights,
                                            samples_sorted=True):
    cvars = np.empty(samples.shape[0])
    for ii in range(samples.shape[0]):
        mean = np.sum(samples[ii, :]*weights[ii, :])
        cvars[ii] = conditional_value_at_risk(
            samples[ii, :], beta, weights[ii, :], samples_sorted)-mean
    return cvars[:, None]


class BayesianBatchDeviationOED(BayesianBatchKLOED):
    """
    A class to compute batch optimal experimental designs for which data
    is not used to inform the choice of subsequent design locations.
    The utility function measures the reduction in the standard deviation of
    predicitions.
    """
    def __init__(self, design_candidates, obs_fun, noise_std,
                 prior_variable, qoi_fun, nouter_loop_samples=1000,
                 ninner_loop_samples=1000, generate_inner_prior_samples=None,
                 econ=False, deviation_fun=oed_variance_deviation):
        r"""
        qoi_fun : callable
        Function with the signature

        `qoi_fun(samples) -> np.ndarray(nsamples, nqoi)`

        That returns evaluations of the forward model. Observations are assumed
        to be :math:`f(z)+\epsilon` where :math:`\epsilon` is additive noise
        nsamples : np.ndarray (nvars, nsamples)
        """
        super().__init__(design_candidates, obs_fun, noise_std,
                         prior_variable, nouter_loop_samples,
                         ninner_loop_samples, generate_inner_prior_samples,
                         econ=econ)
        self.qoi_fun = qoi_fun
        self.deviation_fun = deviation_fun

    def populate(self):
        """
        Compute the data needed to initialize the OED algorithm.
        """
        (self.outer_loop_obs, self.outer_loop_pred_obs,
         self.inner_loop_pred_obs, self.inner_loop_weights,
         self.outer_loop_prior_samples, self.inner_loop_prior_samples,
         self.inner_loop_pred_qois
         ) = precompute_compute_expected_deviation_data(
             self.generate_prior_samples, self.nouter_loop_samples,
             self.obs_fun, self.noise_fun, self.qoi_fun,
             self.ninner_loop_samples,
             generate_inner_prior_samples=self.generate_inner_prior_samples,
             econ=self.econ)

        # Sort inner_loop_pred_qois and use this order to sort
        # inner_loop_prior_samples so that cvar deviation does not have to
        # constantly sort samples
        idx1 = 0
        for ii in range(self.nouter_loop_samples):
            idx2 = idx1 + self.ninner_loop_samples
            qoi_indices = np.argsort(self.inner_loop_pred_qois[idx1:idx2, 0])
            self.inner_loop_pred_qois[idx1:idx2] = \
                self.inner_loop_pred_qois[idx1:idx2][qoi_indices]
            self.inner_loop_prior_samples[:, idx1:idx2] = \
                self.inner_loop_prior_samples[:, idx1:idx2][:, qoi_indices]
            self.inner_loop_pred_obs[idx1:idx2] = \
                self.inner_loop_pred_obs[idx1:idx2][qoi_indices, :]
            self.inner_loop_weights[ii] = \
                self.inner_loop_weights[ii, qoi_indices]
            idx1 = idx2

        self.outer_loop_weights = np.ones(
            (self.inner_loop_weights.shape[0], 1)) / \
            self.inner_loop_weights.shape[0]

    def compute_expected_utility(self, collected_design_indices,
                                 new_design_indices, return_all=False):
        """
        Compute the negative expected deviation in predictions of QoI

        Parameters
        ----------
        collected_design_indices : np.ndarray (nobs)
            The indices into the qoi vector associated with the
            collected observations

        new_design_indices : np.ndarray (nnew_obs)
            The indices into the qoi vector associated with new design
            locations under consideration

        return_all : boolean
             False - return the utilities
             True - used for debugging returns utilities
             and itermediate data useful for testing

        Returns
        -------
        utility : float
            The negative expected deviation
        """
        # unlike open loop design (closed loop batch design)
        # we do not update inner and outer loop weights but rather
        # just compute likelihood for all collected and new design indices
        # If want to update weights then we must have a different set of
        # weights for each inner iteration of the inner loop that is
        # computed using
        # the associated outerloop data
        return compute_negative_expected_deviation_monte_carlo(
            self.loglike_fun, self.outer_loop_obs, self.outer_loop_pred_obs,
            self.inner_loop_pred_obs, self.inner_loop_weights,
            self.outer_loop_weights, self.inner_loop_pred_qois,
            self.deviation_fun, collected_design_indices, new_design_indices,
            return_all)


class BayesianSequentialKLOED(BayesianBatchKLOED):
    """
    A class to compute sequential optimal experimental designs that collect
    data and use this to inform the choice of subsequent design locations.
    """
    def __init__(self, design_candidates, obs_fun, noise_std,
                 prior_variable, obs_process, nouter_loop_samples=1000,
                 ninner_loop_samples=1000, generate_inner_prior_samples=None,
                 econ=False):
        super().__init__(design_candidates, obs_fun, noise_std,
                         prior_variable, nouter_loop_samples,
                         ninner_loop_samples, generate_inner_prior_samples,
                         econ=econ)
        self.obs_process = obs_process
        self.collected_obs = None
        self.inner_importance_weights = None
        self.outer_importance_weights = None
        self.inner_loop_weights_up = None
        self.outer_loop_weights_up = None
        self.evidence_from_prior = 1
        self.evidence = None

    def __compute_evidence(self):
        """
        Compute the evidence associated with using the true collected data.

        Notes
        -----
        This is a private function because calling by user will upset
        evidence calculation

        Always just use the first inner loop sample set to compute evidence.
        To avoid numerical prevision problems recompute evidence with
        all data as opposed to updating evidence just using new data
        """
        log_like_vals = self.loglike_fun(
            self.collected_obs,
            self.inner_loop_pred_obs[:self.ninner_loop_samples,
                                     self.collected_design_indices])
        # compute evidence moving from initial prior to current posterior
        evidence_from_prior = np.sum(
            np.exp(log_like_vals)[:, 0]*self.inner_loop_weights[0, :])
        # compute evidence moving from previous posterior to current posterior
        self.evidence = evidence_from_prior/self.evidence_from_prior
        self.evidence_from_prior = evidence_from_prior

    def compute_importance_weights(self):
        """
        Compute the importance weights used in the computation of the expected
        utility that acccount for the fact we want to use the current posterior
        as the prior in the utility formula.
        """
        self.outer_importance_weights = np.exp(self.loglike_fun(
            self.collected_obs, self.outer_loop_pred_obs[
                :, self.collected_design_indices]))/self.evidence_from_prior
        nobs = self.collected_design_indices.shape[0]
        tmp = self.inner_loop_pred_obs[
            :, self.collected_design_indices].reshape(
                self.nouter_loop_samples, self.ninner_loop_samples, nobs)

        self.inner_importance_weights = np.exp(self.loglike_fun(
            self.collected_obs[:, None, :], tmp))/self.evidence_from_prior

        # # above is a faster version of loop below

        # outer_importance_weights = np.empty((self.nouter_loop_samples, 1))
        # inner_importance_weights = np.empty(
        #     (self.nouter_loop_samples, self.ninner_loop_samples))

        # idx1 = 0
        # for ii in range(self.nouter_loop_samples):
        #     outer_importance_weights[ii] = np.exp(self.loglike_fun(
        #         self.collected_obs,
        #         self.outer_loop_pred_obs[
        #             ii:ii+1, self.collected_design_indices]))/ \
        #             self.evidence_from_prior

        #     idx2 = idx1 + self.ninner_loop_samples
        #     inner_importance_weights[ii, :] = np.exp(self.loglike_fun(
        #         self.collected_obs,
        #         self.inner_loop_pred_obs[
        #             idx1:idx2, self.collected_design_indices]))[:, 0] / \
        #         self.evidence_from_prior
        #     idx1 = idx2
        # np.allclose(self.outer_importance_weights, outer_importance_weights)
        # np.allclose(self.inner_importance_weights, inner_importance_weights)

    def update_observations(self, new_obs):
        """
        Store the newly collected obsevations which will dictate
        the next design point.

        Parameters
        ----------
        new_obs : np.ndarray (1, nnew_obs)
            The new observations

        Notes
        -----
        self.inner_importance_weights contains likelihood vals/evidence
        at inner_loop_samples
        self.inner_loop_weights is the prior quadrature weights which
        for random samples drawn from
        prior is just 1/N and for Gauss Quadrature is the quadrature rule
        weights.

        Similarly for self.outer_importance_weights
        """
        if self.collected_obs is None:
            self.collected_obs = new_obs
        else:
            self.collected_obs = np.hstack(
                (self.collected_obs, new_obs))
        self.__compute_evidence()
        self.compute_importance_weights()
        self.outer_loop_weights_up = \
            self.outer_loop_weights*self.outer_importance_weights
        self.inner_loop_weights_up = \
            self.inner_loop_weights*self.inner_importance_weights

    def set_collected_design_indices(self, indices):
        """
        Set the initial design indices and collect data at the
        corresponding design points.

        Parameters
        ----------
        indices : np.ndarray (nindices, 1)
            The indices corresponding to an initial design
        """
        self.collected_design_indices = indices.copy()
        new_obs = self.obs_process(self.collected_design_indices)
        self.update_observations(new_obs)

    def compute_expected_utility(self, collected_design_indices,
                                 new_design_indices, return_all=False):
        """
        Compute the expected utility. Using the current posterior as the new
        prior.

        Parameters
        ----------
        collected_design_indices : np.ndarray (nobs)
            The indices into the qoi vector associated with the
            collected observations

        new_design_indices : np.ndarray (nnew_obs)
            The indices into the qoi vector associated with new design 
            locations under consideration

        Notes
        -----
        Passing None for collected_design_indices will ensure
        only obs at new_design indices is used to evaluate likelihood
        the data at collected indices is incoroporated into the
        inner and outer loop weights
        """
        # TODO pass in these weights so do not have to do so much
        # multiplications
        return compute_expected_kl_utility_monte_carlo(
            self.loglike_fun, self.outer_loop_obs, self.outer_loop_pred_obs,
            self.inner_loop_pred_obs, self.inner_loop_weights_up,
            self.outer_loop_weights_up,
            None,  # collected_design_indices,
            new_design_indices, return_all)

    def update_design(self):
        return super().update_design()


def get_oed_inner_quadrature_rule(ninner_loop_samples, prior_variable,
                                  quad_method='gauss'):
    nrandom_vars = prior_variable.num_vars()
    ninner_loop_samples_1d = ninner_loop_samples
    if quad_method == "gauss":
        var_trans = AffineRandomVariableTransformation(prior_variable)
        univariate_quad_rules = \
            get_univariate_quadrature_rules_from_variable(
                prior_variable, [ninner_loop_samples_1d]*nrandom_vars)[0]
        x_quad, w_quad = get_tensor_product_quadrature_rule(
            [ninner_loop_samples_1d]*nrandom_vars, nrandom_vars,
            univariate_quad_rules,
            transform_samples=var_trans.map_from_canonical_space)
        return x_quad, w_quad

    degree = {'linear': 1, 'quadratic': 2}[quad_method]
    if prior_variable.is_bounded_continuous_variable():
        alpha = 1
    else:
        alpha = 1-1e-6
    new_ranges = prior_variable.get_statistics(
        "interval", alpha=alpha).flatten()
    x_quad, w_quad = \
        get_tensor_product_piecewise_polynomial_quadrature_rule(
            ninner_loop_samples_1d, new_ranges, degree)
    w_quad *= prior_variable.pdf(x_quad)[:, 0]
    return x_quad, w_quad


def get_posterior_vals_at_inner_loop_samples(
        oed, prior_variable, nn, outer_loop_idx):
    # plot posterior for one realization of the data
    # nn : number of data used to form posterior
    # outer_loop_idx : the outer loop iteration used to generate the data
    assert prior_variable.num_vars() == 2
    evidences, weights = oed.compute_expected_utility(
        oed.collected_design_indices[:nn], np.zeros(0, dtype=int), True)[1:3]
    ninner_loop_samples = weights.shape[1]
    vals = (
        weights[outer_loop_idx, :] /
        oed.inner_loop_weights[outer_loop_idx, :])[:, None]
    # multiply vals by prior.
    vals *= prior_variable.pdf(
        oed.inner_loop_prior_samples[:, :ninner_loop_samples])
    return vals


def get_posterior_2d_interpolant_from_oed_data(
        oed, prior_variable, nn, outer_loop_idx, quad_method):
    # plot posterior for one realization of the data
    # nn : number of data used to form posterior
    # outer_loop_idx : the outer loop iteration used to generate the data
    vals = get_posterior_vals_at_inner_loop_samples(
        oed, prior_variable, nn, outer_loop_idx)
    print(vals.min(), vals.max())
    ninner_loop_samples = vals.shape[0]

    if quad_method == "gauss":
        # interpolate posterior vals onto equidistant mesh for plotting
        nvars = prior_variable.num_vars()
        abscissa_1d = []
        for dd in range(nvars):
            abscissa_1d.append(
                np.unique(
                    oed.inner_loop_prior_samples[dd, :ninner_loop_samples]))
        fun = partial(tensor_product_barycentric_interpolation, abscissa_1d,
                      vals[outer_loop_idx, :][:, None])
        return fun

    quad_methods = ['linear', 'quadratic', 'gauss']
    if quad_method != "linear" and quad_method != "quadratic":
        raise ValueError(f"quad_method must be in {quad_methods}")

    # if using piecewise polynomial quadrature interpolate between using
    # piecewise linear method
    from scipy.interpolate import griddata
    x_quad = oed.inner_loop_prior_samples[:, :ninner_loop_samples]
    def fun(x): return griddata(x_quad.T, vals, x.T, method="linear")
    return fun


def plot_2d_posterior_from_oed_data(
        oed, prior_variable, nn, outer_loop_idx, method):

    if prior_variable.is_bounded_continuous_variable():
        alpha = 1
    else:
        alpha = 0.99
    plot_limits = prior_variable.get_statistics(
        "interval", alpha=alpha).flatten()

    fun = get_posterior_2d_interpolant_from_oed_data(
        oed, prior_variable, nn, outer_loop_idx, method)
    from pyapprox import plt, get_meshgrid_function_data
    X, Y, Z = get_meshgrid_function_data(fun, plot_limits, 30)
    p = plt.contourf(
        X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 21))
    plt.colorbar(p)
    plt.show()


def tensor_product_barycentric_interpolation(abscissa_1d, values, samples):
    nvars = len(abscissa_1d)
    barycentric_weights_1d = []
    for dd in range(nvars):
        interval_length = abscissa_1d[dd].max()-abscissa_1d[dd].min()
        barycentric_weights_1d.append(
            compute_barycentric_weights_1d(
                abscissa_1d[dd], interval_length=interval_length))
    poly_vals = multivariate_barycentric_lagrange_interpolation(
        samples, abscissa_1d, barycentric_weights_1d, values,
        np.arange(nvars))
    return poly_vals
