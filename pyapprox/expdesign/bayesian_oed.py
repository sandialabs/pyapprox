import os
import math
import numpy as np
from scipy.spatial.distance import cdist
from pyapprox.util.pya_numba import njit
from functools import partial
from multiprocessing import Pool, RawArray
from abc import ABC, abstractmethod
from scipy import stats
import time
import itertools

from pyapprox.util.sys_utilities import trace_error_with_msg
from pyapprox.util.utilities import (
    get_tensor_product_quadrature_rule, split_indices)
from pyapprox.variables.risk import (
    conditional_value_at_risk, conditional_value_at_risk_vectorized,
    entropic_risk_measure)
from pyapprox.variables.transforms import AffineTransform
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.interp.tensorprod import (
    get_tensor_product_piecewise_polynomial_quadrature_rule)
from pyapprox.surrogates.polychaos.gpc import (
    get_univariate_quadrature_rules_from_variable)
from pyapprox.surrogates.interp.barycentric_interpolation import (
    compute_barycentric_weights_1d,
    multivariate_barycentric_lagrange_interpolation)
from pyapprox.surrogates.integrate import integrate


def gaussian_loglike_fun_broadcast(
        obs, pred_obs, noise_std, active_indices=None):
    """
    Conmpute the log-likelihood values from a set of real and predicted
    observations

    Parameters
    ----------
    obs : np.ndarray (nsamples, nobs)
        The real observations repeated for each sample

    pred_obs : np.ndarray (nsamples, nobs)
        The observations predicited by the model for a set of samples

    noise_std : float or np.ndarray (nobs, 1)
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
    if (type(noise_std) == np.ndarray and
            noise_std.shape[0] != obs.shape[-1]):
        raise ValueError("noise_std must be provided for each observation")

    if type(noise_std) != np.ndarray:
        noise_std = np.ones((obs.shape[-1], 1), dtype=float)*noise_std

    if active_indices is None:
        # avoid copy if possible
        # using special indexing with array e.g array[:, I] where I is an array
        # makes a copy which is slow
        tmp = 1/(2*noise_std[:, 0]**2)
        llike = 0.5*np.sum(np.log(tmp/np.pi))
        llike += np.sum(-(obs-pred_obs)**2*tmp, axis=-1)
    else:
        tmp = 1/(2*noise_std[active_indices, 0]**2)
        llike = 0.5*np.sum(np.log(tmp/np.pi))
        llike += np.sum(
            -(obs[..., active_indices]-pred_obs[..., active_indices])**2*tmp,
            axis=-1)
    if llike.ndim == 1:
        llike = llike[:, None]
    return llike


@njit(cache=True)
def sq_dists_numba_3d(XX, YY, a, b, active_indices):
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

    b : float
        scalar added to l2 distance

    Returns
    -------
    ss : np.ndarray (LL, MM)
        The scaled distances
    """
    Yshape = YY.shape
    ss = np.empty(Yshape[:2])
    nactive_indices = active_indices.shape[0]
    for ii in range(Yshape[0]):
        for jj in range(Yshape[1]):
            ss[ii, jj] = 0.0
            for kk in range(nactive_indices):
                ss[ii, jj] += a[active_indices[kk]]*(
                    XX[ii, active_indices[kk]] -
                    YY[ii, jj, active_indices[kk]])**2
            ss[ii, jj] = ss[ii, jj]+b
    return ss


def sq_dists_3d(XX, YY, a=1, b=0, active_indices=None):
    assert XX.shape[0] == YY.shape[0]
    assert a.shape[0] == XX.shape[1]
    assert a.shape[0] == YY.shape[2]
    if active_indices is None:
        active_indices = np.arange(YY.shape[2])
    if np.isscalar(a):
        a = np.ones(active_indices.shape[0])

    try:
        from pyapprox.cython.utilities import sq_dists_3d_pyx
        return sq_dists_3d_pyx(XX, YY, active_indices, a, b)
    except(ImportError, ModuleNotFoundError) as e:
        msg = 'sq_dists_3d extension failed'
        trace_error_with_msg(msg, e)

    return sq_dists_numba_3d(XX, YY, a, b, active_indices)


def gaussian_loglike_fun_economial_3D(
        obs, pred_obs, noise_std, active_indices=None):
    if pred_obs.ndim != 3:
        raise ValueError("pred_obs must be 3D")
    # cdist has a lot of overhead and cannot be used with active_indices
    # sq_dists = sq_dists_cdist_3d(obs, pred_obs)

    if type(noise_std) != np.ndarray:
        noise_std = np.ones((pred_obs.shape[-1], 1), dtype=float)*noise_std

    tmp1 = -1/(2*noise_std[:, 0]**2)
    if active_indices is None:
        tmp2 = 0.5*np.sum(np.log(-tmp1/np.pi))
    else:
        tmp2 = 0.5*np.sum(np.log(-tmp1[active_indices]/np.pi))
    llike = sq_dists_3d(obs, pred_obs, tmp1, tmp2, active_indices)
    if llike.ndim == 1:
        llike = llike[:, None]
    return llike


@njit(cache=True)
def sq_dists_numba_3d_XX_prereduced(XX, YY, a, b, active_indices):
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

    b : float
        scalar added to l2 distance

    Returns
    -------
    ss : np.ndarray (LL, MM)
        The scaled distances
    """
    Yshape = YY.shape
    ss = np.empty(Yshape[:2])
    nactive_indices = active_indices.shape[0]
    for ii in range(Yshape[0]):
        for jj in range(Yshape[1]):
            ss[ii, jj] = 0.0
            for kk in range(nactive_indices):
                ss[ii, jj] += a[active_indices[kk]]*(
                    XX[ii, kk] - YY[ii, jj, active_indices[kk]])**2
            ss[ii, jj] = ss[ii, jj]+b
    return ss


def sq_dists_3d_prereduced(XX, YY, a=1, b=0, active_indices=None):
    assert XX.shape[0] == YY.shape[0]
    assert a.shape[0] == XX.shape[1]
    assert a.shape[0] == YY.shape[2]
    if active_indices is None:
        active_indices = np.arange(YY.shape[2])
    if np.isscalar(a):
        a = np.ones(active_indices.shape[0])

    try:
        from pyapprox.cython.utilities import sq_dists_3d_prereduced_pyx
        return sq_dists_3d_prereduced_pyx(XX, YY, active_indices, a, b)
    except(ImportError, ModuleNotFoundError) as e:
        msg = 'sq_dists_3d_prereduced extension failed'
        trace_error_with_msg(msg, e)

    return sq_dists_numba_3d_XX_prereduced(XX, YY, a, b, active_indices)


def gaussian_loglike_fun_3d_prereduced(
        obs, pred_obs, noise_std, active_indices):
    if type(noise_std) != np.ndarray:
        noise_std = np.ones((pred_obs.shape[-1], 1), dtype=float)*noise_std

    tmp1 = -1/(2*noise_std[:, 0]**2)
    tmp2 = 0.5*np.sum(np.log(-tmp1[active_indices]/np.pi))
    llike = sq_dists_numba_3d_XX_prereduced(
        obs, pred_obs, tmp1, tmp2, active_indices)
    if llike.ndim == 1:
        llike = llike[:, None]
    return llike


def compute_weighted_sqeuclidian_distance(obs, pred_obs, noise_std,
                                          active_indices):
    if obs.ndim != 2 or pred_obs.ndim != 2:
        raise ValueError("obs and pred_obs must be 2D arrays")
    if type(noise_std) != np.ndarray or noise_std.ndim != 2:
        msg = "noise_std must be a 2d np.ndarray with one column"
        raise ValueError(msg)

    if active_indices is None:
        weights = 1/(np.sqrt(2)*noise_std[:, 0])
        # avoid copy is possible
        # using special indexing with array makes a copy which is slow
        weighted_obs = obs*weights
        weighted_pred_obs = pred_obs*weights
        sq_dists = cdist(weighted_obs, weighted_pred_obs, "sqeuclidean")
        return sq_dists

    weights = 1/(np.sqrt(2)*noise_std[active_indices, 0])
    weighted_obs = obs[:, active_indices]*weights
    weighted_pred_obs = pred_obs[:, active_indices]*weights
    sq_dists = cdist(weighted_obs, weighted_pred_obs, "sqeuclidean")
    return sq_dists


def gaussian_loglike_fun_economial_2D(
        obs, pred_obs, noise_std, active_indices=None):
    if type(noise_std) != np.ndarray:
        noise_std = np.ones((obs.shape[-1], 1), dtype=float)*noise_std
    sq_dists = compute_weighted_sqeuclidian_distance(
        obs, pred_obs, noise_std, active_indices)
    if active_indices is None:
        llike = (0.5*np.sum(np.log(1/(2*np.pi*noise_std[:, 0]**2))) -
                 sq_dists[0, :])
    else:
        llike = (0.5*np.sum(
            np.log(1/(2*np.pi*noise_std[active_indices, 0]**2))) -
                 sq_dists[0, :])

    if llike.ndim == 1:
        llike = llike[:, None]
    return llike


def gaussian_loglike_fun(obs, pred_obs, noise_std, active_indices=None):
    assert pred_obs.shape[-1] == obs.shape[-1]
    if pred_obs.ndim == 3 and obs.ndim == 2 and obs.shape[0] != 1:
        return gaussian_loglike_fun_economial_3D(
            obs, pred_obs, noise_std, active_indices)
    elif obs.ndim == 2 and pred_obs.ndim == 2 and obs.shape != pred_obs.shape:
        return gaussian_loglike_fun_economial_2D(
            obs, pred_obs, noise_std, active_indices)
    else:
        return gaussian_loglike_fun_broadcast(
            obs, pred_obs, noise_std, active_indices)


@njit(cache=True)
def _evidences(inner_log_likelihood_vals, in_weights):
    MM, NN = inner_log_likelihood_vals.shape
    evidences = np.empty((MM, 1))
    for mm in range(MM):
        evidences[mm, 0] = 0.0
        for nn in range(NN):
            evidences[mm, 0] += math.exp(
                inner_log_likelihood_vals[mm, nn])*in_weights[mm, nn]
    return evidences


def _loglike_fun_from_noiseless_obs(
        noiseless_obs, pred_obs, noise_realizations, noise_std,
        active_indices):
    nactive_indices = active_indices.shape[0]
    obs = noiseless_obs[:, active_indices].copy()
    obs += noise_realizations[:, :nactive_indices]
    if pred_obs.ndim == 3:
        return gaussian_loglike_fun_3d_prereduced(
            obs, pred_obs, noise_std, active_indices)
    return gaussian_loglike_fun(
        obs, pred_obs[:, active_indices], noise_std[active_indices])


@njit(cache=True)
def _compute_evidences_repeated_in_samples(
        out_obs, in_pred_obs, in_weights, active_indices, noise_std):
    nout_samples = out_obs.shape[0]
    nin_samples = in_pred_obs.shape[0]
    nactive_indices = active_indices.shape[0]
    const1 = -1/(2*noise_std[:, 0]**2)
    evidences = np.empty((nout_samples, 1))
    const2 = 0.5*np.sum(np.log(-const1[active_indices]/np.pi))
    for ii in range(nout_samples):
        evidences[ii, 0] = 0.0
        for jj in range(nin_samples):
            loglike_val = const2
            for kk in range(nactive_indices):
                loglike_val += const1[active_indices[kk]]*(
                    out_obs[ii, kk] -
                    in_pred_obs[jj, active_indices[kk]])**2
            evidences[ii, 0] += math.exp(loglike_val)*in_weights[jj, 0]
    return evidences


def _compute_evidences(
        out_pred_obs, in_pred_obs, in_weights,
        out_weights, active_indices, noise_samples, noise_std):
    # warning outerloop_pred_obs has already been reduced down to
    # the active indices but in_pred_obs has not. The cost of
    # copying (from special indexing) is too expensive for the
    # later (and larger) array.

    outer_log_likelihood_vals = _loglike_fun_from_noiseless_obs(
        out_pred_obs, out_pred_obs, noise_samples,
        noise_std, active_indices)

    out_obs = out_pred_obs[:, active_indices].copy()
    out_obs += noise_samples[:, :active_indices.shape[0]]
    evidences = _compute_evidences_repeated_in_samples(
        out_obs, in_pred_obs, in_weights, active_indices, noise_std)

    return outer_log_likelihood_vals, evidences


def _compute_expected_kl_utility_monte_carlo(
        out_pred_obs, in_pred_obs, in_weights, out_weights,
        active_indices, noise_samples, noise_std, data_risk_fun, return_all):
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

    out_pred_obs : np.ndarray (nout_samples, ncandidates)
        The noiseless values of out_obs with noise removed

    in_pred_obs : np.ndarray (nout_samples*nin_samples, ncandidates)
        The noiseless values of obs_fun at all sets of innerloop samples
        used for each outerloop iteration. The values are stacked such
        that np.vstack((inner_loop1_vals, inner_loop2_vals, ...))

    in_weights  : np.ndarray (nout_samples, nin_samples)
        The quadrature weights associated with each in_pred_obs set
        used to compute the inner integral.

    out_weights  : np.ndarray (nout_samples, 1)
        The quadrature weights associated with each out_pred_obs set
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

    outer_log_likelihood_vals, evidences = _compute_evidences(
         out_pred_obs, in_pred_obs, in_weights, out_weights,
         active_indices, noise_samples, noise_std)

    divergences = outer_log_likelihood_vals - np.log(evidences)
    utility_val = data_risk_fun(divergences, out_weights)
    # utility_val = np.sum((outer_log_likelihood_vals - np.log(evidences)) *
    #                      out_weights)
    if not return_all:
        return {"utility_val": utility_val}
    result = {"utility_val": utility_val, "evidences": evidences,
              "divergences": divergences}
    return result


def _precompute_out_quadrature_rule(
        nvars, generate_prior_noise_samples, nout_samples):
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

    nout_samples : integer
        The number of Monte Carlo samples used to compute the outer integral
        over all possible observations

    Returns
    -------
    out_quad_data : tuple(outerquad_x, outerquad_w)
        Tuple containing the points and weights of the outer quadrature rule
        with respect to the joint density of the prior and noise.
        outerquad_x is np.ndarray(nprior_vars, nquad)
        outerquad_w is np.ndarray(nquad, 1)

    out_noise_samples : np.ndarray (max_ncollected_obs, nquad)
    """
    out_samples, out_weights = \
        generate_prior_noise_samples(nout_samples)
    out_prior_samples = out_samples[:nvars]
    out_noise_samples = out_samples[nvars:]
    out_prior_quad_data = (
        out_prior_samples, out_weights)
    return (out_prior_quad_data, out_noise_samples)


def _precompute_expected_kl_utility_data(out_quad_data, in_quad_data, obs_fun):
    r"""
    Parameters
    ----------
    out_quad_data : tuple(outerquad_x, outerquad_w)
        Tuple containing the points and weights of the outer quadrature rule
        with respect to the prior used for outerloop calculations.
        outerquad_x is np.ndarray(nprior_vars, nquad)
        outerquad_w is np.ndarray(nquad, 1)

    in_quad_data : tuple(outerquad_x, outerquad_w)
        Tuple containing the points and weights of the outer quadrature rule
        with respect to the joint prior used for innerloop calculations.
        outerquad_x is np.ndarray(nprior_vars, nquad)
        outerquad_w is np.ndarray(nquad, 1)

    obs_fun : callable
        Function with the signature

        `obs_fun(samples) -> np.ndarray(nsamples, nqoi)`

        That returns noiseless evaluations of the forward model.

    Returns
    -------
    out_pred_obs : np.ndarray (nout_samples, ncandidates)
        The noiseless values of out_obs with noise removed

    in_pred_obs : np.ndarray (nout_samples*nin_samples, ncandidates)
        The noiseless values of obs_fun at all sets of innerloop samples
        used for each outerloop iteration. The values are stacked such
        that np.vstack((inner_loop1_vals, inner_loop2_vals, ...))
    """
    out_prior_samples, out_weights = out_quad_data
    assert out_weights.ndim == 2
    print(f"Running {out_prior_samples.shape[1]} outer model evaluations")
    t0 = time.time()
    out_pred_obs = obs_fun(out_prior_samples)
    print("Predicted observation bounds",
          out_pred_obs.max(), out_pred_obs.min())
    print("Evaluations took", time.time()-t0)

    if out_pred_obs.shape[0] != out_prior_samples.shape[1]:
        msg = "obs_fun is not returning an array with the correct shape."
        msg += f" nrows is {out_pred_obs.shape[0]}. Should be "
        msg += f"{out_prior_samples.shape[1]}"
        raise ValueError(msg)

    print(f"Running {in_quad_data[0].shape[1]} inner model evaluations")
    t0 = time.time()
    in_pred_obs = obs_fun(in_quad_data[0])
    print("Evaluations took", time.time()-t0)

    return out_pred_obs, in_pred_obs


def _precompute_expected_deviation_data(
        out_quad_data, in_quad_data, obs_fun, qoi_fun):
    out_pred_obs, in_pred_obs = _precompute_expected_kl_utility_data(
        out_quad_data, in_quad_data, obs_fun)
    in_samples = in_quad_data[0]
    print(f"Running {in_samples.shape[1]} qoi model evaluations")
    t0 = time.time()
    in_pred_qois = qoi_fun(in_samples)
    print("Evaluations took", time.time()-t0)
    if in_pred_qois.shape[0] != in_samples.shape[1]:
        msg = "qoi_fun is not returning an array with the correct shape"
        raise ValueError(msg)
    return out_pred_obs, in_pred_obs, in_pred_qois


def _evidences_and_weights(inner_log_likelihood_vals, in_weights):
    inner_likelihood_vals = np.exp(inner_log_likelihood_vals)
    evidences = np.einsum(
        "ij,ij->i", inner_likelihood_vals, in_weights)[:, None]
    weights = inner_likelihood_vals*in_weights/evidences
    return evidences, weights


# @njit(cache=True)
# def _evidences_and_weights(inner_log_likelihood_vals, in_weights):
#     MM, NN = inner_log_likelihood_vals.shape
#     evidences = np.empty((MM, 1))
#     weights = in_weights.copy()
#     for mm in range(MM):
#         evidences[mm, 0] = 0.0
#         for nn in range(NN):
#             likelihood_val = math.exp(inner_log_likelihood_vals[mm, nn])
#             weights[mm, nn] *= likelihood_val
#             evidences[mm, 0] += weights[mm, nn]
#     return evidences, weights/evidences


def _compute_negative_expected_deviation_monte_carlo(
        out_pred_obs, in_pred_obs, in_weights, out_weights,
        in_pred_qois, deviation_fun, pred_risk_fun, data_risk_fun,
        noise_samples, noise_std, active_indices, return_all):
    nout_samples = out_pred_obs.shape[0]

    out_obs = out_pred_obs[:, active_indices].copy()
    # print(out_obs.shape, active_indices.shape, noise_samples.shape)
    out_obs += noise_samples[:, :active_indices.shape[0]]
    deviations, evidences = deviation_fun(
        out_obs, in_pred_obs, in_weights, active_indices, noise_std,
        in_pred_qois)
    # deviation.shape = [nout_quad, nprediction_candidates]
    # evidences.shape = [nout_quad, 1]

    # expectation taken with respect to observations
    # assume always want deviation here, but this can be changed
    # expected_obs_deviations = np.sum(deviations*out_weights, axis=0)
    # use einsum because it does not create intermediate arrays
    # expected_obs_deviations = np.einsum(
    #    "ij,i->j", deviations, out_weights[:, 0])
    expected_obs_deviations = data_risk_fun(deviations, out_weights)

    disutility_val = pred_risk_fun(expected_obs_deviations)

    utility_val = -disutility_val
    if not return_all:
        return {'utility_val': utility_val}
    result = {
        'utility_val': utility_val, 'evidences': evidences,
        'deviations': deviations,
        'expected_deviations': expected_obs_deviations}
    return result


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

        `obs_process(design_indices) -> np.ndarray (1, ndesign_indices)`

        where design_samples is np.ndarary (nvars, ndesign_indices)

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
        (collected_design_indices, new_design_indices)).astype(int)
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


def _define_design_data(collected_design_indices,
                        new_design_indices, ndesign_candidates,
                        ndata_per_candidate, noise_std):
    # unlike open loop design (closed loop batch design)
    # we do not update inner and outer loop weights but rather
    # just compute likelihood for all collected and new design indices
    # If want to update weights then we must have a different set of
    # weights for each inner iteration of the inner loop that is
    # computed using
    # the associated outerloop data
    if collected_design_indices is not None:
        design_indices = np.hstack(
            (collected_design_indices, new_design_indices))
    else:
        # assume the observations at the collected_design_indices are
        # already incorporated into the inner and outer loop weights
        design_indices = np.asarray(new_design_indices)

    active_indices = np.hstack([idx*ndata_per_candidate + np.arange(
        ndata_per_candidate) for idx in design_indices])
    # assume same noise_std for each data point associated with a design
    # candidate. Consider adding ndata_per_candidate to self.__init__
    # and defining noise_std correct length there.
    noise_std = np.hstack(
        [[noise_std[idx, 0]]*ndata_per_candidate
         for idx in range(ndesign_candidates)])[:, None]
    return active_indices, noise_std


def oed_data_expectation(deviations, weights):
    """
    Compute the expected deviation for each outer loop sample

    Parameters
    ----------
    deviations : np.ndarray (nout_samples, nqois)
         The samples

    weights : np.ndarray (nout_samples, 1)
        Weights associated with each inner loop sample

    Returns
    -------
    expected_obs_deviations : np.ndarray (nqois, 1)
        The deviation vals
    """
    expected_obs_deviations = np.einsum(
        "ij,i->j", deviations, weights[:, 0])[:, None]
    return expected_obs_deviations


class AbstractBayesianOED(ABC):
    r"""Base Bayesian OED class"""

    def __init__(self, ndesign_candidates, obs_fun, noise_std,
                 prior_variable, out_quad_opts,
                 in_quad_opts, nprocs=1,
                 max_ncollected_obs=2, ndata_per_candidate=1,
                 data_risk_fun=oed_data_expectation):
        """
        Constructor.

        Parameters
        ----------
        ndesign_candidates :
            The number of design candidates

        obs_fun : callable
            Function with the signature

            `obs_fun(samples) -> np.ndarray(nsamples, ndesign_candidates*ndata_per_candidate)`

            That returns noiseless evaluations of the forward model.

        noise_std : float or np.ndarray (nobs, 1)
            The standard deviation of the mean zero Gaussian noise added to
            each observation

        prior_variable : pya.IndependentMarginalsVariable
            The prior variable consisting of independent univariate random
            variables

        generate_inner_prior_samples : callable
           Function with the signature

            `generate_inner_prior_samples(nsamples) -> np.ndarray(
             nvars, nsamples), np.ndarray(nsamples, 1)`

            Generate samples and associated weights used to evaluate
            the evidence computed by the inner loop
            If None then the function generate_outer_prior_samples is used and
            weights are assumed to be 1/nsamples. This function is useful if
            wanting to use multivariate quadrature to evaluate the evidence

        nin_samples : integer
            The number of quadrature samples used for the inner integral that
            computes the evidence for each realiaztion of the predicted
            observations

        nout_samples : integer
            The number of Monte Carlo samples used to compute the outer
            integral over all possible observations

        pre_collected_design_indices : np.ndarray (nobs)
            The indices into the qoi vector associated with the
            collected observations

        econ : boolean
            Make all inner loop samples the same for all outer loop samples.
            This reduces number of evaluations of prediction model. Currently
            this common data is copied and repeated for each outer loop sample
            so the rest of the code can remain the same. Eventually the data
            has to be tiled anyway when computing exepcted utility so this is
            not a big deal.

        nprocs : integer
            The number of threads used to compute OED design. Warning:
            this uses multiprocessing.Pool and seems to provide very little
            benefit and in many cases increases the CPU time.

        max_ncollected_obs : integer
            The maximum number of observations that will be collected.

        outer_quad_type : string
            The type of quadrature used for the outerloop. Choose from
            ["mc", :"qmc:, "gauss"]

        """

        self.ndesign_candidates = ndesign_candidates
        self.ndata_per_candidate = ndata_per_candidate
        if not callable(obs_fun):
            raise ValueError("obs_fun must be a callable function")
        self.obs_fun = obs_fun
        self.noise_std = noise_std
        if isinstance(self.noise_std, (float, int)):
            self.noise_std = np.full(
                (self.ndesign_candidates, 1), self.noise_std)
        if (self.noise_std.shape[0] != self.ndesign_candidates):
            msg = "noise_std must be scalar or given for each design candidate"
            raise ValueError(msg)
        # for now assume homoscedastic noise. TODO. currently
        # noise_variable_marginals is used to create quadrature rule
        # used for all design candidates. IF want to allow for heteroscedastic
        # noise then must scale this quadrule for each design location
        # when computing noisy data in utility functions
        # Also assume same noise applied to each data point associated
        # with a design candidate. See compute_expected_utility
        assert np.allclose(self.noise_std, self.noise_std[0])
        self.prior_variable = prior_variable
        self.max_ncollected_obs = max_ncollected_obs
        noise_variable_marginals = [
            stats.norm(0, self.noise_std[0, 0])]*self.max_ncollected_obs
        self.joint_prior_noise_variable = IndependentMarginalsVariable(
            prior_variable.marginals()+noise_variable_marginals)

        (self.out_quad_data, self.noise_samples, self.in_quad_data,
         self.econ) = self._get_quad_rules(out_quad_opts, in_quad_opts)

        self.nout_samples = self.out_quad_data[0].shape[1]
        self.nin_samples = self.in_quad_data[0].shape[1]
        self.out_prior_samples, self.out_weights = self.out_quad_data
        self.in_samples, self.in_weights = self.in_quad_data

        self.collected_design_indices = None
        self.out_pred_obs = None
        self.in_pred_obs = None

        self.nprocs = self._set_nprocs(nprocs)
        self.data_risk_fun = data_risk_fun

    def _get_quad_rules(self, out_quad_opts, in_quad_opts):
        out_quad_data = integrate(
            out_quad_opts["method"], self.joint_prior_noise_variable,
            *out_quad_opts.get("args", []), **out_quad_opts.get("kwargs", []))
        out_prior_samples = out_quad_data[0][:self.prior_variable.num_vars()]
        out_noise_samples = out_quad_data[0][self.prior_variable.num_vars():].T
        out_quad_data = (
            out_prior_samples, out_quad_data[1])
        in_quad_data = integrate(
            in_quad_opts["method"], self.prior_variable,
            *in_quad_opts.get("args", []), **in_quad_opts.get("kwargs", []))
        econ = True
        return out_quad_data, out_noise_samples, in_quad_data, econ

    def _set_nprocs(self, nprocs):
        if (nprocs > 1 and (
                'OMP_NUM_THREADS' not in os.environ or
                not int(os.environ['OMP_NUM_THREADS']) == 1)):
            msg = 'User set assert_omp=True but OMP_NUM_THREADS has not been '
            msg += 'set to 1. Run script with '
            msg += 'OMP_NUM_THREADS=1 python script.py'
            raise Exception(msg)
        return nprocs

    @abstractmethod
    def populate(self):
        raise NotImplementedError()

    def update_design(self, return_all=False, nnew=1):
        if not hasattr(self, "out_pred_obs"):
            raise ValueError("Must call self.populate before creating designs")
        if self.collected_design_indices is None:
            self.collected_design_indices = np.zeros((0), dtype=int)
        if self.collected_design_indices.shape[0]+nnew > self.max_ncollected_obs:
            msg = "To many new design points requested. Decrease nnew and/or "
            msg += "increase self.max_ncollected_obs"
            raise ValueError(msg)
        utility_vals, selected_indices, results = self.select_design(
            self.collected_design_indices, nnew, return_all)

        self.collected_design_indices = np.hstack(
            (self.collected_design_indices, selected_indices)).astype(int)
        if return_all is False:
            return utility_vals, selected_indices, None
        return utility_vals, selected_indices, results

    def set_collected_design_indices(self, indices):
        self.collected_design_indices = indices.copy()

    @abstractmethod
    def compute_expected_utility(self, collected_design_indices,
                                 new_design_indices, return_all=False):
        raise NotImplementedError()

    def _compute_utilities_parallel_shared(
            self, worker_fun, init_worker, _get_initargs, indices,
            collected_design_indices, return_all):
        nindices = indices.shape[0]
        splits = split_indices(nindices, self.nprocs)
        args = [(indices[splits[ii]:splits[ii+1]], collected_design_indices,
                 return_all, self.ndesign_candidates, self.ndata_per_candidate)
                for ii in range(splits.shape[0]-1)]
        # t0 = time.time()
        with Pool(processes=self.nprocs, initializer=init_worker,
                  initargs=_get_initargs(self)) as pool:
            # print("pool startup", time.time()-t0)
            result = pool.map(worker_fun, args)
        utility_vals = np.hstack([r[0] for r in result])
        results = []
        for r in result:
            results += r[1]
        return utility_vals, results

    # does not work due to ussues with pickling
    # def _compute_utilities_parallel(self, ncandidates,
    #                                 collected_design_indices, return_all):
    #     splits = split_indices(self.ndesign_candidates, self.nprocs)
    #     args = [(splits[ii], splits[ii+1]) for ii in range(splits.shape[0]-1)]
    #     t0 = time.time()
    #     with Pool(processes=self.nprocs) as pool:
    #         print("pool startup", time.time()-t0)
    #         result = pool.map(
    #             partial(self._compute_utilities_serial,
    #                     self.compute_expected_utility,
    #                     collected_design_indices,
    #                     return_all), args)
    #     utility_vals = np.hstack([r[0] for r in result])
    #     results = []
    #     for r in result:
    #         results += r[1]
    #     return utility_vals, results

    def _compute_utilities_serial(self, compute_expected_utility,
                                  collected_design_indices, return_all,
                                  indices):
        import time
        t0 = time.time()
        ncandidates = indices.shape[0]
        utility_vals = -np.ones(ncandidates)*np.inf
        results = [None for ii in range(ncandidates)]
        ii = 0
        for idx in indices:
            results[ii] = compute_expected_utility(
                collected_design_indices, np.asarray(idx, dtype=int),
                return_all=return_all)
            utility_vals[ii] = results[ii]["utility_val"]
            ii += 1
        print("Computing utilities in serial took", time.time()-t0)
        return utility_vals, results

    def compute_utilities(self, ncandidates, collected_design_indices,
                          new_indices, return_all):
        if self.nprocs == 1:
            return self._compute_utilities_serial(
                self.compute_expected_utility,
                collected_design_indices, return_all, new_indices)
        return self._compute_utilities_parallel_shared(
            kl_worker_fun, init_kl_worker, _get_kl_compute_utilities_initargs,
            new_indices, collected_design_indices, return_all)
        # return self._compute_utilities_parallel(
        #     ncandidates, collected_design_indices, return_all)

    def select_design(self, collected_design_indices, nnew, return_all):
        """
        Update an experimental design.

        Parameters
        ----------
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

        results : dict
            Dictionary of useful data used to compute expected utility
            At a minimum it has the keys ["utilties", "evidences", "weights"]
        """
        new_indices = np.asarray(list(itertools.combinations_with_replacement(
            np.arange(self.ndesign_candidates), nnew)))
        utility_vals, results = self.compute_utilities(
            self.ndesign_candidates, collected_design_indices, new_indices,
            return_all)
        selected_index = new_indices[np.argmax(np.round(utility_vals, 16))]
        if not return_all:
            results = None
        return utility_vals, selected_index, results

    def _define_design_data(self, collected_design_indices,
                            new_design_indices):
        return _define_design_data(collected_design_indices,
                                   new_design_indices, self.ndesign_candidates,
                                   self.ndata_per_candidate, self.noise_std)


class OEDSharedData():
    def __init__(self):
        self.attr_names = [
            "in_weights", "out_weights",
            "in_pred_obs", "out_pred_obs", "noise_samples", "noise_std"]

    def set_data(self, data):
        assert len(data) == 2*len(self.attr_names)
        for ii in range(len(self.attr_names)):
            setattr(self, self.attr_names[ii], data[2*ii])
            setattr(self, self.attr_names[ii]+"_shape", data[2*ii+1])

    def clear(self):
        self.set_data([None]*len(self.attr_names))
        self.in_pred_qois = None
        self.set_funs(None, None, None)

    def set_funs(self, deviation_fun, pred_risk_fun,
                 data_risk_fun):
        self.deviation_fun = deviation_fun
        self.pred_risk_fun = pred_risk_fun
        self.data_risk_fun = data_risk_fun

    def set_in_pred_qois(self, in_pred_qois, shape):
        self.in_pred_qois = in_pred_qois
        self.in_pred_qois_shape = shape


global_oed_shared_data = OEDSharedData()


def _create_global_array_from_name(obj, name):
    array = getattr(obj, name)
    X = RawArray('d', int(np.prod(array.shape)))
    # X[:] = array.ravel() very slow
    X_np = np.frombuffer(X).reshape(array.shape)
    np.copyto(X_np, array)
    return X, array.shape


import time
def _get_kl_compute_utilities_initargs(obj):
    # t0 = time.time()
    initargs = []
    for name in global_oed_shared_data.attr_names:
        X, shape = _create_global_array_from_name(obj, name)
        initargs += [X, shape]
    initargs = (initargs, obj.data_risk_fun)
    # print("data time", time.time()-t0)
    return (initargs, )


def init_kl_worker(args):
    data, data_risk_fun = args
    global_oed_shared_data.set_data(data)
    funs = [None, None, data_risk_fun]
    global_oed_shared_data.set_funs(*funs)


def _get_deviation_compute_utilities_initargs(obj):
    # t0 = time.time()
    initargs = _get_kl_compute_utilities_initargs(obj)[0][0]
    X, shape = _create_global_array_from_name(obj, "in_pred_qois")
    initargs = (initargs, [X, shape],
                [obj.deviation_fun, obj.pred_risk_fun, obj.data_risk_fun])
    # print("data time", time.time()-t0)
    return (initargs, )


def init_deviation_worker(initargs):
    data, in_pred_qois, funs = initargs
    global_oed_shared_data.set_data(data)
    global_oed_shared_data.set_in_pred_qois(*in_pred_qois)
    global_oed_shared_data.set_funs(*funs)



def from_buffer(name):
    return np.frombuffer(getattr(global_oed_shared_data, name)).reshape(
        getattr(global_oed_shared_data, name+"_shape"))


def kl_worker_fun(arg):
    t0 = time.time()
    in_weights = from_buffer("in_weights")
    out_weights = from_buffer("out_weights")
    in_pred_obs = from_buffer("in_pred_obs")
    out_pred_obs = from_buffer("out_pred_obs")
    noise_samples = from_buffer("noise_samples")
    noise_std = from_buffer("noise_std")
    indices = arg[0]
    collected_design_indices, return_all = arg[1], arg[2]
    ndesign_candidates, ndata_per_candidate = arg[3], arg[4]
    data_risk_fun = global_oed_shared_data.data_risk_fun
    nindices = indices.shape[0]
    t0 = time.time()
    results = [None for ii in range(nindices)]
    utility_vals = np.full(nindices, -np.inf)
    ii = 0

    noise_std = np.hstack(
        [[noise_std[idx, 0]]*ndata_per_candidate
         for idx in range(ndesign_candidates)])[:, None]
    for new_design_indices in indices:
        if collected_design_indices is not None:
            design_indices = np.hstack(
                (collected_design_indices, new_design_indices))
        else:
            # assume the observations at the collected_design_indices
            # are already incorporated into the inner and outer loop weights
            design_indices = np.asarray(new_design_indices)

        active_indices = np.hstack([idx*ndata_per_candidate + np.arange(
            ndata_per_candidate) for idx in design_indices])
        results[ii] = _compute_expected_kl_utility_monte_carlo(
            out_pred_obs,  in_pred_obs,  in_weights,
            out_weights, active_indices, noise_samples, noise_std,
            data_risk_fun, return_all)
        utility_vals[ii] = results[ii]["utility_val"]
        ii += 1
    print("Worker Took", time.time()-t0, indices[0], indices[-1])
    return utility_vals, results


def deviation_worker_fun(arg):
    t0 = time.time()
    in_weights = from_buffer("in_weights")
    out_weights = from_buffer("out_weights")
    in_pred_obs = from_buffer("in_pred_obs")
    out_pred_obs = from_buffer("out_pred_obs")
    noise_samples = from_buffer("noise_samples")
    noise_std = from_buffer("noise_std")
    in_pred_qois = from_buffer("in_pred_qois")
    indices = arg[0]
    collected_design_indices, return_all = arg[1], arg[2]
    ndesign_candidates, ndata_per_candidate = arg[3], arg[4]
    deviation_fun, pred_risk_fun, data_risk_fun = (
        global_oed_shared_data.deviation_fun,
        global_oed_shared_data.pred_risk_fun,
        global_oed_shared_data.data_risk_fun)

    nindices = indices.shape[0]
    t0 = time.time()
    results = [None for ii in range(nindices)]
    utility_vals = np.full(nindices, -np.inf)
    ii = 0
    noise_std = np.hstack(
        [[noise_std[idx, 0]]*ndata_per_candidate
         for idx in range(ndesign_candidates)])[:, None]
    for new_design_indices in indices:
        if collected_design_indices is not None:
            design_indices = np.hstack(
                (collected_design_indices, new_design_indices))
        else:
            # assume the observations at the collected_design_indices
            # are already incorporated into the inner and outer loop weights
            design_indices = np.asarray(new_design_indices)
        active_indices = np.hstack([idx*ndata_per_candidate + np.arange(
            ndata_per_candidate) for idx in design_indices])
        results[ii] = _compute_negative_expected_deviation_monte_carlo(
            out_pred_obs,  in_pred_obs,  in_weights, out_weights,
            in_pred_qois,  deviation_fun, pred_risk_fun,
            data_risk_fun, noise_samples, noise_std, active_indices, return_all)
        utility_vals[ii] = results[ii]["utility_val"]
        ii += 1
    print("Worker Took", time.time()-t0, indices[0], indices[-1])
    return utility_vals, results


class BayesianBatchKLOED(AbstractBayesianOED):
    r"""
    Compute open-loop OED my maximizing KL divergence between the prior and
    posterior.
    """

    def populate(self):
        (self.out_pred_obs, self.in_pred_obs) = \
            _precompute_expected_kl_utility_data(
                self.out_quad_data, self.in_quad_data, self.obs_fun)
        assert (self.out_pred_obs.shape[1] ==
                self.ndesign_candidates*self.ndata_per_candidate)

    def compute_expected_utility(self, collected_design_indices,
                                 new_design_indices, return_all=False):
        """
        return_all true used for debugging returns more than just utilities
        and also returns itermediate data useful for testing
        """
        active_indices, noise_std = self._define_design_data(
            collected_design_indices, new_design_indices)
        return _compute_expected_kl_utility_monte_carlo(
            self.out_pred_obs,  self.in_pred_obs,  self.in_weights,
            self.out_weights, active_indices, self.noise_samples,
            noise_std, self.data_risk_fun, return_all)


def oed_prediction_average(qoi_vals, weights=None):
    assert qoi_vals.ndim == 2 and qoi_vals.shape[1] == 1
    if weights is None:
        return qoi_vals.mean()

    assert weights.shape[1] == 1
    return np.sum(qoi_vals*weights, axis=0)


class OEDQOIDeviation():
    def __init__(self, name, *args):
        self.name = name
        self._args = args

    def __call__(self, out_obs, in_pred_obs, in_weights, active_indices,
                 noise_std, in_pred_qois):
        if self.name == "variance":
            deviation_fun = _posterior_push_fwd_variance_deviation
        elif self.name == "std_dev":
            deviation_fun = _posterior_push_fwd_standard_deviation
        elif self.name == 'entropic':
            deviation_fun = _posterior_push_fwd_entropic_deviation
        elif self.name == 'cvar':
            deviation_fun = _posterior_push_fwd_cvar_deviation
        else:
            msg = f"deviation: {self.name} is not supported"
            raise NotImplementedError(msg)

        return _compute_posterior_push_fwd_deviation(
            out_obs, in_pred_obs, in_weights, active_indices, noise_std,
            in_pred_qois, deviation_fun, *self._args)



@njit(cache=True)
def _posterior_push_fwd_variance_deviation(qoi_vals, nin_samples, weights):
    qoi_mean = np.zeros(qoi_vals.shape[1])
    deviations = np.zeros(qoi_vals.shape[1])
    for jj in range(nin_samples):
        qoi_mean += qoi_vals[jj, :]*weights[jj]
        deviations += qoi_vals[jj, :]**2*weights[jj]
    deviations -= qoi_mean**2
    return deviations


@njit(cache=True)
def _posterior_push_fwd_standard_deviation(qoi_vals, nin_samples, weights):
    deviations = np.sqrt(_posterior_push_fwd_variance_deviation(
        qoi_vals, nin_samples, weights))
    return deviations


@njit(cache=True)
def _posterior_push_fwd_entropic_deviation(
        qoi_vals, nin_samples, weights):
    qoi_mean = np.zeros(qoi_vals.shape[1])
    deviations = np.zeros(qoi_vals.shape[1])
    for jj in range(nin_samples):
        qoi_mean += qoi_vals[jj, :]*weights[jj]
        deviations += np.exp(qoi_vals[jj, :])*weights[jj]
    deviations = np.log(deviations)-qoi_mean
    return deviations


@njit(cache=True)
def _posterior_push_fwd_cvar_deviation(
        qoi_vals, nin_samples, weights, beta):
    qoi_means = np.zeros(qoi_vals.shape[1])
    # qoi_vars = np.zeros(qoi_vals.shape[1])
    for jj in range(nin_samples):
        qoi_means += qoi_vals[jj, :]*weights[jj]
        # qoi_vars += qoi_vals[jj, :]**2*weights[jj]
    # qoi_vars -= qoi_means**2
    nqoi = qoi_vals.shape[1]
    weights_expanded = np.empty_like(qoi_vals.T)
    for kk in range(nqoi):
        weights_expanded[kk] = weights
    # qoi_vals.copy() is necessary because numba is updating qoi_vals
    risks = conditional_value_at_risk_vectorized(
        qoi_vals.copy().T, beta, weights_expanded,
        samples_sorted=False)
    deviations = risks-qoi_means
    return deviations


@njit(cache=True)
def _compute_posterior_push_fwd_deviation(
        out_obs, in_pred_obs, in_weights, active_indices, noise_std,
        qoi_vals, deviation_fun, *args):
    nout_samples = out_obs.shape[0]
    nin_samples = in_pred_obs.shape[0]
    nactive_indices = active_indices.shape[0]
    const1 = -1/(2*noise_std[:, 0]**2)
    evidences = np.empty((nout_samples, 1))
    deviations = np.empty((nout_samples, qoi_vals.shape[1]))
    const2 = 0.5*np.sum(np.log(-const1[active_indices]/np.pi))
    weights = np.empty(nin_samples)
    for ii in range(nout_samples):
        evidences[ii, 0] = 0.0
        for jj in range(nin_samples):
            loglike_val = const2
            for kk in range(nactive_indices):
                loglike_val += const1[active_indices[kk]]*(
                    out_obs[ii, kk] -
                    in_pred_obs[jj, active_indices[kk]])**2
            weights[jj] = math.exp(loglike_val)*in_weights[jj, 0]
            evidences[ii, 0] += weights[jj]

        weights /= evidences[ii, 0]
        deviations[ii, :] = deviation_fun(
            qoi_vals, nin_samples, weights, *args)

    return deviations, evidences


def oed_variance_deviation(samples, weights):
    """
    Compute the variance deviation for each outer loop sample using the
    corresponding inner loop samples

    Parameters
    ----------
    samples : np.ndarray (nout_samples, nin_samples, nqois)
         The samples

    weights : np.ndarray (nout_samples, nin_samples)
        Weights associated with each inner loop sample

    Returns
    -------
    deviation_vals : np.ndarray (nout_samples, nqois)
        The deviation vals
    """
    # For large arrays variance_3D_pyx is the same speed as einsum
    # implementation below
    try:
        from pyapprox.cython.utilities import variance_3D_pyx
        return variance_3D_pyx(samples, weights)
    except:
        pass
    means = np.einsum(
         "ijk,ij->ik", samples, weights)
    variances = np.einsum(
        "ijk,ij->ik", samples**2, weights)-means**2
    return variances


def oed_entropic_deviation(samples, weights):
    """
    Compute the entropic risk deviation for each outer loop sample using the
    corresponding inner loop samples

    Parameters
    ----------
    samples : np.ndarray (nout_samples, nin_samples, nqois)
         The samples

    weights : np.ndarray (nout_samples, nin_samples)
        Weights associated with each inner loop sample

    Returns
    -------
    deviation_vals : np.ndarray (nout_samples, nqois)
        The deviation vals
    """
    means = np.einsum(
        "ijk,ij->ik", samples, weights)
    risks = np.log(np.einsum(
        "ijk,ij->ik", np.exp(samples), weights))
    return risks-means


def oed_data_cvar(deviations, weights, quantile=None):
    """
    Compute the conditional value of risk of the deviations
    for each outer loop sample

    Parameters
    ----------
    deviations : np.ndarray (nout_samples, nqois)
         The samples

    weights : np.ndarray (nout_samples, 1)
        Weights associated with each inner loop sample

    quantile : float
        The quantile used to compute of the conditional value at risk
        of the deviations for each outerloop obsevation

    Returns
    -------
    cvar_obs_deviations : np.ndarray (nqois, 1)
        The deviation vals
    """
    assert quantile is not None
    cvar_obs_deviations = np.empty((deviations.shape[1], 1))
    for qq in range(deviations.shape[1]):
        cvar_obs_deviations[qq, 0] = conditional_value_at_risk(
            deviations[:, qq], quantile, weights[:, 0], False)
    return cvar_obs_deviations


def oed_standard_deviation(samples, weights):
    """
    Compute the standard deviation for each outer loop sample using the
    corresponding inner loop samples

    Parameters
    ----------
    samples : np.ndarray (nout_samples, nin_samples, nqois)
         The samples

    weights : np.ndarray (nout_samples, nin_samples)
        Weights associated with each inner loop sample

    Returns
    -------
    deviation_vals : np.ndarray (nout_samples, nqois)
        The deviation vals
    """
    variance = oed_variance_deviation(samples, weights)
    # rouding error can cause slightly negative values
    variance[variance < 0] = 0
    return np.sqrt(variance)


def oed_conditional_value_at_risk_deviation(samples, weights, quantile=None,
                                            samples_sorted=True):
    """
    Compute the conditional value at risk deviation for each outer loop
    sample using the corresponding inner loop samples

    Parameters
    ----------
    samples : np.ndarray (nout_samples, nin_samples, nqois)
         The samples

    weights : np.ndarray (nout_samples, nin_samples)
        Weights associated with each inner loop sample

    quantile : float
        The quantile of the conditional value at risk used to
        compute the deviation

    Returns
    -------
    deviation_vals : np.ndarray (nout_samples, nqois)
        The deviation vals
    """
    assert quantile is not None
    if samples.shape[2] > 1 and samples_sorted:
        raise ValueError("samples cannot be sorted if nqoi > 1")
    cvars = np.empty((samples.shape[0], samples.shape[2]))
    for ii in range(samples.shape[0]):
        for qq in range(samples.shape[2]):
            mean = np.sum(samples[ii, :, qq]*weights[ii, :])
            cvars[ii, qq] = (conditional_value_at_risk(
                samples[ii, :, qq], quantile, weights[ii, :], samples_sorted) -
                             mean)
    return cvars


class BayesianBatchDeviationOED(AbstractBayesianOED):
    r"""
    Compute open-loop OED by minimizing the deviation on the push forward
    of the posterior through a QoI model.
    """

    def __init__(self, ndesign_candidates, obs_fun, noise_std,
                 prior_variable, out_quad_opts, in_quad_opts, qoi_fun=None,
                 deviation_fun=oed_standard_deviation,
                 pred_risk_fun=oed_prediction_average,
                 data_risk_fun=oed_data_expectation,
                 nprocs=1, max_ncollected_obs=2, ndata_per_candidate=1):
        r"""
        Constructor.

        Parameters
        ----------
        design_candidates : np.ndarray (nvars, nsamples)
            The location of all design sample candidates

        obs_fun : callable
            Function with the signature

            `obs_fun(samples) -> np.ndarray(nsamples, nqoi)`

            That returns noiseless evaluations of the forward model.

        noise_std : float or np.ndarray (nobs, 1)
            The standard deviation of the mean zero Gaussian noise added to
            each observation

        prior_variable : pya.IndependentMarginalsVariable
            The prior variable consisting of independent univariate random
            variables

        qoi_fun : callable
            Function with the signature

            `qoi_fun(samples) -> np.ndarray(nsamples, nqoi)`

            That returns evaluations of the forward model. Observations are
            assumed to be :math:`f(z)+\epsilon` where :math:`\epsilon` is
            additive noise nsamples : np.ndarray (nvars, nsamples)

        generate_inner_prior_samples : callable
           Function with the signature

            `generate_inner_prior_samples(nsamples) -> np.ndarray(
             nvars, nsamples), np.ndarray(nsamples, 1)`

            Generate samples and associated weights used to evaluate
            the evidence computed by the inner loop
            If None then the function generate_outer_prior_samples is used and
            weights are assumed to be 1/nsamples. This function is useful if
            wanting to use multivariate quadrature to evaluate the evidence

        nin_samples : integer
            The number of quadrature samples used for the inner integral that
            computes the evidence for each realiaztion of the predicted
            observations

        nout_samples : integer
            The number of Monte Carlo samples used to compute the outer
            integral over all possible observations

        pre_collected_design_indices : np.ndarray (nobs)
            The indices into the qoi vector associated with the
            collected observations

        econ : boolean
            Make all inner loop samples the same for all outer loop samples.
            This reduces number of evaluations of prediction model. Currently
            this common data is copied and repeated for each outer loop sample
            so the rest of the code can remain the same. Eventually the data
            has to be tiled anyway when computing exepcted utility so this is
            not a big deal.

         deviation_fun : callable
             Function with the signature

            `deviation_fun(in_pred_qois, weights) ->
             np.ndarray(nout_samples, nqois)`

             where

             in_pred_qois : np.ndarray (
             nout_samples, nin_samples, nqois)
             weights : np.ndarray (nout_samples, nin_samples)

        nprocs : integer
            The number of threads used to compute OED design. Warning:
            this uses multiprocessing.Pool and seems to provide very little
            benefit and in many cases increases the CPU time.

        pred_risk_fun : callable
            Function to compute risk over multiple qoi with the signature

             `pred_risk_fun(expected_deviations) -> float`

            where expected_deviations : np.ndarray (nqois, 1)

        data_risk_fun : callable
            Function to compute risk of deviations over all outerloop samples

             `data_risk_fun(deviations) -> np.ndarray (nqois, 1)`

            where deviations : np.ndarray (nout_samples, nqois)
        """

        super().__init__(ndesign_candidates, obs_fun, noise_std,
                         prior_variable, out_quad_opts,
                         in_quad_opts, nprocs=nprocs,
                         max_ncollected_obs=max_ncollected_obs,
                         ndata_per_candidate=ndata_per_candidate,
                         data_risk_fun=data_risk_fun)
        # qoi fun deafult is None so that same api can be used for KL based OED
        # which does not require qoi_fun
        if not callable(qoi_fun):
            raise ValueError("qoi_fun must be a callable function")
        if not callable(deviation_fun):
            raise ValueError("deviation_fun must be a callable function")
        self.qoi_fun = qoi_fun
        self.deviation_fun = deviation_fun
        self.pred_risk_fun = pred_risk_fun

    def _populate(self):
        """
        Compute the data needed to initialize the OED algorithm.
        """
        (self.out_pred_obs, self.in_pred_obs,
         self.in_pred_qois) = _precompute_expected_deviation_data(
             self.out_quad_data, self.in_quad_data,
             self.obs_fun, self.qoi_fun)
        if (self.out_pred_obs.shape[1] !=
                self.ndesign_candidates*self.ndata_per_candidate):
            msg = "out_pred_obs.shape[1] != "
            msg += "self.ndesign_candidates*self.ndata_per_candidate. "
            msg += f"{self.out_pred_obs.shape[1]}"
            msg += f"!={self.ndesign_candidates}*{self.ndata_per_candidate} "
            msg += "check ndata_per_candidate.\nEach design candidate"
            msg += " must have the same number of data returned by obs_fun"
            raise ValueError(msg)

    def _sort_qoi(self):
        # Sort in_pred_qois and use this order to sort
        # in_samples so that cvar deviation does not have to
        # constantly sort samples
        if self.in_pred_qois.shape[2] != 1:
            raise ValueError("Sorting can only be used for a single QoI")
        return np.argsort(self.in_pred_qois, axis=1)

    def populate(self):
        """
        Compute the data needed to initialize the OED algorithm.
        """
        self._populate()
        # if self.in_pred_qois.shape[1] == 1:
        #     # speeds up calcualtion of avar
        #     self._sort_qoi()

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
        active_indices, noise_std = self._define_design_data(
            collected_design_indices, new_design_indices)
        return _compute_negative_expected_deviation_monte_carlo(
            self.out_pred_obs, self.in_pred_obs, self.in_weights,
            self.out_weights, self.in_pred_qois,
            self.deviation_fun, self.pred_risk_fun, self.data_risk_fun,
            self.noise_samples, noise_std, active_indices, return_all)

    def compute_utilities(self, ncandidates, collected_design_indices,
                          new_indices, return_all):
        if self.nprocs == 1:
            return self._compute_utilities_serial(
                self.compute_expected_utility,
                collected_design_indices, return_all, new_indices)
        return self._compute_utilities_parallel_shared(
            deviation_worker_fun, init_deviation_worker,
            _get_deviation_compute_utilities_initargs,
            new_indices, collected_design_indices, return_all)


class BayesianSequentialOED(AbstractBayesianOED):
    r"""
    Compute sequential optimal experimental designs that collect
    data and use this to inform the choice of subsequent design locations.
    """

    @abstractmethod
    def __init__(self, obs_process):
        if not callable(obs_process):
            raise ValueError("obs_process must be a callable function")
        self.obs_process = obs_process
        self.collected_obs = None
        self.inner_importance_weights = None
        self.outer_importance_weights = None
        self.in_weights_up = None
        self.out_weights_up = None
        self.evidence_from_prior = 1
        self.evidence = None

    def _loglike_fun(self, obs, pred_obs, noise_std):
        return gaussian_loglike_fun(obs, pred_obs, noise_std)

    def _compute_evidence(self):
        """
        Compute the evidence associated with using the true collected data.

        Notes
        -----
        This is a private function because calling by user will upset
        evidence calculation

        Always just use the first inner loop sample set to compute evidence.
        To avoid numerical precision problems recompute evidence with
        all data as opposed to updating evidence just using new data
        """
        # For now only allow one data per design location
        assert np.allclose(
            self.out_pred_obs.shape[1]/self.ndesign_candidates, 1.0,
            atol=1e-14)
        log_like_vals = self._loglike_fun(
            self.collected_obs,
            self.in_pred_obs[:, self.collected_design_indices],
            self.noise_std[self.collected_design_indices])

        # compute evidence moving from initial prior to current posterior
        evidence_from_prior = np.sum(
            np.exp(log_like_vals)[:, 0]*self.in_weights[:, 0])
        # compute evidence moving from previous posterior to current posterior
        self.evidence = evidence_from_prior/self.evidence_from_prior
        self.evidence_from_prior = evidence_from_prior

    def compute_importance_weights(self):
        """
        Compute the importance weights used in the computation of the expected
        utility that acccount for the fact we want to use the current posterior
        as the prior in the utility formula.
        """
        self.outer_importance_weights = np.exp(self._loglike_fun(
            self.collected_obs, self.out_pred_obs[
                :, self.collected_design_indices],
            self.noise_std[self.collected_design_indices]))/(
                self.evidence_from_prior)
        nobs = self.collected_design_indices.shape[0]

        tmp = self.in_pred_obs[:, self.collected_design_indices].reshape(
            1, self.nin_samples, nobs)

        self.inner_importance_weights = (np.exp(
             self._loglike_fun(self.collected_obs, tmp, self.noise_std[
                 self.collected_design_indices]))/(
                 self.evidence_from_prior)).T

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
        at in_samples
        self.in_weights is the prior quadrature weights which
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
        self._compute_evidence()
        self.compute_importance_weights()
        self.out_weights_up = \
            self.out_weights*self.outer_importance_weights
        self.in_weights_up = \
            self.in_weights*self.inner_importance_weights

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

    # def update_design(self, return_all=False, rounding_decimals=16):
    #     return super().update_design(return_all, rounding_decimals)


class BayesianSequentialKLOED(BayesianSequentialOED, BayesianBatchKLOED):
    r"""
    Compute closed-loop OED my maximizing KL divergence between the prior and
    posterior.
    """

    def __init__(self, ndesign_candidates, obs_fun, noise_std,
                 prior_variable, out_quad_opts, in_quad_opts,
                 obs_process=None, nprocs=1, max_ncollected_obs=2):
        r"""
        Constructor.

        Parameters
        ----------
        design_candidates : np.ndarray (nvars, nsamples)
            The location of all design sample candidates

        obs_fun : callable
            Function with the signature

            `obs_fun(samples) -> np.ndarray(nsamples, nqoi)`

            That returns noiseless evaluations of the forward model.

        noise_std : float or np.ndarray (nobs, 1)
            The standard deviation of the mean zero Gaussian noise added to
            each observation

        prior_variable : pya.IndependentMarginalsVariable
            The prior variable consisting of independent univariate random
            variables

        obs_process : callable
            The true data generation model with the signature

            `obs_process(design_indices) -> np.ndarray (1, ndesign_indices)`

            where design_samples is np.ndarary (nvars, ndesign_indices)

        generate_inner_prior_samples : callable
           Function with the signature

            `generate_inner_prior_samples(nsamples) -> np.ndarray(
             nvars, nsamples), np.ndarray(nsamples, 1)`

            Generate samples and associated weights used to evaluate
            the evidence computed by the inner loop
            If None then the function generate_outer_prior_samples is used and
            weights are assumed to be 1/nsamples. This function is useful if
            wanting to use multivariate quadrature to evaluate the evidence

        nin_samples : integer
            The number of quadrature samples used for the inner integral that
            computes the evidence for each realiaztion of the predicted
            observations

        nout_samples : integer
            The number of Monte Carlo samples used to compute the outer
            integral over all possible observations

        pre_collected_design_indices : np.ndarray (nobs)
            The indices into the qoi vector associated with the
            collected observations

        econ : boolean
            Make all inner loop samples the same for all outer loop samples.
            This reduces number of evaluations of prediction model. Currently
            this common data is copied and repeated for each outer loop sample
            so the rest of the code can remain the same. Eventually the data
            has to be tiled anyway when computing exepcted utility so this is
            not a big deal.

        nprocs : integer
            The number of threads used to compute OED design. Warning:
            this uses multiprocessing.Pool and seems to provide very little
            benefit and in many cases increases the CPU time.
        """
        # obs_process default is None so same API can be used as
        # open loop design
        BayesianBatchKLOED.__init__(
            self, ndesign_candidates, obs_fun, noise_std, prior_variable,
            out_quad_opts, in_quad_opts,
            nprocs=nprocs, max_ncollected_obs=max_ncollected_obs)
        BayesianSequentialOED.__init__(self, obs_process)

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
        return _compute_expected_kl_utility_monte_carlo(
            self.out_pred_obs, self.in_pred_obs, self.in_weights_up,
            self.out_weights_up, new_design_indices, self.noise_samples,
            self.noise_std, self.data_risk_fun, return_all)


class BayesianSequentialDeviationOED(
        BayesianSequentialOED, BayesianBatchDeviationOED):
    r"""
    Compute closed-loop OED by minimizing the deviation on the push forward
    of the posterior through a QoI model.
    """
    def __init__(self, ndesign_candidates, obs_fun, noise_std,
                 prior_variable,  out_quad_opts, in_quad_opts,
                 qoi_fun=None, obs_process=None,
                 deviation_fun=oed_standard_deviation,
                 pred_risk_fun=oed_prediction_average,
                 data_risk_fun=oed_data_expectation,
                 nprocs=1, max_ncollected_obs=2):
        r"""
        Constructor.

        Parameters
        ----------
        design_candidates : np.ndarray (nvars, nsamples)
            The location of all design sample candidates

        obs_fun : callable
            Function with the signature

            `obs_fun(samples) -> np.ndarray(nsamples, nqoi)`

            That returns noiseless evaluations of the forward model.

        noise_std : float or np.ndarray (nobs, 1)
            The standard deviation of the mean zero Gaussian noise added to
            each observation

        prior_variable : pya.IndependentMarginalsVariable
            The prior variable consisting of independent univariate random
            variables

        obs_process : callable
            The true data generation model with the signature

            `obs_process(design_indices) -> np.ndarray (1, ndesign_indices)`

            where design_samples is np.ndarary (nvars, ndesign_indices)

        qoi_fun : callable
            Function with the signature

            `qoi_fun(samples) -> np.ndarray(nsamples, nqoi)`

            That returns evaluations of the forward model. Observations are
            assumed to be :math:`f(z)+\epsilon` where :math:`\epsilon` is
            additive noise nsamples : np.ndarray (nvars, nsamples)

        generate_inner_prior_samples : callable
           Function with the signature

            `generate_inner_prior_samples(nsamples) -> np.ndarray(
             nvars, nsamples), np.ndarray(nsamples, 1)`

            Generate samples and associated weights used to evaluate
            the evidence computed by the inner loop
            If None then the function generate_outer_prior_samples is used and
            weights are assumed to be 1/nsamples. This function is useful if
            wanting to use multivariate quadrature to evaluate the evidence

        nin_samples : integer
            The number of quadrature samples used for the inner integral that
            computes the evidence for each realiaztion of the predicted
            observations

        nout_samples : integer
            The number of Monte Carlo samples used to compute the outer
            integral over all possible observations

        pre_collected_design_indices : np.ndarray (nobs)
            The indices into the qoi vector associated with the
            collected observations

        econ : boolean
            Make all inner loop samples the same for all outer loop samples.
            This reduces number of evaluations of prediction model. Currently
            this common data is copied and repeated for each outer loop sample
            so the rest of the code can remain the same. Eventually the data
            has to be tiled anyway when computing exepcted utility so this is
            not a big deal.

         deviation_fun : callable
             Function with the signature

            `deviation_fun(in_pred_qois, weights) ->
             np.ndarray(nout_samples, nqois)`

             where

             in_pred_qois : np.ndarray (
             nout_samples, nin_samples, nqois)
             weights : np.ndarray (nout_samples, nin_samples)

        nprocs : integer
            The number of threads used to compute OED design. Warning:
            this uses multiprocessing.Pool and seems to provide very little
            benefit and in many cases increases the CPU time.

        pred_risk_fun : callable
            Function to compute risk over multiple qoi with the signature

             `pred_risk_fun(expected_deviations) -> float`

            where expected_deviations : np.ndarray (nqois, 1)
        """
        # obs_process default is None so same API can be used as
        # open loop design
        BayesianBatchDeviationOED.__init__(
            self, ndesign_candidates, obs_fun, noise_std,
            prior_variable, out_quad_opts, in_quad_opts, qoi_fun,
            deviation_fun, pred_risk_fun, data_risk_fun,
            nprocs, max_ncollected_obs)
        BayesianSequentialOED.__init__(self, obs_process)

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
        return _compute_negative_expected_deviation_monte_carlo(
            self.out_pred_obs, self.in_pred_obs, self.in_weights_up,
            self.out_weights_up, self.in_pred_qois,
            self.deviation_fun, self.pred_risk_fun, self.data_risk_fun,
            self.noise_samples, self.noise_std, new_design_indices, return_all)


def get_oed_inner_quadrature_rule(nin_samples, prior_variable,
                                  quad_method='gauss'):
    """
    Parameters
    ----------
    quad_method : string
        The method used to compute the inner loop integral needed to
        evaluate the evidence for an outer loop sample. Options are
        ["linear", "quadratic", "gaussian", "monte_carlo"]
        The first 3 construct tensor product quadrature rules from
        univariate rules that are respectively piecewise linear,
        piecewise quadratic or Gauss-quadrature.

    """
    nrandom_vars = prior_variable.num_vars()
    nin_samples_1d = nin_samples
    if quad_method == "gauss":
        var_trans = AffineTransform(prior_variable)
        univariate_quad_rules = \
            get_univariate_quadrature_rules_from_variable(
                prior_variable, [nin_samples_1d]*nrandom_vars)[0]
        x_quad, w_quad = get_tensor_product_quadrature_rule(
            [nin_samples_1d]*nrandom_vars, nrandom_vars,
            univariate_quad_rules, transform_samples=None)
        return x_quad, w_quad[:, None]

    degree = {'linear': 1, 'quadratic': 2}[quad_method]
    if prior_variable.is_bounded_continuous_variable():
        alpha = 1
    else:
        alpha = 1-1e-6
    new_ranges = prior_variable.get_statistics(
        "interval", confidence=alpha).flatten()
    x_quad, w_quad = \
        get_tensor_product_piecewise_polynomial_quadrature_rule(
            nin_samples_1d, new_ranges, degree)
    w_quad *= prior_variable.pdf(x_quad)[:, 0]
    return x_quad, w_quad[:, None]


def get_posterior_weights_at_in_samples(oed, nn, out_idx):
    # plot posterior for one realization of the data
    # nn : number of data used to form posterior
    # out_idx : the outer loop iteration used to generate the data
    assert nn > 0
    active_indices = np.hstack([idx*oed.ndata_per_candidate + np.arange(
        oed.ndata_per_candidate) for idx in oed.collected_design_indices[:nn]])
    outer_log_likelihood_vals, evidences = _compute_evidences(
        oed.out_pred_obs[out_idx:out_idx+1],
        oed.in_pred_obs, oed.in_weights,
        oed.out_weights[out_idx:out_idx+1], active_indices,
        oed.noise_samples[out_idx:out_idx+1], oed.noise_std)
    inner_log_likelihood_vals = _loglike_fun_from_noiseless_obs(
        oed.out_pred_obs[out_idx:out_idx+1], oed.in_pred_obs,
        oed.noise_samples[out_idx:out_idx+1],
        oed.noise_std, active_indices)
    weights = np.exp(inner_log_likelihood_vals)*oed.in_weights/evidences
    return weights


def get_posterior_2d_interpolant_from_oed_data(
        oed, prior_variable, nn, out_idx, quad_method):
    # plot posterior for one realization of the data
    # nn : number of data used to form posterior
    # out_idx : the outer loop iteration used to generate the data
    assert prior_variable.num_vars() == 2
    weights = get_posterior_weights_at_in_samples(oed, nn, out_idx)
    vals = weights/oed.in_weights
    # multiply vals by prior.
    vals *= prior_variable.pdf(oed.in_samples)

    nin_samples = vals.shape[0]

    if quad_method == "gauss":
        # interpolate posterior vals onto equidistant mesh for plotting
        nvars = prior_variable.num_vars()
        abscissa_1d = []
        for dd in range(nvars):
            abscissa_1d.append(
                np.unique(
                    oed.in_samples[dd, :nin_samples]))
        fun = partial(tensor_product_barycentric_interpolation, abscissa_1d,
                      vals)
        return fun

    quad_methods = ['linear', 'quadratic', 'gauss']
    if quad_method != "linear" and quad_method != "quadratic":
        raise ValueError(f"quad_method must be in {quad_methods}")

    # if using piecewise polynomial quadrature interpolate between using
    # piecewise linear method
    from scipy.interpolate import griddata
    x_quad = oed.in_samples
    def fun(x): return griddata(x_quad.T, vals, x.T, method="linear")
    return fun


def plot_2d_posterior_from_oed_data(
        oed, prior_variable, nn, out_idx, method, ax=None,
        oed_results=None):

    from pyapprox.util.visualization import plt, get_meshgrid_function_data
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if prior_variable.is_bounded_continuous_variable():
        alpha = 1
    else:
        alpha = 0.99
    plot_limits = prior_variable.get_statistics(
        "interval", confidence=alpha).flatten()

    fun = get_posterior_2d_interpolant_from_oed_data(
        oed, prior_variable, nn, out_idx, method)
    X, Y, Z = get_meshgrid_function_data(fun, plot_limits, 100)
    p = ax.contourf(
        X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 21))
    plt.colorbar(p, ax=ax)


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


def generate_inner_prior_samples_fixed(x_quad, w_quad, nsamples):
    """
    Wrapper that can be used with functools.partial to create a
    function with the signature generate_inner_samples(nsamples)
    that always returns the same quadrature rule. This function
    will be called many times and creating a quadrature each time
    can be computationally expensive.
    """
    assert nsamples == x_quad.shape[1], (nsamples, x_quad.shape)
    return x_quad, w_quad


def get_deviation_fun(name, opts={}):
    """
    Get the deviation function used to compute the deviation of the
    posterior push-forward for a realization of the observational data

    Parameters
    ----------
    name : string
        Name of the deviation function.
        Must be one of ["std", "cvar", "entropic"]

    opts : dict
         Any options needed by the desired deviation function. cvar requires
         {"quantile", p} where 0<=p<1. No options are needed for the other
         deviation functions

    Returns
    -------
    deviation_fun : callable
        Function with the signature

        `deviation_fun(in_pred_qois, weights) ->
         np.ndarray(nout_samples, nqois)`

         where

         in_pred_qois : np.ndarray (nout_samples, nin_samples, nqois)
         weights : np.ndarray (nout_samples, nin_samples)
    """
    deviation_funs = {
        "std": oed_standard_deviation,
        "cvar": oed_conditional_value_at_risk_deviation,
        "entropic": oed_entropic_deviation}

    if name not in deviation_funs:
        msg = f"{name} not in {deviation_funs.keys()}"
        raise ValueError(msg)

    fun = partial(deviation_funs[name], **opts)
    return fun


def get_data_risk_fun(name, opts={}):
    """
    Get the risk function used to compute the risk of the deviation for all
    outerloop realizations of the observations

    Parameters
    ----------
    name : string
        Name of the deviation function.
        Must be one of ["std", "cvar", "entropic"]

    opts : dict
         Any options needed by the desired deviation function. cvar requires
         {"quantile", p} where 0<=p<1. No options are needed for the other
         deviation functions

    Returns
    -------
    deviation_fun : callable
        Function with the signature

        `deviation_fun(in_pred_qois, weights) ->
         np.ndarray(nout_samples, nqois)`

         where

         in_pred_qois : np.ndarray (nout_samples, nin_samples, nqois)
         weights : np.ndarray (nout_samples, nin_samples)
    """
    risk_funs = {
        "mean": oed_data_expectation,
        "cvar": oed_data_cvar}

    if name not in risk_funs:
        msg = f"{name} not in {risk_funs.keys()}"
        raise ValueError(msg)

    fun = partial(risk_funs[name], **opts)
    return fun


def get_pred_risk_fun(name, **kwargs):
    risk_funs = {
        "mean": oed_prediction_average,
        "cvar": conditional_value_at_risk,
        "entropic": entropic_risk_measure}

    if name not in risk_funs:
        msg = f"{name} not in {risk_funs.keys()}"
        raise ValueError(msg)

    fun = partial(risk_funs[name], **kwargs)
    return fun


def extract_independent_noise_cov(cov, indices):
    """
    When computing laplace approximations we need a covariance matrix that
    treats each observation independent even when indices are the same,
    that is we have two or more observations for the same observation matrix
    """
    nindices = len(indices)
    if np.unique(indices).shape[0] == nindices:
        return cov[np.ix_(indices, indices)]
    new_cov = np.diag(np.diag(cov)[indices])
    return new_cov


def sequential_oed_synthetic_observation_process(
        obs_fun, true_sample, noise_fun, new_design_indices):
    r"""
    Use obs_model to generate all observations then downselect. For true
    observation processes this defeats the purpose of experimental design
    In these cases a custom obs_model must takes design indices as an
    argument.

    Parameters
    ----------
    obs_fun : callable
        Function with the signature

        `obs_fun() -> np.ndarray(nsamples, nqoi)`

        That returns the synethic truth for all design candidates.

    true_sample : np.ndaray (nvars, 1)
        The true sample used to generate the synthetic truth

    new_design_indices : np.ndarray (nnew_obs)
        The indices into the qoi vector associated with new design locations
        under consideration

    noise_fun : callable
        Function with signature

        `noise_fun(values, new_design_indices) -> np.ndarray (values.shape[0], new_design_indices.shape)`

         that returns noise for the new observations. Here
         values : np.ndarray (1, nobs) and
         new_design_indices : np.ndarary (nindices) where nindices<=nobs
    """
    all_obs = obs_fun(true_sample)
    noise = noise_fun(all_obs, new_design_indices)
    obs = all_obs[:, new_design_indices]+noise
    return obs


def gaussian_noise_fun(noise_std, values, active_indices=None):
    """
    Generate gaussian possibly heteroscedastic random noise

    Parameters
    ----------
    noise_std : float or np.ndarray (nobs)
        The standard deviation of the noise at each observation

    values : np.ndarray (nsamples, nobs)
        The observations at variour realizations of the random parameters

    active_indices :np.ndarray (nindices)
        The indices of the active observations with nindices <= nobs

    Returns
    -------
    noise : np.ndarray (nsamples, nindices)
        The noise at the active observations nindices=nobs if
        active_indices is None
    """
    if type(noise_std) == np.ndarray:
        noise_std = noise_std.flatten()
    if active_indices is None:
        return np.random.normal(0, noise_std, (values.shape))
    shape = (values.shape[0], active_indices.shape[0])
    if type(noise_std) != np.ndarray:
        return np.random.normal(0, noise_std, shape)
    return np.random.normal(
        0, noise_std[active_indices], shape)


def get_bayesian_oed_optimizer(
        short_oed_type, ndesign_candidates, obs_fun, noise_std,
        prior_variable, out_quad_opts=None, in_quad_opts=None, nprocs=1,
        pre_collected_design_indices=None, **kwargs):
    r"""
    Initialize a Bayesian OED optimizer.

    Parameters
    ----------
    short_oed_type : string
        The type of experimental design strategy

    design_candidates : np.ndarray (nvars, nsamples)
        The location of all design sample candidates

    obs_fun : callable
        Function with the signature

        `obs_fun(samples) -> np.ndarray(nsamples, nqoi)`

        That returns noiseless evaluations of the forward model.

    noise_std : float or np.ndarray (nobs, 1)
        The standard deviation of the mean zero Gaussian noise added to each
        observation

    nin_samples : integer
        The number of quadrature samples used for the inner integral that
        computes the evidence for each realiaztion of the predicted
        observations

    nout_samples : integer
        The number of Monte Carlo samples used to compute the outer integral
        over all possible observations

    quad_method : string
        The method used to compute the inner loop integral needed to
        evaluate the evidence for an outer loop sample. Options are
        ["linear", "quadratic", "gaussian", "monte_carlo"]
        The first 3 construct tensor product quadrature rules from univariate
        rules that are respectively piecewise linear, piecewise quadratic
        or Gauss-quadrature.

    pre_collected_design_indices : np.ndarray (nobs)
        The indices into the qoi vector associated with the
        collected observations

    kwargs : kwargs
        Key word arguments specific to the OED type

    Returns
    -------
    oed : pyapprox.expdesign.AbstractBayesianOED
        Bayesian OED optimizer object
    """

    if "obs_process" in kwargs:
        oed_type = "closed_loop_" + short_oed_type
    else:
        oed_type = "open_loop_" + short_oed_type

    oed_types = {"open_loop_kl_params": BayesianBatchKLOED,
                 "closed_loop_kl_params": BayesianSequentialKLOED,
                 "open_loop_dev_pred": BayesianBatchDeviationOED,
                 "closed_loop_dev_pred": BayesianSequentialDeviationOED}

    if oed_type not in oed_types:
        msg = f"oed_type {short_oed_type} not supported."
        msg += "Select from [kl_params, dev_pred]"
        raise ValueError(msg)

    if (type(noise_std) == np.ndarray and
            noise_std.shape[0] != ndesign_candidates):
        msg = "noise_std must be specified for each design candiate"
        raise ValueError(msg)

    if out_quad_opts is None:
        out_quad_opts = {
            "method": "quasimontecarlo", "kwargs": {"nsamples": int(1e3)}}
    if in_quad_opts is None:
        in_quad_opts = {
            "method": "quasimontecarlo", "kwargs": {"nsamples": int(1e3)}}

    oed = oed_types[oed_type](
        ndesign_candidates, obs_fun, noise_std, prior_variable,
        out_quad_opts, in_quad_opts, nprocs=nprocs, **kwargs)
    oed.populate()
    if pre_collected_design_indices is not None:
        oed.set_collected_design_indices(np.asarray(pre_collected_design_indices))
    return oed
