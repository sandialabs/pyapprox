import os
import numpy as np
from scipy.spatial.distance import cdist
from pyapprox.util.pya_numba import njit
from functools import partial
from multiprocessing import Pool
from abc import ABC, abstractmethod

from pyapprox.util.sys_utilities import trace_error_with_msg
from pyapprox.util.utilities import get_tensor_product_quadrature_rule
from pyapprox.variables.risk import conditional_value_at_risk
from pyapprox.variables.sampling import (
    generate_independent_random_samples
)
from pyapprox.variables.transforms import (
    AffineTransform
)
from pyapprox.surrogates.interp.tensorprod import (
    get_tensor_product_piecewise_polynomial_quadrature_rule
)
from pyapprox.surrogates.polychaos.gpc import (
    get_univariate_quadrature_rules_from_variable
)
from pyapprox.surrogates.interp.barycentric_interpolation import (
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
        obs, pred_obs, noise_stdev, active_indices):
    if type(noise_stdev) != np.ndarray:
        noise_stdev = np.ones((pred_obs.shape[-1], 1), dtype=float)*noise_stdev

    tmp1 = -1/(2*noise_stdev[:, 0]**2)
    tmp2 = 0.5*np.sum(np.log(-tmp1[active_indices]/np.pi))
    llike = sq_dists_numba_3d_XX_prereduced(
        obs, pred_obs, tmp1, tmp2, active_indices)
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
    if pred_obs.ndim == 3 and obs.ndim == 2 and obs.shape[0] != 1:
        return gaussian_loglike_fun_economial_3D(
            obs, pred_obs, noise_stdev, active_indices)
    elif obs.ndim == 2 and pred_obs.ndim == 2 and obs.shape != pred_obs.shape:
        return gaussian_loglike_fun_economial_2D(
            obs, pred_obs, noise_stdev, active_indices)
    else:
        return gaussian_loglike_fun_broadcast(
            obs, pred_obs, noise_stdev, active_indices)


def __compute_expected_kl_utility_monte_carlo(
        log_likelihood_fun, outer_loop_pred_obs,
        inner_loop_pred_obs, inner_loop_weights, outer_loop_weights,
        active_indices, return_all):

    nouter_loop_samples = outer_loop_pred_obs.shape[0]
    ninner_loop_samples = int(
        inner_loop_pred_obs.shape[0]//nouter_loop_samples)

    outer_log_likelihood_vals = log_likelihood_fun(
        outer_loop_pred_obs, outer_loop_pred_obs, active_indices)
    nobs = outer_loop_pred_obs.shape[1]
    tmp = inner_loop_pred_obs.reshape(
        nouter_loop_samples, ninner_loop_samples, nobs)
    inner_log_likelihood_vals = log_likelihood_fun(
        outer_loop_pred_obs, tmp, active_indices)

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
        return {"utility_val": utility_val}
    weights = np.exp(inner_log_likelihood_vals)*inner_loop_weights/evidences
    result = {"utility_val": utility_val, "evidences": evidences,
              "weights": weights}
    return result


def precompute_expected_kl_utility_data(
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

        That returns noiseless evaluations of the forward model.

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
    print(f"Running {outer_loop_prior_samples.shape[1]} model evaluations")
    outer_loop_pred_obs = obs_fun(outer_loop_prior_samples)

    if outer_loop_pred_obs.shape[0] != outer_loop_prior_samples.shape[1]:
        msg = "obs_fun is not returning an array with the correct shape"
        raise ValueError(msg)

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
        print(f"Running {inner_loop_prior_samples.shape[1]} model evaluations")
        inner_loop_pred_obs = obs_fun(inner_loop_prior_samples)
    else:
        print(f"Running {in_samples.shape[1]} model evaluations")
        shared_inner_loop_pred_obs = obs_fun(in_samples)
        inner_loop_pred_obs = np.tile(
            shared_inner_loop_pred_obs, (nouter_loop_samples, 1))

    return (outer_loop_pred_obs, inner_loop_pred_obs,
            inner_loop_weights, outer_loop_prior_samples,
            inner_loop_prior_samples)


def precompute_expected_deviation_data(
        generate_outer_prior_samples, nouter_loop_samples, obs_fun,
        noise_fun, qoi_fun, ninner_loop_samples,
        generate_inner_prior_samples=None, econ=False):
    (outer_loop_pred_obs, inner_loop_pred_obs,
     inner_loop_weights, outer_loop_prior_samples,
     inner_loop_prior_samples) = precompute_expected_kl_utility_data(
             generate_outer_prior_samples, nouter_loop_samples, obs_fun,
             noise_fun, ninner_loop_samples, generate_inner_prior_samples,
             econ)

    if not econ:
        inner_loop_pred_qois = qoi_fun(inner_loop_prior_samples)
        if inner_loop_pred_qois.shape[0] != inner_loop_prior_samples.shape[1]:
            msg = "qoi_fun is not returning an array with the correct shape. "
            msg += f"expected nrows to be {inner_loop_prior_samples.shape[1]}"
            msg += f" but got {inner_loop_pred_qois.shape[0]}"
            raise ValueError(msg)
    else:
        in_samples = inner_loop_prior_samples[:, :ninner_loop_samples]
        shared_inner_loop_pred_qois = qoi_fun(in_samples)
        if shared_inner_loop_pred_qois.shape[0] != in_samples.shape[1]:
            msg = "qoi_fun is not returning an array with the correct shape"
            raise ValueError(msg)
        inner_loop_pred_qois = np.tile(
            shared_inner_loop_pred_qois, (nouter_loop_samples, 1))

    nqois = inner_loop_pred_qois.shape[1]
    # print(nqois, [nouter_loop_samples, ninner_loop_samples, nqois],
    #       np.prod([nouter_loop_samples, ninner_loop_samples, nqois]))
    # tmp = np.empty((nouter_loop_samples, ninner_loop_samples, nqois))
    # for kk in range(nqois):
    #     tmp[:, :, kk] = inner_loop_pred_qois[:, kk].reshape(
    #         (nouter_loop_samples, ninner_loop_samples))
    # inner_loop_pred_qois = tmp
    inner_loop_pred_qois = inner_loop_pred_qois.reshape(
            (nouter_loop_samples, ninner_loop_samples, nqois))
    # assert np.allclose(tmp, inner_loop_pred_qois.reshape(
    #         (nouter_loop_samples, ninner_loop_samples, nqois)))
    return (outer_loop_pred_obs, inner_loop_pred_obs,
            inner_loop_weights, outer_loop_prior_samples,
            inner_loop_prior_samples, inner_loop_pred_qois)


def compute_expected_kl_utility_monte_carlo(
        log_likelihood_fun, outer_loop_pred_obs,
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
        log_likelihood_fun, outer_loop_pred_obs,
        inner_loop_pred_obs, inner_loop_weights, outer_loop_weights,
        active_indices, return_all)


def __compute_negative_expected_deviation_monte_carlo(
        log_likelihood_fun, outer_loop_pred_obs,
        inner_loop_pred_obs, inner_loop_weights, outer_loop_weights,
        inner_loop_pred_qois, deviation_fun, active_indices, pred_risk_fun,
        return_all, data_risk_fun):
    nouter_loop_samples = outer_loop_pred_obs.shape[0]
    ninner_loop_samples = int(
        inner_loop_pred_obs.shape[0]//nouter_loop_samples)

    nobs = outer_loop_pred_obs.shape[1]
    tmp = inner_loop_pred_obs.reshape(
        nouter_loop_samples, ninner_loop_samples, nobs)

    inner_log_likelihood_vals = log_likelihood_fun(
        outer_loop_pred_obs, tmp, active_indices)
    inner_likelihood_vals = np.exp(inner_log_likelihood_vals)
    evidences = np.einsum(
        "ij,ij->i", inner_likelihood_vals, inner_loop_weights)[:, None]

    weights = inner_likelihood_vals*inner_loop_weights/evidences

    # make deviation_fun operate on columns of samples
    # so that it returns a vector of deviations one for each column
    deviations = deviation_fun(inner_loop_pred_qois, weights)

    # expectation taken with respect to observations
    # assume always want deviation here, but this can be changed
    # expected_obs_deviations = np.sum(deviations*outer_loop_weights, axis=0)
    # use einsum because it does not create intermediate arrays
    # expected_obs_deviations = np.einsum(
    #    "ij,i->j", deviations, outer_loop_weights[:, 0])
    expected_obs_deviations = data_risk_fun(deviations, outer_loop_weights)

    disutility_val = pred_risk_fun(expected_obs_deviations)

    utility_val = -disutility_val
    if not return_all:
        return {'utility_val': utility_val}
    result = {
        'utility_val': utility_val, 'evidences': evidences,
        'weights': weights, 'deviations': deviations,
        'expected_deviations': expected_obs_deviations}
    return result


def compute_negative_expected_deviation_monte_carlo(
        log_likelihood_fun, outer_loop_pred_obs,
        inner_loop_pred_obs, inner_loop_weights, outer_loop_weights,
        inner_loop_pred_qois, deviation_fun, collected_design_indices,
        new_design_indices, pred_risk_fun, return_all, data_risk_fun):
    if collected_design_indices is not None:
        active_indices = np.hstack(
            (collected_design_indices, new_design_indices)).astype(int)
    else:
        # assume the observations at the collected_design_indices are already
        # incorporated into the inner and outer loop weights
        active_indices = np.asarray(new_design_indices, dtype=int)
    return __compute_negative_expected_deviation_monte_carlo(
        log_likelihood_fun, outer_loop_pred_obs,
        inner_loop_pred_obs, inner_loop_weights, outer_loop_weights,
        inner_loop_pred_qois, deviation_fun, active_indices, pred_risk_fun,
        return_all, data_risk_fun)


def select_design(design_candidates, collected_design_indices,
                  compute_expected_utility, max_eval_concurrency=1,
                  return_all=False, rounding_decimals=16):
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

    results : dict
        Dictionary of useful data used to compute expected utility
        At a minimum it has the keys ["utilties", "evidences", "weights"]
    """
    ncandidates = design_candidates.shape[1]

    if max_eval_concurrency > 1:
        pool = Pool(max_eval_concurrency)
        results = pool.map(
            partial(compute_expected_utility, collected_design_indices,
                    return_all=return_all),
            [np.array([ii]) for ii in range(ncandidates)])
        pool.close()
        utility_vals = np.array(
            [results[ii]['utility_val'] for ii in range(ncandidates)])
        # utility_vals = -np.ones(ncandidates)*np.inf
        # results = [None for ii in range(ncandidates)]
        # cnt = 0
        # for ii in range(ncandidates):
        #     if ii not in collected_design_indices:
        #         utility_vals[ii] = pool_results[cnt]['utility_val']
        #         results[ii] = pool_results[cnt]
        #         cnt += 1
    else:
        results = []
        utility_vals = -np.ones(ncandidates)*np.inf
        results = [None for ii in range(ncandidates)]
        for ii in range(ncandidates):
            results[ii] = compute_expected_utility(
                collected_design_indices, np.array([ii], dtype=int),
                return_all=return_all)
            utility_vals[ii] = results[ii]["utility_val"]
            # print(f'Candidate {ii}:', utility_vals[ii])

    selected_index = np.argmax(np.round(utility_vals, rounding_decimals))
    # print(np.round(utility_vals, rounding_decimals))

    if not return_all:
        results = None
    return utility_vals, selected_index, results


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


class AbstractBayesianOED(ABC):
    r"""Base Bayesian OED class"""

    def __init__(self, design_candidates, obs_fun, noise_std,
                 prior_variable, nouter_loop_samples=1000,
                 ninner_loop_samples=1000, generate_inner_prior_samples=None,
                 econ=False, max_eval_concurrency=1):
        """
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

        generate_inner_prior_samples : callable
           Function with the signature

            `generate_inner_prior_samples(nsamples) -> np.ndarray(
             nvars, nsamples), np.ndarray(nsamples, 1)`

            Generate samples and associated weights used to evaluate
            the evidence computed by the inner loop
            If None then the function generate_outer_prior_samples is used and
            weights are assumed to be 1/nsamples. This function is useful if
            wanting to use multivariate quadrature to evaluate the evidence

        ninner_loop_samples : integer
            The number of quadrature samples used for the inner integral that
            computes the evidence for each realiaztion of the predicted
            observations

        nouter_loop_samples : integer
            The number of Monte Carlo samples used to compute the outer
            integral over all possible observations

        quad_method : string
            The method used to compute the inner loop integral needed to
            evaluate the evidence for an outer loop sample. Options are
            ["linear", "quadratic", "gaussian", "monte_carlo"]
            The first 3 construct tensor product quadrature rules from
            univariate rules that are respectively piecewise linear,
            piecewise quadratic or Gauss-quadrature.

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

        max_eval_concurrency : integer
            The number of threads used to compute OED design. Warning:
            this uses multiprocessing.Pool and seems to provide very little
            benefit and in many cases increases the CPU time.
        """

        self.design_candidates = design_candidates
        if not callable(obs_fun):
            raise ValueError("obs_fun must be a callable function")
        self.obs_fun = obs_fun
        self.noise_std = noise_std
        self.prior_variable = prior_variable
        self.nouter_loop_samples = nouter_loop_samples
        self.ninner_loop_samples = ninner_loop_samples
        self.generate_inner_prior_samples = generate_inner_prior_samples
        self.econ = econ

        self.collected_design_indices = None
        self.outer_loop_pred_obs = None
        self.inner_loop_pred_obs = None
        self.inner_loop_weights = None
        self.outer_loop_prior_samples = None
        self.noise_realizations = None
        self.max_eval_concurrency = None

        self.set_max_eval_concurrency(max_eval_concurrency)

    def set_max_eval_concurrency(self, max_eval_concurrency):
        self.max_eval_concurrency = max_eval_concurrency
        if (max_eval_concurrency > 1 and (
                'OMP_NUM_THREADS' not in os.environ or
                not int(os.environ['OMP_NUM_THREADS']) == 1)):
            msg = 'User set assert_omp=True but OMP_NUM_THREADS has not been '
            msg += 'set to 1. Run script with '
            msg += 'OMP_NUM_THREADS=1 python script.py'
            raise Exception(msg)

    def noise_fun(self, values, active_indices=None):
        return gaussian_noise_fun(self.noise_std, values, active_indices)

    def reproducible_noise_fun(self, values, active_indices):
        active_indices = np.atleast_1d(active_indices)
        assert values.shape[0] == self.nouter_loop_samples
        nunique_indices = np.unique(
            active_indices, return_counts=True)[1].max()
        if self.noise_realizations is None:
            # make noise the same each time this funciton is called
            self.noise_realizations = self.noise_fun(values)[:, :, None]
        if self.noise_realizations.shape[2] < nunique_indices:
            self.noise_realizations = np.dstack(
                (self.noise_realizations, self.noise_fun(values)[:, :, None]))
        counts = np.zeros(values.shape[1], dtype=int)
        noise = np.empty((values.shape[0], active_indices.shape[0]))
        for ii in range(active_indices.shape[0]):
            idx = active_indices[ii]
            noise[:, ii] = self.noise_realizations[:, idx, counts[idx]]
            counts[idx] += 1
        return noise

    def generate_prior_samples(self, nsamples):
        return generate_independent_random_samples(
            self.prior_variable, nsamples), np.ones(nsamples)/nsamples

    def loglike_fun(self, obs, pred_obs, active_indices=None):
        return gaussian_loglike_fun(
            obs, pred_obs, self.noise_std, active_indices)

    def __get_outer_loop_obs(self, noiseless_obs, active_indices):
        noise = self.reproducible_noise_fun(noiseless_obs, active_indices)
        return noiseless_obs[:, active_indices] + noise

    def get_outer_loop_obs(self, active_indices):
        return self.__get_outer_loop_obs(
            self.outer_loop_pred_obs, active_indices)

    def loglike_fun_from_noiseless_obs(
            self, noiseless_obs, pred_obs, active_indices):
        # special indexing with active_indices causes a copy
        # this is ok for small 2D noiseless_obs array but is too expensive
        # for 3D pred_obs array so use loglike function that takes reduced
        # 2D array and full 3D array and applied active_indices to 3D array
        # internally. This wierdness is causd by the fact that when there
        # are repeat entries in active_indices we need to generate different
        # noise realization for the same noiseless obs. This is not possible
        # with special indexing as it will just copy the same value twice.
        obs = self.__get_outer_loop_obs(noiseless_obs, active_indices)
        if pred_obs.ndim == 3:
            return gaussian_loglike_fun_3d_prereduced(
                obs, pred_obs, self.noise_std, active_indices)
        return self.loglike_fun(obs, pred_obs[:, active_indices])

    @abstractmethod
    def populate(self):
        pass

    def update_design(self, return_all=False, rounding_decimals=16):
        if not hasattr(self, "outer_loop_pred_obs"):
            raise ValueError("Must call self.populate before creating designs")
        if self.collected_design_indices is None:
            self.collected_design_indices = np.zeros((0), dtype=int)
        utility_vals, selected_index, results = select_design(
            self.design_candidates, self.collected_design_indices,
            self.compute_expected_utility, self.max_eval_concurrency,
            return_all, rounding_decimals)

        new_design_indices = np.array([selected_index], dtype=int)
        self.collected_design_indices = np.hstack(
            (self.collected_design_indices, new_design_indices)).astype(int)
        if return_all is False:
            return utility_vals, new_design_indices, None
        return utility_vals, new_design_indices, results

    def set_collected_design_indices(self, indices):
        self.collected_design_indices = indices.copy()

    @abstractmethod
    def compute_expected_utility(self, collected_design_indices,
                                 new_design_indices, return_all=False):
        raise NotImplementedError()


class BayesianBatchKLOED(AbstractBayesianOED):
    r"""
    Compute open-loop OED my maximizing KL divergence between the prior and
    posterior.
    """

    def populate(self):
        (self.outer_loop_pred_obs,
         self.inner_loop_pred_obs, self.inner_loop_weights,
         self.outer_loop_prior_samples, self.inner_loop_prior_samples) = \
             precompute_expected_kl_utility_data(
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
            self.loglike_fun_from_noiseless_obs, self.outer_loop_pred_obs,
            self.inner_loop_pred_obs, self.inner_loop_weights,
            self.outer_loop_weights, collected_design_indices,
            new_design_indices, return_all)


def oed_prediction_average(qoi_vals, weights=None):
    assert qoi_vals.ndim == 2 and qoi_vals.shape[1] == 1
    if weights is None:
        return qoi_vals.mean()

    assert weights.shape[1] == 1
    return np.sum(qoi_vals*weights, axis=0)


def oed_variance_deviation(samples, weights):
    """
    Compute the variance deviation for each outer loop sample using the
    corresponding inner loop samples

    Parameters
    ----------
    samples : np.ndarray (nouter_loop_samples, ninner_loop_samples, nqois)
         The samples

    weights : np.ndarray (nouter_loop_samples, ninner_loop_samples)
        Weights associated with each innner loop sample

    Returns
    -------
    deviation_vals : np.ndarray (nouter_loop_samples, nqois)
        The deviation vals
    """
    # For large arrays variance_3D_pyx is the same speed as einsum
    # implementation below
    # try:
    #     from pyapprox.cython.utilities import variance_3D_pyx
    #     return variance_3D_pyx(samples, weights)
    # except:
    #     pass
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
    samples : np.ndarray (nouter_loop_samples, ninner_loop_samples, nqois)
         The samples

    weights : np.ndarray (nouter_loop_samples, ninner_loop_samples)
        Weights associated with each innner loop sample

    Returns
    -------
    deviation_vals : np.ndarray (nouter_loop_samples, nqois)
        The deviation vals
    """
    means = np.einsum(
        "ijk,ij->ik", samples, weights)
    risks = np.log(np.einsum(
        "ijk,ij->ik", np.exp(samples), weights))
    return risks-means


def oed_data_expectation(deviations, weights):
    """
    Compute the expected deviation for each outer loop sample

    Parameters
    ----------
    deviations : np.ndarray (nouter_loop_samples, nqois)
         The samples

    weights : np.ndarray (nouter_loop_samples, 1)
        Weights associated with each innner loop sample

    Returns
    -------
    expected_obs_deviations : np.ndarray (nqois, 1)
        The deviation vals
    """
    expected_obs_deviations = np.einsum(
        "ij,i->j", deviations, weights[:, 0])[:, None]
    return expected_obs_deviations


def oed_data_cvar(deviations, weights, quantile=None):
    """
    Compute the conditional value of risk of the deviations
    for each outer loop sample

    Parameters
    ----------
    deviations : np.ndarray (nouter_loop_samples, nqois)
         The samples

    weights : np.ndarray (nouter_loop_samples, 1)
        Weights associated with each innner loop sample

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
    samples : np.ndarray (nouter_loop_samples, ninner_loop_samples, nqois)
         The samples

    weights : np.ndarray (nouter_loop_samples, ninner_loop_samples)
        Weights associated with each innner loop sample

    Returns
    -------
    deviation_vals : np.ndarray (nouter_loop_samples, nqois)
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
    samples : np.ndarray (nouter_loop_samples, ninner_loop_samples, nqois)
         The samples

    weights : np.ndarray (nouter_loop_samples, ninner_loop_samples)
        Weights associated with each innner loop sample

    quantile : float
        The quantile of the conditional value at risk used to
        compute the deviation

    Returns
    -------
    deviation_vals : np.ndarray (nouter_loop_samples, nqois)
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

    def __init__(self, design_candidates, obs_fun, noise_std,
                 prior_variable, qoi_fun=None, nouter_loop_samples=1000,
                 ninner_loop_samples=1000, generate_inner_prior_samples=None,
                 econ=False, deviation_fun=oed_standard_deviation,
                 max_eval_concurrency=1,
                 pred_risk_fun=oed_prediction_average,
                 data_risk_fun=oed_data_expectation):
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

        ninner_loop_samples : integer
            The number of quadrature samples used for the inner integral that
            computes the evidence for each realiaztion of the predicted
            observations

        nouter_loop_samples : integer
            The number of Monte Carlo samples used to compute the outer
            integral over all possible observations

        quad_method : string
            The method used to compute the inner loop integral needed to
            evaluate the evidence for an outer loop sample. Options are
            ["linear", "quadratic", "gaussian", "monte_carlo"]
            The first 3 construct tensor product quadrature rules from
            univariate rules that are respectively piecewise linear,
            piecewise quadratic or Gauss-quadrature.

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

            `deviation_fun(inner_loop_pred_qois, weights) ->
             np.ndarray(nouter_loop_samples, nqois)`

             where

             inner_loop_pred_qois : np.ndarray (
             nouter_loop_samples, ninner_loop_samples, nqois)
             weights : np.ndarray (nouter_loop_samples, ninner_loop_samples)

        max_eval_concurrency : integer
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

            where deviations : np.ndarray (nouter_loop_samples, nqois)
        """

        super().__init__(design_candidates, obs_fun, noise_std,
                         prior_variable, nouter_loop_samples,
                         ninner_loop_samples, generate_inner_prior_samples,
                         econ=econ, max_eval_concurrency=max_eval_concurrency)
        # qoi fun deafult is None so that same api can be used for KL based OED
        # which does not require qoi_fun
        if not callable(qoi_fun):
            raise ValueError("qoi_fun must be a callable function")
        if not callable(deviation_fun):
            raise ValueError("deviation_fun must be a callable function")
        self.qoi_fun = qoi_fun
        self.deviation_fun = deviation_fun
        self.pred_risk_fun = pred_risk_fun
        self.data_risk_fun = data_risk_fun

    def __populate(self):
        """
        Compute the data needed to initialize the OED algorithm.
        """
        print("nouter_loop_samples * ninner_loop_samples: ",
              self.ninner_loop_samples*self.nouter_loop_samples)
        (self.outer_loop_pred_obs,
         self.inner_loop_pred_obs, self.inner_loop_weights,
         self.outer_loop_prior_samples, self.inner_loop_prior_samples,
         self.inner_loop_pred_qois
         ) = precompute_expected_deviation_data(
             self.generate_prior_samples, self.nouter_loop_samples,
             self.obs_fun, self.noise_fun, self.qoi_fun,
             self.ninner_loop_samples,
             generate_inner_prior_samples=self.generate_inner_prior_samples,
             econ=self.econ)

        self.outer_loop_weights = np.ones(
            (self.inner_loop_weights.shape[0], 1)) / \
            self.inner_loop_weights.shape[0]

    def __sort_qoi(self):
        # Sort inner_loop_pred_qois and use this order to sort
        # inner_loop_prior_samples so that cvar deviation does not have to
        # constantly sort samples
        if self.inner_loop_pred_qois.shape[2] != 1:
            raise ValueError("Sorting can only be used for a single QoI")
        idx1 = 0
        for ii in range(self.nouter_loop_samples):
            idx2 = idx1 + self.ninner_loop_samples
            qoi_indices = np.argsort(self.inner_loop_pred_qois[ii, :, 0])
            self.inner_loop_pred_qois[ii] = \
                self.inner_loop_pred_qois[ii, qoi_indices]
            self.inner_loop_prior_samples[:, idx1:idx2] = \
                self.inner_loop_prior_samples[:, idx1:idx2][:, qoi_indices]
            self.inner_loop_pred_obs[idx1:idx2] = \
                self.inner_loop_pred_obs[idx1:idx2][qoi_indices, :]
            self.inner_loop_weights[ii] = \
                self.inner_loop_weights[ii, qoi_indices]
            idx1 = idx2

    def populate(self):
        """
        Compute the data needed to initialize the OED algorithm.
        """
        self.__populate()
        if self.inner_loop_pred_qois.shape[2] == 1:
            # speeds up calcualtion of avar
            self.__sort_qoi()

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
            self.loglike_fun_from_noiseless_obs, self.outer_loop_pred_obs,
            self.inner_loop_pred_obs, self.inner_loop_weights,
            self.outer_loop_weights, self.inner_loop_pred_qois,
            self.deviation_fun, collected_design_indices, new_design_indices,
            self.pred_risk_fun, return_all, self.data_risk_fun)


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
        To avoid numerical precision problems recompute evidence with
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
            self.collected_obs, tmp))/self.evidence_from_prior

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

    # def update_design(self, return_all=False, rounding_decimals=16):
    #     return super().update_design(return_all, rounding_decimals)


class BayesianSequentialKLOED(BayesianSequentialOED, BayesianBatchKLOED):
    r"""
    Compute closed-loop OED my maximizing KL divergence between the prior and
    posterior.
    """

    def __init__(self, design_candidates, obs_fun, noise_std,
                 prior_variable, obs_process=None, nouter_loop_samples=1000,
                 ninner_loop_samples=1000, generate_inner_prior_samples=None,
                 econ=False, max_eval_concurrency=1):
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

        ninner_loop_samples : integer
            The number of quadrature samples used for the inner integral that
            computes the evidence for each realiaztion of the predicted
            observations

        nouter_loop_samples : integer
            The number of Monte Carlo samples used to compute the outer
            integral over all possible observations

        quad_method : string
            The method used to compute the inner loop integral needed to
            evaluate the evidence for an outer loop sample. Options are
            ["linear", "quadratic", "gaussian", "monte_carlo"]
            The first 3 construct tensor product quadrature rules from
            univariate rules that are respectively piecewise linear,
            piecewise quadratic or Gauss-quadrature.

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

        max_eval_concurrency : integer
            The number of threads used to compute OED design. Warning:
            this uses multiprocessing.Pool and seems to provide very little
            benefit and in many cases increases the CPU time.
        """
        # obs_process default is None so same API can be used as
        # open loop design
        BayesianBatchKLOED.__init__(
            self, design_candidates, obs_fun, noise_std, prior_variable,
            nouter_loop_samples, ninner_loop_samples,
            generate_inner_prior_samples, econ, max_eval_concurrency)
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
        return compute_expected_kl_utility_monte_carlo(
            self.loglike_fun_from_noiseless_obs, self.outer_loop_pred_obs,
            self.inner_loop_pred_obs, self.inner_loop_weights_up,
            self.outer_loop_weights_up, None, new_design_indices, return_all)


class BayesianSequentialDeviationOED(
        BayesianSequentialOED, BayesianBatchDeviationOED):
    r"""
    Compute closed-loop OED by minimizing the deviation on the push forward
    of the posterior through a QoI model.
    """
    def __init__(self, design_candidates, obs_fun, noise_std,
                 prior_variable, qoi_fun=None, obs_process=None,
                 nouter_loop_samples=1000, ninner_loop_samples=1000,
                 generate_inner_prior_samples=None, econ=False,
                 deviation_fun=oed_standard_deviation,
                 max_eval_concurrency=1,
                 pred_risk_fun=oed_prediction_average,
                 data_risk_fun=oed_data_expectation):
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

        ninner_loop_samples : integer
            The number of quadrature samples used for the inner integral that
            computes the evidence for each realiaztion of the predicted
            observations

        nouter_loop_samples : integer
            The number of Monte Carlo samples used to compute the outer
            integral over all possible observations

        quad_method : string
            The method used to compute the inner loop integral needed to
            evaluate the evidence for an outer loop sample. Options are
            ["linear", "quadratic", "gaussian", "monte_carlo"]
            The first 3 construct tensor product quadrature rules from
            univariate rules that are respectively piecewise linear,
            piecewise quadratic or Gauss-quadrature.

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

            `deviation_fun(inner_loop_pred_qois, weights) ->
             np.ndarray(nouter_loop_samples, nqois)`

             where

             inner_loop_pred_qois : np.ndarray (
             nouter_loop_samples, ninner_loop_samples, nqois)
             weights : np.ndarray (nouter_loop_samples, ninner_loop_samples)

        max_eval_concurrency : integer
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
            self, design_candidates, obs_fun, noise_std,
            prior_variable, qoi_fun, nouter_loop_samples,
            ninner_loop_samples, generate_inner_prior_samples,
            econ, deviation_fun, max_eval_concurrency, pred_risk_fun, data_risk_fun)
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
        # TODO pass in these weights so do not have to do so much
        # multiplications
        return compute_negative_expected_deviation_monte_carlo(
            self.loglike_fun_from_noiseless_obs, self.outer_loop_pred_obs,
            self.inner_loop_pred_obs, self.inner_loop_weights_up,
            self.outer_loop_weights_up, self.inner_loop_pred_qois,
            self.deviation_fun, None, new_design_indices,
            self.pred_risk_fun, return_all, self.data_risk_fun)


def get_oed_inner_quadrature_rule(ninner_loop_samples, prior_variable,
                                  quad_method='gauss'):
    nrandom_vars = prior_variable.num_vars()
    ninner_loop_samples_1d = ninner_loop_samples
    if quad_method == "gauss":
        var_trans = AffineTransform(prior_variable)
        univariate_quad_rules = \
            get_univariate_quadrature_rules_from_variable(
                prior_variable, [ninner_loop_samples_1d]*nrandom_vars)[0]
        x_quad, w_quad = get_tensor_product_quadrature_rule(
            [ninner_loop_samples_1d]*nrandom_vars, nrandom_vars,
            univariate_quad_rules, transform_samples=None)
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
    assert nn > 0
    results = oed.compute_expected_utility(
        oed.collected_design_indices[:nn-1],
        oed.collected_design_indices[nn-1:nn], True)
    weights = results["weights"]
    return get_posterior_vals_at_inner_loop_samples_base(
        oed, prior_variable, outer_loop_idx, weights)


def get_posterior_vals_at_inner_loop_samples_from_oed_results(
        oed, prior_variable, nn, outer_loop_idx, oed_results):
    """
    oed_results : list(list(dict))
         axis 0: each experimental design step
         axis 1: each design candidate
         dict: the data structures returned by the compute expected utility
              function used. Assumes that weights is returned as the
              is a key, i.e. index [ii][jj]["weights"] exists
    """
    assert nn > 0
    weights = oed_results[nn-1][oed.collected_design_indices[nn-1]]["weights"]
    return get_posterior_vals_at_inner_loop_samples_base(
        oed, prior_variable, outer_loop_idx, weights)


def get_posterior_vals_at_inner_loop_samples_base(
        oed, prior_variable, outer_loop_idx, weights):
    ninner_loop_samples = weights.shape[1]
    vals = (
        weights[outer_loop_idx, :] /
        oed.inner_loop_weights[outer_loop_idx, :])[:, None]
    # multiply vals by prior.
    vals *= prior_variable.pdf(
        oed.inner_loop_prior_samples[:, :ninner_loop_samples])
    return vals


def get_posterior_2d_interpolant_from_oed_data(
        oed, prior_variable, nn, outer_loop_idx, quad_method,
        oed_results=None):
    # plot posterior for one realization of the data
    # nn : number of data used to form posterior
    # outer_loop_idx : the outer loop iteration used to generate the data
    assert prior_variable.num_vars() == 2
    if oed_results is None:
        vals = get_posterior_vals_at_inner_loop_samples(
            oed, prior_variable, nn, outer_loop_idx)
    else:
        vals = get_posterior_vals_at_inner_loop_samples_from_oed_results(
            oed, prior_variable, nn, outer_loop_idx, oed_results)
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
                      vals)
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
        oed, prior_variable, nn, outer_loop_idx, method, ax=None,
        oed_results=None):

    from pyapprox import plt, get_meshgrid_function_data
    if ax is None:
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))

    if prior_variable.is_bounded_continuous_variable():
        alpha = 1
    else:
        alpha = 0.99
    plot_limits = prior_variable.get_statistics(
        "interval", alpha=alpha).flatten()

    fun = get_posterior_2d_interpolant_from_oed_data(
        oed, prior_variable, nn, outer_loop_idx, method, oed_results)
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


def run_bayesian_batch_deviation_oed_deprecated(
        prior_variable, obs_fun, qoi_fun, noise_std,
        design_candidates, pre_collected_design_indices, deviation_fun,
        pred_risk_fun, nexperiments, nouter_loop_samples, ninner_loop_samples,
        quad_method, max_eval_concurrency=1, return_all=False,
        rounding_decimals=16):
    r"""
    Deprecated. Use get_bayesian_oed_optimizer()

    Parameters
    ----------
    prior_variable : pya.IndependentMarginalsVariable
        The prior variable consisting of independent univariate random
        variables

    obs_fun : callable
        Function with the signature

        `obs_fun(samples) -> np.ndarray(nsamples, nobs)`

        which returns evaluations of the noiseless observation model.

    qoi_fun : callable
        Function with the signature

        `qoi_fun(samples) -> np.ndarray(nsamples, nqoi)`

        which returns evaluations of the prediction model.

    noise_std : float or np.ndarray (nobs, 1)
        The standard deviation of the mean zero Gaussian noise added to each
        observation

    design_candidates : np.ndarray (ndesign_vars, nsamples)
        The location of all design sample candidates

    pre_collected_indices : array_like
        The indices of the experiments that must be in the final design

    deviation_fun : callable
        Function with the signature

        `deviation_fun(inner_loop_pred_qois, weights) ->
         np.ndarray(nouter_loop_samples, nqois)`

         where

         inner_loop_pred_qois : np.ndarray (nouter_loop_samples, ninner_loop_samples, nqois)
         weights : np.ndarray (nouter_loop_samples, ninner_loop_samples)

    pred_risk_fun : callable
        Function to compute risk over multiple qoi with the signature

         `pred_risk_fun(expected_deviations) -> float`

        where expected_deviations : np.ndarray (nqois, 1)

    nexperiments : integer
        The number of experiments to be collected

    nouter_loop_samples : integer
        The number of Monte Carlo samples used to compute the outer integral
        over all possible observations

    ninner_loop_samples : integer
        The number of quadrature samples used for the inner integral that
        computes the evidence for each realiaztion of the predicted
        observations. If quad_method is a tensor product rule
        then this parameter actually specifies the number of points in each
        univariate rule so the total number of inner loop samples is
        ninner_loop_samples**nvars

    quad_method : string
        The method used to compute the inner loop integral needed to
        evaluate the evidence for an outer loop sample. Options are
        ["linear", "quadratic", "gaussian", "monte_carlo"]
        The first 3 construct tensor product quadrature rules from univariate
        rules that are respectively piecewise linear, piecewise quadratic
        or Gauss-quadrature.

    max_eval_concurrency : integer
        The number of threads used to compute OED design. Warning:
        this uses multiprocessing.Pool and seems to provide very little benefit
        and in many cases increases the CPU time.

    return_all : boolean
        Return intermediate quantities used to compute experimental design.
        This is primarily intended for testing purposes

    rounding_decimals : integer
        The number of decimal places to round utility_vals to when choosing
        the optimal design. This can be useful when comparing with
        numerical solutions where 2 designs are equivalent analytically
        but numerically there are slight differences that causes design to be
        different

    Returns
    -------
    oed : BayesianBatchDeviationOED
        OED object

    oed_results : list
        Contains the intermediate quantities used to compute experimental
        design at each design iteration. If not return_all then it is a list of
        None. If return_all then for each iteration entry is another list
        containing intermediate quantities for each design candidate.

    """

    # Define OED options
    if quad_method != "monte_carlo":
        x_quad, w_quad = get_oed_inner_quadrature_rule(
            ninner_loop_samples, prior_variable, quad_method)
        ninner_loop_samples = x_quad.shape[1]
        generate_inner_prior_samples = partial(
            generate_inner_prior_samples_fixed, x_quad, w_quad)
        econ = True
    else:
        # use default Monte Carlo sampling
        generate_inner_prior_samples = None
        econ = False

    # Setup OED problem
    oed = BayesianBatchDeviationOED(
        design_candidates, obs_fun, noise_std, prior_variable,
        qoi_fun, nouter_loop_samples, ninner_loop_samples,
        generate_inner_prior_samples=generate_inner_prior_samples,
        econ=econ, deviation_fun=deviation_fun, pred_risk_fun=pred_risk_fun,
        max_eval_concurrency=max_eval_concurrency)
    oed.populate()
    if pre_collected_design_indices is not None:
        oed.set_collected_design_indices(pre_collected_design_indices)

    # Generate experimental design
    if pre_collected_design_indices is None:
        npre_collected_design_indices = 0
    else:
        npre_collected_design_indices = len(pre_collected_design_indices)

    results = []
    for step in range(npre_collected_design_indices, nexperiments):
        results_step = oed.update_design(return_all, rounding_decimals)[2]
        results.append(results_step)

    return oed, results


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

        `deviation_fun(inner_loop_pred_qois, weights) ->
         np.ndarray(nouter_loop_samples, nqois)`

         where

         inner_loop_pred_qois : np.ndarray (nouter_loop_samples, ninner_loop_samples, nqois)
         weights : np.ndarray (nouter_loop_samples, ninner_loop_samples)
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

        `deviation_fun(inner_loop_pred_qois, weights) ->
         np.ndarray(nouter_loop_samples, nqois)`

         where

         inner_loop_pred_qois : np.ndarray (nouter_loop_samples, ninner_loop_samples, nqois)
         weights : np.ndarray (nouter_loop_samples, ninner_loop_samples)
    """
    risk_funs = {
        "mean": oed_data_expectation,
        "cvar": oed_data_cvar}

    if name not in risk_funs:
        msg = f"{name} not in {risk_funs.keys()}"
        raise ValueError(msg)

    fun = partial(risk_funs[name], **opts)
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
        short_oed_type, design_candidates, obs_fun, noise_std,
        prior_variable, nouter_loop_samples,
        ninner_loop_samples, quad_method,
        pre_collected_design_indices=None,
        **kwargs):
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

    ninner_loop_samples : integer
        The number of quadrature samples used for the inner integral that
        computes the evidence for each realiaztion of the predicted
        observations

    nouter_loop_samples : integer
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
            noise_std.shape[0] != design_candidates.shape[1]):
        msg = "noise_std must be specified for each design candiate"
        raise ValueError(msg)

    if quad_method in ["gauss", "linear", "quadratic"]:
        x_quad, w_quad = get_oed_inner_quadrature_rule(
            ninner_loop_samples, prior_variable, quad_method)
        ninner_loop_samples = x_quad.shape[1]
        generate_inner_prior_samples = partial(
            generate_inner_prior_samples_fixed, x_quad, w_quad)
    elif quad_method == "monte_carlo":
        # use default Monte Carlo sampling
        generate_inner_prior_samples = None
    else:
        raise ValueError(f"Incorrect quad_method {quad_method} specified")
    econ = True

    oed = oed_types[oed_type](
        design_candidates, obs_fun, noise_std, prior_variable,
        nouter_loop_samples=nouter_loop_samples,
        ninner_loop_samples=ninner_loop_samples, econ=econ,
        generate_inner_prior_samples=generate_inner_prior_samples,
        max_eval_concurrency=1, **kwargs)
    oed.populate()
    if pre_collected_design_indices is not None:
        oed.set_collected_design_indices(pre_collected_design_indices)
    return oed
