import numpy as np
from itertools import combinations


def _restriction_matrix(ncols, subset):
    # TODO Consider replacing _restriction_matrix.T.dot(A) with
    # special indexing applied to A
    nsubset = len(subset)
    mat = np.zeros((nsubset, ncols))
    for ii in range(nsubset):
        mat[ii, subset[ii]] = 1.0
    return mat


def get_model_subsets(nmodels, max_subset_nmodels=None):
    """
    Parameters
    ----------
    nmodels : integer
        The number of models

    max_subset_nmodels : integer
        The maximum number of in a subset.
    """
    if max_subset_nmodels is None:
        max_subset_nmodels = nmodels
    assert max_subset_nmodels > 0
    assert max_subset_nmodels <= nmodels
    subsets = []
    model_indices = np.arange(nmodels)
    for nsubset_lfmodels in range(1, max_subset_nmodels+1):
        for subset_indices in combinations(
                model_indices, nsubset_lfmodels):
            idx = np.asarray(subset_indices).astype(int)
            subsets.append(idx)
    return subsets


def BLUE_evaluate_models(variable, models, nsamples_per_subset):
    nmodels = len(models)
    subsets = get_model_subsets(nmodels)
    values = []
    for ii, subset in enumerate(subsets):
        if nsamples_per_subset[ii] == 0:
            values.append([])
            continue
        subset_samples = variable.rvs(nsamples_per_subset[ii])
        # todo improve parallelization by using ModelEnsemble
        subset_values = [models[s](subset_samples) for s in subset]
        if np.any([v.shape[1] != 1 for v in subset_values]):
            msg = "values returned by models are incorrect shape"
            raise ValueError(msg)
        values_ii = np.full(
            (nsamples_per_subset[ii], nmodels), np.nan)
        values_ii[:, subset] = np.hstack(subset_values)
        values.append(values_ii)
    return values


def BLUE_Psi(Sigma, costs, reg_blue, nsamples_per_subset):
    nmodels = Sigma.shape[0]
    # get all model subsets
    subsets = get_model_subsets(nmodels)
    subset_costs = [costs[subset].sum() for subset in subsets]
    mat = np.identity(nmodels)*reg_blue
    submats = []
    for ii, subset in enumerate(subsets):
        R = _restriction_matrix(nmodels, subset)
        # TODO why pseudo inverse?
        submat = R.T.dot(np.linalg.pinv(
            Sigma[np.ix_(subset, subset)])).dot(R)/subset_costs[ii]
        submats.append(submat)
        mat += nsamples_per_subset[ii]*submat
    return mat, submats


def BLUE_RHS(Sigma, values):
    """
    Parameters
    ----------
    Sigma : np.ndarray (nmodels, nmodels)
        The covariance between all models (including high-fidelity)

    values : list[np.ndarray[shape=(nsubset_samples, nmodels)]]
        Evaluations of each model for each subset. If a model is not in
        a subset its values must be set to np.nan
    """
    nmodels = Sigma.shape[0]
    subsets = get_model_subsets(nmodels)
    rhs = np.zeros((nmodels))
    for ii, subset in enumerate(subsets):
        R = _restriction_matrix(nmodels, subset)
        print(values, len(values), subset)
        if len(values[ii]) == 0:
            continue
        if np.any(np.isfinite(
                np.delete(values[ii], subset, axis=1))):
            raise ValueError("Values not in subset must be set to np.nan")
        rhs += np.linalg.multi_dot((
            R.T, np.linalg.pinv(Sigma[np.ix_(subset, subset)]),
            (values[ii][:, subset].sum(axis=0)).T))
    return rhs[:, None]


def BLUE_bound_constraint(tol, nsamples_per_subset):
    return nsamples_per_subset-tol


def BLUE_bound_constraint_jac(nsamples_per_subset):
    return np.eye(nsamples_per_subset.shape[0])


def BLUE_cost_constraint(nsamples_per_subset):
    return 1-nsamples_per_subset.sum()


def BLUE_cost_constraint_jac(nsamples_per_subset):
    return -np.ones(nsamples_per_subset.shape[0])


def BLUE_variance(asketch, Sigma, costs, reg_blue, nsamples_per_subset,
                  return_grad=False, return_hess=False):
    """Compute variance of BLUE estimator using Equation 4.13 paper.
    We normalize costs so that the variance is for budget B_ept=1 with respect
    to nsamples_per_subset. This is done because ???
    """
    if return_hess and not return_grad:
        raise ValueError("return_grad must be True if return_hess is True")

    mat, submats = BLUE_Psi(Sigma, costs, reg_blue, nsamples_per_subset)
    assert asketch.ndim == 2 and asketch.shape[1] == 1
    mat_inv = np.linalg.pinv(mat)
    variance = asketch.T.dot(mat_inv).dot(asketch)[0, 0]

    if not return_grad:
        return variance

    aT_mat_inv = asketch.T.dot(mat_inv)
    grad = np.array(
        [-np.linalg.multi_dot((aT_mat_inv, smat, aT_mat_inv.T))[0, 0]
         for smat in submats])

    if not return_hess:
        print(variance, grad.shape)
        return variance, grad

    raise NotImplementedError()
    # nsubsets = len(subsets)
    # temp = np.linalg.multi_dot(
    #     (np.array(submats).reshape(nsubsets*nmodels, nmodels), mat_inv,
    #      asketch)).reshape(nsubsets, nmodels)
    # hess = 2*np.linalg.multi_dot((temp, mat_inv, temp.T))
    # return variance, grad, hess
