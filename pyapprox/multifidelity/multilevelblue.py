import numpy as np
from itertools import combinations
from functools import partial
from scipy.optimize import minimize


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


def BLUE_evaluate_models(
        rvs, models, subsets, nsamples_per_subset, pilot_values=None):
    nmodels = len(models)
    values = []
    if pilot_values is not None:
        npilot_samples = pilot_values.shape[0]
    else:
        npilot_samples = 0
    npilot_samples_used = 0
    for ii, subset in enumerate(subsets):
        if nsamples_per_subset[ii] == 0:
            values.append([])
            continue
        nnew_pilot_samples_to_use = max(0, min(
            npilot_samples-npilot_samples_used, nsamples_per_subset[ii]))
        nremaining_subset_samples = (
            nsamples_per_subset[ii]-nnew_pilot_samples_to_use)
        subset_samples = rvs(nremaining_subset_samples)
        remaining_subset_values = [models[s](subset_samples) for s in subset]
        if np.any([v.shape[1] != 1 for v in remaining_subset_values]):
            msg = "values returned by models are incorrect shape"
            raise ValueError(msg)
        remaining_subset_values = np.hstack(remaining_subset_values)
        if nnew_pilot_samples_to_use > 0:
            idx1 = npilot_samples_used
            idx2 = idx1+nnew_pilot_samples_to_use
            npilot_samples_used += nnew_pilot_samples_to_use
            subset_values = np.vstack((
                pilot_values[:, idx1:idx2], remaining_subset_values))
        else:
            subset_values = remaining_subset_values
        # todo improve parallelization by using ModelEnsemble
        values_ii = np.full(
            (nsamples_per_subset[ii], nmodels), np.nan)
        values_ii[:, subset] = subset_values
        values.append(values_ii)
    return values


def BLUE_Psi(Sigma, costs, reg_blue, subsets, nsamples_per_subset):
    nmodels = Sigma.shape[0]
    # get all model subsets
    mat = np.identity(nmodels)*reg_blue
    submats = []
    for ii, subset in enumerate(subsets):
        R = _restriction_matrix(nmodels, subset)
        submat = np.linalg.multi_dot((
            R.T,
            np.linalg.pinv(Sigma[np.ix_(subset, subset)]),
            R))
        submats.append(submat)
        mat += submat*nsamples_per_subset[ii]
    return mat, submats


def BLUE_betas(Sigma, asketch, reg_blue, subsets, nsamples_per_subset):
    nmodels = Sigma.shape[0]
    Psi = BLUE_Psi(Sigma, None, reg_blue, subsets, nsamples_per_subset)[0]
    Psi_inv = np.linalg.pinv(Psi)
    betas = np.empty((len(subsets), nmodels))
    for ii, subset in enumerate(subsets):
        Sigma_inv = np.linalg.pinv(Sigma[np.ix_(subset, subset)])
        R = _restriction_matrix(nmodels, subset)
        betas[ii] = np.linalg.multi_dot(
            (R.T, Sigma_inv, R, Psi_inv,
             asketch))[:, 0]*nsamples_per_subset[ii]
    return betas


def BLUE_RHS(subsets, Sigma, values):
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
    rhs = np.zeros((nmodels))
    for ii, subset in enumerate(subsets):
        R = _restriction_matrix(nmodels, subset)
        if len(values[ii]) == 0:
            continue
        if np.any(np.isfinite(
                np.delete(values[ii], subset, axis=1))):
            raise ValueError("Values not in subset must be set to np.nan")
        rhs += np.linalg.multi_dot((
            R.T, np.linalg.pinv(Sigma[np.ix_(subset, subset)]),
            (values[ii][:, subset].sum(axis=0)).T))
    return rhs[:, None]


def BLUE_hf_nsamples_constraint(subsets, nsamples_per_subset):
    nhf_samples = 0
    for ii, subset in enumerate(subsets):
        if 0 in subset:
            nhf_samples += nsamples_per_subset[ii]
    # this is usually only violated for small target costs
    return nhf_samples-(1)


def BLUE_hf_nsamples_constraint_jac(subsets, nsamples_per_subset):
    nsubsets = nsamples_per_subset.shape[0]
    grad = np.zeros(nsubsets)
    for ii, subset in enumerate(subsets):
        if 0 in subset:
            grad[ii] = 1.
    return grad


def BLUE_cost_constraint(target_cost, subset_costs, nsamples_per_subset):
    assert nsamples_per_subset.ndim == 1 and subset_costs.ndim == 1
    return target_cost-nsamples_per_subset @ subset_costs


def BLUE_cost_constraint_jac(target_cost, subset_costs, nsamples_per_subset):
    return -subset_costs


def BLUE_variance(asketch, Sigma, costs, reg_blue, subsets,
                  nsamples_per_subset,
                  return_grad=False, return_hess=False):
    """Compute variance of BLUE estimator using Equation 4.13 paper.
    We normalize costs so that the variance is for budget B_ept=1 with respect
    to nsamples_per_subset. I no longer do this
    """
    if return_hess and not return_grad:
        raise ValueError("return_grad must be True if return_hess is True")

    mat, submats = BLUE_Psi(
        Sigma, costs, reg_blue, subsets, nsamples_per_subset)
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
        return variance, grad

    raise NotImplementedError()
    # nsubsets = len(subsets)
    # temp = np.linalg.multi_dot(
    #     (np.array(submats).reshape(nsubsets*nmodels, nmodels), mat_inv,
    #      asketch)).reshape(nsubsets, nmodels)
    # hess = 2*np.linalg.multi_dot((temp, mat_inv, temp.T))
    # return variance, grad, hess


def _get_MLBLUE_bounds(self, subset_costs, target_cost):
    nsubsets = len(subset_costs)
    bounds = [(0, target_cost/subset_costs[ii]) for ii in range(
        nsubsets)]
    return bounds


def AETC_BLUE_allocate_samples(
        beta_Sp, Sigma_S, sigma_S_sq, x_Sp, Lambda_Sp, costs_S,
        reg_blue, constraint_reg):
    nmodels = len(costs_S)

    # np.trace(Gamma) = $\hat{\sigma}^2_S
    k1 = sigma_S_sq*np.trace(
        np.linalg.multi_dot((x_Sp, x_Sp.T, np.linalg.inv(Lambda_Sp))))

    Sigma_Sp = np.zeros((Sigma_S.shape[0]+1, Sigma_S.shape[1]+1))
    Sigma_Sp[1:, 1:] = Sigma_S

    if nmodels == 1:
        # exploitation cost
        exploit_cost = sum(costs_S)
        k2 = exploit_cost*np.trace(
            np.linalg.multi_dot((Sigma_Sp, beta_Sp, beta_Sp.T)))
        return k1, k2, np.ones(1)

    asketch = beta_Sp[1:]  # remove high-fidelity coefficient
    # USE ML BLUES initital guess, this one is wrong should depend on target cost
    init_guess = np.full(2**nmodels-1, (1/(2**nmodels-1)))

    # need to port this implementation to use new MLBLUE implementation
    # that does not scale target cost to 1 and passes subsets to different
    # BLUE functions
    obj = partial(
        BLUE_variance, asketch, Sigma_S, costs_S, reg_blue, return_grad=True)
    constraints = [
        {'type': 'eq',
         'fun': BLUE_cost_constraint,
         'jac': BLUE_cost_constraint_jac}]
    res = minimize(obj, init_guess, jac=True, method="SLSQP",
                   constraints=constraints,
                   bounds=_get_MLBLUE_bounds(target_cost, costs_S))

    k2 = res["fun"]

    # makes sure nsamples_per_subset.sum() == 1 so that when correcting for
    # a given budget, num_samples_per_model is a fraction of the total budet
    nsamples_per_subset_frac = np.maximum(np.zeros_like(res["x"]), res["x"])
    nsamples_per_subset_frac /= nsamples_per_subset_frac.sum()
    return k1, k2, nsamples_per_subset_frac


def AETC_least_squares(hf_values, covariate_values):
    r"""
    Parameters
    ----------
    hf_values : np.ndarray (nsamples, 1)
        Evaluations of the high-fidelity model

    covariate_values : np.ndarray (nsamples, nmodels-1)
        Evaluations of the low-fidelity models (co-variates)

    Returns
    -------
    """
    # number of samples and number of low fidelity models
    # used to compute linear estimator
    nsamples, ncovariates = covariate_values.shape
    # X_S is evaluations of low fidelity models in subset S
    # X_Sp is $X_{S+}=(1, X_S)$
    X_Sp = np.hstack(
        (np.ones((nsamples, 1)), covariate_values))
    # beta_Sp = $\hat{beta}_{S+}=(\hat{b}_S, \hat{\beta}_S)$
    beta_Sp = np.linalg.lstsq(X_Sp, hf_values, rcond=None)[0]
    assert beta_Sp.ndim == 2 and beta_Sp.shape[1] == 1
    # sigma_S_sq = $\hat{\sigma}_S^2$
    # TODO paper uses (nsamples-ncovariates-1)
    sigma_S_sq = (
        (hf_values-X_Sp.dot(beta_Sp))**2).sum()/(nsamples-1)

    # debugging
    # Gamma = np.cov((hf_values-X_Sp.dot(beta_Sp)).T)
    # print(((hf_values-X_Sp.dot(beta_Sp)).T).shape, beta_Sp.shape)
    # print(Gamma, sigma_S_sq)#, X_Sp, hf_values[:, 0], beta_Sp[:, 0])
    # assert np.allclose(np.trace(np.atleast_2d(Gamma)), sigma_S_sq)
    return beta_Sp, sigma_S_sq, X_Sp


def AETC_BLUE_objective_deprecated(
        asketch, Sigma_S, costs, kappa, reg_blue, nsamples_per_subset,
        return_grad=False, return_hess=False):

    if return_hess and not return_grad:
        raise ValueError("return_grad must be True if return_hess is True")

    # TODO why is budget constraint squared, e.g. **2
    budget_constraint = (nsamples_per_subset.sum()-1)**2
    result = BLUE_variance(
        asketch, Sigma_S, costs, reg_blue, nsamples_per_subset, return_grad,
        return_hess)

    if not return_grad:
        variance = result
        return variance + kappa*budget_constraint

    variance = result[0]
    objective = variance + kappa*budget_constraint
    grad = result[1]
    reg_grad = 2*kappa*(nsamples_per_subset.sum()-1)
    grad += reg_grad  # applies reg_grad to each entry

    # print(objective, nsamples_per_subset)

    if not return_hess:
        return objective, grad

    reg_hess = kappa*np.ones((grad.shape[0], grad.shape[0]))
    hess = result[2] + reg_hess
    return objective, grad, hess


def AETC_BLUE_allocate_samples_deprecated(
        beta_Sp, Sigma_S, sigma_S_sq, x_Sp, Lambda_Sp, costs_S,
        reg_blue, constraint_reg):
    kappa = 1e2
    nmodels = len(costs_S)

    # np.trace(Gamma) = $\hat{\sigma}^2_S
    k1 = sigma_S_sq*np.trace(
        np.linalg.multi_dot((x_Sp, x_Sp.T, np.linalg.inv(Lambda_Sp))))

    Sigma_Sp = np.zeros((Sigma_S.shape[0]+1, Sigma_S.shape[1]+1))
    Sigma_Sp[1:, 1:] = Sigma_S

    if nmodels == 1:
        # exploitation cost
        exploit_cost = sum(costs_S)
        k2 = exploit_cost*np.trace(
            np.linalg.multi_dot((Sigma_Sp, beta_Sp, beta_Sp.T)))
        return k1, k2, np.ones(1)

    asketch = beta_Sp[1:]  # remove high-fidelity coefficient
    init_guess = np.full(2**nmodels-1, (1/(2**nmodels-1)))

    obj = partial(
        AETC_BLUE_objective_deprecated, asketch, Sigma_S, costs_S, kappa,
        reg_blue, return_grad=True)
    constraints = {
        'type': 'ineq',
        'fun': partial(BLUE_bound_constraint, constraint_reg),
        'jac': BLUE_bound_constraint_jac}
    res = minimize(obj, init_guess, jac=True, method="SLSQP",
                   constraints=constraints)

    k2 = res["fun"]

    # makes sure nsamples_per_subset.sum() == 1 so that when correcting for
    # a given budget, num_samples_per_model is a fraction of the total budet
    nsamples_per_subset_frac = np.maximum(np.zeros_like(res["x"]), res["x"])
    nsamples_per_subset_frac /= nsamples_per_subset_frac.sum()
    return k1, k2, nsamples_per_subset_frac


def _AETC_subset_oracle_stats(oracle_stats, covariate_subset):
    cov, means = oracle_stats
    Sigma_S = cov[np.ix_(covariate_subset+1, covariate_subset+1)]
    Sp_subset = np.hstack((0, covariate_subset+1))
    x_Sp = means[Sp_subset]
    tmp1 = np.zeros_like(cov)
    tmp1[1:, 1:] = cov[1:, 1:]
    tmp2 = np.vstack((1, means[1:]))
    Lambda_Sp = (tmp1+tmp2.dot(tmp2.T))[np.ix_(Sp_subset, Sp_subset)]
    return Sigma_S, Lambda_Sp, x_Sp


def AETC_optimal_loss(
        total_budget, hf_values, covariate_values, costs, covariate_subset,
        alpha, reg_blue, constraint_reg, oracle_stats):
    r"""
    Parameters
    ----------
    total_budget : float
        The total budget allocated to exploration and exploitation.

    hf_values : np.ndarray (nsamples, 1)
        Evaluations of the high-fidelity model

    covariate_values : np.ndarray (nsamples, nmodels-1)
        Evaluations of the low-fidelity models (co-variates)

    costs : np.ndarray (nmodels)
        The computational cost of evaluating each model at one realization.
        High fidelity is assumed to be first entry

    covariate_subset : np.ndarray (nsubset_models)
        The indices :math:`S\subseteq[1,\ldots,N]` of the low-fidelity models
        in the subset. :math:`s\in S` indexes the columns in values and costs.
        High-fidelity is indexed by :math:`s=0` and cannot be in :math:`S`.

    alpha : float
        Regularization parameter

    """
    # Compute AETC least squares solution beta_Sp and trace of
    # residual covariance sigma_S_sq
    beta_Sp, sigma_S_sq, X_Sp = AETC_least_squares(
        hf_values, covariate_values[:, covariate_subset])

    # Compute Lambda_S = \hat{Lambda}_{S} # TODO in paper why not subscript S+
    nsamples = hf_values.shape[0]

    # Sigma_S = $\hat{\Sigma}_S$
    if oracle_stats is None:
        # x_Sp = $\bar{x}_{S+}$
        x_Sp = X_Sp.mean(axis=0)[:, None]
        Sigma_S = np.atleast_2d(
            np.cov(covariate_values[:, covariate_subset].T))
        # TODO in paper $X_{S+}$=X_Sp.T used here
        Lambda_Sp = X_Sp.T.dot(X_Sp)/nsamples
        # print(Sigma_S, Lambda_Sp, x_Sp)
    else:
        Sigma_S, Lambda_Sp, x_Sp = _AETC_subset_oracle_stats(
            oracle_stats, covariate_subset)
        # print(Sigma_S, Lambda_Sp, x_Sp)

    # extract costs of models in subset
    # covariate_subset+1 is used because high-fidelity assumed
    # to be in first column of covariate values and covariate_subset
    # just indexes low fidelity models starting from 0
    costs_S = costs[covariate_subset+1]

    # find optimal sample allocation
    # only pass in costs_S of subset because exploitation does not
    # further evaluate the high-fidelity model
    # print(covariate_subset, beta_Sp[:, 0])
    # print(X_Sp)
    # print(hf_values[:, 0])
    # print(covariate_subset)
    # k1, k2, nsamples_per_subset_frac = AETC_BLUE_allocate_samples_deprecated(
    k1, k2, nsamples_per_subset_frac = AETC_BLUE_allocate_samples(
        beta_Sp, Sigma_S, sigma_S_sq, x_Sp, Lambda_Sp, costs_S,
        reg_blue, constraint_reg)

    # cost of exploration (exploration evaluates all models)
    explore_cost = costs.sum()

    # estimate optimal exploration rate Equation 4.34
    # k2 is $\gamma_m$ and k1 is $k_m$ in the paper
    explore_rate = max(
        total_budget/(
            explore_cost+np.sqrt(explore_cost*k2/(k1+alpha**(-nsamples)))),
        nsamples)
    # print(explore_rate, 'r', total_budget, explore_cost, alpha, nsamples)

    # estimate optimal loss
    exploit_budget = (total_budget-explore_cost*explore_rate)
    opt_loss = k2/exploit_budget+(k1+alpha**(-nsamples))/explore_rate

    return (opt_loss, nsamples_per_subset_frac, explore_rate, beta_Sp, Sigma_S,
            k2, exploit_budget)
