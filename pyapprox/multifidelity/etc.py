from functools import partial

import numpy as np

from pyapprox.multifidelity.groupacv import MLBLUEEstimator, get_model_subsets
from pyapprox.surrogates.autogp._torch_wrappers import asarray
from pyapprox.multifidelity.stats import MultiOutputMean


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


def _AETC_least_squares(hf_values, covariate_values):
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
    # Gamma = np.cov((hf_values-X_Sp.dot(beta_Sp)).T, ddof=0)
    # print(((hf_values-X_Sp.dot(beta_Sp)).T).shape, beta_Sp.shape)
    # print(Gamma, sigma_S_sq)#, X_Sp, hf_values[:, 0], beta_Sp[:, 0])
    # assert np.allclose(np.trace(np.atleast_2d(Gamma)), sigma_S_sq)
    return beta_Sp, sigma_S_sq, X_Sp


def _AETC_BLUE_allocate_samples(
        beta_Sp, Sigma_S, sigma_S_sq, x_Sp, Lambda_Sp, costs_S,
        reg_blue, constraint_reg, opt_options, exploit_budget):
    nmodels = len(costs_S)

    # np.trace(Gamma) = $\hat{\sigma}^2_S
    k1 = sigma_S_sq*np.trace(
        np.linalg.multi_dot((x_Sp, x_Sp.T, np.linalg.inv(Lambda_Sp))))

    Sigma_Sp = np.zeros((Sigma_S.shape[0]+1, Sigma_S.shape[1]+1))
    Sigma_Sp[1:, 1:] = Sigma_S

    normalize_opt = opt_options.copy().pop("normalize", False)

    if nmodels == 1:
        # exploitation cost
        exploit_cost = sum(costs_S)
        nsamples_per_subset = exploit_budget/exploit_cost
        k2 = exploit_cost*np.trace(
            np.linalg.multi_dot((Sigma_Sp, beta_Sp, beta_Sp.T)))
        # exploit buget cancels out because k2 = var*exploit_budget
        return k1, k2, nsamples_per_subset*np.ones(1)

    asketch = beta_Sp[1:]  # remove high-fidelity coefficient

    stat_S = MultiOutputMean(1)
    stat_S.set_pilot_quantities(Sigma_S)
    est = MLBLUEEstimator(
        stat_S, costs_S, asketch=asketch, reg_blue=reg_blue)
    if normalize_opt:
        target_cost = 1
    else:
        target_cost = exploit_budget
    est.allocate_samples(target_cost, round_nsamples=False,
                         optim_options=opt_options, min_nhf_samples=0)
    nsamples_per_subset = np.maximum(
            np.zeros_like(est._rounded_npartition_samples),
            est._rounded_npartition_samples)
    k2 = est._optimized_criteria
    if not normalize_opt:
        k2 *= exploit_budget
    else:
        nsamples_per_subset *= exploit_budget

    return k1, k2, nsamples_per_subset


def _AETC_optimal_loss(
        total_budget, hf_values, covariate_values, costs, covariate_subset,
        alpha, reg_blue, constraint_reg, oracle_stats, opt_options,
        exploit_budget):
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
    beta_Sp, sigma_S_sq, X_Sp = _AETC_least_squares(
        hf_values, covariate_values[:, covariate_subset])

    # Compute Lambda_S = \hat{Lambda}_{S} # TODO in paper why not subscript S+
    nsamples = hf_values.shape[0]

    # Sigma_S = $\hat{\Sigma}_S$
    if oracle_stats is None:
        # x_Sp = $\bar{x}_{S+}$
        x_Sp = X_Sp.mean(axis=0)[:, None]
        Sigma_S = np.atleast_2d(
            np.cov(covariate_values[:, covariate_subset].T, ddof=1))
        # TODO in paper $X_{S+}$=X_Sp.T used here
        Lambda_Sp = X_Sp.T.dot(X_Sp)/nsamples
    else:
        Sigma_S, Lambda_Sp, x_Sp = _AETC_subset_oracle_stats(
            oracle_stats, covariate_subset)

    # extract costs of models in subset
    # covariate_subset+1 is used because high-fidelity assumed
    # to be in first column of covariate values and covariate_subset
    # just indexes low fidelity models starting from 0
    costs_S = costs[covariate_subset+1]

    # find optimal sample allocation
    # only pass in costs_S of subset because exploitation does not
    # further evaluate the high-fidelity model
    k1, k2, nsamples_per_subset = _AETC_BLUE_allocate_samples(
        beta_Sp, Sigma_S, sigma_S_sq, x_Sp, Lambda_Sp, costs_S,
        reg_blue, constraint_reg, opt_options, exploit_budget)

    # cost of exploration (exploration evaluates all models)
    explore_cost = costs.sum()

    # estimate optimal exploration rate Equation 4.34
    # k2 is $\gamma_m$ and k1 is $k_m$ in the paper
    explore_rate = max(
        total_budget/(
            explore_cost+np.sqrt(explore_cost*k2/(k1+alpha**(-nsamples)))),
        nsamples)

    # estimate optimal loss
    exploit_budget = (total_budget-explore_cost*explore_rate)
    opt_loss = k2/exploit_budget+(k1+alpha**(-nsamples))/explore_rate
    return (opt_loss, nsamples_per_subset, explore_rate, beta_Sp, Sigma_S,
            k1, k2, exploit_budget)


class AETCBLUE():
    def __init__(self, models, rvs, costs=None, oracle_stats=None,
                 reg_blue=1e-15, constraint_reg=0, opt_options={}):
        r"""
        Parameters
        ----------
        models : list
            List of callable functions fun with signature

            ``fun(samples)-> np.ndarary (nsamples, nqoi)``

        where samples is np.ndarray (nvars, nsamples)

        rvs : callable
            Function used to generate random samples with signature

            ``fun(nsamples)-> np.ndarary (nvars, nsamples)``

        costs : iterable
            Iterable containing the time taken to evaluate a single sample
            with each model. If None then each model will be assumed to
            track the evaluation time.

        oracle_stats : list[np.ndarray (nmodels, nmodels), np.ndarray (nmodels, nmodels)]
            This is only used for testing.
            First element is the Oracle covariance between models.
            Second element is the Oracle Lambda_Sp
        """
        self.models = models
        self._nmodels = len(models)
        if not callable(rvs):
            raise ValueError("rvs must be callabe")
        self.rvs = rvs
        self._costs = self._validate_costs(costs)
        self._reg_blue = reg_blue
        self._constraint_reg = constraint_reg
        self._oracle_stats = oracle_stats
        self._opt_options = opt_options

    def _validate_costs(self, costs):
        if costs is None:
            return
        if len(costs) != self._nmodels:
            raise ValueError("costs must be provided for each model")
        return np.asarray(costs)

    def _validate_subsets(self, subsets):
        # subsets are indexes of low fidelity models
        if subsets is None:
            subsets = get_model_subsets(self._nmodels-1)
        validated_subsets, max_ncovariates = [], -np.inf
        for subset in subsets:
            if ((np.unique(subset).shape[0] != len(subset)) or
                    (np.max(subset) >= self._nmodels-1)):
                msg = "subsets provided are not valid. First invalid subset"
                msg += f" {subset}"
                raise ValueError(msg)
            validated_subsets.append(np.asarray(subset))
            max_ncovariates = max(max_ncovariates, len(subset))
        return validated_subsets, max_ncovariates

    def _explore_step(self, total_budget, lf_model_subsets, values, alpha,
                      reg_blue, constraint_reg):
        """
        Parameters
        ----------
        subsets : list[np.ndarray]
           Indices of the low fidelity models in a subset from 0,...,K-2
           e.g. (0) contains only the first low fidelity model and (0, 2)
           contains the first and third. 0 DOES NOT correspond to the
           high-fidelity model
        """

        nsamples = values.shape[0]
        explore_cost = np.sum(self._costs)

        # compute exploitation budget used _AETC_optimal_loss. It is notation
        # the exploit budget if nexplore_samples (redefined below) > nsamples
        # Note exploit budget is for unrounded nsamples_per_subset
        exploit_budget = (total_budget-nsamples*explore_cost)
        if (exploit_budget) < 0:
            raise RuntimeError("Exploitation budget is negative")

        results = []
        for subset in lf_model_subsets:
            result = _AETC_optimal_loss(
                total_budget, values[:, :1], values[:, 1:], self._costs,
                subset, alpha, reg_blue, constraint_reg, self._oracle_stats,
                self._opt_options, exploit_budget)
            (loss, nsamples_per_subset, explore_rate, beta_Sp,
             Sigma_S, k1, k2, exploit_budget) = result
            results.append(result)

        # compute optimal model
        best_subset_idx = np.argmin([result[0] for result in results])
        best_result = results[best_subset_idx]
        (best_loss, best_allocation, best_rate, best_beta_Sp,
         best_Sigma_S, best_k1, best_k2,
         best_exploit_budget) = best_result
        best_cost = self._costs[lf_model_subsets[best_subset_idx]+1].sum()

        best_subset = lf_model_subsets[best_subset_idx]
        # use +1 to accound for subset indexing only lf models
        best_subset_costs = self._costs[best_subset+1]
        best_subset_groups = get_model_subsets(best_subset.shape[0])
        # print(best_subset_groups)
        best_subset_group_costs = asarray([
            best_subset_costs[group].sum() for group in best_subset_groups])

        # recorrect for solving exploitation with unit exploit budget
        best_nsamples_per_subset = asarray(best_allocation)
        rounded_best_nsamples_per_subset = asarray(
            np.floor(best_nsamples_per_subset))
        best_blue_variance = best_k2/best_exploit_budget

        # Incrementing one round at a time is the most optimal
        # but does not allow for parallelism
        # if best_rate <= nsamples:
        #     nexplore_samples = nsamples
        # else:
        #     nexplore_samples = nsamples + 1

        if best_rate > 2*nsamples:
            nexplore_samples = 2*nsamples
        elif best_rate > nsamples:
            nexplore_samples = int(np.ceil((nsamples+best_rate)/2))
        else:
            nexplore_samples = nsamples

        if (total_budget-nexplore_samples*explore_cost) < 0:
            nexplore_samples = int(total_budget/explore_cost)

        return (nexplore_samples, best_subset, best_cost, best_beta_Sp,
                best_Sigma_S, rounded_best_nsamples_per_subset,
                best_nsamples_per_subset,
                best_loss, best_k1,
                best_blue_variance, best_exploit_budget,
                best_subset_group_costs)

    def explore(self, total_budget, lf_model_subsets, alpha=4):
        if self._costs is None:
            # todo extract costs from models
            # costs = ...
            raise NotImplementedError()
        lf_model_subsets, max_ncovariates = self._validate_subsets(
            lf_model_subsets)

        nexplore_samples = max_ncovariates+2
        nexplore_samples_prev = 0
        while ((nexplore_samples - nexplore_samples_prev > 0)):
            nnew_samples = nexplore_samples-nexplore_samples_prev
            new_samples = self.rvs(nnew_samples)
            new_values = [
                model(new_samples) for model in self.models]
            if nexplore_samples_prev == 0:
                samples = new_samples
                values = np.hstack(new_values)
                # will fail if model does not return ndarray (nsamples, nqoi=1)
                assert values.ndim == 2
            else:
                samples = np.hstack((samples, new_samples))
                values = np.vstack((values, np.hstack(new_values)))
            nexplore_samples_prev = nexplore_samples
            result = self._explore_step(
                total_budget, lf_model_subsets, values, alpha, self._reg_blue,
                self._constraint_reg)
            nexplore_samples = result[0]
            last_result = result
        return samples, values, last_result  # akil returns result

    def exploit(self, result):
        best_subset = result[1]
        beta_Sp, Sigma_best_S, rounded_nsamples_per_subset = result[3:6]
        costs_best_S = self._costs[best_subset+1]
        beta_best_S = beta_Sp[1:]
        stat_best_S = MultiOutputMean(1)
        stat_best_S.set_pilot_quantities(Sigma_best_S)
        est = MLBLUEEstimator(
            stat_best_S, costs_best_S, Sigma_best_S, asketch=beta_best_S)
        est._set_optimized_params(rounded_nsamples_per_subset)
        samples_per_model = est.generate_samples_per_model(self.rvs)
        # use +1 to accound for subset indexing only lf models
        values_per_model = [
            self.models[s+1](samples)
            for s, samples in zip(best_subset, samples_per_model)]
        return beta_Sp[0, 0] + est(values_per_model).item()

    def _explore_result_to_dict(self, result):
        result = {
            "nexplore_samples": result[0], "subset": result[1],
            "subset_cost": result[2], "beta_Sp": result[3],
            "sigma_S": result[4], "rounded_nsamples_per_subset": result[5],
            "nsamples_per_subset": result[6],
            "loss": result[7], "k1": result[8],
            # BLUE_variance is for unrounded nsamples_per_subset
            "BLUE_variance": result[9],
            "exploit_budget": result[10], "mlblue_subset_costs": result[11],
            "explore_budget": result[0]*(result[2]+self._costs[0])}
        return result

    def estimate(self, total_budget, subsets=None, return_dict=True):
        samples, values, result = self.explore(total_budget, subsets)
        mean = self.exploit(result)
        if not return_dict:
            return mean, values, result
        # package up result
        result = self._explore_result_to_dict(result)
        return mean, values, result

    def __repr__(self):
        rep = "{0}()".format(self.__class__.__name__)
        return rep
