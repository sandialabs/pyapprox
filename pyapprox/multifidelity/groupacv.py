from functools import partial
from itertools import combinations

import torch
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

try:
    import cvxpy
    _cvx_available = True
except ImportError:
    _cvx_available = False


from pyapprox.surrogates.autogp._torch_wrappers import (
    full, multidot, pinv, solve, hstack, vstack, asarray,
    eye, log, einsum, floor, copy)
from pyapprox.multifidelity.stats import MultiOutputMean


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


def _get_allocation_matrix_is(subsets):
    nsubsets = len(subsets)
    npartitions = nsubsets
    allocation_mat = full(
        (nsubsets, npartitions), 0., dtype=torch.double)
    for ii, subset in enumerate(subsets):
        allocation_mat[ii, ii] = 1.0
    return allocation_mat


def _get_allocation_matrix_nested(subsets):
    # nest partitions according to order of subsets
    nsubsets = len(subsets)
    npartitions = nsubsets
    allocation_mat = full(
        (nsubsets, npartitions), 0., dtype=torch.double)
    for ii, subset in enumerate(subsets):
        allocation_mat[ii, :ii+1] = 1.0
    return allocation_mat


def _nest_subsets(subsets, nmodels):
    for subset in subsets:
        if np.allclose(subset, [0]):
            raise ValueError("Cannot use subset [0]")
    idx = sorted(
        list(range(len(subsets))),
        key=lambda ii: (len(subsets[ii]), tuple(nmodels-subsets[ii])),
        reverse=True)
    return [subsets[ii] for ii in idx], np.array(idx)


def _grouped_acv_beta(nmodels, Sigma, subsets, R, reg, asketch):
    """
    Parameters
    ----------
    nmodels: integer
        The total number of models including the highest fidelity

    Sigma : array (nestimators, nestimators)
        The covariance between all estimators

    reg : float
        Regularization parameter to stabilize matrix inversion
    """
    reg_mat = np.identity(nmodels)*reg
    if asketch.shape != (nmodels, 1):
        raise ValueError("asketch has the wrong shape")

    # TODO instead of applyint R matrices just collect correct rows and columns
    beta = multidot((
        pinv(Sigma), R.T,
        solve(multidot((R, pinv(Sigma), R.T))+reg_mat, asketch[:, 0])))
    return beta


def _grouped_acv_variance(nmodels, Sigma, subsets, R, reg, asketch):
    reg_mat = np.identity(nmodels)*reg
    if asketch.shape != (nmodels, 1):
        raise ValueError("asketch has the wrong shape")

    reg_mat = eye(nmodels)*reg
    return asketch.T @ pinv(multidot((R, pinv(Sigma), R.T))+reg_mat) @ asketch


def _grouped_acv_estimate(
        nmodels, Sigma, reg, subsets, subset_values, R, asketch):
    nsubsets = len(subsets)
    beta = _grouped_acv_beta(nmodels, Sigma, subsets, R, reg, asketch)
    ll, mm = 0, 0
    acv_mean = 0
    for kk in range(nsubsets):
        mm += len(subsets[kk])
        if subset_values[kk].shape[0] > 0:
            subset_mean = subset_values[kk].mean(axis=0)
            acv_mean += (beta[ll:mm]) @ subset_mean
        ll = mm
    return acv_mean


def _grouped_acv_sigma_block(
        subset0, subset1, nsamples_intersect, nsamples_subset0,
        nsamples_subset1, cov):
    nsubset0 = len(subset0)
    nsubset1 = len(subset1)
    block = full((nsubset0, nsubset1), 0.)
    if (nsamples_subset0*nsamples_subset1) == 0:
        return block
    block = cov[np.ix_(subset0, subset1)]*nsamples_intersect/(
                nsamples_subset0*nsamples_subset1)
    return block


def _grouped_acv_sigma(
        nmodels, nsamples_intersect, cov, subsets):
    nsubsets = len(subsets)
    Sigma = [[None for jj in range(nsubsets)] for ii in range(nsubsets)]
    for ii, subset0 in enumerate(subsets):
        N_ii = nsamples_intersect[ii, ii]
        Sigma[ii][ii] = _grouped_acv_sigma_block(
            subset0, subset0, N_ii, N_ii, N_ii, cov)
        for jj, subset1 in enumerate(subsets[:ii]):
            N_jj = nsamples_intersect[jj, jj]
            Sigma[ii][jj] = _grouped_acv_sigma_block(
                subset0, subset1, nsamples_intersect[ii, jj],
                N_ii, N_jj, cov)
            Sigma[jj][ii] = Sigma[ii][jj].T
    Sigma = vstack([hstack(row) for row in Sigma])
    return Sigma


class GroupACVEstimator():
    def __init__(self, stat, costs, reg_blue=0, subsets=None,
                 est_type="is", asketch=None):
        self._cov, self._costs = self._check_cov(stat._cov, costs)
        self.nmodels = len(costs)
        self._reg_blue = reg_blue
        if not isinstance(stat, MultiOutputMean):
            raise ValueError(
                "MLBLUE currently only suppots estimation of means")
        self._stat = stat

        self.subsets, self.allocation_mat = self._set_subsets(
            subsets, est_type)
        self.nsubsets = len(self.subsets)
        self.npartitions = self.allocation_mat.shape[1]
        self.partitions_per_model = self._get_partitions_per_model()
        self.partitions_intersect = (
            self._get_subset_intersecting_partitions())
        self.R = hstack(
            [asarray(_restriction_matrix(self.nmodels, subset).T)
             for ii, subset in enumerate(self.subsets)])
        self._costs = asarray(costs)
        self.subset_costs = self._get_model_subset_costs(
            self.subsets, self._costs)

        # set npatition_samples above small constant,
        # otherwise gradient will not be defined.
        self._npartition_samples_lb = 0  # 1e-5

        self._npartitions = self.nsubsets  # TODO replace .nsubsets everywhere
        self._optimized_criteria = None
        self._asketch = self._validate_asketch(asketch)

        if est_type == "is":
            self._obj_jac = True
        else:
            # hack because currently autogradients do not works so must
            # use finite difference
            self._obj_jac = False

    def _check_cov(self, cov, costs):
        if cov.shape[0] != len(costs):
            print(cov.shape, costs.shape)
            raise ValueError("cov and costs are inconsistent")
        return cov, np.array(costs)

    def _set_subsets(self, subsets, est_type):
        if subsets is None:
            subsets = get_model_subsets(self.nmodels)
        if est_type == "is":
            get_allocation_mat = _get_allocation_matrix_is
        elif est_type == "nested":
            for ii, subset in enumerate(subsets):
                if np.allclose(subset, [0]):
                    del subsets[ii]
                    break
            subsets = _nest_subsets(subsets, self.nmodels)[0]
            get_allocation_mat = _get_allocation_matrix_nested
        else:
            raise ValueError(
                "incorrect est_type {0} specified".format(est_type))
        return subsets,  get_allocation_mat(subsets)

    def _get_partitions_per_model(self):
        # assume npartitions = nsubsets
        npartitions = self.allocation_mat.shape[1]
        partitions_per_model = full((self.nmodels, npartitions), 0.)
        for ii, subset in enumerate(self.subsets):
            partitions_per_model[
                np.ix_(subset, self.allocation_mat[ii] == 1)] = 1
        return partitions_per_model

    def _compute_nsamples_per_model(self, npartition_samples):
        nsamples_per_model = einsum(
            "ji,i->j", self.partitions_per_model, npartition_samples)
        return nsamples_per_model

    def _estimator_cost(self, npartition_samples):
        return sum(
            self._costs*self._compute_nsamples_per_model(npartition_samples))

    def _get_subset_intersecting_partitions(self):
        amat = self.allocation_mat
        npartitions = self.allocation_mat.shape[1]
        partition_intersect = full(
            (self.nsubsets, self.nsubsets, npartitions), 0.)
        for ii, subset_ii in enumerate(self.subsets):
            for jj, subset_jj in enumerate(self.subsets):
                # partitions are shared when sum of allocation entry is 2
                partition_intersect[ii, jj, amat[ii]+amat[jj] == 2] = 1.
        return partition_intersect

    def _nintersect_samples(self, npartition_samples):
        """
        Get the number of samples in the intersection of two subsets.

        Note the number of samples per subset is simply the diagonal of this
        matrix
        """
        return einsum(
            "ijk,k->ij", self.partitions_intersect, npartition_samples)

    def _sigma(self, npartition_samples):
        return _grouped_acv_sigma(
            self.nmodels, self._nintersect_samples(npartition_samples),
            self._cov, self.subsets)

    def _covariance_from_npartition_samples(self, npartition_samples):
        return _grouped_acv_variance(
            self.nmodels, self._sigma(npartition_samples), self.subsets,
            self.R, self._reg_blue, self._asketch)

    def _objective(self, npartition_samples_np, return_grad=True):
        npartition_samples = torch.as_tensor(
            npartition_samples_np.copy(), dtype=torch.double)
        if return_grad:
            npartition_samples.requires_grad = True
        est_var = self._covariance_from_npartition_samples(
            npartition_samples)
        # talking log of objective seems to make scipy strugle to minimize
        val = est_var
        if not return_grad:
            return val.item()
        val.backward()
        grad = npartition_samples.grad.detach().numpy().copy()
        npartition_samples.grad.zero_()
        return val.item(), grad

    @staticmethod
    def _get_model_subset_costs(subsets, costs):
        subset_costs = np.array(
            [costs[subset].sum() for subset in subsets])
        return subset_costs

    def _cost_constraint(
            self, npartition_samples_np, target_cost, return_grad=False):
        # because this is a constraint it must only return grad or val
        # not both unlike usual PyApprox convention
        npartition_samples = torch.as_tensor(
            npartition_samples_np, dtype=torch.double)
        if return_grad:
            npartition_samples.requires_grad = True
        val = (target_cost-self._estimator_cost(npartition_samples))
        if not return_grad:
            return val.item()
        raise NotImplementedError(
            "This feature has been deprecated use self._cost_constraint_jac")
        # val.backward()
        # grad = npartition_samples.grad.detach().numpy().copy()
        # npartition_samples.grad.zero_()
        # assert np.allclose(
        #     grad,
        #     self._cost_constraint_jac(npartition_samples_np, target_cost))
        # return grad

    def _nhf_samples(self, npartition_samples):
        return (self.partitions_per_model[0]*npartition_samples).sum()

    def _nhf_samples_constraint(self, npartition_samples, min_nhf_samples):
        return self._nhf_samples(npartition_samples)-min_nhf_samples

    def _nhf_samples_constraint_jac(self, npartition_samples, min_nhf_samples):
        return self.partitions_per_model[0]

    def _cost_constraint_jac(self, npartition_samples_np, target_cost):
        return -(self._costs[None, :] @ self.partitions_per_model).numpy()

    def _validate_target_cost_min_nhf_samples(
            self, target_cost, min_nhf_samples):
        lb = min_nhf_samples*self._costs[0]
        ub = target_cost
        if ub < lb:
            msg = "target_cost {0} and cost of min_nhf_samples {1} ".format(
                target_cost, lb)
            msg += "are inconsistent"
            raise ValueError(msg)
        return lb, ub

    def _nelder_mead_min_nlf_samples_constraint(
            self, x, min_nlf_samples, ii):
        return (self.partitions_per_model[ii].numpy()*x).sum()-min_nlf_samples

    def _get_nelder_mead_constraints(self, target_cost, min_nhf_samples,
                                     min_nlf_samples, constraint_reg=0):
        cons = [
            {'type': 'ineq',
             'fun': self._cost_constraint,
             'args': (target_cost, )}]
        cons += [
            {'type': 'ineq',
             'fun': self._nelder_mead_min_nlf_samples_constraint,
             'args': [min_nhf_samples, 0]}]
        if min_nlf_samples is not None:
            assert len(min_nlf_samples) == self.nmodels-1
            for ii in range(1, self.nmodels):
                cons += [
                    {'type': 'ineq',
                     'fun': self._nelder_mead_min_nlf_samples_constraint,
                     'args': [min_nlf_samples, ii, ]}]
        return cons

    def _get_constraints(self, target_cost, min_nhf_samples,
                         min_nlf_samples, constraint_reg=0):
        keep_feasible = False
        lb, ub = self._validate_target_cost_min_nhf_samples(
            target_cost, min_nhf_samples)
        cons = [LinearConstraint(
            (self._costs[None, :]@self.partitions_per_model).numpy(),
            lb=lb, ub=ub, keep_feasible=keep_feasible)]
        cons += [LinearConstraint(
            self.partitions_per_model[0].numpy(), lb=min_nhf_samples,
            keep_feasible=keep_feasible)]
        if min_nlf_samples is not None:
            assert len(min_nlf_samples) == self.nmodels-1
            for ii in range(1, self.nmodels):
                cons += [LinearConstraint(
                    self.partitions_per_model[ii].numpy(),
                    lb=min_nlf_samples[ii-1],
                    keep_feasible=keep_feasible)]
        return cons

    def _constrained_objective(self, cons, x):
        # used for gradient free optimizers
        lamda = 1e8
        cons_term = 0
        for con in cons:
            c_val = con["fun"](x, *con["args"])
            if c_val < 0:
                cons_term -= c_val * lamda
        return self._objective(x, return_grad=False) + cons_term

    def _init_guess(self, target_cost):
        # start with the same number of samples per partition

        # get the number of samples per model when 1 sample is in each
        # partition
        nsamples_per_model = self._compute_nsamples_per_model(
            full((self.npartitions,), 1.))
        # nsamples_per_model[0] = max(0, min_nhf_samples)
        cost = (nsamples_per_model*self._costs).sum()

        # the total number of samples per partition is then target_cost/cost
        # we take the floor to make sure we do not exceed the target cost
        return full(
            (self.npartitions,), np.floor(target_cost/cost))

    def _update_init_guess(self, init_guess, constraints, options, bounds):
        method = "nelder-mead"
        # options["xatol"] = 1e-6
        # options["fatol"] = 1e-6
        # options["maxfev"] = 100 * len(init_guess)
        obj = partial(self._constrained_objective, constraints)
        res = minimize(
            obj, init_guess, jac=False,
            method=method, constraints=None, options=options,
            bounds=self._get_bounds(*bounds))
        return res.x

    def _set_optimized_params_base(self, rounded_npartition_samples,
                                   rounded_nsamples_per_model,
                                   rounded_target_cost):
        self._rounded_npartition_samples = rounded_npartition_samples
        self._rounded_nsamples_per_model = rounded_nsamples_per_model
        self._rounded_target_cost = rounded_target_cost
        self._opt_sample_splits = self._sample_splits_per_model()
        self._optimized_sigma = self._sigma(self._rounded_npartition_samples)
        self._optimized_criteria = self._covariance_from_npartition_samples(
            self._rounded_npartition_samples).item()

    def _set_optimized_params(self, npartition_samples, round_nsamples=True):
        if round_nsamples:
            rounded_npartition_samples = floor(npartition_samples)
        else:
            rounded_npartition_samples = npartition_samples
        self._set_optimized_params_base(
            rounded_npartition_samples,
            self._compute_nsamples_per_model(rounded_npartition_samples),
            self._estimator_cost(rounded_npartition_samples))

    def _get_bounds(self, lb, ub):
        # better to use bounds because they are never violated
        # but enforcing bounds as constraints means bounds can be violated
        # bounds = [(0, np.inf) for ii in range(self.npartitions)]
        # optimizer has trouble when ub in np.inf
        bounds = Bounds(np.zeros(self.npartitions)+lb,
                        np.full((self.npartitions,), ub),
                        keep_feasible=True)
        return bounds

    def _validate_asketch(self, asketch):
        if asketch is None:
            asketch = full((self.nmodels, 1), 0)
            asketch[0] = 1.0
        asketch = asarray(asketch)
        if asketch.shape[0] != self._costs.shape[0]:
            raise ValueError("aksetch has the wrong shape")
        if asketch.ndim == 1:
            asketch = asketch[:, None]
        return asketch

    def allocate_samples(self, target_cost,
                         constraint_reg=0, round_nsamples=True,
                         init_guess=None,
                         min_nhf_samples=1, min_nlf_samples=None,
                         optim_options={}):
        """
        Parameters
        ----------
        min_nhf_samples : float
            The minimum number of high-fidelity samples before rounding.
            Unforunately, there is no way to enforce that the min_nhf_samples
            is met after rounding. As the differentiable constraint
            enforces that the sum of the nsamples in each partition involving
            the high-fidelity model is zero. But when each partition nsample
            is rounded the rounded nhf_samples may be less than desired. It
            will be close though
        """
        obj = partial(self._objective, return_grad=self._obj_jac)
        constraints = self._get_constraints(
            target_cost, min_nhf_samples, min_nlf_samples, constraint_reg)
        optim_options_copy = optim_options.copy()
        bounds = optim_options_copy.pop("bounds", [1e-8, 1e10])
        if init_guess is None:
            init_guess = self._init_guess(target_cost)
            init_opts = optim_options_copy.pop("init_guess", {})
            if isinstance(init_opts, dict):
                nelder_mead_constraints = self._get_nelder_mead_constraints(
                    target_cost, min_nhf_samples, min_nlf_samples,
                    constraint_reg)
                init_guess = self._update_init_guess(
                    init_guess, nelder_mead_constraints, init_opts, bounds)
        init_guess = np.maximum(init_guess, self._npartition_samples_lb)
        method = optim_options_copy.pop("method", "trust-constr")
        # import warnings
        # warnings.filterwarnings("error")
        res = minimize(
            obj, init_guess, jac=self._obj_jac,
            method=method, constraints=constraints,
            options=optim_options_copy,
            bounds=self._get_bounds(*bounds))
        if not res.success or np.any(res["x"] < 0):
            # second condition is needed, even though bounds should enforce,
            # positivity, because somtimes trust-constr does not enforce
            # bounds and I am not sure why
            msg = "optimization not successful"
            print(msg)
            print(res)
            raise RuntimeError(msg)

        self._set_optimized_params(asarray(res["x"]), round_nsamples)

    @staticmethod
    def _get_partition_splits(npartition_samples):
        """
        Get the indices, into the flattened array of all samples/values,
        of each indpendent sample partition
        """
        splits = np.hstack(
            (0, np.cumsum(npartition_samples.numpy()))).astype(int)
        return splits

    def generate_samples_per_model(self, rvs, npilot_samples=0):
        ntotal_independent_samples = self._rounded_npartition_samples.sum()
        partition_splits = self._get_partition_splits(
            self._rounded_npartition_samples)
        samples = rvs(ntotal_independent_samples)
        samples_per_model = []
        for ii in range(self.nmodels):
            active_partitions = np.where(self.partitions_per_model[ii])[0]
            samples_per_model.append(np.hstack([
                samples[:, partition_splits[idx]:partition_splits[idx+1]]
                for idx in active_partitions]))
        if npilot_samples == 0:
            return samples_per_model

        if (self.partitions_per_model[0] *
                self._rounded_npartition_samples).max() < npilot_samples:
            msg = "Insert pilot samples currently only supported when only"
            msg += " the largest subset of those containing the "
            msg += "high-fidelity model can fit all pilot samples. "
            msg += "npilot = {0} != {1}".format(
                npilot_samples,
                (self.partitions_per_model[0] *
                 self._rounded_npartition_samples).max())
            raise ValueError(msg)
        return self._remove_pilot_samples(
            npilot_samples, samples_per_model)[0]

    def _sample_splits_per_model(self):
        # for each model get the sample splits in values_per_model
        # that correspond to each partition used in values_per_model.
        # If the model is not evaluated for a partition, then
        # the splits will be [-1, -1]
        partition_splits = self._get_partition_splits(
            self._rounded_npartition_samples)
        splits_per_model = []
        for ii in range(self.nmodels):
            active_partitions = np.where(self.partitions_per_model[ii])[0]
            splits = np.full((self.npartitions, 2), -1, dtype=int)
            lb, ub = 0, 0
            for ii, idx in enumerate(active_partitions):
                ub += partition_splits[idx+1]-partition_splits[idx]
                splits[idx] = [lb, ub]
                lb = ub
            splits_per_model.append(splits)
        return splits_per_model

    def _separate_values_per_model(self, values_per_model):
        if len(values_per_model) != self.nmodels:
            msg = "len(values_per_model) {0} != nmodels {1}".format(
                len(values_per_model), self.nmodels)
            raise ValueError(msg)
        for ii in range(self.nmodels):
            if (values_per_model[ii].shape[0] !=
                    self._rounded_nsamples_per_model[ii]):
                msg = "{0} != {1}".format(
                    "len(values_per_model[{0}]): {1}".format(
                        ii, values_per_model[ii].shape[0]),
                    "nsamples_per_model[{0}]: {1}".format(
                        ii, self._rounded_nsamples_per_model[ii]))
                raise ValueError(msg)

        values_per_subset = []
        for ii, subset in enumerate(self.subsets):
            values = []
            active_partitions = np.where(self.allocation_mat[ii])[0]
            for model_id in subset:
                splits = self._opt_sample_splits[model_id]
                values.append(np.vstack([
                    values_per_model[model_id][
                        splits[idx, 0]:splits[idx, 1], :]
                    for idx in active_partitions]))
            values_per_subset.append(np.hstack(values))
        return values_per_subset

    def _estimate(self, values_per_subset):
        return _grouped_acv_estimate(
            self.nmodels, self._optimized_sigma, self._reg_blue, self.subsets,
            values_per_subset, self.R, self._asketch)

    def __call__(self, values_per_model):
        values_per_subset = self._separate_values_per_model(values_per_model)
        return self._estimate(values_per_subset)

    def _reduce_model_sample_splits(
            self, model_id, partition_id, nsamples_to_reduce):
        """ return splits that occur when removing N samples of
        a partition of a given model"""
        lb, ub = self._opt_sample_splits[model_id][partition_id]
        sample_splits = self._opt_sample_splits[model_id].copy()
        sample_splits[partition_id][0] = (lb+nsamples_to_reduce)
        removed_split = lb, lb+nsamples_to_reduce
        return sample_splits, removed_split

    def _remove_pilot_samples(self, npilot_samples, samples_per_model):
        active_hf_subsets = np.where(self.partitions_per_model[0] == 1)[0]
        partition_id = active_hf_subsets[np.argmax(
            self._rounded_npartition_samples[active_hf_subsets])]
        removed_samples = None
        for model_id in self.subsets[partition_id]:
            if (npilot_samples >
                    self._rounded_npartition_samples[partition_id]):
                msg = "Too many pilot values {0}+>{1}".format(
                    npilot_samples,
                    self._rounded_npartition_samples[partition_id])
                raise ValueError(msg)
            if (samples_per_model[model_id].shape[1] !=
                    self._rounded_nsamples_per_model[model_id]):
                raise ValueError("samples per model has the wrong size")
            splits, removed_split = self._reduce_model_sample_splits(
                model_id, partition_id, npilot_samples)
            # removed samples must be computed before samples_per_model is
            # redefined below
            if removed_samples is None:
                removed_samples = samples_per_model[model_id][
                    :, removed_split[0]:removed_split[1]]
            else:
                assert np.allclose(
                    removed_samples, samples_per_model[model_id][
                        :, removed_split[0]:removed_split[1]])
            samples_per_model[model_id] = np.hstack(
                [samples_per_model[model_id][:, splits[idx, 0]: splits[idx, 1]]
                 for idx in np.where(
                         self.partitions_per_model[model_id] == 1)[0]])
        return samples_per_model, removed_samples

    def insert_pilot_values(self, pilot_values, values_per_model):
        npilot_values = pilot_values[0].shape[0]
        if (self.partitions_per_model[0] *
                self._rounded_npartition_samples).max() < npilot_values:
            msg = "Insert pilot samples currently only supported when only"
            msg += " the largest subset of those containing the "
            msg += "high-fidelity model can fit all pilot samples"
            raise ValueError(msg)

        new_values_per_model = [v.copy() for v in values_per_model]
        active_hf_subsets = np.where(self.partitions_per_model[0] == 1)[0]
        partition_id = active_hf_subsets[np.argmax(
            self._rounded_npartition_samples[active_hf_subsets])]
        for model_id in self.subsets[partition_id]:
            npilot_values = pilot_values[model_id].shape[0]
            if npilot_values != pilot_values[0].shape[0]:
                msg = "Must have the same number of pilot values "
                msg += "for each model"
                raise ValueError(msg)
            if (npilot_values >
                    self._rounded_npartition_samples[partition_id]):
                raise ValueError("Too many pilot values {0}>{1}".format(
                    npilot_values+values_per_model[model_id].shape[0],
                    self._rounded_npartition_samples[partition_id]))
            lb, ub = self._opt_sample_splits[model_id][partition_id]
            # Pilot samples become first samples of the chosen partition
            new_values_per_model[model_id] = np.vstack((
                values_per_model[model_id][:lb], pilot_values[model_id],
                values_per_model[model_id][lb:]))
        return new_values_per_model

    def __repr__(self):
        if self._optimized_criteria is None:
            return "{0}()".format(
                self.__class__.__name__)
        rep = "{0}(criteria={1:.3g}".format(
            self.__class__.__name__, self._optimized_criteria)
        rep += " target_cost={0:.5g}, nsamples={1})".format(
            self._rounded_target_cost,
            self._rounded_nsamples_per_model)
        return rep


class MLBLUEEstimator(GroupACVEstimator):
    def __init__(self, stat, costs, reg_blue=0, subsets=None,
                 asketch=None):
        # Currently stats is ignored.
        super().__init__(stat, costs, reg_blue, subsets, est_type="is",
                         asketch=asketch)
        self._best_model_indices = np.arange(len(costs))

        # compute psi blocks once and store because they are independent
        # of the number of samples per partition/subset
        self._psi_blocks = self._compute_psi_blocks()
        self._psi_blocks_flat = np.hstack(
                [b.flatten()[:, None] for b in self._psi_blocks])

        self._obj_jac = True

    def _compute_psi_blocks(self):
        submats = []
        for ii, subset in enumerate(self.subsets):
            R = _restriction_matrix(self.nmodels, subset)
            submat = np.linalg.multi_dot((
                R.T,
                np.linalg.pinv(self._cov[np.ix_(subset, subset)]),
                R))
            submats.append(submat)
        return submats

    def _psi_matrix(self, npartition_samples_np):
        psi = np.identity(self.nmodels)*self._reg_blue
        psi += (self._psi_blocks_flat@npartition_samples_np).reshape(
            (self.nmodels, self.nmodels))
        # for ii, submat in enumerate(self._psi_blocks):
        #    psi += npartition_samples_np[ii]*submat
        return psi

    def _objective(self, npartition_samples_np, return_grad=True):
        # leverage block diagonal structure to compute gradients efficiently
        psi = self._psi_matrix(npartition_samples_np)
        try:
            psi_inv = np.linalg.inv(psi)
        except np.linalg.LinAlgError:
            # sometimes nelder mead tries values that violate constraints
            # so return large value here
            assert not return_grad
            return 9e16
        variance = np.linalg.multi_dot(
            (self._asketch.T, psi_inv, self._asketch))
        if not return_grad:
            return variance
        aT_psi_inv = self._asketch.T.numpy().dot(psi_inv)
        grad = np.array(
            [-np.linalg.multi_dot((aT_psi_inv, smat, aT_psi_inv.T))[0, 0]
             for smat in self._psi_blocks])
        return variance, grad

    def _cvxpy_psi(self, nsps_cvxpy):
        Psi = self._psi_blocks_flat@nsps_cvxpy
        Psi = cvxpy.reshape(Psi, (self.nmodels, self.nmodels))
        return Psi

    def _cvxpy_spd_constraint(self, nsps_cvxpy, t_cvxpy):
        Psi = self._cvxpy_psi(nsps_cvxpy)
        mat = cvxpy.bmat(
            [[Psi, self._asketch],
             [self._asketch.T, cvxpy.reshape(t_cvxpy, (1, 1))]])
        return mat

    def _minimize_cvxpy(self, target_cost, min_nhf_samples, min_nlf_samples):
        # use notation from https://www.cvxpy.org/examples/basic/sdp.html

        t_cvxpy = cvxpy.Variable(nonneg=True)
        nsps_cvxpy = cvxpy.Variable(self.nsubsets, nonneg=True)
        obj = cvxpy.Minimize(t_cvxpy)
        self._validate_target_cost_min_nhf_samples(
            target_cost, min_nhf_samples)
        constraints = [self.subset_costs@nsps_cvxpy <= target_cost]
        constraints += [
            self.partitions_per_model[0]@nsps_cvxpy >= min_nhf_samples]
        if min_nlf_samples is not None:
            constraints += [
                self.partitions_per_model[ii+1]@nsps_cvxpy >=
                min_nlf_samples[ii] for ii in range(self.nmodels-1)]
        constraints += [self._cvxpy_spd_constraint(
            nsps_cvxpy, t_cvxpy) >> 0]
        prob = cvxpy.Problem(obj, constraints)
        prob.solve(verbose=0, solver="CVXOPT")
        res = dict([("x",  nsps_cvxpy.value), ("fun", t_cvxpy.value)])
        if res["fun"] is None:
            raise RuntimeError("solver did not converge")
        return res

    def allocate_samples(self, target_cost,
                         constraint_reg=0, round_nsamples=True,
                         init_guess=None, min_nhf_samples=1,
                         min_nlf_samples=None, optim_options={}):
        optim_options_copy = optim_options.copy()
        method = optim_options_copy.get(
            "method", "cvxpy" if _cvx_available else "trust-constr")
        if method == "cvxpy":
            if not _cvx_available:
                raise ImportError("must install cvxpy")
            res = self._minimize_cvxpy(
                target_cost, min_nhf_samples, min_nlf_samples)
            return self._set_optimized_params(
                asarray(res["x"]), round_nsamples)
        # TODO
        # when running mlblue with trust-constr make sure to compute
        # contraint jacobians and activate them
        # TODO put all keyword args into optim_options
        return super().allocate_samples(
            target_cost, constraint_reg, round_nsamples,
            init_guess, min_nhf_samples, min_nlf_samples, optim_options_copy)

    def estimate_all_means(self, values_per_subset):
        asketch = copy(self._asketch)
        means = np.empty(self.nmodels)
        for ii in range(self.nmodels):
            self._asketch = full((self.nmodels), 0.)
            self._asketch[ii] = 1.0
            means[ii] = self._estimate(values_per_subset)
        self._asketch = asketch
        return means


#cvxpy requires cmake
#on osx with M1 chip install via
#arch -arm64 brew install cmake
#must also install cvxopt via
#pip install cvxopt
