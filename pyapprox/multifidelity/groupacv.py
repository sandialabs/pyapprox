from abc import ABC, abstractmethod
from itertools import combinations

import torch
import numpy as np

from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.multifidelity.stats import MultiOutputMean
from pyapprox.interface.model import Model
from pyapprox.optimization.pya_minimize import (
    Constraint, ConstrainedOptimizer, OptimizationResult, ChainedOptimizer,
    Optimizer
)


def get_model_subsets(nmodels, bkd, max_subset_nmodels=None):
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
    model_indices = bkd.arange(nmodels)
    for nsubset_lfmodels in range(1, max_subset_nmodels+1):
        for subset_indices in combinations(
                model_indices, nsubset_lfmodels):
            idx = bkd.asarray(subset_indices, dtype=int)
            subsets.append(idx)
    return subsets


def _get_allocation_matrix_is(subsets, bkd):
    nsubsets = len(subsets)
    npartitions = nsubsets
    allocation_mat = bkd.full(
        (nsubsets, npartitions), 0., dtype=torch.double)
    for ii, subset in enumerate(subsets):
        allocation_mat[ii, ii] = 1.0
    return allocation_mat


def _get_allocation_matrix_nested(subsets, bkd):
    # nest partitions according to order of subsets
    nsubsets = len(subsets)
    npartitions = nsubsets
    allocation_mat = bkd.full(
        (nsubsets, npartitions), 0., dtype=torch.double)
    for ii, subset in enumerate(subsets):
        allocation_mat[ii, :ii+1] = 1.0
    return allocation_mat


def _nest_subsets(subsets, nmodels, bkd):
    for subset in subsets:
        if np.allclose(subset, [0]):
            raise ValueError("Cannot use subset [0]")
    idx = sorted(
        list(range(len(subsets))),
        key=lambda ii: (len(subsets[ii]), tuple(nmodels-subsets[ii])),
        reverse=True)
    return [subsets[ii] for ii in idx], bkd.array(idx)


def _grouped_acv_beta(nmodels, Sigma, subsets, R, reg, asketch, bkd):
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
    reg_mat = bkd.eye(nmodels)*reg
    if asketch.shape != (nmodels, 1):
        raise ValueError("asketch has the wrong shape")

    # TODO instead of applyint R matrices just collect correct rows and columns
    beta = bkd.multidot((
        bkd.pinv(Sigma), R.T,
        bkd.solve(bkd.multidot(
            (R, bkd.pinv(Sigma), R.T))+reg_mat, asketch[:, 0])))
    return beta


def _grouped_acv_estimate(
        nmodels, Sigma, reg, subsets, subset_values, R, asketch, bkd):
    nsubsets = len(subsets)
    beta = _grouped_acv_beta(nmodels, Sigma, subsets, R, reg, asketch, bkd)
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
        nsamples_subset1, cov, bkd):
    nsubset0 = len(subset0)
    nsubset1 = len(subset1)
    block = bkd.full((nsubset0, nsubset1), 0.)
    if (nsamples_subset0*nsamples_subset1) == 0:
        return block
    block = cov[np.ix_(subset0, subset1)]*nsamples_intersect/(
                nsamples_subset0*nsamples_subset1)
    return block


def _grouped_acv_sigma(
        nmodels, nsamples_intersect, cov, subsets, bkd):
    nsubsets = len(subsets)
    Sigma = [[None for jj in range(nsubsets)] for ii in range(nsubsets)]
    for ii, subset0 in enumerate(subsets):
        N_ii = nsamples_intersect[ii, ii]
        Sigma[ii][ii] = _grouped_acv_sigma_block(
            subset0, subset0, N_ii, N_ii, N_ii, cov, bkd)
        for jj, subset1 in enumerate(subsets[:ii]):
            N_jj = nsamples_intersect[jj, jj]
            Sigma[ii][jj] = _grouped_acv_sigma_block(
                subset0, subset1, nsamples_intersect[ii, jj],
                N_ii, N_jj, cov, bkd)
            Sigma[jj][ii] = Sigma[ii][jj].T
    Sigma = bkd.vstack([bkd.hstack(row) for row in Sigma])
    return Sigma


class GroupACVEstimator:
    def __init__(self, stat, costs, reg_blue=0, subsets=None,
                 est_type="is", asketch=None, backend=TorchLinAlgMixin):
        self._bkd = backend
        self._cov, self._costs = self._check_cov(stat._cov, costs)
        self._nmodels = len(costs)
        self._reg_blue = reg_blue
        if not isinstance(stat, MultiOutputMean):
            raise ValueError(
                "MLBLUE currently only suppots estimation of means")
        self._stat = stat

        self._subsets, self._allocation_mat = self._set_subsets(
            subsets, est_type)
        self._npartitions = self._allocation_mat.shape[1]
        self._partitions_per_model = self._get_partitions_per_model()
        self._partitions_intersect = (
            self._get_subset_intersecting_partitions())
        self._R = self._bkd.hstack(
            [
                self._restriction_matrix(self.nmodels(), subset).T
                for ii, subset in enumerate(self._subsets)
            ]
        )
        # set npatition_samples above small constant,
        # otherwise gradient will not be defined.
        self._npartition_samples_lb = 0  # 1e-5
        self._optimized_criteria = None
        self._asketch = self._validate_asketch(asketch)
        self._objective = None

    def nsubsets(self):
        return len(self._subsets)

    def npartitions(self):
        return self._npartitions

    def nmodels(self):
        return self._nmodels

    def _restriction_matrix(self, ncols, subset):
        # TODO Consider replacing _restriction_matrix.T.dot(A) with
        # special indexing applied to A
        nsubset = len(subset)
        mat = self._bkd.zeros((nsubset, ncols))
        for ii in range(nsubset):
            mat[ii, subset[ii]] = 1.0
        return mat

    def _check_cov(self, cov, costs):
        if cov.shape[0] != len(costs):
            print(cov.shape, costs.shape)
            raise ValueError("cov and costs are inconsistent")
        return cov, self._bkd.asarray(costs)

    def _set_subsets(self, subsets, est_type):
        if subsets is None:
            subsets = get_model_subsets(self.nmodels(), self._bkd)
        if est_type == "is":
            get_allocation_mat = _get_allocation_matrix_is
        elif est_type == "nested":
            zero = self._bkd.zeros((1,), dtype=int)
            for ii, subset in enumerate(subsets):
                if self._bkd.allclose(subset, zero):
                    del subsets[ii]
                    break
            subsets = _nest_subsets(subsets, self.nmodels(), self._bkd)[0]
            get_allocation_mat = _get_allocation_matrix_nested
        else:
            raise ValueError(
                "incorrect est_type {0} specified".format(est_type))
        return subsets,  get_allocation_mat(subsets, self._bkd)

    def _get_partitions_per_model(self):
        # assume npartitions = nsubsets
        npartitions = self._allocation_mat.shape[1]
        partitions_per_model = self._bkd.full(
            (self.nmodels(), npartitions), 0.
        )
        for ii, subset in enumerate(self._subsets):
            partitions_per_model[
                np.ix_(subset, self._allocation_mat[ii] == 1)] = 1
        return partitions_per_model

    def _compute_nsamples_per_model(self, npartition_samples):
        nsamples_per_model = self._bkd.einsum(
            "ji,i->j", self._partitions_per_model, npartition_samples)
        return nsamples_per_model

    def _estimator_cost(self, npartition_samples):
        return sum(
            self._costs*self._compute_nsamples_per_model(npartition_samples))

    def _get_subset_intersecting_partitions(self):
        amat = self._allocation_mat
        npartitions = self._allocation_mat.shape[1]
        partition_intersect = self._bkd.full(
            (self.nsubsets(), self.nsubsets(), npartitions), 0.)
        for ii, subset_ii in enumerate(self._subsets):
            for jj, subset_jj in enumerate(self._subsets):
                # partitions are shared when sum of allocation entry is 2
                partition_intersect[ii, jj, amat[ii]+amat[jj] == 2] = 1.
        return partition_intersect

    def _nintersect_samples(self, npartition_samples):
        """
        Get the number of samples in the intersection of two subsets.

        Note the number of samples per subset is simply the diagonal of this
        matrix
        """
        return self._bkd.einsum(
            "ijk,k->ij", self._partitions_intersect, npartition_samples)

    def _sigma(self, npartition_samples):
        return _grouped_acv_sigma(
            self.nmodels(), self._nintersect_samples(npartition_samples),
            self._cov, self._subsets, self._bkd)

    def _psi_matrix_from_sigma(self, Sigma):
        # TODO instead of applyint R matrices just collect correct rows and columns
        reg_mat = self._bkd.eye(self.nmodels())*self._reg_blue
        return self._bkd.multidot(
             (self._R, self._bkd.pinv(Sigma), self._R.T))+reg_mat

    def _psi_matrix(self, npartition_samples):
        Sigma = self._sigma(npartition_samples)
        return self._psi_matrix_from_sigma(Sigma)

    def _covariance_from_npartition_samples(self, npartition_samples):
        if self._asketch.shape != (self.nmodels(), 1):
            raise ValueError("asketch has the wrong shape")
        psi_inv = self._bkd.pinv(self._psi_matrix(npartition_samples))
        return self._bkd.multidot(
            (self._asketch.T, psi_inv, self._asketch)
        )

    def _get_model_subset_costs(self, subsets, costs):
        subset_costs = self._bkd.array(
            [costs[subset].sum() for subset in subsets])
        return subset_costs

    def _nelder_mead_min_nlf_samples_constraint(
            self, x, min_nlf_samples, ii):
        return (self._partitions_per_model[ii].numpy()*x).sum()-min_nlf_samples

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
            assert len(min_nlf_samples) == self.nmodels()-1
            for ii in range(1, self.nmodels()):
                cons += [
                    {'type': 'ineq',
                     'fun': self._nelder_mead_min_nlf_samples_constraint,
                     'args': [min_nlf_samples, ii, ]}]
        return cons

    def _init_guess(self, target_cost):
        # start with the same number of samples per partition

        # get the number of samples per model when 1 sample is in each
        # partition
        nsamples_per_model = self._compute_nsamples_per_model(
            self._bkd.full((self.npartitions(),), 1.))
        # nsamples_per_model[0] = max(0, min_nhf_samples)
        cost = (nsamples_per_model*self._costs).sum()

        # the total number of samples per partition is then target_cost/cost
        # we take the floor to make sure we do not exceed the target cost
        return self._bkd.full(
            (self.npartitions(),), self._bkd.floor(target_cost/cost))

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
        # expected scalar type Double but found Float error can occur
        # with torch if npartition samples is not torch.double need to
        # think of a check that works for all backends
        if round_nsamples:
            rounded_npartition_samples = self._bkd.floor(npartition_samples)
        else:
            rounded_npartition_samples = npartition_samples
        self._set_optimized_params_base(
            rounded_npartition_samples,
            self._compute_nsamples_per_model(rounded_npartition_samples),
            self._estimator_cost(rounded_npartition_samples))

    def _validate_asketch(self, asketch):
        if asketch is None:
            asketch = self._bkd.full((self.nmodels(), 1), 0)
            asketch[0] = 1.0
        asketch = self._bkd.array(asketch)
        if asketch.shape[0] != self._costs.shape[0]:
            raise ValueError("aksetch has the wrong shape")
        if asketch.ndim == 1:
            asketch = asketch[:, None]
        return asketch

    def set_optimizer(self, optimizer):
        if (
                not isinstance(optimizer, GroupACVOptimizer)
                and not isinstance(optimizer, ChainedACVOptimizer)
        ):
            raise ValueError(
                "optimizer must be instance of GroupACVOptimizer"
                "or ChainedACVOptimizer"
            )
        self._optimizer = optimizer

    def allocate_samples(
            self, target_cost, min_nhf_samples=1, round_nsamples=True,
            iterate=None,
    ):
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
        if self._optimizer is None:
            raise RuntimeError("must call set_optimizer")
        self._optimizer.set_budget(target_cost, min_nhf_samples)
        result = self._optimizer.minimize(iterate)

        if not result.success or self._bkd.any(result.x < 0):
            raise RuntimeError("optimization not successful")

        self._set_optimized_params(result.x[:, 0], round_nsamples)

    def _get_partition_splits(self, npartition_samples):
        """
        Get the indices, into the flattened array of all samples/values,
        of each indpendent sample partition
        """
        splits = self._bkd.hstack(
            (
                self._bkd.zeros((1,), dtype=int),
                self._bkd.cumsum(npartition_samples, dtype=int)
            )
        )
        return splits

    def generate_samples_per_model(self, rvs, npilot_samples=0):
        ntotal_independent_samples = self._rounded_npartition_samples.sum()
        partition_splits = self._get_partition_splits(
            self._rounded_npartition_samples)
        samples = rvs(ntotal_independent_samples)
        samples_per_model = []
        for ii in range(self.nmodels()):
            active_partitions = self._bkd.where(
                self._partitions_per_model[ii]
            )[0]
            samples_per_model.append(self._bkd.hstack([
                samples[:, partition_splits[idx]:partition_splits[idx+1]]
                for idx in active_partitions]))
        if npilot_samples == 0:
            return samples_per_model

        if (self._partitions_per_model[0] *
                self._rounded_npartition_samples).max() < npilot_samples:
            msg = "Insert pilot samples currently only supported when only"
            msg += " the largest subset of those containing the "
            msg += "high-fidelity model can fit all pilot samples. "
            msg += "npilot = {0} != {1}".format(
                npilot_samples,
                (self._partitions_per_model[0] *
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
        for ii in range(self.nmodels()):
            active_partitions = self._bkd.where(
                self._partitions_per_model[ii]
            )[0]
            splits = self._bkd.full((self.npartitions(), 2), -1, dtype=int)
            lb, ub = 0, 0
            for ii, idx in enumerate(active_partitions):
                ub += partition_splits[idx+1]-partition_splits[idx]
                splits[idx] = self._bkd.array([lb, ub])
                lb = self._bkd.copy(ub)
            splits_per_model.append(splits)
        return splits_per_model

    def _separate_values_per_model(self, values_per_model):
        if len(values_per_model) != self.nmodels():
            msg = "len(values_per_model) {0} != nmodels {1}".format(
                len(values_per_model), self.nmodels())
            raise ValueError(msg)
        for ii in range(self.nmodels()):
            if (values_per_model[ii].shape[0] !=
                    self._rounded_nsamples_per_model[ii]):
                msg = "{0} != {1}".format(
                    "len(values_per_model[{0}]): {1}".format(
                        ii, values_per_model[ii].shape[0]),
                    "nsamples_per_model[{0}]: {1}".format(
                        ii, self._rounded_nsamples_per_model[ii]))
                raise ValueError(msg)

        values_per_subset = []
        for ii, subset in enumerate(self._subsets):
            values = []
            active_partitions = self._bkd.where(self._allocation_mat[ii])[0]
            for model_id in subset:
                splits = self._opt_sample_splits[model_id]
                values.append(self._bkd.vstack([
                    values_per_model[model_id][
                        splits[idx, 0]:splits[idx, 1], :]
                    for idx in active_partitions]))
            values_per_subset.append(self._bkd.hstack(values))
        return values_per_subset

    def _grouped_acv_beta(self, sigma):
        psi_matrix = self._psi_matrix_from_sigma(sigma)
        beta = self._bkd.multidot((
            self._bkd.pinv(sigma), self._R.T,
            self._bkd.solve(psi_matrix, self._asketch[:, 0])))
        return beta

    def _estimate(self, values_per_subset):
        beta = self._grouped_acv_beta(self._optimized_sigma)
        ll, mm = 0, 0
        acv_mean = 0
        for kk in range(self.nsubsets()):
            mm += len(self._subsets[kk])
            if values_per_subset[kk].shape[0] > 0:
                subset_mean = values_per_subset[kk].mean(axis=0)
                acv_mean += (beta[ll:mm]) @ subset_mean
            ll = mm
        return acv_mean

    def __call__(self, values_per_model):
        values_per_subset = self._separate_values_per_model(values_per_model)
        return self._estimate(values_per_subset)

    def _reduce_model_sample_splits(
            self, model_id, partition_id, nsamples_to_reduce):
        """ return splits that occur when removing N samples of
        a partition of a given model"""
        lb, ub = self._opt_sample_splits[model_id][partition_id]
        sample_splits = self._bkd.copy(self._opt_sample_splits[model_id])
        sample_splits[partition_id][0] = (lb+nsamples_to_reduce)
        removed_split = lb, lb+nsamples_to_reduce
        return sample_splits, removed_split

    def _remove_pilot_samples(self, npilot_samples, samples_per_model):
        active_hf_subsets = self._bkd.where(
            self._partitions_per_model[0] == 1
        )[0]
        partition_id = active_hf_subsets[self._bkd.argmax(
            self._rounded_npartition_samples[active_hf_subsets])]
        removed_samples = None
        for model_id in self._subsets[partition_id]:
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
                assert self._bkd.allclose(
                    removed_samples, samples_per_model[model_id][
                        :, removed_split[0]:removed_split[1]])
            samples_per_model[model_id] = self._bkd.hstack(
                [samples_per_model[model_id][:, splits[idx, 0]: splits[idx, 1]]
                 for idx in self._bkd.where(
                         self._partitions_per_model[model_id] == 1)[0]])
        return samples_per_model, removed_samples

    def insert_pilot_values(self, pilot_values, values_per_model):
        npilot_values = pilot_values[0].shape[0]
        if (self._partitions_per_model[0] *
                self._rounded_npartition_samples).max() < npilot_values:
            msg = "Insert pilot samples currently only supported when only"
            msg += " the largest subset of those containing the "
            msg += "high-fidelity model can fit all pilot samples"
            raise ValueError(msg)

        new_values_per_model = [self._bkd.copy(v) for v in values_per_model]
        active_hf_subsets = self._bkd.where(
            self._partitions_per_model[0] == 1
        )[0]
        partition_id = active_hf_subsets[self._bkd.argmax(
            self._rounded_npartition_samples[active_hf_subsets])]
        for model_id in self._subsets[partition_id]:
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
            new_values_per_model[model_id] = self._bkd.vstack((
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
        self._best_model_indices = self._bkd.arange(len(costs))

        # compute psi blocks once and store because they are independent
        # of the number of samples per partition/subset
        self._psi_blocks = self._compute_psi_blocks()
        self._psi_blocks_flat = self._bkd.hstack(
                [b.flatten()[:, None] for b in self._psi_blocks])

        self._obj_jac = True

    def _compute_psi_blocks(self):
        submats = []
        for ii, subset in enumerate(self._subsets):
            R = self._restriction_matrix(self.nmodels(), subset)
            submat = self._bkd.multidot((
                R.T,
                self._bkd.pinv(self._cov[np.ix_(subset, subset)]),
                R))
            submats.append(submat)
        return submats

    def _psi_matrix(self, npartition_samples):
        psi = self._bkd.eye(self.nmodels())*self._reg_blue
        psi += (self._psi_blocks_flat @ npartition_samples).reshape(
            (self.nmodels(), self.nmodels()))
        return psi

    def estimate_all_means(self, values_per_subset):
        asketch = self._bkd.copy(self._asketch)
        means = self._bkd.empty(self.nmodels())
        for ii in range(self.nmodels()):
            self._asketch = self._bkd.full((self.nmodels()), 0.)
            self._asketch[ii] = 1.0
            means[ii] = self._estimate(values_per_subset)
        self._asketch = asketch
        return means


class GroupACVObjective(Model):
    def __init__(self):
        super().__init__()
        self._est = None
        self._bkd = None

    def nqoi(self):
        return 1

    def set_estimator(self, estimator):
        self._est = estimator
        self._bkd = self._est._bkd
        self._jacobian_implemented = self._bkd.jacobian_implemented()
        self._hessian_implemented = self._bkd.hessian_implemented()

    def _values(self, npartition_samples):
        return self._est._covariance_from_npartition_samples(
            npartition_samples[:, 0]
        )

    def _jacobian(self, npartition_samples):
        return self._bkd.grad(
            self._est._covariance_from_npartition_samples,
            npartition_samples[:, 0],
        )[1]

    def _hessian(self, npartition_samples):
        return self._bkd.hessian(
            self._est._covariance_from_npartition_samples,
            npartition_samples[:, 0],
        )[None, ...]


class GroupACVConstraint(Constraint):
    def __init__(self, bounds, keep_feasible=True):
        super().__init__(bounds, keep_feasible)
        self._est = None
        # self._jacobian_implemented = self. jacobian_implemented()

    def set_estimator(self, estimator):
        self._est = estimator
        self._bkd = self._est._bkd


class GroupACVCostContstraint(GroupACVConstraint):
    def __init__(self, bounds, keep_feasible=True):
        if bounds.shape[0] != self.nqoi():
            # the number of columns is checked with call to super().__init__
            raise ValueError("Bounds must have shape (2, 2)")
        super().__init__(bounds, keep_feasible)
        self._target_cost = None
        self._min_nhf_samples = None
        self._jacobian_implemented = True
        self._hessian_implemented = True

    def set_budget(self, target_cost, min_nhf_samples):
        self._target_cost = target_cost
        self._min_nhf_samples = min_nhf_samples
        self._validate_target_cost_min_nhf_samples()

    def _validate_target_cost_min_nhf_samples(self):
        lb = self._min_nhf_samples*self._est._costs[0]
        ub = self._target_cost
        if ub < lb:
            msg = "target_cost {0} & cost of min_nhf_samples {1} ".format(
                self._target_cost, lb
            )
            msg += "are inconsistent"
            raise ValueError(msg)

    def nqoi(self):
        return 2

    def _values(self, npartition_samples):
        return self._bkd.array(
            [
                self._target_cost-self._est._estimator_cost(
                    npartition_samples[:, 0]
                ),
                self._bkd.sum(
                    self._est._partitions_per_model[0]*npartition_samples[:, 0]
                ) - self._min_nhf_samples
            ]
        )[None, :]

    def _jacobian(self, npartition_samples):
        return self._bkd.vstack(
            (
                -(self._est._costs[None, :] @ self._est._partitions_per_model),
                self._est._partitions_per_model[0][None, :]
            )
        )

    def _hessian(self, npartition_samples):
        return self._bkd.zeros(
            (
                self.nqoi(),
                npartition_samples.shape[0],
                npartition_samples.shape[0]
             )
        )


class GroupACVOptimizer(Optimizer):
    def __init__(self):
        super().__init__()
        self._target_cost = None
        self._min_nhf_samples = None
        self._est = None

    def set_budget(self, target_cost, min_nhf_samples=1):
        self._target_cost = target_cost
        self._min_nhf_samples = min_nhf_samples

    def set_estimator(self, est):
        self._est = est
        self._bkd = self._est._bkd

    @abstractmethod
    def _minimize(self, iterate):
        raise NotImplementedError

    def minimize(self, iterate):
        result = self._minimize(iterate)
        if not isinstance(result, OptimizationResult):
            raise RuntimeError(
                "{0}.minimize did not return OptimizationResult".format(self)
            )
        return result


class GroupACVGradientOptimizer(GroupACVOptimizer):
    def __init__(self, optimizer):
        super().__init__()
        if not isinstance(optimizer, ConstrainedOptimizer):
            raise ValueError(
                "optimizer must be an instance of ConstrainedOptimizer"
            )
        self._optimizer = optimizer
        self._constraint = None

    def _minimize(self, iterate):
        return self._optimizer.minimize(iterate)

    def set_budget(self, target_cost, min_nhf_samples=1):
        super().set_budget(target_cost, min_nhf_samples)
        self._constraint.set_budget(target_cost, min_nhf_samples)

    def set_estimator(self, est):
        super().set_estimator(est)
        objective = GroupACVObjective()
        objective.set_estimator(self._est)
        self._optimizer.set_objective_function(objective)
        self._constraint = GroupACVCostContstraint(
            bounds=self._bkd.array([[0, np.inf], [0, np.inf]])
        )
        self._constraint.set_estimator(self._est)
        self._optimizer.set_constraints([self._constraint])
        self._optimizer.set_bounds(
            self._bkd.reshape(
                self._bkd.repeat(
                    self._bkd.array([0, np.inf]), self._est.npartitions()),
                (self._est.npartitions(), 2)
            )
        )


class MLBLUESPDOptimizer(GroupACVOptimizer):
    def __init__(self):
        try:
            import cvxpy
        except ImportError:
            raise ValueError(
                "MLBLUESPDOptimizer can only be used when optinal dependency"
                "cvxpy is installed")
        self._cvxpy = cvxpy

        super().__init__()
        self._min_nlf_samples = None

    def _cvxpy_psi(self, nsps_cvxpy):
        Psi = self._est._psi_blocks_flat@nsps_cvxpy
        Psi = self._cvxpy.reshape(
            Psi, (self._est.nmodels(), self._est.nmodels())
        )
        return Psi

    def _cvxpy_spd_constraint(self, nsps_cvxpy, t_cvxpy):
        Psi = self._cvxpy_psi(nsps_cvxpy)
        mat = self._cvxpy.bmat(
            [[Psi, self._est._asketch],
             [self._est._asketch.T, self._cvxpy.reshape(t_cvxpy, (1, 1))]])
        return mat

    def _minimize(self, iterate):
        if iterate is not None:
            raise ValueError("iterate must be None")
        t_cvxpy = self._cvxpy.Variable(nonneg=True)
        nsps_cvxpy = self._cvxpy.Variable(self._est.nsubsets(), nonneg=True)
        obj = self._cvxpy.Minimize(t_cvxpy)
        subset_costs = self._est._get_model_subset_costs(
            self._est._subsets, self._est._costs)
        constraints = [subset_costs@nsps_cvxpy <= self._target_cost]
        constraints += [
            self._est._partitions_per_model[0]@nsps_cvxpy
            >= self._min_nhf_samples
        ]
        if self._min_nlf_samples is not None:
            constraints += [
                self._est._partitions_per_model[ii+1]@nsps_cvxpy >=
                self._min_nlf_samples[ii] for ii in range(self.nmodels()-1)]
        constraints += [self._cvxpy_spd_constraint(
            nsps_cvxpy, t_cvxpy) >> 0]
        prob = self._cvxpy.Problem(obj, constraints)
        prob.solve(verbose=0, solver="CVXOPT")
        if t_cvxpy.value is None:
            raise RuntimeError("solver did not converge")
        result = OptimizationResult(
            {
                "x": self._bkd.array(nsps_cvxpy.value)[:, None],
                "fun": t_cvxpy.value,
                "success": True,
            }
        )
        return result


class ChainedACVOptimizer(ChainedOptimizer):
    def __init__(self, optimizer1, optimizer2):
        if not isinstance(optimizer1, GroupACVOptimizer):
            raise ValueError(
                "optimizer1 must be an instance of GroupACVOptimizer"
            )
        if not isinstance(optimizer2, GroupACVOptimizer):
            raise ValueError(
                "optimizer2 must be an instance of GroupACVOptimizer"
            )
        super().__init__(optimizer1, optimizer2)

    def set_budget(self, target_cost, min_nhf_samples):
        self._optimizer1.set_budget(target_cost, min_nhf_samples)
        self._optimizer2.set_budget(target_cost, min_nhf_samples)

#cvxpy requires cmake
#on osx with M1 chip install via
#arch -arm64 brew install cmake
#must also install cvxopt via
#pip install cvxopt
