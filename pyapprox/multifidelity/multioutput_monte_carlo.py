import copy
import warnings
from abc import abstractmethod
from functools import partial
from itertools import combinations
from multiprocessing import Pool

import torch
import numpy as np
from scipy.optimize import minimize

from pyapprox.util.utilities import get_correlation_from_covariance
from pyapprox.multifidelity.stats import (
    MultiOutputMean, MultiOutputVariance, MultiOutputMeanAndVariance,
    _nqoi_nqoi_subproblem)
from pyapprox.multifidelity._optim import (
    _allocate_samples_mlmc,
    _allocate_samples_mfmc,
    _check_mfmc_model_costs_and_correlations,
    _cast_to_integers,
    _get_sample_allocation_matrix_mlmc,
    _get_sample_allocation_matrix_mfmc,
    _get_acv_recursion_indices)


def _combine_acv_values(reorder_allocation_mat, npartition_samples,
                        acv_values):
    r"""
    Extract the unique values from the sets
    :math:`f_\alpha(\mathcal{Z}_\alpha), `f_\alpha(\mathcal{Z}_\alpha^*)`
    for each model :math:`\alpha=0,\ldots,M`
    """
    nmodels = len(acv_values)
    values_per_model = [None for ii in range(nmodels)]
    values_per_model[0] = acv_values[0][1]
    for ii in range(1, nmodels):
        lb, ub = 0, 0
        lb2, ub2 = 0, 0
        values_per_model[ii] = []
        for jj in range(nmodels):
            found = False
            if reorder_allocation_mat[jj, 2*ii] == 1:
                ub = lb + int(npartition_samples[jj])
                values_per_model[ii] += [acv_values[ii][0][lb:ub]]
                lb = ub
                found = True
            if reorder_allocation_mat[jj, 2*ii+1] == 1:
                # there is no need to enter here is samle set has already
                # been added by acv_values[ii][0], hence the use of elseif here
                ub2 = lb2 + int(npartition_samples[jj])
                if not found:
                    values_per_model[ii] += [acv_values[ii][1][lb2:ub2]]
                lb2 = ub2
        values_per_model[ii] = np.vstack(values_per_model[ii])
    return values_per_model


def _combine_acv_samples(reorder_allocation_mat, npartition_samples,
                         acv_samples):
    r"""
    Extract the unique amples from the sets
:math:`\mathcal{Z}_\alpha, `\mathcal{Z}_\alpha^*` for each model
    :math:`\alpha=0,\ldots,M`
    """
    nmodels = len(acv_samples)
    samples_per_model = [None for ii in range(nmodels)]
    samples_per_model[0] = acv_samples[0][1]
    for ii in range(1, nmodels):
        lb, ub = 0, 0
        lb2, ub2 = 0, 0
        samples_per_model[ii] = []
        for jj in range(nmodels):
            found = False
            if reorder_allocation_mat[jj, 2*ii] == 1:
                ub = lb + int(npartition_samples[jj])
                samples_per_model[ii] += [acv_samples[ii][0][:, lb:ub]]
                lb = ub
                found = True
            if reorder_allocation_mat[jj, 2*ii+1] == 1:
                ub2 = lb2 + int(npartition_samples[jj])
                if not found:
                    # Only add samples if they were not in Z_m^*
                    samples_per_model[ii] += [acv_samples[ii][1][:, lb2:ub2]]
                lb2 = ub2
        samples_per_model[ii] = np.hstack(samples_per_model[ii])
    return samples_per_model


def _get_allocation_matrix_gmf(recursion_index):
    nmodels = len(recursion_index)+1
    mat = np.zeros((nmodels, 2*nmodels))
    for ii in range(nmodels):
        mat[ii, 2*ii+1] = 1.0
    for ii in range(1, nmodels):
        mat[:, 2*ii] = mat[:, recursion_index[ii-1]*2+1]
    for ii in range(2, 2*nmodels):
        II = np.where(mat[:, ii] == 1)[0][-1]
        mat[:II, ii] = 1.0
    return mat


def _get_allocation_matrix_acvis(recursion_index):
    nmodels = len(recursion_index)+1
    mat = np.zeros((nmodels, 2*nmodels))
    for ii in range(nmodels):
        mat[ii, 2*ii+1] = 1
    for ii in range(1, nmodels):
        mat[:, 2*ii] = mat[:, recursion_index[ii-1]*2+1]
    for ii in range(1, nmodels):
        mat[:, 2*ii+1] = np.maximum(mat[:, 2*ii], mat[:, 2*ii+1])
    return mat


def _get_allocation_matrix_acvrd(recursion_index):
    nmodels = len(recursion_index)+1
    allocation_mat = np.zeros((nmodels, 2*nmodels))
    for ii in range(nmodels):
        allocation_mat[ii, 2*ii+1] = 1
    for ii in range(1, nmodels):
        allocation_mat[:, 2*ii] = (
            allocation_mat[:, recursion_index[ii-1]*2+1])
    return allocation_mat


def log_determinant_variance(variance):
    val = torch.logdet(variance)
    return val


def determinant_variance(variance):
    return torch.det(variance)


def log_trace_variance(variance):
    val = torch.log(torch.trace(variance))
    if not torch.isfinite(val):
        raise RuntimeError("trace is negative")
    return val


def log_linear_combination_diag_variance(weights, variance):
    # must be used with partial, e.g.
    # opt_criteria = partial(log_linear_combination_diag_variance, weights)
    return torch.log(torch.multi_dot(weights, torch.diag(variance)))


class MCEstimator():
    def __init__(self, stat, costs, cov, opt_criteria=None):
        r"""
        Parameters
        ----------
        stat : :class:`~pyapprox.multifidelity.multioutput_monte_carlo.MultiOutputStatistic`
            Object defining what statistic will be calculated

        costs : np.ndarray (nmodels)
            The relative costs of evaluating each model

        cov : np.ndarray (nmodels*nqoi, nmodels)
            The covariance C between each of the models. The highest fidelity
            model is the first model, i.e. covariance between its QoI
            is cov[:nqoi, :nqoi]

        opt_criteria : callable
            Function of the the covariance between the high-fidelity
            QoI estimators with signature

            ``opt_criteria(variance) -> float

            where variance is np.ndarray with size that depends on
            what statistics are being estimated. E.g. when estimating means
            then variance shape is (nqoi, nqoi), when estimating variances
            then variance shape is (nqoi**2, nqoi**2), when estimating mean
            and variance then shape (nqoi+nqoi**2, nqoi+nqoi**2)
        """
        # public variables (will be backwards compatible)
        self._stat = stat

        # private variables (no guarantee that these variables
        #                    will exist in the future)
        self._cov, self._costs, self._nmodels, self._nqoi = self._check_cov(
            cov, costs)
        self._optimization_criteria = self._set_optimization_criteria(
            opt_criteria)

        self._nsamples_per_model = None
        self._rounded_npartition_samples = None
        self._rounded_target_cost = None
        self._optimized_criteria = None
        self._optimized_covariance = None
        self._model_labels = None

    def _check_cov(self, cov, costs):
        nmodels = len(costs)
        if cov.shape[0] % nmodels:
            msg = "cov.shape {0} and costs.shape {1} are inconsistent".format(
                cov.shape, costs.shape)
            raise ValueError(msg)
        return (torch.as_tensor(cov, dtype=torch.double).clone(),
                torch.as_tensor(costs, dtype=torch.double),
                nmodels, cov.shape[0]//nmodels)

    def _set_optimization_criteria(self, opt_criteria):
        if opt_criteria is None:
            opt_criteria = log_trace_variance
            # opt_criteria = log_determinant_variance
        return opt_criteria

    def _covariance_from_npartition_samples(self, npartition_samples):
        """
        Get the variance of the Monte Carlo estimator from costs and cov
        and npartition_samples
        """
        return self._stat.high_fidelity_estimator_covariance(
            npartition_samples[0])

    def optimized_covariance(self):
        """
        Return the estimator covariance at the optimal sample allocation
        computed using self.allocate_samples()
        """
        return self._optimized_covariance

    def allocate_samples(self, target_cost, verbosity=0):
        self._nsamples_per_model = np.asarray(
            [int(np.floor(target_cost/self._costs[0]))])
        self._rounded_npartition_samples = self._nsamples_per_model
        est_covariance = self._covariance_from_npartition_samples(
            self._rounded_npartition_samples)
        self._optimized_covariance = est_covariance
        optimized_criteria = self._optimization_criteria(est_covariance)
        self._rounded_target_cost = self._costs[0]*self._nsamples_per_model[0]
        self._optimized_criteria = optimized_criteria

    def generate_samples_per_model(self, rvs):
        """
        Returns the samples needed to the model

        Parameters
        ----------
        rvs : callable
            Function with signature

            `rvs(nsamples)->np.ndarray(nvars, nsamples)`

        Returns
        -------
        samples_per_model : list[np.ndarray] (1)
            List with one entry np.ndarray (nvars, nsamples_per_model[0])
        """
        return [rvs(self._nsamples_per_model)]

    def __call__(self, values):
        if not isinstance(values, np.ndarray):
            raise ValueError("values must be an np.ndarray type={0}".format(
                type(values)))
        if values.ndim != 2 or values.shape[0] != self._nsamples_per_model[0]:
            msg = "values has the incorrect shape {0} expected {1}".format(
                values.shape, (self._nsamples_per_model[0], self._nqoi))
            raise ValueError(msg)
        return self._stat.sample_estimate(values)

    def __repr__(self):
        return "{0}(stat={1}, nqoi={2})".format(
            self.__class__.__name__, self._stat, self._nqoi)


class ACVEstimator(MCEstimator):
    def __init__(self, stat, costs, cov,
                 recursion_index=None, opt_criteria=None,
                 tree_depth=None, allow_failures=False):
        """
        Constructor.

        Parameters
        ----------
        stat : :class:`~pyapprox.multifidelity.multioutput_monte_carlo.MultiOutputStatistic`
            Object defining what statistic will be calculated

        costs : np.ndarray (nmodels)
            The relative costs of evaluating each model

        cov : np.ndarray (nmodels*nqoi, nmodels)
            The covariance C between each of the models. The highest fidelity
            model is the first model, i.e. covariance between its QoI
            is cov[:nqoi, :nqoi]

        recursion_index : np.ndarray (nmodels-1)
            The recusion index that specifies which ACV estimator is used

        opt_criteria : callable
            Function of the the covariance between the high-fidelity
            QoI estimators with signature

            ``opt_criteria(variance) -> float

            where variance is np.ndarray with size that depends on
            what statistics are being estimated. E.g. when estimating means
            then variance shape is (nqoi, nqoi), when estimating variances
            then variance shape is (nqoi**2, nqoi**2), when estimating mean
            and variance then shape (nqoi+nqoi**2, nqoi+nqoi**2)

        tree_depth: integer (default=None)
            The maximum depth of the recursion tree.
            If not None, then recursion_index is ignored.

        allow_failures: boolean (default=False)
            Allow optimization of estimators to fail when enumerating
            each recursion tree. This is useful for estimators, like MFMC,
            that have optimization that enforce constraints on the structure
            of the model ensemble
        """
        super().__init__(stat, costs, cov, opt_criteria=opt_criteria)
        self._set_initial_guess(None)

        if tree_depth is not None and recursion_index is not None:
            msg = "Only tree_depth or recurusion_index must be specified"
            raise ValueError(msg)
        if tree_depth is None:
            self._set_recursion_index(recursion_index)
        self._tree_depth = tree_depth
        self._allow_failures = allow_failures

        self._rounded_partition_ratios = None
        self._rounded_npartition_samples = None

    def _weights(self, CF, cf):
        return -torch.linalg.multi_dot(
            (torch.linalg.pinv(CF), cf.T)).T
        # try:
        #     direct solve is usually not a good idea because of ill
        #     conditioning which can be larges especially for mean_variance
        #
        #     return -torch.linalg.solve(CF, cf.T).T
        # except (torch._C._LinAlgError):
        #     return -torch.linalg.multi_dot(
        #         (torch.linalg.pinv(CF), cf.T)).T
        # try:
        #     weights = -torch.linalg.multi_dot(
        #         (torch.linalg.pinv(CF), cf.T))
        # except (np.linalg.LinAlgError, RuntimeError):
        #     weights = torch.ones(cf.T.shape, dtype=torch.double)*1e16
        # return weights.T

    @staticmethod
    def _get_partition_indices(npartition_samples):
        """
        Get the indices, into the flattened array of all samples/values,
        of each indpendent sample partition
        """
        ntotal_independent_samples = npartition_samples.sum()
        total_indices = torch.arange(ntotal_independent_samples)
        # round the cumsum to make sure values like 3.9999999999999999
        # do not get rounded down to 3
        indices = np.split(
            total_indices,
            np.round(np.cumsum(npartition_samples.numpy()[:-1])).astype(int))
        return [torch.as_tensor(idx, dtype=int) for idx in indices]

    def _get_partition_indices_per_acv_subset(self):
        r"""
        Get the indices, into the flattened array of all samples/values
        for each model, of each acv subset
        :math:`\mathcal{Z}_\alpha,\mathcal{Z}_\alpha^*`
        """
        partition_indices = self._get_partition_indices(
            self._rounded_npartition_samples)
        partition_indices_per_model = [
            np.array([], dtype=int), partition_indices[0]]
        for ii in range(1, self._nmodels):
            active_partitions = np.where(
                (self._allocation_mat[:, 2*ii] == 1) |
                (self._allocation_mat[:, 2*ii+1] == 1))[0]
            subset_indices = [None for ii in range(self._nmodels)]
            lb, ub = 0, 0
            for idx in active_partitions:
                ub += partition_indices[idx].shape[0]
                subset_indices[idx] = np.arange(lb, ub)
                lb = ub
            active_partitions_1 = np.where(
                (self._allocation_mat[:, 2*ii] == 1))[0]
            active_partitions_2 = np.where(
                (self._allocation_mat[:, 2*ii+1] == 1))[0]
            indices_1 = np.hstack(
                [subset_indices[idx] for idx in active_partitions_1])
            indices_2 = np.hstack(
                [subset_indices[idx] for idx in active_partitions_2])
            partition_indices_per_model += [indices_1, indices_2]
        return partition_indices_per_model

    def _partition_ratios_to_model_ratios(self, partition_ratios):
        """
        Convert the partition ratios defining the number of samples per
        partition relative to the number of samples in the
        highest-fidelity model partition
        to ratios defining the number of samples per mdoel
        relative to the number of highest-fidelity model samples
        """
        model_ratios = torch.empty_like(partition_ratios, dtype=torch.double)
        for ii in range(1, self._nmodels):
            active_partitions = np.where(
                (self._allocation_mat[1:, 2*ii] == 1) |
                (self._allocation_mat[1:, 2*ii+1] == 1))[0]
            model_ratios[ii-1] = partition_ratios[active_partitions].sum()
            if ((self._allocation_mat[0, 2*ii] == 1) or
                    (self._allocation_mat[0, 2*ii+1] == 1)):
                model_ratios[ii-1] += 1
        return model_ratios

    def _get_num_high_fidelity_samples_from_partition_ratios(
            self, target_cost, partition_ratios):
        model_ratios = self._partition_ratios_to_model_ratios(partition_ratios)
        return target_cost/(
            self._costs[0]+(model_ratios*self._costs[1:]).sum())

    def _npartition_samples_from_partition_ratios(
            self, target_cost, partition_ratios):
        nhf_samples = (
            self._get_num_high_fidelity_samples_from_partition_ratios(
                target_cost, partition_ratios))
        npartition_samples = torch.empty(
            partition_ratios.shape[0]+1, dtype=torch.double)
        npartition_samples[0] = nhf_samples
        npartition_samples[1:] = partition_ratios*nhf_samples
        return npartition_samples

    @staticmethod
    def _covariance_non_optimal_weights(
            hf_est_covar, weights, CF, cf):
        # The expression below, e.g. Equation 8
        # from Dixon 2024, can be used for non optimal control variate weights
        # Warning: Even though this function is general,
        # it should only ever be used for MLMC, because
        # expression for optimal weights is more efficient
        return (
            hf_est_covar
            + torch.linalg.multi_dot((weights, CF, weights.T))
            + torch.linalg.multi_dot((cf, weights.T))
            + torch.linalg.multi_dot((weights, cf.T))
        )

    def _covariance_from_partition_ratios(self, target_cost, partition_ratios):
        """
        Get the variance of the Monte Carlo estimator from costs and cov
        and nsamples ratios. Needed for optimization.

        Parameters
        ----------
        target_cost : float
            The total cost budget

        partition_ratios : np.ndarray (nmodels-1)
            The sample ratios r used to specify the number of samples
            in the indepedent sample partitions

        Returns
        -------
        variance : float
            The variance of the estimator
            """
        npartition_samples = self._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios)
        return self._covariance_from_npartition_samples(npartition_samples)

    def _separate_values_per_model(self, values_per_model):
        r"""
        Seperate values per model into the acv subsets associated with
        :math:`\mathcal{Z}_\alpha,\mathcal{Z}_\alpha^*`
        """
        if len(values_per_model) != self._nmodels:
            msg = "len(values_per_model) {0} != nmodels {1}".format(
                len(values_per_model), self._nmodels)
            raise ValueError(msg)
        for ii in range(self._nmodels):
            if values_per_model[ii].shape[0] != self._nsamples_per_model[ii]:
                msg = "{0} != {1}".format(
                    "len(values_per_model[{0}]): {1}".format(
                        ii, values_per_model[ii].shape[0]),
                    "nsamples_per_model[ii]: {0}".format(
                        self._nsamples_per_model[ii]))
                raise ValueError(msg)

        acv_partition_indices = self._get_partition_indices_per_acv_subset()
        nacv_subsets = len(acv_partition_indices)
        acv_values = [
            values_per_model[ii//2][acv_partition_indices[ii]]
            for ii in range(nacv_subsets)]
        return acv_values

    def _separate_samples_per_model(self, samples_per_model):
        if len(samples_per_model) != self._nmodels:
            msg = "len(samples_per_model) {0} != nmodels {1}".format(
                len(samples_per_model), self._nmodels)
            raise ValueError(msg)
        for ii in range(self._nmodels):
            if samples_per_model[ii].shape[1] != self._nsamples_per_model[ii]:
                msg = "{0} != {1}".format(
                    "len(samples_per_model[{0}]): {1}".format(
                        ii, samples_per_model[ii].shape[0]),
                    "nsamples_per_model[ii]: {0}".format(
                        self._nsamples_per_model[ii]))
                raise ValueError(msg)

        acv_partition_indices = self._get_partition_indices_per_acv_subset()
        nacv_subsets = len(acv_partition_indices)
        acv_samples = [
            samples_per_model[ii//2][:, acv_partition_indices[ii]]
            for ii in range(nacv_subsets)]
        return acv_samples

    def generate_samples_per_model(self, rvs):
        """
        Returns the unique samples needed to evaluate each model.

        Parameters
        ----------
        rvs : callable
            Function with signature

            `rvs(nsamples)->np.ndarray(nvars, nsamples)`

        Returns
        -------
        samples_per_model : list[np.ndarray] (nmodels)
            List of np.ndarray (nvars, nsamples_per_model[ii])
        """
        ntotal_independent_samples = self._rounded_npartition_samples.sum()
        independent_samples = rvs(ntotal_independent_samples)
        samples_per_model = []
        partition_indices = self._get_partition_indices(
            self._rounded_npartition_samples)
        for ii in range(self._nmodels):
            active_partitions = np.where(
                (self._allocation_mat[:, 2*ii] == 1) |
                (self._allocation_mat[:, 2*ii+1] == 1))[0]
            indices = np.hstack(
                [partition_indices[idx] for idx in active_partitions])
            if indices.shape[0] != self._nsamples_per_model[ii]:
                msg = "Rounding has caused {0} != {1}".format(
                    indices.shape[0], self._nsamples_per_model[ii])
                raise RuntimeError(msg)
            samples_per_model.append(independent_samples[:, indices])
        return samples_per_model

    def _compute_single_model_nsamples(self, npartition_samples, model_id):
        active_partitions = np.where(
            (self._allocation_mat[:, 2*model_id] == 1) |
            (self._allocation_mat[:, 2*model_id+1] == 1))[0]
        return npartition_samples[active_partitions].sum()

    def _compute_single_model_nsamples_from_partition_ratios(
            self, partition_ratios, target_cost, model_id):
        npartition_samples = self._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios)
        return self._compute_single_model_nsamples(
            npartition_samples, model_id)

    def _compute_nsamples_per_model(self, npartition_samples):
        nsamples_per_model = np.empty(self._nmodels)
        for ii in range(self._nmodels):
            nsamples_per_model[ii] = self._compute_single_model_nsamples(
                npartition_samples, ii)
        return nsamples_per_model

    def _estimate(self, values_per_model, weights):
        nmodels = len(values_per_model)
        acv_values = self._separate_values_per_model(values_per_model)
        deltas = np.hstack(
            [self._stat.sample_estimate(acv_values[2*ii]) -
             self._stat.sample_estimate(acv_values[2*ii+1])
             for ii in range(1, nmodels)])
        est = (self._stat.sample_estimate(acv_values[1]) +
               weights.numpy().dot(deltas))
        return est

    def __call__(self, values_per_model):
        r"""
        Return the value of the Monte Carlo like estimator

        Parameters
        ----------
        values_per_model : list (nmodels)
            The unique values of each model

        Returns
        -------
        est : np.ndarray (nqoi, nqoi)
            The covariance of the estimator values for
            each high-fidelity model QoI
        """
        return self._estimate(values_per_model, self._optimized_weights)

    def __repr__(self):
        if self._optimized_criteria is None:
            return "{0}(stat={1}, recursion_index={2})".format(
                self.__class__.__name__, self._stat, self._recursion_index)
        rep = "{0}(stat={1}, recursion_index={2}, criteria={3:.3g}".format(
            self.__class__.__name__, self._stat, self._recursion_index,
            self._optimized_criteria)
        rep += " target_cost={0:.5g}, ratios={1}, nsamples={2})".format(
            self._rounded_target_cost,
            self._rounded_partition_ratios,
            self._nsamples_per_model)
        return rep

    @abstractmethod
    def _create_allocation_matrix(self, recursion_index):
        r"""
        Return the allocation matrix corresponding to
        self._nsamples_per_model set by _set_optimized_params

        Returns
        -------
        mat : np.ndarray (nmodels, 2*nmodels)
            For columns :math:`2j, j=0,\ldots,M-1` the ith row contains a
            flag specifiying if :math:`z_i^\star\subseteq z_j^\star`
            For columns :math:`2j+1, j=0,\ldots,M-1` the ith row contains a
            flag specifiying if :math:`z_i\subseteq z_j`
        """
        raise NotImplementedError

    def _get_allocation_matrix(self):
        """return allocation matrix as torch tensor"""
        return torch.as_tensor(self._allocation_mat, dtype=torch.double)

    def _set_recursion_index(self, index):
        """Set the recursion index of the parameterically defined ACV
        Estimator.

        This function intializes the allocation matrix.

        Parameters
        ----------
        index : np.ndarray (nmodels-1)
            The recusion index
        """
        if index is None:
            index = np.zeros(self._nmodels-1, dtype=int)
        else:
            index = np.asarray(index)
        if index.shape[0] != self._nmodels-1:
            msg = "index {0} is the wrong shape. Should be {1}".format(
                index, self._nmodels-1)
            raise ValueError(msg)
        self._create_allocation_matrix(index)
        self._recursion_index = index

    def _covariance_from_npartition_samples(self, npartition_samples):
        CF, cf = self._stat._get_discrepancy_covariances(
            self, npartition_samples)
        weights = self._weights(CF, cf)
        return (self._stat.high_fidelity_estimator_covariance(
            npartition_samples[0]) + torch.linalg.multi_dot((weights, cf.T)))

    def combine_acv_samples(self, acv_samples):
        return _combine_acv_samples(
            self._allocation_mat, self._rounded_npartition_samples,
            acv_samples)

    def combine_acv_values(self, acv_values):
        return _combine_acv_values(
            self._allocation_mat, self._rounded_npartition_samples, acv_values)

    def bootstrap(self, values_per_model, nbootstraps=1000):
        raise NotImplementedError()

    def _bootstrap_acv_estimator(
            self, values_per_model, partition_indices_per_model,
            npartition_samples, reorder_allocation_mat,
            nbootstraps):
        r"""
        Approximate the variance of the Monte Carlo estimate of the mean using
        bootstraping

        Parameters
        ----------

        nbootstraps : integer
            The number of boostraps used to compute estimator variance

        Returns
        -------
        bootstrap_mean : float
            The bootstrap estimate of the estimator mean

        bootstrap_var : float
            The bootstrap estimate of the estimator variance
        """
        nmodels = len(values_per_model)
        npartitions = len(npartition_samples)
        npartition_samples = _cast_to_integers(npartition_samples)
        # preallocate memory so do not have to do it repeatedly
        permuted_partition_indices = [
            np.empty(npartition_samples[jj], dtype=int)
            for jj in range(npartitions)]
        permuted_values_per_model = [v.copy() for v in values_per_model]
        active_partitions = []
        for ii in range(nmodels):
            active_partitions.append(np.where(
                (reorder_allocation_mat[:, 2*ii] == 1) |
                (reorder_allocation_mat[:, 2*ii+1] == 1))[0])

        estimator_vals = np.empty((nbootstraps, self._stat._nqoi))
        for kk in range(nbootstraps):
            for jj in range(npartitions):
                n_jj = npartition_samples[jj]
                permuted_partition_indices[jj][:] = (
                    np.random.choice(np.arange(n_jj, dtype=int), size=(n_jj),
                                     replace=True))
            for ii in range(nmodels):
                for idx in active_partitions[ii]:
                    II = np.where(partition_indices_per_model[ii] == idx)[0]
                    permuted_values_per_model[ii][II] = values_per_model[ii][
                        II[permuted_partition_indices[idx]]]
                    permuted_acv_values = _separate_model_values_acv(
                        reorder_allocation_mat, permuted_values_per_model,
                        partition_indices_per_model)
                    estimator_vals[kk] = self(permuted_acv_values)
                    bootstrap_mean = estimator_vals.mean()
                    bootstrap_var = estimator_vals.var()
        return bootstrap_mean, bootstrap_var

    def _objective(self, target_cost, x, return_grad=True):
        partition_ratios = torch.as_tensor(x, dtype=torch.double)
        if return_grad:
            partition_ratios.requires_grad = True
            covariance = self._covariance_from_partition_ratios(
                target_cost, partition_ratios)
            val = self._optimization_criteria(covariance)
        if not return_grad:
            return val.item()
        val.backward()
        grad = partition_ratios.grad.detach().numpy().copy()
        partition_ratios.grad.zero_()
        return val.item(), grad

    def _set_initial_guess(self, initial_guess):
        if initial_guess is not None:
            self._initial_guess = torch.as_tensor(
                initial_guess, dtype=torch.double)
        else:
            self._initial_guess = None

    def _allocate_samples_opt_minimize(
            self, costs, target_cost, initial_guess, optim_method,
            optim_options, cons):
        if optim_options is None:
            if optim_method == "SLSQP":
                optim_options = {'disp': False, 'ftol': 1e-10,
                                 'maxiter': 10000, "iprint": 0}
            elif optim_method == "trust-constr":
                optim_options = {'disp': False, 'gtol': 1e-10,
                                 'maxiter': 10000, "verbose": 0}
            else:
                raise ValueError(f"{optim_method} not supported")

        if target_cost < costs.sum():
            msg = "Target cost does not allow at least one sample from "
            msg += "each model"
            raise ValueError(msg)

        nmodels = len(costs)
        nunknowns = len(initial_guess)
        assert nunknowns == nmodels-1
        bounds = None  # [(0, np.inf) for ii in range(nunknowns)]

        return_grad = True
        with warnings.catch_warnings():
            # ignore scipy warnings
            warnings.simplefilter("ignore")
            opt = minimize(
                partial(self._objective, target_cost, return_grad=return_grad),
                initial_guess, method=optim_method, jac=return_grad,
                bounds=bounds, constraints=cons, options=optim_options)
        return opt

    def _get_initial_guess(self, initial_guess, cov, costs, target_cost):
        if initial_guess is not None:
            return initial_guess
        return np.full((self._nmodels-1,), 1)

    def _allocate_samples_opt(self, cov, costs, target_cost,
                              cons=[],
                              initial_guess=None,
                              optim_options=None, optim_method='trust-constr'):
        initial_guess = self._get_initial_guess(
            initial_guess, cov, costs, target_cost)
        assert optim_method == "SLSQP" or optim_method == "trust-constr"
        opt = self._allocate_samples_opt_minimize(
            costs, target_cost, initial_guess, optim_method, optim_options,
            cons)
        partition_ratios = torch.as_tensor(opt.x, dtype=torch.double)
        if not opt.success:  # and (opt.status!=8 or not np.isfinite(opt.fun)):
            raise RuntimeError('optimizer failed'+f'{opt}')
        else:
            val = opt.fun
        return partition_ratios, val

    def _allocate_samples_user_init_guess(self, cons, target_cost, **kwargs):
        opt = self._allocate_samples_opt(
            self._cov, self._costs, target_cost, cons,
            initial_guess=self._initial_guess, **kwargs)
        try:
            opt = self._allocate_samples_opt(
                self._cov, self._costs, target_cost, cons,
                initial_guess=self._initial_guess, **kwargs)
            return opt
        except RuntimeError:
            return None, np.inf

    def _allocate_samples_mfmc(self, cons, target_cost, **kwargs):
        # TODO convert MFMC allocation per model to npartition_samples
        assert False
        if (not (_check_mfmc_model_costs_and_correlations(
                self._costs,
                get_correlation_from_covariance(self._cov.numpy()))) or
            len(self._cov) != len(self._costs)):
            # second condition above will not be true for multiple qoi
            return None, np.inf
        mfmc_model_ratios = torch.as_tensor(_allocate_samples_mfmc(
            self._cov, self._costs, target_cost)[0], dtype=torch.double)
        mfmc_initial_guess = MFMCEstimator._mfmc_ratios_to_npartition_ratios(
            mfmc_model_ratios)
        try:
            opt = self._allocate_samples_opt(
                self._cov, self._costs, target_cost, cons,
                initial_guess=mfmc_initial_guess, **kwargs)
            return opt
        except RuntimeError:
            return None, np.inf

    @abstractmethod
    def _get_specific_constraints(self, target_cost):
        raise NotImplementedError()

    @staticmethod
    def _constraint_jacobian(constraint_fun, partition_ratios_np, *args):
        partition_ratios = torch.as_tensor(
            partition_ratios_np, dtype=torch.double)
        partition_ratios.requires_grad = True
        val = constraint_fun(partition_ratios, *args, return_numpy=False)
        val.backward()
        jac = partition_ratios.grad.detach().numpy().copy()
        partition_ratios.grad.zero_()
        return jac

    def _acv_npartition_samples_constraint(
            self, partition_ratios_np, target_cost, min_nsamples, partition_id,
            return_numpy=True):
        partition_ratios = torch.as_tensor(
            partition_ratios_np, dtype=torch.double)
        nsamples = self._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios)[partition_id]
        val = nsamples-min_nsamples
        if return_numpy:
            return val.item()
        return val

    def _acv_npartition_samples_constraint_jac(
            self, partition_ratios_np, target_cost, min_nsamples,
            partition_id):
        return self._constraint_jacobian(
            self._acv_npartition_samples_constraint, partition_ratios_np,
            target_cost, min_nsamples, partition_id)

    def _npartition_ratios_constaint(self, partition_ratios_np, ratio_id):
        # needs to be positiv e
        return partition_ratios_np[ratio_id]-1e-8

    def _npartition_ratios_constaint_jac(
            self, partition_ratios_np, ratio_id):
        jac = np.zeros(partition_ratios_np.shape[0], dtype=float)
        jac[ratio_id] = 1.0
        return jac

    def _get_constraints(self, target_cost):
        # Ensure the first partition has enough samples to compute
        # the desired statistic
        if isinstance(
                self._stat,
                (MultiOutputVariance, MultiOutputMeanAndVariance)):
            min_nhf_samples = 2
        else:
            min_nhf_samples = 1
            cons = [
                {'type': 'ineq',
                 'fun': self._acv_npartition_samples_constraint,
                 'jac': self._acv_npartition_samples_constraint_jac,
                 'args': (target_cost, min_nhf_samples, 0)}]

        # Ensure that remaining partitions have at least one sample
        cons += [
            {'type': 'ineq',
             'fun': self._acv_npartition_samples_constraint,
             'jac': self._acv_npartition_samples_constraint_jac,
             'args': (target_cost, 1, ii)}
            for ii in range(1, self._nmodels)]

        # Ensure ratios are positive
        cons += [
            {'type': 'ineq',
             'fun': self._npartition_ratios_constaint,
             'jac': self._npartition_ratios_constaint_jac,
             'args': (ii,)}
            for ii in range(self._nmodels-1)]

        # Note target cost is satisfied by construction using the above
        # constraints because nsamples is determined based on target cost
        cons += self._get_specific_constraints(target_cost)
        return cons

    def _allocate_samples(self, target_cost, **kwargs):
        cons = self._get_constraints(target_cost)
        opts = []
        # kwargs["optim_method"] = "trust-constr"
        # opt_user_tr = self._allocate_samples_user_init_guess(
        #     cons, target_cost, **kwargs)
        # opts.append(opt_user_tr)

        if True:  # opt_user_tr[0] is None:
            kwargs["optim_method"] = "SLSQP"
            opt_user_sq = self._allocate_samples_user_init_guess(
                cons, target_cost, **kwargs)
            opts.append(opt_user_sq)

        # kwargs["optim_method"] = "trust-constr"
        # opt_mfmc_tr = self._allocate_samples_mfmc(cons, target_cost, **kwargs)
        # opts.append(opt_mfmc_tr)
        # if opt_mfmc_tr[0] is None:
        #     kwargs["optim_method"] = "SLSQP"
        #     opt_mfmc_sq = self._allocate_samples_mfmc(
        #         cons, target_cost, **kwargs)
        #     opts.append(opt_mfmc_sq)
        obj_vals = np.array([o[1] for o in opts])
        if not np.any(np.isfinite(obj_vals)):
            raise RuntimeError(
                "no solution found from multiple initial guesses {0}")
        II = np.argmin(obj_vals)
        return opts[II]

    def _round_partition_ratios(self, target_cost, partition_ratios):
        npartition_samples = self._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios)
        if ((npartition_samples[0] < 1.-1e-8)):
            raise RuntimeError("Rounding will cause nhf samples to be zero")
        rounded_npartition_samples = np.round(
            npartition_samples.numpy()).astype(int)
        rounded_target_cost = (
            self._compute_nsamples_per_model(rounded_npartition_samples) *
            self._costs.numpy()).sum()
        rounded_partition_ratios = (
            rounded_npartition_samples[1:]/rounded_npartition_samples[0])
        return rounded_partition_ratios, rounded_target_cost

    def _set_optimized_params(self, partition_ratios, target_cost):
        """
        Set the parameters needed to generate samples for evaluating the
        estimator

        Parameters
        ----------
        rounded_nsample_ratios : np.ndarray (nmodels-1, dtype=int)
            The sample ratios r used to specify the number of samples in
            the independent sample partitions.
            the ith model.

        rounded_target_cost : float
            The cost of the new sample allocation

        Sets attrributes
        ----------------
        self._rounded_partition_ratios : np.ndarray (nmodels-1)
            The optimal partition ratios rounded so that each partition
            contains an integer number of samples

        self._rounded_cost : float
            The computational cost of the estimator using the rounded
            partition_ratios

        self._rounded_npartition_samples :  np.ndarray (nmodels)
            The number of samples in each partition corresponding to the
            rounded partition_ratios

        self._optimized_criteria: float
            The value of the sample allocation objective using the rounded
            partition_ratios

        self._rounded_nsamples_per_model : np.ndarray (nmodels)
            The number of samples allocated to each model using the rounded
            partition_ratios
        """
        self._rounded_partition_ratios, self._rounded_target_cost = (
            self._round_partition_ratios(
                target_cost,
                torch.as_tensor(partition_ratios, dtype=torch.double)))
        self._optimized_covariance = self._covariance_from_partition_ratios(
            self._rounded_target_cost, torch.as_tensor(
                self._rounded_partition_ratios, dtype=torch.double))
        self._optimized_criteria = self._optimization_criteria(
            self._optimized_covariance)
        self._rounded_npartition_samples = torch.round(
            self._npartition_samples_from_partition_ratios(
                self._rounded_target_cost,
                torch.as_tensor(
                    self._rounded_partition_ratios,
                    dtype=torch.double)))
        self._nsamples_per_model = torch.as_tensor(
            self._compute_nsamples_per_model(self._rounded_npartition_samples),
            dtype=torch.int)
        self._optimized_CF, self._optimized_cf = (
            self._stat._get_discrepancy_covariances(
                self,  self._rounded_npartition_samples))
        self._optimized_weights = self._weights(
            self._optimized_CF, self._optimized_cf)

    def _allocate_samples_for_single_recursion(self, target_cost, verbosity=0):
        partition_ratios, obj_val = self._allocate_samples(
            target_cost)
        self._set_optimized_params(partition_ratios, target_cost)

    def _allocate_samples_for_all_recursion_indices(
            self, target_cost, verbosity):
        best_variance = torch.as_tensor(np.inf, dtype=torch.double)
        best_result = None
        for index in _get_acv_recursion_indices(
                self._nmodels, self._tree_depth):
            self._set_recursion_index(index)
            try:
                self._allocate_samples_for_single_recursion(
                    target_cost, verbosity)
            except RuntimeError as e:
                # typically solver fails because trying to use
                # uniformative model as a recursive control variate
                if not self._allow_failures:
                    raise e
                self._optimized_criteria = torch.as_tensor([np.inf])
                if verbosity > 0:
                    print("Optimizer failed")
            if verbosity > 0:
                msg = "Recursion: {0} Objective: best {1}, current {2}".format(
                    index, best_variance.item(),
                    self._optimized_criteria.item())
                print(msg)
            if self._optimized_criteria < best_variance:
                best_result = [self._rounded_partition_ratios,
                               self._rounded_target_cost,
                               self._optimized_criteria, index]
                best_variance = self._optimized_criteria
        if best_result is None:
            raise RuntimeError("No solutions were found")
        self._set_recursion_index(best_result[3])
        self._set_optimized_params(
            torch.as_tensor(best_result[0], dtype=torch.double),
            *best_result[1:3])
        return best_result[:3]

    def allocate_samples(self, target_cost, verbosity=0):
        if self._tree_depth is not None:
            return self._allocate_samples_for_all_recursion_indices(
                target_cost, verbosity)
        return self._allocate_samples_for_single_recursion(
            target_cost, verbosity)


class GMFEstimator(ACVEstimator):
    def _create_allocation_matrix(self, recursion_index):
        self._allocation_mat = _get_allocation_matrix_gmf(
            recursion_index)

    def _get_specific_constraints(self, target_cost):
        return []


class GISEstimator(ACVEstimator):
    """
    The GIS estimator from Gorodetsky et al. and Bomorito et al
    """
    def _create_allocation_matrix(self, recursion_index):
        self._allocation_mat = _get_allocation_matrix_acvis(
            recursion_index)

    def _get_specific_constraints(self, target_cost):
        return []


class GRDEstimator(ACVEstimator):
    """
    The GRD estimator.
    """
    def _create_allocation_matrix(self, recursion_index):
        self._allocation_mat = _get_allocation_matrix_acvrd(
            recursion_index)

    def _get_specific_constraints(self, target_cost):
        return []


class MFMCEstimator(GMFEstimator):
    def __init__(self, stat, costs, cov, opt_criteria=None,
                 opt_qoi=0):
        # Use the sample analytical sample allocation for estimating a scalar
        # mean when estimating any statistic
        super().__init__(stat, costs, cov,
                         recursion_index=None, opt_criteria=None)
        # The qoi index used to generate the sample allocation
        self._opt_qoi = opt_qoi

    def _allocate_samples(self, target_cost):
        # nsample_ratios returned will be listed in according to
        # self.model_order which is what self.get_rsquared requires
        nqoi = self._cov.shape[0]//len(self._costs)
        nsample_ratios, val = _allocate_samples_mfmc(
            self._cov.numpy()[self._opt_qoi::nqoi, self._opt_qoi::nqoi],
            self._costs.numpy(), target_cost)
        nsample_ratios = (
            self._mfmc_ratios_to_npartition_ratios(nsample_ratios))
        return torch.as_tensor(nsample_ratios, dtype=torch.double), val

    @staticmethod
    def _mfmc_ratios_to_npartition_ratios(ratios):
        partition_ratios = np.hstack((ratios[0]-1, np.diff(ratios)))
        return partition_ratios

    def _get_allocation_matrix(self):
        return _get_sample_allocation_matrix_mfmc(self._nmodels)


class MLMCEstimator(ACVEstimator):
    def __init__(self, stat, costs, cov, opt_criteria=None,
                 opt_qoi=0):
        """
        Use the sample analytical sample allocation for estimating a scalar
        mean when estimating any statistic

        Use optimal ACV weights instead of all weights=-1 used by
        classical MLMC.
        """
        super().__init__(stat, costs, cov,
                         recursion_index=None, opt_criteria=None)
        # The qoi index used to generate the sample allocation
        self._opt_qoi = opt_qoi

    @staticmethod
    def _weights(CF, cf):
        # raise NotImplementedError("check weights size is correct")
        return -torch.ones(cf.shape, dtype=torch.double)

    def _covariance_from_npartition_samples(self, npartition_samples):
        CF, cf = self._stat._get_discrepancy_covariances(
            self, npartition_samples)
        weights = self._weights(CF, cf)
        # cannot use formulation of variance that uses optimal weights
        # must use the more general expression below, e.g. Equation 8
        # from Dixon 2024.
        return self._covariance_non_optimal_weights(
            self._stat.high_fidelity_estimator_covariance(
                npartition_samples[0]), weights, CF, cf)

    def _allocate_samples(self, target_cost):
        nqoi = self._cov.shape[0]//len(self._costs)
        nsample_ratios, val = _allocate_samples_mlmc(
            self._cov.numpy()[self._opt_qoi::nqoi, self._opt_qoi::nqoi],
            self._costs.numpy(), target_cost)
        return torch.as_tensor(nsample_ratios, dtype=torch.double), val

    def _create_allocation_matrix(self):
        return _get_sample_allocation_matrix_mlmc(self._nmodels)

    @staticmethod
    def _mlmc_ratios_to_npartition_ratios(ratios):
        partition_ratios = [ratios[0]-1]
        for ii in range(1, len(ratios)):
            partition_ratios.append(ratios[ii]-partition_ratios[ii-1])
        return np.hstack(partition_ratios)


class BestModelSubsetEstimator():
    def __init__(self, estimator_type, stat_type, costs, cov,
                 max_nmodels, *est_args, **est_kwargs):

        self.best_est = None

        self._estimator_type = estimator_type
        self._stat_type = stat_type
        self._candidate_cov, self._candidate_costs = cov, np.asarray(costs)
        # self._ncandidate_nmodels is the number of total models
        self._ncandidate_models = len(self._candidate_costs)
        self._nqoi = self._candidate_cov.shape[0]//self._ncandidate_models
        self._max_nmodels = max_nmodels
        self._args = est_args
        self._allow_failures = est_kwargs.get("allow_failures", False)
        if "allow_failures" in est_kwargs:
            del est_kwargs["allow_failures"]
        self._kwargs = est_kwargs
        self._best_model_indices = None
        self._all_model_labels = None

    @property
    def model_labels(self):
        return [self._all_model_labels[idx]
                for idx in self._best_model_indices]

    @model_labels.setter
    def model_labels(self, labels):
        self._all_model_labels = labels

    def _get_model_subset_estimator(self, qoi_idx,
                                    nsubset_lfmodels, allocate_kwargs,
                                    target_cost, lf_model_subset_indices):
        idx = np.hstack(([0], lf_model_subset_indices)).astype(int)
        subset_cov = _nqoi_nqoi_subproblem(
            self._candidate_cov, self._ncandidate_models, self._nqoi,
            idx, qoi_idx)
        subset_costs = self._candidate_costs[idx]
        sub_args = multioutput_stats[self._stat_type]._args_model_subset(
            self._ncandidate_models, self._nqoi, idx, *self._args)
        sub_kwargs = copy.deepcopy(self._kwargs)
        if "recursion_index" in sub_kwargs:
            index = sub_kwargs["recursion_index"]
            if (np.allclose(index, np.arange(len(index))) or
                    np.allclose(index, np.zeros(len(index)))):
                sub_kwargs["recursion_index"] = index[:nsubset_lfmodels]
            else:
                msg = "model selection can only be used with recursion indices"
                msg += " (0, 1, 2, ...) or (0, ..., 0) or tree_depth is"
                msg += " not None"
                # There is no logical way to reduce a recursion index to use
                # a subset of model unless they are one of these two indices
                # or tree_depth is not None so that all possible recursion
                # indices are considered
                raise ValueError(msg)
        if "tree_depth" in sub_kwargs:
            sub_kwargs["tree_depth"] = min(
                sub_kwargs["tree_depth"], nsubset_lfmodels)
        try:
            est = get_estimator(
                self._estimator_type, self._stat_type, self._nqoi,
                subset_costs, subset_cov, *sub_args, **sub_kwargs)
        except ValueError as e:
            if allocate_kwargs.get("verbosity", 0) > 0:
                print(e)
            # Some estiamtors, e.g. MFMC, fail when certain criteria
            # are not satisfied
            return None
        try:
            est.allocate_samples(target_cost, **allocate_kwargs)
            if allocate_kwargs.get("verbosity", 0) > 0:
                msg = "Model: {0} Objective: {1}".format(
                    idx, est._optimized_criteria.item())
                print(msg)
            return est
        except (RuntimeError, ValueError) as e:
            if self._allow_failures:
                return None
            raise e

    def _get_best_models_for_acv_estimator(
            self, target_cost, **allocate_kwargs):
        if self._max_nmodels is None:
            max_nmodels = self._ncandidate_nmodels
        else:
            max_nmodels = self._max_nmodels
            lf_model_indices = np.arange(1, self._ncandidate_models)
        best_criteria = np.inf
        best_est, best_model_indices = None, None
        qoi_idx = np.arange(self._nqoi)
        nprocs = allocate_kwargs.get("nprocs", 1)
        if allocate_kwargs.get("verbosity", 0) > 0:
            print(f"Finding best model using {nprocs} processors")
        if "nprocs" in allocate_kwargs:
            del allocate_kwargs["nprocs"]
        for nsubset_lfmodels in range(1, max_nmodels):
            if nprocs > 1:
                pool = Pool(nprocs)
                indices = list(
                    combinations(lf_model_indices, nsubset_lfmodels))
                result = pool.map(
                    partial(self._get_model_subset_estimator,
                            qoi_idx, nsubset_lfmodels, allocate_kwargs,
                            target_cost), indices)
                pool.close()
                criteria = [
                    np.array(est._optimized_criteria)
                    if est is not None else np.inf for est in result]
                II = np.argmin(criteria)
                if not np.isfinite(criteria[II]):
                    best_est = None
                else:
                    best_est = result[II]
                    best_model_indices = np.hstack(
                        ([0], indices[II])).astype(int)
                    best_criteria = best_est._optimized_criteria
                continue

            for lf_model_subset_indices in combinations(
                    lf_model_indices, nsubset_lfmodels):
                est = self._get_model_subset_estimator(
                    qoi_idx, nsubset_lfmodels, allocate_kwargs,
                    target_cost, lf_model_subset_indices)
                if est is not None and est._optimized_criteria < best_criteria:
                    best_est = est
                    best_model_indices = np.hstack(
                        ([0], lf_model_subset_indices)).astype(int)
                    best_criteria = best_est._optimized_criteria
        if best_est is None:
            raise RuntimeError("No solutions found for any model subset")
        return best_est, best_model_indices

    def allocate_samples(self, target_cost, **allocate_kwargs):
        if self._estimator_type == "mc":
            best_model_indices = np.array([0])
            args = multioutput_stats[self._stat_type]._args_model_subset(
                self._ncandidate_models, self._nqoi, best_model_indices,
                *self._args)
            best_est = get_estimator(
                self._estimator_type, self._stat_type, self._nqoi,
                self._candidate_costs[:1], self._candidate_cov[:1, :1],
                *args, **self._kwargs)
            best_est.allocate_samples(target_cost)

        else:
            best_est, best_model_indices = (
                self._get_best_models_for_acv_estimator(
                    target_cost, **allocate_kwargs))
        # self._optimized_criteria = best_est._optimized_criteria
        # self._rounded_target_cost = best_est.rounded_target_cost
        self.best_est = best_est
        self._best_model_indices = best_model_indices
        self._set_best_est_attributes()

    def _set_best_est_attributes(self):
        # allow direct access of important self.best_est attributes
        # __call__ cannot be set using this approach.
        attr_list = [
            # public functions
            "combine_acv_samples",
            "combine_acv_values",
            "generate_samples_per_model",
            "bootstrap",
            # private functions and variables
            "_separate_model_values",
            "_covariance_from_npartition_samples",
            "_covariance_from_ratios",
            "_nsample_ratios", "_stat",
            "_nmodels", "_cov", "_npartition_samples"
            "_nsamples_per_model", "_costs",
            "_optimized_criteria",
            "_rounded_target_cost",
            "_get_allocation_matrix"]
        for attr in attr_list:
            setattr(self, attr, getattr(self.best_est, attr))

    def __repr__(self):
        if self._optimized_criteria is None:
            return "{0}".format(self.__class__.__name__)
        return "{0}(est={1}, subset={2})".format(
            self.__class__.__name__, self.best_est, self._best_model_indices)

    def __call__(self, values):
        return self.best_est(values)


multioutput_estimators = {
    "gmf": GMFEstimator,
    "gis": GISEstimator,
    "grd": GRDEstimator,
    "mfmc": MFMCEstimator,
    "mlmc": MLMCEstimator,
    "mc": MCEstimator}


multioutput_stats = {
    "mean": MultiOutputMean,
    "variance": MultiOutputVariance,
    "mean_variance": MultiOutputMeanAndVariance,
}


def get_estimator(estimator_type, stat_type, nqoi, costs, cov, *args,
                  max_nmodels=None, **kwargs):
    if estimator_type not in multioutput_estimators:
        msg = f"Estimator {estimator_type} not supported. "
        msg += f"Must be one of {multioutput_estimators.keys()}"
        raise ValueError(msg)

    if stat_type not in multioutput_stats:
        msg = f"Statistic {stat_type} not supported. "
        msg += f"Must be one of {multioutput_stats.keys()}"
        raise ValueError(msg)

    if max_nmodels is None:
        stat = multioutput_stats[stat_type](nqoi, cov, *args)
        return multioutput_estimators[estimator_type](
            stat, costs, cov, **kwargs)

    return BestModelSubsetEstimator(
        estimator_type, stat_type, costs, cov,
        max_nmodels, *args, **kwargs)


class SingleQoiAndStatComparisonCriteria():
    def __init__(self, stat_type, qoi_idx):
        """
        Compare estimators based on the variance of a single statistic
        for a single QoI even though mutiple QoI may have been used to compute
        multiple statistics

        Parameters
        ----------
        stat_type: str
            The stat type. Must be one of ["mean", "variance", "mean_variance"]

        qoi_idx: integer
            The index of the QoI as it appears in the covariance matrix
        """
        self._stat_type = stat_type
        self._qoi_idx = qoi_idx

    def __call__(self, est_covariance, est):
        if self._stat_type != "mean" and isinstance(
                est._stat, MultiOutputMeanAndVariance):
            return (
                est_covariance[est.nqoi+self._qoi_idx, est._nqoi+self._qoi_idx])
        elif (isinstance(
                est._stat, (MultiOutputVariance, MultiOutputMean)) or
              self._stat_type == "mean"):
            return est_covariance[self._qoi_idx, self._qoi_idx]
        raise ValueError("{0} not supported".format(est._stat))

    def __repr__(self):
        return "{0}(stat={1}, qoi={2})".format(
            self.__class__.__name__, self._stat_type, self._qoi_idx)


def _estimate_components(variable, est, funs, ii):
    """
    Notes
    -----
    To create reproducible results when running numpy.random in parallel
    must use RandomState. If not the results will be non-deterministic.
    This is happens because of a race condition. numpy.random.* uses only
    one global PRNG that is shared across all the threads without
    synchronization. Since the threads are running in parallel, at the same
    time, and their access to this global PRNG is not synchronized between
    them, they are all racing to access the PRNG state (so that the PRNG's
    state might change behind other threads' backs). Giving each thread its
    own PRNG (RandomState) solves this problem because there is no longer
    any state that's shared by multiple threads without synchronization.
    Also see new features
    https://docs.scipy.org/doc/numpy/reference/random/parallel.html
    https://docs.scipy.org/doc/numpy/reference/random/multithreading.html
    """
    random_state = np.random.RandomState(ii)
    samples_per_model = est.generate_samples_per_model(
        partial(variable.rvs, random_state=random_state))
    values_per_model = [
        fun(samples) for fun, samples in zip(funs, samples_per_model)]

    mc_est = est._stat.sample_estimate
    if isinstance(est, ACVEstimator):
        est_val = est(values_per_model)
        acv_values = est._separate_values_per_model(values_per_model)
        Q = mc_est(acv_values[1])
        delta = np.hstack([mc_est(acv_values[2*ii]) -
                           mc_est(acv_values[2*ii+1])
                           for ii in range(1, est._nmodels)])
    else:
        est_val = est(values_per_model[0])
        Q = mc_est(values_per_model[0])
        delta = Q*0
    return est_val, Q, delta


def _estimate_components_loop(
        variable, ntrials, est, funs, max_eval_concurrency):
    if max_eval_concurrency == 1:
        Q = []
        delta = []
        estimator_vals = []
        for ii in range(ntrials):
            est_val, Q_val, delta_val = _estimate_components(
                variable, est, funs, ii)
            estimator_vals.append(est_val)
            Q.append(Q_val)
            delta.append(delta_val)
        Q = np.array(Q)
        delta = np.array(delta)
        estimator_vals = np.array(estimator_vals)
        return estimator_vals, Q, delta

    from multiprocessing import Pool
    # set flat funs to none so funs can be pickled
    pool = Pool(max_eval_concurrency)
    func = partial(_estimate_components, variable, est, funs)
    result = pool.map(func, list(range(ntrials)))
    pool.close()
    estimator_vals = np.asarray([r[0] for r in result])
    Q = np.asarray([r[1] for r in result])
    delta = np.asarray([r[2] for r in result])
    return estimator_vals, Q, delta


def numerically_compute_estimator_variance(
        funs, variable, est, ntrials=1e3, max_eval_concurrency=1,
        return_all=False):
    r"""
    Numerically estimate the variance of an approximate control variate
    estimator.

    Parameters
    ----------
    funs : list [callable]
        List of functions with signature

        `fun(samples) -> np.ndarray (nsamples, nqoi)`

    where samples has shape (nvars, nsamples)

    est : :class:`pyapprox.multifidelity.multioutput_monte_carlo.MCEstimator`
        A Monte Carlo like estimator for computing sample based statistics

    ntrials : integer
        The number of times to compute estimator using different randomly
        generated set of samples

    max_eval_concurrency : integer
        The number of processors used to compute realizations of the estimators
        which can be run independently and in parallel.

    Returns
    -------
    hf_covar_numer : np.ndarray (nstats, nstats)
        The estimator covariance of the single high-fidelity Monte Carlo
        estimator

    hf_covar : np.ndarray (nstats, nstats)
        The analytical value of the estimator covariance of the single
       high-fidelity Monte Carlo estimator


    covar_numer : np.ndarray (nstats, nstats)
        The estimator covariance of est

    hf_covar : np.ndarray (nstats, nstats)
        The analytical value of the estimator covariance of est

    est_vals : np.ndarray (ntrials, nstats)
        The values for the est for each trial. Only returned if return_all=True

    Q0 : np.ndarray (ntrials, nstats)
        The values for the single fidelity MC estimator for each trial.
        Only returned if return_all=True

    delta : np.ndarray (ntrials, nstats)
        The values for the differences between the low-fidelty estimators
        :math:`\mathcal{Z}_\alpha` and :math:`\mathcal{Z}_\alpha^*`
        for each trial. Only returned if return_all=True
    """
    est_vals, Q0, delta = _estimate_components_loop(
        variable, ntrials, est, funs, max_eval_concurrency)

    hf_covar_numer = np.cov(Q0, ddof=1, rowvar=False)
    hf_covar = est._stat.high_fidelity_estimator_covariance(
        est._rounded_npartition_samples[0])

    covar_numer = np.cov(est_vals, ddof=1, rowvar=False)
    covar = est._covariance_from_npartition_samples(
        est._rounded_npartition_samples).numpy()

    if not return_all:
        return hf_covar_numer, hf_covar, covar_numer, covar
    return hf_covar_numer, hf_covar, covar_numer, covar, est_vals, Q0, delta


# COMMON TORCH AUTOGRAD MISTAKES
# Do not use hstack to form a vector
# The following will create an numerical error in gradient
# but not error is thrown
# torch.hstack([nhf_samples, nhf_samlpes*npartition_ratios])
# So instead use
# npartition_samples = torch.empty(
# partition_ratios.shape[0]+1, dtype=torch.double)
# npartition_samples[0] = nhf_samples
# npartition_samples[1:] = partition_ratios*nhf_samples
