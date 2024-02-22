import warnings
from abc import abstractmethod
from functools import partial

import torch
import numpy as np
from scipy.optimize import minimize, Bounds

from pyapprox.util.utilities import get_correlation_from_covariance
from pyapprox.multifidelity.stats import (
    MultiOutputVariance, MultiOutputMeanAndVariance)
from pyapprox.multifidelity._visualize import (
    _plot_allocation_matrix, _plot_model_recursion)
from pyapprox.multifidelity._optim import (
    _allocate_samples_mlmc,
    _allocate_samples_mfmc,
    _check_mfmc_model_costs_and_correlations,
    _get_sample_allocation_matrix_mlmc,
    _get_sample_allocation_matrix_mfmc,
    _get_acv_recursion_indices)
from pyapprox.surrogates.autogp._torch_wrappers import asarray


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
    # return torch.logdet(variance)
    # Only compute large eigvalues as the variance will
    # be singular when estimating variance or mean+variance
    # because of the duplicate entries in
    # the covariance matrix
    eigvals = torch.linalg.eigh(variance)[0]
    val = torch.log(eigvals[eigvals > 1e-14]).sum()
    return val


def determinant_variance(variance):
    # return torch.det(variance)
    eigvals = torch.linalg.eigh(variance)[0]
    return eigvals[eigvals > 1e-14].prod()


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
    def __init__(self, stat, costs, opt_criteria=None):
        r"""
        Parameters
        ----------
        stat : :class:`~pyapprox.multifidelity.multioutput_monte_carlo.MultiOutputStatistic`
            Object defining what statistic will be calculated

        costs : np.ndarray (nmodels)
            The relative costs of evaluating each model

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
            self._stat._cov, costs)
        self._optimization_criteria = self._set_optimization_criteria(
            opt_criteria)

        self._rounded_nsamples_per_model = None
        self._rounded_npartition_samples = None
        self._rounded_target_cost = None
        self._optimized_criteria = None
        self._optimized_covariance = None
        self._model_labels = None
        self._npartitions = 1

    def _check_cov(self, cov, costs):
        costs = np.atleast_1d(costs)
        if costs.ndim != 1:
            raise ValueError("costs is not a 1D iterable")
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
            opt_criteria = log_determinant_variance
            # opt_criteria = log_trace_variance
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

    def allocate_samples(self, target_cost, optim_options={}):
        self._rounded_nsamples_per_model = np.asarray(
            [int(np.floor(target_cost/self._costs[0]))])
        self._rounded_npartition_samples = self._rounded_nsamples_per_model
        est_covariance = self._covariance_from_npartition_samples(
            self._rounded_npartition_samples)
        self._optimized_covariance = est_covariance
        optimized_criteria = self._optimization_criteria(est_covariance)
        self._rounded_target_cost = (
            self._costs[0]*self._rounded_nsamples_per_model[0])
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
        return [rvs(self._rounded_nsamples_per_model)]

    def __call__(self, values):
        if not isinstance(values, np.ndarray):
            raise ValueError(
                "values must be an np.ndarray but type={0}".format(
                    type(values)))
        if ((values.ndim != 2) or
                (values.shape[0] != self._rounded_nsamples_per_model[0])):
            msg = "values has the incorrect shape {0} expected {1}".format(
                values.shape,
                (self._rounded_nsamples_per_model[0], self._nqoi))
            raise ValueError(msg)
        return self._stat.sample_estimate(values)

    def bootstrap(self, values, nbootstraps=1000):
        r"""
        Approximate the variance of the estimator using
        bootstraping. The accuracy of bootstapping depends on the number
        of values per model. As it gets large the boostrapped statistics
        will approach the theoretical values.

        Parameters
        ----------
        values : [np.ndarray(nsamples, nqoi)]
            A single entry list containing the unique values of each model.
            The list is required to allow consistent interface with
            multi-fidelity estimators

        nbootstraps : integer
            The number of boostraps used to compute estimator variance

        Returns
        -------
        bootstrap_stats : float
            The bootstrap estimate of the estimator

        bootstrap_covar : float
            The bootstrap estimate of the estimator covariance
        """
        nbootstraps = int(nbootstraps)
        estimator_vals = np.empty((nbootstraps, self._stat._nqoi))
        nsamples = values[0].shape[0]
        indices = np.arange(nsamples)
        for kk in range(nbootstraps):
            bootstrapped_indices = np.random.choice(
                indices, size=nsamples, replace=True)
            estimator_vals[kk] = self._stat.sample_estimate(
                values[0][bootstrapped_indices])
        bootstrap_mean = estimator_vals.mean(axis=0)
        bootstrap_covar = np.cov(estimator_vals, rowvar=False, ddof=1)
        return bootstrap_mean, bootstrap_covar

    def __repr__(self):
        if self._optimized_criteria is None:
            return "{0}(stat={1}, nqoi={2})".format(
                self.__class__.__name__, self._stat, self._nqoi)
        rep = "{0}(stat={1}, criteria={2:.3g}".format(
            self.__class__.__name__, self._stat, self._optimized_criteria)
        rep += " target_cost={0:.5g}, nsamples={1})".format(
            self._rounded_target_cost,
            self._rounded_nsamples_per_model[0])
        return rep


class CVEstimator(MCEstimator):
    def __init__(self, stat, costs, lowfi_stats=None, opt_criteria=None):
        super().__init__(stat, costs, opt_criteria=opt_criteria)
        self._lowfi_stats = lowfi_stats

        self._optimized_CF = None
        self._optimized_cf = None
        self._optimized_weights = None

        self._best_model_indices = np.arange(len(costs))

    def _get_discrepancy_covariances(self, npartition_samples):
        return self._stat._get_cv_discrepancy_covariances(npartition_samples)

    def _covariance_from_npartition_samples(self, npartition_samples):
        CF, cf = self._get_discrepancy_covariances(npartition_samples)
        weights = self._weights(CF, cf)
        return (self._stat.high_fidelity_estimator_covariance(
            npartition_samples[0]) + torch.linalg.multi_dot((weights, cf.T)))

    def _set_optimized_params_base(self, rounded_npartition_samples,
                                   rounded_nsamples_per_model,
                                   rounded_target_cost):
        r"""
        Set the parameters needed to generate samples for evaluating the
        estimator

        Parameters
        ----------
        rounded_npartition_samples : np.ndarray (npartitions, dtype=int)
            The number of samples in the independent sample partitions.

        rounded_nsamples_per_model :  np.ndarray (nmodels)
            The number of samples allocated to each model

        rounded_target_cost : float
            The cost of the new sample allocation

        Sets attributes
        ----------------
        self._rounded_target_cost : float
            The computational cost of the estimator using the rounded
            npartition_samples

        self._rounded_npartition_samples :  np.ndarray (npartitions)
            The number of samples in each partition corresponding to the
            rounded partition_ratios

        self._rounded_nsamples_per_model :  np.ndarray (nmodels)
            The number of samples allocated to each model

        self._optimized_covariance : np.ndarray (nstats, nstats)
            The optimal estimator covariance

        self._optimized_criteria: float
            The value of the sample allocation objective using the rounded
            partition_ratios

        self._rounded_nsamples_per_model : np.ndarray (nmodels)
            The number of samples allocated to each model using the rounded
            partition_ratios

        self._optimized_CF : np.ndarray (nstats*(nmodels-1),nstats*(nmodels-1))
            The covariance between :math:`\Delta_i`, :math:`\Delta_j`

        self._optimized_cf : np.ndarray (nstats, nstats*(nmodels-1))
            The covariance between :math:`Q_0`, :math:`\Delta_j`

        self._optimized_weights : np.ndarray (nstats, nmodels-1)
            The optimal control variate weights
        """
        self._rounded_npartition_samples = rounded_npartition_samples
        self._rounded_nsamples_per_model = rounded_nsamples_per_model
        self._rounded_target_cost = rounded_target_cost
        self._optimized_covariance = self._covariance_from_npartition_samples(
            self._rounded_npartition_samples)
        self._optimized_criteria = self._optimization_criteria(
            self._optimized_covariance)
        self._optimized_CF, self._optimized_cf = (
            self._get_discrepancy_covariances(
                self._rounded_npartition_samples))
        self._optimized_weights = self._weights(
            self._optimized_CF, self._optimized_cf)

    def _estimator_cost(self, npartition_samples):
        return (npartition_samples[0]*self._costs).sum()

    def _set_optimized_params(self, rounded_npartition_samples):
        rounded_target_cost = self._estimator_cost(rounded_npartition_samples)
        self._set_optimized_params_base(
            rounded_npartition_samples, rounded_npartition_samples,
            rounded_target_cost)

    def allocate_samples(self, target_cost, optim_options={}):
        npartition_samples = [target_cost/self._costs.sum()]
        rounded_npartition_samples = [int(np.floor(npartition_samples[0]))]
        if isinstance(self._stat,
                      (MultiOutputVariance, MultiOutputMeanAndVariance)):
            min_nhf_samples = 2
        else:
            min_nhf_samples = 1
        if rounded_npartition_samples[0] < min_nhf_samples:
            msg = "target_cost is to small. Not enough samples of each model"
            msg += " can be taken {0} < {1}".format(
                npartition_samples[0], min_nhf_samples)
            raise ValueError(msg)

        rounded_nsamples_per_model = np.full(
            (self._nmodels,), rounded_npartition_samples[0])
        rounded_target_cost = (
            self._costs*rounded_nsamples_per_model).sum()
        self._set_optimized_params_base(
            rounded_npartition_samples, rounded_nsamples_per_model,
            rounded_target_cost)

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
        samples = rvs(self._rounded_nsamples_per_model[0])
        return [samples.copy() for ii in range(self._nmodels)]

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

    def _estimate(self, values_per_model, weights, bootstrap=False):
        if len(values_per_model) != self._nmodels:
            print(len(self._lowfi_stats), self._nmodels)
            msg = "Must provide the values for each model."
            msg += " {0} != {1}".format(len(values_per_model), self._nmodels)
            raise ValueError(msg)
        nsamples = values_per_model[0].shape[0]
        for values in values_per_model[1:]:
            if values.shape[0] != nsamples:
                msg = "Must provide the same number of samples for each model"
                raise ValueError(msg)
        indices = np.arange(nsamples)
        if bootstrap:
            indices = np.random.choice(
                indices, size=indices.shape[0], replace=True)

        deltas = np.hstack(
            [self._stat.sample_estimate(values_per_model[ii][indices]) -
             self._lowfi_stats[ii-1] for ii in range(1, self._nmodels)])
        est = (self._stat.sample_estimate(values_per_model[0][indices]) +
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

    def insert_pilot_values(self, pilot_values, values_per_model):
        """
        Only add pilot values to the fist indepedent partition and thus
        only to models that use that partition
        """
        new_values_per_model = []
        for ii in range(self._nmodels):
            active_partition = (
                (self._allocation_mat[0, 2*ii] == 1) or
                (self._allocation_mat[0, 2*ii+1] == 1))
            if active_partition:
                new_values_per_model.append(np.vstack((
                    pilot_values[ii], values_per_model[ii])))
            else:
                new_values_per_model.append(values_per_model[ii].copy())
        return new_values_per_model

    def __repr__(self):
        if self._optimized_criteria is None:
            return "{0}(stat={1}, recursion_index={2})".format(
                self.__class__.__name__, self._stat, self._recursion_index)
        rep = "{0}(stat={1}, criteria={2:.3g}".format(
            self.__class__.__name__, self._stat, self._optimized_criteria)
        rep += " target_cost={0:.5g}, nsamples={1})".format(
            self._rounded_target_cost,
            self._rounded_nsamples_per_model[0])
        return rep

    def bootstrap(self, values_per_model, nbootstraps=1000):
        r"""
        Approximate the variance of the estimator using
        bootstraping. The accuracy of bootstapping depends on the number
        of values per model. As it gets large the boostrapped statistics
        will approach the theoretical values.

        Parameters
        ----------
        values_per_model : list (nmodels)
            The unique values of each model

        nbootstraps : integer
            The number of boostraps used to compute estimator variance

        Returns
        -------
        bootstrap_stats : float
            The bootstrap estimate of the estimator

        bootstrap_covar : float
            The bootstrap estimate of the estimator covariance
        """
        nbootstraps = int(nbootstraps)
        estimator_vals = np.empty((nbootstraps, self._stat._nqoi))
        for kk in range(nbootstraps):
            estimator_vals[kk] = self._estimate(
                values_per_model, self._optimized_weights, bootstrap=True)
        bootstrap_mean = estimator_vals.mean(axis=0)
        bootstrap_covar = np.cov(estimator_vals, rowvar=False, ddof=1)
        return bootstrap_mean, bootstrap_covar


class ACVEstimator(CVEstimator):
    def __init__(self, stat, costs,
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
        super().__init__(stat, costs, None, opt_criteria=opt_criteria)
        if tree_depth is not None and recursion_index is not None:
            msg = "Only tree_depth or recurusion_index must be specified"
            raise ValueError(msg)
        if tree_depth is None:
            self._set_recursion_index(recursion_index)
        self._tree_depth = tree_depth
        self._allow_failures = allow_failures

        self._rounded_partition_ratios = None
        self._npartitions = self._nmodels
        self._objective_scaling = 1.0

    def _get_discrepancy_covariances(self, npartition_samples):
        return self._stat._get_acv_discrepancy_covariances(
            self._get_allocation_matrix(), npartition_samples)

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

    def _get_partition_indices_per_acv_subset(self, bootstrap=False):
        r"""
        Get the indices, into the flattened array of all samples/values
        for each model, of each acv subset
        :math:`\mathcal{Z}_\alpha,\mathcal{Z}_\alpha^*`
        """
        partition_indices = self._get_partition_indices(
            self._rounded_npartition_samples)
        if bootstrap:
            npartitions = len(self._rounded_npartition_samples)
            random_partition_indices = [
                None for jj in range(npartitions)]
            random_partition_indices[0] = np.random.choice(
                np.arange(partition_indices[0].shape[0], dtype=int),
                size=partition_indices[0].shape[0], replace=True)
            partition_indices_per_acv_subset = [
                np.array([], dtype=int),
                partition_indices[0][random_partition_indices[0]]]
        else:
            partition_indices_per_acv_subset = [
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
                if bootstrap:
                    if random_partition_indices[idx] is None:
                        # make sure the same random permutation for partition
                        # idx is used for all acv_subsets
                        random_partition_indices[idx] = np.random.choice(
                            np.arange(ub-lb, dtype=int), size=ub-lb,
                            replace=True)
                    subset_indices[idx] = (
                        subset_indices[idx][random_partition_indices[idx]])
                lb = ub
            active_partitions_1 = np.where(
                (self._allocation_mat[:, 2*ii] == 1))[0]
            active_partitions_2 = np.where(
                (self._allocation_mat[:, 2*ii+1] == 1))[0]
            indices_1 = np.hstack(
                [subset_indices[idx] for idx in active_partitions_1])
            indices_2 = np.hstack(
                [subset_indices[idx] for idx in active_partitions_2])
            partition_indices_per_acv_subset += [indices_1, indices_2]
        return partition_indices_per_acv_subset

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

    def _separate_values_per_model(self, values_per_model, bootstrap=False):
        r"""
        Seperate values per model into the acv subsets associated with
        :math:`\mathcal{Z}_\alpha,\mathcal{Z}_\alpha^*`
        """
        if len(values_per_model) != self._nmodels:
            msg = "len(values_per_model) {0} != nmodels {1}".format(
                len(values_per_model), self._nmodels)
            raise ValueError(msg)
        for ii in range(self._nmodels):
            if (values_per_model[ii].shape[0] !=
                    self._rounded_nsamples_per_model[ii]):
                msg = "{0} != {1}".format(
                    "len(values_per_model[{0}]): {1}".format(
                        ii, values_per_model[ii].shape[0]),
                    "nsamples_per_model[ii]: {0}".format(
                        self._rounded_nsamples_per_model[ii]))
                raise ValueError(msg)

        acv_partition_indices = self._get_partition_indices_per_acv_subset(
            bootstrap)
        nacv_subsets = len(acv_partition_indices)
        # atleast_2d is needed for when acv_partition_indices[ii].shape[0] == 1
        # in this case python automatically reduces the values array from
        # shape (1, N) to (N)
        acv_values = [
            np.atleast_2d(values_per_model[ii//2][acv_partition_indices[ii]])
            for ii in range(nacv_subsets)]
        return acv_values

    def _separate_samples_per_model(self, samples_per_model):
        if len(samples_per_model) != self._nmodels:
            msg = "len(samples_per_model) {0} != nmodels {1}".format(
                len(samples_per_model), self._nmodels)
            raise ValueError(msg)
        for ii in range(self._nmodels):
            if (samples_per_model[ii].shape[1] !=
                    self._rounded_nsamples_per_model[ii]):
                msg = "{0} != {1}".format(
                    "len(samples_per_model[{0}]): {1}".format(
                        ii, samples_per_model[ii].shape[0]),
                    "nsamples_per_model[ii]: {0}".format(
                        self._rounded_nsamples_per_model[ii]))
                raise ValueError(msg)

        acv_partition_indices = self._get_partition_indices_per_acv_subset()
        nacv_subsets = len(acv_partition_indices)
        acv_samples = [
            samples_per_model[ii//2][:, acv_partition_indices[ii]]
            for ii in range(nacv_subsets)]
        return acv_samples

    def generate_samples_per_model(self, rvs, npilot_samples=0):
        ntotal_independent_samples = (
            self._rounded_npartition_samples.sum()-npilot_samples)
        independent_samples = rvs(ntotal_independent_samples)
        samples_per_model = []
        rounded_npartition_samples = self._rounded_npartition_samples.clone()
        if npilot_samples > rounded_npartition_samples[0]:
            raise ValueError(
                "npilot_samples is larger than optimized first partition size")
        rounded_npartition_samples[0] -= npilot_samples
        rounded_nsamples_per_model = self._compute_nsamples_per_model(
            rounded_npartition_samples)
        partition_indices = self._get_partition_indices(
            rounded_npartition_samples)
        for ii in range(self._nmodels):
            active_partitions = np.where(
                (self._allocation_mat[:, 2*ii] == 1) |
                (self._allocation_mat[:, 2*ii+1] == 1))[0]
            indices = np.hstack(
                [partition_indices[idx] for idx in active_partitions])
            if indices.shape[0] != rounded_nsamples_per_model[ii]:
                msg = "Rounding has caused {0} != {1}".format(
                    indices.shape[0], rounded_nsamples_per_model[ii])
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

    def _estimate(self, values_per_model, weights, bootstrap=False):
        nmodels = len(values_per_model)
        acv_values = self._separate_values_per_model(
            values_per_model, bootstrap)
        deltas = np.hstack(
            [self._stat.sample_estimate(acv_values[2*ii]) -
             self._stat.sample_estimate(acv_values[2*ii+1])
             for ii in range(1, nmodels)])
        est = (self._stat.sample_estimate(acv_values[1]) +
               weights.numpy().dot(deltas))
        return est

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
            self._rounded_nsamples_per_model.numpy())
        return rep

    @abstractmethod
    def _create_allocation_matrix(self, recursion_index):
        r"""
        Return the allocation matrix corresponding to
        self._rounded_nsamples_per_model set by _set_optimized_params

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

    def combine_acv_samples(self, acv_samples):
        return _combine_acv_samples(
            self._allocation_mat, self._rounded_npartition_samples,
            acv_samples)

    def combine_acv_values(self, acv_values):
        return _combine_acv_values(
            self._allocation_mat, self._rounded_npartition_samples, acv_values)

    def plot_allocation(self, ax, show_npartition_samples=False, **kwargs):
        if show_npartition_samples:
            if self._rounded_npartition_samples is None:
                msg = "set_optimized_params must be called"
                raise ValueError(msg)
            return _plot_allocation_matrix(
                self._allocation_mat, self._rounded_npartition_samples, ax,
                **kwargs)

        return _plot_allocation_matrix(
                self._allocation_mat, None, ax, **kwargs)

    def plot_recursion_dag(self, ax):
        return _plot_model_recursion(self._recursion_index, ax)

    def _objective(self, target_cost, x, return_grad=True):
        partition_ratios = torch.as_tensor(x, dtype=torch.double)
        if return_grad:
            partition_ratios.requires_grad = True
        covariance = self._covariance_from_partition_ratios(
            target_cost, partition_ratios)
        val = self._optimization_criteria(covariance)*self._objective_scaling
        if not return_grad:
            return val.item()
        val.backward()
        grad = partition_ratios.grad.detach().numpy().copy()
        partition_ratios.grad.zero_()
        return val.item(), grad

    def _allocate_samples_minimize(
            self, costs, target_cost, cons, optim_options):

        # take copy of options so do not effect options dictionary
        # provided by user. Items will then be popped from opts
        # so that remaining opts are those needed for scipy opt
        optim_opts_copy = optim_options.copy()
        # default guess is to use nedler mead
        init_guess = optim_opts_copy.pop(
            "init_guess", {"disp": False, "maxiter": 500})
        optim_method = optim_opts_copy.pop("method", "SLSQP")
        if optim_method != "SLSQP" and optim_method != "trust-constr":
            raise ValueError(f"{optim_method} not supported")

        # the robustness of optimization can be improved by scaling
        # the objective.
        self._objective_scaling = optim_opts_copy.pop(
            "scaling", 1e-2)

        if target_cost < costs.sum():
            msg = "Target cost does not allow at least one sample from "
            msg += "each model"
            raise ValueError(msg)

        nunknowns = self._nmodels-1
        # lower and upper bounds can cause some edge cases to
        # not solve reliably
        bounds = Bounds(
            np.zeros(nunknowns)+1e-10, np.full((nunknowns), np.inf),
            keep_feasible=True)

        return_grad = True
        with warnings.catch_warnings():
            # ignore scipy warnings
            warnings.simplefilter("ignore")
            if isinstance(init_guess, dict):
                # get rough initial guess from global optimizer
                default_init_guess = asarray(np.full((self._nmodels-1,), 1.))
                opt = minimize(
                    partial(self._objective, target_cost, return_grad=False),
                    default_init_guess, method="nelder-mead", jac=False,
                    bounds=bounds, constraints=cons, options=init_guess)
                init_guess = opt.x
            if init_guess.shape[0] != self._nmodels-1:
                raise ValueError(
                    "init_guess {0} has the wrong shape".format(init_guess))
            opt = minimize(
                partial(self._objective, target_cost, return_grad=return_grad),
                init_guess, method=optim_method, jac=return_grad,
                bounds=bounds, constraints=cons, options=optim_options)
        return opt

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
        # needs to be positive
        return partition_ratios_np[ratio_id]-0

    def _npartition_ratios_constaint_jac(
            self, partition_ratios_np, ratio_id):
        jac = np.zeros(partition_ratios_np.shape[0], dtype=float)
        jac[ratio_id] = 1.0
        return jac

    def _get_constraints(self, target_cost):
        # Ensure the each partition has enough samples to compute
        # the desired statistic. Techinically we only need the number
        # of samples in each acv subset have enough. But this constraint
        # is easy to implement and not really restrictive practically
        if isinstance(
                self._stat,
                (MultiOutputVariance, MultiOutputMeanAndVariance)):
            partition_min_nsamples = 2.
        else:
            partition_min_nsamples = 1.
        cons = [
            {'type': 'ineq',
             'fun': self._acv_npartition_samples_constraint,
             'jac': self._acv_npartition_samples_constraint_jac,
             'args': (target_cost, partition_min_nsamples, ii)}
            for ii in range(self._nmodels)]

        # Better to enforce this with bounds
        # Ensure ratios are positive
        # cons += [
        #     {'type': 'ineq',
        #      'fun': self._npartition_ratios_constaint,
        #      'jac': self._npartition_ratios_constaint_jac,
        #      'args': (ii,)}
        #     for ii in range(self._nmodels-1)]

        # Note target cost is satisfied by construction using the above
        # constraints because nsamples is determined based on target cost
        cons += self._get_specific_constraints(target_cost)
        return cons

    def _allocate_samples(self, target_cost, optim_options):
        cons = self._get_constraints(target_cost)
        opt = self._allocate_samples_minimize(
            self._costs, target_cost, cons, optim_options)
        partition_ratios = torch.as_tensor(opt.x, dtype=torch.double)
        if not opt.success:
            raise RuntimeError('{0} optimizer failed {1}'.format(self, opt))
        else:
            val = opt.fun
        return partition_ratios, val

    def _round_partition_ratios(self, target_cost, partition_ratios):
        npartition_samples = self._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios)
        if ((npartition_samples[0] < 1-1e-8)):
            msg = "Rounding will cause nhf samples to be zero {0}".format(
                npartition_samples)
            raise RuntimeError(msg)
        rounded_npartition_samples = np.floor(
            npartition_samples.numpy()+1e-8).astype(int)
        assert rounded_npartition_samples[0] >= 1
        rounded_target_cost = (
            self._compute_nsamples_per_model(rounded_npartition_samples) *
            self._costs.numpy()).sum()
        rounded_partition_ratios = (
            rounded_npartition_samples[1:]/rounded_npartition_samples[0])
        return rounded_partition_ratios, rounded_target_cost

    def _estimator_cost(self, npartition_samples):
        nsamples_per_model = self._compute_nsamples_per_model(
            asarray(npartition_samples))
        return (nsamples_per_model*self._costs.numpy()).sum()

    def _set_optimized_params(self, partition_ratios, target_cost):
        """
        Set the parameters needed to generate samples for evaluating the
        estimator

        Parameters
        ----------
        rounded_nsample_ratios : np.ndarray (nmodels-1, dtype=int)
            The sample ratios r used to specify the number of samples in
            the independent sample partitions.

        rounded_target_cost : float
            The cost of the new sample allocation

        Sets attrributes
        ----------------
        self._rounded_partition_ratios : np.ndarray (nmodels-1)
            The optimal partition ratios rounded so that each partition
            contains an integer number of samples

        And all attributes set by super()._set_optimized_params. See
        the docstring of that function for further details
        """
        self._rounded_partition_ratios, rounded_target_cost = (
            self._round_partition_ratios(
                target_cost,
                torch.as_tensor(partition_ratios, dtype=torch.double)))
        rounded_npartition_samples = (
            self._npartition_samples_from_partition_ratios(
                rounded_target_cost,
                torch.as_tensor(self._rounded_partition_ratios,
                                dtype=torch.double)))
        # round because sometimes round_partition_ratios
        # will produce floats slightly smaller
        # than an integer so when converted to an integer will produce
        # values 1 smaller than the correct value
        rounded_npartition_samples = np.round(rounded_npartition_samples)
        rounded_nsamples_per_model = torch.as_tensor(
            self._compute_nsamples_per_model(rounded_npartition_samples),
            dtype=torch.int)
        super()._set_optimized_params_base(
            rounded_npartition_samples, rounded_nsamples_per_model,
            rounded_target_cost)

    def _allocate_samples_for_single_recursion(
            self, target_cost, optim_options):
        partition_ratios, obj_val = self._allocate_samples(
            target_cost, optim_options)
        self._set_optimized_params(partition_ratios, target_cost)

    def get_all_recursion_indices(self):
        return _get_acv_recursion_indices(self._nmodels, self._tree_depth)

    def _allocate_samples_for_all_recursion_indices(
            self, target_cost, optim_options):
        verbosity = optim_options.get("verbosity", 0)
        best_criteria = torch.as_tensor(np.inf, dtype=torch.double)
        best_result = None
        for index in self.get_all_recursion_indices():
            self._set_recursion_index(index)
            try:
                self._allocate_samples_for_single_recursion(
                    target_cost, optim_options)
            except RuntimeError as e:
                # typically solver fails because trying to use
                # uniformative model as a recursive control variate
                if not self._allow_failures:
                    raise e
                self._optimized_criteria = torch.as_tensor(
                    np.inf, dtype=torch.double)
                if verbosity > 0:
                    print("Optimizer failed")
            if verbosity > 2:
                msg = "\t\t Recursion: {0} Objective: best {1}, current {2}".format(
                    index, best_criteria.item(),
                    self._optimized_criteria.item())
                print(msg)
            if self._optimized_criteria < best_criteria:
                best_result = [self._rounded_partition_ratios,
                               self._rounded_target_cost,
                               self._optimized_criteria, index]
                best_criteria = self._optimized_criteria
        if best_result is None:
            raise RuntimeError("No solutions were found")
        self._set_recursion_index(best_result[3])
        self._set_optimized_params(
            torch.as_tensor(best_result[0], dtype=torch.double),
            target_cost)

    def allocate_samples(self, target_cost, optim_options={}):
        if self._tree_depth is not None:
            return self._allocate_samples_for_all_recursion_indices(
                target_cost, optim_options)
        return self._allocate_samples_for_single_recursion(
            target_cost, optim_options)


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
    def __init__(self, stat, costs, opt_criteria=None, opt_qoi=0):
        # Use the sample analytical sample allocation for estimating a scalar
        # mean when estimating any statistic
        nmodels = len(costs)
        super().__init__(stat, costs,
                         recursion_index=np.arange(nmodels-1),
                         opt_criteria=None)
        # The qoi index used to generate the sample allocation
        self._opt_qoi = opt_qoi

    def _allocate_samples(self, target_cost, optim_options={}):
        # nsample_ratios returned will be listed in according to
        # self.model_order which is what self.get_rsquared requires
        nqoi = self._cov.shape[0]//len(self._costs)
        if not _check_mfmc_model_costs_and_correlations(
                self._costs,
                get_correlation_from_covariance(self._cov.numpy())):
            raise ValueError("models do not admit a hierarchy")
        nsample_ratios, val = _allocate_samples_mfmc(
            self._cov.numpy()[self._opt_qoi::nqoi, self._opt_qoi::nqoi],
            self._costs.numpy(), target_cost)
        nsample_ratios = (
            self._native_ratios_to_npartition_ratios(nsample_ratios))
        return torch.as_tensor(nsample_ratios, dtype=torch.double), val

    @staticmethod
    def _native_ratios_to_npartition_ratios(ratios):
        partition_ratios = np.hstack((ratios[0]-1, np.diff(ratios)))
        return partition_ratios

    def _get_allocation_matrix(self):
        return _get_sample_allocation_matrix_mfmc(self._nmodels)


class MLMCEstimator(GRDEstimator):
    def __init__(self, stat, costs, opt_criteria=None,
                 opt_qoi=0):
        """
        Use the sample analytical sample allocation for estimating a scalar
        mean when estimating any statistic

        Use optimal ACV weights instead of all weights=-1 used by
        classical MLMC.
        """
        nmodels = len(costs)
        super().__init__(stat, costs,
                         recursion_index=np.arange(nmodels-1),
                         opt_criteria=None)
        # The qoi index used to generate the sample allocation
        self._opt_qoi = opt_qoi

    @staticmethod
    def _weights(CF, cf):
        # raise NotImplementedError("check weights size is correct")
        return -torch.ones(cf.shape, dtype=torch.double)

    def _covariance_from_npartition_samples(self, npartition_samples):
        CF, cf = self._get_discrepancy_covariances(npartition_samples)
        weights = self._weights(CF, cf)
        # cannot use formulation of variance that uses optimal weights
        # must use the more general expression below, e.g. Equation 8
        # from Dixon 2024.
        return self._covariance_non_optimal_weights(
            self._stat.high_fidelity_estimator_covariance(
                npartition_samples[0]), weights, CF, cf)

    def _allocate_samples(self, target_cost, optim_options={}):
        nqoi = self._cov.shape[0]//len(self._costs)
        nsample_ratios, val = _allocate_samples_mlmc(
            self._cov.numpy()[self._opt_qoi::nqoi, self._opt_qoi::nqoi],
            self._costs.numpy(), target_cost)
        return torch.as_tensor(nsample_ratios, dtype=torch.double), val

    def _create_allocation_matrix(self, dummy):
        self._allocation_mat = _get_sample_allocation_matrix_mlmc(
            self._nmodels)

    @staticmethod
    def _native_ratios_to_npartition_ratios(ratios):
        partition_ratios = [ratios[0]-1]
        for ii in range(1, len(ratios)):
            partition_ratios.append(ratios[ii]-partition_ratios[ii-1])
        return np.hstack(partition_ratios)


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
