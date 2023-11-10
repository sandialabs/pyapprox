import torch
import copy
from abc import abstractmethod
from functools import partial
from itertools import combinations
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

from pyapprox.interface.wrappers import ModelEnsemble
from pyapprox.util.utilities import get_correlation_from_covariance
from pyapprox.multifidelity.control_variate_monte_carlo import (
    get_nsamples_per_model,
    separate_model_values_acv, separate_samples_per_model_acv,
    generate_samples_acv, round_nsample_ratios, bootstrap_acv_estimator,
    get_nhf_samples, allocate_samples_mfmc,
    check_mfmc_model_costs_and_correlations,
    acv_sample_allocation_nhf_samples_constraint,
    acv_sample_allocation_nhf_samples_constraint_jac,
    acv_sample_allocation_gmf_ratio_constraint,
    acv_sample_allocation_gmf_ratio_constraint_jac,
    acv_sample_allocation_nlf_gt_nhf_ratio_constraint,
    acv_sample_allocation_nlf_gt_nhf_ratio_constraint_jac,
    get_acv_initial_guess, get_npartition_samples_mlmc,
    get_sample_allocation_matrix_mlmc, allocate_samples_mlmc,
    get_sample_allocation_matrix_mfmc, get_npartition_samples_mfmc,
    get_acv_recursion_indices,
    combine_acv_values, combine_acv_samples, cast_to_integers)
from pyapprox.multifidelity.stats import (
    MultiOutputMean, MultiOutputVariance, MultiOutputMeanAndVariance,
    _nqoi_nqoi_subproblem)


def _round_nsample_ratios(target_cost, costs, nsample_ratios):
    """
    Return sample ratios that produce integer sample allocations.
    The cost of the returned allocation will not usually equal target cost

    Parameters
    ----------
    target_cost : float
        The total cost budget

    costs : np.ndarray (nmodels)
        The relative costs of evaluating each model

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the
        lower fidelity models, e.g. N_i = r_i*nhf_samples, i=1,...,nmodels-1

    Returns
    -------
    nsample_ratios_floor : float
         ratios r used to specify INTEGER number of samples of the lower
         fidelity models. These ratios will also force nhf_samples to
         be an integer

    rounded_target_cost : float
         The cost of the new sample allocation
    """
    nsamples_float = get_nsamples_per_model(
        target_cost, costs, nsample_ratios, False)
    nsamples_floor = nsamples_float.astype(int)
    # ensure all low-fidelity samples > nhf_samples after rounding
    if nsamples_floor[0] < 1 and nsamples_float[0] < 1-1e-8:
        raise RuntimeError("Rounding likely caused nhf samples to be zero")
    elif nsamples_floor[0] < 1:
        nsamples_floor[0] = 1
    II = np.where(nsamples_floor[1:] == nsamples_floor[0])[0]+1
    nsamples_floor[II] = 2
    nsample_ratios_floor = nsamples_floor[1:]/nsamples_floor[0]
    rounded_target_cost = nsamples_floor[0]*(costs[0]+np.dot(
        nsample_ratios_floor, costs[1:]))
    return nsample_ratios_floor, rounded_target_cost


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
        self.stat = stat

        # private variables (no guarantee that these variables
        #                    will exist in the future)
        self._cov, self._costs, self._nmodels, self._nqoi = self._check_cov(
            cov, costs)
        self._optimization_criteria = self._set_optimization_criteria(
            opt_criteria)

        self._nsamples_per_model = None
        self._rounded_target_cost = None
        self._optimized_criteria = None
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
        return opt_criteria

    def _get_number_of_high_fidelity_samples_from_ratios(
            self, target_cost, nsample_ratios):
        return target_cost/(
            self._costs[0]+(nsample_ratios*self._costs[1:]).sum())

    def _npartition_samples_from_ratios(self, target_cost, nsample_ratios):
        nhf_samples = self._get_number_of_high_fidelity_samples_from_ratios(
            target_cost, nsample_ratios)
        return torch.hstack(
            [torch.as_tensor([nhf_samples], dtype=torch.double),
             nsample_ratios*nhf_samples])

    def _covariance_from_ratios(self, target_cost, nsample_ratios):
        """
        Get the variance of the Monte Carlo estimator from costs and cov
        and nsamples ratios. Needed for optimization.

        Parameters
        ----------
        target_cost : float
            The total cost budget

        nsample_ratios : np.ndarray or torch.tensor (nmodels-1)
            The sample ratios r used to specify the number of samples of the
            lower fidelity models, e.g. N_i = r_i*nhf_samples,
            i=1,...,nmodels-1

        Returns
        -------
        variance : float
            The variance of the estimator
            """
        npartition_samples = self._npartition_samples_from_ratios(
            target_cost, nsample_ratios)
        return self._covariance_from_npartition_samples(npartition_samples)

    def _covariance_from_npartition_samples(self, npartition_samples):
        """
        Get the variance of the Monte Carlo estimator from costs and cov
        and npartition_samples
        """
        return self.stat.high_fidelity_estimator_covariance(
            npartition_samples[0])

    def allocate_samples(self, target_cost, verbosity=0):
        self._nsamples_per_model = np.asarray(
            [int(np.floor(target_cost/self._costs[0]))])
        nsample_ratios = np.zeros(0)
        # Fot single fidelity MC
        # self._npartition_samples = self._nsamples_per_model
        est_covariance = self._covariance_from_npartition_samples(
            self._nsamples_per_model)
        optimized_criteria = self._optimization_criteria(est_covariance)
        self._rounded_target_cost = self._costs[0]*self._nsamples_per_model[0]
        self._optimized_criteria = optimized_criteria
        return nsample_ratios, est_covariance, self._rounded_target_cost

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
        if values.ndim != 2 or values.shape[0] != self._nsamples_per_model:
            msg = "values has the incorrect shape {0}".format(
                values.shape)
            print(self._nsamples_per_model)
            raise ValueError(msg)
        return self.stat.sample_estimate(values)

    def __repr__(self):
        return "{0}(stat={1}, nqoi={2})".format(
            self.__class__.__name__, self.stat, self.nqoi)


def acv_variance_sample_allocation_nhf_samples_constraint(ratios, *args):
    target_cost, costs = args
    # add to ensure that when constraint is violated by small numerical value
    # nhf samples generated from ratios will be greater than 1
    nhf_samples = get_nhf_samples(target_cost, costs, ratios)
    eps = 0
    val = nhf_samples-(2+eps)
    return val


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

        self._rounded_nsample_ratios = None
        self._npartition_samples = None

    def _weights(self, CF, cf):
        return -torch.linalg.multi_dot(
            (torch.linalg.pinv(CF), cf.T)).T
        # try:
        #     direct solve is usually not a good idea because of ill
        #     conditioning which can be larges especsially for meanvariance
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

    def _get_partition_indices(self):
        """
        Get the indices, into the flattened array of all samples/values,
        of each indpendent sample partition
        """
        ntotal_independent_samples = self._npartition_samples.sum()
        total_indices = torch.arange(ntotal_independent_samples)
        indices = np.split(
            total_indices,
            np.cumsum(self._npartition_samples.numpy()).astype(int))
        return [torch.as_tensor(idx, dtype=int) for idx in indices]

    def _get_partition_indices_per_acv_subset(self):
        r"""
        Get the indices, into the flattened array of all samples/values,
        of each acv subset :math:`\mathcal{Z}_\alpha,\mathcal{Z}_\alpha^*`
        """
        partition_indices = self._get_partition_indices()
        partition_indices_per_model = [np.array([], dtype=int)]
        for ii in range(1, 2*self._nmodels):
            active_partitions = np.where(
                (self._allocation_mat[:, ii] == 1))[0]
            indices = np.hstack(
                [partition_indices[idx] for idx in active_partitions])
            partition_indices_per_model.append(indices)
        return partition_indices_per_model

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

    def _separate_samples_per_model(self, samples):
        return self._separate_values_per_model(samples, True)

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
        ntotal_independent_samples = self._npartition_samples.sum()
        independent_samples = rvs(ntotal_independent_samples)
        samples_per_model = []
        partition_indices = self._get_partition_indices()
        for ii in range(self._nmodels):
            active_partitions = np.where(
                (self._allocation_mat[:, 2*ii] == 1) |
                (self._allocation_mat[:, 2*ii+1] == 1))[0]
            indices = np.hstack(
                [partition_indices[idx] for idx in active_partitions])
            samples_per_model.append(independent_samples[:, indices])
        return samples_per_model

    def _compute_nsamples_per_model(self):
        partition_indices = self._get_partition_indices()
        nsamples_per_model = np.empty(self._nmodels, dtype=int)
        for ii in range(self._nmodels):
            active_partitions = np.where(
                (self._allocation_mat[:, 2*ii] == 1) |
                (self._allocation_mat[:, 2*ii+1] == 1))[0]
            indices = np.hstack(
                [partition_indices[idx] for idx in active_partitions])
            nsamples_per_model[ii] = indices.shape[0]
        return nsamples_per_model

    def _estimate(self, values_per_model, weights):
        nmodels = len(values_per_model)
        acv_values = self._separate_values_per_model(values_per_model)
        deltas = np.hstack(
            [self.stat.sample_estimate(acv_values[2*ii]) -
             self.stat.sample_estimate(acv_values[2*ii+1])
             for ii in range(1, nmodels)])
        est = (self.stat.sample_estimate(acv_values[1]) +
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
        CF, cf = self.stat.get_discrepancy_covariances(
            self, self._npartition_samples)
        weights = self._weights(CF, cf)
        return self._estimate(values_per_model, weights)

    def __repr__(self):
        if self._optimized_criteria is None:
            return "{0}(stat={1}, recursion_index={2})".format(
                self.__class__.__name__, self.stat, self._recursion_index)
        rep = "{0}(stat={1}, recursion_index={2}, criteria={3:.3g}".format(
            self.__class__.__name__, self.stat, self._recursion_index,
            self._optimized_criteria)
        rep += " target_cost={0:.5g}, ratios={1}, nsamples={2})".format(
            self._rounded_target_cost,
            self._rounded_nsample_ratios,
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
        CF, cf = self.stat.get_discrepancy_covariances(
            self, npartition_samples)
        weights = self._weights(CF, cf)
        return (self.stat.high_fidelity_estimator_covariance(
            npartition_samples[0]) + torch.linalg.multi_dot((weights, cf.T)))

    def combine_acv_samples(self, acv_samples):
        return combine_acv_samples(
            self._allocation_mat, self._npartition_samples, acv_samples)

    def combine_acv_values(self, acv_values):
        return combine_acv_values(
            self._allocation_mat, self._npartition_samples, acv_values)

    def _generate_data(self, samples_per_model_wo_pilot,
                       values_per_model_wo_pilot, pilot_data,
                       partition_indices_per_model):
        samples_per_model, values_per_model = self.insert_pilot_samples(
            samples_per_model_wo_pilot, values_per_model_wo_pilot,
            pilot_data)
        reorder_allocation_mat = self._get_allocation_matrix(
            self._nsamples_per_model)
        acv_values = separate_model_values_acv(
            reorder_allocation_mat, values_per_model,
            partition_indices_per_model)
        acv_samples = separate_samples_per_model_acv(
            reorder_allocation_mat, samples_per_model,
            partition_indices_per_model)
        return acv_samples, acv_values

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
        npartition_samples = cast_to_integers(npartition_samples)
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

        estimator_vals = np.empty((nbootstraps, self.stat.nqoi))
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
            permuted_acv_values = separate_model_values_acv(
                reorder_allocation_mat, permuted_values_per_model,
                partition_indices_per_model)
            estimator_vals[kk] = self(permuted_acv_values)
        bootstrap_mean = estimator_vals.mean()
        bootstrap_var = estimator_vals.var()
        return bootstrap_mean, bootstrap_var

    def _objective(self, target_cost, x, return_grad=True):
        ratios = torch.tensor(x, dtype=torch.double)
        if return_grad:
            ratios.requires_grad = True
        variance = self._covariance_from_ratios(target_cost, ratios)
        val = self._optimization_criteria(variance)
        if not return_grad:
            return val.item()
        val.backward()
        grad = ratios.grad.detach().numpy().copy()
        ratios.grad.zero_()
        return val.item(), grad

    def _set_initial_guess(self, initial_guess):
        if initial_guess is not None:
            self.initial_guess = torch.as_tensor(
                initial_guess, dtype=torch.double)
        else:
            self.initial_guess = None

    def _allocate_samples_opt_minimize(
            self, costs, target_cost, initial_guess, optim_method,
            optim_options, cons):
        if optim_options is None:
            if optim_method == "SLSQP":
                optim_options = {'disp': False, 'ftol': 1e-10,
                                 'maxiter': 10000, "iprint": 0}
            elif optim_method == "trust-constr":
                optim_options = {'disp': False, 'gtol': 1e-10,
                                 'maxiter': 10000}
            else:
                raise ValueError(f"{optim_method} not supported")

        if target_cost < costs.sum():
            msg = "Target cost does not allow at least one sample from "
            msg += "each model"
            raise ValueError(msg)

        nmodels = len(costs)
        nunknowns = len(initial_guess)
        max_nhf = target_cost/costs[0]
        bounds = [(1+1/(max_nhf),
                   np.ceil(target_cost/cost)) for cost in costs[1:]]
        assert nunknowns == nmodels-1

        # constraint
        # nhf*r-nhf >= 1
        # nhf*(r-1) >= 1
        # r-1 >= 1/nhf
        # r >= 1+1/nhf
        # smallest lower bound when nhf = max_nhf

        return_grad = True
        opt = minimize(
            partial(self._objective, target_cost, return_grad=return_grad),
            initial_guess, method=optim_method, jac=return_grad,
            bounds=bounds, constraints=cons, options=optim_options)
        return opt

    def _allocate_samples_opt(self, cov, costs, target_cost,
                              cons=[],
                              initial_guess=None,
                              optim_options=None, optim_method='trust-constr'):
        initial_guess = get_acv_initial_guess(
            initial_guess, cov, costs, target_cost)
        assert optim_method == "SLSQP" or optim_method == "trust-constr"
        opt = self._allocate_samples_opt_minimize(
            costs, target_cost, initial_guess, optim_method, optim_options,
            cons)
        nsample_ratios = torch.as_tensor(opt.x, dtype=torch.double)
        if not opt.success:  # and (opt.status!=8 or not np.isfinite(opt.fun)):
            raise RuntimeError('optimizer failed'+f'{opt}')
        else:
            val = opt.fun  # self.get_variance(target_cost, nsample_ratios)
        return nsample_ratios, val

    def _allocate_samples_user_init_guess(self, cons, target_cost, **kwargs):
        opt = self._allocate_samples_opt(
                self._cov, self._costs, target_cost, cons,
                initial_guess=self.initial_guess, **kwargs)
        try:
            opt = self._allocate_samples_opt(
                self._cov, self._costs, target_cost, cons,
                initial_guess=self.initial_guess, **kwargs)
            return opt
        except RuntimeError:
            return None, np.inf

    def _allocate_samples_mfmc(self, cons, target_cost, **kwargs):
        # TODO convert MFMC allocation per model to npartition_samples
        if (not (check_mfmc_model_costs_and_correlations(
                self._costs,
                get_correlation_from_covariance(self._cov.numpy()))) or
                len(self._cov) != len(self._costs)):
            # second condition above will not be true for multiple qoi
            return None, np.inf
        mfmc_initial_guess = torch.as_tensor(allocate_samples_mfmc(
            self._cov, self._costs, target_cost)[0], dtype=torch.double)
        try:
            opt = self._allocate_samples_opt(
                self._cov, self._costs, target_cost, cons,
                initial_guess=mfmc_initial_guess, **kwargs)
            return opt
        except RuntimeError:
            return None, np.inf

    def _allocate_samples(self, target_cost, **kwargs):
        cons = self.get_constraints(target_cost)
        opts = []
        kwargs["optim_method"] = "trust-constr"
        opt_user_tr = self._allocate_samples_user_init_guess(
            cons, target_cost, **kwargs)
        opts.append(opt_user_tr)
        if opt_user_tr[0] is None:
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

    @staticmethod
    def _scipy_wrapper(fun, xx, *args):
        # convert argument to fun to tensor before passing to fun
        return fun(torch.as_tensor(xx, dtype=torch.double), *args)

    @abstractmethod
    def _get_constraints(self, target_cost):
        raise NotImplementedError()

    def get_constraints(self, target_cost):
        if isinstance(
                self.stat, (MultiOutputVariance, MultiOutputMeanAndVariance)):
            cons = [
                {'type': 'ineq',
                 'fun': partial(
                     self._scipy_wrapper,
                     acv_variance_sample_allocation_nhf_samples_constraint),
                 'jac': partial(
                     self._scipy_wrapper,
                     acv_sample_allocation_nhf_samples_constraint_jac),
                 'args': (target_cost, self._costs)}]
        else:
            cons = [
                {'type': 'ineq',
                 'fun': partial(
                     self._scipy_wrapper,
                     acv_sample_allocation_nhf_samples_constraint),
                 'jac': partial(
                     self._scipy_wrapper,
                     acv_sample_allocation_nhf_samples_constraint_jac),
                 'args': (target_cost, self._costs)}]
        cons += self._get_constraints(target_cost)

        # Ensure that all low-fidelity models have at least one more sample
        # than high-fidelity model. Otherwise Fmat will not be invertable after
        # rounding to integers
        cons += [
            {'type': 'ineq',
             'fun': partial(
                 self._scipy_wrapper,
                 acv_sample_allocation_nlf_gt_nhf_ratio_constraint),
             'jac': partial(
                 self._scipy_wrapper,
                 acv_sample_allocation_nlf_gt_nhf_ratio_constraint_jac),
             'args': (ii, target_cost, self._costs)}
            for ii in range(1, self._nmodels)]
        return cons

    def _set_optimized_params(self, rounded_nsample_ratios,
                              rounded_target_cost,
                              optimized_criteria):
        """
        Set the parameters needed to generate samples for evaluating the
        estimator

        rounded_nsample_ratios : np.ndarray (nmodels-1, dtype=int)
            The sample ratios r used to specify the number of samples of the
            lower fidelity models, e.g. N_i = r_i*nhf_samples,
            i=1,...,nmodels-1. For model i>0 nsample_ratio*nhf_samples equals
            the number of samples in the two different discrepancies involving
            the ith model. They correspond to rounded_target_cost.

        rounded_target_cost : float
            The cost of the new sample allocation

        optimized_criteria : float
            A function ofthe estimator covariance
            using the integer sample allocations, e.g. Trace(est_cov)
        """
        self._rounded_nsample_ratios = rounded_nsample_ratios.numpy()
        self._rounded_target_cost = rounded_target_cost
        self._npartition_samples = self._npartition_samples_from_ratios(
            self._rounded_target_cost, rounded_nsample_ratios)
        self._nsamples_per_model = torch.as_tensor(
            self._compute_nsamples_per_model(), dtype=torch.int)
        self._optimized_criteria = optimized_criteria

    def _allocate_samples_for_single_recursion(self, target_cost, verbosity=0):
        """
        Determine the samples (integers) that must be allocated to
        each model to compute the Monte Carlo like estimator

        Parameters
        ----------
        target_cost : float
            The total cost budget

        Returns
        -------
        nsample_ratios : np.ndarray (nmodels-1, dtype=int)
            The sample ratios r used to specify the number of samples of the
            lower fidelity models, e.g. N_i = r_i*nhf_samples,
            i=1,...,nmodels-1. For model i>0 nsample_ratio*nhf_samples equals
            the number of samples in the two different discrepancies involving
            the ith model.

        variance : float
            The variance of the estimator using the integer sample allocations

        rounded_target_cost : float
            The cost of the new sample allocation
        """
        nsample_ratios, obj_val = self._allocate_samples(
            target_cost)
        nsample_ratios = nsample_ratios.detach().numpy()
        rounded_nsample_ratios, rounded_target_cost = _round_nsample_ratios(
            target_cost, self._costs.numpy(), nsample_ratios)
        rounded_nsample_ratios = torch.as_tensor(
            rounded_nsample_ratios, dtype=torch.double)
        covariance = self._covariance_from_ratios(
            rounded_target_cost, rounded_nsample_ratios)
        val = self._optimization_criteria(covariance)
        self._set_optimized_params(
            rounded_nsample_ratios, rounded_target_cost, val)
        return rounded_nsample_ratios, covariance, rounded_target_cost

    def _allocate_samples_for_all_recursion_indices(
            self, target_cost, verbosity):
        best_variance = torch.as_tensor(np.inf, dtype=torch.double)
        best_result = None
        for index in get_acv_recursion_indices(
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
                best_result = [self._rounded_nsample_ratios,
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

    def _get_constraints(self, target_cost):
        # Must ensure that the samples of any model acting as a recursive
        # control variate has at least one more sample than its parent.
        # Otherwise Fmat will not be invertable sample ratios are rounded to
        # integers. Fmat is not invertable when two or more sample ratios
        # are equal
        cons = [
            {'type': 'ineq',
             'fun': partial(self._scipy_wrapper,
                            acv_sample_allocation_gmf_ratio_constraint),
             'jac': partial(self._scipy_wrapper,
                            acv_sample_allocation_gmf_ratio_constraint_jac),
             'args': (ii, jj, target_cost, self._costs)}
            for ii, jj in zip(range(1, self._nmodels), self._recursion_index)
            if jj > 0]
        return cons


class GISEstimator(ACVEstimator):
    """
    The GIS estimator from Gorodetsky et al. and Bomorito et al
    """
    def _create_allocation_matrix(self, recursion_index):
        self._allocation_mat = _get_allocation_matrix_acvis(
            recursion_index)

    def _get_constraints(self, target_cost):
        return []


class GRDEstimator(ACVEstimator):
    """
    The GRD estimator.
    """
    def _create_allocation_matrix(self, recursion_index):
        self._allocation_mat = _get_allocation_matrix_acvrd(
            recursion_index)

    def _get_constraints(self, target_cost):
        # raise NotImplementedError
        return []


class MFMCEstimator(GMFEstimator):
    def __init__(self, stat, costs, variable, cov, opt_criteria=None,
                 opt_qoi=0):
        # Use the sample analytical sample allocation for estimating a scalar
        # mean when estimating any statistic
        super().__init__(stat, costs, variable, cov,
                         recursion_index=None, opt_criteria=None)
        # The qoi index used to generate the sample allocation
        self._opt_qoi = opt_qoi

    def _allocate_samples(self, target_cost):
        # nsample_ratios returned will be listed in according to
        # self.model_order which is what self.get_rsquared requires
        nqoi = self._cov.shape[0]//len(self._costs)
        nsample_ratios, val = allocate_samples_mfmc(
            self._cov.numpy()[self._opt_qoi::nqoi, self._opt_qoi::nqoi],
            self._costs.numpy(), target_cost)
        return torch.as_tensor(nsample_ratios, dtype=torch.double), val

    def _get_allocation_matrix(self):
        return get_sample_allocation_matrix_mfmc(self._nmodels)

    def _get_npartition_samples(self, nsamples_per_model):
        return get_npartition_samples_mfmc(nsamples_per_model)


class ACVMLMCEstimator(ACVEstimator):
    def __init__(self, stat, costs, variable, cov, opt_criteria=None,
                 opt_qoi=0):
        """
        Use the sample analytical sample allocation for estimating a scalar
        mean when estimating any statistic

        Use optimal ACV weights instead of all weights=-1 used by
        classical MLMC.
        """
        super().__init__(stat, costs, variable, cov,
                         recursion_index=None, opt_criteria=None)
        # The qoi index used to generate the sample allocation
        self._opt_qoi = opt_qoi

    def _allocate_samples(self, target_cost):
        nqoi = self._cov.shape[0]//len(self._costs)
        nsample_ratios, val = allocate_samples_mlmc(
            self._cov.numpy()[self._opt_qoi::nqoi, self._opt_qoi::nqoi],
            self._costs.numpy(), target_cost)
        return torch.as_tensor(nsample_ratios, dtype=torch.double), val

    def _create_allocation_matrix(self):
        return get_sample_allocation_matrix_mlmc(self._nmodels)

    def _get_allocation_matrix(self):
        return get_sample_allocation_matrix_mlmc(self._nmodels)

    def _get_npartition_samples(self, nsamples_per_model):
        return get_npartition_samples_mlmc(nsamples_per_model)

    def _get_variance(self, npartition_samples):
        CF, cf = self.stat.get_discrepancy_covariances(
            self, npartition_samples)
        weights = self._weights(CF, cf)
        return (
            self.stat.high_fidelity_estimator_covariance(npartition_samples[0])
            + torch.linalg.multi_dot((weights, CF, weights.T))
            + torch.linalg.multi_dot((cf, weights.T))
            + torch.linalg.multi_dot((weights, cf.T))
        )


class MLMCEstimator(ACVMLMCEstimator):
    """
    The classical MLMC estimator that weits all control variate weights to -1
    """
    def _weights(self, CF, cf):
        return -torch.ones(cf.shape, dtype=torch.double)


class BestModelSubsetEstimator():
    def __init__(self, estimator_type, stat_type, variable, costs, cov,
                 max_nmodels, *est_args, **est_kwargs):
        self.estimator_type = estimator_type
        self.stat_type = stat_type
        self._candidate_cov, self._candidate_costs = cov, np.asarray(costs)
        self.variable = variable
        # self._ncandidate_nmodels is the number of total models
        # self._nmodels returns number of models in best subset
        self._ncandidate_models = len(self._candidate_costs)
        self.nqoi = self._candidate_cov.shape[0]//self._ncandidate_models
        self.max_nmodels = max_nmodels
        self.args = est_args
        self._allow_failures = est_kwargs.get("allow_failures", False)
        if "allow_failures" in est_kwargs:
            del est_kwargs["allow_failures"]
        self.kwargs = est_kwargs

        # self._optimized_criteria = None
        # self._rounded_target_cost = None
        self.best_est = None
        self.best_model_indices = None
        self._all_model_labels = None

    @property
    def model_labels(self):
        return [self._all_model_labels[idx] for idx in self.best_model_indices]

    @model_labels.setter
    def model_labels(self, labels):
        self._all_model_labels = labels

    def _get_model_subset_estimator(self, qoi_idx,
                                    nsubset_lfmodels, allocate_kwargs,
                                    target_cost, lf_model_subset_indices):
        idx = np.hstack(([0], lf_model_subset_indices)).astype(int)
        subset_cov = _nqoi_nqoi_subproblem(
            self._candidate_cov, self._ncandidate_models, self.nqoi,
            idx, qoi_idx)
        subset_costs = self._candidate_costs[idx]
        sub_args = multioutput_stats[self.stat_type].args_model_subset(
            self._ncandidate_models, self.nqoi, idx, *self.args)
        sub_kwargs = copy.deepcopy(self.kwargs)
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
                self.estimator_type, self.stat_type, self.variable,
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
                    idx, est.optimized_criteria.item())
                print(msg)
            return est
        except (RuntimeError, ValueError) as e:
            if self._allow_failures:
                return None
            raise e

    def _get_best_models_for_acv_estimator(
            self, target_cost, **allocate_kwargs):
        if self.max_nmodels is None:
            max_nmodels = self._ncandidate_nmodels
        else:
            max_nmodels = self.max_nmodels
            lf_model_indices = np.arange(1, self._ncandidate_models)
        best_criteria = np.inf
        best_est, best_model_indices = None, None
        qoi_idx = np.arange(self.nqoi)
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
                    np.array(est.optimized_criteria)
                    if est is not None else np.inf for est in result]
                II = np.argmin(criteria)
                if not np.isfinite(criteria[II]):
                    best_est = None
                else:
                    best_est = result[II]
                    best_model_indices = np.hstack(
                        ([0], indices[II])).astype(int)
                    best_criteria = best_est.optimized_criteria
                continue

            for lf_model_subset_indices in combinations(
                    lf_model_indices, nsubset_lfmodels):
                est = self._get_model_subset_estimator(
                    qoi_idx, nsubset_lfmodels, allocate_kwargs,
                    target_cost, lf_model_subset_indices)
                if est is not None and est.optimized_criteria < best_criteria:
                    best_est = est
                    best_model_indices = np.hstack(
                        ([0], lf_model_subset_indices)).astype(int)
                    best_criteria = best_est.optimized_criteria
        if best_est is None:
            raise RuntimeError("No solutions found for any model subset")
        return best_est, best_model_indices

    def allocate_samples(self, target_cost, **allocate_kwargs):
        if self.estimator_type == "mc":
            best_model_indices = np.array([0])
            args = multioutput_stats[self.stat_type].args_model_subset(
                self._ncandidate_models, self.nqoi, best_model_indices,
                *self.args)
            best_est = get_estimator(
                self.estimator_type, self.stat_type, self.variable,
                self._candidate_costs[:1], self._candidate_cov[:1, :1],
                *args, **self.kwargs)
            best_est.allocate_samples(target_cost)

        else:
            best_est, best_model_indices = (
                self._get_best_models_for_acv_estimator(
                    target_cost, **allocate_kwargs))
        # self._optimized_criteria = best_est.optimized_criteria
        # self._rounded_target_cost = best_est.rounded_target_cost
        self.best_est = best_est
        self.best_model_indices = best_model_indices
        self._set_best_est_attributes()

    def _set_best_est_attributes(self):
        # allow direct access of important self.best_est attributes
        # __call__ cannot be set using this approach.
        attr_list = ["separate_model_values", "separate_model_samples",
                     "_get_variance", "combine_acv_samples",
                     "combine_acv_values", "_generate_estimator_samples",
                     "bootstrap", "nsample_ratios", "stat",
                     "nmodels", "cov", "nsamples_per_model", "costs",
                     "get_variance", "optimized_criteria",
                     "rounded_target_cost",
                     "_get_allocation_matrix",
                     "_get_npartition_samples"]
        for attr in attr_list:
            setattr(self, attr, getattr(self.best_est, attr))

    def generate_data(self, functions, pilot_data=[None, None]):
        if pilot_data[0] is not None:
            # downselect pilot data to just contain data from selected models
            pilot_data_subset = [
                pilot_data[0],
                [pilot_data[1][idx] for idx in self.best_model_indices]]
        else:
            pilot_data_subset = [None, None]
        return self.best_est.generate_data(
            [functions[idx] for idx in self.best_model_indices],
            pilot_data_subset)

    def __repr__(self):
        if self._optimized_criteria is None:
            return "{0}".format(self.__class__.__name__)
        return "{0}(est={1}, subset={2})".format(
            self.__class__.__name__, self.best_est, self.best_model_indices)

    def combine_pilot_data(
            self, samples_per_model_wo_pilot, values_per_model_wo_pilot,
            pilot_data):
        """
        pilot data contains the data from all models. Internally we downselect
        the correct data
        """
        pilot_samples, pilot_values = pilot_data
        pilot_data = [pilot_samples,
                      [pilot_values[idx] for idx in self.best_model_indices]]
        return self.best_est.combine_pilot_data(
            samples_per_model_wo_pilot, values_per_model_wo_pilot,
            pilot_data)

    def __call__(self, values):
        return self.best_est(values)


multioutput_estimators = {
    "gmf": GMFEstimator,
    "gis": GISEstimator,
    "grd": GRDEstimator,
    "mfmc": MFMCEstimator,
    "mlmc": MLMCEstimator,
    "acvmlmc": ACVMLMCEstimator,
    "mc": MCEstimator}


multioutput_stats = {
    "mean": MultiOutputMean,
    "variance": MultiOutputVariance,
    "mean_variance": MultiOutputMeanAndVariance,
}


def get_estimator(estimator_type, stat_type, costs, cov, *args,
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
        nmodels = len(costs)
        stat = multioutput_stats[stat_type](nmodels, cov, *args)
        return multioutput_estimators[estimator_type](
            stat, costs, cov, **kwargs)

    return BestModelSubsetEstimator(
        estimator_type, stat_type, costs, cov,
        max_nmodels, *args, **kwargs)


def plot_estimator_variances(optimized_estimators,
                             est_labels, ax, ylabel=None,
                             relative_id=0, cost_normalization=1,
                             criteria=determinant_variance):
    """
    Plot variance as a function of the total cost for a set of estimators.

    Parameters
    ----------
    optimized_estimators : list
         Each entry is a list of optimized estimators for a set of target costs

    est_labels : list (nestimators)
        String used to label each estimator

    relative_id the model id used to normalize variance
    """
    from pyapprox.util.configure_plots import mathrm_label
    linestyles = ['-', '--', ':', '-.', (0, (5, 10)), '-']
    nestimators = len(est_labels)
    est_criteria = []
    for ii in range(nestimators):
        est_total_costs = np.array(
            [est.rounded_target_cost for est in optimized_estimators[ii]])
        est_criteria.append(np.array(
            [criteria(est._get_variance(est.nsamples_per_model), est)
             for est in optimized_estimators[ii]]))
    est_total_costs *= cost_normalization
    for ii in range(nestimators):
        ax.loglog(est_total_costs,
                  est_criteria[ii]/est_criteria[relative_id][0],
                  label=est_labels[ii], ls=linestyles[ii], marker='o')
    if ylabel is None:
        ylabel = mathrm_label("Estimator variance")
    ax.set_xlabel(mathrm_label("Target cost"))
    ax.set_ylabel(ylabel)
    ax.legend()


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
        self.stat_type = stat_type
        self.qoi_idx = qoi_idx

    def __call__(self, est_covariance, est):
        if self.stat_type != "mean" and isinstance(
                est.stat, MultiOutputMeanAndVariance):
            return est_covariance[est.nqoi+self.qoi_idx, est.nqoi+self.qoi_idx]
        elif (isinstance(
                est.stat, (MultiOutputVariance, MultiOutputMean)) or
              self.stat_type == "mean"):
            return est_covariance[self.qoi_idx, self.qoi_idx]
        raise ValueError("{0} not supported".format(est.stat))

    def __repr__(self):
        return "{0}(stat={1}, qoi={2})".format(
            self.__class__.__name__, self.stat_type, self.qoi_idx)


def plot_estimator_variance_reductions(optimized_estimators,
                                       est_labels, ax, ylabel=None,
                                       criteria=determinant_variance,
                                       **bar_kawrgs):
    """
    Plot variance as a function of the total cost for a set of estimators.

    Parameters
    ----------
    optimized_estimators : list
         Each entry is a list of optimized estimators for a set of target costs

    est_labels : list (nestimators)
        String used to label each estimator

    """
    from pyapprox.util.configure_plots import mathrm_label
    var_red, est_criterias, sf_criterias = [], [], []
    optimized_estimators = optimized_estimators.copy()
    est_labels = est_labels.copy()
    nestimators = len(est_labels)
    for ii in range(nestimators):
        assert len(optimized_estimators[ii]) == 1
        est = optimized_estimators[ii][0]
        est_criteria = criteria(est._get_variance(est.nsamples_per_model), est)
        nhf_samples = int(est.rounded_target_cost/est.costs[0])
        sf_criteria = criteria(
            est.stat.high_fidelity_estimator_covariance(
                [nhf_samples]), est)
        var_red.append(sf_criteria/est_criteria)
        sf_criterias.append(sf_criteria)
        est_criterias.append(est_criteria)
    rects = ax.bar(est_labels, var_red, **bar_kawrgs)
    rects = [r for r in rects]  # convert to list
    from pyapprox.multifidelity.monte_carlo_estimators import _autolabel
    _autolabel(ax, rects, ['$%1.2f$' % (v) for v in var_red])
    if ylabel is None:
        ylabel = mathrm_label("Estimator variance reduction")
    ax.set_ylabel(ylabel)
    return var_red, est_criterias, sf_criterias
