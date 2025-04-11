import copy
from abc import abstractmethod, ABC
from functools import partial
from typing import List, Union, Tuple

import numpy as np

from pyapprox.util.utilities import get_correlation_from_covariance
from pyapprox.multifidelity.stats import (
    MultiOutputVariance,
    MultiOutputMeanAndVariance,
    MultiOutputStatistic,
)
from pyapprox.multifidelity._visualize import (
    _plot_allocation_matrix,
    _plot_model_recursion,
)
from pyapprox.multifidelity._optim import (
    _allocate_samples_mlmc,
    _allocate_samples_mfmc,
    _check_mfmc_model_costs_and_correlations,
    _get_sample_allocation_matrix_mlmc,
    _get_sample_allocation_matrix_mfmc,
    _get_acv_recursion_indices,
)
from pyapprox.util.backends.template import Array, BackendMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.interface.model import SingleSampleModel
from pyapprox.optimization.scipy import (
    ScipyConstrainedOptimizer,
    ScipyConstrainedNelderMeadOptimizer,
)
from pyapprox.optimization.minimize import (
    Constraint,
    Optimizer,
    ChainedOptimizer,
    OptimizationResult,
)


def _combine_acv_values(
    reorder_allocation_mat: Array,
    npartition_samples: Array,
    acv_values: List,
    bkd: BackendMixin,
) -> List[Array]:
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
            if reorder_allocation_mat[jj, 2 * ii] == 1:
                ub = lb + int(npartition_samples[jj])
                values_per_model[ii] += [acv_values[ii][0][lb:ub]]
                lb = ub
                found = True
            if reorder_allocation_mat[jj, 2 * ii + 1] == 1:
                # there is no need to enter here is samle set has already
                # been added by acv_values[ii][0], hence the use of elseif here
                ub2 = lb2 + int(npartition_samples[jj])
                if not found:
                    values_per_model[ii] += [acv_values[ii][1][lb2:ub2]]
                lb2 = ub2
        values_per_model[ii] = bkd.vstack(values_per_model[ii])
    return values_per_model


def _combine_acv_samples(
    reorder_allocation_mat: Array,
    npartition_samples: Array,
    acv_samples: List,
    bkd: BackendMixin,
) -> List[Array]:
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
            if reorder_allocation_mat[jj, 2 * ii] == 1:
                ub = lb + int(npartition_samples[jj])
                samples_per_model[ii] += [acv_samples[ii][0][:, lb:ub]]
                lb = ub
                found = True
            if reorder_allocation_mat[jj, 2 * ii + 1] == 1:
                ub2 = lb2 + int(npartition_samples[jj])
                if not found:
                    # Only add samples if they were not in Z_m^*
                    samples_per_model[ii] += [acv_samples[ii][1][:, lb2:ub2]]
                    lb2 = ub2
                    samples_per_model[ii] = bkd.hstack(samples_per_model[ii])
    return samples_per_model


def _get_allocation_matrix_gmf(
    recursion_index: Array, bkd: BackendMixin
) -> Array:
    nmodels = len(recursion_index) + 1
    mat = bkd.zeros((nmodels, 2 * nmodels))
    for ii in range(nmodels):
        mat[ii, 2 * ii + 1] = 1.0
    for ii in range(1, nmodels):
        mat[:, 2 * ii] = mat[:, recursion_index[ii - 1] * 2 + 1]
    for ii in range(2, 2 * nmodels):
        II = bkd.where(mat[:, ii] == 1)[0][-1]
        mat[:II, ii] = 1.0
    return mat


def _get_allocation_matrix_acvis(
    recursion_index: Array, bkd: BackendMixin
) -> Array:
    nmodels = len(recursion_index) + 1
    mat = bkd.zeros((nmodels, 2 * nmodels))
    for ii in range(nmodels):
        mat[ii, 2 * ii + 1] = 1
    for ii in range(1, nmodels):
        mat[:, 2 * ii] = mat[:, recursion_index[ii - 1] * 2 + 1]
    for ii in range(1, nmodels):
        mat[:, 2 * ii + 1] = bkd.maximum(mat[:, 2 * ii], mat[:, 2 * ii + 1])
    return mat


def _get_allocation_matrix_acvrd(
    recursion_index: Array, bkd: BackendMixin
) -> Array:
    nmodels = len(recursion_index) + 1
    allocation_mat = bkd.zeros((nmodels, 2 * nmodels))
    for ii in range(nmodels):
        allocation_mat[ii, 2 * ii + 1] = 1
    for ii in range(1, nmodels):
        allocation_mat[:, 2 * ii] = allocation_mat[
            :, recursion_index[ii - 1] * 2 + 1
        ]
    return allocation_mat


class MCEstimator:
    def __init__(
        self,
        stat: MultiOutputStatistic,
        costs: Union[List, Array],
        opt_criteria: callable = None,
    ):
        r"""
        Parameters
        ----------
        stat : :class:`~pyapprox.multifidelity.multioutput_monte_carlo.MultiOutputStatistic`
            Object defining what statistic will be calculated

        costs : Array (nmodels)
            The relative costs of evaluating each model
        """
        self._bkd = stat._bkd

        self._stat, self._costs = self._check_inputs(stat, costs)
        self._optimization_criteria = self._log_determinant_variance

        self._rounded_nsamples_per_model = None
        self._rounded_npartition_samples = None
        self._rounded_target_cost = None
        self._optimized_criteria = None
        self._optimized_covariance = None
        self._model_labels = None
        self._npartitions = 1

    def _check_inputs(
        self, stat: MultiOutputStatistic, costs: Union[List, Array]
    ) -> Tuple[Array, Array, int, int]:
        if not isinstance(stat, MultiOutputStatistic):
            raise ValueError(
                "stat must be an instance of MultiOutputStatistic"
            )

        costs = self._bkd.atleast1d(costs)
        if costs.ndim != 1:
            raise ValueError("costs is not a 1D iterable")
        self._nmodels = stat._nmodels
        return stat, costs

    def _log_determinant_variance(self, variance: Array) -> float:
        # Only compute large eigvalues as the variance will
        # be singular when estimating variance or mean+variance
        # because of the duplicate entries in
        # the covariance matrix
        eigvals = self._bkd.eigh(variance)[0]
        val = self._bkd.log(eigvals[eigvals > 1e-14]).sum()
        return val

    def _covariance_from_npartition_samples(
        self, npartition_samples: Array
    ) -> Array:
        """
        Get the variance of the Monte Carlo estimator from costs and cov
        and npartition_samples
        """
        return self._stat.high_fidelity_estimator_covariance(
            npartition_samples[0]
        )

    def optimized_covariance(self) -> float:
        """
        Return the estimator covariance at the optimal sample allocation
        computed using self.allocate_samples()
        """
        return self._optimized_covariance

    def allocate_samples(self, target_cost: float):
        """
        Find the optimal number of samples that minimize the metric of the
        estimator covvariance for the specficied target cost.

        Parameters
        ----------
        target_cost : float
            The total computational budget that can be used to compute the
            estimator
        """
        self._rounded_nsamples_per_model = self._bkd.asarray(
            [int(np.floor(target_cost / self._costs[0]))]
        )
        self._rounded_npartition_samples = self._rounded_nsamples_per_model
        est_covariance = self._covariance_from_npartition_samples(
            self._rounded_npartition_samples
        )
        self._optimized_covariance = est_covariance
        optimized_criteria = self._optimization_criteria(est_covariance)
        self._rounded_target_cost = (
            self._costs[0] * self._rounded_nsamples_per_model[0]
        )
        self._optimized_criteria = optimized_criteria

    def generate_samples_per_model(self, rvs: callable) -> List[Array]:
        """
        Returns the samples needed to the model

        Parameters
        ----------
        rvs : callable
            Function with signature

            `rvs(nsamples)->Array (nvars, nsamples)`

        Returns
        -------
        samples_per_model : list[Array] (1)
            List with one entry Array (nvars, nsamples_per_model[0])
        """
        return [rvs(self._rounded_nsamples_per_model[0])]

    def __call__(self, values: Array) -> Array:
        """
        Return the value of the estimator using a set of model evaluations.

        Parameters
        ----------
        values: Array
            The values of each model output at the optimal number of samples

        Return
        ------
        stat_value: Array
            The value of the estimate statistic
        """
        if not isinstance(values, self._bkd.array_type()):
            raise ValueError(
                "values must be an {0} but type={1}".format(
                    self._bkd.array_type(), type(values)
                )
            )
        if (values.ndim != 2) or (
            values.shape[0] != self._rounded_nsamples_per_model[0]
        ):
            msg = "values has the incorrect shape {0} expected {1}".format(
                values.shape,
                (self._rounded_nsamples_per_model[0], self._stat._nqoi),
            )
            raise ValueError(msg)
        return self._stat.sample_estimate(values)

    def bootstrap(
        self, values: List[Array], nbootstraps: int = 1000
    ) -> Tuple[Array, Array]:
        r"""
        Approximate the variance of the estimator using
        bootstraping. The accuracy of bootstapping depends on the number
        of values per model. As it gets large the boostrapped statistics
        will approach the theoretical values.

        Parameters
        ----------
        values : [Array (nsamples, nqoi)]
            A single entry list containing the unique values of each model.
            The list is required to allow consistent interface with
            multi-fidelity estimators

        nbootstraps : integer
            The number of boostraps used to compute estimator variance

        Returns
        -------
        bootstrap_stats : Array
            The bootstrap estimate of the estimator

        bootstrap_covar : Array
            The bootstrap estimate of the estimator covariance
        """
        nbootstraps = int(nbootstraps)
        estimator_vals = self._bkd.empty((nbootstraps, self._stat._nqoi))
        nsamples = values[0].shape[0]
        indices = self._bkd.arange(nsamples, dtype=int)
        for kk in range(nbootstraps):
            bootstrapped_indices = self._bkd.array(
                np.random.choice(indices, size=nsamples, replace=True),
                dtype=int,
            )
            estimator_vals[kk] = self._stat.sample_estimate(
                values[0][bootstrapped_indices]
            )
            bootstrap_mean = estimator_vals.mean(axis=0)
            bootstrap_covar = self._bkd.cov(
                estimator_vals, rowvar=False, ddof=1
            )
        return bootstrap_mean, bootstrap_covar

    def __repr__(self):
        if self._optimized_criteria is None:
            return "{0}(stat={1}, nqoi={2})".format(
                self.__class__.__name__, self._stat, self._stat._nqoi
            )
        rep = "{0}(stat={1}, criteria={2:.3g}".format(
            self.__class__.__name__, self._stat, self._optimized_criteria
        )
        rep += " target_cost={0:.5g}, nsamples={1})".format(
            self._rounded_target_cost, self._rounded_nsamples_per_model[0]
        )
        return rep


class CVEstimator(MCEstimator):
    def __init__(
        self,
        stat: MultiOutputStatistic,
        costs: Union[List, Array],
        lowfi_stats: Array = None,
        opt_criteria: callable = None,
    ):
        super().__init__(stat, costs, opt_criteria=opt_criteria)
        if lowfi_stats is not None:
            if lowfi_stats.shape != (self._nmodels - 1, self._stat.nstats()):
                raise ValueError(
                    "lowfi_stats must be a 2D Array with shape {0} "
                    "but has shape {1}".format(
                        (self._nmodels - 1, self._stat.nstats()),
                        lowfi_stats.shape,
                    )
                )
        self._lowfi_stats = lowfi_stats

        self._optimized_CF = None
        self._optimized_cf = None
        self._optimized_weights = None

        self._best_model_indices = self._bkd.arange(len(costs), dtype=int)

    def _get_discrepancy_covariances(
        self, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        return self._stat._get_cv_discrepancy_covariances(npartition_samples)

    def _covariance_from_npartition_samples(
        self, npartition_samples: Array
    ) -> Array:
        CF, cf = self._get_discrepancy_covariances(npartition_samples)
        weights = self._weights(CF, cf)
        return self._stat.high_fidelity_estimator_covariance(
            npartition_samples[0]
        ) + self._bkd.multidot((weights, cf.T))

    def _set_optimized_params_base(
        self,
        rounded_npartition_samples: Array,
        rounded_nsamples_per_model: Array,
        rounded_target_cost: float,
    ):
        r"""
        Set the parameters needed to generate samples for evaluating the
        estimator

        Parameters
        ----------
        rounded_npartition_samples : Array (npartitions, dtype=int)
            The number of samples in the independent sample partitions.

        rounded_nsamples_per_model :  Array (nmodels)
            The number of samples allocated to each model

        rounded_target_cost : float
            The cost of the new sample allocation

        Sets attributes
        ----------------
        self._rounded_target_cost : float
            The computational cost of the estimator using the rounded
            npartition_samples

        self._rounded_npartition_samples :  Array (npartitions)
            The number of samples in each partition corresponding to the
            rounded partition_ratios

        self._rounded_nsamples_per_model :  Array (nmodels)
            The number of samples allocated to each model

        self._optimized_covariance : Array (nstats, nstats)
            The optimal estimator covariance

        self._optimized_criteria: float
            The value of the sample allocation objective using the rounded
            partition_ratios

        self._rounded_nsamples_per_model : Array (nmodels)
            The number of samples allocated to each model using the rounded
            partition_ratios

        self._optimized_CF : Array (nstats*(nmodels-1),nstats*(nmodels-1))
            The covariance between :math:`\Delta_i`, :math:`\Delta_j`

        self._optimized_cf : Array (nstats, nstats*(nmodels-1))
            The covariance between :math:`Q_0`, :math:`\Delta_j`

        self._optimized_weights : Array (nstats, nmodels-1)
            The optimal control variate weights
        """
        self._rounded_npartition_samples = rounded_npartition_samples
        self._rounded_nsamples_per_model = rounded_nsamples_per_model
        self._rounded_target_cost = rounded_target_cost
        self._optimized_covariance = self._covariance_from_npartition_samples(
            self._rounded_npartition_samples
        )
        self._optimized_criteria = self._optimization_criteria(
            self._optimized_covariance
        )
        self._optimized_CF, self._optimized_cf = (
            self._get_discrepancy_covariances(self._rounded_npartition_samples)
        )
        self._optimized_weights = self._weights(
            self._optimized_CF, self._optimized_cf
        )

    def _estimator_cost(self, npartition_samples: Array) -> float:
        return (npartition_samples[0] * self._costs).sum()

    def _set_optimized_params(self, rounded_npartition_samples: Array):
        rounded_target_cost = self._estimator_cost(rounded_npartition_samples)
        self._set_optimized_params_base(
            rounded_npartition_samples,
            rounded_npartition_samples,
            rounded_target_cost,
        )

    def allocate_samples(self, target_cost: float):
        npartition_samples = [target_cost / self._costs.sum()]
        rounded_npartition_samples = [int(np.floor(npartition_samples[0]))]
        if isinstance(
            self._stat, (MultiOutputVariance, MultiOutputMeanAndVariance)
        ):
            min_nhf_samples = 2
        else:
            min_nhf_samples = 1
        if rounded_npartition_samples[0] < min_nhf_samples:
            msg = "target_cost is to small. Not enough samples of each model"
            msg += " can be taken {0} < {1}".format(
                npartition_samples[0], min_nhf_samples
            )
            raise ValueError(msg)

        rounded_nsamples_per_model = self._bkd.full(
            (self._nmodels,), rounded_npartition_samples[0]
        )
        rounded_target_cost = (self._costs * rounded_nsamples_per_model).sum()
        self._set_optimized_params_base(
            rounded_npartition_samples,
            rounded_nsamples_per_model,
            rounded_target_cost,
        )

    def generate_samples_per_model(self, rvs: callable) -> List[Array]:
        """
        Returns the samples needed to the model

        Parameters
        ----------
        rvs : callable
            Function with signature

            `rvs(nsamples)->Array (nvars, nsamples)`

        Returns
        -------
        samples_per_model : list[Array] (1)
            List with one entry Array (nvars, nsamples_per_model[0])
        """
        samples = rvs(self._rounded_nsamples_per_model[0])
        return [self._bkd.copy(samples) for ii in range(self._nmodels)]

    def _weights(self, CF, cf):
        return -self._bkd.multidot((self._bkd.pinv(CF), cf.T)).T

    def _covariance_non_optimal_weights(
        self, hf_est_covar: Array, weights: Array, CF: Array, cf: Array
    ) -> Array:
        # The expression below, e.g. Equation 8
        # from Dixon 2024, can be used for non optimal control variate weights
        # Warning: Even though this function is general,
        # it should only ever be used for MLMC, because
        # expression for optimal weights is more efficient
        return (
            hf_est_covar
            + self._bkd.multidot((weights, CF, weights.T))
            + self._bkd.multidot((cf, weights.T))
            + self._bkd.multidot((weights, cf.T))
        )

    def _estimate(
        self,
        values_per_model: List[Array],
        weights: Array,
        bootstrap: bool = False,
    ) -> Array:
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
            indices = self._bkd.arange(nsamples, dtype=int)
        if bootstrap:
            indices = self._bkd.array(
                np.random.choice(indices, size=indices.shape[0], replace=True),
                dtype=int,
            )

        deltas = self._bkd.hstack(
            [
                self._stat.sample_estimate(values_per_model[ii][indices])
                - self._lowfi_stats[ii - 1]
                for ii in range(1, self._nmodels)
            ]
        )
        est = (
            self._stat.sample_estimate(values_per_model[0][indices])
            + weights @ deltas
        )
        return est

    def __call__(self, values_per_model: List[Array]) -> Array:
        r"""
        Return the value of the Monte Carlo like estimator

        Parameters
        ----------
        values_per_model : list (nmodels)
            The unique values of each model

        Returns
        -------
        est : Array (nqoi, nqoi)
            The covariance of the estimator values for
            each high-fidelity model QoI
        """
        for vals in values_per_model:
            if not isinstance(vals, self._bkd.array_type()):
                raise ValueError(
                    "vals must be an instance of {0}".format(
                        self._bkd.array_type()
                    )
                )
        return self._estimate(values_per_model, self._optimized_weights)

    def insert_pilot_values(
        self, pilot_values: List[Array], values_per_model: List[Array]
    ) -> List[Array]:
        """
        Only add pilot values to the fist indepedent partition and thus
        only to models that use that partition
        """
        new_values_per_model = []
        for ii in range(self._nmodels):
            active_partition = (self._allocation_mat[0, 2 * ii] == 1) or (
                self._allocation_mat[0, 2 * ii + 1] == 1
            )
            if active_partition:
                new_values_per_model.append(
                    self._bkd.vstack((pilot_values[ii], values_per_model[ii]))
                )
            else:
                new_values_per_model.append(values_per_model[ii].copy())
        return new_values_per_model

    def __repr__(self):
        if self._optimized_criteria is None:
            return "{0}(stat={1}, recursion_index={2})".format(
                self.__class__.__name__, self._stat, self._recursion_index
            )
        rep = "{0}(stat={1}, criteria={2:.3g}".format(
            self.__class__.__name__, self._stat, self._optimized_criteria
        )
        rep += " target_cost={0:.5g}, nsamples={1})".format(
            self._rounded_target_cost, self._rounded_nsamples_per_model[0]
        )
        return rep

    def bootstrap(
        self,
        values_per_model: List[Array],
        nbootstraps: int = 1000,
        mode: str = "values",
        pilot_values: List[Array] = None,
    ):
        modes = ["values", "values_weights", "weights"]
        if mode not in modes:
            raise ValueError("mode must be in {0}".format(modes))
        if pilot_values is not None and mode not in modes[1:]:
            raise ValueError(
                "pilot_values given by mode not in {0}".format(modes[1:])
            )
        bootstrap_vals = mode in modes[:2]
        bootstrap_weights = mode in modes[1:]
        nbootstraps = int(nbootstraps)
        estimator_vals = []
        if bootstrap_weights:
            npilot_samples = pilot_values[0].shape[0]
            self_stat = copy.deepcopy(self._stat)
            weights_list = []
        for kk in range(nbootstraps):
            if bootstrap_weights:
                indices = self._bkd.array(
                    np.random.choice(
                        np.arange(npilot_samples, dtype=int),
                        size=npilot_samples,
                        replace=True,
                    ),
                    dtype=int,
                )
                boostrap_pilot_values = [
                    vals[indices] for vals in pilot_values
                ]
                self._stat.set_pilot_quantities(
                    *self._stat.compute_pilot_quantities(boostrap_pilot_values)
                )
                CF, cf = self._get_discrepancy_covariances(
                    self._rounded_npartition_samples
                )
                weights = self._weights(CF, cf)
                weights_list.append(weights.flatten())
            else:
                weights = self._optimized_weights
            estimator_vals.append(
                self._estimate(
                    values_per_model, weights, bootstrap=bootstrap_vals
                ).flatten()
            )
        estimator_vals = self._bkd.stack(estimator_vals)
        bootstrap_values_mean = estimator_vals.mean(axis=0)
        bootstrap_values_covar = self._bkd.cov(
            estimator_vals, rowvar=False, ddof=1
        )
        if bootstrap_weights:
            self._stat = self_stat
            weights_list = self._bkd.stack(weights_list)
            bootstrap_weights_mean = weights_list.mean(axis=0)
            bootstrap_weights_covar = self._bkd.cov(
                weights_list, rowvar=False, ddof=1
            )
            return (
                bootstrap_values_mean,
                bootstrap_values_covar,
                bootstrap_weights_mean,
                bootstrap_weights_covar,
            )
        return (bootstrap_values_mean, bootstrap_values_covar)


class ACVObjective(SingleSampleModel, ABC):
    def __init__(
        self,
        scaling: float = 1,
        backend: BackendMixin = TorchMixin,
    ):
        self._scaling = scaling
        super().__init__(backend)

    def set_target_cost(self, target_cost: float):
        self._target_cost = target_cost

    def nvars(self) -> int:
        return self._est._nmodels - 1

    def set_estimator(self, est: "ACVEstimator"):
        if not isinstance(est, ACVEstimator):
            raise ValueError("est must be an instance of ACVEstimator")
        if not est._bkd.jacobian_implemented():
            raise ValueError(
                "Optimization requires "
                "est._bkd.jacobian_implemented() be true"
            )
        self._est = est
        self._bkd = est._bkd

    def jacobian_implemented(self) -> bool:
        return True

    def apply_hessian_implemented(self) -> bool:
        # autograd implementation is slow so turn off
        return False  # self._bkd.hvp_implemented()

    @abstractmethod
    def _optimization_criteria(self, est_covariance: Array) -> float:
        raise NotImplementedError

    def _objective_value(self, partition_ratios: Array) -> Array:
        if partition_ratios.shape[1] != 1:
            raise ValueError(
                "partition_ratios must be a 2D array with one column"
            )
        est_covariance = self._est._covariance_from_partition_ratios(
            self._target_cost, partition_ratios[:, 0]
        )
        return self._optimization_criteria(est_covariance) * self._scaling

    def _evaluate(self, partition_ratios: Array) -> Array:
        return self._bkd.atleast2d(self._objective_value(partition_ratios))

    def _jacobian(self, partition_ratios: Array) -> Array:
        return self._bkd.jacobian(self._objective_value, partition_ratios).T

    def _apply_hessian(self, partition_ratios: Array, vvec: Array) -> Array:
        return self._bkd.hvp(self._objective_value, partition_ratios, vvec)

    def nqoi(self) -> int:
        return 1


class ACVLogDeterminantObjective(ACVObjective):
    def _optimization_criteria(self, est_covariance: Array) -> float:
        # Only compute large eigvalues as the variance will
        # be singular when estimating variance or mean+variance
        # because of the duplicate entries in
        # the covariance matrix
        eigvals = self._bkd.eigh(est_covariance)[0]
        return self._bkd.log(eigvals[eigvals > 1e-14]).sum()


class ACVPartitionConstraint(Constraint):
    def __init__(
        self,
        est: "ACVEstimator",
        target_cost: float,
    ):
        if not isinstance(est, ACVEstimator):
            raise ValueError("est must be an instance of ACVEstimator")
        if not est._bkd.jacobian_implemented():
            raise ValueError(
                "Optimization requires "
                "est._bkd.jacobian_implemented() be true"
            )
        self._est = est
        self._target_cost = target_cost
        bkd = est._bkd
        bounds = bkd.stack(
            (
                bkd.zeros(self._est._npartitions),
                bkd.full((self._est._npartitions,), np.inf),
            ),
            axis=1,
        )
        super().__init__(bounds, keep_feasible=True, backend=est._bkd)

    def jacobian_implemented(self) -> bool:
        return True

    def apply_hessian_implemented(self) -> bool:
        # autograd implementation is slow so turn off
        return False  # self._bkd.hvp_implemented()

    def nvars(self) -> int:
        return self._est._nmodels - 1

    def _eval_constraint(self, partition_ratios: Array) -> Array:
        if partition_ratios.ndim != 1:
            raise ValueError("partition_ratios.ndim != 1")
        nsamples = self._est._npartition_samples_from_partition_ratios(
            self._target_cost, partition_ratios
        )
        vals = nsamples - self._est._stat.min_nsamples()
        return vals

    def _values(self, partition_ratios: Array) -> Array:
        return self._eval_constraint(partition_ratios[:, 0])[None, :]

    def _jacobian(self, partition_ratios: Array) -> Array:
        return self._bkd.jacobian(
            self._eval_constraint, partition_ratios[:, 0]
        )

    def _constraint_dot_product(
        self, weights: Array, partition_ratios: Array
    ) -> Array:
        if weights.ndim != 1:
            raise ValueError("weights.ndim != 1")
        return weights @ self._eval_constraint(partition_ratios)

    def _weighted_hessian(
        self, partition_ratios: Array, weights: Array
    ) -> Array:
        hess = self._bkd.hessian(
            partial(self._constraint_dot_product, weights[:, 0]),
            partition_ratios[:, 0],
        )
        return hess

    def nqoi(self):
        return self._est._npartitions


class ACVEstimator(CVEstimator):
    def __init__(
        self,
        stat: MultiOutputStatistic,
        costs: Union[List, Array],
        recursion_index: Array = None,
        opt_criteria: callable = None,
        tree_depth: int = None,
        allow_failures: bool = False,
    ):
        """
        Constructor.

        Parameters
        ----------
        stat : :class:`~pyapprox.multifidelity.multioutput_monte_carlo.MultiOutputStatistic`
            Object defining what statistic will be calculated

        costs : Array (nmodels)
            The relative costs of evaluating each model

        cov : Array (nmodels*nqoi, nmodels)
            The covariance C between each of the models. The highest fidelity
            model is the first model, i.e. covariance between its QoI
            is cov[:nqoi, :nqoi]

        recursion_index : Array (nmodels-1)
            The recusion index that specifies which ACV estimator is used

        opt_criteria : callable
            Function of the the covariance between the high-fidelity
            QoI estimators with signature

            ``opt_criteria(variance) -> float

            where variance is Array with size that depends on
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
        self._optimizer = None

    def _get_discrepancy_covariances(self, npartition_samples: Array) -> Array:
        return self._stat._get_acv_discrepancy_covariances(
            self._get_allocation_matrix(), npartition_samples
        )

    def _get_partition_indices(self, npartition_samples: Array) -> Array:
        """
        Get the indices, into the flattened array of all samples/values,
        of each indpendent sample partition
        """
        ntotal_independent_samples = npartition_samples.sum()
        total_indices = self._bkd.arange(ntotal_independent_samples, dtype=int)
        # round the cumsum to make sure values like 3.9999999999999999
        # do not get rounded down to 3
        indices = self._bkd.split(
            total_indices,
            self._bkd.array(
                self._bkd.round(self._bkd.cumsum(npartition_samples[:-1])),
                dtype=int,
            ),
        )
        return [self._bkd.asarray(idx, dtype=int) for idx in indices]

    def _get_partition_indices_per_acv_subset(
        self, bootstrap: bool = False
    ) -> Array:
        r"""
        Get the indices, into the flattened array of all samples/values
        for each model, of each acv subset
        :math:`\mathcal{Z}_\alpha,\mathcal{Z}_\alpha^*`
        """
        partition_indices = self._get_partition_indices(
            self._rounded_npartition_samples
        )
        if bootstrap:
            npartitions = len(self._rounded_npartition_samples)
            random_partition_indices = [None for jj in range(npartitions)]
            random_partition_indices[0] = self._bkd.array(
                np.random.choice(
                    np.arange(partition_indices[0].shape[0], dtype=int),
                    size=partition_indices[0].shape[0],
                    replace=True,
                ),
                dtype=int,
            )
            partition_indices_per_acv_subset = [
                self._bkd.array([], dtype=int),
                partition_indices[0][random_partition_indices[0]],
            ]
        else:
            partition_indices_per_acv_subset = [
                self._bkd.array([], dtype=int),
                partition_indices[0],
            ]
        for ii in range(1, self._nmodels):
            active_partitions = self._bkd.where(
                (self._allocation_mat[:, 2 * ii] == 1)
                | (self._allocation_mat[:, 2 * ii + 1] == 1)
            )[0]
            subset_indices = [None for ii in range(self._nmodels)]
            lb, ub = 0, 0
            for idx in active_partitions:
                ub += partition_indices[idx].shape[0]
                subset_indices[idx] = self._bkd.arange(lb, ub, dtype=int)
                if bootstrap:
                    if random_partition_indices[idx] is None:
                        # make sure the same random permutation for partition
                        # idx is used for all acv_subsets
                        random_partition_indices[idx] = self._bkd.array(
                            np.random.choice(
                                np.arange(ub - lb, dtype=int),
                                size=ub - lb,
                                replace=True,
                            ),
                            dtype=int,
                        )
                    subset_indices[idx] = subset_indices[idx][
                        random_partition_indices[idx]
                    ]
                lb = ub
            active_partitions_1 = self._bkd.where(
                (self._allocation_mat[:, 2 * ii] == 1)
            )[0]
            active_partitions_2 = self._bkd.where(
                (self._allocation_mat[:, 2 * ii + 1] == 1)
            )[0]
            indices_1 = self._bkd.hstack(
                [subset_indices[idx] for idx in active_partitions_1]
            )
            indices_2 = self._bkd.hstack(
                [subset_indices[idx] for idx in active_partitions_2]
            )
            partition_indices_per_acv_subset += [indices_1, indices_2]
        return partition_indices_per_acv_subset

    def _partition_ratios_to_model_ratios(
        self, partition_ratios: Array
    ) -> Array:
        """
        Convert the partition ratios defining the number of samples per
        partition relative to the number of samples in the
        highest-fidelity model partition
        to ratios defining the number of samples per mdoel
        relative to the number of highest-fidelity model samples
        """
        model_ratios = self._bkd.empty(partition_ratios.shape)
        for ii in range(1, self._nmodels):
            active_partitions = self._bkd.where(
                (self._allocation_mat[1:, 2 * ii] == 1)
                | (self._allocation_mat[1:, 2 * ii + 1] == 1)
            )[0]
            model_ratios[ii - 1] = partition_ratios[active_partitions].sum()
            if (self._allocation_mat[0, 2 * ii] == 1) or (
                self._allocation_mat[0, 2 * ii + 1] == 1
            ):
                model_ratios[ii - 1] += 1
        return model_ratios

    def _get_num_high_fidelity_samples_from_partition_ratios(
        self, target_cost: int, partition_ratios: Array
    ) -> float:
        model_ratios = self._partition_ratios_to_model_ratios(partition_ratios)
        return target_cost / (
            self._costs[0] + (model_ratios * self._costs[1:]).sum()
        )

    def _npartition_samples_from_partition_ratios(
        self, target_cost: int, partition_ratios: Array
    ) -> Array:
        nhf_samples = (
            self._get_num_high_fidelity_samples_from_partition_ratios(
                target_cost, partition_ratios
            )
        )
        npartition_samples = self._bkd.empty(partition_ratios.shape[0] + 1)
        npartition_samples[0] = nhf_samples
        npartition_samples[1:] = partition_ratios * nhf_samples
        return npartition_samples

    def _covariance_from_partition_ratios(
        self, target_cost: int, partition_ratios: Array
    ) -> Array:
        """
        Get the variance of the Monte Carlo estimator from costs and cov
        and nsamples ratios. Needed for optimization.

        Parameters
        ----------
        target_cost : float
            The total cost budget

        partition_ratios : Array (nmodels-1)
            The sample ratios r used to specify the number of samples
            in the indepedent sample partitions

        Returns
        -------
        variance : float
            The variance of the estimator
        """
        npartition_samples = self._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios
        )
        # if self._bkd.any(npartition_samples < 0):
        #     print(partition_ratios, "SSSS")
        return self._covariance_from_npartition_samples(npartition_samples)

    def _separate_values_per_model(
        self, values_per_model: List[Array], bootstrap: bool = False
    ) -> List[Array]:
        r"""
        Seperate values per model into the acv subsets associated with
        :math:`\mathcal{Z}_\alpha,\mathcal{Z}_\alpha^*`
        """
        if len(values_per_model) != self._nmodels:
            msg = "len(values_per_model) {0} != nmodels {1}".format(
                len(values_per_model), self._nmodels
            )
            raise ValueError(msg)
        for ii in range(self._nmodels):
            if (
                values_per_model[ii].shape[0]
                != self._rounded_nsamples_per_model[ii]
            ):
                msg = "{0} != {1}".format(
                    "len(values_per_model[{0}]): {1}".format(
                        ii, values_per_model[ii].shape[0]
                    ),
                    "nsamples_per_model[ii]: {0}".format(
                        self._rounded_nsamples_per_model[ii]
                    ),
                )
                raise ValueError(msg)

        acv_partition_indices = self._get_partition_indices_per_acv_subset(
            bootstrap
        )
        nacv_subsets = len(acv_partition_indices)
        # atleast_2d is needed for when acv_partition_indices[ii].shape[0] == 1
        # in this case python automatically reduces the values array from
        # shape (1, N) to (N)
        acv_values = [
            self._bkd.atleast2d(
                values_per_model[ii // 2][acv_partition_indices[ii]]
            )
            for ii in range(nacv_subsets)
        ]
        return acv_values

    def _separate_samples_per_model(
        self, samples_per_model: List[Array]
    ) -> List[Array]:
        if len(samples_per_model) != self._nmodels:
            msg = "len(samples_per_model) {0} != nmodels {1}".format(
                len(samples_per_model), self._nmodels
            )
            raise ValueError(msg)
        for ii in range(self._nmodels):
            if (
                samples_per_model[ii].shape[1]
                != self._rounded_nsamples_per_model[ii]
            ):
                msg = "{0} != {1}".format(
                    "len(samples_per_model[{0}]): {1}".format(
                        ii, samples_per_model[ii].shape[0]
                    ),
                    "nsamples_per_model[ii]: {0}".format(
                        self._rounded_nsamples_per_model[ii]
                    ),
                )
                raise ValueError(msg)

        acv_partition_indices = self._get_partition_indices_per_acv_subset()
        nacv_subsets = len(acv_partition_indices)
        acv_samples = [
            samples_per_model[ii // 2][:, acv_partition_indices[ii]]
            for ii in range(nacv_subsets)
        ]
        return acv_samples

    def generate_samples_per_model(
        self, rvs: callable, npilot_samples: int = 0
    ) -> List[Array]:
        ntotal_independent_samples = (
            self._rounded_npartition_samples.sum() - npilot_samples
        )
        independent_samples = rvs(ntotal_independent_samples)
        samples_per_model = []
        rounded_npartition_samples = self._rounded_npartition_samples.clone()
        if npilot_samples > rounded_npartition_samples[0]:
            raise ValueError(
                "npilot_samples is larger than optimized first partition size"
            )
        rounded_npartition_samples[0] -= npilot_samples
        rounded_nsamples_per_model = self._compute_nsamples_per_model(
            rounded_npartition_samples
        )
        partition_indices = self._get_partition_indices(
            rounded_npartition_samples
        )
        for ii in range(self._nmodels):
            active_partitions = self._bkd.where(
                (self._allocation_mat[:, 2 * ii] == 1)
                | (self._allocation_mat[:, 2 * ii + 1] == 1)
            )[0]
            indices = self._bkd.hstack(
                [partition_indices[idx] for idx in active_partitions]
            )
            if indices.shape[0] != rounded_nsamples_per_model[ii]:
                msg = "Rounding has caused {0} != {1}".format(
                    indices.shape[0], rounded_nsamples_per_model[ii]
                )
                raise RuntimeError(msg)
            samples_per_model.append(independent_samples[:, indices])
        return samples_per_model

    def _compute_single_model_nsamples(
        self, npartition_samples: Array, model_id: int
    ) -> float:
        active_partitions = self._bkd.where(
            (self._allocation_mat[:, 2 * model_id] == 1)
            | (self._allocation_mat[:, 2 * model_id + 1] == 1)
        )[0]
        return npartition_samples[active_partitions].sum()

    def _compute_single_model_nsamples_from_partition_ratios(
        self, partition_ratios: Array, target_cost: float, model_id: int
    ) -> float:
        npartition_samples = self._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios
        )
        return self._compute_single_model_nsamples(
            npartition_samples, model_id
        )

    def _compute_nsamples_per_model(self, npartition_samples: Array) -> Array:
        nsamples_per_model = self._bkd.empty(self._nmodels)
        for ii in range(self._nmodels):
            nsamples_per_model[ii] = self._compute_single_model_nsamples(
                npartition_samples, ii
            )
        return nsamples_per_model

    def _estimate(
        self,
        values_per_model: List[Array],
        weights: Array,
        bootstrap: bool = False,
    ):
        nmodels = len(values_per_model)
        acv_values = self._separate_values_per_model(
            values_per_model, bootstrap
        )
        deltas = self._bkd.hstack(
            [
                self._stat.sample_estimate(acv_values[2 * ii])
                - self._stat.sample_estimate(acv_values[2 * ii + 1])
                for ii in range(1, nmodels)
            ]
        )
        est = self._stat.sample_estimate(acv_values[1]) + weights @ deltas
        return est

    def __repr__(self):
        if self._optimized_criteria is None:
            if not hasattr(self, "_recursion_index"):
                return "{0}(stat={1})".format(
                    self.__class__.__name__, self._stat
                )
            return "{0}(stat={1}, recursion_index={2})".format(
                self.__class__.__name__, self._stat, self._recursion_index
            )
        rep = "{0}(stat={1}, recursion_index={2}, criteria={3:.3g}".format(
            self.__class__.__name__,
            self._stat,
            self._recursion_index,
            self._optimized_criteria,
        )
        rep += " target_cost={0:.5g}, ratios={1}, nsamples={2})".format(
            self._rounded_target_cost,
            self._rounded_partition_ratios,
            self._rounded_nsamples_per_model,
        )
        return rep

    @abstractmethod
    def _create_allocation_matrix(self, recursion_index: Array) -> Array:
        r"""
        Return the allocation matrix corresponding to
        self._rounded_nsamples_per_model set by _set_optimized_params

        Returns
        -------
        mat : Array (nmodels, 2*nmodels)
            For columns :math:`2j, j=0,\ldots,M-1` the ith row contains a
            flag specifiying if :math:`z_i^\star\subseteq z_j^\star`
            For columns :math:`2j+1, j=0,\ldots,M-1` the ith row contains a
            flag specifiying if :math:`z_i\subseteq z_j`
        """
        raise NotImplementedError

    def _get_allocation_matrix(self) -> Array:
        """return allocation matrix as torch tensor"""
        return self._bkd.asarray(self._allocation_mat)

    def _set_recursion_index(self, index: Array):
        """Set the recursion index of the parameterically defined ACV
        Estimator.

        This function intializes the allocation matrix.

        Parameters
        ----------
        index : Array (nmodels-1)
            The recusion index
        """
        if index is None:
            index = self._bkd.zeros(self._nmodels - 1, dtype=int)
        else:
            index = self._bkd.asarray(index, dtype=int)
        if self._nmodels is None:
            raise RuntimeError("must call stat.set_pilot_quantities()")
        if index.shape[0] != self._nmodels - 1:
            msg = "index {0} is the wrong shape. Should be {1}".format(
                index, self._nmodels - 1
            )
            raise ValueError(msg)
        self._create_allocation_matrix(index)
        self._recursion_index = index

    def combine_acv_samples(self, acv_samples: List[Array]) -> List[Array]:
        return _combine_acv_samples(
            self._allocation_mat,
            self._rounded_npartition_samples,
            acv_samples,
            self._bkd,
        )

    def combine_acv_values(self, acv_values: List[Array]) -> List[Array]:
        return _combine_acv_values(
            self._allocation_mat,
            self._rounded_npartition_samples,
            acv_values,
            self._bkd,
        )

    def plot_allocation(
        self, ax, show_npartition_samples: bool = False, **kwargs
    ):
        if show_npartition_samples:
            if self._rounded_npartition_samples is None:
                msg = "set_optimized_params must be called"
                raise ValueError(msg)
            return _plot_allocation_matrix(
                self._allocation_mat,
                self._rounded_npartition_samples,
                ax,
                **kwargs,
            )

        return _plot_allocation_matrix(
            self._allocation_mat, None, ax, **kwargs
        )

    def plot_recursion_dag(self, ax):
        return _plot_model_recursion(
            self._bkd.to_numpy(self._recursion_index), ax
        )

    # def _objective(self, target_cost, x, return_grad=True):
    #     partition_ratios = self._bkd.asarray(x)
    #     if return_grad:
    #         partition_ratios.requires_grad = True
    #     covariance = self._covariance_from_partition_ratios(
    #         target_cost, partition_ratios
    #     )
    #     val = self._optimization_criteria(covariance) * self._objective_scaling
    #     if not return_grad:
    #         return val.item()
    #     val.backward()
    #     grad = partition_ratios.grad.detach().numpy().copy()
    #     partition_ratios.grad.zero_()
    #     return val.item(), grad

    def get_npartition_bounds(self) -> Array:
        nunknowns = self._npartitions - 1
        lower_bound = 1e-3  # this can impact the ability to find a solution
        bounds = self._bkd.stack(
            (
                self._bkd.full((nunknowns,), lower_bound),
                self._bkd.full((nunknowns,), np.inf),
            ),
            axis=1,
        )
        return bounds

    def get_default_optimizer(self) -> Optimizer:
        # nunknowns = self._npartitions - 1
        # local_optimizer = ScipyConstrainedOptimizer(opts={"gtol": 1e-9})
        # init_gen = RandomUniformOptimzerIterateGenerator(
        #     nunknowns, backend=self._bkd
        # )
        # init_gen.set_bounds(self.get_npartition_bounds())
        # optimizer = ConstrainedMultiStartOptimizer(local_optimizer)
        # optimizer.set_initial_iterate_generator(init_gen)
        global_optimizer = ScipyConstrainedNelderMeadOptimizer(
            opts={"maxiter": 500}
        )
        local_optimizer = ScipyConstrainedOptimizer()
        optimizer = ChainedOptimizer(global_optimizer, local_optimizer)
        optimizer.set_verbosity(0)
        return optimizer

    def set_optimizer(self, optimizer: Optimizer):
        if not isinstance(optimizer, Optimizer):
            raise ValueError("Optimizer must be an instance of Optimizer")
        self._optimizer = optimizer

    def _allocate_samples_minimize(
        self, target_cost: float
    ) -> OptimizationResult:
        if target_cost < self._bkd.sum(self._costs):
            msg = "Target cost does not allow at least one sample from "
            msg += "each model"
            raise ValueError(msg)

        # TODO Pass in optimizer once tests pass
        if self._optimizer is None:
            self.set_optimizer(self.get_default_optimizer())
        scaling = self._optimizer._opts.get("scaling", 1)
        objective = ACVLogDeterminantObjective(scaling=scaling)
        objective.set_target_cost(target_cost)
        objective.set_estimator(self)
        self._optimizer.set_objective_function(objective)
        constraints = [ACVPartitionConstraint(self, target_cost)]
        self._optimizer.set_constraints(constraints)
        self._optimizer.set_bounds(self.get_npartition_bounds())
        init_iterate = self._bkd.full((self._nmodels - 1, 1), 1.0)
        result = self._optimizer.minimize(init_iterate)
        return result

    @abstractmethod
    def _get_specific_constraints(self, target_cost: float):
        raise NotImplementedError()

    # def _constraint_jacobian(
    #     self, constraint_fun, partition_ratios_np: Array, *args
    # ):
    #     partition_ratios = self._bkd.asarray(partition_ratios_np)
    #     partition_ratios.requires_grad = True
    #     val = constraint_fun(partition_ratios, *args, return_numpy=False)
    #     val.backward()
    #     jac = partition_ratios.grad.detach().numpy().copy()
    #     partition_ratios.grad.zero_()
    #     return jac

    # def _acv_npartition_samples_constraint(
    #     self,
    #     partition_ratios_np,
    #     target_cost,
    #     min_nsamples,
    #     partition_id,
    #     return_numpy=True,
    # ):
    #     partition_ratios = self._bkd.asarray(partition_ratios_np)
    #     nsamples = self._npartition_samples_from_partition_ratios(
    #         target_cost, partition_ratios
    #     )[partition_id]
    #     val = nsamples - min_nsamples
    #     if return_numpy:
    #         return val.item()
    #     return val

    # def _acv_npartition_samples_constraint_jac(
    #     self, partition_ratios_np, target_cost, min_nsamples, partition_id
    # ):
    #     return self._constraint_jacobian(
    #         self._acv_npartition_samples_constraint,
    #         partition_ratios_np,
    #         target_cost,
    #         min_nsamples,
    #         partition_id,
    #     )

    # def _npartition_ratios_constaint(self, partition_ratios_np, ratio_id):
    #     # needs to be positive
    #     return partition_ratios_np[ratio_id] - 0

    # def _npartition_ratios_constaint_jac(self, partition_ratios_np, ratio_id):
    #     jac = self._bkd.zeros(partition_ratios_np.shape[0], dtype=float)
    #     jac[ratio_id] = 1.0
    #     return jac

    # def _get_constraints(self, target_cost):
    #     # Ensure the each partition has enough samples to compute
    #     # the desired statistic. Techinically we only need the number
    #     # of samples in each acv subset have enough. But this constraint
    #     # is easy to implement and not really restrictive practically
    #     if isinstance(
    #         self._stat, (MultiOutputVariance, MultiOutputMeanAndVariance)
    #     ):
    #         partition_min_nsamples = 2.0
    #     else:
    #         partition_min_nsamples = 1.0
    #     cons = [
    #         {
    #             "type": "ineq",
    #             "fun": self._acv_npartition_samples_constraint,
    #             "jac": self._acv_npartition_samples_constraint_jac,
    #             "args": (target_cost, partition_min_nsamples, ii),
    #         }
    #         for ii in range(self._nmodels)
    #     ]

    #     # Better to enforce this with bounds
    #     # Ensure ratios are positive
    #     # cons += [
    #     #     {'type': 'ineq',
    #     #      'fun': self._npartition_ratios_constaint,
    #     #      'jac': self._npartition_ratios_constaint_jac,
    #     #      'args': (ii,)}
    #     #     for ii in range(self._nmodels-1)]

    #     # Note target cost is satisfied by construction using the above
    #     # constraints because nsamples is determined based on target cost
    #     cons += self._get_specific_constraints(target_cost)
    #     return cons

    def _allocate_samples(self, target_cost: float):
        opt_result = self._allocate_samples_minimize(target_cost)
        partition_ratios = opt_result.x[:, 0]
        if not opt_result.success:
            raise RuntimeError(
                "{0} optimizer failed {1} with message {2}".format(
                    self, opt_result, opt_result.message
                )
            )
        else:
            val = opt_result.fun
        return partition_ratios, val

    def _round_partition_ratios(
        self, target_cost: float, partition_ratios: Array
    ):
        npartition_samples = self._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios
        )
        if npartition_samples[0] < 1 - 1e-8:
            msg = "Rounding will cause nhf samples to be zero {0}".format(
                npartition_samples
            )
            raise RuntimeError(msg)
        rounded_npartition_samples = self._bkd.asarray(
            self._bkd.floor(npartition_samples + 1e-8), dtype=int
        )
        assert rounded_npartition_samples[0] >= 1
        rounded_target_cost = (
            self._compute_nsamples_per_model(rounded_npartition_samples)
            * self._costs
        ).sum()
        rounded_partition_ratios = (
            rounded_npartition_samples[1:] / rounded_npartition_samples[0]
        )
        return rounded_partition_ratios, rounded_target_cost

    def _estimator_cost(self, npartition_samples: Array) -> float:
        nsamples_per_model = self._compute_nsamples_per_model(
            npartition_samples
        )
        return (nsamples_per_model * self._costs).sum()

    def _set_optimized_params(
        self, partition_ratios: Array, target_cost: float
    ):
        """
        Set the parameters needed to generate samples for evaluating the
        estimator

        Parameters
        ----------
        rounded_nsample_ratios : Array (nmodels-1, dtype=int)
            The sample ratios r used to specify the number of samples in
            the independent sample partitions.

        rounded_target_cost : float
            The cost of the new sample allocation

        Sets attrributes
        ----------------
        self._rounded_partition_ratios : Array (nmodels-1)
            The optimal partition ratios rounded so that each partition
            contains an integer number of samples

        And all attributes set by super()._set_optimized_params. See
        the docstring of that function for further details
        """
        self._rounded_partition_ratios, rounded_target_cost = (
            self._round_partition_ratios(
                target_cost, self._bkd.asarray(partition_ratios)
            )
        )
        rounded_npartition_samples = (
            self._npartition_samples_from_partition_ratios(
                rounded_target_cost,
                self._bkd.asarray(self._rounded_partition_ratios),
            )
        )
        # round because sometimes round_partition_ratios
        # will produce floats slightly smaller
        # than an integer so when converted to an integer will produce
        # values 1 smaller than the correct value
        rounded_npartition_samples = self._bkd.round(
            rounded_npartition_samples
        )
        rounded_nsamples_per_model = self._bkd.asarray(
            self._compute_nsamples_per_model(rounded_npartition_samples),
            dtype=int,
        )
        super()._set_optimized_params_base(
            rounded_npartition_samples,
            rounded_nsamples_per_model,
            rounded_target_cost,
        )

    def _allocate_samples_for_single_recursion(self, target_cost: float):
        partition_ratios, obj_val = self._allocate_samples(target_cost)
        self._set_optimized_params(partition_ratios, target_cost)

    def get_all_recursion_indices(self) -> List[Array]:
        return _get_acv_recursion_indices(self._nmodels, self._tree_depth)

    def _allocate_samples_for_all_recursion_indices(self, target_cost: float):
        best_criteria = self._bkd.asarray(np.inf)
        best_result = None
        for index in self.get_all_recursion_indices():
            self._set_recursion_index(index)
            try:
                self._allocate_samples_for_single_recursion(target_cost)
            except RuntimeError as e:
                # typically solver fails because trying to use
                # uniformative model as a recursive control variate
                if not self._allow_failures:
                    raise e
                self._optimized_criteria = self._bkd.asarray(np.inf)
                if self._optimizer._verbosity > 0:
                    print("Optimizer failed")
            if self._optimizer._verbosity > 2:
                msg = "\t\t Recursion: {0} Objective: best {1}, current {2}".format(
                    index,
                    best_criteria.item(),
                    self._optimized_criteria.item(),
                )
                print(msg)
            if self._optimized_criteria < best_criteria:
                best_result = [
                    self._rounded_partition_ratios,
                    self._rounded_target_cost,
                    self._optimized_criteria,
                    index,
                ]
                best_criteria = self._optimized_criteria
        if best_result is None:
            raise RuntimeError("No solutions were found")
        self._set_recursion_index(best_result[3])
        self._set_optimized_params(
            self._bkd.asarray(best_result[0]), target_cost
        )

    def allocate_samples(self, target_cost: float):
        if self._tree_depth is not None:
            return self._allocate_samples_for_all_recursion_indices(
                target_cost
            )
        return self._allocate_samples_for_single_recursion(target_cost)


class GMFEstimator(ACVEstimator):
    def _create_allocation_matrix(self, recursion_index: Array) -> Array:
        self._allocation_mat = _get_allocation_matrix_gmf(
            recursion_index, self._bkd
        )

    def _get_specific_constraints(self, target_cost: float):
        return []


class GISEstimator(ACVEstimator):
    """
    The GIS estimator from Gorodetsky et al. and Bomorito et al
    """

    def _create_allocation_matrix(self, recursion_index: Array) -> Array:
        self._allocation_mat = _get_allocation_matrix_acvis(
            recursion_index, self._bkd
        )

    def _get_specific_constraints(self, target_cost: float):
        return []


class GRDEstimator(ACVEstimator):
    """
    The GRD estimator.
    """

    def _create_allocation_matrix(self, recursion_index: Array) -> Array:
        self._allocation_mat = _get_allocation_matrix_acvrd(
            recursion_index, self._bkd
        )

    def _get_specific_constraints(self, target_cost: float):
        return []


class MFMCEstimator(GMFEstimator):
    def __init__(
        self,
        stat: MultiOutputStatistic,
        costs: Union[List, Array],
        opt_criteria=None,
        opt_qoi: int = 0,
    ):
        # Use the sample analytical sample allocation for estimating a scalar
        # mean when estimating any statistic
        nmodels = len(costs)
        super().__init__(
            stat,
            costs,
            recursion_index=stat._bkd.arange(nmodels - 1, dtype=int),
            opt_criteria=None,
        )
        # The qoi index used to generate the sample allocation
        self._opt_qoi = opt_qoi

    def _allocate_samples(self, target_cost: float):
        # nsample_ratios returned will be listed in according to
        # self.model_order which is what self.get_rsquared requires
        if not _check_mfmc_model_costs_and_correlations(
            self._costs,
            get_correlation_from_covariance(self._stat._cov, self._bkd),
        ):
            raise ValueError("models do not admit a hierarchy")
        nsample_ratios, val = _allocate_samples_mfmc(
            self._stat._cov[
                self._opt_qoi :: self._stat._nqoi,
                self._opt_qoi :: self._stat._nqoi,
            ],
            self._costs,
            target_cost,
            self._bkd,
        )
        nsample_ratios = self._native_ratios_to_npartition_ratios(
            nsample_ratios
        )
        return nsample_ratios, val

    def _native_ratios_to_npartition_ratios(self, ratios: Array):
        partition_ratios = self._bkd.hstack(
            (ratios[0] - 1, self._bkd.diff(ratios))
        )
        return partition_ratios

    def _get_allocation_matrix(self):
        return _get_sample_allocation_matrix_mfmc(self._nmodels, self._bkd)


class MLMCEstimator(GRDEstimator):
    def __init__(
        self,
        stat: MultiOutputStatistic,
        costs: Union[List, Array],
        opt_criteria=None,
        opt_qoi: int = 0,
    ):
        """
        Use the sample analytical sample allocation for estimating a scalar
        mean when estimating any statistic

        Use optimal ACV weights instead of all weights=-1 used by
        classical MLMC.
        """
        nmodels = len(costs)
        super().__init__(
            stat,
            costs,
            recursion_index=stat._bkd.arange(nmodels - 1),
            opt_criteria=None,
        )
        # The qoi index used to generate the sample allocation
        self._opt_qoi = opt_qoi

    def _weights(self, CF: Array, cf: Array) -> Array:
        # raise NotImplementedError("check weights size is correct")
        return -self._bkd.ones(cf.shape)

    def _covariance_from_npartition_samples(
        self, npartition_samples: Array
    ) -> Array:
        CF, cf = self._get_discrepancy_covariances(npartition_samples)
        weights = self._weights(CF, cf)
        # cannot use formulation of variance that uses optimal weights
        # must use the more general expression below, e.g. Equation 8
        # from Dixon 2024.
        return self._covariance_non_optimal_weights(
            self._stat.high_fidelity_estimator_covariance(
                npartition_samples[0]
            ),
            weights,
            CF,
            cf,
        )

    def _allocate_samples(self, target_cost: float):
        nsample_ratios, val = _allocate_samples_mlmc(
            self._stat._cov[
                self._opt_qoi :: self._stat._nqoi,
                self._opt_qoi :: self._stat._nqoi,
            ],
            self._costs,
            target_cost,
            self._bkd,
        )
        return self._bkd.asarray(nsample_ratios), val

    def _create_allocation_matrix(self, dummy: Array) -> Array:
        self._allocation_mat = _get_sample_allocation_matrix_mlmc(
            self._nmodels, self._bkd
        )

    def _native_ratios_to_npartition_ratios(self, ratios: Array) -> Array:
        partition_ratios = [ratios[0] - 1]
        for ii in range(1, len(ratios)):
            partition_ratios.append(ratios[ii] - partition_ratios[ii - 1])
        return self._bkd.hstack(partition_ratios)


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
