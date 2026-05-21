"""ACVEstimator template and FittedACVEstimator.

ACVEstimator is the immutable template: it maps (target_cost, partition_ratios)
to estimator covariance but stores no allocation state.

FittedACVEstimator composes (template, ACVAllocationResult) and eagerly
computes weights/covariance from the discrete (post-rounding) counts.
"""

from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from pyapprox.statest.acv.result import ACVAllocationResult
from pyapprox.statest.acv.utils import (
    _combine_acv_samples,
    _combine_acv_values,
)
from pyapprox.statest.cv_estimator import CVEstimator
from pyapprox.statest.statistics import (
    MultiOutputStatistic,
)
from pyapprox.util.backends.protocols import Array, Backend


class ACVEstimator(CVEstimator[Array], Generic[Array]):
    """Approximate Control Variate estimator template.

    Immutable after construction. Maps (target_cost, partition_ratios) to
    estimator covariance. Stores no allocation state.
    """

    def __init__(
        self,
        stat: MultiOutputStatistic[Array],
        costs: Union[List[float], Array],
        recursion_index: Optional[Array] = None,
        npartitions_lower_bound: float = 1e-2,
    ):
        """
        Constructor.

        Parameters
        ----------
        stat : MultiOutputStatistic
            Object defining what statistic will be calculated

        costs : Array (nmodels)
            The relative costs of evaluating each model

        recursion_index : Array (nmodels-1)
            The recusion index that specifies which ACV estimator is used

        npartitions_lower_bound : float
            Lower bound for partition ratios during optimization.
        """
        super().__init__(stat, costs, None)
        self._recursion_index: Optional[Array] = None
        self._set_recursion_index(recursion_index)
        self._npartitions = self._nmodels
        self._npartitions_lower_bound = npartitions_lower_bound

    # === Template API (pure, allocation-parameterized) ===

    def covariance_at(
        self, target_cost: float, partition_ratios: Array
    ) -> Array:
        """Compute estimator covariance from continuous partition ratios.

        For use during optimization. Supports autodiff gradients.

        Parameters
        ----------
        target_cost : float
            The target computational budget.
        partition_ratios : Array
            Continuous partition ratios from optimization.

        Returns
        -------
        Array
            The estimator covariance matrix.
        """
        return self._covariance_from_partition_ratios(target_cost, partition_ratios)

    def npartition_samples_at(
        self, target_cost: float, partition_ratios: Array
    ) -> Array:
        """Compute continuous npartition_samples from partition ratios.

        For use during optimization. Supports autodiff gradients.

        Parameters
        ----------
        target_cost : float
            The target computational budget.
        partition_ratios : Array
            Continuous partition ratios from optimization.

        Returns
        -------
        Array
            Continuous sample counts per partition.
        """
        return self._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios
        )

    def allocation_matrix(self) -> Array:
        """Return the allocation matrix.

        Returns
        -------
        Array
            Allocation matrix. Shape: (npartitions, 2*nmodels).
        """
        return self._get_allocation_matrix()

    # === Internal covariance computation ===

    def _get_discrepancy_covariances(
        self, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        return self._stat._get_acv_discrepancy_covariances(
            self._get_allocation_matrix(), npartition_samples
        )

    def _get_partition_indices(self, npartition_samples: Array) -> Array:
        """
        Get the indices, into the flattened array of all samples/values,
        of each indpendent sample partition
        """
        ntotal_independent_samples = self._bkd.sum(npartition_samples)
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

    def _partition_ratios_to_model_ratios(self, partition_ratios: Array) -> Array:
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
            model_ratios[ii - 1] = self._bkd.sum(partition_ratios[active_partitions])
            if (self._allocation_mat[0, 2 * ii] == 1) or (
                self._allocation_mat[0, 2 * ii + 1] == 1
            ):
                model_ratios[ii - 1] += 1
        return model_ratios

    def _get_num_high_fidelity_samples_from_partition_ratios(
        self, target_cost: float, partition_ratios: Array
    ) -> Array:
        model_ratios = self._partition_ratios_to_model_ratios(partition_ratios)
        return target_cost / (
            self._costs[0]
            + self._bkd.sum(model_ratios * self._costs[1:])
        )

    def _npartition_samples_from_partition_ratios(
        self, target_cost: float, partition_ratios: Array
    ) -> Array:
        nhf_samples = self._get_num_high_fidelity_samples_from_partition_ratios(
            target_cost, partition_ratios
        )
        npartition_samples = self._bkd.empty((partition_ratios.shape[0] + 1,))
        npartition_samples[0] = nhf_samples
        npartition_samples[1:] = partition_ratios * nhf_samples
        return npartition_samples

    def _covariance_from_partition_ratios(
        self, target_cost: float, partition_ratios: Array
    ) -> Array:
        """
        Get the variance of the Monte Carlo estimator from costs and cov
        and nsamples ratios. Needed for optimization.
        """
        npartition_samples = self._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios
        )
        return self._covariance_from_npartition_samples(npartition_samples)

    def _compute_single_model_nsamples(
        self, npartition_samples: Array, model_id: int
    ) -> Array:
        active_partitions = self._bkd.where(
            (self._allocation_mat[:, 2 * model_id] == 1)
            | (self._allocation_mat[:, 2 * model_id + 1] == 1)
        )[0]
        return self._bkd.sum(npartition_samples[active_partitions])

    def _compute_single_model_nsamples_from_partition_ratios(
        self, partition_ratios: Array, target_cost: float, model_id: int
    ) -> Array:
        npartition_samples = self._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios
        )
        return self._compute_single_model_nsamples(npartition_samples, model_id)

    def _compute_nsamples_per_model(self, npartition_samples: Array) -> Array:
        nsamples_per_model = self._bkd.empty((self._nmodels,))
        for ii in range(self._nmodels):
            nsamples_per_model[ii] = self._compute_single_model_nsamples(
                npartition_samples, ii
            )
        return nsamples_per_model

    def _estimate_from_acv_values(
        self, acv_values: List[Array], weights: Array, nmodels: int
    ) -> Array:
        deltas = self._bkd.hstack(
            [
                self._stat.sample_estimate(acv_values[2 * ii])
                - self._stat.sample_estimate(acv_values[2 * ii + 1])
                for ii in range(1, nmodels)
            ]
        )
        est = self._stat.sample_estimate(acv_values[1]) + weights @ deltas
        return est

    def _estimator_cost(self, npartition_samples: Array) -> Array:
        nsamples_per_model = self._compute_nsamples_per_model(npartition_samples)
        return self._bkd.sum(nsamples_per_model * self._costs)

    def get_npartition_bounds(self, total_cost: float) -> Array:
        nunknowns = self._npartitions - 1
        bounds = self._bkd.stack(
            (
                self._bkd.full((nunknowns,), self._npartitions_lower_bound),
                total_cost / self._costs[1:],
            ),
            axis=1,
        )
        return bounds

    def get_all_recursion_indices(self) -> List[Array]:
        from pyapprox.statest.acv._recursion_indices import (
            _get_acv_recursion_indices,
        )

        return _get_acv_recursion_indices(self._nmodels, None)

    @abstractmethod
    def _create_allocation_matrix(self, recursion_index: Array) -> None:
        r"""
        Set the allocation matrix corresponding to
        the recursion index.
        """
        raise NotImplementedError

    def _get_allocation_matrix(self) -> Array:
        """return allocation matrix as backend array"""
        return self._bkd.asarray(self._allocation_mat)

    def _set_recursion_index(self, index: Array) -> None:
        """Set the recursion index of the parameterically defined ACV
        Estimator.
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

    def __repr__(self) -> str:
        return "{0}(stat={1}, recursion_index={2})".format(
            self.__class__.__name__, self._stat, self._recursion_index
        )


class FittedACVEstimator(Generic[Array]):
    """Frozen ACV estimator with a fixed allocation.

    Composes (template: ACVEstimator, allocation: ACVAllocationResult).
    Eagerly computes weights and covariance from discrete (post-rounding) counts.
    """

    def __init__(
        self,
        template: ACVEstimator[Array],
        allocation: ACVAllocationResult[Array],
    ) -> None:
        if not allocation.success:
            raise ValueError(f"Cannot create fitted estimator from failed allocation: "
                             f"{allocation.message}")
        bkd = template._bkd
        if not bkd.is_integer_dtype(allocation.npartition_samples):
            raise TypeError(
                f"allocation.npartition_samples must be integer-typed, "
                f"got dtype={allocation.npartition_samples.dtype}"
            )
        if not bkd.is_integer_dtype(allocation.nsamples_per_model):
            raise TypeError(
                f"allocation.nsamples_per_model must be integer-typed, "
                f"got dtype={allocation.nsamples_per_model.dtype}"
            )
        self._template = template
        self._allocation = allocation
        self._bkd = bkd
        self._stat = template._stat

        CF, cf = template._get_discrepancy_covariances(allocation.npartition_samples)
        self._weights_val = template._optimal_weights(CF, cf)
        self._covariance_val = template._covariance_from_npartition_samples(
            allocation.npartition_samples
        )
        self._criteria_val = template._optimization_criteria(self._covariance_val)

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def covariance(self) -> Array:
        """Return the estimator covariance at the allocated sample count."""
        return self._covariance_val

    def weights(self) -> Array:
        """Return the optimal control variate weights."""
        return self._weights_val

    def nsamples_per_model(self) -> Array:
        """Return the number of samples allocated to each model."""
        return self._allocation.nsamples_per_model

    def npartition_samples(self) -> Array:
        """Return the number of samples per partition."""
        return self._allocation.npartition_samples

    def partition_ratios(self) -> Array:
        """Return the partition ratios from the allocation."""
        return self._allocation.partition_ratios

    def objective_value(self) -> Array:
        """Return the objective value from the allocation."""
        return self._allocation.objective_value

    def actual_cost(self) -> float:
        """Return the actual cost of the allocation."""
        return self._allocation.actual_cost

    def allocation_matrix(self) -> Array:
        """Return the allocation matrix from the template."""
        return self._template.allocation_matrix()

    def generate_samples_per_model(
        self, rvs: Callable[..., Any], npilot_samples: int = 0
    ) -> List[Array]:
        npartition_samples = self._allocation.npartition_samples
        ntotal_independent_samples = self._bkd.to_int(
            self._bkd.sum(npartition_samples) - npilot_samples
        )
        independent_samples = rvs(ntotal_independent_samples)
        samples_per_model = []
        adjusted_npartition_samples = self._bkd.copy(npartition_samples)
        if npilot_samples > adjusted_npartition_samples[0]:
            raise ValueError(
                "npilot_samples is larger than optimized first partition size"
            )
        adjusted_npartition_samples[0] -= npilot_samples
        adjusted_nsamples_per_model = self._template._compute_nsamples_per_model(
            adjusted_npartition_samples
        )
        partition_indices = self._template._get_partition_indices(
            adjusted_npartition_samples
        )
        alloc_mat = self._template._allocation_mat
        for ii in range(self._template._nmodels):
            active_partitions = self._bkd.where(
                (alloc_mat[:, 2 * ii] == 1)
                | (alloc_mat[:, 2 * ii + 1] == 1)
            )[0]
            indices = self._bkd.hstack(
                [partition_indices[idx] for idx in active_partitions]
            )
            if indices.shape[0] != self._bkd.to_int(adjusted_nsamples_per_model[ii]):
                msg = "Rounding has caused {0} != {1}".format(
                    indices.shape[0], adjusted_nsamples_per_model[ii]
                )
                raise RuntimeError(msg)
            samples_per_model.append(independent_samples[:, indices])
        return samples_per_model

    def _get_partition_indices_per_acv_subset(
        self, bootstrap: bool = False
    ) -> List[Array]:
        r"""
        Get the indices, into the flattened array of all samples/values
        for each model, of each acv subset
        :math:`\mathcal{Z}_\alpha,\mathcal{Z}_\alpha^*`
        """
        npartition_samples = self._allocation.npartition_samples
        alloc_mat = self._template._allocation_mat
        nmodels = self._template._nmodels

        partition_indices = self._template._get_partition_indices(npartition_samples)
        if bootstrap:
            npartitions = len(npartition_samples)
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
        for ii in range(1, nmodels):
            active_partitions = self._bkd.where(
                (alloc_mat[:, 2 * ii] == 1)
                | (alloc_mat[:, 2 * ii + 1] == 1)
            )[0]
            subset_indices = [None for _ in range(nmodels)]
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
                (alloc_mat[:, 2 * ii] == 1)
            )[0]
            active_partitions_2 = self._bkd.where(
                (alloc_mat[:, 2 * ii + 1] == 1)
            )[0]
            indices_1 = self._bkd.hstack(
                [subset_indices[idx] for idx in active_partitions_1]
            )
            indices_2 = self._bkd.hstack(
                [subset_indices[idx] for idx in active_partitions_2]
            )
            partition_indices_per_acv_subset += [indices_1, indices_2]
        return partition_indices_per_acv_subset

    def _separate_values_per_model(
        self, values_per_model: List[Array], bootstrap: bool = False
    ) -> List[Array]:
        r"""
        Separate values per model into the acv subsets associated with
        :math:`\mathcal{Z}_\alpha,\mathcal{Z}_\alpha^*`

        Parameters
        ----------
        values_per_model : List[Array]
            Model values. Each array has shape (nqoi, nsamples).
        bootstrap : bool
            Whether to use bootstrap resampling.

        Returns
        -------
        List[Array]
            ACV subset values. Each array has shape (nqoi, nsamples_subset).
        """
        nmodels = self._template._nmodels
        nsamples_per_model = self._allocation.nsamples_per_model
        if len(values_per_model) != nmodels:
            msg = "len(values_per_model) {0} != nmodels {1}".format(
                len(values_per_model), nmodels
            )
            raise ValueError(msg)
        for ii in range(nmodels):
            expected = self._bkd.to_int(nsamples_per_model[ii])
            if values_per_model[ii].shape[1] != expected:
                msg = "{0} != {1}".format(
                    "values_per_model[{0}].shape[1]: {1}".format(
                        ii, values_per_model[ii].shape[1]
                    ),
                    "nsamples_per_model[ii]: {0}".format(
                        nsamples_per_model[ii]
                    ),
                )
                raise ValueError(msg)

        acv_partition_indices = self._get_partition_indices_per_acv_subset(bootstrap)
        nacv_subsets = len(acv_partition_indices)
        # Index along axis=1 (samples axis) for (nqoi, nsamples)
        acv_values = [
            values_per_model[ii // 2][:, acv_partition_indices[ii]]
            for ii in range(nacv_subsets)
        ]
        return acv_values

    def _separate_samples_per_model(
        self, samples_per_model: List[Array]
    ) -> List[Array]:
        nmodels = self._template._nmodels
        nsamples_per_model = self._allocation.nsamples_per_model
        if len(samples_per_model) != nmodels:
            msg = "len(samples_per_model) {0} != nmodels {1}".format(
                len(samples_per_model), nmodels
            )
            raise ValueError(msg)
        for ii in range(nmodels):
            expected = self._bkd.to_int(nsamples_per_model[ii])
            if samples_per_model[ii].shape[1] != expected:
                msg = "{0} != {1}".format(
                    "len(samples_per_model[{0}]): {1}".format(
                        ii, samples_per_model[ii].shape[0]
                    ),
                    "nsamples_per_model[ii]: {0}".format(
                        nsamples_per_model[ii]
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

    def combine_acv_samples(self, acv_samples: List[Array]) -> List[Array]:
        return _combine_acv_samples(
            self._template._allocation_mat,
            self._allocation.npartition_samples,
            acv_samples,
            self._bkd,
        )

    def combine_acv_values(self, acv_values: List[Array]) -> List[Array]:
        return _combine_acv_values(
            self._template._allocation_mat,
            self._allocation.npartition_samples,
            acv_values,
            self._bkd,
        )

    def _estimate(
        self,
        values_per_model: List[Array],
        weights: Array,
        bootstrap: bool = False,
    ) -> Array:
        acv_values = self._separate_values_per_model(values_per_model, bootstrap)
        return self._template._estimate_from_acv_values(
            acv_values, weights, len(values_per_model)
        )

    def __call__(self, values_per_model: List[Array]) -> Array:
        r"""
        Return the value of the ACV estimator.

        Parameters
        ----------
        values_per_model : list (nmodels)
            The values of each model. Each array has shape (nqoi, nsamples).

        Returns
        -------
        est : Array (nstats,)
            The estimator value.
        """
        for vals in values_per_model:
            if not isinstance(vals, self._bkd.array_type()):
                raise ValueError(
                    "vals must be an instance of {0}".format(self._bkd.array_type())
                )
        return self._estimate(values_per_model, self._weights_val)

    def insert_pilot_values(
        self, pilot_values: List[Array], values_per_model: List[Array]
    ) -> List[Array]:
        """Delegate to template's insert_pilot_values."""
        return self._template.insert_pilot_values(pilot_values, values_per_model)

    def bootstrap(
        self,
        values_per_model: List[Array],
        nbootstraps: int = 1000,
        mode: str = "values",
        pilot_values: Optional[List[Array]] = None,
    ) -> Union[
        Tuple[Array, Array],
        Tuple[Array, Array, Array, Array],
    ]:
        """Bootstrap variance estimation.

        Parameters
        ----------
        values_per_model : List[Array]
            Model values. Each array has shape (nqoi, nsamples).
        nbootstraps : int
            Number of bootstrap iterations.
        mode : str
            Bootstrap mode.
        pilot_values : List[Array], optional
            Pilot values. Each array has shape (nqoi, npilot).
        """
        import copy

        modes = ["values", "values_weights", "weights"]
        if mode not in modes:
            raise ValueError("mode must be in {0}".format(modes))
        if pilot_values is not None and mode not in modes[1:]:
            raise ValueError("pilot_values given by mode not in {0}".format(modes[1:]))
        bootstrap_vals = mode in modes[:2]
        bootstrap_weights = mode in modes[1:]
        nbootstraps = int(nbootstraps)
        estimator_vals = []
        if bootstrap_weights:
            npilot_samples = pilot_values[0].shape[1]
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
                boostrap_pilot_values = [vals[:, indices] for vals in pilot_values]
                self._stat.set_pilot_quantities(
                    *self._stat.compute_pilot_quantities(boostrap_pilot_values)
                )
                CF, cf = self._template._get_discrepancy_covariances(
                    self._allocation.npartition_samples
                )
                weights = self._template._optimal_weights(CF, cf)
                weights_list.append(self._bkd.flatten(weights))
            else:
                weights = self._weights_val
            estimator_vals.append(
                self._bkd.flatten(self._estimate(
                    values_per_model, weights, bootstrap=bootstrap_vals
                ))
            )
        estimator_vals = self._bkd.stack(estimator_vals)
        bootstrap_values_mean = estimator_vals.mean(axis=0)
        bootstrap_values_covar = self._bkd.cov(estimator_vals, rowvar=False, ddof=1)
        if bootstrap_weights:
            self._template._stat = self_stat
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

    def __repr__(self) -> str:
        rep = "{0}(criteria={1:.3g}".format(
            self.__class__.__name__,
            self._criteria_val,
        )
        rep += " target_cost={0:.5g}, nsamples={1})".format(
            self._allocation.actual_cost,
            self._allocation.nsamples_per_model,
        )
        return rep
