"""ACVEstimator base class for approximate control variate estimation.

This module provides the ACVEstimator class which extends CVEstimator
with approximate control variate functionality including sample allocation
optimization.
"""

from abc import abstractmethod
from typing import Callable, Generic, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np

from pyapprox.typing.util.backends.protocols import Array

from pyapprox.typing.statest.statistics import (
    MultiOutputStatistic,
)
from pyapprox.typing.statest.cv_estimator import CVEstimator
from pyapprox.typing.statest.acv.optimization import (
    _combine_acv_values,
    _combine_acv_samples,
)

if TYPE_CHECKING:
    from pyapprox.typing.statest.acv.allocation import ACVAllocationResult


class ACVEstimator(CVEstimator[Array], Generic[Array]):
    """Approximate Control Variate estimator base class."""

    def __init__(
        self,
        stat: MultiOutputStatistic[Array],
        costs: Union[List, Array],
        recursion_index: Array = None,
        opt_criteria: Callable = None,
        tree_depth: int = None,
        allow_failures: bool = False,
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

        opt_criteria : callable
            Function of the the covariance between the high-fidelity
            QoI estimators with signature

            ``opt_criteria(variance) -> float

        tree_depth: integer (default=None)
            The maximum depth of the recursion tree.
            If not None, then recursion_index is ignored.

        allow_failures: boolean (default=False)
            Allow optimization of estimators to fail when enumerating
            each recursion tree.
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
        self._npartitions_lower_bound = npartitions_lower_bound

        # ACV-specific source of truth
        self._partition_ratios: Optional[Array] = None
        self._npartition_samples: Optional[Array] = None

    # === Allocation management (new API) ===

    def set_allocation(self, allocation: "ACVAllocationResult[Array]") -> None:
        """Set allocation from ACVAllocationResult.

        Parameters
        ----------
        allocation : ACVAllocationResult
            The allocation result. Must have success=True.

        Raises
        ------
        ValueError
            If allocation.success is False.
        """
        if not allocation.success:
            raise ValueError(f"Cannot set failed allocation: {allocation.message}")
        self._allocation = allocation
        self._partition_ratios = allocation.partition_ratios
        self._npartition_samples = allocation.npartition_samples
        self._nsamples_per_model = allocation.nsamples_per_model
        self._target_cost = allocation.actual_cost
        self._invalidate_cache()

        # Update legacy attributes for backward compatibility
        self._rounded_partition_ratios = allocation.partition_ratios
        self._rounded_npartition_samples = allocation.npartition_samples
        self._rounded_nsamples_per_model = allocation.nsamples_per_model
        self._rounded_target_cost = allocation.actual_cost
        # Pre-compute legacy attributes for backward compatibility
        self._optimized_covariance = self._covariance_from_npartition_samples(
            allocation.npartition_samples
        )
        self._optimized_criteria = self._optimization_criteria(
            self._optimized_covariance
        )
        self._optimized_CF, self._optimized_cf = (
            self._get_discrepancy_covariances(allocation.npartition_samples)
        )
        self._optimized_weights = self._weights(
            self._optimized_CF, self._optimized_cf
        )

    def allocation(self) -> "ACVAllocationResult[Array]":
        """Get current allocation.

        Returns
        -------
        ACVAllocationResult
            The current allocation.

        Raises
        ------
        RuntimeError
            If no allocation has been set.
        """
        if self._allocation is None:
            raise RuntimeError(
                "No allocation set. Call allocate_samples() or set_allocation() first."
            )
        return self._allocation

    @property
    def has_allocation(self) -> bool:
        """Check if allocation has been set.

        Returns
        -------
        bool
            True if an allocation has been set, False otherwise.
        """
        return (
            self._allocation is not None
            or self._nsamples_per_model is not None
        )

    def _compute_discrepancy_covariances(self) -> Tuple[Array, Array]:
        """Compute and cache CF, cf discrepancy covariances for ACV."""
        self._ensure_allocation()
        if self._cached_discrepancy_covariances is None:
            npartition_samples = (
                self._npartition_samples
                if self._npartition_samples is not None
                else self._rounded_npartition_samples
            )
            self._cached_discrepancy_covariances = self._get_discrepancy_covariances(
                npartition_samples
            )
        return self._cached_discrepancy_covariances

    def optimized_covariance(self) -> Array:
        """Lazily compute and return optimized covariance for ACV."""
        self._ensure_allocation()
        if self._cached_covariance is None:
            npartition_samples = (
                self._npartition_samples
                if self._npartition_samples is not None
                else self._rounded_npartition_samples
            )
            self._cached_covariance = self._covariance_from_npartition_samples(
                npartition_samples
            )
        return self._cached_covariance

    def allocation_matrix(self) -> Array:
        """Return the allocation matrix.

        Returns
        -------
        Array
            Allocation matrix. Shape: (npartitions, 2*nmodels).
        """
        return self._get_allocation_matrix()

    def npartition_samples(self) -> Array:
        """Get current partition sample allocation.

        Returns
        -------
        Array
            Number of samples in each partition. Shape (npartitions,).

        Raises
        ------
        RuntimeError
            If allocation has not been set.
        """
        self._ensure_allocation()
        return self._npartition_samples

    # === Optimization mode (continuous, stateless) ===

    def covariance_from_ratios(
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

    def npartition_samples_from_ratios(
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

    # === Evaluation mode (discrete, stateful) ===

    def covariance(self) -> Array:
        """Compute estimator covariance using stored discrete allocation.

        Returns
        -------
        Array
            The estimator covariance matrix.

        Raises
        ------
        RuntimeError
            If no allocation has been set.
        """
        return self._covariance_from_npartition_samples(
            self.allocation().npartition_samples
        )

    # === End of new allocation API ===

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
        """
        npartition_samples = self._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios
        )
        return self._covariance_from_npartition_samples(npartition_samples)

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
        if len(values_per_model) != self._nmodels:
            msg = "len(values_per_model) {0} != nmodels {1}".format(
                len(values_per_model), self._nmodels
            )
            raise ValueError(msg)
        for ii in range(self._nmodels):
            if (
                values_per_model[ii].shape[1]
                != self._rounded_nsamples_per_model[ii]
            ):
                msg = "{0} != {1}".format(
                    "values_per_model[{0}].shape[1]: {1}".format(
                        ii, values_per_model[ii].shape[1]
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
        # Index along axis=1 (samples axis) for (nqoi, nsamples)
        acv_values = [
            values_per_model[ii // 2][:, acv_partition_indices[ii]]
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
        self, rvs: Callable, npilot_samples: int = 0
    ) -> List[Array]:
        ntotal_independent_samples = int(
            self._rounded_npartition_samples.sum() - npilot_samples
        )
        independent_samples = rvs(ntotal_independent_samples)
        samples_per_model = []
        rounded_npartition_samples = self._bkd.copy(
            self._rounded_npartition_samples
        )
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

    def _estimate(
        self,
        values_per_model: List[Array],
        weights: Array,
        bootstrap: bool = False,
    ) -> Array:
        acv_values = self._separate_values_per_model(
            values_per_model, bootstrap
        )
        return self._estimate_from_acv_values(
            acv_values, weights, len(values_per_model)
        )

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
        """
        raise NotImplementedError

    def _get_allocation_matrix(self) -> Array:
        """return allocation matrix as backend array"""
        return self._bkd.asarray(self._allocation_mat)

    def _set_recursion_index(self, index: Array):
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

    def get_default_optimizer(self):
        from pyapprox.typing.optimization.minimize.scipy.diffevol import (
            ScipyDifferentialEvolutionOptimizer,
        )
        from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
            ScipyTrustConstrOptimizer,
        )
        from pyapprox.typing.optimization.minimize.chained.chained_optimizer import (
            ChainedOptimizer,
        )

        global_optimizer = ScipyDifferentialEvolutionOptimizer(
            maxiter=3, raise_on_failure=False
        )
        local_optimizer = ScipyTrustConstrOptimizer()
        optimizer = ChainedOptimizer(global_optimizer, local_optimizer)
        return optimizer

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    def _estimator_cost(self, npartition_samples: Array) -> float:
        nsamples_per_model = self._compute_nsamples_per_model(
            npartition_samples
        )
        return (nsamples_per_model * self._costs).sum()

    def _allocate_samples_for_single_recursion(self, target_cost: float):
        from pyapprox.typing.statest.acv.allocation import (
            ACVAllocator,
            default_allocator_factory,
        )

        if self._optimizer is not None:
            allocator = ACVAllocator(self, optimizer=self._optimizer)
        else:
            allocator = default_allocator_factory(self)
        result = allocator.allocate(target_cost)
        if not result.success:
            raise RuntimeError(
                "{0} optimizer failed with message {1}".format(
                    self, result.message
                )
            )
        self.set_allocation(result)

    def _get_optimizer_verbosity(self) -> int:
        """Get verbosity level from the optimizer.

        Returns
        -------
        int
            Verbosity level, or 0 if optimizer doesn't support verbosity.
        """
        if hasattr(self._optimizer, "local_optimizer_verbosity"):
            return self._optimizer.local_optimizer_verbosity()
        if hasattr(self._optimizer, "_verbosity"):
            return self._optimizer._verbosity
        return 0

    def get_all_recursion_indices(self) -> List[Array]:
        from pyapprox.multifidelity._optim import _get_acv_recursion_indices
        return _get_acv_recursion_indices(self._nmodels, self._tree_depth)

    def _allocate_samples_for_all_recursion_indices(self, target_cost: float):
        from pyapprox.typing.statest.acv.allocation import (
            ACVAllocator,
            default_allocator_factory,
        )

        best_obj = self._bkd.asarray(np.inf)
        best_result = None
        best_index = None
        for index in self.get_all_recursion_indices():
            self._set_recursion_index(index)
            if self._optimizer is not None:
                allocator = ACVAllocator(self, optimizer=self._optimizer)
            else:
                allocator = default_allocator_factory(self)
            result = allocator.allocate(target_cost)
            if not result.success:
                if not self._allow_failures:
                    raise RuntimeError(
                        "{0} optimizer failed with message {1}".format(
                            self, result.message
                        )
                    )
                if self._get_optimizer_verbosity() > 0:
                    print("Optimizer failed")
                continue
            obj_val = result.objective_value[0]
            if self._get_optimizer_verbosity() > 2:
                msg = "\t\t Recursion: {0} Objective: best {1}, current {2}".format(
                    index,
                    best_obj.item(),
                    obj_val.item(),
                )
                print(msg)
            if obj_val < best_obj:
                best_obj = obj_val
                best_result = result
                best_index = index
        if best_result is None:
            raise RuntimeError("No solutions were found")
        self._set_recursion_index(best_index)
        self.set_allocation(best_result)

    def allocate_samples(self, target_cost: float):
        if self._tree_depth is not None:
            return self._allocate_samples_for_all_recursion_indices(
                target_cost
            )
        return self._allocate_samples_for_single_recursion(target_cost)
