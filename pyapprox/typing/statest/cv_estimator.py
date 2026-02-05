"""Control variate estimator for multi-fidelity estimation.

This module provides the CVEstimator class for computing statistics
using control variate sampling with known low-fidelity statistics.
"""

import copy
from typing import Callable, Generic, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np

from pyapprox.typing.util.backends.protocols import Array

from pyapprox.typing.statest.statistics import (
    MultiOutputStatistic,
    MultiOutputVariance,
    MultiOutputMeanAndVariance,
)
from pyapprox.typing.statest.mc_estimator import MCEstimator

if TYPE_CHECKING:
    from pyapprox.typing.statest.allocation import CVAllocationResult


class CVEstimator(MCEstimator[Array], Generic[Array]):
    """Control variate estimator with known low-fidelity statistics."""

    def __init__(
        self,
        stat: MultiOutputStatistic[Array],
        costs: Union[List, Array],
        lowfi_stats: Array = None,
        opt_criteria: Callable = None,
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

        # Source of truth (define the allocation state)
        self._allocation: Optional["CVAllocationResult[Array]"] = None
        self._nsamples_per_model: Optional[Array] = None
        self._target_cost: Optional[float] = None

        # Cached computed values (invalidated when allocation changes)
        self._cached_weights: Optional[Array] = None
        self._cached_covariance: Optional[Array] = None
        self._cached_criteria: Optional[Array] = None
        self._cached_discrepancy_covariances: Optional[Tuple[Array, Array]] = None

        # Legacy attributes (kept for backward compatibility during transition)
        self._optimized_CF: Optional[Array] = None
        self._optimized_cf: Optional[Array] = None
        self._optimized_weights: Optional[Array] = None

        self._best_model_indices = self._bkd.arange(len(costs), dtype=int)

    def _invalidate_cache(self) -> None:
        """Invalidate all cached computed values."""
        self._cached_weights = None
        self._cached_covariance = None
        self._cached_criteria = None
        self._cached_discrepancy_covariances = None

    def set_allocation(self, allocation: "CVAllocationResult[Array]") -> None:
        """Set allocation from AllocationResult.

        Parameters
        ----------
        allocation : CVAllocationResult
            The allocation result. Must have success=True.

        Raises
        ------
        ValueError
            If allocation.success is False.
        """
        if not allocation.success:
            raise ValueError(f"Cannot set failed allocation: {allocation.message}")
        self._allocation = allocation
        self._nsamples_per_model = allocation.nsamples_per_model
        self._target_cost = allocation.actual_cost
        self._invalidate_cache()
        # Update legacy attributes for backward compatibility
        self._rounded_nsamples_per_model = allocation.nsamples_per_model
        self._rounded_npartition_samples = allocation.nsamples_per_model
        self._rounded_target_cost = allocation.actual_cost

    @property
    def has_allocation(self) -> bool:
        """Check if allocation has been set."""
        return self._nsamples_per_model is not None

    def _ensure_allocation(self) -> None:
        """Raise if no allocation has been set."""
        if not self.has_allocation:
            raise RuntimeError(
                "No allocation set. Call allocate_samples() or set_allocation() first."
            )

    def _compute_discrepancy_covariances(self) -> Tuple[Array, Array]:
        """Compute and cache CF, cf discrepancy covariances."""
        self._ensure_allocation()
        if self._cached_discrepancy_covariances is None:
            self._cached_discrepancy_covariances = self._get_discrepancy_covariances(
                self._nsamples_per_model
            )
        return self._cached_discrepancy_covariances

    def optimized_weights(self) -> Array:
        """Lazily compute and return optimized weights."""
        self._ensure_allocation()
        if self._cached_weights is None:
            CF, cf = self._compute_discrepancy_covariances()
            self._cached_weights = self._weights(CF, cf)
        return self._cached_weights

    def optimized_covariance(self) -> Array:
        """Lazily compute and return optimized covariance."""
        self._ensure_allocation()
        if self._cached_covariance is None:
            self._cached_covariance = self._covariance_from_nsamples_per_model(
                self._nsamples_per_model
            )
        return self._cached_covariance

    def optimized_criteria(self) -> Array:
        """Lazily compute and return optimization criteria value."""
        self._ensure_allocation()
        if self._cached_criteria is None:
            self._cached_criteria = self._optimization_criteria(
                self.optimized_covariance()
            )
        return self._cached_criteria

    def _get_discrepancy_covariances(
        self, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        return self._stat._get_cv_discrepancy_covariances(npartition_samples)

    def _covariance_from_nsamples_per_model(
        self, nsamples_per_model: Array
    ) -> Array:
        """Compute covariance from nsamples_per_model for CV estimator."""
        CF, cf = self._get_discrepancy_covariances(nsamples_per_model)
        weights = self._weights(CF, cf)
        return self._stat.high_fidelity_estimator_covariance(
            nsamples_per_model[0]
        ) + self._bkd.multidot([weights, cf.T])

    def _covariance_from_npartition_samples(
        self, npartition_samples: Array
    ) -> Array:
        """Compute covariance from npartition_samples.

        For CV estimator, npartition_samples = nsamples_per_model.
        """
        return self._covariance_from_nsamples_per_model(npartition_samples)

    def _set_optimized_params_base(
        self,
        rounded_npartition_samples: Array,
        rounded_nsamples_per_model: Array,
        rounded_target_cost: float,
    ):
        r"""
        Set the parameters needed to generate samples for evaluating the
        estimator. This method updates both the new lazy caching system
        and legacy attributes for backward compatibility.

        Parameters
        ----------
        rounded_npartition_samples : Array (npartitions, dtype=int)
            The number of samples in the independent sample partitions.

        rounded_nsamples_per_model :  Array (nmodels)
            The number of samples allocated to each model

        rounded_target_cost : float
            The cost of the new sample allocation
        """
        # Update source of truth
        self._nsamples_per_model = rounded_nsamples_per_model
        self._target_cost = rounded_target_cost
        self._invalidate_cache()

        # Legacy attributes (kept for backward compatibility)
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

    def nsamples_per_model(self) -> Array:
        """Return the number of samples allocated to each model.

        Returns
        -------
        Array
            Number of samples per model. Shape: (nmodels,)

        Raises
        ------
        ValueError
            If allocate_samples has not been called.
        """
        if not hasattr(self, "_rounded_nsamples_per_model"):
            raise ValueError("Call allocate_samples first.")
        return self._rounded_nsamples_per_model

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
        """Allocate samples for given budget.

        Parameters
        ----------
        target_cost : float
            Total computational budget.
        """
        from pyapprox.typing.statest.allocation import CVAllocationResult

        npartition_samples = [target_cost / self._costs.sum()]
        rounded_npartition_samples = [
            int(self._bkd.floor(npartition_samples[0]))
        ]
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

        # Create allocation result and set it
        allocation = CVAllocationResult(
            nsamples_per_model=rounded_nsamples_per_model,
            actual_cost=float(rounded_target_cost),
            objective_value=self._bkd.zeros((1,)),
            success=True,
            message="",
        )
        self._allocation = allocation

        # Still call the legacy method for backward compatibility
        self._set_optimized_params_base(
            rounded_npartition_samples,
            rounded_nsamples_per_model,
            rounded_target_cost,
        )

    def generate_samples_per_model(self, rvs: Callable) -> List[Array]:
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
        # print(self._bkd.cond(CF), "ACV COND")
        # return -self._bkd.multidot((self._bkd.pinv(CF), cf.T)).T
        return -self._bkd.solve(CF, cf.T).T

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
            + self._bkd.multidot([weights, CF, weights.T])
            + self._bkd.multidot([cf, weights.T])
            + self._bkd.multidot([weights, cf.T])
        )

    def _estimate(
        self,
        values_per_model: List[Array],
        weights: Array,
        bootstrap: bool = False,
    ) -> Array:
        """Compute CV estimate.

        Parameters
        ----------
        values_per_model : List[Array]
            Model values. Each array has shape (nqoi, nsamples).
        weights : Array
            Control variate weights.
        bootstrap : bool
            Whether to use bootstrap resampling.

        Returns
        -------
        Array
            Estimate. Shape: (nstats,)
        """
        if len(values_per_model) != self._nmodels:
            print(len(self._lowfi_stats), self._nmodels)
            msg = "Must provide the values for each model."
            msg += " {0} != {1}".format(len(values_per_model), self._nmodels)
            raise ValueError(msg)
        nsamples = values_per_model[0].shape[1]
        for values in values_per_model[1:]:
            if values.shape[1] != nsamples:
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
                self._stat.sample_estimate(values_per_model[ii][:, indices])
                - self._lowfi_stats[ii - 1]
                for ii in range(1, self._nmodels)
            ]
        )
        est = (
            self._stat.sample_estimate(values_per_model[0][:, indices])
            + weights @ deltas
        )
        return est

    def __call__(self, values_per_model: List[Array]) -> Array:
        r"""
        Return the value of the Monte Carlo like estimator

        Parameters
        ----------
        values_per_model : list (nmodels)
            The unique values of each model. Each array has shape (nqoi, nsamples).

        Returns
        -------
        est : Array (nstats,)
            The estimator value.
        """
        for vals in values_per_model:
            if not isinstance(vals, self._bkd.array_type()):
                raise ValueError(
                    "vals must be an instance of {0}".format(
                        self._bkd.array_type()
                    )
                )
        return self._estimate(values_per_model, self.optimized_weights())

    def insert_pilot_values(
        self, pilot_values: List[Array], values_per_model: List[Array]
    ) -> List[Array]:
        """
        Only add pilot values to the first independent partition and thus
        only to models that use that partition.

        Parameters
        ----------
        pilot_values : List[Array]
            Pilot values for each model. Each array has shape (nqoi, npilot).
        values_per_model : List[Array]
            Model values. Each array has shape (nqoi, nsamples).

        Returns
        -------
        List[Array]
            Combined values. Each array has shape (nqoi, npilot + nsamples).
        """
        new_values_per_model = []
        for ii in range(self._nmodels):
            active_partition = (self._allocation_mat[0, 2 * ii] == 1) or (
                self._allocation_mat[0, 2 * ii + 1] == 1
            )
            if active_partition:
                # hstack along axis=1 (samples axis) for (nqoi, nsamples)
                new_values_per_model.append(
                    self._bkd.hstack((pilot_values[ii], values_per_model[ii]))
                )
            else:
                new_values_per_model.append(self._bkd.copy(values_per_model[ii]))
        return new_values_per_model

    def __repr__(self):
        """String representation handling unset allocation."""
        if not self.has_allocation:
            return f"{self.__class__.__name__}(allocation=None)"
        # Convert Array to float for formatting
        criteria_val = float(self._bkd.to_numpy(self.optimized_criteria()).flat[0])
        return (
            f"{self.__class__.__name__}("
            f"criteria={criteria_val:.3g}, "
            f"target_cost={self._target_cost:.5g})"
        )

    def bootstrap(
        self,
        values_per_model: List[Array],
        nbootstraps: int = 1000,
        mode: str = "values",
        pilot_values: List[Array] = None,
    ):
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
                boostrap_pilot_values = [
                    vals[:, indices] for vals in pilot_values
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
                weights = self.optimized_weights()
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
