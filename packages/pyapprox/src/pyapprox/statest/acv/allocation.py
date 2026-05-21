"""Sample allocation optimization for ACV estimators.

This module separates allocation optimization from estimation, providing:
- Allocator: Abstract base for allocation strategies
- ACVAllocator: Optimization-based allocator
- AnalyticalAllocator: Closed-form allocator for MFMC/MLMC
"""

import copy
from abc import ABC, abstractmethod
from typing import Generic, List, Optional

from pyapprox.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)
from pyapprox.optimization.minimize.result_protocol import (
    OptimizerResultProtocol,
)
from pyapprox.statest.acv.base import ACVEstimator
from pyapprox.statest.acv.optimization import (
    ACVLogDeterminantObjective,
    ACVObjective,
    ACVPartitionConstraint,
)
from pyapprox.statest.acv.result import ACVAllocationResult
from pyapprox.util.backends.protocols import Array, Backend


def _clone_estimator_for_torch(
    estimator: ACVEstimator[Array],
) -> ACVEstimator[Array]:
    """Create a torch-backed clone of an estimator for optimization.

    Reuses the same logic as ACVEstimator._clone_for_torch_optimization:
    shallow copy with TorchBkd, converting costs, stat, allocation_mat,
    and recursion_index.

    Parameters
    ----------
    estimator : ACVEstimator
        The estimator (any backend) to clone.

    Returns
    -------
    ACVEstimator
        A shallow copy backed by TorchBkd.
    """
    import torch

    from pyapprox.statest.statistics import (
        MultiOutputMean,
        MultiOutputMeanAndVariance,
        MultiOutputVariance,
    )
    from pyapprox.util.backends.torch import TorchBkd

    torch_bkd = TorchBkd()
    clone = copy.copy(estimator)
    clone._bkd = torch_bkd
    clone._costs = torch_bkd.asarray(
        estimator._bkd.to_numpy(estimator._costs), dtype=torch.double
    )

    # Create fresh torch stat with re-derived pilot quantities
    nqoi = estimator._stat.nqoi()
    stat = estimator._stat

    def _to_torch_double(arr: Array) -> Array:
        return torch_bkd.asarray(estimator._bkd.to_numpy(arr), dtype=torch.double)

    if isinstance(stat, MultiOutputMeanAndVariance):
        clone._stat = MultiOutputMeanAndVariance(nqoi, torch_bkd, tril=stat._tril)
        clone._stat.set_pilot_quantities(
            _to_torch_double(stat._cov),
            _to_torch_double(stat._W),
            _to_torch_double(stat._B),
        )
    elif isinstance(stat, MultiOutputVariance):
        clone._stat = MultiOutputVariance(nqoi, torch_bkd, tril=stat._tril)
        clone._stat.set_pilot_quantities(
            _to_torch_double(stat._cov),
            _to_torch_double(stat._W),
        )
    elif isinstance(stat, MultiOutputMean):
        clone._stat = MultiOutputMean(nqoi, torch_bkd)
        clone._stat.set_pilot_quantities(
            _to_torch_double(stat._cov),
        )
    else:
        raise TypeError(
            f"Unsupported stat type for torch optimization: {type(stat).__name__}"
        )

    # Convert allocation matrix
    if hasattr(estimator, "_allocation_mat") and estimator._allocation_mat is not None:
        clone._allocation_mat = _to_torch_double(estimator._allocation_mat)

    # Convert recursion index
    if (
        hasattr(estimator, "_recursion_index")
        and estimator._recursion_index is not None
    ):
        clone._recursion_index = _to_torch_double(estimator._recursion_index)

    return clone


def _convert_result_to_backend(
    result: ACVAllocationResult[Array],
    bkd: Backend[Array],
) -> ACVAllocationResult[Array]:
    """Convert an ACVAllocationResult's arrays to a different backend.

    Parameters
    ----------
    result : ACVAllocationResult
        The result to convert (typically from TorchBkd optimization).
    bkd : Backend
        The target backend.

    Returns
    -------
    ACVAllocationResult
        A new result with all Array fields converted to the target backend.
    """
    from pyapprox.util.backends.torch import TorchBkd

    src_bkd = TorchBkd()
    return ACVAllocationResult(
        partition_ratios=bkd.asarray(src_bkd.to_numpy(result.partition_ratios)),
        continuous_npartition_samples=bkd.asarray(
            src_bkd.to_numpy(result.continuous_npartition_samples)
        ),
        objective_value=bkd.asarray(src_bkd.to_numpy(result.objective_value)),
        npartition_samples=bkd.asarray(
            src_bkd.to_numpy(result.npartition_samples), dtype=int
        ),
        nsamples_per_model=bkd.asarray(
            src_bkd.to_numpy(result.nsamples_per_model), dtype=int
        ),
        target_cost=result.target_cost,
        actual_cost=result.actual_cost,
        success=result.success,
        message=result.message,
    )


def _build_allocation_result(
    est: ACVEstimator[Array],
    bkd: Backend[Array],
    target_cost: float,
    partition_ratios: Array,
    objective_value: Array,
) -> ACVAllocationResult[Array]:
    """Convert continuous partition ratios to a discrete allocation result.

    Uses rounding (not floor) and clamps HF samples to at least 1 so that
    near-integer continuous values like 0.99 round up instead of down.
    """
    continuous = est._npartition_samples_from_partition_ratios(
        target_cost, partition_ratios
    )
    nhf_continuous = bkd.to_float(continuous[0])
    if nhf_continuous < 1 - 1e-3:
        nmodels = est._nmodels
        npartitions = est._npartitions
        return ACVAllocationResult(
            partition_ratios=bkd.zeros((nmodels - 1,)),
            continuous_npartition_samples=bkd.zeros((npartitions,)),
            objective_value=bkd.array([float("inf")]),
            npartition_samples=bkd.zeros((npartitions,), dtype=int),
            nsamples_per_model=bkd.zeros((nmodels,), dtype=int),
            target_cost=target_cost,
            actual_cost=0.0,
            success=False,
            message=f"Would give {nhf_continuous:.2f} HF samples",
        )

    # Floor all partitions to stay within budget, but clamp HF (index 0)
    # to at least 1 so near-integer values like 0.999 don't floor to 0.
    npartition_samples = bkd.asarray(
        bkd.floor(continuous + 1e-8), dtype=int
    )
    if bkd.to_int(npartition_samples[0]) < 1:
        npartition_samples[0] = 1
    nsamples_per_model = bkd.asarray(
        est._compute_nsamples_per_model(npartition_samples), dtype=int
    )
    actual_cost = bkd.to_float(
        bkd.sum(nsamples_per_model * est._costs)
    )

    return ACVAllocationResult(
        partition_ratios=partition_ratios,
        continuous_npartition_samples=continuous,
        objective_value=objective_value,
        npartition_samples=npartition_samples,
        nsamples_per_model=nsamples_per_model,
        target_cost=target_cost,
        actual_cost=actual_cost,
        success=True,
        message="",
    )


class Allocator(ABC, Generic[Array]):
    """Abstract base for sample allocation strategies."""

    @abstractmethod
    def allocate(self, target_cost: float) -> ACVAllocationResult[Array]:
        """Allocate samples for the estimator.

        Parameters
        ----------
        target_cost : float
            The total computational budget.

        Returns
        -------
        AllocationResult
            The allocation result containing partition ratios, sample counts,
            and metadata.
        """
        raise NotImplementedError


class ACVAllocator(Allocator[Array]):
    """Optimization-based allocator for ACV estimators.

    Parameters
    ----------
    estimator : ACVEstimator
        The estimator to optimize allocation for.
    optimizer : optional
        Optimizer to use. If None, uses default chained optimizer.
    objective : optional
        Objective function. If None, uses ACVLogDeterminantObjective.
    """

    def __init__(
        self,
        estimator: ACVEstimator[Array],
        optimizer: Optional[BindableOptimizerProtocol[Array]] = None,
        objective: Optional[ACVObjective[Array]] = None,
    ) -> None:
        self._est = estimator
        self._bkd: Backend[Array] = estimator._bkd
        self._optimizer = optimizer
        self._objective = objective

    def _get_optimizer(self) -> BindableOptimizerProtocol[Array]:
        if self._optimizer is not None:
            return self._optimizer
        return self._default_optimizer()

    def _default_optimizer(self) -> BindableOptimizerProtocol[Array]:
        from pyapprox.optimization.minimize.chained.chained_optimizer import (
            ChainedOptimizer,
        )
        from pyapprox.optimization.minimize.scipy.diffevol import (
            ScipyDifferentialEvolutionOptimizer,
        )
        from pyapprox.optimization.minimize.scipy.trust_constr import (
            ScipyTrustConstrOptimizer,
        )

        return ChainedOptimizer(
            ScipyDifferentialEvolutionOptimizer(maxiter=3, raise_on_failure=False),
            ScipyTrustConstrOptimizer(),
        )

    def _get_objective(
        self, optimizer: BindableOptimizerProtocol[Array]
    ) -> ACVObjective[Array]:
        if self._objective is not None:
            obj = self._objective
        else:
            scaling = getattr(optimizer, "_scaling", 1)
            obj = ACVLogDeterminantObjective(scaling=scaling, bkd=self._bkd)
        obj.set_estimator(self._est)
        return obj

    def allocate(self, target_cost: float) -> ACVAllocationResult[Array]:
        """Allocate samples using optimization.

        Automatically clones the estimator to TorchBkd when the estimator
        uses NumpyBkd, since the trust-constr optimizer requires autodiff
        gradients. The result arrays are converted back to the original
        backend.

        Parameters
        ----------
        target_cost : float
            The total computational budget.

        Returns
        -------
        AllocationResult
            The allocation result. Check `success` field for status.
        """
        from pyapprox.util.backends.numpy import NumpyBkd

        if isinstance(self._bkd, NumpyBkd):
            return self._allocate_with_torch_clone(target_cost)
        return self._allocate_impl(target_cost)

    def _allocate_with_torch_clone(
        self, target_cost: float
    ) -> ACVAllocationResult[Array]:
        """Run allocation via a TorchBkd clone, then convert result back."""
        import torch

        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            torch_est = _clone_estimator_for_torch(self._est)
            torch_allocator = ACVAllocator(torch_est, optimizer=self._optimizer)
            torch_result = torch_allocator._allocate_impl(target_cost)
            if not torch_result.success:
                return self._failure_result(target_cost, torch_result.message)
            return _convert_result_to_backend(torch_result, self._bkd)
        finally:
            torch.set_default_dtype(prev_dtype)

    def _allocate_impl(self, target_cost: float) -> ACVAllocationResult[Array]:
        """Core allocation logic (requires autodiff-capable backend)."""
        if target_cost < self._bkd.to_float(self._bkd.sum(self._est._costs)):
            return self._failure_result(
                target_cost, "Budget too small for one sample per model"
            )

        optimizer = self._get_optimizer()
        objective = self._get_objective(optimizer)

        try:
            opt_result = self._run_optimization(target_cost, optimizer, objective)
        except Exception as e:
            return self._failure_result(target_cost, str(e))

        if not opt_result.success():
            return self._failure_result(target_cost, opt_result.message())

        partition_ratios = opt_result.optima()[:, 0]
        # Optimizer returns float, wrap in Array to preserve interface
        objective_value = self._bkd.array([opt_result.fun()])

        return self._build_result(target_cost, partition_ratios, objective_value)

    def _run_optimization(
        self,
        target_cost: float,
        optimizer: BindableOptimizerProtocol[Array],
        objective: ACVObjective[Array],
    ) -> OptimizerResultProtocol[Array]:
        objective.set_target_cost(target_cost)
        constraint = ACVPartitionConstraint(self._est, target_cost)
        constraints: List[ACVPartitionConstraint[Array]] = [constraint]
        bounds = self._est.get_npartition_bounds(target_cost)
        optimizer.bind(objective, bounds, constraints)
        init_iterate = self._bkd.full((self._est._nmodels - 1, 1), 1.0)
        init_iterate = self._ensure_feasible_init(init_iterate, constraint, bounds)
        return optimizer.minimize(init_iterate)

    def _ensure_feasible_init(
        self,
        init_iterate: Array,
        constraint: ACVPartitionConstraint[Array],
        bounds: Array,
    ) -> Array:
        """Ensure the initial iterate satisfies constraints and bounds.

        If the default initial guess (all ones) is infeasible, projects it
        to the midpoint of the bounds which is more likely to be feasible.

        Parameters
        ----------
        init_iterate : Array, shape (nvars, 1)
            The initial iterate to check.
        constraint : ACVPartitionConstraint
            The partition constraint.
        bounds : Array, shape (nvars, 2)
            Variable bounds [lower, upper] per row.

        Returns
        -------
        Array, shape (nvars, 1)
            A feasible initial iterate.
        """
        constraint_vals = constraint(init_iterate)
        lb = constraint.lb()
        if self._bkd.all_bool(constraint_vals[:, 0] >= lb):
            return init_iterate

        # Try midpoint of bounds
        mid = ((bounds[:, 0] + bounds[:, 1]) / 2.0)[:, None]
        constraint_vals_mid = constraint(mid)
        if self._bkd.all_bool(constraint_vals_mid[:, 0] >= lb):
            return mid

        # Try lower bounds (smallest feasible ratios)
        lo = bounds[:, 0:1]
        constraint_vals_lo = constraint(lo)
        if self._bkd.all_bool(constraint_vals_lo[:, 0] >= lb):
            return lo

        # Return original; optimizer will handle infeasibility
        return init_iterate

    def _build_result(
        self,
        target_cost: float,
        partition_ratios: Array,
        objective_value: Array,
    ) -> ACVAllocationResult[Array]:
        return _build_allocation_result(
            self._est, self._bkd, target_cost, partition_ratios,
            objective_value,
        )

    def _failure_result(
        self, target_cost: float, message: str
    ) -> ACVAllocationResult[Array]:
        nmodels = self._est._nmodels
        npartitions = self._est._npartitions
        return ACVAllocationResult(
            partition_ratios=self._bkd.zeros((nmodels - 1,)),
            continuous_npartition_samples=self._bkd.zeros((npartitions,)),
            objective_value=self._bkd.array([float("inf")]),
            npartition_samples=self._bkd.zeros((npartitions,), dtype=int),
            nsamples_per_model=self._bkd.zeros((nmodels,), dtype=int),
            target_cost=target_cost,
            actual_cost=0.0,
            success=False,
            message=message,
        )


class AnalyticalAllocator(Allocator[Array]):
    """Analytical (closed-form) allocator for MFMC/MLMC.

    Uses the estimator's `_allocate_samples_analytical` method for
    closed-form allocation formulas.

    Parameters
    ----------
    estimator : ACVEstimator
        The estimator to allocate for. Must have `_allocate_samples_analytical`.
    """

    def __init__(self, estimator: ACVEstimator[Array]):
        self._est = estimator
        self._bkd: Backend[Array] = estimator._bkd

    def allocate(self, target_cost: float) -> ACVAllocationResult[Array]:
        """Allocate samples using analytical formula.

        Parameters
        ----------
        target_cost : float
            The total computational budget.

        Returns
        -------
        AllocationResult
            The allocation result. Check `success` field for status.
        """
        if target_cost < self._bkd.to_float(self._bkd.sum(self._est._costs)):
            return self._failure_result(
                target_cost, "Budget too small for one sample per model"
            )

        try:
            partition_ratios, objective_value = self._est._allocate_samples_analytical(
                target_cost
            )
        except Exception as e:
            return self._failure_result(target_cost, str(e))

        return self._build_result(target_cost, partition_ratios, objective_value)

    def _build_result(
        self,
        target_cost: float,
        partition_ratios: Array,
        objective_value: Array,
    ) -> ACVAllocationResult[Array]:
        return _build_allocation_result(
            self._est, self._bkd, target_cost, partition_ratios,
            objective_value,
        )

    def _failure_result(
        self, target_cost: float, message: str
    ) -> ACVAllocationResult[Array]:
        nmodels = self._est._nmodels
        npartitions = self._est._npartitions
        return ACVAllocationResult(
            partition_ratios=self._bkd.zeros((nmodels - 1,)),
            continuous_npartition_samples=self._bkd.zeros((npartitions,)),
            objective_value=self._bkd.array([float("inf")]),
            npartition_samples=self._bkd.zeros((npartitions,), dtype=int),
            nsamples_per_model=self._bkd.zeros((nmodels,), dtype=int),
            target_cost=target_cost,
            actual_cost=0.0,
            success=False,
            message=message,
        )


def default_allocator_factory(
    estimator: ACVEstimator[Array],
    optimizer: Optional[BindableOptimizerProtocol[Array]] = None,
) -> Allocator[Array]:
    """Create appropriate allocator for estimator type.

    Dispatches by capability: estimators with `_allocate_samples_analytical`
    get AnalyticalAllocator, everything else gets ACVAllocator.

    Parameters
    ----------
    estimator : ACVEstimator
        The estimator to create an allocator for.
    optimizer : optional
        Optimizer to use for optimization-based allocators (ACVAllocator).
        If None, uses the default chained optimizer. Ignored when an
        analytical allocator is selected.

    Returns
    -------
    Allocator
        The most efficient allocator for the given estimator type.
    """
    if hasattr(estimator, "_allocate_samples_analytical"):
        return AnalyticalAllocator(estimator)

    return ACVAllocator(estimator, optimizer=optimizer)
