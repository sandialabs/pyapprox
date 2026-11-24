"""Sample allocation optimization for ACV estimators.

This module separates allocation optimization from estimation, providing:
- Allocator: Abstract base for allocation strategies
- ACVAllocator: Optimization-based allocator on the estimator's backend
- ACVAllocatorViaAutodiff: Optimization-based allocator for an estimator
  on any backend, optimizing a copy of it on an autodiff backend
- ViaAutodiffAllocatorFactory: Allocator factory that optimizes on an
  autodiff backend
- AnalyticalAllocator: Closed-form allocator for MFMC/MLMC
"""

from abc import ABC, abstractmethod
from typing import (
    Callable,
    Generic,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

from pyapprox.interface.functions.autograd import (
    WithAutogradJacobian,
    WithAutogradJacobianConstraint,
)
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
from pyapprox.statest.protocols import TargetArray
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array, Array_co, Backend


def _convert_result_to_backend(
    result: ACVAllocationResult[TargetArray],
    bkd: Backend[Array],
) -> ACVAllocationResult[Array]:
    """Convert an ACVAllocationResult's arrays to a different backend.

    Parameters
    ----------
    result : ACVAllocationResult
        The result to convert (typically from autodiff-backend
        optimization).
    bkd : Backend
        The target backend.

    Returns
    -------
    ACVAllocationResult
        A new result with all Array fields converted to the target backend.
    """
    return ACVAllocationResult(
        partition_ratios=bkd.asarray(result.partition_ratios),
        continuous_npartition_samples=bkd.asarray(result.continuous_npartition_samples),
        objective_value=bkd.asarray(result.objective_value),
        npartition_samples=bkd.asarray(
            result.npartition_samples, dtype=bkd.int64_dtype()
        ),
        nsamples_per_model=bkd.asarray(
            result.nsamples_per_model, dtype=bkd.int64_dtype()
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
            npartition_samples=bkd.zeros((npartitions,), dtype=bkd.int64_dtype()),
            nsamples_per_model=bkd.zeros((nmodels,), dtype=bkd.int64_dtype()),
            target_cost=target_cost,
            actual_cost=0.0,
            success=False,
            message=f"Would give {nhf_continuous:.2f} HF samples",
        )

    # Floor all partitions to stay within budget, but clamp HF (index 0)
    # to at least 1 so near-integer values like 0.999 don't floor to 0.
    npartition_samples = bkd.asarray(
        bkd.floor(continuous + 1e-8), dtype=bkd.int64_dtype()
    )
    if bkd.to_int(npartition_samples[0]) < 1:
        npartition_samples[0] = 1
    nsamples_per_model = bkd.asarray(
        est._compute_nsamples_per_model(npartition_samples), dtype=bkd.int64_dtype()
    )
    actual_cost = bkd.to_float(bkd.sum(nsamples_per_model * est._costs))

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

    Optimizes on the estimator's own backend. On a backend with autodiff
    the objective and constraint get autograd jacobians; on one without,
    the optimizer must not need them -- a gradient-free optimizer, or
    one that approximates them. :class:`ACVAllocatorViaAutodiff` runs this
    on an autodiff backend for an estimator on any backend.

    Parameters
    ----------
    estimator : ACVEstimator
        The estimator to optimize allocation for.
    optimizer : optional
        Optimizer to use. If None, uses a chained optimizer: a few
        differential-evolution generations, then trust-constr, which
        approximates jacobians by finite differences when the backend
        supplies none.
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
            ScipyDifferentialEvolutionOptimizer[Array](
                maxiter=3, raise_on_failure=False
            ),
            ScipyTrustConstrOptimizer[Array](),
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
        bounds = self._est.get_npartition_bounds(target_cost)
        # Autograd is a composition source: ACV objectives/constraints
        # declare no analytical derivatives, so on an autodiff-capable
        # backend wrap them to fill the jacobian from autograd.
        bound_obj: Union[ACVObjective[Array], WithAutogradJacobian[Array]] = objective
        bound_con: Union[
            ACVPartitionConstraint[Array],
            WithAutogradJacobianConstraint[Array],
        ] = constraint
        if isinstance(self._bkd, AutodiffBackend):
            if objective.derivatives().jacobian is None:
                bound_obj = WithAutogradJacobian(objective, self._bkd)
            if constraint.derivatives().jacobian is None:
                bound_con = WithAutogradJacobianConstraint(constraint, self._bkd)
        optimizer.bind(bound_obj, bounds, [bound_con])
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
        if self._bkd.to_bool(self._bkd.all(constraint_vals[:, 0] >= lb)):
            return init_iterate

        # Try midpoint of bounds
        mid = ((bounds[:, 0] + bounds[:, 1]) / 2.0)[:, None]
        constraint_vals_mid = constraint(mid)
        if self._bkd.to_bool(self._bkd.all(constraint_vals_mid[:, 0] >= lb)):
            return mid

        # Try lower bounds (smallest feasible ratios)
        lo = bounds[:, 0:1]
        constraint_vals_lo = constraint(lo)
        if self._bkd.to_bool(self._bkd.all(constraint_vals_lo[:, 0] >= lb)):
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
            self._est,
            self._bkd,
            target_cost,
            partition_ratios,
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
            npartition_samples=self._bkd.zeros(
                (npartitions,), dtype=self._bkd.int64_dtype()
            ),
            nsamples_per_model=self._bkd.zeros(
                (nmodels,), dtype=self._bkd.int64_dtype()
            ),
            target_cost=target_cost,
            actual_cost=0.0,
            success=False,
            message=message,
        )


class ACVAllocatorViaAutodiff(Allocator[Array], Generic[Array, TargetArray]):
    """Optimization-based allocator for an estimator on any backend,
    optimizing on an autodiff backend.

    Moves a copy of the estimator to ``autodiff_bkd``, allocates there with
    :class:`ACVAllocator` -- so the objective and constraint get autograd
    jacobians -- and returns the result on the estimator's own backend.
    The optimizer and objective therefore work on ``autodiff_bkd``'s
    arrays, whatever backend the estimator uses.

    Parameters
    ----------
    estimator : ACVEstimator
        The estimator to optimize allocation for, on any backend.
    autodiff_bkd : AutodiffBackend
        The backend the optimization runs on; its configured dtype and
        device govern the solve.
    optimizer : optional
        Optimizer on ``autodiff_bkd``'s arrays. If None, uses the default
        of :class:`ACVAllocator`.
    objective : optional
        Objective on ``autodiff_bkd``'s arrays. If None, uses
        ACVLogDeterminantObjective.
    """

    def __init__(
        self,
        estimator: ACVEstimator[Array],
        autodiff_bkd: AutodiffBackend[TargetArray],
        optimizer: Optional[BindableOptimizerProtocol[TargetArray]] = None,
        objective: Optional[ACVObjective[TargetArray]] = None,
    ) -> None:
        if not isinstance(autodiff_bkd, AutodiffBackend):
            raise TypeError(
                "autodiff_bkd must satisfy AutodiffBackend, got "
                f"{type(autodiff_bkd).__name__}"
            )
        self._est = estimator
        self._autodiff_bkd = autodiff_bkd
        self._optimizer = optimizer
        self._objective = objective

    def allocate(self, target_cost: float) -> ACVAllocationResult[Array]:
        """Allocate samples using optimization on the autodiff backend.

        Parameters
        ----------
        target_cost : float
            The total computational budget.

        Returns
        -------
        AllocationResult
            The allocation result on the estimator's backend. Check
            `success` field for status.
        """
        autodiff_result = ACVAllocator(
            self._est.with_backend(self._autodiff_bkd),
            optimizer=self._optimizer,
            objective=self._objective,
        ).allocate(target_cost)
        return _convert_result_to_backend(autodiff_result, self._est.bkd())


@runtime_checkable
class AnalyticallyAllocatable(Protocol[Array_co]):
    """An estimator whose optimal sample allocation has a closed form."""

    def allocate_samples_analytical(
        self, target_cost: float
    ) -> Tuple[Array_co, Array_co]:
        """Return ``(partition_ratios, objective_value)`` for a budget.

        ``partition_ratios`` has shape ``(nmodels - 1,)`` and
        ``objective_value`` shape ``(1,)``.
        """
        ...


class AnalyticalAllocator(Allocator[Array]):
    """Analytical (closed-form) allocator for MFMC/MLMC.

    Uses the estimator's closed-form allocation formulas.

    Parameters
    ----------
    estimator : ACVEstimator
        The estimator to allocate for. Must satisfy
        :class:`AnalyticallyAllocatable`.
    """

    def __init__(self, estimator: ACVEstimator[Array]):
        if not isinstance(estimator, AnalyticallyAllocatable):
            raise TypeError(
                "estimator must satisfy AnalyticallyAllocatable, got "
                f"{type(estimator).__name__}"
            )
        self._est = estimator
        self._bkd: Backend[Array] = estimator._bkd
        self._allocate_samples: Callable[[float], Tuple[Array, Array]] = (
            estimator.allocate_samples_analytical
        )

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
            partition_ratios, objective_value = self._allocate_samples(target_cost)
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
            self._est,
            self._bkd,
            target_cost,
            partition_ratios,
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
            npartition_samples=self._bkd.zeros(
                (npartitions,), dtype=self._bkd.int64_dtype()
            ),
            nsamples_per_model=self._bkd.zeros(
                (nmodels,), dtype=self._bkd.int64_dtype()
            ),
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

    Dispatches by capability: estimators satisfying AnalyticallyAllocatable
    get AnalyticalAllocator, everything else gets ACVAllocator on the
    estimator's own backend (autograd jacobians on an autodiff backend,
    the optimizer's finite differences otherwise). Pass a
    :class:`ViaAutodiffAllocatorFactory` as ``allocator_factory`` to
    optimize an estimator on an autodiff backend instead.

    Parameters
    ----------
    estimator : ACVEstimator
        The estimator to create an allocator for.
    optimizer : optional
        Optimizer to use for the optimization-based allocator, on the
        estimator's backend. If None, uses the default chained optimizer.
        Ignored when an analytical allocator is selected.

    Returns
    -------
    Allocator
        The most efficient allocator for the given estimator type.
    """
    if isinstance(estimator, AnalyticallyAllocatable):
        return AnalyticalAllocator(estimator)

    return ACVAllocator(estimator, optimizer=optimizer)


class ViaAutodiffAllocatorFactory(Generic[TargetArray]):
    """Allocator factory that optimizes on an autodiff backend.

    Called with an estimator, like :func:`default_allocator_factory`:
    estimators satisfying AnalyticallyAllocatable get AnalyticalAllocator;
    everything else gets :class:`ACVAllocatorViaAutodiff`, which optimizes
    a copy of the estimator on ``autodiff_bkd`` with autograd jacobians
    whatever the estimator's backend. Pass it as ``allocator_factory`` to
    a search.

    Parameters
    ----------
    autodiff_bkd : AutodiffBackend
        The backend the optimization runs on.
    optimizer : optional
        Optimizer on ``autodiff_bkd``'s arrays. If None, uses the default
        chained optimizer. Ignored when an analytical allocator is
        selected.
    """

    def __init__(
        self,
        autodiff_bkd: AutodiffBackend[TargetArray],
        optimizer: Optional[BindableOptimizerProtocol[TargetArray]] = None,
    ) -> None:
        if not isinstance(autodiff_bkd, AutodiffBackend):
            raise TypeError(
                "autodiff_bkd must satisfy AutodiffBackend, got "
                f"{type(autodiff_bkd).__name__}"
            )
        self._autodiff_bkd = autodiff_bkd
        self._optimizer = optimizer

    def __call__(self, estimator: ACVEstimator[Array]) -> Allocator[Array]:
        """Create the allocator for ``estimator``, on any backend."""
        if isinstance(estimator, AnalyticallyAllocatable):
            return AnalyticalAllocator(estimator)

        return ACVAllocatorViaAutodiff(
            estimator, self._autodiff_bkd, optimizer=self._optimizer
        )
