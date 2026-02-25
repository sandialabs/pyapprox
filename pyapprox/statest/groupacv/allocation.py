"""Sample allocation optimization for GroupACV estimators."""

from dataclasses import dataclass
from typing import Generic, Optional, TYPE_CHECKING

from pyapprox.util.backends.protocols import Array

if TYPE_CHECKING:
    from pyapprox.statest.groupacv.base import GroupACVEstimator
    from pyapprox.statest.groupacv.optimization import (
        GroupACVObjective,
        GroupACVCostConstraint,
    )
    from pyapprox.optimization.minimize.protocols import (
        BindableOptimizerProtocol,
    )
    from pyapprox.optimization.minimize.chained.chained_optimizer import (
        ChainedOptimizer,
    )


def default_groupacv_optimizer() -> "ChainedOptimizer":
    """Create the default optimizer for GroupACV sample allocation.

    Returns
    -------
    ChainedOptimizer
        A chained optimizer with differential evolution followed by
        trust-constr refinement.
    """
    from pyapprox.optimization.minimize.chained.chained_optimizer import (
        ChainedOptimizer,
    )
    from pyapprox.optimization.minimize.scipy.trust_constr import (
        ScipyTrustConstrOptimizer,
    )
    from pyapprox.optimization.minimize.scipy.diffevol import (
        ScipyDifferentialEvolutionOptimizer,
    )

    global_opt = ScipyDifferentialEvolutionOptimizer(
        maxiter=100,
        polish=False,
        seed=1,
        tol=1e-8,
        raise_on_failure=False,
    )
    local_opt = ScipyTrustConstrOptimizer(
        gtol=1e-8,
        maxiter=1000,
    )
    return ChainedOptimizer(global_opt, local_opt)


@dataclass
class GroupACVAllocationResult(Generic[Array]):
    """Allocation result for GroupACV estimators.

    Attributes
    ----------
    npartition_samples : Array
        Partition sample counts. Shape (npartitions,).
    nsamples_per_model : Array
        Sample counts per model. Shape (nmodels,).
    actual_cost : float
        Actual computational cost.
    objective_value : Array
        Objective value. Shape (1,).
    success : bool
        Whether allocation succeeded.
    message : str
        Status message.
    """

    npartition_samples: Array
    nsamples_per_model: Array
    actual_cost: float
    objective_value: Array  # Shape (1,) - keeps autograd graph
    success: bool
    message: str = ""


class GroupACVAllocationOptimizer(Generic[Array]):
    """Optimizer for GroupACV sample allocation.

    Separates allocation optimization from estimation, allowing:
    - Custom optimizer injection
    - Custom objective functions
    - Independent testing of optimization logic

    Parameters
    ----------
    estimator : GroupACVEstimator
        The estimator to optimize allocation for.
    optimizer : BindableOptimizerProtocol, optional
        Optimizer to use. If None, uses default chained optimizer
        (differential evolution + trust-constr).
    objective : GroupACVObjective, optional
        Objective function. If None, uses estimator's default_objective().
    constraint : GroupACVCostConstraint, optional
        Constraint function. If None, creates default cost constraint.

    Examples
    --------
    >>> # With default optimizer
    >>> allocator = GroupACVAllocationOptimizer(est)
    >>> result = allocator.optimize(target_cost=1000, min_nhf_samples=10)
    >>> est.set_allocation(result)

    >>> # With custom optimizer
    >>> from pyapprox.optimization.minimize.scipy.trust_constr import (
    ...     ScipyTrustConstrOptimizer
    ... )
    >>> optimizer = ScipyTrustConstrOptimizer(gtol=1e-8, maxiter=1000)
    >>> allocator = GroupACVAllocationOptimizer(est, optimizer=optimizer)
    >>> result = allocator.optimize(target_cost=1000, min_nhf_samples=10)
    >>> est.set_allocation(result)
    """

    def __init__(
        self,
        estimator: "GroupACVEstimator[Array]",
        optimizer: Optional["BindableOptimizerProtocol[Array]"] = None,
        objective: Optional["GroupACVObjective[Array]"] = None,
        constraint: Optional["GroupACVCostConstraint[Array]"] = None,
    ):
        self._est = estimator
        self._bkd = estimator._bkd

        # Use default optimizer if not provided
        if optimizer is None:
            optimizer = default_groupacv_optimizer()
        self._optimizer = optimizer

        # Use default objective if not provided
        if objective is None:
            objective = estimator.default_objective()
        self._objective = objective
        self._objective.set_estimator(estimator)

        # Use default constraint if not provided
        if constraint is None:
            from pyapprox.statest.groupacv.optimization import (
                GroupACVCostConstraint,
            )

            constraint = GroupACVCostConstraint(self._bkd)
        self._constraint = constraint
        self._constraint.set_estimator(estimator)

    def optimize(
        self,
        target_cost: float,
        min_nhf_samples: int = 1,
        init_guess: Optional[Array] = None,
        round_nsamples: bool = True,
    ) -> GroupACVAllocationResult[Array]:
        """Find optimal sample allocation.

        Parameters
        ----------
        target_cost : float
            Maximum computational budget.
        min_nhf_samples : int, optional
            Minimum high-fidelity samples. Default is 1.
        init_guess : Array, optional
            Initial guess for optimizer. Shape (npartitions, 1).
            If None, uses default initial guess.
        round_nsamples : bool, optional
            Whether to round result to integers. Default is True.

        Returns
        -------
        GroupACVAllocationResult
            Optimization result with npartition_samples.

        Raises
        ------
        RuntimeError
            If optimization fails and success is False.
        """
        # Configure constraint with budget
        min_nhf = max(self._est._stat.min_nsamples(), min_nhf_samples)
        self._constraint.set_budget(target_cost, min_nhf)

        # Set up bounds
        max_npartition_samples = target_cost / float(self._est._costs.min()) + 1
        bounds = self._bkd.array(
            [[0.0, max_npartition_samples]] * self._est.npartitions()
        )

        # Bind optimizer
        self._optimizer.bind(self._objective, bounds, [self._constraint])

        # Get initial guess
        if init_guess is None:
            init_guess = self._est._init_guess(target_cost)

        # Run optimization
        result = self._optimizer.minimize(init_guess)

        if not result.success() or self._bkd.any_bool(result.optima() < 0):
            nsamples_per_model = self._est._compute_nsamples_per_model(
                init_guess[:, 0]
            )
            return GroupACVAllocationResult(
                npartition_samples=init_guess[:, 0],
                nsamples_per_model=nsamples_per_model,
                actual_cost=float(self._est._estimator_cost(init_guess[:, 0])),
                objective_value=self._bkd.array([float("inf")]),
                success=False,
                message=f"Optimization failed: {result.message()}",
            )

        # Extract result (optimizer returns (nvars, 1), we store (nvars,))
        npartition_samples = result.optima()[:, 0]

        # Round if requested
        if round_nsamples:
            npartition_samples = self._bkd.floor(npartition_samples + 1e-4)

        # Compute nsamples_per_model and actual_cost
        nsamples_per_model = self._est._compute_nsamples_per_model(
            npartition_samples
        )
        actual_cost = float(self._est._estimator_cost(npartition_samples))

        # Compute objective at solution
        # Pass as (nvars, 1) as expected by objective
        obj_value = self._objective(npartition_samples[:, None])

        return GroupACVAllocationResult(
            npartition_samples=npartition_samples,
            nsamples_per_model=nsamples_per_model,
            actual_cost=actual_cost,
            objective_value=obj_value.flatten(),  # Shape (1,)
            success=True,
            message="",
        )
