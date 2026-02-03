"""Sample allocation optimization for GroupACV estimators."""

from dataclasses import dataclass
from typing import Generic, Optional, TYPE_CHECKING

from pyapprox.typing.util.backends.protocols import Array

if TYPE_CHECKING:
    from pyapprox.typing.statest.groupacv.base import GroupACVEstimator
    from pyapprox.typing.statest.groupacv.optimization import (
        GroupACVObjective,
        GroupACVCostConstraint,
    )
    from pyapprox.typing.optimization.minimize.protocols import (
        BindableOptimizerProtocol,
    )
    from pyapprox.typing.optimization.minimize.chained.chained_optimizer import (
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
    from pyapprox.typing.optimization.minimize.chained.chained_optimizer import (
        ChainedOptimizer,
    )
    from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
        ScipyTrustConstrOptimizer,
    )
    from pyapprox.typing.optimization.minimize.scipy.diffevol import (
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
class AllocationResult(Generic[Array]):
    """Result of sample allocation optimization.

    Attributes
    ----------
    npartition_samples : Array
        Optimal partition sample counts. Shape (npartitions,).
    objective_value : Array
        Objective value at solution. Shape (1,). Kept as Array for autograd.
    success : bool
        Whether optimization succeeded.
    message : str
        Optional message from optimizer.
    """

    npartition_samples: Array
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
    >>> est.set_npartition_samples(result.npartition_samples)

    >>> # With custom optimizer
    >>> from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
    ...     ScipyTrustConstrOptimizer
    ... )
    >>> optimizer = ScipyTrustConstrOptimizer(gtol=1e-8, maxiter=1000)
    >>> allocator = GroupACVAllocationOptimizer(est, optimizer=optimizer)
    >>> result = allocator.optimize(target_cost=1000, min_nhf_samples=10)
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
            from pyapprox.typing.statest.groupacv.optimization import (
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
    ) -> AllocationResult[Array]:
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
        AllocationResult
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
            return AllocationResult(
                npartition_samples=init_guess[:, 0],
                objective_value=self._bkd.array([float("inf")]),
                success=False,
                message=f"Optimization failed: {result.message()}",
            )

        # Extract result (optimizer returns (nvars, 1), we store (nvars,))
        npartition_samples = result.optima()[:, 0]

        # Round if requested
        if round_nsamples:
            npartition_samples = self._bkd.floor(npartition_samples + 1e-4)

        # Compute objective at solution
        # Pass as (nvars, 1) as expected by objective
        obj_value = self._objective(npartition_samples[:, None])

        return AllocationResult(
            npartition_samples=npartition_samples,
            objective_value=obj_value.flatten(),  # Shape (1,)
            success=True,
            message="",
        )
