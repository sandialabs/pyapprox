"""Sample allocation optimization for GroupACV estimators."""

from typing import TYPE_CHECKING, Generic, Optional

from pyapprox.statest.groupacv._allocation_common import (
    raw_bounds,
    solve_in_variable_space,
)
from pyapprox.statest.groupacv.optimization import (
    GroupACVCostConstraint,
    GroupACVLogDetObjective,
    GroupACVObjective,
)
from pyapprox.statest.groupacv.result import GroupACVAllocationResult
from pyapprox.statest.groupacv.variable_space import (
    AllocationProblemConfig,
    BudgetConstraintForm,
)
from pyapprox.util.backends.protocols import Array

if TYPE_CHECKING:
    from pyapprox.optimization.minimize.protocols import (
        BindableOptimizerProtocol,
    )
    from pyapprox.statest.groupacv.base import BaseGroupACVEstimator


def default_groupacv_optimizer() -> "BindableOptimizerProtocol[Array]":
    """Create the default optimizer for GroupACV sample allocation.

    Returns
    -------
    BindableOptimizerProtocol
        A chained optimizer with differential evolution followed by
        trust-constr refinement.
    """
    from pyapprox.optimization.minimize.chained.chained_optimizer import (
        ChainedOptimizer,
    )
    from pyapprox.optimization.minimize.scipy.diffevol import (
        ScipyDifferentialEvolutionOptimizer,
    )
    from pyapprox.optimization.minimize.scipy.trust_constr import (
        ScipyTrustConstrOptimizer,
    )

    global_opt: ScipyDifferentialEvolutionOptimizer[Array] = (
        ScipyDifferentialEvolutionOptimizer(
            maxiter=100,
            polish=False,
            seed=1,
            tol=1e-8,
            raise_on_failure=False,
        )
    )
    local_opt: ScipyTrustConstrOptimizer[Array] = ScipyTrustConstrOptimizer(
        gtol=1e-8,
        maxiter=1000,
    )
    return ChainedOptimizer(global_opt, local_opt)


class GroupACVAllocationOptimizer(Generic[Array]):
    """Optimizer for GroupACV sample allocation.

    Separates allocation optimization from estimation, allowing:
    - Custom optimizer injection
    - Custom objective functions
    - Independent testing of optimization logic

    Parameters
    ----------
    estimator : BaseGroupACVEstimator
        The estimator to optimize allocation for.
    optimizer : BindableOptimizerProtocol, optional
        Optimizer to use. If None, uses default chained optimizer
        (differential evolution + trust-constr).
    objective : GroupACVObjective, optional
        Objective function. If None, uses GroupACVLogDetObjective.
    constraint : GroupACVCostConstraint, optional
        Constraint function. If None, creates default cost constraint.

    Examples
    --------
    >>> # With default optimizer
    >>> allocator = GroupACVAllocationOptimizer(est)
    >>> result = allocator.optimize(target_cost=1000, min_nhf_samples=10)
    >>> fitted = FittedGroupACVEstimator(est, result)

    >>> # With custom optimizer
    >>> from pyapprox.optimization.minimize.scipy.trust_constr import (
    ...     ScipyTrustConstrOptimizer
    ... )
    >>> optimizer = ScipyTrustConstrOptimizer(gtol=1e-8, maxiter=1000)
    >>> allocator = GroupACVAllocationOptimizer(est, optimizer=optimizer)
    >>> result = allocator.optimize(target_cost=1000, min_nhf_samples=10)
    >>> fitted = FittedGroupACVEstimator(est, result)
    """

    def __init__(
        self,
        estimator: "BaseGroupACVEstimator[Array]",
        optimizer: Optional["BindableOptimizerProtocol[Array]"] = None,
        objective: Optional["GroupACVObjective[Array]"] = None,
        constraint: Optional["GroupACVCostConstraint[Array]"] = None,
        problem_config: Optional[AllocationProblemConfig] = None,
    ):
        self._est = estimator
        self._bkd = estimator._bkd

        # Use default optimizer if not provided
        if optimizer is None:
            optimizer = default_groupacv_optimizer()
        self._optimizer: "BindableOptimizerProtocol[Array]" = optimizer

        # Use default objective if not provided
        if objective is None:
            objective = GroupACVLogDetObjective(estimator._bkd)
        self._objective: GroupACVObjective[Array] = objective
        self._objective.set_estimator(estimator)

        # Use default constraint if not provided
        if constraint is None:
            constraint = GroupACVCostConstraint(self._bkd)
        self._constraint: GroupACVCostConstraint[Array] = constraint
        self._constraint.set_estimator(estimator)

        # Use default problem config if not provided
        if problem_config is None:
            problem_config = AllocationProblemConfig()
        self._config: AllocationProblemConfig = problem_config

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
            Optimization result with npartition_samples. ``success`` is
            False if the optimizer failed, or if the rounded allocation
            costs more than ``target_cost``; the budget is checked here
            rather than assumed from the optimizer's exit status. The
            check applies only when ``round_nsamples`` is True, since a
            relaxed allocation sits on the budget boundary and lands on
            either side of it by the solver's convergence tolerance.

        Raises
        ------
        RuntimeError
            If optimization fails and success is False.
        """
        bkd = self._bkd

        # 1. Budget setup
        min_nhf = max(self._est._stat.min_nsamples(), min_nhf_samples)
        self._constraint.set_budget(target_cost, min_nhf)

        # 2. Apply budget form (resets lb/ub to correct state)
        budget_form: BudgetConstraintForm[Array] = (
            self._config.build_budget_form()
        )
        budget_form.adjust_bounds(self._constraint)

        # 3. Compute raw n-space bounds from the budget
        n_bounds = raw_bounds(self._est, self._config, target_cost)

        # 4-5. Move into the optimizer's variable space, solve, and
        # transform the answer back to sample counts.
        if init_guess is None:
            init_guess = self._est._init_guess(target_cost)
        solution = solve_in_variable_space(
            self._objective,
            self._constraint,
            n_bounds,
            init_guess,
            self._optimizer,
            self._config,
            self._est,
        )

        # 6. Handle failure
        if not solution.succeeded():
            nsamples_per_model = self._est._compute_nsamples_per_model(
                init_guess[:, 0]
            )
            return GroupACVAllocationResult(
                npartition_samples=init_guess[:, 0],
                nsamples_per_model=nsamples_per_model,
                actual_cost=bkd.to_float(
                    self._est._estimator_cost(init_guess[:, 0])
                ),
                objective_value=bkd.array([float("inf")]),
                success=False,
                message=solution.message(),
            )

        npartition_samples = solution.npartition_samples()
        relaxed_npartition_samples = npartition_samples

        # Round if requested
        if round_nsamples:
            npartition_samples = bkd.asarray(
                bkd.floor(npartition_samples + 1e-4),
                dtype=bkd.int64_dtype(),
            )

        # Compute nsamples_per_model and actual_cost
        nps_float = bkd.asarray(
            npartition_samples, dtype=bkd.double_dtype()
        )
        nsamples_per_model = self._est._compute_nsamples_per_model(nps_float)
        actual_cost = bkd.to_float(self._est._estimator_cost(nps_float))

        # Compute objective at solution (in n-space)
        obj_value = self._objective(nps_float[:, None])

        if round_nsamples:
            nsamples_per_model = bkd.asarray(
                nsamples_per_model, dtype=bkd.int64_dtype()
            )

        # The optimizer is injectable, so the budget is only respected to
        # the extent the supplied solver respects its constraints: one
        # that reports convergence at an infeasible point would otherwise
        # be relayed as a success. Check the guarantee here rather than
        # trusting the solver, mirroring the tolerance-driven allocator.
        #
        # Only the rounded allocation is checked, and it is checked
        # exactly. Integer sample counts make the cost a sum of integer
        # multiples of the model costs, so there is no continuum near the
        # budget to converge into: an allocation is either within budget
        # or over it by at least the cheapest model's cost. That makes
        # any excess a real overspend rather than convergence noise, and
        # removes the need for a tolerance to distinguish the two.
        #
        # A relaxed allocation has no such floor. It sits on the
        # constraint boundary and lands on either side of it by an amount
        # set by the solver's own convergence tolerance -- measured at
        # 3e-8 relative under log scaling, where the returned counts are
        # exp() of the optimizer's variables. Rejecting that would fail
        # legitimate solves over a fraction of one sample.
        if round_nsamples and actual_cost > target_cost:
            return GroupACVAllocationResult(
                npartition_samples=npartition_samples,
                nsamples_per_model=nsamples_per_model,
                actual_cost=actual_cost,
                objective_value=bkd.flatten(obj_value),
                success=False,
                message=(
                    "Rounded allocation exceeds target_cost: {0} > "
                    "{1}".format(actual_cost, target_cost)
                ),
                relaxed_npartition_samples=relaxed_npartition_samples,
            )

        return GroupACVAllocationResult(
            npartition_samples=npartition_samples,
            nsamples_per_model=nsamples_per_model,
            actual_cost=actual_cost,
            objective_value=bkd.flatten(obj_value),
            success=True,
            message="",
            relaxed_npartition_samples=relaxed_npartition_samples,
        )
