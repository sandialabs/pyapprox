"""Tolerance-driven sample allocation for GroupACV estimators.

:class:`~pyapprox.statest.groupacv.allocation.GroupACVAllocationOptimizer`
minimizes an estimator criterion subject to a budget. This module solves
the dual problem, minimizing cost subject to an accuracy requirement, for
a caller who knows the accuracy they need rather than the budget they
have.

The two directions are separate classes rather than two methods on one.
They put different objects in the objective and constraint roles -- cost
becomes the objective and the criterion becomes the constraint -- so a
single class would carry four role-slots of which any one call uses two,
and its ``_objective`` attribute would mean the minimized quantity in one
method and the constrained quantity in the other.
"""

from typing import TYPE_CHECKING, Generic, Optional

from pyapprox.statest.groupacv._allocation_common import (
    raw_bounds,
    solve_in_variable_space,
)
from pyapprox.statest.groupacv.optimization import (
    GroupACVCostObjective,
    GroupACVLogDetObjective,
    GroupACVObjective,
    GroupACVToleranceConstraint,
)
from pyapprox.statest.groupacv.result import GroupACVToleranceResult
from pyapprox.statest.groupacv.variable_space import AllocationProblemConfig
from pyapprox.util.backends.protocols import Array

if TYPE_CHECKING:
    from pyapprox.optimization.minimize.protocols import (
        BindableOptimizerProtocol,
    )
    from pyapprox.statest.groupacv.base import BaseGroupACVEstimator


# Bound on the search for a feasible reference allocation. The search
# scales a uniform allocation geometrically, so the cap admits
# allocations far beyond any affordable budget before declaring a
# tolerance unreachable.
_MAX_REFERENCE_DOUBLINGS = 60

# Slack allowed when judging whether a relaxed solution met the
# requirement. The optimum lies on the constraint boundary, so a
# converged solver lands within rounding of it -- measured at ~1e-13 --
# while a solver that stopped without restoring feasibility misses by
# ~1e-1. Anything between those leaves the result reported as a
# failure, which is the safe direction.
_FEASIBILITY_RTOL = 1e-8


def default_tolerance_optimizer() -> "BindableOptimizerProtocol[Array]":
    """Create the default optimizer for tolerance-driven allocation.

    Sequential least squares, the same local solver the budget-driven
    path uses, and without that path's global phase: cost is linear in
    the sample counts, so there are no local minima for a global search
    to escape, and the feasible reference allocation already starts the
    solve inside the constraint.

    ``ftol`` sits well below the scipy default deliberately. It is an
    absolute tolerance on the objective, and on the budget-driven path
    the objective is an estimator variance small enough that the
    default exceeds its scale, so the solver reports convergence
    without having moved. The value is kept the same here.

    Pairs with the log-space variables the allocator configures by
    default. Measured on a nested estimator: this solver reaches the
    optimum under log scaling and stalls at the reference allocation
    without it, while a trust region does the reverse. Neither
    dominates, which is why both the solver and the variable scaling
    are caller-replaceable.
    """
    from pyapprox.optimization.minimize.scipy.slsqp import (
        ScipySLSQPOptimizer,
    )

    optimizer: ScipySLSQPOptimizer[Array] = ScipySLSQPOptimizer(
        maxiter=1000, ftol=1e-10
    )
    return optimizer


class GroupACVToleranceAllocator(Generic[Array]):
    """Find the cheapest GroupACV allocation meeting an accuracy requirement.

    The objective is always the total cost, so it is built internally
    from the estimator's cost model and is not a constructor argument.
    Only the optimizer, the problem configuration, and the accuracy
    requirement itself are supplied by the caller: an injected objective
    could silently stop the allocator minimizing cost while its name
    still claimed otherwise.
    """

    def __init__(
        self,
        estimator: "BaseGroupACVEstimator[Array]",
        criterion: Optional[GroupACVObjective[Array]] = None,
        optimizer: Optional["BindableOptimizerProtocol[Array]"] = None,
        problem_config: Optional[AllocationProblemConfig] = None,
    ) -> None:
        self._est = estimator
        self._bkd = estimator._bkd
        if criterion is None:
            criterion = GroupACVLogDetObjective(self._bkd)
        criterion.set_estimator(estimator)
        self._criterion = criterion
        self._cost_objective: GroupACVCostObjective[Array] = (
            GroupACVCostObjective(self._bkd)
        )
        self._cost_objective.set_estimator(estimator)
        self._constraint: GroupACVToleranceConstraint[Array] = (
            GroupACVToleranceConstraint(criterion, self._bkd)
        )
        self._constraint.set_estimator(estimator)
        if optimizer is None:
            optimizer = default_tolerance_optimizer()
        self._optimizer = optimizer
        if problem_config is None:
            # Log-space variables, matching the budget-driven recipe.
            # Model costs span orders of magnitude, so in raw sample
            # counts the gradient with respect to the cheap partitions
            # is far smaller than the one with respect to the
            # high-fidelity partition, and a solver stops early on the
            # directions that barely register. Measured on a nested
            # estimator: raw counts stop at the reference allocation,
            # log-space reaches the optimum.
            problem_config = AllocationProblemConfig(variable_scaling="log")
        self._config = problem_config

    def criterion(self) -> GroupACVObjective[Array]:
        """Return the criterion the tolerance is applied to."""
        return self._criterion

    def _criterion_value(self, npartition_samples_1d: Array) -> float:
        """Criterion value at a 1D allocation."""
        return self._bkd.to_float(
            self._criterion(npartition_samples_1d[:, None])
        )

    def _uniform_allocation(self, multiplier: float) -> Array:
        """Uniform allocation with ``multiplier`` samples per partition."""
        return self._bkd.full((self._est.npartitions(),), multiplier)

    def _feasible_reference(
        self, tolerance: float, min_nhf_samples: int
    ) -> Array:
        """Return a uniform allocation meeting the requirement.

        Scaling a uniform allocation up decreases the estimator
        covariance in every partition, so growing the multiplier
        geometrically reaches the requirement whenever any allocation
        does. The result bounds the optimum from above -- it is
        feasible, so the cheapest feasible allocation costs no more --
        and is a feasible starting point for the solve.
        """
        multiplier = float(
            max(min_nhf_samples, self._config.resolve_bounds_lb(self._est._stat))
        )
        multiplier = max(multiplier, 1.0)
        for _ in range(_MAX_REFERENCE_DOUBLINGS):
            allocation = self._uniform_allocation(multiplier)
            if self._criterion_value(allocation) <= tolerance:
                return allocation
            multiplier *= 2.0
        raise ValueError(
            f"tolerance {tolerance} is not achievable: the criterion is "
            f"still unmet at {multiplier:g} samples in every partition. The "
            "estimator covariance has a nonzero infimum for this model set, "
            "so no budget attains this tolerance; loosen it or add models."
        )

    def _round_up_to_tolerance(
        self, relaxed: Array, tolerance: float, min_nhf_samples: int
    ) -> Array:
        """Round up to integer sample counts.

        Rounds up rather than down. The relaxed solution sits on the
        accuracy boundary, so discarding fractional parts would land
        just inside the infeasible side -- the opposite of the
        budget-driven path, which rounds down to stay under budget.

        Rounding up is the whole of the step. A relaxed solution that
        genuinely sits on the boundary is feasible once its counts are
        raised to the next integer, so if the result still misses the
        tolerance the relaxed point was infeasible by more than a
        fractional sample -- the optimizer stopped short of the
        constraint. The caller checks for that and reports it, rather
        than adding samples until the requirement is met: repair would
        return a feasible but needlessly expensive allocation while
        reporting success, hiding the optimizer failure behind a
        silently suboptimal answer when cost is the quantity the caller
        asked to minimize.
        """
        bkd = self._bkd
        bounds_lb = self._config.resolve_bounds_lb(self._est._stat)
        npartition_samples = bkd.ceil(relaxed - 1e-10)
        return bkd.maximum(
            npartition_samples, bkd.full(relaxed.shape, float(bounds_lb))
        )

    def _build_result(
        self,
        npartition_samples: Array,
        relaxed: Optional[Array],
        success: bool,
        message: str,
        round_nsamples: bool,
    ) -> GroupACVToleranceResult[Array]:
        """Assemble a result from a final allocation."""
        bkd = self._bkd
        nsamples_per_model = self._est._compute_nsamples_per_model(
            npartition_samples
        )
        total_cost = bkd.to_float(
            self._est._estimator_cost(npartition_samples)
        )
        constraint_value = bkd.flatten(
            self._criterion(npartition_samples[:, None])
        )
        if round_nsamples:
            npartition_samples = bkd.asarray(
                npartition_samples, dtype=bkd.int64_dtype()
            )
            nsamples_per_model = bkd.asarray(
                nsamples_per_model, dtype=bkd.int64_dtype()
            )
        return GroupACVToleranceResult(
            npartition_samples=npartition_samples,
            nsamples_per_model=nsamples_per_model,
            total_cost=total_cost,
            constraint_value=constraint_value,
            success=success,
            message=message,
            relaxed_npartition_samples=relaxed,
        )

    def allocate_for_tolerance(
        self,
        tolerance: float,
        min_nhf_samples: int = 1,
        init_guess: Optional[Array] = None,
        round_nsamples: bool = True,
        max_cost: Optional[float] = None,
    ) -> GroupACVToleranceResult[Array]:
        """Find the cheapest allocation whose criterion is at or below
        ``tolerance``.

        Parameters
        ----------
        tolerance : float
            Largest acceptable criterion value, in the units of this
            allocator's criterion. With the default log-determinant
            criterion the value is on a log scale and is normally
            negative.
        min_nhf_samples : int, optional
            Minimum high-fidelity samples. Default is 1.
        init_guess : Array, optional
            Initial guess, shape (npartitions, 1). Defaults to a
            feasible uniform allocation found by scaling.
        round_nsamples : bool, optional
            Whether to round the result to integers. Default is True.
        max_cost : float, optional
            Cost ceiling for the search. Supplying it skips the search
            for a feasible reference allocation.

        Returns
        -------
        GroupACVToleranceResult
            The allocation, its cost, and its achieved criterion value.

        Raises
        ------
        ValueError
            If no allocation attains the tolerance.
        """
        bkd = self._bkd
        min_nhf = max(self._est._stat.min_nsamples(), min_nhf_samples)
        self._constraint.set_tolerance(tolerance, min_nhf)

        reference = self._feasible_reference(tolerance, min_nhf)
        if max_cost is None:
            cost_ceiling = 2.0 * bkd.to_float(
                self._est._estimator_cost(reference)
            )
        else:
            cost_ceiling = max_cost

        floor_allocation = self._uniform_allocation(
            max(float(min_nhf), self._config.resolve_bounds_lb(self._est._stat))
        )
        if self._criterion_value(floor_allocation) <= tolerance:
            return self._build_result(
                floor_allocation,
                floor_allocation,
                True,
                "tolerance met at the minimum feasible allocation",
                round_nsamples,
            )

        n_bounds = raw_bounds(self._est, self._config, cost_ceiling)
        if init_guess is None:
            init_guess = reference[:, None]
        solution = solve_in_variable_space(
            self._cost_objective,
            self._constraint,
            n_bounds,
            init_guess,
            self._optimizer,
            self._config,
            self._est,
        )
        if not solution.succeeded():
            # Unlike the budget-driven path, the fallback allocation is
            # known to meet the tolerance, so its criterion is a real
            # number rather than an infinity.
            return self._build_result(
                reference,
                reference,
                False,
                solution.message(),
                round_nsamples,
            )

        relaxed = solution.npartition_samples()
        # An optimizer reporting convergence has not necessarily reached
        # a feasible point: a solver whose step computation breaks down
        # can stop while the requirement is still violated and report
        # that as its answer. The guarantee is that a successful result
        # meets the tolerance, so it is checked here rather than assumed
        # from the solver's own status. The comparison carries a
        # relative slack because the solution sits *on* the boundary,
        # where a converged answer lands within rounding of it; without
        # that, an exact test rejects good solutions over a difference
        # of order 1e-13. Genuine breakdowns miss by many orders more.
        if self._criterion_value(relaxed) > tolerance + _FEASIBILITY_RTOL * max(
            abs(tolerance), 1.0
        ):
            return self._build_result(
                reference,
                reference,
                False,
                "optimizer stopped at an allocation that misses the "
                "tolerance",
                round_nsamples,
            )
        if not round_nsamples:
            return self._build_result(relaxed, relaxed, True, "", False)

        rounded = self._round_up_to_tolerance(relaxed, tolerance, min_nhf)
        if self._criterion_value(rounded) > tolerance:
            return self._build_result(
                rounded,
                relaxed,
                False,
                "tolerance not met after rounding up to integer sample "
                "counts: the relaxed solution missed it by more than a "
                "fractional sample, so the optimizer stopped short of "
                "the accuracy constraint",
                True,
            )
        return self._build_result(rounded, relaxed, True, "", True)
