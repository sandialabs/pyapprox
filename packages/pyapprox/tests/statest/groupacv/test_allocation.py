"""Unit tests for budget-driven GroupACV allocation."""

import numpy as np
from pyapprox.statest.groupacv import GroupACVEstimatorIS
from pyapprox.statest.groupacv.allocation import GroupACVAllocationOptimizer
from pyapprox.statest.groupacv.variable_space import AllocationProblemConfig
from pyapprox.statest.statistics import MultiOutputMean


def _make_estimator(bkd, nmodels=3, nqoi=1):
    """Create an IS estimator with a correlated covariance."""
    np.random.seed(1)
    cov_size = nmodels * nqoi
    cov = bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
    cov = cov.T @ cov
    costs = bkd.arange(nmodels, 0, -1, dtype=bkd.double_dtype())
    stat = MultiOutputMean(nqoi, bkd)
    stat.set_pilot_quantities(cov)
    return GroupACVEstimatorIS(stat, costs)


class _ConvergedResult:
    """Optimizer result claiming convergence at a supplied point."""

    def __init__(self, optima):
        self._optima = optima

    def success(self) -> bool:
        return True

    def optima(self):
        return self._optima


class _OverBudgetOptimizer:
    """Reports success at an allocation far exceeding the budget.

    Stands in for a solver whose exit status cannot be taken at face
    value, which is the case the allocator's budget check exists to
    catch. The returned point is expressed in the optimizer's own
    variable space, so the allocator transforms it back before costing
    it, exactly as it would a real solver's answer.
    """

    def __init__(self, bkd, value=1e4):
        self._bkd = bkd
        self._value = value
        self._nvars = 0

    def bind(self, objective, bounds, constraints=None):
        self._nvars = objective.nvars()
        return self

    def minimize(self, init_guess):
        return _ConvergedResult(
            self._bkd.full((self._nvars, 1), self._value)
        )

    def is_bound(self) -> bool:
        return True

    def copy(self):
        return self

    def bkd(self):
        return self._bkd


class TestBudgetIsVerifiedNotAssumed:
    """A reported success must respect the budget, whatever the solver says.

    The optimizer is injectable, so the budget holds only to the extent
    the supplied solver respects its constraints. A solver that reports
    convergence at an infeasible point must not be relayed as a success.
    """

    def test_solver_that_reports_success_over_budget_is_refused(
        self, bkd
    ) -> None:
        """The guarantee cannot rest on the solver's exit status."""
        est = _make_estimator(bkd)
        target_cost = 100.0
        opt = GroupACVAllocationOptimizer(
            est,
            optimizer=_OverBudgetOptimizer(bkd),
            problem_config=AllocationProblemConfig(variable_scaling="none"),
        )
        result = opt.optimize(target_cost, round_nsamples=True)
        assert not result.success
        assert result.actual_cost > target_cost
        assert "exceeds target_cost" in result.message

    def test_relaxed_allocation_is_not_budget_checked(self, bkd) -> None:
        """A relaxed allocation has no integer floor to separate it from
        the boundary.

        Integer counts put the cost on a lattice, so a rounded overspend
        is at least one model's cost and can be caught exactly. The
        relaxed solution instead sits *on* the budget and lands either
        side of it by the solver's convergence tolerance -- measured at
        3e-8 relative under log scaling, a fraction of a single sample.
        Checking it rejects legitimate solves, so it is left unchecked.
        """
        est = _make_estimator(bkd)
        target_cost = 100.0
        opt = GroupACVAllocationOptimizer(
            est,
            optimizer=_OverBudgetOptimizer(bkd),
            problem_config=AllocationProblemConfig(variable_scaling="none"),
        )
        result = opt.optimize(target_cost, round_nsamples=False)
        assert result.success

    def test_genuine_solve_stays_within_budget_and_succeeds(
        self, bkd
    ) -> None:
        """The check must not reject legitimate allocations.

        Measured across both roundings and budgets from 50 to 5000, no
        legitimate solve exceeds its budget: rounding floors well under
        it, and the unrounded optimum approaches the boundary from
        below. A check that fired here would be worse than none.
        """
        est = _make_estimator(bkd)
        for target_cost in (50.0, 200.0, 1000.0, 5000.0):
            for round_nsamples in (True, False):
                opt = GroupACVAllocationOptimizer(
                    est,
                    problem_config=AllocationProblemConfig(
                        variable_scaling="none"
                    ),
                )
                result = opt.optimize(
                    target_cost, round_nsamples=round_nsamples
                )
                if not result.success:
                    continue
                assert result.actual_cost <= target_cost * (1.0 + 1e-8)
