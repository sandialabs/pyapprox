"""Unit tests for tolerance-driven GroupACV allocation."""

import numpy as np
import pytest

from pyapprox.interface.functions.autograd import WithAutogradJacobian
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.optimization.minimize.objective.validation import (
    validate_objective,
)
from pyapprox.optimization.minimize.scipy.slsqp import ScipySLSQPOptimizer
from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.statest.cv_estimator import CVEstimator
from pyapprox.statest.groupacv import GroupACVEstimatorIS
from pyapprox.statest.groupacv.allocation import GroupACVAllocationOptimizer
from pyapprox.statest.groupacv.mlblue import MLBLUEEstimator
from pyapprox.statest.groupacv.optimization import (
    GroupACVCostConstraint,
    GroupACVCostObjective,
    GroupACVLogDetObjective,
    GroupACVRequirementConstraint,
)
from pyapprox.statest.groupacv.tolerance_allocation import (
    GroupACVToleranceAllocator,
    default_tolerance_optimizer,
)
from pyapprox.statest.groupacv.variable_space import AllocationProblemConfig
from pyapprox.statest.groupacv.variants import GroupACVEstimatorNested
from pyapprox.statest.known import KnownMean
from pyapprox.statest.statistics import MultiOutputMean
from pyapprox.statest.tolerance import (
    CVToleranceAllocator,
    LogDeterminantConstraint,
    MaxMarginalStandardErrorConstraint,
    TraceConstraint,
)


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


def _make_nested_estimator(bkd, nmodels=3, nqoi=1):
    """Nested estimator, whose criteria have no analytical derivatives.

    The capability check requires an identity allocation matrix, which
    only independent sampling produces.
    """
    np.random.seed(1)
    cov_size = nmodels * nqoi
    cov = bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
    cov = cov.T @ cov
    costs = bkd.arange(nmodels, 0, -1, dtype=bkd.double_dtype())
    stat = MultiOutputMean(nqoi, bkd)
    stat.set_pilot_quantities(cov)
    return GroupACVEstimatorNested(stat, costs)


def _make_all_known_mlblue(bkd, nmodels=3, nqoi=1, nsamples=5000):
    """MLBLUE with every low-fidelity mean known, plus the matching CV.

    With all statistics known, concentrating the whole allocation on the
    all-models subset reproduces the control-variate estimator exactly,
    which makes the two families comparable on the same problem.
    """
    np.random.seed(0)
    pilot = [np.random.normal(0, 1, (nqoi, nsamples))]
    for k in range(1, nmodels):
        pilot.append(
            pilot[0] * (0.9**k)
            + 0.1 * np.random.normal(0, 1, (nqoi, nsamples))
        )
    stat = MultiOutputMean(nqoi, bkd)
    stat.set_pilot_quantities(
        *stat.compute_pilot_quantities([bkd.array(p) for p in pilot])
    )
    costs = bkd.array([4.0, 2.0, 1.0][:nmodels])
    known = [
        KnownMean(m, bkd.array(np.mean(pilot[m], axis=1)))
        for m in range(1, nmodels)
    ]
    mlblue = MLBLUEEstimator(stat, costs, known_quantities=known)
    return mlblue, CVEstimator(stat, costs)


class TestGroupACVCostObjective:
    """The cost objective, dual of the cost constraint's budget row."""

    def test_satisfies_objective_protocol(self, bkd) -> None:
        """nqoi must be 1; the optimizer rejects anything else."""
        est = _make_estimator(bkd)
        obj = GroupACVCostObjective(bkd)
        obj.set_estimator(est)
        validate_objective(obj)
        bkd.assert_allclose(
            bkd.asarray([obj.nqoi(), obj.nvars()]),
            bkd.asarray([1, est.npartitions()]),
        )

    def test_value_is_estimator_cost(self, bkd) -> None:
        est = _make_estimator(bkd)
        obj = GroupACVCostObjective(bkd)
        obj.set_estimator(est)
        iterate = est._init_guess(100.0)
        bkd.assert_allclose(
            obj(iterate),
            bkd.atleast_2d(est._estimator_cost(iterate[:, 0])),
            rtol=1e-12,
        )

    def test_sign_is_opposite_the_cost_constraint(self, bkd) -> None:
        """The constraint reports budget minus cost; the objective cost."""
        est = _make_estimator(bkd)
        target_cost = 100.0
        obj = GroupACVCostObjective(bkd)
        obj.set_estimator(est)
        con = GroupACVCostConstraint(bkd)
        con.set_estimator(est)
        con.set_budget(target_cost, 1)
        iterate = est._init_guess(target_cost)
        bkd.assert_allclose(
            obj(iterate)[0, 0],
            target_cost - con(iterate)[0, 0],
            rtol=1e-12,
        )
        bkd.assert_allclose(
            obj.derivatives().jacobian(iterate),
            -con.jacobian(iterate)[0:1, :],
            rtol=1e-12,
        )

    def test_bundle_is_second_order(self, bkd) -> None:
        """hessian as well as hvp: the rescaling wrappers key on hessian."""
        est = _make_estimator(bkd)
        obj = GroupACVCostObjective(bkd)
        obj.set_estimator(est)
        derivs = obj.derivatives()
        assert derivs.jacobian is not None
        assert derivs.hessian is not None
        assert derivs.hvp is not None

    def test_curvature_is_zero(self, bkd) -> None:
        """Cost is linear in the sample counts."""
        est = _make_estimator(bkd)
        obj = GroupACVCostObjective(bkd)
        obj.set_estimator(est)
        iterate = est._init_guess(100.0)
        nvars = obj.nvars()
        bkd.assert_allclose(
            obj.derivatives().hessian(iterate),
            bkd.zeros((nvars, nvars)),
            atol=1e-14,
        )

    def test_requires_estimator(self, bkd) -> None:
        obj = GroupACVCostObjective(bkd)
        with pytest.raises(RuntimeError, match="set_estimator"):
            obj.nvars()




class TestGroupACVToleranceAllocator:
    """Tolerance-driven allocation."""

    def test_default_suits_a_curved_constraint(self, bkd) -> None:
        """A trust region, in the forward path's log-space variables.

        The variables are shared with the budget-driven path, since the
        two directions share a feasible set. The solver is not: here the
        accuracy constraint is the curved part, which a trust region
        using its Hessian handles and sequential least squares, which
        linearizes it, does not reliably.
        """
        assert isinstance(
            default_tolerance_optimizer(), ScipyTrustConstrOptimizer
        )
        alloc = GroupACVToleranceAllocator(_make_estimator(bkd))
        assert alloc._config.variable_scaling == "log"

    def test_optimizer_is_swappable(self, bkd) -> None:
        """The solver is injected, so a different one still applies."""
        est = _make_estimator(bkd)
        tolerance = -2.0
        requirement = LogDeterminantConstraint(tolerance, bkd)
        default = GroupACVToleranceAllocator(est).allocate_for_tolerance(
            requirement, round_nsamples=False
        )
        swapped = GroupACVToleranceAllocator(
            est, optimizer=ScipySLSQPOptimizer(maxiter=2000, ftol=1e-10)
        ).allocate_for_tolerance(requirement, round_nsamples=False)
        assert swapped.success
        assert float(swapped.constraint_value[0]) <= tolerance + 1e-6
        assert swapped.total_cost <= default.total_cost * (1 + 1e-2)

    def test_requirement_is_satisfied(self, bkd) -> None:
        est = _make_estimator(bkd)
        alloc = GroupACVToleranceAllocator(est)
        tolerance = -2.0
        result = alloc.allocate_for_tolerance(
            LogDeterminantConstraint(tolerance, bkd)
        )
        assert result.success
        assert float(result.constraint_value[0]) <= tolerance + 1e-9

    def test_rounding_up_is_what_preserves_the_requirement(
        self, bkd
    ) -> None:
        """Flooring the relaxed solution would violate the tolerance.

        The relaxed optimum sits on the boundary, so discarding
        fractional parts moves the requirement the wrong way. This is the
        opposite of the budget-driven path, which floors to stay under
        budget.
        """
        est = _make_estimator(bkd)
        alloc = GroupACVToleranceAllocator(est)
        tolerance = -2.0
        requirement = LogDeterminantConstraint(tolerance, bkd)
        relaxed = alloc.allocate_for_tolerance(
            requirement, round_nsamples=False
        )
        floored = bkd.floor(relaxed.npartition_samples)
        floored_value = bkd.to_float(
            requirement.value(est._covariance_from_npartition_samples(floored))
        )
        rounded = alloc.allocate_for_tolerance(requirement)
        assert float(rounded.constraint_value[0]) <= tolerance + 1e-9
        assert floored_value > tolerance

    def test_requirement_met_at_the_sample_floor(self, bkd) -> None:
        """A loose tolerance returns the cheapest feasible allocation."""
        est = _make_estimator(bkd)
        alloc = GroupACVToleranceAllocator(est)
        result = alloc.allocate_for_tolerance(
            LogDeterminantConstraint(1e6, bkd)
        )
        assert result.success
        assert "minimum feasible allocation" in result.message

    def test_unreachable_tolerance_raises(self, bkd) -> None:
        est = _make_estimator(bkd)
        alloc = GroupACVToleranceAllocator(est)
        with pytest.raises(ValueError, match="not achievable"):
            alloc.allocate_for_tolerance(LogDeterminantConstraint(-1e9, bkd))

    def test_max_cost_skips_the_reference_search(self, bkd) -> None:
        """Supplying a ceiling avoids probing for a feasible reference."""
        est = _make_estimator(bkd)
        alloc = GroupACVToleranceAllocator(est)
        requirement = LogDeterminantConstraint(-2.0, bkd)
        unbounded = alloc.allocate_for_tolerance(requirement)
        bounded = alloc.allocate_for_tolerance(
            requirement, max_cost=10.0 * unbounded.total_cost
        )
        assert bounded.success
        assert float(bounded.constraint_value[0]) <= -2.0 + 1e-9

    def test_result_reports_cost_and_achieved_accuracy(self, bkd) -> None:
        """The minimized quantity and the guarantee are separate fields."""
        est = _make_estimator(bkd)
        alloc = GroupACVToleranceAllocator(est)
        result = alloc.allocate_for_tolerance(LogDeterminantConstraint(-2.0, bkd))
        bkd.assert_allclose(
            bkd.asarray([result.total_cost]),
            bkd.asarray(
                [
                    bkd.to_float(
                        est._estimator_cost(
                            bkd.asarray(
                                result.npartition_samples,
                                dtype=bkd.double_dtype(),
                            )
                        )
                    )
                ]
            ),
            rtol=1e-10,
        )
        assert result.constraint_value.shape == (1,)


class TestAutogradComposition:
    """Derivatives absent from the covariance are composed, not skipped.

    Analytical derivatives require the statistic to supply sigma-block
    derivatives and the estimator to be independent-sample, so a nested
    estimator has none. In the budget-driven direction only the
    objective can lack them; here the requirement sits in the constraint
    role, so the constraint can lack them too.
    """

    def test_nested_criterion_has_no_analytical_jacobian(
        self, torch_bkd
    ) -> None:
        """The precondition, so the test below is not vacuous."""
        est = _make_nested_estimator(torch_bkd)
        criterion = GroupACVLogDetObjective(torch_bkd)
        criterion.set_estimator(est)
        assert criterion.derivatives().jacobian is None

    def test_autograd_supplies_the_missing_jacobian(
        self, torch_bkd
    ) -> None:
        est = _make_nested_estimator(torch_bkd)
        criterion = GroupACVLogDetObjective(torch_bkd)
        criterion.set_estimator(est)
        wrapped = WithAutogradJacobian(criterion, torch_bkd)
        assert wrapped.derivatives().jacobian is not None

    def test_constraint_capability_follows_the_covariance(
        self, torch_bkd
    ) -> None:
        """Absent stays absent rather than becoming a zero stub."""
        est = _make_nested_estimator(torch_bkd)
        constraint = GroupACVRequirementConstraint(
            LogDeterminantConstraint(-1.0, torch_bkd)
        )
        constraint.set_estimator(est)
        assert constraint.derivatives().jacobian is None
        assert constraint.derivatives().whvp is None

    def test_allocation_succeeds_without_analytical_derivatives(
        self, torch_bkd
    ) -> None:
        """The cost objective is always analytic, so the composition
        that matters here is the constraint's.

        Names its solver: without analytical constraint Hessians the
        trust-region default works from a quasi-Newton approximation,
        and this test is about the composition, not the solver.
        """
        est = _make_nested_estimator(torch_bkd)
        alloc = GroupACVToleranceAllocator(
            est, optimizer=ScipySLSQPOptimizer(maxiter=2000, ftol=1e-10)
        )
        tolerance = -2.0
        result = alloc.allocate_for_tolerance(
            LogDeterminantConstraint(tolerance, torch_bkd)
        )
        assert result.success
        assert float(result.constraint_value[0]) <= tolerance + 1e-9


class _ConvergedResult:
    """Optimizer result claiming convergence at a supplied point."""

    def __init__(self, optima):
        self._optima = optima

    def success(self) -> bool:
        return True

    def optima(self):
        return self._optima


class _InfeasibleOptimizer:
    """Reports success at an allocation too small to meet any tolerance.

    Stands in for a solver whose exit status cannot be taken at face
    value, which is the case the allocator's own feasibility check
    exists to catch.
    """

    def __init__(self, bkd):
        self._bkd = bkd
        self._nvars = 0

    def bind(self, objective, bounds, constraints=None):
        self._nvars = objective.nvars()
        return self

    def minimize(self, init_guess):
        return _ConvergedResult(self._bkd.full((self._nvars, 1), 1.0))

    def is_bound(self) -> bool:
        return True

    def copy(self):
        return self

    def bkd(self):
        return self._bkd


class TestFeasibilityIsVerifiedNotAssumed:
    """A reported success must meet the tolerance, whatever the solver says.

    Sequential least squares searches along a ray on a merit function
    that trades cost against constraint violation, so its iterates
    leave the feasible set by design and it can stop outside. It has no
    feasibility tolerance to tighten and no feasible-path mode. The
    allocator therefore checks the returned allocation itself instead
    of trusting the solver's exit status.
    """

    def test_solver_that_reports_success_while_infeasible_is_refused(
        self, torch_bkd
    ) -> None:
        """The guarantee cannot rest on the solver's exit status.

        Sequential least squares happens to self-report this failure
        (scipy status 8), but nothing in the optimizer protocol
        requires that, and a solver reporting convergence at a point
        that misses the requirement must not be relayed as a success.
        """
        est = _make_estimator(torch_bkd)
        alloc = GroupACVToleranceAllocator(
            est, optimizer=_InfeasibleOptimizer(torch_bkd)
        )
        tolerance = -2.0
        result = alloc.allocate_for_tolerance(
            LogDeterminantConstraint(tolerance, torch_bkd),
            round_nsamples=False,
        )
        assert not result.success
        assert "misses the tolerance" in result.message

    def test_converged_boundary_solutions_are_accepted(
        self, torch_bkd
    ) -> None:
        """The check must not reject solutions that merely round.

        The optimum lies on the constraint boundary, so a converged
        answer lands a rounding error outside it. An exact comparison
        would discard those.
        """
        est = _make_estimator(torch_bkd)
        for optimizer in (
            ScipySLSQPOptimizer(maxiter=1000, ftol=1e-10),
            ScipyTrustConstrOptimizer(gtol=1e-8, maxiter=1000),
        ):
            result = GroupACVToleranceAllocator(
                est, optimizer=optimizer
            ).allocate_for_tolerance(
                LogDeterminantConstraint(-2.0, torch_bkd), round_nsamples=False
            )
            assert result.success
            assert float(result.constraint_value[0]) <= -2.0 + 1e-6


class TestDuality:
    """The two directions solve the same problem from opposite sides."""

    @pytest.mark.slow_on("TorchBkd")
    def test_inverse_recovers_the_forward_cost(self, bkd) -> None:
        """Feeding the forward criterion back returns the forward cost.

        Compared without rounding: the two directions round opposite
        ways, so integer results differ by design.
        """
        est = _make_estimator(bkd)
        target_cost = 200.0
        forward = GroupACVAllocationOptimizer(est).optimize(
            target_cost, round_nsamples=False
        )
        tolerance = float(forward.objective_value[0])
        inverse = GroupACVToleranceAllocator(est).allocate_for_tolerance(
            LogDeterminantConstraint(tolerance, bkd), round_nsamples=False
        )
        assert inverse.success
        assert inverse.total_cost <= forward.actual_cost * (1 + 1e-3)
        assert float(inverse.constraint_value[0]) <= tolerance + 1e-6

    def test_relaxed_allocation_sits_on_the_boundary(self, bkd) -> None:
        """The relaxed solution spends exactly enough, and no more.

        Without this the duality check passes by construction: the
        forward allocation is itself feasible for the inverse, so the
        inverse cannot report a higher cost even if it were far from
        optimal. Minimality is asserted on the relaxed solution rather
        than the rounded one, because rounding up necessarily buys more
        accuracy than asked for and so leaves every partition
        individually reducible.
        """
        est = _make_estimator(bkd)
        alloc = GroupACVToleranceAllocator(est)
        tolerance = -2.0
        result = alloc.allocate_for_tolerance(
            LogDeterminantConstraint(tolerance, bkd), round_nsamples=False
        )
        assert result.success
        achieved = float(result.constraint_value[0])
        assert achieved <= tolerance + 1e-6
        # Spending less would miss the requirement: the constraint is
        # active, so the achieved value sits at the tolerance rather
        # than below it.
        assert achieved >= tolerance - 1e-3

    def test_scaling_the_allocation_down_breaks_the_requirement(
        self, bkd
    ) -> None:
        """A uniformly cheaper allocation is infeasible.

        The integer result carries slack from rounding up, so single
        partitions can be trimmed; a proportional reduction cannot be
        absorbed and shows the allocation is not simply oversized.
        """
        est = _make_estimator(bkd)
        alloc = GroupACVToleranceAllocator(est)
        tolerance = -2.0
        requirement = LogDeterminantConstraint(tolerance, bkd)
        result = alloc.allocate_for_tolerance(requirement)
        assert result.success
        samples = bkd.asarray(
            result.npartition_samples, dtype=bkd.double_dtype()
        )
        smaller = est._covariance_from_npartition_samples(0.9 * samples)
        assert bkd.to_float(requirement.value(smaller)) > tolerance


class TestCrossFamilyEquivalence:
    """MLBLUE with all statistics known reduces to control variates.

    Concentrating the allocation on the all-models subset reproduces the
    CV estimator exactly, so the two families must agree on the cost of
    meeting an accuracy requirement. The paths share no covariance code,
    which is what makes this an independent check rather than one that
    passes by construction.
    """

    def test_concentrated_mlblue_matches_cv(self, bkd) -> None:
        """The precondition, asserted before comparing allocations."""
        mlblue, cv = _make_all_known_mlblue(bkd)
        nsamples = 50.0
        full_subset = mlblue.npartitions() - 1
        npartition_samples = bkd.zeros((mlblue.npartitions(),))
        npartition_samples[full_subset] = nsamples
        bkd.assert_allclose(
            mlblue._covariance_from_npartition_samples(npartition_samples),
            cv._covariance_from_nsamples_per_model(
                bkd.full((cv._nmodels,), nsamples)
            ),
            rtol=1e-10,
        )

    def test_concentrating_beats_spreading(self, bkd) -> None:
        """The regime precondition: CV's fixed shape is optimal here.

        Where a spread allocation wins, the two families legitimately
        diverge and the cost comparison below would compare different
        optima.
        """
        mlblue, _ = _make_all_known_mlblue(bkd)
        npartitions = mlblue.npartitions()
        budget = 3500.0
        partition_costs = bkd.einsum(
            "m,mp->p", mlblue._costs, mlblue._partitions_per_model
        )
        concentrated = bkd.zeros((npartitions,))
        concentrated[npartitions - 1] = budget / bkd.to_float(
            partition_costs[npartitions - 1]
        )
        best = float(
            mlblue._covariance_from_npartition_samples(concentrated)[0, 0]
        )
        np.random.seed(3)
        for _ in range(200):
            weights = np.random.dirichlet(np.ones(npartitions))
            spread = bkd.array(
                [
                    budget * weights[i] / bkd.to_float(partition_costs[i])
                    for i in range(npartitions)
                ]
            )
            value = float(
                mlblue._covariance_from_npartition_samples(spread)[0, 0]
            )
            if value > 0:
                assert best <= value * (1 + 1e-9)

    def test_same_cost_for_the_same_requirement(self, bkd) -> None:
        """Both families reach an accuracy target at the same cost."""
        mlblue, cv = _make_all_known_mlblue(bkd)
        requirement = MaxMarginalStandardErrorConstraint(0.02, bkd)
        cv_result = CVToleranceAllocator(cv).allocate_for_tolerance(
            requirement
        )
        # The same requirement object prices both families.
        result = GroupACVToleranceAllocator(mlblue).allocate_for_tolerance(
            requirement, round_nsamples=False
        )
        assert result.success
        # The control-variate allocator must take whole samples of every
        # model, so it pays at least what the relaxed solve does.
        assert result.total_cost <= cv_result.actual_cost() * (1 + 1e-6)


class TestVariableSpaceConsistency:
    """The answer must not depend on the optimizer's coordinates."""

    @pytest.mark.parametrize(
        "variable_scaling", ["none", "constraint_only", "full", "log"]
    )
    def test_tolerance_met_in_every_space(self, bkd, variable_scaling) -> None:
        """A wrong constraint Hessian shows up as a space-dependent answer."""
        est = _make_estimator(bkd)
        alloc = GroupACVToleranceAllocator(
            est,
            problem_config=AllocationProblemConfig(
                variable_scaling=variable_scaling
            ),
        )
        tolerance = -2.0
        result = alloc.allocate_for_tolerance(
            LogDeterminantConstraint(tolerance, bkd)
        )
        assert result.success
        assert float(result.constraint_value[0]) <= tolerance + 1e-9


def _make_scaled_estimator(bkd, scales, nmodels=3, estimator_cls=None):
    """An estimator whose QoIs differ in size by the factors ``scales``.

    Statistics of very different size are what separate a bound on each
    of them from a bound on their total.
    """
    np.random.seed(3)
    nqoi = len(scales)
    base = np.random.normal(0, 1, (nqoi, 4000)) * np.asarray(scales)[:, None]
    values = [
        bkd.asarray(base * 0.9**k + 0.3 * np.random.normal(0, 1, base.shape))
        for k in range(nmodels)
    ]
    stat = MultiOutputMean(nqoi, bkd)
    stat.set_pilot_quantities(*stat.compute_pilot_quantities(values))
    costs = bkd.asarray([1.0, 0.1, 0.01][:nmodels])
    cls = GroupACVEstimatorIS if estimator_cls is None else estimator_cls
    return cls(stat, costs)


REQUIREMENTS = [
    lambda bkd: MaxMarginalStandardErrorConstraint(0.05, bkd),
    lambda bkd: TraceConstraint(0.01, bkd),
    lambda bkd: LogDeterminantConstraint(-12.0, bkd),
]
REQUIREMENT_IDS = ["max-marginal", "trace", "log-det"]


class TestGroupACVRequirementConstraint:
    """The rows of a requirement, composed with the covariance."""

    @pytest.mark.parametrize("make", REQUIREMENTS, ids=REQUIREMENT_IDS)
    def test_derivatives(self, bkd, make) -> None:
        est = _make_scaled_estimator(bkd, [1.0, 5.0])
        constraint = GroupACVRequirementConstraint(make(bkd))
        constraint.set_estimator(est)
        constraint.set_min_nhf_samples(2)
        checker = DerivativeChecker(constraint)
        errors = checker.check_derivatives(
            bkd.asarray(np.linspace(20, 60, est.npartitions())[:, None]),
            weights=bkd.asarray(
                np.linspace(0.7, 1.3, constraint.nqoi())[:, None]
            ),
        )
        assert float(checker.error_ratio(errors[0])) <= 1e-6
        assert float(checker.error_ratio(errors[1])) <= 1e-6

    def test_bounds_are_lower_only(self, bkd) -> None:
        """The requirement rides in the values, not in finite upper
        bounds, and the floor row has none."""
        est = _make_scaled_estimator(bkd, [1.0, 5.0])
        constraint = GroupACVRequirementConstraint(
            MaxMarginalStandardErrorConstraint(0.05, bkd)
        )
        constraint.set_estimator(est)
        bkd.assert_allclose(constraint.lb(), bkd.zeros((constraint.nqoi(),)))
        assert bool(np.all(np.isinf(bkd.to_numpy(constraint.ub()))))

    def test_floor_row_matches_the_cost_constraint(self, bkd) -> None:
        """The minimum-sample row is shared with the budget direction."""
        est = _make_scaled_estimator(bkd, [1.0, 5.0])
        constraint = GroupACVRequirementConstraint(
            MaxMarginalStandardErrorConstraint(0.05, bkd)
        )
        constraint.set_estimator(est)
        constraint.set_min_nhf_samples(1)
        cost_constraint = GroupACVCostConstraint(bkd)
        cost_constraint.set_estimator(est)
        cost_constraint.set_budget(100.0, 1)
        iterate = est._init_guess(100.0)
        bkd.assert_allclose(
            constraint(iterate)[-1, 0], cost_constraint(iterate)[1, 0],
            rtol=1e-12,
        )

    def test_one_row_per_statistic_then_the_floor(self, bkd) -> None:
        est = _make_scaled_estimator(bkd, [1.0, 5.0, 25.0])
        constraint = GroupACVRequirementConstraint(
            MaxMarginalStandardErrorConstraint(0.05, bkd)
        )
        constraint.set_estimator(est)
        assert constraint.nqoi() == 3 + 1

    def test_no_analytical_derivatives_without_independent_sampling(
        self, bkd
    ) -> None:
        """Left to autograd, which the solve composes."""
        constraint = GroupACVRequirementConstraint(
            MaxMarginalStandardErrorConstraint(0.05, bkd)
        )
        constraint.set_estimator(_make_nested_estimator(bkd))
        assert constraint.derivatives().jacobian is None


class TestAllocateForRequirement:
    """Cost to a requirement stated as Monte Carlo states it."""

    @pytest.mark.parametrize("make", REQUIREMENTS, ids=REQUIREMENT_IDS)
    def test_requirement_is_met(self, bkd, make) -> None:
        req = make(bkd)
        est = _make_scaled_estimator(bkd, [1.0, 5.0])
        result = GroupACVToleranceAllocator(est).allocate_for_tolerance(req)
        assert result.success
        covariance = est._covariance_from_npartition_samples(
            bkd.asarray(result.npartition_samples, dtype=bkd.double_dtype())
        )
        assert bkd.all_bool(req.rows(covariance) <= 0.0)
        bkd.assert_allclose(result.constraint_value, req.value(covariance))

    def test_every_statistic_is_bounded_not_their_total(self, bkd) -> None:
        """A trace bound of ``k eps^2`` lets the largest statistic exceed
        ``eps``; one row per statistic does not."""
        eps = 0.05
        est = _make_scaled_estimator(bkd, [1.0, 5.0, 25.0])
        alloc = GroupACVToleranceAllocator(est)
        per_statistic = alloc.allocate_for_tolerance(
            MaxMarginalStandardErrorConstraint(eps, bkd)
        )
        total = alloc.allocate_for_tolerance(TraceConstraint(3 * eps**2, bkd))

        def worst(result):
            covariance = est._covariance_from_npartition_samples(
                bkd.asarray(
                    result.npartition_samples, dtype=bkd.double_dtype()
                )
            )
            return bkd.to_float(bkd.max(bkd.sqrt(bkd.diag(covariance))))

        assert per_statistic.success and total.success
        assert worst(per_statistic) <= eps
        assert worst(total) > eps

    def test_without_analytical_derivatives(self, torch_bkd) -> None:
        """A nested estimator solves through the autograd jacobian.

        Names its solver, as the criterion path's composition test does:
        with no analytical Hessian neither shipped solver is reliable on
        nested estimators across requirements, and this test is about the
        composition reaching the requirement path.
        """
        bkd = torch_bkd
        est = _make_scaled_estimator(
            bkd, [1.0, 5.0], estimator_cls=GroupACVEstimatorNested
        )
        req = LogDeterminantConstraint(-12.0, bkd)
        result = GroupACVToleranceAllocator(
            est, optimizer=ScipySLSQPOptimizer(maxiter=2000, ftol=1e-10)
        ).allocate_for_tolerance(req)
        assert result.success
        assert float(result.constraint_value[0]) <= -12.0

    def test_requirement_must_satisfy_the_protocol(self, numpy_bkd) -> None:
        alloc = GroupACVToleranceAllocator(_make_estimator(numpy_bkd))
        with pytest.raises(TypeError, match="ToleranceConstraintProtocol"):
            alloc.allocate_for_tolerance(0.05)
