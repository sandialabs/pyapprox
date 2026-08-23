"""Unit tests for tolerance-driven GroupACV allocation."""

import numpy as np
import pytest
from pyapprox.interface.functions.autograd import WithAutogradJacobian
from pyapprox.interface.functions.derivatives import Derivatives
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
    GroupACVToleranceConstraint,
    GroupACVTraceObjective,
)
from pyapprox.statest.groupacv.tolerance_allocation import (
    GroupACVToleranceAllocator,
    default_tolerance_optimizer,
)
from pyapprox.statest.groupacv.variable_space import AllocationProblemConfig
from pyapprox.statest.groupacv.variants import GroupACVEstimatorNested
from pyapprox.statest.statistics import MultiOutputMean
from pyapprox.statest.tolerance import (
    CVToleranceAllocator,
    MaxMarginalStandardErrorConstraint,
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
    known = {
        (m, "mean"): bkd.array(np.mean(pilot[m], axis=1))
        for m in range(1, nmodels)
    }
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


class TestGroupACVToleranceConstraint:
    """The criterion held at or below a tolerance."""

    def _make(self, bkd, tolerance=-2.0):
        est = _make_estimator(bkd)
        criterion = GroupACVLogDetObjective(bkd)
        con = GroupACVToleranceConstraint(criterion, bkd)
        con.set_estimator(est)
        con.set_tolerance(tolerance, 1)
        return est, criterion, con

    def test_row_layout_matches_cost_constraint(self, bkd) -> None:
        """Two rows, so the existing bound handling applies unchanged."""
        _, _, con = self._make(bkd)
        bkd.assert_allclose(bkd.asarray([con.nqoi()]), bkd.asarray([2]))

    def test_bounds_are_lower_only(self, bkd) -> None:
        """The requirement rides in the value, not in a finite upper
        bound: the SLSQP adapter drops an upper bound unless every row
        has one, and the sample-count row has none."""
        _, _, con = self._make(bkd)
        bkd.assert_allclose(con.lb(), bkd.zeros((2,)))
        assert bool(np.all(np.isinf(bkd.to_numpy(con.ub()))))

    def test_first_row_is_tolerance_minus_criterion(self, bkd) -> None:
        est, criterion, con = self._make(bkd, tolerance=-2.0)
        iterate = est._init_guess(100.0)
        bkd.assert_allclose(
            con(iterate)[0, 0],
            -2.0 - criterion(iterate)[0, 0],
            rtol=1e-12,
        )

    def test_second_row_matches_cost_constraint(self, bkd) -> None:
        """The minimum-sample row is shared with the budget direction."""
        est, _, con = self._make(bkd)
        cost_con = GroupACVCostConstraint(bkd)
        cost_con.set_estimator(est)
        cost_con.set_budget(100.0, 1)
        iterate = est._init_guess(100.0)
        bkd.assert_allclose(
            con(iterate)[1, 0], cost_con(iterate)[1, 0], rtol=1e-12
        )

    def test_jacobian_negates_the_criterion(self, bkd) -> None:
        est, criterion, con = self._make(bkd)
        iterate = est._init_guess(100.0)
        jac = con.derivatives().jacobian(iterate)
        bkd.assert_allclose(
            jac[0:1, :],
            -criterion.derivatives().jacobian(iterate),
            rtol=1e-10,
        )

    def test_whvp_is_nonzero(self, bkd) -> None:
        """The criterion has curvature, unlike the cost constraint.

        A zero weighted Hessian here would be the wrapper bug the
        variable-space chain rules were fixed for, reintroduced at the
        constraint itself.
        """
        est, _, con = self._make(bkd)
        iterate = est._init_guess(100.0)
        vec = bkd.full((con.nvars(), 1), 0.5)
        weights = bkd.array([[1.0], [0.0]])
        whvp = con.derivatives().whvp(iterate, vec, weights)
        assert float(bkd.max(bkd.abs(whvp))) > 1e-8

    def test_capability_absent_without_criterion_derivatives(
        self, bkd
    ) -> None:
        """A criterion without analytical derivatives yields no bundle
        fields, rather than zero-valued ones.

        GroupACV objectives report no derivatives unless the statistic
        supplies sigma-block derivatives and the estimator is
        independent-sample, so a nested estimator exercises this.
        """
        est = _make_estimator(bkd)
        criterion = GroupACVLogDetObjective(bkd)

        class _NoDerivatives(type(criterion)):
            def _build_derivatives(self):
                return Derivatives.none()

        bare = _NoDerivatives(bkd)
        bare.set_estimator(est)
        assert bare.derivatives().jacobian is None
        con = GroupACVToleranceConstraint(bare, bkd)
        con.set_estimator(est)
        assert con.derivatives().jacobian is None
        assert con.derivatives().whvp is None

    def test_normalization_survives_signed_tolerance(self, bkd) -> None:
        """Log-scale tolerances are signed and may sit near zero."""
        for tolerance in (-12.0, 0.0, 3.0):
            _, _, con = self._make(bkd, tolerance=tolerance)
            norm = con.normalization()
            assert bool(np.all(bkd.to_numpy(norm) > 0))


class TestGroupACVToleranceAllocator:
    """Tolerance-driven allocation."""

    def test_default_matches_the_budget_driven_recipe(self, bkd) -> None:
        """Same solver and same variable scaling as the forward path.

        The two directions share a feasible set, so a configuration
        that conditions one conditions the other.
        """
        assert isinstance(default_tolerance_optimizer(), ScipySLSQPOptimizer)
        alloc = GroupACVToleranceAllocator(_make_estimator(bkd))
        assert alloc._config.variable_scaling == "log"

    def test_optimizer_is_swappable(self, bkd) -> None:
        """The solver is injected, so a different one still applies."""
        est = _make_estimator(bkd)
        tolerance = -2.0
        default = GroupACVToleranceAllocator(est).allocate_for_tolerance(
            tolerance, round_nsamples=False
        )
        swapped = GroupACVToleranceAllocator(
            est, optimizer=ScipyTrustConstrOptimizer(gtol=1e-8, maxiter=1000)
        ).allocate_for_tolerance(tolerance, round_nsamples=False)
        assert swapped.success
        assert float(swapped.constraint_value[0]) <= tolerance + 1e-6
        assert swapped.total_cost <= default.total_cost * (1 + 1e-2)

    def test_requirement_is_satisfied(self, bkd) -> None:
        est = _make_estimator(bkd)
        alloc = GroupACVToleranceAllocator(est)
        tolerance = -2.0
        result = alloc.allocate_for_tolerance(tolerance)
        assert result.success
        assert float(result.constraint_value[0]) <= tolerance + 1e-9

    def test_rounding_up_is_what_preserves_the_requirement(
        self, bkd
    ) -> None:
        """Flooring the relaxed solution would violate the tolerance.

        The relaxed optimum sits on the boundary, so discarding
        fractional parts moves the criterion the wrong way. This is the
        opposite of the budget-driven path, which floors to stay under
        budget.
        """
        est = _make_estimator(bkd)
        alloc = GroupACVToleranceAllocator(est)
        tolerance = -2.0
        relaxed = alloc.allocate_for_tolerance(
            tolerance, round_nsamples=False
        )
        floored = bkd.floor(relaxed.npartition_samples)
        floored_value = alloc._criterion_value(floored)
        rounded = alloc.allocate_for_tolerance(tolerance)
        assert float(rounded.constraint_value[0]) <= tolerance + 1e-9
        assert floored_value > tolerance

    def test_requirement_met_at_the_sample_floor(self, bkd) -> None:
        """A loose tolerance returns the cheapest feasible allocation."""
        est = _make_estimator(bkd)
        alloc = GroupACVToleranceAllocator(est)
        result = alloc.allocate_for_tolerance(1e6)
        assert result.success
        assert "minimum feasible allocation" in result.message

    def test_unreachable_tolerance_raises(self, bkd) -> None:
        est = _make_estimator(bkd)
        alloc = GroupACVToleranceAllocator(est)
        with pytest.raises(ValueError, match="not achievable"):
            alloc.allocate_for_tolerance(-1e9)

    def test_max_cost_skips_the_reference_search(self, bkd) -> None:
        """Supplying a ceiling avoids probing for a feasible reference."""
        est = _make_estimator(bkd)
        alloc = GroupACVToleranceAllocator(est)
        unbounded = alloc.allocate_for_tolerance(-2.0)
        bounded = alloc.allocate_for_tolerance(
            -2.0, max_cost=10.0 * unbounded.total_cost
        )
        assert bounded.success
        assert float(bounded.constraint_value[0]) <= -2.0 + 1e-9

    def test_result_reports_cost_and_achieved_accuracy(self, bkd) -> None:
        """The minimized quantity and the guarantee are separate fields."""
        est = _make_estimator(bkd)
        alloc = GroupACVToleranceAllocator(est)
        result = alloc.allocate_for_tolerance(-2.0)
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
    """Derivatives absent from a criterion are composed, not skipped.

    Analytical derivatives require the statistic to supply sigma-block
    derivatives and the estimator to be independent-sample, so a nested
    estimator has none. In the budget-driven direction only the
    objective can lack them; here the criterion sits in the constraint
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

    def test_constraint_capability_follows_the_criterion(
        self, torch_bkd
    ) -> None:
        """Absent stays absent rather than becoming a zero stub."""
        est = _make_nested_estimator(torch_bkd)
        criterion = GroupACVLogDetObjective(torch_bkd)
        constraint = GroupACVToleranceConstraint(criterion, torch_bkd)
        constraint.set_estimator(est)
        constraint.set_tolerance(-1.0, 1)
        assert constraint.derivatives().jacobian is None
        assert constraint.derivatives().whvp is None

    def test_allocation_succeeds_without_analytical_derivatives(
        self, torch_bkd
    ) -> None:
        """The cost objective is always analytic, so the composition
        that matters here is the constraint's."""
        est = _make_nested_estimator(torch_bkd)
        alloc = GroupACVToleranceAllocator(est)
        tolerance = -2.0
        result = alloc.allocate_for_tolerance(tolerance)
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
            tolerance, round_nsamples=False
        )
        assert not result.success
        assert "misses the tolerance" in result.message

    def test_slsqp_breakdown_in_raw_counts_is_reported_as_failure(
        self, torch_bkd
    ) -> None:
        """In raw sample counts the solve stops outside the feasible set.

        The cheapest allocation leaves two subsets unsampled, so their
        partitions sit on the lower bound of 1e-8. Those active bounds
        together with the accuracy constraint leave the quadratic
        subproblem without a descent direction, and the solve stops
        while the requirement is still unmet (scipy status 8). The
        allocator refuses that rather than relaying it as a success.

        Log-space variables avoid it by construction: the same bound
        becomes log(1e-8), roughly -18, and the unsampled partitions
        settle near -16, so nothing is on a boundary.
        """
        est = _make_nested_estimator(torch_bkd)
        alloc = GroupACVToleranceAllocator(
            est,
            optimizer=ScipySLSQPOptimizer(maxiter=1000, ftol=1e-10),
            problem_config=AllocationProblemConfig(variable_scaling="none"),
        )
        result = alloc.allocate_for_tolerance(-2.0, round_nsamples=False)
        assert not result.success

    def test_default_configuration_converges_where_raw_counts_do_not(
        self, torch_bkd
    ) -> None:
        """The default pairing solves the problem the above one fails."""
        est = _make_nested_estimator(torch_bkd)
        result = GroupACVToleranceAllocator(est).allocate_for_tolerance(
            -2.0, round_nsamples=False
        )
        assert result.success
        assert float(result.constraint_value[0]) <= -2.0 + 1e-6

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
            ).allocate_for_tolerance(-2.0, round_nsamples=False)
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
            tolerance, round_nsamples=False
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
            tolerance, round_nsamples=False
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
        result = alloc.allocate_for_tolerance(tolerance)
        assert result.success
        samples = bkd.asarray(
            result.npartition_samples, dtype=bkd.double_dtype()
        )
        assert alloc._criterion_value(0.9 * samples) > tolerance


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
        target_se = 0.02
        cv_result = CVToleranceAllocator(cv).allocate_for_tolerance(
            MaxMarginalStandardErrorConstraint(target_se, bkd)
        )
        groupacv = GroupACVToleranceAllocator(
            mlblue, criterion=GroupACVTraceObjective(bkd)
        )
        # nqoi is 1, so the trace of the estimator covariance is the
        # variance and the requirement is the squared standard error.
        result = groupacv.allocate_for_tolerance(
            target_se**2, round_nsamples=False
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
        result = alloc.allocate_for_tolerance(tolerance)
        assert result.success
        assert float(result.constraint_value[0]) <= tolerance + 1e-9
