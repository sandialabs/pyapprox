"""Unit tests for tolerance-driven (inverse) MC and CV allocation."""

import numpy as np
import pytest
from pyapprox.statest.allocation import CVAllocator, MCAllocator
from pyapprox.statest.cv_estimator import CVEstimator
from pyapprox.statest.mc_estimator import MCEstimator
from pyapprox.statest.statistics import (
    MultiOutputMean,
    MultiOutputMeanAndVariance,
    MultiOutputVariance,
)
from pyapprox.statest.tolerance import (
    CVToleranceAllocator,
    LogDeterminantConstraint,
    MaxMarginalStandardErrorConstraint,
    MCToleranceAllocator,
    ToleranceConstraintProtocol,
    TraceConstraint,
)

STATS = [
    (MultiOutputMean, 1),
    (MultiOutputMean, 3),
    (MultiOutputVariance, 2),
    (MultiOutputMeanAndVariance, 2),
]


def _fit_stat(bkd, cls, nqoi, nmodels=1, nsamples=3000):
    """Build a statistic with pilot quantities from correlated samples."""
    pilot = [np.random.normal(0, 1, (nqoi, nsamples))]
    for k in range(1, nmodels):
        pilot.append(
            pilot[0] * (0.9**k) + 0.1 * np.random.normal(0, 1, (nqoi, nsamples))
        )
    stat = cls(nqoi, bkd)
    stat.set_pilot_quantities(
        *stat.compute_pilot_quantities([bkd.array(p) for p in pilot])
    )
    return stat


class TestConstraintForms:
    """The three shipped accuracy requirements."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_satisfy_protocol(self, bkd) -> None:
        for constraint in (
            MaxMarginalStandardErrorConstraint(0.1, bkd),
            TraceConstraint(0.1, bkd),
            LogDeterminantConstraint(-5.0, bkd),
        ):
            assert isinstance(constraint, ToleranceConstraintProtocol)

    def test_values(self, bkd) -> None:
        """Each form reduces the covariance to its own scalar summary."""
        cov = bkd.array([[4.0, 0.5], [0.5, 9.0]])
        bkd.assert_allclose(
            MaxMarginalStandardErrorConstraint(1.0, bkd).value(cov),
            bkd.asarray([3.0]),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            TraceConstraint(1.0, bkd).value(cov),
            bkd.asarray([13.0]),
            rtol=1e-12,
        )

    def test_value_shape_is_1d(self, bkd) -> None:
        cov = bkd.eye(3) * 2.0
        for constraint in (
            MaxMarginalStandardErrorConstraint(0.1, bkd),
            TraceConstraint(0.1, bkd),
            LogDeterminantConstraint(-5.0, bkd),
        ):
            assert constraint.value(cov).shape == (1,)

    @pytest.mark.parametrize(
        "cls", [MaxMarginalStandardErrorConstraint, TraceConstraint]
    )
    def test_nonpositive_tolerance_raises(self, bkd, cls) -> None:
        """Squared/positive-scale requirements cannot be met by any budget."""
        with pytest.raises(ValueError, match="must be positive"):
            cls(0.0, bkd)

    def test_log_determinant_accepts_negative_tolerance(self, bkd) -> None:
        """Log-scale tolerances are signed; negative is the normal case."""
        constraint = LogDeterminantConstraint(-12.0, bkd)
        bkd.assert_allclose(
            bkd.asarray([constraint.tolerance()]), bkd.asarray([-12.0])
        )

    def test_rank_detects_singular_covariance(self, bkd) -> None:
        """The eigenvalue filter is what keeps singular covariances finite."""
        constraint = LogDeterminantConstraint(-1.0, bkd)
        full = bkd.array([[4.0, 0.0], [0.0, 9.0]])
        singular = bkd.array([[1.0, 1.0], [1.0, 1.0]])
        bkd.assert_allclose(
            bkd.asarray([constraint.rank(full)]), bkd.asarray([2])
        )
        bkd.assert_allclose(
            bkd.asarray([constraint.rank(singular)]), bkd.asarray([1])
        )
        assert np.isfinite(bkd.to_float(constraint.value(singular)))


class _WorstQoIVariance:
    """An accuracy requirement defined outside the library.

    Stands for a caller's own form: it is never imported, registered or
    named by the allocator, and reaches it only by satisfying the
    protocol.
    """

    def __init__(self, tolerance, bkd):
        self._tolerance = tolerance
        self._bkd = bkd

    def bkd(self):
        return self._bkd

    def value(self, covariance):
        return self._bkd.atleast_1d(self._bkd.max(self._bkd.diag(covariance)))

    def tolerance(self):
        return self._tolerance

    def description(self):
        return f"worst per-QoI variance <= {self._tolerance}"


class TestCallerDefinedRequirements:
    """New requirements need no change to the library.

    The extension seam is the protocol, not a table of known forms, so
    a caller supplies an object rather than registering a name.
    """

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_satisfies_the_protocol_without_registration(self, bkd) -> None:
        assert isinstance(
            _WorstQoIVariance(0.01, bkd), ToleranceConstraintProtocol
        )

    def test_allocates_against_a_caller_defined_requirement(
        self, bkd
    ) -> None:
        stat = _fit_stat(bkd, MultiOutputMean, 2)
        alloc = MCToleranceAllocator(MCEstimator(stat, [1.0]))
        constraint = _WorstQoIVariance(0.01, bkd)
        fitted = alloc.allocate_for_tolerance(constraint)
        achieved = bkd.to_float(constraint.value(fitted.covariance()))
        assert achieved <= 0.01

    def test_caller_defined_requirement_is_also_minimal(self, bkd) -> None:
        """Rounding up stops at the first sufficient count, as for the
        shipped forms."""
        stat = _fit_stat(bkd, MultiOutputMean, 2)
        alloc = MCToleranceAllocator(MCEstimator(stat, [1.0]))
        constraint = _WorstQoIVariance(0.01, bkd)
        fitted = alloc.allocate_for_tolerance(constraint)
        nsamples = int(bkd.to_int(fitted.nsamples_per_model()[0]))
        below = bkd.to_float(
            constraint.value(alloc._covariance(float(nsamples - 1)))
        )
        assert below > 0.01


class TestMCToleranceAllocator:
    """Inverse allocation for MC estimators."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _allocator(self, bkd, cls=MultiOutputMean, nqoi=1, cost=1.0):
        stat = _fit_stat(bkd, cls, nqoi)
        return MCToleranceAllocator(MCEstimator(stat, [cost])), stat

    def test_rejects_non_estimator(self, bkd) -> None:
        with pytest.raises(TypeError, match="requires MCEstimator"):
            MCToleranceAllocator("not an estimator")

    def test_rejects_bare_float_tolerance(self, bkd) -> None:
        """Units are ambiguous without a constraint object."""
        alloc, _ = self._allocator(bkd)
        with pytest.raises(TypeError, match="ToleranceConstraintProtocol"):
            alloc.allocate_for_tolerance(1e-3)

    @pytest.mark.parametrize("tolerance", [0.1, 0.05])
    def test_marginal_standard_error_matches_closed_form(
        self, bkd, tolerance
    ) -> None:
        """For a mean, variance is sigma^2/n so n is ceil(sigma^2/tol^2)."""
        alloc, stat = self._allocator(bkd)
        fitted = alloc.allocate_for_tolerance(
            MaxMarginalStandardErrorConstraint(tolerance, bkd)
        )
        sigma_sq = bkd.to_float(stat._cov[0, 0])
        expected = int(np.ceil(sigma_sq / tolerance**2))
        bkd.assert_allclose(
            bkd.asarray([fitted.nsamples_per_model()[0]]),
            bkd.asarray([expected]),
        )

    @pytest.mark.parametrize("nqoi", [1, 3])
    @pytest.mark.parametrize("tolerance", [-5.0, -12.0])
    def test_log_determinant_matches_closed_form(
        self, bkd, nqoi, tolerance
    ) -> None:
        """logdet(Sigma/n) = logdet(Sigma) - nqoi*log(n) for a mean."""
        alloc, stat = self._allocator(bkd, nqoi=nqoi)
        fitted = alloc.allocate_for_tolerance(
            LogDeterminantConstraint(tolerance, bkd)
        )
        sigma = bkd.to_numpy(stat._cov[:nqoi, :nqoi])
        logdet = float(np.linalg.slogdet(sigma)[1])
        expected = int(np.ceil(np.exp((logdet - tolerance) / nqoi)))
        bkd.assert_allclose(
            bkd.asarray([fitted.nsamples_per_model()[0]]),
            bkd.asarray([expected]),
        )

    @pytest.mark.parametrize("cls,nqoi", STATS)
    @pytest.mark.parametrize(
        "form,tolerance",
        [
            (MaxMarginalStandardErrorConstraint, 0.05),
            (TraceConstraint, 0.01),
            (LogDeterminantConstraint, -20.0),
        ],
    )
    def test_requirement_is_satisfied(
        self, bkd, cls, nqoi, form, tolerance
    ) -> None:
        """The returned allocation meets the requirement, never approximates it."""
        alloc, _ = self._allocator(bkd, cls=cls, nqoi=nqoi)
        constraint = form(tolerance, bkd)
        fitted = alloc.allocate_for_tolerance(constraint)
        achieved = bkd.to_float(constraint.value(fitted.covariance()))
        assert achieved <= tolerance + 1e-12

    @pytest.mark.parametrize("cls,nqoi", STATS)
    @pytest.mark.parametrize(
        "form,tolerance",
        [
            (MaxMarginalStandardErrorConstraint, 0.05),
            (TraceConstraint, 0.01),
            (LogDeterminantConstraint, -20.0),
        ],
    )
    def test_allocation_is_minimal(
        self, bkd, cls, nqoi, form, tolerance
    ) -> None:
        """One fewer sample fails, so rounding up did not overshoot.

        This is what distinguishes rounding up from rounding down: the
        budget-driven allocators floor to stay under budget, which here
        would land just inside the infeasible side.
        """
        alloc, stat = self._allocator(bkd, cls=cls, nqoi=nqoi)
        constraint = form(tolerance, bkd)
        fitted = alloc.allocate_for_tolerance(constraint)
        nsamples = int(bkd.to_int(fitted.nsamples_per_model()[0]))
        if nsamples <= stat.min_nsamples():
            pytest.skip("allocation sits on the sample floor")
        below = bkd.to_float(
            constraint.value(alloc._covariance(float(nsamples - 1)))
        )
        assert below > tolerance

    def test_marginal_bound_is_per_statistic(self, bkd) -> None:
        """A log-determinant target does not bound the widest marginal.

        The determinant is a volume, so it can be met while one marginal
        is still wider than requested. This is why an accuracy
        requirement on marginals is a separate constraint form rather
        than a rescaling of the default criterion.
        """
        alloc, stat = self._allocator(
            bkd, cls=MultiOutputMeanAndVariance, nqoi=2
        )
        target_se = 0.1
        nstats = stat.nstats()
        # The isotropic log-determinant with every marginal at target_se.
        equivalent_logdet = nstats * np.log(target_se**2)
        by_logdet = alloc.allocate_for_tolerance(
            LogDeterminantConstraint(equivalent_logdet, bkd)
        )
        marginal = MaxMarginalStandardErrorConstraint(target_se, bkd)
        assert bkd.to_float(marginal.value(by_logdet.covariance())) > target_se

        by_marginal = alloc.allocate_for_tolerance(marginal)
        assert (
            bkd.to_float(marginal.value(by_marginal.covariance())) <= target_se
        )

    def test_requirement_met_at_sample_floor(self, bkd) -> None:
        """A loose requirement returns the statistic's minimum sample count."""
        alloc, stat = self._allocator(
            bkd, cls=MultiOutputMeanAndVariance, nqoi=2
        )
        fitted = alloc.allocate_for_tolerance(TraceConstraint(1e6, bkd))
        bkd.assert_allclose(
            bkd.asarray([fitted.nsamples_per_model()[0]]),
            bkd.asarray([stat.min_nsamples()]),
        )

    def test_unreachable_requirement_raises(self, bkd) -> None:
        """A tolerance no budget attains fails loudly."""
        stat = MultiOutputMean(1, bkd)
        # A covariance with a nonzero infimum: the estimator variance
        # cannot fall below this no matter how many samples are drawn.
        stat.set_pilot_quantities(bkd.eye(1) * 1.0)
        alloc = MCToleranceAllocator(MCEstimator(stat, [1.0]))
        with pytest.raises(ValueError, match="not achievable"):
            alloc.allocate_for_tolerance(
                LogDeterminantConstraint(-1e9, bkd)
            )

    def test_round_trip_with_budget_allocator(self, bkd) -> None:
        """Allocating at the returned cost reproduces the sample count."""
        alloc, _ = self._allocator(bkd, cost=2.0)
        fitted = alloc.allocate_for_tolerance(
            MaxMarginalStandardErrorConstraint(0.05, bkd)
        )
        forward = MCAllocator(alloc._template).allocate(fitted.actual_cost())
        bkd.assert_allclose(
            forward.nsamples_per_model(), fitted.nsamples_per_model()
        )

    def test_actual_cost_reflects_model_cost(self, bkd) -> None:
        alloc, _ = self._allocator(bkd, cost=3.0)
        fitted = alloc.allocate_for_tolerance(TraceConstraint(0.01, bkd))
        nsamples = bkd.to_float(fitted.nsamples_per_model()[0])
        bkd.assert_allclose(
            bkd.asarray([fitted.actual_cost()]),
            bkd.asarray([3.0 * nsamples]),
            rtol=1e-12,
        )


class TestCVToleranceAllocator:
    """Inverse allocation for CV estimators."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _allocator(self, bkd, nmodels=2, nqoi=1, cls=MultiOutputMean):
        stat = _fit_stat(bkd, cls, nqoi, nmodels=nmodels)
        costs = bkd.array([2.0, 1.0][:nmodels])
        return CVToleranceAllocator(CVEstimator(stat, costs))

    def test_rejects_non_estimator(self, bkd) -> None:
        with pytest.raises(TypeError, match="requires CVEstimator"):
            CVToleranceAllocator("not an estimator")

    @pytest.mark.parametrize("cls,nqoi", STATS)
    @pytest.mark.parametrize(
        "form,tolerance",
        [
            (MaxMarginalStandardErrorConstraint, 0.05),
            (TraceConstraint, 0.005),
            (LogDeterminantConstraint, -12.0),
        ],
    )
    def test_requirement_is_satisfied(
        self, bkd, cls, nqoi, form, tolerance
    ) -> None:
        """Every statistic, not just means: the mean-and-variance
        covariance has no closed-form inverse, so it exercises the
        numerical solve rather than a formula."""
        alloc = self._allocator(bkd, cls=cls, nqoi=nqoi)
        constraint = form(tolerance, bkd)
        fitted = alloc.allocate_for_tolerance(constraint)
        achieved = bkd.to_float(constraint.value(fitted.covariance()))
        assert achieved <= tolerance + 1e-12

    @pytest.mark.parametrize("cls,nqoi", STATS)
    def test_allocation_is_minimal(self, bkd, cls, nqoi) -> None:
        """One fewer sample of every model fails the requirement."""
        alloc = self._allocator(bkd, cls=cls, nqoi=nqoi)
        constraint = MaxMarginalStandardErrorConstraint(0.05, bkd)
        fitted = alloc.allocate_for_tolerance(constraint)
        nsamples = int(bkd.to_int(fitted.nsamples_per_model()[0]))
        if nsamples <= alloc._min_nsamples:
            pytest.skip("allocation sits on the sample floor")
        below = bkd.to_float(
            constraint.value(alloc._covariance(float(nsamples - 1)))
        )
        assert below > 0.05

    def test_every_model_gets_the_same_count(self, bkd) -> None:
        alloc = self._allocator(bkd, nmodels=2)
        fitted = alloc.allocate_for_tolerance(
            MaxMarginalStandardErrorConstraint(0.05, bkd)
        )
        counts = fitted.nsamples_per_model()
        bkd.assert_allclose(counts, bkd.full((2,), counts[0], dtype=int))

    def test_actual_cost_sums_model_costs(self, bkd) -> None:
        alloc = self._allocator(bkd, nmodels=2)
        fitted = alloc.allocate_for_tolerance(TraceConstraint(0.005, bkd))
        nsamples = bkd.to_float(fitted.nsamples_per_model()[0])
        bkd.assert_allclose(
            bkd.asarray([fitted.actual_cost()]),
            bkd.asarray([3.0 * nsamples]),
            rtol=1e-12,
        )

    def test_round_trip_with_budget_allocator(self, bkd) -> None:
        alloc = self._allocator(bkd)
        fitted = alloc.allocate_for_tolerance(TraceConstraint(0.005, bkd))
        forward = CVAllocator(alloc._template).allocate(fitted.actual_cost())
        bkd.assert_allclose(
            forward.nsamples_per_model(), fitted.nsamples_per_model()
        )

    def test_cheaper_than_mc_for_same_requirement(self, bkd) -> None:
        """Control variates reach a requirement for less than plain MC."""
        cv_alloc = self._allocator(bkd, nmodels=2)
        mc_stat = _fit_stat(bkd, MultiOutputMean, 1, nmodels=1)
        mc_alloc = MCToleranceAllocator(MCEstimator(mc_stat, [2.0]))
        constraint = MaxMarginalStandardErrorConstraint(0.05, bkd)
        cv_cost = cv_alloc.allocate_for_tolerance(constraint).actual_cost()
        mc_cost = mc_alloc.allocate_for_tolerance(constraint).actual_cost()
        assert cv_cost <= mc_cost
