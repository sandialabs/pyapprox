"""Unit tests for tolerance-driven (inverse) MC and CV allocation."""

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
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

    def nrows(self, nstats):
        return nstats

    def rows(self, covariance):
        return self._bkd.diag(covariance) / self._tolerance - 1.0

    def row_derivatives(self, covariance, directions):
        return self._bkd.einsum("dii->id", directions) / self._tolerance

    def row_second_derivatives(self, covariance, directions):
        n, d = covariance.shape[0], directions.shape[0]
        return self._bkd.zeros((n, d, d))


REQUIREMENTS = [
    lambda bkd: MaxMarginalStandardErrorConstraint(1.2, bkd),
    lambda bkd: TraceConstraint(4.0, bkd),
    lambda bkd: LogDeterminantConstraint(0.5, bkd),
    lambda bkd: _WorstQoIVariance(1.5, bkd),
]
REQUIREMENT_IDS = ["max-marginal", "trace", "log-det", "caller-defined"]
NSTATS = 3
NDIRECTIONS = 2


def _covariance_and_directions(bkd):
    """A well-conditioned covariance, and symmetric directions about it."""
    rng = np.random.RandomState(7)
    factor = rng.normal(0, 1, (NSTATS, NSTATS))
    covariance = factor @ factor.T / NSTATS + np.eye(NSTATS)

    def symmetric(scale):
        raw = rng.normal(0, scale, (NDIRECTIONS, NSTATS, NSTATS))
        return raw + np.transpose(raw, (0, 2, 1))

    return (
        bkd.asarray(covariance),
        bkd.asarray(symmetric(0.2)),
        bkd.asarray(symmetric(0.1)),
    )


class _RowsAlongACurve:
    r"""A requirement's rows as a function of coefficients ``x``.

    ``Sigma(x) = Sigma0 + sum_m (x_m V_m + x_m^2 W_m + x_m^3 W_m)``:
    nonlinear in ``x``, so even a row linear in the covariance has a
    second derivative here, and the derivatives below exercise the chain
    rule an allocator uses -- the rows along the covariance's first
    derivatives, plus along its second derivatives, plus the rows' own
    curvature along pairs of first derivatives. The cubic term keeps
    that second derivative varying with ``x``: with only the quadratic
    one, a row linear in the covariance has an exactly linear gradient,
    whose finite difference is exact and so tests nothing.
    """

    def __init__(self, requirement, covariance, linear, quadratic, bkd):
        self._req = requirement
        self._covariance = covariance
        self._linear = linear
        self._quadratic = quadratic
        self._bkd = bkd
        self._derivs = Derivatives.second_order_weighted(
            jacobian=self.jacobian, whvp=self.whvp
        )

    def derivatives(self):
        return self._derivs

    def bkd(self):
        return self._bkd

    def nvars(self):
        return NDIRECTIONS

    def nqoi(self):
        return self._req.nrows(NSTATS)

    def _sigma(self, x):
        return (
            self._covariance
            + self._bkd.einsum("m,mij->ij", x, self._linear)
            + self._bkd.einsum("m,mij->ij", x**2 + x**3, self._quadratic)
        )

    def _first(self, x):
        """``d Sigma / d x_m``, one per coefficient."""
        return self._linear + (2.0 * x + 3.0 * x**2)[:, None, None] * (
            self._quadratic
        )

    def _second(self, x):
        """``d2 Sigma / d x_m^2``; the mixed ones are zero."""
        return (2.0 + 6.0 * x)[:, None, None] * self._quadratic

    def __call__(self, samples):
        return self._req.rows(self._sigma(samples[:, 0]))[:, None]

    def jacobian(self, sample):
        x = sample[:, 0]
        return self._req.row_derivatives(self._sigma(x), self._first(x))

    def whvp(self, sample, vec, weights):
        x = sample[:, 0]
        sigma = self._sigma(x)
        curvature = self._req.row_second_derivatives(sigma, self._first(x))
        # d2 Sigma / dx_m dx_p is nonzero on the diagonal only.
        along_second = self._req.row_derivatives(sigma, self._second(x))
        hessians = curvature + self._bkd.einsum(
            "jm,mp->jmp", along_second, self._bkd.eye(NDIRECTIONS)
        )
        weighted = self._bkd.einsum("j,jmp->mp", weights[:, 0], hessians)
        return weighted @ vec


class TestRowContract:
    """What every requirement must satisfy, shipped or caller-defined.

    A new requirement joins ``REQUIREMENTS`` and is held to the same
    checks; nothing in an allocator changes.
    """

    @pytest.mark.parametrize("make", REQUIREMENTS, ids=REQUIREMENT_IDS)
    def test_shapes(self, bkd, make) -> None:
        req = make(bkd)
        covariance, directions, _ = _covariance_and_directions(bkd)
        nrows = req.nrows(NSTATS)
        assert req.rows(covariance).shape == (nrows,)
        assert req.row_derivatives(covariance, directions).shape == (
            nrows,
            NDIRECTIONS,
        )
        assert req.row_second_derivatives(covariance, directions).shape == (
            nrows,
            NDIRECTIONS,
            NDIRECTIONS,
        )

    @pytest.mark.parametrize("make", REQUIREMENTS, ids=REQUIREMENT_IDS)
    def test_rows_and_value_agree_on_what_is_met(self, bkd, make) -> None:
        """Across the boundary, from far inside to far outside it."""
        req = make(bkd)
        covariance, _, _ = _covariance_and_directions(bkd)
        verdicts = []
        for scale in np.logspace(-3, 3, 61):
            scaled = covariance * scale
            by_rows = bool(bkd.all_bool(req.rows(scaled) <= 0.0))
            by_value = bkd.to_float(req.value(scaled)) <= req.tolerance()
            assert by_rows == by_value
            verdicts.append(by_rows)
        assert any(verdicts) and not all(verdicts)

    @pytest.mark.parametrize("make", REQUIREMENTS, ids=REQUIREMENT_IDS)
    def test_rows_do_not_rise_as_the_covariance_shrinks(
        self, bkd, make
    ) -> None:
        req = make(bkd)
        covariance, _, _ = _covariance_and_directions(bkd)
        shrunk = covariance - 0.5 * covariance
        assert bkd.all_bool(req.rows(shrunk) <= req.rows(covariance))

    @pytest.mark.parametrize("make", REQUIREMENTS, ids=REQUIREMENT_IDS)
    def test_derivatives(self, bkd, make) -> None:
        covariance, linear, quadratic = _covariance_and_directions(bkd)
        curve = _RowsAlongACurve(make(bkd), covariance, linear, quadratic, bkd)
        checker = DerivativeChecker(curve)
        errors = checker.check_derivatives(
            bkd.asarray([[0.3], [-0.2]]),
            weights=bkd.asarray(
                np.linspace(0.5, 1.5, curve.nqoi())[:, None]
            ),
        )
        assert float(checker.error_ratio(errors[0])) <= 1e-6
        assert float(checker.error_ratio(errors[1])) <= 1e-6

    def test_directions_must_be_a_stack(self, numpy_bkd) -> None:
        req = MaxMarginalStandardErrorConstraint(1.0, numpy_bkd)
        covariance, directions, _ = _covariance_and_directions(numpy_bkd)
        with pytest.raises(ValueError, match="ndirections"):
            req.row_derivatives(covariance, directions[0])


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
