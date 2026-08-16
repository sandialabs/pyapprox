"""Tolerance-driven (inverse) allocation for MC and CV estimators.

The budget-driven allocators in :mod:`pyapprox.statest.allocation` answer
*minimize the estimator covariance subject to cost <= target_cost*. The
allocators here answer the dual question, *minimize cost subject to an
accuracy requirement*, for a caller who knows the accuracy they need
rather than the budget they have.

The accuracy requirement is supplied by the caller as a constraint object
satisfying :class:`ToleranceConstraintProtocol`, never as a bare float.
Different scalarizations of the same estimator covariance disagree by
orders of magnitude -- a log-determinant tolerance is on a log scale and
is normally negative, while a trace tolerance is in squared estimate
units -- so a bare number carries no usable meaning on its own. Requiring
the constraint object makes the units explicit at the call site.
"""

from typing import Generic, Protocol, Tuple, TypeVar, runtime_checkable

from pyapprox.statest.cv_estimator import CVEstimator, FittedCVEstimator
from pyapprox.statest.mc_estimator import FittedMCEstimator, MCEstimator
from pyapprox.util.backends.protocols import Array, ArrayProtocol, Backend


@runtime_checkable
class ToleranceConstraintProtocol(Protocol, Generic[Array]):
    """An accuracy requirement on an estimator covariance matrix.

    Implementations must be **monotone decreasing in the sample count**:
    drawing more samples may never increase :meth:`value`. The inverse
    allocation brackets and then solves for the smallest sample count
    meeting the requirement, and both steps rely on that monotonicity.
    A non-monotone requirement makes the problem ill-posed rather than
    merely harder to solve.
    """

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        ...

    def value(self, covariance: Array) -> Array:
        """The constrained quantity, as a shape ``(1,)`` array."""
        ...

    def tolerance(self) -> float:
        """The largest acceptable :meth:`value`."""
        ...

    def description(self) -> str:
        """Human-readable requirement, used in error messages."""
        ...


# Contravariant because the protocol only consumes covariance matrices;
# the shared Array typevar is invariant and cannot express that.
RankArray_contra = TypeVar(
    "RankArray_contra", bound=ArrayProtocol, contravariant=True
)


@runtime_checkable
class FilteredRankProtocol(Protocol, Generic[RankArray_contra]):
    """A requirement that ignores covariance directions below a floor.

    Such a requirement is monotone only while the ignored set stays
    fixed, so the inverse allocation checks :meth:`rank` at both ends of
    its search range. Requirements that use every direction do not
    implement this and skip the check.
    """

    def rank(self, covariance: RankArray_contra) -> int:
        """Number of covariance directions the requirement accounts for."""
        ...


class MaxMarginalStandardErrorConstraint(Generic[Array]):
    r"""Require every marginal standard error to be at or below a tolerance.

    Constrains :math:`\max_i \sqrt{\Sigma_{ii}}`, where :math:`\Sigma` is
    the estimator covariance. Unlike a determinant or trace, this bounds
    each statistic individually, so a requested tolerance is a genuine
    guarantee on every entry rather than on an aggregate that can hide a
    single wide marginal.

    The tolerance is in the same units as the estimated statistic.
    """

    def __init__(self, tolerance: float, bkd: Backend[Array]) -> None:
        if tolerance <= 0.0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")
        self._tolerance = tolerance
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def value(self, covariance: Array) -> Array:
        bkd = self._bkd
        return bkd.atleast_1d(bkd.max(bkd.sqrt(bkd.diag(covariance))))

    def tolerance(self) -> float:
        return self._tolerance

    def description(self) -> str:
        return f"max marginal standard error <= {self._tolerance}"


class TraceConstraint(Generic[Array]):
    """Require the trace of the estimator covariance to be at or below a
    tolerance.

    The tolerance is in squared estimate units, and aggregates over all
    statistics rather than bounding any one of them.
    """

    def __init__(self, tolerance: float, bkd: Backend[Array]) -> None:
        if tolerance <= 0.0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")
        self._tolerance = tolerance
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def value(self, covariance: Array) -> Array:
        bkd = self._bkd
        return bkd.atleast_1d(bkd.trace(covariance))

    def tolerance(self) -> float:
        return self._tolerance

    def description(self) -> str:
        return f"trace of estimator covariance <= {self._tolerance}"


class LogDeterminantConstraint(Generic[Array]):
    """Require the log determinant of the estimator covariance to be at or
    below a tolerance.

    Matches the default optimization criterion used by the budget-driven
    allocators, so a tolerance expressed this way is directly comparable
    with the values those allocators report. The tolerance is on a log
    scale and is normally negative; it is a volume measure and does not
    bound any individual marginal standard error.

    Eigenvalues at or below ``1e-14`` are excluded, matching
    :func:`~pyapprox.statest.statistics.log_determinant_variance`. That
    filter is what keeps the value finite for the singular covariances
    produced by the variance statistics, whose entries repeat.
    """

    def __init__(self, tolerance: float, bkd: Backend[Array]) -> None:
        self._tolerance = tolerance
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def value(self, covariance: Array) -> Array:
        bkd = self._bkd
        eigvals = bkd.eigh(covariance)[0]
        return bkd.atleast_1d(bkd.sum(bkd.log(eigvals[eigvals > 1e-14])))

    def rank(self, covariance: Array) -> int:
        """Number of eigenvalues above the conditioning floor.

        The inverse allocation compares this across the search bracket:
        a change means the filter admitted or dropped a direction, which
        would break monotonicity of :meth:`value`.
        """
        bkd = self._bkd
        eigvals = bkd.eigh(covariance)[0]
        return int(bkd.to_int(bkd.sum(bkd.asarray(eigvals > 1e-14, dtype=int))))

    def tolerance(self) -> float:
        return self._tolerance

    def description(self) -> str:
        return f"log determinant of estimator covariance <= {self._tolerance}"


# Bound on the search for the smallest sufficient sample count. The
# bracket doubles from the sample floor, so the cap admits sample counts
# far beyond any affordable budget before declaring a tolerance
# unreachable.
_MAX_BRACKET_DOUBLINGS = 200


class ToleranceAllocatorMixin(Generic[Array]):
    """Shared inverse-allocation solve for fixed-shape estimators.

    Subclasses supply the estimator-specific pieces: how a sample count
    maps to an estimator covariance, and what one sample of every model
    costs. The search itself -- bracket, solve, round up, verify -- is
    common to MC and CV because both allocate the same number of samples
    to every model, leaving a single scalar unknown.
    """

    _bkd: Backend[Array]
    _min_nsamples: int

    def _covariance(self, nsamples: float) -> Array:
        """Estimator covariance at a (possibly non-integer) sample count."""
        raise NotImplementedError

    def _cost(self, nsamples: float) -> float:
        """Cost of drawing ``nsamples`` samples of every model."""
        raise NotImplementedError

    def _slack(
        self, constraint: ToleranceConstraintProtocol[Array], nsamples: float
    ) -> float:
        """Signed distance to the requirement; non-negative is feasible."""
        value = constraint.value(self._covariance(nsamples))
        return constraint.tolerance() - self._bkd.to_float(value)

    def _bracket(
        self, constraint: ToleranceConstraintProtocol[Array]
    ) -> Tuple[float, float]:
        """Return sample counts straddling the requirement.

        The lower end is infeasible and the upper end feasible, so a
        monotone requirement has its crossing between them.
        """
        lb = float(self._min_nsamples)
        ub = lb
        for _ in range(_MAX_BRACKET_DOUBLINGS):
            ub *= 2.0
            if self._slack(constraint, ub) >= 0.0:
                return lb, ub
            lb = ub
        raise ValueError(
            f"{constraint.description()} is not achievable: the requirement "
            f"is still unmet at {ub:g} samples of every model. The estimator "
            "covariance has a nonzero infimum for this problem, so no budget "
            "attains this tolerance; loosen it or add models."
        )

    def _check_rank_stable(
        self,
        constraint: ToleranceConstraintProtocol[Array],
        lb: float,
        ub: float,
    ) -> None:
        """Reject a bracket whose endpoints filter different directions.

        Constraints that discard near-zero eigenvalues are monotone only
        while the discarded set is fixed. If it changes across the
        bracket the requirement is not monotone there, and the solve
        would return a sample count that does not mean what it claims.
        """
        if not isinstance(constraint, FilteredRankProtocol):
            return
        rank_lb = constraint.rank(self._covariance(lb))
        rank_ub = constraint.rank(self._covariance(ub))
        if rank_lb != rank_ub:
            raise ValueError(
                f"{constraint.description()} is not monotone over the search "
                f"range: {rank_lb} covariance directions exceed the "
                f"conditioning floor at {lb:g} samples but {rank_ub} do at "
                f"{ub:g}. The requirement cannot be inverted reliably here."
            )

    def _solve_relaxed(
        self,
        constraint: ToleranceConstraintProtocol[Array],
        lb: float,
        ub: float,
    ) -> float:
        """Smallest real sample count meeting the requirement."""
        from scipy.optimize import brentq

        def residual(nsamples: float) -> float:
            return self._slack(constraint, nsamples)

        return float(brentq(residual, lb, ub, xtol=1e-10, rtol=1e-12))

    def _round_up(
        self, constraint: ToleranceConstraintProtocol[Array], relaxed: float
    ) -> int:
        """Smallest integer sample count meeting the requirement.

        Rounds up rather than down. The relaxed solution sits on the
        requirement boundary, so discarding its fractional part would
        land just inside the infeasible side -- the opposite of the
        budget-driven allocators, which round down to stay under budget.

        Rounding up is the whole of the step. The relaxed count is a
        root located to ``xtol``, so raising it to the next integer
        clears a requirement that decreases in the sample count. If the
        result still misses, the requirement is not monotone and the
        premise the bracket-and-solve rests on does not hold; scanning
        upwards from here would return a larger count while leaving that
        broken premise unreported.
        """
        bkd = self._bkd
        nsamples = int(bkd.to_int(bkd.ceil(bkd.asarray(relaxed) - 1e-10)))
        nsamples = max(nsamples, self._min_nsamples)
        if self._slack(constraint, float(nsamples)) < 0.0:
            raise ValueError(
                f"{constraint.description()} was not met after rounding up "
                f"from the relaxed solution {relaxed:g}; the requirement is "
                "not monotone in the sample count."
            )
        return nsamples

    def _nsamples_for_tolerance(
        self, constraint: ToleranceConstraintProtocol[Array]
    ) -> int:
        """Smallest integer sample count satisfying ``constraint``."""
        if not isinstance(constraint, ToleranceConstraintProtocol):
            raise TypeError(
                "constraint must satisfy ToleranceConstraintProtocol, got "
                f"{type(constraint).__name__}"
            )
        if self._slack(constraint, float(self._min_nsamples)) >= 0.0:
            return self._min_nsamples
        lb, ub = self._bracket(constraint)
        self._check_rank_stable(constraint, lb, ub)
        return self._round_up(constraint, self._solve_relaxed(constraint, lb, ub))


class MCToleranceAllocator(ToleranceAllocatorMixin[Array], Generic[Array]):
    """Allocate the cheapest MC sample count meeting an accuracy requirement.

    The dual of :class:`~pyapprox.statest.allocation.MCAllocator`, which
    maximizes accuracy under a budget.
    """

    def __init__(self, template: MCEstimator[Array]) -> None:
        if not isinstance(template, MCEstimator):
            raise TypeError(
                "MCToleranceAllocator requires MCEstimator, got "
                f"{type(template).__name__}"
            )
        self._template = template
        self._bkd = template._bkd
        self._min_nsamples = template._stat.min_nsamples()

    def _covariance(self, nsamples: float) -> Array:
        return self._template._covariance_from_npartition_samples(
            self._bkd.asarray([nsamples])
        )

    def _cost(self, nsamples: float) -> float:
        return self._bkd.to_float(self._template._costs[0] * nsamples)

    def allocate_for_tolerance(
        self, constraint: ToleranceConstraintProtocol[Array]
    ) -> FittedMCEstimator[Array]:
        """Return the cheapest fitted estimator meeting ``constraint``.

        Parameters
        ----------
        constraint : ToleranceConstraintProtocol
            The accuracy requirement, carrying its own tolerance and
            units. Required rather than a bare tolerance so the units
            are unambiguous at the call site.

        Returns
        -------
        FittedMCEstimator
            Fitted estimator whose covariance satisfies ``constraint``.

        Raises
        ------
        ValueError
            If no sample count attains the requirement.
        """
        nsamples = self._nsamples_for_tolerance(constraint)
        nsamples_per_model = self._bkd.asarray([nsamples], dtype=int)
        return FittedMCEstimator(
            self._template, nsamples_per_model, self._cost(float(nsamples))
        )


class CVToleranceAllocator(ToleranceAllocatorMixin[Array], Generic[Array]):
    """Allocate the cheapest CV sample count meeting an accuracy requirement.

    The dual of :class:`~pyapprox.statest.allocation.CVAllocator`. Every
    model receives the same number of samples, so the cost of one sample
    is the sum of the per-model costs.
    """

    def __init__(self, template: CVEstimator[Array]) -> None:
        if not isinstance(template, CVEstimator):
            raise TypeError(
                "CVToleranceAllocator requires CVEstimator, got "
                f"{type(template).__name__}"
            )
        self._template = template
        self._bkd = template._bkd
        self._min_nsamples = template._stat.min_nsamples()

    def _covariance(self, nsamples: float) -> Array:
        nsamples_per_model = self._bkd.full(
            (self._template._nmodels,), nsamples
        )
        return self._template._covariance_from_nsamples_per_model(
            nsamples_per_model
        )

    def _cost(self, nsamples: float) -> float:
        return self._bkd.to_float(self._bkd.sum(self._template._costs) * nsamples)

    def allocate_for_tolerance(
        self, constraint: ToleranceConstraintProtocol[Array]
    ) -> FittedCVEstimator[Array]:
        """Return the cheapest fitted estimator meeting ``constraint``.

        Parameters
        ----------
        constraint : ToleranceConstraintProtocol
            The accuracy requirement, carrying its own tolerance and
            units.

        Returns
        -------
        FittedCVEstimator
            Fitted estimator whose covariance satisfies ``constraint``.

        Raises
        ------
        ValueError
            If no sample count attains the requirement.
        """
        nsamples = self._nsamples_for_tolerance(constraint)
        nsamples_per_model = self._bkd.full(
            (self._template._nmodels,), nsamples, dtype=int
        )
        return FittedCVEstimator(
            self._template, nsamples_per_model, self._cost(float(nsamples))
        )
