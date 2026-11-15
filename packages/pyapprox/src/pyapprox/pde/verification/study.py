r"""Convergence studies and the checks that judge them.

A study is the sequence of (mesh size, error) pairs a discretizer gives
for one problem and one error measure. A check judges it:

- :class:`Converges`: the least-squares slope of :math:`\log e` against
  :math:`\log h` reaches the expected rate, so mesh sizes need not halve
  exactly;
- :class:`Recovers`: every level is exact to a tolerance, for solutions
  in the discrete space;
- :class:`Stalls`: the slope stays below a bound, the wrong-model control.

Errors are plain floats: the stack integrated them.

A quantity study records a quantity of interest's value at each level
(:class:`QuantityStudy`). A truth turns those values into a study of
errors: :class:`ExactValue` (from the exact solution, e.g. by
:func:`pyapprox.pde.verification.exact.exact_value`) or
:class:`ReferenceValue` (a much finer solve or published data, with its
own uncertainty). Without either, :func:`extrapolate` estimates the
order and the limit from the values alone (Richardson), and
:class:`ObservedOrder` judges that order.
"""

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, TypeVar, runtime_checkable

from pyapprox.pde.verification.protocols import (
    ApproximationProtocol,
    DiscretizerProtocol,
    QuantityApproximationProtocol,
    QuantityDiscretizerProtocol,
)
from pyapprox.pde.verification.quantities import QuantityProtocol

P = TypeVar("P")


@runtime_checkable
class ErrorMeasureProtocol(Protocol):
    """Which error of an approximation a study records."""

    def measure(self, approximation: ApproximationProtocol) -> float:
        """The error of ``approximation``."""
        ...


class L2Error:
    r""":math:`\|u^* - u_h\|_{L^2}`."""

    def measure(self, approximation: ApproximationProtocol) -> float:
        r""":math:`\|u^* - u_h\|_{L^2}`, from the stack."""
        return approximation.l2_error()

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class NodalError:
    r""":math:`\max_i |u^*(x_i) - u_h(x_i)|`."""

    def measure(self, approximation: ApproximationProtocol) -> float:
        """The largest nodal error, from the stack."""
        return approximation.nodal_error()

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


@dataclass(frozen=True)
class Study:
    """Mesh sizes and errors, coarsest first.

    Raises
    ------
    ValueError
        If there are no levels, the lengths differ, a mesh size is not
        positive, or an error is negative or not finite.
    """

    h: Tuple[float, ...]
    errors: Tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.h or len(self.h) != len(self.errors):
            raise ValueError(
                f"a study needs one error per mesh size, at least one; got "
                f"{len(self.h)} sizes and {len(self.errors)} errors"
            )
        if any(not h > 0.0 for h in self.h):
            raise ValueError(f"mesh sizes must be positive, got {self.h}")
        if any(not (math.isfinite(e) and e >= 0.0) for e in self.errors):
            raise ValueError(f"errors must be finite and non-negative: {self.errors}")


@dataclass(frozen=True)
class QuantityStudy:
    """Mesh sizes and a quantity's values, coarsest first.

    Raises
    ------
    ValueError
        If there are no levels, the lengths differ, a mesh size is not
        positive, or a value is not finite.
    """

    h: Tuple[float, ...]
    values: Tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.h or len(self.h) != len(self.values):
            raise ValueError(
                f"a study needs one value per mesh size, at least one; got "
                f"{len(self.h)} sizes and {len(self.values)} values"
            )
        if any(not h > 0.0 for h in self.h):
            raise ValueError(f"mesh sizes must be positive, got {self.h}")
        if any(not math.isfinite(v) for v in self.values):
            raise ValueError(f"values must be finite: {self.values}")


@runtime_checkable
class TruthProtocol(Protocol):
    """What a quantity's values are compared with."""

    def errors(self, study: QuantityStudy) -> Study:
        """The error of each level's value."""
        ...


class ExactValue:
    """The quantity's exact value; errors are distances to it.

    Parameters
    ----------
    value : float
        The exact value.

    Raises
    ------
    ValueError
        If ``value`` is not finite.
    """

    def __init__(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError(f"an exact value must be finite, got {value}")
        self._value = value

    def errors(self, study: QuantityStudy) -> Study:
        """:math:`|Q_h - Q|` at each level."""
        return Study(study.h, tuple(abs(v - self._value) for v in study.values))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._value!r})"


class ReferenceValue:
    """A reference value known to within ``uncertainty`` (a much finer
    solve, or published data). An error at or below the uncertainty is not
    resolved by the reference, so it is refused rather than reported.

    Parameters
    ----------
    value : float
        The reference value.
    uncertainty : float
        How far the reference may be from the true value.

    Raises
    ------
    ValueError
        If ``value`` is not finite or ``uncertainty`` is not positive and
        finite.
    """

    def __init__(self, value: float, uncertainty: float) -> None:
        if not math.isfinite(value):
            raise ValueError(f"a reference value must be finite, got {value}")
        if not (math.isfinite(uncertainty) and uncertainty > 0.0):
            raise ValueError(
                f"the uncertainty must be positive and finite, got {uncertainty}"
            )
        self._value, self._uncertainty = value, uncertainty

    def errors(self, study: QuantityStudy) -> Study:
        r""":math:`|Q_h - Q_{\mathrm{ref}}|` at each level.

        Raises
        ------
        ValueError
            If an error is not above the reference's uncertainty.
        """
        errors = tuple(abs(v - self._value) for v in study.values)
        unresolved = [e for e in errors if e <= self._uncertainty]
        if unresolved:
            raise ValueError(
                f"errors {unresolved} are within the reference's uncertainty "
                f"{self._uncertainty:.3e}; drop those levels or use a finer "
                "reference"
            )
        return Study(study.h, errors)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._value!r}, "
            f"uncertainty={self._uncertainty!r})"
        )


@dataclass(frozen=True)
class CheckResult:
    """A check's verdict, what it observed, and why."""

    passed: bool
    observed: float
    message: str


def fitted_rate(study: Study, last: Optional[int] = None) -> float:
    r"""The least-squares slope of :math:`\log e` against :math:`\log h`.

    Parameters
    ----------
    study : Study
        The study.
    last : int, optional
        Fit only the finest ``last`` levels. Default: all.

    Raises
    ------
    ValueError
        If fewer than two levels are fitted, the fitted mesh sizes are all
        equal, or a fitted error is zero.
    """
    h, errors = study.h, study.errors
    if last is not None:
        h, errors = h[-last:], errors[-last:]
    if len(h) < 2:
        raise ValueError(f"a rate needs at least two levels, got {len(h)}")
    if any(e == 0.0 for e in errors):
        raise ValueError(f"a rate needs nonzero errors, got {errors}")
    xs = [math.log(v) for v in h]
    ys = [math.log(e) for e in errors]
    xmean, ymean = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - xmean) ** 2 for x in xs)
    if sxx == 0.0:
        raise ValueError(f"a rate needs distinct mesh sizes, got {h}")
    return sum((x - xmean) * (y - ymean) for x, y in zip(xs, ys)) / sxx


@runtime_checkable
class CheckProtocol(Protocol):
    """Judges a study."""

    def verify(self, study: Study) -> CheckResult:
        """The verdict on ``study``."""
        ...


class Converges:
    """The fitted rate is at least ``rate - atol``.

    Parameters
    ----------
    rate : float
        The expected rate.
    atol : float
        How far below ``rate`` the fitted rate may fall.
    last : int, optional
        Fit only the finest ``last`` levels, past a pre-asymptotic start.
    """

    def __init__(self, rate: float, atol: float = 0.1, last: Optional[int] = None):
        self._rate, self._atol, self._last = rate, atol, last

    def verify(self, study: Study) -> CheckResult:
        """Pass if the fitted rate reaches ``rate - atol``."""
        observed = fitted_rate(study, self._last)
        passed = observed >= self._rate - self._atol
        return CheckResult(
            passed,
            observed,
            f"fitted rate {observed:.3f}, expected at least "
            f"{self._rate - self._atol:.3f}; errors {study.errors}",
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(rate={self._rate}, atol={self._atol}, "
            f"last={self._last})"
        )


class Recovers:
    """Every level's error is at most ``atol``.

    Parameters
    ----------
    atol : float
        The largest error allowed.
    """

    def __init__(self, atol: float) -> None:
        self._atol = atol

    def verify(self, study: Study) -> CheckResult:
        """Pass if the largest error is at most ``atol``."""
        observed = max(study.errors)
        return CheckResult(
            observed <= self._atol,
            observed,
            f"largest error {observed:.3e}, allowed {self._atol:.3e}",
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(atol={self._atol})"


class Stalls:
    """The fitted rate stays below ``below``: the error does not fall as
    a correct model's would. A level with zero error is not a stall.

    Parameters
    ----------
    below : float
        The rate a stall stays under.
    """

    def __init__(self, below: float) -> None:
        self._below = below

    def verify(self, study: Study) -> CheckResult:
        """Pass if the fitted rate is below ``below``."""
        if any(e == 0.0 for e in study.errors):
            return CheckResult(False, 0.0, f"an error is zero: {study.errors}")
        observed = fitted_rate(study)
        return CheckResult(
            observed < self._below,
            observed,
            f"fitted rate {observed:.3f}, a stall stays below {self._below}; "
            f"errors {study.errors}",
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(below={self._below})"


def run_study(
    problem: P, discretizer: DiscretizerProtocol[P], measure: ErrorMeasureProtocol
) -> Study:
    """Solve ``problem`` at every level of ``discretizer`` and record
    ``measure``.

    Raises
    ------
    TypeError
        If ``discretizer``, ``measure`` or an approximation is of the wrong
        type.
    """
    if not isinstance(discretizer, DiscretizerProtocol):
        raise TypeError(
            "discretizer must satisfy DiscretizerProtocol, got "
            f"{type(discretizer).__name__}"
        )
    if not isinstance(measure, ErrorMeasureProtocol):
        raise TypeError(
            f"measure must satisfy ErrorMeasureProtocol, got {type(measure).__name__}"
        )
    hs, errors = [], []
    for approximation in discretizer.approximations(problem):
        if not isinstance(approximation, ApproximationProtocol):
            raise TypeError(
                "each approximation must satisfy ApproximationProtocol, got "
                f"{type(approximation).__name__}"
            )
        hs.append(approximation.h())
        errors.append(measure.measure(approximation))
    return Study(tuple(hs), tuple(errors))


def verify(
    problem: P,
    discretizer: DiscretizerProtocol[P],
    measure: ErrorMeasureProtocol,
    check: CheckProtocol,
) -> CheckResult:
    """Run the study and judge it with ``check``.

    Raises
    ------
    TypeError
        If ``check`` is of the wrong type (and as :func:`run_study`).
    """
    if not isinstance(check, CheckProtocol):
        raise TypeError(f"check must satisfy CheckProtocol, got {type(check).__name__}")
    return check.verify(run_study(problem, discretizer, measure))


def run_quantity_study(
    problem: P,
    discretizer: QuantityDiscretizerProtocol[P],
    quantity: QuantityProtocol,
) -> QuantityStudy:
    """Solve ``problem`` at every level of ``discretizer`` and record
    ``quantity``'s value.

    Raises
    ------
    TypeError
        If ``discretizer``, ``quantity`` or an approximation is of the
        wrong type.
    """
    if not isinstance(discretizer, QuantityDiscretizerProtocol):
        raise TypeError(
            "discretizer must satisfy QuantityDiscretizerProtocol, got "
            f"{type(discretizer).__name__}"
        )
    if not isinstance(quantity, QuantityProtocol):
        raise TypeError(
            f"quantity must satisfy QuantityProtocol, got {type(quantity).__name__}"
        )
    hs, values = [], []
    for approximation in discretizer.approximations(problem):
        if not isinstance(approximation, QuantityApproximationProtocol):
            raise TypeError(
                "each approximation must satisfy QuantityApproximationProtocol, "
                f"got {type(approximation).__name__}"
            )
        hs.append(approximation.h())
        values.append(approximation.quantity(quantity))
    return QuantityStudy(tuple(hs), tuple(values))


_RATIO_RTOL = 1e-6


@dataclass(frozen=True)
class Extrapolation:
    """Richardson's estimate from the three finest levels.

    Attributes
    ----------
    order : float
        The observed order :math:`p`.
    limit : float
        The extrapolated value :math:`Q^*`.
    gci : float
        Roache's grid convergence index on the finest value, as an
        absolute band: :math:`F_s |Q_3 - Q_2| / (r^p - 1)`.
    """

    order: float
    limit: float
    gci: float


def _asymptotic_failure(values: Tuple[float, ...]) -> Optional[str]:
    """Why the successive differences of ``values`` are not in the
    asymptotic range (one sign, shrinking), or ``None`` if they are."""
    diffs = [b - a for a, b in zip(values, values[1:])]
    if any(d == 0.0 for d in diffs):
        return f"values stop changing (roundoff?): {values}"
    if any((a > 0.0) != (b > 0.0) for a, b in zip(diffs, diffs[1:])):
        return f"oscillatory: successive differences change sign; values {values}"
    if any(abs(b) >= abs(a) for a, b in zip(diffs, diffs[1:])):
        return f"not converging: successive differences do not shrink; values {values}"
    return None


def extrapolate(study: QuantityStudy, safety: float = 1.25) -> Extrapolation:
    r"""Richardson extrapolation from the three finest levels.

    With a constant refinement ratio :math:`r = h_1/h_2 = h_2/h_3` and
    values :math:`Q_1, Q_2, Q_3` (finest last), :math:`p = \log((Q_2 -
    Q_1)/(Q_3 - Q_2))/\log r` and :math:`Q^* = Q_3 + (Q_3 - Q_2)/(r^p -
    1)`.

    Parameters
    ----------
    study : QuantityStudy
        At least three levels.
    safety : float
        Roache's factor of safety :math:`F_s`; 1.25 for three levels.

    Raises
    ------
    ValueError
        If there are fewer than three levels, the ratio of the finest
        three is not constant or not above one, or their differences are
        not in the asymptotic range (zero, oscillatory or not shrinking).
    """
    if len(study.h) < 3:
        raise ValueError(f"extrapolation needs three levels, got {len(study.h)}")
    h1, h2, h3 = study.h[-3:]
    r = h1 / h2
    if not r > 1.0 or abs(h2 / h3 - r) > _RATIO_RTOL * r:
        raise ValueError(
            f"extrapolation needs a constant refinement ratio above one; the "
            f"finest mesh sizes {(h1, h2, h3)} give {h1 / h2} and {h2 / h3}"
        )
    finest = study.values[-3:]
    failure = _asymptotic_failure(finest)
    if failure is not None:
        raise ValueError(f"cannot extrapolate: {failure}")
    q1, q2, q3 = finest
    growth = (q2 - q1) / (q3 - q2)
    order = math.log(growth) / math.log(r)
    limit = q3 + (q3 - q2) / (growth - 1.0)
    return Extrapolation(order, limit, safety * abs(q3 - q2) / (growth - 1.0))


@runtime_checkable
class QuantityCheckProtocol(Protocol):
    """Judges a quantity study."""

    def verify(self, study: QuantityStudy) -> CheckResult:
        """The verdict on ``study``."""
        ...


class ObservedOrder:
    """Richardson's observed order is at least ``order - atol``.

    First confirms the asymptotic range: over the levels used, successive
    differences keep one sign and shrink. An oscillatory or diverging
    sequence fails with that reason instead of an order.

    Parameters
    ----------
    order : float
        The expected order.
    atol : float
        How far below ``order`` the observed order may fall.
    last : int, optional
        Use only the finest ``last`` levels (at least three), past a
        pre-asymptotic start.
    """

    def __init__(
        self, order: float, atol: float = 0.1, last: Optional[int] = None
    ) -> None:
        if last is not None and last < 3:
            raise ValueError(f"an observed order needs three levels, got last={last}")
        self._order, self._atol, self._last = order, atol, last

    def verify(self, study: QuantityStudy) -> CheckResult:
        """Pass if the sequence is asymptotic and its order reaches
        ``order - atol``; the message gives the limit and the GCI band.

        Raises
        ------
        ValueError
            As :func:`extrapolate`, for too few levels or a ratio that is
            not constant.
        """
        if self._last is not None:
            study = QuantityStudy(study.h[-self._last :], study.values[-self._last :])
        failure = _asymptotic_failure(study.values)
        if failure is not None and len(study.values) >= 3:
            return CheckResult(False, math.nan, failure)
        estimate = extrapolate(study)
        passed = estimate.order >= self._order - self._atol
        return CheckResult(
            passed,
            estimate.order,
            f"observed order {estimate.order:.3f}, expected at least "
            f"{self._order - self._atol:.3f}; limit {estimate.limit:.12g} "
            f"(GCI band {estimate.gci:.3e}); values {study.values}",
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(order={self._order}, atol={self._atol}, "
            f"last={self._last})"
        )


def verify_quantity(
    problem: P,
    discretizer: QuantityDiscretizerProtocol[P],
    quantity: QuantityProtocol,
    check: QuantityCheckProtocol,
) -> CheckResult:
    """Run the quantity study and judge it with ``check``.

    Raises
    ------
    TypeError
        If ``check`` is of the wrong type (and as
        :func:`run_quantity_study`).
    """
    if not isinstance(check, QuantityCheckProtocol):
        raise TypeError(
            f"check must satisfy QuantityCheckProtocol, got {type(check).__name__}"
        )
    return check.verify(run_quantity_study(problem, discretizer, quantity))
