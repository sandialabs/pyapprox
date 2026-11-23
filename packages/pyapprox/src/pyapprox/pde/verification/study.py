r"""Convergence studies and the checks that judge them.

A study is a table: one row per level, holding the level's
:class:`~pyapprox.pde.verification.protocols.Sizes`, and one labeled
column per error measure. :func:`run_study` solves each level once and
records every measure; :meth:`Study.from_records` builds the same table
from plain numbers, e.g. a result file of a code run elsewhere. A check
judges one column against one
:class:`~pyapprox.pde.verification.protocols.Axis` of the sizes (the
mesh size, the degree, a time step, a subdomain's own size, or an axis a
stack declares):

- :class:`Converges`: the least-squares slope of :math:`\log e` against
  the log of the axis matches the expected rate, from both sides, so
  sizes need not halve exactly and a rate above the expected one is
  reported rather than passed;
- :class:`DecaysExponentially`: the slope of :math:`\log e` against the
  axis (the degree) reaches an exponential decay, for spectral methods
  and p-refinement;
- :class:`Recovers`: every level is exact to a tolerance, for solutions
  in the discrete space;
- :class:`Stalls`: the slope stays below a bound, the wrong-model control.

Errors are plain floats: the stack integrated them. Check results hold
only booleans, numbers and strings, so they serialize to JSON.

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
from typing import (
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    runtime_checkable,
)

from pyapprox.pde.verification.protocols import (
    DEGREE,
    ApproximationProtocol,
    Axis,
    DiscretizerProtocol,
    H,
    QuantityApproximationProtocol,
    QuantityDiscretizerProtocol,
    Sizes,
)
from pyapprox.pde.verification.quantities import QuantityProtocol

P = TypeVar("P")
R = TypeVar("R")
A = TypeVar("A", bound=ApproximationProtocol)


@runtime_checkable
class ErrorMeasureProtocol(Protocol):
    """Which error of an approximation a study records, under which
    column label."""

    def measure(self, approximation: ApproximationProtocol) -> float:
        """The error of ``approximation``."""
        ...

    def label(self) -> str:
        """The study column the error is recorded in."""
        ...


class L2Error:
    r""":math:`\|u^* - u_h\|_{L^2}`, in column ``"l2"``."""

    def measure(self, approximation: ApproximationProtocol) -> float:
        r""":math:`\|u^* - u_h\|_{L^2}`, from the stack."""
        return approximation.l2_error()

    def label(self) -> str:
        """``"l2"``."""
        return "l2"

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class NodalError:
    r""":math:`\max_i |u^*(x_i) - u_h(x_i)|`, in column ``"nodal"``."""

    def measure(self, approximation: ApproximationProtocol) -> float:
        """The largest nodal error, from the stack."""
        return approximation.nodal_error()

    def label(self) -> str:
        """``"nodal"``."""
        return "nodal"

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


@dataclass(frozen=True)
class Column:
    """One measure's errors, one per level, coarsest first."""

    label: str
    values: Tuple[float, ...]


@dataclass(frozen=True)
class Study:
    """A table of levels, coarsest first: each level's sizes, and one
    column of errors per measure.

    Raises
    ------
    ValueError
        If there are no levels or no columns, a column's length is not the
        number of levels, two columns share a label, or an error is
        negative or not finite.
    """

    sizes: Tuple[Sizes, ...]
    columns: Tuple[Column, ...]

    def __post_init__(self) -> None:
        if not self.sizes or not self.columns:
            raise ValueError(
                f"a study needs at least one level and one column; got "
                f"{len(self.sizes)} levels and {len(self.columns)} columns"
            )
        labels = [column.label for column in self.columns]
        if len(set(labels)) != len(labels):
            raise ValueError(f"column labels must be unique, got {labels}")
        for column in self.columns:
            if len(column.values) != len(self.sizes):
                raise ValueError(
                    f"a study needs one error per level; column {column.label!r} "
                    f"has {len(column.values)} for {len(self.sizes)} levels"
                )
            if any(not (math.isfinite(e) and e >= 0.0) for e in column.values):
                raise ValueError(
                    f"errors must be finite and non-negative; column "
                    f"{column.label!r} has {column.values}"
                )

    @classmethod
    def from_records(
        cls, records: Sequence[Tuple[Sizes, Mapping[str, float]]]
    ) -> "Study":
        """The study of plain records, one per level, coarsest first: the
        level's sizes and its errors by column label.

        Raises
        ------
        ValueError
            If the records do not all have the same labels (and as the
            constructor).
        """
        if not records:
            raise ValueError("a study needs at least one level; got no records")
        labels = list(records[0][1])
        for _, errors in records:
            if list(errors) != labels:
                raise ValueError(
                    f"every record needs the labels {labels}, got {list(errors)}"
                )
        return cls(
            tuple(sizes for sizes, _ in records),
            tuple(
                Column(label, tuple(float(errors[label]) for _, errors in records))
                for label in labels
            ),
        )

    def labels(self) -> Tuple[str, ...]:
        """The column labels, in order."""
        return tuple(column.label for column in self.columns)

    def column(self, label: Optional[str] = None) -> Tuple[float, ...]:
        """The errors in column ``label``; by default the only column.

        Raises
        ------
        ValueError
            If no label is given and there are several columns, or no
            column has the label.
        """
        if label is None:
            if len(self.columns) != 1:
                raise ValueError(f"name one of the columns {list(self.labels())}")
            return self.columns[0].values
        for column in self.columns:
            if column.label == label:
                return column.values
        raise ValueError(f"no column {label!r}; the study has {list(self.labels())}")

    def axis(self, axis: Axis) -> Tuple[float, ...]:
        """Each level's size along ``axis``."""
        return tuple(sizes.of(axis) for sizes in self.sizes)


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


def _error_study(h: Tuple[float, ...], errors: Tuple[float, ...]) -> Study:
    """A quantity's errors against mesh sizes, in column ``"error"``."""
    return Study(tuple(Sizes({H: v}) for v in h), (Column("error", errors),))


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
        """:math:`|Q_h - Q|` at each level, in column ``"error"``."""
        return _error_study(study.h, tuple(abs(v - self._value) for v in study.values))

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
        r""":math:`|Q_h - Q_{\mathrm{ref}}|` at each level, in column
        ``"error"``.

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
        return _error_study(study.h, errors)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._value!r}, uncertainty={self._uncertainty!r})"
        )


@dataclass(frozen=True)
class CheckResult:
    """A check's verdict, what it observed (``None`` when it observed no
    number, e.g. a sequence outside the asymptotic range), and why. Its
    fields are a boolean, a finite number or ``None``, and a string, so
    ``dataclasses.asdict`` gives a strict JSON record."""

    passed: bool
    observed: Optional[float]
    message: str


def _series(
    study: Study, column: Optional[str], axis: Axis, last: Optional[int]
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """The sizes along ``axis`` and the errors of ``column``, of the
    finest ``last`` levels (default all)."""
    sizes, errors = study.axis(axis), study.column(column)
    if last is not None:
        sizes, errors = sizes[-last:], errors[-last:]
    return sizes, errors


def _slope(xs: Sequence[float], ys: Sequence[float], what: str) -> float:
    """The least-squares slope of ``ys`` against ``xs``.

    Raises
    ------
    ValueError
        If there are fewer than two points or the ``xs`` are all equal.
    """
    if len(xs) < 2:
        raise ValueError(f"a {what} needs at least two levels, got {len(xs)}")
    xmean, ymean = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - xmean) ** 2 for x in xs)
    if sxx == 0.0:
        raise ValueError(f"a {what} needs distinct sizes, got {tuple(xs)}")
    return sum((x - xmean) * (y - ymean) for x, y in zip(xs, ys)) / sxx


def fitted_rate(
    study: Study,
    column: Optional[str] = None,
    axis: Axis = H,
    last: Optional[int] = None,
) -> float:
    r"""The least-squares slope of :math:`\log e` against the log of the
    sizes along ``axis``.

    Parameters
    ----------
    study : Study
        The study.
    column : str, optional
        The column fitted; default the study's only column.
    axis : Axis
        The sizes fitted against. Default: the mesh size.
    last : int, optional
        Fit only the finest ``last`` levels. Default: all.

    Raises
    ------
    ValueError
        If fewer than two levels are fitted, the fitted sizes are all
        equal, or a fitted error is zero.
    """
    sizes, errors = _series(study, column, axis, last)
    if any(e == 0.0 for e in errors):
        raise ValueError(f"a rate needs nonzero errors, got {errors}")
    return _slope([math.log(v) for v in sizes], [math.log(e) for e in errors], "rate")


@runtime_checkable
class CheckProtocol(Protocol):
    """Judges a study."""

    def verify(self, study: Study) -> CheckResult:
        """The verdict on ``study``."""
        ...


class Converges:
    """The fitted rate is ``rate`` within ``atol``, from both sides.

    A rate well above the expected one is a finding (superconvergence on
    a structured mesh or at nodes, or a measure weaker than intended), so
    it fails unless the case names the rate it allows.

    Parameters
    ----------
    rate : float
        The expected rate.
    atol : float
        How far the fitted rate may fall below ``rate``, or rise above the
        highest allowed rate.
    last : int, optional
        Fit only the finest ``last`` levels, past a pre-asymptotic start.
    axis : Axis
        The sizes fitted against. Default: the mesh size.
    column : str, optional
        The column judged; default the study's only column.
    superconvergence : float, optional
        The highest rate allowed, when the case expects more than
        ``rate`` may be seen. Default: ``rate``.

    Raises
    ------
    ValueError
        If ``superconvergence`` is below ``rate``.
    """

    def __init__(
        self,
        rate: float,
        atol: float = 0.1,
        last: Optional[int] = None,
        axis: Axis = H,
        column: Optional[str] = None,
        superconvergence: Optional[float] = None,
    ) -> None:
        if superconvergence is not None and superconvergence < rate:
            raise ValueError(
                f"superconvergence {superconvergence} is below the rate {rate}"
            )
        self._rate, self._atol, self._last = rate, atol, last
        self._axis, self._column = axis, column
        self._highest = rate if superconvergence is None else superconvergence

    def verify(self, study: Study) -> CheckResult:
        """Pass if the fitted rate lies in ``[rate - atol, highest +
        atol]``."""
        observed = fitted_rate(study, self._column, self._axis, self._last)
        low, high = self._rate - self._atol, self._highest + self._atol
        return CheckResult(
            low <= observed <= high,
            observed,
            f"fitted rate {observed:.3f} in {self._axis.name}, expected "
            f"between {low:.3f} and {high:.3f}; errors {study.column(self._column)}",
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(rate={self._rate}, atol={self._atol}, "
            f"last={self._last}, axis={self._axis!r}, column={self._column!r}, "
            f"superconvergence={self._highest})"
        )


class DecaysExponentially:
    r"""The error falls at least exponentially along ``axis`` (the degree
    by default): the least-squares slope of :math:`\log e` against the
    sizes is at most ``-rate``.

    Spectral methods and p-refinement on smooth solutions converge as
    :math:`e \sim C\exp(-\rho p)`, which no algebraic rate in :math:`h`
    states: the slope of :math:`\log e` against :math:`\log h` grows
    without bound, while against :math:`p` it is the constant
    :math:`-\rho`, with :math:`\rho = \ln r` for a solution analytic
    inside the Bernstein ellipse of parameter :math:`r`. A spectral method
    of :math:`N` nodes has degree :math:`N - 1`, so the slope per node is
    the slope per degree. For an entire solution the convergence is
    faster than any exponential and the slope steepens with :math:`p`;
    ``rate`` is a floor, which that only clears more easily. Fit levels
    above round-off, where the curve flattens.

    Parameters
    ----------
    rate : float
        The decay :math:`\rho` a correct model reaches, per unit of the
        axis.
    last : int, optional
        Fit only the finest ``last`` levels.
    axis : Axis
        The sizes fitted against. Default: the degree.
    column : str, optional
        The column judged; default the study's only column.
    """

    def __init__(
        self,
        rate: float,
        last: Optional[int] = None,
        axis: Axis = DEGREE,
        column: Optional[str] = None,
    ) -> None:
        self._rate, self._last, self._axis, self._column = rate, last, axis, column

    def verify(self, study: Study) -> CheckResult:
        """Pass if the fitted decay reaches ``rate``; fail on a zero error,
        whose logarithm the fit cannot take.

        Raises
        ------
        ValueError
            If fewer than two levels are fitted or the fitted sizes are
            all equal.
        """
        sizes, errors = _series(study, self._column, self._axis, self._last)
        if any(e == 0.0 for e in errors):
            return CheckResult(False, 0.0, f"an error is zero: {errors}")
        observed = -_slope(sizes, [math.log(e) for e in errors], "decay")
        return CheckResult(
            observed >= self._rate,
            observed,
            f"fitted decay {observed:.3f} per {self._axis.name}, expected at "
            f"least {self._rate:.3f}; sizes {sizes}, errors {errors}",
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(rate={self._rate}, last={self._last}, "
            f"axis={self._axis!r}, column={self._column!r})"
        )


class Recovers:
    """Every level's error is at most ``atol``.

    Parameters
    ----------
    atol : float
        The largest error allowed.
    column : str, optional
        The column judged; default the study's only column.
    """

    def __init__(self, atol: float, column: Optional[str] = None) -> None:
        self._atol, self._column = atol, column

    def verify(self, study: Study) -> CheckResult:
        """Pass if the largest error is at most ``atol``."""
        observed = max(study.column(self._column))
        return CheckResult(
            observed <= self._atol,
            observed,
            f"largest error {observed:.3e}, allowed {self._atol:.3e}",
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(atol={self._atol}, column={self._column!r})"


class Stalls:
    """The fitted rate stays below ``below``: the error does not fall as
    a correct model's would. A level with zero error is not a stall.

    Parameters
    ----------
    below : float
        The rate a stall stays under.
    axis : Axis
        The sizes fitted against. Default: the mesh size.
    column : str, optional
        The column judged; default the study's only column.
    """

    def __init__(
        self, below: float, axis: Axis = H, column: Optional[str] = None
    ) -> None:
        self._below, self._axis, self._column = below, axis, column

    def verify(self, study: Study) -> CheckResult:
        """Pass if the fitted rate is below ``below``."""
        errors = study.column(self._column)
        if any(e == 0.0 for e in errors):
            return CheckResult(False, 0.0, f"an error is zero: {errors}")
        observed = fitted_rate(study, self._column, self._axis)
        return CheckResult(
            observed < self._below,
            observed,
            f"fitted rate {observed:.3f} in {self._axis.name}, a stall stays "
            f"below {self._below}; errors {errors}",
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(below={self._below}, axis={self._axis!r}, "
            f"column={self._column!r})"
        )


def run_study(
    problem: P,
    discretizer: DiscretizerProtocol[P, R, A],
    measures: Sequence[ErrorMeasureProtocol],
    requests: Optional[Sequence[R]] = None,
) -> Study:
    """Solve ``problem`` once per level and record every measure.

    Parameters
    ----------
    problem
        The problem.
    discretizer : DiscretizerProtocol
        The stack.
    measures : Sequence[ErrorMeasureProtocol]
        One column each, under its label.
    requests : Sequence, optional
        The levels, coarsest first. Default: every level of the
        discretizer's own family (``discretizer.requests(problem)``).

    Raises
    ------
    TypeError
        If ``discretizer``, a measure, a level's sizes or an approximation
        is of the wrong type.
    ValueError
        If there are no measures or no levels, or two measures share a
        label.
    """
    if not isinstance(discretizer, DiscretizerProtocol):
        raise TypeError(
            "discretizer must satisfy DiscretizerProtocol, got "
            f"{type(discretizer).__name__}"
        )
    for measure in measures:
        if not isinstance(measure, ErrorMeasureProtocol):
            raise TypeError(
                "each measure must satisfy ErrorMeasureProtocol, got "
                f"{type(measure).__name__}"
            )
    if not measures:
        raise ValueError("a study needs at least one measure")
    levels = discretizer.requests(problem) if requests is None else requests
    records: List[Tuple[Sizes, Mapping[str, float]]] = []
    for request in levels:
        sizes = discretizer.sizes(problem, request)
        if not isinstance(sizes, Sizes):
            raise TypeError(f"sizes must be Sizes, got {type(sizes).__name__}")
        approximation = discretizer.solve(problem, request)
        if not isinstance(approximation, ApproximationProtocol):
            raise TypeError(
                "each approximation must satisfy ApproximationProtocol, got "
                f"{type(approximation).__name__}"
            )
        errors = {}
        for measure in measures:
            label = measure.label()
            if label in errors:
                raise ValueError(f"two measures share the label {label!r}")
            errors[label] = measure.measure(approximation)
        records.append((sizes, errors))
    return Study.from_records(records)


def verify(
    problem: P,
    discretizer: DiscretizerProtocol[P, R, A],
    measure: ErrorMeasureProtocol,
    check: CheckProtocol,
    requests: Optional[Sequence[R]] = None,
) -> CheckResult:
    """Run the study of one measure and judge it with ``check``.

    Raises
    ------
    TypeError
        If ``check`` is of the wrong type (and as :func:`run_study`).
    """
    if not isinstance(check, CheckProtocol):
        raise TypeError(f"check must satisfy CheckProtocol, got {type(check).__name__}")
    return check.verify(run_study(problem, discretizer, [measure], requests))


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
            return CheckResult(False, None, failure)
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
