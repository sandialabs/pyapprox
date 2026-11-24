r"""Convergence studies and the checks that judge them.

A study is a table: one row per level, holding the level's
:class:`~pyapprox.pde.verification.protocols.Sizes`, and one labeled
column per measure. A column is of a :class:`Kind`: errors (distances to
the truth, each optionally beside the truth's norm in the same norm) or
values (a quantity of interest, an iteration count). Measures are written
by the stack whose approximations they read (:class:`MeasureProtocol` is
generic in that type), so a new norm or a per-field error is a new
measure in one stack, and a measure paired with another stack is a type
error. :func:`run_study` solves each level once and records every
measure; :meth:`Study.from_records` builds the same table from plain
numbers, e.g. a result file of a code run elsewhere. A check judges one
error column against one
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
- :class:`Recovers`: every level is exact to a tolerance, relative to the
  truth's norm and absolute, for solutions in the discrete space;
- :class:`Stalls`: the slope stays below a bound, the wrong-model control.

Numbers are plain floats: the stack integrated them. Check results hold
only booleans, numbers and strings, so they serialize to JSON.

A quantity of interest is a measure like any other: :func:`quantity_value`
records its value at each level in a value column, from the same solves
as the error columns. A truth appends that column's errors to the study:
:class:`ExactValue` (from the exact solution, e.g. by
:func:`pyapprox.pde.verification.exact.exact_value`) or
:class:`ReferenceValue` (a much finer solve or published data, with its
own uncertainty), after which every check above applies. Without either,
:func:`extrapolate` estimates the order and the limit from the values
alone along a shrinking axis (Richardson), and :class:`ObservedOrder`
judges that order.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

from pyapprox.pde.verification.protocols import (
    DEGREE,
    Axis,
    DiscretizerProtocol,
    H,
    Sizes,
)
from pyapprox.pde.verification.quantities import QuantityProtocol

P = TypeVar("P")
R = TypeVar("R")
A = TypeVar("A")
A_contra = TypeVar("A_contra", contravariant=True)


class Kind(Enum):
    """What a study column holds."""

    ERROR = "error"
    """Distances to the truth: finite and non-negative."""
    VALUE = "value"
    """Values computed from the solution (a quantity of interest, an
    iteration count): finite, of either sign."""


@dataclass(frozen=True)
class Measurement:
    """One level's number for one column, and for an error the norm of
    the truth it is measured against (in the same norm), which a relative
    tolerance is taken of."""

    value: float
    norm: Optional[float] = None


@runtime_checkable
class MeasureProtocol(Protocol[A_contra]):
    """What a study records of a stack's approximation, under which
    column label.

    Generic in the approximation it reads, so a measure is written by the
    stack that owns the approximation, and pairing it with another
    stack's discretizer is a type error."""

    def label(self) -> str:
        """The study column the measurement is recorded in."""
        ...

    def kind(self) -> Kind:
        """Whether the column holds errors or values."""
        ...

    def measure(self, approximation: A_contra) -> Measurement:
        """The measurement of ``approximation``."""
        ...


class Measure(Generic[A]):
    """A measure from a label, a reader of the stack's approximation and
    a column kind: what a stack's measure factories return, so a new norm
    or a new value column is a function, not a class.

    Parameters
    ----------
    label : str
        The study column.
    read : Callable
        The measurement of one approximation.
    kind : Kind
        Default: :attr:`Kind.ERROR`.

    Raises
    ------
    ValueError
        If ``label`` is empty.
    TypeError
        If ``read`` is not callable.
    """

    def __init__(
        self,
        label: str,
        read: Callable[[A], Measurement],
        kind: Kind = Kind.ERROR,
    ) -> None:
        if not label:
            raise ValueError("a measure needs a non-empty label")
        if not callable(read):
            raise TypeError(f"read must be callable, got {type(read).__name__}")
        self._label, self._read, self._kind = label, read, kind

    def label(self) -> str:
        """The study column."""
        return self._label

    def kind(self) -> Kind:
        """Whether the column holds errors or values."""
        return self._kind

    def measure(self, approximation: A) -> Measurement:
        """The measurement of ``approximation``."""
        return self._read(approximation)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._label!r}, kind={self._kind})"


def field_label(norm: str, fields: Optional[Sequence[str]]) -> str:
    """``norm`` for all fields, ``"norm(f1,f2)"`` for some: the default
    label of a stack's per-field error measures.

    Raises
    ------
    ValueError
        If ``fields`` is given empty.
    """
    if fields is None:
        return norm
    if not fields:
        raise ValueError("name at least one field, or none for all")
    return f"{norm}({','.join(fields)})"


@dataclass(frozen=True)
class Column:
    """One measure's numbers, one per level, coarsest first, and for an
    error column optionally the truth's norm at each level."""

    label: str
    values: Tuple[float, ...]
    kind: Kind = Kind.ERROR
    norms: Optional[Tuple[float, ...]] = None


@dataclass(frozen=True)
class Study:
    """A table of levels, coarsest first: each level's sizes, and one
    column per measure.

    Raises
    ------
    ValueError
        If there are no levels or no columns, a column's values or norms
        are not one per level, two columns share a label, a value is not
        finite, an error or a norm is negative, or a value column has
        norms.
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
            for what, numbers in (("value", column.values), ("norm", column.norms)):
                if numbers is not None and len(numbers) != len(self.sizes):
                    raise ValueError(
                        f"a study needs one {what} per level; column "
                        f"{column.label!r} has {len(numbers)} for "
                        f"{len(self.sizes)} levels"
                    )
            if any(not math.isfinite(v) for v in column.values):
                raise ValueError(
                    f"values must be finite; column {column.label!r} has "
                    f"{column.values}"
                )
            if column.kind is Kind.ERROR and any(e < 0.0 for e in column.values):
                raise ValueError(
                    f"errors must be non-negative; column {column.label!r} has "
                    f"{column.values}"
                )
            if column.norms is not None:
                if column.kind is not Kind.ERROR:
                    raise ValueError(
                        f"only an error column has norms; {column.label!r} holds values"
                    )
                if any(not (math.isfinite(v) and v >= 0.0) for v in column.norms):
                    raise ValueError(
                        f"norms must be finite and non-negative; column "
                        f"{column.label!r} has {column.norms}"
                    )

    @classmethod
    def from_records(
        cls,
        records: Sequence[Tuple[Sizes, Mapping[str, Union[float, Measurement]]]],
        kinds: Optional[Mapping[str, Kind]] = None,
    ) -> "Study":
        """The study of plain records, one per level, coarsest first: the
        level's sizes and its numbers by column label, each a float or a
        :class:`Measurement` carrying its truth's norm.

        Parameters
        ----------
        records : Sequence
            One ``(sizes, numbers)`` pair per level.
        kinds : Mapping[str, Kind], optional
            Each column's kind; a column not named holds errors.

        Raises
        ------
        ValueError
            If the records do not all have the same labels, a column gives
            norms at some levels only, or ``kinds`` names an unknown
            column (and as the constructor).
        """
        if not records:
            raise ValueError("a study needs at least one level; got no records")
        labels = list(records[0][1])
        for _, numbers in records:
            if list(numbers) != labels:
                raise ValueError(
                    f"every record needs the labels {labels}, got {list(numbers)}"
                )
        chosen = dict(kinds or {})
        unknown = set(chosen) - set(labels)
        if unknown:
            raise ValueError(f"kinds names unknown columns {sorted(unknown)}")
        columns = []
        for label in labels:
            measured = [
                entry if isinstance(entry, Measurement) else Measurement(float(entry))
                for entry in (numbers[label] for _, numbers in records)
            ]
            norms = [m.norm for m in measured]
            given = [n for n in norms if n is not None]
            if given and len(given) != len(norms):
                raise ValueError(
                    f"column {label!r} gives norms at some levels only: {norms}"
                )
            columns.append(
                Column(
                    label,
                    tuple(float(m.value) for m in measured),
                    chosen.get(label, Kind.ERROR),
                    tuple(float(n) for n in given) if given else None,
                )
            )
        return cls(tuple(sizes for sizes, _ in records), tuple(columns))

    def labels(self) -> Tuple[str, ...]:
        """The column labels, in order."""
        return tuple(column.label for column in self.columns)

    def find(self, label: Optional[str] = None) -> Column:
        """Column ``label``; by default the only column.

        Raises
        ------
        ValueError
            If no label is given and there are several columns, or no
            column has the label.
        """
        if label is None:
            if len(self.columns) != 1:
                raise ValueError(f"name one of the columns {list(self.labels())}")
            return self.columns[0]
        for column in self.columns:
            if column.label == label:
                return column
        raise ValueError(f"no column {label!r}; the study has {list(self.labels())}")

    def column(self, label: Optional[str] = None) -> Tuple[float, ...]:
        """The numbers in column ``label``; by default the only column.

        Raises
        ------
        ValueError
            If no label is given and there are several columns, or no
            column has the label.
        """
        return self.find(label).values

    def errors(self, label: Optional[str] = None) -> Tuple[float, ...]:
        """The errors in column ``label``; by default the only column.

        Raises
        ------
        ValueError
            As :meth:`column`, or if the column holds values.
        """
        column = self.find(label)
        if column.kind is not Kind.ERROR:
            raise ValueError(
                f"column {column.label!r} holds values, not errors; turn it into "
                "errors against a truth first"
            )
        return column.values

    def values(self, label: Optional[str] = None) -> Tuple[float, ...]:
        """The values in column ``label``; by default the only column.

        Raises
        ------
        ValueError
            As :meth:`column`, or if the column holds errors.
        """
        column = self.find(label)
        if column.kind is not Kind.VALUE:
            raise ValueError(
                f"column {column.label!r} holds errors, not values; Richardson "
                "and the truths read a quantity's signed values"
            )
        return column.values

    def with_column(self, column: Column) -> "Study":
        """This study with ``column`` appended.

        Raises
        ------
        ValueError
            As the constructor (a duplicate label, a length mismatch).
        """
        return Study(self.sizes, self.columns + (column,))

    def last(self, nlevels: Optional[int]) -> "Study":
        """The finest ``nlevels`` levels (default all), every column.

        Raises
        ------
        ValueError
            If ``nlevels`` is not positive.
        """
        if nlevels is None:
            return self
        if nlevels < 1:
            raise ValueError(f"keep at least one level, got {nlevels}")
        return Study(
            self.sizes[-nlevels:],
            tuple(
                Column(
                    c.label,
                    c.values[-nlevels:],
                    c.kind,
                    None if c.norms is None else c.norms[-nlevels:],
                )
                for c in self.columns
            ),
        )

    def norms(self, label: Optional[str] = None) -> Optional[Tuple[float, ...]]:
        """The truth's norm at each level of error column ``label``, or
        ``None`` if the column has none.

        Raises
        ------
        ValueError
            As :meth:`errors`.
        """
        self.errors(label)
        return self.find(label).norms

    def axis(self, axis: Axis) -> Tuple[float, ...]:
        """Each level's size along ``axis``."""
        return tuple(sizes.of(axis) for sizes in self.sizes)


@runtime_checkable
class QuantityReaderProtocol(Protocol):
    """An approximation that evaluates a quantity of interest from its own
    solution, with its own quadrature and geometry."""

    def quantity(self, quantity: QuantityProtocol) -> float:
        """The value of ``quantity``."""
        ...


def quantity_value(
    quantity: QuantityProtocol, label: Optional[str] = None
) -> Measure[QuantityReaderProtocol]:
    """A value column of ``quantity`` at each level, read from any stack
    whose approximations evaluate quantities. A truth
    (:class:`ExactValue`, :class:`ReferenceValue`) turns it into errors;
    :class:`ObservedOrder` judges it without one.

    Parameters
    ----------
    quantity : QuantityProtocol
        What to evaluate.
    label : str, optional
        The column. Default: ``repr(quantity)``.

    Raises
    ------
    TypeError
        If ``quantity`` does not satisfy ``QuantityProtocol``.
    """
    if not isinstance(quantity, QuantityProtocol):
        raise TypeError(
            f"quantity must satisfy QuantityProtocol, got {type(quantity).__name__}"
        )
    return Measure(
        label or repr(quantity),
        lambda approximation: Measurement(approximation.quantity(quantity)),
        Kind.VALUE,
    )


@runtime_checkable
class TruthProtocol(Protocol):
    """What a value column is compared with."""

    def errors(
        self, study: Study, column: Optional[str] = None, label: Optional[str] = None
    ) -> Study:
        """``study`` with an error column: the distance of each value in
        ``column`` to the truth."""
        ...


def _with_errors(
    study: Study,
    column: Optional[str],
    label: Optional[str],
    errors: Sequence[float],
    norm: float,
) -> Study:
    source = study.find(column).label
    return study.with_column(
        Column(
            label or f"{source} error",
            tuple(errors),
            Kind.ERROR,
            (abs(norm),) * len(errors),
        )
    )


class ExactValue:
    """The quantity's exact value; errors are distances to it, beside its
    magnitude as the norm, so :class:`Recovers` can be relative.

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

    def errors(
        self, study: Study, column: Optional[str] = None, label: Optional[str] = None
    ) -> Study:
        """``study`` with :math:`|Q_h - Q|` of value column ``column``
        (default the only column) appended, labeled ``label`` (default
        ``"<column> error"``).

        Raises
        ------
        ValueError
            If the column holds errors (and as :meth:`Study.with_column`).
        """
        values = study.values(column)
        return _with_errors(
            study, column, label, [abs(v - self._value) for v in values], self._value
        )

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

    def errors(
        self, study: Study, column: Optional[str] = None, label: Optional[str] = None
    ) -> Study:
        r"""``study`` with :math:`|Q_h - Q_{\mathrm{ref}}|` of value column
        ``column`` (default the only column) appended, labeled ``label``
        (default ``"<column> error"``).

        Raises
        ------
        ValueError
            If an error is not above the reference's uncertainty, or the
            column holds errors (and as :meth:`Study.with_column`).
        """
        errors = [abs(v - self._value) for v in study.values(column)]
        unresolved = [e for e in errors if e <= self._uncertainty]
        if unresolved:
            raise ValueError(
                f"errors {unresolved} are within the reference's uncertainty "
                f"{self._uncertainty:.3e}; drop those levels or use a finer "
                "reference"
            )
        return _with_errors(study, column, label, errors, self._value)

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
    sizes, errors = study.axis(axis), study.errors(column)
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
            f"between {low:.3f} and {high:.3f}; errors {study.errors(self._column)}",
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
    r"""Every level's error :math:`e` is at most
    :math:`\mathrm{rtol}\,\|u^*\| + \mathrm{atol}`, with :math:`\|u^*\|`
    the norm the column records beside each error.

    The relative part makes the tolerance independent of the solution's
    scale; the absolute part lets a field that is zero (a pressure that
    vanishes) pass. Run it on the coarse levels, where round-off has not
    grown with the number of unknowns.

    Parameters
    ----------
    atol : float
        The absolute part. Default: 0.
    rtol : float
        The relative part, at least the solver's tolerance. Default: 0.
    column : str, optional
        The column judged; default the study's only column.

    Raises
    ------
    ValueError
        If a tolerance is negative or not finite, or both are zero.
    """

    def __init__(
        self, atol: float = 0.0, rtol: float = 0.0, column: Optional[str] = None
    ) -> None:
        for name, value in (("atol", atol), ("rtol", rtol)):
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be non-negative and finite, got {value}")
        if atol == 0.0 and rtol == 0.0:
            raise ValueError("Recovers needs a positive atol or rtol")
        self._atol, self._rtol, self._column = atol, rtol, column

    def verify(self, study: Study) -> CheckResult:
        """Pass if every level's error is within its tolerance; the
        observed number is the largest ratio of error to tolerance.

        Raises
        ------
        ValueError
            If ``rtol`` is positive and the column records no norms.
        """
        errors, norms = study.errors(self._column), study.norms(self._column)
        if self._rtol > 0.0 and norms is None:
            raise ValueError(
                "a relative tolerance needs the norms of the truth; the column "
                "records none"
            )
        scales = norms if norms is not None else (0.0,) * len(errors)
        allowed = [self._rtol * n + self._atol for n in scales]
        ratios = [
            e / a if a > 0.0 else (0.0 if e == 0.0 else math.inf)
            for e, a in zip(errors, allowed)
        ]
        worst = max(ratios)
        return CheckResult(
            worst <= 1.0,
            worst if math.isfinite(worst) else None,
            f"errors {errors}, allowed {tuple(allowed)}",
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(atol={self._atol}, rtol={self._rtol}, "
            f"column={self._column!r})"
        )


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
        errors = study.errors(self._column)
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
    measures: Sequence[MeasureProtocol[A]],
    requests: Optional[Sequence[R]] = None,
) -> Study:
    """Solve ``problem`` once per level and record every measure.

    Parameters
    ----------
    problem
        The problem.
    discretizer : DiscretizerProtocol
        The stack.
    measures : Sequence[MeasureProtocol]
        Measures of the stack's approximations, one column each, under
        its label and of its kind.
    requests : Sequence, optional
        The levels, coarsest first. Default: every level of the
        discretizer's own family (``discretizer.requests(problem)``).

    Raises
    ------
    TypeError
        If ``discretizer``, a measure, a level's sizes or a measurement is
        of the wrong type.
    ValueError
        If there are no measures or no levels, two measures share a label,
        or a value measure reports a norm (and as the :class:`Study`).
    """
    if not isinstance(discretizer, DiscretizerProtocol):
        raise TypeError(
            "discretizer must satisfy DiscretizerProtocol, got "
            f"{type(discretizer).__name__}"
        )
    for measure in measures:
        if not isinstance(measure, MeasureProtocol):
            raise TypeError(
                "each measure must satisfy MeasureProtocol, got "
                f"{type(measure).__name__}"
            )
    if not measures:
        raise ValueError("a study needs at least one measure")
    labels = [measure.label() for measure in measures]
    if len(set(labels)) != len(labels):
        raise ValueError(f"measure labels must be unique, got {labels}")
    levels = discretizer.requests(problem) if requests is None else requests
    records: List[Tuple[Sizes, Mapping[str, Union[float, Measurement]]]] = []
    for request in levels:
        sizes = discretizer.sizes(problem, request)
        if not isinstance(sizes, Sizes):
            raise TypeError(f"sizes must be Sizes, got {type(sizes).__name__}")
        approximation = discretizer.solve(problem, request)
        numbers: Dict[str, Union[float, Measurement]] = {}
        for label, measure in zip(labels, measures):
            measurement = measure.measure(approximation)
            if not isinstance(measurement, Measurement):
                raise TypeError(
                    f"measure {label!r} must return a Measurement, got "
                    f"{type(measurement).__name__}"
                )
            numbers[label] = measurement
        records.append((sizes, numbers))
    return Study.from_records(
        records, {label: measure.kind() for label, measure in zip(labels, measures)}
    )


def verify(
    problem: P,
    discretizer: DiscretizerProtocol[P, R, A],
    measure: MeasureProtocol[A],
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


def extrapolate(
    study: Study,
    column: Optional[str] = None,
    axis: Axis = H,
    safety: float = 1.25,
) -> Extrapolation:
    r"""Richardson extrapolation of value column ``column`` from the three
    finest levels.

    With a constant refinement ratio :math:`r = s_1/s_2 = s_2/s_3` of the
    sizes along ``axis`` and values :math:`Q_1, Q_2, Q_3` (finest last),
    :math:`p = \log((Q_2 - Q_1)/(Q_3 - Q_2))/\log r` and :math:`Q^* = Q_3 +
    (Q_3 - Q_2)/(r^p - 1)`. The axis must shrink with refinement (a mesh
    size or a time step), so the order is per unit of its logarithm.

    Parameters
    ----------
    study : Study
        At least three levels.
    column : str, optional
        A value column; default the study's only column.
    axis : Axis
        The sizes. Default: the mesh size.
    safety : float
        Roache's factor of safety :math:`F_s`; 1.25 for three levels.

    Raises
    ------
    ValueError
        If there are fewer than three levels, the column holds errors, the
        ratio of the finest three sizes is not constant or not above one,
        or their differences are not in the asymptotic range (zero,
        oscillatory or not shrinking).
    """
    values, sizes = study.values(column), study.axis(axis)
    if len(values) < 3:
        raise ValueError(f"extrapolation needs three levels, got {len(values)}")
    s1, s2, s3 = sizes[-3:]
    r = s1 / s2
    if not r > 1.0 or abs(s2 / s3 - r) > _RATIO_RTOL * r:
        raise ValueError(
            f"extrapolation needs a constant refinement ratio above one; the "
            f"finest sizes along {axis.name} {(s1, s2, s3)} give {s1 / s2} and "
            f"{s2 / s3}"
        )
    finest = values[-3:]
    failure = _asymptotic_failure(finest)
    if failure is not None:
        raise ValueError(f"cannot extrapolate: {failure}")
    q1, q2, q3 = finest
    growth = (q2 - q1) / (q3 - q2)
    order = math.log(growth) / math.log(r)
    limit = q3 + (q3 - q2) / (growth - 1.0)
    return Extrapolation(order, limit, safety * abs(q3 - q2) / (growth - 1.0))


class ObservedOrder:
    """Richardson's observed order of a value column is at least ``order -
    atol``: a check without a truth.

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
    axis : Axis
        The sizes, shrinking with refinement. Default: the mesh size.
    column : str, optional
        The value column judged; default the study's only column.

    Raises
    ------
    ValueError
        If ``last`` is below three.
    """

    def __init__(
        self,
        order: float,
        atol: float = 0.1,
        last: Optional[int] = None,
        axis: Axis = H,
        column: Optional[str] = None,
    ) -> None:
        if last is not None and last < 3:
            raise ValueError(f"an observed order needs three levels, got last={last}")
        self._order, self._atol, self._last = order, atol, last
        self._axis, self._column = axis, column

    def verify(self, study: Study) -> CheckResult:
        """Pass if the sequence is asymptotic and its order reaches
        ``order - atol``; the message gives the limit and the GCI band.

        Raises
        ------
        ValueError
            As :func:`extrapolate`, for too few levels, an error column or
            a ratio that is not constant.
        """
        study = study.last(self._last)
        values = study.values(self._column)
        failure = _asymptotic_failure(values)
        if failure is not None and len(values) >= 3:
            return CheckResult(False, None, failure)
        estimate = extrapolate(study, self._column, self._axis)
        passed = estimate.order >= self._order - self._atol
        return CheckResult(
            passed,
            estimate.order,
            f"observed order {estimate.order:.3f} in {self._axis.name}, expected "
            f"at least {self._order - self._atol:.3f}; limit "
            f"{estimate.limit:.12g} (GCI band {estimate.gci:.3e}); values {values}",
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(order={self._order}, atol={self._atol}, "
            f"last={self._last}, axis={self._axis!r}, column={self._column!r})"
        )
