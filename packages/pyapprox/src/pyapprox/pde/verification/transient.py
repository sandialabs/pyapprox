r"""The time axis of a verification study.

A transient stack names a level by a :class:`TransientLevel`: its own
space request and a uniform :class:`TimeGrid`. It reports both the mesh
size and the time step as the level's
:class:`~pyapprox.pde.verification.protocols.Sizes`, so every check reads
either axis unchanged (``Converges(r, axis=DT)``). Its approximation is a
series of the steady approximation type, one per time
(:class:`SeriesProtocol`), read one step at a time.

Plans are request lists, passed to
:func:`~pyapprox.pde.verification.study.run_study` as ``requests``:

- :func:`refine_space` keeps one grid across the space levels (a fine
  time step isolates the spatial order :math:`p`);
- :func:`refine_time` divides the time step at one space level (a fine
  space level isolates the temporal order :math:`r`);
- :func:`refine_jointly` sets :math:`\Delta t \propto h^q` from each
  level's mesh size, read before any solve; with
  :math:`e \approx A h^p + B \Delta t^r` the rate in :math:`h` is
  :math:`\min(p, q r)`, so :math:`q = p / r` keeps both terms.

Lists concatenate, so one study can hold several plans' levels; that is
what :func:`fit_joint` needs to separate :math:`A` from :math:`B`, which a
joint plan alone makes proportional.

A steady measure becomes a measure of a series by :func:`max_over_steps`
(the error the convergence theorems bound, the primary transient measure)
or :func:`final_time` (which can pass near zero for an error that changes
sign as it crosses the final time).
"""

import math
from dataclasses import dataclass
from typing import (
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    runtime_checkable,
)

from pyapprox.pde.verification.protocols import DT, Axis, H, Sizes
from pyapprox.pde.verification.study import (
    Kind,
    Measure,
    Measurement,
    MeasureProtocol,
    Study,
)

A = TypeVar("A")
A_co = TypeVar("A_co", covariant=True)
S = TypeVar("S")
P = TypeVar("P")
P_contra = TypeVar("P_contra", contravariant=True)
S_contra = TypeVar("S_contra", contravariant=True)

_DIVIDES_RTOL = 1e-10


@dataclass(frozen=True)
class TimeGrid:
    """A uniform grid of times from ``t0`` to ``tfinal`` in steps of
    ``dt``, which must divide the interval.

    Raises
    ------
    ValueError
        If a time is not finite, ``tfinal`` is not after ``t0``, ``dt`` is
        not positive and finite, or ``dt`` does not divide
        ``tfinal - t0`` to a relative 1e-10.
    """

    t0: float
    tfinal: float
    dt: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.t0) and math.isfinite(self.tfinal)):
            raise ValueError(
                f"times must be finite, got t0={self.t0}, tfinal={self.tfinal}"
            )
        if not self.tfinal > self.t0:
            raise ValueError(
                f"tfinal must be after t0, got t0={self.t0}, tfinal={self.tfinal}"
            )
        if not (math.isfinite(self.dt) and self.dt > 0.0):
            raise ValueError(f"dt must be positive and finite, got {self.dt}")
        span = self.tfinal - self.t0
        nsteps = round(span / self.dt)
        if nsteps < 1 or abs(nsteps * self.dt - span) > _DIVIDES_RTOL * span:
            raise ValueError(
                f"dt={self.dt} does not divide the interval [{self.t0}, {self.tfinal}]"
            )

    def nsteps(self) -> int:
        """The number of steps."""
        return round((self.tfinal - self.t0) / self.dt)

    def times(self) -> Tuple[float, ...]:
        """The ``nsteps() + 1`` times, ``t0`` and ``tfinal`` included
        exactly."""
        nsteps = self.nsteps()
        return tuple(self.t0 + n * self.dt for n in range(nsteps)) + (self.tfinal,)


@dataclass(frozen=True)
class TransientLevel(Generic[S]):
    """A request for one level of a transient study: the stack's own space
    request (e.g. a :class:`~pyapprox.pde.verification.protocols.
    SpaceLevel`) and the time grid."""

    space: S
    grid: TimeGrid


@runtime_checkable
class SeriesProtocol(Protocol[A_co]):
    """A transient approximation: one steady approximation per time of
    its grid, the initial time included, built when asked for."""

    def ntimes(self) -> int:
        """The number of times, the initial one included."""
        ...

    def times(self) -> Sequence[float]:
        """The times, first to last."""
        ...

    def at(self, n: int) -> A_co:
        """The approximation at time ``n``, counted from the initial
        time."""
        ...


@runtime_checkable
class SpaceSizesProtocol(Protocol[P_contra, S_contra]):
    """A transient stack that sizes a space level without a time grid, so
    a joint plan can choose each level's time step from it."""

    def space_sizes(self, problem: P_contra, space: S_contra) -> Sizes:
        """The sizes of space level ``space``, without solving."""
        ...


def refine_space(spaces: Sequence[S], grid: TimeGrid) -> List[TransientLevel[S]]:
    """One level per space request, each on ``grid``.

    Raises
    ------
    ValueError
        If ``spaces`` is empty.
    TypeError
        If ``grid`` is not a :class:`TimeGrid`.
    """
    if not spaces:
        raise ValueError("a plan needs at least one space level")
    if not isinstance(grid, TimeGrid):
        raise TypeError(f"grid must be a TimeGrid, got {type(grid).__name__}")
    return [TransientLevel(space, grid) for space in spaces]


def refine_time(
    space: S, grid: TimeGrid, nlevels: int, ratio: int = 2
) -> List[TransientLevel[S]]:
    """``nlevels`` levels at space request ``space``, the first on
    ``grid`` and each next one's step ``ratio`` times smaller.

    Raises
    ------
    ValueError
        If ``nlevels`` is below 2 or ``ratio`` below 2.
    TypeError
        If ``grid`` is not a :class:`TimeGrid`.
    """
    if nlevels < 2:
        raise ValueError(f"a refinement needs at least two levels, got {nlevels}")
    if ratio < 2:
        raise ValueError(f"the step ratio is an integer of at least 2, got {ratio}")
    if not isinstance(grid, TimeGrid):
        raise TypeError(f"grid must be a TimeGrid, got {type(grid).__name__}")
    return [
        TransientLevel(space, TimeGrid(grid.t0, grid.tfinal, grid.dt / ratio**level))
        for level in range(nlevels)
    ]


def refine_jointly(
    problem: P,
    sizer: SpaceSizesProtocol[P, S],
    spaces: Sequence[S],
    q: float,
    dt0: float,
    t0: float,
    tfinal: float,
    axis: Axis = H,
) -> List[TransientLevel[S]]:
    r"""One level per space request, with :math:`\Delta t = \Delta t_0
    (h / h_0)^q` from each level's size :math:`h` along ``axis``, read
    from ``sizer`` before any solve, and rounded down to divide the
    interval. The level's sizes report the step it runs.

    Parameters
    ----------
    problem
        The problem the levels are sized for.
    sizer : SpaceSizesProtocol
        The stack.
    spaces : Sequence
        Its space requests, coarsest first.
    q : float
        The exponent; :math:`p / r` keeps both error terms.
    dt0 : float
        The step at the first level.
    t0, tfinal : float
        The interval.
    axis : Axis
        The space size :math:`h`. Default: the mesh size.

    Raises
    ------
    TypeError
        If ``sizer`` does not satisfy :class:`SpaceSizesProtocol`.
    ValueError
        If ``spaces`` is empty, ``q`` or ``dt0`` is not positive and
        finite, or the sizes do not shrink from level to level (and as
        :class:`TimeGrid`).
    """
    if not isinstance(sizer, SpaceSizesProtocol):
        raise TypeError(
            f"sizer must satisfy SpaceSizesProtocol, got {type(sizer).__name__}"
        )
    if not spaces:
        raise ValueError("a plan needs at least one space level")
    for name, value in (("q", q), ("dt0", dt0)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be positive and finite, got {value}")
    hs = [sizer.space_sizes(problem, space).of(axis) for space in spaces]
    if any(fine >= coarse for coarse, fine in zip(hs, hs[1:])):
        raise ValueError(f"a joint plan needs shrinking sizes, got {hs}")
    span = tfinal - t0
    levels = []
    for space, h in zip(spaces, hs):
        target = dt0 * (h / hs[0]) ** q
        nsteps = max(1, math.ceil(span / target * (1.0 - _DIVIDES_RTOL)))
        levels.append(TransientLevel(space, TimeGrid(t0, tfinal, span / nsteps)))
    return levels


def _checked(label: str, measurement: object) -> Measurement:
    if not isinstance(measurement, Measurement):
        raise TypeError(
            f"measure {label!r} must return a Measurement, got "
            f"{type(measurement).__name__}"
        )
    return measurement


def _steady(measure: MeasureProtocol[A]) -> str:
    if not isinstance(measure, MeasureProtocol):
        raise TypeError(
            f"measure must satisfy MeasureProtocol, got {type(measure).__name__}"
        )
    return measure.label()


def max_over_steps(
    measure: MeasureProtocol[A], label: Optional[str] = None
) -> Measure[SeriesProtocol[A]]:
    r"""The largest error of ``measure`` over every time of a series, the
    initial one included: the error in :math:`L^\infty(0, T; X)` that the
    convergence theorems bound. Its norm is the largest of the truth's
    norms over the times, so a relative tolerance is relative in the same
    norm. Steps are read one at a time.

    Parameters
    ----------
    measure : MeasureProtocol
        A steady error measure.
    label : str, optional
        The column. Default: ``"max_t <label>"``.

    Raises
    ------
    TypeError
        If ``measure`` does not satisfy ``MeasureProtocol``.
    ValueError
        If ``measure`` holds values; when measuring, if the series has no
        times or gives norms at some times only.
    """
    steady = _steady(measure)
    if measure.kind() is not Kind.ERROR:
        raise ValueError(
            f"the maximum over steps is of errors; {steady!r} holds values"
        )

    def read(series: SeriesProtocol[A]) -> Measurement:
        ntimes = series.ntimes()
        if ntimes < 1:
            raise ValueError("a series needs at least one time")
        value, norms = 0.0, []
        for n in range(ntimes):
            step = _checked(steady, measure.measure(series.at(n)))
            value = max(value, step.value)
            norms.append(step.norm)
        given = [norm for norm in norms if norm is not None]
        if given and len(given) != len(norms):
            raise ValueError(f"measure {steady!r} gives norms at some times only")
        return Measurement(value, max(given) if given else None)

    return Measure(label or f"max_t {steady}", read, Kind.ERROR)


def final_time(
    measure: MeasureProtocol[A], label: Optional[str] = None
) -> Measure[SeriesProtocol[A]]:
    """``measure`` at the last time of a series, with its own norm: an
    error or a value (a quantity of interest at the final time). An error
    that changes sign as it crosses the final time is near zero there, so
    prefer :func:`max_over_steps` for convergence.

    Parameters
    ----------
    measure : MeasureProtocol
        A steady measure.
    label : str, optional
        The column. Default: ``"<label>(T)"``.

    Raises
    ------
    TypeError
        If ``measure`` does not satisfy ``MeasureProtocol``.
    ValueError
        When measuring, if the series has no times.
    """
    steady = _steady(measure)

    def read(series: SeriesProtocol[A]) -> Measurement:
        ntimes = series.ntimes()
        if ntimes < 1:
            raise ValueError("a series needs at least one time")
        return _checked(steady, measure.measure(series.at(ntimes - 1)))

    return Measure(label or f"{steady}(T)", read, measure.kind())


@dataclass(frozen=True)
class JointFit:
    r"""The constants of :math:`e \approx A h^p + B \Delta t^r` fitted with
    :math:`p` and :math:`r` fixed, both non-negative, and the largest
    relative misfit :math:`|\hat e / e - 1|` over the levels. Its fields
    are finite numbers, so ``dataclasses.asdict`` gives a strict JSON
    record."""

    space: float
    time: float
    misfit: float


def fit_joint(
    study: Study,
    p: float,
    r: float,
    column: Optional[str] = None,
    space: Axis = H,
    time: Axis = DT,
    last: Optional[int] = None,
) -> JointFit:
    r"""Fit :math:`e \approx A h^p + B \Delta t^r` to error column
    ``column``, minimizing the relative misfit, so a term hidden by a joint
    rate shows as a constant.

    The levels must vary :math:`h^p` and :math:`\Delta t^r`
    independently: a joint plan with :math:`q = p / r` makes them
    proportional, so fit a study whose requests concatenate it with a
    space-only or time-only plan.

    Parameters
    ----------
    study : Study
        The study.
    p, r : float
        The spatial and temporal orders.
    column : str, optional
        The error column; default the study's only column.
    space, time : Axis
        The axes of :math:`h` and :math:`\Delta t`.
    last : int, optional
        Fit only the finest ``last`` levels. Default: all.

    Raises
    ------
    ValueError
        If ``p`` or ``r`` is not positive and finite, there are fewer than
        two levels, an error is zero, or the two terms are proportional
        across the levels.
    """
    for name, value in (("p", p), ("r", r)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be positive and finite, got {value}")
    chosen = study.last(last)
    errors = chosen.errors(column)
    if len(errors) < 2:
        raise ValueError(f"a joint fit needs at least two levels, got {len(errors)}")
    if any(e == 0.0 for e in errors):
        raise ValueError(f"a joint fit needs nonzero errors, got {errors}")
    xs = [h**p / e for h, e in zip(chosen.axis(space), errors)]
    ys = [dt**r / e for dt, e in zip(chosen.axis(time), errors)]
    sxx, syy = sum(x * x for x in xs), sum(y * y for y in ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sx, sy = sum(xs), sum(ys)
    det = sxx * syy - sxy * sxy
    if det <= 1e-10 * sxx * syy:
        raise ValueError(
            "h^p and dt^r are proportional across the levels; add levels that "
            "refine one alone"
        )
    a, b = (syy * sx - sxy * sy) / det, (sxx * sy - sxy * sx) / det
    if a < 0.0 or b < 0.0:
        candidates = [(sx / sxx, 0.0), (0.0, sy / syy)]
        a, b = min(
            candidates,
            key=lambda ab: sum(
                (ab[0] * x + ab[1] * y - 1.0) ** 2 for x, y in zip(xs, ys)
            ),
        )
    misfit = max(abs(a * x + b * y - 1.0) for x, y in zip(xs, ys))
    return JointFit(a, b, misfit)
