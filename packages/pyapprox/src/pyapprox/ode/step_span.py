r"""Where a time step lies: its size and the times it reads, exactly.

A step's size is given, never recovered from times: subtracting two
floating-point times gives back a slightly different size from step to
step, so a method would integrate with a size other than the one asked
for, and two steps of the same size would not be recognized as such (a
factorization of :math:`M - \Delta t\, a_{ii} J` could not be reused, and
a multistep method's coefficients would vary in their last bits).

So sizes are the stored truth and times are derived from them by one
rule, :math:`t_{k+1} = t_k + h_k`, applied in order from the start of a
trajectory. A :class:`StepSpan` holds the time of the oldest state a step
reads and the sizes from there on, oldest first; every time it reports is
computed by adding the sizes forward, in the order the trajectory was
built, so it equals the time a store records for that state bit for bit.
Nothing subtracts times, except :meth:`StepSpan.from_times`, which checks
that adding the sizes back reproduces every time exactly.

Times derived this way drift from :math:`t_0 + k h` by about
:math:`k\,\varepsilon` relative (1.7e-8 at :math:`t = 1000` after
:math:`10^6` steps of 0.001); a time that must be hit exactly is a stop
of the step plan (see :mod:`pyapprox.ode.step_plan`).
"""

from dataclasses import dataclass
from typing import Sequence, Tuple


def _advance(time: float, size: float) -> float:
    """The one rule by which times are made: :math:`t + h`."""
    return time + size


@dataclass(frozen=True)
class StepSpan:
    r"""A step's place in time: the time of the oldest state it reads and
    the sizes from there to the step's end.

    Lag :math:`l` is the step's end for :math:`l = 0` and the stored state
    :math:`y_{n+1-l}` otherwise; a step reading :math:`k` stored states
    has :math:`k + 1` lags and :math:`k` sizes.

    Parameters
    ----------
    origin : float
        :math:`t_{n+1-k}`, the time of the oldest state the step reads.
    sizes : tuple of float
        :math:`(h_{n+1-k}, \dots, h_n)`, oldest first; the last is the
        step's own size.

    Raises
    ------
    ValueError
        If there is no size, or adding a size does not advance the time
        (a size not positive, or too small to change the time it is added
        to).
    """

    origin: float
    sizes: Tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.sizes:
            raise ValueError("a step span needs at least one size")
        time = self.origin
        for size in self.sizes:
            later = _advance(time, size)
            if not later > time:
                raise ValueError(
                    f"a step of size {size} from t = {time} does not advance the "
                    "time"
                )
            time = later

    @staticmethod
    def first(origin: float, size: float) -> "StepSpan":
        """The span of a step from ``origin`` reading one state."""
        return StepSpan(origin, (size,))

    @staticmethod
    def from_times(times: Sequence[float]) -> "StepSpan":
        r"""The span through the increasing ``times`` (oldest first, the
        step's end last), with sizes their differences.

        Raises
        ------
        ValueError
            If fewer than two times are given, or adding the differences
            back does not reproduce every time exactly (the times do not
            determine their sizes in floating point).
        """
        if len(times) < 2:
            raise ValueError(f"a step span needs at least two times, got {len(times)}")
        sizes = tuple(later - earlier for earlier, later in zip(times, times[1:]))
        span = StepSpan(float(times[0]), sizes)
        rebuilt = span.times()
        if rebuilt != tuple(float(time) for time in times):
            raise ValueError(
                f"the times {tuple(times)} do not determine their sizes exactly: "
                f"adding the differences back gives {rebuilt}"
            )
        return span

    def nlags(self) -> int:
        """The number of lags: one more than the stored states read."""
        return len(self.sizes) + 1

    def times(self) -> Tuple[float, ...]:
        """Every lag's time, oldest first, the step's end last."""
        times = [self.origin]
        for size in self.sizes:
            times.append(_advance(times[-1], size))
        return tuple(times)

    def lag_time(self, lag: int) -> float:
        r""":math:`t_{n+1-\mathrm{lag}}`: the step's end at lag 0.

        Raises
        ------
        ValueError
            If ``lag`` is not between 0 and the number of states read.
        """
        if not 0 <= lag < self.nlags():
            raise ValueError(f"lag must be between 0 and {len(self.sizes)}, got {lag}")
        return self.times()[-1 - lag]

    def time(self) -> float:
        """:math:`t_n`, the step's start."""
        return self.lag_time(1)

    def end(self) -> float:
        """:math:`t_{n+1}`, the step's end: its start plus its size."""
        return self.lag_time(0)

    def deltat(self) -> float:
        """:math:`h_n`, the step's own size, as given."""
        return self.sizes[-1]

    def advanced(self, size: float, nstates: int) -> "StepSpan":
        """The next step's span: of size ``size``, from this step's end,
        reading ``nstates`` stored states (at most one more than this step
        read). The origin moves forward by adding the sizes dropped, so the
        times stay those of the same trajectory.

        Raises
        ------
        ValueError
            If ``nstates`` is not between 1 and this step's lags.
        """
        if not 1 <= nstates <= self.nlags():
            raise ValueError(
                f"the next step reads between 1 and {self.nlags()} states, "
                f"not {nstates}"
            )
        sizes = self.sizes + (size,)
        dropped = len(sizes) - nstates
        origin = self.origin
        for old in sizes[:dropped]:
            origin = _advance(origin, old)
        return StepSpan(origin, sizes[dropped:])
