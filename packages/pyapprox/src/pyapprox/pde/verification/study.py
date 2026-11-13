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
"""

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, TypeVar, runtime_checkable

from pyapprox.pde.verification.protocols import (
    ApproximationProtocol,
    DiscretizerProtocol,
)

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
