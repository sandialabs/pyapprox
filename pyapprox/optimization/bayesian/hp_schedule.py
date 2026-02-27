"""Hyperparameter refit schedules for Bayesian optimization.

Controls when full HP optimization occurs vs cheap Cholesky-only refits
during the ask/tell loop.
"""

from typing import Protocol, Set, runtime_checkable


@runtime_checkable
class HPRefitScheduleProtocol(Protocol):
    """Protocol for hyperparameter refit schedules.

    Determines whether full hyperparameter optimization should run
    at a given tell() step, or whether a cheap Cholesky-only refit
    suffices.
    """

    def should_optimize(self, step: int) -> bool:
        """Return True if full HP optimization should run at this step.

        Parameters
        ----------
        step : int
            Zero-based tell() call index.

        Returns
        -------
        bool
            True to run full optimization, False for Cholesky-only refit.
        """
        ...


class AlwaysOptimizeSchedule:
    """Optimize hyperparameters at every tell() call.

    This reproduces the default (pre-schedule) behavior.
    """

    def should_optimize(self, step: int) -> bool:
        """Return True for every step."""
        return True


class EveryKSchedule:
    """Optimize every K-th tell() call. Always optimizes on step 0.

    Parameters
    ----------
    k : int
        Optimization interval. Must be >= 1.
    """

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._k = k

    def should_optimize(self, step: int) -> bool:
        """Return True if step is a multiple of k."""
        return step % self._k == 0


class GeometricSchedule:
    """Geometrically increasing intervals between HP optimizations.

    Dense optimization early (when HPs change fast), sparse later
    (when HPs have stabilized). Always optimizes on step 0.

    With base=1.5: optimize at steps 0, 1, 2, 3, 5, 8, 12, 18, 27, ...

    Parameters
    ----------
    base : float
        Geometric growth factor for the interval. Must be > 1.0.
    """

    def __init__(self, base: float = 1.5) -> None:
        if base <= 1.0:
            raise ValueError(f"base must be > 1.0, got {base}")
        self._base = base
        self._optimize_steps: Set[int] = set()
        self._computed_up_to = -1
        self._precompute_up_to(100)

    def should_optimize(self, step: int) -> bool:
        """Return True if step is in the geometric schedule."""
        if step > self._computed_up_to:
            self._precompute_up_to(step * 2)
        return step in self._optimize_steps

    def _precompute_up_to(self, cap: int) -> None:
        """Extend the set of optimization steps up to cap."""
        interval = 1.0
        s = 0
        while s <= cap:
            step_int = int(round(s))
            self._optimize_steps.add(step_int)
            s += interval
            interval *= self._base
        self._computed_up_to = cap
