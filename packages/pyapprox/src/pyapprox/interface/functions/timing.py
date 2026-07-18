"""Function timing wrapper module.

Provides transparent timing wrappers for ObjectiveProtocol objects,
recording per-method timing statistics with batch-size awareness.

Classes
-------
- MethodTimer: Per-method timing with median/total/count/reset
- FunctionTimer: Aggregates MethodTimers by method name
- TimedFunction: Wrapper for ObjectiveProtocol; derivative capability is
  mirrored from the wrapped function's ``Derivatives`` bundle with each
  populated field wrapped in a timing recorder. A derivative-free
  function participates by returning ``Derivatives.none()``.

Functions
---------
- timed(): Factory wrapping a function in TimedFunction

Composition
-----------
The correct composition order is ``timed(make_parallel(fn))``, NOT
``make_parallel(timed(fn))``. The latter breaks because multiprocessing
pickles the timer into worker processes and the state is lost.
"""

# TODO: should this be moved to the interface.wrappers module

import time
from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Tuple

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.protocols.objective import (
    ObjectiveProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class MethodTimer:
    """Per-method timing tracker with batch-aware statistics.

    Each record is a ``(elapsed_seconds, n_evals)`` tuple. Individual
    calls record ``n_evals=1``. Batch calls record
    ``n_evals=batch_size``.

    Do not mix individual and batch calls on the same MethodTimer.
    Use separate method names (e.g., ``jacobian`` vs ``jacobian_batch``).
    """

    def __init__(self) -> None:
        self._records: List[Tuple[float, int]] = []

    def record(self, elapsed: float, n_evals: int = 1) -> None:
        """Append a timing record."""
        self._records.append((elapsed, n_evals))

    def median(self) -> float:
        """Median per-evaluation time.

        For individual calls (all records have n_evals=1): true median
        of per-call times computed from a sorted list.

        For batch calls (any record has n_evals>1): weighted mean across
        batches, i.e. ``total_time() / total_evals()``. This is the best
        available estimate when individual times are not observable.

        Returns
        -------
        float
            Median per-evaluation time in seconds.

        Raises
        ------
        ValueError
            If no records exist.
        """
        if not self._records:
            raise ValueError("No records to compute median from.")
        if all(n == 1 for _, n in self._records):
            times = sorted(t for t, _ in self._records)
            n = len(times)
            mid = n // 2
            if n % 2 == 1:
                return times[mid]
            return (times[mid - 1] + times[mid]) / 2.0
        return self.total_time() / self.total_evals()

    def total_time(self) -> float:
        """Sum of elapsed times across all records."""
        return sum(t for t, _ in self._records)

    def total_evals(self) -> int:
        """Sum of n_evals across all records."""
        return sum(n for _, n in self._records)

    def call_count(self) -> int:
        """Number of records (i.e. method invocations)."""
        return len(self._records)

    def reset(self) -> None:
        """Clear all records."""
        self._records = []

    def __repr__(self) -> str:
        if not self._records:
            return "MethodTimer(empty)"
        return (
            f"MethodTimer(call_count={self.call_count()}, "
            f"total_time={self.total_time():.4f}s, "
            f"median={self.median():.4f}s)"
        )


class FunctionTimer:
    """Aggregator of MethodTimers keyed by method name.

    Auto-creates MethodTimer instances on first access via ``get()``.
    """

    def __init__(self) -> None:
        self._timers: Dict[str, MethodTimer] = {}

    def get(self, method_name: str) -> MethodTimer:
        """Get or create a MethodTimer for the given method name."""
        if method_name not in self._timers:
            self._timers[method_name] = MethodTimer()
        return self._timers[method_name]

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return summary dict with stats per method.

        Returns
        -------
        dict
            ``{method_name: {median, total_time, total_evals, call_count}}``
            for each method that has been timed.
        """
        result: Dict[str, Dict[str, float]] = {}
        for name, timer in self._timers.items():
            if timer.call_count() > 0:
                result[name] = {
                    "median": timer.median(),
                    "total_time": timer.total_time(),
                    "total_evals": float(timer.total_evals()),
                    "call_count": float(timer.call_count()),
                }
        return result

    def reset(self) -> None:
        """Reset all MethodTimers."""
        for timer in self._timers.values():
            timer.reset()

    def __repr__(self) -> str:
        if not self._timers:
            return "FunctionTimer(empty)"
        parts = []
        for name, timer in self._timers.items():
            if timer.call_count() > 0:
                parts.append(
                    f"{name}(n={timer.total_evals()}, median={timer.median():.4f}s)"
                )
        if not parts:
            return "FunctionTimer(empty)"
        return f"FunctionTimer({', '.join(parts)})"


@dataclass(frozen=True)
class _TimedUnary(Generic[Array]):
    """Timing wrapper for a ``(samples) -> Array`` bundle field.

    ``batch`` selects whether n_evals is the number of columns (batch
    fields) or 1 (single-sample fields).
    """

    fn: Callable[[Array], Array]
    timer: FunctionTimer
    name: str
    batch: bool

    def __call__(self, samples: Array) -> Array:
        n_evals = samples.shape[1] if self.batch else 1
        t0 = time.perf_counter()
        result = self.fn(samples)
        self.timer.get(self.name).record(time.perf_counter() - t0, n_evals)
        return result


@dataclass(frozen=True)
class _TimedBinary(Generic[Array]):
    """Timing wrapper for a ``(samples, vecs) -> Array`` bundle field."""

    fn: Callable[[Array, Array], Array]
    timer: FunctionTimer
    name: str
    batch: bool

    def __call__(self, samples: Array, vecs: Array) -> Array:
        n_evals = samples.shape[1] if self.batch else 1
        t0 = time.perf_counter()
        result = self.fn(samples, vecs)
        self.timer.get(self.name).record(time.perf_counter() - t0, n_evals)
        return result


@dataclass(frozen=True)
class _TimedTernary(Generic[Array]):
    """Timing wrapper for ``(samples, vecs, weights) -> Array`` fields."""

    fn: Callable[[Array, Array, Array], Array]
    timer: FunctionTimer
    name: str
    batch: bool

    def __call__(self, samples: Array, vecs: Array, weights: Array) -> Array:
        n_evals = samples.shape[1] if self.batch else 1
        t0 = time.perf_counter()
        result = self.fn(samples, vecs, weights)
        self.timer.get(self.name).record(time.perf_counter() - t0, n_evals)
        return result


class TimedFunction(Generic[Array]):
    """Transparent timing wrapper for ObjectiveProtocol objects.

    Derivative capability is mirrored from the wrapped function's
    ``Derivatives`` bundle: each populated field is re-exposed through
    ``derivatives()`` wrapped in a timing recorder keyed by the field
    name. An ``inexact`` suite is propagated unchanged.

    Parameters
    ----------
    function : ObjectiveProtocol[Array]
        The function to wrap. A derivative-free function participates
        by implementing ``derivatives()`` returning ``Derivatives.none()``.
    timer : FunctionTimer, optional
        Shared timer instance. If None, creates a new one.
        Pass a shared timer to aggregate stats across multiple
        functions (e.g. all models in a multifidelity ensemble).
    """

    def __init__(
        self,
        function: ObjectiveProtocol[Array],
        timer: Optional[FunctionTimer] = None,
    ) -> None:
        self._function = function
        self._timer = timer if timer is not None else FunctionTimer()
        if not isinstance(function, ObjectiveProtocol):
            raise TypeError(
                f"{type(function).__name__} must satisfy ObjectiveProtocol "
                "(a FunctionProtocol exposing derivatives()). If the "
                "function has no derivative capability, add a "
                "derivatives() method returning Derivatives.none()."
            )
        fd = function.derivatives()
        t = self._timer
        self._derivs: Derivatives[Array] = Derivatives(
            jacobian=None
            if fd.jacobian is None
            else _TimedUnary(fd.jacobian, t, "jacobian", False),
            jacobian_batch=None
            if fd.jacobian_batch is None
            else _TimedUnary(fd.jacobian_batch, t, "jacobian_batch", True),
            jvp=None
            if fd.jvp is None
            else _TimedBinary(fd.jvp, t, "jvp", False),
            hvp=None
            if fd.hvp is None
            else _TimedBinary(fd.hvp, t, "hvp", False),
            whvp=None
            if fd.whvp is None
            else _TimedTernary(fd.whvp, t, "whvp", False),
            hessian=None
            if fd.hessian is None
            else _TimedUnary(fd.hessian, t, "hessian", False),
            hessian_batch=None
            if fd.hessian_batch is None
            else _TimedUnary(fd.hessian_batch, t, "hessian_batch", True),
            hvp_batch=None
            if fd.hvp_batch is None
            else _TimedBinary(fd.hvp_batch, t, "hvp_batch", True),
            whvp_batch=None
            if fd.whvp_batch is None
            else _TimedTernary(fd.whvp_batch, t, "whvp_batch", True),
            inexact=fd.inexact,
        )

    def derivatives(self) -> Derivatives[Array]:
        """Return the timed derivative bundle mirroring the wrapped
        function."""
        return self._derivs

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._function.bkd()

    def nvars(self) -> int:
        """Return the number of variables."""
        return int(self._function.nvars())

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        return int(self._function.nqoi())

    def timer(self) -> FunctionTimer:
        """Return the FunctionTimer."""
        return self._timer

    def wrapped(self) -> ObjectiveProtocol[Array]:
        """Return the wrapped function."""
        return self._function

    def __call__(self, samples: Array) -> Array:
        """Evaluate the function and record timing.

        Records n_evals equal to the number of samples.
        """
        n_evals = samples.shape[1]
        t0 = time.perf_counter()
        result: Array = self._function(samples)
        self._timer.get("__call__").record(time.perf_counter() - t0, n_evals)
        return result

    def __repr__(self) -> str:
        return f"TimedFunction({self._function!r})"


def timed(
    function: ObjectiveProtocol[Array],
    timer: Optional[FunctionTimer] = None,
) -> TimedFunction[Array]:
    """Wrap a function with timing instrumentation.

    The wrapper is transparent — evaluation is delegated unchanged and
    the wrapped function's derivative bundle is mirrored with timing
    recorders on every populated field.

    The correct composition order is ``timed(make_parallel(fn))``, NOT
    ``make_parallel(timed(fn))``. The latter breaks because
    multiprocessing pickles the timer into worker processes and the
    state is lost.

    Parameters
    ----------
    function : ObjectiveProtocol[Array]
        The function to wrap. A derivative-free function participates
        by implementing ``derivatives()`` returning ``Derivatives.none()``.
    timer : FunctionTimer, optional
        Shared timer instance. If None, creates a new one.
        Pass a shared timer to aggregate stats across multiple
        functions (e.g. all models in a multifidelity ensemble).

    Returns
    -------
    TimedFunction[Array]
        Timed wrapper. Access stats via ``.timer()``.

    Raises
    ------
    TypeError
        If function does not satisfy ObjectiveProtocol.
    """
    return TimedFunction(function, timer)
