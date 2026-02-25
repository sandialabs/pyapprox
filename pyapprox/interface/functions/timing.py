"""Function timing wrapper module.

Provides transparent timing wrappers for FunctionProtocol objects,
recording per-method timing statistics with batch-size awareness.

Classes
-------
- MethodTimer: Per-method timing with median/total/count/reset
- FunctionTimer: Aggregates MethodTimers by method name
- TimedFunction: Wrapper for FunctionProtocol
- TimedFunctionWithJacobian: Wrapper adding jacobian timing
- TimedFunctionWithJacobianAndHVP: Wrapper adding hvp/hessian timing
- TimedFunctionWithJVP: Wrapper adding jvp timing
- TimedFunctionWithJacobianAndWHVP: Wrapper adding whvp timing

Functions
---------
- timed(): Factory that auto-selects wrapper based on protocol

Composition
-----------
The correct composition order is ``timed(make_parallel(fn))``, NOT
``make_parallel(timed(fn))``. The latter breaks because multiprocessing
pickles the timer into worker processes and the state is lost.
"""

import time
from typing import Any, Dict, Generic, List, Optional, Tuple

from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.interface.functions.protocols.hessian import (
    FunctionWithJacobianAndHVPProtocol,
    FunctionWithJacobianAndWHVPProtocol,
)
from pyapprox.interface.functions.protocols.jacobian import (
    FunctionWithJacobianProtocol,
    FunctionWithJVPProtocol,
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
                    f"{name}(n={timer.total_evals()}, "
                    f"median={timer.median():.4f}s)"
                )
        if not parts:
            return "FunctionTimer(empty)"
        return f"FunctionTimer({', '.join(parts)})"


class TimedFunction(Generic[Array]):
    """Transparent timing wrapper for FunctionProtocol objects.

    Parameters
    ----------
    function : FunctionProtocol[Array]
        The function to wrap.
    timer : FunctionTimer, optional
        Shared timer instance. If None, creates a new one.
        Pass a shared timer to aggregate stats across multiple
        functions (e.g. all models in a multifidelity ensemble).
    """

    def __init__(
        self,
        function: FunctionProtocol[Array],
        timer: Optional[FunctionTimer] = None,
    ) -> None:
        self._function = function
        self._timer = timer if timer is not None else FunctionTimer()

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

    def wrapped(self) -> FunctionProtocol[Array]:
        """Return the wrapped function."""
        return self._function

    def __call__(self, samples: Array) -> Array:
        """Evaluate the function and record timing.

        Records n_evals equal to the number of samples.
        """
        n_evals = samples.shape[1]
        t0 = time.perf_counter()
        result: Array = self._function(samples)
        self._timer.get("__call__").record(
            time.perf_counter() - t0, n_evals
        )
        return result

    def __repr__(self) -> str:
        return f"TimedFunction({self._function!r})"


class TimedFunctionWithJacobian(TimedFunction[Array]):
    """Timing wrapper for FunctionWithJacobianProtocol objects."""

    _function: Any

    def __init__(
        self,
        function: FunctionWithJacobianProtocol[Array],
        timer: Optional[FunctionTimer] = None,
    ) -> None:
        super().__init__(function, timer)
        if hasattr(self._function, "jacobian_batch"):
            self.jacobian_batch = self._jacobian_batch

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian and record timing."""
        t0 = time.perf_counter()
        result: Array = self._function.jacobian(sample)
        self._timer.get("jacobian").record(time.perf_counter() - t0, 1)
        return result

    def _jacobian_batch(self, samples: Array) -> Array:
        """Compute batch Jacobians and record timing."""
        n_evals = samples.shape[1]
        t0 = time.perf_counter()
        result: Array = self._function.jacobian_batch(samples)
        self._timer.get("jacobian_batch").record(
            time.perf_counter() - t0, n_evals
        )
        return result

    def __repr__(self) -> str:
        return f"TimedFunctionWithJacobian({self._function!r})"


class TimedFunctionWithJacobianAndHVP(TimedFunctionWithJacobian[Array]):
    """Timing wrapper for FunctionWithJacobianAndHVPProtocol objects."""

    def __init__(
        self,
        function: FunctionWithJacobianAndHVPProtocol[Array],
        timer: Optional[FunctionTimer] = None,
    ) -> None:
        super().__init__(function, timer)
        if hasattr(self._function, "hvp_batch"):
            self.hvp_batch = self._hvp_batch
        if hasattr(self._function, "hessian_batch"):
            self.hessian_batch = self._hessian_batch

    def hvp(self, sample: Array, vec: Array) -> Array:
        """Compute HVP and record timing."""
        t0 = time.perf_counter()
        result: Array = self._function.hvp(sample, vec)
        self._timer.get("hvp").record(time.perf_counter() - t0, 1)
        return result

    def _hvp_batch(self, samples: Array, vecs: Array) -> Array:
        """Compute batch HVPs and record timing."""
        n_evals = samples.shape[1]
        t0 = time.perf_counter()
        result: Array = self._function.hvp_batch(samples, vecs)
        self._timer.get("hvp_batch").record(
            time.perf_counter() - t0, n_evals
        )
        return result

    def _hessian_batch(self, samples: Array) -> Array:
        """Compute batch Hessians and record timing."""
        n_evals = samples.shape[1]
        t0 = time.perf_counter()
        result: Array = self._function.hessian_batch(samples)
        self._timer.get("hessian_batch").record(
            time.perf_counter() - t0, n_evals
        )
        return result

    def __repr__(self) -> str:
        return f"TimedFunctionWithJacobianAndHVP({self._function!r})"


class TimedFunctionWithJVP(TimedFunction[Array]):
    """Timing wrapper for FunctionWithJVPProtocol objects."""

    _function: Any

    def __init__(
        self,
        function: FunctionWithJVPProtocol[Array],
        timer: Optional[FunctionTimer] = None,
    ) -> None:
        super().__init__(function, timer)

    def jvp(self, sample: Array, vec: Array) -> Array:
        """Compute JVP and record timing."""
        t0 = time.perf_counter()
        result: Array = self._function.jvp(sample, vec)
        self._timer.get("jvp").record(time.perf_counter() - t0, 1)
        return result

    def __repr__(self) -> str:
        return f"TimedFunctionWithJVP({self._function!r})"


class TimedFunctionWithJacobianAndWHVP(
    TimedFunctionWithJacobianAndHVP[Array]
):
    """Timing wrapper for FunctionWithJacobianAndWHVPProtocol objects.

    Extends TimedFunctionWithJacobianAndHVP so that hvp timing is
    preserved for functions that have both hvp and whvp.
    """

    def __init__(
        self,
        function: FunctionWithJacobianAndWHVPProtocol[Array],
        timer: Optional[FunctionTimer] = None,
    ) -> None:
        super().__init__(function, timer)  # type: ignore[arg-type]
        if hasattr(self._function, "whvp_batch"):
            self.whvp_batch = self._whvp_batch

    def whvp(self, sample: Array, vec: Array, weights: Array) -> Array:
        """Compute weighted HVP and record timing."""
        t0 = time.perf_counter()
        result: Array = self._function.whvp(sample, vec, weights)
        self._timer.get("whvp").record(time.perf_counter() - t0, 1)
        return result

    def _whvp_batch(
        self, samples: Array, vecs: Array, weights: Array
    ) -> Array:
        """Compute batch weighted HVPs and record timing."""
        n_evals = samples.shape[1]
        t0 = time.perf_counter()
        result: Array = self._function.whvp_batch(samples, vecs, weights)
        self._timer.get("whvp_batch").record(
            time.perf_counter() - t0, n_evals
        )
        return result

    def __repr__(self) -> str:
        return f"TimedFunctionWithJacobianAndWHVP({self._function!r})"


def timed(
    function: FunctionProtocol[Array],
    timer: Optional[FunctionTimer] = None,
) -> TimedFunction[Array]:
    """Wrap a function with timing instrumentation.

    Returns a wrapper satisfying the same protocols as the input.
    The wrapper is transparent — all method signatures and return
    types are identical.

    The correct composition order is ``timed(make_parallel(fn))``, NOT
    ``make_parallel(timed(fn))``. The latter breaks because
    multiprocessing pickles the timer into worker processes and the
    state is lost.

    Parameters
    ----------
    function : FunctionProtocol[Array]
        The function to wrap.
    timer : FunctionTimer, optional
        Shared timer instance. If None, creates a new one.
        Pass a shared timer to aggregate stats across multiple
        functions (e.g. all models in a multifidelity ensemble).

    Returns
    -------
    TimedFunction[Array]
        Timed wrapper (or subclass). Access stats via ``.timer()``.

    Raises
    ------
    TypeError
        If function does not satisfy FunctionProtocol.
    """
    if not isinstance(function, FunctionProtocol):
        raise TypeError(
            f"function must satisfy FunctionProtocol, "
            f"got {type(function).__name__}"
        )
    if isinstance(function, FunctionWithJacobianAndWHVPProtocol):
        return TimedFunctionWithJacobianAndWHVP(function, timer)
    if isinstance(function, FunctionWithJacobianAndHVPProtocol):
        return TimedFunctionWithJacobianAndHVP(function, timer)
    if isinstance(function, FunctionWithJVPProtocol):
        return TimedFunctionWithJVP(function, timer)
    if isinstance(function, FunctionWithJacobianProtocol):
        return TimedFunctionWithJacobian(function, timer)
    return TimedFunction(function, timer)
