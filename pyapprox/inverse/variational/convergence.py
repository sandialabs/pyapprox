"""VI convergence monitor and optimizer integration.

Provides VIConvergenceMonitor for tracking optimization progress and
running convergence checks, plus integration points for ROL (via
StatusTest subclass) and scipy (via callback factory).
"""

from typing import Any, Callable, Generic, List, Optional

from pyapprox.inverse.variational.convergence_protocols import (
    ConvergenceCheckProtocol,
    ConvergenceCheckResult,
)
from pyapprox.util.backends.protocols import Array, Backend


class VIConvergenceMonitor(Generic[Array]):
    """Tracks ELBO history and runs convergence checks periodically.

    Records ELBO values each iteration and periodically runs a
    convergence check to determine whether early stopping is warranted.

    Parameters
    ----------
    convergence_check : ConvergenceCheckProtocol[Array]
        The diagnostic check to run.
    check_every : int
        Run the diagnostic every ``check_every`` iterations.
    min_iterations : int
        Minimum iterations before allowing early stopping.
    """

    def __init__(
        self,
        convergence_check: ConvergenceCheckProtocol[Array],
        check_every: int = 10,
        min_iterations: int = 20,
    ) -> None:
        self._check = convergence_check
        self._check_every = check_every
        self._min_iterations = min_iterations
        self._elbo_history: List[float] = []
        self._check_history: List[ConvergenceCheckResult] = []
        self._triggered = False
        self._last_result: Optional[ConvergenceCheckResult] = None
        self._iteration = 0

    def record_iteration(self, params: Array, elbo: float) -> None:
        """Record an ELBO value for the current iteration.

        Parameters
        ----------
        params : Array
            Current variational parameters (not stored, for API
            consistency).
        elbo : float
            Current ELBO value (positive = better).
        """
        self._elbo_history.append(elbo)
        self._iteration += 1

    def recent_improvement(self) -> float:
        """Compute recent ELBO improvement.

        Returns the difference between the latest ELBO and the ELBO
        ``check_every`` iterations ago. Positive means improving.

        Returns
        -------
        float
            Recent ELBO improvement.
        """
        if len(self._elbo_history) < 2:
            return float("inf")
        lookback = min(self._check_every, len(self._elbo_history) - 1)
        return self._elbo_history[-1] - self._elbo_history[-1 - lookback]

    def should_check(self) -> bool:
        """Whether it's time to run a convergence check.

        Returns
        -------
        bool
            True if enough iterations have passed.
        """
        return (
            self._iteration >= self._min_iterations
            and self._iteration % self._check_every == 0
        )

    def run_check(self, params: Array) -> ConvergenceCheckResult:
        """Run the convergence diagnostic.

        Parameters
        ----------
        params : Array
            Current variational parameters, shape ``(nvars, 1)``.

        Returns
        -------
        ConvergenceCheckResult
        """
        improvement = self.recent_improvement()
        result = self._check.check(params, improvement)
        self._check_history.append(result)
        self._last_result = result
        if result.should_stop:
            self._triggered = True
        return result

    @property
    def triggered(self) -> bool:
        """Whether early stopping has been triggered."""
        return self._triggered

    @property
    def check_history(self) -> List[ConvergenceCheckResult]:
        """List of all check results."""
        return self._check_history

    @property
    def last_result(self) -> Optional[ConvergenceCheckResult]:
        """Most recent check result, or None if no checks run."""
        return self._last_result

    @property
    def elbo_history(self) -> List[float]:
        """List of recorded ELBO values."""
        return self._elbo_history


def make_rol_convergence_status_test(
    monitor: VIConvergenceMonitor,
    bkd: Backend,
) -> object:
    """Create a ROL StatusTest that delegates to a VIConvergenceMonitor.

    Parameters
    ----------
    monitor : VIConvergenceMonitor
        The convergence monitor.
    bkd : Backend
        Computational backend.

    Returns
    -------
    object
        A ROL StatusTest instance.
    """
    import pyrol.pyrol

    class _ROLStatusTest(pyrol.pyrol.ROL.StatusTest_double_t):
        def __init__(
            self,
            mon: VIConvergenceMonitor,
            backend: Backend,
        ) -> None:
            super().__init__()
            self._monitor = mon
            self._bkd = backend

        def check(self, state: Any) -> bool:
            params = self._bkd.asarray(state.iterateVec.array)[:, None]
            elbo = -state.value
            self._monitor.record_iteration(params, elbo)
            if self._monitor.should_check():
                self._monitor.run_check(params)
            return not self._monitor.triggered

    return _ROLStatusTest(monitor, bkd)


def make_scipy_convergence_callback(
    monitor: VIConvergenceMonitor,
    bkd: Backend,
) -> Callable:
    """Create a scipy trust-constr callback from a VIConvergenceMonitor.

    The returned callback has signature ``callback(x, state) -> bool``
    where returning True stops the optimizer (``status=3``).

    Parameters
    ----------
    monitor : VIConvergenceMonitor
        The convergence monitor.
    bkd : Backend
        Computational backend.

    Returns
    -------
    callable
        Callback for ``scipy.optimize.minimize(..., callback=...)``.
    """

    def callback(x: Any, state: Any) -> bool:
        params = bkd.asarray(x)[:, None]
        elbo = -float(state.fun)
        monitor.record_iteration(params, elbo)
        if monitor.should_check():
            monitor.run_check(params)
        return monitor.triggered

    return callback
