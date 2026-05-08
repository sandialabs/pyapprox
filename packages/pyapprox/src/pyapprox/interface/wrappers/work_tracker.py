"""Work tracking utilities for model evaluations.

This module provides:
- WorkTracker: Track evaluation counts and wall times
- TrackedModel: Transparent wrapper that records to WorkTracker
"""

import time
from typing import TYPE_CHECKING, Dict, Generic, List, Optional

from pyapprox.util.backends.protocols import Array, Backend

if TYPE_CHECKING:
    from pyapprox.interface.functions.protocols.function import (
        FunctionProtocol,
    )


class WorkTracker(Generic[Array]):
    """Track model evaluation counts and wall times.

    This class records the number of evaluations and wall times for
    different evaluation types (values, jacobian, hessian, hvp, jvp).

    Parameters
    ----------
    bkd : Backend[Array]
        The backend for array operations.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> tracker = WorkTracker(bkd)
    >>> tracker.record("values", 0.5)
    >>> tracker.record("values", 0.3)
    >>> tracker.num_evaluations("values")
    2
    >>> tracker.total_time("values")
    0.8
    >>> tracker.mean_time("values")
    0.4
    """

    _EVAL_TYPES = ("values", "jacobian", "hessian", "hvp", "jvp", "whvp")

    def __init__(self, bkd: Backend[Array]) -> None:
        """Initialize the work tracker.

        Parameters
        ----------
        bkd : Backend[Array]
            The backend for array operations.
        """
        self._bkd = bkd
        self._wall_times: Dict[str, List[float]] = {key: [] for key in self._EVAL_TYPES}

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def record(self, eval_type: str, wall_time: float) -> None:
        """Record a wall time for an evaluation.

        Parameters
        ----------
        eval_type : str
            The type of evaluation ("values", "jacobian", "hessian",
            "hvp", "jvp", "whvp").
        wall_time : float
            The wall time in seconds.

        Raises
        ------
        ValueError
            If eval_type is not recognized.
        """
        if eval_type not in self._wall_times:
            raise ValueError(
                f"Unknown eval_type: {eval_type}. Must be one of {self._EVAL_TYPES}"
            )
        self._wall_times[eval_type].append(wall_time)

    def num_evaluations(self, eval_type: str = "values") -> int:
        """Return the number of evaluations for a given type.

        Parameters
        ----------
        eval_type : str, optional
            The type of evaluation. Default is "values".

        Returns
        -------
        int
            The number of evaluations.
        """
        return len(self._wall_times[eval_type])

    def total_time(self, eval_type: str = "values") -> float:
        """Return the total wall time for a given evaluation type.

        Parameters
        ----------
        eval_type : str, optional
            The type of evaluation. Default is "values".

        Returns
        -------
        float
            The total wall time in seconds.
        """
        return sum(self._wall_times[eval_type])

    def mean_time(self, eval_type: str = "values") -> float:
        """Return the mean wall time for a given evaluation type.

        Parameters
        ----------
        eval_type : str, optional
            The type of evaluation. Default is "values".

        Returns
        -------
        float
            The mean wall time in seconds. Returns 0.0 if no evaluations.
        """
        n = self.num_evaluations(eval_type)
        if n == 0:
            return 0.0
        return self.total_time(eval_type) / n

    def wall_times(self, eval_type: str = "values") -> Array:
        """Return all wall times as an array.

        Parameters
        ----------
        eval_type : str, optional
            The type of evaluation. Default is "values".

        Returns
        -------
        Array
            Array of wall times.
        """
        return self._bkd.asarray(self._wall_times[eval_type])

    def reset(self, eval_type: Optional[str] = None) -> None:
        """Reset the tracker.

        Parameters
        ----------
        eval_type : str, optional
            If provided, reset only this evaluation type.
            If None, reset all types.
        """
        if eval_type is not None:
            self._wall_times[eval_type] = []
        else:
            for key in self._wall_times:
                self._wall_times[key] = []

    def __repr__(self) -> str:
        """Return string representation."""
        parts = []
        for eval_type in self._EVAL_TYPES:
            n = self.num_evaluations(eval_type)
            if n > 0:
                mean = self.mean_time(eval_type)
                parts.append(f"{eval_type}(n={n}, mean={mean:.4f}s)")
        if not parts:
            return "WorkTracker(empty)"
        return f"WorkTracker({', '.join(parts)})"


class TrackedModel(Generic[Array]):
    """Transparent wrapper that tracks evaluations.

    This wrapper passes all calls through to the wrapped model
    while recording wall times to a WorkTracker.

    Parameters
    ----------
    model : FunctionProtocol[Array]
        The model to wrap. Must have bkd(), nvars(), nqoi(), and __call__.
    tracker : WorkTracker[Array]
        The tracker to record to.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.interface.functions.fromcallable.function import (
    ...     FunctionFromCallable,
    ... )
    >>> bkd = NumpyBkd()
    >>> def my_fun(samples):
    ...     return samples[0:1, :] ** 2
    >>> model = FunctionFromCallable(nqoi=1, nvars=2, fun=my_fun, bkd=bkd)
    >>> tracker = WorkTracker(bkd)
    >>> tracked = TrackedModel(model, tracker)
    >>> samples = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
    >>> values = tracked(samples)
    >>> tracker.num_evaluations("values")
    1
    """

    def __init__(
        self,
        model: "FunctionProtocol[Array]",
        tracker: WorkTracker[Array],
    ) -> None:
        """Initialize the tracked model.

        Parameters
        ----------
        model : FunctionProtocol[Array]
            The model to wrap.
        tracker : WorkTracker[Array]
            The tracker to record to.
        """
        self._model = model
        self._tracker = tracker
        self._setup_derivative_methods()

    def _setup_derivative_methods(self) -> None:
        """Dynamically add derivative methods if model has them."""
        if hasattr(self._model, "jacobian"):
            self.jacobian = self._jacobian
        if hasattr(self._model, "hessian"):
            self.hessian = self._hessian
        if hasattr(self._model, "hvp"):
            self.hvp = self._hvp
        if hasattr(self._model, "jvp"):
            self.jvp = self._jvp
        if hasattr(self._model, "whvp"):
            self.whvp = self._whvp

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._model.bkd()

    def nvars(self) -> int:
        """Return the number of variables."""
        return int(self._model.nvars())

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        return int(self._model.nqoi())

    def tracker(self) -> WorkTracker[Array]:
        """Return the tracker."""
        return self._tracker

    def wrapped(self) -> "FunctionProtocol[Array]":
        """Return the wrapped model."""
        return self._model

    def __call__(self, samples: Array) -> Array:
        """Evaluate the model and track time.

        Parameters
        ----------
        samples : Array
            Input samples of shape (nvars, nsamples).

        Returns
        -------
        Array
            Output values of shape (nqoi, nsamples).
        """
        start = time.perf_counter()
        values: Array = self._model(samples)
        elapsed = time.perf_counter() - start
        self._tracker.record("values", elapsed)
        return values

    def _jacobian(self, sample: Array) -> Array:
        """Compute Jacobian and track time."""
        start = time.perf_counter()
        result: Array = self._model.jacobian(sample)
        elapsed = time.perf_counter() - start
        self._tracker.record("jacobian", elapsed)
        return result

    def _hessian(self, sample: Array) -> Array:
        """Compute Hessian and track time."""
        start = time.perf_counter()
        result: Array = self._model.hessian(sample)
        elapsed = time.perf_counter() - start
        self._tracker.record("hessian", elapsed)
        return result

    def _hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product and track time."""
        start = time.perf_counter()
        result: Array = self._model.hvp(sample, vec)
        elapsed = time.perf_counter() - start
        self._tracker.record("hvp", elapsed)
        return result

    def _jvp(self, sample: Array, vec: Array) -> Array:
        """Compute Jacobian-vector product and track time."""
        start = time.perf_counter()
        result: Array = self._model.jvp(sample, vec)
        elapsed = time.perf_counter() - start
        self._tracker.record("jvp", elapsed)
        return result

    def _whvp(self, sample: Array, vec: Array, weights: Array) -> Array:
        """Compute weighted Hessian-vector product and track time."""
        start = time.perf_counter()
        result: Array = self._model.whvp(sample, vec, weights)
        elapsed = time.perf_counter() - start
        self._tracker.record("whvp", elapsed)
        return result

    def __repr__(self) -> str:
        """Return string representation."""
        return f"TrackedModel({self._model!r}, {self._tracker!r})"
