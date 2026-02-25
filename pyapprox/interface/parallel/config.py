"""Configuration for parallel execution.

This module provides ParallelConfig for configuring parallel execution
backends and settings.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Union

from pyapprox.interface.parallel.protocols import (
    ParallelBackendProtocol,
)


@dataclass
class ParallelConfig:
    """Configuration for parallel execution.

    Parameters
    ----------
    backend : {"joblib_processes", "joblib_threads", "futures", "mpire", "sequential"}
        Parallel execution backend to use.
        - "joblib_processes": Joblib with process-based parallelism (default)
        - "joblib_threads": Joblib with thread-based parallelism
        - "futures": stdlib concurrent.futures (no extra dependencies)
        - "mpire": Progress bars, better worker management
        - "sequential": No parallelism (for debugging)
    n_jobs : int
        Number of parallel workers. -1 means use all CPUs.
    chunk_size : int, optional
        Number of samples per chunk. If None, auto-determined.

    Examples
    --------
    >>> config = ParallelConfig(backend="joblib_processes", n_jobs=4)
    >>> backend = config.get_parallel_backend()
    >>> results = backend.map(my_func, items)
    """

    backend: Literal[
        "joblib_processes",
        "joblib_threads",
        "futures",
        "mpire",
        "sequential",
    ] = "joblib_processes"
    n_jobs: int = -1
    chunk_size: Optional[int] = None

    def get_parallel_backend(
        self,
    ) -> Union[ParallelBackendProtocol, "SequentialBackend"]:
        """Create and return the configured parallel backend.

        Returns
        -------
        ParallelBackendProtocol
            Configured backend instance.

        Raises
        ------
        ValueError
            If backend is unknown.
        """
        if self.backend in ("joblib_processes", "joblib_threads"):
            from pyapprox.interface.parallel.joblib_backend import (
                JoblibBackend,
            )

            prefer = "threads" if self.backend == "joblib_threads" else "processes"
            return JoblibBackend(
                n_jobs=self.n_jobs,
                prefer=prefer,
            )
        elif self.backend == "futures":
            from pyapprox.interface.parallel.futures_backend import (
                FuturesBackend,
            )

            return FuturesBackend(n_jobs=self.n_jobs)
        elif self.backend == "mpire":
            from pyapprox.interface.parallel.mpire_backend import (
                MpireBackend,
            )

            return MpireBackend(n_jobs=self.n_jobs)
        elif self.backend == "sequential":
            return SequentialBackend()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")


class SequentialBackend:
    """Sequential (non-parallel) backend for debugging.

    Executes functions sequentially without any parallelism.
    Useful for debugging and small workloads.
    """

    def map(self, func, items, n_jobs=-1):
        """Apply function to each item sequentially.

        Parameters
        ----------
        func : Callable
            Function to apply.
        items : Sequence
            Items to process.
        n_jobs : int, optional
            Ignored (for API compatibility).

        Returns
        -------
        list
            Results in same order as input.
        """
        return [func(item) for item in items]

    def starmap(self, func, items, n_jobs=-1):
        """Apply function with unpacked arguments sequentially.

        Parameters
        ----------
        func : Callable
            Function to apply with unpacked arguments.
        items : Sequence[tuple]
            Sequence of argument tuples.
        n_jobs : int, optional
            Ignored (for API compatibility).

        Returns
        -------
        list
            Results in same order as input.
        """
        return [func(*args) for args in items]

    def backend_name(self) -> str:
        """Return backend identifier string."""
        return "sequential"
