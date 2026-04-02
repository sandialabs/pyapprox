"""Protocol definitions for parallel execution backends.

This module defines the ParallelBackendProtocol that all parallel
execution backends must implement.
"""

from typing import (
    Any,
    Callable,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

T = TypeVar("T")


@runtime_checkable
class ParallelBackendProtocol(Protocol):
    """Protocol for parallel execution backends.

    All parallel backends (joblib, mpire, etc.) must implement this
    interface to be used with the parallel execution infrastructure.

    Methods
    -------
    map(func, items, n_jobs)
        Apply function to each item in parallel.
    starmap(func, items, n_jobs)
        Apply function with unpacked arguments in parallel.
    backend_name()
        Return backend identifier string.
    """

    def map(
        self,
        func: Callable[[T], T],
        items: Sequence[T],
        n_jobs: int = -1,
    ) -> Sequence[T]:
        """Apply function to each item in parallel.

        Parameters
        ----------
        func : Callable[[T], T]
            Function to apply to each item.
        items : Sequence[T]
            Items to process.
        n_jobs : int, optional
            Number of parallel workers. -1 means use all CPUs.
            Default is -1.

        Returns
        -------
        Sequence[T]
            Results in same order as input items.
        """
        ...

    def starmap(
        self,
        func: Callable[..., T],
        items: Sequence[tuple[Any, ...]],
        n_jobs: int = -1,
    ) -> Sequence[T]:
        """Apply function with unpacked tuple arguments in parallel.

        Parameters
        ----------
        func : Callable[..., T]
            Function to apply with unpacked arguments.
        items : Sequence[tuple]
            Sequence of argument tuples.
        n_jobs : int, optional
            Number of parallel workers. -1 means use all CPUs.
            Default is -1.

        Returns
        -------
        Sequence[T]
            Results in same order as input items.
        """
        ...

    def backend_name(self) -> str:
        """Return backend identifier string.

        Returns
        -------
        str
            Human-readable backend name with configuration info.
        """
        ...
