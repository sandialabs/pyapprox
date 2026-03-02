"""Mpire-based parallel execution backend.

This module provides a parallel execution backend using mpire,
which offers a modern API with built-in progress bars and
better worker state management.
"""

from typing import Callable, List, Sequence, TypeVar

T = TypeVar("T")


class MpireBackend:
    """Parallel execution backend using mpire.

    Mpire provides a modern parallel processing API with
    built-in progress bars, better exception handling, and
    efficient worker state management.

    Parameters
    ----------
    n_jobs : int, optional
        Number of parallel workers. -1 means use all CPUs.
        Default is -1.
    progress_bar : bool, optional
        Whether to show progress bar. Default is False.

    Examples
    --------
    >>> backend = MpireBackend(n_jobs=4, progress_bar=True)
    >>> results = backend.map(lambda x: x * 2, [1, 2, 3, 4])
    >>> list(results)
    [2, 4, 6, 8]
    """

    def __init__(
        self,
        n_jobs: int = -1,
        progress_bar: bool = False,
    ):
        self._n_jobs = n_jobs
        self._progress_bar = progress_bar

    def _get_n_jobs(self, n_jobs: int) -> int:
        """Resolve number of jobs, handling -1 for all CPUs."""
        import os

        n = n_jobs if n_jobs > 0 else self._n_jobs
        if n == -1:
            return os.cpu_count() or 1
        return n

    def map(
        self,
        func: Callable[[T], T],
        items: Sequence[T],
        n_jobs: int = -1,
    ) -> List[T]:
        """Apply function to each item in parallel.

        Parameters
        ----------
        func : Callable[[T], T]
            Function to apply to each item.
        items : Sequence[T]
            Items to process.
        n_jobs : int, optional
            Number of parallel workers. If -1 or not specified,
            uses the instance default. Default is -1.

        Returns
        -------
        List[T]
            Results in same order as input items.
        """
        from pyapprox.util.optional_deps import import_optional_dependency

        import_optional_dependency(
            "mpire", feature_name="MpireBackend", extra_name="parallel"
        )
        from mpire import WorkerPool

        n = self._get_n_jobs(n_jobs)
        # mpire's map unpacks tuples, so wrap single items in tuples
        wrapped_items = [(item,) for item in items]
        with WorkerPool(n_jobs=n) as pool:
            return pool.map(
                func,
                wrapped_items,
                progress_bar=self._progress_bar,
                concatenate_numpy_output=False,
            )

    def starmap(
        self,
        func: Callable[..., T],
        items: Sequence[tuple],
        n_jobs: int = -1,
    ) -> List[T]:
        """Apply function with unpacked tuple arguments in parallel.

        Parameters
        ----------
        func : Callable[..., T]
            Function to apply with unpacked arguments.
        items : Sequence[tuple]
            Sequence of argument tuples.
        n_jobs : int, optional
            Number of parallel workers. If -1 or not specified,
            uses the instance default. Default is -1.

        Returns
        -------
        List[T]
            Results in same order as input items.
        """
        from pyapprox.util.optional_deps import import_optional_dependency

        import_optional_dependency(
            "mpire", feature_name="MpireBackend", extra_name="parallel"
        )
        from mpire import WorkerPool

        n = self._get_n_jobs(n_jobs)
        # mpire's map already unpacks tuples like starmap
        with WorkerPool(n_jobs=n) as pool:
            return pool.map(
                func,
                list(items),
                progress_bar=self._progress_bar,
                concatenate_numpy_output=False,
            )

    def backend_name(self) -> str:
        """Return backend identifier string.

        Returns
        -------
        str
            Human-readable backend name with configuration.
        """
        return f"mpire(n_jobs={self._n_jobs}, progress_bar={self._progress_bar})"
