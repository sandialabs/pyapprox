"""concurrent.futures-based parallel execution backend.

This module provides a parallel execution backend using loky's
ProcessPoolExecutor, which extends the standard library
ProcessPoolExecutor with cloudpickle serialization (needed for
closures and local functions).
"""

import os
from typing import Callable, List, Sequence, TypeVar

T = TypeVar("T")


class FuturesBackend:
    """Parallel execution backend using loky ProcessPoolExecutor.

    Uses loky's ProcessPoolExecutor which extends the standard library
    executor with cloudpickle serialization, enabling pickling of
    closures and locally-defined functions.

    Parameters
    ----------
    n_jobs : int, optional
        Number of parallel workers. -1 means use all CPUs.
        Default is -1.

    Examples
    --------
    >>> backend = FuturesBackend(n_jobs=4)
    >>> results = backend.map(lambda x: x * 2, [1, 2, 3, 4])
    >>> list(results)
    [2, 4, 6, 8]
    """

    def __init__(self, n_jobs: int = -1):
        self._n_jobs = n_jobs

    def _get_n_jobs(self, n_jobs: int = -1) -> int:
        """Resolve n_jobs to a concrete worker count.

        Parameters
        ----------
        n_jobs : int
            Requested worker count. If -1 or 0, uses instance default.

        Returns
        -------
        int
            Resolved worker count.
        """
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
        from joblib.externals.loky import ProcessPoolExecutor

        items_list = list(items)
        if not items_list:
            return []
        n = self._get_n_jobs(n_jobs)
        with ProcessPoolExecutor(max_workers=n) as executor:
            return list(executor.map(func, items_list))

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
        from joblib.externals.loky import ProcessPoolExecutor

        items_list = list(items)
        if not items_list:
            return []
        n = self._get_n_jobs(n_jobs)
        with ProcessPoolExecutor(max_workers=n) as executor:
            futures = [executor.submit(func, *args) for args in items_list]
            return [f.result() for f in futures]

    def backend_name(self) -> str:
        """Return backend identifier string.

        Returns
        -------
        str
            Human-readable backend name with configuration.
        """
        return f"futures(n_jobs={self._n_jobs})"
