"""Joblib-based parallel execution backend.

This module provides a parallel execution backend using joblib,
which is commonly used in the scikit-learn ecosystem and provides
good numpy array handling and caching capabilities.
"""

from typing import Any, Callable, List, Sequence, TypeVar

T = TypeVar("T")


class JoblibBackend:
    """Parallel execution backend using joblib.

    Joblib provides efficient parallel execution with good numpy
    array serialization and optional result caching.

    Parameters
    ----------
    n_jobs : int, optional
        Number of parallel workers. -1 means use all CPUs.
        Default is -1.
    prefer : str, optional
        Preferred backend: "processes" or "threads".
        Default is "processes".
    verbose : int, optional
        Verbosity level. Default is 0.

    Examples
    --------
    >>> backend = JoblibBackend(n_jobs=4)
    >>> results = backend.map(lambda x: x * 2, [1, 2, 3, 4])
    >>> list(results)
    [2, 4, 6, 8]
    """

    def __init__(
        self,
        n_jobs: int = -1,
        prefer: str = "processes",
        verbose: int = 0,
    ):
        self._n_jobs = n_jobs
        self._prefer = prefer
        self._verbose = verbose

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
            "joblib", feature_name="JoblibBackend", extra_name="parallel"
        )
        from joblib import Parallel, delayed

        n = n_jobs if n_jobs > 0 else self._n_jobs
        result: list[T] = Parallel(
            n_jobs=n,
            prefer=self._prefer,
            verbose=self._verbose,
        )(delayed(func)(item) for item in items)
        return result

    def starmap(
        self,
        func: Callable[..., T],
        items: Sequence[tuple[Any, ...]],
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
            "joblib", feature_name="JoblibBackend", extra_name="parallel"
        )
        from joblib import Parallel, delayed

        n = n_jobs if n_jobs > 0 else self._n_jobs
        return Parallel(
            n_jobs=n,
            prefer=self._prefer,
            verbose=self._verbose,
        )(delayed(func)(*args) for args in items)

    def backend_name(self) -> str:
        """Return backend identifier string.

        Returns
        -------
        str
            Human-readable backend name with configuration.
        """
        return f"joblib(n_jobs={self._n_jobs}, prefer={self._prefer})"
