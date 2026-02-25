"""Benchmark registry for discovering and instantiating benchmarks.

Benchmarks are registered as factory functions that take a backend
and return a fixed benchmark instance.
"""

from typing import Dict, Callable, Generic, List, Any

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.benchmarks.protocols import BenchmarkProtocol

BenchmarkFactory = Callable[[Backend[Array]], Any]


class BenchmarkRegistry(Generic[Array]):
    """Registry for benchmark factories.

    Benchmarks are registered as factory functions that take a backend
    and return a fixed benchmark instance.

    Examples
    --------
    >>> @BenchmarkRegistry.register("ishigami_3d", category="sensitivity")
    ... def _ishigami_factory(bkd):
    ...     return create_ishigami_benchmark(bkd)
    >>> benchmark = BenchmarkRegistry.get("ishigami_3d", bkd)
    """

    _benchmarks: Dict[str, BenchmarkFactory] = {}
    _categories: Dict[str, List[str]] = {}
    _descriptions: Dict[str, str] = {}

    @classmethod
    def register(
        cls,
        name: str,
        category: str,
        description: str = "",
    ) -> Callable[[BenchmarkFactory], BenchmarkFactory]:
        """Decorator to register a benchmark factory.

        Parameters
        ----------
        name
            Unique name for the benchmark.
        category
            Category for grouping (e.g., "sensitivity", "optimization").
        description
            Short description of the benchmark.

        Returns
        -------
        Callable
            Decorator function.
        """

        def decorator(factory: BenchmarkFactory) -> BenchmarkFactory:
            cls._benchmarks[name] = factory
            if category not in cls._categories:
                cls._categories[category] = []
            cls._categories[category].append(name)
            cls._descriptions[name] = description
            return factory

        return decorator

    @classmethod
    def get(cls, name: str, bkd: Backend[Array]) -> Any:
        """Get a benchmark instance by name.

        Parameters
        ----------
        name
            Name of the benchmark.
        bkd
            Backend to use for array operations.

        Returns
        -------
        Any
            Benchmark instance.

        Raises
        ------
        KeyError
            If benchmark name is not registered.
        """
        if name not in cls._benchmarks:
            raise KeyError(
                f"Unknown benchmark: {name}. "
                f"Available: {list(cls._benchmarks.keys())}"
            )
        return cls._benchmarks[name](bkd)

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered benchmark names."""
        return list(cls._benchmarks.keys())

    @classmethod
    def list_category(cls, category: str) -> List[str]:
        """List benchmarks in a category."""
        return cls._categories.get(category, [])

    @classmethod
    def categories(cls) -> List[str]:
        """List all categories."""
        return list(cls._categories.keys())

    @classmethod
    def description(cls, name: str) -> str:
        """Get description for a benchmark."""
        return cls._descriptions.get(name, "")

    @classmethod
    def satisfying(
        cls, *protocols: type, bkd: Backend[Array],
    ) -> List[Any]:
        """Return benchmark instances satisfying all given protocols.

        Parameters
        ----------
        *protocols
            One or more ``@runtime_checkable`` protocol classes.
        bkd
            Backend for instantiation.

        Returns
        -------
        List[Any]
            Benchmark instances that satisfy every protocol.
        """
        results: List[Any] = []
        for name in cls._benchmarks:
            bm = cls._benchmarks[name](bkd)
            if all(isinstance(bm, p) for p in protocols):
                results.append(bm)
        return results

    @classmethod
    def names_satisfying(
        cls, *protocols: type, bkd: Backend[Array],
    ) -> List[str]:
        """Return names of benchmarks satisfying all given protocols.

        Parameters
        ----------
        *protocols
            One or more ``@runtime_checkable`` protocol classes.
        bkd
            Backend for instantiation.

        Returns
        -------
        List[str]
            Names of matching benchmarks.
        """
        names: List[str] = []
        for name in cls._benchmarks:
            bm = cls._benchmarks[name](bkd)
            if all(isinstance(bm, p) for p in protocols):
                names.append(name)
        return names

    @classmethod
    def clear(cls) -> None:
        """Clear all registered benchmarks (for testing)."""
        cls._benchmarks.clear()
        cls._categories.clear()
        cls._descriptions.clear()
