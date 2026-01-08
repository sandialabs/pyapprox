"""Benchmark dataclasses.

These dataclasses provide concrete implementations that satisfy the benchmark
protocols. They use composition, not inheritance.
"""

from dataclasses import dataclass
from typing import Generic, TypeVar, Sequence, Any

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.typing.benchmarks.protocols import (
    DomainProtocol,
    GroundTruthProtocol,
    ConstraintProtocol,
)

GT = TypeVar("GT", bound=GroundTruthProtocol)


@dataclass
class BoxDomain(Generic[Array]):
    """Rectangular domain with bounds."""

    _bounds: Array
    _bkd: Backend[Array]

    def bounds(self) -> Array:
        """Return bounds of shape (nvars, 2)."""
        return self._bounds

    def nvars(self) -> int:
        """Return number of variables."""
        return int(self._bounds.shape[0])

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd


@dataclass
class Benchmark(Generic[Array, GT]):
    """Base benchmark: function + domain + ground truth (no prior)."""

    _name: str
    _function: FunctionProtocol[Array]
    _domain: DomainProtocol[Array]
    _ground_truth: GT
    _description: str = ""
    _reference: str = ""

    def name(self) -> str:
        """Return the benchmark name."""
        return self._name

    def function(self) -> FunctionProtocol[Array]:
        """Return the benchmark function."""
        return self._function

    def domain(self) -> DomainProtocol[Array]:
        """Return the domain specification."""
        return self._domain

    def ground_truth(self) -> GT:
        """Return the ground truth object."""
        return self._ground_truth

    def description(self) -> str:
        """Return the benchmark description."""
        return self._description

    def reference(self) -> str:
        """Return the benchmark reference."""
        return self._reference


@dataclass
class BenchmarkWithPrior(Generic[Array, GT]):
    """Benchmark with prior distribution (for UQ, sensitivity analysis)."""

    _name: str
    _function: FunctionProtocol[Array]
    _domain: DomainProtocol[Array]
    _ground_truth: GT
    _prior: Any
    _description: str = ""
    _reference: str = ""

    def name(self) -> str:
        """Return the benchmark name."""
        return self._name

    def function(self) -> FunctionProtocol[Array]:
        """Return the benchmark function."""
        return self._function

    def domain(self) -> DomainProtocol[Array]:
        """Return the domain specification."""
        return self._domain

    def ground_truth(self) -> GT:
        """Return the ground truth object."""
        return self._ground_truth

    def prior(self) -> Any:
        """Return the prior distribution."""
        return self._prior

    def description(self) -> str:
        """Return the benchmark description."""
        return self._description

    def reference(self) -> str:
        """Return the benchmark reference."""
        return self._reference


@dataclass
class ConstrainedBenchmark(Generic[Array, GT]):
    """Optimization benchmark with constraints."""

    _name: str
    _function: FunctionProtocol[Array]
    _domain: DomainProtocol[Array]
    _ground_truth: GT
    _constraints: Sequence[ConstraintProtocol[Array]]
    _description: str = ""
    _reference: str = ""

    def name(self) -> str:
        """Return the benchmark name."""
        return self._name

    def function(self) -> FunctionProtocol[Array]:
        """Return the benchmark function."""
        return self._function

    def domain(self) -> DomainProtocol[Array]:
        """Return the domain specification."""
        return self._domain

    def ground_truth(self) -> GT:
        """Return the ground truth object."""
        return self._ground_truth

    def constraints(self) -> Sequence[ConstraintProtocol[Array]]:
        """Return the list of constraints."""
        return self._constraints

    def description(self) -> str:
        """Return the benchmark description."""
        return self._description

    def reference(self) -> str:
        """Return the benchmark reference."""
        return self._reference
