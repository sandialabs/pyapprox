"""Benchmark protocols for the PyApprox typing module.

This module defines minimal protocols specific to benchmarks. Function protocols
are reused from pyapprox.typing.interface.functions.protocols.
"""

from typing import Protocol, runtime_checkable, Generic, Sequence, Any

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.interface.functions.protocols.function import (
    FunctionProtocol,
)


@runtime_checkable
class DomainProtocol(Protocol, Generic[Array]):
    """Input domain specification."""

    def bounds(self) -> Array:
        """Return bounds of shape (nvars, 2)."""
        ...

    def nvars(self) -> int:
        """Return number of variables."""
        ...

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        ...


@runtime_checkable
class GroundTruthProtocol(Protocol):
    """Ground truth for a benchmark - what is known/computable."""

    def available(self) -> Sequence[str]:
        """Return list of available ground truth properties."""
        ...

    def get(self, name: str) -> Any:
        """Get a ground truth value by name."""
        ...


@runtime_checkable
class BenchmarkProtocol(Protocol, Generic[Array]):
    """Fixed benchmark instance with ground truth."""

    def name(self) -> str:
        """Return the benchmark name."""
        ...

    def function(self) -> FunctionProtocol[Array]:
        """Return the benchmark function."""
        ...

    def ground_truth(self) -> GroundTruthProtocol:
        """Return the ground truth object."""
        ...

    def domain(self) -> DomainProtocol[Array]:
        """Return the domain specification."""
        ...


@runtime_checkable
class BenchmarkWithPriorProtocol(Protocol, Generic[Array]):
    """Benchmark with probability distribution (for UQ, sensitivity)."""

    def name(self) -> str:
        """Return the benchmark name."""
        ...

    def function(self) -> FunctionProtocol[Array]:
        """Return the benchmark function."""
        ...

    def ground_truth(self) -> GroundTruthProtocol:
        """Return the ground truth object."""
        ...

    def domain(self) -> DomainProtocol[Array]:
        """Return the domain specification."""
        ...

    def prior(self) -> Any:
        """Return the prior distribution."""
        ...


@runtime_checkable
class ConstraintProtocol(Protocol, Generic[Array]):
    """Single constraint function."""

    def __call__(self, samples: Array) -> Array:
        """Evaluate constraint at samples."""
        ...

    def constraint_type(self) -> str:
        """Return constraint type: 'eq' or 'ineq'."""
        ...


@runtime_checkable
class ConstrainedBenchmarkProtocol(Protocol, Generic[Array]):
    """Benchmark with constraints (extends base via composition)."""

    def name(self) -> str:
        """Return the benchmark name."""
        ...

    def function(self) -> FunctionProtocol[Array]:
        """Return the benchmark function."""
        ...

    def ground_truth(self) -> GroundTruthProtocol:
        """Return the ground truth object."""
        ...

    def domain(self) -> DomainProtocol[Array]:
        """Return the domain specification."""
        ...

    def constraints(self) -> Sequence[ConstraintProtocol[Array]]:
        """Return the list of constraints."""
        ...
