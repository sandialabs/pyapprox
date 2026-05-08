"""Benchmark protocols for the PyApprox benchmarks module.

This module defines minimal protocols specific to benchmarks. Function protocols
are reused from pyapprox.interface.functions.protocols.

Structural protocols:
    DomainProtocol — input domain specification
    ConstraintProtocol — single constraint function

OED protocols:
    HasExactEIG — benchmark provides analytical expected information gain
"""

from __future__ import annotations

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


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
class ConstraintProtocol(Protocol, Generic[Array]):
    """Single constraint function."""

    def __call__(self, samples: Array) -> Array:
        """Evaluate constraint at samples."""
        ...

    def constraint_type(self) -> str:
        """Return constraint type: 'eq' or 'ineq'."""
        ...


@runtime_checkable
class HasExactEIG(Protocol, Generic[Array]):
    """Benchmark provides analytical expected information gain."""

    def exact_eig(self, weights: Array) -> float:
        """Return analytical EIG for given design weights."""
        ...
