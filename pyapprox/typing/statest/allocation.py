"""Allocation result types for statistical estimators."""

from dataclasses import dataclass
from typing import Protocol, Generic, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array


@runtime_checkable
class AllocationResultProtocol(Protocol, Generic[Array]):
    """Minimum interface all allocation results must satisfy."""

    @property
    def nsamples_per_model(self) -> Array:
        """Number of samples per model. Shape (nmodels,)."""
        ...

    @property
    def actual_cost(self) -> float:
        """Actual computational cost after rounding."""
        ...

    @property
    def success(self) -> bool:
        """Whether allocation succeeded."""
        ...

    @property
    def message(self) -> str:
        """Status message or error description."""
        ...


@dataclass(frozen=True)
class CVAllocationResult(Generic[Array]):
    """Allocation result for CV estimators.

    Attributes
    ----------
    nsamples_per_model : Array
        Number of samples per model. Shape (nmodels,).
    actual_cost : float
        Actual computational cost.
    objective_value : Array
        Objective value. Shape (1,) to preserve autograd.
    success : bool
        Whether allocation succeeded.
    message : str
        Status message.
    """

    nsamples_per_model: Array
    actual_cost: float
    objective_value: Array
    success: bool
    message: str = ""
