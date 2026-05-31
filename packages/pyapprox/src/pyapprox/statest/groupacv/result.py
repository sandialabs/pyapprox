"""GroupACV allocation result dataclass."""

from dataclasses import dataclass
from typing import Generic

from pyapprox.util.backends.protocols import Array


@dataclass(frozen=True)
class GroupACVAllocationResult(Generic[Array]):
    """Allocation result for GroupACV estimators.

    Attributes
    ----------
    npartition_samples : Array
        Partition sample counts. Shape (npartitions,).
    nsamples_per_model : Array
        Sample counts per model. Shape (nmodels,).
    actual_cost : float
        Actual computational cost.
    objective_value : Array
        Objective value. Shape (1,).
    success : bool
        Whether allocation succeeded.
    message : str
        Status message.
    """

    npartition_samples: Array
    nsamples_per_model: Array
    actual_cost: float
    objective_value: Array  # Shape (1,) - keeps autograd graph
    success: bool
    message: str = ""
