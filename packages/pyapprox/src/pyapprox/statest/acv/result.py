"""ACV allocation result (leaf module — imports nothing from acv)."""

from dataclasses import dataclass
from typing import Generic

from pyapprox.util.backends.protocols import Array


@dataclass(frozen=True)
class ACVAllocationResult(Generic[Array]):
    """Result of allocation optimization for a single estimator configuration.

    Continuous Attributes (Optimization)
    -------------------------------------
    partition_ratios : Array, shape (nmodels-1,)
        Continuous ratios from optimization. Used for covariance computation
        during optimization where gradients are required.

    continuous_npartition_samples : Array, shape (npartitions,)
        Continuous sample counts. For reference/analysis only.

    objective_value : Array, shape (1,)
        Optimal objective function value (e.g., log-determinant of covariance).
        Kept as Array to preserve autograd computation graph.

    Discrete Attributes (Evaluation)
    --------------------------------
    npartition_samples : Array, shape (npartitions,), dtype=int
        Integer sample counts per partition. Use for sample generation.

    nsamples_per_model : Array, shape (nmodels,), dtype=int
        Integer sample counts per model.

    Metadata
    --------
    target_cost : float
        Requested computational budget.

    actual_cost : float
        Actual cost after rounding to integers.

    success : bool
        Whether allocation succeeded.

    message : str
        Status message or error description.
    """

    # Continuous (optimization)
    partition_ratios: Array
    continuous_npartition_samples: Array
    objective_value: Array  # Shape (1,) - keeps autograd graph

    # Discrete (evaluation)
    npartition_samples: Array
    nsamples_per_model: Array

    # Metadata
    target_cost: float
    actual_cost: float
    success: bool
    message: str = ""
