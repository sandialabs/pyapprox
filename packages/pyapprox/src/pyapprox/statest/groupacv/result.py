"""GroupACV allocation result dataclass."""

from dataclasses import dataclass
from typing import Generic, Optional

from pyapprox.util.backends.protocols import Array


@dataclass(frozen=True)
class GroupACVAllocationResult(Generic[Array]):
    """Allocation result for GroupACV estimators.

    Sample counts appear in two forms. Continuous counts are the relaxation
    explored during optimization, where gradients are required; the template
    estimator's math is float-only and rejects integer input. Discrete counts
    are the post-rounding numbers of samples actually drawn, and are what
    :class:`FittedGroupACVEstimator` requires.

    Continuous Attributes (Optimization)
    ------------------------------------
    relaxed_npartition_samples : Array or None
        Continuous (unrounded) partition sample counts. Shape (npartitions,).
        Always stored when optimization succeeds. Same as
        ``npartition_samples`` when ``round_nsamples=False``.
    objective_value : Array
        Objective value. Shape (1,).

    Discrete Attributes (Evaluation)
    --------------------------------
    npartition_samples : Array
        Partition sample counts. Shape (npartitions,). Use for sample
        generation. Integer when ``round_nsamples=True``, float otherwise.
    nsamples_per_model : Array
        Sample counts per model. Shape (nmodels,). Integer when
        ``round_nsamples=True``, float otherwise.

    Other Attributes
    ----------------
    actual_cost : float
        Actual computational cost.
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
    relaxed_npartition_samples: Optional[Array] = None
