"""GroupACV allocation result dataclasses."""

from dataclasses import dataclass
from typing import Generic, Optional, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Array_co


@runtime_checkable
class GroupACVAllocationProtocol(Protocol[Array_co]):
    """What a fitted estimator needs from an allocation.

    Whichever direction produced it. A budget-driven allocation and a
    tolerance-driven one minimize different quantities and stay distinct
    types -- the searches rank by the budget direction's criterion, and a
    cost passed off as one would rank wrongly -- but both are sample
    counts, and the counts alone fix an estimator's weights, its samples
    and its covariance.

    Covariant in the array type, since it only produces arrays.
    Read-only properties, so the frozen result dataclasses satisfy it.
    """

    @property
    def npartition_samples(self) -> Array_co:
        """Samples per partition, shape ``(npartitions,)``."""
        ...

    @property
    def nsamples_per_model(self) -> Array_co:
        """Samples per model, shape ``(nmodels,)``."""
        ...

    @property
    def success(self) -> bool:
        """Whether the allocation succeeded."""
        ...

    @property
    def message(self) -> str:
        """Why it did not, when it did not."""
        ...


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


@dataclass(frozen=True)
class GroupACVToleranceResult(Generic[Array]):
    """Result of a tolerance-driven GroupACV allocation.

    Deliberately a distinct type from
    :class:`GroupACVAllocationResult` rather than a reuse of it. The two
    directions minimize different quantities, and the searches rank
    candidate allocations by the budget-driven result's
    ``objective_value``. Were a cost reported in that field, those
    comparisons would rank by cost while appearing to rank by accuracy
    and silently pick the wrong allocation. Separate types make handing
    a tolerance-driven result to a search a type error instead.

    Continuous Attributes (Optimization)
    ------------------------------------
    relaxed_npartition_samples : Array or None
        Continuous (unrounded) partition sample counts.
        Shape (npartitions,). Stored when optimization succeeds.

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
    total_cost : float
        Cost of the returned allocation. This is the minimized quantity.
    constraint_value : Array
        Achieved value of the accuracy requirement. Shape (1,). At or
        below the requested tolerance whenever ``success`` is True.
    success : bool
        Whether allocation succeeded: the optimizer converged to an
        allocation meeting the requirement. When False, the allocation
        still meets it -- it is the best feasible one available -- but
        is not shown to be the cheapest.
    message : str
        Status message, including the optimizer's own reason on failure.
    final_iterate : Array or None
        Where the optimizer stopped, in sample counts, shape
        (npartitions,), whether or not it succeeded; None when no solve
        ran. Restarting from it (``init_guess=final_iterate[:, None]``)
        shows whether more steps would finish a failed solve.
    """

    npartition_samples: Array
    nsamples_per_model: Array
    total_cost: float
    constraint_value: Array  # Shape (1,) - keeps autograd graph
    success: bool
    message: str = ""
    relaxed_npartition_samples: Optional[Array] = None
    final_iterate: Optional[Array] = None
