"""CandidateInfo dataclass for adaptive sparse grid refinement.

Carries all information needed by error indicators to evaluate
a candidate subspace without accessing grid internals.
"""

from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple

from pyapprox.util.backends.protocols import Array
from pyapprox.surrogates.sparsegrids.subspace import (
    TensorProductSubspace,
)

# Type alias for config indices (tuple of ints)
ConfigIdx = Tuple[int, ...]


@dataclass
class CandidateInfo(Generic[Array]):
    """Information about a candidate subspace for error evaluation.

    Attributes
    ----------
    candidate_index : Array
        Full multi-index of the candidate, shape (nvars_index,).
    candidate_subspace : TensorProductSubspace[Array]
        The candidate subspace.
    all_samples : Array
        All unique samples across selected + this candidate,
        shape (nvars_physical, n_total).
    new_samples : Array
        Samples unique to this candidate, shape (nvars_physical, n_new).
    new_sample_local_indices : List[int]
        Local indices within candidate_subspace that are new.
    selected_surrogate : object
        CombinationSurrogate built from selected indices only.
    sel_plus_candidate_surrogate : object
        CombinationSurrogate built from selected + this candidate.

    config_idx : Optional[ConfigIdx]
        Config index for MF grids, None for SF.
    model_cost : Optional[float]
        Per-sample cost from CostModelProtocol, None for SF.
    subspace_cost : Optional[float]
        model_cost * len(new_sample_local_indices), None for SF.
    parent_indices : Optional[List[Array]]
        Reserved for future use. Always None for now.
    """

    candidate_index: Array
    candidate_subspace: TensorProductSubspace[Array]
    all_samples: Array
    new_samples: Array
    new_sample_local_indices: List[int]
    selected_surrogate: object
    sel_plus_candidate_surrogate: object

    # MF fields (None for single-fidelity)
    config_idx: Optional[ConfigIdx] = None
    model_cost: Optional[float] = None
    subspace_cost: Optional[float] = None

    # Reserved
    parent_indices: Optional[List[Array]] = None
