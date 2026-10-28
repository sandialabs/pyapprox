"""What an error indicator is told about a candidate subspace.

A candidate carries its own backward box, which is everything needed to
score it: the change any linear functional of the interpolant undergoes
when the candidate is added is a signed sum over that box.

Indicators also take a second argument, the grid, typed as
``AdaptiveGridViewProtocol``: what any indicator may read from it.
"""

from dataclasses import dataclass
from typing import (
    Dict,
    Generic,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from pyapprox.surrogates.sparsegrids.subspace import (
    TensorProductSubspace,
)
from pyapprox.util.backends.protocols import Array

# Type alias for config indices (tuple of ints)
ConfigIdx = Tuple[int, ...]

# Which subspaces a grid accessor reports on. The grid view protocol
# takes this rather than str so the fitter, which accepts only these
# values, satisfies it.
SubsetType = Literal["selected", "candidate", "all"]


@dataclass(frozen=True)
class Candidate(Generic[Array]):
    """A candidate subspace and the box that scores it.

    Frozen value object; fields are read directly.

    Attributes
    ----------
    index : Array
        Full multi-index, including config dimensions, shape (nvars,).
    subspace : TensorProductSubspace[Array]
        The candidate's subspace, with values set.
    box : Sequence[Tuple[int, TensorProductSubspace[Array]]]
        (sign, subspace) over the candidate's backward box. Summing any
        per-subspace quantity against these signs gives the change that
        quantity undergoes when the candidate is added.
    new_sample_local_indices : Sequence[int]
        Indices within ``subspace``'s samples that are new to the grid.
    config_idx : Optional[ConfigIdx]
        Config index for multi-fidelity grids, None for single fidelity.
    cost : float
        Per-sample model cost times the number of new samples.
    """

    index: Array
    subspace: TensorProductSubspace[Array]
    box: Sequence[Tuple[int, TensorProductSubspace[Array]]]
    new_sample_local_indices: Sequence[int]
    config_idx: Optional[ConfigIdx]
    cost: float


@dataclass(frozen=True, eq=False)
class SmolyakSelection(Generic[Array]):
    """The selected set's Smolyak terms at one moment.

    Identity-compared and hashable by identity, so consumers can memoize
    against it in a WeakKeyDictionary: the same object is handed to every
    candidate within a round, and a new one replaces it after a
    promotion.

    Attributes
    ----------
    terms : Tuple[Tuple[int, TensorProductSubspace[Array]], ...]
        (coefficient, subspace) for subspaces with a nonzero
        coefficient.
    """

    terms: Tuple[Tuple[int, TensorProductSubspace[Array]], ...]


@runtime_checkable
class AdaptiveGridViewProtocol(Protocol[Array]):
    """What an error indicator may read from the grid.

    Every indicator's ``grid`` parameter has this type, whether or not
    it reads anything, so all indicators share one call signature and a
    wrapper passes ``grid`` through unchanged. The fitter satisfies it
    and passes itself. It lists the methods indicators use; one that
    needs more of the grid adds a method here, which the fitter must
    then provide.
    """

    def get_samples(self, subset: SubsetType = "all") -> Dict[ConfigIdx, Array]:
        """Return the grid's unique samples, one array per config."""
        ...

    def selection(self) -> SmolyakSelection[Array]:
        """Return the selected set's Smolyak terms for this round."""
        ...
