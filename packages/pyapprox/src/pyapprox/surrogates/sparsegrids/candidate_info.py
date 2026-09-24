"""What an error indicator is told about a candidate subspace.

A candidate carries its own backward box, which is everything needed to
score it: the change any linear functional of the interpolant undergoes
when the candidate is added is a signed sum over that box.

Indicators that need more than the candidate take a second argument, a
read-only view of the grid. Each declares the narrowest protocol it
actually uses, so its signature states its dependencies.
"""

from dataclasses import dataclass
from typing import (
    Dict,
    Generic,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

from pyapprox.surrogates.sparsegrids.subspace import (
    TensorProductSubspace,
)
from pyapprox.util.backends.protocols import Array

# Type alias for config indices (tuple of ints)
ConfigIdx = Tuple[int, ...]

# Covariant counterpart of Array, for protocols that only return arrays.
ArrayCov = TypeVar("ArrayCov", covariant=True)


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
class SampleSourceProtocol(Protocol[ArrayCov]):
    """Supplies the grid's samples.

    The multi-fidelity fitter returns one array per config; the
    single-fidelity wrapper returns a bare array. Consumers that do not
    care which must handle both.

    The type variable is covariant because the array appears only in
    return position: something that supplies arrays of a subtype is a
    valid supplier of the supertype. The invariant project-wide
    ``Array`` cannot express that in a protocol this shape.
    """

    def get_samples(
        self, subset: str = "all"
    ) -> Union[ArrayCov, Dict[ConfigIdx, ArrayCov]]: ...


@runtime_checkable
class SelectionSourceProtocol(Protocol[Array]):
    """Supplies the current selected-set snapshot."""

    def selection(self) -> SmolyakSelection[Array]: ...


@runtime_checkable
class AdaptiveGridViewProtocol(Protocol[Array]):
    """Read-only view of an adaptive grid, passed to error indicators.

    The fitter satisfies this and passes itself. Indicators should type
    their ``grid`` parameter with the narrowest protocol they use --- or
    with ``object`` when they need nothing from the grid --- rather than
    with this one.
    """

    def get_samples(
        self, subset: str = "all"
    ) -> Union[Array, Dict[ConfigIdx, Array]]: ...

    def get_values(
        self, subset: str = "all"
    ) -> Union[Optional[Array], Dict[ConfigIdx, Optional[Array]]]: ...

    def get_selected_indices(self) -> Array: ...

    def get_candidate_indices(self) -> Optional[Array]: ...

    def nvars_physical(self) -> int: ...

    def selection(self) -> SmolyakSelection[Array]: ...
