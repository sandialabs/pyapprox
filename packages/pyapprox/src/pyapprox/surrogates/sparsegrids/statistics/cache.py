"""Per-subspace memoization for statistics used during refinement.

Scoring re-examines every candidate each round, and a candidate's score
is a signed sum over its backward box. The same subspace appears in many
boxes across many rounds, so the statistic it contributes is computed
once and reused.
"""

from typing import (
    Callable,
    Generic,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    runtime_checkable,
)
from weakref import WeakKeyDictionary

from pyapprox.surrogates.sparsegrids.subspace import (
    TensorProductSubspace,
)
from pyapprox.util.backends.protocols import Array

T = TypeVar("T")


S = TypeVar("S", bound="SignedSummableProtocol")


@runtime_checkable
class SignedSummableProtocol(Protocol):
    """A statistic that can be combined over a backward box.

    ``box_sum`` needs exactly two operations: multiplication by a +1/-1
    sign on the left, and addition to another value of the same kind.
    Backend arrays satisfy this, and so does anything else carrying its
    own arithmetic, such as a ``PolynomialChaosExpansion``.

    Both methods are typed to return the implementing type rather than
    the protocol, so folding a sequence of them keeps the caller's
    statistic type instead of widening to the protocol at the first
    operation.
    """

    def __rmul__(self: S, sign: int) -> S:
        ...

    def __add__(self: S, other: S) -> S:
        ...


class SubspaceCache(Generic[Array, T]):
    """Memoize a statistic against the subspace object that produced it.

    Keyed by subspace identity rather than by multi-index: two grids, or
    two fidelities of one grid, can hold different subspaces carrying
    equal indices, and their statistics must not collide.

    Entries are valid for the subspace's lifetime because subspace
    values are write-once. The mapping holds weak references, so a
    subspace dropped by the grid is not kept alive by having been
    measured.

    Generic in the backend array type and in the statistic type: a
    moment is an array, while a PCE conversion is a map from
    multi-index to coefficients.

    Parameters
    ----------
    statistic : Callable[[TensorProductSubspace[Array]], T]
        Computes the statistic for one subspace. Called at most once per
        subspace.

    Examples
    --------
    >>> from pyapprox.surrogates.sparsegrids.statistics.subspace_moments import (
    ...     subspace_mean,
    ... )
    >>> cache = SubspaceCache(subspace_mean)
    >>> # cache.get(subspace) computes on first call, reuses after
    """

    def __init__(
        self, statistic: Callable[[TensorProductSubspace[Array]], T]
    ) -> None:
        self._statistic = statistic
        self._values: "WeakKeyDictionary[TensorProductSubspace[Array], T]" = (
            WeakKeyDictionary()
        )

    def get(self, subspace: TensorProductSubspace[Array]) -> T:
        """Return the statistic for a subspace, computing it if needed.

        Parameters
        ----------
        subspace : TensorProductSubspace[Array]
            Subspace to measure. Must have values set.

        Returns
        -------
        T
            The memoized statistic.
        """
        cached = self._values.get(subspace)
        if cached is None:
            cached = self._statistic(subspace)
            self._values[subspace] = cached
        return cached

    def nentries(self) -> int:
        """Return how many subspaces are currently memoized."""
        return len(self._values)

    def __repr__(self) -> str:
        return f"SubspaceCache(nentries={self.nentries()})"


def box_sum(
    cache: "SubspaceCache[Array, S]",
    terms: Sequence[Tuple[int, TensorProductSubspace[Array]]],
) -> S:
    """Return the signed sum of a cached statistic over a box.

    For a statistic s that is linear in the subspaces, the change from
    adding subspace k is the signed sum over its backward box:

        Delta s = sum_e (-1)^|e| s_{k-e}

    The statistic only has to carry its own arithmetic: this folds with
    a sign multiply and an add, and never indexes, reshapes or calls a
    backend method. An array satisfies that, and so does an expansion
    that adds by grouping like terms across differing index sets.

    Parameters
    ----------
    cache : SubspaceCache[Array, S]
        Cache supplying each subspace's statistic.
    terms : Sequence[Tuple[int, TensorProductSubspace[Array]]]
        (sign, subspace) pairs, as produced from a backward box.

    Returns
    -------
    S
        The signed sum, of whatever kind the statistic is.

    Raises
    ------
    ValueError
        If terms is empty, since there is nothing to fold from.
    """
    if len(terms) == 0:
        raise ValueError("cannot sum over an empty box")
    sign, subspace = terms[0]
    total = sign * cache.get(subspace)
    for sign, subspace in terms[1:]:
        total = total + sign * cache.get(subspace)
    return total
