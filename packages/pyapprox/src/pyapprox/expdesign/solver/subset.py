"""Searches over subsets of design variables.

Each solver chooses ``k`` of the candidate design variables, a sensor or a
group of observations each, to minimize a subset objective:

- ``ExhaustiveSubsetSolver`` scores every ``k``-subset, so it is exact and
  costs ``C(n, k)`` evaluations.
- ``GreedySubsetSolver`` adds the best candidate, or the best set of
  ``batch_size`` candidates, keeping earlier choices fixed.
- ``ExchangeSubsetSolver`` improves a given subset by swapping one chosen
  candidate for an unchosen one until no swap helps.

Greedy and exchange take the incremental protocol, so an objective with a
cheap update is used by the same code as one that rescores.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Generic, List, Optional, Sequence, Tuple

from pyapprox.expdesign.protocols.subset import (
    IncrementalSubsetObjectiveProtocol,
    State,
    SubsetObjectiveProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


@dataclass(frozen=True)
class SubsetSearchResult:
    """The subset a search chose.

    Attributes
    ----------
    subset : Tuple[int, ...]
        Indices of the chosen design variables, sorted.
    value : float
        The objective at ``subset``.
    nevaluations : int
        Number of subsets the search scored.
    """

    subset: Tuple[int, ...]
    value: float
    nevaluations: int


def _check_size(k: int, ncandidates: int) -> None:
    if not 1 <= k <= ncandidates:
        raise ValueError(f"k must lie in [1, {ncandidates}], got {k}")


def _check_incremental(objective: object) -> None:
    if not isinstance(objective, IncrementalSubsetObjectiveProtocol):
        raise TypeError(
            "objective must satisfy IncrementalSubsetObjectiveProtocol, got "
            f"{type(objective).__name__}; wrap a subset objective in "
            "ReevaluatingIncremental"
        )


def _argmin(values: Array, bkd: Backend[Array]) -> Tuple[int, float]:
    """Index and value of the first smallest entry."""
    index = bkd.to_int(bkd.argmin(values))
    return index, bkd.to_float(values[index])


class ExhaustiveSubsetSolver(Generic[Array]):
    """The best ``k``-subset, by scoring all ``C(n, k)`` of them.

    Ties go to the first subset in lexicographic order.

    Parameters
    ----------
    objective : SubsetObjectiveProtocol[Array]
        The objective to minimize.
    """

    def __init__(self, objective: SubsetObjectiveProtocol[Array]) -> None:
        if not isinstance(objective, SubsetObjectiveProtocol):
            raise TypeError(
                "objective must satisfy SubsetObjectiveProtocol, got "
                f"{type(objective).__name__}"
            )
        self._objective = objective

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._objective.bkd()

    def solve(self, k: int) -> SubsetSearchResult:
        """The best subset of ``k`` design variables."""
        ncandidates = self._objective.ncandidates()
        _check_size(k, ncandidates)
        best_subset: Tuple[int, ...] = ()
        best_value = float("inf")
        nevaluations = 0
        for subset in combinations(range(ncandidates), k):
            value = self._objective.value(subset)
            nevaluations += 1
            if value < best_value:
                best_subset, best_value = subset, value
        return SubsetSearchResult(best_subset, best_value, nevaluations)


class GreedySubsetSolver(Generic[Array, State]):
    """Grow a subset by the best addition at each step.

    Each step scores every set of ``batch_size`` unchosen candidates added
    to those already chosen, keeps the best, and fixes it. With
    ``batch_size = 1`` this is the classical greedy search; with
    ``batch_size = k`` it is exhaustive. The last step adds only what
    remains to reach ``k``. Ties go to the first addition in
    lexicographic order.

    Parameters
    ----------
    objective : IncrementalSubsetObjectiveProtocol[Array, State]
        The objective to minimize. Wrap a plain subset objective in
        ``ReevaluatingIncremental``.
    batch_size : int
        Number of candidates added per step. Default 1.
    """

    def __init__(
        self,
        objective: IncrementalSubsetObjectiveProtocol[Array, State],
        batch_size: int = 1,
    ) -> None:
        _check_incremental(objective)
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self._objective = objective
        self._batch_size = batch_size

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._objective.bkd()

    def solve(self, k: int) -> SubsetSearchResult:
        """A subset of ``k`` design variables, grown greedily."""
        bkd = self.bkd()
        ncandidates = self._objective.ncandidates()
        _check_size(k, ncandidates)
        state = self._objective.initial_state()
        chosen: List[int] = []
        value = float("inf")
        nevaluations = 0
        while len(chosen) < k:
            size = min(self._batch_size, k - len(chosen))
            remaining = [ii for ii in range(ncandidates) if ii not in chosen]
            additions = list(combinations(remaining, size))
            values = self._objective.values_after(state, additions)
            nevaluations += len(additions)
            index, value = _argmin(values, bkd)
            state = self._objective.add(state, additions[index])
            chosen.extend(additions[index])
        return SubsetSearchResult(tuple(sorted(chosen)), value, nevaluations)


class ExchangeSubsetSolver(Generic[Array, State]):
    """Improve a subset by swaps until none helps.

    Each sweep removes every set of ``b = swap_size`` chosen candidates and
    refills it with every set of ``b`` candidates not kept, removed ones
    included. Refilling with some removed candidates makes a smaller swap,
    so the ``C(k, b) C(n - k + b, b)`` scored subsets are all those that
    differ from the current one in at most ``b`` candidates; some are
    scored more than once. It makes the best swap if it lowers the value.
    Every accepted swap strictly lowers the value, so no subset repeats and
    the search ends in at most ``C(n, k)`` sweeps at a subset no such swap
    improves. With ``swap_size = 1`` this is the classical exchange; with
    ``swap_size = k`` one sweep reaches every ``k``-subset, so the search
    ends at the exhaustive optimum. A ``swap_size`` above ``k`` is capped
    at ``k``.

    Parameters
    ----------
    objective : IncrementalSubsetObjectiveProtocol[Array, State]
        The objective to minimize. Wrap a plain subset objective in
        ``ReevaluatingIncremental``.
    swap_size : int
        Number of candidates exchanged per swap. Default 1.
    """

    def __init__(
        self,
        objective: IncrementalSubsetObjectiveProtocol[Array, State],
        swap_size: int = 1,
    ) -> None:
        _check_incremental(objective)
        if swap_size < 1:
            raise ValueError(f"swap_size must be positive, got {swap_size}")
        self._objective = objective
        self._swap_size = swap_size

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._objective.bkd()

    def _best_swap(
        self, chosen: Tuple[int, ...]
    ) -> Tuple[Optional[Tuple[int, ...]], float, int]:
        """The best subset one swap from ``chosen``, its value and cost."""
        bkd = self.bkd()
        size = min(self._swap_size, len(chosen))
        best_subset: Optional[Tuple[int, ...]] = None
        best_value = float("inf")
        nevaluations = 0
        for removed in combinations(chosen, size):
            kept = [ii for ii in chosen if ii not in removed]
            pool = [ii for ii in range(self._objective.ncandidates()) if ii not in kept]
            additions = list(combinations(pool, size))
            state = self._objective.add(self._objective.initial_state(), kept)
            values = self._objective.values_after(state, additions)
            nevaluations += len(additions)
            index, value = _argmin(values, bkd)
            if value < best_value:
                best_subset = tuple(sorted([*kept, *additions[index]]))
                best_value = value
        return best_subset, best_value, nevaluations

    def solve(self, initial: Sequence[int]) -> SubsetSearchResult:
        """A subset of ``len(initial)`` design variables, improved by swaps.

        Parameters
        ----------
        initial : Sequence[int]
            Distinct indices to start from, for example a greedy result.
        """
        ncandidates = self._objective.ncandidates()
        _check_size(len(initial), ncandidates)
        if len(set(initial)) != len(initial) or not all(
            0 <= ii < ncandidates for ii in initial
        ):
            raise ValueError(
                f"initial must hold distinct indices in [0, {ncandidates}), got "
                f"{list(initial)}"
            )
        chosen = tuple(sorted(initial))
        empty = self._objective.initial_state()
        value = self.bkd().to_float(self._objective.values_after(empty, [chosen])[0])
        nevaluations = 1
        if len(chosen) == ncandidates:
            return SubsetSearchResult(chosen, value, nevaluations)
        while True:
            swapped, swapped_value, cost = self._best_swap(chosen)
            nevaluations += cost
            if swapped is None or not swapped_value < value:
                return SubsetSearchResult(chosen, value, nevaluations)
            chosen, value = swapped, swapped_value
