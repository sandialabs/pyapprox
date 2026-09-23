"""Turning a candidate's error into the value that orders the queue.

Error is approximation mathematics: how much the surrogate moves when a
candidate is added. Priority is a policy: what that movement is worth
given what the candidate costs. Separating them lets either be replaced
without touching the other, and keeps cost out of every indicator.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.surrogates.sparsegrids.candidate_info import Candidate
from pyapprox.util.backends.protocols import Array


@runtime_checkable
class PriorityProtocol(Protocol[Array]):
    """Orders candidates from their error and their cost.

    Higher priority is refined sooner.
    """

    def __call__(self, error: float, candidate: Candidate[Array]) -> float: ...


class CostWeightedPriority(Generic[Array]):
    """Priority is error per unit cost.

    A candidate that moves the surrogate twice as much but costs three
    times as much is refined later, which is what makes refinement
    budget-aware rather than purely accuracy-driven.

    Falls back to the raw error when the cost is zero or negative, so a
    grid with no cost model still orders by error alone.
    """

    def __call__(self, error: float, candidate: Candidate[Array]) -> float:
        if candidate.cost > 0:
            return error / candidate.cost
        return error

    def __repr__(self) -> str:
        return "CostWeightedPriority()"
