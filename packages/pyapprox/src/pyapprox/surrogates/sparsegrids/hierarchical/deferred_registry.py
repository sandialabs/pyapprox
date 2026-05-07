"""Deferred refinement registry with multi-blocker support."""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class DeferredTask:
    """A refinement task blocked by one or more incomplete subspaces."""

    point_id: int
    direction: int
    target_subspace: Tuple[int, ...]
    blockers: Set[Tuple[int, ...]] = field(default_factory=set)


class DeferredRefinementRegistry:
    """Tracks deferred refinements waiting on subspace completion.

    A refinement of point P in direction d targets subspace S. If S is
    not yet admissible, some backward-neighbor subspaces of S are
    incomplete. This registry holds the task until ALL blockers are
    resolved.
    """

    def __init__(self) -> None:
        self._by_blocker: Dict[Tuple[int, ...], List[DeferredTask]] = {}
        self._all_tasks: List[DeferredTask] = []

    def defer(
        self,
        point_id: int,
        direction: int,
        target_subspace: Tuple[int, ...],
        blockers: Set[Tuple[int, ...]],
    ) -> None:
        """Register a deferred refinement with all its blockers."""
        task = DeferredTask(
            point_id, direction, target_subspace, set(blockers)
        )
        self._all_tasks.append(task)
        for b in blockers:
            self._by_blocker.setdefault(b, []).append(task)

    def notify_complete(
        self, completed_subspace: Tuple[int, ...]
    ) -> List[DeferredTask]:
        """Notify that a subspace is complete.

        Removes this subspace from each task's blocker set. Returns
        tasks whose blocker set is now empty (fully unblocked).
        """
        released: List[DeferredTask] = []
        tasks = self._by_blocker.pop(completed_subspace, [])
        for task in tasks:
            task.blockers.discard(completed_subspace)
            if not task.blockers:
                released.append(task)
                self._all_tasks.remove(task)
        return released

    def empty(self) -> bool:
        return len(self._all_tasks) == 0

    def n_deferred(self) -> int:
        return len(self._all_tasks)

    def by_blocker(
        self, blocker: Tuple[int, ...]
    ) -> List[DeferredTask]:
        return list(self._by_blocker.get(blocker, []))
