"""Priority queue for adaptive index refinement.

This module provides a priority queue implementation for managing
candidate indices in adaptive sparse grids and PCE refinement.
"""

import heapq
from typing import Generic, List, Tuple

from pyapprox.util.backends.protocols import Array


class PriorityQueue(Generic[Array]):
    """Priority queue for candidate index ranking.

    Uses a min-heap internally. To get max-priority behavior
    (higher priority = selected first), priorities are negated.

    Parameters
    ----------
    max_priority : bool
        If True (default), higher priorities are selected first.
        If False, lower priorities are selected first.

    Examples
    --------
    >>> queue = PriorityQueue()
    >>> queue.put(priority=0.5, error=0.1, index_id=0)
    >>> queue.put(priority=0.8, error=0.2, index_id=1)
    >>> queue.get()  # Returns highest priority first
    (0.8, 0.2, 1)
    """

    def __init__(self, max_priority: bool = True):
        self._heap: List[Tuple[float, int, float, float, int]] = []
        self._counter = 0  # Tiebreaker for equal priorities
        self._max_priority = max_priority

    def empty(self) -> bool:
        """Return True if the queue is empty.

        Returns
        -------
        bool
            True if no items in queue.
        """
        return len(self._heap) == 0

    def __len__(self) -> int:
        """Return the number of items in the queue."""
        return len(self._heap)

    def put(self, priority: float, error: float, index_id: int) -> None:
        """Add a candidate to the queue.

        Parameters
        ----------
        priority : float
            Refinement priority.
        error : float
            Error estimate for this candidate.
        index_id : int
            Identifier for the candidate index.
        """
        # Negate priority for max-heap behavior
        heap_priority = -priority if self._max_priority else priority
        # Use counter as tiebreaker for FIFO ordering on equal priorities
        heapq.heappush(
            self._heap, (heap_priority, self._counter, priority, error, index_id)
        )
        self._counter += 1

    def get(self) -> Tuple[float, float, int]:
        """Remove and return the highest priority candidate.

        Returns
        -------
        Tuple[float, float, int]
            (priority, error, index_id) for the highest priority candidate.

        Raises
        ------
        IndexError
            If the queue is empty.
        """
        if self.empty():
            raise IndexError("Priority queue is empty")
        _, _, priority, error, index_id = heapq.heappop(self._heap)
        return priority, error, index_id

    def peek(self) -> Tuple[float, float, int]:
        """Return the highest priority candidate without removing it.

        Returns
        -------
        Tuple[float, float, int]
            (priority, error, index_id) for the highest priority candidate.

        Raises
        ------
        IndexError
            If the queue is empty.
        """
        if self.empty():
            raise IndexError("Priority queue is empty")
        _, _, priority, error, index_id = self._heap[0]
        return priority, error, index_id

    def clear(self) -> None:
        """Remove all items from the queue."""
        self._heap.clear()
        self._counter = 0

    def __repr__(self) -> str:
        """Show all entries sorted by priority (highest first if max_priority)."""
        if self.empty():
            return "PriorityQueue(empty)"
        # Sort a copy of the heap so highest-priority entries come first
        entries = sorted(self._heap)
        lines = [f"PriorityQueue({len(entries)} entries):"]
        for heap_pri, counter, priority, error, index_id in entries:
            lines.append(
                f"  idx={index_id}  priority={priority:.6e}  error={error:.6e}"
            )
        return "\n".join(lines)
