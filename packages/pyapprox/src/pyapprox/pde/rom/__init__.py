"""Projection reduced-order models of discretized PDEs.

Snapshots
---------
- SnapshotSet: full-order states with the parameter and time of each
- collect_steady_snapshots / collect_transient_snapshots
- concatenate_snapshot_sets

Lifts
-----
- LiftProtocol: the field carrying the essential boundary data
- AlgebraicLift: the data on the constrained DOFs, zero elsewhere
- homogenize: subtract the lift, checking the snapshots carry its data
"""

from .lift import AlgebraicLift, LiftProtocol, homogenize
from .snapshots import (
    SnapshotSet,
    TrajectorySolverProtocol,
    collect_steady_snapshots,
    collect_transient_snapshots,
    concatenate_snapshot_sets,
)

__all__ = [
    "SnapshotSet",
    "TrajectorySolverProtocol",
    "collect_steady_snapshots",
    "collect_transient_snapshots",
    "concatenate_snapshot_sets",
    "LiftProtocol",
    "AlgebraicLift",
    "homogenize",
]
