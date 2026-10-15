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

Trial spaces
------------
- TrialSpaceProtocol: a fixed reference plus the span of a basis
- LinearTrialSpace: over any BasisOperatorProtocol, projecting in a metric
- fit_pod_trial_space: POD through fit_kle_encoder
- snapshot_spectrum / projection_errors / reduced_gram: truncation
  diagnostics and the reduced mass
"""

from .lift import AlgebraicLift, LiftProtocol, homogenize
from .snapshots import (
    SnapshotSet,
    TrajectorySolverProtocol,
    collect_steady_snapshots,
    collect_transient_snapshots,
    concatenate_snapshot_sets,
)
from .trial_space import (
    LinearTrialSpace,
    TrialSpaceProtocol,
    fit_pod_trial_space,
    projection_errors,
    reduced_gram,
    snapshot_spectrum,
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
    "TrialSpaceProtocol",
    "LinearTrialSpace",
    "fit_pod_trial_space",
    "snapshot_spectrum",
    "projection_errors",
    "reduced_gram",
]
