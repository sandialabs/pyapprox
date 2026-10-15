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
- TrialSpaceProtocol: a decoder with its tangent and curvature, linear
  or curved
- LinearTrialSpaceProtocol: the affine case, a reference plus a basis
- TangentOperatorProtocol: a decoder tangent as actions
- LinearTrialSpace: over any BasisOperatorProtocol, projecting in a metric
- fit_pod_trial_space: POD through fit_kle_encoder
- snapshot_spectrum / projection_errors / reduced_gram: truncation
  diagnostics and the reduced mass

Reduced residuals
-----------------
- ReducedResidualProtocol: the projected residual and its Jacobian
- ReducedResidualDerivatives: its optional parameter Jacobian
- ParameterizedResidualProtocol: the raw full-order residual consumed
- ProjectedResidual: the exact projection, the reference for every
  cheaper reduced residual

Affine projection
-----------------
- AffineProjectedResidual / fit_affine_projection: precomputed pieces of
  a residual affine in the state and the parameters
- ComposedParameterizedResidualProtocol: a raw residual exposing the
  operator and parameterization whose declarations are checked

Checks
------
- ReducedResidualCheckProtocol: compares a fitted reduced residual with
  a reference, run on a fit's result
- AgreementCheck: value and Jacobian at given parameters and codes

Steady solves
-------------
- solve_reduced_steady / ReducedNewtonResidual: Newton on any reduced
  residual
- full_state: g(t) + phi(z)

Hyper-reduction
---------------
- ElementSampledResidual: the reduced residual summed over a sample of
  elements, each with its own reduction (ECSW now, DEIM later)
- unit_weight_residual / ecsw_training_matrix / fit_ecsw / build_ecsw:
  ECSW in three steps (training matrix, weights, residual)
- ElementTargetProtocol / ResidualTargets / JacobianTargets: what the
  weights must reproduce
- ECSWFit: the hyper-reduced residual and its fit
"""

from .affine import (
    AffineProjectedResidual,
    ComposedParameterizedResidualProtocol,
    fit_affine_projection,
)
from .checks import AgreementCheck, ReducedResidualCheckProtocol
from .ecsw import (
    ECSWFit,
    ElementTargetProtocol,
    JacobianTargets,
    ResidualTargets,
    build_ecsw,
    ecsw_training_matrix,
    fit_ecsw,
    unit_weight_residual,
)
from .lift import AlgebraicLift, LiftProtocol, homogenize
from .projection import (
    ParameterizedResidualProtocol,
    ProjectedResidual,
    ReducedResidualDerivatives,
    ReducedResidualProtocol,
)
from .sampled import ElementSampledResidual
from .snapshots import (
    SnapshotSet,
    TrajectorySolverProtocol,
    collect_steady_snapshots,
    collect_transient_snapshots,
    concatenate_snapshot_sets,
)
from .steady import ReducedNewtonResidual, full_state, solve_reduced_steady
from .trial_space import (
    LinearTrialSpace,
    LinearTrialSpaceProtocol,
    TangentOperatorProtocol,
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
    "LinearTrialSpaceProtocol",
    "TangentOperatorProtocol",
    "LinearTrialSpace",
    "fit_pod_trial_space",
    "snapshot_spectrum",
    "projection_errors",
    "reduced_gram",
    "ReducedResidualProtocol",
    "ReducedResidualDerivatives",
    "ParameterizedResidualProtocol",
    "ProjectedResidual",
    "AffineProjectedResidual",
    "ComposedParameterizedResidualProtocol",
    "fit_affine_projection",
    "ReducedResidualCheckProtocol",
    "AgreementCheck",
    "ReducedNewtonResidual",
    "solve_reduced_steady",
    "full_state",
    "ElementSampledResidual",
    "ECSWFit",
    "ElementTargetProtocol",
    "ResidualTargets",
    "JacobianTargets",
    "unit_weight_residual",
    "ecsw_training_matrix",
    "fit_ecsw",
    "build_ecsw",
]
