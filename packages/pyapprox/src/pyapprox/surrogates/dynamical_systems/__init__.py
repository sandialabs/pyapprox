"""Dynamical systems learning: surrogates for ODE right-hand sides."""

from pyapprox.surrogates.dynamical_systems.batched_ode_residual import (
    BatchedBoundODEResidual,
)
from pyapprox.surrogates.dynamical_systems.dataset import SnapshotDataset
from pyapprox.surrogates.dynamical_systems.encoders import (
    IdentityEncoder,
    LinearEncoder,
)
from pyapprox.surrogates.dynamical_systems.losses import (
    DerivativeMatchingLoss,
    TrajectoryMatchingLoss,
)
from pyapprox.surrogates.dynamical_systems.protocols import (
    EncoderProtocol,
    LearnedFunctionProtocol,
)

__all__ = [
    "BatchedBoundODEResidual",
    "DerivativeMatchingLoss",
    "EncoderProtocol",
    "IdentityEncoder",
    "LearnedFunctionProtocol",
    "LinearEncoder",
    "SnapshotDataset",
    "TrajectoryMatchingLoss",
]
