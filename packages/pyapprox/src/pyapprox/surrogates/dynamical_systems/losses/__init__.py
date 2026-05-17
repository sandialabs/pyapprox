"""Loss functions for dynamical systems learning."""

from pyapprox.surrogates.dynamical_systems.losses.derivative_matching import (
    DerivativeMatchingLoss,
)
from pyapprox.surrogates.dynamical_systems.losses.trajectory_matching import (
    TrajectoryMatchingLoss,
)

__all__ = ["DerivativeMatchingLoss", "TrajectoryMatchingLoss"]
