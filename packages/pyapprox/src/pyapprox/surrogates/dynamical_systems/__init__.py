"""Dynamical systems learning: surrogates for ODE right-hand sides."""

from pyapprox.surrogates.dynamical_systems.batched_ode_residual import (
    BatchedBoundODEResidual,
)
from pyapprox.surrogates.dynamical_systems.dataset import SnapshotDataset
from pyapprox.surrogates.dynamical_systems.encoders import (
    IdentityEncoder,
    LinearEncoder,
)
from pyapprox.surrogates.dynamical_systems.fitters import (
    FixedPoissonVariableHamiltonianDerivativeMatchingFitter,
    VariablePoissonFixedHamiltonianDerivativeMatchingFitter,
)
from pyapprox.surrogates.dynamical_systems.losses import (
    DerivativeMatchingLoss,
    TrajectoryMatchingLoss,
)
from pyapprox.surrogates.dynamical_systems.protocols import (
    EncoderProtocol,
    LearnedFunctionProtocol,
)
from pyapprox.surrogates.dynamical_systems.surrogates.fixed_poisson_variable_hamiltonian import (  # noqa: E501
    FixedPoissonVariableHamiltonianSurrogate,
)
from pyapprox.surrogates.dynamical_systems.surrogates.variable_poisson_fixed_hamiltonian import (  # noqa: E501
    VariablePoissonFixedHamiltonianSurrogate,
)

__all__ = [
    "BatchedBoundODEResidual",
    "DerivativeMatchingLoss",
    "EncoderProtocol",
    "FixedPoissonVariableHamiltonianDerivativeMatchingFitter",
    "FixedPoissonVariableHamiltonianSurrogate",
    "IdentityEncoder",
    "LearnedFunctionProtocol",
    "LinearEncoder",
    "SnapshotDataset",
    "TrajectoryMatchingLoss",
    "VariablePoissonFixedHamiltonianDerivativeMatchingFitter",
    "VariablePoissonFixedHamiltonianSurrogate",
]
