"""Pointwise constitutive models (solver-neutral).

Provides protocols, implementations, and a registry for pluggable
hyperelastic stress models used by collocation and Galerkin physics,
plus typed coefficient function families (diffusion, velocity,
reaction) for ADR-type physics.
"""

from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    ConstantDiffusion,
    ConstantVelocity,
    CoordinateDiffusion,
    CoordinateVelocity,
    DiffusionFunctionProtocol,
    LinearReaction,
    NodalFieldDiffusion,
    NodalFieldVelocity,
    ReactionFunctionProtocol,
    ReactionFunctionWithSecondDerivativeProtocol,
    StateDependentDiffusionProtocol,
    VelocityFunctionProtocol,
)
from pyapprox.pde.constitutive.neo_hookean import (
    NeoHookeanStress,
)
from pyapprox.pde.constitutive.protocols import (
    StressModelProtocol,
    StressModelWithSensitivityProtocol,
    StressModelWithTangentProtocol,
    SymbolicStressModelProtocol,
)
from pyapprox.pde.constitutive.registry import (
    create_stress_model,
    list_stress_models,
    register_stress_model,
)

# Auto-register built-in stress models
register_stress_model("neo_hookean", lambda **kw: NeoHookeanStress(**kw))

__all__ = [
    # Coefficient function protocols
    "DiffusionFunctionProtocol",
    "StateDependentDiffusionProtocol",
    "VelocityFunctionProtocol",
    "ReactionFunctionProtocol",
    "ReactionFunctionWithSecondDerivativeProtocol",
    # Coefficient function implementations
    "ConstantDiffusion",
    "CoordinateDiffusion",
    "NodalFieldDiffusion",
    "ConstantVelocity",
    "CoordinateVelocity",
    "NodalFieldVelocity",
    "LinearReaction",
    "CallableReaction",
    # Protocols
    "StressModelProtocol",
    "StressModelWithSensitivityProtocol",
    "StressModelWithTangentProtocol",
    "SymbolicStressModelProtocol",
    # Implementations
    "NeoHookeanStress",
    # Registry
    "register_stress_model",
    "create_stress_model",
    "list_stress_models",
]
