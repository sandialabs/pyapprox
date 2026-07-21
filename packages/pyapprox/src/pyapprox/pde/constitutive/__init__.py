"""Pointwise constitutive models (solver-neutral).

Provides protocols, implementations, and a registry for pluggable
hyperelastic stress models used by collocation and Galerkin physics.
"""

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
