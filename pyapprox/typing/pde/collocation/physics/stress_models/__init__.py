"""Hyperelastic stress models for spectral collocation physics.

Provides protocols, implementations, and a registry for pluggable
constitutive models used by HyperelasticityPhysics.
"""

from pyapprox.typing.pde.collocation.physics.stress_models.protocols import (
    StressModelProtocol,
    StressModelWithTangentProtocol,
    SymbolicStressModelProtocol,
)
from pyapprox.typing.pde.collocation.physics.stress_models.neo_hookean import (
    NeoHookeanStress,
)
from pyapprox.typing.pde.collocation.physics.stress_models.registry import (
    register_stress_model,
    create_stress_model,
    list_stress_models,
)

# Auto-register built-in stress models
register_stress_model(
    "neo_hookean", lambda **kw: NeoHookeanStress(**kw)
)

__all__ = [
    # Protocols
    "StressModelProtocol",
    "StressModelWithTangentProtocol",
    "SymbolicStressModelProtocol",
    # Implementations
    "NeoHookeanStress",
    # Registry
    "register_stress_model",
    "create_stress_model",
    "list_stress_models",
]
