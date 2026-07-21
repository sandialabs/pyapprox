"""Temporary re-export shim: stress models moved to pyapprox.pde.constitutive.

Import from ``pyapprox.pde.constitutive`` instead. This shim is removed
once external callers have migrated.
"""

from pyapprox.pde.constitutive import (
    NeoHookeanStress,
    StressModelProtocol,
    StressModelWithSensitivityProtocol,
    StressModelWithTangentProtocol,
    SymbolicStressModelProtocol,
    create_stress_model,
    list_stress_models,
    register_stress_model,
)

__all__ = [
    "StressModelProtocol",
    "StressModelWithSensitivityProtocol",
    "StressModelWithTangentProtocol",
    "SymbolicStressModelProtocol",
    "NeoHookeanStress",
    "register_stress_model",
    "create_stress_model",
    "list_stress_models",
]
