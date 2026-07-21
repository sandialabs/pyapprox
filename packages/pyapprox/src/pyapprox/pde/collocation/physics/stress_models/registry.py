"""Temporary re-export shim: moved to pyapprox.pde.constitutive.registry."""

from pyapprox.pde.constitutive.registry import (
    create_stress_model,
    list_stress_models,
    register_stress_model,
)

__all__ = [
    "register_stress_model",
    "create_stress_model",
    "list_stress_models",
]
