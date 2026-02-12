"""Manufactured solution integration for Galerkin finite element tests.

Adapts the collocation manufactured solutions for use with Galerkin tests.
"""

from pyapprox.typing.pde.galerkin.manufactured.adapter import (
    GalerkinManufacturedSolutionAdapter,
    GalerkinHyperelasticityAdapter,
    create_adr_manufactured_test,
    create_helmholtz_manufactured_test,
    create_hyperelasticity_manufactured_test,
)

__all__ = [
    "GalerkinManufacturedSolutionAdapter",
    "GalerkinHyperelasticityAdapter",
    "create_adr_manufactured_test",
    "create_helmholtz_manufactured_test",
    "create_hyperelasticity_manufactured_test",
]
