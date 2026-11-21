"""Manufactured solution integration for Galerkin finite element tests.

Adapts the collocation manufactured solutions for use with Galerkin tests.

Superseded by :mod:`pyapprox.pde.verification` and its Galerkin adapter,
``pyapprox.pde.models.galerkin.verification``. New code must not import
this package; it is deleted once its remaining users move.
"""

from pyapprox.util.optional_deps import package_available

__all__: list[str]

if package_available("skfem"):
    from pyapprox.pde.galerkin.manufactured.adapter import (
        GalerkinManufacturedSolution,
        create_adr_manufactured_test,
        create_helmholtz_manufactured_test,
        create_hyperelasticity_manufactured_test,
    )

    __all__ = [
        "GalerkinManufacturedSolution",
        "create_adr_manufactured_test",
        "create_helmholtz_manufactured_test",
        "create_hyperelasticity_manufactured_test",
    ]
else:
    __all__ = []
