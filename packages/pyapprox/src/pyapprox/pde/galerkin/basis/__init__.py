"""Finite element basis implementations for Galerkin methods."""

from pyapprox.util.optional_deps import package_available

__all__: list[str]

if package_available("skfem"):
    from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis
    from pyapprox.pde.galerkin.basis.vector_lagrange import VectorLagrangeBasis

    __all__ = [
        "LagrangeBasis",
        "VectorLagrangeBasis",
    ]
else:
    __all__ = []
