"""Mesh implementations for Galerkin finite element methods."""

from pyapprox.util.optional_deps import package_available

__all__: list[str]

if package_available("skfem"):
    from pyapprox.pde.galerkin.mesh.obstructed import (
        ObstructedMesh2D,
    )
    from pyapprox.pde.galerkin.mesh.structured import (
        StructuredMesh1D,
        StructuredMesh2D,
        StructuredMesh3D,
    )
    from pyapprox.pde.galerkin.mesh.unstructured import (
        UnstructuredMesh2D,
    )

    __all__ = [
        "StructuredMesh1D",
        "StructuredMesh2D",
        "StructuredMesh3D",
        "UnstructuredMesh2D",
        "ObstructedMesh2D",
    ]
else:
    __all__ = []
