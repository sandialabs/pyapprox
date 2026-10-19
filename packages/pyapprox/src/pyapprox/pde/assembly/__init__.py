"""Assembly of element contributions into a linear-algebra stack."""

from pyapprox.pde.assembly.protocols import (
    AssemblyStackProtocol,
    BulkMatrixSinkProtocol,
    BulkVectorSinkProtocol,
    ElementDofMapProtocol,
    MatrixSinkProtocol,
    VectorSinkProtocol,
)
from pyapprox.pde.assembly.scipy_stack import (
    ScipyAssemblyStack,
    ScipyMatrixSink,
    ScipyVectorSink,
)

__all__ = [
    "AssemblyStackProtocol",
    "BulkMatrixSinkProtocol",
    "BulkVectorSinkProtocol",
    "ElementDofMapProtocol",
    "MatrixSinkProtocol",
    "ScipyAssemblyStack",
    "ScipyMatrixSink",
    "ScipyVectorSink",
    "VectorSinkProtocol",
]
