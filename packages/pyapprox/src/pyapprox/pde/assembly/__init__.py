"""Assembly of element contributions into a linear-algebra stack."""

from pyapprox.pde.assembly.protocols import (
    AssemblyStackProtocol,
    BulkCapableAssemblyStackProtocol,
    BulkCapableMatrixSinkProtocol,
    BulkMatrixSinkProtocol,
    BulkVectorSinkProtocol,
    ElementDofMapProtocol,
    JacobianAssemblerProtocol,
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
    "BulkCapableAssemblyStackProtocol",
    "BulkCapableMatrixSinkProtocol",
    "BulkMatrixSinkProtocol",
    "BulkVectorSinkProtocol",
    "ElementDofMapProtocol",
    "JacobianAssemblerProtocol",
    "MatrixSinkProtocol",
    "ScipyAssemblyStack",
    "ScipyMatrixSink",
    "ScipyVectorSink",
    "VectorSinkProtocol",
]
