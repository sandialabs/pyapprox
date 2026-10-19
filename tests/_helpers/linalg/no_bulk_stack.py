"""The SciPy assembly stack without its bulk capability.

A distributed stack has no whole-matrix add, so its sinks' ``bulk()`` is
``None`` and an assembler must fall back to element blocks. This stack
is the single-process SciPy stack with only that changed, so the
fallback is tested without a distributed stack.
"""

from typing import Generic, Optional

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.assembly.protocols import (
    BulkMatrixSinkProtocol,
    BulkVectorSinkProtocol,
)
from pyapprox.pde.assembly.scipy_stack import (
    ScipyAssemblyStack,
    ScipyMatrixSink,
    ScipyVectorSink,
)
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    OperatorProperties,
    SpaceDescriptor,
)
from pyapprox.util.linalg.scipy_stack import (
    CSRFormat,
    NumpyRowSet,
    NumpyVectorSpace,
    S,
    ScipyCSROperator,
)

_Dofs = NDArray[np.int64]


class NoBulkMatrixSink(Generic[S]):
    """A SciPy matrix sink that accepts element blocks only."""

    def __init__(self, sink: ScipyMatrixSink[S]) -> None:
        self._sink: ScipyMatrixSink[S] = sink

    def add_element_matrices(
        self,
        row_dofs: _Dofs,
        col_dofs: _Dofs,
        values: NDArray[S],
        weights: Optional[NDArray[S]] = None,
    ) -> None:
        self._sink.add_element_matrices(row_dofs, col_dofs, values, weights)

    def bulk(self) -> Optional[BulkMatrixSinkProtocol]:
        return None

    def zero(self) -> None:
        self._sink.zero()

    def assemble(
        self, properties: Optional[OperatorProperties] = None
    ) -> ScipyCSROperator[S]:
        return self._sink.assemble(properties)


class NoBulkVectorSink(Generic[S]):
    """A SciPy vector sink that accepts element vectors only."""

    def __init__(self, sink: ScipyVectorSink[S]) -> None:
        self._sink: ScipyVectorSink[S] = sink

    def add_element_vectors(
        self,
        row_dofs: _Dofs,
        values: NDArray[S],
        weights: Optional[NDArray[S]] = None,
    ) -> None:
        self._sink.add_element_vectors(row_dofs, values, weights)

    def bulk(self) -> Optional[BulkVectorSinkProtocol[NDArray[S]]]:
        return None

    def zero(self) -> None:
        self._sink.zero()

    def assemble(self) -> NDArray[S]:
        return self._sink.assemble()


class NoBulkAssemblyStack(Generic[S]):
    """The SciPy assembly stack whose sinks have no bulk capability."""

    def __init__(self, scalar: type[S]) -> None:
        self._stack: ScipyAssemblyStack[S] = ScipyAssemblyStack(scalar)

    def space(self, descriptor: SpaceDescriptor) -> NumpyVectorSpace[S]:
        return self._stack.space(descriptor)

    def matrix_sink(
        self,
        range: SpaceDescriptor,
        domain: SpaceDescriptor,
        row_dofs: _Dofs,
        col_dofs: _Dofs,
    ) -> NoBulkMatrixSink[S]:
        return NoBulkMatrixSink(
            self._stack.matrix_sink(range, domain, row_dofs, col_dofs)
        )

    def vector_sink(self, descriptor: SpaceDescriptor) -> NoBulkVectorSink[S]:
        return NoBulkVectorSink(self._stack.vector_sink(descriptor))

    def row_set(self, descriptor: SpaceDescriptor, rows: _Dofs) -> NumpyRowSet:
        return self._stack.row_set(descriptor, rows)

    def prolongation(self, assembly: SpaceDescriptor) -> ScipyCSROperator[S]:
        return self._stack.prolongation(assembly)

    def solve_operator(
        self, op: ActionOperatorProtocol[NDArray[S], CSRFormat]
    ) -> ActionOperatorProtocol[NDArray[S], CSRFormat]:
        return self._stack.solve_operator(op)
