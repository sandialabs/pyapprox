"""The mock assembly stack: element and bulk assembly into its own format.

The same 1D mesh as the SciPy stack's tests (6 nodes, 5 two-node
elements); bulk input arrives as a SciPy sparse matrix and is converted
into the mock's triplets.
"""

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from pyapprox.pde.assembly import (
    AssemblyStackProtocol,
    MatrixSinkProtocol,
    VectorSinkProtocol,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import RowReplacedOperator, SpaceDescriptor
from tests._helpers.linalg.mock_stack import MockAssemblyStack, MockLU, MockRowSet, peek

_BKD = NumpyBkd()
_U = SpaceDescriptor("u", 6)
_DOFS = np.array([[e, e + 1] for e in range(5)])


def _blocks(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(5, 2, 2))


def _dense_sum(values: np.ndarray) -> np.ndarray:
    total = np.zeros((6, 6))
    for dofs, block in zip(_DOFS, values):
        total[np.ix_(dofs, dofs)] += block
    return total


class TestMockAssembly:
    def test_satisfies_protocols(self) -> None:
        stack = MockAssemblyStack()
        assert isinstance(stack, AssemblyStackProtocol)
        assert isinstance(stack.matrix_sink(_U, _U, _DOFS, _DOFS), MatrixSinkProtocol)
        assert isinstance(stack.vector_sink(_U), VectorSinkProtocol)

    def test_element_and_bulk_assembly_agree(self) -> None:
        values = _blocks(0)
        stack = MockAssemblyStack()
        by_element = stack.matrix_sink(_U, _U, _DOFS, _DOFS)
        by_element.add_element_matrices(_DOFS, _DOFS, values)
        rows = np.repeat(_DOFS[:, :, None], 2, axis=2).ravel()
        cols = np.repeat(_DOFS[:, None, :], 2, axis=1).ravel()
        by_bulk = stack.matrix_sink(_U, _U, _DOFS, _DOFS)
        by_bulk.bulk().add_assembled(
            coo_matrix((values.ravel(), (rows, cols)), shape=(6, 6)).tocsr()
        )
        expected = _dense_sum(values)
        _BKD.assert_allclose(
            by_element.assemble().format().dense(), expected, rtol=1e-14
        )
        _BKD.assert_allclose(by_bulk.assemble().format().dense(), expected, rtol=1e-14)

    def test_pattern_refuses_outside_entries(self) -> None:
        sink = MockAssemblyStack().matrix_sink(_U, _U, _DOFS, _DOFS)
        with pytest.raises(ValueError, match="outside the sink's sparsity pattern"):
            sink.add_element_matrices(
                np.array([[0]]), np.array([[5]]), np.ones((1, 1, 1))
            )

    def test_assembled_system_is_constrained_and_solved(self) -> None:
        """Assemble, replace the end rows, factor and solve, never looking
        inside a vector except to check the answer."""
        stiffness = np.array([[1.0, -1.0], [-1.0, 1.0]])
        stack = MockAssemblyStack()
        sink = stack.matrix_sink(_U, _U, _DOFS, _DOFS)
        sink.add_element_matrices(_DOFS, _DOFS, np.repeat(stiffness[None], 5, axis=0))
        load = stack.vector_sink(_U)
        load.add_element_vectors(_DOFS, np.full((5, 2), 0.5))
        op = RowReplacedOperator(
            stack.solve_operator(sink.assemble()), MockRowSet(_U, [0, 5])
        )
        b = MockRowSet(_U, [0, 5]).zero_rows(load.assemble())
        x, info = MockLU().setup(op).solve(b)
        assert info.converged
        reference = _dense_sum(np.repeat(stiffness[None], 5, axis=0))
        reference[[0, 5], :] = 0.0
        reference[[0, 5], [0, 5]] = 1.0
        _BKD.assert_allclose(reference @ peek(x), peek(b), rtol=1e-10)
