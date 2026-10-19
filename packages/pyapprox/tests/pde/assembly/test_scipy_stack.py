"""Tests for the single-process SciPy assembly stack.

The element data is a 1D mesh of 6 nodes and 5 two-node elements with
random element blocks; references are explicit dense sums.
"""

from typing import Union

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from pyapprox.pde.assembly import (
    AssemblyStackProtocol,
    MatrixSinkProtocol,
    ScipyAssemblyStack,
    VectorSinkProtocol,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import (
    OperatorProperties,
    RowSetProtocol,
    SpaceDescriptor,
)
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from tests._helpers.linalg.no_bulk_stack import NoBulkAssemblyStack

_BKD = NumpyBkd()
_U = SpaceDescriptor("u", 6)
_DOFS = np.array([[e, e + 1] for e in range(5)])


def _blocks(seed: int, nrow: int = 2, ncol: int = 2) -> NDArray[np.float64]:
    return np.random.default_rng(seed).normal(size=(5, nrow, ncol))


def _dense_sum(
    rows: NDArray[np.int64],
    cols: NDArray[np.int64],
    values: NDArray[np.float64],
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    total = np.zeros(shape)
    for r, c, v in zip(rows, cols, values):
        total[np.ix_(r, c)] += v
    return total


def _spd_blocks() -> NDArray[np.float64]:
    """1D P1 stiffness plus mass on unit elements: SPD once assembled."""
    stiffness = np.array([[1.0, -1.0], [-1.0, 1.0]])
    mass = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0
    return np.repeat((stiffness + mass)[None], 5, axis=0)


class TestMatrixSink:
    def test_satisfies_protocols(self) -> None:
        stack = ScipyAssemblyStack(np.float64)
        assert isinstance(stack, AssemblyStackProtocol)
        assert isinstance(stack.matrix_sink(_U, _U, _DOFS, _DOFS), MatrixSinkProtocol)
        assert isinstance(stack.vector_sink(_U), VectorSinkProtocol)

    def test_element_adds_match_the_dense_sum(self) -> None:
        values = _blocks(0)
        sink = ScipyAssemblyStack(np.float64).matrix_sink(_U, _U, _DOFS, _DOFS)
        sink.add_element_matrices(_DOFS, _DOFS, values)
        op = sink.assemble()
        _BKD.assert_allclose(
            op.format().matrix().toarray(),
            _dense_sum(_DOFS, _DOFS, values, (6, 6)),
            rtol=1e-14,
        )
        assert op.domain() == _U and op.range() == _U

    def test_bulk_and_element_adds_give_the_same_operator(self) -> None:
        values = _blocks(1)
        stack = ScipyAssemblyStack(np.float64)
        by_element = stack.matrix_sink(_U, _U, _DOFS, _DOFS)
        by_element.add_element_matrices(_DOFS, _DOFS, values)
        rows = np.repeat(_DOFS[:, :, None], 2, axis=2).ravel()
        cols = np.repeat(_DOFS[:, None, :], 2, axis=1).ravel()
        whole = coo_matrix((values.ravel(), (rows, cols)), shape=(6, 6)).tocsr()
        by_bulk = stack.matrix_sink(_U, _U, _DOFS, _DOFS)
        bulk = by_bulk.bulk()
        assert bulk is not None
        bulk.add_assembled(whole)
        _BKD.assert_allclose(
            by_bulk.assemble().format().matrix().toarray(),
            by_element.assemble().format().matrix().toarray(),
            rtol=1e-14,
        )

    def test_weights_scale_each_element(self) -> None:
        values = _blocks(2)
        weights = np.array([1.0, 0.0, 2.0, 0.5, 3.0])
        sink = ScipyAssemblyStack(np.float64).matrix_sink(_U, _U, _DOFS, _DOFS)
        sink.add_element_matrices(_DOFS, _DOFS, values, weights)
        _BKD.assert_allclose(
            sink.assemble().format().matrix().toarray(),
            _dense_sum(_DOFS, _DOFS, values * weights[:, None, None], (6, 6)),
            rtol=1e-14,
        )

    def test_pattern_is_fixed_at_creation(self) -> None:
        """Tridiagonal: 6 diagonal entries plus 2 x 5 off-diagonal. Adds
        never grow it, and an entry outside it is refused."""
        sink = ScipyAssemblyStack(np.float64).matrix_sink(_U, _U, _DOFS, _DOFS)
        assert sink.nnz() == 16
        sink.add_element_matrices(_DOFS, _DOFS, _blocks(3))
        assert sink.nnz() == 16
        with pytest.raises(ValueError, match="outside the sink's sparsity pattern"):
            sink.add_element_matrices(
                np.array([[0]]), np.array([[5]]), np.ones((1, 1, 1))
            )
        bulk = sink.bulk()
        assert bulk is not None
        with pytest.raises(ValueError, match="outside"):
            bulk.add_assembled(csr_matrix(([1.0], ([0], [5])), shape=(6, 6)))

    def test_zero_reuses_the_pattern_and_assembled_operators_are_independent(
        self,
    ) -> None:
        values = _blocks(4)
        sink = ScipyAssemblyStack(np.float64).matrix_sink(_U, _U, _DOFS, _DOFS)
        sink.add_element_matrices(_DOFS, _DOFS, values)
        first = sink.assemble()
        expected = first.format().matrix().toarray().copy()
        sink.add_element_matrices(_DOFS, _DOFS, values)
        _BKD.assert_allclose(first.format().matrix().toarray(), expected)
        sink.zero()
        sink.add_element_matrices(_DOFS, _DOFS, values)
        _BKD.assert_allclose(sink.assemble().format().matrix().toarray(), expected)

    def test_rectangular_sink(self) -> None:
        """Three one-row cells, each reading a stencil of two states."""
        cells = SpaceDescriptor("cells", 3)
        rows = np.array([[0], [1], [2]])
        cols = np.array([[0, 1], [2, 3], [4, 5]])
        values = np.random.default_rng(5).normal(size=(3, 1, 2))
        sink = ScipyAssemblyStack(np.float64).matrix_sink(cells, _U, rows, cols)
        sink.add_element_matrices(rows, cols, values)
        op = sink.assemble()
        assert op.range() == cells and op.domain() == _U
        _BKD.assert_allclose(
            op.format().matrix().toarray(),
            _dense_sum(rows, cols, values, (3, 6)),
            rtol=1e-14,
        )

    def test_assemble_declares_given_properties(self) -> None:
        sink = ScipyAssemblyStack(np.float64).matrix_sink(_U, _U, _DOFS, _DOFS)
        sink.add_element_matrices(_DOFS, _DOFS, _spd_blocks())
        spd = OperatorProperties(symmetric=True, spd=True)
        assert sink.assemble(spd).properties() == spd

    def test_rejects_bad_inputs(self) -> None:
        stack = ScipyAssemblyStack(np.float64)
        with pytest.raises(ValueError, match=r"\[0, 6\)"):
            stack.matrix_sink(_U, _U, _DOFS + 1, _DOFS)
        with pytest.raises(ValueError, match="same elements"):
            stack.matrix_sink(_U, _U, _DOFS, _DOFS[:3])
        sink = stack.matrix_sink(_U, _U, _DOFS, _DOFS)
        with pytest.raises(ValueError, match="values must have shape"):
            sink.add_element_matrices(_DOFS, _DOFS, np.ones((5, 2, 3)))
        with pytest.raises(ValueError, match="weights must have shape"):
            sink.add_element_matrices(_DOFS, _DOFS, _blocks(6), np.ones(4))
        bulk = sink.bulk()
        assert bulk is not None
        with pytest.raises(ValueError, match="shape"):
            bulk.add_assembled(csr_matrix((5, 6)))


class TestVectorSink:
    def test_element_and_bulk_adds_match(self) -> None:
        values = np.random.default_rng(7).normal(size=(5, 2))
        weights = np.array([1.0, 2.0, 0.0, 1.0, 0.5])
        stack = ScipyAssemblyStack(np.float64)
        sink = stack.vector_sink(_U)
        sink.add_element_vectors(_DOFS, values, weights)
        expected = np.zeros(6)
        for dofs, value, weight in zip(_DOFS, values, weights):
            expected[dofs] += weight * value
        result = sink.assemble()
        assert result.shape == (6, 1)
        _BKD.assert_allclose(result[:, 0], expected, rtol=1e-14)
        bulk_sink = stack.vector_sink(_U)
        bulk = bulk_sink.bulk()
        assert bulk is not None
        bulk.add_assembled(expected)
        _BKD.assert_allclose(bulk_sink.assemble(), result, rtol=1e-14)
        sink.zero()
        _BKD.assert_allclose(sink.assemble(), np.zeros((6, 1)))

    def test_rejects_bad_shapes(self) -> None:
        sink = ScipyAssemblyStack(np.float64).vector_sink(_U)
        with pytest.raises(ValueError, match="values must have shape"):
            sink.add_element_vectors(_DOFS, np.ones((5, 3)))
        bulk = sink.bulk()
        assert bulk is not None
        with pytest.raises(ValueError, match="vector must have shape"):
            bulk.add_assembled(np.ones(5))


class TestProlongation:
    def test_identity_means_one_space(self) -> None:
        """With P = I the solve space is the assembly space and the
        assembled operator is solved as it is."""
        stack = ScipyAssemblyStack(np.float64)
        p = stack.prolongation(_U)
        assert p.domain() == _U and p.range() == _U and p.properties().identity
        sink = stack.matrix_sink(_U, _U, _DOFS, _DOFS)
        sink.add_element_matrices(_DOFS, _DOFS, _spd_blocks())
        op = sink.assemble()
        assert stack.solve_operator(op) is op

    def test_nonidentity_prolongation_gives_ptap(self) -> None:
        """Node 5 constrained to the average of nodes 3 and 4 (as a hanging
        node would be): 6 assembly DOFs, 5 solve DOFs."""
        solve = SpaceDescriptor("u-true", 5)
        p = np.zeros((6, 5))
        p[:5, :5] = np.eye(5)
        p[5, 3] = p[5, 4] = 0.5
        stack = ScipyAssemblyStack(np.float64, {_U: (solve, csr_matrix(p))})
        assert stack.prolongation(_U).domain() == solve
        sink = stack.matrix_sink(_U, _U, _DOFS, _DOFS)
        sink.add_element_matrices(_DOFS, _DOFS, _spd_blocks())
        assembled = sink.assemble(OperatorProperties(symmetric=True))
        solved = stack.solve_operator(assembled)
        assert solved.domain() == solve and solved.range() == solve
        assert solved.properties().symmetric
        a = assembled.format().matrix().toarray()
        _BKD.assert_allclose(
            solved.format().matrix().toarray(), p.T @ a @ p, rtol=1e-14
        )
        b = np.random.default_rng(8).normal(size=(5, 1))
        x, info = ScipySparseLU().setup(solved).solve(b)
        assert info.converged
        _BKD.assert_allclose(p.T @ a @ p @ x, b, rtol=1e-10)

    def test_rejects_a_misshapen_prolongation(self) -> None:
        with pytest.raises(ValueError, match="prolongation"):
            ScipyAssemblyStack(
                np.float64, {_U: (SpaceDescriptor("t", 5), csr_matrix((5, 5)))}
            )


_Stack = Union[ScipyAssemblyStack[np.float64], NoBulkAssemblyStack[np.float64]]


@pytest.mark.parametrize(
    "stack",
    [ScipyAssemblyStack(np.float64), NoBulkAssemblyStack(np.float64)],
    ids=["scipy", "no-bulk"],
)
class TestRowSet:
    def test_selects_the_given_rows(self, stack: _Stack) -> None:
        rows = stack.row_set(_U, np.array([4, 1, 4], dtype=np.int64))
        assert isinstance(rows, RowSetProtocol)
        assert rows.count() == 2 and rows.descriptor() == _U
        x = np.arange(12.0).reshape(6, 2)
        expected = x.copy()
        expected[[1, 4]] = 0.0
        _BKD.assert_allclose(rows.zero_rows(x), expected)

    def test_rejects_rows_outside_the_space(self, stack: _Stack) -> None:
        with pytest.raises(ValueError, match=r"\[0, 6\)"):
            stack.row_set(_U, np.array([6], dtype=np.int64))


class TestNoBulkStack:
    def test_sinks_have_no_bulk_capability(self) -> None:
        stack = NoBulkAssemblyStack(np.float64)
        assert isinstance(stack, AssemblyStackProtocol)
        sink = stack.matrix_sink(_U, _U, _DOFS, _DOFS)
        assert isinstance(sink, MatrixSinkProtocol)
        assert sink.bulk() is None
        assert stack.vector_sink(_U).bulk() is None
        values = _blocks(9)
        sink.add_element_matrices(_DOFS, _DOFS, values)
        _BKD.assert_allclose(
            sink.assemble().format().matrix().toarray(),
            _dense_sum(_DOFS, _DOFS, values, (6, 6)),
        )
