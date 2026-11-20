"""Tests for the dense backend matrix sink.

The element data is a 1D mesh of 6 nodes and 5 two-node elements, with
a rectangular (6 x 4) case whose columns repeat; references are explicit
dense sums. Element values keep their torch graph through assembly.
"""

import numpy as np
import pytest
import torch

from pyapprox.pde.assembly.backend_dense_sink import (
    BackendDenseMatrixAssembly,
    BackendDenseMatrixSink,
)
from pyapprox.pde.assembly.protocols import MatrixSinkProtocol
from pyapprox.pde.coupling.ports import PortAssemblyProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.linalg.backend_dense import backend_dtype_name
from pyapprox.util.linalg.operators import OperatorProperties, SpaceDescriptor

_ROWS = [[e, e + 1] for e in range(5)]
# Column DOFs of a 4-wide space, repeated across elements.
_COLS = [[e % 4, (e + 2) % 4, (e + 3) % 4] for e in range(5)]


def _space(bkd: Backend[Array], size: int) -> SpaceDescriptor:
    return SpaceDescriptor("u", size, backend_dtype_name(bkd))


def _dofs(bkd: Backend[Array], dofs: list[list[int]]) -> Array:
    return bkd.asarray(dofs, dtype=bkd.int64_dtype())


def _dense_sum(
    rows: list[list[int]],
    cols: list[list[int]],
    values: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    total = np.zeros(shape)
    for r, c, v in zip(rows, cols, values):
        total[np.ix_(r, c)] += v
    return total


def _sink(bkd: Backend[Array]) -> BackendDenseMatrixSink[Array]:
    return BackendDenseMatrixAssembly(bkd).matrix_sink(
        _space(bkd, 6), _space(bkd, 4), _dofs(bkd, _ROWS), _dofs(bkd, _COLS)
    )


def test_satisfies_protocols(bkd: Backend[Array]) -> None:
    assert isinstance(BackendDenseMatrixAssembly(bkd), PortAssemblyProtocol)
    sink = _sink(bkd)
    assert isinstance(sink, MatrixSinkProtocol)
    assert sink.bulk() is None


def test_weighted_adds_match_the_dense_sum(bkd: Backend[Array]) -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(5, 2, 3))
    weights = rng.uniform(0.5, 2.0, size=5)
    sink = _sink(bkd)
    sink.add_element_matrices(
        _dofs(bkd, _ROWS), _dofs(bkd, _COLS), bkd.asarray(values), bkd.asarray(weights)
    )
    sink.add_element_matrices(_dofs(bkd, _ROWS), _dofs(bkd, _COLS), bkd.asarray(values))
    op = sink.assemble(OperatorProperties())
    expected = _dense_sum(_ROWS, _COLS, values * (1.0 + weights)[:, None, None], (6, 4))
    bkd.assert_allclose(op.format().matrix(), bkd.asarray(expected), rtol=1e-14)
    assert op.range().size == 6 and op.domain().size == 4


def test_zero_resets_and_assemble_is_independent(bkd: Backend[Array]) -> None:
    values = bkd.asarray(np.ones((5, 2, 3)))
    sink = _sink(bkd)
    sink.add_element_matrices(_dofs(bkd, _ROWS), _dofs(bkd, _COLS), values)
    first = sink.assemble().format().matrix()
    sink.add_element_matrices(_dofs(bkd, _ROWS), _dofs(bkd, _COLS), values)
    bkd.assert_allclose(sink.assemble().format().matrix(), 2.0 * first)
    sink.zero()
    bkd.assert_allclose(sink.assemble().format().matrix(), bkd.zeros((6, 4)))
    assert bkd.to_float(bkd.sum(first)) == 30.0


def test_refusals(bkd: Backend[Array]) -> None:
    assembly = BackendDenseMatrixAssembly(bkd)
    rows, cols = _dofs(bkd, _ROWS), _dofs(bkd, _COLS)
    u6, u4 = _space(bkd, 6), _space(bkd, 4)
    other = "float32" if backend_dtype_name(bkd) == "float64" else "float64"
    with pytest.raises(ValueError, match="dtype"):
        assembly.matrix_sink(SpaceDescriptor("u", 6, other), u4, rows, cols)
    with pytest.raises(ValueError, match="lie in"):
        assembly.matrix_sink(u4, u4, rows, cols)
    with pytest.raises(ValueError, match="shape"):
        assembly.matrix_sink(u6, u4, bkd.reshape(rows, (-1,)), cols)
    with pytest.raises(ValueError, match="same elements"):
        assembly.matrix_sink(u6, u4, rows[:3], cols)
    sink = assembly.matrix_sink(u6, u4, rows, cols)
    with pytest.raises(ValueError, match="values must have shape"):
        sink.add_element_matrices(rows, cols, bkd.zeros((5, 2, 2)))
    with pytest.raises(ValueError, match="weights must have shape"):
        sink.add_element_matrices(rows, cols, bkd.zeros((5, 2, 3)), bkd.ones((4,)))
    with pytest.raises(ValueError, match="same elements"):
        sink.add_element_matrices(rows, cols[:2], bkd.zeros((5, 2, 3)))


def test_assembly_keeps_the_graph_through_the_values(torch_bkd: TorchBkd) -> None:
    bkd = torch_bkd
    rng = np.random.default_rng(1)
    values = torch.tensor(rng.normal(size=(5, 2, 3)), requires_grad=True)
    probe = torch.tensor(rng.normal(size=(6, 4)))
    sink = _sink(bkd)
    sink.add_element_matrices(_dofs(bkd, _ROWS), _dofs(bkd, _COLS), values)
    matrix = sink.assemble().format().matrix()
    (grad,) = torch.autograd.grad((matrix * probe).sum(), [values])
    # d/dvalues[e, a, b] of sum(probe * A) = probe[rows[e, a], cols[e, b]].
    expected = probe[torch.tensor(_ROWS)[:, :, None], torch.tensor(_COLS)[:, None, :]]
    bkd.assert_allclose(grad, expected, rtol=1e-14)
