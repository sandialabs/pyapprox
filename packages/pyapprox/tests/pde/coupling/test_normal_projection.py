r"""The normal-projection transfer: a scalar skeleton read by a vector
port, on both backends.

Four port nodes with unit normals that vary node to node (as on a
curve), over a scalar skeleton of six DOFs read by matching rows. Checks:
the scatter is the skeleton value along each normal, s λ n; the gather
of a vector block adds s n · v at the port's skeleton DOFs; <E λ, r> =
<λ, E^T r>; with an inner projection map the same holds and the
composition is the inner map spread along n; an autodiff graph through
the skeleton values survives. Refusals: an inner transfer outside the
protocol, normals of the wrong shape or not of unit length.
"""

import math

import pytest

from pyapprox.pde.coupling.ports import (
    BackendPortRows,
    MatchingNodalTransfer,
    NormalProjectionTransfer,
    PortMatrixTransfer,
)
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseOperator
from pyapprox.util.linalg.operators import SpaceDescriptor

_SKELETON = SpaceDescriptor("skeleton", 6)
_DOFS = [4, 1, 2, 5]
_ANGLES = [0.0, 0.3, 1.1, 2.0]


def _normals(bkd: Backend[Array]) -> Array:
    return bkd.asarray([[math.cos(a), math.sin(a)] for a in _ANGLES])


def _transfer(
    bkd: Backend[Array], sign: float = 1.0
) -> NormalProjectionTransfer[Array, Array]:
    inner = MatchingNodalTransfer(BackendPortRows(_SKELETON, _DOFS, bkd))
    return NormalProjectionTransfer(inner, _normals(bkd), bkd, sign)


def _lam(bkd: Backend[Array]) -> Array:
    return bkd.asarray([[0.5], [-1.0], [2.0], [0.0], [3.0], [-0.25]])


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_scatter_spreads_along_the_normals(bkd: Backend[Array], sign: float) -> None:
    transfer = _transfer(bkd, sign)
    assert transfer.nport() == 8
    lam = _lam(bkd)
    normals = _normals(bkd)
    expected = bkd.reshape(sign * lam[_DOFS] * normals, (-1, 1))
    bkd.assert_allclose(transfer.scatter(lam), expected, rtol=1e-14)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_gather_reads_the_normal_component(bkd: Backend[Array], sign: float) -> None:
    transfer = _transfer(bkd, sign)
    v = bkd.asarray([[1.0, 2.0], [-0.5, 0.25], [3.0, -1.0], [0.0, 4.0]])
    gathered = transfer.gather_add(bkd.reshape(v, (-1, 1)), bkd.zeros((6, 1)))
    normal = bkd.sum(v * _normals(bkd), axis=1)
    expected = bkd.zeros((6, 1))
    for a, dof in enumerate(_DOFS):
        expected = bkd.index_add(
            expected,
            bkd.asarray([dof], dtype=bkd.int64_dtype()),
            bkd.reshape(sign * normal[a], (1, 1)),
        )
    bkd.assert_allclose(gathered, expected, rtol=1e-14, atol=1e-15)


def test_gather_is_the_adjoint_of_scatter(bkd: Backend[Array]) -> None:
    transfer = _transfer(bkd, -1.0)
    lam = bkd.hstack([_lam(bkd), 2.0 * _lam(bkd) - 1.0])
    block = bkd.reshape(bkd.linspace(-1.0, 2.0, 16), (8, 2))
    lhs = bkd.sum(transfer.scatter(lam) * block)
    rhs = bkd.sum(lam * transfer.gather_add(block, bkd.zeros((6, 2))))
    bkd.assert_allclose(bkd.reshape(lhs, (1,)), bkd.reshape(rhs, (1,)), rtol=1e-13)


def test_an_inner_map_is_spread_along_the_normals(bkd: Backend[Array]) -> None:
    # The inner transfer averages neighboring skeleton DOFs onto the nodes.
    rows = BackendPortRows(_SKELETON, [0, 1, 2, 3, 4, 5], bkd)
    dense = bkd.asarray(
        [
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
        ]
    )
    local = BackendDenseOperator(
        dense,
        SpaceDescriptor("port-rows", 6),
        bkd,
        range=SpaceDescriptor("port", 4),
    )
    transfer = NormalProjectionTransfer(
        PortMatrixTransfer(rows, local), _normals(bkd), bkd
    )
    lam = _lam(bkd)
    expected = bkd.reshape((dense @ lam) * _normals(bkd), (-1, 1))
    bkd.assert_allclose(transfer.scatter(lam), expected, rtol=1e-14)
    block = bkd.reshape(bkd.linspace(0.0, 1.0, 8), (8, 1))
    lhs = bkd.sum(transfer.scatter(lam) * block)
    rhs = bkd.sum(lam * transfer.gather_add(block, bkd.zeros((6, 1))))
    bkd.assert_allclose(bkd.reshape(lhs, (1,)), bkd.reshape(rhs, (1,)), rtol=1e-13)


def test_the_autodiff_graph_survives(torch_bkd: AutodiffBackend[Array]) -> None:
    bkd = torch_bkd
    transfer = _transfer(bkd, -1.0)
    block = bkd.reshape(bkd.linspace(-1.0, 1.0, 8), (8, 1))

    def total(lam: Array) -> Array:
        return bkd.reshape(
            bkd.sum(transfer.scatter(bkd.reshape(lam, (6, 1))) * block), (1,)
        )

    # d/d lam <E lam, block> = E^T block.
    expected = bkd.reshape(transfer.gather_add(block, bkd.zeros((6, 1))), (1, 6))
    bkd.assert_allclose(
        bkd.jacobian(total, bkd.reshape(_lam(bkd), (6,))), expected, rtol=1e-13
    )


def test_refusals(bkd: Backend[Array]) -> None:
    inner = MatchingNodalTransfer(BackendPortRows(_SKELETON, _DOFS, bkd))
    with pytest.raises(TypeError):
        NormalProjectionTransfer(object(), _normals(bkd), bkd)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        NormalProjectionTransfer(inner, _normals(bkd)[:3], bkd)
    with pytest.raises(ValueError, match="unit"):
        NormalProjectionTransfer(inner, 2.0 * _normals(bkd), bkd)
