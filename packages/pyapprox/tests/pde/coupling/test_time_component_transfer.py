r"""A transfer on a skeleton with several values per DOF (one trace per
unknown of a space in time), on both backends.

The plain transfer is the normal projection of a scalar six-DOF skeleton
onto four port nodes with varying unit normals (as in
``test_normal_projection``), three components per DOF. Checks: component
``k`` of the scatter is the plain scatter of component ``k`` (entries
``q * d + k``); likewise the gather, added to the given vector;
<E λ, r> = <λ, E^T r>; one component is the plain transfer; an autodiff
graph through the skeleton values survives. Refusals: an inner transfer
outside the protocol, fewer than one component.
"""

import math

import pytest

from pyapprox.pde.coupling.ports import (
    BackendPortRows,
    MatchingNodalTransfer,
    NormalProjectionTransfer,
    TimeComponentTransfer,
)
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import SpaceDescriptor

_SKELETON = SpaceDescriptor("skeleton", 6)
_DOFS = [4, 1, 2, 5]
_ANGLES = [0.0, 0.3, 1.1, 2.0]
_Q = 3


def _plain(bkd: Backend[Array]) -> NormalProjectionTransfer[Array, Array]:
    normals = bkd.asarray([[math.cos(a), math.sin(a)] for a in _ANGLES])
    inner = MatchingNodalTransfer(BackendPortRows(_SKELETON, _DOFS, bkd))
    return NormalProjectionTransfer(inner, normals, bkd, -1.0)


def _lam(bkd: Backend[Array]) -> Array:
    """Three components of a six-DOF skeleton, interleaved, two columns."""
    return bkd.reshape(bkd.linspace(-1.0, 2.0, 6 * _Q * 2), (6 * _Q, 2))


def test_components_are_transferred_alone(bkd: Backend[Array]) -> None:
    plain = _plain(bkd)
    transfer = TimeComponentTransfer(plain, _Q, bkd)
    assert transfer.nport() == plain.nport() * _Q
    lam = _lam(bkd)
    scattered = transfer.scatter(lam)
    block = bkd.reshape(bkd.linspace(0.5, -1.5, plain.nport() * _Q * 2), (-1, 2))
    y = bkd.reshape(bkd.linspace(1.0, 3.0, 6 * _Q * 2), (6 * _Q, 2))
    gathered = transfer.gather_add(block, y)
    for k in range(_Q):
        bkd.assert_allclose(scattered[k::_Q], plain.scatter(lam[k::_Q]), rtol=1e-14)
        expected = plain.gather_add(block[k::_Q], y[k::_Q])
        bkd.assert_allclose(gathered[k::_Q], expected, rtol=1e-14)


def test_gather_is_the_adjoint_of_scatter(bkd: Backend[Array]) -> None:
    transfer = TimeComponentTransfer(_plain(bkd), _Q, bkd)
    lam = _lam(bkd)
    block = bkd.reshape(bkd.linspace(-2.0, 1.0, transfer.nport() * 2), (-1, 2))
    lhs = bkd.sum(transfer.scatter(lam) * block)
    rhs = bkd.sum(lam * transfer.gather_add(block, bkd.zeros((6 * _Q, 2))))
    bkd.assert_allclose(bkd.reshape(lhs, (1,)), bkd.reshape(rhs, (1,)), rtol=1e-13)


def test_one_component_is_the_plain_transfer(bkd: Backend[Array]) -> None:
    plain = _plain(bkd)
    transfer = TimeComponentTransfer(plain, 1, bkd)
    lam = bkd.reshape(bkd.linspace(-1.0, 1.0, 6), (6, 1))
    bkd.assert_allclose(transfer.scatter(lam), plain.scatter(lam), rtol=1e-14)


def test_the_autodiff_graph_survives(torch_bkd: AutodiffBackend[Array]) -> None:
    bkd = torch_bkd
    transfer = TimeComponentTransfer(_plain(bkd), _Q, bkd)
    n = 6 * _Q
    block = bkd.reshape(bkd.linspace(-1.0, 1.0, transfer.nport()), (-1, 1))

    def total(lam: Array) -> Array:
        return bkd.reshape(
            bkd.sum(transfer.scatter(bkd.reshape(lam, (n, 1))) * block), (1,)
        )

    # d/d lam <E lam, block> = E^T block.
    expected = bkd.reshape(transfer.gather_add(block, bkd.zeros((n, 1))), (1, n))
    lam = bkd.reshape(_lam(bkd)[:, :1], (n,))
    bkd.assert_allclose(bkd.jacobian(total, lam), expected, rtol=1e-13)


def test_refusals(bkd: Backend[Array]) -> None:
    with pytest.raises(TypeError, match="PortTransferProtocol"):
        TimeComponentTransfer(object(), _Q, bkd)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 1"):
        TimeComponentTransfer(_plain(bkd), 0, bkd)
