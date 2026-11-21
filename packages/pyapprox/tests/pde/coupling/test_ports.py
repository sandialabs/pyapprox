"""Port transfers through port blocks, on dense NumPy and Torch and the
opaque mock stack.

Checks: the dot identity <E lam, r> = <lam, E^T r> for each subdomain,
and sum_i E_i^T E_i = diag(multiplicity): each continuous skeleton DOF
is read by exactly the subdomains incident to the entity it lies on.
"""

from dataclasses import dataclass
from typing import Any, Callable, Generic, Sequence, Union

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.pde.coupling.basis import lobatto_lagrange_basis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.ports import (
    BackendPortRows,
    EntityDofsProtocol,
    MatchingNodalTransfer,
    PortRowsProtocol,
    PortTransferProtocol,
    SubdomainEntitiesProtocol,
    subdomain_port_dofs,
)
from pyapprox.pde.coupling.spaces import (
    ContinuousSkeletonSpace,
    FacetSkeletonSpace,
    UniformBases,
)
from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, ArrayProtocol, Backend
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox.util.linalg.operators import SpaceDescriptor
from tests._helpers.coupling import MockPortRows
from tests._helpers.linalg.mock_stack import MockFormat, MockVector
from tests._helpers.linalg.stack_case import A, F, TypedStackCase, V
from tests._helpers.linalg.stacks import dense_case, mock_case


@dataclass
class _Stack(Generic[V, F, A]):
    """A stack case plus its port rows, so one test body runs on each.
    Port blocks are the case's backend arrays ``A``."""

    case: TypedStackCase[V, F, A]
    rows: Callable[[SpaceDescriptor, Sequence[int]], PortRowsProtocol[V, A]]


DenseStack = _Stack[ArrayProtocol, DenseFormat[ArrayProtocol], ArrayProtocol]
"""The dense stack over NumPy or Torch, named without importing Torch."""


def _dense_stack(bkd: Backend[Array]) -> _Stack[Array, DenseFormat[Array], Array]:
    def rows(
        descriptor: SpaceDescriptor, dofs: Sequence[int]
    ) -> BackendPortRows[Array]:
        return BackendPortRows(descriptor, dofs, bkd)

    return _Stack(dense_case(bkd), rows)


_AnyStack = Union[_Stack[MockVector, MockFormat, NDArray[Any]], DenseStack]


@pytest.fixture(params=["dense-numpy", "dense-torch", "mock"])
def stack(request: pytest.FixtureRequest) -> _AnyStack:
    if request.param == "mock":
        return _Stack(mock_case(), MockPortRows)
    fixture = "numpy_bkd" if request.param == "dense-numpy" else "torch_bkd"
    return _dense_stack(request.getfixturevalue(fixture))


def _skeleton(
    ncells: tuple[int, ...], npoints: int
) -> tuple[TensorBoxPartition, ContinuousSkeletonSpace[NDArray[Any]]]:
    topo = TensorBoxPartition(ncells)
    bases = UniformBases(lobatto_lagrange_basis(topo.ndim() - 1, npoints, NumpyBkd()))
    return topo, ContinuousSkeletonSpace(
        topo, bases, VertexWeightNumbering(topo, bases)
    )


@pytest.mark.parametrize("ncells,npoints", [((3,), 2), ((2, 2), 3), ((2, 2, 2), 2)])
def test_dot_identity(
    stack: _Stack[V, F, A], ncells: tuple[int, ...], npoints: int
) -> None:
    case = stack.case
    topo, space = _skeleton(ncells, npoints)
    descriptor = space.descriptor()
    vectors = case.space(descriptor)
    for s in range(topo.nsubdomains()):
        dofs = subdomain_port_dofs(topo, space, s)
        transfer = MatchingNodalTransfer(stack.rows(descriptor, dofs))
        assert isinstance(transfer, PortTransferProtocol)
        assert transfer.nport() == len(dofs)
        lam = case.vec(descriptor, 2, seed=s)
        r = case.bkd.asarray(np.random.default_rng(100 + s).normal(size=(len(dofs), 2)))
        lhs = case.bkd.sum(transfer.scatter(lam) * r)
        back = transfer.gather_add(r, vectors.zeros(2))
        rhs = case.bkd.sum(case.array(lam) * case.array(back))
        case.bkd.assert_allclose(
            case.bkd.reshape(lhs, (1,)), case.bkd.reshape(rhs, (1,)), rtol=1e-12
        )


@pytest.mark.parametrize("ncells", [(3,), (2, 2), (3, 2), (2, 2, 2)])
def test_round_trip_counts_multiplicity(
    stack: _Stack[V, F, A], ncells: tuple[int, ...]
) -> None:
    case = stack.case
    topo, space = _skeleton(ncells, 3)
    descriptor = space.descriptor()
    multiplicity = np.zeros((descriptor.size, 1))
    for e in range(topo.nentities()):
        for dof in space.entity_dofs(e):
            multiplicity[dof] = len(topo.incident_subdomains(e))
    lam = case.vec(descriptor, 1, seed=7)
    total = case.space(descriptor).zeros(1)
    for s in range(topo.nsubdomains()):
        rows = stack.rows(descriptor, subdomain_port_dofs(topo, space, s))
        assert isinstance(rows, PortRowsProtocol)
        transfer = MatchingNodalTransfer(rows)
        total = transfer.gather_add(transfer.scatter(lam), total)
    case.assert_vector(total, case.dense(multiplicity) * case.array(lam), rtol=1e-12)


def test_backend_port_rows_reject_bad_input(numpy_bkd: Backend[Array]) -> None:
    descriptor = SpaceDescriptor("skeleton", 4)
    with pytest.raises(ValueError):
        BackendPortRows(descriptor, [0, 0], numpy_bkd)
    with pytest.raises(ValueError):
        BackendPortRows(descriptor, [4], numpy_bkd)
    rows = BackendPortRows(descriptor, [1, 3], numpy_bkd)
    with pytest.raises(ValueError):
        rows.take(numpy_bkd.zeros((3, 1)))
    with pytest.raises(ValueError):
        rows.add_to(numpy_bkd.zeros((4, 1)), numpy_bkd.zeros((3, 1)))
    with pytest.raises(TypeError):
        MatchingNodalTransfer(object())  # type: ignore[arg-type]


def test_backend_port_rows_keep_the_autodiff_graph(
    torch_bkd: AutodiffBackend[Array],
) -> None:
    bkd = torch_bkd
    rows = BackendPortRows(SpaceDescriptor("skeleton", 5), [3, 0], bkd)

    def total(params: Array) -> Array:
        x, block = bkd.reshape(params[:5], (5, 1)), bkd.reshape(params[5:], (2, 1))
        return bkd.reshape(bkd.sum(rows.add_to(x, block * rows.take(x))), (1,))

    # out = x + P^T (block * P x): d sum(out)/dx = 1 + P^T block,
    # d sum(out)/d block = P x.
    params = bkd.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 2.0, -1.0])
    bkd.assert_allclose(
        bkd.jacobian(total, params),
        bkd.asarray([[0.0, 1.0, 1.0, 3.0, 1.0, 3.0, 0.0]]),
    )


def test_both_spaces_give_entity_dofs() -> None:
    topo = TensorBoxPartition((2, 2))
    bases = UniformBases(lobatto_lagrange_basis(1, 3, NumpyBkd()))
    assert isinstance(topo, SubdomainEntitiesProtocol)
    for space in (
        ContinuousSkeletonSpace(topo, bases, VertexWeightNumbering(topo, bases)),
        FacetSkeletonSpace(topo, bases),
    ):
        assert isinstance(space, EntityDofsProtocol)
        dofs = subdomain_port_dofs(topo, space, 0)
        assert list(dofs) == sorted(set(dofs))


class _TwoTriangleSkeleton:
    """A hand-numbered interface, not a box partition: three subdomains
    meeting along two edges (entities 0 and 1) at a vertex (entity 2)
    with one DOF, each edge carrying two interior DOFs."""

    def subdomain_entities(self, subdomain: int) -> tuple[int, ...]:
        return ((0, 2), (0, 1, 2), (1, 2))[subdomain]

    def entity_dofs(self, entity: int) -> tuple[int, ...]:
        return ((1, 2), (3, 4), (0,))[entity]


def test_port_dofs_need_no_box_partition() -> None:
    skeleton = _TwoTriangleSkeleton()
    assert subdomain_port_dofs(skeleton, skeleton, 0) == (0, 1, 2)
    assert subdomain_port_dofs(skeleton, skeleton, 1) == (0, 1, 2, 3, 4)
    assert subdomain_port_dofs(skeleton, skeleton, 2) == (0, 3, 4)


def test_port_dofs_reject_bad_input() -> None:
    topo, space = _skeleton((2,), 2)
    with pytest.raises(TypeError, match="SubdomainEntitiesProtocol"):
        subdomain_port_dofs(object(), space, 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="EntityDofsProtocol"):
        subdomain_port_dofs(topo, object(), 0)  # type: ignore[arg-type]
