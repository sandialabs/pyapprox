"""The component skeleton space, both backends.

Checks, on 2x2 partitions with cubic faces: the assembled mass equals
kron(scalar mass, I_k) for a continuous (k = 2) and a facet (k = 3)
space; entity and port DOFs are the scalar ones expanded node-major;
k = 1 reproduces the scalar space; refusals.
"""

import pytest

from pyapprox.pde.coupling.assembly import DenseMassAssembler, MassBlocksProtocol
from pyapprox.pde.coupling.basis import TensorLagrangeBasis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.ports import EntityDofsProtocol, subdomain_port_dofs
from pyapprox.pde.coupling.spaces import (
    ComponentSkeletonSpace,
    ContinuousSkeletonSpace,
    FacetSkeletonSpace,
    UniformBases,
)
from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.util.backends.protocols import Array, Backend


def _continuous(
    bkd: Backend[Array],
) -> tuple[TensorBoxPartition, ContinuousSkeletonSpace[Array]]:
    topo = TensorBoxPartition((2, 2))
    bases = UniformBases(TensorLagrangeBasis(1, 3, bkd))
    return topo, ContinuousSkeletonSpace(
        topo, bases, VertexWeightNumbering(topo, bases)
    )


def test_continuous_mass_is_kron(bkd: Backend[Array]) -> None:
    _, scalar = _continuous(bkd)
    space = ComponentSkeletonSpace(scalar, 2)
    assert isinstance(space, MassBlocksProtocol)
    assembler = DenseMassAssembler(bkd)
    expected = bkd.kron(assembler.assemble(scalar).format().matrix(), bkd.eye(2))
    mass = assembler.assemble(space).format().matrix()
    assert space.descriptor().size == 2 * scalar.descriptor().size
    bkd.assert_allclose(mass, expected, rtol=1e-14)


def test_facet_mass_is_kron(bkd: Backend[Array]) -> None:
    topo = TensorBoxPartition((2, 2))
    scalar = FacetSkeletonSpace(topo, UniformBases(TensorLagrangeBasis(1, 3, bkd)))
    space = ComponentSkeletonSpace(scalar, 3)
    assembler = DenseMassAssembler(bkd)
    expected = bkd.kron(assembler.assemble(scalar).format().matrix(), bkd.eye(3))
    bkd.assert_allclose(
        assembler.assemble(space).format().matrix(), expected, rtol=1e-14
    )


def test_dofs_are_node_major(bkd: Backend[Array]) -> None:
    topo, scalar = _continuous(bkd)
    space = ComponentSkeletonSpace(scalar, 2)
    assert isinstance(space, EntityDofsProtocol)
    assert space.ncomponents() == 2
    for entity in range(topo.nentities()):
        assert space.entity_dofs(entity) == tuple(
            d for s in scalar.entity_dofs(entity) for d in space.node_dofs(s)
        )
    assert space.node_dofs(3) == (6, 7)
    for subdomain in range(4):
        scalar_ports = subdomain_port_dofs(topo, scalar, subdomain)
        assert subdomain_port_dofs(topo, space, subdomain) == tuple(
            2 * d + c for d in scalar_ports for c in range(2)
        )


def test_one_component_is_scalar(bkd: Backend[Array]) -> None:
    topo, scalar = _continuous(bkd)
    space = ComponentSkeletonSpace(scalar, 1)
    assert space.descriptor() == scalar.descriptor()
    for entity in range(topo.nentities()):
        assert space.entity_dofs(entity) == scalar.entity_dofs(entity)
    assembler = DenseMassAssembler(bkd)
    bkd.assert_allclose(
        assembler.assemble(space).format().matrix(),
        assembler.assemble(scalar).format().matrix(),
        rtol=1e-14,
    )


def test_rejects_bad_input(bkd: Backend[Array]) -> None:
    _, scalar = _continuous(bkd)
    with pytest.raises(ValueError, match="at least 1"):
        ComponentSkeletonSpace(scalar, 0)
    with pytest.raises(TypeError, match="ScalarSpaceProtocol"):
        ComponentSkeletonSpace(object(), 2)  # type: ignore[arg-type]
    space = ComponentSkeletonSpace(scalar, 2)
    nscalar = scalar.descriptor().size
    with pytest.raises(IndexError, match="out of range"):
        space.node_dofs(nscalar)
    with pytest.raises(IndexError, match="out of range"):
        space.node_dofs(-1)
