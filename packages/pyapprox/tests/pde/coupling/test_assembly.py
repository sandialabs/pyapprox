"""The dense mass assembler, both backends.

Checks: overlapping blocks sum where they share DOFs, equal to
G^T blockdiag(blocks) G with G the explicit 0/1 copy to block rows; a
continuous space's assembled mass equals that product built from its own
blocks; refusals.
"""

from typing import Generic, Sequence

import pytest

from pyapprox.pde.coupling.assembly import DenseMassAssembler, MassBlocksProtocol
from pyapprox.pde.coupling.basis import lobatto_lagrange_basis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.spaces import (
    ContinuousSkeletonSpace,
    MassBlock,
    UniformBases,
)
from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import SpaceDescriptor


class _Blocks(Generic[Array]):
    def __init__(self, size: int, blocks: Sequence[MassBlock[Array]]) -> None:
        self._descriptor, self._blocks = SpaceDescriptor("skeleton", size), blocks

    def descriptor(self) -> SpaceDescriptor:
        return self._descriptor

    def mass_blocks(self) -> Sequence[MassBlock[Array]]:
        return self._blocks


def _copy_then_sum(
    bkd: Backend[Array], size: int, blocks: Sequence[MassBlock[Array]]
) -> Array:
    """G^T blockdiag(blocks) G, G stacking each block's 0/1 row copy."""
    total = bkd.zeros((size, size))
    for block in blocks:
        g = bkd.asarray(
            [[1.0 if d == col else 0.0 for col in range(size)] for d in block.dofs]
        )
        total = total + bkd.dot(g.T, bkd.dot(block.matrix, g))
    return total


def test_overlapping_blocks_add(bkd: Backend[Array]) -> None:
    blocks = [
        MassBlock((3, 0), bkd.asarray([[2.0, 1.0], [1.0, 4.0]])),
        MassBlock(
            (0, 1, 3),
            bkd.asarray([[1.0, 0.5, 0.0], [0.5, 3.0, 0.2], [0.0, 0.2, 5.0]]),
        ),
    ]
    source = _Blocks(4, blocks)
    assert isinstance(source, MassBlocksProtocol)
    mass = DenseMassAssembler(bkd).assemble(source)
    expected = _copy_then_sum(bkd, 4, blocks)
    bkd.assert_allclose(mass.format().matrix(), expected, rtol=1e-14)
    # DOF 2 is in no block: its row and column stay zero.
    bkd.assert_allclose(expected[2], bkd.zeros((4,)))


def test_continuous_space_mass(bkd: Backend[Array]) -> None:
    topo = TensorBoxPartition((2, 2))
    bases = UniformBases(lobatto_lagrange_basis(1, 3, bkd))
    space = ContinuousSkeletonSpace(topo, bases, VertexWeightNumbering(topo, bases))
    size = space.descriptor().size
    mass = DenseMassAssembler(bkd).assemble(space)
    bkd.assert_allclose(
        mass.format().matrix(),
        _copy_then_sum(bkd, size, space.mass_blocks()),
        rtol=1e-14,
    )


def test_rejects_bad_input(bkd: Backend[Array]) -> None:
    assembler = DenseMassAssembler(bkd)
    with pytest.raises(ValueError, match="shape"):
        assembler.assemble(_Blocks(3, [MassBlock((0, 1), bkd.eye(3))]))
    with pytest.raises(ValueError, match="outside"):
        assembler.assemble(_Blocks(3, [MassBlock((0, 3), bkd.eye(2))]))
    with pytest.raises(TypeError, match="MassBlocksProtocol"):
        assembler.assemble(object())  # type: ignore[arg-type]
