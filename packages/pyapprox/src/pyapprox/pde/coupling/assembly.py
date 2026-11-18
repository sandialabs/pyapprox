r"""Assemblers: skeleton mass blocks into an operator of a vector stack.

A skeleton space gives its mass as dense blocks, each with the global
DOFs of its rows and columns, the hand-off every assembly library
accepts (a local matrix and its indices). An assembler belongs to a
vector stack and forms that stack's operator, summing blocks where
their DOFs overlap. :class:`DenseMassAssembler` is the dense backend
one; a sparse or distributed stack supplies its own.
"""

from typing import Generic, Protocol, Sequence, runtime_checkable

from pyapprox.pde.coupling.spaces import MassBlock
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseOperator
from pyapprox.util.linalg.operators import OperatorProperties, SpaceDescriptor


@runtime_checkable
class MassBlocksProtocol(Protocol, Generic[Array]):
    """What a mass assembler reads of a skeleton space."""

    def descriptor(self) -> SpaceDescriptor: ...

    def mass_blocks(self) -> Sequence[MassBlock[Array]]: ...


class DenseMassAssembler(Generic[Array]):
    """Assembles the mass as a dense backend operator, declared
    symmetric positive definite.

    Parameters
    ----------
    bkd
        The backend of the operator and of the blocks.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def assemble(self, space: MassBlocksProtocol[Array]) -> BackendDenseOperator[Array]:
        """The sum of ``space``'s blocks, each at its DOFs.

        Raises
        ------
        TypeError
            If ``space`` does not satisfy :class:`MassBlocksProtocol`.
        ValueError
            If a block's shape does not match its DOFs, or a DOF lies
            outside the space.
        """
        if not isinstance(space, MassBlocksProtocol):
            raise TypeError(
                "space must satisfy MassBlocksProtocol, got "
                f"{type(space).__name__}"
            )
        bkd, descriptor = self._bkd, space.descriptor()
        size = descriptor.size
        matrix = bkd.zeros((size, size))
        for block in space.mass_blocks():
            n = len(block.dofs)
            if tuple(block.matrix.shape) != (n, n):
                raise ValueError(
                    f"a block on {n} DOFs has shape {tuple(block.matrix.shape)}"
                )
            if any(not 0 <= d < size for d in block.dofs):
                raise ValueError(
                    f"block DOFs {block.dofs} lie outside [0, {size})"
                )
            index = bkd.asarray(list(block.dofs), dtype=bkd.int64_dtype())
            rows, cols = bkd.reshape(index, (n, 1)), bkd.reshape(index, (1, n))
            # A block's own DOFs are distinct, so the update adds once each.
            matrix = bkd.index_update(
                matrix, (rows, cols), matrix[rows, cols] + block.matrix
            )
        props = OperatorProperties(symmetric=True, spd=True)
        return BackendDenseOperator(matrix, descriptor, bkd, properties=props)
