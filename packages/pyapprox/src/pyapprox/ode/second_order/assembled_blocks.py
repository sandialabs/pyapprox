r"""The two state blocks stored as one vector of :math:`2n` rows, on any
stack.

:class:`AssembledStateBlocks` satisfies
:class:`~pyapprox.ode.second_order.reduction.StateBlocksProtocol` for any
stack: :math:`w = [u; v]` is one of the stack's vectors, the displacement
first, and a block operator is a
:class:`~pyapprox.util.linalg.operators.BlockAssembled` whose format is
assembled by the stack's format and factored by the stack's solver at
size :math:`2n`. A stack supplies its spaces, its row partition, the
identity on one block and a solver; :func:`backend_dense_state_blocks`
supplies them for the dense backend stack, and
:mod:`~pyapprox.ode.second_order.scipy_blocks` for the SciPy CSR stack.
"""

from typing import Generic, Optional, TypeVar

from pyapprox.ode.residual_derivatives import backend_state_space
from pyapprox.ode.second_order.reduction import Blocks
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendDenseOperator,
    BackendRowPartition,
    DenseFormat,
)
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    BlockAssembled,
    FormatProtocol,
    InnerProductSpaceProtocol,
    LinearSolverProtocol,
    RowPartitionProtocol,
)

V = TypeVar("V")
A = TypeVar("A")
F = TypeVar("F", bound=FormatProtocol)


class AssembledStateBlocks(Generic[V, A, F]):
    r""":math:`w = [u; v]` as one of the stack's vectors, block operators
    assembled at size :math:`2n` in the stack's format.

    Parameters
    ----------
    half_space, space : InnerProductSpaceProtocol
        The space of one block (size :math:`n`) and of :math:`w`
        (:math:`2n`).
    partition : RowPartitionProtocol
        Splits :math:`w` into its two blocks.
    identity : ActionOperatorProtocol
        The identity on one block, in format ``F``.
    solver : LinearSolverProtocol
        Solves with operators of format ``F`` of size :math:`2n`.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    ValueError
        If the partition does not have two parts of ``half_space`` over
        ``space``, or the identity is not on ``half_space``.
    """

    def __init__(
        self,
        half_space: InnerProductSpaceProtocol[V, A],
        space: InnerProductSpaceProtocol[V, A],
        partition: RowPartitionProtocol[V],
        identity: ActionOperatorProtocol[V, F],
        solver: LinearSolverProtocol[V, F],
    ) -> None:
        if not isinstance(partition, RowPartitionProtocol):
            raise TypeError(
                "partition must satisfy RowPartitionProtocol, got "
                f"{type(partition).__name__}"
            )
        if not isinstance(identity, ActionOperatorProtocol):
            raise TypeError(
                "identity must satisfy ActionOperatorProtocol, got "
                f"{type(identity).__name__}"
            )
        if not isinstance(solver, LinearSolverProtocol):
            raise TypeError(
                f"solver must satisfy LinearSolverProtocol, got {type(solver).__name__}"
            )
        half = half_space.descriptor()
        parts = partition.parts()
        if len(parts) != 2:
            raise ValueError(f"the partition must have two parts, got {len(parts)}")
        for part in parts:
            half.require_same(part.descriptor(), "part of the state partition")
        space.descriptor().require_same(
            partition.space().descriptor(), "space of the state partition"
        )
        half.require_same(identity.domain(), "domain of the block identity")
        half.require_same(identity.range(), "range of the block identity")
        self._half, self._space = half_space, space
        self._partition, self._identity, self._solver = partition, identity, solver

    def half_space(self) -> InnerProductSpaceProtocol[V, A]:
        return self._half

    def space(self) -> InnerProductSpaceProtocol[V, A]:
        return self._space

    def displacement(self, state: V) -> V:
        return self._partition.split(state)[0]

    def velocity(self, state: V) -> V:
        return self._partition.split(state)[1]

    def join(self, displacement: V, velocity: V) -> V:
        return self._partition.join([displacement, velocity])

    def identity(self) -> ActionOperatorProtocol[V, F]:
        return self._identity

    def assemble(self, blocks: Blocks[V, F]) -> BlockAssembled[V, F]:
        """The :math:`2n` block operator of ``blocks``, a None block zero."""
        return BlockAssembled(blocks, self._partition)

    def solver(self) -> LinearSolverProtocol[V, F]:
        return self._solver

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nstates={self._half.descriptor().size})"


def backend_dense_state_blocks(
    bkd: Backend[Array],
    nstates: int,
    solver: Optional[LinearSolverProtocol[Array, DenseFormat[Array]]] = None,
) -> AssembledStateBlocks[Array, Array, DenseFormat[Array]]:
    """The assembled storage on the dense backend stack.

    Parameters
    ----------
    bkd : Backend
        The backend of the states.
    nstates : int
        :math:`n`, the size of one block.
    solver : LinearSolverProtocol, optional
        Solves at size :math:`2n`; a dense LU by default.
    """
    half = backend_state_space(bkd, nstates)
    space = backend_state_space(bkd, 2 * nstates)
    return AssembledStateBlocks(
        half,
        space,
        BackendRowPartition(space, [half, half], bkd),
        BackendDenseOperator.identity(half.descriptor(), bkd),
        BackendDenseLU(bkd) if solver is None else solver,
    )
