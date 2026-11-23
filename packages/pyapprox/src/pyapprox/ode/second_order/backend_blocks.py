r"""The two state blocks stored as one dense :math:`2n` backend array.

:class:`BackendDenseStateBlocks` satisfies
:class:`~pyapprox.ode.second_order.reduction.StateBlocksProtocol` on the
backend stack: :math:`w = [u; v]` is one array of :math:`2n` rows, the
displacement first, and block operators are assembled into one dense
matrix solved by an LU of size :math:`2n`.
"""

from typing import Generic, Optional

from pyapprox.ode.residual_derivatives import backend_state_space
from pyapprox.ode.second_order.reduction import Blocks
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendDenseOperator,
    BackendVectorSpace,
    DenseFormat,
)
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    LinearSolverProtocol,
    OperatorProperties,
)


class BackendDenseStateBlocks(Generic[Array]):
    """:math:`w = [u; v]` as one dense backend array of :math:`2n` rows.

    Parameters
    ----------
    bkd : Backend
        The backend of the states.
    nstates : int
        :math:`n`, the size of one block.
    solver : LinearSolverProtocol, optional
        Solves with the assembled operators; a dense LU by default.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        nstates: int,
        solver: Optional[LinearSolverProtocol[Array, DenseFormat[Array]]] = None,
    ) -> None:
        self._bkd, self._n = bkd, nstates
        self._half: BackendVectorSpace[Array] = backend_state_space(bkd, nstates)
        self._space: BackendVectorSpace[Array] = backend_state_space(bkd, 2 * nstates)
        self._identity = BackendDenseOperator.identity(self._half.descriptor(), bkd)
        self._solver: LinearSolverProtocol[Array, DenseFormat[Array]] = (
            BackendDenseLU(bkd) if solver is None else solver
        )

    def half_space(self) -> BackendVectorSpace[Array]:
        return self._half

    def space(self) -> BackendVectorSpace[Array]:
        return self._space

    def displacement(self, state: Array) -> Array:
        return state[: self._n]

    def velocity(self, state: Array) -> Array:
        return state[self._n :]

    def join(self, displacement: Array, velocity: Array) -> Array:
        return self._bkd.concatenate([displacement, velocity], axis=0)

    def identity(self) -> BackendDenseOperator[Array]:
        return self._identity

    def assemble(
        self,
        blocks: Blocks[Array, DenseFormat[Array]],
        properties: OperatorProperties,
    ) -> ActionOperatorProtocol[Array, DenseFormat[Array]]:
        """The dense :math:`2n` matrix of ``blocks``, a None block zero."""
        bkd = self._bkd
        rows = [bkd.hstack([self._matrix(block) for block in row]) for row in blocks]
        return BackendDenseOperator(
            bkd.vstack(rows), self._space.descriptor(), bkd, properties=properties
        )

    def _matrix(
        self, block: Optional[ActionOperatorProtocol[Array, DenseFormat[Array]]]
    ) -> Array:
        if block is None:
            return self._bkd.zeros((self._n, self._n))
        return block.format().matrix()

    def solver(self) -> LinearSolverProtocol[Array, DenseFormat[Array]]:
        return self._solver

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nstates={self._n})"
