r"""Block-diagonal operators on the dense backend stack.

An operator of ``k`` independent ``m x m`` blocks acts on vectors of
size :math:`k m` laid out block after block, the same backend arrays of
shape ``(k m, ncols)`` as :mod:`~pyapprox.util.linalg.backend_dense`.
Storing the ``(k, m, m)`` blocks costs :math:`k m^2` rather than
:math:`(k m)^2`, and a solve :math:`k m^3` rather than :math:`(k m)^3`.
Every operation is a batched backend expression, so a torch graph
through the blocks and the operands is preserved; ``out`` is ignored.
"""

import warnings
from typing import Generic, Optional, Sequence, TypeVar

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendRowSet,
    require_backend_dtype,
)
from pyapprox.util.linalg.operators.algebra import Transposed
from pyapprox.util.linalg.operators.descriptor import SpaceDescriptor
from pyapprox.util.linalg.operators.protocols import (
    ActionOperatorProtocol,
    OperatorProperties,
    RowSetProtocol,
)
from pyapprox.util.linalg.operators.solvers import (
    SolveInfo,
    SolverCapabilities,
    SolverHandleBase,
    SolverHandleProtocol,
    require_format,
    require_square,
)

# The vector type of a row set given to a row edit; see FormatProtocol.
W = TypeVar("W")

_SYMMETRY_RTOL = 1e-12


def _blockwise(x: Array, nblocks: int, block_size: int, bkd: Backend[Array]) -> Array:
    """A ``(k m, ncols)`` block of vectors as ``(k, m, ncols)``."""
    return bkd.reshape(x, (nblocks, block_size, x.shape[1]))


def _block_keep_mask(
    rows: RowSetProtocol[W], nblocks: int, block_size: int, bkd: Backend[Array]
) -> Array:
    """The keep mask of a backend row set, as ``(k, m, 1)``."""
    if not isinstance(rows, BackendRowSet):
        raise TypeError(
            "BlockDiagonalFormat row edits need a BackendRowSet, "
            f"got {type(rows).__name__}"
        )
    return bkd.reshape(rows.keep_mask(), (nblocks, block_size, 1))


class BlockDiagonalFormat(Generic[Array]):
    """Block-diagonal storage, the format of
    :class:`BackendBlockDiagonalOperator`.

    Every combination keeps the structure: a scaled sum adds blocks, a
    transpose transposes each block, and a row edit touches rows within
    their own blocks (a replaced row becomes a row of the identity,
    which is block-diagonal).

    Parameters
    ----------
    blocks : Array
        Shape ``(k, m, m)``.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, blocks: Array, bkd: Backend[Array]) -> None:
        self._blocks = blocks
        self._bkd = bkd

    def blocks(self) -> Array:
        """The stored ``(k, m, m)`` blocks. Not to be modified."""
        return self._blocks

    def scaled_sum(
        self,
        coefficient: float,
        others: Sequence[tuple[float, "BlockDiagonalFormat[Array]"]],
    ) -> "BlockDiagonalFormat[Array]":
        total = coefficient * self._blocks
        for c, other in others:
            if tuple(other._blocks.shape) != tuple(self._blocks.shape):
                raise ValueError(
                    "block-diagonal operators in a sum must have the same "
                    f"blocks, got {tuple(self._blocks.shape)} and "
                    f"{tuple(other._blocks.shape)}"
                )
            total = total + c * other._blocks
        return BlockDiagonalFormat(total, self._bkd)

    def transposed(self) -> "BlockDiagonalFormat[Array]":
        return BlockDiagonalFormat(
            self._bkd.transpose(self._blocks, (0, 2, 1)), self._bkd
        )

    def rows_replaced(self, rows: RowSetProtocol[W]) -> "BlockDiagonalFormat[Array]":
        nblocks, block_size = self._blocks.shape[0], self._blocks.shape[1]
        keep = _block_keep_mask(rows, nblocks, block_size, self._bkd)
        eye = self._bkd.eye(block_size, dtype=self._bkd.default_dtype())
        return BlockDiagonalFormat(self._blocks * keep + eye * (1.0 - keep), self._bkd)

    def rows_zeroed(self, rows: RowSetProtocol[W]) -> "BlockDiagonalFormat[Array]":
        nblocks, block_size = self._blocks.shape[0], self._blocks.shape[1]
        keep = _block_keep_mask(rows, nblocks, block_size, self._bkd)
        return BlockDiagonalFormat(self._blocks * keep, self._bkd)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={tuple(self._blocks.shape)})"


class BackendBlockDiagonalOperator(Generic[Array]):
    """A linear operator of independent square blocks.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.protocols.DiagonalCapableProtocol`
    with backend arrays as vectors and :class:`BlockDiagonalFormat` as its
    format.

    Parameters
    ----------
    blocks : Array
        Shape ``(k, m, m)``.
    descriptor : SpaceDescriptor
        Domain and range, of size ``k m``; its dtype must be the
        backend's configured dtype.
    bkd : Backend[Array]
        Computational backend.
    properties : OperatorProperties, optional
        Declared properties. ``symmetric``, ``diagonal`` and ``identity``
        are verified; ``spd`` is not (that would cost a factorization).

    Raises
    ------
    ValueError
        If ``blocks`` is not ``(k, m, m)`` with ``k m`` the descriptor's
        size, the descriptor's dtype is not the backend's, or a verified
        property does not hold.
    """

    def __init__(
        self,
        blocks: Array,
        descriptor: SpaceDescriptor,
        bkd: Backend[Array],
        properties: Optional[OperatorProperties] = None,
    ) -> None:
        require_backend_dtype(descriptor, bkd, "BackendBlockDiagonalOperator")
        if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2]:
            raise ValueError(
                f"blocks must have shape (k, m, m), got {tuple(blocks.shape)}"
            )
        covered = blocks.shape[0] * blocks.shape[1]
        if covered != descriptor.size:
            raise ValueError(
                f"blocks {tuple(blocks.shape)} cover {covered} entries, the "
                f"descriptor {descriptor.size}"
            )
        self._descriptor = descriptor
        self._nblocks: int = blocks.shape[0]
        self._block_size: int = blocks.shape[1]
        self._bkd = bkd
        self._properties = OperatorProperties() if properties is None else properties
        self._verify_properties(blocks)
        self._format = BlockDiagonalFormat(blocks, bkd)

    def _verify_properties(self, blocks: Array) -> None:
        if blocks.shape[0] == 0:
            return
        bkd, props = self._bkd, self._properties
        eye = bkd.eye(self._block_size, dtype=bkd.default_dtype())
        # Checks only: converting the reductions to floats detaches
        # nothing that is returned.
        if props.symmetric:
            transposed = bkd.transpose(blocks, (0, 2, 1))
            asymmetry = bkd.to_float(bkd.max(bkd.abs(blocks - transposed)))
            scale = bkd.to_float(bkd.max(bkd.abs(blocks)))
            if asymmetry > _SYMMETRY_RTOL * max(scale, 1.0):
                raise ValueError(
                    f"blocks declared symmetric are not: max |B - B^T| = "
                    f"{asymmetry:.3e} against max |B| = {scale:.3e}"
                )
        if props.diagonal:
            if bkd.to_float(bkd.max(bkd.abs(blocks - blocks * eye))) > 0.0:
                raise ValueError("blocks declared diagonal have off-diagonal nonzeros")
        if props.identity:
            if bkd.to_float(bkd.max(bkd.abs(blocks - eye))) > 0.0:
                raise ValueError("blocks declared identity differ from the identity")

    def nblocks(self) -> int:
        """The number of blocks, ``k``."""
        return self._nblocks

    def block_size(self) -> int:
        """The size of each block, ``m``."""
        return self._block_size

    def domain(self) -> SpaceDescriptor:
        return self._descriptor

    def range(self) -> SpaceDescriptor:
        return self._descriptor

    def properties(self) -> OperatorProperties:
        return self._properties

    def format(self) -> BlockDiagonalFormat[Array]:
        return self._format

    def _check_apply(self, x: Array, out: Optional[Array]) -> None:
        size = self._descriptor.size
        if x.ndim != 2 or x.shape[0] != size:
            raise ValueError(
                f"input must have shape ({size}, ncols), got {tuple(x.shape)}"
            )
        if out is x:
            raise ValueError("out must not alias the input of apply")

    def _apply(self, subscripts: str, x: Array) -> Array:
        bkd = self._bkd
        product = bkd.einsum(
            subscripts,
            self._format.blocks(),
            _blockwise(x, self._nblocks, self._block_size, bkd),
        )
        return bkd.reshape(product, (self._descriptor.size, x.shape[1]))

    def apply(self, x: Array, out: Optional[Array] = None) -> Array:
        self._check_apply(x, out)
        return self._apply("kij,kjc->kic", x)

    def apply_transpose(self, y: Array, out: Optional[Array] = None) -> Array:
        self._check_apply(y, out)
        return self._apply("kji,kjc->kic", y)

    def diagonal(self) -> Array:
        return self._bkd.reshape(
            self._bkd.einsum("kii->ki", self._format.blocks()),
            (self._descriptor.size, 1),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nblocks={self._nblocks}, "
            f"block_size={self._block_size}, {self._properties!r})"
        )


class _BlockDiagonalLUHandle(SolverHandleBase[Array]):
    """Batched LU factors of every block, solving with A or, sharing
    them, A^T."""

    def __init__(
        self,
        op: ActionOperatorProtocol[Array, BlockDiagonalFormat[Array]],
        factors: Optional[tuple[Array, Array]],
        failure: str,
        transposed: bool,
        block_shape: tuple[int, int],
        bkd: Backend[Array],
    ) -> None:
        super().__init__(op)
        self._op = op
        self._factors = factors
        self._failure = failure
        self._transposed = transposed
        self._block_shape = block_shape
        self._bkd = bkd

    def solve(
        self, rhs: Array, tol: Optional[float] = None, x0: Optional[Array] = None
    ) -> tuple[Array, SolveInfo]:
        size = self.domain().size
        if rhs.ndim != 2 or rhs.shape[0] != size:
            raise ValueError(
                f"rhs must have shape ({size}, ncols), got {tuple(rhs.shape)}"
            )
        if self._factors is None:
            return rhs * float("nan"), SolveInfo(False, message=self._failure)
        lu, pivots = self._factors
        nblocks, block_size = self._block_shape
        bkd = self._bkd
        solution = bkd.reshape(
            bkd.lu_solve(
                lu,
                pivots,
                _blockwise(rhs, nblocks, block_size, bkd),
                transpose=self._transposed,
            ),
            (size, rhs.shape[1]),
        )
        if not bkd.all_bool(bkd.isfinite(solution)):
            return solution, SolveInfo(False, message="non-finite solution")
        return solution, SolveInfo(True)

    def transpose(self) -> "_BlockDiagonalLUHandle[Array]":
        return _BlockDiagonalLUHandle(
            Transposed(self._op),
            self._factors,
            self._failure,
            not self._transposed,
            self._block_shape,
            self._bkd,
        )


class BlockDiagonalLU(Generic[Array]):
    """Batched LU of every block, for block-diagonal operators.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.solvers.LinearSolverProtocol`
    for :class:`BlockDiagonalFormat`. The ``k`` blocks are factored once,
    together, at ``setup``: :math:`k m^3` work, against :math:`(k m)^3`
    for the dense operator. The backend's LU is differentiable, so a
    torch graph through the blocks and the right-hand side is preserved.
    A singular block gives handles whose solves report
    ``converged=False``.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def capabilities(self) -> SolverCapabilities:
        return SolverCapabilities(transpose=True)

    def setup(
        self, op: ActionOperatorProtocol[Array, BlockDiagonalFormat[Array]]
    ) -> SolverHandleProtocol[Array]:
        fmt = require_format(op, BlockDiagonalFormat, "BlockDiagonalLU")
        require_square(op, "BlockDiagonalLU")
        blocks = fmt.blocks()
        factors: Optional[tuple[Array, Array]] = None
        failure = ""
        try:
            # An exactly singular block only warns here; the solve then
            # reports it as non-finite.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                factors = self._bkd.lu_factor(blocks)
        except (RuntimeError, ValueError) as error:
            failure = f"factorization failed: {error}"
        return _BlockDiagonalLUHandle(
            op,
            factors,
            failure,
            False,
            (blocks.shape[0], blocks.shape[1]),
            self._bkd,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
