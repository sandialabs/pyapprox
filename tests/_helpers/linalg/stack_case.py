"""A linear-algebra stack as a test case: what stack-generic suites read.

A :class:`TypedStackCase` holds a stack's constructors (vector spaces,
stored operators, row sets, a direct solver) and the conversions a test
uses to build and inspect its vectors, so one test body runs on every
stack. This module imports protocols only: a suite written against it
pulls in no concrete stack, and a new stack supplies its own case.
``tests/_helpers/linalg/stacks.py`` builds the cases of the stacks in
this repository.
"""

from dataclasses import dataclass
from typing import Callable, Generic, Optional, Protocol, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray

from pyapprox.util.backends.protocols import ArrayProtocol, Backend
from pyapprox.util.linalg.operators import (
    DiagonalCapableProtocol,
    FormatProtocol,
    InnerProductSpaceProtocol,
    LinearMapProtocol,
    LinearSolverProtocol,
    OperatorProperties,
    RowSetProtocol,
    SpaceDescriptor,
)


def random_matrix(nrows: int, ncols: int, seed: int) -> NDArray[np.float64]:
    """A random matrix with about half its entries zero."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(nrows, ncols))
    matrix[rng.random((nrows, ncols)) < 0.5] = 0.0
    return matrix


def spd_matrix(n: int, seed: int) -> NDArray[np.float64]:
    """A random symmetric positive definite matrix."""
    a = np.random.default_rng(seed).normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


V = TypeVar("V")
F = TypeVar("F", bound=FormatProtocol)
A = TypeVar("A", bound=ArrayProtocol)
F_co = TypeVar("F_co", covariant=True)


class _Build(Protocol[V, F_co]):
    """Builds a stack's stored operator from a dense NumPy matrix."""

    def __call__(
        self,
        matrix: NDArray[np.float64],
        domain: SpaceDescriptor,
        range: Optional[SpaceDescriptor] = None,
        properties: Optional[OperatorProperties] = None,
    ) -> DiagonalCapableProtocol[V, F_co]: ...


@dataclass
class TypedStackCase(Generic[V, F, A]):
    """A stack's constructors, and the boundary tests use to inspect it.

    ``V`` is the stack's vector type, ``F`` the format of its stored
    operators, and ``A`` the backend array type tests compute expected
    values in (also the stack's reduction type). For array stacks
    ``V`` is ``A`` and the conversions are the identity; for an opaque
    stack they are the only way a test reads or builds a vector.
    """

    build: _Build[V, F]
    identity: Callable[[SpaceDescriptor], DiagonalCapableProtocol[V, F]]
    from_diagonal: Callable[[V, SpaceDescriptor], DiagonalCapableProtocol[V, F]]
    space: Callable[[SpaceDescriptor], InnerProductSpaceProtocol[V, A]]
    rows: Callable[[SpaceDescriptor, Sequence[int]], RowSetProtocol[V]]
    direct_solver: Callable[[], LinearSolverProtocol[V, F]]
    format_type: type[F]
    format_dense: Callable[[F], A]
    """A stored format's matrix as a dense backend array."""
    to_array: Callable[[V], A]
    from_array: Callable[[A, SpaceDescriptor], V]
    bkd: Backend[A]
    in_place: bool

    def vec(self, descriptor: SpaceDescriptor, ncols: int, seed: int) -> V:
        """A random block of ``ncols`` vectors of ``descriptor``."""
        rng = np.random.default_rng(seed)
        return self.vector(
            self.bkd.asarray(rng.normal(size=(descriptor.size, ncols))), descriptor
        )

    def array(self, x: V) -> A:
        """The entries of ``x`` as a backend array, for assertions."""
        return self.to_array(x)

    def vector(self, array: A, descriptor: SpaceDescriptor) -> V:
        """A vector of ``descriptor`` holding ``array``, for setup."""
        return self.from_array(array, descriptor)

    def dense(self, matrix: NDArray[np.float64]) -> A:
        """``matrix`` as a backend array, for explicit comparisons."""
        return self.bkd.asarray(matrix)

    def times(self, matrix: A, x: V) -> A:
        """The reference product of a dense backend matrix and ``x``."""
        return self.bkd.dot(matrix, self.array(x))

    def assert_vector(
        self, actual: V, expected: A, rtol: float = 1e-7, atol: float = 0.0
    ) -> None:
        """Assert a stack vector equals a backend array."""
        self.bkd.assert_allclose(self.array(actual), expected, rtol=rtol, atol=atol)

    def assert_dot_identity(self, op: LinearMapProtocol[V], seed: int) -> None:
        r"""Assert :math:`\langle A x, y \rangle = \langle x, A^\top y \rangle`
        for random single vectors."""
        bkd = self.bkd
        x = self.vec(op.domain(), 1, seed)
        y = self.vec(op.range(), 1, seed + 1)
        lhs = bkd.sum(self.array(op.apply(x)) * self.array(y))
        rhs = bkd.sum(self.array(x) * self.array(op.apply_transpose(y)))
        bkd.assert_allclose(bkd.reshape(lhs, (1,)), bkd.reshape(rhs, (1,)), rtol=1e-12)
