"""Linear-algebra stacks for conformance tests of the operator layer.

Each :class:`StackCase` builds a stack's vector space and stored
operators from plain NumPy data, so one test body runs on every stack.
Use the ``case`` fixture of ``tests/util/linalg/operators/conftest.py``.
"""

from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, Protocol, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, ArrayProtocol, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendDenseOperator,
    BackendRowSet,
    BackendVectorSpace,
    DenseFormat,
)
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    DiagonalCapableProtocol,
    LinearSolverProtocol,
    OperatorProperties,
    RowSetProtocol,
    SpaceDescriptor,
    VectorSpaceProtocol,
)
from pyapprox.util.linalg.scipy_stack import (
    CSRFormat,
    NumpyRowSet,
    NumpyVectorSpace,
    ScipyCSROperator,
    ScipySparseLU,
)
from tests._helpers.linalg.mock_stack import (
    MockFormat,
    MockLU,
    MockOperator,
    MockRowSet,
    MockVector,
    MockVectorSpace,
    peek,
    wrap,
)

STACK_NAMES = ("csr", "dense-numpy", "dense-torch", "mock")


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
F = TypeVar("F")
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
    space: Callable[[SpaceDescriptor], VectorSpaceProtocol[V, A]]
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

    def assert_dot_identity(
        self, op: ActionOperatorProtocol[V, Any], seed: int
    ) -> None:
        r"""Assert :math:`\langle A x, y \rangle = \langle x, A^\top y \rangle`
        for random single vectors."""
        bkd = self.bkd
        x = self.vec(op.domain(), 1, seed)
        y = self.vec(op.range(), 1, seed + 1)
        lhs = bkd.sum(self.array(op.apply(x)) * self.array(y))
        rhs = bkd.sum(self.array(x) * self.array(op.apply_transpose(y)))
        bkd.assert_allclose(
            bkd.reshape(lhs, (1,)), bkd.reshape(rhs, (1,)), rtol=1e-12
        )


StackCase = TypedStackCase[Any, Any, Any]
"""What the parametrized ``case`` fixture yields: one test body runs on
stacks with different vector types, so at that boundary the types can
only be ``Any``. Everything inside the helper is typed per stack."""


def _same(x: Any) -> Any:
    return x


def _same_in(array: Any, descriptor: SpaceDescriptor) -> Any:
    return array


def csr_case() -> TypedStackCase[NDArray[Any], CSRFormat, NDArray[Any]]:
    def build(
        matrix: NDArray[np.float64],
        domain: SpaceDescriptor,
        range: Optional[SpaceDescriptor] = None,
        properties: Optional[OperatorProperties] = None,
    ) -> ScipyCSROperator:
        return ScipyCSROperator(csr_matrix(matrix), domain, range, properties)

    def format_dense(fmt: CSRFormat) -> NDArray[Any]:
        dense: NDArray[Any] = fmt.matrix().toarray()
        return dense

    return TypedStackCase(
        build,
        ScipyCSROperator.identity,
        ScipyCSROperator.from_diagonal,
        NumpyVectorSpace,
        NumpyRowSet,
        ScipySparseLU,
        CSRFormat,
        format_dense,
        _same,
        _same_in,
        NumpyBkd(),
        True,
    )


def dense_case(
    bkd: Backend[Array],
) -> TypedStackCase[Array, DenseFormat[Array], Array]:
    def build(
        matrix: NDArray[np.float64],
        domain: SpaceDescriptor,
        range: Optional[SpaceDescriptor] = None,
        properties: Optional[OperatorProperties] = None,
    ) -> BackendDenseOperator[Array]:
        return BackendDenseOperator(bkd.asarray(matrix), domain, bkd, range, properties)

    return TypedStackCase(
        build,
        lambda descriptor: BackendDenseOperator.identity(descriptor, bkd),
        lambda d, descriptor: BackendDenseOperator.from_diagonal(d, descriptor, bkd),
        lambda descriptor: BackendVectorSpace(descriptor, bkd),
        lambda descriptor, rows: BackendRowSet(descriptor, rows, bkd),
        lambda: BackendDenseLU(bkd),
        DenseFormat,
        lambda fmt: fmt.matrix(),
        _same,
        _same_in,
        bkd,
        False,
    )


def mock_case() -> TypedStackCase[MockVector, MockFormat, NDArray[Any]]:
    """The opaque mock stack; tests see its vectors only through
    ``array`` and ``vector``."""

    def build(
        matrix: NDArray[np.float64],
        domain: SpaceDescriptor,
        range: Optional[SpaceDescriptor] = None,
        properties: Optional[OperatorProperties] = None,
    ) -> MockOperator:
        return MockOperator.from_dense(matrix, domain, range, properties)

    def format_dense(fmt: MockFormat) -> NDArray[Any]:
        return fmt.dense()

    return TypedStackCase(
        build,
        MockOperator.identity,
        MockOperator.from_diagonal,
        MockVectorSpace,
        MockRowSet,
        MockLU,
        MockFormat,
        format_dense,
        peek,
        wrap,
        NumpyBkd(),
        True,
    )
