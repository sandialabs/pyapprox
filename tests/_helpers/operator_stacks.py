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

STACK_NAMES = ("csr", "dense-numpy", "dense-torch")


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


V = TypeVar("V", bound=ArrayProtocol)
F = TypeVar("F")
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
class TypedStackCase(Generic[V, F]):
    """A stack's constructors and the backend its vectors live in.

    ``V`` is the stack's vector type (also its reduction type) and ``F``
    the format of its stored operators.
    """

    build: _Build[V, F]
    identity: Callable[[SpaceDescriptor], DiagonalCapableProtocol[V, F]]
    from_diagonal: Callable[[V, SpaceDescriptor], DiagonalCapableProtocol[V, F]]
    space: Callable[[SpaceDescriptor], VectorSpaceProtocol[V, V]]
    rows: Callable[[SpaceDescriptor, Sequence[int]], RowSetProtocol[V]]
    direct_solver: Callable[[], LinearSolverProtocol[V, F]]
    format_type: type[F]
    format_dense: Callable[[F], V]
    """A stored format's matrix as a dense backend array."""
    bkd: Backend[V]
    in_place: bool

    def vec(self, n: int, ncols: int, seed: int) -> V:
        rng = np.random.default_rng(seed)
        return self.bkd.asarray(rng.normal(size=(n, ncols)))

    def dense(self, matrix: NDArray[np.float64]) -> V:
        """``matrix`` as a backend array, for explicit comparisons."""
        return self.bkd.asarray(matrix)


StackCase = TypedStackCase[Any, Any]
"""What the parametrized ``case`` fixture yields: one test body runs on
stacks with different vector types, so at that boundary the types can
only be ``Any``. Everything inside the helper is typed per stack."""


def csr_case() -> TypedStackCase[NDArray[Any], CSRFormat]:
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
        NumpyBkd(),
        True,
    )


def dense_case(bkd: Backend[Array]) -> TypedStackCase[Array, DenseFormat[Array]]:
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
        bkd,
        False,
    )
