"""Linear-algebra stacks for conformance tests of the operator layer.

Each :class:`StackCase` builds a stack's vector space and stored
operators from plain NumPy data, so one test body runs on every stack.
Use the ``case`` fixture of ``tests/util/linalg/operators/conftest.py``.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseOperator,
    BackendRowSet,
    BackendVectorSpace,
    DenseFormat,
)
from pyapprox.util.linalg.operators import (
    Layout,
    OperatorProperties,
    RowSetProtocol,
    VectorSpaceProtocol,
)
from pyapprox.util.linalg.scipy_stack import (
    CSRFormat,
    NumpyRowSet,
    NumpyVectorSpace,
    ScipyCSROperator,
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


@dataclass
class StackCase:
    """A stack's constructors and the backend its vectors live in."""

    build: Callable[..., Any]
    identity: Callable[[Layout], Any]
    from_diagonal: Callable[[Any, Layout], Any]
    space: Callable[[Layout], VectorSpaceProtocol[Any, Any]]
    rows: Callable[[Layout, Sequence[int]], RowSetProtocol[Any]]
    format_type: type
    format_dense: Callable[[Any], Any]
    """A stored format's matrix as a dense backend array."""
    bkd: Backend[Any]
    in_place: bool

    def vec(self, n: int, ncols: int, seed: int) -> Any:
        rng = np.random.default_rng(seed)
        return self.bkd.asarray(rng.normal(size=(n, ncols)))

    def dense(self, matrix: NDArray[np.float64]) -> Any:
        """``matrix`` as a backend array, for explicit comparisons."""
        return self.bkd.asarray(matrix)


def csr_case() -> StackCase:
    def build(
        matrix: NDArray[np.float64],
        domain: Layout,
        range: Optional[Layout] = None,
        properties: Optional[OperatorProperties] = None,
    ) -> ScipyCSROperator:
        return ScipyCSROperator(csr_matrix(matrix), domain, range, properties)

    return StackCase(
        build,
        ScipyCSROperator.identity,
        ScipyCSROperator.from_diagonal,
        NumpyVectorSpace,
        NumpyRowSet,
        CSRFormat,
        lambda fmt: fmt.matrix().toarray(),
        NumpyBkd(),
        True,
    )


def dense_case(bkd: Backend[Any]) -> StackCase:
    def build(
        matrix: NDArray[np.float64],
        domain: Layout,
        range: Optional[Layout] = None,
        properties: Optional[OperatorProperties] = None,
    ) -> BackendDenseOperator[Any]:
        return BackendDenseOperator(bkd.asarray(matrix), domain, bkd, range, properties)

    return StackCase(
        build,
        lambda layout: BackendDenseOperator.identity(layout, bkd),
        lambda d, layout: BackendDenseOperator.from_diagonal(d, layout, bkd),
        lambda layout: BackendVectorSpace(layout, bkd),
        lambda layout, rows: BackendRowSet(layout, rows, bkd),
        DenseFormat,
        lambda fmt: fmt.matrix(),
        bkd,
        False,
    )
