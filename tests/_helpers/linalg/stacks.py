"""The linear-algebra stacks of this repository, as test cases.

Each case builds a stack's vector space and stored operators from plain
NumPy data (``TypedStackCase``, in ``stack_case.py`` with the type
variables suites annotate with). The ``case`` fixture of
``tests/util/linalg/operators/conftest.py`` serves these.
"""

from typing import Any, Optional, TypeVar, Union

import numpy as np
import pytest
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
    OperatorProperties,
    SpaceDescriptor,
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
from tests._helpers.linalg.stack_case import TypedStackCase

STACK_NAMES = ("csr", "dense-numpy", "dense-torch", "mock")
ARRAY_STACK_NAMES = ("csr", "dense-numpy", "dense-torch")
"""The stacks whose vectors are backend arrays, for code that still
takes arrays where a stack vector belongs."""


# NumPy cases compute expected values with NumpyBkd, whose array type is
# NDArray[Any]: the dtype varies (float values, integer indices).
CsrCase = TypedStackCase[NDArray[np.float64], CSRFormat, NDArray[Any]]
DenseCase = TypedStackCase[ArrayProtocol, DenseFormat[ArrayProtocol], ArrayProtocol]
"""The dense case over some backend array: NumPy or Torch, decided by the
fixture's backend, named here without importing Torch."""
MockCase = TypedStackCase[MockVector, MockFormat, NDArray[Any]]
StackCase = Union[CsrCase, DenseCase, MockCase]
"""What the parametrized ``case`` fixture yields. Test bodies annotate
their parameter as ``TypedStackCase[V, F, A]`` instead, so each body is
checked against any stack rather than a union of these."""

T = TypeVar("T")


def _same(x: T) -> T:
    return x


def _same_in(array: T, descriptor: SpaceDescriptor) -> T:
    return array


def csr_case() -> CsrCase:
    def build(
        matrix: NDArray[np.float64],
        domain: SpaceDescriptor,
        range: Optional[SpaceDescriptor] = None,
        properties: Optional[OperatorProperties] = None,
    ) -> ScipyCSROperator[np.float64]:
        return ScipyCSROperator(
            csr_matrix(matrix), domain, range, properties, scalar=np.float64
        )

    def format_dense(fmt: CSRFormat) -> NDArray[np.float64]:
        # scipy's sparse matrices are untyped.
        dense: NDArray[np.float64] = fmt.matrix().toarray()
        return dense

    def space(descriptor: SpaceDescriptor) -> NumpyVectorSpace[np.float64]:
        return NumpyVectorSpace(descriptor, np.float64)

    def identity(descriptor: SpaceDescriptor) -> ScipyCSROperator[np.float64]:
        return ScipyCSROperator.identity(descriptor, np.float64)

    return TypedStackCase(
        build,
        identity,
        ScipyCSROperator.from_diagonal,
        space,
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


def mock_case() -> MockCase:
    """The opaque mock stack; tests see its vectors only through
    ``array`` and ``vector``."""

    def build(
        matrix: NDArray[np.float64],
        domain: SpaceDescriptor,
        range: Optional[SpaceDescriptor] = None,
        properties: Optional[OperatorProperties] = None,
    ) -> MockOperator:
        return MockOperator.from_dense(matrix, domain, range, properties)

    def format_dense(fmt: MockFormat) -> NDArray[np.float64]:
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


def stack_case(request: pytest.FixtureRequest) -> StackCase:
    """The case of the stack named by ``request.param`` (one of
    ``STACK_NAMES``): the body of a ``case`` fixture parametrized over
    them. The dense stacks take the ``numpy_bkd`` and ``torch_bkd``
    fixtures, so a missing Torch skips its case."""
    if request.param == "csr":
        return csr_case()
    if request.param == "mock":
        return mock_case()
    fixture = "numpy_bkd" if request.param == "dense-numpy" else "torch_bkd"
    return dense_case(request.getfixturevalue(fixture))
