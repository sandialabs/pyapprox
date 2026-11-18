"""Pinned skeleton rows on a small residual, both backends, with a
matrix-free and a dense inner Jacobian.

Checks: pinned rows of the value read state - g and the others are
untouched; the Jacobian and its transpose equal the explicit matrix
(I - P) J + P and its transpose, and keep the inner format (a dense
Jacobian stays factorable); Newton's first step puts the pinned DOFs
at g.
"""

from typing import Callable, Generic, Optional, TypeVar

import pytest

from pyapprox.implicitfunction.residual_derivatives import (
    ResidualDerivatives,
    ResidualJacobian,
)
from pyapprox.pde.coupling.pinning import PinnedSkeletonResidual
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseOperator,
    BackendRowSet,
    BackendVectorSpace,
    DenseFormat,
)
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    LinearizationPoint,
    LinearizedOperator,
    MatrixFree,
    OperatorProperties,
    SpaceDescriptor,
)

F = TypeVar("F", bound=FormatProtocol)

_DESCRIPTOR = SpaceDescriptor("skeleton", 4)
_PINNED = [1, 3]
_POINT = LinearizationPoint(0, 0)


class _Matrix(Generic[Array]):
    """A dense matrix applied as a matrix-free operator."""

    def __init__(self, matrix: Array, bkd: Backend[Array]) -> None:
        self._matrix, self._bkd = matrix, bkd

    def domain(self) -> SpaceDescriptor:
        return _DESCRIPTOR

    def range(self) -> SpaceDescriptor:
        return _DESCRIPTOR

    def properties(self) -> OperatorProperties:
        return OperatorProperties()

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(self, x: Array, out: Optional[Array] = None) -> Array:
        return self._bkd.dot(self._matrix, x)

    def apply_transpose(self, y: Array, out: Optional[Array] = None) -> Array:
        return self._bkd.dot(self._matrix.T, y)


class _Cubic(Generic[Array, F]):
    """R(x) = A x + x^3 - b, its Jacobian wrapped by ``wrap``."""

    def __init__(
        self,
        bkd: Backend[Array],
        wrap: Callable[[Array], ActionOperatorProtocol[Array, F]],
    ) -> None:
        self._bkd, self._wrap = bkd, wrap
        self._a = bkd.asarray(
            [
                [4.0, 1.0, 0.0, 2.0],
                [1.0, 5.0, 1.0, 0.0],
                [0.0, 3.0, 6.0, 1.0],
                [1.0, 0.0, 2.0, 7.0],
            ]
        )
        self._b = bkd.asarray([[1.0], [2.0], [3.0], [4.0]])

    def __call__(self, state: Array) -> Array:
        return self._bkd.dot(self._a, state) + state**3 - self._b

    def jacobian_matrix(self, state: Array) -> Array:
        return self._a + self._bkd.diag(3.0 * state[:, 0] ** 2)

    def _jacobian(
        self, state: Array, point: LinearizationPoint
    ) -> LinearizedOperator[Array, F]:
        return LinearizedOperator(self._wrap(self.jacobian_matrix(state)), point)

    def derivatives(self) -> ResidualDerivatives[Array, F]:
        return ResidualDerivatives.state_only(ResidualJacobian.general(self._jacobian))

    def parameters_version(self) -> int:
        return 0


def _matrix_free(bkd: Backend[Array]) -> _Cubic[Array, MatrixFree]:
    def wrap(matrix: Array) -> _Matrix[Array]:
        return _Matrix(matrix, bkd)

    return _Cubic(bkd, wrap)


def _dense(bkd: Backend[Array]) -> _Cubic[Array, DenseFormat[Array]]:
    def wrap(matrix: Array) -> BackendDenseOperator[Array]:
        return BackendDenseOperator(matrix, _DESCRIPTOR, bkd)

    return _Cubic(bkd, wrap)


def _pin(
    inner: _Cubic[Array, F], bkd: Backend[Array]
) -> tuple[PinnedSkeletonResidual[Array, F], Array]:
    space = BackendVectorSpace(_DESCRIPTOR, bkd)
    g = bkd.asarray([[0.5], [-2.0]])
    # Entries off the pinned rows are never read.
    values = bkd.asarray([[99.0], [0.5], [99.0], [-2.0]])
    rows = BackendRowSet(_DESCRIPTOR, _PINNED, bkd)
    return PinnedSkeletonResidual(inner, space, rows, values), g


def _x(bkd: Backend[Array]) -> Array:
    return bkd.asarray([[0.3], [-0.7], [1.1], [0.2]])


@pytest.mark.parametrize("make", [_matrix_free, _dense])
def test_value_replaces_pinned_rows(
    bkd: Backend[Array], make: Callable[[Backend[Array]], _Cubic[Array, F]]
) -> None:
    inner = make(bkd)
    pinned, g = _pin(inner, bkd)
    x = _x(bkd)
    value, raw = pinned(x), inner(x)
    bkd.assert_allclose(value[_PINNED], x[_PINNED] - g, rtol=1e-14)
    bkd.assert_allclose(value[[0, 2]], raw[[0, 2]], rtol=1e-14)


@pytest.mark.parametrize("make", [_matrix_free, _dense])
def test_jacobian_and_transpose_match_the_matrix(
    bkd: Backend[Array], make: Callable[[Backend[Array]], _Cubic[Array, F]]
) -> None:
    inner = make(bkd)
    pinned, _ = _pin(inner, bkd)
    x = _x(bkd)
    expected = inner.jacobian_matrix(x)
    for k in _PINNED:
        expected[k] = bkd.eye(4)[k]
    jacobian = pinned.derivatives().jacobian
    assert jacobian is not None
    op = jacobian(x, _POINT)
    probe = bkd.asarray([[1.0, 0.5], [-2.0, 1.0], [0.25, 3.0], [4.0, -1.0]])
    bkd.assert_allclose(op.apply(probe), bkd.dot(expected, probe), rtol=1e-13)
    bkd.assert_allclose(
        op.apply_transpose(probe), bkd.dot(expected.T, probe), rtol=1e-13
    )


def test_dense_jacobian_stays_stored(bkd: Backend[Array]) -> None:
    inner = _dense(bkd)
    pinned, _ = _pin(inner, bkd)
    x = _x(bkd)
    jacobian = pinned.derivatives().jacobian
    assert jacobian is not None
    stored = jacobian(x, _POINT).format()
    assert isinstance(stored, DenseFormat)
    bkd.assert_allclose(
        stored.matrix(), jacobian(x, _POINT).apply(bkd.eye(4)), rtol=1e-14
    )


@pytest.mark.parametrize("make", [_matrix_free, _dense])
def test_newton_step_lands_pinned_dofs_on_g(
    bkd: Backend[Array], make: Callable[[Backend[Array]], _Cubic[Array, F]]
) -> None:
    pinned, g = _pin(make(bkd), bkd)
    x = _x(bkd)
    jacobian = pinned.derivatives().jacobian
    assert jacobian is not None
    matrix = jacobian(x, _POINT).apply(bkd.eye(4))
    step = bkd.solve(matrix, -pinned(x))
    bkd.assert_allclose((x + step)[_PINNED], g, rtol=1e-13)


def test_rejects_bad_input(bkd: Backend[Array]) -> None:
    inner = _matrix_free(bkd)
    space = BackendVectorSpace(_DESCRIPTOR, bkd)
    rows = BackendRowSet(_DESCRIPTOR, _PINNED, bkd)
    values = bkd.zeros((4, 1))
    other = BackendRowSet(SpaceDescriptor("other", 4), _PINNED, bkd)
    with pytest.raises(ValueError):
        PinnedSkeletonResidual(inner, space, other, values)
    with pytest.raises(ValueError, match="single"):
        PinnedSkeletonResidual(inner, space, rows, bkd.zeros((4, 2)))
    with pytest.raises(TypeError, match="ResidualProtocol"):
        PinnedSkeletonResidual(object(), space, rows, values)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="RowSetProtocol"):
        PinnedSkeletonResidual(inner, space, object(), values)  # type: ignore[arg-type]
