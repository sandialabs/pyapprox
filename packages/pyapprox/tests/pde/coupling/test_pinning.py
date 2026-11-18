"""Pinned skeleton rows on a small dense residual, both backends.

Checks: pinned rows of the value read state - g and the others are
untouched; the Jacobian and its transpose equal the explicit matrix
(I - P^T P) J + P^T P and its transpose; Newton's first step puts the
pinned DOFs at g.
"""

from typing import Generic, Optional

from pyapprox.implicitfunction.residual_derivatives import (
    ResidualDerivatives,
    ResidualJacobian,
)
from pyapprox.pde.coupling.pinning import PinnedSkeletonResidual
from pyapprox.pde.coupling.ports import BackendPortRows
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import (
    LinearizationPoint,
    LinearizedOperator,
    MatrixFree,
    OperatorProperties,
    SpaceDescriptor,
)

_DESCRIPTOR = SpaceDescriptor("skeleton", 4)
_PINNED = [1, 3]


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


class _Cubic(Generic[Array]):
    """R(x) = A x + x^3 - b."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
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
    ) -> LinearizedOperator[Array, MatrixFree]:
        return LinearizedOperator(
            _Matrix(self.jacobian_matrix(state), self._bkd), point
        )

    def derivatives(self) -> ResidualDerivatives[Array, MatrixFree]:
        return ResidualDerivatives.state_only(ResidualJacobian.general(self._jacobian))

    def parameters_version(self) -> int:
        return 0


def _pinned(
    bkd: Backend[Array],
) -> tuple[_Cubic[Array], PinnedSkeletonResidual[Array, Array], Array]:
    inner = _Cubic(bkd)
    g = bkd.asarray([[0.5], [-2.0]])
    rows = BackendPortRows(_DESCRIPTOR, _PINNED, bkd)
    return inner, PinnedSkeletonResidual(inner, rows, g), g


def test_value_replaces_pinned_rows(bkd: Backend[Array]) -> None:
    inner, pinned, g = _pinned(bkd)
    x = bkd.asarray([[0.3], [-0.7], [1.1], [0.2]])
    value, raw = pinned(x), inner(x)
    bkd.assert_allclose(value[_PINNED], x[_PINNED] - g, rtol=1e-14)
    bkd.assert_allclose(value[[0, 2]], raw[[0, 2]], rtol=1e-14)


def test_jacobian_and_transpose_match_the_matrix(bkd: Backend[Array]) -> None:
    inner, pinned, _ = _pinned(bkd)
    x = bkd.asarray([[0.3], [-0.7], [1.1], [0.2]])
    expected = inner.jacobian_matrix(x)
    for k in _PINNED:
        expected[k] = bkd.eye(4)[k]
    jacobian = pinned.derivatives().jacobian
    assert jacobian is not None
    op = jacobian(x, LinearizationPoint(0, 0))
    probe = bkd.asarray([[1.0, 0.5], [-2.0, 1.0], [0.25, 3.0], [4.0, -1.0]])
    bkd.assert_allclose(op.apply(probe), bkd.dot(expected, probe), rtol=1e-13)
    bkd.assert_allclose(
        op.apply_transpose(probe), bkd.dot(expected.T, probe), rtol=1e-13
    )


def test_newton_step_lands_pinned_dofs_on_g(bkd: Backend[Array]) -> None:
    _, pinned, g = _pinned(bkd)
    x = bkd.asarray([[0.3], [-0.7], [1.1], [0.2]])
    jacobian = pinned.derivatives().jacobian
    assert jacobian is not None
    op = jacobian(x, LinearizationPoint(0, 0))
    matrix = op.apply(bkd.eye(4))
    step = bkd.solve(matrix, -pinned(x))
    bkd.assert_allclose((x + step)[_PINNED], g, rtol=1e-13)
