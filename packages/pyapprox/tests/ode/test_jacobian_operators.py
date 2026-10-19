"""DenseJacobianOperator and MatrixFreeJacobianOperator on Lotka-Volterra.

The model supplies both its Jacobian matrix and its actions, so the two
wrappers can be compared with each other; each is also checked against a
central difference of the residual, which shares no code with either,
and the matrix-free transpose against the dot identity.
"""

import numpy as np
import pytest

from pyapprox.ode.dense_jacobian_operator import (
    DenseJacobianOperator,
    DenseJacobianSnapshot,
)
from pyapprox.ode.matrix_free_jacobian_operator import (
    MatrixFreeJacobianOperator,
    MatrixFreeJacobianSnapshot,
)
from pyapprox.ode.protocols import JacobianOperatorProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseOperator,
    BackendVectorSpace,
    DenseFormat,
)
from pyapprox.util.linalg.operators import (
    GMRES,
    ActionOperatorProtocol,
    FormatProtocol,
    JacobianSnapshotProtocol,
    LinearizationPoint,
    MatrixFree,
    ScaledSum,
    SpaceDescriptor,
)
from tests._helpers.ode_models import LotkaVolterra

_Y = SpaceDescriptor("y", 2)
_POINT = LinearizationPoint(0, ("step", 3))


def _state(bkd: Backend[Array]) -> Array:
    return bkd.asarray([10.0, 5.0])


def _block(bkd: Backend[Array], seed: int) -> Array:
    return bkd.asarray(np.random.default_rng(seed).normal(size=(2, 3)))


def _central_difference(
    model: LotkaVolterra[Array], bkd: Backend[Array], direction: Array
) -> Array:
    """Columns of :math:`(f(y + h d) - f(y - h d)) / 2h`, one per column
    of ``direction``."""
    y, step = _state(bkd), 1e-6
    columns = [
        (
            model(y + step * direction[:, k], 0.0)
            - model(y - step * direction[:, k], 0.0)
        )
        / (2 * step)
        for k in range(direction.shape[1])
    ]
    return bkd.stack(columns, axis=1)


class TestDenseJacobianOperator:
    def test_satisfies_the_protocols(self, bkd: Backend[Array]) -> None:
        producer: JacobianOperatorProtocol[Array, Array, DenseFormat[Array]] = (
            DenseJacobianOperator(LotkaVolterra(bkd), _Y, bkd)
        )
        assert isinstance(producer, JacobianOperatorProtocol)
        op = producer.jacobian_operator(_state(bkd), 0.0, _POINT)
        assert isinstance(op, JacobianSnapshotProtocol)
        assert isinstance(op, DenseJacobianSnapshot)
        assert op.linearization() == _POINT
        assert isinstance(op.format(), DenseFormat)

    def test_action_is_the_residual_derivative(self, bkd: Backend[Array]) -> None:
        model = LotkaVolterra(bkd)
        op = DenseJacobianOperator(model, _Y, bkd).jacobian_operator(
            _state(bkd), 0.0, _POINT
        )
        direction = _block(bkd, 0)
        bkd.assert_allclose(
            op.apply(direction),
            _central_difference(model, bkd, direction),
            rtol=1e-7,
            atol=1e-8,
        )

    def test_refuses_a_jacobian_of_the_wrong_shape(self, bkd: Backend[Array]) -> None:
        producer = DenseJacobianOperator(
            LotkaVolterra(bkd), SpaceDescriptor("y", 3), bkd
        )
        with pytest.raises(ValueError, match="shape"):
            producer.jacobian_operator(_state(bkd), 0.0, _POINT)

    def test_refuses_a_model_without_a_jacobian(self, bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="JacobianMatrixProtocol"):
            DenseJacobianOperator(object(), _Y, bkd)  # type: ignore[arg-type]


class _WrongShapeModel(LotkaVolterra[Array]):
    def apply_jacobian(self, y: Array, t: float, v: Array) -> Array:
        return v[0:1]


class TestMatrixFreeJacobianOperator:
    def test_satisfies_the_protocols(self, bkd: Backend[Array]) -> None:
        producer: JacobianOperatorProtocol[Array, Array, MatrixFree] = (
            MatrixFreeJacobianOperator(LotkaVolterra(bkd), _Y)
        )
        assert isinstance(producer, JacobianOperatorProtocol)
        op = producer.jacobian_operator(_state(bkd), 0.0, _POINT)
        assert isinstance(op, JacobianSnapshotProtocol)
        assert isinstance(op, MatrixFreeJacobianSnapshot)
        assert op.linearization() == _POINT
        assert op.format() == MatrixFree()
        assert (op.domain(), op.range()) == (_Y, _Y)

    def test_agrees_with_the_matrix(self, bkd: Backend[Array]) -> None:
        model = LotkaVolterra(bkd)
        y = _state(bkd)
        free = MatrixFreeJacobianOperator(model, _Y).jacobian_operator(y, 0.0, _POINT)
        dense = DenseJacobianOperator(model, _Y, bkd).jacobian_operator(y, 0.0, _POINT)
        x = _block(bkd, 1)
        bkd.assert_allclose(free.apply(x), dense.apply(x), rtol=1e-14)
        bkd.assert_allclose(
            free.apply_transpose(x), dense.apply_transpose(x), rtol=1e-14
        )

    def test_action_and_transpose_are_consistent(self, bkd: Backend[Array]) -> None:
        r"""The action against a central difference; the transpose against
        :math:`\langle J x, w \rangle = \langle x, J^T w \rangle`."""
        model = LotkaVolterra(bkd)
        op = MatrixFreeJacobianOperator(model, _Y).jacobian_operator(
            _state(bkd), 0.0, _POINT
        )
        x, w = _block(bkd, 2), _block(bkd, 3)
        bkd.assert_allclose(
            op.apply(x), _central_difference(model, bkd, x), rtol=1e-7, atol=1e-8
        )
        bkd.assert_allclose(
            bkd.sum(op.apply(x) * w, axis=0),
            bkd.sum(x * op.apply_transpose(w), axis=0),
            rtol=1e-13,
        )

    def test_refuses_wrong_shapes(self, bkd: Backend[Array]) -> None:
        op = MatrixFreeJacobianOperator(LotkaVolterra(bkd), _Y).jacobian_operator(
            _state(bkd), 0.0, _POINT
        )
        with pytest.raises(ValueError, match="block of shape"):
            op.apply(bkd.zeros((2,)))
        with pytest.raises(ValueError, match="block of shape"):
            op.apply_transpose(bkd.zeros((3, 1)))
        bad = MatrixFreeJacobianOperator(_WrongShapeModel(bkd), _Y).jacobian_operator(
            _state(bkd), 0.0, _POINT
        )
        with pytest.raises(ValueError, match="returned shape"):
            bad.apply(_block(bkd, 4))

    def test_refuses_a_model_without_actions(self, bkd: Backend[Array]) -> None:
        class _MatrixOnly:
            def jacobian(self, y: Array, t: float) -> Array:
                return y

        with pytest.raises(TypeError, match="JacobianActionProtocol"):
            MatrixFreeJacobianOperator(_MatrixOnly(), _Y)  # type: ignore[arg-type]


class TestImplicitStepSolve:
    def test_gmres_solves_the_matrix_free_newton_system(
        self, bkd: Backend[Array]
    ) -> None:
        r""":math:`(I - \Delta t J) x = b` with the matrix-free Jacobian,
        never forming :math:`J`, against a dense solve."""
        dt = 0.1
        model = LotkaVolterra(bkd)
        y = _state(bkd)
        jacobian = MatrixFreeJacobianOperator(model, _Y).jacobian_operator(
            y, 0.0, _POINT
        )
        terms: list[tuple[float, ActionOperatorProtocol[Array, FormatProtocol]]] = [
            (1.0, BackendDenseOperator.identity(_Y, bkd)),
            (-dt, jacobian),
        ]
        newton = ScaledSum(terms, BackendVectorSpace(_Y, bkd))
        b = bkd.asarray([[1.0], [-2.0]])
        x, info = GMRES(BackendVectorSpace(_Y, bkd), bkd).setup(newton).solve(b)
        assert info.converged
        dense = bkd.eye(2, dtype=bkd.default_dtype()) - dt * model.jacobian(y, 0.0)
        bkd.assert_allclose(x, bkd.solve(dense, b), rtol=1e-9)
