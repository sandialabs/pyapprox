"""The mock stack: opaque vectors, its own format, and generic code on it.

The generic combinators, constraints, solvers and cache run here on
vectors that cannot be indexed or converted, so any hidden reliance on
array storage fails.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    CachingSolver,
    DiagonalCapableProtocol,
    DiagonalSolver,
    LinearizationPoint,
    LinearizedOperator,
    LinearSolverProtocol,
    Product,
    RowReplacedOperator,
    RowSetProtocol,
    ScaledSum,
    SolverHandleProtocol,
    SpaceDescriptor,
    SpaceMismatchError,
    Transposed,
    VectorSpaceProtocol,
)
from pyapprox.util.linalg.scipy_stack import ScipyCSROperator, ScipySparseLU
from tests._helpers.linalg.mock_stack import (
    MockLU,
    MockOperator,
    MockRowSet,
    MockVector,
    MockVectorSpace,
    peek,
    wrap,
)
from tests._helpers.linalg.stacks import random_matrix, spd_matrix

_BKD = NumpyBkd()
_U = SpaceDescriptor("u", 5)


def _vec(seed: int, ncols: int = 1, space: SpaceDescriptor = _U) -> MockVector:
    return wrap(np.random.default_rng(seed).normal(size=(space.size, ncols)), space)


class TestOpacity:
    def test_vectors_refuse_inspection(self) -> None:
        x = _vec(0)
        # Each of these is a type error on purpose: the runtime must refuse
        # it too.
        with pytest.raises(TypeError):
            x[0]
        with pytest.raises(TypeError):
            x @ x
        with pytest.raises(TypeError):
            x + x
        with pytest.raises(TypeError, match="opaque"):
            np.asarray(x)
        assert not hasattr(x, "shape")

    def test_operations_check_the_space(self) -> None:
        """A plain array cannot say which space it belongs to; a mock
        vector can, so every operation checks it."""
        space = MockVectorSpace(_U)
        other = _vec(1, space=SpaceDescriptor("v", 5))
        stale = _vec(2, space=SpaceDescriptor("u", 5, version=1))
        with pytest.raises(SpaceMismatchError, match="name"):
            space.norm(other)
        with pytest.raises(SpaceMismatchError, match="version"):
            space.axpby(1.0, _vec(3), 1.0, stale)
        op = MockOperator.from_dense(spd_matrix(5, 4), _U)
        with pytest.raises(SpaceMismatchError):
            op.apply(other)


class TestProtocols:
    def test_mock_classes_satisfy_the_protocols(self) -> None:
        op = MockOperator.from_dense(spd_matrix(5, 5), _U)
        assert isinstance(MockVectorSpace(_U), VectorSpaceProtocol)
        assert isinstance(MockRowSet(_U, [0]), RowSetProtocol)
        assert isinstance(op, ActionOperatorProtocol)
        assert isinstance(op, DiagonalCapableProtocol)
        assert isinstance(MockLU(), LinearSolverProtocol)
        assert isinstance(MockLU().setup(op), SolverHandleProtocol)

    def test_solvers_refuse_each_others_formats(self) -> None:
        mock = MockOperator.from_dense(spd_matrix(5, 6), _U)
        csr = ScipyCSROperator(csr_matrix(spd_matrix(5, 6)), _U)
        with pytest.raises(
            TypeError, match="needs an operator in CSRFormat, got one in MockFormat"
        ):
            ScipySparseLU().setup(mock)
        with pytest.raises(
            TypeError, match="needs an operator in MockFormat, got one in CSRFormat"
        ):
            MockLU().setup(csr)


class TestGenericCodeOnTheMock:
    def test_space_writes_into_out(self) -> None:
        space = MockVectorSpace(_U)
        x, y = _vec(7), _vec(8)
        expected = 2.0 * peek(x) - peek(y)
        assert space.axpby(2.0, x, -1.0, y, out=y) is y
        _BKD.assert_allclose(peek(y), expected)
        with pytest.raises(ValueError, match="alias"):
            space.axpby(1.0, x, 1.0, y, out=x)

    def test_constrained_sum_factored_and_solved(self) -> None:
        """(M + 0.5 K) with rows {0, 4} replaced, factored by MockLU: the
        shape of an implicit step with essential rows."""
        m, k = spd_matrix(5, 9), random_matrix(5, 5, 10) + 5.0 * np.eye(5)
        space = MockVectorSpace(_U)
        newton = ScaledSum(
            [
                (1.0, MockOperator.from_dense(m, _U)),
                (0.5, MockOperator.from_dense(k, _U)),
            ],
            space,
        )
        constrained = RowReplacedOperator(newton, MockRowSet(_U, [0, 4]))
        reference = m + 0.5 * k
        reference[[0, 4], :] = 0.0
        reference[[0, 4], [0, 4]] = 1.0
        _BKD.assert_allclose(constrained.format().dense(), reference, rtol=1e-14)
        b = _vec(11)
        handle = MockLU().setup(constrained)
        x, info = handle.solve(b)
        assert info.converged
        _BKD.assert_allclose(reference @ peek(x), peek(b), rtol=1e-10)
        xt = handle.apply_transpose(b)
        _BKD.assert_allclose(reference.T @ peek(xt), peek(b), rtol=1e-10)

    def test_handle_inside_a_product(self) -> None:
        op = MockOperator.from_dense(random_matrix(5, 5, 12) + 5.0 * np.eye(5), _U)
        product = Product([op, MockLU().setup(op)])
        x = _vec(13, ncols=2)
        _BKD.assert_allclose(peek(product.apply(x)), peek(x), rtol=1e-10)
        _BKD.assert_allclose(peek(Transposed(product).apply(x)), peek(x), rtol=1e-10)

    def test_cache_and_diagonal_solver(self) -> None:
        cache = CachingSolver(MockLU(), capacity=1)
        op = MockOperator.from_dense(spd_matrix(5, 14), _U)
        for _ in range(3):
            cache.setup(LinearizedOperator(op, LinearizationPoint(0)))
        assert (cache.setups(), cache.hits()) == (1, 2)
        d = wrap(np.arange(1.0, 6.0)[:, None], _U)
        handle = DiagonalSolver(MockVectorSpace(_U)).setup(
            MockOperator.from_diagonal(d, _U)
        )
        b = _vec(15)
        _BKD.assert_allclose(peek(handle.apply(b)), peek(b) / peek(d), rtol=1e-14)
