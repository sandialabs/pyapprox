"""Conformance tests for solvers and solver handles on every stack."""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendDenseOperator,
)
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    DiagonalSolver,
    MatrixFree,
    OperatorProperties,
    Product,
    RowReplacedOperator,
    SolveFailedError,
    SolverHandleProtocol,
    SpaceDescriptor,
    SpaceMismatchError,
    as_matrix_free,
)
from pyapprox.util.linalg.scipy_stack import ScipyCSROperator, ScipySparseLU
from tests._helpers.linalg.stacks import (
    A,
    F,
    TypedStackCase,
    V,
    random_matrix,
    spd_matrix,
)

_U = SpaceDescriptor("u", 5)
_V = SpaceDescriptor("v", 5)
_SPD = OperatorProperties(symmetric=True, spd=True)
# Direct solves are exact up to rounding: a relative residual of 1e-10
# leaves four decades for the condition numbers of these 5x5 matrices.
_DIRECT_RTOL = 1e-10


def _well_conditioned(seed: int) -> NDArray[np.float64]:
    """A random nonsymmetric matrix, shifted to keep it well conditioned."""
    return random_matrix(5, 5, seed) + 5.0 * np.eye(5)


def _assert_solves(case: TypedStackCase[V, F, A], matrix: A, x: V, b: V) -> None:
    """``matrix x = b`` to relative residual ``_DIRECT_RTOL``; ``x`` and
    ``b`` are stack vectors, ``matrix`` a backend array."""
    bkd = case.bkd
    rhs = case.array(b)
    residual = case.times(matrix, x) - rhs
    bkd.assert_allclose(
        bkd.reshape(bkd.sum(residual * residual), (1,)),
        bkd.zeros((1,)),
        atol=_DIRECT_RTOL**2 * bkd.to_float(bkd.sum(rhs * rhs)),
    )


class TestDirectSolver:
    def test_solve_and_transpose(self, case: TypedStackCase[V, F, A]) -> None:
        a = _well_conditioned(0)
        handle = case.direct_solver().setup(case.build(a, _U))
        assert isinstance(handle, SolverHandleProtocol)
        assert case.direct_solver().capabilities().transpose
        b = case.vec(_U, 3, 1)
        x, info = handle.solve(b)
        assert info.converged and info.iterations == 0
        _assert_solves(case, case.dense(a), x, b)
        transposed = handle.transpose()
        assert transposed is not None
        xt, info = transposed.solve(b)
        assert info.converged
        _assert_solves(case, case.dense(a).T, xt, b)

    def test_handle_is_the_inverse_operator(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        """apply is the solve, from the operator's range (u) to its domain
        (v); the transposed solve takes vectors of v."""
        a = _well_conditioned(2)
        handle = case.direct_solver().setup(case.build(a, _V, _U))
        assert isinstance(handle, ActionOperatorProtocol)
        assert handle.domain() == _U and handle.range() == _V
        assert handle.format() == MatrixFree()
        b, bt = case.vec(_U, 2, 3), case.vec(_V, 2, 4)
        _assert_solves(case, case.dense(a), handle.apply(b), b)
        _assert_solves(case, case.dense(a).T, handle.apply_transpose(bt), bt)
        case.assert_vector(handle.apply(b), case.array(handle.solve(b)[0]))

    def test_keeps_the_operators_properties(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        handle = case.direct_solver().setup(
            case.build(spd_matrix(5, 4), _U, properties=_SPD)
        )
        assert handle.properties() == _SPD

    def test_composes_as_an_inverse(self, case: TypedStackCase[V, F, A]) -> None:
        """A A^{-1} x = x, with the handle as a product factor."""
        op = case.build(_well_conditioned(5), _U)
        product = Product([op, case.direct_solver().setup(op)])
        x = case.vec(_U, 2, 6)
        case.assert_vector(product.apply(x), case.array(x), rtol=1e-10)
        case.assert_vector(product.apply_transpose(x), case.array(x), rtol=1e-10)

    def test_row_replaced_operator_is_factored(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        """The row-replaced storage is formed at setup; the solution takes
        the right-hand side on the replaced rows."""
        bkd = case.bkd
        a = _well_conditioned(7)
        op = RowReplacedOperator(case.build(a, _U), case.rows(_U, [0, 3]))
        b = case.vec(_U, 1, 8)
        x, info = case.direct_solver().setup(op).solve(b)
        assert info.converged
        case.assert_vector(op.apply(x), case.array(b), rtol=1e-10)
        keep = case.dense(np.array([[0.0], [1.0], [1.0], [0.0], [1.0]]))
        bkd.assert_allclose(
            (1 - keep) * case.array(x), (1 - keep) * case.array(b), atol=1e-12
        )

    def test_singular_matrix_reports_without_raising(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        a = _well_conditioned(9)
        a[2, :] = 0.0
        handle = case.direct_solver().setup(case.build(a, _U))
        _, info = handle.solve(case.vec(_U, 1, 10))
        assert not info.converged
        assert info.message
        with pytest.raises(SolveFailedError):
            handle.apply(case.vec(_U, 1, 11))

    def test_refuses_matrix_free_operators(self, case: TypedStackCase[V, F, A]) -> None:
        with pytest.raises(TypeError, match="got one in MatrixFree"):
            case.direct_solver().setup(as_matrix_free(case.identity(_U)))  # type: ignore[arg-type]

    def test_refuses_non_square(self, case: TypedStackCase[V, F, A]) -> None:
        with pytest.raises(ValueError, match="square"):
            case.direct_solver().setup(
                case.build(random_matrix(3, 5, 12), _U, SpaceDescriptor("p", 3))
            )


class TestFormatPairingAtSetup:
    def test_sparse_lu_refuses_dense_storage(self, numpy_bkd: Backend[Array]) -> None:
        dense = BackendDenseOperator(numpy_bkd.asarray(np.eye(5)), _U, numpy_bkd)
        with pytest.raises(
            TypeError, match="needs an operator in CSRFormat, got one in DenseFormat"
        ):
            ScipySparseLU().setup(dense)  # type: ignore[arg-type]

    def test_dense_lu_refuses_csr_storage(self, numpy_bkd: Backend[Array]) -> None:
        sparse = ScipyCSROperator(csr_matrix(np.eye(5)), _U, scalar=np.float64)
        with pytest.raises(
            TypeError, match="needs an operator in DenseFormat, got one in CSRFormat"
        ):
            BackendDenseLU(numpy_bkd).setup(sparse)  # type: ignore[arg-type]


class TestDiagonalSolver:
    def test_solves_with_the_diagonal(self, case: TypedStackCase[V, F, A]) -> None:
        entries: NDArray[np.float64] = np.arange(1.0, 6.0, dtype=np.float64)[:, None]
        d = case.dense(entries)
        handle = DiagonalSolver(case.space(_U)).setup(
            case.from_diagonal(case.vector(d, _U), _U)
        )
        b = case.vec(_U, 2, 13)
        x, info = handle.solve(b)
        assert info.converged
        case.assert_vector(x, case.array(b) / d, rtol=1e-14)
        transposed = handle.transpose()
        assert transposed is not None
        case.assert_vector(transposed.apply(b), case.array(b) / d, rtol=1e-14)

    @pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
    def test_zero_diagonal_reports_without_raising(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        """The division gives inf, which all_finite turns into a report."""
        d = case.dense(np.array([[1.0], [0.0], [2.0], [3.0], [4.0]]))
        handle = DiagonalSolver(case.space(_U)).setup(
            case.from_diagonal(case.vector(d, _U), _U)
        )
        _, info = handle.solve(case.vec(_U, 1, 14))
        assert not info.converged

    def test_refuses_an_operator_not_declared_diagonal(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        with pytest.raises(TypeError, match="declared diagonal"):
            DiagonalSolver(case.space(_U)).setup(case.build(spd_matrix(5, 15), _U))
        with pytest.raises(SpaceMismatchError):
            DiagonalSolver(case.space(_V)).setup(case.identity(_U))
