"""Conformance tests for solvers and solver handles on every stack."""

from typing import Any

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pyapprox.util.backends.protocols import Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendDenseOperator,
)
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    DiagonalSolver,
    Layout,
    LayoutMismatchError,
    MatrixFree,
    OperatorProperties,
    Product,
    RowReplacedOperator,
    SolveFailedError,
    SolverHandleProtocol,
    as_matrix_free,
)
from pyapprox.util.linalg.scipy_stack import ScipyCSROperator, ScipySparseLU
from tests._helpers.operator_stacks import StackCase, random_matrix, spd_matrix

_U = Layout("u", 5)
_V = Layout("v", 5)
_SPD = OperatorProperties(symmetric=True, spd=True)
# Direct solves are exact up to rounding: a relative residual of 1e-10
# leaves four decades for the condition numbers of these 5x5 matrices.
_DIRECT_RTOL = 1e-10


def _well_conditioned(seed: int) -> np.ndarray[Any, np.dtype[np.float64]]:
    """A random nonsymmetric matrix, shifted to keep it well conditioned."""
    return random_matrix(5, 5, seed) + 5.0 * np.eye(5)


def _assert_solves(case: StackCase, matrix: Any, x: Any, b: Any) -> None:
    bkd = case.bkd
    residual = bkd.dot(matrix, x) - b
    bkd.assert_allclose(
        bkd.reshape(bkd.sum(residual * residual), (1,)),
        bkd.zeros((1,)),
        atol=_DIRECT_RTOL**2 * bkd.to_float(bkd.sum(b * b)),
    )


class TestDirectSolver:
    def test_solve_and_transpose(self, case: StackCase) -> None:
        a = _well_conditioned(0)
        handle = case.direct_solver().setup(case.build(a, _U))
        assert isinstance(handle, SolverHandleProtocol)
        assert case.direct_solver().capabilities().transpose
        b = case.vec(5, 3, 1)
        x, info = handle.solve(b)
        assert info.converged and info.iterations == 0
        _assert_solves(case, case.dense(a), x, b)
        transposed = handle.transpose()
        assert transposed is not None
        xt, info = transposed.solve(b)
        assert info.converged
        _assert_solves(case, case.dense(a).T, xt, b)

    def test_handle_is_the_inverse_operator(self, case: StackCase) -> None:
        """apply is the solve, from the operator's range to its domain."""
        bkd = case.bkd
        a = _well_conditioned(2)
        handle = case.direct_solver().setup(case.build(a, _V, _U))
        assert isinstance(handle, ActionOperatorProtocol)
        assert handle.domain() == _U and handle.range() == _V
        assert handle.format() == MatrixFree()
        b = case.vec(5, 2, 3)
        _assert_solves(case, case.dense(a), handle.apply(b), b)
        _assert_solves(case, case.dense(a).T, handle.apply_transpose(b), b)
        bkd.assert_allclose(handle.apply(b), handle.solve(b)[0])

    def test_keeps_the_operators_properties(self, case: StackCase) -> None:
        handle = case.direct_solver().setup(
            case.build(spd_matrix(5, 4), _U, properties=_SPD)
        )
        assert handle.properties() == _SPD

    def test_composes_as_an_inverse(self, case: StackCase) -> None:
        """A A^{-1} x = x, with the handle as a product factor."""
        bkd = case.bkd
        op = case.build(_well_conditioned(5), _U)
        product = Product([op, case.direct_solver().setup(op)])
        x = case.vec(5, 2, 6)
        bkd.assert_allclose(product.apply(x), x, rtol=1e-10)
        bkd.assert_allclose(product.apply_transpose(x), x, rtol=1e-10)

    def test_row_replaced_operator_is_factored(self, case: StackCase) -> None:
        """The row-replaced storage is formed at setup; the solution takes
        the right-hand side on the replaced rows."""
        bkd = case.bkd
        a = _well_conditioned(7)
        op = RowReplacedOperator(case.build(a, _U), case.rows(_U, [0, 3]))
        b = case.vec(5, 1, 8)
        x, info = case.direct_solver().setup(op).solve(b)
        assert info.converged
        bkd.assert_allclose(op.apply(x), b, rtol=1e-10)
        keep = case.dense(np.array([[0.0], [1.0], [1.0], [0.0], [1.0]]))
        bkd.assert_allclose((1 - keep) * x, (1 - keep) * b, atol=1e-12)

    def test_singular_matrix_reports_without_raising(self, case: StackCase) -> None:
        a = _well_conditioned(9)
        a[2, :] = 0.0
        handle = case.direct_solver().setup(case.build(a, _U))
        _, info = handle.solve(case.vec(5, 1, 10))
        assert not info.converged
        assert info.message
        with pytest.raises(SolveFailedError):
            handle.apply(case.vec(5, 1, 11))

    def test_refuses_matrix_free_operators(self, case: StackCase) -> None:
        with pytest.raises(TypeError, match="got one in MatrixFree"):
            case.direct_solver().setup(as_matrix_free(case.identity(_U)))

    def test_refuses_non_square(self, case: StackCase) -> None:
        with pytest.raises(ValueError, match="square"):
            case.direct_solver().setup(
                case.build(random_matrix(3, 5, 12), _U, Layout("p", 3))
            )


class TestFormatPairingAtSetup:
    def test_sparse_lu_refuses_dense_storage(self, numpy_bkd: Backend[Any]) -> None:
        dense = BackendDenseOperator(numpy_bkd.asarray(np.eye(5)), _U, numpy_bkd)
        with pytest.raises(
            TypeError, match="needs an operator in CSRFormat, got one in DenseFormat"
        ):
            ScipySparseLU().setup(dense)

    def test_dense_lu_refuses_csr_storage(self, numpy_bkd: Backend[Any]) -> None:
        sparse = ScipyCSROperator(csr_matrix(np.eye(5)), _U)
        with pytest.raises(
            TypeError, match="needs an operator in DenseFormat, got one in CSRFormat"
        ):
            BackendDenseLU(numpy_bkd).setup(sparse)


class TestDiagonalSolver:
    def test_solves_with_the_diagonal(self, case: StackCase) -> None:
        bkd = case.bkd
        d = case.dense(np.arange(1.0, 6.0)[:, None])
        handle = DiagonalSolver(case.space(_U)).setup(case.from_diagonal(d, _U))
        b = case.vec(5, 2, 13)
        x, info = handle.solve(b)
        assert info.converged
        bkd.assert_allclose(x, b / d, rtol=1e-14)
        transposed = handle.transpose()
        assert transposed is not None
        bkd.assert_allclose(transposed.apply(b), b / d, rtol=1e-14)

    @pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
    def test_zero_diagonal_reports_without_raising(self, case: StackCase) -> None:
        """The division gives inf, which all_finite turns into a report."""
        d = case.dense(np.array([[1.0], [0.0], [2.0], [3.0], [4.0]]))
        handle = DiagonalSolver(case.space(_U)).setup(case.from_diagonal(d, _U))
        _, info = handle.solve(case.vec(5, 1, 14))
        assert not info.converged

    def test_refuses_an_operator_not_declared_diagonal(self, case: StackCase) -> None:
        with pytest.raises(TypeError, match="declared diagonal"):
            DiagonalSolver(case.space(_U)).setup(case.build(spd_matrix(5, 15), _U))
        with pytest.raises(LayoutMismatchError):
            DiagonalSolver(case.space(_V)).setup(case.identity(_U))


class TestBackendGraph:
    def test_solve_preserves_the_graph(self, torch_bkd: Backend[Any]) -> None:
        r"""d(y^T A^{-1} b)/db = A^{-T} y through autograd."""
        import torch

        bkd = torch_bkd
        a = _well_conditioned(16)
        handle = BackendDenseLU(bkd).setup(
            BackendDenseOperator(bkd.asarray(a), _U, bkd)
        )
        rng = np.random.default_rng(17)
        b = torch.tensor(rng.normal(size=(5, 1)), requires_grad=True)
        y = torch.tensor(rng.normal(size=(5, 1)))
        (grad,) = torch.autograd.grad(torch.sum(y * handle.apply(b)), b)
        bkd.assert_allclose(grad, handle.apply_transpose(y), rtol=1e-10)
