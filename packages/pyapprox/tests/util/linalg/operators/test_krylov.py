"""Conformance tests for the Krylov solvers on every stack.

Conjugate gradients on SPD operators, GMRES on nonsymmetric ones, both on
stored and matrix-free operators; and the true-residual test that turns a
solve conjugate gradients cannot do (a strongly nonsymmetric operator)
into ``converged=False``.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import (
    GMRES,
    ActionOperatorProtocol,
    ConjugateGradient,
    FormatProtocol,
    OperatorProperties,
    SolverHandleProtocol,
    SpaceDescriptor,
    SpaceMismatchError,
    as_matrix_free,
)
from pyapprox.util.linalg.scipy_stack import NumpyVectorSpace, ScipyCSROperator
from tests._helpers.linalg.stacks import (
    A,
    F,
    TypedStackCase,
    V,
    random_matrix,
    spd_matrix,
)

_U = SpaceDescriptor("u", 6)
_SPD = OperatorProperties(symmetric=True, spd=True)


def _nonsymmetric(seed: int, n: int = 6) -> NDArray[np.float64]:
    """Nonsymmetric, shifted to keep it well conditioned."""
    return random_matrix(n, n, seed) + 4.0 * np.eye(n)


def _advection_newton(n: int) -> NDArray[np.float64]:
    r""":math:`I - \Delta t J` for 1D advection-diffusion, central
    differences, Peclet 500 and :math:`\Delta t = 10^{-3}`: its skew part
    exceeds its symmetric part, which defeats conjugate gradients."""
    h, nu, a, dt = 1.0 / (n + 1), 1.0, 500.0, 1e-3
    jacobian = (
        np.diag(-2.0 * nu / h**2 * np.ones(n))
        + np.diag((nu / h**2 + a / (2 * h)) * np.ones(n - 1), -1)
        + np.diag((nu / h**2 - a / (2 * h)) * np.ones(n - 1), 1)
    )
    newton: NDArray[np.float64] = np.eye(n) - dt * jacobian
    return newton


def _assert_solves(
    case: TypedStackCase[V, F, A], matrix: A, x: V, b: V, rtol: float = 1e-9
) -> None:
    """``matrix x = b`` to ``rtol`` relative to the largest entry of ``b``,
    the scale the solvers' norm-relative tolerance works at."""
    bkd, rhs = case.bkd, case.array(b)
    scale = bkd.to_float(bkd.max(bkd.abs(rhs)))
    bkd.assert_allclose(case.times(matrix, x), rhs, rtol=0.0, atol=rtol * scale)


class TestConjugateGradient:
    @pytest.mark.parametrize("matrix_free", [False, True], ids=["stored", "free"])
    def test_solves_spd_and_transpose(
        self, case: TypedStackCase[V, F, A], matrix_free: bool
    ) -> None:
        a = spd_matrix(6, 0)
        stored = case.build(a, _U, properties=_SPD)
        op: ActionOperatorProtocol[V, FormatProtocol] = (
            as_matrix_free(stored) if matrix_free else stored
        )
        handle = ConjugateGradient(case.space(_U), case.bkd).setup(op)
        assert isinstance(handle, SolverHandleProtocol)
        b = case.vec(_U, 1, 1)
        x, info = handle.solve(b)
        assert info.converged and 0 < info.iterations <= 12
        assert info.residual_norm is not None
        _assert_solves(case, case.dense(a), x, b)
        transposed = handle.transpose()
        assert transposed is not None
        xt, info = transposed.solve(b)
        assert info.converged
        _assert_solves(case, case.dense(a).T, xt, b)

    def test_reports_a_strongly_nonsymmetric_operator(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        """The recurrence drifts; the true residual exposes it."""
        space = SpaceDescriptor("w", 40)
        op = case.build(_advection_newton(40), space)
        handle = ConjugateGradient(case.space(space), case.bkd, maxiter=200).setup(op)
        _, info = handle.solve(case.vec(space, 1, 2))
        assert not info.converged
        assert "relative residual" in info.message


class TestGMRES:
    @pytest.mark.parametrize("matrix_free", [False, True], ids=["stored", "free"])
    @pytest.mark.parametrize("restart", [50, 2], ids=["full", "restarted"])
    def test_solves_nonsymmetric_and_transpose(
        self, case: TypedStackCase[V, F, A], matrix_free: bool, restart: int
    ) -> None:
        a = _nonsymmetric(3)
        stored = case.build(a, _U)
        op: ActionOperatorProtocol[V, FormatProtocol] = (
            as_matrix_free(stored) if matrix_free else stored
        )
        handle = GMRES(case.space(_U), case.bkd, restart=restart).setup(op)
        b = case.vec(_U, 1, 4)
        x, info = handle.solve(b)
        assert info.converged
        _assert_solves(case, case.dense(a), x, b)
        transposed = handle.transpose()
        assert transposed is not None
        xt, info = transposed.solve(b)
        assert info.converged
        _assert_solves(case, case.dense(a).T, xt, b)

    def test_solves_what_conjugate_gradients_cannot(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        space = SpaceDescriptor("w", 40)
        a = _advection_newton(40)
        handle = GMRES(case.space(space), case.bkd).setup(case.build(a, space))
        b = case.vec(space, 1, 5)
        x, info = handle.solve(b)
        assert info.converged
        _assert_solves(case, case.dense(a), x, b, rtol=1e-8)

    def test_reports_too_few_iterations(self, case: TypedStackCase[V, F, A]) -> None:
        handle = GMRES(case.space(_U), case.bkd, maxiter=1).setup(
            case.build(_nonsymmetric(6), _U)
        )
        _, info = handle.solve(case.vec(_U, 1, 7))
        assert not info.converged and info.iterations == 1


@pytest.mark.parametrize("solver", [ConjugateGradient, GMRES], ids=["cg", "gmres"])
class TestShared:
    def test_zero_rhs_and_exact_guess(
        self,
        case: TypedStackCase[V, F, A],
        solver: type[ConjugateGradient[V, A]] | type[GMRES[V, A]],
    ) -> None:
        a = spd_matrix(6, 8)
        op = case.build(a, _U, properties=_SPD)
        handle = solver(case.space(_U), case.bkd).setup(op)
        x, info = handle.solve(case.vector(case.bkd.zeros((6, 1)), _U))
        assert info.converged and info.iterations == 0
        case.assert_vector(x, case.bkd.zeros((6, 1)))
        b = case.vec(_U, 1, 9)
        exact, _ = handle.solve(b)
        _, info = handle.solve(b, tol=1e-8, x0=exact)
        assert info.converged and info.iterations == 0

    def test_solves_a_block_column_by_column(
        self,
        case: TypedStackCase[V, F, A],
        solver: type[ConjugateGradient[V, A]] | type[GMRES[V, A]],
    ) -> None:
        a = spd_matrix(6, 13)
        handle = solver(case.space(_U), case.bkd).setup(
            case.build(a, _U, properties=_SPD)
        )
        b = case.vec(_U, 3, 14)
        x, info = handle.solve(b)
        assert info.converged and info.residual_norm is not None
        _assert_solves(case, case.dense(a), x, b)
        one, single = handle.solve(case.space(_U).column(b, 1))
        assert info.iterations >= single.iterations > 0
        _assert_solves(case, case.dense(a), one, case.space(_U).column(b, 1))
        # Each column starts from its own column of x0.
        _, info = handle.solve(b, x0=x)
        assert info.converged and info.iterations == 0
        with pytest.raises(ValueError, match="3 columns"):
            handle.solve(b, x0=case.vec(_U, 2, 15))

    def test_reports_the_failing_column(
        self,
        case: TypedStackCase[V, F, A],
        solver: type[ConjugateGradient[V, A]] | type[GMRES[V, A]],
    ) -> None:
        a = spd_matrix(6, 16)
        handle = solver(case.space(_U), case.bkd, maxiter=1).setup(
            case.build(a, _U, properties=_SPD)
        )
        space = case.space(_U)
        b = case.vec(_U, 1, 17)
        block = space.concatenate([space.zeros(1), b])
        _, info = handle.solve(block)
        assert not info.converged
        assert info.message.startswith("column 1: relative residual")

    def test_refuses_mismatched_spaces(
        self,
        case: TypedStackCase[V, F, A],
        solver: type[ConjugateGradient[V, A]] | type[GMRES[V, A]],
    ) -> None:
        op = case.build(spd_matrix(6, 10), _U, properties=_SPD)
        with pytest.raises(SpaceMismatchError):
            solver(case.space(SpaceDescriptor("v", 6)), case.bkd).setup(op)
        with pytest.raises(ValueError, match="square"):
            solver(case.space(_U), case.bkd).setup(
                case.build(random_matrix(3, 6, 12), _U, SpaceDescriptor("p", 3))
            )


class TestSettings:
    def test_refuses_bad_settings(self) -> None:
        bkd = NumpyBkd()
        space = NumpyVectorSpace(_U, np.float64)
        with pytest.raises(ValueError, match="tol"):
            ConjugateGradient(space, bkd, tol=0.0)
        with pytest.raises(ValueError, match="maxiter"):
            GMRES(space, bkd, maxiter=0)
        with pytest.raises(ValueError, match="restart"):
            GMRES(space, bkd, restart=0)
        with pytest.raises(TypeError, match="InnerProductSpaceProtocol"):
            GMRES(bkd, bkd)  # type: ignore[arg-type]

    def test_refuses_complex_scalars(self) -> None:
        bkd = NumpyBkd()
        complex_space = SpaceDescriptor("z", 2, dtype="complex128")
        space = NumpyVectorSpace(complex_space, np.complex128)
        op = ScipyCSROperator.identity(complex_space, np.complex128)
        with pytest.raises(TypeError, match="real scalars"):
            ConjugateGradient(space, bkd).setup(op)
        with pytest.raises(TypeError, match="real scalars"):
            GMRES(space, bkd).setup(op)
