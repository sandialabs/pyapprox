"""ScipyCG and ScipyGMRES: SciPy's Krylov methods behind the solver protocol.

They are bound to the NumPy/SciPy stack, so they are tested there only:
the same behavior as the generic solvers (true-residual test, transposed
solves, stored and matrix-free operators) and agreement with them.
"""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import (
    GMRES,
    ActionOperatorProtocol,
    FormatProtocol,
    LinearSolverProtocol,
    MatrixFree,
    OperatorProperties,
    SpaceDescriptor,
    SpaceMismatchError,
    as_matrix_free,
)
from pyapprox.util.linalg.scipy_stack import (
    NumpyVectorSpace,
    ScipyCG,
    ScipyCSROperator,
    ScipyGMRES,
)
from tests._helpers.linalg.stacks import random_matrix, spd_matrix

_BKD = NumpyBkd()
_U = SpaceDescriptor("u", 6)
_SPD = OperatorProperties(symmetric=True, spd=True)


def _operator(
    a: NDArray[np.float64], matrix_free: bool, properties: OperatorProperties
) -> ActionOperatorProtocol[NDArray[np.float64], FormatProtocol]:
    stored = ScipyCSROperator(
        csr_matrix(a), _U, properties=properties, scalar=np.float64
    )
    return as_matrix_free(stored) if matrix_free else stored


def _rhs(seed: int, n: int = 6) -> NDArray[np.float64]:
    rhs: NDArray[np.float64] = np.random.default_rng(seed).normal(size=(n, 1))
    return rhs


def _advection_newton(n: int) -> NDArray[np.float64]:
    """The strongly nonsymmetric Newton matrix of the generic Krylov tests."""
    h, nu, a, dt = 1.0 / (n + 1), 1.0, 500.0, 1e-3
    jacobian = (
        np.diag(-2.0 * nu / h**2 * np.ones(n))
        + np.diag((nu / h**2 + a / (2 * h)) * np.ones(n - 1), -1)
        + np.diag((nu / h**2 - a / (2 * h)) * np.ones(n - 1), 1)
    )
    newton: NDArray[np.float64] = np.eye(n) - dt * jacobian
    return newton


@pytest.mark.parametrize("matrix_free", [False, True], ids=["stored", "free"])
class TestSolves:
    def test_cg_solves_spd_and_transpose(self, matrix_free: bool) -> None:
        a = spd_matrix(6, 0)
        handle = ScipyCG().setup(_operator(a, matrix_free, _SPD))
        b = _rhs(1)
        x, info = handle.solve(b)
        assert info.converged and info.iterations > 0
        _BKD.assert_allclose(a @ x, b, rtol=0.0, atol=1e-9 * np.abs(b).max())
        transposed = handle.transpose()
        assert transposed is not None
        xt, info = transposed.solve(b)
        assert info.converged
        _BKD.assert_allclose(a.T @ xt, b, rtol=0.0, atol=1e-9 * np.abs(b).max())

    @pytest.mark.parametrize("restart", [50, 2], ids=["full", "restarted"])
    def test_gmres_solves_nonsymmetric_and_transpose(
        self, matrix_free: bool, restart: int
    ) -> None:
        a = random_matrix(6, 6, 3) + 4.0 * np.eye(6)
        handle = ScipyGMRES(restart=restart).setup(
            _operator(a, matrix_free, OperatorProperties())
        )
        b = _rhs(4)
        x, info = handle.solve(b)
        assert info.converged
        _BKD.assert_allclose(a @ x, b, rtol=0.0, atol=1e-9 * np.abs(b).max())
        transposed = handle.transpose()
        assert transposed is not None
        xt, info = transposed.solve(b)
        assert info.converged
        _BKD.assert_allclose(a.T @ xt, b, rtol=0.0, atol=1e-9 * np.abs(b).max())


class TestBehavior:
    def test_satisfies_the_protocol_and_pairs_with_matrix_free(self) -> None:
        solver: LinearSolverProtocol[NDArray[np.float64], MatrixFree] = ScipyGMRES()
        assert isinstance(solver, LinearSolverProtocol)
        assert solver.capabilities().transpose

    def test_agrees_with_the_generic_gmres(self) -> None:
        a = random_matrix(6, 6, 5) + 4.0 * np.eye(6)
        op = _operator(a, False, OperatorProperties())
        b = _rhs(6)
        scipy_x, _ = ScipyGMRES().setup(op).solve(b)
        generic_x, _ = GMRES(NumpyVectorSpace(_U, np.float64), _BKD).setup(op).solve(b)
        _BKD.assert_allclose(scipy_x, generic_x, rtol=1e-8)

    def test_cg_reports_a_strongly_nonsymmetric_operator(self) -> None:
        space = SpaceDescriptor("w", 40)
        op = ScipyCSROperator(
            csr_matrix(_advection_newton(40)), space, scalar=np.float64
        )
        _, info = ScipyCG(maxiter=200).setup(op).solve(_rhs(7, 40))
        assert not info.converged and "relative residual" in info.message
        _, info = ScipyGMRES().setup(op).solve(_rhs(7, 40))
        assert info.converged

    def test_maxiter_counts_operator_applications(self) -> None:
        a = random_matrix(6, 6, 8) + 4.0 * np.eye(6)
        handle = ScipyGMRES(maxiter=2, restart=50).setup(
            _operator(a, False, OperatorProperties())
        )
        _, info = handle.solve(_rhs(9))
        assert not info.converged and info.iterations <= 2

    def test_zero_rhs_and_exact_guess(self) -> None:
        a = spd_matrix(6, 10)
        handle = ScipyCG().setup(_operator(a, False, _SPD))
        x, info = handle.solve(np.zeros((6, 1)))
        assert info.converged and info.iterations == 0
        _BKD.assert_allclose(x, np.zeros((6, 1)))
        b = _rhs(11)
        exact = np.asarray(np.linalg.solve(a, b), dtype=np.float64)
        _, info = handle.solve(b, tol=1e-8, x0=exact)
        assert info.converged and info.iterations == 0

    def test_refuses_blocks_mismatches_and_bad_settings(self) -> None:
        handle = ScipyGMRES().setup(_operator(spd_matrix(6, 12), False, _SPD))
        with pytest.raises(ValueError, match="one right-hand side"):
            handle.solve(np.ones((6, 2)))
        rectangular = ScipyCSROperator(
            csr_matrix(random_matrix(3, 6, 13)),
            _U,
            SpaceDescriptor("p", 3),
            scalar=np.float64,
        )
        with pytest.raises(ValueError, match="square"):
            ScipyCG().setup(rectangular)
        swapped = ScipyCSROperator(
            csr_matrix(np.eye(6)), _U, SpaceDescriptor("v", 6), scalar=np.float64
        )
        with pytest.raises(SpaceMismatchError):
            ScipyGMRES().setup(swapped)
        with pytest.raises(ValueError, match="tol"):
            ScipyCG(tol=-1.0)
        with pytest.raises(ValueError, match="restart"):
            ScipyGMRES(restart=0)
