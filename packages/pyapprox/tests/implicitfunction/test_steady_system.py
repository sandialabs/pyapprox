r"""The steady problem :math:`R(u; p) = 0` solved by Newton on the stack.

The pinned quadratic of ``tests._helpers.implicitfunction_cases``
(:math:`F_i = (A u)_i + p_0 u_i^2 + p_1^2 c_i`, rows :math:`\{0, 2\}`
pinned to :math:`(p_0^2 + 1, p_0 p_1)`, :math:`p = (0.7, -0.4)`), dense
LU through the backend, Newton with ``atol`` 1e-13 from zero:

- Nonlinear: the solve converges, :math:`\|R(u^*)\| < 10^{-12}`, the
  pinned rows hold the data; each Newton step sets up a new operator and
  the solved point one more, kept for repeated requests; its handle
  solves with :math:`R_u(u^*)`.
- Affine (:math:`p_0 = 0`, constant Jacobian): one step, one setup for
  the solve, the solved point and a second solve; a new one after
  ``set_param``.
- Refusals: the handle of an unconverged solve or of one whose
  parameters changed since; a residual without a Jacobian, or of the
  wrong type, and a space without an inner product.
"""

import pytest

from pyapprox.implicitfunction.residual_derivatives import ResidualDerivatives
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseLU, DenseFormat
from pyapprox.util.linalg.operators import CachingSolver, LinearizationPoint
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.implicitfunction_cases import (
    QUADRATIC_PARAM,
    QuadraticResidual,
    pinned_quadratic,
)
from tests._helpers.linalg.counting_solver import CountingSolver


def _counting_lu(
    bkd: Backend[Array],
) -> CountingSolver[Array, DenseFormat[Array]]:
    return CountingSolver(BackendDenseLU(bkd))


def _newton(maxiters: int = 20) -> Newton[Array, Array]:
    return Newton(atol=1e-13, rtol=0.0, maxiters=maxiters)


class TestANonlinearSolve:
    def test_it_converges_to_the_pinned_solution(self, bkd: Backend[Array]) -> None:
        f, pinned = pinned_quadratic(bkd)
        lu = _counting_lu(bkd)
        system = SteadySystem(pinned, f.states, lu)
        assert not system.is_affine()
        point = system.solve(_newton(), bkd.zeros((3, 1)))
        assert point.converged() and point.parameters_version() == 0
        state = point.state()
        assert bkd.to_float(bkd.norm(pinned(state))) < 1e-12
        p0, p1 = QUADRATIC_PARAM
        bkd.assert_allclose(state[0, 0], bkd.asarray(p0**2 + 1.0), atol=1e-14)
        bkd.assert_allclose(state[2, 0], bkd.asarray(p0 * p1), atol=1e-14)
        iterations = point.info().iterations
        assert iterations > 1 and lu.setups == iterations

    def test_the_handle_solves_with_the_jacobian_there(
        self, bkd: Backend[Array]
    ) -> None:
        f, pinned = pinned_quadratic(bkd)
        lu = _counting_lu(bkd)
        point = SteadySystem(pinned, f.states, lu).solve(_newton(), bkd.zeros((3, 1)))
        before = lu.setups
        handle = point.jacobian_handle()
        assert point.jacobian_handle() is handle and lu.setups == before + 1
        rhs = bkd.asarray([[1.0], [-2.0], [0.5]])
        x, info = handle.solve(rhs)
        assert info.converged
        jacobian = pinned.derivatives().jacobian
        assert jacobian is not None
        applied = jacobian(point.state(), LinearizationPoint(0, ("check",))).apply(x)
        bkd.assert_allclose(applied, rhs, rtol=1e-12)


class TestAnAffineSolve:
    def test_one_setup_until_the_parameters_change(self, bkd: Backend[Array]) -> None:
        f, pinned = pinned_quadratic(bkd, affine=True)
        lu = _counting_lu(bkd)
        system = SteadySystem(pinned, f.states, lu)
        assert system.is_affine()
        point = system.solve(_newton(), bkd.zeros((3, 1)))
        assert point.converged() and point.info().iterations == 1
        assert bkd.to_float(bkd.norm(pinned(point.state()))) < 1e-12
        point.jacobian_handle()
        system.solve(_newton(), bkd.ones((3, 1)))
        assert lu.setups == 1
        f.set_param(bkd.reshape(bkd.asarray([0.0, 0.3]), (2, 1)))
        moved = system.solve(_newton(), bkd.zeros((3, 1)))
        assert moved.parameters_version() == 1 and lu.setups == 2
        assert bkd.to_float(bkd.norm(pinned(moved.state()))) < 1e-12


class TestASharedCache:
    def test_systems_sharing_a_cache_keep_their_own_setups(
        self, bkd: Backend[Array]
    ) -> None:
        """An unpinned and a pinned system, both with constant Jacobians at
        one parameter version, set up through one caching solver: neither
        is given the other's factorization."""
        f, pinned = pinned_quadratic(bkd, affine=True)
        lu = _counting_lu(bkd)
        shared = CachingSolver(lu, capacity=4)
        unpinned_point = SteadySystem(f, f.states, shared).solve(
            _newton(), bkd.zeros((3, 1))
        )
        pinned_point = SteadySystem(pinned, f.states, shared).solve(
            _newton(), bkd.zeros((3, 1))
        )
        assert bkd.to_float(bkd.norm(f(unpinned_point.state()))) < 1e-12
        assert bkd.to_float(bkd.norm(pinned(pinned_point.state()))) < 1e-12
        assert lu.setups == 2


class TestRefusals:
    def test_an_unconverged_solve_has_no_handle(self, bkd: Backend[Array]) -> None:
        f, pinned = pinned_quadratic(bkd)
        point = SteadySystem(pinned, f.states, BackendDenseLU(bkd)).solve(
            _newton(maxiters=1), bkd.zeros((3, 1))
        )
        assert not point.converged()
        with pytest.raises(ValueError, match="did not converge"):
            point.jacobian_handle()

    def test_a_stale_solve_has_no_handle(self, bkd: Backend[Array]) -> None:
        f, pinned = pinned_quadratic(bkd)
        point = SteadySystem(pinned, f.states, BackendDenseLU(bkd)).solve(
            _newton(), bkd.zeros((3, 1))
        )
        f.set_param(f.p)
        with pytest.raises(ValueError, match="changed"):
            point.jacobian_handle()

    def test_wrong_inputs(self, bkd: Backend[Array]) -> None:
        f: QuadraticResidual[Array] = QuadraticResidual(bkd)
        lu: BackendDenseLU[Array] = BackendDenseLU(bkd)
        with pytest.raises(TypeError, match="ResidualProtocol"):
            SteadySystem(3.0, f.states, lu)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="InnerProductSpaceProtocol"):
            SteadySystem(f, 3.0, lu)  # type: ignore[arg-type]

        class _NoJacobian(QuadraticResidual[Array]):
            def derivatives(self) -> ResidualDerivatives[Array, DenseFormat[Array]]:
                return ResidualDerivatives.none()

        with pytest.raises(TypeError, match="no jacobian"):
            SteadySystem(_NoJacobian(bkd), f.states, lu)
