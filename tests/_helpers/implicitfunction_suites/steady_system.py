r"""Stack-generic suites: the steady problem :math:`R(u; p) = 0` solved by
Newton on the stack.

The pinned quadratic of ``tests._helpers.implicitfunction_stack_cases``
(:math:`F_i = (A u)_i + p_0 u_i^2 + p_1^2 c_i`, rows :math:`\{0, 2\}`
pinned to :math:`(p_0^2 + 1, p_0 p_1)`, :math:`p = (0.7, -0.4)`), the
stack's direct solver counting its setups, under ``KeepAll``, Newton with
``atol`` 1e-13 from zero:

- Nonlinear: the solve converges, :math:`\|R(u^*)\| < 10^{-12}`, the
  pinned rows hold the data; each Newton step sets up a new operator and
  the solved point one more, kept for repeated requests; its handle
  solves with :math:`R_u(u^*)`.
- Affine (:math:`p_0 = 0`, constant Jacobian): one step, one setup for
  the solve, the solved point and a second solve; after ``set_param`` and
  ``clear``, a new one.
- Two systems sharing one strategy (an unpinned and a pinned residual,
  both affine): neither is given the other's setup.
- Refusals: the handle of an unconverged solve or of one whose
  parameters changed since; a residual without a Jacobian, or of the
  wrong type, and a space without an inner product.
"""

import pytest

from pyapprox.implicitfunction.residual_derivatives import ResidualDerivatives
from pyapprox.implicitfunction.setups import KeepAll, RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.util.linalg.operators import LinearizationPoint
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.implicitfunction_stack_cases import (
    QUADRATIC_PARAM,
    STATES,
    QuadraticResidual,
    pinned_quadratic,
)
from tests._helpers.linalg.counting_solver import CountingSolver
from tests._helpers.linalg.stack_case import A, F, TypedStackCase, V


def _newton(maxiters: int = 20) -> Newton[V, A]:
    return Newton(atol=1e-13, rtol=0.0, maxiters=maxiters)


def _norm(f: QuadraticResidual[V, F, A], x: V) -> float:
    return f.bkd.to_float(f.states.norm(x)[0])


class NonlinearSolveSuite:
    def test_it_converges_to_the_pinned_solution(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        f, pinned = pinned_quadratic(case)
        lu = CountingSolver(case.direct_solver())
        system = SteadySystem(pinned, f.states, KeepAll(lu))
        assert not system.is_affine()
        point = system.solve(_newton(), f.states.zeros(1))
        assert point.converged() and point.parameters_version() == 0
        state = case.array(point.state())
        assert _norm(f, pinned(point.state())) < 1e-12
        p0, p1 = QUADRATIC_PARAM
        case.bkd.assert_allclose(state[0, 0], case.bkd.asarray(p0**2 + 1.0), atol=1e-14)
        case.bkd.assert_allclose(state[2, 0], case.bkd.asarray(p0 * p1), atol=1e-14)
        iterations = point.info().iterations
        assert iterations > 1 and lu.setups == iterations

    def test_the_handle_solves_with_the_jacobian_there(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        f, pinned = pinned_quadratic(case)
        lu = CountingSolver(case.direct_solver())
        system = SteadySystem(pinned, f.states, KeepAll(lu))
        point = system.solve(_newton(), f.states.zeros(1))
        before = lu.setups
        handle = point.jacobian_handle()
        assert point.jacobian_handle() is handle and lu.setups == before + 1
        rhs = case.bkd.asarray([[1.0], [-2.0], [0.5]])
        x, info = handle.solve(case.vector(rhs, STATES))
        assert info.converged
        jacobian = pinned.derivatives().jacobian
        assert jacobian is not None
        applied = jacobian(point.state(), LinearizationPoint(0, ("check",))).apply(x)
        case.assert_vector(applied, rhs, rtol=1e-12)


class AffineSolveSuite:
    def test_one_setup_until_the_parameters_change(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        f, pinned = pinned_quadratic(case, affine=True)
        lu = CountingSolver(case.direct_solver())
        setups: KeepAll[V, F] = KeepAll(lu)
        system = SteadySystem(pinned, f.states, setups)
        assert system.is_affine()
        point = system.solve(_newton(), f.states.zeros(1))
        assert point.converged() and point.info().iterations == 1
        assert _norm(f, pinned(point.state())) < 1e-12
        point.jacobian_handle()
        ones = case.vector(case.bkd.ones((3, 1)), STATES)
        system.solve(_newton(), ones)
        assert lu.setups == 1
        f.set_param(case.bkd.asarray([0.0, 0.3]))
        setups.clear()
        moved = system.solve(_newton(), f.states.zeros(1))
        assert moved.parameters_version() == 1 and lu.setups == 2
        assert _norm(f, pinned(moved.state())) < 1e-12


class SharedStrategySuite:
    def test_systems_sharing_a_strategy_keep_their_own_setups(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        """An unpinned and a pinned system, both with constant Jacobians,
        set up through one strategy: neither is given the other's
        factorization."""
        f, pinned = pinned_quadratic(case, affine=True)
        lu = CountingSolver(case.direct_solver())
        shared: KeepAll[V, F] = KeepAll(lu)
        unpinned_point = SteadySystem(f, f.states, shared).solve(
            _newton(), f.states.zeros(1)
        )
        pinned_point = SteadySystem(pinned, f.states, shared).solve(
            _newton(), f.states.zeros(1)
        )
        assert _norm(f, f(unpinned_point.state())) < 1e-12
        assert _norm(f, pinned(pinned_point.state())) < 1e-12
        assert lu.setups == 2


class RefusalsSuite:
    def test_an_unconverged_solve_has_no_handle(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        f, pinned = pinned_quadratic(case)
        system = SteadySystem(pinned, f.states, RecomputeAlways(case.direct_solver()))
        point = system.solve(_newton(maxiters=1), f.states.zeros(1))
        assert not point.converged()
        with pytest.raises(ValueError, match="did not converge"):
            point.jacobian_handle()

    def test_a_stale_solve_has_no_handle(self, case: TypedStackCase[V, F, A]) -> None:
        f, pinned = pinned_quadratic(case)
        system = SteadySystem(pinned, f.states, RecomputeAlways(case.direct_solver()))
        point = system.solve(_newton(), f.states.zeros(1))
        f.set_param(f.p[:, 0])
        with pytest.raises(ValueError, match="changed"):
            point.jacobian_handle()

    def test_wrong_inputs(self, case: TypedStackCase[V, F, A]) -> None:
        f: QuadraticResidual[V, F, A] = QuadraticResidual(case)
        setups: RecomputeAlways[V, F] = RecomputeAlways(case.direct_solver())
        with pytest.raises(TypeError, match="ResidualProtocol"):
            SteadySystem(3.0, f.states, setups)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="InnerProductSpaceProtocol"):
            SteadySystem(f, 3.0, setups)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="SetupStrategyProtocol"):
            SteadySystem(f, f.states, case.direct_solver())  # type: ignore[arg-type]

        class _NoJacobian(QuadraticResidual[V, F, A]):
            def derivatives(self) -> ResidualDerivatives[V, F]:
                return ResidualDerivatives.none()

        with pytest.raises(TypeError, match="no jacobian"):
            SteadySystem(_NoJacobian(case), f.states, setups)
