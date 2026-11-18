r"""Stack-generic suites: the adjoint gradient and Hessian-vector product
of a steady functional.

The pinned quadratic of ``tests._helpers.implicitfunction_stack_cases``
(:math:`F_i = (A u)_i + p_0 u_i^2 + p_1^2 c_i`, rows :math:`\{0, 2\}`
pinned to :math:`(p_0^2 + 1, p_0 p_1)`, :math:`p = (0.7, -0.4)`), solved
by Newton (``atol`` 1e-13) with the stack's direct solver, and :math:`Q =
g + r` with :math:`g = \tfrac12 u^\top D u + p_1 u_1`, :math:`D =
\mathrm{diag}(1, 2, 3)`, :math:`r = \tfrac{3}{20}\|p\|^2`:

- The gradient is the derivative of :math:`p \mapsto Q(u^*(p); p)` and
  the HVP the derivative of the gradient, by central finite differences
  (minimum relative error below 1e-6); the Hessian is symmetric to 1e-12.
- Wrong-model control: the direct :math:`Q_p` alone, without the
  adjoint's :math:`R_p^\top \lambda`, misses the gradient by more than
  1e-2 relative.
- Setups, cumulative after a solve of k Newton steps, the adjoint, two
  HVPs and a tangent: ``KeepAll`` k + 1 (every derivative solve at the
  point shares one setup), ``RecomputeAlways`` and ``KeepStateFree``
  k + 6. Affine, a solve, the adjoint and an HVP: ``KeepAll`` and
  ``KeepStateFree`` 1, ``RecomputeAlways`` 4. The helper clears its
  strategy whenever it sets the parameters.
- Levels: first-order bundles give no Hessian; a functional of the
  parameters alone needs no adjoint solve.
- Refusals: a residual without a parameter Jacobian, a term without
  derivatives, a stale adjoint, a multi-column direction; arguments of
  the wrong type, a point among them.
"""

from typing import Callable, Generic, List, Tuple

import pytest

from pyapprox.implicitfunction.functional_terms import ParameterTermDerivatives
from pyapprox.implicitfunction.setups import (
    KeepAll,
    KeepStateFree,
    RecomputeAlways,
    SetupStrategyProtocol,
    SteadyRequest,
)
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_derivatives import SteadyDerivatives
from pyapprox.implicitfunction.steady_system import SteadyPoint, SteadySystem
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.implicitfunction_stack_cases import (
    QUADRATIC_PARAM,
    QuadraticParameterTerm,
    QuadraticResidual,
    QuadraticStateTerm,
    pinned_quadratic,
    quadratic_functional,
)
from tests._helpers.implicitfunction_suites.checks import min_fd_error
from tests._helpers.linalg.counting_solver import CountingSolver
from tests._helpers.linalg.stack_case import A, F, TypedStackCase, V


class _Solved(Generic[V, F, A]):
    """The pinned quadratic, its system and the functional's derivatives,
    on one stack; gradients and Hessians returned as backend arrays."""

    def __init__(
        self,
        case: TypedStackCase[V, F, A],
        residual_order: int = 2,
        functional_order: int = 2,
        affine: bool = False,
        strategy: Callable[
            [CountingSolver[V, F]],
            SetupStrategyProtocol[SteadyRequest[V, F], V, F],
        ] = KeepAll,
    ) -> None:
        self.case, self.bkd = case, case.bkd
        self.f, self.pinned = pinned_quadratic(case, residual_order, affine)
        self.lu: CountingSolver[V, F] = CountingSolver(case.direct_solver())
        self.setups = strategy(self.lu)
        self.system = SteadySystem(self.pinned, self.f.states, self.setups)
        self.functional = quadratic_functional(self.f, functional_order)
        self.derivs = SteadyDerivatives(self.pinned, self.f.states, self.functional)

    def direction(self, column: List[float]) -> V:
        """A parameter direction."""
        return self.f.parameters.vector(
            self.bkd.reshape(self.bkd.asarray(column), (2, 1))
        )

    def solve(self, p: A) -> SteadyPoint[V, A]:
        # Whoever sets the parameters drops the setups kept for the old ones.
        self.f.set_param(p[:, 0])
        self.setups.clear()
        newton: Newton[V, A] = Newton(atol=1e-13, rtol=0.0, maxiters=30)
        point = self.system.solve(newton, self.f.states.zeros(1))
        assert point.converged()
        return point

    def gradient(self, p: A) -> A:
        point = self.solve(p)
        return self.case.array(self.derivs.gradient(self.derivs.adjoint(point)))

    def hessian(self, p: A) -> A:
        point = self.solve(p)
        hessian = self.derivs.hessian()
        assert hessian is not None
        adjoint = self.derivs.adjoint(point)
        columns = [[1.0, 0.0], [0.0, 1.0]]
        return self.bkd.hstack(
            [
                self.case.array(hessian.hvp(adjoint, self.direction(column)))
                for column in columns
            ]
        )


def _param(case: TypedStackCase[V, F, A]) -> A:
    return case.bkd.reshape(case.bkd.asarray(QUADRATIC_PARAM), (2, 1))


class TheGradientSuite:
    def test_it_is_the_derivative_of_the_solved_value(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        solved = _Solved(case)

        def value(p: A) -> A:
            return solved.derivs.value(solved.solve(p))

        def jacobian(p: A) -> A:
            return solved.gradient(p).T

        assert min_fd_error(case, 1, 2, value, jacobian, _param(case)) < 1e-6

    def test_wrong_model_control(self, case: TypedStackCase[V, F, A]) -> None:
        """Dropping the adjoint term leaves only the direct derivative."""
        solved = _Solved(case)
        bkd = case.bkd
        p = _param(case)
        gradient = solved.gradient(p)
        state = solved.system.space().copy(solved.solve(p).solution())
        direct = QuadraticParameterTerm(solved.f).derivatives().wrt_param
        by_state = QuadraticStateTerm(solved.f).derivatives().wrt_param
        assert direct is not None and by_state is not None
        partial = by_state(state)
        assert partial is not None
        one = bkd.ones((1, 1))
        missing = case.array(direct().apply_transpose(one)) + case.array(
            partial.apply_transpose(one)
        )
        miss = bkd.to_float(bkd.norm(gradient - missing))
        assert miss > 1e-2 * bkd.to_float(bkd.norm(gradient))


class TheHessianSuite:
    def test_it_is_the_derivative_of_the_gradient(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        solved = _Solved(case)
        error = min_fd_error(
            case, 2, 2, solved.gradient, solved.hessian, _param(case)
        )
        assert error < 1e-6

    def test_it_is_symmetric(self, case: TypedStackCase[V, F, A]) -> None:
        solved = _Solved(case)
        hessian = solved.hessian(_param(case))
        case.bkd.assert_allclose(hessian, hessian.T, rtol=1e-12, atol=1e-14)


class TheTangentSuite:
    def test_it_is_the_gradient_along_the_direction(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        """Forward and adjoint give the same derivative: dQ v = grad q . v."""
        solved = _Solved(case)
        bkd = case.bkd
        point = solved.solve(_param(case))
        gradient = case.array(solved.derivs.gradient(solved.derivs.adjoint(point)))
        direction = solved.direction([0.5, -0.2])
        bkd.assert_allclose(
            solved.derivs.tangent(point, direction),
            gradient.T @ case.array(direction),
            rtol=1e-12,
        )
        two = solved.f.parameters.vector(bkd.ones((2, 2)))
        with pytest.raises(ValueError, match="one parameter vector"):
            solved.derivs.tangent(point, two)


class SetupsSuite:
    def test_every_derivative_solve_shares_the_point_handle(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        solved = _Solved(case)
        point = solved.solve(_param(case))
        iterations = point.info().iterations
        assert solved.lu.setups == iterations
        hessian = solved.derivs.hessian()
        assert hessian is not None
        adjoint = solved.derivs.adjoint(point)
        solved.derivs.gradient(adjoint)
        hessian.hvp(adjoint, solved.direction([1.0, 1.0]))
        hessian.hvp(adjoint, solved.direction([0.5, 0.5]))
        solved.derivs.tangent(point, solved.direction([1.0, 1.0]))
        assert solved.lu.setups == iterations + 1

    def test_a_constant_jacobian_is_set_up_once(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        solved = _Solved(case, affine=True)
        point = solved.solve(case.bkd.reshape(case.bkd.asarray([0.0, -0.4]), (2, 1)))
        hessian = solved.derivs.hessian()
        assert hessian is not None
        adjoint = solved.derivs.adjoint(point)
        solved.derivs.gradient(adjoint)
        hessian.hvp(adjoint, solved.direction([1.0, 1.0]))
        assert solved.lu.setups == 1

    @pytest.mark.parametrize(
        "strategy, beyond_newton, affine",
        [
            (RecomputeAlways, 6, 4),
            (KeepStateFree, 6, 1),
            (KeepAll, 1, 1),
        ],
        ids=["RecomputeAlways", "KeepStateFree", "KeepAll"],
    )
    def test_counts_per_strategy(
        self,
        case: TypedStackCase[V, F, A],
        strategy: Callable[
            [CountingSolver[V, F]],
            SetupStrategyProtocol[SteadyRequest[V, F], V, F],
        ],
        beyond_newton: int,
        affine: int,
    ) -> None:
        """Nonlinear: k Newton setups, then the adjoint, two HVPs (two
        solves each) and a tangent, 6 requests at the solution. Affine: one
        Newton step, the adjoint and one HVP, 4 requests."""
        solved = _Solved(case, strategy=strategy)
        point = solved.solve(_param(case))
        k = point.info().iterations
        hessian = solved.derivs.hessian()
        assert hessian is not None
        adjoint = solved.derivs.adjoint(point)
        hessian.hvp(adjoint, solved.direction([1.0, 1.0]))
        hessian.hvp(adjoint, solved.direction([0.5, 0.5]))
        solved.derivs.tangent(point, solved.direction([1.0, 1.0]))
        assert solved.lu.setups == k + beyond_newton
        flat = _Solved(case, affine=True, strategy=strategy)
        point = flat.solve(case.bkd.reshape(case.bkd.asarray([0.0, -0.4]), (2, 1)))
        hessian = flat.derivs.hessian()
        assert hessian is not None
        hessian.hvp(flat.derivs.adjoint(point), flat.direction([1.0, 1.0]))
        assert flat.lu.setups == affine


class LevelsSuite:
    @pytest.mark.parametrize("orders", [(1, 2), (2, 1)])
    def test_first_order_gives_no_hessian(
        self, case: TypedStackCase[V, F, A], orders: Tuple[int, int]
    ) -> None:
        solved = _Solved(case, residual_order=orders[0], functional_order=orders[1])
        assert solved.derivs.hessian() is None
        solved.gradient(_param(case))

    def test_a_parameter_functional_needs_no_adjoint_solve(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        solved = _Solved(case)
        functional: StateFunctional[V, A] = StateFunctional.from_terms(
            case.bkd, 1, parameter_terms=(QuadraticParameterTerm(solved.f),)
        )
        derivs = SteadyDerivatives(solved.pinned, solved.f.states, functional)
        p = _param(case)
        point = solved.solve(p)
        setups = solved.lu.setups
        gradient = derivs.gradient(derivs.adjoint(point))
        assert solved.lu.setups == setups
        case.assert_vector(gradient, 0.3 * p, rtol=1e-14)


class RefusalsSuite:
    def test_a_residual_without_a_parameter_jacobian(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        f, pinned = pinned_quadratic(case, order=0)
        with pytest.raises(TypeError, match="no parameter Jacobian"):
            SteadyDerivatives(pinned, f.states, quadratic_functional(f))

    def test_a_term_without_derivatives(self, case: TypedStackCase[V, F, A]) -> None:
        f, pinned = pinned_quadratic(case)
        with pytest.raises(TypeError, match="QuadraticStateTerm.*only evaluated"):
            SteadyDerivatives(pinned, f.states, quadratic_functional(f, order=0))

        class _ValueOnly(QuadraticParameterTerm[V, F, A]):
            def derivatives(self) -> ParameterTermDerivatives[V, A]:
                return ParameterTermDerivatives.none()

        functional: StateFunctional[V, A] = StateFunctional.from_terms(
            case.bkd, 1, parameter_terms=(_ValueOnly(f),)
        )
        with pytest.raises(TypeError, match="_ValueOnly.*only evaluated"):
            SteadyDerivatives(pinned, f.states, functional)

    def test_a_stale_adjoint(self, case: TypedStackCase[V, F, A]) -> None:
        solved = _Solved(case)
        adjoint = solved.derivs.adjoint(solved.solve(_param(case)))
        solved.f.set_param(solved.f.p[:, 0])
        with pytest.raises(ValueError, match="changed"):
            solved.derivs.gradient(adjoint)

    def test_wrong_inputs(self, case: TypedStackCase[V, F, A]) -> None:
        solved = _Solved(case)
        bkd = case.bkd
        adjoint = solved.derivs.adjoint(solved.solve(_param(case)))
        hessian = solved.derivs.hessian()
        assert hessian is not None
        with pytest.raises(ValueError, match="one parameter vector"):
            hessian.hvp(adjoint, solved.f.parameters.vector(bkd.ones((2, 2))))
        with pytest.raises(ValueError, match="weights must have shape"):
            solved.derivs.adjoint(adjoint.point, bkd.ones((2, 1)))
        f: QuadraticResidual[V, F, A] = QuadraticResidual(case)
        functional = quadratic_functional(f)
        with pytest.raises(TypeError, match="ResidualProtocol"):
            SteadyDerivatives(3.0, f.states, functional)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="InnerProductSpaceProtocol"):
            SteadyDerivatives(f, 3.0, functional)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="StateFunctional"):
            SteadyDerivatives(f, f.states, 3.0)  # type: ignore[arg-type]
        for method in (solved.derivs.value, solved.derivs.adjoint):
            with pytest.raises(TypeError, match="SteadyPointProtocol"):
                method(3.0)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="SteadyPointProtocol"):
            solved.derivs.tangent(
                3.0,  # type: ignore[arg-type]
                solved.direction([1.0, 1.0]),
            )
