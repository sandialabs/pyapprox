r"""The adjoint gradient and Hessian-vector product of a steady functional.

The pinned quadratic of ``tests._helpers.implicitfunction_cases``
(:math:`F_i = (A u)_i + p_0 u_i^2 + p_1^2 c_i`, rows :math:`\{0, 2\}`
pinned to :math:`(p_0^2 + 1, p_0 p_1)`, :math:`p = (0.7, -0.4)`), solved
by Newton (``atol`` 1e-13) with dense LU, and :math:`Q = g + r` with
:math:`g = \tfrac12 u^\top D u + p_1 u_1`, :math:`D = \mathrm{diag}(1, 2,
3)`, :math:`r = \tfrac{3}{20}\|p\|^2`:

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
  ``KeepStateFree`` 1, ``RecomputeAlways`` 4. The case clears its
  strategy whenever it sets the parameters.
- Levels: first-order bundles give no Hessian; a functional of the
  parameters alone needs no adjoint solve.
- Refusals: a residual without a parameter Jacobian, a term without
  derivatives, a stale adjoint, a multi-column direction.
"""

from typing import Callable, Generic, Tuple

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
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseLU, DenseFormat
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.implicitfunction_cases import (
    QUADRATIC_PARAM,
    QuadraticParameterTerm,
    QuadraticResidual,
    QuadraticStateTerm,
    pinned_quadratic,
    quadratic_functional,
)
from tests._helpers.linalg.counting_solver import CountingSolver

_StrategyOf = Callable[
    [CountingSolver[Array, DenseFormat[Array]]],
    SetupStrategyProtocol[
        SteadyRequest[Array, DenseFormat[Array]], Array, DenseFormat[Array]
    ],
]


class _Case(Generic[Array]):
    """The pinned quadratic, its system and the functional's derivatives."""

    def __init__(
        self,
        bkd: Backend[Array],
        residual_order: int = 2,
        functional_order: int = 2,
        affine: bool = False,
        strategy: "_StrategyOf[Array]" = KeepAll,
    ) -> None:
        self.bkd = bkd
        self.f, self.pinned = pinned_quadratic(bkd, residual_order, affine)
        self.lu: CountingSolver[Array, DenseFormat[Array]] = CountingSolver(
            BackendDenseLU(bkd)
        )
        self.setups = strategy(self.lu)
        self.system = SteadySystem(self.pinned, self.f.states, self.setups)
        self.functional = quadratic_functional(self.f, functional_order)
        self.derivs = SteadyDerivatives(self.system, self.functional)

    def solve(self, p: Array) -> SteadyPoint[Array, Array]:
        # Whoever sets the parameters drops the setups kept for the old ones.
        self.f.set_param(p)
        self.setups.clear()
        point = self.system.solve(
            Newton(atol=1e-13, rtol=0.0, maxiters=30), self.bkd.zeros((3, 1))
        )
        assert point.converged()
        return point

    def gradient(self, p: Array) -> Array:
        point = self.solve(p)
        return self.derivs.gradient(self.derivs.adjoint(point))

    def hessian(self, p: Array) -> Array:
        point = self.solve(p)
        hessian = self.derivs.hessian()
        assert hessian is not None
        adjoint = self.derivs.adjoint(point)
        eye = self.bkd.eye(2)
        return self.bkd.hstack(
            [hessian.hvp(adjoint, eye[:, j : j + 1]) for j in range(2)]
        )


def _param(bkd: Backend[Array]) -> Array:
    return bkd.reshape(bkd.asarray(QUADRATIC_PARAM), (2, 1))


def _min_error(
    bkd: Backend[Array],
    nqoi: int,
    fun: Callable[[Array], Array],
    jacobian: Callable[[Array], Array],
) -> float:
    errors = DerivativeChecker(
        FunctionWithJacobianFromCallable(
            nqoi=nqoi, nvars=2, fun=fun, jacobian=jacobian, bkd=bkd
        )
    ).check_derivatives(_param(bkd), relative=True, central=True)[0]
    return bkd.to_float(bkd.min(errors))


class TestTheGradient:
    def test_it_is_the_derivative_of_the_solved_value(
        self, bkd: Backend[Array]
    ) -> None:
        case = _Case(bkd)

        def value(p: Array) -> Array:
            return case.derivs.value(case.solve(p))

        def jacobian(p: Array) -> Array:
            return case.gradient(p).T

        assert _min_error(bkd, 1, value, jacobian) < 1e-6

    def test_wrong_model_control(self, bkd: Backend[Array]) -> None:
        """Dropping the adjoint term leaves only the direct derivative."""
        case = _Case(bkd)
        p = _param(bkd)
        gradient = case.gradient(p)
        state = case.system.space().copy(case.solve(p).solution())
        direct = QuadraticParameterTerm(case.f).derivatives().wrt_param
        by_state = QuadraticStateTerm(case.f).derivatives().wrt_param
        assert direct is not None and by_state is not None
        partial = by_state(state)
        assert partial is not None
        one = bkd.ones((1, 1))
        missing = direct().apply_transpose(one) + partial.apply_transpose(one)
        miss = bkd.to_float(bkd.norm(gradient - missing))
        assert miss > 1e-2 * bkd.to_float(bkd.norm(gradient))


class TestTheHessian:
    def test_it_is_the_derivative_of_the_gradient(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        assert _min_error(bkd, 2, case.gradient, case.hessian) < 1e-6

    def test_it_is_symmetric(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        hessian = case.hessian(_param(bkd))
        bkd.assert_allclose(hessian, hessian.T, rtol=1e-12, atol=1e-14)


class TestTheTangent:
    def test_it_is_the_gradient_along_the_direction(self, bkd: Backend[Array]) -> None:
        """Forward and adjoint give the same derivative: dQ v = grad q . v."""
        case = _Case(bkd)
        p = _param(bkd)
        point = case.solve(p)
        gradient = case.derivs.gradient(case.derivs.adjoint(point))
        direction = bkd.asarray([[0.5], [-0.2]])
        bkd.assert_allclose(
            case.derivs.tangent(point, direction),
            gradient.T @ direction,
            rtol=1e-12,
        )
        with pytest.raises(ValueError, match="one parameter vector"):
            case.derivs.tangent(point, bkd.ones((2, 2)))


class TestSetups:
    def test_every_derivative_solve_shares_the_point_handle(
        self, bkd: Backend[Array]
    ) -> None:
        case = _Case(bkd)
        point = case.solve(_param(bkd))
        iterations = point.info().iterations
        assert case.lu.setups == iterations
        hessian = case.derivs.hessian()
        assert hessian is not None
        adjoint = case.derivs.adjoint(point)
        case.derivs.gradient(adjoint)
        hessian.hvp(adjoint, bkd.ones((2, 1)))
        hessian.hvp(adjoint, bkd.zeros((2, 1)) + 0.5)
        case.derivs.tangent(point, bkd.ones((2, 1)))
        assert case.lu.setups == iterations + 1

    def test_a_constant_jacobian_is_set_up_once(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd, affine=True)
        point = case.solve(bkd.reshape(bkd.asarray([0.0, -0.4]), (2, 1)))
        hessian = case.derivs.hessian()
        assert hessian is not None
        adjoint = case.derivs.adjoint(point)
        case.derivs.gradient(adjoint)
        hessian.hvp(adjoint, bkd.ones((2, 1)))
        assert case.lu.setups == 1

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
        bkd: Backend[Array],
        strategy: "_StrategyOf[Array]",
        beyond_newton: int,
        affine: int,
    ) -> None:
        """Nonlinear: k Newton setups, then the adjoint, two HVPs (two
        solves each) and a tangent, 6 requests at the solution. Affine: one
        Newton step, the adjoint and one HVP, 4 requests."""
        case = _Case(bkd, strategy=strategy)
        point = case.solve(_param(bkd))
        k = point.info().iterations
        hessian = case.derivs.hessian()
        assert hessian is not None
        adjoint = case.derivs.adjoint(point)
        hessian.hvp(adjoint, bkd.ones((2, 1)))
        hessian.hvp(adjoint, bkd.zeros((2, 1)) + 0.5)
        case.derivs.tangent(point, bkd.ones((2, 1)))
        assert case.lu.setups == k + beyond_newton
        flat = _Case(bkd, affine=True, strategy=strategy)
        point = flat.solve(bkd.reshape(bkd.asarray([0.0, -0.4]), (2, 1)))
        hessian = flat.derivs.hessian()
        assert hessian is not None
        hessian.hvp(flat.derivs.adjoint(point), bkd.ones((2, 1)))
        assert flat.lu.setups == affine


class TestLevels:
    @pytest.mark.parametrize("orders", [(1, 2), (2, 1)])
    def test_first_order_gives_no_hessian(
        self, bkd: Backend[Array], orders: Tuple[int, int]
    ) -> None:
        case = _Case(bkd, residual_order=orders[0], functional_order=orders[1])
        assert case.derivs.hessian() is None
        case.gradient(_param(bkd))

    def test_a_parameter_functional_needs_no_adjoint_solve(
        self, bkd: Backend[Array]
    ) -> None:
        case = _Case(bkd)
        functional: StateFunctional[Array, Array] = StateFunctional.from_terms(
            bkd, 1, parameter_terms=(QuadraticParameterTerm(case.f),)
        )
        derivs = SteadyDerivatives(case.system, functional)
        p = _param(bkd)
        point = case.solve(p)
        setups = case.lu.setups
        gradient = derivs.gradient(derivs.adjoint(point))
        assert case.lu.setups == setups
        bkd.assert_allclose(gradient, 0.3 * p, rtol=1e-14)


class TestRefusals:
    def test_a_residual_without_a_parameter_jacobian(self, bkd: Backend[Array]) -> None:
        f, pinned = pinned_quadratic(bkd, order=0)
        system = SteadySystem(pinned, f.states, RecomputeAlways(BackendDenseLU(bkd)))
        with pytest.raises(TypeError, match="no parameter Jacobian"):
            SteadyDerivatives(system, quadratic_functional(f))

    def test_a_term_without_derivatives(self, bkd: Backend[Array]) -> None:
        f, pinned = pinned_quadratic(bkd)
        system = SteadySystem(pinned, f.states, RecomputeAlways(BackendDenseLU(bkd)))
        with pytest.raises(TypeError, match="QuadraticStateTerm.*only evaluated"):
            SteadyDerivatives(system, quadratic_functional(f, order=0))

        class _ValueOnly(QuadraticParameterTerm[Array]):
            def derivatives(self) -> ParameterTermDerivatives[Array, Array]:
                return ParameterTermDerivatives.none()

        functional: StateFunctional[Array, Array] = StateFunctional.from_terms(
            bkd, 1, parameter_terms=(_ValueOnly(f),)
        )
        with pytest.raises(TypeError, match="_ValueOnly.*only evaluated"):
            SteadyDerivatives(system, functional)

    def test_a_stale_adjoint(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        adjoint = case.derivs.adjoint(case.solve(_param(bkd)))
        case.f.set_param(case.f.p)
        with pytest.raises(ValueError, match="changed"):
            case.derivs.gradient(adjoint)

    def test_wrong_inputs(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        adjoint = case.derivs.adjoint(case.solve(_param(bkd)))
        hessian = case.derivs.hessian()
        assert hessian is not None
        with pytest.raises(ValueError, match="one parameter vector"):
            hessian.hvp(adjoint, bkd.ones((2, 2)))
        with pytest.raises(ValueError, match="weights must have shape"):
            case.derivs.adjoint(adjoint.point, bkd.ones((2, 1)))
        f: QuadraticResidual[Array] = QuadraticResidual(bkd)
        with pytest.raises(TypeError, match="SteadySystem"):
            SteadyDerivatives(3.0, quadratic_functional(f))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="StateFunctional"):
            SteadyDerivatives(case.system, 3.0)  # type: ignore[arg-type]
