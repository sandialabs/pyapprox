r"""A linear observation of a residual, :math:`Q = H F(u)`, on a steady
solution.

The pinned quadratic of ``tests._helpers.implicitfunction_cases``
(:math:`F_i = (A u)_i + p_0 u_i^2 + p_1^2 c_i`, rows :math:`\{0, 2\}`
pinned, :math:`p = (0.7, -0.4)`), solved by Newton (``atol`` 1e-13) with
dense LU. :math:`H = W^\top`, :math:`W = [e_0, e_2]`, observes the
unpinned :math:`F` on the pinned rows, where it does not vanish at the
solution: the analog of an essential boundary's flux. The term is built
from the solved pinned residual (``pinned_rows_residual``). Weights
:math:`\omega = (1.5, -0.5)` make it scalar for the adjoint.

- The value is :math:`F` evaluated directly on rows 0 and 2 (1e-14), and
  :math:`F` on the free row vanishes (1e-12).
- The adjoint gradient of :math:`\omega^\top Q` is the derivative of the
  solved value, its HVP the derivative of the gradient, and the forward
  tangent of :math:`Q` the derivative of both entries, by central finite
  differences (minimum relative error below 1e-6).
- The term's bundle follows the residual's: none without a parameter
  Jacobian, first order without second derivatives, and a state-state
  block skipped where the residual states it zero (the affine quadratic,
  checked directly: it is affine only at :math:`p_0 = 0`, so finite
  differences in :math:`p_0` would test a false declaration).
- It reads the solve's own :math:`F`: after ``set_param`` on the solved
  problem, the term's value moves with it.
- Refusals: a residual, observation or space not satisfying its
  protocol; an observation of another space; weights of the wrong shape
  or nonzero off the pinned rows (a free row, where :math:`F` vanishes);
  a residual that is not pinned.
"""

from typing import Callable, Generic

import pytest

from pyapprox.implicitfunction.observation_terms import RowWeights, StateEntries
from pyapprox.implicitfunction.observed_residual import (
    ObservedResidual,
    pinned_rows_residual,
)
from pyapprox.implicitfunction.setups import RecomputeAlways
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
from pyapprox.util.linalg.backend_dense import BackendDenseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.implicitfunction_cases import (
    QUADRATIC_PARAM,
    QuadraticResidual,
    pinned_quadratic,
)

_PINNED = [0, 2]
_WEIGHTS = [[1.5], [-0.5]]


def _unit_columns(bkd: Backend[Array]) -> Array:
    """:math:`W = [e_0, e_2]`: each pinned row read on its own."""
    return bkd.asarray([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])


class _Case(Generic[Array]):
    """The pinned quadratic, its system, and the observed unpinned
    residual."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self.bkd = bkd
        self.f, self.pinned = pinned_quadratic(bkd)
        self.system = SteadySystem(
            self.pinned, self.f.states, RecomputeAlways(BackendDenseLU(bkd))
        )
        self.term = pinned_rows_residual(bkd, self.pinned, _unit_columns(bkd))
        functional: StateFunctional[Array, Array] = StateFunctional.from_terms(
            bkd, 2, state_terms=(self.term,)
        )
        self.derivs = SteadyDerivatives(self.pinned, self.f.states, functional)

    def param(self) -> Array:
        return self.bkd.reshape(self.bkd.asarray(QUADRATIC_PARAM), (2, 1))

    def solve(self, p: Array) -> SteadyPoint[Array, Array]:
        self.f.set_param(p[:, 0])
        point = self.system.solve(
            Newton(atol=1e-13, rtol=0.0, maxiters=30), self.bkd.zeros((3, 1))
        )
        assert point.converged()
        return point

    def weights(self) -> Array:
        return self.bkd.asarray(_WEIGHTS)


def _min_error(
    case: _Case[Array],
    nqoi: int,
    fun: Callable[[Array], Array],
    jacobian: Callable[[Array], Array],
) -> float:
    bkd = case.bkd
    errors = DerivativeChecker(
        FunctionWithJacobianFromCallable(
            nqoi=nqoi, nvars=2, fun=fun, jacobian=jacobian, bkd=bkd
        )
    ).check_derivatives(case.param(), relative=True, central=True)[0]
    return bkd.to_float(bkd.min(errors))


class TestTheValue:
    def test_reads_the_unpinned_residual(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        point = case.solve(case.param())
        u = case.system.space().copy(point.solution())
        raw = case.f(u)
        bkd.assert_allclose(case.term.value(u), raw[_PINNED], rtol=1e-14)
        bkd.assert_allclose(raw[1:2], bkd.zeros((1, 1)), atol=1e-12)
        assert case.term.nqoi() == 2

    def test_reads_the_solves_parameters(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        u = case.system.space().copy(case.solve(case.param()).solution())
        before = case.term.value(u)
        case.f.set_param(bkd.asarray([0.2, 0.9]))
        bkd.assert_allclose(case.term.value(u), case.f(u)[_PINNED], rtol=1e-14)
        assert bkd.to_float(bkd.max(bkd.abs(case.term.value(u) - before))) > 0.1


class TestTheDerivatives:
    def test_gradient_and_hvp(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        derivs, weights = case.derivs, case.weights()
        hessian = derivs.hessian()
        assert hessian is not None

        def value(p: Array) -> Array:
            return weights.T @ derivs.value(case.solve(p))

        def gradient(p: Array) -> Array:
            return derivs.gradient(derivs.adjoint(case.solve(p), weights))

        def gradient_row(p: Array) -> Array:
            return gradient(p).T

        def hessian_matrix(p: Array) -> Array:
            adjoint = derivs.adjoint(case.solve(p), weights)
            eye = bkd.eye(2)
            return bkd.hstack(
                [hessian.hvp(adjoint, eye[:, j : j + 1]) for j in range(2)]
            )

        assert _min_error(case, 1, value, gradient_row) < 1e-6
        assert _min_error(case, 2, gradient, hessian_matrix) < 1e-6

    def test_tangent(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        derivs = case.derivs

        def value(p: Array) -> Array:
            return derivs.value(case.solve(p))

        def jacobian(p: Array) -> Array:
            point = case.solve(p)
            eye = bkd.eye(2)
            return bkd.hstack(
                [derivs.tangent(point, eye[:, j : j + 1]) for j in range(2)]
            )

        assert _min_error(case, 2, value, jacobian) < 1e-6


class TestTheBundle:
    def test_follows_the_residual(self, bkd: Backend[Array]) -> None:
        entries = StateEntries(bkd, 3, _PINNED)
        f0: QuadraticResidual[Array] = QuadraticResidual(bkd, order=0)
        f1: QuadraticResidual[Array] = QuadraticResidual(bkd, order=1)
        f2: QuadraticResidual[Array] = QuadraticResidual(bkd, order=2)
        none = ObservedResidual(f0, entries, f0.states)
        assert none.derivatives().wrt_state is None
        first = ObservedResidual(f1, entries, f1.states)
        assert first.derivatives().wrt_state is not None
        assert first.derivatives().wrt_param is not None
        assert first.derivatives().state_state is None
        second = ObservedResidual(f2, entries, f2.states)
        assert second.derivatives().state_state is not None

    def test_a_stated_zero_block_is_none(self, bkd: Backend[Array]) -> None:
        # The affine quadratic declares its state curvature zero, so the
        # term's state-state block is skipped, not applied.
        f = QuadraticResidual(bkd, order=2, affine=True)
        term = ObservedResidual(f, StateEntries(bkd, 3, _PINNED), f.states)
        state_state = term.derivatives().state_state
        assert state_state is not None
        u, w = bkd.ones((3, 1)), bkd.ones((3, 1))
        assert state_state(u, bkd.asarray(_WEIGHTS), w) is None


class TestRefusals:
    def test_protocols(self, bkd: Backend[Array]) -> None:
        f: QuadraticResidual[Array] = QuadraticResidual(bkd)
        entries = StateEntries(bkd, 3, [0])
        with pytest.raises(TypeError, match="ResidualProtocol"):
            ObservedResidual(object(), entries, f.states)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="StateObservationProtocol"):
            ObservedResidual(f, object(), f.states)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="VectorSpaceProtocol"):
            ObservedResidual(f, entries, object())  # type: ignore[arg-type]

    def test_an_observation_of_another_space(self, bkd: Backend[Array]) -> None:
        f: QuadraticResidual[Array] = QuadraticResidual(bkd)
        with pytest.raises(ValueError, match="observation"):
            ObservedResidual(f, StateEntries(bkd, 4, [0]), f.states)

    def test_weights(self, bkd: Backend[Array]) -> None:
        _, pinned = pinned_quadratic(bkd)
        rows = pinned.pin().rows()
        with pytest.raises(ValueError, match="shape"):
            RowWeights(bkd, rows, bkd.ones((3,)))
        with pytest.raises(ValueError, match="shape"):
            RowWeights(bkd, rows, bkd.ones((4, 1)))
        # Row 1 is free: F vanishes there, so reading it is an error.
        with pytest.raises(ValueError, match="vanish off the rows"):
            pinned_rows_residual(bkd, pinned, bkd.asarray([[1.0], [0.5], [0.0]]))
        with pytest.raises(TypeError, match="PinnedResidual"):
            pinned_rows_residual(
                bkd,
                QuadraticResidual(bkd),  # type: ignore[arg-type]
                _unit_columns(bkd),
            )
