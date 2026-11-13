r"""The observation, misfit and Tikhonov terms on a steady solution.

The pinned quadratic of ``tests._helpers.implicitfunction_cases``
(:math:`F_i = (A u)_i + p_0 u_i^2 + p_1^2 c_i`, rows :math:`\{0, 2\}`
pinned, :math:`p = (0.7, -0.4)`), solved by Newton (``atol`` 1e-13) with
dense LU; :math:`H` observes entries 1 and 2 of :math:`u`.

- :math:`Q = \|H u - d\|^2 / (2 \cdot 0.5^2) + (0.3/2)\, p^\top W p`, with
  :math:`d = (0.5, -0.3)` and :math:`W = [[2, 0.5], [0.5, 1]]`: its value
  is the two formulas evaluated directly (1e-14), its adjoint gradient
  the derivative of the solved value and its HVP the derivative of the
  gradient, by central finite differences (minimum relative error below
  1e-6).
- :math:`Q = H u` (two entries): its forward tangent along each
  parameter is the derivative of the solved value, likewise.
- Refusals: empty or out-of-range entries; a misfit of data of the wrong
  shape, of a non-operator or with non-positive noise; an observed value
  of a non-operator; a non-square or asymmetric weight, an unset or
  non-column parameter.
"""

from typing import Callable, Generic

import pytest

from pyapprox.implicitfunction.observation_terms import (
    ObservationMisfit,
    ObservedValue,
    StateEntries,
    StateObservationProtocol,
)
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_derivatives import SteadyDerivatives
from pyapprox.implicitfunction.steady_system import SteadyPoint, SteadySystem
from pyapprox.implicitfunction.tikhonov_term import TikhonovTerm
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.implicitfunction_cases import QUADRATIC_PARAM, pinned_quadratic

_ROWS = [1, 2]
_DATA = [[0.5], [-0.3]]
_WEIGHT = [[2.0, 0.5], [0.5, 1.0]]


class _Case(Generic[Array]):
    """The pinned quadratic, its system, and a Tikhonov term kept at the
    residual's parameters."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self.bkd = bkd
        self.f, self.pinned = pinned_quadratic(bkd)
        self.system = SteadySystem(
            self.pinned, self.f.states, RecomputeAlways(BackendDenseLU(bkd))
        )
        self.observation = StateEntries(bkd, 3, _ROWS)
        self.tikhonov = TikhonovTerm(bkd, 0.3, bkd.asarray(_WEIGHT))

    def solve(self, p: Array) -> SteadyPoint[Array, Array]:
        self.f.set_param(p[:, 0])
        self.tikhonov.set_param(p)
        point = self.system.solve(
            Newton(atol=1e-13, rtol=0.0, maxiters=30), self.bkd.zeros((3, 1))
        )
        assert point.converged()
        return point

    def misfit_functional(self) -> StateFunctional[Array, Array]:
        misfit = ObservationMisfit(
            self.bkd, self.observation, self.bkd.asarray(_DATA), noise_std=0.5
        )
        return StateFunctional.from_terms(
            self.bkd, 1, parameter_terms=(self.tikhonov,), state_terms=(misfit,)
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


class TestTheMisfitAndTikhonovTerms:
    def test_the_value(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        p = _param(bkd)
        derivs = SteadyDerivatives(case.pinned, case.f.states, case.misfit_functional())
        point = case.solve(p)
        u = case.system.space().copy(point.solution())
        residual = u[1:3] - bkd.asarray(_DATA)
        expected = 2.0 * bkd.sum(residual * residual, keepdims=True) + 0.15 * (
            p.T @ bkd.asarray(_WEIGHT) @ p
        )
        bkd.assert_allclose(derivs.value(point), expected, rtol=1e-14)

    def test_gradient_and_hvp(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        derivs = SteadyDerivatives(case.pinned, case.f.states, case.misfit_functional())
        hessian = derivs.hessian()
        assert hessian is not None

        def value(p: Array) -> Array:
            return derivs.value(case.solve(p))

        def gradient(p: Array) -> Array:
            return derivs.gradient(derivs.adjoint(case.solve(p)))

        def gradient_row(p: Array) -> Array:
            return gradient(p).T

        def hessian_matrix(p: Array) -> Array:
            adjoint = derivs.adjoint(case.solve(p))
            eye = bkd.eye(2)
            return bkd.hstack(
                [hessian.hvp(adjoint, eye[:, j : j + 1]) for j in range(2)]
            )

        assert _min_error(bkd, 1, value, gradient_row) < 1e-6
        assert _min_error(bkd, 2, gradient, hessian_matrix) < 1e-6


class TestTheObservedValue:
    def test_the_tangent(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        functional: StateFunctional[Array, Array] = StateFunctional.from_terms(
            bkd, 2, state_terms=(ObservedValue(case.observation),)
        )
        derivs = SteadyDerivatives(case.pinned, case.f.states, functional)

        def value(p: Array) -> Array:
            return derivs.value(case.solve(p))

        def jacobian(p: Array) -> Array:
            point = case.solve(p)
            eye = bkd.eye(2)
            return bkd.hstack(
                [derivs.tangent(point, eye[:, j : j + 1]) for j in range(2)]
            )

        assert _min_error(bkd, 2, value, jacobian) < 1e-6


class TestRefusals:
    def test_state_entries(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="at least one entry"):
            StateEntries(bkd, 2, [])
        with pytest.raises(ValueError, match="not all in"):
            StateEntries(bkd, 2, [0, 2])
        assert isinstance(StateEntries(bkd, 2, [0]), StateObservationProtocol)

    def test_misfit_and_observed_value(self, bkd: Backend[Array]) -> None:
        entries = StateEntries(bkd, 2, [0])
        datum = bkd.asarray([[1.0]])
        with pytest.raises(ValueError, match="must have shape"):
            ObservationMisfit(bkd, entries, bkd.asarray([[1.0], [2.0]]))
        with pytest.raises(TypeError, match="StateObservationProtocol"):
            ObservationMisfit(bkd, object(), datum)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="positive"):
            ObservationMisfit(bkd, entries, datum, noise_std=0.0)
        with pytest.raises(TypeError, match="StateObservationProtocol"):
            ObservedValue(object())  # type: ignore[arg-type]

    def test_tikhonov(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="square"):
            TikhonovTerm(bkd, 1.0, bkd.ones((2, 3)))
        with pytest.raises(ValueError, match="symmetric"):
            TikhonovTerm(bkd, 1.0, bkd.asarray([[1.0, 2.0], [0.0, 1.0]]))
        term = TikhonovTerm(bkd, 1.0)
        with pytest.raises(ValueError, match="set_param"):
            term.value()
        with pytest.raises(ValueError, match="column"):
            term.set_param(bkd.ones((2,)))
