r"""The tableau stepper's forward step.

Tableaux: every library tableau satisfies the order conditions of its
order (computed from its coefficients: first order :math:`\sum b = 1`,
second :math:`\sum b c = 1/2`, third :math:`\sum b c^2 = 1/3` and
:math:`b^\top A c = 1/6`, fourth the four further conditions) and has the
declared structure (explicit, trivial first stage, stiffly accurate);
malformed tableaux are refused.

Linear step: on :math:`M \dot y = K y` (3 states, a nontrivial mass) one
step from :math:`y_n` equals the closed-form Runge-Kutta update, stages
:math:`(I - \Delta t\, A \otimes M^{-1}K)\, Y = 1 \otimes y_n` and
:math:`y_{n+1} = y_n + \Delta t\, (b^\top \otimes M^{-1}K)\, Y`, built
with dense NumPy from the tableau alone, for every tableau and both
backends.

Nonlinear step: on Lotka-Volterra with a nontrivial mass, the step
residual vanishes at the computed unknown for every tableau, and the
error at T = 0.2 against an RK4 reference with dt = 1/1600 falls at the
method's order as dt halves from 0.05 (rates within 0.15 of 1, 1, 2, 2,
2, 4, 3).

Opaque stack: on the mock stack, with a residual taking stack vectors
(identity form), forward and backward Euler steps of a linear system
equal their closed forms, so the stepper uses only the space, the
operators, the solvers and the form.
"""

from typing import Any, Callable, Generic, List

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.ode.mass_form_system import MassFormSystem, dense_mass_form_system
from pyapprox.ode.mass_matrix import ConstantDenseMassMatrix
from pyapprox.ode.residual_derivatives import (
    ODEResidualDerivatives,
    backend_state_space,
    dense_jacobian,
)
from pyapprox.ode.residual_form import IdentityForm
from pyapprox.ode.runge_kutta import RungeKuttaStepper
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.tableau import (
    ButcherTableau,
    alexander_sdirk3,
    backward_euler,
    classical_rk4,
    crank_nicolson,
    forward_euler,
    heun,
    implicit_midpoint,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox.util.linalg.operators import (
    BlockVector,
    LinearizationPoint,
    LinearizedOperator,
    SpaceDescriptor,
)
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.linalg.mock_stack import (
    MockFormat,
    MockLU,
    MockOperator,
    MockVector,
    MockVectorSpace,
    peek,
    wrap,
)
from tests._helpers.ode_models import LotkaVolterra

_Np = NDArray[np.float64]

# (tableau, order, explicit, trivial first stage, stiffly accurate)
_TABLEAUX: List[tuple[Callable[[], ButcherTableau], int, bool, bool, bool]] = [
    (forward_euler, 1, True, True, False),
    (backward_euler, 1, False, False, True),
    (crank_nicolson, 2, False, True, True),
    (implicit_midpoint, 2, False, False, False),
    (heun, 2, True, True, False),
    (classical_rk4, 4, True, True, False),
    (alexander_sdirk3, 3, False, False, True),
]
_IDS = [make().name for make, *_ in _TABLEAUX]

_MASS = [[2.0, 0.5, 0.0], [0.5, 1.5, 0.2], [0.0, 0.2, 1.0]]
_K = [[-1.0, 0.3, 0.0], [0.2, -0.8, 0.1], [0.0, 0.4, -1.2]]


def _arrays(tableau: ButcherTableau) -> tuple[_Np, _Np, _Np]:
    return np.array(tableau.a), np.array(tableau.b), np.array(tableau.c)


class TestTableaux:
    @pytest.mark.parametrize(
        "make, order, explicit, trivial, stiff", _TABLEAUX, ids=_IDS
    )
    def test_order_conditions_and_structure(
        self,
        make: Callable[[], ButcherTableau],
        order: int,
        explicit: bool,
        trivial: bool,
        stiff: bool,
    ) -> None:
        tableau = make()
        a, b, c = _arrays(tableau)
        conditions = [(b.sum(), 1.0)]
        if order >= 2:
            conditions.append((b @ c, 1 / 2))
        if order >= 3:
            conditions += [(b @ c**2, 1 / 3), (b @ a @ c, 1 / 6)]
        if order >= 4:
            conditions += [
                (b @ c**3, 1 / 4),
                (b @ (c * (a @ c)), 1 / 8),
                (b @ a @ c**2, 1 / 12),
                (b @ a @ a @ c, 1 / 24),
            ]
        for value, expected in conditions:
            assert abs(value - expected) < 1e-13
        assert tableau.is_explicit() == explicit
        assert tableau.has_trivial_first_stage() == trivial
        assert tableau.is_stiffly_accurate() == stiff

    def test_malformed_tableaux_are_refused(self) -> None:
        with pytest.raises(ValueError, match="same number of stages"):
            ButcherTableau(a=((1.0,),), b=(0.5, 0.5), c=(1.0,), name="bad")
        with pytest.raises(ValueError, match="above its diagonal"):
            ButcherTableau(
                a=((0.25, 0.25), (0.25, 0.25)), b=(0.5, 0.5), c=(0.5, 0.5), name="bad"
            )
        with pytest.raises(ValueError, match="not the row sum"):
            ButcherTableau(a=((1.0,),), b=(1.0,), c=(0.5,), name="bad")


class _LinearModel(Generic[Array]):
    """:math:`M \\dot y = K y` on the backend, with a dense Jacobian."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._k = bkd.asarray(_K)
        self._mass = ConstantDenseMassMatrix(bkd.asarray(_MASS), bkd)
        states = backend_state_space(bkd, 3)
        self._derivatives: ODEResidualDerivatives[Array, DenseFormat[Array]] = (
            ODEResidualDerivatives.state_only(
                dense_jacobian(self.jacobian, states, bkd)
            )
        )

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __call__(self, state: Array, time: float) -> Array:
        return self._bkd.dot(self._k, state)

    def jacobian(self, state: Array, time: float) -> Array:
        return self._k

    def mass_matrix(self) -> ConstantDenseMassMatrix[Array]:
        return self._mass

    def derivatives(self) -> ODEResidualDerivatives[Array, DenseFormat[Array]]:
        return self._derivatives


def _stepper(
    tableau: ButcherTableau,
    system: MassFormSystem[Array, Array, DenseFormat[Array]],
) -> RungeKuttaStepper[Array, Array, DenseFormat[Array]]:
    newton: Newton[Array, Array] = Newton(atol=1e-13, rtol=0.0, maxiters=30)
    return RungeKuttaStepper(tableau, system, None if tableau.is_explicit() else newton)


def _closed_form_step(tableau: ButcherTableau, y: _Np, deltat: float) -> _Np:
    """The Runge-Kutta update of a linear system from the tableau alone."""
    a, b, _ = _arrays(tableau)
    z = np.linalg.solve(np.array(_MASS), np.array(_K))
    n, s = z.shape[0], len(b)
    stages = np.linalg.solve(
        np.eye(s * n) - deltat * np.kron(a, z), np.kron(np.ones(s), y)
    )
    return y + deltat * np.kron(b, z) @ stages


def _new_state(
    stepper: RungeKuttaStepper[Array, Array, DenseFormat[Array]],
    unknown: BlockVector[Array],
) -> Array:
    return stepper.readout().apply(unknown).blocks[0]


class TestLinearStep:
    @pytest.mark.parametrize("make", [make for make, *_ in _TABLEAUX], ids=_IDS)
    def test_one_step_is_the_closed_form_update(
        self, bkd: Backend[Array], make: Callable[[], ButcherTableau]
    ) -> None:
        tableau = make()
        stepper = _stepper(tableau, dense_mass_form_system(_LinearModel(bkd)))
        y = np.array([1.0, -0.5, 2.0])
        unknown = stepper.step(bkd.asarray(y[:, None]), 0.3, 0.1)
        bkd.assert_allclose(
            _new_state(stepper, unknown)[:, 0],
            bkd.asarray(_closed_form_step(tableau, y, 0.1)),
            rtol=1e-11,
        )

    def test_an_implicit_tableau_needs_a_nonlinear_solver(
        self, bkd: Backend[Array]
    ) -> None:
        with pytest.raises(TypeError, match="needs a nonlinear solver"):
            RungeKuttaStepper(
                backward_euler(), dense_mass_form_system(_LinearModel(bkd))
            )


def _lotka_volterra(bkd: Backend[Array]) -> LotkaVolterra[Array]:
    return LotkaVolterra(bkd, mass=bkd.asarray([[2.0, 0.5], [0.5, 1.0]]))


class TestNonlinearStep:
    @pytest.mark.parametrize("make", [make for make, *_ in _TABLEAUX], ids=_IDS)
    def test_the_step_residual_vanishes_at_the_step(
        self, bkd: Backend[Array], make: Callable[[], ButcherTableau]
    ) -> None:
        stepper = _stepper(make(), dense_mass_form_system(_lotka_volterra(bkd)))
        y = bkd.asarray([[10.0], [5.0]])
        unknown = stepper.step(y, 0.2, 0.05)
        window = StepWindow(0.2, 0.05, (y,), unknown)
        residual = stepper.residual_space().norm(stepper.step_residual(window))
        assert bkd.to_float(residual[0, 0]) < 1e-10

    @pytest.mark.parametrize(
        "make, order", [(make, order) for make, order, *_ in _TABLEAUX], ids=_IDS
    )
    def test_convergence_order(
        self, numpy_bkd: NumpyBkd, make: Callable[[], ButcherTableau], order: int
    ) -> None:
        bkd = numpy_bkd
        system = dense_mass_form_system(_lotka_volterra(bkd))

        def solve(tableau: ButcherTableau, deltat: float) -> NDArray[Any]:
            stepper = _stepper(tableau, system)
            y = bkd.asarray([[10.0], [5.0]])
            nsteps = round(0.2 / deltat)
            for n in range(nsteps):
                y = _new_state(stepper, stepper.step(y, n * deltat, deltat))
            return y

        reference = solve(classical_rk4(), 0.2 / 1600)
        errors = [
            float(np.linalg.norm(solve(make(), dt) - reference))
            for dt in (0.05, 0.025, 0.0125)
        ]
        rates = [np.log2(e0 / e1) for e0, e1 in zip(errors, errors[1:])]
        assert all(abs(rate - order) < 0.15 for rate in rates), rates


def _mock_rhs_operator() -> MockOperator:
    return MockOperator.from_dense(np.array(_K), _MOCK_STATE)


_MOCK_STATE = SpaceDescriptor("state", 3, "float64")


class _MockLinear:
    """:math:`\\dot y = K y` taking the mock stack's vectors directly."""

    def __init__(self) -> None:
        self._k = _mock_rhs_operator()
        self._derivatives: ODEResidualDerivatives[MockVector, MockFormat] = (
            ODEResidualDerivatives.state_only(self._jacobian)
        )

    def __call__(self, state: MockVector, time: float) -> MockVector:
        return self._k.apply(state)

    def _jacobian(
        self, state: MockVector, time: float, point: LinearizationPoint
    ) -> LinearizedOperator[MockVector, MockFormat]:
        return LinearizedOperator(self._k, point)

    def derivatives(self) -> ODEResidualDerivatives[MockVector, MockFormat]:
        return self._derivatives


class TestOnAnOpaqueStack:
    @pytest.mark.parametrize("make", [forward_euler, backward_euler])
    def test_euler_steps_are_their_closed_forms(
        self, make: Callable[[], ButcherTableau]
    ) -> None:
        rhs = _MockLinear()
        jacobian = rhs.derivatives().jacobian
        assert jacobian is not None
        system: MassFormSystem[MockVector, NDArray[Any], MockFormat] = MassFormSystem(
            rhs=rhs,
            mass=MockOperator.identity(_MOCK_STATE),
            jacobian=jacobian,
            space=MockVectorSpace(_MOCK_STATE),
            solver=MockLU(),
            form=IdentityForm(),
        )
        tableau = make()
        newton: Newton[MockVector, NDArray[Any]] = Newton(atol=1e-13, rtol=0.0)
        stepper = RungeKuttaStepper(
            tableau, system, None if tableau.is_explicit() else newton
        )
        y = np.array([[1.0], [-0.5], [2.0]])
        unknown = stepper.step(wrap(y, _MOCK_STATE), 0.0, 0.1)
        k = np.array(_K)
        expected = (
            y + 0.1 * k @ y
            if tableau.is_explicit()
            else np.linalg.solve(np.eye(3) - 0.1 * k, y)
        )
        new_state = stepper.readout().apply(unknown).blocks[0]
        np.testing.assert_allclose(peek(new_state), expected, rtol=1e-12)
