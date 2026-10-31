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
backends; the system is affine (its Jacobian is constant), so a step
evaluates f once per implicit stage and once per slope it reads.

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

from typing import Any, Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.ode.mass_form_system import dense_mass_form_system
from pyapprox.ode.runge_kutta import RungeKuttaStepper
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.tableau import (
    ButcherTableau,
    backward_euler,
    classical_rk4,
    forward_euler,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.linalg.mock_stack import (
    peek,
    wrap,
)
from tests._helpers.ode.runge_kutta_cases import (
    LINEAR_K,
    LINEAR_MASS,
    MOCK_STATE,
    TABLEAU_IDS,
    TABLEAUX,
    LinearMassModel,
    lotka_volterra_with_mass,
    mock_stepper,
    new_state,
    rk_stepper,
)

_Np = NDArray[np.float64]


def _arrays(tableau: ButcherTableau) -> tuple[_Np, _Np, _Np]:
    return np.array(tableau.a), np.array(tableau.b), np.array(tableau.c)


class TestTableaux:
    @pytest.mark.parametrize(
        "make, order, explicit, trivial, stiff", TABLEAUX, ids=TABLEAU_IDS
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


def _closed_form_step(tableau: ButcherTableau, y: _Np, deltat: float) -> _Np:
    """The Runge-Kutta update of a linear system from the tableau alone."""
    a, b, _ = _arrays(tableau)
    z = np.linalg.solve(np.array(LINEAR_MASS), np.array(LINEAR_K))
    n, s = z.shape[0], len(b)
    stages = np.linalg.solve(
        np.eye(s * n) - deltat * np.kron(a, z), np.kron(np.ones(s), y)
    )
    return y + deltat * np.kron(b, z) @ stages


class _CountingLinear(LinearMassModel[Array]):
    """:class:`LinearMassModel` counting evaluations of f."""

    def __init__(self, bkd: Backend[Array]) -> None:
        super().__init__(bkd)
        self.nevals = 0

    def __call__(self, state: Array, time: float) -> Array:
        self.nevals += 1
        return super().__call__(state, time)


_AFFINE_EVALS = [1, 1, 2, 2, 2, 4, 5]
"""f evaluations in one step of an affine system, in ``TABLEAUX`` order:
one per implicit stage (Newton's residual at its guess; the affine path
takes the full step with no convergence check) plus one per stage slope a
later stage or the update reads. Forward Euler 0 + 1, backward Euler 1 +
0 (stiffly accurate), Crank-Nicolson 1 + 1 (explicit first slope, read
by the second stage), implicit midpoint 1 + 1 (the update reads the
slope), Heun 0 + 2, RK4 0 + 4, SDIRK3 3 + 2 (stiffly accurate; slopes
one and two are read)."""


class TestLinearStep:
    @pytest.mark.parametrize("make", [make for make, *_ in TABLEAUX], ids=TABLEAU_IDS)
    def test_one_step_is_the_closed_form_update(
        self, bkd: Backend[Array], make: Callable[[], ButcherTableau]
    ) -> None:
        tableau = make()
        stepper = rk_stepper(tableau, dense_mass_form_system(LinearMassModel(bkd)))
        y = np.array([1.0, -0.5, 2.0])
        unknown = stepper.step((bkd.asarray(y[:, None]),), (0.4, 0.3))
        bkd.assert_allclose(
            new_state(stepper, unknown)[:, 0],
            bkd.asarray(_closed_form_step(tableau, y, 0.1)),
            rtol=1e-11,
        )

    @pytest.mark.parametrize(
        "make, nevals", list(zip([make for make, *_ in TABLEAUX], _AFFINE_EVALS))
    )
    def test_an_affine_step_evaluates_f_once_per_solve_and_read_slope(
        self, bkd: Backend[Array], make: Callable[[], ButcherTableau], nevals: int
    ) -> None:
        model = _CountingLinear(bkd)
        stepper = rk_stepper(make(), dense_mass_form_system(model))
        assert stepper.system().is_affine()
        stepper.step((bkd.asarray([[1.0], [-0.5], [2.0]]),), (0.4, 0.3))
        assert model.nevals == nevals

    def test_an_implicit_tableau_needs_a_nonlinear_solver(
        self, bkd: Backend[Array]
    ) -> None:
        with pytest.raises(TypeError, match="needs a nonlinear solver"):
            RungeKuttaStepper(
                backward_euler(), dense_mass_form_system(LinearMassModel(bkd))
            )


class TestNonlinearStep:
    @pytest.mark.parametrize("make", [make for make, *_ in TABLEAUX], ids=TABLEAU_IDS)
    def test_the_step_residual_vanishes_at_the_step(
        self, bkd: Backend[Array], make: Callable[[], ButcherTableau]
    ) -> None:
        stepper = rk_stepper(
            make(), dense_mass_form_system(lotka_volterra_with_mass(bkd))
        )
        y = bkd.asarray([[10.0], [5.0]])
        unknown = stepper.step((y,), (0.25, 0.2))
        window = StepWindow((0.25, 0.2), (y,), unknown)
        residual = stepper.residual_space().norm(stepper.step_residual(window))
        assert bkd.to_float(residual[0, 0]) < 1e-10

    @pytest.mark.parametrize(
        "make, order", [(make, order) for make, order, *_ in TABLEAUX], ids=TABLEAU_IDS
    )
    def test_convergence_order(
        self, numpy_bkd: NumpyBkd, make: Callable[[], ButcherTableau], order: int
    ) -> None:
        bkd = numpy_bkd
        system = dense_mass_form_system(lotka_volterra_with_mass(bkd))

        def solve(tableau: ButcherTableau, deltat: float) -> NDArray[Any]:
            stepper = rk_stepper(tableau, system)
            y = bkd.asarray([[10.0], [5.0]])
            nsteps = round(0.2 / deltat)
            for n in range(nsteps):
                times = ((n + 1) * deltat, n * deltat)
                y = new_state(stepper, stepper.step((y,), times))
            return y

        reference = solve(classical_rk4(), 0.2 / 1600)
        errors = [
            float(np.linalg.norm(solve(make(), dt) - reference))
            for dt in (0.05, 0.025, 0.0125)
        ]
        rates = [np.log2(e0 / e1) for e0, e1 in zip(errors, errors[1:])]
        assert all(abs(rate - order) < 0.15 for rate in rates), rates


class TestOnAnOpaqueStack:
    @pytest.mark.parametrize("make", [forward_euler, backward_euler])
    def test_euler_steps_are_their_closed_forms(
        self, make: Callable[[], ButcherTableau]
    ) -> None:
        tableau = make()
        stepper = mock_stepper(tableau)
        y = np.array([[1.0], [-0.5], [2.0]])
        unknown = stepper.step((wrap(y, MOCK_STATE),), (0.1, 0.0))
        k = np.array(LINEAR_K)
        expected = (
            y + 0.1 * k @ y
            if tableau.is_explicit()
            else np.linalg.solve(np.eye(3) - 0.1 * k, y)
        )
        stepped = stepper.readout().apply(unknown).blocks[0]
        np.testing.assert_allclose(peek(stepped), expected, rtol=1e-12)
