r"""The tableau stepper's step-equation derivatives.

Hand-derived: on Lotka-Volterra with a mass, at a converged step from
(10, 5), t = 0.2, dt = 0.05, backward Euler (unknown Y, the new state)
has :math:`\partial R/\partial Y = M - \Delta t J(Y, t + \Delta t)`,
:math:`\partial R/\partial y_n = -M` and
:math:`\partial R/\partial p = -\Delta t f_p(Y, t + \Delta t)`; forward
Euler (trivial first stage, the unknown a final block) has
:math:`M`, :math:`-M - \Delta t J(y_n, t)` and
:math:`-\Delta t f_p(y_n, t)`. Each block is formed by applying the
stepper's operator to identity columns and compared with the model's own
matrices.

Step map, against the textbook update: on :math:`M \dot y = K y` the
step's tangent :math:`-E\,\mathrm{diag}^{-1}\,\mathrm{off\_diag}` equals
the step-map matrix
:math:`\Phi = I + \Delta t (b^\top \otimes Z)(I - \Delta t A \otimes Z)^{-1}
(\mathbf{1} \otimes I)`, :math:`Z = M^{-1}K`, built from the tableau
alone, and the adjoint composition
:math:`-\mathrm{off\_diag}^\top\,\mathrm{diag}^{-\top} E^\top` equals
:math:`\Phi^\top`, for every tableau and both backends: the stage-level
derivatives, the readout, the diagonal handle and their transposes
reproduce the discrete adjoint of the Runge-Kutta update.

Checker: every tableau's derivatives, second order included, pass the
stepper checker at its converged step (forward-difference errors below
1e-6, dot tests and handle errors below 1e-12).
"""

from typing import Callable

import pytest

from pyapprox.ode.mass_form_system import dense_mass_form_system
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.tableau import ButcherTableau, backward_euler, forward_euler
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import BlockVector
from tests._helpers.ode.runge_kutta_cases import (
    TABLEAU_IDS,
    TABLEAUX,
    LinearMassModel,
    backend_checker,
    linear_step_map,
    lotka_volterra_step,
    rk_stepper,
)

_TABLEAU_MAKERS = [make for make, *_ in TABLEAUX]


def _single_block_matrix(
    bkd: Backend[Array],
    op_apply: Callable[[BlockVector[Array]], BlockVector[Array]],
    n: int,
) -> Array:
    """The matrix of an operator between one-block spaces, column by
    column on the identity."""
    return op_apply(BlockVector((bkd.eye(n),))).blocks[0]


class TestHandDerived:
    def test_backward_euler(self, bkd: Backend[Array]) -> None:
        case = lotka_volterra_step(bkd, backward_euler())
        window, model = case.window, case.model
        derivs = case.stepper.derivatives()
        assert derivs.off_diag is not None and derivs.param is not None
        stage = window.unknown.blocks[0][:, 0]
        time = window.time() + window.deltat()
        mass = model.mass_matrix().as_matrix()
        bkd.assert_allclose(
            _single_block_matrix(bkd, case.stepper.diag(window).apply, 2),
            mass - window.deltat() * model.jacobian(stage, time),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            _single_block_matrix(bkd, derivs.off_diag(window, 1).apply, 2),
            -mass,
            rtol=1e-12,
        )
        bkd.assert_allclose(
            _single_block_matrix(bkd, derivs.param(window).apply, 4),
            -window.deltat() * model.param_jacobian(stage, time),
            rtol=1e-12,
        )

    def test_forward_euler(self, bkd: Backend[Array]) -> None:
        case = lotka_volterra_step(bkd, forward_euler())
        window, model = case.window, case.model
        derivs = case.stepper.derivatives()
        assert derivs.off_diag is not None and derivs.param is not None
        state = window.past_state(1)[:, 0]
        mass = model.mass_matrix().as_matrix()
        bkd.assert_allclose(
            _single_block_matrix(bkd, case.stepper.diag(window).apply, 2),
            mass,
            rtol=1e-12,
        )
        bkd.assert_allclose(
            _single_block_matrix(bkd, derivs.off_diag(window, 1).apply, 2),
            -mass - window.deltat() * model.jacobian(state, window.time()),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            _single_block_matrix(bkd, derivs.param(window).apply, 4),
            -window.deltat() * model.param_jacobian(state, window.time()),
            rtol=1e-12,
        )


class TestStepMap:
    @pytest.mark.parametrize("make", _TABLEAU_MAKERS, ids=TABLEAU_IDS)
    def test_tangent_and_adjoint_are_the_textbook_step_map(
        self, bkd: Backend[Array], make: Callable[[], ButcherTableau]
    ) -> None:
        tableau, deltat = make(), 0.1
        stepper = rk_stepper(tableau, dense_mass_form_system(LinearMassModel(bkd)))
        state = bkd.asarray([[1.0], [-0.5], [2.0]])
        times = (0.3 + deltat, 0.3)
        window = StepWindow(times, (state,), stepper.step((state,), times))
        derivs = stepper.derivatives()
        assert derivs.off_diag is not None
        off_diag = derivs.off_diag(window, 1)
        handle = stepper.diag_handle(window)
        transposed = handle.transpose()
        assert transposed is not None
        readout = stepper.readout()
        phi = bkd.asarray(linear_step_map(tableau, deltat))
        eye = bkd.eye(3)
        # Tangent: dy_{n+1}/dy_n = -E diag^{-1} off_diag, column by column.
        tangent = readout.apply(handle.apply(off_diag.apply(BlockVector((eye,)))))
        bkd.assert_allclose(-tangent.blocks[0], phi, rtol=1e-11, atol=1e-13)
        # Adjoint: its transpose, -off_diag^T diag^{-T} E^T.
        adjoint = off_diag.apply_transpose(
            transposed.apply(readout.apply_transpose(BlockVector((eye,))))
        )
        bkd.assert_allclose(-adjoint.blocks[0], phi.T, rtol=1e-11, atol=1e-13)


class TestCheckerPasses:
    @pytest.mark.parametrize("make", _TABLEAU_MAKERS, ids=TABLEAU_IDS)
    def test_every_tableau(
        self, bkd: Backend[Array], make: Callable[[], ButcherTableau]
    ) -> None:
        case = lotka_volterra_step(bkd, make())
        report = backend_checker(case, bkd).check(case.window, case.adj, case.param)
        assert "param_param" in report.fd_errors
        assert report.failures() == []
