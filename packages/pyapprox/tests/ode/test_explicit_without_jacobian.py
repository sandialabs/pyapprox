r"""An explicit forward solve needs no Jacobian.

On :math:`M \dot y = K y` (``LinearMassModel``: 3 states, a nontrivial
mass), the same model with an empty derivatives bundle, from (1, -0.5, 2)
over ``FixedStep(0.1, (0, 0.5))``:

- Forward Euler, Heun and Adams-Bashforth 2 give trajectories equal,
  bitwise, to the model with its Jacobian: an explicit step solves with
  the mass alone.
- The steppers' derivatives bundles are empty, so the sweeps, which need
  ``off_diag``, refuse.
- Binding an implicit method (backward Euler, BDF2) refuses, naming the
  right-hand side.
"""

from typing import List

import pytest

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.ode.integration import Integrator
from pyapprox.ode.linear_multistep import LinearMultistepStepper
from pyapprox.ode.mass_form_system import dense_mass_form_system
from pyapprox.ode.multistep_methods import adams_bashforth2, bdf2
from pyapprox.ode.residual_derivatives import ODEResidualDerivatives
from pyapprox.ode.runge_kutta import RungeKuttaStepper
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.tableau import backward_euler, forward_euler, heun
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.ode.runge_kutta_cases import LinearMassModel


class _NoJacobian(LinearMassModel[Array]):
    """The same right-hand side, declaring no derivatives."""

    def derivatives(self) -> ODEResidualDerivatives[Array, DenseFormat[Array]]:
        return ODEResidualDerivatives.none()


def _trajectory(
    bkd: Backend[Array], model: LinearMassModel[Array], method: str
) -> List[Array]:
    system = dense_mass_form_system(model)
    store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
    init = bkd.asarray([[1.0], [-0.5], [2.0]])
    steps = FixedStep(0.1, (0.0, 0.5))
    setups = RecomputeAlways(system.solver)
    if method == "AB2":
        lmm = LinearMultistepStepper(adams_bashforth2(), system, setups)
        Integrator(lmm).solve(init, steps, store)
    else:
        tableau = forward_euler() if method == "FE" else heun()
        rk = RungeKuttaStepper(tableau, system, setups)
        Integrator(rk).solve(init, steps, store)
    return [store.state(n) for n in range(store.ntimes())]


class TestAnExplicitForwardSolve:
    @pytest.mark.parametrize("method", ["FE", "Heun", "AB2"])
    def test_it_matches_the_model_with_a_jacobian(
        self, bkd: Backend[Array], method: str
    ) -> None:
        without = _trajectory(bkd, _NoJacobian(bkd), method)
        with_jacobian = _trajectory(bkd, LinearMassModel(bkd), method)
        assert len(without) == len(with_jacobian) == 6
        for got, expected in zip(without, with_jacobian):
            bkd.assert_allclose(got, expected, rtol=0.0, atol=0.0)

    def test_the_steppers_have_no_derivatives(self, bkd: Backend[Array]) -> None:
        system = dense_mass_form_system(_NoJacobian(bkd))
        setups = RecomputeAlways(system.solver)
        for derivatives in (
            RungeKuttaStepper(heun(), system, setups).derivatives(),
            LinearMultistepStepper(adams_bashforth2(), system, setups).derivatives(),
        ):
            assert derivatives.off_diag is None and derivatives.param is None


class TestAnImplicitMethodRefuses:
    @pytest.mark.parametrize("which", ["RK", "LMM"])
    def test_binding_names_the_right_hand_side(
        self, bkd: Backend[Array], which: str
    ) -> None:
        system = dense_mass_form_system(_NoJacobian(bkd))
        newton: Newton[Array, Array] = Newton()
        setups = RecomputeAlways(system.solver)
        with pytest.raises(TypeError, match="_NoJacobian's derivatives bundle has"):
            if which == "RK":
                RungeKuttaStepper(backward_euler(), system, setups, newton)
            else:
                LinearMultistepStepper(bdf2(), system, setups, newton)
