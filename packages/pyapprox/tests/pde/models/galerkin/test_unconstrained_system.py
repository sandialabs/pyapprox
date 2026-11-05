r"""The unconstrained Galerkin system, and the boundary flux read from it.

The heat problem of ``tests._helpers.galerkin_heat`` (1D, P1 on 10
cells, log-KLE diffusivity at p = (0.4, -0.3, 0.2), forcing (1 + x)(1 +
sin 5t), u(0) = 0, moving right end 0.5 sin 4t), solved on the new path:
the adapter's ``mass_form_system()`` (neutralized rows) under the tableau
stepper with the essential rows pinned at the step end, dt = 0.002 to
T = 0.1, Newton tolerance 1e-13.

- The unconstrained right-hand side is the spatial residual as
  assembled: its value, Jacobian and parameter Jacobian equal the
  adapter's off the essential rows, and differ on them; its Jacobian is
  declared at the adapter's level; its bundle passes the residual
  checker; one ``set_param`` on either system moves both versions.
- The flux, ``StepResidualTerm`` of the unconstrained stepper with the
  end entries of the last block, summed over both ends and every step,
  equals the content change minus the source integrated by the method's
  own stages, :math:`\int u_h(T) - \int u_h(0) - \sum_n \Delta t \sum_i
  b_i\, 1.5 (1 + \sin 5(t_n + c_i \Delta t))`, to 1e-10 relative, for
  backward and forward Euler, Crank-Nicolson, implicit midpoint, Heun,
  RK4 and SDIRK3, and for forward Euler with a lumped mass. None of the
  balance's terms is assembled. Wrong-model control: the same term on
  the solved stepper reads the pinned rows' residual, zero, and misses
  the balance.
- Its gradient and Hessian-vector product through the solved stepper's
  sweeps equal the old model's (``GalerkinTransientForwardModel`` with
  ``TimeIntegratedEssentialBoundaryFluxFunctional``) to 1e-9 for
  backward Euler and Crank-Nicolson, whose trajectories the paths share.
- The term passes the term checker for backward Euler and Heun over five
  steps.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.ode.integration import Integrator
from pyapprox.ode.observation_terms import StateEntries
from pyapprox.ode.pinned_rows import RowPinning
from pyapprox.ode.protocols import DifferentiableStepperProtocol
from pyapprox.ode.residual_derivatives_check import ODEResidualBundleChecker
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.ode.runge_kutta import RungeKuttaStepper
from pyapprox.ode.step_plan import GivenSizes
from pyapprox.ode.step_residual_terms import LastBlock, StepResidualTerm
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
from pyapprox.ode.trajectory_derivatives import (
    ResidualInitialState,
    TrajectoryDerivatives,
)
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.ode.trajectory_terms_check import TrajectoryTermChecker
from pyapprox.pde.galerkin.time_integration import skfem_discretization
from pyapprox.pde.models.galerkin import (
    GalerkinParameterizedResidual,
    GalerkinTransientForwardModel,
    TimeIntegratedEssentialBoundaryFluxFunctional,
)
from pyapprox.pde.models.galerkin.physics_adapter import (
    GalerkinPhysicsToODEResidualWithSetParamAdapter,
    create_galerkin_physics_ode_residual,
)
from pyapprox.pde.models.galerkin.unconstrained_system import (
    GalerkinUnconstrainedRightHandSide,
    galerkin_unconstrained_system,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import LinearizationPoint
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.galerkin_heat import (
    HEAT_PARAM,
    MovingEndHeat,
    heat_source_total,
)
from tests._helpers.ode.runge_kutta_cases import backend_sample

_Np = NDArray[Any]
_DELTAT, _FINAL_TIME = 0.002, 0.1
_TIME = 0.03
_POINT = LinearizationPoint(0, ("test",))

_BALANCED = [
    backward_euler,
    forward_euler,
    crank_nicolson,
    implicit_midpoint,
    heun,
    classical_rk4,
    alexander_sdirk3,
]
_IDS = ["BE", "FE", "CN", "IM", "Heun", "RK4", "SDIRK3"]


def _newton() -> Newton[_Np, _Np]:
    return Newton(atol=1e-13, rtol=0.0, maxiters=20)


class _Case:
    """The solved and unconstrained systems over one discretization and
    one parameter state, the solved trajectory, and the flux term."""

    def __init__(
        self,
        bkd: NumpyBkd,
        tableau: ButcherTableau,
        lumped: bool = False,
        final_time: float = _FINAL_TIME,
    ) -> None:
        self.bkd, self.tableau = bkd, tableau
        self.heat = MovingEndHeat(bkd)
        self.discretization = skfem_discretization(self.heat.system, lumped)
        self.adapter: GalerkinPhysicsToODEResidualWithSetParamAdapter[_Np, Any] = (
            create_galerkin_physics_ode_residual(
                self.discretization, ScipySparseLU(), self.heat.parameterization
            )
        )
        self.adapter.set_param(HEAT_PARAM[:, 0])
        implicit = not tableau.is_explicit()
        self.solver = RungeKuttaStepper(
            tableau,
            self.adapter.mass_form_system(),
            _newton() if implicit else None,
            RowPinning(self.adapter.pinned_rows()),
        )
        self.unconstrained = RungeKuttaStepper(
            tableau,
            galerkin_unconstrained_system(
                self.discretization,
                ScipySparseLU(),
                self.adapter.parameter_state(),
            ),
            _newton() if implicit else None,
        )
        nsteps = round(final_time / _DELTAT)
        self.steps = GivenSizes.from_times([_DELTAT * n for n in range(nsteps + 1)])
        self.store = self.solve()

    def solve(self) -> InMemoryTrajectoryStore[_Np]:
        store: InMemoryTrajectoryStore[_Np] = InMemoryTrajectoryStore()
        Integrator(self.solver).solve(
            self.bkd.reshape(self.bkd.asarray(self.heat.init_state), (-1, 1)),
            self.steps,
            store,
        )
        return store

    def window(self, step: int) -> StepWindow[_Np]:
        return self.store.window(step, min(self.solver.depth(), step + 1))

    def flux(
        self, stepper: Optional[DifferentiableStepperProtocol[_Np, _Np]] = None
    ) -> StepResidualTerm[_Np, _Np]:
        """Both ends' entries of the last block of ``stepper``'s residual,
        the unconstrained stepper's by default."""
        stepper = self.unconstrained if stepper is None else stepper
        ends = StateEntries(
            self.bkd,
            self.heat.system.nstates(),
            [int(dof) for dof in self.heat.ends],
        )
        return StepResidualTerm(stepper, LastBlock(ends, stepper.residual_space()))

    def functional(self) -> TrajectoryFunctional[_Np, _Np]:
        """Each end's flux, summed over the steps: two entries, totaled by
        unit weights."""
        return TrajectoryFunctional.from_terms(self.bkd, 2, step_terms=(self.flux(),))

    def total(self, term: StepResidualTerm[_Np, _Np]) -> float:
        return sum(
            float(np.sum(term.value(step, self.window(step))))
            for step in range(self.store.nsteps())
        )

    def balance(self) -> float:
        """Content change minus the source integrated by the stages."""
        store, tableau = self.store, self.tableau
        source = 0.0
        for step in range(store.nsteps()):
            start = store.time(step)
            deltat = store.time(step + 1) - start
            source += deltat * sum(
                b * heat_source_total(start + c * deltat)
                for b, c in zip(tableau.b, tableau.c)
            )
        last = store.state(store.nsteps())
        return self.heat.content(last) - self.heat.content(store.state(0)) - source


class TestTheRightHandSide:
    def test_it_is_the_assembled_residual(self, numpy_bkd: NumpyBkd) -> None:
        """Off the essential rows the adapter's neutralized f equals the
        raw residual; on them the adapter carries g_dot, the raw one the
        assembled rows."""
        bkd = numpy_bkd
        case = _Case(bkd, backward_euler(), final_time=_DELTAT)
        raw = case.unconstrained.system().rhs
        state = bkd.asarray(np.linspace(0.1, 0.9, case.heat.system.nstates()))
        interior = np.setdiff1d(np.arange(len(state)), case.heat.ends)
        unconstrained, neutralized = raw(state, _TIME), case.adapter(state, _TIME)
        bkd.assert_allclose(
            unconstrained,
            case.heat.system.spatial_operator().spatial_residual(state, _TIME),
        )
        bkd.assert_allclose(unconstrained[interior], neutralized[interior])
        bkd.assert_allclose(
            neutralized[case.heat.ends],
            bkd.asarray([0.0, 2.0 * np.cos(4.0 * _TIME)]),
        )
        gap = unconstrained[case.heat.ends] - neutralized[case.heat.ends]
        assert float(np.max(np.abs(gap))) > 1e-2

    def test_its_jacobians_keep_the_essential_rows(self, numpy_bkd: NumpyBkd) -> None:
        """The state and parameter Jacobians equal the adapter's off the
        essential rows and are not zero on them; the state Jacobian is
        declared at the adapter's level."""
        bkd = numpy_bkd
        case = _Case(bkd, backward_euler(), final_time=_DELTAT)
        raw = case.unconstrained.system().rhs.derivatives()
        solved = case.adapter.derivatives()
        assert raw.jacobian is not None and solved.jacobian is not None
        assert repr(raw.jacobian) == repr(solved.jacobian)
        assert raw.param is not None and solved.param is not None
        state = bkd.asarray(np.linspace(0.1, 0.9, case.heat.system.nstates()))
        rng = np.random.default_rng(3)
        direction = bkd.asarray(rng.normal(size=(len(state), 1)))
        param_direction = bkd.asarray(rng.normal(size=(3, 1)))
        interior = np.setdiff1d(np.arange(len(state)), case.heat.ends)
        for applied, zeroed in [
            (
                raw.jacobian(state, _TIME, _POINT).apply(direction),
                solved.jacobian(state, _TIME, _POINT).apply(direction),
            ),
            (
                raw.param(state, _TIME).apply(param_direction),
                solved.param(state, _TIME).apply(param_direction),
            ),
        ]:
            bkd.assert_allclose(applied[interior], zeroed[interior])
            bkd.assert_allclose(zeroed[case.heat.ends], bkd.zeros((2, 1)))
            assert float(np.max(np.abs(applied[case.heat.ends]))) > 1e-3

    def test_its_bundle_passes_the_residual_checker(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        case = _Case(bkd, backward_euler(), final_time=_DELTAT)
        raw = case.unconstrained.system().rhs
        assert isinstance(raw, GalerkinUnconstrainedRightHandSide)
        nstates = case.heat.system.nstates()
        rng = np.random.default_rng(5)
        report = ODEResidualBundleChecker(raw).check(
            bkd.asarray(np.linspace(0.1, 0.9, nstates)),
            _TIME,
            adj=bkd.asarray(rng.normal(size=nstates)),
            param=bkd.asarray(HEAT_PARAM[:, 0]),
        )
        assert report.failures() == []

    def test_one_set_param_moves_both_systems(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        case = _Case(bkd, backward_euler(), final_time=_DELTAT)
        raw = case.unconstrained.system().rhs
        state = bkd.asarray(np.linspace(0.1, 0.9, case.heat.system.nstates()))
        before = raw(state, _TIME)
        version = case.solver.parameters_version()
        assert case.unconstrained.parameters_version() == version
        case.adapter.set_param(bkd.asarray([0.1, 0.2, -0.1]))
        assert case.solver.parameters_version() == version + 1
        assert case.unconstrained.parameters_version() == version + 1
        assert float(np.max(np.abs(raw(state, _TIME) - before))) > 1e-3
        assert isinstance(raw, GalerkinUnconstrainedRightHandSide)
        raw.set_param(bkd.asarray(HEAT_PARAM[:, 0]))
        assert case.adapter.parameters_version() == version + 2
        bkd.assert_allclose(raw(state, _TIME), before)

    def test_without_parameters(self, numpy_bkd: NumpyBkd) -> None:
        heat = MovingEndHeat(numpy_bkd)
        raw: GalerkinUnconstrainedRightHandSide[_Np, Any] = (
            GalerkinUnconstrainedRightHandSide(skfem_discretization(heat.system))
        )
        assert raw.parameters_version() == 0
        assert raw.derivatives().param is None
        assert raw.derivatives().jacobian is not None
        with pytest.raises(ValueError, match="without parameters"):
            raw.set_param(numpy_bkd.asarray(HEAT_PARAM[:, 0]))

    def test_refusals(self, numpy_bkd: NumpyBkd) -> None:
        heat, other = MovingEndHeat(numpy_bkd), MovingEndHeat(numpy_bkd)
        discretization = skfem_discretization(heat.system)
        with pytest.raises(TypeError, match="GalerkinStackDiscretization"):
            GalerkinUnconstrainedRightHandSide(heat.system)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="ParameterState"):
            GalerkinUnconstrainedRightHandSide(
                discretization, heat.parameterization  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="does not hold"):
            GalerkinUnconstrainedRightHandSide(
                discretization, ParameterState(other.parameterization)
            )


class TestTheFlux:
    @pytest.mark.parametrize("make", _BALANCED, ids=_IDS)
    def test_it_balances_content_and_source(
        self, numpy_bkd: NumpyBkd, make: Callable[[], ButcherTableau]
    ) -> None:
        case = _Case(numpy_bkd, make())
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([case.total(case.flux())]),
            numpy_bkd.asarray([case.balance()]),
            rtol=1e-10,
        )

    def test_it_balances_with_a_lumped_mass(self, numpy_bkd: NumpyBkd) -> None:
        """Lumping keeps the mass's row sums, so the content is still
        :math:`\\int u_h`."""
        case = _Case(numpy_bkd, forward_euler(), lumped=True)
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([case.total(case.flux())]),
            numpy_bkd.asarray([case.balance()]),
            rtol=1e-10,
        )

    def test_the_solved_systems_residual_misses_it(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Wrong-model control: the solved stepper's residual vanishes on
        the pinned rows, so the same term there reads no flux."""
        case = _Case(numpy_bkd, backward_euler())
        pinned = case.total(case.flux(case.solver))
        assert abs(pinned) < 1e-10
        assert abs(case.balance() - pinned) > 1e-2


class _OldModel:
    """The old path's model of the total end flux, at the same steps."""

    def __init__(self, heat: MovingEndHeat, method: str) -> None:
        bkd = NumpyBkd()
        system = heat.system
        functional = TimeIntegratedEssentialBoundaryFluxFunctional(
            GalerkinParameterizedResidual(
                system.spatial_operator(), heat.parameterization
            ),
            system,
            bkd.asarray(heat.ends),
            bkd.ones((1, 2)),
        )
        self.model = GalerkinTransientForwardModel(
            skfem_discretization(system),
            ScipySparseLU(),
            heat.parameterization,
            bkd.asarray(heat.init_state),
            TimeIntegrationConfig(
                method=method,
                init_time=0.0,
                final_time=_FINAL_TIME,
                deltat=_DELTAT,
                newton_tol=1e-13,
                newton_maxiter=20,
                verbosity=0,
            ),
            functional=functional,
        )


_DIRECTION = np.array([[0.7], [-1.1], [0.4]])


class TestAgainstTheOldFlux:
    @pytest.mark.parametrize(
        "make, method",
        [(backward_euler, "backward_euler"), (crank_nicolson, "crank_nicolson")],
        ids=["BE", "CN"],
    )
    def test_value_gradient_and_hvp(
        self,
        numpy_bkd: NumpyBkd,
        make: Callable[[], ButcherTableau],
        method: str,
    ) -> None:
        bkd = numpy_bkd
        case = _Case(bkd, make())
        functional = case.functional()
        sweeps: TrajectoryDerivatives[_Np, _Np] = TrajectoryDerivatives(
            case.solver,
            initial_state=ResidualInitialState(case.adapter, BackendColumnForm(bkd)),
        )
        value = case.total(case.flux())
        both_ends = bkd.ones((2, 1))
        gradient = sweeps.gradient(case.store, functional, weights=both_ends)
        hvp = sweeps.hvp(
            case.store, functional, bkd.asarray(_DIRECTION), weights=both_ends
        )
        old = _OldModel(case.heat, method).model
        param = bkd.asarray(HEAT_PARAM)
        derivatives = old.derivatives()
        assert derivatives.jacobian is not None and derivatives.hvp is not None
        bkd.assert_allclose(bkd.asarray([value]), old(param)[0], rtol=1e-9)
        bkd.assert_allclose(gradient, derivatives.jacobian(param).T, rtol=1e-9)
        bkd.assert_allclose(
            hvp, derivatives.hvp(param, bkd.asarray(_DIRECTION)), rtol=1e-9
        )


class TestConformance:
    @pytest.mark.parametrize("make", [backward_euler, heun], ids=["BE", "Heun"])
    def test_the_term_passes_the_checker(
        self, numpy_bkd: NumpyBkd, make: Callable[[], ButcherTableau]
    ) -> None:
        bkd = numpy_bkd
        case = _Case(bkd, make(), final_time=5 * _DELTAT)

        def set_param(param: _Np) -> None:
            case.adapter.set_param(param[:, 0])

        checker: TrajectoryTermChecker[_Np, _Np] = TrajectoryTermChecker(
            case.unconstrained, bkd, backend_sample(bkd), set_param
        )
        report = checker.check(
            case.functional(),
            case.store,
            weights=bkd.asarray([[1.0], [-0.5]]),
            param=bkd.asarray(HEAT_PARAM),
        )
        assert report.failures() == []
