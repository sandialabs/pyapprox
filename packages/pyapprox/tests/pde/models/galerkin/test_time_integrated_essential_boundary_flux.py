"""The time-integrated essential-boundary flux: discrete conservation.

1D heat equation on [0, 1], P1 on 10 cells,

    u_t = (kappa(x) u')' + f(x, t),   kappa = exp(W p) (log-KLE),
    f = (1 + x)(1 + sin(5 t)),

with u(0, t) = 0 and a moving right end u(1, t) = 0.5 sin(4 t), from
u(x, 0) = 0.3 sin(pi x), over [0, 0.1] with dt = 0.002 (stable for
the explicit methods), at p = (0.4, -0.3, 0.2). The flux is
``time_integrated_essential_boundary_flux``, a step term summed over the
steps of a ``solve_galerkin_transient`` trajectory.

Summed over every row, each step's non-essential rows vanish at the
solution, and the interpolant of 1 annihilates the diffusion term. So
with unit weights on both ends the flux must equal, for every Runge-Kutta
method with weights b and nodes c,

    Q_total = integral u_h(T) - integral u_h(0)
              - sum_n sum_i dt b_i 1.5 (1 + sin(5 (t_n + c_i dt))),

the change of total content minus the source integrated by the method's
own rule; integral (1 + x) dx = 1.5, and the content integrals use the
trapezoid rule on the nodes, exact for P1 (and kept by a row-sum lumped
mass). None of these terms is computed by the assembly. Checked to 1e-10
for backward and forward Euler, Crank-Nicolson, implicit midpoint and
Heun, and for forward Euler with a lumped mass. Wrong-model control:
integrating the steady flux instead misses the balance by more than
1e-3.

Through ``TransientImplicitMap`` and ``ImplicitMapFunction``: the scalar
flux's Jacobian and HVP pass central finite differences along a fixed
direction for every method, implicit midpoint included (the V of second
order, judged against each curve's predicted rounding); each end's flux
as its own entry gives a Jacobian whose adjoint rows equal its tangent
columns (1e-10), and which with the weighted HVP (weights (1, -0.5))
passes the same check; a 1e-5 relative defect in the HVP fails it; the
function survives a pickle round trip. Refusals: a non-essential DOF,
weights of the wrong width, a method that is not one.
"""

import pickle
from typing import Any, Callable, List, Tuple

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.implicitfunction.implicit_map import (
    AdjointRows,
    ArrayParameters,
    ImplicitMapFunction,
    TangentColumns,
)
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.step_residual_terms import StepResidualTerm
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import (
    ButcherTableau,
    backward_euler,
    crank_nicolson,
    forward_euler,
    heun,
    implicit_midpoint,
)
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.ode.transient_implicit_map import (
    FixedTrajectoryFunctional,
    TransientImplicitMap,
)
from pyapprox.pde.boundary import BoundarySignal
from pyapprox.pde.constitutive.coefficient_functions import (
    NodalFieldDiffusion,
    TimeDependent,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.models.galerkin.essential_boundary_flux import (
    time_integrated_essential_boundary_flux,
)
from pyapprox.pde.models.galerkin.parameterized_residual import (
    GalerkinParameterizedResidual,
)
from pyapprox.pde.models.galerkin.transient_system import (
    GalerkinTransientSystem,
    galerkin_transient_system,
    solve_galerkin_transient,
)
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.adjoint_checks import (
    NumpyArray,
    central_derivatives_pass_fd,
)

_PARAM = np.array([[0.4], [-0.3], [0.2]])
_DIRECTION = np.array([[0.7], [-1.1], [0.4]])
_DELTAT, _FINAL_TIME = 0.002, 0.1
_METHODS: List[Tuple[Callable[[], ButcherTableau], str]] = [
    (backward_euler, "backward_euler"),
    (forward_euler, "forward_euler"),
    (crank_nicolson, "crank_nicolson"),
    (implicit_midpoint, "implicit_midpoint"),
    (heun, "heun"),
]
_IDS = [name for _, name in _METHODS]


def _forcing(coords: NumpyArray, time: float) -> NumpyArray:
    value: NumpyArray = (1.0 + coords[0]) * (1.0 + np.sin(5.0 * time))
    return value


def _right_value(coords: NumpyArray, time: float) -> NumpyArray:
    return np.full(coords.shape[1], 0.5 * np.sin(4.0 * time))


def _right_rate(coords: NumpyArray, time: float) -> NumpyArray:
    return np.full(coords.shape[1], 2.0 * np.cos(4.0 * time))


def _log_kle_map(
    bkd: NumpyBkd, basis: LagrangeBasis[NumpyArray]
) -> TransformedFieldMap[NumpyArray]:
    coords = bkd.to_numpy(basis.dof_coordinates())[0]
    modes = np.stack(
        [0.4 * np.sin((k + 1) * np.pi * coords) / (k + 1) for k in range(3)],
        axis=1,
    )
    kle = MeshKLEFieldMap(
        bkd, bkd.asarray(np.zeros(coords.shape[0])), bkd.asarray(modes)
    )
    exp = _ExpTransform(bkd)
    return TransformedFieldMap(kle, exp, exp, bkd, transform_deriv2=exp)


def _method(make: Callable[[], ButcherTableau]) -> RungeKutta[NumpyArray, NumpyArray]:
    tableau = make()
    return RungeKutta(
        tableau,
        None if tableau.is_explicit() else Newton(atol=1e-13, rtol=0.0, maxiters=20),
    )


class _Problem:
    """The heat problem, its parameters, both end DOFs and its new-path
    system."""

    def __init__(self, bkd: NumpyBkd, lumped_mass: bool = False) -> None:
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        self.basis = LagrangeBasis(mesh, degree=1)
        physics = AdvectionDiffusionReaction(
            basis=self.basis,
            diffusivity=NodalFieldDiffusion(self.basis),
            bkd=bkd,
            forcing=TimeDependent(_forcing),
        )
        right = BoundarySignal(
            TimeDependent(_right_value), [TimeDependent(_right_rate)]
        )
        self.system: GalerkinSystem[NumpyArray] = compose_galerkin_system(
            physics,
            [
                DirichletBC(self.basis, "left", 0.0, bkd),
                DirichletBC(self.basis, "right", right, bkd),
            ],
        )
        self.param_obj = AdvectionDiffusionParameterization(
            physics, diffusivity_map=_log_kle_map(bkd, self.basis), bkd=bkd
        )
        parameters = ParameterState(self.param_obj)
        parameters.set_param(_PARAM[:, 0])
        self.transient: GalerkinTransientSystem[NumpyArray, Any] = (
            galerkin_transient_system(
                skfem_discretization(self.system, lumped_mass),
                ScipySparseLU(),
                parameters,
            )
        )
        self.ends = bkd.concatenate(
            [self.basis.get_dofs("left"), self.basis.get_dofs("right")]
        )
        x = bkd.to_numpy(self.basis.dof_coordinates())[0]
        self.init_state = bkd.asarray(0.3 * np.sin(np.pi * x)[:, None])
        self.bkd = bkd

    def flux(
        self, method: RungeKutta[NumpyArray, NumpyArray], weights: NumpyArray
    ) -> StepResidualTerm[NumpyArray, NumpyArray]:
        return time_integrated_essential_boundary_flux(
            self.transient,
            method,
            RecomputeAlways(self.transient.unconstrained.solver),
            self.ends,
            weights,
        )

    def steps(self) -> FixedStep:
        return FixedStep(_DELTAT, (0.0, _FINAL_TIME))

    def solve(
        self, method: RungeKutta[NumpyArray, NumpyArray]
    ) -> InMemoryTrajectoryStore[NumpyArray]:
        store: InMemoryTrajectoryStore[NumpyArray] = InMemoryTrajectoryStore()
        solve_galerkin_transient(
            self.transient,
            method,
            RecomputeAlways(self.transient.solved.solver),
            self.steps(),
            self.init_state,
            store,
        )
        return store

    def content(self, column: NumpyArray) -> float:
        """integral of u_h over [0, 1] (trapezoid on the nodes, exact)."""
        x = self.bkd.to_numpy(self.basis.dof_coordinates())[0]
        order = np.argsort(x)
        values = self.bkd.to_numpy(column)[:, 0]
        return float(np.trapezoid(values[order], x[order]))

    def function(
        self,
        make: Callable[[], ButcherTableau],
        weights: NumpyArray,
    ) -> TransientImplicitMap[NumpyArray, NumpyArray, Any]:
        """The flux under ``weights`` as a transient map."""
        method = _method(make)
        functional = TrajectoryFunctional.from_terms(
            self.bkd, weights.shape[0], step_terms=(self.flux(method, weights),)
        )
        return TransientImplicitMap(
            self.transient,
            self.transient,
            method,
            RecomputeAlways(self.transient.solved.solver),
            self.steps(),
            self.init_state,
            FixedTrajectoryFunctional(functional),
        )


def _total(
    term: StepResidualTerm[NumpyArray, NumpyArray],
    store: InMemoryTrajectoryStore[NumpyArray],
) -> NumpyArray:
    """The flux summed over the stored steps, shape ``(nqoi, 1)``."""
    total: NumpyArray = sum(
        (term.value(step, store.window(step)) for step in range(store.nsteps())),
        start=np.zeros((term.nqoi(), 1)),
    )
    return total


def _balance(
    problem: _Problem,
    tableau: ButcherTableau,
    store: InMemoryTrajectoryStore[NumpyArray],
) -> float:
    """Content change minus the source integrated by the method's rule."""
    source = sum(
        _DELTAT * b * 1.5 * (1.0 + np.sin(5.0 * (store.time(n) + c * _DELTAT)))
        for n in range(store.nsteps())
        for b, c in zip(tableau.b, tableau.c)
    )
    last = store.ntimes() - 1
    return (
        problem.content(store.state(last))
        - problem.content(store.state(0))
        - float(source)
    )


class TestDiscreteConservation:
    @pytest.mark.parametrize("make, name", _METHODS, ids=_IDS)
    def test_total_flux_balances_content_and_source(
        self, numpy_bkd: NumpyBkd, make: Callable[[], ButcherTableau], name: str
    ) -> None:
        bkd = numpy_bkd
        problem = _Problem(bkd)
        method = _method(make)
        store = problem.solve(method)
        bkd.assert_allclose(
            _total(problem.flux(method, bkd.ones((1, 2))), store),
            bkd.asarray([[_balance(problem, make(), store)]]),
            rtol=1e-10,
        )

    def test_lumped_mass_balances_with_its_own_mass(self, numpy_bkd: NumpyBkd) -> None:
        """Forward Euler with a row-sum lumped mass: the flux reads the
        lumped scheme's residual, and lumping keeps the content, so the
        balance still holds."""
        bkd = numpy_bkd
        problem = _Problem(bkd, lumped_mass=True)
        method = _method(forward_euler)
        store = problem.solve(method)
        bkd.assert_allclose(
            _total(problem.flux(method, bkd.ones((1, 2))), store),
            bkd.asarray([[_balance(problem, forward_euler(), store)]]),
            rtol=1e-10,
        )

    def test_integrated_steady_flux_breaks_conservation(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Wrong-model control: integrating the steady flux -F_E over
        time (backward Euler's rule, the step-end state) drops M u_t on
        the end rows, and misses the balance by far more than the solver
        tolerance."""
        bkd = numpy_bkd
        problem = _Problem(bkd)
        method = _method(backward_euler)
        store = problem.solve(method)
        residual = GalerkinParameterizedResidual(
            problem.system.spatial_operator(), problem.param_obj
        )
        steady_integrated = 0.0
        for n in range(1, store.ntimes()):
            rows = residual.residual(
                store.state(n)[:, 0], store.time(n), bkd.asarray(_PARAM[:, 0])
            )
            steady_integrated -= _DELTAT * bkd.to_float(bkd.sum(rows[problem.ends]))
        balanced = bkd.to_float(
            _total(problem.flux(method, bkd.ones((1, 2))), store)[0, 0]
        )
        assert abs(steady_integrated - balanced) > 1e-3


class TestRefusals:
    def test_refuses_bad_arguments(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        problem = _Problem(bkd)
        method = _method(backward_euler)
        setups = RecomputeAlways(problem.transient.unconstrained.solver)
        with pytest.raises(ValueError, match="essential"):
            time_integrated_essential_boundary_flux(
                problem.transient,
                method,
                setups,
                bkd.asarray([3], dtype=bkd.int64_dtype()),
                bkd.ones((1, 1)),
            )
        with pytest.raises(ValueError, match="weights must have shape"):
            problem.flux(method, bkd.ones((1, 3)))
        with pytest.raises(TypeError, match="SteppingMethodProtocol"):
            time_integrated_essential_boundary_flux(
                problem.transient,
                object(),  # type: ignore[arg-type]
                setups,
                problem.ends,
                bkd.ones((1, 2)),
            )


class TestFunctionDerivatives:
    """Gradients and HVPs of the total end flux with respect to the
    log-KLE diffusivity, against finite differences of full solves."""

    @pytest.mark.parametrize("make, name", _METHODS, ids=_IDS)
    def test_derivatives_match_fd(
        self, numpy_bkd: NumpyBkd, make: Callable[[], ButcherTableau], name: str
    ) -> None:
        bkd = numpy_bkd
        problem = _Problem(bkd)
        function = ImplicitMapFunction(
            problem.function(make, bkd.ones((1, 2))), ArrayParameters(bkd)
        )
        derivatives = function.derivatives()
        assert derivatives.jacobian is not None and derivatives.hvp is not None
        reports = central_derivatives_pass_fd(
            function, bkd.asarray(_PARAM), bkd.asarray(_DIRECTION)
        )
        assert len(reports) == 2
        assert all(report.passed for report in reports), reports

    def test_vector_flux_jacobian(self, numpy_bkd: NumpyBkd) -> None:
        """Each end's flux as its own entry: the tangent-column Jacobian
        passes finite differences and equals the adjoint rows."""
        bkd = numpy_bkd
        problem = _Problem(bkd)
        implicit = problem.function(backward_euler, bkd.eye(2))
        columns = ImplicitMapFunction(implicit, ArrayParameters(bkd), TangentColumns())
        rows = ImplicitMapFunction(implicit, ArrayParameters(bkd), AdjointRows())
        param = bkd.asarray(_PARAM)
        reports = central_derivatives_pass_fd(
            columns, param, bkd.asarray(_DIRECTION), bkd.asarray([[1.0], [-0.5]])
        )
        assert len(reports) == 2
        assert all(report.passed for report in reports), reports
        by_columns = columns.derivatives().jacobian
        by_rows = rows.derivatives().jacobian
        assert by_columns is not None and by_rows is not None
        bkd.assert_allclose(by_rows(param), by_columns(param), rtol=1e-10)

    def test_a_slightly_wrong_hvp_fails(self, numpy_bkd: NumpyBkd) -> None:
        """Wrong-model control for the check: the Crank-Nicolson flux with
        its HVP scaled by 1 + 1e-5 (the Jacobian kept) fails, the
        Jacobian still passes."""
        bkd = numpy_bkd
        problem = _Problem(bkd)
        function = ImplicitMapFunction(
            problem.function(crank_nicolson, bkd.ones((1, 2))), ArrayParameters(bkd)
        )
        reports = central_derivatives_pass_fd(
            _ScaledHVP(function, 1.0 + 1e-5),
            bkd.asarray(_PARAM),
            bkd.asarray(_DIRECTION),
        )
        assert reports[0].passed and not reports[1].passed


class _ScaledHVP:
    """``function`` with its HVP scaled by ``factor``."""

    def __init__(
        self, function: ImplicitMapFunction[Any, NumpyArray, NumpyArray], factor: float
    ) -> None:
        self._function, self._factor = function, factor
        derivatives = function.derivatives()
        jacobian, hvp = derivatives.jacobian, derivatives.hvp
        assert jacobian is not None and hvp is not None
        self._jacobian, self._hvp = jacobian, hvp

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()

    def nvars(self) -> int:
        return self._function.nvars()

    def nqoi(self) -> int:
        return self._function.nqoi()

    def __call__(self, samples: NumpyArray) -> NumpyArray:
        return self._function(samples)

    def derivatives(self) -> Derivatives[NumpyArray]:
        return Derivatives.second_order(jacobian=self._jacobian, hvp=self._scaled)

    def _scaled(self, sample: NumpyArray, vec: NumpyArray) -> NumpyArray:
        return self._factor * self._hvp(sample, vec)


class TestPickle:
    def test_round_trip_keeps_the_value(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        problem = _Problem(bkd)
        function = ImplicitMapFunction(
            problem.function(crank_nicolson, bkd.ones((1, 2))), ArrayParameters(bkd)
        )
        restored = pickle.loads(pickle.dumps(function))
        param = bkd.asarray(_PARAM)
        bkd.assert_allclose(restored(param), function(param))
