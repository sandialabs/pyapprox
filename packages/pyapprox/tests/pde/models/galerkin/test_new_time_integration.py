r"""The Galerkin heat problem on the new time integration, against the old.

1D heat equation on [0, 1], P1 on 10 cells, diffusivity exp(W p) from a
three-term log-KLE at p = (0.4, -0.3, 0.2), forcing (1 + x)(1 + sin 5t),
u(0) = 0 and a moving right end 0.5 sin 4t (its rate supplied), from
0.3 sin(pi x), dt = 0.001 to T = 0.02, Newton tolerance 1e-13, the
symmetric row treatment. The essential rows are handled at the stored
states by one of the two strategies that need no row replacement inside
the step:

- B, the mass-consistent projection;
- C, the essential pin.

(Strategy A, the BC wrapper pinning each step's end inside the solve,
is rebuilt on the new path as the step-end row transform.)

For each strategy and each method both paths have (backward and forward
Euler, Crank-Nicolson, implicit midpoint, Heun), the new path (the
adapter's ``mass_form_system()`` under the tableau stepper,
``Integrator`` with the constraint, ``TrajectoryDerivatives``) must
reproduce the old one (``TimeIntegrator`` with the tiered stepper and the
same constraint, its adjoint): the trajectory to 1e-10 and the gradient
and Hessian-vector product of the value at the DOF next to the moving end
to 1e-9, relative.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Callable, List, Literal

import numpy as np
from numpy.typing import NDArray

from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.integration import Integrator, ResidualFormConstraint
from pyapprox.ode.operator.time_adjoint_hvp import TimeAdjointOperatorWithHVP
from pyapprox.ode.protocols import StateConstraintProtocol
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.ode.runge_kutta import RungeKuttaStepper
from pyapprox.ode.stepper_table import create_stepper
from pyapprox.ode.tableau import (
    ButcherTableau,
    backward_euler,
    crank_nicolson,
    forward_euler,
    heun,
    implicit_midpoint,
)
from pyapprox.ode.trajectory_derivatives import (
    ResidualInitialState,
    TrajectoryDerivatives,
)
from pyapprox.ode.trajectory_functional import (
    BackendSecondOrderTrajectoryFunctional,
    BackendTrajectoryFunctional,
)
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
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
from pyapprox.pde.galerkin.boundary import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration import (
    SymmetricNeutralizedRows,
    skfem_discretization,
)
from pyapprox.pde.galerkin.time_integration.state_constraints import (
    EssentialPin,
    MassConsistentProjection,
)
from pyapprox.pde.models.galerkin.physics_adapter import (
    GalerkinPhysicsToODEResidualWithSetParamAdapter,
    create_galerkin_physics_ode_residual,
)
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.newton import NewtonSolver
from pyapprox.util.rootfinding.nonlinear import Newton

_Np = NDArray[Any]
_NPARAMS = 3
_PARAM = np.array([[0.4], [-0.3], [0.2]])
_DELTAT, _FINAL_TIME = 0.001, 0.02
_Strategy = Literal["projection", "pin"]

# The old path's stepper name for each tableau, for the same method.
_METHODS: List[tuple[Callable[[], ButcherTableau], str]] = [
    (backward_euler, "backward_euler"),
    (forward_euler, "forward_euler"),
    (crank_nicolson, "crank_nicolson"),
    (implicit_midpoint, "implicit_midpoint"),
    (heun, "heun"),
]


def _forcing(x: _Np, t: float) -> _Np:
    value: _Np = (1.0 + x[0]) * (1.0 + np.sin(5.0 * t))
    return value


def _right_value(x: _Np, t: float) -> _Np:
    value: _Np = np.full(x.shape[1], 0.5 * np.sin(4.0 * t))
    return value


def _right_rate(x: _Np, t: float) -> _Np:
    value: _Np = np.full(x.shape[1], 2.0 * np.cos(4.0 * t))
    return value


class _Heat:
    """The parameterized heat system, its initial state and the DOF next
    to the moving right end."""

    def __init__(self, bkd: NumpyBkd) -> None:
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=NodalFieldDiffusion(basis),
            bkd=bkd,
            forcing=TimeDependent(_forcing),
        )
        right = BoundarySignal(
            TimeDependent(_right_value), [TimeDependent(_right_rate)]
        )
        self.system: GalerkinSystem[_Np] = compose_galerkin_system(
            physics,
            [
                DirichletBC(basis, "left", 0.0, bkd),
                DirichletBC(basis, "right", right, bkd),
            ],
        )
        x = np.asarray(basis.dof_coordinates())[0]
        modes = np.stack(
            [0.4 * np.sin((k + 1) * np.pi * x) / (k + 1) for k in range(_NPARAMS)],
            axis=1,
        )
        exp = _ExpTransform(bkd)
        self.parameterization = AdvectionDiffusionParameterization(
            physics,
            diffusivity_map=TransformedFieldMap(
                MeshKLEFieldMap(bkd, bkd.zeros((x.shape[0],)), bkd.asarray(modes)),
                exp,
                exp,
                bkd,
                transform_deriv2=exp,
            ),
            bkd=bkd,
        )
        self.init_state: _Np = 0.3 * np.sin(np.pi * x)
        interior = np.setdiff1d(
            np.arange(x.shape[0]), np.asarray(self.system.constraint_set().dofs())
        )
        self.near_right = int(interior[np.argmax(x[interior])])

    def adapter(self) -> GalerkinPhysicsToODEResidualWithSetParamAdapter[_Np, Any]:
        adapter = create_galerkin_physics_ode_residual(
            skfem_discretization(self.system),
            ScipySparseLU(),
            self.parameterization,
            row_treatment=SymmetricNeutralizedRows,
        )
        adapter.set_param(_PARAM[:, 0])
        return adapter

    def constraint(
        self,
        adapter: GalerkinPhysicsToODEResidualWithSetParamAdapter[_Np, Any],
        strategy: _Strategy,
    ) -> StateConstraintProtocol[_Np]:
        if strategy == "projection":
            return MassConsistentProjection(adapter, _NPARAMS)
        return EssentialPin(self.system, _NPARAMS)

    def functional(self) -> EndpointFunctional[_Np]:
        """The value at the DOF next to the moving end."""
        return EndpointFunctional(
            self.near_right, self.system.nstates(), _NPARAMS, NumpyBkd()
        )


class _Paths:
    """The heat problem on both paths under one strategy and method."""

    def __init__(
        self,
        bkd: NumpyBkd,
        strategy: _Strategy,
        make: Callable[[], ButcherTableau],
        old_name: str,
    ) -> None:
        heat = _Heat(bkd)
        self._heat, self._bkd = heat, bkd
        nsteps = round(_FINAL_TIME / _DELTAT)
        times = [_DELTAT * n for n in range(nsteps + 1)]
        # The new path.
        adapter = heat.adapter()
        tableau = make()
        newton: Newton[_Np, _Np] = Newton(atol=1e-13, rtol=0.0, maxiters=20)
        self._stepper = RungeKuttaStepper(
            tableau,
            adapter.mass_form_system(),
            None if tableau.is_explicit() else newton,
        )
        form = BackendColumnForm(bkd)
        constraint = ResidualFormConstraint(heat.constraint(adapter, strategy), form)
        self._store: InMemoryTrajectoryStore[_Np] = InMemoryTrajectoryStore()
        Integrator(self._stepper, constraint).solve(
            bkd.reshape(bkd.asarray(heat.init_state), (-1, 1)), times, self._store
        )
        self._sweeps: TrajectoryDerivatives[_Np, _Np] = TrajectoryDerivatives(
            self._stepper, constraint, ResidualInitialState(adapter, form)
        )
        # The old path, on its own adapter.
        old_adapter = heat.adapter()
        old_newton = NewtonSolver(create_stepper(old_name, old_adapter))
        old_newton.set_options(maxiters=20, atol=1e-13, rtol=0.0)
        self._integrator = TimeIntegrator(
            0.0,
            _FINAL_TIME,
            _DELTAT,
            old_newton,
            constraint=heat.constraint(old_adapter, strategy),
        )
        self._old_states, _ = self._integrator.solve(heat.init_state)

    def new_states(self) -> _Np:
        return np.hstack([self._store.state(n) for n in range(self._store.ntimes())])

    def old_states(self) -> _Np:
        return self._old_states

    def new_gradient(self) -> _Np:
        functional = BackendTrajectoryFunctional(
            self._heat.functional(), self._store, _PARAM
        )
        return self._sweeps.gradient(self._store, functional)

    def old_gradient(self) -> _Np:
        operator = TimeAdjointOperatorWithHVP(self._integrator, self._heat.functional())
        return np.reshape(operator.jacobian(self._heat.init_state, _PARAM), (-1, 1))

    def new_hvp(self, direction: _Np) -> _Np:
        functional = BackendSecondOrderTrajectoryFunctional(
            self._heat.functional(), self._store, _PARAM
        )
        return self._sweeps.hvp(self._store, functional, direction)

    def old_hvp(self, direction: _Np) -> _Np:
        operator = TimeAdjointOperatorWithHVP(self._integrator, self._heat.functional())
        product = operator.hvp(self._heat.init_state, _PARAM, direction)
        return np.reshape(product, (-1, 1))


_DIRECTION = np.array([[0.7], [-1.1], [0.4]])


@pytest.mark.parametrize("strategy", ["projection", "pin"])
@pytest.mark.parametrize("make, old_name", _METHODS, ids=[n for _, n in _METHODS])
class TestAgainstTheOldPath:
    def test_trajectory(
        self,
        numpy_bkd: NumpyBkd,
        strategy: _Strategy,
        make: Callable[[], ButcherTableau],
        old_name: str,
    ) -> None:
        paths = _Paths(numpy_bkd, strategy, make, old_name)
        numpy_bkd.assert_allclose(
            paths.new_states(), paths.old_states(), rtol=1e-10, atol=1e-13
        )

    def test_gradient(
        self,
        numpy_bkd: NumpyBkd,
        strategy: _Strategy,
        make: Callable[[], ButcherTableau],
        old_name: str,
    ) -> None:
        paths = _Paths(numpy_bkd, strategy, make, old_name)
        numpy_bkd.assert_allclose(paths.new_gradient(), paths.old_gradient(), rtol=1e-9)

    def test_hvp(
        self,
        numpy_bkd: NumpyBkd,
        strategy: _Strategy,
        make: Callable[[], ButcherTableau],
        old_name: str,
    ) -> None:
        paths = _Paths(numpy_bkd, strategy, make, old_name)
        numpy_bkd.assert_allclose(
            paths.new_hvp(_DIRECTION), paths.old_hvp(_DIRECTION), rtol=1e-9
        )
