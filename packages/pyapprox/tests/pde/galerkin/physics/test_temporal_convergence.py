"""Temporal convergence rate tests for Galerkin FEM time integration.

Verifies expected convergence rates for all 5 time stepping methods:
- Backward Euler: O(dt) — first order
- Crank-Nicolson: O(dt^2) — second order
- Implicit midpoint: O(dt^2) — second order
- Forward Euler: O(dt) — first order
- Heun (RK2): O(dt^2) — second order

Uses manufactured solutions that are spatially exact on P2 with nx=4
(quadratic in x), so the only error is from temporal discretization.
Time dependence is cubic (T^3) so both first- and second-order methods
see nonzero truncation error.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from pyapprox.implicitfunction.setups import KeepStateFree
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.protocols import StateConstraintProtocol
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepper_table import create_stepper
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
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration.neutralized_rows import (
    SymmetricNeutralizedRows,
)
from pyapprox.pde.galerkin.time_integration.physics_adapter import (
    GalerkinPhysicsToODEResidualAdapter,
)
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
    skfem_ode_residual,
)
from pyapprox.pde.galerkin.time_integration.state_constraints import (
    EssentialPin,
    MassConsistentProjection,
)
from pyapprox.pde.models.galerkin.transient_system import (
    galerkin_transient_system,
    solve_galerkin_transient,
)
from pyapprox.pde.models.galerkin.verification import (
    boundary_conditions,
    exact_dof_values,
    forcing,
)
from pyapprox.pde.verification.conditions import ConditionProtocol, Dirichlet
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import Divergence, TermProtocol, TimeDerivative
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.newton import NewtonSolver
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.markers import slow_test

_Np = NDArray[Any]
# The new path's tableau for each method name the old-path strategies use.
_TABLEAUS: Dict[str, Callable[[], ButcherTableau]] = {
    "backward_euler": backward_euler,
    "crank_nicolson": crank_nicolson,
    "implicit_midpoint": implicit_midpoint,
    "forward_euler": forward_euler,
    "heun": heun,
}


def _compute_convergence_rates(
    dt_values: List[float], errors: List[float]
) -> NDArray[np.floating[Any]]:
    """Compute convergence rates from dt refinement study.

    Parameters
    ----------
    dt_values : list of float
        Time step sizes (decreasing).
    errors : list of float
        Corresponding errors.

    Returns
    -------
    rates : np.ndarray
        Convergence rates between successive refinements.
    """
    dt_arr = np.array(dt_values)
    err_arr = np.array(errors)
    rates: NDArray[np.floating[Any]] = np.log(err_arr[:-1] / err_arr[1:]) / np.log(
        dt_arr[:-1] / dt_arr[1:]
    )
    return rates


def _setup(
    bkd: NumpyBkd, solution: str, nx: int = 4, diffusivity: float = 0.1
) -> Tuple[GalerkinSystem[_Np], Callable[[float], _Np]]:
    r"""The diffusion problem :math:`u_t = (D u_x)_x + f` on [0, 1], P2,
    Dirichlet at both ends, with manufactured ``solution``: its system
    and the exact DOF values at a time, shape (ndofs,)."""
    terms: List[TermProtocol] = [
        TimeDerivative("u"),
        Divergence("u", [f"-({diffusivity})*u_x"]),
    ]
    conditions: Dict[str, ConditionProtocol] = {
        "left": Dirichlet(),
        "right": Dirichlet(),
    }
    problem = ManufacturedProblem({"u": solution}, terms, Box([(0.0, 1.0)]), conditions)
    mesh = StructuredMesh1D(nx=nx, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=2)
    physics = AdvectionDiffusionReaction(
        basis=basis, diffusivity=diffusivity, bkd=bkd, forcing=forcing(problem, "u")
    )
    system = compose_galerkin_system(physics, boundary_conditions(problem, basis, "u"))

    def exact_at_time(time: float) -> _Np:
        return exact_dof_values(problem, basis, ["u"], time)

    return system, exact_at_time


def _final_time_errors(
    bkd: NumpyBkd, solution: str, method: str, dt_values: List[float]
) -> List[float]:
    """Error at t = 1 of each step size, stepped by
    ``solve_galerkin_transient`` from the exact initial state."""
    system, exact_at_time = _setup(bkd, solution)
    transient = galerkin_transient_system(skfem_discretization(system), ScipySparseLU())
    tableau = _TABLEAUS[method]()
    stepping: RungeKutta[_Np, _Np] = RungeKutta(
        tableau,
        None if tableau.is_explicit() else Newton(atol=1e-10, rtol=0.0, maxiters=20),
    )
    errors = []
    for dt in dt_values:
        store: InMemoryTrajectoryStore[_Np] = InMemoryTrajectoryStore()
        solve_galerkin_transient(
            transient,
            stepping,
            KeepStateFree(transient.solved.solver),
            FixedStep(dt, (0.0, 1.0)),
            exact_at_time(0.0)[:, None],
            store,
        )
        last = store.ntimes() - 1
        errors.append(
            float(
                np.linalg.norm(
                    store.state(last)[:, 0] - exact_at_time(store.times()[last])
                )
            )
        )
    return errors


# =========================================================================
# Test 1: Homogeneous Dirichlet BCs
# u(x,t) = (1-x)*x*(1+T^3) — P2 exact on nx=4, cubic in time
# u(0,t) = u(1,t) = 0
# =========================================================================

# Implicit methods: larger dt is stable
IMPLICIT_CONVERGENCE_CASES = [
    ("backward_euler", 1.0),
    ("crank_nicolson", 2.0),
    ("implicit_midpoint", 2.0),
]

# Explicit methods: dt must satisfy CFL
# D=0.1, h=0.25 (nx=4) → spectral radius ~80 → dt_max ≈ 0.025
EXPLICIT_CONVERGENCE_CASES = [
    ("forward_euler", 1.0),
    ("heun", 2.0),
]


@slow_test
class TestTemporalConvergenceImplicit:
    """Temporal convergence for implicit methods with homogeneous Dirichlet.

    u(x,t) = (1-x)*x*(1+T^3), D=0.1, no velocity, no reaction.
    P2 on nx=4 is spatially exact. Error is purely temporal.
    """

    @pytest.mark.parametrize(
        "method,expected_order",
        IMPLICIT_CONVERGENCE_CASES,
    )
    def test_convergence_homogeneous_dirichlet(
        self,
        numpy_bkd: NumpyBkd,
        method: str,
        expected_order: float,
    ) -> None:
        dt_values = [0.1, 0.05, 0.025]
        errors = _final_time_errors(numpy_bkd, "(1-x)*x*(1+t**3)", method, dt_values)
        rates = _compute_convergence_rates(dt_values, errors)

        min_expected = expected_order - 0.15
        assert np.all(rates > min_expected), (
            f"{method}: rates={rates}, errors={errors}, expected > {min_expected}"
        )


@slow_test
class TestTemporalConvergenceExplicit:
    """Temporal convergence for explicit methods with homogeneous Dirichlet.

    u(x,t) = (1-x)*x*(1+T^3), D=0.1, no velocity, no reaction.
    P2 on nx=4 is spatially exact. Error is purely temporal.
    CFL-constrained dt values: 0.01, 0.005, 0.0025.
    """

    @pytest.mark.parametrize(
        "method,expected_order",
        EXPLICIT_CONVERGENCE_CASES,
    )
    def test_convergence_homogeneous_dirichlet(
        self,
        numpy_bkd: NumpyBkd,
        method: str,
        expected_order: float,
    ) -> None:
        dt_values = [0.01, 0.005, 0.0025]
        errors = _final_time_errors(numpy_bkd, "(1-x)*x*(1+t**3)", method, dt_values)
        rates = _compute_convergence_rates(dt_values, errors)

        min_expected = expected_order - 0.15
        assert np.all(rates > min_expected), (
            f"{method}: rates={rates}, errors={errors}, expected > {min_expected}"
        )


# =========================================================================
# Test 2: Time-varying nonzero Dirichlet BCs
# u(x,t) = x*(1+T^3) — P2 exact on nx=4, cubic in time
# u(0,t) = 0, u(1,t) = 1+t^3 (nonzero, time-varying)
# =========================================================================


@slow_test
class TestTemporalConvergenceNonzeroDirichlet:
    """Temporal convergence with time-varying nonzero Dirichlet BCs.

    u(x,t) = x*(1+T^3), D=0.1, no velocity, no reaction.
    u(0,t) = 0, u(1,t) = 1+t^3.
    P2 on nx=4 is spatially exact. Error is purely temporal.
    Tests both BE (order 1) and CN (order 2).
    """

    @pytest.mark.parametrize(
        "method,expected_order",
        IMPLICIT_CONVERGENCE_CASES,
    )
    def test_convergence_nonzero_dirichlet(
        self,
        numpy_bkd: NumpyBkd,
        method: str,
        expected_order: float,
    ) -> None:
        dt_values = [0.1, 0.05, 0.025]
        errors = _final_time_errors(numpy_bkd, "x*(1+t**3)", method, dt_values)
        rates = _compute_convergence_rates(dt_values, errors)

        min_expected = expected_order - 0.15
        assert np.all(rates > min_expected), (
            f"{method}: rates={rates}, errors={errors}, expected > {min_expected}"
        )


# =========================================================================
# Test 3: Time-varying Dirichlet data with stiffness coupling
# u(x,t) = (x + x*(1-x))*(1+T^3) — P2 exact on nx=4, cubic in time
# u(0,t) = 0, u(1,t) = 1+t^3; u_xx = -2(1+t^3), so the boundary values
# enter the interior through the stiffness as well as the mass.
# =========================================================================

_COUPLED_SOLUTION = "(x+x*(1-x))*(1+t**3)"


class TestTemporalConvergenceCoupledDirichlet:
    """Every method keeps its order with time-varying Dirichlet data that
    reaches the interior through the stiffness, not only the mass.

    Not marked slow, so it runs on every push: it is the oracle against
    which boundary-condition strategies are compared. Implicit methods use
    dt in {0.1, 0.05, 0.025}; explicit ones the CFL-limited
    {0.01, 0.005, 0.0025} (D = 0.1, P2 on nx = 4).
    """

    @pytest.mark.parametrize(
        "method,expected_order",
        IMPLICIT_CONVERGENCE_CASES + EXPLICIT_CONVERGENCE_CASES,
    )
    def test_order(
        self, numpy_bkd: NumpyBkd, method: str, expected_order: float
    ) -> None:
        explicit = (method, expected_order) in EXPLICIT_CONVERGENCE_CASES
        dt_values = [0.01, 0.005, 0.0025] if explicit else [0.1, 0.05, 0.025]
        errors = _final_time_errors(numpy_bkd, _COUPLED_SOLUTION, method, dt_values)
        rates = _compute_convergence_rates(dt_values, errors)
        assert np.all(rates > expected_order - 0.15), (
            f"{method}: rates={rates}, errors={errors}"
        )


_Constraint = StateConstraintProtocol[NDArray[Any]]


def _projection(
    adapter: GalerkinPhysicsToODEResidualAdapter[NDArray[Any], Any],
) -> _Constraint:
    return MassConsistentProjection(adapter, 0)


def _pin(
    adapter: GalerkinPhysicsToODEResidualAdapter[NDArray[Any], Any],
) -> _Constraint:
    return EssentialPin(adapter.system(), 0)


def _strategy_errors(
    bkd: NumpyBkd,
    method: str,
    dt_values: List[float],
    constraint_of: Callable[
        [GalerkinPhysicsToODEResidualAdapter[NDArray[Any], Any]], _Constraint
    ],
) -> List[float]:
    """Error at t = 1 of each step size, stepping the symmetric row
    treatment without the BC wrapper, the stored states constrained."""
    system, exact_at_time = _setup(bkd, _COUPLED_SOLUTION)
    adapter = skfem_ode_residual(system, row_treatment=SymmetricNeutralizedRows)
    constraint = constraint_of(adapter)
    errors = []
    for dt in dt_values:
        newton = NewtonSolver(create_stepper(method, adapter))
        newton.set_options(maxiters=20, atol=1e-10, rtol=0.0)
        integrator = TimeIntegrator(0.0, 1.0, dt, newton, constraint=constraint)
        solutions, times = integrator.solve(bkd.asarray(exact_at_time(0.0)))
        u_exact = exact_at_time(float(times[-1]))
        errors.append(float(np.linalg.norm(bkd.to_numpy(solutions[:, -1]) - u_exact)))
    return errors


class TestTemporalConvergenceStrategies:
    """The symmetric row treatment keeps every method's order, with either
    the mass-consistent projection (exactly conservative for explicit
    steppers) or the plain pin: the same oracle, rates and step sizes as
    :class:`TestTemporalConvergenceCoupledDirichlet`, which covers
    today's strategy."""

    @pytest.mark.parametrize(
        "constraint_of", [_projection, _pin], ids=["projection", "pin"]
    )
    @pytest.mark.parametrize(
        "method,expected_order",
        IMPLICIT_CONVERGENCE_CASES + EXPLICIT_CONVERGENCE_CASES,
    )
    def test_order(
        self,
        numpy_bkd: NumpyBkd,
        method: str,
        expected_order: float,
        constraint_of: Callable[
            [GalerkinPhysicsToODEResidualAdapter[NDArray[Any], Any]], _Constraint
        ],
    ) -> None:
        explicit = (method, expected_order) in EXPLICIT_CONVERGENCE_CASES
        dt_values = [0.01, 0.005, 0.0025] if explicit else [0.1, 0.05, 0.025]
        errors = _strategy_errors(numpy_bkd, method, dt_values, constraint_of)
        rates = _compute_convergence_rates(dt_values, errors)
        assert np.all(rates > expected_order - 0.15), (
            f"{method}: rates={rates}, errors={errors}"
        )
