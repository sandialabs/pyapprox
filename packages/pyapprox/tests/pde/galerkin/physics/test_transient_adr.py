"""Parametrized transient ADR tests (1D + 2D).

This module contains:
1. 8 parametrized 1D transient tests replicating legacy
   test_finite_elements.py test_transient_advection_diffusion_reaction
2. 4 parametrized 2D transient tests (new coverage)
3. Crank-Nicolson variants (1D + 2D)
4. Explicit integrator tests (Forward Euler + Heun, with/without advection)
5. Low-level manual Newton test using the BC-enforcing residual wrapper

Problems are ``pde.verification`` manufactured problems. Every case but
the manual-Newton one steps ``solve_galerkin_transient``: backward Euler
or Crank-Nicolson, and the explicit methods with CFL-constrained dt.
Errors are relative Euclidean norms of the final DOF vector.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.ode.implicit_steppers import (
    BackwardEulerHVP,
    CrankNicolsonHVP,
)
from pyapprox.ode.step_context import StepContext
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import (
    ButcherTableau,
    backward_euler,
    crank_nicolson,
    forward_euler,
    heun,
)
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration.bc_time_residual_adapter import (
    create_galerkin_bc_enforcing_residual,
)
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
    skfem_ode_residual,
)
from pyapprox.pde.models.galerkin.transient_system import (
    galerkin_transient_system,
    solve_galerkin_transient,
)
from pyapprox.pde.models.galerkin.verification import (
    boundary_conditions,
    exact_dof_values,
    forcing,
    state_law,
    vector_field,
)
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import (
    Divergence,
    Source,
    TermProtocol,
    TimeDerivative,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.galerkin_verification import coded
from tests._helpers.markers import slow_test

_Np = NDArray[Any]
_INTERVAL = Box([(0.0, 1.0)])
_SQUARE = Box([(0.0, 1.0)] * 2)
_SOLUTION_1D = "(1-x)*x*(1+t)"
_SOLUTION_2D = "(1-x)*x*(1-y)*y*(1+t)"

# =========================================================================
# Helpers
# =========================================================================


def _setup(
    bkd: NumpyBkd,
    box: Box,
    solution: str,
    codes: str,
    nx: int,
    diffusivity: float,
    velocity: Optional[Sequence[str]] = None,
    reaction: Optional[str] = None,
) -> Tuple[GalerkinSystem[_Np], Callable[[float], _Np]]:
    r"""Transient ADR :math:`u_t + \nabla\cdot(-D\nabla u) + v\cdot\nabla u
    - R(u) = f` on ``box`` with P2 on ``nx`` cells per side, the sides'
    conditions by code (D, N, R with alpha 1): its system and the exact
    DOF values at a time, shape (ndofs,)."""
    coords = ("x", "y")[: box.ndim()]
    terms: List[TermProtocol] = [
        TimeDerivative("u"),
        Divergence("u", [f"-({diffusivity})*u_{c}" for c in coords]),
    ]
    if velocity is not None:
        transport = " + ".join(f"({v})*u_{c}" for v, c in zip(velocity, coords))
        terms.append(Source("u", transport))
    if reaction is not None:
        terms.append(Source("u", f"-({reaction})"))
    problem = ManufacturedProblem({"u": solution}, terms, box, coded(codes, box))
    mesh = (
        StructuredMesh1D(nx=nx, bounds=(0.0, 1.0), bkd=bkd)
        if box.ndim() == 1
        else StructuredMesh2D(nx=nx, ny=nx, bounds=[(0.0, 1.0)] * 2, bkd=bkd)
    )
    basis = LagrangeBasis(mesh, degree=2)
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=diffusivity,
        bkd=bkd,
        velocity=None if velocity is None else vector_field(problem, velocity),
        reaction=None if reaction is None else state_law(problem, reaction, "u"),
        forcing=forcing(problem, "u"),
    )
    system = compose_galerkin_system(physics, boundary_conditions(problem, basis, "u"))

    def exact_at_time(time: float) -> _Np:
        return exact_dof_values(problem, basis, ["u"], time)

    return system, exact_at_time


def _relative_error(
    system: GalerkinSystem[_Np],
    exact_at_time: Callable[[float], _Np],
    make: Callable[[], ButcherTableau],
    deltat: float,
    final_time: float,
) -> float:
    """Relative error of the final state stepped by
    ``solve_galerkin_transient`` from the exact initial state."""
    transient = galerkin_transient_system(skfem_discretization(system), ScipySparseLU())
    tableau = make()
    method: RungeKutta[_Np, _Np] = RungeKutta(
        tableau,
        None if tableau.is_explicit() else Newton(atol=1e-10, rtol=0.0, maxiters=20),
    )
    store: InMemoryTrajectoryStore[_Np] = InMemoryTrajectoryStore()
    solve_galerkin_transient(
        transient,
        method,
        RecomputeAlways(transient.solved.solver),
        FixedStep(deltat, (0.0, final_time)),
        exact_at_time(0.0)[:, None],
        store,
    )
    last = store.ntimes() - 1
    exact = exact_at_time(store.times()[last])
    return float(
        np.linalg.norm(store.state(last)[:, 0] - exact) / np.linalg.norm(exact)
    )


# =========================================================================
# 1D Transient ADR Cases
# =========================================================================

# Format: (name, boundary codes, velocity or None, reaction or None)
# Fixed: u = (1-x)*x*(1+t) on [0, 1], D = 4, P2 on 32 cells.
_V_1D = ("(1+x)/10",)
TRANSIENT_ADR_1D_CASES: List[
    Tuple[str, str, Optional[Tuple[str, ...]], Optional[str]]
] = [
    ("1d_DD_noV_noR", "DD", None, None),
    ("1d_DD_noV_R", "DD", None, "u**2"),
    ("1d_DD_V_noR", "DD", _V_1D, None),
    ("1d_DD_V_R", "DD", _V_1D, "u**2"),
    ("1d_DN_noV_noR", "DN", None, None),
    ("1d_DN_noV_R", "DN", None, "u**2"),
    ("1d_DN_V_noR", "DN", _V_1D, None),
    ("1d_DN_V_R", "DN", _V_1D, "u**2"),
]


class TestTransientADR1D:
    """Parametrized 1D transient ADR tests with P2 + backward Euler.

    Replicates legacy test_finite_elements.py
    test_transient_advection_diffusion_reaction (8 cases).
    """

    @pytest.mark.parametrize(
        "name,codes,velocity,reaction",
        TRANSIENT_ADR_1D_CASES,
    )
    def test_transient_adr_1d(
        self,
        numpy_bkd: NumpyBkd,
        name: str,
        codes: str,
        velocity: Optional[Tuple[str, ...]],
        reaction: Optional[str],
    ) -> None:
        """Test transient ADR with manufactured solution."""
        system, exact_at_time = _setup(
            numpy_bkd, _INTERVAL, _SOLUTION_1D, codes, 32, 4.0, velocity, reaction
        )
        rel_error = _relative_error(system, exact_at_time, backward_euler, 1.0, 5.0)
        # Nonlinear reaction with Neumann BCs has slightly larger error
        # due to spatial discretization of u^2 term
        tol = 2e-6 if reaction is not None and "N" in codes else 1e-6
        assert rel_error < tol


# =========================================================================
# 2D Transient ADR Cases
# =========================================================================

# Format: (name, boundary codes, velocity or None)
# Fixed: u = (1-x)*x*(1-y)*y*(1+t) on [0, 1]^2, D = 4, P2 on 16 x 16,
#        no reaction.
_V_2D = ("(1+x)/10", "(1+y)/10")
TRANSIENT_ADR_2D_CASES: List[Tuple[str, str, Optional[Tuple[str, ...]]]] = [
    ("2d_DDDD_noV", "DDDD", None),
    ("2d_DDDD_V", "DDDD", _V_2D),
    ("2d_DRDN_noV", "DRDN", None),
    ("2d_DRDN_V", "DRDN", _V_2D),
]


class TestTransientADR2D:
    """Parametrized 2D transient ADR tests with P2 + backward Euler."""

    @pytest.mark.parametrize(
        "name,codes,velocity",
        TRANSIENT_ADR_2D_CASES,
    )
    def test_transient_adr_2d(
        self,
        numpy_bkd: NumpyBkd,
        name: str,
        codes: str,
        velocity: Optional[Tuple[str, ...]],
    ) -> None:
        """Test 2D transient ADR with manufactured solution."""
        system, exact_at_time = _setup(
            numpy_bkd, _SQUARE, _SOLUTION_2D, codes, 16, 4.0, velocity
        )
        rel_error = _relative_error(system, exact_at_time, backward_euler, 1.0, 5.0)
        assert rel_error < 1e-6


# =========================================================================
# 1D Transient ADR with Crank-Nicolson
# =========================================================================

TRANSIENT_ADR_1D_CN_CASES: List[Tuple[str, Optional[Tuple[str, ...]]]] = [
    ("1d_DD_noV_noR_CN", None),
    ("1d_DD_V_noR_CN", _V_1D),
]


class TestTransientADR1D_CN:
    """1D transient ADR tests with P2 + Crank-Nicolson."""

    @pytest.mark.parametrize("name,velocity", TRANSIENT_ADR_1D_CN_CASES)
    def test_transient_adr_1d_cn(
        self,
        numpy_bkd: NumpyBkd,
        name: str,
        velocity: Optional[Tuple[str, ...]],
    ) -> None:
        system, exact_at_time = _setup(
            numpy_bkd, _INTERVAL, _SOLUTION_1D, "DD", 32, 4.0, velocity
        )
        rel_error = _relative_error(system, exact_at_time, crank_nicolson, 1.0, 5.0)
        assert rel_error < 1e-6


# =========================================================================
# 2D Transient ADR with Crank-Nicolson
# =========================================================================


class TestTransientADR2D_CN:
    """2D transient ADR tests with P2 + Crank-Nicolson."""

    def test_transient_adr_2d_cn(self, numpy_bkd: NumpyBkd) -> None:
        """All sides Dirichlet, no velocity."""
        system, exact_at_time = _setup(
            numpy_bkd, _SQUARE, _SOLUTION_2D, "DDDD", 16, 4.0
        )
        rel_error = _relative_error(system, exact_at_time, crank_nicolson, 1.0, 5.0)
        assert rel_error < 1e-6


# =========================================================================
# 1D Transient ADR with Explicit Methods (Forward Euler, Heun)
# =========================================================================

# Format: (name, tableau, velocity or None)
# Fixed: u = (1-x)*x*(1+t) on [0, 1], D = 0.1, Dirichlet both ends,
#        P2 on 4 cells (exact in space), no reaction.
# D=0.1, h=0.25 → CFL diffusive: dt < h²/(2D) = 0.3125
# With |v|=0.5 → CFL advective: dt < h/|v| = 0.5
# Combined: dt < 0.3125. Use dt=0.01.
TRANSIENT_ADR_1D_EXPLICIT_CASES: List[
    Tuple[str, Callable[[], ButcherTableau], Optional[Tuple[str, ...]]]
] = [
    ("1d_DD_noV_FE", forward_euler, None),
    ("1d_DD_noV_Heun", heun, None),
    ("1d_DD_V_FE", forward_euler, ("0.5",)),
    ("1d_DD_V_Heun", heun, ("0.5",)),
]


@slow_test
class TestTransientADRExplicit1D:
    """1D transient ADR tests with P2 + explicit methods (FE, Heun).

    CFL-constrained dt. P2 on nx=4 exactly represents u=(1-x)*x*(1+t),
    so spatial error is zero. Remaining error is purely temporal
    discretization.
    """

    @pytest.mark.parametrize(
        "name,make,velocity",
        TRANSIENT_ADR_1D_EXPLICIT_CASES,
    )
    def test_transient_adr_1d_explicit(
        self,
        numpy_bkd: NumpyBkd,
        name: str,
        make: Callable[[], ButcherTableau],
        velocity: Optional[Tuple[str, ...]],
    ) -> None:
        system, exact_at_time = _setup(
            numpy_bkd, _INTERVAL, _SOLUTION_1D, "DD", 4, 0.1, velocity
        )
        rel_error = _relative_error(system, exact_at_time, make, 0.01, 1.0)
        assert rel_error < 1e-5


# =========================================================================
# Manual Newton test using the BC-enforcing residual wrapper
# =========================================================================


class TestManualNewtonWithConstraint:
    """Low-level test of the BC-enforcing residual with manual Newton.

    Verifies that the wrapper correctly applies Dirichlet constraints
    to the assembled Newton system at a low level.
    """

    def test_manual_newton_backward_euler(self, numpy_bkd: NumpyBkd) -> None:
        """Manual Newton with BE + the BC-enforcing residual."""
        bkd = numpy_bkd
        system, exact_at_time = _setup(bkd, _INTERVAL, _SOLUTION_1D, "DD", 32, 4.0)

        # Manual setup: adapter + stepper + constrained wrapper
        ode_adapter = skfem_ode_residual(system)
        stepper = BackwardEulerHVP(ode_adapter)
        constrained = create_galerkin_bc_enforcing_residual(
            stepper, system.constraint_set(), bkd
        )

        y = bkd.asarray(exact_at_time(0.0))
        dt = 1.0
        nsteps = 5
        t = 0.0

        for step in range(nsteps):
            t_np1 = t + dt
            # Set stepper with unmodified prev_state
            constrained.bind(StepContext(t_prev=t, deltat=dt, y_prev=y))
            # Initial guess with Dirichlet values injected
            constraint_set = system.constraint_set()
            d_dofs = constraint_set.dofs()
            d_vals = constraint_set.values(t_np1)
            d_dofs_np = bkd.to_numpy(d_dofs).astype(np.intp)
            y_new_np = bkd.to_numpy(y).copy()
            if len(d_dofs_np) > 0:
                y_new_np[d_dofs_np] = bkd.to_numpy(d_vals)
            y_new = bkd.asarray(y_new_np.astype(np.float64))

            for newton_iter in range(10):
                res = constrained(y_new)
                res_norm = float(np.linalg.norm(bkd.to_numpy(res)))
                if res_norm < 1e-10:
                    break
                dy = constrained.linsolve(y_new, res)
                y_new = y_new - dy
            y = y_new
            t += dt

        u_exact_final = exact_at_time(t)
        u_num = bkd.to_numpy(y)

        u_norm = np.linalg.norm(u_exact_final)
        rel_error = np.linalg.norm(u_num - u_exact_final) / u_norm

        assert rel_error < 1e-6

    def test_manual_newton_crank_nicolson(self, numpy_bkd: NumpyBkd) -> None:
        """Manual Newton with CN + the BC-enforcing residual."""
        bkd = numpy_bkd
        system, exact_at_time = _setup(bkd, _INTERVAL, _SOLUTION_1D, "DD", 32, 4.0)

        ode_adapter = skfem_ode_residual(system)
        stepper = CrankNicolsonHVP(ode_adapter)
        constrained = create_galerkin_bc_enforcing_residual(
            stepper, system.constraint_set(), bkd
        )

        y = bkd.asarray(exact_at_time(0.0))
        dt = 1.0
        nsteps = 5
        t = 0.0

        for step in range(nsteps):
            t_np1 = t + dt
            constrained.bind(StepContext(t_prev=t, deltat=dt, y_prev=y))
            constraint_set = system.constraint_set()
            d_dofs = constraint_set.dofs()
            d_vals = constraint_set.values(t_np1)
            d_dofs_np = bkd.to_numpy(d_dofs).astype(np.intp)
            y_new_np = bkd.to_numpy(y).copy()
            if len(d_dofs_np) > 0:
                y_new_np[d_dofs_np] = bkd.to_numpy(d_vals)
            y_new = bkd.asarray(y_new_np.astype(np.float64))

            for newton_iter in range(10):
                res = constrained(y_new)
                res_norm = float(np.linalg.norm(bkd.to_numpy(res)))
                if res_norm < 1e-10:
                    break
                dy = constrained.linsolve(y_new, res)
                y_new = y_new - dy
            y = y_new
            t += dt

        u_exact_final = exact_at_time(t)
        u_num = bkd.to_numpy(y)

        u_norm = np.linalg.norm(u_exact_final)
        rel_error = np.linalg.norm(u_num - u_exact_final) / u_norm

        assert rel_error < 1e-6
