"""Parametrized transient ADR tests (1D + 2D).

This module contains:
1. 8 parametrized 1D transient tests replicating legacy
   test_finite_elements.py test_transient_advection_diffusion_reaction
2. 4 parametrized 2D transient tests (new coverage)
3. Crank-Nicolson variants (1D + 2D)
4. Explicit integrator tests (Forward Euler + Heun, with/without advection)
5. Low-level manual Newton test using ConstrainedTimeStepResidual wrapper

Implicit cases use GalerkinModel.solve_transient() with backward Euler or
Crank-Nicolson. Explicit cases use GalerkinModel with CFL-constrained dt.
"""

import unittest
from typing import List, Tuple

import numpy as np
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.manufactured.adapter import (
    GalerkinManufacturedSolutionAdapter,
    create_adr_manufactured_test,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration import (
    ConstrainedTimeStepResidual,
    GalerkinModel,
    GalerkinPhysicsODEAdapter,
    TimeIntegrationConfig,
)
from pyapprox.pde.time.implicit_steppers import (
    BackwardEulerResidual,
    CrankNicolsonResidual,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.test_utils import slow_test

# =========================================================================
# Helpers
# =========================================================================


def _parse_reaction(react_str):
    """Convert reaction string to physics callable pair."""
    if react_str == "0*u":
        return None
    elif react_str == "u**2":
        return (lambda x, u: u**2, lambda x, u: 2 * u)
    else:
        raise ValueError(f"Unknown react_str: {react_str}")


def _has_velocity(vel_strs):
    """Check if velocity strings represent non-negligible velocity."""
    return not all("1e-16" in v for v in vel_strs)


def _setup_1d_problem(
    bkd, bounds, sol_str, diff_str, react_str, vel_strs, bndry_types, nx, diffusivity
):
    """Create physics, model, and exact solution for a 1D problem."""
    functions, _ = create_adr_manufactured_test(
        bounds=bounds,
        sol_str=sol_str,
        diff_str=diff_str,
        react_str=react_str,
        vel_strs=vel_strs,
        bkd=bkd,
        time_dependent=True,
    )

    mesh = StructuredMesh1D(nx=nx, bounds=(bounds[0], bounds[1]), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=2)

    adapter = GalerkinManufacturedSolutionAdapter(
        basis, functions, bkd, time_dependent=True
    )
    bc_set = adapter.create_boundary_conditions(bndry_types, robin_alpha=1.0)

    velocity = adapter.velocity_for_galerkin() if _has_velocity(vel_strs) else None
    reaction = _parse_reaction(react_str)

    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=diffusivity,
        bkd=bkd,
        velocity=velocity,
        reaction=reaction,
        forcing=adapter.forcing_for_galerkin(),
        boundary_conditions=bc_set.all_conditions(),
    )

    model = GalerkinModel(physics, bkd)

    exact_sol_func = adapter.solution_function()
    dof_coords = bkd.to_numpy(basis.dof_coordinates())

    def exact_at_time(t):
        u = exact_sol_func(dof_coords, t)
        if hasattr(u, "shape") and u.ndim > 1:
            return u[:, 0] if u.shape[1] == 1 else u.flatten()
        return u

    return physics, model, exact_at_time


# =========================================================================
# 1D Transient ADR Cases
# =========================================================================

# Format: (name, bndry_types, vel_strs, react_str)
# Fixed: bounds=[0,1], sol_str="(1-x)*x*(1+T)", diff_str="4+1e-16*x"
TRANSIENT_ADR_1D_CASES: List[Tuple[str, List[str], List[str], str]] = [
    ("1d_DD_noV_noR", ["D", "D"], ["0+1e-16*x"], "0*u"),
    ("1d_DD_noV_R", ["D", "D"], ["0+1e-16*x"], "u**2"),
    ("1d_DD_V_noR", ["D", "D"], ["(1+x)/10"], "0*u"),
    ("1d_DD_V_R", ["D", "D"], ["(1+x)/10"], "u**2"),
    ("1d_DN_noV_noR", ["D", "N"], ["0+1e-16*x"], "0*u"),
    ("1d_DN_noV_R", ["D", "N"], ["0+1e-16*x"], "u**2"),
    ("1d_DN_V_noR", ["D", "N"], ["(1+x)/10"], "0*u"),
    ("1d_DN_V_R", ["D", "N"], ["(1+x)/10"], "u**2"),
]


class TestTransientADR1D(ParametrizedTestCase):
    """Parametrized 1D transient ADR tests with P2 + backward Euler.

    Replicates legacy test_finite_elements.py
    test_transient_advection_diffusion_reaction (8 cases).
    Uses GalerkinModel.solve_transient() for time integration.
    """

    @parametrize(
        "name,bndry_types,vel_strs,react_str",
        TRANSIENT_ADR_1D_CASES,
    )
    def test_transient_adr_1d(
        self,
        name: str,
        bndry_types: List[str],
        vel_strs: List[str],
        react_str: str,
    ) -> None:
        """Test transient ADR with manufactured solution."""
        bkd = NumpyBkd()
        _, model, exact_at_time = _setup_1d_problem(
            bkd,
            bounds=[0.0, 1.0],
            sol_str="(1-x)*x*(1+T)",
            diff_str="4+1e-16*x",
            react_str=react_str,
            vel_strs=vel_strs,
            bndry_types=bndry_types,
            nx=32,
            diffusivity=4.0,
        )

        y0 = bkd.asarray(exact_at_time(0.0))
        config = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=5.0,
            deltat=1.0,
        )
        solutions, times = model.solve_transient(y0, config)

        u_exact_final = exact_at_time(float(times[-1]))
        u_num = bkd.to_numpy(solutions[:, -1])

        u_norm = np.linalg.norm(u_exact_final)
        if u_norm > 1e-10:
            rel_error = np.linalg.norm(u_num - u_exact_final) / u_norm
        else:
            rel_error = np.linalg.norm(u_num - u_exact_final)

        # Nonlinear reaction with Neumann BCs has slightly larger error
        # due to spatial discretization of u^2 term
        tol = 2e-6 if "R" in name and "N" in "".join(bndry_types) else 1e-6
        self.assertLess(
            rel_error,
            tol,
            f"Test {name}: rel_error={rel_error:.2e} at t={float(times[-1])} "
            f"should be < {tol}",
        )


# =========================================================================
# 2D Transient ADR Cases
# =========================================================================

# Format: (name, bndry_types, vel_strs)
# Fixed: bounds=[0,1,0,1], sol_str="(1-x)*x*(1-y)*y*(1+T)",
#        diff_str="4+1e-16*x+1e-16*y", react_str="0*u"
TRANSIENT_ADR_2D_CASES: List[Tuple[str, List[str], List[str]]] = [
    ("2d_DDDD_noV", ["D", "D", "D", "D"], ["0+1e-16*x", "0+1e-16*y"]),
    ("2d_DDDD_V", ["D", "D", "D", "D"], ["(1+x)/10", "(1+y)/10"]),
    ("2d_DRDN_noV", ["D", "R", "D", "N"], ["0+1e-16*x", "0+1e-16*y"]),
    ("2d_DRDN_V", ["D", "R", "D", "N"], ["(1+x)/10", "(1+y)/10"]),
]


class TestTransientADR2D(ParametrizedTestCase):
    """Parametrized 2D transient ADR tests with P2 + backward Euler.

    Uses GalerkinModel.solve_transient() for time integration.
    """

    @parametrize(
        "name,bndry_types,vel_strs",
        TRANSIENT_ADR_2D_CASES,
    )
    def test_transient_adr_2d(
        self,
        name: str,
        bndry_types: List[str],
        vel_strs: List[str],
    ) -> None:
        """Test 2D transient ADR with manufactured solution."""
        bkd = NumpyBkd()
        bounds = [0.0, 1.0, 0.0, 1.0]
        sol_str = "(1-x)*x*(1-y)*y*(1+T)"
        diff_str = "4+1e-16*x+1e-16*y"
        react_str = "0*u"

        functions, _ = create_adr_manufactured_test(
            bounds=bounds,
            sol_str=sol_str,
            diff_str=diff_str,
            react_str=react_str,
            vel_strs=vel_strs,
            bkd=bkd,
            time_dependent=True,
        )

        mesh = StructuredMesh2D(
            nx=16,
            ny=16,
            bounds=[(bounds[0], bounds[1]), (bounds[2], bounds[3])],
            bkd=bkd,
        )
        basis = LagrangeBasis(mesh, degree=2)

        adapter = GalerkinManufacturedSolutionAdapter(
            basis, functions, bkd, time_dependent=True
        )
        bc_set = adapter.create_boundary_conditions(bndry_types, robin_alpha=1.0)

        velocity = adapter.velocity_for_galerkin() if _has_velocity(vel_strs) else None

        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=4.0,
            bkd=bkd,
            velocity=velocity,
            forcing=adapter.forcing_for_galerkin(),
            boundary_conditions=bc_set.all_conditions(),
        )

        model = GalerkinModel(physics, bkd)

        exact_sol_func = adapter.solution_function()
        dof_coords = bkd.to_numpy(basis.dof_coordinates())

        def exact_at_time(t):
            u = exact_sol_func(dof_coords, t)
            if hasattr(u, "shape") and u.ndim > 1:
                return u[:, 0] if u.shape[1] == 1 else u.flatten()
            return u

        y0 = bkd.asarray(exact_at_time(0.0))
        config = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=5.0,
            deltat=1.0,
        )
        solutions, times = model.solve_transient(y0, config)

        u_exact_final = exact_at_time(float(times[-1]))
        u_num = bkd.to_numpy(solutions[:, -1])

        u_norm = np.linalg.norm(u_exact_final)
        if u_norm > 1e-10:
            rel_error = np.linalg.norm(u_num - u_exact_final) / u_norm
        else:
            rel_error = np.linalg.norm(u_num - u_exact_final)

        self.assertLess(
            rel_error,
            1e-6,
            f"Test {name}: rel_error={rel_error:.2e} at t={float(times[-1])} "
            f"should be < 1e-6",
        )


# =========================================================================
# 1D Transient ADR with Crank-Nicolson
# =========================================================================

TRANSIENT_ADR_1D_CN_CASES: List[Tuple[str, List[str], List[str], str]] = [
    ("1d_DD_noV_noR_CN", ["D", "D"], ["0+1e-16*x"], "0*u"),
    ("1d_DD_V_noR_CN", ["D", "D"], ["(1+x)/10"], "0*u"),
]


class TestTransientADR1D_CN(ParametrizedTestCase):
    """1D transient ADR tests with P2 + Crank-Nicolson.

    Uses GalerkinModel.solve_transient() for time integration.
    """

    @parametrize(
        "name,bndry_types,vel_strs,react_str",
        TRANSIENT_ADR_1D_CN_CASES,
    )
    def test_transient_adr_1d_cn(
        self,
        name: str,
        bndry_types: List[str],
        vel_strs: List[str],
        react_str: str,
    ) -> None:
        bkd = NumpyBkd()
        _, model, exact_at_time = _setup_1d_problem(
            bkd,
            bounds=[0.0, 1.0],
            sol_str="(1-x)*x*(1+T)",
            diff_str="4+1e-16*x",
            react_str=react_str,
            vel_strs=vel_strs,
            bndry_types=bndry_types,
            nx=32,
            diffusivity=4.0,
        )

        y0 = bkd.asarray(exact_at_time(0.0))
        config = TimeIntegrationConfig(
            method="crank_nicolson",
            init_time=0.0,
            final_time=5.0,
            deltat=1.0,
        )
        solutions, times = model.solve_transient(y0, config)

        u_exact_final = exact_at_time(float(times[-1]))
        u_num = bkd.to_numpy(solutions[:, -1])

        u_norm = np.linalg.norm(u_exact_final)
        if u_norm > 1e-10:
            rel_error = np.linalg.norm(u_num - u_exact_final) / u_norm
        else:
            rel_error = np.linalg.norm(u_num - u_exact_final)

        self.assertLess(
            rel_error,
            1e-6,
            f"Test {name}: rel_error={rel_error:.2e} at t={float(times[-1])} "
            f"should be < 1e-6",
        )


# =========================================================================
# 2D Transient ADR with Crank-Nicolson
# =========================================================================

TRANSIENT_ADR_2D_CN_CASES: List[Tuple[str, List[str], List[str]]] = [
    ("2d_DDDD_noV_CN", ["D", "D", "D", "D"], ["0+1e-16*x", "0+1e-16*y"]),
]


class TestTransientADR2D_CN(ParametrizedTestCase):
    """2D transient ADR tests with P2 + Crank-Nicolson.

    Uses GalerkinModel.solve_transient() for time integration.
    """

    @parametrize(
        "name,bndry_types,vel_strs",
        TRANSIENT_ADR_2D_CN_CASES,
    )
    def test_transient_adr_2d_cn(
        self,
        name: str,
        bndry_types: List[str],
        vel_strs: List[str],
    ) -> None:
        bkd = NumpyBkd()
        bounds = [0.0, 1.0, 0.0, 1.0]
        sol_str = "(1-x)*x*(1-y)*y*(1+T)"
        diff_str = "4+1e-16*x+1e-16*y"
        react_str = "0*u"

        functions, _ = create_adr_manufactured_test(
            bounds=bounds,
            sol_str=sol_str,
            diff_str=diff_str,
            react_str=react_str,
            vel_strs=vel_strs,
            bkd=bkd,
            time_dependent=True,
        )

        mesh = StructuredMesh2D(
            nx=16,
            ny=16,
            bounds=[(bounds[0], bounds[1]), (bounds[2], bounds[3])],
            bkd=bkd,
        )
        basis = LagrangeBasis(mesh, degree=2)

        adapter = GalerkinManufacturedSolutionAdapter(
            basis, functions, bkd, time_dependent=True
        )
        bc_set = adapter.create_boundary_conditions(bndry_types, robin_alpha=1.0)

        velocity = adapter.velocity_for_galerkin() if _has_velocity(vel_strs) else None

        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=4.0,
            bkd=bkd,
            velocity=velocity,
            forcing=adapter.forcing_for_galerkin(),
            boundary_conditions=bc_set.all_conditions(),
        )

        model = GalerkinModel(physics, bkd)

        exact_sol_func = adapter.solution_function()
        dof_coords = bkd.to_numpy(basis.dof_coordinates())

        def exact_at_time(t):
            u = exact_sol_func(dof_coords, t)
            if hasattr(u, "shape") and u.ndim > 1:
                return u[:, 0] if u.shape[1] == 1 else u.flatten()
            return u

        y0 = bkd.asarray(exact_at_time(0.0))
        config = TimeIntegrationConfig(
            method="crank_nicolson",
            init_time=0.0,
            final_time=5.0,
            deltat=1.0,
        )
        solutions, times = model.solve_transient(y0, config)

        u_exact_final = exact_at_time(float(times[-1]))
        u_num = bkd.to_numpy(solutions[:, -1])

        u_norm = np.linalg.norm(u_exact_final)
        if u_norm > 1e-10:
            rel_error = np.linalg.norm(u_num - u_exact_final) / u_norm
        else:
            rel_error = np.linalg.norm(u_num - u_exact_final)

        self.assertLess(
            rel_error,
            1e-6,
            f"Test {name}: rel_error={rel_error:.2e} at t={float(times[-1])} "
            f"should be < 1e-6",
        )


# =========================================================================
# 1D Transient ADR with Explicit Methods (Forward Euler, Heun)
# =========================================================================

# Format: (name, method, vel_strs, has_vel)
# Fixed: bounds=[0,1], sol_str="(1-x)*x*(1+T)", diff_str="0.1+1e-16*x",
#        react_str="0*u", bndry_types=["D","D"], nx=4 (P2 exact)
# D=0.1, h=0.25 → CFL diffusive: dt < h²/(2D) = 0.3125
# With |v|=0.5 → CFL advective: dt < h/|v| = 0.5
# Combined: dt < 0.3125. Use dt=0.01.
TRANSIENT_ADR_1D_EXPLICIT_CASES: List[Tuple[str, str, List[str], bool]] = [
    ("1d_DD_noV_FE", "forward_euler", ["0+1e-16*x"], False),
    ("1d_DD_noV_Heun", "heun", ["0+1e-16*x"], False),
    ("1d_DD_V_FE", "forward_euler", ["0.5"], True),
    ("1d_DD_V_Heun", "heun", ["0.5"], True),
]


@slow_test
class TestTransientADRExplicit1D(ParametrizedTestCase):
    """1D transient ADR tests with P2 + explicit methods (FE, Heun).

    Uses GalerkinModel.solve_transient() with CFL-constrained dt.
    P2 on nx=4 exactly represents u=(1-x)*x*(1+T), so spatial error is
    zero. Remaining error is purely temporal discretization.
    """

    @parametrize(
        "name,method,vel_strs,has_vel",
        TRANSIENT_ADR_1D_EXPLICIT_CASES,
    )
    def test_transient_adr_1d_explicit(
        self,
        name: str,
        method: str,
        vel_strs: List[str],
        has_vel: bool,
    ) -> None:
        bkd = NumpyBkd()
        _, model, exact_at_time = _setup_1d_problem(
            bkd,
            bounds=[0.0, 1.0],
            sol_str="(1-x)*x*(1+T)",
            diff_str="0.1+1e-16*x",
            react_str="0*u",
            vel_strs=vel_strs,
            bndry_types=["D", "D"],
            nx=4,
            diffusivity=0.1,
        )

        y0 = bkd.asarray(exact_at_time(0.0))
        config = TimeIntegrationConfig(
            method=method,
            init_time=0.0,
            final_time=1.0,
            deltat=0.01,
        )
        solutions, times = model.solve_transient(y0, config)

        u_exact_final = exact_at_time(float(times[-1]))
        u_num = bkd.to_numpy(solutions[:, -1])

        u_norm = np.linalg.norm(u_exact_final)
        if u_norm > 1e-10:
            rel_error = np.linalg.norm(u_num - u_exact_final) / u_norm
        else:
            rel_error = np.linalg.norm(u_num - u_exact_final)

        self.assertLess(
            rel_error,
            1e-5,
            f"Test {name}: rel_error={rel_error:.2e} at t={float(times[-1])} "
            f"should be < 1e-5",
        )


# =========================================================================
# Manual Newton test using ConstrainedTimeStepResidual wrapper
# =========================================================================


class TestManualNewtonWithConstraint(unittest.TestCase):
    """Low-level test of ConstrainedTimeStepResidual with manual Newton.

    Verifies that the wrapper correctly applies Dirichlet constraints
    to the assembled Newton system at a low level.
    """

    def test_manual_newton_backward_euler(self) -> None:
        """Manual Newton with BE + ConstrainedTimeStepResidual."""
        bkd = NumpyBkd()
        _, _, exact_at_time = _setup_1d_problem(
            bkd,
            bounds=[0.0, 1.0],
            sol_str="(1-x)*x*(1+T)",
            diff_str="4+1e-16*x",
            react_str="0*u",
            vel_strs=["0+1e-16*x"],
            bndry_types=["D", "D"],
            nx=32,
            diffusivity=4.0,
        )

        # Also need physics for manual setup
        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="(1-x)*x*(1+T)",
            diff_str="4+1e-16*x",
            react_str="0*u",
            vel_strs=["0+1e-16*x"],
            bkd=bkd,
            time_dependent=True,
        )
        mesh = StructuredMesh1D(nx=32, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=2)
        adapter = GalerkinManufacturedSolutionAdapter(
            basis, functions, bkd, time_dependent=True
        )
        bc_set = adapter.create_boundary_conditions(["D", "D"])
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=4.0,
            bkd=bkd,
            forcing=adapter.forcing_for_galerkin(),
            boundary_conditions=bc_set.all_conditions(),
        )

        # Manual setup: adapter + stepper + constrained wrapper
        ode_adapter = GalerkinPhysicsODEAdapter(physics)
        stepper = BackwardEulerResidual(ode_adapter)
        constrained = ConstrainedTimeStepResidual(stepper, ode_adapter)

        y = bkd.asarray(exact_at_time(0.0))
        dt = 1.0
        nsteps = 5
        t = 0.0

        for step in range(nsteps):
            t_np1 = t + dt
            # Set stepper with unmodified prev_state
            stepper.set_time(t, dt, y)
            # Set constraint time
            constrained.set_bc_time(t_np1)
            # Initial guess with Dirichlet values injected
            d_dofs, d_vals = ode_adapter.dirichlet_dof_info(t_np1)
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

        self.assertLess(
            rel_error,
            1e-6,
            f"Manual Newton: rel_error={rel_error:.2e} at t={t} should be < 1e-6",
        )

    def test_manual_newton_crank_nicolson(self) -> None:
        """Manual Newton with CN + ConstrainedTimeStepResidual."""
        bkd = NumpyBkd()

        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="(1-x)*x*(1+T)",
            diff_str="4+1e-16*x",
            react_str="0*u",
            vel_strs=["0+1e-16*x"],
            bkd=bkd,
            time_dependent=True,
        )
        mesh = StructuredMesh1D(nx=32, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=2)
        adapter = GalerkinManufacturedSolutionAdapter(
            basis, functions, bkd, time_dependent=True
        )
        bc_set = adapter.create_boundary_conditions(["D", "D"])
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=4.0,
            bkd=bkd,
            forcing=adapter.forcing_for_galerkin(),
            boundary_conditions=bc_set.all_conditions(),
        )

        exact_sol_func = adapter.solution_function()
        dof_coords = bkd.to_numpy(basis.dof_coordinates())

        def exact_at_time(t):
            u = exact_sol_func(dof_coords, t)
            if hasattr(u, "shape") and u.ndim > 1:
                return u[:, 0] if u.shape[1] == 1 else u.flatten()
            return u

        ode_adapter = GalerkinPhysicsODEAdapter(physics)
        stepper = CrankNicolsonResidual(ode_adapter)
        constrained = ConstrainedTimeStepResidual(stepper, ode_adapter)

        y = bkd.asarray(exact_at_time(0.0))
        dt = 1.0
        nsteps = 5
        t = 0.0

        for step in range(nsteps):
            t_np1 = t + dt
            stepper.set_time(t, dt, y)
            constrained.set_bc_time(t_np1)
            d_dofs, d_vals = ode_adapter.dirichlet_dof_info(t_np1)
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

        self.assertLess(
            rel_error,
            1e-6,
            f"Manual Newton CN: rel_error={rel_error:.2e} at t={t} should be < 1e-6",
        )


from pyapprox.util.test_utils import load_tests  # noqa: F401

if __name__ == "__main__":
    unittest.main()
