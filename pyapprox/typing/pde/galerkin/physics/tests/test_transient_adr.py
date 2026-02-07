"""Parametrized transient ADR tests (1D + 2D).

This module contains:
1. 8 parametrized 1D transient tests replicating legacy
   test_finite_elements.py test_transient_advection_diffusion_reaction
2. 4 parametrized 2D transient tests (new coverage)

All cases use backward Euler time stepping with Newton iteration.
"""

import unittest
from typing import List, Tuple

import numpy as np
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.typing.pde.galerkin.basis import LagrangeBasis
from pyapprox.typing.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.typing.pde.galerkin.manufactured.adapter import (
    GalerkinManufacturedSolutionAdapter,
    create_adr_manufactured_test,
)
from pyapprox.typing.pde.galerkin.time_integration import (
    GalerkinPhysicsODEAdapter,
)
from pyapprox.typing.pde.time.implicit_steppers import (
    BackwardEulerResidual,
    CrankNicolsonResidual,
)


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


# =========================================================================
# 1D Transient ADR Cases
# =========================================================================

# Format: (name, bndry_types, vel_strs, react_str)
# Fixed: bounds=[0,1], sol_str="(1-x)*x*(1+T)", diff_str="4+1e-16*x"
TRANSIENT_ADR_1D_CASES: List[
    Tuple[str, List[str], List[str], str]
] = [
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
        bounds = [0.0, 1.0]
        sol_str = "(1-x)*x*(1+T)"
        diff_str = "4+1e-16*x"
        nx = 32

        # Create manufactured solution
        functions, _ = create_adr_manufactured_test(
            bounds=bounds,
            sol_str=sol_str,
            diff_str=diff_str,
            react_str=react_str,
            vel_strs=vel_strs,
            bkd=bkd,
            time_dependent=True,
        )

        # Create mesh and basis
        mesh = StructuredMesh1D(
            nx=nx, bounds=(bounds[0], bounds[1]), bkd=bkd
        )
        basis = LagrangeBasis(mesh, degree=2)

        # Create adapter
        adapter = GalerkinManufacturedSolutionAdapter(
            basis, functions, bkd, time_dependent=True
        )
        bc_set = adapter.create_boundary_conditions(
            bndry_types, robin_alpha=1.0
        )

        # Build physics
        velocity = (
            adapter.velocity_for_galerkin()
            if _has_velocity(vel_strs)
            else None
        )
        reaction = _parse_reaction(react_str)

        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=4.0,
            bkd=bkd,
            velocity=velocity,
            reaction=reaction,
            forcing=adapter.forcing_for_galerkin(),
            boundary_conditions=bc_set.all_conditions(),
        )

        # Time stepping
        ode_adapter = GalerkinPhysicsODEAdapter(physics)
        stepper = BackwardEulerResidual(ode_adapter)

        exact_sol_func = adapter.solution_function()
        dof_coords = bkd.to_numpy(basis.dof_coordinates())

        def exact_at_time(t):
            u = exact_sol_func(dof_coords, t)
            if hasattr(u, "shape") and u.ndim > 1:
                return u[:, 0] if u.shape[1] == 1 else u.flatten()
            return u

        y = bkd.asarray(exact_at_time(0.0))
        dt = 1.0
        nsteps = 5
        t = 0.0

        for step in range(nsteps):
            stepper.set_time(t, dt, y)
            y_new = bkd.copy(y)
            for newton_iter in range(10):
                res = stepper(y_new)
                res_norm = float(np.linalg.norm(bkd.to_numpy(res)))
                if res_norm < 1e-10:
                    break
                jac = stepper.jacobian(y_new)
                dy = bkd.solve(jac, -res)
                y_new = y_new + dy
            y = y_new
            t += dt

        # Compare to exact at final time
        u_exact_final = exact_at_time(t)
        u_num = bkd.to_numpy(y)

        u_norm = np.linalg.norm(u_exact_final)
        if u_norm > 1e-10:
            rel_error = np.linalg.norm(u_num - u_exact_final) / u_norm
        else:
            rel_error = np.linalg.norm(u_num - u_exact_final)

        self.assertLess(
            rel_error, 1e-6,
            f"Test {name}: rel_error={rel_error:.2e} at t={t} "
            f"should be < 1e-6",
        )


# =========================================================================
# 2D Transient ADR Cases
# =========================================================================

# Format: (name, bndry_types, vel_strs)
# Fixed: bounds=[0,1,0,1], sol_str="(1-x)*x*(1-y)*y*(1+T)",
#        diff_str="4+1e-16*x+1e-16*y", react_str="0*u"
TRANSIENT_ADR_2D_CASES: List[
    Tuple[str, List[str], List[str]]
] = [
    ("2d_DDDD_noV", ["D", "D", "D", "D"], ["0+1e-16*x", "0+1e-16*y"]),
    ("2d_DDDD_V", ["D", "D", "D", "D"], ["(1+x)/10", "(1+y)/10"]),
    ("2d_DRDN_noV", ["D", "R", "D", "N"], ["0+1e-16*x", "0+1e-16*y"]),
    ("2d_DRDN_V", ["D", "R", "D", "N"], ["(1+x)/10", "(1+y)/10"]),
]


class TestTransientADR2D(ParametrizedTestCase):
    """Parametrized 2D transient ADR tests with P2 + backward Euler.

    New coverage — no legacy 2D transient ADR tests exist.
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

        # Create manufactured solution
        functions, _ = create_adr_manufactured_test(
            bounds=bounds,
            sol_str=sol_str,
            diff_str=diff_str,
            react_str=react_str,
            vel_strs=vel_strs,
            bkd=bkd,
            time_dependent=True,
        )

        # Create mesh and basis
        mesh = StructuredMesh2D(
            nx=16, ny=16,
            bounds=[(bounds[0], bounds[1]), (bounds[2], bounds[3])],
            bkd=bkd,
        )
        basis = LagrangeBasis(mesh, degree=2)

        # Create adapter
        adapter = GalerkinManufacturedSolutionAdapter(
            basis, functions, bkd, time_dependent=True
        )
        bc_set = adapter.create_boundary_conditions(
            bndry_types, robin_alpha=1.0
        )

        # Build physics
        velocity = (
            adapter.velocity_for_galerkin()
            if _has_velocity(vel_strs)
            else None
        )

        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=4.0,
            bkd=bkd,
            velocity=velocity,
            forcing=adapter.forcing_for_galerkin(),
            boundary_conditions=bc_set.all_conditions(),
        )

        # Time stepping
        ode_adapter = GalerkinPhysicsODEAdapter(physics)
        stepper = BackwardEulerResidual(ode_adapter)

        exact_sol_func = adapter.solution_function()
        dof_coords = bkd.to_numpy(basis.dof_coordinates())

        def exact_at_time(t):
            u = exact_sol_func(dof_coords, t)
            if hasattr(u, "shape") and u.ndim > 1:
                return u[:, 0] if u.shape[1] == 1 else u.flatten()
            return u

        y = bkd.asarray(exact_at_time(0.0))
        dt = 1.0
        nsteps = 5
        t = 0.0

        for step in range(nsteps):
            stepper.set_time(t, dt, y)
            y_new = bkd.copy(y)
            for newton_iter in range(10):
                res = stepper(y_new)
                res_norm = float(np.linalg.norm(bkd.to_numpy(res)))
                if res_norm < 1e-10:
                    break
                jac = stepper.jacobian(y_new)
                dy = bkd.solve(jac, -res)
                y_new = y_new + dy
            y = y_new
            t += dt

        # Compare to exact at final time
        u_exact_final = exact_at_time(t)
        u_num = bkd.to_numpy(y)

        u_norm = np.linalg.norm(u_exact_final)
        if u_norm > 1e-10:
            rel_error = np.linalg.norm(u_num - u_exact_final) / u_norm
        else:
            rel_error = np.linalg.norm(u_num - u_exact_final)

        self.assertLess(
            rel_error, 1e-6,
            f"Test {name}: rel_error={rel_error:.2e} at t={t} "
            f"should be < 1e-6",
        )


# =========================================================================
# 1D Transient ADR with Crank-Nicolson
# =========================================================================

TRANSIENT_ADR_1D_CN_CASES: List[
    Tuple[str, List[str], List[str], str]
] = [
    ("1d_DD_noV_noR_CN", ["D", "D"], ["0+1e-16*x"], "0*u"),
    ("1d_DD_V_noR_CN", ["D", "D"], ["(1+x)/10"], "0*u"),
]


class TestTransientADR1D_CN(ParametrizedTestCase):
    """1D transient ADR tests with P2 + Crank-Nicolson."""

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
        bounds = [0.0, 1.0]
        sol_str = "(1-x)*x*(1+T)"
        diff_str = "4+1e-16*x"
        nx = 32

        functions, _ = create_adr_manufactured_test(
            bounds=bounds,
            sol_str=sol_str,
            diff_str=diff_str,
            react_str=react_str,
            vel_strs=vel_strs,
            bkd=bkd,
            time_dependent=True,
        )

        mesh = StructuredMesh1D(
            nx=nx, bounds=(bounds[0], bounds[1]), bkd=bkd
        )
        basis = LagrangeBasis(mesh, degree=2)

        adapter = GalerkinManufacturedSolutionAdapter(
            basis, functions, bkd, time_dependent=True
        )
        bc_set = adapter.create_boundary_conditions(
            bndry_types, robin_alpha=1.0
        )

        velocity = (
            adapter.velocity_for_galerkin()
            if _has_velocity(vel_strs)
            else None
        )
        reaction = _parse_reaction(react_str)

        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=4.0,
            bkd=bkd,
            velocity=velocity,
            reaction=reaction,
            forcing=adapter.forcing_for_galerkin(),
            boundary_conditions=bc_set.all_conditions(),
        )

        ode_adapter = GalerkinPhysicsODEAdapter(physics)
        stepper = CrankNicolsonResidual(ode_adapter)

        exact_sol_func = adapter.solution_function()
        dof_coords = bkd.to_numpy(basis.dof_coordinates())

        def exact_at_time(t):
            u = exact_sol_func(dof_coords, t)
            if hasattr(u, "shape") and u.ndim > 1:
                return u[:, 0] if u.shape[1] == 1 else u.flatten()
            return u

        y = bkd.asarray(exact_at_time(0.0))
        dt = 1.0
        nsteps = 5
        t = 0.0

        for step in range(nsteps):
            stepper.set_time(t, dt, y)
            y_new = bkd.copy(y)
            for newton_iter in range(10):
                res = stepper(y_new)
                res_norm = float(np.linalg.norm(bkd.to_numpy(res)))
                if res_norm < 1e-10:
                    break
                jac = stepper.jacobian(y_new)
                dy = bkd.solve(jac, -res)
                y_new = y_new + dy
            y = y_new
            t += dt

        u_exact_final = exact_at_time(t)
        u_num = bkd.to_numpy(y)

        u_norm = np.linalg.norm(u_exact_final)
        if u_norm > 1e-10:
            rel_error = np.linalg.norm(u_num - u_exact_final) / u_norm
        else:
            rel_error = np.linalg.norm(u_num - u_exact_final)

        self.assertLess(
            rel_error, 1e-6,
            f"Test {name}: rel_error={rel_error:.2e} at t={t} "
            f"should be < 1e-6",
        )


# =========================================================================
# 2D Transient ADR with Crank-Nicolson
# =========================================================================

TRANSIENT_ADR_2D_CN_CASES: List[
    Tuple[str, List[str], List[str]]
] = [
    ("2d_DDDD_noV_CN", ["D", "D", "D", "D"], ["0+1e-16*x", "0+1e-16*y"]),
]


class TestTransientADR2D_CN(ParametrizedTestCase):
    """2D transient ADR tests with P2 + Crank-Nicolson."""

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
            nx=16, ny=16,
            bounds=[(bounds[0], bounds[1]), (bounds[2], bounds[3])],
            bkd=bkd,
        )
        basis = LagrangeBasis(mesh, degree=2)

        adapter = GalerkinManufacturedSolutionAdapter(
            basis, functions, bkd, time_dependent=True
        )
        bc_set = adapter.create_boundary_conditions(
            bndry_types, robin_alpha=1.0
        )

        velocity = (
            adapter.velocity_for_galerkin()
            if _has_velocity(vel_strs)
            else None
        )

        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=4.0,
            bkd=bkd,
            velocity=velocity,
            forcing=adapter.forcing_for_galerkin(),
            boundary_conditions=bc_set.all_conditions(),
        )

        ode_adapter = GalerkinPhysicsODEAdapter(physics)
        stepper = CrankNicolsonResidual(ode_adapter)

        exact_sol_func = adapter.solution_function()
        dof_coords = bkd.to_numpy(basis.dof_coordinates())

        def exact_at_time(t):
            u = exact_sol_func(dof_coords, t)
            if hasattr(u, "shape") and u.ndim > 1:
                return u[:, 0] if u.shape[1] == 1 else u.flatten()
            return u

        y = bkd.asarray(exact_at_time(0.0))
        dt = 1.0
        nsteps = 5
        t = 0.0

        for step in range(nsteps):
            stepper.set_time(t, dt, y)
            y_new = bkd.copy(y)
            for newton_iter in range(10):
                res = stepper(y_new)
                res_norm = float(np.linalg.norm(bkd.to_numpy(res)))
                if res_norm < 1e-10:
                    break
                jac = stepper.jacobian(y_new)
                dy = bkd.solve(jac, -res)
                y_new = y_new + dy
            y = y_new
            t += dt

        u_exact_final = exact_at_time(t)
        u_num = bkd.to_numpy(y)

        u_norm = np.linalg.norm(u_exact_final)
        if u_norm > 1e-10:
            rel_error = np.linalg.norm(u_num - u_exact_final) / u_norm
        else:
            rel_error = np.linalg.norm(u_num - u_exact_final)

        self.assertLess(
            rel_error, 1e-6,
            f"Test {name}: rel_error={rel_error:.2e} at t={t} "
            f"should be < 1e-6",
        )


from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


if __name__ == "__main__":
    unittest.main()
