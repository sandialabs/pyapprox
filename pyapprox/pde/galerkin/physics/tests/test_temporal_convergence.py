"""Temporal convergence rate tests for Galerkin FEM time integration.

Verifies expected convergence rates for all 4 time stepping methods:
- Backward Euler: O(dt) — first order
- Crank-Nicolson: O(dt^2) — second order
- Forward Euler: O(dt) — first order
- Heun (RK2): O(dt^2) — second order

Uses manufactured solutions that are spatially exact on P2 with nx=4
(quadratic in x), so the only error is from temporal discretization.
Time dependence is cubic (T^3) so both first- and second-order methods
see nonzero truncation error.
"""

import unittest

import numpy as np
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.manufactured.adapter import (
    GalerkinManufacturedSolutionAdapter,
    create_adr_manufactured_test,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration import (
    GalerkinModel,
    TimeIntegrationConfig,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.test_utils import slow_test


def _compute_convergence_rates(dt_values, errors):
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
    rates = np.log(err_arr[:-1] / err_arr[1:]) / np.log(dt_arr[:-1] / dt_arr[1:])
    return rates


def _setup_physics_and_model(
    bkd,
    sol_str,
    diff_str,
    vel_strs,
    bndry_types,
    nx=4,
    diffusivity=0.1,
):
    """Create Galerkin physics and model for a 1D diffusion problem.

    Returns
    -------
    model : GalerkinModel
    exact_at_time : callable
        Returns exact DOF values at given time.
    """
    bounds = [0.0, 1.0]
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

    mesh = StructuredMesh1D(nx=nx, bounds=(bounds[0], bounds[1]), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=2)

    adapter = GalerkinManufacturedSolutionAdapter(
        basis, functions, bkd, time_dependent=True
    )
    bc_set = adapter.create_boundary_conditions(bndry_types, robin_alpha=1.0)

    has_vel = not all("1e-16" in v for v in vel_strs)
    velocity = adapter.velocity_for_galerkin() if has_vel else None

    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=diffusivity,
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

    return model, exact_at_time


# =========================================================================
# Test 1: Homogeneous Dirichlet BCs
# u(x,t) = (1-x)*x*(1+T^3) — P2 exact on nx=4, cubic in time
# u(0,t) = u(1,t) = 0
# =========================================================================

# Implicit methods: larger dt is stable
IMPLICIT_CONVERGENCE_CASES = [
    ("backward_euler", 1.0),
    ("crank_nicolson", 2.0),
]

# Explicit methods: dt must satisfy CFL
# D=0.1, h=0.25 (nx=4) → spectral radius ~80 → dt_max ≈ 0.025
EXPLICIT_CONVERGENCE_CASES = [
    ("forward_euler", 1.0),
    ("heun", 2.0),
]


@slow_test
class TestTemporalConvergenceImplicit(ParametrizedTestCase):
    """Temporal convergence for implicit methods with homogeneous Dirichlet.

    u(x,t) = (1-x)*x*(1+T^3), D=0.1, no velocity, no reaction.
    P2 on nx=4 is spatially exact. Error is purely temporal.
    """

    @parametrize(
        "method,expected_order",
        IMPLICIT_CONVERGENCE_CASES,
    )
    def test_convergence_homogeneous_dirichlet(
        self,
        method: str,
        expected_order: float,
    ) -> None:
        bkd = NumpyBkd()
        model, exact_at_time = _setup_physics_and_model(
            bkd,
            sol_str="(1-x)*x*(1+T**3)",
            diff_str="0.1+1e-16*x",
            vel_strs=["0+1e-16*x"],
            bndry_types=["D", "D"],
        )

        dt_values = [0.1, 0.05, 0.025]
        final_time = 1.0
        errors = []

        for dt in dt_values:
            y0 = bkd.asarray(exact_at_time(0.0))
            config = TimeIntegrationConfig(
                method=method,
                init_time=0.0,
                final_time=final_time,
                deltat=dt,
            )
            solutions, times = model.solve_transient(y0, config)

            u_exact = exact_at_time(float(times[-1]))
            u_num = bkd.to_numpy(solutions[:, -1])
            error = np.linalg.norm(u_num - u_exact)
            errors.append(error)

        rates = _compute_convergence_rates(dt_values, errors)

        min_expected = expected_order - 0.15
        self.assertTrue(
            np.all(rates > min_expected),
            f"{method}: rates={rates}, errors={errors}, expected > {min_expected}",
        )


@slow_test
class TestTemporalConvergenceExplicit(ParametrizedTestCase):
    """Temporal convergence for explicit methods with homogeneous Dirichlet.

    u(x,t) = (1-x)*x*(1+T^3), D=0.1, no velocity, no reaction.
    P2 on nx=4 is spatially exact. Error is purely temporal.
    CFL-constrained dt values: 0.01, 0.005, 0.0025.
    """

    @parametrize(
        "method,expected_order",
        EXPLICIT_CONVERGENCE_CASES,
    )
    def test_convergence_homogeneous_dirichlet(
        self,
        method: str,
        expected_order: float,
    ) -> None:
        bkd = NumpyBkd()
        model, exact_at_time = _setup_physics_and_model(
            bkd,
            sol_str="(1-x)*x*(1+T**3)",
            diff_str="0.1+1e-16*x",
            vel_strs=["0+1e-16*x"],
            bndry_types=["D", "D"],
        )

        dt_values = [0.01, 0.005, 0.0025]
        final_time = 1.0
        errors = []

        for dt in dt_values:
            y0 = bkd.asarray(exact_at_time(0.0))
            config = TimeIntegrationConfig(
                method=method,
                init_time=0.0,
                final_time=final_time,
                deltat=dt,
            )
            solutions, times = model.solve_transient(y0, config)

            u_exact = exact_at_time(float(times[-1]))
            u_num = bkd.to_numpy(solutions[:, -1])
            error = np.linalg.norm(u_num - u_exact)
            errors.append(error)

        rates = _compute_convergence_rates(dt_values, errors)

        min_expected = expected_order - 0.15
        self.assertTrue(
            np.all(rates > min_expected),
            f"{method}: rates={rates}, errors={errors}, expected > {min_expected}",
        )


# =========================================================================
# Test 2: Time-varying nonzero Dirichlet BCs
# u(x,t) = x*(1+T^3) — P2 exact on nx=4, cubic in time
# u(0,t) = 0, u(1,t) = 1+t^3 (nonzero, time-varying)
# =========================================================================


@slow_test
class TestTemporalConvergenceNonzeroDirichlet(ParametrizedTestCase):
    """Temporal convergence with time-varying nonzero Dirichlet BCs.

    u(x,t) = x*(1+T^3), D=0.1, no velocity, no reaction.
    u(0,t) = 0, u(1,t) = 1+t^3.
    P2 on nx=4 is spatially exact. Error is purely temporal.
    Tests both BE (order 1) and CN (order 2).
    """

    @parametrize(
        "method,expected_order",
        IMPLICIT_CONVERGENCE_CASES,
    )
    def test_convergence_nonzero_dirichlet(
        self,
        method: str,
        expected_order: float,
    ) -> None:
        bkd = NumpyBkd()
        model, exact_at_time = _setup_physics_and_model(
            bkd,
            sol_str="x*(1+T**3)",
            diff_str="0.1+1e-16*x",
            vel_strs=["0+1e-16*x"],
            bndry_types=["D", "D"],
        )

        dt_values = [0.1, 0.05, 0.025]
        final_time = 1.0
        errors = []

        for dt in dt_values:
            y0 = bkd.asarray(exact_at_time(0.0))
            config = TimeIntegrationConfig(
                method=method,
                init_time=0.0,
                final_time=final_time,
                deltat=dt,
            )
            solutions, times = model.solve_transient(y0, config)

            u_exact = exact_at_time(float(times[-1]))
            u_num = bkd.to_numpy(solutions[:, -1])
            error = np.linalg.norm(u_num - u_exact)
            errors.append(error)

        rates = _compute_convergence_rates(dt_values, errors)

        min_expected = expected_order - 0.15
        self.assertTrue(
            np.all(rates > min_expected),
            f"{method}: rates={rates}, errors={errors}, expected > {min_expected}",
        )


from pyapprox.util.test_utils import load_tests  # noqa: F401

if __name__ == "__main__":
    unittest.main()
