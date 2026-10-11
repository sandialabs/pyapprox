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

from pyapprox.pde.boundary import BoundaryConditionRole
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary import NormalFluxLawProtocol
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.manufactured.adapter import (
    GalerkinManufacturedSolution,
    create_adr_manufactured_test,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration import (
    GalerkinModel,
    TimeIntegrationConfig,
)
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.markers import slow_test

_BOUNDARY_NAMES = ["left", "right"]
_ROBIN_ALPHA = 1.0


def _dirichlet(
    ms: GalerkinManufacturedSolution[Array], name: str, law: NormalFluxLawProtocol
) -> BoundaryConditionRole[Array]:
    return ms.dirichlet(name)


def _neumann(
    ms: GalerkinManufacturedSolution[Array], name: str, law: NormalFluxLawProtocol
) -> BoundaryConditionRole[Array]:
    return ms.neumann(name, law)


def _robin(
    ms: GalerkinManufacturedSolution[Array], name: str, law: NormalFluxLawProtocol
) -> BoundaryConditionRole[Array]:
    return ms.robin(name, _ROBIN_ALPHA, law)


# Test cases name BC types by code, in boundary order.
_BC_BUILDERS: Dict[
    str,
    Callable[
        [GalerkinManufacturedSolution[Array], str, NormalFluxLawProtocol],
        BoundaryConditionRole[Array],
    ],
] = {"D": _dirichlet, "N": _neumann, "R": _robin}


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
    bkd: Backend[Array],
    sol_str: str,
    diff_str: str,
    vel_strs: List[str],
    bndry_types: List[str],
    nx: int = 4,
    diffusivity: float = 0.1,
) -> Tuple[GalerkinModel[Array], Callable[[float], NDArray[np.floating[Any]]]]:
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

    ms = GalerkinManufacturedSolution(basis, functions, bkd)

    has_vel = not all("1e-16" in v for v in vel_strs)
    velocity = ms.velocity() if has_vel else None

    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=diffusivity,
        bkd=bkd,
        velocity=velocity,
        forcing=ms.forcing(),
    )
    law = physics.flux_law()
    bcs = [
        _BC_BUILDERS[code](ms, name, law)
        for code, name in zip(bndry_types, _BOUNDARY_NAMES)
    ]
    system = compose_galerkin_system(physics, bcs)

    model = GalerkinModel(system, bkd)

    exact_sol_func = functions["solution"]
    dof_coords = bkd.to_numpy(basis.dof_coordinates())

    def exact_at_time(t: float) -> NDArray[np.floating[Any]]:
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
        numpy_bkd,
        method: str,
        expected_order: float,
    ) -> None:
        bkd = numpy_bkd
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
                newton_tol=1e-10,
                newton_maxiter=20,
                lumped_mass=False,
                verbosity=0,
            )
            solutions, times = model.solve_transient(y0, config)

            u_exact = exact_at_time(float(times[-1]))
            u_num = bkd.to_numpy(solutions[:, -1])
            error = np.linalg.norm(u_num - u_exact)
            errors.append(error)

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
        numpy_bkd,
        method: str,
        expected_order: float,
    ) -> None:
        bkd = numpy_bkd
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
                newton_tol=1e-10,
                newton_maxiter=20,
                lumped_mass=False,
                verbosity=0,
            )
            solutions, times = model.solve_transient(y0, config)

            u_exact = exact_at_time(float(times[-1]))
            u_num = bkd.to_numpy(solutions[:, -1])
            error = np.linalg.norm(u_num - u_exact)
            errors.append(error)

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
        numpy_bkd,
        method: str,
        expected_order: float,
    ) -> None:
        bkd = numpy_bkd
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
                newton_tol=1e-10,
                newton_maxiter=20,
                lumped_mass=False,
                verbosity=0,
            )
            solutions, times = model.solve_transient(y0, config)

            u_exact = exact_at_time(float(times[-1]))
            u_num = bkd.to_numpy(solutions[:, -1])
            error = np.linalg.norm(u_num - u_exact)
            errors.append(error)

        rates = _compute_convergence_rates(dt_values, errors)

        min_expected = expected_order - 0.15
        assert np.all(rates > min_expected), (
            f"{method}: rates={rates}, errors={errors}, expected > {min_expected}"
        )
