"""Smoke tests for GalerkinModel.

Validates that GalerkinModel.solve_transient produces the same results
as the manual time-stepping loop for all 4 integration methods.
"""


import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Callable, Tuple

import numpy as np
from numpy.typing import NDArray
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

from tests._helpers.markers import slow_test

_ExactAtTime = Callable[[float], NDArray[np.floating[Any]]]


def _setup_adr_physics(
    bkd: NumpyBkd, nx: int = 32
) -> Tuple[AdvectionDiffusionReaction[Any], _ExactAtTime]:
    """Create a simple 1D ADR physics with manufactured solution."""
    bounds = [0.0, 1.0]
    sol_str = "(1-x)*x*(1+T)"
    diff_str = "4+1e-16*x"
    react_str = "0*u"
    vel_strs = ["0+1e-16*x"]

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
    bc_set = adapter.create_boundary_conditions(["D", "D"], robin_alpha=1.0)

    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=4.0,
        bkd=bkd,
        forcing=adapter.forcing_for_galerkin(),
        boundary_conditions=bc_set.all_conditions(),
    )

    exact_sol_func = adapter.solution_function()
    dof_coords = bkd.to_numpy(basis.dof_coordinates())

    def exact_at_time(t: float) -> NDArray[np.floating[Any]]:
        u = exact_sol_func(dof_coords, t)
        if hasattr(u, "shape") and u.ndim > 1:
            return np.asarray(u[:, 0] if u.shape[1] == 1 else u.flatten())
        return np.asarray(u)

    return physics, exact_at_time


IMPLICIT_CASES = [
    "backward_euler",
    "crank_nicolson",
    "implicit_midpoint",
]


class TestGalerkinModelImplicit:
    """Test GalerkinModel with implicit methods."""

    @pytest.mark.parametrize(
        "method",
        IMPLICIT_CASES,
    )
    def test_solve_transient_implicit(self, numpy_bkd: NumpyBkd, method: str) -> None:
        """GalerkinModel matches exact solution for time-linear problem."""
        bkd = numpy_bkd
        physics, exact_at_time = _setup_adr_physics(bkd)

        model = GalerkinModel(physics, bkd)

        y0 = bkd.asarray(exact_at_time(0.0))
        config: TimeIntegrationConfig[Any] = TimeIntegrationConfig(
            method=method,
            init_time=0.0,
            final_time=5.0,
            deltat=1.0,
            newton_tol=1e-10,
            newton_maxiter=20,
            lumped_mass=False,
            verbosity=0,
        )
        solutions, times = model.solve_transient(y0, config)

        u_exact_final = exact_at_time(float(times[-1]))
        u_num = bkd.to_numpy(solutions[:, -1])

        u_norm = np.linalg.norm(u_exact_final)
        rel_error = np.linalg.norm(u_num - u_exact_final) / u_norm

        assert rel_error < 1e-6


EXPLICIT_CASES = [
    "forward_euler",
    "heun",
]


@slow_test
class TestGalerkinModelExplicit:
    """Test GalerkinModel with explicit methods (CFL-constrained)."""

    @pytest.mark.parametrize(
        "method",
        EXPLICIT_CASES,
    )
    def test_solve_transient_explicit(self, numpy_bkd: NumpyBkd, method: str) -> None:
        """GalerkinModel matches exact solution for time-linear problem.

        Uses nx=4, P2 which exactly represents u=(1-x)*x*(1+T).
        Zero spatial error + linear-in-time → machine precision.
        CFL: h=0.25, D=4, dt < h²/(2D) = 0.0078 → dt=1e-5 well within.
        """
        bkd = numpy_bkd
        physics, exact_at_time = _setup_adr_physics(bkd, nx=4)

        model = GalerkinModel(physics, bkd)

        y0 = bkd.asarray(exact_at_time(0.0))
        config: TimeIntegrationConfig[Any] = TimeIntegrationConfig(
            method=method,
            init_time=0.0,
            final_time=5e-4,
            deltat=1e-5,
            newton_tol=1e-10,
            newton_maxiter=20,
            lumped_mass=False,
            verbosity=0,
        )
        solutions, times = model.solve_transient(y0, config)

        u_exact_final = exact_at_time(float(times[-1]))
        u_num = bkd.to_numpy(solutions[:, -1])

        u_norm = np.linalg.norm(u_exact_final)
        rel_error = np.linalg.norm(u_num - u_exact_final) / u_norm

        assert rel_error < 1e-10


class TestExplicitUnifiedPipeline:
    """Explicit methods through the one BC-enforcing pipeline."""

    def _setup_constant_in_space(
        self, bkd: NumpyBkd
    ) -> Tuple[AdvectionDiffusionReaction[Any], _ExactAtTime]:
        """u = 1+T: constant in space, so row-sum lumping is exact and
        the Dirichlet values g(t) = 1+t vary in time."""
        bounds = [0.0, 1.0]
        functions, _ = create_adr_manufactured_test(
            bounds=bounds,
            sol_str="(1+T)+1e-16*x",
            diff_str="1e-2+1e-16*x",
            react_str="0*u",
            vel_strs=["0+1e-16*x"],
            bkd=bkd,
            time_dependent=True,
        )
        mesh = StructuredMesh1D(nx=8, bounds=(bounds[0], bounds[1]), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        adapter = GalerkinManufacturedSolutionAdapter(
            basis, functions, bkd, time_dependent=True
        )
        bc_set = adapter.create_boundary_conditions(["D", "D"])
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=1e-2,
            bkd=bkd,
            forcing=adapter.forcing_for_galerkin(),
            boundary_conditions=bc_set.all_conditions(),
        )
        exact_sol_func = adapter.solution_function()
        dof_coords = bkd.to_numpy(basis.dof_coordinates())

        def exact_at_time(t: float) -> NDArray[np.floating[Any]]:
            u = exact_sol_func(dof_coords, t)
            if hasattr(u, "shape") and u.ndim > 1:
                return np.asarray(
                    u[:, 0] if u.shape[1] == 1 else u.flatten()
                )
            return np.asarray(u)

        return physics, exact_at_time

    @pytest.mark.parametrize("lumped", [False, True])
    def test_forward_euler_time_varying_dirichlet(
        self, numpy_bkd: NumpyBkd, lumped: bool
    ) -> None:
        """Constraint rows impose g(t_{n+1}) exactly; lumped and
        consistent mass both reproduce the constant-in-space solution
        to machine precision (row-sum lumping preserves constants)."""
        bkd = numpy_bkd
        physics, exact_at_time = self._setup_constant_in_space(bkd)
        model = GalerkinModel(physics, bkd)
        y0 = bkd.asarray(exact_at_time(0.0))
        config: TimeIntegrationConfig[Any] = TimeIntegrationConfig(
            method="forward_euler",
            init_time=0.0,
            final_time=0.5,
            deltat=0.05,
            newton_tol=1e-10,
            newton_maxiter=20,
            lumped_mass=lumped,
            verbosity=0,
        )
        solutions, times = model.solve_transient(y0, config)

        u_exact_final = exact_at_time(float(times[-1]))
        u_num = bkd.to_numpy(solutions[:, -1])
        rel_error = np.linalg.norm(u_num - u_exact_final) / np.linalg.norm(
            u_exact_final
        )
        assert rel_error < 1e-12

        # Dirichlet DOFs hit g(t) exactly at every stored time
        cs = physics.constraint_set()
        dofs = bkd.to_numpy(cs.dofs())
        for jj, t in enumerate(bkd.to_numpy(times)):
            g = bkd.to_numpy(cs.values(float(t)))
            bkd.assert_allclose(
                bkd.asarray(bkd.to_numpy(solutions[:, jj])[dofs]),
                bkd.asarray(g),
                rtol=1e-12,
                atol=1e-13,
            )

    def test_heun_time_varying_dirichlet_witness(self, numpy_bkd: NumpyBkd) -> None:
        """Stage-BC correctness witness: Heun + consistent mass +
        time-varying Dirichlet g(t) = 1+t. The manufactured analytic
        g_dot makes the stage slopes exact; the solution is linear in
        time and exactly representable (P2), so the trajectory is
        machine-precise. Without the g_dot-carrying stage rows this
        errs at O(dt) per step."""
        bkd = numpy_bkd
        bounds = [0.0, 1.0]
        functions, _ = create_adr_manufactured_test(
            bounds=bounds,
            sol_str="(1+x*(1-x))*(1+T)",
            diff_str="1e-2+1e-16*x",
            react_str="0*u",
            vel_strs=["0+1e-16*x"],
            bkd=bkd,
            time_dependent=True,
        )
        mesh = StructuredMesh1D(nx=8, bounds=(bounds[0], bounds[1]), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=2)
        adapter = GalerkinManufacturedSolutionAdapter(
            basis, functions, bkd, time_dependent=True
        )
        bc_set = adapter.create_boundary_conditions(["D", "D"])
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=1e-2,
            bkd=bkd,
            forcing=adapter.forcing_for_galerkin(),
            boundary_conditions=bc_set.all_conditions(),
        )
        exact_sol_func = adapter.solution_function()
        dof_coords = bkd.to_numpy(basis.dof_coordinates())

        model = GalerkinModel(physics, bkd)
        y0 = bkd.asarray(exact_sol_func(dof_coords, 0.0).flatten())
        config: TimeIntegrationConfig[Any] = TimeIntegrationConfig(
            method="heun",
            init_time=0.0,
            final_time=0.5,
            deltat=0.05,
            newton_tol=1e-10,
            newton_maxiter=20,
            lumped_mass=False,
            verbosity=0,
        )
        solutions, times = model.solve_transient(y0, config)

        u_exact = exact_sol_func(dof_coords, float(times[-1])).flatten()
        u_num = bkd.to_numpy(solutions[:, -1])
        rel_error = np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact)
        assert rel_error < 1e-12
