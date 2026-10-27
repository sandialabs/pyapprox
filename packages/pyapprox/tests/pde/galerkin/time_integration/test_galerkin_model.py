"""Smoke tests for the Galerkin steady and transient models.

Validates that GalerkinTransientModel.solve_transient produces the same
results as the manual time-stepping loop for all 4 integration methods,
and how both models are constructed.
"""


import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Callable, Tuple

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.manufactured import (
    GalerkinManufacturedSolution,
    create_adr_manufactured_test,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration import (
    GalerkinSteadyModel,
    GalerkinTransientModel,
    TimeIntegrationConfig,
    skfem_ode_residual,
)
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.markers import slow_test

_ExactAtTime = Callable[[float], NDArray[np.floating[Any]]]


def _setup_adr_system(
    bkd: NumpyBkd, nx: int = 32
) -> Tuple[GalerkinSystem[NDArray[Any]], _ExactAtTime]:
    """Create a simple 1D ADR system with manufactured solution."""
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

    ms = GalerkinManufacturedSolution(basis, functions, bkd)

    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=4.0,
        bkd=bkd,
        forcing=ms.forcing(),
    )
    system = compose_galerkin_system(
        physics, [ms.dirichlet("left"), ms.dirichlet("right")]
    )

    exact_sol_func = functions["solution"]
    dof_coords = bkd.to_numpy(basis.dof_coordinates())

    def exact_at_time(t: float) -> NDArray[np.floating[Any]]:
        u = exact_sol_func(dof_coords, t)
        if hasattr(u, "shape") and u.ndim > 1:
            return np.asarray(u[:, 0] if u.shape[1] == 1 else u.flatten())
        return np.asarray(u)

    return system, exact_at_time


IMPLICIT_CASES = [
    "backward_euler",
    "crank_nicolson",
    "implicit_midpoint",
]


class TestGalerkinTransientModelImplicit:
    """Test GalerkinTransientModel with implicit methods."""

    @pytest.mark.parametrize(
        "method",
        IMPLICIT_CASES,
    )
    def test_solve_transient_implicit(self, numpy_bkd: NumpyBkd, method: str) -> None:
        """GalerkinTransientModel matches exact solution for time-linear problem."""
        bkd = numpy_bkd
        system, exact_at_time = _setup_adr_system(bkd)

        model = GalerkinTransientModel(skfem_ode_residual(system))

        y0 = bkd.asarray(exact_at_time(0.0))
        config: TimeIntegrationConfig[Any] = TimeIntegrationConfig(
            method=method,
            init_time=0.0,
            final_time=5.0,
            deltat=1.0,
            newton_tol=1e-10,
            newton_maxiter=20,
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
class TestGalerkinTransientModelExplicit:
    """Test GalerkinTransientModel with explicit methods (CFL-constrained)."""

    @pytest.mark.parametrize(
        "method",
        EXPLICIT_CASES,
    )
    def test_solve_transient_explicit(self, numpy_bkd: NumpyBkd, method: str) -> None:
        """GalerkinTransientModel matches exact solution for time-linear problem.

        Uses nx=4, P2 which exactly represents u=(1-x)*x*(1+T).
        Zero spatial error + linear-in-time → machine precision.
        CFL: h=0.25, D=4, dt < h²/(2D) = 0.0078 → dt=1e-5 well within.
        """
        bkd = numpy_bkd
        system, exact_at_time = _setup_adr_system(bkd, nx=4)

        model = GalerkinTransientModel(skfem_ode_residual(system))

        y0 = bkd.asarray(exact_at_time(0.0))
        config: TimeIntegrationConfig[Any] = TimeIntegrationConfig(
            method=method,
            init_time=0.0,
            final_time=5e-4,
            deltat=1e-5,
            newton_tol=1e-10,
            newton_maxiter=20,
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
    ) -> Tuple[GalerkinSystem[NDArray[Any]], _ExactAtTime]:
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
        ms = GalerkinManufacturedSolution(basis, functions, bkd)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=1e-2,
            bkd=bkd,
            forcing=ms.forcing(),
        )
        system = compose_galerkin_system(
            physics, [ms.dirichlet("left"), ms.dirichlet("right")]
        )
        exact_sol_func = functions["solution"]
        dof_coords = bkd.to_numpy(basis.dof_coordinates())

        def exact_at_time(t: float) -> NDArray[np.floating[Any]]:
            u = exact_sol_func(dof_coords, t)
            if hasattr(u, "shape") and u.ndim > 1:
                return np.asarray(
                    u[:, 0] if u.shape[1] == 1 else u.flatten()
                )
            return np.asarray(u)

        return system, exact_at_time

    @pytest.mark.parametrize("lumped", [False, True])
    def test_forward_euler_time_varying_dirichlet(
        self, numpy_bkd: NumpyBkd, lumped: bool
    ) -> None:
        """Constraint rows impose g(t_{n+1}) exactly; lumped and
        consistent mass both reproduce the constant-in-space solution
        to machine precision (row-sum lumping preserves constants)."""
        bkd = numpy_bkd
        system, exact_at_time = self._setup_constant_in_space(bkd)
        model = GalerkinTransientModel(skfem_ode_residual(system, lumped_mass=lumped))
        y0 = bkd.asarray(exact_at_time(0.0))
        config: TimeIntegrationConfig[Any] = TimeIntegrationConfig(
            method="forward_euler",
            init_time=0.0,
            final_time=0.5,
            deltat=0.05,
            newton_tol=1e-10,
            newton_maxiter=20,
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
        cs = system.constraint_set()
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
        ms = GalerkinManufacturedSolution(basis, functions, bkd)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=1e-2,
            bkd=bkd,
            forcing=ms.forcing(),
        )
        system = compose_galerkin_system(
            physics, [ms.dirichlet("left"), ms.dirichlet("right")]
        )
        exact_sol_func = functions["solution"]
        dof_coords = bkd.to_numpy(basis.dof_coordinates())

        model = GalerkinTransientModel(skfem_ode_residual(system))
        y0 = bkd.asarray(exact_sol_func(dof_coords, 0.0).flatten())
        config: TimeIntegrationConfig[Any] = TimeIntegrationConfig(
            method="heun",
            init_time=0.0,
            final_time=0.5,
            deltat=0.05,
            newton_tol=1e-10,
            newton_maxiter=20,
            verbosity=0,
        )
        solutions, times = model.solve_transient(y0, config)

        u_exact = exact_sol_func(dof_coords, float(times[-1])).flatten()
        u_num = bkd.to_numpy(solutions[:, -1])
        rel_error = np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact)
        assert rel_error < 1e-12


class TestModelConstruction:
    def test_steady_model_rejects_a_non_system(self) -> None:
        with pytest.raises(TypeError, match="GalerkinSteadySystemProtocol"):
            GalerkinSteadyModel(object())  # type: ignore[arg-type]

    def test_steady_model_takes_its_backend_from_the_system(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        system, _ = _setup_adr_system(numpy_bkd, nx=4)
        assert GalerkinSteadyModel(system).bkd() is system.bkd()

    def test_transient_model_rejects_a_non_adapter(self, numpy_bkd: NumpyBkd) -> None:
        system, _ = _setup_adr_system(numpy_bkd, nx=4)
        with pytest.raises(TypeError, match="GalerkinODEResidualProtocol"):
            GalerkinTransientModel(system)  # type: ignore[arg-type]

    def test_transient_model_takes_everything_from_its_adapter(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """The system and backend come from the adapter, so they are
        never passed twice."""
        system, _ = _setup_adr_system(numpy_bkd, nx=4)
        adapter = skfem_ode_residual(system)
        model = GalerkinTransientModel(adapter)
        assert model.adapter() is adapter
        assert model.system() is system
        assert model.bkd() is system.bkd()
        assert model.nstates() == system.nstates()
