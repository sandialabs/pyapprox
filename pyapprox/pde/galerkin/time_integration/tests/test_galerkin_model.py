"""Smoke tests for GalerkinModel.

Validates that GalerkinModel.solve_transient produces the same results
as the manual time-stepping loop for all 4 integration methods.
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


def _setup_adr_physics(bkd, nx=32):
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

    def exact_at_time(t):
        u = exact_sol_func(dof_coords, t)
        if hasattr(u, "shape") and u.ndim > 1:
            return u[:, 0] if u.shape[1] == 1 else u.flatten()
        return u

    return physics, exact_at_time


IMPLICIT_CASES = [
    ("backward_euler",),
    ("crank_nicolson",),
]


class TestGalerkinModelImplicit(ParametrizedTestCase):
    """Test GalerkinModel with implicit methods."""

    @parametrize(
        "method",
        IMPLICIT_CASES,
    )
    def test_solve_transient_implicit(self, method: str) -> None:
        """GalerkinModel matches exact solution for time-linear problem."""
        bkd = NumpyBkd()
        physics, exact_at_time = _setup_adr_physics(bkd)

        model = GalerkinModel(physics, bkd)

        y0 = bkd.asarray(exact_at_time(0.0))
        config = TimeIntegrationConfig(
            method=method,
            init_time=0.0,
            final_time=5.0,
            deltat=1.0,
        )
        solutions, times = model.solve_transient(y0, config)

        u_exact_final = exact_at_time(float(times[-1]))
        u_num = bkd.to_numpy(solutions[:, -1])

        u_norm = np.linalg.norm(u_exact_final)
        rel_error = np.linalg.norm(u_num - u_exact_final) / u_norm

        self.assertLess(
            rel_error,
            1e-6,
            f"Method {method}: rel_error={rel_error:.2e} should be < 1e-6",
        )


EXPLICIT_CASES = [
    ("forward_euler",),
    ("heun",),
]


@slow_test
class TestGalerkinModelExplicit(ParametrizedTestCase):
    """Test GalerkinModel with explicit methods (CFL-constrained)."""

    @parametrize(
        "method",
        EXPLICIT_CASES,
    )
    def test_solve_transient_explicit(self, method: str) -> None:
        """GalerkinModel matches exact solution for time-linear problem.

        Uses nx=4, P2 which exactly represents u=(1-x)*x*(1+T).
        Zero spatial error + linear-in-time → machine precision.
        CFL: h=0.25, D=4, dt < h²/(2D) = 0.0078 → dt=1e-5 well within.
        """
        bkd = NumpyBkd()
        physics, exact_at_time = _setup_adr_physics(bkd, nx=4)

        model = GalerkinModel(physics, bkd)

        y0 = bkd.asarray(exact_at_time(0.0))
        config = TimeIntegrationConfig(
            method=method,
            init_time=0.0,
            final_time=5e-4,
            deltat=1e-5,
        )
        solutions, times = model.solve_transient(y0, config)

        u_exact_final = exact_at_time(float(times[-1]))
        u_num = bkd.to_numpy(solutions[:, -1])

        u_norm = np.linalg.norm(u_exact_final)
        rel_error = np.linalg.norm(u_num - u_exact_final) / u_norm

        self.assertLess(
            rel_error,
            1e-10,
            f"Method {method}: rel_error={rel_error:.2e} should be < 1e-10",
        )


from pyapprox.util.test_utils import load_tests  # noqa: F401

if __name__ == "__main__":
    unittest.main()
