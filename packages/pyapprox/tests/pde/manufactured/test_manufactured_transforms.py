"""Transient advection-diffusion on a polar (non-affine) mesh.

Backward Euler and Crank-Nicolson on the half annulus 1 <= r <= 2,
-pi/2 <= theta <= pi/2, with time-dependent Dirichlet data. The steady
curvilinear cases are in the collocation verification catalogs, on full
rings.
"""

import math
from typing import List, Tuple

import pytest

from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary import DirichletBC
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms.polar import PolarTransform
from pyapprox.pde.collocation.physics import AdvectionDiffusionReaction
from pyapprox.pde.collocation.physics.coefficients import NodalSupplier
from pyapprox.pde.collocation.protocols.boundary import BoundaryConditionProtocol
from pyapprox.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.pde.manufactured import ManufacturedAdvectionDiffusionReaction
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.markers import slow_test


def _setup(
    bkd: Backend[Array],
) -> Tuple[
    TransformedMesh2D[Array],
    AdvectionDiffusionReaction[Array],
    ManufacturedAdvectionDiffusionReaction[Array],
]:
    """25 x 25 nodes on the half annulus, u = x^2 y^2 (1 + t), D = 1,
    forcing and Dirichlet data from the manufactured solution."""
    transform = PolarTransform(
        r_bounds=(1.0, 2.0), theta_bounds=(-math.pi / 2, math.pi / 2), bkd=bkd
    )
    mesh = TransformedMesh2D(25, 25, bkd, transform)
    basis = ChebyshevBasis2D(mesh, bkd)
    # Linear in time: backward Euler and Crank-Nicolson are exact.
    man_sol = ManufacturedAdvectionDiffusionReaction(
        sol_str="x**2*y**2*(1 + T)",
        nvars=2,
        diff_str="1.0",
        react_str="0",
        vel_strs=["0", "0"],
        bkd=bkd,
        oned=True,
    )
    points = mesh.points()
    physics = AdvectionDiffusionReaction(
        basis,
        bkd,
        diffusion=1.0,
        forcing=NodalSupplier(TimeDependent(man_sol.functions["forcing"]), points),
    )
    bcs: List[BoundaryConditionProtocol[Array]] = []
    for side in range(4):
        boundary_idx = mesh.boundary_indices(side)
        side_points = points[:, boundary_idx]

        def values(t: float, pts: Array = side_points) -> Array:
            side_values: Array = man_sol.functions["solution"](pts, t)
            return side_values

        bcs.append(DirichletBC(bkd, boundary_idx, values))
    physics.set_boundary_conditions(bcs)
    return mesh, physics, man_sol


def _final_error(bkd: Backend[Array], method: str, deltat: float) -> None:
    """Integrate to t = 1 and compare with the exact solution there."""
    mesh, physics, man_sol = _setup(bkd)
    points = mesh.points()
    config: TimeIntegrationConfig[Array] = TimeIntegrationConfig(
        final_time=1.0,
        deltat=deltat,
        method=method,
        init_time=0.0,
        newton_tol=1e-10,
        newton_maxiter=20,
        verbosity=0,
    )
    initial: Array = man_sol.functions["solution"](points, 0.0)
    solutions, times = CollocationModel(physics, bkd).solve_transient(initial, config)
    bkd.assert_allclose(times[-1:], bkd.asarray([1.0]))
    exact: Array = man_sol.functions["solution"](points, 1.0)
    bkd.assert_allclose(solutions[:, -1], exact, atol=1e-8)


@slow_test
def test_transient_polar_backward_euler(bkd: Backend[Array]) -> None:
    """Backward Euler, step 0.25: exact in time for a solution linear in
    time, so only the spatial error (below 1e-8) remains."""
    _final_error(bkd, "backward_euler", 0.25)


@pytest.mark.slow_on("TorchBkd")
def test_transient_polar_crank_nicolson(bkd: Backend[Array]) -> None:
    """Crank-Nicolson, step 0.5: exact in time for a solution linear in
    time, so only the spatial error (below 1e-8) remains."""
    _final_error(bkd, "crank_nicolson", 0.5)
