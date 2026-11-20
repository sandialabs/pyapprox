"""Integration tests for Robin BCs with full PDE solves: time-dependent
Robin data on the transient path, and Robin sides on a polar mesh.

Steady Neumann and Robin solves on boxes, in both the gradient and flux
conventions, are verified by the collocation catalogs
(``tests/pde/models/collocation/test_verification_*.py``).
"""

import math
from typing import List

from pyapprox.pde.collocation.basis import (
    ChebyshevBasis1D,
    ChebyshevBasis2D,
)
from pyapprox.pde.collocation.boundary import (
    DirichletBC,
    flux_robin_bc,
    gradient_robin_bc,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    TransformedMesh2D,
    create_uniform_mesh_1d,
)
from pyapprox.pde.collocation.mesh.transforms.polar import PolarTransform
from pyapprox.pde.collocation.physics import AdvectionDiffusionReaction
from pyapprox.pde.collocation.physics.coefficients import (
    NodalField,
    NodalSupplier,
)
from pyapprox.pde.collocation.protocols.boundary import BoundaryConditionProtocol
from pyapprox.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)
from pyapprox.pde.constitutive.advection import ConservativeAdvection
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.pde.manufactured import (
    ManufacturedAdvectionDiffusionReaction,
)
from pyapprox.util.backends.protocols import Array, Backend


class TestTransientRobinSolve:
    """Integration tests for time-dependent Robin BCs."""

    def _solve_1d_transient_robin(
        self,
        convention: str,
        diffusion: float,
        velocity_val: int,
        bkd: Backend[Array],
    ) -> None:
        """Solve 1D transient with Robin left, Dirichlet right.

        u = (x^2+1)*(1+T), linear in time => backward Euler exact.
        """
        npts = 15
        tmesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(tmesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        vel_strs = [str(velocity_val)]
        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="(x**2 + 1)*(1 + T)",
            nvars=1,
            diff_str=str(diffusion),
            react_str="u",
            vel_strs=vel_strs,
            bkd=bkd,
            oned=True,
            # With zero velocity both forms add nothing, so one form
            # serves every case.
            advection_form=ConservativeAdvection(),
        )

        pts = tmesh.points()

        velocity = [bkd.full((npts,), float(velocity_val))] if velocity_val else None
        physics = AdvectionDiffusionReaction(
            basis,
            bkd,
            diffusion=diffusion,
            reaction=1.0,
            velocity=velocity,
            forcing=NodalSupplier(TimeDependent(man_sol.functions["forcing"]), pts),
            advection_form=ConservativeAdvection(),
        )

        # Left: time-dependent Robin, alpha=1, beta=1
        left_idx = umesh.boundary_indices(0)
        left_normals = tmesh.boundary_normals(0)
        left_pts = pts[:, left_idx]

        def robin_values_fn(t: float) -> Array:
            values: Array = man_sol.robin_values(
                left_pts, left_normals, 1.0, 1.0, time=t, convention=convention
            )
            return values

        if convention == "gradient":
            bc_left = gradient_robin_bc(
                bkd,
                left_idx,
                left_normals,
                [basis.derivative_matrix()],
                1.0,
                1.0,
                robin_values_fn,
            )
        else:
            bc_left = flux_robin_bc(
                bkd,
                left_idx,
                left_normals,
                physics,
                1.0,
                1.0,
                robin_values_fn,
            )

        # Right: time-dependent Dirichlet
        right_idx = umesh.boundary_indices(1)
        right_pts = pts[:, right_idx]

        def right_dirichlet_fn(t: float) -> Array:
            values: Array = man_sol.functions["solution"](right_pts, t)
            return values

        bc_right = DirichletBC(bkd, right_idx, right_dirichlet_fn)
        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)
        u_initial = man_sol.functions["solution"](pts, 0.0)
        config: TimeIntegrationConfig[Array] = TimeIntegrationConfig(
            method="backward_euler",
            final_time=1.0,
            deltat=0.25,
            init_time=0.0,
            newton_tol=1e-10,
            newton_maxiter=20,
            verbosity=0,
        )

        solutions, times = model.solve_transient(u_initial, config)
        t_final = float(bkd.to_numpy(times[-1]))
        u_exact_final = man_sol.functions["solution"](pts, t_final)
        bkd.assert_allclose(solutions[:, -1], u_exact_final, atol=1e-8)

    def test_transient_gradient_robin(self, bkd: Backend[Array]) -> None:
        """1D transient gradient Robin, D=1, no velocity."""
        self._solve_1d_transient_robin("gradient", 1.0, 0, bkd)

    def test_transient_flux_robin(self, bkd: Backend[Array]) -> None:
        """1D transient flux Robin, D=2, v=1."""
        self._solve_1d_transient_robin("flux", 2.0, 1, bkd)


class TestCurvilinearRobinSolve:
    """Integration tests for Robin BCs on curvilinear meshes."""

    def test_polar_gradient_robin_solve(self, bkd: Backend[Array]) -> None:
        """Polar domain r in [1,2], theta in [-pi/2,pi/2].

        u = x^2*y^2, D=1, reaction=u.
        Robin on r_min (varying normals), Dirichlet on other 3 sides.
        """
        npts = 25
        transform = PolarTransform(
            r_bounds=(1.0, 2.0),
            theta_bounds=(-math.pi / 2, math.pi / 2),
            bkd=bkd,
        )
        tmesh = TransformedMesh2D(npts, npts, bkd, transform)
        basis = ChebyshevBasis2D(tmesh, bkd)

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="x**2*y**2",
            nvars=2,
            diff_str="1.0",
            react_str="u",
            vel_strs=["0", "0"],
            bkd=bkd,
            oned=True,
        )

        pts = tmesh.points()
        u_exact = man_sol.functions["solution"](pts)
        forcing = man_sol.functions["forcing"](pts)

        physics = AdvectionDiffusionReaction(
            basis,
            bkd,
            diffusion=1.0,
            reaction=1.0,
            forcing=NodalField(forcing),
        )

        Dx = basis.derivative_matrix(1, 0)
        Dy = basis.derivative_matrix(1, 1)

        # r_min (side 0): gradient Robin with varying normals
        left_idx = tmesh.boundary_indices(0)
        left_normals = tmesh.boundary_normals(0)
        robin_val = man_sol.robin_values(
            pts[:, left_idx],
            left_normals,
            1.0,
            1.0,
            convention="gradient",
        )
        bc_left = gradient_robin_bc(
            bkd,
            left_idx,
            left_normals,
            [Dx, Dy],
            1.0,
            1.0,
            robin_val,
        )

        # Other 3 sides: Dirichlet
        bcs: List[BoundaryConditionProtocol[Array]] = [bc_left]
        for side in range(1, 4):
            idx = tmesh.boundary_indices(side)
            bcs.append(DirichletBC(bkd, idx, u_exact[idx]))

        physics.set_boundary_conditions(bcs)

        model = CollocationModel(physics, bkd)
        u_numerical = model.solve_steady(
            bkd.zeros((npts * npts,)),
            tol=1e-10,
            maxiter=50,
        )
        bkd.assert_allclose(u_numerical, u_exact, atol=1e-6)
