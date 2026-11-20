"""Integration test for periodic BCs on a curvilinear mesh: an annulus
periodic in θ, through PolarTransform.

Periodic boxes are verified by the collocation ADR catalog
(``tests/pde/models/collocation/test_verification_adr.py``).
"""

import math

from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary import DirichletBC, PeriodicBC
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms.polar import PolarTransform
from pyapprox.pde.collocation.physics import AdvectionDiffusionReaction
from pyapprox.pde.collocation.time_integration import CollocationModel
from pyapprox.pde.manufactured import (
    ManufacturedAdvectionDiffusionReaction,
)
from pyapprox.util.backends.protocols import Array, Backend


class TestPeriodicBCTransforms:
    """Integration tests for periodic BCs on transformed domains."""

    def test_annular_periodic_theta(self, bkd: Backend[Array]) -> None:
        """θ-periodic annular domain, r ∈ [1, 2], θ ∈ [0, 2π].

        u = x = r*cos(θ), D = 1, reaction = u
        Dirichlet at r_min and r_max, periodic in θ.
        """
        npts_r, npts_theta = 20, 30
        transform = PolarTransform(
            r_bounds=(1.0, 2.0),
            theta_bounds=(0.0, 2 * math.pi),
            bkd=bkd,
        )
        mesh = TransformedMesh2D(npts_r, npts_theta, bkd, transform)
        basis = ChebyshevBasis2D(mesh, bkd)

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="x",
            nvars=2,
            diff_str="1.0",
            react_str="u",
            vel_strs=["0", "0"],
            bkd=bkd,
            oned=True,
        )

        pts = mesh.points()
        u_exact = man_sol.functions["solution"](pts)
        forcing = man_sol.functions["forcing"](pts)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, reaction=1.0, forcing=lambda t: forcing
        )

        # r boundaries: Dirichlet from manufactured solution
        r_min_idx = mesh.boundary_indices(0)
        r_max_idx = mesh.boundary_indices(1)
        bc_r_min = DirichletBC(bkd, r_min_idx, u_exact[r_min_idx])
        bc_r_max = DirichletBC(bkd, r_max_idx, u_exact[r_max_idx])

        # θ boundaries: periodic (θ is reference dim 1 = y direction)
        theta_min_idx = mesh.boundary_indices(2)
        theta_max_idx = mesh.boundary_indices(3)
        Dy = basis.derivative_matrix(1, 1)
        bc_theta = PeriodicBC(bkd, theta_min_idx, theta_max_idx, Dy)

        physics.set_boundary_conditions([bc_r_min, bc_r_max, bc_theta])

        model = CollocationModel(physics, bkd)
        initial_guess = bkd.zeros((npts_r * npts_theta,))
        u_numerical = model.solve_steady(initial_guess, tol=1e-8, maxiter=50)

        bkd.assert_allclose(u_numerical, u_exact, atol=1e-4)
