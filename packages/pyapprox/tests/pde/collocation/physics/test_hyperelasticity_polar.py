"""Neo-Hookean hyperelasticity on a quarter-annulus (polar transform).

Verifies:
1. Manufactured solution with all-Dirichlet BCs (baseline)
2. Manufactured solution with mixed (traction + Dirichlet) BCs
"""

import math
from typing import List, Tuple

from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary.dirichlet import DirichletBC
from pyapprox.pde.collocation.boundary.hyperelastic_traction import (
    hyperelastic_traction_neumann_bc,
)
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms import PolarTransform
from pyapprox.pde.collocation.physics.coefficients import NodalField
from pyapprox.pde.collocation.physics.hyperelasticity import (
    HyperelasticityPhysics,
)
from pyapprox.pde.collocation.physics.neo_hookean_law import NeoHookeanLaw
from pyapprox.pde.collocation.protocols.boundary import (
    BoundaryConditionProtocol,
)
from pyapprox.pde.collocation.time_integration import CollocationModel
from pyapprox.pde.constitutive import (
    NeoHookeanStress,
)
from pyapprox.pde.manufactured.hyperelasticity import (
    ManufacturedHyperelasticityEquations,
)
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.markers import slow_test


def _setup_polar_mesh_and_basis(
    bkd: Backend[Array],
    npts_r: int,
    npts_theta: int,
    r_inner: float,
    r_outer: float,
) -> Tuple[TransformedMesh2D[Array], ChebyshevBasis2D[Array]]:
    """Create polar mesh and 2D Chebyshev basis on quarter-annulus."""
    transform = PolarTransform(
        (r_inner, r_outer),
        (0.0, math.pi / 2.0),
        bkd,
    )
    mesh = TransformedMesh2D(npts_r, npts_theta, bkd, transform)
    basis = ChebyshevBasis2D(mesh, bkd)
    return mesh, basis


@slow_test
class TestHyperelasticityPolar:
    """Neo-Hookean hyperelasticity on the quarter-annulus."""

    # ------------------------------------------------------------------
    # Manufactured solution with all-Dirichlet BCs (baseline)
    # ------------------------------------------------------------------

    def test_manufactured_all_dirichlet(self, bkd: Backend[Array]) -> None:
        """Manufactured hyperelastic solution with all-Dirichlet BCs."""
        npts_r, npts_theta = 12, 12
        r_inner, r_outer = 1.0, 2.0
        lamda_val, mu_val = 1.0, 1.0

        mesh, basis = _setup_polar_mesh_and_basis(
            bkd,
            npts_r,
            npts_theta,
            r_inner,
            r_outer,
        )
        npts = basis.npts()
        pts = mesh.points()

        stress_model: NeoHookeanStress[Array] = NeoHookeanStress(
            lamda=lamda_val, mu=mu_val
        )

        # Small polynomial displacement
        man_sol = ManufacturedHyperelasticityEquations(
            sol_strs=[
                "0.01*x**2*y",
                "0.01*x*y**2",
            ],
            nvars=2,
            stress_model=stress_model,
            bkd=bkd,
            oned=True,
        )

        u_exact = man_sol.functions["solution"](pts)  # (npts, 2)
        forcing = man_sol.functions["forcing"](pts)  # (npts, 2)
        forcing_flat = bkd.concatenate([forcing[:, 0], forcing[:, 1]])

        physics = HyperelasticityPhysics(
            basis,
            bkd,
            NeoHookeanLaw(bkd),
            mu=mu_val,
            lamda=lamda_val,
            forcing=NodalField(forcing_flat),
        )

        # All-Dirichlet BCs on all 4 sides, both components
        bcs: List[BoundaryConditionProtocol[Array]] = []
        for side in range(4):
            bnd_idx = mesh.boundary_indices(side)
            bnd_idx_np = bkd.to_numpy(bnd_idx).astype(int)
            u_vals = u_exact[bnd_idx_np, 0]
            bcs.append(DirichletBC(bkd, bnd_idx, u_vals))
            v_vals = u_exact[bnd_idx_np, 1]
            v_idx = bnd_idx + npts
            bcs.append(DirichletBC(bkd, v_idx, v_vals))
        physics.set_boundary_conditions(bcs)

        model = CollocationModel(physics, bkd)
        exact_flat = bkd.concatenate([u_exact[:, 0], u_exact[:, 1]])
        sol = model.solve_steady(
            bkd.zeros((2 * npts,)),
            tol=1e-10,
            maxiter=50,
        )
        bkd.assert_allclose(sol, exact_flat, atol=1e-7)

    # ------------------------------------------------------------------
    # Manufactured solution with mixed BCs
    # ------------------------------------------------------------------

    def test_manufactured_mixed_bcs(self, bkd: Backend[Array]) -> None:
        """Manufactured hyperelastic solution with cylinder-style BCs."""
        npts_r, npts_theta = 12, 12
        r_inner, r_outer = 1.0, 2.5
        lamda_val, mu_val = 1.5, 0.8

        mesh, basis = _setup_polar_mesh_and_basis(
            bkd,
            npts_r,
            npts_theta,
            r_inner,
            r_outer,
        )
        npts = basis.npts()
        pts = mesh.points()

        stress_model: NeoHookeanStress[Array] = NeoHookeanStress(
            lamda=lamda_val, mu=mu_val
        )

        man_sol = ManufacturedHyperelasticityEquations(
            sol_strs=[
                "0.01*x*(x**2 + y**2)",
                "0.01*y*(x**2 + y**2)",
            ],
            nvars=2,
            stress_model=stress_model,
            bkd=bkd,
            oned=True,
        )

        u_exact = man_sol.functions["solution"](pts)  # (npts, 2)
        forcing = man_sol.functions["forcing"](pts)  # (npts, 2)
        forcing_flat = bkd.concatenate([forcing[:, 0], forcing[:, 1]])
        flux = man_sol.functions["flux"](pts)  # (2, npts, 2)

        physics = HyperelasticityPhysics(
            basis,
            bkd,
            NeoHookeanLaw(bkd),
            mu=mu_val,
            lamda=lamda_val,
            forcing=NodalField(forcing_flat),
        )

        bcs: List[BoundaryConditionProtocol[Array]] = []

        # Inner (bnd 0): traction with exact manufactured PK1 traction
        inner_idx = mesh.boundary_indices(0)
        inner_normals = mesh.boundary_normals(0)
        inner_idx_np = bkd.to_numpy(inner_idx).astype(int)
        for comp in (0, 1):
            # t_comp = P_{comp+1, 1}*nx + P_{comp+1, 2}*ny
            traction_vals = (
                flux[comp, inner_idx_np, 0] * inner_normals[:, 0]
                + flux[comp, inner_idx_np, 1] * inner_normals[:, 1]
            )
            bcs.append(
                hyperelastic_traction_neumann_bc(
                    bkd,
                    inner_idx,
                    inner_normals,
                    physics,
                    npts,
                    comp,
                    values=traction_vals,
                )
            )

        # Outer (bnd 1): traction with exact manufactured PK1 traction
        outer_idx = mesh.boundary_indices(1)
        outer_normals = mesh.boundary_normals(1)
        outer_idx_np = bkd.to_numpy(outer_idx).astype(int)
        for comp in (0, 1):
            traction_vals = (
                flux[comp, outer_idx_np, 0] * outer_normals[:, 0]
                + flux[comp, outer_idx_np, 1] * outer_normals[:, 1]
            )
            bcs.append(
                hyperelastic_traction_neumann_bc(
                    bkd,
                    outer_idx,
                    outer_normals,
                    physics,
                    npts,
                    comp,
                    values=traction_vals,
                )
            )

        # Bottom (bnd 2): traction t_x (comp 0), Dirichlet v (comp 1)
        bottom_idx = mesh.boundary_indices(2)
        bottom_normals = mesh.boundary_normals(2)
        bottom_idx_np = bkd.to_numpy(bottom_idx).astype(int)
        traction_bottom_x = (
            flux[0, bottom_idx_np, 0] * bottom_normals[:, 0]
            + flux[0, bottom_idx_np, 1] * bottom_normals[:, 1]
        )
        bcs.append(
            hyperelastic_traction_neumann_bc(
                bkd,
                bottom_idx,
                bottom_normals,
                physics,
                npts,
                component=0,
                values=traction_bottom_x,
            )
        )
        bottom_v_idx = bottom_idx + npts
        bottom_v_exact = u_exact[bottom_idx_np, 1]
        bcs.append(DirichletBC(bkd, bottom_v_idx, bottom_v_exact))

        # Top (bnd 3): Dirichlet u (comp 0), traction t_y (comp 1)
        top_idx = mesh.boundary_indices(3)
        top_normals = mesh.boundary_normals(3)
        top_idx_np = bkd.to_numpy(top_idx).astype(int)
        top_u_exact = u_exact[top_idx_np, 0]
        bcs.append(DirichletBC(bkd, top_idx, top_u_exact))
        traction_top_y = (
            flux[1, top_idx_np, 0] * top_normals[:, 0]
            + flux[1, top_idx_np, 1] * top_normals[:, 1]
        )
        bcs.append(
            hyperelastic_traction_neumann_bc(
                bkd,
                top_idx,
                top_normals,
                physics,
                npts,
                component=1,
                values=traction_top_y,
            )
        )

        physics.set_boundary_conditions(bcs)

        model = CollocationModel(physics, bkd)
        exact_flat = bkd.concatenate([u_exact[:, 0], u_exact[:, 1]])
        sol = model.solve_steady(
            bkd.zeros((2 * npts,)),
            tol=1e-10,
            maxiter=50,
        )
        bkd.assert_allclose(sol, exact_flat, atol=1e-7)
