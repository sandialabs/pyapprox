"""Linear elasticity on a quarter-annulus (polar transform).

Verifies:
1. Manufactured solution with mixed (traction + Dirichlet) BCs on polar domain
2. Manufactured solution with all-Dirichlet BCs (baseline)
3. Variable Lame Jacobian on polar domain via DerivativeChecker
"""

import math
from typing import Generic, List, Tuple

import numpy as np

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary import (
    traction_neumann_bc,
    zero_dirichlet_bc,
)
from pyapprox.pde.collocation.boundary.dirichlet import DirichletBC
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms import PolarTransform
from pyapprox.pde.collocation.physics import LinearElasticityPhysics
from pyapprox.pde.collocation.physics.coefficients import NodalField
from pyapprox.pde.collocation.protocols.boundary import (
    BoundaryConditionProtocol,
)
from pyapprox.pde.collocation.time_integration import CollocationModel
from pyapprox.pde.manufactured import (
    ManufacturedLinearElasticityEquations,
)
from pyapprox.util.backends.protocols import Array, Backend


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


class _PhysWrapper(Generic[Array]):
    """The residual of a physics with its conditions, as a function of
    the state, for DerivativeChecker."""

    def __init__(
        self, phys: LinearElasticityPhysics[Array], bk: Backend[Array]
    ) -> None:
        self._p = phys
        self._b = bk

    def bkd(self) -> Backend[Array]:
        return self._b

    def nvars(self) -> int:
        return self._p.nstates()

    def nqoi(self) -> int:
        return self._p.nstates()

    def __call__(self, samples: Array) -> Array:
        return self._b.stack(
            [self._p.residual(samples[:, i], 0.0) for i in range(samples.shape[1])],
            axis=1,
        )

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.first_order(jacobian=self.jacobian)

    def jacobian(self, sample: Array) -> Array:
        return self._p.jacobian(sample[:, 0], 0.0)


class TestLinearElasticityPolar:
    """Linear elasticity on the quarter-annulus."""

    # ------------------------------------------------------------------
    # Manufactured solution with mixed BCs (same structure as the
    # pressurized cylinder)
    # ------------------------------------------------------------------

    def test_manufactured_mixed_bcs(self, bkd: Backend[Array]) -> None:
        """Manufactured solution with cylinder-style BCs on polar domain."""
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

        # Manufactured solution: nonzero everywhere on quarter-annulus
        man_sol = ManufacturedLinearElasticityEquations(
            sol_strs=[
                "x*(x**2 + y**2)",
                "y*(x**2 + y**2)",
            ],
            nvars=2,
            lambda_str=str(lamda_val),
            mu_str=str(mu_val),
            bkd=bkd,
            oned=True,
        )

        # Exact solution and forcing
        u_exact = man_sol.functions["solution"](pts)  # (npts, 2)
        forcing = man_sol.functions["forcing"](pts)  # (npts, 2)
        forcing_flat = bkd.concatenate([forcing[:, 0], forcing[:, 1]])

        physics = LinearElasticityPhysics(
            basis,
            bkd,
            lamda=lamda_val,
            mu=mu_val,
            forcing=NodalField(forcing_flat),
        )

        bcs: List[BoundaryConditionProtocol[Array]] = []

        # Inner boundary (bnd 0): traction with exact manufactured values
        inner_idx = mesh.boundary_indices(0)
        inner_normals = mesh.boundary_normals(0)
        inner_pts = pts[:, bkd.to_numpy(inner_idx).astype(int)]
        inner_traction = man_sol.traction_values(
            inner_pts,
            inner_normals,
        )  # (nbnd, 2)
        for comp in (0, 1):
            bcs.append(
                traction_neumann_bc(
                    bkd,
                    inner_idx,
                    inner_normals,
                    physics,
                    npts,
                    comp,
                    values=inner_traction[:, comp],
                )
            )

        # Outer boundary (bnd 1): traction with exact manufactured values
        outer_idx = mesh.boundary_indices(1)
        outer_normals = mesh.boundary_normals(1)
        outer_pts = pts[:, bkd.to_numpy(outer_idx).astype(int)]
        outer_traction = man_sol.traction_values(
            outer_pts,
            outer_normals,
        )
        for comp in (0, 1):
            bcs.append(
                traction_neumann_bc(
                    bkd,
                    outer_idx,
                    outer_normals,
                    physics,
                    npts,
                    comp,
                    values=outer_traction[:, comp],
                )
            )

        # Bottom (bnd 2, theta=0): traction t_x (comp 0) + Dirichlet v (comp 1)
        bottom_idx = mesh.boundary_indices(2)
        bottom_normals = mesh.boundary_normals(2)
        bottom_pts = pts[:, bkd.to_numpy(bottom_idx).astype(int)]
        bottom_traction = man_sol.traction_values(
            bottom_pts,
            bottom_normals,
        )
        bcs.append(
            traction_neumann_bc(
                bkd,
                bottom_idx,
                bottom_normals,
                physics,
                npts,
                component=0,
                values=bottom_traction[:, 0],
            )
        )
        # Dirichlet v at bottom
        bottom_v_idx = bottom_idx + npts
        bottom_v_exact = u_exact[bkd.to_numpy(bottom_idx).astype(int), 1]
        bcs.append(DirichletBC(bkd, bottom_v_idx, bottom_v_exact))

        # Top (bnd 3, theta=pi/2): Dirichlet u (comp 0) + traction t_y (comp 1)
        top_idx = mesh.boundary_indices(3)
        top_normals = mesh.boundary_normals(3)
        top_pts = pts[:, bkd.to_numpy(top_idx).astype(int)]
        top_traction = man_sol.traction_values(
            top_pts,
            top_normals,
        )
        # Dirichlet u at top
        top_u_exact = u_exact[bkd.to_numpy(top_idx).astype(int), 0]
        bcs.append(DirichletBC(bkd, top_idx, top_u_exact))
        bcs.append(
            traction_neumann_bc(
                bkd,
                top_idx,
                top_normals,
                physics,
                npts,
                component=1,
                values=top_traction[:, 1],
            )
        )

        physics.set_boundary_conditions(bcs)

        # Solve
        model = CollocationModel(physics, bkd)
        exact_flat = bkd.concatenate([u_exact[:, 0], u_exact[:, 1]])
        sol = model.solve_steady(bkd.zeros((2 * npts,)), tol=1e-10, maxiter=50)
        bkd.assert_allclose(sol, exact_flat, atol=1e-8)

    # ------------------------------------------------------------------
    # Manufactured solution with all-Dirichlet BCs (baseline)
    # ------------------------------------------------------------------

    def test_manufactured_all_dirichlet(self, bkd: Backend[Array]) -> None:
        """Manufactured solution on polar domain with all-Dirichlet BCs."""
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

        # Low-order polynomial smooth on the quarter-annulus
        man_sol = ManufacturedLinearElasticityEquations(
            sol_strs=[
                "x**2 + x*y",
                "y**2 - x*y",
            ],
            nvars=2,
            lambda_str=str(lamda_val),
            mu_str=str(mu_val),
            bkd=bkd,
            oned=True,
        )

        u_exact = man_sol.functions["solution"](pts)
        forcing = man_sol.functions["forcing"](pts)
        forcing_flat = bkd.concatenate([forcing[:, 0], forcing[:, 1]])

        physics = LinearElasticityPhysics(
            basis,
            bkd,
            lamda=lamda_val,
            mu=mu_val,
            forcing=NodalField(forcing_flat),
        )

        # Dirichlet BCs on all 4 sides, both components
        bcs: List[BoundaryConditionProtocol[Array]] = []
        for side in range(4):
            bnd_idx = mesh.boundary_indices(side)
            bnd_idx_np = bkd.to_numpy(bnd_idx).astype(int)
            # u-component
            u_vals = u_exact[bnd_idx_np, 0]
            bcs.append(DirichletBC(bkd, bnd_idx, u_vals))
            # v-component
            v_vals = u_exact[bnd_idx_np, 1]
            v_idx = bnd_idx + npts
            bcs.append(DirichletBC(bkd, v_idx, v_vals))
        physics.set_boundary_conditions(bcs)

        model = CollocationModel(physics, bkd)
        exact_flat = bkd.concatenate([u_exact[:, 0], u_exact[:, 1]])
        sol = model.solve_steady(bkd.zeros((2 * npts,)), tol=1e-10, maxiter=50)
        bkd.assert_allclose(sol, exact_flat, atol=1e-8)

    # ------------------------------------------------------------------
    # Variable Lame Jacobian on polar domain
    # ------------------------------------------------------------------

    def test_variable_lame_jacobian_on_polar(self, bkd: Backend[Array]) -> None:
        """DerivativeChecker validates variable-Lame Jacobian on polar."""
        npts_r, npts_theta = 6, 6
        r_inner, r_outer = 1.0, 2.0

        mesh, basis = _setup_polar_mesh_and_basis(
            bkd,
            npts_r,
            npts_theta,
            r_inner,
            r_outer,
        )
        npts = basis.npts()
        pts = mesh.points()
        x = pts[0, :]
        y = pts[1, :]

        # Spatially varying Lame params
        mu_field = 1.0 + 0.3 * bkd.sin(math.pi * x) * bkd.cos(math.pi * y)
        lam_field = 1.5 + 0.2 * bkd.cos(math.pi * x) * bkd.sin(math.pi * y)

        physics = LinearElasticityPhysics(
            basis,
            bkd,
            lamda=NodalField(lam_field),
            mu=NodalField(mu_field),
        )

        # All-Dirichlet for clean Jacobian test
        bcs: List[BoundaryConditionProtocol[Array]] = []
        for side in range(4):
            bnd_idx = mesh.boundary_indices(side)
            bcs.append(zero_dirichlet_bc(bkd, bnd_idx))
            bcs.append(zero_dirichlet_bc(bkd, bnd_idx + npts))
        physics.set_boundary_conditions(bcs)

        # Wrap physics for DerivativeChecker
        wrapper = _PhysWrapper(physics, bkd)
        np.random.seed(42)
        state_np = np.random.randn(2 * npts) * 0.1
        sample = bkd.asarray(state_np)[:, None]

        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, verbosity=0)
        assert checker.error_ratio(errors[0]) <= 1e-5
