"""Transient ADR manufactured solutions on the collocation transient path.

The steady cases are verified by the collocation ADR catalog
(``tests/pde/models/collocation/test_verification_adr.py``).
"""

from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.pde.collocation.mesh import TransformedMesh1D, create_uniform_mesh_1d
from pyapprox.pde.collocation.physics import AdvectionDiffusionReaction
from pyapprox.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)
from pyapprox.pde.manufactured import (
    ManufacturedAdvectionDiffusionReaction,
)
from pyapprox.util.backends.protocols import Array, Backend


class TestManufacturedADRTransient:
    """Test transient ADR manufactured solutions.

    For transient tests, we use polynomial solutions in BOTH space and time:
    - u = (1 - 0.5*T)*(1-x**2)*x  (polynomial in space and time)

    This gives machine precision residuals.
    """

    def test_transient_diffusion_residual(self, bkd: Backend[Array]) -> None:
        """Test transient manufactured solution residual equals du/dt.

        For transient problems, residual = L(u) + f = du/dt at exact solution.
        Uses polynomial solution in both space and time.
        """
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        # Transient polynomial solution: u = (1 - 0.5*T)*(1-x**2)*x
        # du/dt = -0.5*(1-x**2)*x
        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="(1 - 0.5*T)*(1 - x**2)*x",
            nvars=1,
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
            oned=True,
        )

        nodes = bkd.reshape(basis.nodes(), (1, -1))
        time = 0.5

        u_exact = man_sol.functions["solution"](nodes, time)

        # The forcing includes both spatial and temporal terms
        physics = AdvectionDiffusionReaction(
            basis,
            bkd,
            diffusion=1.0,
            forcing=lambda t: man_sol.functions["forcing"](nodes, t),
        )

        left_idx = umesh.boundary_indices(0)
        right_idx = umesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        # The residual should equal du/dt
        residual = physics.residual(u_exact, time)

        # Expected: du/dt = -0.5*(1-x**2)*x
        # Compute expected at nodes
        x = basis.nodes()
        expected_dudt = -0.5 * (1 - x**2) * x

        # Interior points only (BCs modify boundary residual)
        bkd.assert_allclose(residual[1:-1], expected_dudt[1:-1], atol=1e-12)

    def test_transient_diffusion_solve(self, bkd: Backend[Array]) -> None:
        """Test transient solve with manufactured solution.

        Uses polynomial solution for machine precision verification.
        """
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        # Polynomial transient solution
        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="(1 - 0.5*T)*(1 - x**2)*x",
            nvars=1,
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
            oned=True,
        )

        nodes = bkd.reshape(basis.nodes(), (1, -1))
        time_start = 0.0
        time_end = 0.1  # Short time to stay in valid range

        u0 = man_sol.functions["solution"](nodes, time_start)
        u_exact_end = man_sol.functions["solution"](nodes, time_end)

        physics = AdvectionDiffusionReaction(
            basis,
            bkd,
            diffusion=1.0,
            forcing=lambda t: man_sol.functions["forcing"](nodes, t),
        )

        left_idx = umesh.boundary_indices(0)
        right_idx = umesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)

        # Solve with backward Euler (implicit, stable)
        config: TimeIntegrationConfig[Array] = TimeIntegrationConfig(
            method="backward_euler",
            deltat=0.01,
            init_time=time_start,
            final_time=time_end,
            newton_tol=1e-10,
            newton_maxiter=20,
            verbosity=0,
        )
        solutions, times = model.solve_transient(u0, config)

        # Final solution should match exact
        u_final = solutions[:, -1]
        bkd.assert_allclose(
            u_final, u_exact_end, atol=1e-3
        )  # Time discretization error
