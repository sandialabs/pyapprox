"""Tests for the 1D linear elastic bar zoo factory."""


from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.interface.functions.protocols import (
    FunctionProtocol,
    FunctionWithJacobianProtocol,
)
from pyapprox.pde.field_maps.kle_factory import (
    create_lognormal_kle_field_map,
)
from pyapprox.pde.zoo.elastic_bar_1d import (
    create_linear_elastic_bar_1d,
)


def _make_kle_field_map(bkd, mesh, num_kle_terms=2):
    """Helper: create lognormal KLE field map on mesh nodes."""
    physical_pts = mesh.points()  # shape (1, npts)
    npts = physical_pts.shape[1]
    # Map [0, L] -> [0, 1] for correlation kernel
    x_min = physical_pts[0, 0]
    x_max = physical_pts[0, -1]
    length = x_max - x_min
    mesh_coords = (physical_pts - x_min) / length  # (1, npts)
    mean_log = bkd.zeros((npts,))
    return create_lognormal_kle_field_map(
        mesh_coords,
        mean_log,
        bkd,
        num_kle_terms=num_kle_terms,
        sigma=0.3,
    )


class TestElasticBar1D:
    def test_manufactured_solution(self, bkd):
        """Manufactured solution: solve and compare to exact."""
        npts = 20
        length = 2.0
        E_val = 3.0

        from pyapprox.pde.collocation.basis import ChebyshevBasis1D
        from pyapprox.pde.collocation.boundary import (
            flux_neumann_bc,
            zero_dirichlet_bc,
        )
        from pyapprox.pde.collocation.manufactured_solutions import (
            ManufacturedAdvectionDiffusionReaction,
        )
        from pyapprox.pde.collocation.mesh import (
            AffineTransform1D,
            TransformedMesh1D,
        )
        from pyapprox.pde.collocation.physics.advection_diffusion import (
            AdvectionDiffusionReaction,
        )
        from pyapprox.pde.collocation.time_integration import (
            CollocationModel,
        )

        # Manufactured solution: u(x) = x*(2-x), vanishes at x=0
        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="x*(2-x)",
            nvars=1,
            diff_str=str(E_val),
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
            oned=True,
        )

        transform = AffineTransform1D((0.0, length), bkd)
        mesh = TransformedMesh1D(npts, bkd, transform)
        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = mesh.points()  # (1, npts)

        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        physics = AdvectionDiffusionReaction(
            basis,
            bkd,
            diffusion=E_val,
            forcing=lambda t: forcing,
        )

        # Left: Dirichlet u(0) = 0
        left_idx = mesh.boundary_indices(0)
        bc_left = zero_dirichlet_bc(bkd, left_idx)

        # Right: flux Neumann from manufactured solution
        right_idx = mesh.boundary_indices(1)
        right_normals = mesh.boundary_normals(1)
        neumann_val = man_sol.neumann_values(
            nodes[:, right_idx],
            right_normals,
            convention="flux",
        )
        bc_right = flux_neumann_bc(
            bkd,
            right_idx,
            right_normals,
            physics,
            neumann_val,
        )

        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)
        u_num = model.solve_steady(bkd.zeros((npts,)), tol=1e-12, maxiter=50)
        bkd.assert_allclose(u_num, u_exact, atol=1e-10)

    def test_residual_at_exact(self, bkd):
        """Residual is near zero at the exact manufactured solution."""
        npts = 15
        length = 1.0
        E_val = 2.0

        from pyapprox.pde.collocation.basis import ChebyshevBasis1D
        from pyapprox.pde.collocation.boundary import (
            flux_neumann_bc,
            zero_dirichlet_bc,
        )
        from pyapprox.pde.collocation.manufactured_solutions import (
            ManufacturedAdvectionDiffusionReaction,
        )
        from pyapprox.pde.collocation.mesh import (
            AffineTransform1D,
            TransformedMesh1D,
        )
        from pyapprox.pde.collocation.physics.advection_diffusion import (
            AdvectionDiffusionReaction,
        )

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="x*(1-x)",
            nvars=1,
            diff_str=str(E_val),
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
            oned=True,
        )

        transform = AffineTransform1D((0.0, length), bkd)
        mesh = TransformedMesh1D(npts, bkd, transform)
        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = mesh.points()

        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        physics = AdvectionDiffusionReaction(
            basis,
            bkd,
            diffusion=E_val,
            forcing=lambda t: forcing,
        )

        left_idx = mesh.boundary_indices(0)
        bc_left = zero_dirichlet_bc(bkd, left_idx)

        right_idx = mesh.boundary_indices(1)
        right_normals = mesh.boundary_normals(1)
        neumann_val = man_sol.neumann_values(
            nodes[:, right_idx],
            right_normals,
            convention="flux",
        )
        bc_right = flux_neumann_bc(
            bkd,
            right_idx,
            right_normals,
            physics,
            neumann_val,
        )

        physics.set_boundary_conditions([bc_left, bc_right])

        residual = physics.residual(u_exact, 0.0)
        residual_bc, _ = physics.apply_boundary_conditions(
            residual,
            physics.jacobian(u_exact, 0.0),
            u_exact,
            0.0,
        )
        bkd.assert_allclose(
            residual_bc,
            bkd.zeros((npts,)),
            atol=1e-10,
        )

    def test_constant_E_analytical(self, bkd):
        """Constant E: u(x) = f*x*(2L-x)/(2E) + T*x/E (analytical)."""
        npts = 20
        length = 1.5
        E_val = 4.0
        f_val = 2.0
        T_val = 3.0

        from pyapprox.pde.collocation.basis import ChebyshevBasis1D
        from pyapprox.pde.collocation.boundary import (
            flux_neumann_bc,
            zero_dirichlet_bc,
        )
        from pyapprox.pde.collocation.mesh import (
            AffineTransform1D,
            TransformedMesh1D,
        )
        from pyapprox.pde.collocation.physics.advection_diffusion import (
            AdvectionDiffusionReaction,
        )
        from pyapprox.pde.collocation.time_integration import (
            CollocationModel,
        )

        transform = AffineTransform1D((0.0, length), bkd)
        mesh = TransformedMesh1D(npts, bkd, transform)
        basis = ChebyshevBasis1D(mesh, bkd)
        x = mesh.points()[0, :]  # physical coords, shape (npts,)

        # Analytical: u(x) = f*x*(2L-x)/(2E) + T*x/E
        u_exact = f_val * x * (2.0 * length - x) / (2.0 * E_val) + T_val * x / E_val

        physics = AdvectionDiffusionReaction(
            basis,
            bkd,
            diffusion=E_val,
            forcing=lambda t: f_val * bkd.ones((npts,)),
        )
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        right_normals = mesh.boundary_normals(1)
        physics.set_boundary_conditions(
            [
                zero_dirichlet_bc(bkd, left_idx),
                flux_neumann_bc(bkd, right_idx, right_normals, physics, -T_val),
            ]
        )

        model = CollocationModel(physics, bkd)
        u_num = model.solve_steady(bkd.zeros((npts,)), tol=2e-12, maxiter=50)
        bkd.assert_allclose(u_num, u_exact, atol=1e-10)

    def test_param_jacobian(self, bkd):
        """DerivativeChecker on KLE-parameterized elastic bar."""
        npts = 20
        length = 1.0
        num_kle_terms = 2

        from pyapprox.pde.collocation.mesh import (
            AffineTransform1D,
            TransformedMesh1D,
        )

        transform = AffineTransform1D((0.0, length), bkd)
        mesh = TransformedMesh1D(npts, bkd, transform)
        field_map = _make_kle_field_map(bkd, mesh, num_kle_terms)

        fwd = create_linear_elastic_bar_1d(
            bkd=bkd,
            npts=npts,
            length=length,
            E_mean_field=1.0,
            forcing=lambda t: bkd.ones((npts,)),
            traction=1.0,
            field_map=field_map,
        )

        assert hasattr(fwd, "jacobian")
        assert isinstance(fwd, FunctionWithJacobianProtocol)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=fwd.nqoi(),
            nvars=fwd.nvars(),
            fun=fwd,
            jacobian=fwd.jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = bkd.array([0.1, -0.1])[:, None]
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        assert ratio <= 1e-5

    def test_factory_produces_valid_model(self, bkd):
        """Zoo factory produces a working forward model with KLE."""
        npts = 20
        length = 1.0
        num_kle_terms = 2

        from pyapprox.pde.collocation.mesh import (
            AffineTransform1D,
            TransformedMesh1D,
        )

        transform = AffineTransform1D((0.0, length), bkd)
        mesh = TransformedMesh1D(npts, bkd, transform)
        field_map = _make_kle_field_map(bkd, mesh, num_kle_terms)

        fwd = create_linear_elastic_bar_1d(
            bkd=bkd,
            npts=npts,
            length=length,
            E_mean_field=1.0,
            forcing=lambda t: bkd.ones((npts,)),
            traction=1.0,
            field_map=field_map,
        )

        assert isinstance(fwd, FunctionProtocol)
        assert isinstance(fwd, FunctionWithJacobianProtocol)

        samples = bkd.zeros((num_kle_terms, 1))
        result = fwd(samples)
        assert result.shape[0] == fwd.nqoi()
        assert result.shape[1] == 1

    def test_isinstance_protocols(self, bkd):
        """Zoo model satisfies expected protocols."""
        npts = 15
        length = 1.0

        from pyapprox.pde.collocation.mesh import (
            AffineTransform1D,
            TransformedMesh1D,
        )

        transform = AffineTransform1D((0.0, length), bkd)
        mesh = TransformedMesh1D(npts, bkd, transform)
        field_map = _make_kle_field_map(bkd, mesh)

        fwd = create_linear_elastic_bar_1d(
            bkd=bkd,
            npts=npts,
            length=length,
            E_mean_field=1.0,
            forcing=lambda t: bkd.ones((npts,)),
            traction=0.5,
            field_map=field_map,
        )

        assert isinstance(fwd, FunctionProtocol)
        assert hasattr(fwd, "jacobian")
