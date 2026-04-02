"""Tests for the 1D hyperelastic bar zoo factory."""


import numpy as np
import pytest

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
from pyapprox.pde.zoo.hyperelastic_bar_1d import (
    create_hyperelastic_bar_1d,
)


def _make_kle_field_map(bkd, mesh, num_kle_terms=2):
    """Helper: create lognormal KLE field map on mesh nodes."""
    physical_pts = mesh.points()  # shape (1, npts)
    npts = physical_pts.shape[1]
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


class TestHyperelasticBar1D:
    def test_manufactured_solution(self, bkd):
        """Manufactured solution: solve and compare to exact."""
        npts = 25
        length = 1.0
        lamda_val = 2.0
        mu_val = 1.0

        from pyapprox.pde.collocation.basis import ChebyshevBasis1D
        from pyapprox.pde.collocation.boundary import (
            flux_neumann_bc,
            zero_dirichlet_bc,
        )
        from pyapprox.pde.collocation.manufactured_solutions.hyperelasticity import (
            ManufacturedHyperelasticityEquations,
        )
        from pyapprox.pde.collocation.mesh import (
            AffineTransform1D,
            TransformedMesh1D,
        )
        from pyapprox.pde.collocation.physics.hyperelasticity import (
            HyperelasticityPhysics,
        )
        from pyapprox.pde.collocation.physics.stress_models import (
            NeoHookeanStress,
        )
        from pyapprox.pde.collocation.time_integration import (
            CollocationModel,
        )

        # u(x) = 0.1*x*(1-x), vanishes at x=0
        stress_model = NeoHookeanStress(lamda=lamda_val, mu=mu_val)
        man_sol = ManufacturedHyperelasticityEquations(
            sol_strs=["0.1*x*(1-x)"],
            nvars=1,
            stress_model=stress_model,
            bkd=bkd,
            oned=True,
        )

        transform = AffineTransform1D((0.0, length), bkd)
        mesh = TransformedMesh1D(npts, bkd, transform)
        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = mesh.points()  # (1, npts)

        # oned=True + VectorSolutionMixin => (npts, ncomponents)
        u_exact = man_sol.functions["solution"](nodes)[:, 0]
        forcing = man_sol.functions["forcing"](nodes)[:, 0]

        physics = HyperelasticityPhysics(
            basis,
            bkd,
            stress_model,
            forcing=lambda t: forcing,
        )

        # Left: Dirichlet u(0) = 0
        left_idx = mesh.boundary_indices(0)
        bc_left = zero_dirichlet_bc(bkd, left_idx)

        # Right: traction BC from manufactured solution
        right_idx = mesh.boundary_indices(1)
        right_normals = mesh.boundary_normals(1)
        # Evaluate flux (PK1 stress) at right boundary
        flux = man_sol.functions["flux"](nodes)
        # flux shape: (1, npts, 1) for 1D with oned=True
        traction_val = flux[0, right_idx, 0] * right_normals[:, 0]
        bc_right = flux_neumann_bc(
            bkd,
            right_idx,
            right_normals,
            physics,
            traction_val,
        )

        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)
        u_num = model.solve_steady(
            bkd.zeros((npts,)),
            tol=1e-12,
            maxiter=100,
        )
        bkd.assert_allclose(u_num, u_exact, atol=1e-9)

    def test_residual_at_exact(self, bkd):
        """Residual is near zero at the exact manufactured solution."""
        npts = 20
        length = 1.0
        lamda_val = 3.0
        mu_val = 2.0

        from pyapprox.pde.collocation.basis import ChebyshevBasis1D
        from pyapprox.pde.collocation.boundary import (
            flux_neumann_bc,
            zero_dirichlet_bc,
        )
        from pyapprox.pde.collocation.manufactured_solutions.hyperelasticity import (
            ManufacturedHyperelasticityEquations,
        )
        from pyapprox.pde.collocation.mesh import (
            AffineTransform1D,
            TransformedMesh1D,
        )
        from pyapprox.pde.collocation.physics.hyperelasticity import (
            HyperelasticityPhysics,
        )
        from pyapprox.pde.collocation.physics.stress_models import (
            NeoHookeanStress,
        )

        stress_model = NeoHookeanStress(lamda=lamda_val, mu=mu_val)
        man_sol = ManufacturedHyperelasticityEquations(
            sol_strs=["0.05*x*(1-x)"],
            nvars=1,
            stress_model=stress_model,
            bkd=bkd,
            oned=True,
        )

        transform = AffineTransform1D((0.0, length), bkd)
        mesh = TransformedMesh1D(npts, bkd, transform)
        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = mesh.points()

        u_exact = man_sol.functions["solution"](nodes)[:, 0]
        forcing = man_sol.functions["forcing"](nodes)[:, 0]

        physics = HyperelasticityPhysics(
            basis,
            bkd,
            stress_model,
            forcing=lambda t: forcing,
        )

        left_idx = mesh.boundary_indices(0)
        bc_left = zero_dirichlet_bc(bkd, left_idx)

        right_idx = mesh.boundary_indices(1)
        right_normals = mesh.boundary_normals(1)
        flux = man_sol.functions["flux"](nodes)
        traction_val = flux[0, right_idx, 0] * right_normals[:, 0]
        bc_right = flux_neumann_bc(
            bkd,
            right_idx,
            right_normals,
            physics,
            traction_val,
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

    def test_param_jacobian(self, bkd):
        """DerivativeChecker on KLE-parameterized hyperelastic bar."""
        npts = 20
        length = 1.0
        num_kle_terms = 2
        E_mean = 3.0
        nu = 0.3

        from pyapprox.pde.collocation.mesh import (
            AffineTransform1D,
            TransformedMesh1D,
        )

        transform = AffineTransform1D((0.0, length), bkd)
        mesh = TransformedMesh1D(npts, bkd, transform)
        field_map = _make_kle_field_map(bkd, mesh, num_kle_terms)

        fwd = create_hyperelastic_bar_1d(
            bkd=bkd,
            npts=npts,
            length=length,
            E_mean=E_mean,
            poisson_ratio=nu,
            forcing=lambda t: bkd.ones((npts,)),
            traction=0.5,
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
        np.random.seed(42)
        sample = bkd.array([0.1, -0.1])[:, None]
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        assert ratio <= 1e-5

    def test_small_strain_matches_linear(self, bkd):
        """Small traction: hyperelastic solution ~ linear elastic solution.

        Uses nu=0 so the 1D Neo-Hookean linearized modulus (2*mu + lambda)
        equals E. With nu>0, the 3D Lame-based effective 1D modulus differs
        from the plane-stress E used by the linear bar.
        """
        npts = 25
        length = 1.0
        E_val = 3.0
        nu = 0.0  # ensures linearized Neo-Hookean modulus = E
        small_traction = 0.01
        num_kle_terms = 2

        from pyapprox.pde.collocation.mesh import (
            AffineTransform1D,
            TransformedMesh1D,
        )
        from pyapprox.pde.zoo.elastic_bar_1d import (
            create_linear_elastic_bar_1d,
        )

        transform = AffineTransform1D((0.0, length), bkd)
        mesh = TransformedMesh1D(npts, bkd, transform)
        field_map = _make_kle_field_map(bkd, mesh, num_kle_terms)

        # Hyperelastic bar
        fwd_hyper = create_hyperelastic_bar_1d(
            bkd=bkd,
            npts=npts,
            length=length,
            E_mean=E_val,
            poisson_ratio=nu,
            forcing=lambda t: bkd.zeros((npts,)),
            traction=small_traction,
            field_map=field_map,
        )

        # Linear elastic bar
        fwd_linear = create_linear_elastic_bar_1d(
            bkd=bkd,
            npts=npts,
            length=length,
            E_mean_field=E_val,
            forcing=lambda t: bkd.zeros((npts,)),
            traction=small_traction,
            field_map=field_map,
        )

        # Evaluate at KLE coefficients = 0 (mean field)
        sample = bkd.zeros((num_kle_terms, 1))
        sol_hyper = fwd_hyper(sample)
        sol_linear = fwd_linear(sample)

        # Should match to ~O(traction^2) relative error
        # atol handles the Dirichlet node where both solutions are ~0
        bkd.assert_allclose(sol_hyper, sol_linear, rtol=1e-2, atol=1e-12)

    def test_flux_neumann_bc(self, bkd):
        """After solve, PK1 stress at right boundary matches traction."""
        npts = 25
        length = 1.0
        E_mean = 3.0
        nu = 0.3
        T_val = 0.5
        num_kle_terms = 2

        from pyapprox.pde.collocation.basis import ChebyshevBasis1D
        from pyapprox.pde.collocation.mesh import (
            AffineTransform1D,
            TransformedMesh1D,
        )
        from pyapprox.pde.collocation.physics.stress_models import (
            NeoHookeanStress,
        )

        transform = AffineTransform1D((0.0, length), bkd)
        mesh = TransformedMesh1D(npts, bkd, transform)
        basis = ChebyshevBasis1D(mesh, bkd)
        field_map = _make_kle_field_map(bkd, mesh, num_kle_terms)

        fwd = create_hyperelastic_bar_1d(
            bkd=bkd,
            npts=npts,
            length=length,
            E_mean=E_mean,
            poisson_ratio=nu,
            forcing=lambda t: bkd.zeros((npts,)),
            traction=T_val,
            field_map=field_map,
        )

        # Solve at mean (KLE coeffs = 0): KLE gives E=exp(0)=1.0
        sample = bkd.zeros((num_kle_terms, 1))
        sol = fwd(sample)  # shape (nqoi, 1)
        u = sol[:, 0]

        # Compute PK1 stress at right boundary using E from KLE
        Dx = basis.derivative_matrix(1, 0)
        F = 1.0 + Dx @ u

        # KLE at zero coeffs: E = exp(mean_log=0) = 1.0
        E_actual = 1.0
        dmu_dE = 1.0 / (2.0 * (1.0 + nu))
        dlam_dE = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu_val = E_actual * dmu_dE
        lam_val = E_actual * dlam_dE
        stress_model = NeoHookeanStress(lamda=lam_val, mu=mu_val)
        P = stress_model.compute_stress_1d(F, bkd)

        # At x=L with n=+1: flux.n = P, should equal prescribed traction
        right_idx = mesh.boundary_indices(1)
        bkd.assert_allclose(
            P[right_idx],
            T_val * bkd.ones((1,)),
            rtol=1e-8,
        )

    def test_set_mu_nonpositive_raises(self, bkd):
        """Setting non-positive mu raises ValueError."""
        from pyapprox.pde.collocation.basis import ChebyshevBasis1D
        from pyapprox.pde.collocation.mesh import (
            AffineTransform1D,
            TransformedMesh1D,
        )
        from pyapprox.pde.collocation.physics.hyperelasticity import (
            HyperelasticityPhysics,
        )
        from pyapprox.pde.collocation.physics.stress_models import (
            NeoHookeanStress,
        )

        transform = AffineTransform1D((0.0, 1.0), bkd)
        mesh = TransformedMesh1D(10, bkd, transform)
        basis = ChebyshevBasis1D(mesh, bkd)
        stress_model = NeoHookeanStress(lamda=1.0, mu=1.0)
        physics = HyperelasticityPhysics(basis, bkd, stress_model)

        with pytest.raises(ValueError):
            physics.set_mu(bkd.array([-0.1]))

        with pytest.raises(ValueError):
            physics.set_lamda(bkd.array([-0.1]))

    def test_factory_produces_valid_model(self, bkd):
        """Zoo factory produces a working forward model."""
        npts = 20
        length = 1.0
        num_kle_terms = 2
        E_mean = 3.0
        nu = 0.3

        from pyapprox.pde.collocation.mesh import (
            AffineTransform1D,
            TransformedMesh1D,
        )

        transform = AffineTransform1D((0.0, length), bkd)
        mesh = TransformedMesh1D(npts, bkd, transform)
        field_map = _make_kle_field_map(bkd, mesh, num_kle_terms)

        fwd = create_hyperelastic_bar_1d(
            bkd=bkd,
            npts=npts,
            length=length,
            E_mean=E_mean,
            poisson_ratio=nu,
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
