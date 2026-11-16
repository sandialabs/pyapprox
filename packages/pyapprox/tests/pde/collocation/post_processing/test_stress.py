"""Tests for 2D stress post-processing."""

import math

import pytest

from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary import (
    traction_neumann_bc,
    zero_dirichlet_bc,
)
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms import PolarTransform
from pyapprox.pde.collocation.physics import LinearElasticityPhysics
from pyapprox.pde.collocation.post_processing.stress import (
    StressPostProcessor2D,
)
from pyapprox.pde.collocation.time_integration import CollocationModel

# ======================================================================
# Helpers
# ======================================================================


def _setup_lame_problem(bkd, npts_r=12, npts_theta=12):
    """Set up and solve the Lame analytical problem on a quarter-annulus."""
    r_inner, r_outer = 1.0, 2.0
    pressure = 1.0
    E_mean = 1.0
    poisson_ratio = 0.3

    dmu_dE = 1.0 / (2.0 * (1.0 + poisson_ratio))
    dlam_dE = poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    mu = E_mean * dmu_dE
    lamda = E_mean * dlam_dE

    transform = PolarTransform(
        (r_inner, r_outer),
        (0.0, math.pi / 2.0),
        bkd,
    )
    mesh = TransformedMesh2D(npts_r, npts_theta, bkd, transform)
    basis = ChebyshevBasis2D(mesh, bkd)

    physics = LinearElasticityPhysics(
        basis,
        bkd,
        lamda=lamda,
        mu=mu,
    )

    # Apply cylinder BCs
    D_matrices = [
        basis.derivative_matrix(1, 0),
        basis.derivative_matrix(1, 1),
    ]
    npts = basis.npts()
    bcs = []

    outer_idx = mesh.boundary_indices(1)
    outer_normals = mesh.boundary_normals(1)
    for comp in (0, 1):
        bcs.append(
            traction_neumann_bc(
                bkd,
                outer_idx,
                outer_normals,
                physics,
                npts,
                comp,
                values=0.0,
            )
        )

    bottom_idx = mesh.boundary_indices(2)
    bottom_normals = mesh.boundary_normals(2)
    bcs.append(
        traction_neumann_bc(
            bkd,
            bottom_idx,
            bottom_normals,
            physics,
            npts,
            component=0,
            values=0.0,
        )
    )

    top_idx = mesh.boundary_indices(3)
    top_normals = mesh.boundary_normals(3)
    bcs.append(
        traction_neumann_bc(
            bkd,
            top_idx,
            top_normals,
            physics,
            npts,
            component=1,
            values=0.0,
        )
    )

    inner_idx = mesh.boundary_indices(0)
    inner_normals = mesh.boundary_normals(0)
    for comp in (0, 1):
        pressure_vals = -pressure * inner_normals[:, comp]
        bcs.append(
            traction_neumann_bc(
                bkd,
                inner_idx,
                inner_normals,
                physics,
                npts,
                comp,
                values=pressure_vals,
            )
        )

    bottom_v_idx = bottom_idx + npts
    bcs.append(zero_dirichlet_bc(bkd, bottom_v_idx))
    bcs.append(zero_dirichlet_bc(bkd, top_idx))
    physics.set_boundary_conditions(bcs)

    # Solve
    model = CollocationModel(physics, bkd)
    init_state = bkd.zeros((2 * npts,))
    state = model.solve_steady(init_state, tol=1e-10, maxiter=50)

    # Create post-processor
    curv_basis = transform.unit_curvilinear_basis(
        mesh.reference_points(),
    )
    proc = StressPostProcessor2D(
        D_matrices[0],
        D_matrices[1],
        bkd=bkd,
        curvilinear_basis=curv_basis,
    )

    return {
        "state": state,
        "proc": proc,
        # (mu, lamda) at the nodes, the material the state was solved with
        "material": (bkd.full((npts,), mu), bkd.full((npts,), lamda)),
        "mesh": mesh,
        "basis": basis,
        "transform": transform,
        "r_inner": r_inner,
        "r_outer": r_outer,
        "pressure": pressure,
        "lamda": lamda,
        "mu": mu,
        "npts": npts,
        "bkd": bkd,
        "D_matrices": D_matrices,
    }


def _lame_stresses(bkd, pts, r_inner, r_outer, pressure):
    """Analytical Lame stresses in polar coordinates.

    Returns sigma_rr, sigma_tt, each shape (npts,).
    """
    x = pts[0, :]
    y = pts[1, :]
    r = bkd.sqrt(x**2 + y**2)
    Ri, Ro = r_inner, r_outer
    A = pressure * Ri**2 / (Ro**2 - Ri**2)
    B = -pressure * Ri**2 * Ro**2 / (Ro**2 - Ri**2)
    sigma_rr = A + B / r**2
    sigma_tt = A - B / r**2
    return sigma_rr, sigma_tt


# ======================================================================
# Tests
# ======================================================================


class TestStressPostProcessor2D:
    def test_hoop_stress_lame(self, bkd):
        """Hoop stress matches analytical Lame solution."""
        setup = _setup_lame_problem(bkd, npts_r=14, npts_theta=14)
        sigma_tt = setup["proc"].hoop_stress(setup["state"], *setup["material"])
        pts = setup["mesh"].points()
        _, sigma_tt_exact = _lame_stresses(
            bkd,
            pts,
            setup["r_inner"],
            setup["r_outer"],
            setup["pressure"],
        )
        bkd.assert_allclose(sigma_tt, sigma_tt_exact, rtol=1e-6)


    @pytest.mark.slow_on("TorchBkd")

    def test_radial_stress_lame(self, bkd):
        """Radial stress matches analytical Lame solution."""
        setup = _setup_lame_problem(bkd, npts_r=16, npts_theta=16)
        sigma_rr = setup["proc"].radial_stress(setup["state"], *setup["material"])
        pts = setup["mesh"].points()
        sigma_rr_exact, _ = _lame_stresses(
            bkd,
            pts,
            setup["r_inner"],
            setup["r_outer"],
            setup["pressure"],
        )
        bkd.assert_allclose(sigma_rr, sigma_rr_exact, atol=1e-6)

    def test_strain_energy_density_positive(self, bkd):
        """Strain energy density is positive for the Lame solution."""
        setup = _setup_lame_problem(bkd)
        psi = setup["proc"].strain_energy_density(setup["state"], *setup["material"])
        assert float(bkd.min(psi)) > 0.0

    def test_hoop_stress_jacobian_fd(self, bkd):
        """Hoop stress Jacobian matches finite differences."""
        setup = _setup_lame_problem(bkd, npts_r=8, npts_theta=8)
        proc = setup["proc"]
        state = setup["state"]
        npts = setup["npts"]

        mat = setup["material"]
        jac = proc.hoop_stress_state_jacobian(state, *mat)  # (npts, 2*npts)
        assert jac.shape == (npts, 2 * npts)

        # Finite difference check at a few DOFs
        eps = 1e-7
        for dof in [0, npts // 2, npts, 3 * npts // 2]:
            if dof >= 2 * npts:
                continue
            state_p = bkd.copy(state)
            state_p[dof] = state_p[dof] + eps
            state_m = bkd.copy(state)
            state_m[dof] = state_m[dof] - eps
            fd = (
                proc.hoop_stress(state_p, *mat) - proc.hoop_stress(state_m, *mat)
            ) / (2.0 * eps)
            bkd.assert_allclose(jac[:, dof], fd, rtol=1e-5, atol=1e-8)

    def test_strain_energy_density_jacobian_fd(self, bkd):
        """Strain energy density Jacobian matches finite differences."""
        setup = _setup_lame_problem(bkd, npts_r=8, npts_theta=8)
        proc = setup["proc"]
        state = setup["state"]
        npts = setup["npts"]

        mat = setup["material"]
        jac = proc.strain_energy_density_state_jacobian(state, *mat)
        assert jac.shape == (npts, 2 * npts)

        eps = 1e-7
        for dof in [0, npts // 2, npts, 3 * npts // 2]:
            if dof >= 2 * npts:
                continue
            state_p = bkd.copy(state)
            state_p[dof] = state_p[dof] + eps
            state_m = bkd.copy(state)
            state_m[dof] = state_m[dof] - eps
            fd = (
                proc.strain_energy_density(state_p, *mat)
                - proc.strain_energy_density(state_m, *mat)
            ) / (2.0 * eps)
            bkd.assert_allclose(jac[:, dof], fd, rtol=1e-4, atol=1e-8)

    def test_cartesian_stress_shapes(self, bkd):
        """Cartesian stresses have correct shapes."""
        setup = _setup_lame_problem(bkd)
        sxx, sxy, syy = setup["proc"].cartesian_stress(
            setup["state"], *setup["material"]
        )
        npts = setup["npts"]
        assert sxx.shape == (npts,)
        assert sxy.shape == (npts,)
        assert syy.shape == (npts,)

    def test_cartesian_stress_jacobian_shapes(self, bkd):
        """Cartesian stress Jacobians have correct shapes."""
        setup = _setup_lame_problem(bkd)
        npts = setup["npts"]
        dsxx, dsxy, dsyy = setup["proc"].cartesian_stress_state_jacobian(
            *setup["material"]
        )
        assert dsxx.shape == (npts, 2 * npts)
        assert dsxy.shape == (npts, 2 * npts)
        assert dsyy.shape == (npts, 2 * npts)

    def test_no_curvilinear_basis_raises(self, bkd):
        """hoop_stress raises when curvilinear_basis not provided."""
        setup = _setup_lame_problem(bkd)
        proc_no_curv = StressPostProcessor2D(
            setup["D_matrices"][0],
            setup["D_matrices"][1],
            bkd=bkd,
        )
        with pytest.raises(ValueError):
            proc_no_curv.hoop_stress(setup["state"], *setup["material"])

    def test_lame_jacobians_fd(self, bkd):
        """The pointwise Lame derivatives of the hoop stress and the strain
        energy density equal central differences in uniform mu and
        lamda shifts (each quantity is pointwise in the material)."""
        setup = _setup_lame_problem(bkd, npts_r=8, npts_theta=8)
        proc = setup["proc"]
        state = setup["state"]
        mu, lamda = setup["material"]
        eps = 1e-6
        for name in ("hoop_stress", "strain_energy_density"):
            value = getattr(proc, name)
            dmu, dlamda = getattr(proc, name + "_lame_jacobian")(state, mu, lamda)
            fd_mu = (value(state, mu + eps, lamda) - value(state, mu - eps, lamda)) / (
                2.0 * eps
            )
            fd_lamda = (
                value(state, mu, lamda + eps) - value(state, mu, lamda - eps)
            ) / (2.0 * eps)
            bkd.assert_allclose(dmu, fd_mu, rtol=1e-6, atol=1e-10)
            bkd.assert_allclose(dlamda, fd_lamda, rtol=1e-6, atol=1e-10)

    def test_stress_scales_with_material(self, bkd):
        """Linear stress is homogeneous of degree one in (mu, lamda) at
        fixed strain, and the strain energy density likewise."""
        setup = _setup_lame_problem(bkd, npts_r=8, npts_theta=8)
        proc = setup["proc"]
        state = setup["state"]
        mu, lamda = setup["material"]
        for name in ("hoop_stress", "strain_energy_density"):
            value = getattr(proc, name)
            bkd.assert_allclose(
                value(state, 3.0 * mu, 3.0 * lamda),
                3.0 * value(state, mu, lamda),
                rtol=1e-12,
            )
