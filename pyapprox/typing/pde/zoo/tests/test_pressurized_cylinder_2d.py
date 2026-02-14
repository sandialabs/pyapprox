"""Tests for the 2D linear elastic pressurized cylinder zoo factory.

Verifies:
1. Lame analytical solution with full cylinder BC structure
2. Manufactured solution with mixed (traction + Dirichlet) BCs on polar domain
3. Manufactured solution with all-Dirichlet BCs (baseline)
4. Variable Lame Jacobian on polar domain via DerivativeChecker
5. Parameter Jacobian on full forward model via DerivativeChecker
6. Factory produces valid model with correct protocol compliance
"""

import math
import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.interface.functions.protocols import (
    FunctionWithJacobianProtocol,
)
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.typing.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.typing.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.typing.pde.collocation.mesh import TransformedMesh2D
from pyapprox.typing.pde.collocation.mesh.transforms import PolarTransform
from pyapprox.typing.pde.collocation.boundary import (
    zero_dirichlet_bc,
    traction_neumann_bc,
)
from pyapprox.typing.pde.collocation.boundary.dirichlet import DirichletBC
from pyapprox.typing.pde.collocation.physics import LinearElasticityPhysics
from pyapprox.typing.pde.collocation.manufactured_solutions import (
    ManufacturedLinearElasticityEquations,
)
from pyapprox.typing.pde.collocation.time_integration import CollocationModel
from pyapprox.typing.pde.zoo.pressurized_cylinder_2d import (
    create_linear_pressurized_cylinder_2d,
)
from pyapprox.typing.pde.field_maps.kle_factory import (
    create_lognormal_kle_field_map,
)


# ======================================================================
# Helpers
# ======================================================================

def _setup_polar_mesh_and_basis(bkd, npts_r, npts_theta, r_inner, r_outer):
    """Create polar mesh and 2D Chebyshev basis on quarter-annulus."""
    transform = PolarTransform(
        (r_inner, r_outer), (0.0, math.pi / 2.0), bkd,
    )
    mesh = TransformedMesh2D(npts_r, npts_theta, bkd, transform)
    basis = ChebyshevBasis2D(mesh, bkd)
    return mesh, basis


def _lame_analytical(bkd, pts, r_inner, r_outer, pressure, lamda, mu):
    """Compute analytical Lame solution in Cartesian coordinates.

    Returns (u_x, u_y) each of shape (npts,) for plane strain.

    Parameters
    ----------
    pts : Array
        Physical coordinates, shape (2, npts).
    """
    x = pts[0, :]
    y = pts[1, :]
    r = bkd.sqrt(x**2 + y**2)

    Ri, Ro = r_inner, r_outer
    A = pressure * Ri**2 / (Ro**2 - Ri**2)
    B = -pressure * Ri**2 * Ro**2 / (Ro**2 - Ri**2)

    # Plane strain Poisson ratio from Lame params
    nu = lamda / (2.0 * (lamda + mu))

    # Radial displacement (plane strain)
    u_r = (1.0 / (2.0 * mu)) * ((1.0 - 2.0 * nu) * A * r - B / r)

    # Convert to Cartesian
    u_x = u_r * x / r
    u_y = u_r * y / r
    return u_x, u_y


def _make_kle_field_map_2d(bkd, mesh, num_kle_terms=2):
    """Create lognormal KLE field map on 2D polar mesh nodes."""
    physical_pts = mesh.points()  # (2, npts)
    npts = physical_pts.shape[1]

    # Normalize coordinates to [0, 1]^2 for correlation kernel
    x = physical_pts[0, :]
    y = physical_pts[1, :]
    x_min, x_max = float(bkd.min(x)), float(bkd.max(x))
    y_min, y_max = float(bkd.min(y)), float(bkd.max(y))
    x_range = max(x_max - x_min, 1e-12)
    y_range = max(y_max - y_min, 1e-12)
    x_norm = (x - x_min) / x_range
    y_norm = (y - y_min) / y_range
    mesh_coords = bkd.stack([x_norm, y_norm], axis=0)  # (2, npts)

    mean_log = bkd.zeros((npts,))
    return create_lognormal_kle_field_map(
        mesh_coords, mean_log, bkd,
        num_kle_terms=num_kle_terms, sigma=0.3,
    )


def _apply_cylinder_bcs(bkd, physics, mesh, basis, lamda, mu, inner_pressure):
    """Apply the full cylinder BC set (traction + symmetry Dirichlet).

    Same BC structure as create_linear_pressurized_cylinder_2d.
    """
    D_matrices = [
        basis.derivative_matrix(1, 0),
        basis.derivative_matrix(1, 1),
    ]
    npts = basis.npts()
    bcs = []

    # (a) Outer (bnd 1): stress-free
    outer_idx = mesh.boundary_indices(1)
    outer_normals = mesh.boundary_normals(1)
    for comp in (0, 1):
        bcs.append(traction_neumann_bc(
            bkd, outer_idx, outer_normals, D_matrices,
            lamda, mu, npts, comp, values=0.0,
        ))

    # (b) Bottom (bnd 2): shear-free t_x=0
    bottom_idx = mesh.boundary_indices(2)
    bottom_normals = mesh.boundary_normals(2)
    bcs.append(traction_neumann_bc(
        bkd, bottom_idx, bottom_normals, D_matrices,
        lamda, mu, npts, component=0, values=0.0,
    ))

    # (c) Top (bnd 3): shear-free t_y=0
    top_idx = mesh.boundary_indices(3)
    top_normals = mesh.boundary_normals(3)
    bcs.append(traction_neumann_bc(
        bkd, top_idx, top_normals, D_matrices,
        lamda, mu, npts, component=1, values=0.0,
    ))

    # (d) Inner (bnd 0): pressure traction
    inner_idx = mesh.boundary_indices(0)
    inner_normals = mesh.boundary_normals(0)
    for comp in (0, 1):
        pressure_vals = -inner_pressure * inner_normals[:, comp]
        bcs.append(traction_neumann_bc(
            bkd, inner_idx, inner_normals, D_matrices,
            lamda, mu, npts, comp, values=pressure_vals,
        ))

    # (e) Bottom Dirichlet v=0
    bottom_v_idx = bottom_idx + npts
    bcs.append(zero_dirichlet_bc(bkd, bottom_v_idx))

    # (f) Top Dirichlet u=0
    bcs.append(zero_dirichlet_bc(bkd, top_idx))

    physics.set_boundary_conditions(bcs)


# ======================================================================
# Test class
# ======================================================================

class TestPressurizedCylinder2D(Generic[Array], unittest.TestCase):
    """Tests for 2D linear elastic pressurized cylinder."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    # ------------------------------------------------------------------
    # Lame analytical solution
    # ------------------------------------------------------------------

    def test_lame_solution(self):
        """Solve with constant E, compare to Lame analytical solution."""
        bkd = self._bkd
        npts_r, npts_theta = 16, 16
        r_inner, r_outer = 1.0, 2.0
        pressure = 1.0
        lamda, mu = 1.0, 1.0

        mesh, basis = _setup_polar_mesh_and_basis(
            bkd, npts_r, npts_theta, r_inner, r_outer,
        )
        npts = basis.npts()
        pts = mesh.points()

        physics = LinearElasticityPhysics(
            basis, bkd, lamda=lamda, mu=mu,
        )
        _apply_cylinder_bcs(
            bkd, physics, mesh, basis, lamda, mu, pressure,
        )

        model = CollocationModel(physics, bkd)
        init_state = bkd.zeros((2 * npts,))
        sol = model.solve_steady(init_state, tol=1e-10, maxiter=50)

        # Analytical Lame solution
        u_x_exact, u_y_exact = _lame_analytical(
            bkd, pts, r_inner, r_outer, pressure, lamda, mu,
        )
        exact_state = bkd.concatenate([u_x_exact, u_y_exact])
        bkd.assert_allclose(sol, exact_state, atol=1e-8)

    # ------------------------------------------------------------------
    # Manufactured solution with mixed BCs (same structure as factory)
    # ------------------------------------------------------------------

    def test_manufactured_mixed_bcs(self):
        """Manufactured solution with cylinder-style BCs on polar domain."""
        bkd = self._bkd
        npts_r, npts_theta = 12, 12
        r_inner, r_outer = 1.0, 2.5
        lamda_val, mu_val = 1.5, 0.8

        mesh, basis = _setup_polar_mesh_and_basis(
            bkd, npts_r, npts_theta, r_inner, r_outer,
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
        forcing = man_sol.functions["forcing"](pts)    # (npts, 2)
        forcing_flat = bkd.concatenate([forcing[:, 0], forcing[:, 1]])

        physics = LinearElasticityPhysics(
            basis, bkd, lamda=lamda_val, mu=mu_val,
            forcing=lambda t: forcing_flat,
        )

        D_matrices = [
            basis.derivative_matrix(1, 0),
            basis.derivative_matrix(1, 1),
        ]
        bcs = []

        # Inner boundary (bnd 0): traction with exact manufactured values
        inner_idx = mesh.boundary_indices(0)
        inner_normals = mesh.boundary_normals(0)
        inner_pts = pts[:, bkd.to_numpy(inner_idx).astype(int)]
        inner_traction = man_sol.traction_values(
            inner_pts, inner_normals,
        )  # (nbnd, 2)
        for comp in (0, 1):
            bcs.append(traction_neumann_bc(
                bkd, inner_idx, inner_normals, D_matrices,
                lamda_val, mu_val, npts, comp,
                values=inner_traction[:, comp],
            ))

        # Outer boundary (bnd 1): traction with exact manufactured values
        outer_idx = mesh.boundary_indices(1)
        outer_normals = mesh.boundary_normals(1)
        outer_pts = pts[:, bkd.to_numpy(outer_idx).astype(int)]
        outer_traction = man_sol.traction_values(
            outer_pts, outer_normals,
        )
        for comp in (0, 1):
            bcs.append(traction_neumann_bc(
                bkd, outer_idx, outer_normals, D_matrices,
                lamda_val, mu_val, npts, comp,
                values=outer_traction[:, comp],
            ))

        # Bottom (bnd 2, theta=0): traction t_x (comp 0) + Dirichlet v (comp 1)
        bottom_idx = mesh.boundary_indices(2)
        bottom_normals = mesh.boundary_normals(2)
        bottom_pts = pts[:, bkd.to_numpy(bottom_idx).astype(int)]
        bottom_traction = man_sol.traction_values(
            bottom_pts, bottom_normals,
        )
        bcs.append(traction_neumann_bc(
            bkd, bottom_idx, bottom_normals, D_matrices,
            lamda_val, mu_val, npts, component=0,
            values=bottom_traction[:, 0],
        ))
        # Dirichlet v at bottom
        bottom_v_idx = bottom_idx + npts
        bottom_v_exact = u_exact[bkd.to_numpy(bottom_idx).astype(int), 1]
        bcs.append(DirichletBC(bkd, bottom_v_idx, bottom_v_exact))

        # Top (bnd 3, theta=pi/2): Dirichlet u (comp 0) + traction t_y (comp 1)
        top_idx = mesh.boundary_indices(3)
        top_normals = mesh.boundary_normals(3)
        top_pts = pts[:, bkd.to_numpy(top_idx).astype(int)]
        top_traction = man_sol.traction_values(
            top_pts, top_normals,
        )
        # Dirichlet u at top
        top_u_exact = u_exact[bkd.to_numpy(top_idx).astype(int), 0]
        bcs.append(DirichletBC(bkd, top_idx, top_u_exact))
        bcs.append(traction_neumann_bc(
            bkd, top_idx, top_normals, D_matrices,
            lamda_val, mu_val, npts, component=1,
            values=top_traction[:, 1],
        ))

        physics.set_boundary_conditions(bcs)

        # Solve
        model = CollocationModel(physics, bkd)
        exact_flat = bkd.concatenate([u_exact[:, 0], u_exact[:, 1]])
        sol = model.solve_steady(bkd.zeros((2 * npts,)), tol=1e-10, maxiter=50)
        bkd.assert_allclose(sol, exact_flat, atol=1e-8)

    # ------------------------------------------------------------------
    # Manufactured solution with all-Dirichlet BCs (baseline)
    # ------------------------------------------------------------------

    def test_manufactured_all_dirichlet(self):
        """Manufactured solution on polar domain with all-Dirichlet BCs."""
        bkd = self._bkd
        npts_r, npts_theta = 12, 12
        r_inner, r_outer = 1.0, 2.0
        lamda_val, mu_val = 1.0, 1.0

        mesh, basis = _setup_polar_mesh_and_basis(
            bkd, npts_r, npts_theta, r_inner, r_outer,
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
            basis, bkd, lamda=lamda_val, mu=mu_val,
            forcing=lambda t: forcing_flat,
        )

        # Dirichlet BCs on all 4 sides, both components
        bcs = []
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

    def test_variable_lame_jacobian_on_polar(self):
        """DerivativeChecker validates variable-Lame Jacobian on polar."""
        bkd = self._bkd
        npts_r, npts_theta = 6, 6
        r_inner, r_outer = 1.0, 2.0

        mesh, basis = _setup_polar_mesh_and_basis(
            bkd, npts_r, npts_theta, r_inner, r_outer,
        )
        npts = basis.npts()
        pts = mesh.points()
        x = pts[0, :]
        y = pts[1, :]

        # Spatially varying Lame params
        mu_field = 1.0 + 0.3 * bkd.sin(math.pi * x) * bkd.cos(math.pi * y)
        lam_field = 1.5 + 0.2 * bkd.cos(math.pi * x) * bkd.sin(math.pi * y)

        physics = LinearElasticityPhysics(
            basis, bkd, lamda=1.0, mu=1.0,
        )
        physics.set_mu(mu_field)
        physics.set_lamda(lam_field)

        # All-Dirichlet for clean Jacobian test
        bcs = []
        for side in range(4):
            bnd_idx = mesh.boundary_indices(side)
            bcs.append(zero_dirichlet_bc(bkd, bnd_idx))
            bcs.append(zero_dirichlet_bc(bkd, bnd_idx + npts))
        physics.set_boundary_conditions(bcs)

        # Wrap physics for DerivativeChecker
        class _PhysWrapper:
            def __init__(self, phys, bk):
                self._p = phys
                self._b = bk
            def bkd(self):
                return self._b
            def nvars(self):
                return self._p.nstates()
            def nqoi(self):
                return self._p.nstates()
            def __call__(self, samples):
                return self._b.stack(
                    [self._p.residual(samples[:, i], 0.0)
                     for i in range(samples.shape[1])],
                    axis=1,
                )
            def jacobian(self, sample):
                return self._p.jacobian(sample[:, 0], 0.0)

        wrapper = _PhysWrapper(physics, bkd)
        np.random.seed(42)
        state_np = np.random.randn(2 * npts) * 0.1
        sample = bkd.asarray(state_np)[:, None]

        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, verbosity=0)
        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-5)

    # ------------------------------------------------------------------
    # Parameter Jacobian on full forward model
    # ------------------------------------------------------------------

    def test_param_jacobian(self):
        """DerivativeChecker on KLE-parameterized cylinder forward model."""
        bkd = self._bkd
        npts_r, npts_theta = 8, 8
        r_inner, r_outer = 1.0, 2.0
        E_mean = 1.0
        nu = 0.3
        pressure = 0.5
        num_kle_terms = 2

        mesh, basis = _setup_polar_mesh_and_basis(
            bkd, npts_r, npts_theta, r_inner, r_outer,
        )
        field_map = _make_kle_field_map_2d(bkd, mesh, num_kle_terms)

        fwd = create_linear_pressurized_cylinder_2d(
            bkd=bkd,
            npts_r=npts_r,
            npts_theta=npts_theta,
            r_inner=r_inner,
            r_outer=r_outer,
            E_mean=E_mean,
            poisson_ratio=nu,
            inner_pressure=pressure,
            field_map=field_map,
        )

        self.assertTrue(isinstance(fwd, FunctionWithJacobianProtocol))

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
        self.assertLessEqual(ratio, 1e-5)

    # ------------------------------------------------------------------
    # Factory produces valid model
    # ------------------------------------------------------------------

    def test_factory_produces_valid_model(self):
        """Zoo factory returns valid SteadyForwardModel with correct shapes."""
        bkd = self._bkd
        npts_r, npts_theta = 6, 6
        r_inner, r_outer = 1.0, 2.0
        num_kle_terms = 2

        mesh, basis = _setup_polar_mesh_and_basis(
            bkd, npts_r, npts_theta, r_inner, r_outer,
        )
        field_map = _make_kle_field_map_2d(bkd, mesh, num_kle_terms)

        fwd = create_linear_pressurized_cylinder_2d(
            bkd=bkd,
            npts_r=npts_r,
            npts_theta=npts_theta,
            r_inner=r_inner,
            r_outer=r_outer,
            E_mean=1.0,
            poisson_ratio=0.3,
            inner_pressure=1.0,
            field_map=field_map,
        )

        # Check protocol compliance
        self.assertTrue(isinstance(fwd, FunctionWithJacobianProtocol))

        # Check dimensions
        npts = npts_r * npts_theta
        self.assertEqual(fwd.nvars(), num_kle_terms)
        self.assertEqual(fwd.nqoi(), 2 * npts)

        # Evaluate at zero parameters
        sample = bkd.zeros((num_kle_terms, 1))
        result = fwd(sample)
        self.assertEqual(result.shape, (2 * npts, 1))

        # Jacobian shape
        jac = fwd.jacobian(sample)
        self.assertEqual(jac.shape, (2 * npts, num_kle_terms))


# ======================================================================
# Backend-specific test classes
# ======================================================================

class TestPressurizedCylinder2DNumpy(
    TestPressurizedCylinder2D[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPressurizedCylinder2DTorch(
    TestPressurizedCylinder2D[torch.Tensor]
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()
