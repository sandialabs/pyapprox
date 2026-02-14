"""Tests for the 2D hyperelastic pressurized cylinder zoo factory.

Verifies:
1. Manufactured solution with all-Dirichlet BCs (baseline)
2. Manufactured solution with mixed (traction + Dirichlet) BCs
3. Small pressure: hyperelastic QoI matches linear QoI
4. Parameter Jacobian on full forward model via DerivativeChecker
5. Factory produces valid model with correct protocol compliance
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
from pyapprox.typing.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.typing.pde.collocation.boundary.dirichlet import DirichletBC
from pyapprox.typing.pde.collocation.boundary.hyperelastic_traction import (
    hyperelastic_traction_neumann_bc,
)
from pyapprox.typing.pde.collocation.physics.hyperelasticity import (
    HyperelasticityPhysics,
)
from pyapprox.typing.pde.collocation.physics.stress_models import (
    NeoHookeanStress,
)
from pyapprox.typing.pde.collocation.manufactured_solutions.hyperelasticity import (
    ManufacturedHyperelasticityEquations,
)
from pyapprox.typing.pde.collocation.time_integration import CollocationModel
from pyapprox.typing.pde.zoo.hyperelastic_cylinder_2d import (
    create_hyperelastic_pressurized_cylinder_2d,
)
from pyapprox.typing.pde.zoo.pressurized_cylinder_2d import (
    create_linear_pressurized_cylinder_2d,
)
from pyapprox.typing.pde.field_maps.kle_factory import (
    create_lognormal_kle_field_map,
)
from pyapprox.typing.optimization.implicitfunction.functionals.elasticity_2d import (
    OuterWallRadialDisplacementFunctional,
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


def _make_kle_field_map_2d(bkd, mesh, num_kle_terms=2):
    """Create lognormal KLE field map on 2D polar mesh nodes."""
    physical_pts = mesh.points()  # (2, npts)
    npts = physical_pts.shape[1]

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


def _make_outer_wall_functional(bkd, mesh, npts, nparams):
    """Create OuterWallRadialDisplacementFunctional for comparison tests."""
    outer_idx = mesh.boundary_indices(1)
    pts = mesh.points()
    outer_pts = pts[:, bkd.to_numpy(outer_idx).astype(int)]
    r = bkd.sqrt(outer_pts[0, :] ** 2 + outer_pts[1, :] ** 2)
    cos_theta = outer_pts[0, :] / r
    sin_theta = outer_pts[1, :] / r
    return OuterWallRadialDisplacementFunctional(
        outer_idx, cos_theta, sin_theta, npts, nparams, bkd,
    )


# ======================================================================
# Test class
# ======================================================================

class TestHyperelasticCylinder2D(Generic[Array], unittest.TestCase):
    """Tests for 2D hyperelastic pressurized cylinder."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    # ------------------------------------------------------------------
    # Manufactured solution with all-Dirichlet BCs (baseline)
    # ------------------------------------------------------------------

    def test_manufactured_all_dirichlet(self):
        """Manufactured hyperelastic solution with all-Dirichlet BCs."""
        bkd = self._bkd
        npts_r, npts_theta = 12, 12
        r_inner, r_outer = 1.0, 2.0
        lamda_val, mu_val = 1.0, 1.0

        mesh, basis = _setup_polar_mesh_and_basis(
            bkd, npts_r, npts_theta, r_inner, r_outer,
        )
        npts = basis.npts()
        pts = mesh.points()

        stress_model = NeoHookeanStress(lamda=lamda_val, mu=mu_val)

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
        forcing = man_sol.functions["forcing"](pts)    # (npts, 2)
        forcing_flat = bkd.concatenate([forcing[:, 0], forcing[:, 1]])

        physics = HyperelasticityPhysics(
            basis, bkd, stress_model, forcing=lambda t: forcing_flat,
        )

        # All-Dirichlet BCs on all 4 sides, both components
        bcs = []
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
            bkd.zeros((2 * npts,)), tol=1e-10, maxiter=50,
        )
        bkd.assert_allclose(sol, exact_flat, atol=1e-7)

    # ------------------------------------------------------------------
    # Manufactured solution with mixed BCs
    # ------------------------------------------------------------------

    def test_manufactured_mixed_bcs(self):
        """Manufactured hyperelastic solution with cylinder-style BCs."""
        bkd = self._bkd
        npts_r, npts_theta = 12, 12
        r_inner, r_outer = 1.0, 2.5
        lamda_val, mu_val = 1.5, 0.8

        mesh, basis = _setup_polar_mesh_and_basis(
            bkd, npts_r, npts_theta, r_inner, r_outer,
        )
        npts = basis.npts()
        pts = mesh.points()
        D_matrices = [
            basis.derivative_matrix(1, 0),
            basis.derivative_matrix(1, 1),
        ]

        stress_model = NeoHookeanStress(lamda=lamda_val, mu=mu_val)

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
        forcing = man_sol.functions["forcing"](pts)    # (npts, 2)
        forcing_flat = bkd.concatenate([forcing[:, 0], forcing[:, 1]])
        flux = man_sol.functions["flux"](pts)  # (2, npts, 2)

        physics = HyperelasticityPhysics(
            basis, bkd, stress_model, forcing=lambda t: forcing_flat,
        )

        bcs = []

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
            bcs.append(hyperelastic_traction_neumann_bc(
                bkd, inner_idx, inner_normals, D_matrices,
                stress_model, npts, comp, values=traction_vals,
            ))

        # Outer (bnd 1): traction with exact manufactured PK1 traction
        outer_idx = mesh.boundary_indices(1)
        outer_normals = mesh.boundary_normals(1)
        outer_idx_np = bkd.to_numpy(outer_idx).astype(int)
        for comp in (0, 1):
            traction_vals = (
                flux[comp, outer_idx_np, 0] * outer_normals[:, 0]
                + flux[comp, outer_idx_np, 1] * outer_normals[:, 1]
            )
            bcs.append(hyperelastic_traction_neumann_bc(
                bkd, outer_idx, outer_normals, D_matrices,
                stress_model, npts, comp, values=traction_vals,
            ))

        # Bottom (bnd 2): traction t_x (comp 0), Dirichlet v (comp 1)
        bottom_idx = mesh.boundary_indices(2)
        bottom_normals = mesh.boundary_normals(2)
        bottom_idx_np = bkd.to_numpy(bottom_idx).astype(int)
        traction_bottom_x = (
            flux[0, bottom_idx_np, 0] * bottom_normals[:, 0]
            + flux[0, bottom_idx_np, 1] * bottom_normals[:, 1]
        )
        bcs.append(hyperelastic_traction_neumann_bc(
            bkd, bottom_idx, bottom_normals, D_matrices,
            stress_model, npts, component=0, values=traction_bottom_x,
        ))
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
        bcs.append(hyperelastic_traction_neumann_bc(
            bkd, top_idx, top_normals, D_matrices,
            stress_model, npts, component=1, values=traction_top_y,
        ))

        physics.set_boundary_conditions(bcs)

        model = CollocationModel(physics, bkd)
        exact_flat = bkd.concatenate([u_exact[:, 0], u_exact[:, 1]])
        sol = model.solve_steady(
            bkd.zeros((2 * npts,)), tol=1e-10, maxiter=50,
        )
        bkd.assert_allclose(sol, exact_flat, atol=1e-7)

    # ------------------------------------------------------------------
    # Small pressure: hyperelastic ≈ linear
    # ------------------------------------------------------------------

    def test_small_pressure_matches_linear(self):
        """Low pressure: hyperelastic QoI matches linear QoI."""
        bkd = self._bkd
        npts_r, npts_theta = 10, 10
        r_inner, r_outer = 1.0, 2.0
        E_mean = 1.0
        nu = 0.3
        pressure = 1e-3
        num_kle_terms = 2

        mesh, basis = _setup_polar_mesh_and_basis(
            bkd, npts_r, npts_theta, r_inner, r_outer,
        )
        npts = basis.npts()
        field_map = _make_kle_field_map_2d(bkd, mesh, num_kle_terms)

        functional_hyper = _make_outer_wall_functional(
            bkd, mesh, npts, num_kle_terms,
        )
        functional_linear = _make_outer_wall_functional(
            bkd, mesh, npts, num_kle_terms,
        )

        fwd_hyper = create_hyperelastic_pressurized_cylinder_2d(
            bkd=bkd,
            npts_r=npts_r, npts_theta=npts_theta,
            r_inner=r_inner, r_outer=r_outer,
            E_mean=E_mean, poisson_ratio=nu,
            inner_pressure=pressure,
            field_map=field_map,
            functional=functional_hyper,
        )
        fwd_linear = create_linear_pressurized_cylinder_2d(
            bkd=bkd,
            npts_r=npts_r, npts_theta=npts_theta,
            r_inner=r_inner, r_outer=r_outer,
            E_mean=E_mean, poisson_ratio=nu,
            inner_pressure=pressure,
            field_map=field_map,
            functional=functional_linear,
        )

        sample = bkd.zeros((num_kle_terms, 1))
        qoi_hyper = fwd_hyper(sample)
        qoi_linear = fwd_linear(sample)

        bkd.assert_allclose(qoi_hyper, qoi_linear, rtol=1e-2)

    # ------------------------------------------------------------------
    # Parameter Jacobian via DerivativeChecker
    # ------------------------------------------------------------------

    def test_param_jacobian(self):
        """DerivativeChecker on KLE-parameterized hyperelastic cylinder."""
        bkd = self._bkd
        npts_r, npts_theta = 8, 8
        r_inner, r_outer = 1.0, 2.0
        E_mean = 1.0
        nu = 0.3
        pressure = 0.1
        num_kle_terms = 2

        mesh, basis = _setup_polar_mesh_and_basis(
            bkd, npts_r, npts_theta, r_inner, r_outer,
        )
        field_map = _make_kle_field_map_2d(bkd, mesh, num_kle_terms)

        fwd = create_hyperelastic_pressurized_cylinder_2d(
            bkd=bkd,
            npts_r=npts_r, npts_theta=npts_theta,
            r_inner=r_inner, r_outer=r_outer,
            E_mean=E_mean, poisson_ratio=nu,
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

        fwd = create_hyperelastic_pressurized_cylinder_2d(
            bkd=bkd,
            npts_r=npts_r, npts_theta=npts_theta,
            r_inner=r_inner, r_outer=r_outer,
            E_mean=1.0, poisson_ratio=0.3,
            inner_pressure=1.0,
            field_map=field_map,
        )

        self.assertTrue(isinstance(fwd, FunctionWithJacobianProtocol))

        npts = npts_r * npts_theta
        self.assertEqual(fwd.nvars(), num_kle_terms)
        self.assertEqual(fwd.nqoi(), 2 * npts)

        sample = bkd.zeros((num_kle_terms, 1))
        result = fwd(sample)
        self.assertEqual(result.shape, (2 * npts, 1))

        jac = fwd.jacobian(sample)
        self.assertEqual(jac.shape, (2 * npts, num_kle_terms))


# ======================================================================
# Backend-specific test classes
# ======================================================================

class TestHyperelasticCylinder2DNumpy(
    TestHyperelasticCylinder2D[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestHyperelasticCylinder2DTorch(
    TestHyperelasticCylinder2D[torch.Tensor]
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()
