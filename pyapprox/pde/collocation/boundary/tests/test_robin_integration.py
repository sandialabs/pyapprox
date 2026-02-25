"""Integration tests for Robin/Neumann BCs with full PDE solves.

Tests gradient and flux conventions with manufactured solutions,
verifying that the numerical solution matches the exact solution.
"""

import math
import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401
from pyapprox.pde.collocation.basis import (
    ChebyshevBasis1D,
    ChebyshevBasis2D,
)
from pyapprox.pde.collocation.mesh import (
    create_uniform_mesh_1d,
    TransformedMesh1D,
    TransformedMesh2D,
)
from pyapprox.pde.collocation.boundary import (
    DirichletBC,
    constant_dirichlet_bc,
    gradient_neumann_bc,
    gradient_robin_bc,
    flux_neumann_bc,
    flux_robin_bc,
    traction_robin_bc,
    traction_neumann_bc,
)
from pyapprox.pde.collocation.mesh.transforms.polar import PolarTransform
from pyapprox.pde.collocation.mesh import create_uniform_mesh_2d
from pyapprox.pde.collocation.physics import AdvectionDiffusionReaction
from pyapprox.pde.collocation.physics.linear_elasticity import (
    LinearElasticityPhysics,
)
from pyapprox.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)
from pyapprox.pde.collocation.manufactured_solutions import (
    ManufacturedAdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.manufactured_solutions.linear_elasticity import (
    ManufacturedLinearElasticityEquations,
)


class TestGradientNeumannSolve(Generic[Array], unittest.TestCase):
    """Integration tests for gradient Neumann BCs."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_1d_gradient_neumann_solve(self):
        """1D diffusion: Neumann left, Dirichlet right.

        u = x^2 + x, D = 1
        du/dx = 2x + 1
        At x=-1: grad(u).n = (2*(-1)+1)*(-1) = 1
        At x=+1: u = 2 (Dirichlet)
        """
        bkd = self.bkd()
        npts = 10
        tmesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(tmesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="x**2 + x",
            nvars=1,
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes().reshape(1, -1)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, forcing=lambda t: forcing
        )

        # Left: gradient Neumann, grad(u).n = g
        left_idx = umesh.boundary_indices(0)
        left_normals = tmesh.boundary_normals(0)
        neumann_val = man_sol.neumann_values(
            tmesh.points()[:, left_idx], left_normals, convention="gradient"
        )
        bc_left = gradient_neumann_bc(
            bkd, left_idx, left_normals,
            [basis.derivative_matrix()], neumann_val
        )

        # Right: Dirichlet
        right_idx = umesh.boundary_indices(1)
        bc_right = constant_dirichlet_bc(bkd, right_idx, float(u_exact[right_idx[0]]))

        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)
        initial_guess = bkd.zeros((npts,))
        u_numerical = model.solve_steady(initial_guess, tol=1e-12, maxiter=50)

        bkd.assert_allclose(u_numerical, u_exact, atol=1e-10)

    def test_2d_gradient_neumann_solve(self):
        """2D diffusion: Neumann on left, Dirichlet on other 3 sides.

        u = (1 - x^2)*y, D = 1
        At x=-1: du/dx = -2x*y = 2y at x=-1, grad(u).n = 2y*(-1) = -2y
        """
        bkd = self.bkd()
        npts_x, npts_y = 10, 10
        tmesh = TransformedMesh2D(npts_x, npts_y, bkd)
        basis = ChebyshevBasis2D(tmesh, bkd)

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="(1 - x**2)*y",
            nvars=2,
            diff_str="1.0",
            react_str="0",
            vel_strs=["0", "0"],
            bkd=bkd,
            oned=True,
        )

        pts = tmesh.points()
        u_exact = man_sol.functions["solution"](pts)
        forcing = man_sol.functions["forcing"](pts)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, forcing=lambda t: forcing
        )

        Dx = basis.derivative_matrix(1, 0)
        Dy = basis.derivative_matrix(1, 1)
        D_list = [Dx, Dy]

        # Left: gradient Neumann
        left_idx = tmesh.boundary_indices(0)
        left_normals = tmesh.boundary_normals(0)
        left_pts = pts[:, left_idx]
        neumann_left = man_sol.neumann_values(
            left_pts, left_normals, convention="gradient"
        )
        bc_left = gradient_neumann_bc(
            bkd, left_idx, left_normals, D_list, neumann_left
        )

        # Other 3 sides: Dirichlet
        bcs = [bc_left]
        for side in range(1, 4):
            idx = tmesh.boundary_indices(side)
            bc = constant_dirichlet_bc(bkd, idx, 0.0)
            # Use exact values for non-zero boundaries
            vals = u_exact[idx]
            from pyapprox.pde.collocation.boundary import DirichletBC
            bc = DirichletBC(bkd, idx, lambda t, v=vals: v)
            bcs.append(bc)

        physics.set_boundary_conditions(bcs)

        model = CollocationModel(physics, bkd)
        initial_guess = bkd.zeros((npts_x * npts_y,))
        u_numerical = model.solve_steady(initial_guess, tol=1e-12, maxiter=50)

        bkd.assert_allclose(u_numerical, u_exact, atol=1e-9)


class TestGradientRobinSolve(Generic[Array], unittest.TestCase):
    """Integration tests for gradient Robin BCs."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_1d_gradient_robin_solve(self):
        """1D diffusion: Robin left, Dirichlet right.

        u = x^2 + 2x + 1, D = 1
        Robin: alpha*u + beta*grad(u).n = g, alpha=1, beta=1
        At x=-1: u=0, du/dx=0, grad(u).n=0*(-1)=0, g=1*0+1*0=0
        At x=+1: u=4 (Dirichlet)
        """
        bkd = self.bkd()
        npts = 10
        tmesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(tmesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="x**2 + 2*x + 1",
            nvars=1,
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes().reshape(1, -1)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, forcing=lambda t: forcing
        )

        # Left: Robin with alpha=1, beta=1
        left_idx = umesh.boundary_indices(0)
        left_normals = tmesh.boundary_normals(0)
        robin_val = man_sol.robin_values(
            tmesh.points()[:, left_idx], left_normals,
            alpha=1.0, beta=1.0, convention="gradient"
        )
        bc_left = gradient_robin_bc(
            bkd, left_idx, left_normals,
            [basis.derivative_matrix()], 1.0, 1.0, robin_val
        )

        # Right: Dirichlet
        right_idx = umesh.boundary_indices(1)
        bc_right = constant_dirichlet_bc(bkd, right_idx, float(u_exact[right_idx[0]]))

        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)
        initial_guess = bkd.zeros((npts,))
        u_numerical = model.solve_steady(initial_guess, tol=1e-12, maxiter=50)

        bkd.assert_allclose(u_numerical, u_exact, atol=1e-10)


class TestFluxNeumannSolve(Generic[Array], unittest.TestCase):
    """Integration tests for flux Neumann BCs."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_1d_flux_neumann_solve(self):
        """1D diffusion: flux Neumann left, Dirichlet right.

        u = x^2, D = 2
        flux = -D*du/dx = -2*2x = -4x
        At x=-1: flux = 4, flux.n = 4*(-1) = -4
        At x=+1: u = 1 (Dirichlet)
        """
        bkd = self.bkd()
        npts = 10
        tmesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(tmesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="x**2",
            nvars=1,
            diff_str="2.0",
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes().reshape(1, -1)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=2.0, forcing=lambda t: forcing
        )

        # Left: flux Neumann
        left_idx = umesh.boundary_indices(0)
        left_normals = tmesh.boundary_normals(0)
        neumann_val = man_sol.neumann_values(
            tmesh.points()[:, left_idx], left_normals, convention="flux"
        )
        bc_left = flux_neumann_bc(
            bkd, left_idx, left_normals, physics, neumann_val
        )

        # Right: Dirichlet
        right_idx = umesh.boundary_indices(1)
        bc_right = constant_dirichlet_bc(bkd, right_idx, float(u_exact[right_idx[0]]))

        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)
        initial_guess = bkd.zeros((npts,))
        u_numerical = model.solve_steady(initial_guess, tol=1e-12, maxiter=50)

        bkd.assert_allclose(u_numerical, u_exact, atol=1e-10)

    def test_1d_flux_neumann_with_velocity(self):
        """1D ADR: flux Neumann left, Dirichlet right.

        u = x^2, D = 1, v = 2
        flux = -D*du/dx + v*u = -2x + 2x^2
        At x=-1: flux = 2 + 2 = 4, flux.n = 4*(-1) = -4
        At x=+1: u = 1 (Dirichlet)
        """
        bkd = self.bkd()
        npts = 10
        tmesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(tmesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="x**2",
            nvars=1,
            diff_str="1.0",
            react_str="0",
            vel_strs=["2"],
            bkd=bkd,
            oned=True,
            conservative=True,
        )

        nodes = basis.nodes().reshape(1, -1)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        velocity = [bkd.full((npts,), 2.0)]
        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, velocity=velocity,
            forcing=lambda t: forcing
        )

        # Left: flux Neumann
        left_idx = umesh.boundary_indices(0)
        left_normals = tmesh.boundary_normals(0)
        neumann_val = man_sol.neumann_values(
            tmesh.points()[:, left_idx], left_normals, convention="flux"
        )
        bc_left = flux_neumann_bc(
            bkd, left_idx, left_normals, physics, neumann_val
        )

        # Right: Dirichlet
        right_idx = umesh.boundary_indices(1)
        bc_right = constant_dirichlet_bc(bkd, right_idx, float(u_exact[right_idx[0]]))

        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)
        initial_guess = bkd.zeros((npts,))
        u_numerical = model.solve_steady(initial_guess, tol=1e-12, maxiter=50)

        bkd.assert_allclose(u_numerical, u_exact, atol=1e-10)


class TestFluxRobinSolve(Generic[Array], unittest.TestCase):
    """Integration tests for flux Robin BCs."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_1d_flux_robin_solve(self):
        """1D diffusion: flux Robin left, Dirichlet right.

        u = x^2 + x, D = 1
        flux = -du/dx = -(2x+1)
        At x=-1: u=0, flux=1, flux.n=1*(-1)=-1
        Robin: alpha*u + beta*flux.n = g with alpha=2, beta=1
        g = 2*0 + 1*(-1) = -1
        At x=+1: u=2 (Dirichlet)
        """
        bkd = self.bkd()
        npts = 10
        tmesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(tmesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="x**2 + x",
            nvars=1,
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes().reshape(1, -1)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, forcing=lambda t: forcing
        )

        # Left: flux Robin, alpha=2, beta=1
        left_idx = umesh.boundary_indices(0)
        left_normals = tmesh.boundary_normals(0)
        robin_val = man_sol.robin_values(
            tmesh.points()[:, left_idx], left_normals,
            alpha=2.0, beta=1.0, convention="flux"
        )
        bc_left = flux_robin_bc(
            bkd, left_idx, left_normals, physics, 2.0, 1.0, robin_val
        )

        # Right: Dirichlet
        right_idx = umesh.boundary_indices(1)
        bc_right = constant_dirichlet_bc(bkd, right_idx, float(u_exact[right_idx[0]]))

        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)
        initial_guess = bkd.zeros((npts,))
        u_numerical = model.solve_steady(initial_guess, tol=1e-12, maxiter=50)

        bkd.assert_allclose(u_numerical, u_exact, atol=1e-10)

    def test_1d_flux_robin_with_velocity(self):
        """1D ADR: flux Robin left, Dirichlet right.

        u = x^2, D = 1, v = 3
        flux = -du/dx + v*u = -2x + 3x^2
        At x=-1: u=1, flux=2+3=5, flux.n=5*(-1)=-5
        Robin: alpha*u + beta*flux.n = g with alpha=1, beta=1
        g = 1*1 + 1*(-5) = -4
        At x=+1: u=1 (Dirichlet)
        """
        bkd = self.bkd()
        npts = 10
        tmesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(tmesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="x**2",
            nvars=1,
            diff_str="1.0",
            react_str="0",
            vel_strs=["3"],
            bkd=bkd,
            oned=True,
            conservative=True,
        )

        nodes = basis.nodes().reshape(1, -1)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        velocity = [bkd.full((npts,), 3.0)]
        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, velocity=velocity,
            forcing=lambda t: forcing
        )

        # Left: flux Robin, alpha=1, beta=1
        left_idx = umesh.boundary_indices(0)
        left_normals = tmesh.boundary_normals(0)
        robin_val = man_sol.robin_values(
            tmesh.points()[:, left_idx], left_normals,
            alpha=1.0, beta=1.0, convention="flux"
        )
        bc_left = flux_robin_bc(
            bkd, left_idx, left_normals, physics, 1.0, 1.0, robin_val
        )

        # Right: Dirichlet
        right_idx = umesh.boundary_indices(1)
        bc_right = constant_dirichlet_bc(bkd, right_idx, float(u_exact[right_idx[0]]))

        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)
        initial_guess = bkd.zeros((npts,))
        u_numerical = model.solve_steady(initial_guess, tol=1e-12, maxiter=50)

        bkd.assert_allclose(u_numerical, u_exact, atol=1e-10)


class TestMixedBCSolve(Generic[Array], unittest.TestCase):
    """Integration tests for mixed Dirichlet + Robin/Neumann BCs."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def _dirichlet_bcs_on_sides(self, tmesh, u_exact, sides):
        """Create Dirichlet BCs from exact solution on given sides."""
        bkd = self.bkd()
        bcs = []
        for side in sides:
            idx = tmesh.boundary_indices(side)
            vals = u_exact[idx]
            bcs.append(DirichletBC(bkd, idx, lambda t, v=vals: v))
        return bcs

    def _solve_2d_mixed_robin(self, convention, diffusion):
        """Solve 2D diffusion with Robin on left, Dirichlet on other 3."""
        bkd = self.bkd()
        npts = 12
        tmesh = TransformedMesh2D(npts, npts, bkd)
        basis = ChebyshevBasis2D(tmesh, bkd)

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="x**2 + y**2 + x*y", nvars=2,
            diff_str=str(diffusion), react_str="u",
            vel_strs=["0", "0"], bkd=bkd, oned=True,
        )

        pts = tmesh.points()
        u_exact = man_sol.functions["solution"](pts)
        forcing = man_sol.functions["forcing"](pts)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=diffusion, reaction=1.0,
            forcing=lambda t: forcing,
        )

        left_idx = tmesh.boundary_indices(0)
        left_normals = tmesh.boundary_normals(0)
        left_pts = pts[:, left_idx]
        robin_val = man_sol.robin_values(
            left_pts, left_normals, 1.0, 1.0, convention=convention,
        )

        if convention == "gradient":
            Dx = basis.derivative_matrix(1, 0)
            Dy = basis.derivative_matrix(1, 1)
            bc_left = gradient_robin_bc(
                bkd, left_idx, left_normals, [Dx, Dy], 1.0, 1.0, robin_val,
            )
        else:
            bc_left = flux_robin_bc(
                bkd, left_idx, left_normals, physics, 1.0, 1.0, robin_val,
            )

        bcs = [bc_left] + self._dirichlet_bcs_on_sides(tmesh, u_exact, [1, 2, 3])
        physics.set_boundary_conditions(bcs)

        model = CollocationModel(physics, bkd)
        u_numerical = model.solve_steady(
            bkd.zeros((npts * npts,)), tol=1e-10, maxiter=50,
        )
        bkd.assert_allclose(u_numerical, u_exact, atol=1e-8)

    def test_mixed_dirichlet_gradient_robin(self):
        """2D: 3 Dirichlet + 1 gradient Robin, D=1, reaction=u."""
        self._solve_2d_mixed_robin("gradient", 1.0)

    def test_mixed_dirichlet_flux_robin(self):
        """2D: 3 Dirichlet + 1 flux Robin, D=2, reaction=u."""
        self._solve_2d_mixed_robin("flux", 2.0)

    def test_variable_diffusion_flux_neumann(self):
        """1D with D(x) = 1+x^2, flux Neumann left, Dirichlet right."""
        bkd = self.bkd()
        npts = 15
        tmesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(tmesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="x**2 + 1", nvars=1, diff_str="1 + x**2",
            react_str="0", vel_strs=["0"], bkd=bkd, oned=True,
        )

        nodes = basis.nodes().reshape(1, -1)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0 + basis.nodes() ** 2,
            forcing=lambda t: forcing,
        )

        left_idx = umesh.boundary_indices(0)
        left_normals = tmesh.boundary_normals(0)
        neumann_val = man_sol.neumann_values(
            tmesh.points()[:, left_idx], left_normals, convention="flux",
        )
        bc_left = flux_neumann_bc(
            bkd, left_idx, left_normals, physics, neumann_val,
        )

        right_idx = umesh.boundary_indices(1)
        bc_right = constant_dirichlet_bc(
            bkd, right_idx, float(u_exact[right_idx[0]]),
        )

        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)
        u_numerical = model.solve_steady(
            bkd.zeros((npts,)), tol=1e-12, maxiter=50,
        )
        bkd.assert_allclose(u_numerical, u_exact, atol=1e-9)


class TestTransientRobinSolve(Generic[Array], unittest.TestCase):
    """Integration tests for time-dependent Robin BCs."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def _solve_1d_transient_robin(self, convention, diffusion, velocity_val):
        """Solve 1D transient with Robin left, Dirichlet right.

        u = (x^2+1)*(1+T), linear in time => backward Euler exact.
        """
        bkd = self.bkd()
        npts = 15
        tmesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(tmesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        vel_strs = [str(velocity_val)]
        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="(x**2 + 1)*(1 + T)", nvars=1,
            diff_str=str(diffusion), react_str="u",
            vel_strs=vel_strs, bkd=bkd, oned=True,
            conservative=(velocity_val != 0),
        )

        pts = tmesh.points()

        def forcing_fn(t):
            return man_sol.functions["forcing"](pts, t)

        velocity = [bkd.full((npts,), float(velocity_val))] if velocity_val else None
        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=diffusion, reaction=1.0,
            velocity=velocity, forcing=forcing_fn,
        )

        # Left: time-dependent Robin, alpha=1, beta=1
        left_idx = umesh.boundary_indices(0)
        left_normals = tmesh.boundary_normals(0)
        left_pts = pts[:, left_idx]

        def robin_values_fn(t, lp=left_pts, ln=left_normals, ms=man_sol, c=convention):
            return ms.robin_values(lp, ln, 1.0, 1.0, time=t, convention=c)

        if convention == "gradient":
            bc_left = gradient_robin_bc(
                bkd, left_idx, left_normals,
                [basis.derivative_matrix()], 1.0, 1.0, robin_values_fn,
            )
        else:
            bc_left = flux_robin_bc(
                bkd, left_idx, left_normals, physics, 1.0, 1.0, robin_values_fn,
            )

        # Right: time-dependent Dirichlet
        right_idx = umesh.boundary_indices(1)
        right_pts = pts[:, right_idx]

        def right_dirichlet_fn(t, rp=right_pts, ms=man_sol):
            return ms.functions["solution"](rp, t)

        bc_right = DirichletBC(bkd, right_idx, right_dirichlet_fn)
        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)
        u_initial = man_sol.functions["solution"](pts, 0.0)
        config = TimeIntegrationConfig(
            method="backward_euler", final_time=1.0, deltat=0.25,
        )

        solutions, times = model.solve_transient(u_initial, config)
        t_final = float(bkd.to_numpy(times[-1]))
        u_exact_final = man_sol.functions["solution"](pts, t_final)
        bkd.assert_allclose(solutions[:, -1], u_exact_final, atol=1e-8)

    def test_transient_gradient_robin(self):
        """1D transient gradient Robin, D=1, no velocity."""
        self._solve_1d_transient_robin("gradient", 1.0, 0)

    def test_transient_flux_robin(self):
        """1D transient flux Robin, D=2, v=1."""
        self._solve_1d_transient_robin("flux", 2.0, 1)


class TestCurvilinearRobinSolve(Generic[Array], unittest.TestCase):
    """Integration tests for Robin BCs on curvilinear meshes."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_polar_gradient_robin_solve(self):
        """Polar domain r in [1,2], theta in [-pi/2,pi/2].

        u = x^2*y^2, D=1, reaction=u.
        Robin on r_min (varying normals), Dirichlet on other 3 sides.
        """
        bkd = self.bkd()
        npts = 25
        transform = PolarTransform(
            r_bounds=(1.0, 2.0),
            theta_bounds=(-math.pi / 2, math.pi / 2),
            bkd=bkd,
        )
        tmesh = TransformedMesh2D(npts, npts, bkd, transform)
        basis = ChebyshevBasis2D(tmesh, bkd)

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="x**2*y**2", nvars=2, diff_str="1.0",
            react_str="u", vel_strs=["0", "0"], bkd=bkd, oned=True,
        )

        pts = tmesh.points()
        u_exact = man_sol.functions["solution"](pts)
        forcing = man_sol.functions["forcing"](pts)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, reaction=1.0,
            forcing=lambda t: forcing,
        )

        Dx = basis.derivative_matrix(1, 0)
        Dy = basis.derivative_matrix(1, 1)

        # r_min (side 0): gradient Robin with varying normals
        left_idx = tmesh.boundary_indices(0)
        left_normals = tmesh.boundary_normals(0)
        robin_val = man_sol.robin_values(
            pts[:, left_idx], left_normals, 1.0, 1.0, convention="gradient",
        )
        bc_left = gradient_robin_bc(
            bkd, left_idx, left_normals, [Dx, Dy], 1.0, 1.0, robin_val,
        )

        # Other 3 sides: Dirichlet
        bcs = [bc_left]
        for side in range(1, 4):
            idx = tmesh.boundary_indices(side)
            vals = u_exact[idx]
            bcs.append(DirichletBC(bkd, idx, lambda t, v=vals: v))

        physics.set_boundary_conditions(bcs)

        model = CollocationModel(physics, bkd)
        u_numerical = model.solve_steady(
            bkd.zeros((npts * npts,)), tol=1e-10, maxiter=50,
        )
        bkd.assert_allclose(u_numerical, u_exact, atol=1e-6)


class TestElasticityTractionBC(Generic[Array], unittest.TestCase):
    """Integration tests for elasticity traction Robin/Neumann BCs."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def _setup_elasticity_problem(self):
        """Set up 2D elasticity problem with solution nonzero on left boundary.

        Solution: u = [x^2*(1-y^2), x*y*(1-y^2)]
        At x=-1 (left): u_x = 1-y^2, u_y = -y*(1-y^2) -- nonzero
        At y=+/-1 (top/bottom): u = 0 -- simplifies Dirichlet
        """
        bkd = self.bkd()
        npts = 10
        tmesh = TransformedMesh2D(npts, npts, bkd)
        basis = ChebyshevBasis2D(tmesh, bkd)
        umesh = create_uniform_mesh_2d(
            (npts, npts), (-1.0, 1.0, -1.0, 1.0), bkd
        )

        lamda, mu = 1.0, 1.0
        man_sol = ManufacturedLinearElasticityEquations(
            sol_strs=["x**2*(1 - y**2)", "x*y*(1 - y**2)"],
            nvars=2,
            lambda_str=str(lamda),
            mu_str=str(mu),
            bkd=bkd,
            oned=True,
        )

        pts = tmesh.points()
        u_exact = man_sol.functions["solution"](pts)  # (npts_total, 2)
        forcing = man_sol.functions["forcing"](pts)    # (npts_total, 2)
        npts_total = basis.npts()
        u_exact_flat = bkd.concatenate([u_exact[:, 0], u_exact[:, 1]])
        forcing_flat = bkd.concatenate([forcing[:, 0], forcing[:, 1]])

        physics = LinearElasticityPhysics(
            basis, bkd, lamda=lamda, mu=mu, forcing=lambda t: forcing_flat
        )

        Dx = basis.derivative_matrix(1, 0)
        Dy = basis.derivative_matrix(1, 1)

        return {
            "bkd": bkd, "tmesh": tmesh, "basis": basis, "umesh": umesh,
            "man_sol": man_sol, "pts": pts, "u_exact_flat": u_exact_flat,
            "physics": physics, "npts_total": npts_total,
            "lamda": lamda, "mu": mu, "D_list": [Dx, Dy],
        }

    def _dirichlet_bcs_for_elasticity(self, umesh, u_exact_flat, npts, sides):
        """Create Dirichlet BCs for both components on given sides."""
        bkd = self.bkd()
        bcs = []
        for side in sides:
            boundary_idx = umesh.boundary_indices(side)
            # u-component
            vals_u = u_exact_flat[boundary_idx]
            bcs.append(DirichletBC(bkd, boundary_idx, vals_u))
            # v-component (offset by npts)
            boundary_idx_v = boundary_idx + npts
            vals_v = u_exact_flat[boundary_idx_v]
            bcs.append(DirichletBC(bkd, boundary_idx_v, vals_v))
        return bcs

    def test_elasticity_traction_neumann(self):
        """Traction Neumann on left, Dirichlet on other 3 sides.

        Tests pure traction (alpha=0, beta=1) with nonzero displacement
        at the Robin boundary.
        """
        s = self._setup_elasticity_problem()
        bkd = s["bkd"]

        left_idx = s["umesh"].boundary_indices(0)
        left_normals = s["tmesh"].boundary_normals(0)
        left_pts = s["pts"][:, left_idx]

        # Exact traction values at left boundary
        traction_vals = s["man_sol"].traction_values(left_pts, left_normals)

        # Traction Neumann for each component
        bc_tx = traction_neumann_bc(
            bkd, left_idx, left_normals, s["D_list"],
            s["lamda"], s["mu"], s["npts_total"], 0, traction_vals[:, 0]
        )
        bc_ty = traction_neumann_bc(
            bkd, left_idx, left_normals, s["D_list"],
            s["lamda"], s["mu"], s["npts_total"], 1, traction_vals[:, 1]
        )

        # Dirichlet on sides 1, 2, 3
        bcs = [bc_tx, bc_ty] + self._dirichlet_bcs_for_elasticity(
            s["umesh"], s["u_exact_flat"], s["npts_total"], [1, 2, 3]
        )
        s["physics"].set_boundary_conditions(bcs)

        model = CollocationModel(s["physics"], bkd)
        u_numerical = model.solve_steady(
            bkd.zeros((2 * s["npts_total"],)), tol=1e-10, maxiter=50
        )
        bkd.assert_allclose(u_numerical, s["u_exact_flat"], atol=1e-8)

    def test_elasticity_traction_robin(self):
        """Traction Robin (alpha=1, beta=1) on left, Dirichlet on other 3.

        Tests full Robin with nonzero alpha*u contribution at boundary.
        """
        s = self._setup_elasticity_problem()
        bkd = s["bkd"]

        alpha, beta = 1.0, 1.0
        left_idx = s["umesh"].boundary_indices(0)
        left_normals = s["tmesh"].boundary_normals(0)
        left_pts = s["pts"][:, left_idx]

        # Exact Robin values at left boundary
        robin_vals = s["man_sol"].robin_values(
            left_pts, left_normals, alpha, beta
        )

        # Traction Robin for each component
        bc_rx = traction_robin_bc(
            bkd, left_idx, left_normals, s["D_list"],
            s["lamda"], s["mu"], s["npts_total"], 0,
            alpha, beta, robin_vals[:, 0]
        )
        bc_ry = traction_robin_bc(
            bkd, left_idx, left_normals, s["D_list"],
            s["lamda"], s["mu"], s["npts_total"], 1,
            alpha, beta, robin_vals[:, 1]
        )

        # Dirichlet on sides 1, 2, 3
        bcs = [bc_rx, bc_ry] + self._dirichlet_bcs_for_elasticity(
            s["umesh"], s["u_exact_flat"], s["npts_total"], [1, 2, 3]
        )
        s["physics"].set_boundary_conditions(bcs)

        model = CollocationModel(s["physics"], bkd)
        u_numerical = model.solve_steady(
            bkd.zeros((2 * s["npts_total"],)), tol=1e-10, maxiter=50
        )
        bkd.assert_allclose(u_numerical, s["u_exact_flat"], atol=1e-8)


# NumPy backend
class TestGradientNeumannSolveNumpy(TestGradientNeumannSolve[NDArray[Any]]):
    __test__ = True
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGradientRobinSolveNumpy(TestGradientRobinSolve[NDArray[Any]]):
    __test__ = True
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFluxNeumannSolveNumpy(TestFluxNeumannSolve[NDArray[Any]]):
    __test__ = True
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFluxRobinSolveNumpy(TestFluxRobinSolve[NDArray[Any]]):
    __test__ = True
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMixedBCSolveNumpy(TestMixedBCSolve[NDArray[Any]]):
    __test__ = True
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTransientRobinSolveNumpy(TestTransientRobinSolve[NDArray[Any]]):
    __test__ = True
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCurvilinearRobinSolveNumpy(TestCurvilinearRobinSolve[NDArray[Any]]):
    __test__ = True
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestElasticityTractionBCNumpy(TestElasticityTractionBC[NDArray[Any]]):
    __test__ = True
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# Torch backend
class TestGradientNeumannSolveTorch(TestGradientNeumannSolve[torch.Tensor]):
    __test__ = True
    def bkd(self) -> TorchBkd:
        return TorchBkd()
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


class TestGradientRobinSolveTorch(TestGradientRobinSolve[torch.Tensor]):
    __test__ = True
    def bkd(self) -> TorchBkd:
        return TorchBkd()
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


class TestFluxNeumannSolveTorch(TestFluxNeumannSolve[torch.Tensor]):
    __test__ = True
    def bkd(self) -> TorchBkd:
        return TorchBkd()
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


class TestFluxRobinSolveTorch(TestFluxRobinSolve[torch.Tensor]):
    __test__ = True
    def bkd(self) -> TorchBkd:
        return TorchBkd()
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


class TestMixedBCSolveTorch(TestMixedBCSolve[torch.Tensor]):
    __test__ = True
    def bkd(self) -> TorchBkd:
        return TorchBkd()
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


class TestTransientRobinSolveTorch(TestTransientRobinSolve[torch.Tensor]):
    __test__ = True
    def bkd(self) -> TorchBkd:
        return TorchBkd()
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


class TestCurvilinearRobinSolveTorch(TestCurvilinearRobinSolve[torch.Tensor]):
    __test__ = True
    def bkd(self) -> TorchBkd:
        return TorchBkd()
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


class TestElasticityTractionBCTorch(TestElasticityTractionBC[torch.Tensor]):
    __test__ = True
    def bkd(self) -> TorchBkd:
        return TorchBkd()
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


if __name__ == "__main__":
    unittest.main()
