"""Integration tests for Robin/Neumann BCs with full PDE solves.

Tests gradient and flux conventions with manufactured solutions,
verifying that the numerical solution matches the exact solution.
"""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.pde.collocation.basis import (
    ChebyshevBasis1D,
    ChebyshevBasis2D,
)
from pyapprox.typing.pde.collocation.mesh import (
    create_uniform_mesh_1d,
    TransformedMesh1D,
    TransformedMesh2D,
)
from pyapprox.typing.pde.collocation.boundary import (
    constant_dirichlet_bc,
    gradient_neumann_bc,
    gradient_robin_bc,
    flux_neumann_bc,
    flux_robin_bc,
)
from pyapprox.typing.pde.collocation.physics import AdvectionDiffusionReaction
from pyapprox.typing.pde.collocation.time_integration import CollocationModel
from pyapprox.typing.pde.collocation.manufactured_solutions import (
    ManufacturedAdvectionDiffusionReaction,
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
            from pyapprox.typing.pde.collocation.boundary import DirichletBC
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


if __name__ == "__main__":
    unittest.main()
