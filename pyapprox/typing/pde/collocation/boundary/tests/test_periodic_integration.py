"""Integration tests for periodic BCs on curvilinear meshes.

Tests PeriodicBC with:
1. Rectangular periodic domain via AffineTransform2D
2. Annular θ-periodic domain via PolarTransform
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
from pyapprox.typing.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.typing.pde.collocation.mesh import TransformedMesh2D
from pyapprox.typing.pde.collocation.mesh.transforms.affine import AffineTransform2D
from pyapprox.typing.pde.collocation.mesh.transforms.polar import PolarTransform
from pyapprox.typing.pde.collocation.boundary import DirichletBC, PeriodicBC
from pyapprox.typing.pde.collocation.physics import AdvectionDiffusionReaction
from pyapprox.typing.pde.collocation.time_integration import CollocationModel
from pyapprox.typing.pde.collocation.manufactured_solutions import (
    ManufacturedAdvectionDiffusionReaction,
)


class TestPeriodicBCTransforms(Generic[Array], unittest.TestCase):
    """Integration tests for periodic BCs on transformed domains."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_rectangular_periodic_adr(self):
        """Full periodic ADR on [0, 2π]².

        u = cos(x)*cos(y), D = 1, reaction = 3*u
        Solution is naturally 2π-periodic in both x and y.
        Forcing f = -u (non-trivial), so u=0 is not a solution.
        """
        bkd = self.bkd()
        npts = 20
        transform = AffineTransform2D(
            (0.0, 2 * math.pi, 0.0, 2 * math.pi), bkd
        )
        mesh = TransformedMesh2D(npts, npts, bkd, transform)
        basis = ChebyshevBasis2D(mesh, bkd)

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="cos(x)*cos(y)",
            nvars=2,
            diff_str="1.0",
            react_str="3*u",
            vel_strs=["0", "0"],
            bkd=bkd,
            oned=True,
        )

        pts = mesh.points()
        u_exact = man_sol.functions["solution"](pts)
        forcing = man_sol.functions["forcing"](pts)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, reaction=3.0, forcing=lambda t: forcing
        )

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bottom_idx = mesh.boundary_indices(2)
        top_idx = mesh.boundary_indices(3)

        Dx = basis.derivative_matrix(1, 0)
        Dy = basis.derivative_matrix(1, 1)

        bc_x = PeriodicBC(bkd, left_idx, right_idx, Dx)
        bc_y = PeriodicBC(bkd, bottom_idx, top_idx, Dy)
        physics.set_boundary_conditions([bc_x, bc_y])

        model = CollocationModel(physics, bkd)
        initial_guess = bkd.zeros((npts * npts,))
        u_numerical = model.solve_steady(initial_guess, tol=1e-10, maxiter=50)

        bkd.assert_allclose(u_numerical, u_exact, atol=1e-6)

    def test_annular_periodic_theta(self):
        """θ-periodic annular domain, r ∈ [1, 2], θ ∈ [0, 2π].

        u = x = r*cos(θ), D = 1, reaction = u
        Dirichlet at r_min and r_max, periodic in θ.
        """
        bkd = self.bkd()
        npts_r, npts_theta = 20, 30
        transform = PolarTransform(
            r_bounds=(1.0, 2.0),
            theta_bounds=(0.0, 2 * math.pi),
            bkd=bkd,
        )
        mesh = TransformedMesh2D(npts_r, npts_theta, bkd, transform)
        basis = ChebyshevBasis2D(mesh, bkd)

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="x",
            nvars=2,
            diff_str="1.0",
            react_str="u",
            vel_strs=["0", "0"],
            bkd=bkd,
            oned=True,
        )

        pts = mesh.points()
        u_exact = man_sol.functions["solution"](pts)
        forcing = man_sol.functions["forcing"](pts)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, reaction=1.0, forcing=lambda t: forcing
        )

        # r boundaries: Dirichlet from manufactured solution
        r_min_idx = mesh.boundary_indices(0)
        r_max_idx = mesh.boundary_indices(1)
        bc_r_min = DirichletBC(bkd, r_min_idx, u_exact[r_min_idx])
        bc_r_max = DirichletBC(bkd, r_max_idx, u_exact[r_max_idx])

        # θ boundaries: periodic (θ is reference dim 1 = y direction)
        theta_min_idx = mesh.boundary_indices(2)
        theta_max_idx = mesh.boundary_indices(3)
        Dy = basis.derivative_matrix(1, 1)
        bc_theta = PeriodicBC(bkd, theta_min_idx, theta_max_idx, Dy)

        physics.set_boundary_conditions([bc_r_min, bc_r_max, bc_theta])

        model = CollocationModel(physics, bkd)
        initial_guess = bkd.zeros((npts_r * npts_theta,))
        u_numerical = model.solve_steady(initial_guess, tol=1e-8, maxiter=50)

        bkd.assert_allclose(u_numerical, u_exact, atol=1e-4)


# NumPy backend
class TestPeriodicBCTransformsNumpy(TestPeriodicBCTransforms[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# Torch backend
class TestPeriodicBCTransformsTorch(TestPeriodicBCTransforms[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


if __name__ == "__main__":
    unittest.main()
