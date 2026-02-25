"""Parameterized tests for Reaction-Diffusion manufactured solutions with physics.

Verifies:
1. Residual = 0 at exact solution (using polynomial solutions for machine precision)
2. Jacobian correctness via DerivativeChecker
3. Multiple test cases via parameterization

The key design:
- Same reaction object is used by both physics and manufactured solution
- This ensures consistent forcing computation and residual evaluation
"""

import unittest
from typing import Any, Generic, Tuple

import sympy as sp
import torch
from numpy.typing import NDArray
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.mesh import (
    create_uniform_mesh_1d,
    TransformedMesh1D,
)
from pyapprox.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.pde.collocation.physics import (
    TwoSpeciesReactionDiffusionPhysics,
    LinearReaction,
    FitzHughNagumoReaction,
)
from pyapprox.pde.collocation.manufactured_solutions import (
    ManufacturedTwoSpeciesReactionDiffusion,
)
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)


class PhysicsDerivativeWrapper(Generic[Array]):
    """Wrapper to adapt physics interface for DerivativeChecker.

    DerivativeChecker expects:
    - bkd() method
    - nvars() method
    - nqoi() method
    - __call__(samples) for batch evaluation returning (nqoi, nsamples)
    - jacobian(sample) for single sample returning (nqoi, nvars)
    """

    def __init__(self, physics, time=0.0):
        self._physics = physics
        self._time = time
        self._backend = physics._bkd

    def bkd(self):
        return self._backend

    def nvars(self):
        return self._physics.npts() * 2  # Two species

    def nqoi(self):
        return self._physics.npts() * 2  # Two species

    def __call__(self, samples):
        # samples shape: (nvars, nsamples), return (nqoi, nsamples)
        if samples.ndim == 2:
            return self._backend.stack(
                [self._physics.residual(samples[:, i], self._time)
                 for i in range(samples.shape[1])],
                axis=1
            )
        # Single sample: return (nqoi, 1)
        return self._physics.residual(samples, self._time).reshape(-1, 1)

    def jacobian(self, sample):
        # sample shape: (nvars, 1), return (nqoi, nvars)
        if sample.ndim == 2:
            sample = sample[:, 0]
        return self._physics.jacobian(sample, self._time)


class TestManufacturedReactionDiffusion1D(Generic[Array], unittest.TestCase):
    """Test 1D reaction-diffusion physics with manufactured solutions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_linear_reaction_residual(self):
        """Test reaction-diffusion residual with linear reaction."""
        bkd = self.bkd()
        npts = 15
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        # Linear reaction: R0 = a00*u0 + a01*u1, R1 = a10*u0 + a11*u1
        reaction = LinearReaction(
            a00=1.0, a01=-0.5, a10=0.5, a11=-1.0, bkd=bkd
        )

        # Create manufactured solution with same reaction
        man_sol = ManufacturedTwoSpeciesReactionDiffusion(
            sol_strs=["(1 - x**2)", "0.5*(1 - x**2)"],
            nvars=1,
            diff_strs=["1.0", "0.5"],
            reaction=reaction,
            bkd=bkd,
            oned=True,
        )

        # Get manufactured solution values
        u_exact = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))

        # Unpack: forcing is (npts, 2) with oned=True
        forcing0 = forcing[:, 0]
        forcing1 = forcing[:, 1]

        # Create physics with manufactured forcing
        physics = TwoSpeciesReactionDiffusionPhysics(
            basis, bkd,
            diffusion0=1.0, diffusion1=0.5,
            reaction=reaction,
            forcing0=lambda t: forcing0,
            forcing1=lambda t: forcing1,
        )

        # Set Dirichlet BCs at boundaries
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bcs = [
            # u0 BCs
            zero_dirichlet_bc(bkd, left_idx),
            zero_dirichlet_bc(bkd, right_idx),
            # u1 BCs (shifted by npts)
            zero_dirichlet_bc(bkd, left_idx + npts),
            zero_dirichlet_bc(bkd, right_idx + npts),
        ]
        physics.set_boundary_conditions(bcs)

        # Combine exact solution into state vector [u0, u1]
        state_exact = bkd.hstack([u_exact[:, 0], u_exact[:, 1]])

        residual = physics.residual(state_exact, 0.0)
        jac = physics.jacobian(state_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, jac, state_exact, 0.0
        )

        # Check interior residual only (boundaries are modified by BCs)
        boundary_indices = {0, npts - 1, npts, 2 * npts - 1}
        interior = [i for i in range(2 * npts) if i not in boundary_indices]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior])

        bkd.assert_allclose(
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-10
        )

    def test_linear_reaction_jacobian(self):
        """Test reaction-diffusion Jacobian via derivative checker."""
        bkd = self.bkd()
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        reaction = LinearReaction(
            a00=1.0, a01=-0.5, a10=0.5, a11=-1.0, bkd=bkd
        )

        man_sol = ManufacturedTwoSpeciesReactionDiffusion(
            sol_strs=["(1 - x**2)", "0.5*(1 - x**2)"],
            nvars=1,
            diff_strs=["1.0", "0.5"],
            reaction=reaction,
            bkd=bkd,
            oned=True,
        )

        u_exact = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))

        physics = TwoSpeciesReactionDiffusionPhysics(
            basis, bkd,
            diffusion0=1.0, diffusion1=0.5,
            reaction=reaction,
            forcing0=lambda t: forcing[:, 0],
            forcing1=lambda t: forcing[:, 1],
        )

        state_exact = bkd.hstack([u_exact[:, 0], u_exact[:, 1]])
        wrapper = PhysicsDerivativeWrapper(physics)
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(state_exact.reshape(-1, 1))
        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-6)

    def test_fitzhugh_nagumo_residual(self):
        """Test reaction-diffusion residual with FitzHugh-Nagumo reaction."""
        bkd = self.bkd()
        npts = 15
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        # FitzHugh-Nagumo reaction
        reaction = FitzHughNagumoReaction(
            alpha=0.1, eps=0.01, beta=0.5, gamma=1.0, bkd=bkd
        )

        # Solution must stay in reasonable range for FHN
        man_sol = ManufacturedTwoSpeciesReactionDiffusion(
            sol_strs=["0.5*(1 - x**2)", "0.3*(1 - x**2)"],
            nvars=1,
            diff_strs=["0.01", "0.0"],  # FHN typically has D0 > 0, D1 = 0
            reaction=reaction,
            bkd=bkd,
            oned=True,
        )

        u_exact = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))

        physics = TwoSpeciesReactionDiffusionPhysics(
            basis, bkd,
            diffusion0=0.01, diffusion1=0.0,
            reaction=reaction,
            forcing0=lambda t: forcing[:, 0],
            forcing1=lambda t: forcing[:, 1],
        )

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bcs = [
            zero_dirichlet_bc(bkd, left_idx),
            zero_dirichlet_bc(bkd, right_idx),
            zero_dirichlet_bc(bkd, left_idx + npts),
            zero_dirichlet_bc(bkd, right_idx + npts),
        ]
        physics.set_boundary_conditions(bcs)

        state_exact = bkd.hstack([u_exact[:, 0], u_exact[:, 1]])
        residual = physics.residual(state_exact, 0.0)
        jac = physics.jacobian(state_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, jac, state_exact, 0.0
        )

        boundary_indices = {0, npts - 1, npts, 2 * npts - 1}
        interior = [i for i in range(2 * npts) if i not in boundary_indices]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior])

        bkd.assert_allclose(
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-10
        )

    def test_fitzhugh_nagumo_jacobian(self):
        """Test FitzHugh-Nagumo Jacobian via derivative checker."""
        bkd = self.bkd()
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        reaction = FitzHughNagumoReaction(
            alpha=0.1, eps=0.01, beta=0.5, gamma=1.0, bkd=bkd
        )

        man_sol = ManufacturedTwoSpeciesReactionDiffusion(
            sol_strs=["0.5*(1 - x**2)", "0.3*(1 - x**2)"],
            nvars=1,
            diff_strs=["0.01", "0.0"],
            reaction=reaction,
            bkd=bkd,
            oned=True,
        )

        u_exact = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))

        physics = TwoSpeciesReactionDiffusionPhysics(
            basis, bkd,
            diffusion0=0.01, diffusion1=0.0,
            reaction=reaction,
            forcing0=lambda t: forcing[:, 0],
            forcing1=lambda t: forcing[:, 1],
        )

        state_exact = bkd.hstack([u_exact[:, 0], u_exact[:, 1]])
        wrapper = PhysicsDerivativeWrapper(physics)
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(state_exact.reshape(-1, 1))
        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-6)


class TestReactionDiffusion1DParameterized(ParametrizedTestCase):
    """Parameterized 1D reaction-diffusion residual tests."""

    def bkd(self):
        return NumpyBkd()

    @parametrize(
        "name,sol0_str,sol1_str,a00,a01,a10,a11,D0,D1,npts",
        [
            ("linear_basic", "(1 - x**2)", "0.5*(1 - x**2)",
             1.0, -0.5, 0.5, -1.0, 1.0, 0.5, 15),
            ("linear_symmetric", "(1 - x**2)", "(1 - x**2)",
             -1.0, 0.5, 0.5, -1.0, 1.0, 1.0, 15),
            ("linear_diagonal", "(1 - x**2)", "0.3*(1 - x**2)",
             -2.0, 0.0, 0.0, -1.0, 0.5, 0.2, 15),
            ("quartic_sol", "(1 - x**2)**2", "0.5*(1 - x**2)**2",
             1.0, -0.5, 0.5, -1.0, 1.0, 0.5, 20),
            ("higher_diffusion", "(1 - x**2)", "0.5*(1 - x**2)",
             1.0, -0.5, 0.5, -1.0, 5.0, 2.5, 15),
        ],
    )
    def test_linear_reaction_residual(
        self, name, sol0_str, sol1_str, a00, a01, a10, a11, D0, D1, npts
    ):
        """Test 1D linear reaction-diffusion residual for parameterized cases."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        reaction = LinearReaction(a00=a00, a01=a01, a10=a10, a11=a11, bkd=bkd)

        man_sol = ManufacturedTwoSpeciesReactionDiffusion(
            sol_strs=[sol0_str, sol1_str],
            nvars=1,
            diff_strs=[str(D0), str(D1)],
            reaction=reaction,
            bkd=bkd,
            oned=True,
        )

        u_exact = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))

        physics = TwoSpeciesReactionDiffusionPhysics(
            basis, bkd,
            diffusion0=D0, diffusion1=D1,
            reaction=reaction,
            forcing0=lambda t: forcing[:, 0],
            forcing1=lambda t: forcing[:, 1],
        )

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bcs = [
            zero_dirichlet_bc(bkd, left_idx),
            zero_dirichlet_bc(bkd, right_idx),
            zero_dirichlet_bc(bkd, left_idx + npts),
            zero_dirichlet_bc(bkd, right_idx + npts),
        ]
        physics.set_boundary_conditions(bcs)

        state_exact = bkd.hstack([u_exact[:, 0], u_exact[:, 1]])
        residual = physics.residual(state_exact, 0.0)
        jac = physics.jacobian(state_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, jac, state_exact, 0.0
        )

        boundary_indices = {0, npts - 1, npts, 2 * npts - 1}
        interior = [i for i in range(2 * npts) if i not in boundary_indices]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior])

        bkd.assert_allclose(
            interior_residual,
            bkd.zeros(interior_residual.shape),
            atol=1e-9,
        )

    @parametrize(
        "name,sol0_str,sol1_str,alpha,eps,beta,gamma,D0,npts",
        [
            ("fhn_basic", "0.5*(1 - x**2)", "0.3*(1 - x**2)",
             0.1, 0.01, 0.5, 1.0, 0.01, 15),
            ("fhn_high_alpha", "0.5*(1 - x**2)", "0.3*(1 - x**2)",
             0.3, 0.01, 0.5, 1.0, 0.01, 15),
            ("fhn_high_eps", "0.5*(1 - x**2)", "0.3*(1 - x**2)",
             0.1, 0.1, 0.5, 1.0, 0.01, 15),
            ("fhn_quartic", "0.3*(1 - x**2)**2", "0.2*(1 - x**2)**2",
             0.1, 0.01, 0.5, 1.0, 0.01, 20),
        ],
    )
    def test_fitzhugh_nagumo_residual(
        self, name, sol0_str, sol1_str, alpha, eps, beta, gamma, D0, npts
    ):
        """Test 1D FitzHugh-Nagumo residual for parameterized cases."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        reaction = FitzHughNagumoReaction(
            alpha=alpha, eps=eps, beta=beta, gamma=gamma, bkd=bkd
        )

        man_sol = ManufacturedTwoSpeciesReactionDiffusion(
            sol_strs=[sol0_str, sol1_str],
            nvars=1,
            diff_strs=[str(D0), "0.0"],
            reaction=reaction,
            bkd=bkd,
            oned=True,
        )

        u_exact = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))

        physics = TwoSpeciesReactionDiffusionPhysics(
            basis, bkd,
            diffusion0=D0, diffusion1=0.0,
            reaction=reaction,
            forcing0=lambda t: forcing[:, 0],
            forcing1=lambda t: forcing[:, 1],
        )

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bcs = [
            zero_dirichlet_bc(bkd, left_idx),
            zero_dirichlet_bc(bkd, right_idx),
            zero_dirichlet_bc(bkd, left_idx + npts),
            zero_dirichlet_bc(bkd, right_idx + npts),
        ]
        physics.set_boundary_conditions(bcs)

        state_exact = bkd.hstack([u_exact[:, 0], u_exact[:, 1]])
        residual = physics.residual(state_exact, 0.0)
        jac = physics.jacobian(state_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, jac, state_exact, 0.0
        )

        boundary_indices = {0, npts - 1, npts, 2 * npts - 1}
        interior = [i for i in range(2 * npts) if i not in boundary_indices]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior])

        bkd.assert_allclose(
            interior_residual,
            bkd.zeros(interior_residual.shape),
            atol=1e-9,
        )

    @parametrize(
        "name,a00,a01,a10,a11,npts",
        [
            ("jacobian_basic", 1.0, -0.5, 0.5, -1.0, 10),
            ("jacobian_symmetric", -1.0, 0.5, 0.5, -1.0, 10),
            ("jacobian_diagonal", -2.0, 0.0, 0.0, -1.0, 10),
        ],
    )
    def test_linear_reaction_jacobian(self, name, a00, a01, a10, a11, npts):
        """Test 1D linear reaction Jacobian via DerivativeChecker."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        reaction = LinearReaction(a00=a00, a01=a01, a10=a10, a11=a11, bkd=bkd)

        man_sol = ManufacturedTwoSpeciesReactionDiffusion(
            sol_strs=["(1 - x**2)", "0.5*(1 - x**2)"],
            nvars=1,
            diff_strs=["1.0", "0.5"],
            reaction=reaction,
            bkd=bkd,
            oned=True,
        )

        u_exact = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))

        physics = TwoSpeciesReactionDiffusionPhysics(
            basis, bkd,
            diffusion0=1.0, diffusion1=0.5,
            reaction=reaction,
            forcing0=lambda t: forcing[:, 0],
            forcing1=lambda t: forcing[:, 1],
        )

        state_exact = bkd.hstack([u_exact[:, 0], u_exact[:, 1]])
        wrapper = PhysicsDerivativeWrapper(physics)
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(state_exact.reshape(-1, 1))
        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-6)


# Concrete backend implementations
class TestManufacturedReactionDiffusion1DNumpy(
    TestManufacturedReactionDiffusion1D[NDArray[Any]]
):
    """NumPy backend tests for 1D reaction-diffusion."""

    __test__ = True

    def bkd(self):
        return NumpyBkd()


class TestManufacturedReactionDiffusion1DTorch(
    TestManufacturedReactionDiffusion1D[torch.Tensor]
):
    """PyTorch backend tests for 1D reaction-diffusion."""

    __test__ = True

    def bkd(self):
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


if __name__ == "__main__":
    unittest.main()
