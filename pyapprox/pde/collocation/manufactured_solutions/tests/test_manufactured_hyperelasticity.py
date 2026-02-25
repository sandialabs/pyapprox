"""Tests for hyperelasticity physics with manufactured solutions.

Verifies:
1. Residual = 0 at exact solution (1D, 2D, 3D)
2. Jacobian correctness via DerivativeChecker (1D, 2D)
3. Numerical solve recovers manufactured solution (1D, 2D)
4. Multiple test cases via parameterization
5. Dual backend support (NumPy and PyTorch)
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.pde.collocation.basis import (
    ChebyshevBasis1D,
    ChebyshevBasis2D,
    ChebyshevBasis3D,
)
from pyapprox.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.pde.collocation.manufactured_solutions.hyperelasticity import (
    ManufacturedHyperelasticityEquations,
)
from pyapprox.pde.collocation.mesh import (
    AffineTransform1D,
    AffineTransform2D,
    AffineTransform3D,
    TransformedMesh1D,
    TransformedMesh2D,
    TransformedMesh3D,
    create_uniform_mesh_1d,
    create_uniform_mesh_2d,
    create_uniform_mesh_3d,
)
from pyapprox.pde.collocation.physics.hyperelasticity import (
    HyperelasticityPhysics,
)
from pyapprox.pde.collocation.physics.stress_models import (
    NeoHookeanStress,
)
from pyapprox.pde.collocation.time_integration import CollocationModel
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class PhysicsDerivativeWrapper(Generic[Array]):
    """Adapt physics interface for DerivativeChecker."""

    def __init__(self, physics, time=0.0):
        self._physics = physics
        self._time = time
        self._backend = physics._bkd

    def bkd(self):
        return self._backend

    def nvars(self):
        return self._physics.nstates()

    def nqoi(self):
        return self._physics.nstates()

    def __call__(self, samples):
        if samples.ndim == 2:
            return self._backend.stack(
                [
                    self._physics.residual(samples[:, i], self._time)
                    for i in range(samples.shape[1])
                ],
                axis=1,
            )
        return self._physics.residual(samples, self._time).reshape(-1, 1)

    def jacobian(self, sample):
        if sample.ndim == 2:
            sample = sample[:, 0]
        return self._physics.jacobian(sample, self._time)


# ------------------------------------------------------------------
# Helper: create basis and mesh for [0,1]^d
# ------------------------------------------------------------------


def _setup_1d(npts, bkd):
    """Create 1D basis on [0,1] and return basis, mesh_bc, nodes."""
    transform = AffineTransform1D((0.0, 1.0), bkd)
    mesh = TransformedMesh1D(npts, bkd, transform)
    basis = ChebyshevBasis1D(mesh, bkd)
    mesh_bc = create_uniform_mesh_1d(npts, (0.0, 1.0), bkd)
    nodes = mesh.points()  # (1, npts)
    return basis, mesh_bc, nodes


def _setup_2d(npts_x, npts_y, bkd):
    """Create 2D basis on [0,1]^2 and return basis, mesh_bc, nodes."""
    transform = AffineTransform2D((0.0, 1.0, 0.0, 1.0), bkd)
    mesh = TransformedMesh2D(npts_x, npts_y, bkd, transform)
    basis = ChebyshevBasis2D(mesh, bkd)
    mesh_bc = create_uniform_mesh_2d((npts_x, npts_y), (0.0, 1.0, 0.0, 1.0), bkd)
    nodes = mesh.points()  # (2, npts)
    return basis, mesh_bc, nodes


def _setup_3d(npts_x, npts_y, npts_z, bkd):
    """Create 3D basis on [0,1]^3 and return basis, mesh_bc, nodes."""
    transform = AffineTransform3D((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), bkd)
    mesh = TransformedMesh3D(npts_x, npts_y, npts_z, bkd, transform)
    basis = ChebyshevBasis3D(mesh, bkd)
    mesh_bc = create_uniform_mesh_3d(
        (npts_x, npts_y, npts_z),
        (0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        bkd,
    )
    nodes = mesh.points()  # (3, npts)
    return basis, mesh_bc, nodes


def _apply_homogeneous_dirichlet_bcs(physics, mesh_bc, npts, ndim, bkd):
    """Apply zero Dirichlet BCs on all boundaries for all components."""
    nboundaries = mesh_bc.nboundaries()
    bcs = []
    for side in range(nboundaries):
        boundary_idx = mesh_bc.boundary_indices(side)
        for comp in range(ndim):
            bc_idx = bkd.asarray([int(idx) + comp * npts for idx in boundary_idx])
            bcs.append(zero_dirichlet_bc(bkd, bc_idx))
    physics.set_boundary_conditions(bcs)


def _get_interior_indices(mesh_bc, npts, ndim):
    """Get interior indices (exclude all boundary points for all components)."""
    nboundaries = mesh_bc.nboundaries()
    boundary_indices = set()
    for side in range(nboundaries):
        for idx in mesh_bc.boundary_indices(side):
            for comp in range(ndim):
                boundary_indices.add(int(idx) + comp * npts)
    return [i for i in range(ndim * npts) if i not in boundary_indices]


# ------------------------------------------------------------------
# 1D tests
# ------------------------------------------------------------------


class TestHyperelasticity1D(Generic[Array], unittest.TestCase):
    """Test 1D hyperelasticity with Neo-Hookean manufactured solutions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_residual_polynomial_1d(self):
        """Residual = 0 at exact polynomial solution.

        Note: Neo-Hookean forcing involves transcendental functions (log, 1/J),
        so spectral approximation error is larger than for polynomial forcings.
        Using npts=16 gives sufficient accuracy for atol=1e-8.
        """
        bkd = self.bkd()
        npts = 16

        stress = NeoHookeanStress(lamda=1.0, mu=1.0)
        basis, mesh_bc, nodes = _setup_1d(npts, bkd)

        man_sol = ManufacturedHyperelasticityEquations(
            sol_strs=["0.1*x**2*(1-x)**2"],
            nvars=1,
            stress_model=stress,
            bkd=bkd,
            oned=True,
        )

        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        # For 1D vector mixin: u_exact shape is (npts, 1), forcing is (npts, 1)
        u_exact_flat = u_exact[:, 0]
        forcing_flat = forcing[:, 0]

        physics = HyperelasticityPhysics(
            basis, bkd, stress, forcing=lambda t: forcing_flat
        )
        _apply_homogeneous_dirichlet_bcs(physics, mesh_bc, npts, 1, bkd)

        residual = physics.residual(u_exact_flat, 0.0)
        jacobian = physics.jacobian(u_exact_flat, 0.0)
        residual_bc, _ = physics.apply_boundary_conditions(
            residual, jacobian, u_exact_flat, 0.0
        )

        interior = _get_interior_indices(mesh_bc, npts, 1)
        interior_res = bkd.asarray([residual_bc[i] for i in interior])

        bkd.assert_allclose(interior_res, bkd.zeros(interior_res.shape), atol=1e-8)

    def test_jacobian_1d(self):
        """Jacobian correctness via DerivativeChecker."""
        bkd = self.bkd()
        npts = 8

        stress = NeoHookeanStress(lamda=1.0, mu=1.0)
        basis, _, _ = _setup_1d(npts, bkd)

        physics = HyperelasticityPhysics(basis, bkd, stress)
        wrapper = PhysicsDerivativeWrapper(physics)
        nstates = physics.nstates()
        sample = bkd.asarray([[0.01 * float(i) for i in range(nstates)]]).T

        # Use smaller fd_eps to avoid log(negative) from large perturbations
        np.random.seed(42)
        fd_eps = bkd.flip(bkd.logspace(-13, -2, 12))
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, fd_eps=fd_eps, verbosity=0)
        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-5)

    def test_solve_1d(self):
        """Numerical solve recovers manufactured solution."""
        bkd = self.bkd()
        npts = 16

        stress = NeoHookeanStress(lamda=1.0, mu=1.0)
        basis, mesh_bc, nodes = _setup_1d(npts, bkd)

        man_sol = ManufacturedHyperelasticityEquations(
            sol_strs=["0.1*x**2*(1-x)**2"],
            nvars=1,
            stress_model=stress,
            bkd=bkd,
            oned=True,
        )

        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        u_exact_flat = u_exact[:, 0]
        forcing_flat = forcing[:, 0]

        physics = HyperelasticityPhysics(
            basis, bkd, stress, forcing=lambda t: forcing_flat
        )
        _apply_homogeneous_dirichlet_bcs(physics, mesh_bc, npts, 1, bkd)

        model = CollocationModel(physics, bkd)
        initial_guess = bkd.zeros((npts,))
        u_numerical = model.solve_steady(initial_guess, tol=1e-10, maxiter=50)

        bkd.assert_allclose(u_numerical, u_exact_flat, atol=1e-8)


# ------------------------------------------------------------------
# 2D tests
# ------------------------------------------------------------------


class TestHyperelasticity2D(Generic[Array], unittest.TestCase):
    """Test 2D hyperelasticity with Neo-Hookean manufactured solutions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_residual_polynomial_2d(self):
        """Residual = 0 at exact polynomial solution."""
        bkd = self.bkd()
        npts_x, npts_y = 10, 10

        stress = NeoHookeanStress(lamda=1.0, mu=1.0)
        basis, mesh_bc, nodes = _setup_2d(npts_x, npts_y, bkd)

        man_sol = ManufacturedHyperelasticityEquations(
            sol_strs=[
                "0.1*x**2*(1-x)**2*y**2*(1-y)**2",
                "0.05*x**2*(1-x)**2*y**2*(1-y)**2",
            ],
            nvars=2,
            stress_model=stress,
            bkd=bkd,
            oned=True,
        )

        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        npts = basis.npts()
        u_exact_flat = bkd.concatenate([u_exact[:, 0], u_exact[:, 1]])
        forcing_flat = bkd.concatenate([forcing[:, 0], forcing[:, 1]])

        physics = HyperelasticityPhysics(
            basis, bkd, stress, forcing=lambda t: forcing_flat
        )
        _apply_homogeneous_dirichlet_bcs(physics, mesh_bc, npts, 2, bkd)

        residual = physics.residual(u_exact_flat, 0.0)
        jacobian = physics.jacobian(u_exact_flat, 0.0)
        residual_bc, _ = physics.apply_boundary_conditions(
            residual, jacobian, u_exact_flat, 0.0
        )

        interior = _get_interior_indices(mesh_bc, npts, 2)
        interior_res = bkd.asarray([residual_bc[i] for i in interior])

        bkd.assert_allclose(interior_res, bkd.zeros(interior_res.shape), atol=1e-8)

    def test_jacobian_2d(self):
        """Jacobian correctness via DerivativeChecker."""
        bkd = self.bkd()
        npts_x, npts_y = 6, 6

        stress = NeoHookeanStress(lamda=1.0, mu=1.0)
        basis, _, _ = _setup_2d(npts_x, npts_y, bkd)

        physics = HyperelasticityPhysics(basis, bkd, stress)
        wrapper = PhysicsDerivativeWrapper(physics)
        nstates = physics.nstates()
        sample = bkd.asarray([[0.01 * float(i) for i in range(nstates)]]).T

        # Use smaller fd_eps to avoid log(negative) from large perturbations
        np.random.seed(42)
        fd_eps = bkd.flip(bkd.logspace(-13, -2, 12))
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, fd_eps=fd_eps, verbosity=0)
        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-5)

    def test_solve_2d(self):
        """Numerical solve recovers manufactured solution."""
        bkd = self.bkd()
        npts_x, npts_y = 10, 10

        stress = NeoHookeanStress(lamda=1.0, mu=1.0)
        basis, mesh_bc, nodes = _setup_2d(npts_x, npts_y, bkd)

        man_sol = ManufacturedHyperelasticityEquations(
            sol_strs=[
                "0.1*x**2*(1-x)**2*y**2*(1-y)**2",
                "0.05*x**2*(1-x)**2*y**2*(1-y)**2",
            ],
            nvars=2,
            stress_model=stress,
            bkd=bkd,
            oned=True,
        )

        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        npts = basis.npts()
        u_exact_flat = bkd.concatenate([u_exact[:, 0], u_exact[:, 1]])
        forcing_flat = bkd.concatenate([forcing[:, 0], forcing[:, 1]])

        physics = HyperelasticityPhysics(
            basis, bkd, stress, forcing=lambda t: forcing_flat
        )
        _apply_homogeneous_dirichlet_bcs(physics, mesh_bc, npts, 2, bkd)

        model = CollocationModel(physics, bkd)
        initial_guess = bkd.zeros((2 * npts,))
        u_numerical = model.solve_steady(initial_guess, tol=1e-10, maxiter=50)

        bkd.assert_allclose(u_numerical, u_exact_flat, atol=1e-7)


# ------------------------------------------------------------------
# 3D tests (residual only, no Jacobian/solve)
# ------------------------------------------------------------------


class TestHyperelasticity3D(Generic[Array], unittest.TestCase):
    """Test 3D hyperelasticity residual with manufactured solutions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_residual_polynomial_3d(self):
        """Residual = 0 at exact polynomial solution."""
        bkd = self.bkd()
        npts_x, npts_y, npts_z = 6, 6, 6

        stress = NeoHookeanStress(lamda=1.0, mu=1.0)
        basis, mesh_bc, nodes = _setup_3d(npts_x, npts_y, npts_z, bkd)

        man_sol = ManufacturedHyperelasticityEquations(
            sol_strs=[
                "0.1*x**2*(1-x)**2*y**2*(1-y)**2*z**2*(1-z)**2",
                "0.05*x**2*(1-x)**2*y**2*(1-y)**2*z**2*(1-z)**2",
                "0.03*x**2*(1-x)**2*y**2*(1-y)**2*z**2*(1-z)**2",
            ],
            nvars=3,
            stress_model=stress,
            bkd=bkd,
            oned=True,
        )

        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        npts = basis.npts()
        u_exact_flat = bkd.concatenate([u_exact[:, i] for i in range(3)])
        forcing_flat = bkd.concatenate([forcing[:, i] for i in range(3)])

        physics = HyperelasticityPhysics(
            basis, bkd, stress, forcing=lambda t: forcing_flat
        )
        _apply_homogeneous_dirichlet_bcs(physics, mesh_bc, npts, 3, bkd)

        residual = physics.residual(u_exact_flat, 0.0)

        # Apply BCs by hand (no jacobian for 3D)
        for bc in physics.boundary_conditions():
            residual = bc.apply_to_residual(residual, u_exact_flat, 0.0)

        interior = _get_interior_indices(mesh_bc, npts, 3)
        interior_res = bkd.asarray([residual[i] for i in interior])

        bkd.assert_allclose(interior_res, bkd.zeros(interior_res.shape), atol=1e-7)


# ------------------------------------------------------------------
# Parameterized tests
# ------------------------------------------------------------------


class TestHyperelasticityParameterized(ParametrizedTestCase):
    """Parameterized tests for residual and Jacobian."""

    def bkd(self):
        return NumpyBkd()

    @parametrize(
        "name,sol_str,lamda,mu,npts_1d",
        [
            ("1d_basic", "0.1*x**2*(1-x)**2", 1.0, 1.0, 16),
            ("1d_soft", "0.2*x**2*(1-x)**2", 0.5, 0.5, 16),
            ("1d_stiff", "0.05*x**2*(1-x)**2", 5.0, 10.0, 16),
        ],
    )
    def test_residual_1d(self, name, sol_str, lamda, mu, npts_1d):
        """Parameterized 1D residual test."""
        bkd = self.bkd()
        stress = NeoHookeanStress(lamda=lamda, mu=mu)
        basis, mesh_bc, nodes = _setup_1d(npts_1d, bkd)

        man_sol = ManufacturedHyperelasticityEquations(
            sol_strs=[sol_str],
            nvars=1,
            stress_model=stress,
            bkd=bkd,
            oned=True,
        )

        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        u_exact_flat = u_exact[:, 0]
        forcing_flat = forcing[:, 0]

        physics = HyperelasticityPhysics(
            basis, bkd, stress, forcing=lambda t: forcing_flat
        )
        _apply_homogeneous_dirichlet_bcs(physics, mesh_bc, basis.npts(), 1, bkd)

        residual = physics.residual(u_exact_flat, 0.0)
        jacobian = physics.jacobian(u_exact_flat, 0.0)
        residual_bc, _ = physics.apply_boundary_conditions(
            residual, jacobian, u_exact_flat, 0.0
        )

        interior = _get_interior_indices(mesh_bc, basis.npts(), 1)
        interior_res = bkd.asarray([residual_bc[i] for i in interior])

        bkd.assert_allclose(interior_res, bkd.zeros(interior_res.shape), atol=1e-8)

    @parametrize(
        "name,u_str,v_str,lamda,mu,npts_1d",
        [
            (
                "2d_basic",
                "0.1*x**2*(1-x)**2*y**2*(1-y)**2",
                "0.05*x**2*(1-x)**2*y**2*(1-y)**2",
                1.0,
                1.0,
                10,
            ),
            (
                "2d_high_lambda",
                "0.05*x**2*(1-x)**2*y**2*(1-y)**2",
                "0.03*x**2*(1-x)**2*y**2*(1-y)**2",
                10.0,
                1.0,
                10,
            ),
        ],
    )
    def test_residual_2d(self, name, u_str, v_str, lamda, mu, npts_1d):
        """Parameterized 2D residual test."""
        bkd = self.bkd()
        stress = NeoHookeanStress(lamda=lamda, mu=mu)
        basis, mesh_bc, nodes = _setup_2d(npts_1d, npts_1d, bkd)

        man_sol = ManufacturedHyperelasticityEquations(
            sol_strs=[u_str, v_str],
            nvars=2,
            stress_model=stress,
            bkd=bkd,
            oned=True,
        )

        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        npts = basis.npts()
        u_exact_flat = bkd.concatenate([u_exact[:, 0], u_exact[:, 1]])
        forcing_flat = bkd.concatenate([forcing[:, 0], forcing[:, 1]])

        physics = HyperelasticityPhysics(
            basis, bkd, stress, forcing=lambda t: forcing_flat
        )
        _apply_homogeneous_dirichlet_bcs(physics, mesh_bc, npts, 2, bkd)

        residual = physics.residual(u_exact_flat, 0.0)
        jacobian = physics.jacobian(u_exact_flat, 0.0)
        residual_bc, _ = physics.apply_boundary_conditions(
            residual, jacobian, u_exact_flat, 0.0
        )

        interior = _get_interior_indices(mesh_bc, npts, 2)
        interior_res = bkd.asarray([residual_bc[i] for i in interior])

        bkd.assert_allclose(interior_res, bkd.zeros(interior_res.shape), atol=1e-8)

    @parametrize(
        "name,lamda,mu,npts_1d",
        [
            ("1d_jac", 1.0, 1.0, 8),
            ("1d_jac_stiff", 5.0, 10.0, 8),
            ("2d_jac", 1.0, 1.0, 6),
            ("2d_jac_high_mu", 0.5, 2.5, 6),
        ],
    )
    def test_jacobian(self, name, lamda, mu, npts_1d):
        """Parameterized Jacobian test via DerivativeChecker."""
        bkd = self.bkd()
        stress = NeoHookeanStress(lamda=lamda, mu=mu)

        if name.startswith("1d"):
            basis, _, _ = _setup_1d(npts_1d, bkd)
        else:
            basis, _, _ = _setup_2d(npts_1d, npts_1d, bkd)

        physics = HyperelasticityPhysics(basis, bkd, stress)
        wrapper = PhysicsDerivativeWrapper(physics)
        nstates = physics.nstates()
        sample = bkd.asarray([[0.01 * float(i) for i in range(nstates)]]).T

        # Use smaller fd_eps to avoid log(negative) from large perturbations
        np.random.seed(42)
        fd_eps = bkd.flip(bkd.logspace(-13, -2, 12))
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, fd_eps=fd_eps, verbosity=0)
        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-5)


# ------------------------------------------------------------------
# Concrete backend classes
# ------------------------------------------------------------------


class TestHyperelasticity1DNumpy(TestHyperelasticity1D[NDArray[Any]]):
    __test__ = True

    def bkd(self):
        return NumpyBkd()


class TestHyperelasticity1DTorch(TestHyperelasticity1D[torch.Tensor]):
    __test__ = True

    def bkd(self):
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)


class TestHyperelasticity2DNumpy(TestHyperelasticity2D[NDArray[Any]]):
    __test__ = True

    def bkd(self):
        return NumpyBkd()


class TestHyperelasticity2DTorch(TestHyperelasticity2D[torch.Tensor]):
    __test__ = True

    def bkd(self):
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)


class TestHyperelasticity3DNumpy(TestHyperelasticity3D[NDArray[Any]]):
    __test__ = True

    def bkd(self):
        return NumpyBkd()


class TestHyperelasticity3DTorch(TestHyperelasticity3D[torch.Tensor]):
    __test__ = True

    def bkd(self):
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)


if __name__ == "__main__":
    unittest.main()
